# frozen_string_literal: true

require_relative "sampler"
require_relative "descriptive"
require_relative "cholesky"

module SkiftetStatistical
  # The value of each arm as a linear sum of several rates, V_a = sum over parts p of w_p * r_(a, p), when both the
  # rates and the weights are uncertain. The rates are one joint normal over every (arm, part) pair (each arm's part
  # means and one covariance over all pairs, so arms that share their evidence stay correlated); the weights come from
  # named parameters drawn afresh in every joint draw. Because each draw takes its weights and its rates together,
  # P(best) and the expected loss hedge every uncertain weight at once instead of judging the arms at one assumed
  # weight.
  #
  #   SkiftetStatistical::LinearValue.summary(
  #     arms: %i[a b], parts: %i[p q],
  #     means: { a: [ 0.12, 0.031 ], b: [ 0.105, 0.044 ] },
  #     covariance: [ [ 4e-4, 6e-5, 2e-4, 1e-5 ], [ 6e-5, 1e-4, 2e-5, 4e-5 ],     # arm-major:
  #                   [ 2e-4, 2e-5, 5e-4, -5e-5 ], [ 1e-5, 4e-5, -5e-5, 1.2e-4 ] ], # (a, p), (a, q), (b, p), (b, q)
  #     parameters: { base: { fixed: 1.5 }, worth: { uniform: [ 2.0, 6.0 ] },
  #                   lift: { normal: [ 0.5, 0.2 ], clip: [ 0.0, 1.0 ] } },
  #     weights: ->(drawn) { { p: drawn.fetch(:base), q: drawn.fetch(:worth) * drawn.fetch(:lift) } },
  #     draws: 20_000, rng: Random.new(7))
  #   # => { a: { point: 0.242, mean: 0.2423..., lower: 0.1516..., upper: 0.3587..., p_best: 0.47715,
  #   #           expected_loss: 0.0188... },
  #   #      b: { point: 0.2455, mean: 0.2461..., lower: 0.1460..., upper: 0.3736..., p_best: 0.52285,
  #   #           expected_loss: 0.0150... } }
  #
  # THE DRAW ORDER IS THE CONTRACT (as JointBeta's). Per draw:
  # 1. each parameter in the order given: `fixed: x` takes nothing; `uniform: [lo, hi]` one `rng.rand` u, giving
  #    lo + (hi - lo) * u; `normal: [mean, sd]` one Sampler#gaussian g, giving mean + sd * g; then its `clip: [lo, hi]`
  #    when it has one (any kind may; a clip bound may be infinite);
  # 2. the part weights, `weights.call(drawn)` with the drawn values under the parameters' own names;
  # 3. one joint normal draw of every (arm, part) rate, z = means + L * n: n is K * P Sampler#gaussian values in
  #    arm-major order, L the Cholesky factor of the covariance (Cholesky.factor, once per call);
  # 4. each arm's value V_a = sum over parts p of w_p * z_(a, p).
  # A part with no variance is drawn at its mean and still takes its normal, so the numbers taken never depend on the
  # data. A seeded caller gets the same rows only while this order holds, never with a draw added, dropped or moved.
  #
  # Where Cholesky.factor refuses the covariance, L is Cholesky.root's square root S of it, read the same way with the
  # same K * P normals: the draw order holds and the joint normal is the same one, but S is not triangular, so its rows
  # are not the rows the factor would give. That is a covariance of lower rank whose rounding the given order cannot
  # carry. `rounding:` (a finite number of at least 0, 0.0 by default) is the largest rounding bound of the covariance's
  # entries, read only there: a caller with SharedCredit's estimate hands `estimate.rounding`, so a covariance summed
  # from rows is never refused for its own rounding. An entry cleared to 0 under its bound may be that bound and its own
  # rounding from the exact one, so Cholesky.root is given twice `rounding`. A covariance that neither takes raises.
  #
  # Everything is checked before the first random number (ArgumentError): 1 to 7 arms and 1 to 12 parts, each named
  # once by a Symbol or a String (a part shared by one weight is summed into one by the caller); `means` an Array of
  # P finite numbers per arm; `covariance` (K * P) rows of K * P, symmetric and positive semidefinite (Cholesky);
  # each parameter one kind of finite numbers (uniform lo <= hi, normal sd >= 0, clip lo <= hi); `weights` callable,
  # answering a Hash with a finite number for every part (read by Symbol or String; a key for no part is left out).
  # A finite number is a real Numeric that stays finite as a Float (an Integer or Rational beyond a Float's range, a
  # Complex, a NaN and an infinity are refused alike), the one number check the library's modules share.
  # The weights are first read at the parameters' centres, so a callable that leaves out a part is refused before any
  # draw; a draw whose weights leave one out raises too.
  module LinearValue
    MAX_ARMS = 7
    MAX_PARTS = 12
    # How many times its rounding bound an entry of a covariance may be from the exact one: cleared to 0 under the
    # bound, it may have been a bound and a half away. Cholesky.root is given this much.
    ENTRY_ROUNDING = 2.0

    class << self
      # Per arm (the arms as given): `point` the value at every parameter's centre (the fixed value, the uniform's
      # midpoint, the normal's mean, each clipped) and the part means; over the `draws` joint draws, `mean` the mean of
      # the drawn values, `lower` and `upper` their equal-tailed `credible_mass` interval (Descriptive.percentile),
      # `p_best` the share of draws the arm is best in (a tie split equally) and `expected_loss` the mean of the draw's
      # best value minus the arm's. Built on the rows each_draw yields: the same seed gives the same rows.
      def summary(arms:, parts:, means:, covariance:, parameters:, weights:, draws:, rng:, credible_mass: 0.95, rounding: 0.0)
        model = Model.new(arms: arms, parts: parts, means: means, covariance: covariance, parameters: parameters,
                          weights: weights, rounding: rounding)
        draws = draw_count(draws)
        raise ArgumentError, "summary needs at least one draw" if draws.zero?

        tail = (1.0 - mass(credible_mass)) / 2.0
        size = model.arms.size
        samples = Array.new(size) { Array.new(draws) }
        wins = Array.new(size, 0.0)
        losses = Array.new(size, 0.0)

        draw = 0
        model.each_row(draws, rng) do |row|
          best = row.max
          leaders = 0
          row.each_with_index do |value, index|
            samples[index][draw] = value
            losses[index] += best - value
            leaders += 1 if value == best
          end
          share = 1.0 / leaders
          row.each_with_index { |value, index| wins[index] += share if value == best }
          draw += 1
        end

        model.arms.each_with_index.to_h do |arm, index|
          values = samples[index]
          [ arm, {
            point: model.point[index],
            mean: values.sum / draws,
            lower: Descriptive.percentile(values, tail * 100.0),
            upper: Descriptive.percentile(values, (1.0 - tail) * 100.0),
            p_best: wins[index] / draws,
            expected_loss: losses[index] / draws
          } ]
        end
      end

      # Yields one Array per draw, each arm's value V in `arms` order, and returns nil. Without a block the rows are
      # drawn at once and an Enumerator over them is answered, so reading it again gives the same rows and the RNG is
      # left where the block form leaves it.
      def each_draw(arms:, parts:, means:, covariance:, parameters:, weights:, draws:, rng:, rounding: 0.0, &block)
        model = Model.new(arms: arms, parts: parts, means: means, covariance: covariance, parameters: parameters,
                          weights: weights, rounding: rounding)
        draws = draw_count(draws)
        unless block
          rows = []
          model.each_row(draws, rng) { |row| rows << row }
          return rows.each
        end

        model.each_row(draws, rng, &block)
        nil
      end

      private

      def draw_count(draws)
        raise ArgumentError, "draws must be an Integer >= 0, got #{draws.inspect}" unless draws.is_a?(Integer) && draws >= 0

        draws
      end

      def mass(credible_mass)
        unless credible_mass.is_a?(Numeric) && credible_mass.real? && credible_mass > 0 && credible_mass < 1
          raise ArgumentError, "credible_mass must be a number above 0 and below 1, got #{credible_mass.inspect}"
        end

        credible_mass.to_f
      end
    end

    # One call's checked inputs: the factor (taken once), the flat arm-major means, the parameters and the value at
    # their centres. Its #each_row is the one draw order both public methods read.
    class Model
      KINDS = %i[fixed uniform normal].freeze
      KEYS = (KINDS + %i[clip]).freeze

      attr_reader :arms, :point

      def initialize(arms:, parts:, means:, covariance:, parameters:, weights:, rounding:)
        @arms = names(arms, "arms", MAX_ARMS)
        @parts = names(parts, "parts", MAX_PARTS)
        @size = @arms.size * @parts.size
        @means = flat_means(means)
        @parameters = read_parameters(parameters)
        raise ArgumentError, "weights must respond to #call, got #{weights.class}" unless weights.respond_to?(:call)

        @weights = weights
        @lower, @bands = factor_bands(covariance, read_rounding(rounding))
        @point = values(part_weights(centres), @means)
      end

      def each_row(draws, rng)
        sampler = Sampler.new(rng)
        draws.times do
          weight = part_weights(draw_parameters(rng, sampler))
          normals = Array.new(@size) { sampler.gaussian }
          yield values(weight, rates(normals))
        end
      end

      private

      def names(list, label, most)
        raise ArgumentError, "#{label} must be an Array, got #{list.class}" unless list.is_a?(Array)
        raise ArgumentError, "#{label} must name 1 to #{most}, got #{list.size}" unless list.size.between?(1, most)

        list.each do |name|
          raise ArgumentError, "#{label} must be Symbols or Strings, got #{name.inspect}" unless name.is_a?(Symbol) || name.is_a?(String)
        end
        raise ArgumentError, "#{label} must each be named once" unless list.map(&:to_s).uniq.size == list.size

        list.dup.freeze
      end

      def flat_means(means)
        raise ArgumentError, "means must be a Hash of arm => [one mean per part], got #{means.class}" unless means.is_a?(Hash)

        @arms.flat_map do |arm|
          row = lookup(means, arm) { raise ArgumentError, "means has no row for arm #{arm.inspect}" }
          unless row.is_a?(Array) && row.size == @parts.size
            raise ArgumentError, "means of arm #{arm.inspect} must be #{@parts.size} numbers, one per part"
          end

          row.map { |value| finite(value, "a mean of arm #{arm.inspect}") }
        end
      end

      # [name, kind, first, second, clip] per parameter, in the order given.
      def read_parameters(parameters)
        raise ArgumentError, "parameters must be a Hash of name => distribution, got #{parameters.class}" unless parameters.is_a?(Hash)

        parameters.map { |name, spec| read_parameter(name, spec) }
      end

      def read_parameter(name, spec)
        raise ArgumentError, "parameter #{name.inspect} must be a Hash, got #{spec.inspect}" unless spec.is_a?(Hash)

        spec = spec.to_h { |key, value| [ key.respond_to?(:to_sym) ? key.to_sym : key, value ] }
        unknown = spec.keys - KEYS
        raise ArgumentError, "parameter #{name.inspect} has unknown key(s) #{unknown.inspect}" unless unknown.empty?

        kinds = spec.keys & KINDS
        raise ArgumentError, "parameter #{name.inspect} must have one of #{KINDS.inspect}" unless kinds.size == 1

        kind = kinds.first
        label = "parameter #{name.inspect}"
        first, second =
          if kind == :fixed
            [ finite(spec.fetch(:fixed), label), nil ]
          else
            pair(spec.fetch(kind), "#{label} #{kind}")
          end
        raise ArgumentError, "#{label}: uniform needs lo <= hi" if kind == :uniform && first > second
        raise ArgumentError, "#{label}: normal needs sd >= 0" if kind == :normal && second.negative?

        [ name, kind, first, second, spec.key?(:clip) ? clip(spec.fetch(:clip), label) : nil ]
      end

      def pair(value, label)
        raise ArgumentError, "#{label} must be two numbers, got #{value.inspect}" unless value.is_a?(Array) && value.size == 2

        value.map { |number| finite(number, label) }
      end

      def clip(value, label)
        unless value.is_a?(Array) && value.size == 2 && value.all? { |bound| clip_bound?(bound) }
          raise ArgumentError, "#{label} clip must be two numbers [lo, hi], got #{value.inspect}"
        end

        bounds = value.map(&:to_f)
        raise ArgumentError, "#{label} clip needs lo <= hi" if bounds[0] > bounds[1]

        bounds
      end

      # A real number, an infinity allowed only as an infinity: one no Float holds is not read as one.
      def clip_bound?(bound)
        number?(bound) && !bound.to_f.nan? && (bound.to_f.finite? || !bound.finite?)
      end

      def read_rounding(value)
        number = finite(value, "rounding")
        raise ArgumentError, "rounding must be at least 0, got #{value.inspect}" if number.negative?

        number
      end

      # The factor's rows and, per row, the band [first, last] of its nonzero columns (first > last for a row of
      # zeros), so a row skips the zero blocks of independent arms; a zero inside the band adds a zero to its sum.
      def factor_bands(covariance, rounding)
        unless covariance.is_a?(Array) && covariance.size == @size
          raise ArgumentError, "covariance must be #{@size} rows of #{@size} (arms x parts, arm-major)"
        end

        lower = square_root(covariance, rounding).map(&:freeze).freeze
        bands = lower.map do |row|
          nonzero = row.each_index.reject { |column| row[column].zero? }
          nonzero.empty? ? [ 0, -1 ].freeze : [ nonzero.first, nonzero.last ].freeze
        end
        [ lower, bands.freeze ]
      end

      # The factor where it takes the covariance, so a covariance it takes draws its rows; the root where it refuses.
      def square_root(covariance, rounding)
        Cholesky.factor(covariance)
      rescue ArgumentError
        Cholesky.root(covariance, rounding: ENTRY_ROUNDING * rounding)
      end

      # z = means + L * n, each row summed left to right over its band.
      def rates(normals)
        Array.new(@size) do |index|
          row = @lower[index]
          column, last = @bands[index]
          total = 0.0
          while column <= last
            total += row[column] * normals[column]
            column += 1
          end
          @means[index] + total
        end
      end

      def centres
        @parameters.to_h do |name, kind, first, second, bounds|
          centre = kind == :uniform ? (first + second) / 2.0 : first
          [ name, bounds ? centre.clamp(*bounds) : centre ]
        end
      end

      def draw_parameters(rng, sampler)
        @parameters.to_h do |name, kind, first, second, bounds|
          value =
            case kind
            when :fixed then first
            when :uniform then first + ((second - first) * rng.rand)
            else first + (second * sampler.gaussian)
            end
          [ name, bounds ? value.clamp(*bounds) : value ]
        end
      end

      def part_weights(drawn)
        given = @weights.call(drawn)
        raise ArgumentError, "the weights callable must answer a Hash of part => weight, got #{given.class}" unless given.is_a?(Hash)

        @parts.map do |part|
          weight = lookup(given, part) { raise ArgumentError, "the weights callable left out part #{part.inspect}" }
          finite(weight, "the weight of part #{part.inspect}")
        end
      end

      # V_a = sum over parts p of w_p * rate_(a, p), the rates arm-major, summed left to right over the parts.
      def values(weight, rates)
        parts = weight.size
        Array.new(@arms.size) do |arm|
          offset = arm * parts
          total = 0.0
          part = 0
          while part < parts
            total += weight[part] * rates[offset + part]
            part += 1
          end
          total
        end
      end

      def lookup(hash, name)
        return hash[name] if hash.key?(name)

        other = name.is_a?(Symbol) ? name.to_s : name.to_sym
        return hash[other] if hash.key?(other)

        yield
      end

      def number?(value)
        value.is_a?(Numeric) && value.real?
      end

      # A finite real number, finite as a Float too (an Integer or Rational beyond a Float's range is not), as a Float:
      # the library's one number check.
      def finite(value, label)
        number = value.to_f if number?(value) && value.finite?
        return number if number&.finite?

        raise ArgumentError, "#{label} must be a finite number, got #{value.inspect}"
      end
    end
    private_constant :Model
  end
end
