# frozen_string_literal: true

require_relative "numbers"
require_relative "inputs"
require_relative "beta_function"
require_relative "gauss_legendre"
require_relative "significance"

module SkiftetStatistical
  # Which of several independent Beta variables is the largest: the exact probability of each member, the seedless
  # pick that probability describes, and the greedy pick.
  #
  # The pick takes three kinds of uniforms from its caller, never from an RNG: one exploration uniform, a priority per
  # member and a uniform per member. Below epsilon it explores (the member with the smallest priority, so each member
  # equally often); otherwise it takes the member whose Beta quantile at its own uniform is the largest (a quantile at
  # a uniform is a draw from that Beta). Over uniform inputs a member c is therefore picked with probability
  #
  #   epsilon / k + (1 - epsilon) * integral of f_c(x) * product over j != c of F_j(x) dx
  #
  # (f the Beta density, F its distribution function), and that is what probabilities answers, so a caller can log
  # the exact probability of the pick it made. Two callers handing the same uniforms to members of equal parameters
  # pick alike.
  #
  # A Beta with a parameter well below 1 holds much of its mass closer to an end of (0, 1) than a Float can say near
  # 1, and closer than a bisection of (0, 1) can say near 0: Q(0.9; 1, 0.05) is 1 - 1e-20. So both answers work on the
  # two halves of (0, 1) apart, each in the distance w from its own end: w = x up to 1/2, with the member's own (alpha,
  # beta), and w = 1 - x beyond, with the reflected (beta, alpha), whose density and distribution function at w are
  # the member's at 1 - w read without rounding.
  #
  # The integral is composite Gauss-Legendre at NODES nodes a panel over each half, in v = log w, the panels broken at
  # the half's quantiles at BREAKS of every member, then normalised so the members sum to 1. In v a density's pole at
  # an end is a plain exponential, whatever its parameter, so the panels hold it. BREAKS is symmetric, so together the
  # two halves break the range at every member's nine quantiles, on the range from min_j Q_j(TAIL) to the same
  # distance from 1 at the top. That leaves at most TAIL of a member out at each end, the largest error that remains:
  # at most 4e-9 of an answer with twenty members.
  #
  # 0.2.0's distribution function, which stays as it is, stops its continued fraction at 300 steps: near the mean of
  # Beta(10**6, 2 * 10**6) that reads 7e-7 off, and an answer 3.6e-8 off. So the integral and the pick alike read this
  # module's own distribution function, 0.2.0's arithmetic with the fraction run to convergence.
  #
  # Both are exact to 1e-8 absolute for every member they take: closed forms measured over parameters from 1e-6 up read
  # at most 3e-9 off, sharp pairs against Evan Miller's exact sum up to alpha + beta = 3 * 10**6 at most 2.1e-9.
  # probabilities and pick refuse what lies beyond those measures, and greedy, which integrates nothing, takes it: a
  # parameter below SMALLEST_PARAMETER and alpha + beta above LARGEST_TOTAL.
  #
  #   SkiftetStatistical::BetaArgmax.probabilities([ [ 1.0, 1.0 ], [ 2.0, 1.0 ] ])         # => [1/3, 2/3]
  #   SkiftetStatistical::BetaArgmax.pick([ [ 2.0, 8.0 ], [ 8.0, 2.0 ] ], explore: 0.5, priorities: [ 0.3, 0.6 ],
  #                                       uniforms: [ 0.5, 0.5 ], epsilon: 0.1)              # => 1
  #   SkiftetStatistical::BetaArgmax.greedy([ [ 1.0, 3.0 ], [ 3.0, 1.0 ] ])                # => 1
  #
  # Members are given as [[alpha, beta], ...], 1 to MAX_MEMBERS of them, alpha and beta finite numbers > 0, inside the
  # limits above for probabilities and pick. Every answer breaks a tie to the lower index.
  #
  # Since 0.4.0 three more answers sit beside these, each new and each leaving probabilities, pick and greedy as they
  # were (pick still compares its own position, below):
  #
  # expected_maximum is E[max] = 1 - the integral over (0, 1) of the product of the members' distribution functions,
  # by the same panels and the same rule. Each half is integrated in the distance w from its own end, so the part below
  # 1/2 is the integral of the product, and the part above is 1/2 less the integral of 1 - the product of the
  # reflected members' complements: both integrands vanish at w = 0, so leaving out what lies below the half's lowest
  # quantile at TAIL costs at most TAIL a member.
  #
  # position is a comparison key of a draw: where the member's quantile at a uniform lies, on the scale pick compares,
  # which rounds at neither end. It solves the same distribution function as the integral, by Newton's method in v
  # inside a bracket that always holds the root, halving the bracket whenever a step would leave it or fails to halve
  # the step before (Numerical Recipes' rtsafe), from a Normal start. The bracket comes from the leading term
  # w**p / (p * B(p, q)), which bounds the distribution function from above when q >= 1 and from below when q < 1, the
  # other end the joint (or, for q < 1, the leading term less (1 - q) * log(2) / p, since (1 - w)**(q - 1) <= 2**(1 - q)
  # on (0, 1/2]). It is a separate method from the bisection pick compares, faster, and within the distribution
  # function's noise of it except near the ends of (0, 1).
  #
  # distribution is that distribution function itself, F(x) of one member: the one probabilities, pick, position and
  # expected_maximum read, so a caller integrating against it reads the same function, bit for bit.
  #
  # The three read their numbers by the library's one number reader (SkiftetStatistical::Numbers, the input domain)
  # and hold a member to the limits probabilities and pick take, an ArgumentError otherwise.
  #
  #   SkiftetStatistical::BetaArgmax.expected_maximum([ [ 1.0, 1.0 ], [ 1.0, 1.0 ] ])      # => 2/3
  #   SkiftetStatistical::BetaArgmax.position(0.5, 2.0, 2.0)                               # => 0.0, the draw 1/2
  #   SkiftetStatistical::BetaArgmax.distribution(0.3, 2.0, 8.0)                           # => 0.8039967659999996
  module BetaArgmax
    # The probability left out of the integration range at each end.
    TAIL = 1e-10
    # The quantile levels at which every member breaks the panels.
    BREAKS = [ TAIL, 1e-4, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0 - 1e-4, 1.0 - TAIL ].freeze
    # Gauss-Legendre nodes a panel.
    NODES = 20
    # The most members one call takes.
    MAX_MEMBERS = 20
    # The smallest alpha or beta probabilities and pick take. Further down a member's distribution function comes
    # within rounding of its ends: a draw's position stops rising with its uniform, pick can raise and a probability
    # can round below 0 (measured down to 1e-305; none of it from here up).
    SMALLEST_PARAMETER = 1e-6
    # The largest alpha + beta they take, as far as the measures go. lgamma's rounding reaches the normalising factor
    # and grows with the total: the worst answer reads 2e-9 off here, 7e-9 at 10**7 and 1.5e-8 at 2 * 10**7.
    LARGEST_TOTAL = 3_000_000.0
    # Where the two halves meet, in v: log(1/2).
    JOINT = Math.log(0.5)
    # Below this a quantile is solved in logs. The bisection ends on a bracket of 2**-80, so it is exact to a Float
    # above here and floors at 2**-81 far below.
    BISECTION_FLOOR = 2.0**-30
    # A bound, not a tuning knob: under BISECTION_FLOOR the log distribution function is nearly a line in v, so Newton
    # takes a few steps from its leading term.
    NEWTON_STEPS = 50
    NEWTON_TOLERANCE = 1e-14
    # A bound, not a tuning knob: up to LARGEST_TOTAL the fraction converges within 600 steps (measured).
    FRACTION_STEPS = 20_000
    # position stops once a step in v is this small relative to max(1, |v|). A Newton step this small leaves an error
    # far below it, and a halving step leaves at most it.
    POSITION_TOLERANCE = 1e-13
    # A bound, not a tuning knob: halving alone narrows the widest bracket (about 1e9 wide, at SMALLEST_PARAMETER and a
    # uniform of 1e-300) to POSITION_TOLERANCE within 80 steps.
    POSITION_STEPS = 200

    class << self
      # P(member c is the largest), mixed with epsilon / k, for every member in order; [1.0] for one member.
      # epsilon lies in [0, 1]. An integral that rounds below 0 counts as 0, so the answers lie in [0, 1] and sum to 1.
      def probabilities(params, epsilon: 0.0)
        members = integrable(members(params))
        epsilon = unit(epsilon, "epsilon")
        return [ 1.0 ] if members.size == 1

        integrals = integrals(members).map { |integral| integral.negative? ? 0.0 : integral }
        total = integrals.sum
        unless total.finite? && total.positive?
          raise ArgumentError, "the members' parameters are beyond what the rule can integrate, got #{params.inspect}"
        end

        k = members.size
        integrals.map { |integral| (epsilon / k) + ((1.0 - epsilon) * integral / total) }
      end

      # The index picked: below epsilon (explore < epsilon) the member with the smallest priority, else the member
      # with the largest quantile at its uniform, compared by position so that no two draws tie by rounding. explore
      # and every uniform lie in [0, 1], every priority is a finite number, one priority and one uniform per member.
      def pick(params, explore:, priorities:, uniforms:, epsilon:)
        members = integrable(members(params))
        explore = unit(explore, "explore")
        epsilon = unit(epsilon, "epsilon")
        priorities = per_member(priorities, members.size, "priorities").map { |value| finite(value, "each priority") }
        uniforms = per_member(uniforms, members.size, "uniforms").map { |value| unit(value, "each uniform") }

        return first_smallest(priorities) if explore < epsilon

        first_largest(members.each_with_index.map { |(alpha, beta), j| pick_position(uniforms[j], alpha, beta) })
      end

      # The index of the largest mean alpha / (alpha + beta).
      def greedy(params)
        first_largest(members(params).map { |alpha, beta| alpha / (alpha + beta) })
      end

      # E[max] of the members, 1 - the integral over (0, 1) of the product of their distribution functions; one
      # member's mean exactly.
      def expected_maximum(params)
        members = integrable(members(read_members(params)))
        if members.size == 1
          alpha, beta = members.first
          return alpha / (alpha + beta)
        end

        below = area(members) { |cdfs| cdfs.inject(:*) }
        above = area(members.map(&:reverse)) { |cdfs| 1.0 - cdfs.inject(1.0) { |product, cdf| product * (1.0 - cdf) } }
        0.5 - below + above
      end

      # Where the draw of Beta(alpha, beta) at uniform lies: log(2 * x) up to 1/2, then -log(2 * (1 - x)), so
      # x = exp(p + JOINT) for a position p < 0, else 1 - exp(JOINT - p). Minus infinity at a uniform of 0 and infinity
      # at 1. uniform lies in [0, 1]; alpha and beta inside the limits pick takes.
      def position(uniform, alpha, beta)
        uniform = unit(Numbers.read(uniform, "uniform"), "uniform")
        alpha, beta = member(alpha, beta)

        locate(uniform, alpha, beta)
      end

      # The Beta(alpha, beta) distribution function at x, I_x(alpha, beta): the one this module integrates, picks and
      # positions on (0.2.0's arithmetic with the fraction run to convergence), read at v = log x. 0.0 at or below 0
      # and 1.0 at or above 1. x is a finite number; alpha and beta inside the limits pick takes.
      def distribution(x, alpha, beta)
        x = Numbers.read(x, "x")
        alpha, beta = member(alpha, beta)
        return 0.0 unless x.positive?
        return 1.0 if x >= 1.0

        cdf(Math.log(x), x, alpha, beta, log_constant(alpha, beta))
      end

      private

      # The members a method new in 0.4.0 takes, read for the readers below (members, integrable), which then bound
      # them as they bound every member: params an Array of [alpha, beta] pairs, each read by the library's one
      # reading of a pair (Numbers.pair, its numbers in the input domain) and answered as its Floats, so a value those
      # readers cannot ask (one that answers no method, say) is an ArgumentError too.
      def read_members(params)
        unless Array === params
          raise ArgumentError, "params must be a non-empty Array of [alpha, beta] pairs, got #{Inputs.shown(params)}"
        end

        Inputs.elements(params).each_with_index.map { |pair, i| Numbers.pair(pair, "member #{i}", %w[alpha beta]) }
      end

      # One member's alpha and beta, read (read_members) and held inside the limits pick takes (members, integrable).
      def member(alpha, beta)
        integrable(members(read_members([ [ alpha, beta ] ]))).first
      end

      # Each member's integral of f_c * product of the others' F: its part below 1/2 plus its part above, where an
      # other member's F(1 - w) is 1 minus the reflected member's distribution function at w.
      def integrals(members)
        below = half(members) { |cdf| cdf }
        above = half(members.map(&:reverse)) { |cdf| 1.0 - cdf }
        below.zip(above).map(&:sum)
      end

      # Each member's integral over one half, given the members' parameters in that half: over w in (0, 1/2] in
      # v = log w, whose density is f(w) * w = exp(constant + p * v + (q - 1) * log(1 - w)). The block turns a member's
      # distribution function at w into its factor in the others' product.
      def half(shapes)
        rule = GaussLegendre.nodes(NODES)
        constants = shapes.map { |p, q| log_constant(p, q) }
        sums = Array.new(shapes.size, 0.0)
        points(shapes).each_cons(2) do |left, right|
          half_width = (right - left) / 2.0
          middle = (left + right) / 2.0
          panel = Array.new(shapes.size, 0.0)
          rule.each do |node, weight|
            v = middle + (half_width * node)
            w = Math.exp(v)
            factors = shapes.each_with_index.map { |(p, q), j| yield cdf(v, w, p, q, constants[j]) }
            shapes.each_with_index do |(p, q), c|
              density = Math.exp(constants[c] + (p * v) + ((q - 1.0) * Math.log(1.0 - w)))
              next if density.zero?

              others = 1.0
              factors.each_with_index { |factor, j| others *= factor unless j == c }
              panel[c] += weight * density * others
            end
          end
          panel.each_with_index { |value, c| sums[c] += half_width * value }
        end
        sums
      end

      # The integral over one half of a function of the members' distribution functions, given the members'
      # parameters in that half: over w in (0, 1/2] in v = log w, so dw = w dv. The block turns the members'
      # distribution functions at w into the integrand, which must vanish as w goes to 0. The panels are those of
      # points, run up to the joint whatever the members, since the integrand need not vanish there.
      def area(shapes)
        rule = GaussLegendre.nodes(NODES)
        constants = shapes.map { |p, q| log_constant(p, q) }
        sum = 0.0
        (points(shapes).select { |v| v < JOINT } << JOINT).sort.uniq.each_cons(2) do |left, right|
          half_width = (right - left) / 2.0
          middle = (left + right) / 2.0
          panel = 0.0
          rule.each do |node, weight|
            v = middle + (half_width * node)
            w = Math.exp(v)
            panel += weight * w * yield(shapes.each_with_index.map { |(p, q), j| cdf(v, w, p, q, constants[j]) })
          end
          sum += half_width * panel
        end
        sum
      end

      # The panels of one half in v: every member's quantiles at BREAKS that lie in it, sorted, each once, closed by
      # the joint unless every member's top quantile stops short of it. A member's quantiles rise with the level, so
      # the smallest is min_j Q_j(TAIL) and, short of the joint, the largest is max_j Q_j(1 - TAIL). No level below
      # the joint for any member leaves the half with no panels: all of them lie past it.
      def points(shapes)
        open_at_joint = false
        breaks = shapes.flat_map do |p, q|
          at_joint = cdf(JOINT, 0.5, p, q, log_constant(p, q))
          open_at_joint ||= BREAKS.last >= at_joint
          BREAKS.select { |level| level < at_joint }.map { |level| log_quantile(level, p, q) }
        end
        return breaks.sort.uniq unless open_at_joint

        (breaks.select { |v| v < JOINT } << JOINT).sort.uniq
      end

      # Where a draw lies, on a scale that rises with x and rounds at neither end: log(2 * x) up to 1/2, then
      # -log(2 * (1 - x)), each read through the quantile of its own half.
      # (0.3.0's private position, the key pick compares, renamed in 0.4.0 only because the public position took its
      # name; its code is 0.3.0's.)
      def pick_position(uniform, alpha, beta)
        return -Float::INFINITY if uniform.zero?
        return Float::INFINITY if uniform == 1.0

        constant = log_constant(alpha, beta)
        return log_quantile(uniform, alpha, beta) - JOINT if uniform < cdf(JOINT, 0.5, alpha, beta, constant)

        # 1 - uniform rounds to 1 only under 2**-53; x itself says that draw, unless beta is as small as the uniform.
        level = 1.0 - uniform
        return JOINT - log_quantile(level, beta, alpha) if level < 1.0

        JOINT - Math.log(1.0 - quantile(uniform, alpha, beta, constant))
      end

      # The public position, its inputs read: each half's quantile solved by log_position.
      def locate(uniform, alpha, beta)
        return -Float::INFINITY if uniform.zero?
        return Float::INFINITY if uniform == 1.0

        constant = log_constant(alpha, beta)
        return log_position(uniform, alpha, beta, constant) - JOINT if uniform < cdf(JOINT, 0.5, alpha, beta, constant)

        # 1 - uniform rounds to 1 only under 2**-53; x itself says that draw, unless beta is as small as the uniform.
        level = 1.0 - uniform
        return JOINT - log_position(level, beta, alpha, log_constant(beta, alpha)) if level < 1.0

        JOINT - Math.log(1.0 - quantile(uniform, alpha, beta, constant))
      end

      # log Q(level; p, q) for a level in (0, 1) whose quantile lies in (0, 1/2]: the root in v of
      # log I_w(p, q) - log(level), by Newton's method inside a bracket that always holds it (see the module's notes).
      # A step that would leave the bracket, or that is not at most half the step before the last, halves the bracket
      # instead, so the answer is found however flat or steep the distribution function is. The bracket's first ends
      # hold the root by arithmetic, not by a value read there, and the leading term is exact when q is 1, so a step
      # that passes an end not yet tried (by rounding, or by overshooting) tries that end itself.
      def log_position(level, p, q, constant)
        target = Math.log(level)
        leading = (target - constant + Math.log(p)) / p
        if q >= 1.0
          low = [ leading, JOINT ].min
          high = JOINT
        else
          low = [ leading - ((1.0 - q) * Math.log(2.0) / p), JOINT ].min
          high = [ leading, JOINT ].min
        end
        low_tried = high_tried = false

        total = p + q
        start = (p / total) + (Math.sqrt(p * q / (total * total * (total + 1.0))) * Significance.normal_quantile(level))
        v = (start.positive? ? Math.log(start) : leading).clamp(low, high)

        step = Float::INFINITY
        POSITION_STEPS.times do
          w = Math.exp(v)
          below = cdf(v, w, p, q, constant)
          gap = Math.log(below) - target
          return v if gap.zero?

          if gap.negative?
            low = v
            low_tried = true
          else
            high = v
            high_tried = true
          end
          newton = gap / Math.exp(constant + (p * v) + ((q - 1.0) * Math.log(1.0 - w)) - Math.log(below))
          landing = v - newton
          landing = low if !low_tried && landing < low
          landing = high if !high_tried && landing > high
          before = step
          step = v - landing
          if newton.finite? && landing.between?(low, high) && (2.0 * step).abs <= before.abs
            v = landing
          else
            step = (high - low) / 2.0
            v = low + step
          end
          break if step.abs <= POSITION_TOLERANCE * [ 1.0, v.abs ].max
        end
        v
      end

      # log Q(level; p, q) for a level in (0, 1) whose quantile lies in (0, 1/2]: the bisection while that can say it,
      # else Newton on the log distribution function in v, from its leading term.
      def log_quantile(level, p, q)
        constant = log_constant(p, q)
        x = quantile(level, p, q, constant)
        return Math.log(x) if x >= BISECTION_FLOOR

        target = Math.log(level)
        v = (target - constant + Math.log(p)) / p
        NEWTON_STEPS.times do
          w = Math.exp(v)
          log_cdf = Math.log(cdf(v, w, p, q, constant))
          step = (log_cdf - target) / Math.exp(constant + (p * v) + ((q - 1.0) * Math.log(1.0 - w)) - log_cdf)
          break unless step.finite?

          v -= step
          break if step.abs <= NEWTON_TOLERANCE * v.abs
        end
        v
      end

      # 0.2.0's quantile, on this module's distribution function.
      def quantile(level, p, q, constant)
        low = 0.0
        high = 1.0
        BetaFunction::QUANTILE_STEPS.times do
          middle = (low + high) / 2.0
          if cdf(Math.log(middle), middle, p, q, constant) < level
            low = middle
          else
            high = middle
          end
        end
        (low + high) / 2.0
      end

      # I_w(p, q) at w = exp(v), v kept beside it: 0.2.0's arithmetic, from the member's constant, with the fraction
      # run to convergence. Below a Float's smallest normal the distribution function is its leading term
      # w**p / (p * B(p, q)) to a Float's precision (the next term is q * w of it), and w itself may be 0.
      def cdf(v, w, p, q, constant)
        return [ Math.exp(constant + (p * v) - Math.log(p)), 1.0 ].min if w < Float::MIN

        front = Math.exp(constant + (p * v) + (q * Math.log(1.0 - w)))
        return front * fraction(w, p, q) / p if w < (p + 1.0) / (p + q + 2.0)

        1.0 - (front * fraction(1.0 - w, q, p) / q)
      end

      # 0.2.0's continued fraction (Numerical Recipes' betacf by Lentz's method), but for how long it runs.
      def fraction(x, alpha, beta)
        qab = alpha + beta
        qap = alpha + 1.0
        qam = alpha - 1.0

        c = 1.0
        d = 1.0 / nonzero(1.0 - (qab * x / qap))
        h = d

        (1..FRACTION_STEPS).each do |m|
          m2 = 2 * m

          numerator = m * (beta - m) * x / ((qam + m2) * (alpha + m2))
          d = 1.0 / nonzero(1.0 + (numerator * d))
          c = nonzero(1.0 + (numerator / c))
          h *= d * c

          numerator = -(alpha + m) * (qab + m) * x / ((alpha + m2) * (qap + m2))
          d = 1.0 / nonzero(1.0 + (numerator * d))
          c = nonzero(1.0 + (numerator / c))
          delta = d * c
          h *= delta

          break if (delta - 1.0).abs < BetaFunction::EPSILON
        end

        h
      end

      def nonzero(value)
        value.abs < BetaFunction::TINY ? BetaFunction::TINY : value
      end

      # log(1 / B(p, q)).
      def log_constant(p, q)
        Math.lgamma(p + q)[0] - Math.lgamma(p)[0] - Math.lgamma(q)[0]
      end

      def first_largest(values)
        best = 0
        values.each_with_index { |value, i| best = i if value > values[best] }
        best
      end

      def first_smallest(values)
        best = 0
        values.each_with_index { |value, i| best = i if value < values[best] }
        best
      end

      def members(params)
        unless params.is_a?(Array) && !params.empty?
          raise ArgumentError, "params must be a non-empty Array of [alpha, beta] pairs, got #{params.inspect}"
        end
        if params.size > MAX_MEMBERS
          raise ArgumentError, "params may hold at most #{MAX_MEMBERS} members, got #{params.size}"
        end

        params.each_with_index.map do |pair, i|
          unless pair.is_a?(Array) && pair.size == 2
            raise ArgumentError, "member #{i} must be an [alpha, beta] pair, got #{pair.inspect}"
          end

          pair.zip(%w[alpha beta]).map { |value, name| positive(value, "member #{i}'s #{name}") }
        end
      end

      def integrable(members)
        members.each_with_index do |(alpha, beta), i|
          next if [ alpha, beta ].min >= SMALLEST_PARAMETER && alpha + beta <= LARGEST_TOTAL

          raise ArgumentError, "member #{i} is beyond what can be answered to 1e-8: alpha and beta must be at least " \
                               "#{SMALLEST_PARAMETER} and sum to at most #{LARGEST_TOTAL}, got #{alpha}, #{beta}"
        end
      end

      def per_member(values, size, name)
        unless values.is_a?(Array) && values.size == size
          raise ArgumentError, "#{name} must be an Array of one value per member (#{size}), got #{values.inspect}"
        end

        values
      end

      # A finite real number, as the other modules read one (a NaN, an infinity, a Complex and a non-number are
      # refused alike), and finite as a Float too (an Integer beyond a Float's range is not).
      def finite(value, name)
        number = value.to_f if value.is_a?(Numeric) && value.real? && value.finite?
        return number if number&.finite?

        raise ArgumentError, "#{name} must be a finite number, got #{value.inspect}"
      end

      def positive(value, name)
        value = finite(value, name)
        raise ArgumentError, "#{name} must be > 0, got #{value}" unless value.positive?

        value
      end

      def unit(value, name)
        value = finite(value, name)
        raise ArgumentError, "#{name} must lie in [0, 1], got #{value}" unless value.between?(0.0, 1.0)

        value
      end
    end
  end
end
