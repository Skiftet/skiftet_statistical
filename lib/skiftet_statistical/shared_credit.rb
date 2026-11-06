# frozen_string_literal: true

module SkiftetStatistical
  # Shared-credit marginal inverse-propensity estimates: each arm's mean of several parts, with one covariance over
  # every (arm, part) pair, when one row may count for several arms at once. A row is
  # `{ credit: { arm => w >= 0 }, values: [x_1, ..., x_P] }`. It counts for every arm it credits, with that arm's weight:
  # in an ordinary test a row credits only the arm drawn, with 1 / p; with shared credit it credits every arm whose own
  # choice is what the row shows, each with 1 over the probability that the row showed it. One estimator serves both.
  #
  #   SkiftetStatistical::SharedCredit.estimate(
  #     [ { credit: { a: 2.0 }, values: [ 1.0, 0.0 ] },
  #       { credit: { a: 1.0, b: 1.0 }, values: [ 0.0, 2.0 ] },
  #       { credit: { b: 4.0 }, values: [ 1.0, 1.0 ] } ],
  #     arms: %i[a b], parts: %i[p q])
  #   # => { means: { a: [ 0.667, 0.667 ], b: [ 0.8, 1.2 ] },
  #   #      covariance: [ [ 0.0988, -0.1975, 0.0356, -0.0356 ], ... ],   # arm-major: (a, p), (a, q), (b, p), (b, q)
  #   #      weight_sums: { a: 3.0, b: 5.0 }, effective: { a: 1.8, b: 1.4706 } }
  #
  # Per arm a and part p the Hajek mean r_ap = sum_i w_ai * x_ip / W_a, with W_a = sum_i w_ai. The covariance of every
  # pair of means is the sandwich sum_i psi_ap(i) * psi_bq(i) over the influence values
  # psi_ap(i) = w_ai * (x_ip - r_ap) / W_a, so two arms that share rows covary. It is laid out arm-major (index
  # arm_index * P + part_index), as LinearValue reads it, computed on one triangle and mirrored, so it is exactly
  # symmetric. Kish's effective size W_a^2 / sum_i w_ai^2 stands beside it. An arm no row credits has nil means, zero
  # rows and columns, a weight sum of 0.0 and an effective size of 0.0.
  #
  # The covariance comes from sufficient sums, kept per arm (W_a and sum_i w_ai * x_ip) and per pair of arms a, b that
  # some row credits together (sum_i w_ai * w_bi, its products with each x_ip and with each x_ip * x_iq). A zero credit
  # and a zero value add nothing and are skipped, so the sums stay sparse. The sums are public, through an
  # Accumulator, for a caller that folds its rows once into a store and never reads them twice; `estimate` is built on
  # it, so the two cannot differ. `from_sums` answers the same estimate for an ordinary test from per-arm sums.
  #
  # An entry of the covariance is four terms that cancel, so its rounding is relative to the terms and not to the
  # answer: where the exact sandwich is 0 (an arm whose credited rows hold one value, as an arm of one row does)
  # the sums leave rounding of either sign, and a negative variance no Cholesky factor takes. An entry within
  # (8 + n) * Float::EPSILON of the sum of its terms' sizes is that rounding and answers exactly 0.0, n being the
  # largest of its two arms' Kish's effective sizes and rows folded. The 8 is the rounding of the products and the
  # rates, which every entry has; the n is the rounding of the sums, which grows with every row they add, in step with
  # the rows and not with their square root when the rows are alike. Every row counts whatever its credit: after a heavy
  # row each light one still rounds at the heavy sum's scale, so one row of credit 100 before a hundred of credit 1 are
  # 101 rows of rounding at a Kish's size of 4. `from_sums` folds no rows, so its n is Kish's size rounded up.
  #
  # The snap clears what falls below the sums' rounding bound, and that bound cannot tell rounding from a real variance
  # as small: at a coefficient of variation near 1e-6 over 1e5 rows a real variance sits under it and reads 0. Rows of
  # small counts are far from that case. A pair (arm, part) whose variance is 0 holds no covariance either (it is the
  # same cancellation), so its row and column answer exactly 0.0 too.
  #
  # An entry the snap keeps still carries its rounding, and where the exact matrix is of lower rank (fewer rows than
  # cells, two parts one offset apart) that rounding can take a pivot below 0 by more than a Cholesky factor allows.
  # Estimate#rounding is the largest bound the snap used, in the covariance's own units: a caller hands it on,
  # `LinearValue.summary(..., rounding: estimate.rounding)`, so a covariance answered on valid rows is not refused for
  # its own rounding.
  #
  # `contrast` gives one difference with its plug-in variance; `contrast_for_fit` gives it with one per-visit variance
  # pooled over the two arms, for a fit that pools contrasts across cells, and leaves out an arm of a small Kish's size.
  # `with_prior` answers the estimate under a prior pooled from its own credited arms, every arm below a strength
  # topped up to it, for drawing where an arm holds only a handful of visits.
  #
  # An estimate is a Hash of exactly those four keys that also names its parts (Estimate#parts) and its rounding
  # (Estimate#rounding), so `linear` and `contrast` read a Hash of weights by name. Read back from JSON it is a plain
  # Hash, which holds neither.
  #
  # Arms and parts are named by Symbols or Strings, each once; the answers name the arms as given. Every name inside a
  # row or a sum is read by Symbol or String. Every number is a finite real number, finite as a Float too (a NaN, an
  # infinity, a Complex, an Integer beyond a Float's range and a non-number are refused alike); every refusal is an
  # ArgumentError.
  #
  # Every result is finite. Where the sums' products leave a Float's range (above Float::MAX, or below Float::MIN, where
  # a Float loses its digits and then reads 0) nothing is answered: a row whose products do not fit is refused before
  # any sum changes, as is a sum that would pass Float::MAX, and `estimate`, `linear` and `contrast` raise ArgumentError
  # rather than answer an infinity, a NaN, or a 0.0 that overflow or underflow made. Credits near 1e-155 or 1e155 are
  # out of range for that reason: they are squared. A caller with such numbers rescales them first.
  module SharedCredit
    class << self
      # The estimate of `rows`, any Enumerable, read once.
      def estimate(rows, arms:, parts:)
        accumulator = Accumulator.new(arms: arms, parts: parts)
        raise ArgumentError, "rows must be an Enumerable of rows, got #{rows.class}" unless rows.respond_to?(:each)

        rows.each { |row| accumulator.add(row) }
        accumulator.estimate
      end

      # An empty Accumulator for these arms and parts.
      def accumulator(arms:, parts:)
        Accumulator.new(arms: arms, parts: parts)
      end

      # The estimate of an ordinary test from per-arm Hajek sums,
      # `{ arm => { w:, w2:, a: { part => }, b: { part => }, c: { [part, part] => } } }`: w = sum 1/p, w2 = sum 1/p^2,
      # a = sum x/p, b = sum x/p^2, c = sum x_p * x_q / p^2. Within an arm r_p = a_p / w and
      # cov(r_p, r_q) = (c_pq - r_p * b_q - r_q * b_p + r_p * r_q * w2) / w^2; across arms 0. These are the sums an
      # Accumulator keeps for rows that each credit one arm, so the answer is that accumulator's estimate.
      def from_sums(sums, arms:, parts:)
        Accumulator.from_sums(sums, arms: arms, parts: parts).estimate
      end

      # Each arm's value V_a = sum_p w_p * r_ap at fixed weights, and the K x K covariance of the values,
      # cov(V_a, V_b) = sum_p sum_q w_p * w_q * cov(r_ap, r_bq), computed on one triangle and mirrored. `weights` holds
      # one finite number per part: an Array in the parts' order, or a Hash by part name (by Symbol or String, in any
      # order, every part once and nothing else). A Hash needs an estimate that names its parts, as `estimate` and
      # `from_sums` answer it; one read back from JSON names none and takes an Array. An arm with nil means has a nil
      # value. A variance that rounding takes below zero answers 0.0. A value or a covariance that overflows raises.
      def linear(estimate, weights:)
        arms, rates, covariance, size = read_estimate(estimate)
        weight = read_weights(weights, size, estimate.is_a?(Estimate) ? estimate.parts : nil)
        count = arms.size
        values = Array.new(count, 0.0)
        result = Array.new(count) { Array.new(count, 0.0) }

        count.times do |first|
          values[first] = rates[first] && Read.result(dot(weight, rates[first]))
          (first...count).each do |second|
            total = 0.0
            size.times do |part|
              row = covariance[(first * size) + part]
              offset = second * size
              inner = 0.0
              size.times { |other| inner += weight[other] * row[offset + other] }
              total += weight[part] * inner
            end
            Read.result(total)
            total = 0.0 if first == second && total.negative?
            result[first][second] = total
            result[second][first] = total
          end
        end

        { means: arms.zip(values).to_h, covariance: result }
      end

      # One difference V_first - V_second at fixed weights (as `linear` reads them) with its variance
      # var(V_first) + var(V_second) - 2 * cov(V_first, V_second); both nil when either arm has nil means. A variance
      # that rounding takes below zero answers 0.0. One that overflows raises.
      def contrast(estimate, first, second, weights:)
        linear = linear(estimate, weights: weights)
        names = linear[:means].keys
        i = arm_index(names, first)
        j = arm_index(names, second)
        mean_first = linear[:means][names[i]]
        mean_second = linear[:means][names[j]]
        return { mean: nil, variance: nil } if mean_first.nil? || mean_second.nil?

        covariance = linear[:covariance]
        variance = Read.result(covariance[i][i] + covariance[j][j] - (2.0 * covariance[i][j]))
        { mean: Read.result(mean_first - mean_second), variance: variance.negative? ? 0.0 : variance }
      end

      # One difference V_arm - V_reference at fixed weights (as `linear` reads them), for a fit that pools contrasts
      # across cells, with the null-homoscedastic sandwich variance: one per-visit variance of the weighted part value,
      # pooled over both arms, times each arm's sum of squared normalised weights. `contrast` stays as it is.
      #
      # Why: an arm held at a small floor share has a small Kish's size, and its plug-in variance moves with its own
      # estimate (a rare 0/1 part that the arm has not yet seen reads a variance near 0), so `contrast`'s z is too wide
      # under an adaptive allocation, and a test across cells rejects too often. Pooled, the per-visit variance comes
      # mostly from the larger arm, and the small arm's estimate barely moves it.
      #
      # With u_ai = w_ai / W_a, the sandwich variance of V_a is v_a = sum_i u_ai^2 * (y_i - V_a)^2, y_i = sum_p
      # weight_p * x_ip, and sum_i u_ai^2 = 1 / n_a, n_a Kish's size (the estimate's `effective`). So the arm's per-visit variance is
      # s_a^2 = v_a * n_a, the pooled one s^2 = (n_a * s_a^2 + n_r * s_r^2) / (n_a + n_r), each arm weighing as its
      # Kish's size, and the variance s^2 * (1 / n_a + 1 / n_r). Where the two per-visit variances agree it is `contrast`'s.
      #
      # Answers { mean:, variance: }, or nil when either arm has nil means or a Kish's size under `minimum_effective`
      # (a finite number at least 0): that contrast is then left out of the fit. An arm against itself answers
      # { mean: 0.0, variance: 0.0 }, as the sandwich s^2 * sum_i (u_ai - u_ai)^2 says. The estimate holds no sum of the
      # credit two arms share, so rows that credit both are read as if apart: the sandwich's term -2 * s^2 * sum_i u_ai *
      # u_ri, which only lowers the variance (a shared row moves both arms alike), is left out. The estimate needs
      # `effective` beside its means and covariance (every estimate this module answers has it, read back from JSON
      # too). A variance that overflows raises.
      def contrast_for_fit(estimate, arm, reference, weights:, minimum_effective: 20)
        minimum = Read.weight(minimum_effective, "minimum_effective")
        linear = linear(estimate, weights: weights)
        names = linear[:means].keys
        i = arm_index(names, arm)
        j = arm_index(names, reference)
        sizes = read_effective(estimate, names)
        mean_arm = linear[:means][names[i]]
        mean_reference = linear[:means][names[j]]
        return nil if mean_arm.nil? || mean_reference.nil?

        size_arm = sizes[i]
        size_reference = sizes[j]
        return nil if size_arm < minimum || size_reference < minimum || !size_arm.positive? || !size_reference.positive?
        return { mean: 0.0, variance: 0.0 } if i == j

        covariance = linear[:covariance]
        per_visit_arm = covariance[i][i] * size_arm
        per_visit_reference = covariance[j][j] * size_reference
        pooled = Read.result(((size_arm * per_visit_arm) + (size_reference * per_visit_reference)) / (size_arm + size_reference))
        variance = Read.result(pooled * ((1.0 / size_arm) + (1.0 / size_reference)))
        { mean: Read.result(mean_arm - mean_reference), variance: variance }
      end

      # The estimate under a prior pooled from its own credited arms, each arm's evidence topped up to `strength`
      # visits, for drawing (LinearValue) where an arm holds only a handful of visits.
      #
      #   estimate = SkiftetStatistical::SharedCredit.estimate(rows, arms: %i[a b], parts: %i[converted])
      #   SkiftetStatistical::SharedCredit.with_prior(estimate, strength: 20)
      #
      # Why: the sandwich of an arm of one visit, or of visits that all hold one value, is exactly 0, so every draw
      # takes that arm's means as certain, and an arm of one visit against an arm of three reads P(best) near 0.99 or
      # near 0.01 by the order of their values alone.
      #
      # The prior, per pair of parts p and q, from the credited arms a with their Kish's sizes n_a (N their sum), their
      # means r_a and their covariance blocks: its mean m0_p = sum_a n_a * r_ap / N, and its per-visit covariance
      # V0_pq = sum_a n_a * (n_a * cov(r_ap, r_aq) + g_ap * g_aq) / N with g_a = r_a - m0: each arm's per-visit
      # covariance (its sandwich times its Kish's size, as `contrast_for_fit` reads it) and its distance from the
      # mean, so a spread between the arms widens the prior. A part every credited arm holds at one mean has that mean
      # as m0_p, exactly. An arm of Kish's size n below the strength k takes e = k - n visits of the prior, the Normal
      # mean's rule under e visits at m0 and a per-visit covariance V0, taken jointly over its parts:
      #   means (e * m0 + n * r_a) / k,
      #   covariance block (n / k)^2 * cov(r_a) + (e / k) * (V0 + (n / k) * g_a * g_a^T) / k,
      # and every covariance with another arm b scaled by n / k (and by n_b / k when b is topped up too), as the shrunk
      # means are. An arm at or above the strength, and an arm with no credit, keeps its means and its own covariance
      # block exactly, bit for bit; where every credited arm is at or above it, the answer holds the given means and
      # covariance, so every draw LinearValue takes from it is the draw it takes from the estimate. In an ordinary test
      # (no row credits two arms) an arm at or above the strength also draws the same values in every draw.
      #
      # Takes: an estimate in `estimate`'s shape (an Estimate, or a plain Hash such as one read back from JSON), read by
      # Symbol or String, with its weight sums and Kish's sizes; `strength` a finite real number at least 0, in visits
      # (0 tops up nothing).
      #
      # Answers an Estimate: the same arms in the same order, parts (none when the estimate is read back from JSON),
      # weight sums and Kish's sizes (the evidence's own, not the prior's: a contrast for a fit reads the given estimate,
      # not this one), its means and covariance under the prior, and its rounding the given one's (0.0 for a plain Hash)
      # or, where larger, the rounding of an entry it changed.
      #
      # Guarantees (exact: plain arithmetic in a fixed order, no draw and no Math function, so the same estimate and
      # strength answer the same Floats everywhere): the rule above, the covariance exactly symmetric wherever the given
      # one is (as every estimate `estimate` answers is); an arm at or above the strength, and an arm with no credit,
      # keeps its means and covariance block bit for bit, and with every credited arm at or above it the answer holds
      # the given means, covariance and rounding, so every seeded draw and summary LinearValue takes from it is the one
      # it takes from the estimate; a part every credited arm holds at one mean keeps that mean exactly; the given
      # estimate is not changed.
      #
      # Refuses, as an ArgumentError: a strength below 0 or not a finite real number; an estimate that is not a Hash, or
      # whose means, covariance, weight sums or Kish's sizes are missing or malformed (the readers `linear` and
      # `contrast_for_fit` use, with their words); an arm with means but a Kish's size of 0; and an estimate whose
      # figures rise past a Float's largest value under the prior ("the sums leave a Float's range"). Figures so small
      # that the prior's products fall below Float::MIN read as 0 and are not refused; no estimate `estimate` answers
      # holds them (it refuses such rows itself).
      def with_prior(estimate, strength:)
        visits = Read.weight(strength, "strength")
        arms, rates, covariance, size = read_estimate(estimate)
        sizes = read_effective(estimate, arms)
        weight_sums = Read.lookup(estimate, :weight_sums) { raise ArgumentError, "the estimate has no weight sums" }
        raise ArgumentError, "the estimate's weight sums must be a Hash of arm => sum, got #{weight_sums.class}" unless weight_sums.is_a?(Hash)

        credited = arms.each_index.select { |arm| rates[arm] }
        credited.each do |arm|
          raise ArgumentError, "arm #{arms[arm].inspect} has means but a Kish's size of 0" unless sizes[arm].positive?
        end
        given = estimate.is_a?(Estimate) ? estimate.rounding : 0.0
        topped = credited.select { |arm| sizes[arm] < visits }
        means, covariance, rounding =
          if topped.empty? then [ rates, covariance, given ]
          else TopUp.new(rates, covariance, size, sizes, credited, visits, given).answer
          end

        Estimate.new(estimate.is_a?(Estimate) ? estimate.parts : nil, {
          means: arms.zip(means).to_h,
          covariance: covariance,
          weight_sums: weight_sums.dup,
          effective: Read.lookup(estimate, :effective) { {} }.dup
        }, rounding)
      end

      private

      # Kish's size per arm of an estimate, in the order of `names` (its means' arms), read by Symbol or String: every
      # arm once, nothing else, each a finite number at least 0.
      def read_effective(estimate, names)
        effective = Read.lookup(estimate, :effective) { raise ArgumentError, "the estimate has no effective sizes" }
        raise ArgumentError, "the estimate's effective sizes must be a Hash of arm => size, got #{effective.class}" unless effective.is_a?(Hash)

        given = Read.keyed(effective, names.each_with_index.to_h { |name, index| [ name.to_s, index ] }, "the estimate's effective sizes").to_h
        names.each_with_index.map do |name, index|
          value = given.fetch(index) { raise ArgumentError, "the estimate has no effective size for arm #{name.inspect}" }
          Read.weight(value, "the effective size of arm #{name.inspect}")
        end
      end

      # [arms, rates (nil or P Floats per arm), covariance (Floats), P] of an estimate's means and covariance (read by
      # Symbol or String key; its other keys are not read).
      def read_estimate(estimate)
        raise ArgumentError, "an estimate must be a Hash { means:, covariance: }, got #{estimate.class}" unless estimate.is_a?(Hash)

        means = Read.lookup(estimate, :means) { raise ArgumentError, "the estimate has no means" }
        covariance = Read.lookup(estimate, :covariance) { raise ArgumentError, "the estimate has no covariance" }
        raise ArgumentError, "the estimate's means must be a Hash of arm => means, got #{means.class}" unless means.is_a?(Hash)

        arms = Read.names(means.keys, "the estimate's arms")
        size = covariance.is_a?(Array) && !covariance.empty? && (covariance.size % arms.size).zero? ? covariance.size / arms.size : 0
        raise ArgumentError, "the estimate's covariance must be a multiple of #{arms.size} rows, arm-major" if size.zero?

        rows = covariance.each_with_index.map do |row, i|
          raise ArgumentError, "covariance row #{i} must be #{covariance.size} numbers" unless row.is_a?(Array) && row.size == covariance.size

          row.each_with_index.map { |value, j| Read.number(value, "covariance entry (#{i}, #{j})") }
        end
        rates = arms.map do |arm|
          row = means[arm]
          next nil if row.nil?
          raise ArgumentError, "the means of arm #{arm.inspect} must be #{size} numbers" unless row.is_a?(Array) && row.size == size

          row.map { |value| Read.number(value, "a mean of arm #{arm.inspect}") }
        end
        [ arms, rates, rows, size ]
      end

      def read_weights(weights, size, parts)
        case weights
        when Array
          raise ArgumentError, "weights must hold #{size} numbers, one per part, got #{weights.size}" unless weights.size == size

          weights.each_with_index.map { |value, index| Read.number(value, "weight #{index}") }
        when Hash then named_weights(weights, size, parts)
        else raise ArgumentError, "weights must be an Array or a Hash of one number per part, got #{weights.class}"
        end
      end

      # The weights of a Hash in the parts' order: every part named once, nothing else.
      def named_weights(weights, size, parts)
        unless parts && parts.size == size
          raise ArgumentError, "a Hash of weights needs an estimate that names its #{size} parts (one read back from JSON " \
                               "names none): give an Array in the parts' order"
        end

        given = Read.keyed(weights, parts.each_with_index.to_h { |part, index| [ part.to_s, index ] }, "weights").to_h
        parts.each_with_index.map do |part, index|
          value = given.fetch(index) { raise ArgumentError, "weights have no weight for part #{part.inspect}" }
          Read.number(value, "the weight of part #{part.inspect}")
        end
      end

      def dot(weight, rates)
        total = 0.0
        weight.each_index { |part| total += weight[part] * rates[part] }
        total
      end

      def arm_index(names, arm)
        index = (arm.is_a?(Symbol) || arm.is_a?(String)) && names.index { |name| name.to_s == arm.to_s }
        raise ArgumentError, "the estimate names no arm #{arm.inspect}" unless index

        index
      end
    end

    # The reading every part of this module shares.
    module Read
      module_function

      # The names as given, checked: an Array of one or more Symbols or Strings, each named once.
      def names(list, label)
        raise ArgumentError, "#{label} must be an Array, got #{list.class}" unless list.is_a?(Array)
        raise ArgumentError, "#{label} must name at least one" if list.empty?

        list.each do |name|
          raise ArgumentError, "#{label} must be Symbols or Strings, got #{name.inspect}" unless name.is_a?(Symbol) || name.is_a?(String)
        end
        raise ArgumentError, "#{label} must each be named once" unless list.map(&:to_s).uniq.size == list.size

        list.dup.freeze
      end

      # A finite real number, finite as a Float too (an Integer or Rational beyond a Float's range is not), as a Float.
      def number(value, label)
        number = value.to_f if value.is_a?(Numeric) && value.real? && value.finite?
        return number if number&.finite?

        raise ArgumentError, "#{label} must be a finite number, got #{value.inspect}"
      end

      def weight(value, label)
        number = number(value, label)
        raise ArgumentError, "#{label} must be at least 0, got #{value.inspect}" if number.negative?

        number
      end

      # A sum or a result, which must be finite.
      def result(value)
        return value if value.finite?

        raise ArgumentError, "the sums leave a Float's range (#{value}): rescale the credits or the values"
      end

      # first * second, exactly 0.0 only when a factor is 0: a product that leaves a Float's range either way is
      # refused.
      def times(first, second)
        product = first * second
        return product if product.abs.between?(Float::MIN, Float::MAX) || first.zero? || second.zero?

        raise ArgumentError, "the sums leave a Float's range (#{first} * #{second}): rescale the credits or the values"
      end

      # value / divisor, read as `times` reads a product. The divisor is above 0.
      def over(value, divisor)
        quotient = value / divisor
        return quotient if quotient.abs.between?(Float::MIN, Float::MAX) || value.zero?

        raise ArgumentError, "the sums leave a Float's range (#{value} / #{divisor}): rescale the credits or the values"
      end

      # [[index, value]] for every key of `hash`, each a Symbol or String naming one entry of `index` (name => index)
      # once.
      def keyed(hash, index, label)
        raise ArgumentError, "#{label} must be a Hash, got #{hash.class}" unless hash.is_a?(Hash)

        seen = {}
        hash.map do |key, value|
          position = index[key.to_s] if key.is_a?(Symbol) || key.is_a?(String)
          raise ArgumentError, "#{label} names #{key.inspect}, which is none of #{index.keys.inspect}" if position.nil?
          raise ArgumentError, "#{label} names #{key.inspect} twice" if seen.key?(position)

          seen[position] = true
          [ position, value ]
        end
      end

      # { name => value } of a Hash whose keys are the names `allowed` (by Symbol or String), each at most once, with
      # every one of `required`.
      def fields(hash, allowed, required, label)
        found = keyed(hash, allowed.each_with_index.to_h, label).to_h { |position, value| [ allowed[position], value ] }
        missing = required - found.keys
        raise ArgumentError, "#{label} has no #{missing.join(', ')}" unless missing.empty?

        found
      end

      def lookup(hash, name)
        return hash[name] if hash.key?(name)

        other = name.is_a?(Symbol) ? name.to_s : name.to_sym
        return hash[other] if hash.key?(other)

        yield
      end
    end
    private_constant :Read

    # SharedCredit.with_prior's arithmetic on a read estimate: the prior pooled from the credited arms and every arm
    # below the strength topped up to it. Rates are per arm (nil for an arm with no credit), the covariance arm-major,
    # sizes the arms' Kish's sizes, `credited` the arms with rates, `visits` the strength and `given` the estimate's
    # rounding.
    class TopUp
      # The units of Float::EPSILON a changed entry may hold as the rounding of its own products and quotients, before
      # one more per arm summed into the prior.
      CANCELLATION = 8

      def initialize(rates, covariance, size, sizes, credited, visits, given)
        @rates = rates
        @covariance = covariance
        @size = size
        @sizes = sizes
        @credited = credited
        @visits = visits
        @given = given
        @total = credited.sum(0.0) { |arm| sizes[arm] }
        @centre = Array.new(size) { |part| centre(part) }
        @gaps = rates.map { |row| row && row.each_index.map { |part| Read.result(row[part] - @centre[part]) } }
        @shares = sizes.each_index.map { |arm| rates[arm] && sizes[arm] < visits ? sizes[arm] / visits : 1.0 }
      end

      # [means, covariance, rounding] under the prior.
      def answer
        spread, spread_magnitude = prior_covariance
        inherited = @credited.sum(0.0) { |arm| @sizes[arm] * @sizes[arm] } / @total * @given
        means = @rates.each_index.map { |arm| topped?(arm) ? shrunk(arm) : @rates[arm] }
        covariance = @covariance.map(&:dup)
        rounding = @given
        cells = @rates.size * @size
        cells.times do |row|
          first, part = row.divmod(@size)
          next unless @rates[first]

          (row...cells).each do |column|
            second, other = column.divmod(@size)
            next unless @rates[second] && (topped?(first) || topped?(second))

            value, bound = entry(first, second, part, other, @covariance[row][column], spread, spread_magnitude, inherited)
            covariance[row][column] = value
            covariance[column][row] = value
            rounding = [ rounding, bound ].max
          end
        end
        [ means, covariance, Read.result(rounding) ]
      end

      private

      def topped?(arm)
        @shares[arm] < 1.0
      end

      # m0_p, the visit-weighted mean of the credited arms; the one mean they all hold, exactly, where they agree.
      def centre(part)
        values = @credited.map { |arm| @rates[arm][part] }
        return values.first if values.uniq.size == 1

        Read.result(@credited.sum(0.0) { |arm| @sizes[arm] * @rates[arm][part] } / @total)
      end

      # [V0, the magnitudes of V0's terms]: per pair of parts, sum_a n_a * (n_a * cov(r_ap, r_aq) + g_ap * g_aq) / N,
      # on one triangle and mirrored.
      def prior_covariance
        spread = Array.new(@size) { Array.new(@size, 0.0) }
        magnitudes = Array.new(@size) { Array.new(@size, 0.0) }
        @size.times do |part|
          (part...@size).each do |other|
            value = 0.0
            magnitude = 0.0
            @credited.each do |arm|
              own = @sizes[arm] * @covariance[(arm * @size) + part][(arm * @size) + other]
              gap = @gaps[arm][part] * @gaps[arm][other]
              value += @sizes[arm] * (own + gap)
              magnitude += @sizes[arm] * (own.abs + gap.abs)
            end
            spread[part][other] = spread[other][part] = Read.result(value / @total)
            magnitudes[part][other] = magnitudes[other][part] = Read.result(magnitude / @total)
          end
        end
        [ spread, magnitudes ]
      end

      # (e * m0 + n * r_a) / k, held between the two means it weighs as they are given, so it never rounds past either
      # and a rate at the prior's mean keeps it exactly.
      def shrunk(arm)
        extra = @visits - @sizes[arm]
        @rates[arm].each_index.map do |part|
          rate = @rates[arm][part]
          mean = Read.result(((extra * @centre[part]) + (@sizes[arm] * rate)) / @visits)
          mean.clamp(*[ @centre[part], rate ].minmax)
        end
      end

      # [the entry under the prior, its rounding bound]: the given entry scaled by both arms' n / k, and within a
      # topped-up arm's block (e / k) * (V0_pq + (n / k) * g_p * g_q) / k added.
      def entry(first, second, part, other, given_entry, spread, spread_magnitude, inherited)
        scale = @shares[first] * @shares[second]
        scaled = scale * given_entry
        value = scaled
        terms = scaled.abs
        bound = scale * @given
        if first == second
          extra = (@visits - @sizes[first]) / @visits
          gap = @shares[first] * @gaps[first][part] * @gaps[first][other]
          value += extra * (spread[part][other] + gap) / @visits
          terms += extra * (spread_magnitude[part][other] + gap.abs) / @visits
          bound += extra * inherited / @visits
        end
        [ Read.result(value), Read.result(bound + ((CANCELLATION + @rates.size) * Float::EPSILON * terms)) ]
      end
    end
    private_constant :TopUp

    # An estimate: a Hash of exactly means, covariance, weight_sums and effective, which also names the parts its means
    # and covariance are laid out by, and holds `rounding`: the largest bound the snap used, in the covariance's units,
    # how far an entry of the covariance may be off. Its keys and its equality are the Hash's.
    class Estimate < Hash
      attr_reader :parts, :rounding

      def initialize(parts, fields, rounding = 0.0)
        super()
        @parts = parts
        @rounding = rounding
        update(fields)
      end
    end

    # The sufficient sums of a set of rows, folded one row at a time (#add) or two sets at once (#merge), stored as a
    # JSON-safe Hash (#to_h, .from_h) and turned into the estimate (#estimate) at any time. A row is read once and never
    # kept.
    #
    # `to_h` is the sparse sums, every name as a String and every number a Float, nothing kept that is 0:
    #   { "format" => 1, "arms" => [ arm, ... ], "parts" => [ part, ... ],
    #     "sums" => { arm => { "w" => sum w, "x" => { part => sum w * x } } },
    #     "pairs" => { arm => { arm2 => { "w" => sum w * w2, "x" => { part => sum w * w2 * x },
    #                                     "xx" => { part => { part2 => sum w * w2 * x * x2 } } } } } }
    # with an arm under "sums" for every arm some row credits, a pair under "pairs" for every two arms some row credits
    # together (an arm with itself included), the first arm of a pair and the first part of a product never later in
    # the arms' or the parts' order than the second. An arm's sums also hold "rows" => the rows that credited it, kept
    # only where the sums do not say it already: without the key the rows are Kish's size rounded up, as they are where
    # the credits are near alike. So the snap's n survives a store, and a heavy row's arm keeps its count.
    class Accumulator
      FORMAT = 1
      # Two orders of one pair of parts handed to from_sums may differ by rounding, no more.
      MIRROR = 1e-12
      # The units of Float::EPSILON a covariance entry may hold as the rounding of its products and rates, before the
      # rounding of its sums (one unit per row of Kish's size) is added.
      CANCELLATION = 8

      # One pair of arms' sums: the summed products of their credits, those times each value (one per part), and those
      # times each two values (the upper triangle of parts by parts).
      Pair = Struct.new(:weight, :values, :squares)
      private_constant :Pair

      attr_reader :arms, :parts

      class << self
        # The accumulator `to_h` gave, its arms and parts named by Symbols.
        def from_h(hash)
          raise ArgumentError, "an accumulator's hash must be a Hash, got #{hash.class}" unless hash.is_a?(Hash)

          fields = Read.fields(hash, %w[format arms parts sums pairs], %w[format arms parts sums pairs], "the accumulator's hash")
          format = fields["format"]
          raise ArgumentError, "the accumulator's format must be #{FORMAT}, got #{format.inspect}" unless format.is_a?(Integer) && format == FORMAT

          arms = Read.names(fields["arms"], "the accumulator's arms").map(&:to_sym)
          parts = Read.names(fields["parts"], "the accumulator's parts").map(&:to_sym)
          accumulator = new(arms: arms, parts: parts)
          accumulator.send(:restore_h, fields["sums"], fields["pairs"])
          accumulator
        end

        # The accumulator of an ordinary test's per-arm sums (SharedCredit.from_sums reads them): an arm left out of
        # `sums`, or whose w is 0 with every other sum 0, has no credit; every other arm needs w2 above 0, every part in
        # a and b, and every pair of parts in c in either order (both orders given must agree to rounding).
        def from_sums(sums, arms:, parts:)
          accumulator = new(arms: arms, parts: parts)
          accumulator.send(:restore_sums, sums)
          accumulator
        end
      end

      def initialize(arms:, parts:)
        @arms = Read.names(arms, "arms")
        @parts = Read.names(parts, "parts")
        @arm_index = @arms.each_with_index.to_h { |arm, index| [ arm.to_s, index ] }.freeze
        @part_index = @parts.each_with_index.to_h { |part, index| [ part.to_s, index ] }.freeze
        @weights = Array.new(@arms.size, 0.0)
        @rows = Array.new(@arms.size, 0.0)
        @values = Array.new(@arms.size) { Array.new(@parts.size, 0.0) }
        @pairs = {}
      end

      # Folds one row in and answers the accumulator. The whole row is checked before any sum changes.
      def add(row)
        credited, present = read_row(row)
        check_fold(credited, present)
        credited.each do |arm, weight|
          @weights[arm] += weight
          @rows[arm] += 1.0
          sums = @values[arm]
          present.each { |part, value| sums[part] += weight * value }
        end
        credited.each_with_index do |(first, first_weight), position|
          credited[position..].each do |second, second_weight|
            fold(pair(first, second), first_weight * second_weight, present)
          end
        end
        self
      end

      # A new accumulator holding both sets of sums; neither is changed. Both must name the same arms and parts in the
      # same order (by name); the new one names them as this one does.
      def merge(other)
        raise ArgumentError, "an accumulator merges only another Accumulator, got #{other.class}" unless other.is_a?(Accumulator)
        unless other.arms.map(&:to_s) == @arms.map(&:to_s) && other.parts.map(&:to_s) == @parts.map(&:to_s)
          raise ArgumentError, "an accumulator merges only one of the same arms and parts, in the same order"
        end

        merged = Accumulator.new(arms: @arms, parts: @parts)
        merged.absorb(self)
        merged.absorb(other)
        merged
      end

      def to_h
        sums = {}
        @arms.each_with_index do |arm, index|
          next unless @weights[index].positive?

          sums[arm.to_s] = { "w" => @weights[index], "x" => sparse(@values[index]) }
          sums[arm.to_s]["rows"] = @rows[index] unless @rows[index] == implied_rows(index)
        end
        pairs = {}
        @pairs.keys.sort.each do |first, second|
          pair = @pairs[[ first, second ]]
          squares = {}
          @parts.each_index do |part|
            row = sparse(pair.squares[part], part)
            squares[@parts[part].to_s] = row unless row.empty?
          end
          (pairs[@arms[first].to_s] ||= {})[@arms[second].to_s] =
            { "w" => pair.weight, "x" => sparse(pair.values), "xx" => squares }
        end
        { "format" => FORMAT, "arms" => @arms.map(&:to_s), "parts" => @parts.map(&:to_s), "sums" => sums, "pairs" => pairs }
      end

      # { means:, covariance:, weight_sums:, effective: } of the rows folded so far, the arms as given.
      def estimate
        size = @parts.size
        rates = @weights.each_index.map do |arm|
          weight = @weights[arm]
          weight.positive? ? @values[arm].map { |sum| Read.over(sum, weight) } : nil
        end
        effective = @weights.each_index.map do |arm|
          weight = @weights[arm]
          weight.positive? ? Read.over(Read.times(weight, weight), @pairs[[ arm, arm ]].weight) : 0.0
        end
        cells = @arms.size * size
        covariance = Array.new(cells) { Array.new(cells, 0.0) }
        rounding = 0.0
        cells.times do |row|
          first, part = row.divmod(size)
          next unless rates[first]

          (row...cells).each do |column|
            second, other = column.divmod(size)
            pair = @pairs[[ first, second ]]
            next unless pair

            rows = [ effective[first], effective[second], @rows[first], @rows[second] ].max
            scale = @weights[first] * @weights[second]
            sum, bound = entry(pair, rates[first][part], rates[second][other], part, other, rows)
            value = Read.over(sum, scale)
            covariance[row][column] = value
            covariance[column][row] = value
            rounding = [ rounding, Read.result(bound / scale) ].max
          end
        end

        clear_constant(covariance)
        Estimate.new(@parts, {
          means: @arms.zip(rates).to_h,
          covariance: covariance,
          weight_sums: @arms.zip(@weights).to_h,
          effective: @arms.zip(effective).to_h
        }, rounding)
      end

      protected

      attr_reader :weights, :rows, :values, :pairs

      def absorb(other)
        other.weights.each_with_index { |weight, arm| @weights[arm] = Read.result(@weights[arm] + weight) }
        other.rows.each_with_index { |count, arm| @rows[arm] += count }
        other.values.each_with_index do |sums, arm|
          mine = @values[arm]
          sums.each_with_index { |sum, part| mine[part] = Read.result(mine[part] + sum) }
        end
        other.pairs.each do |(first, second), source|
          target = pair(first, second)
          target.weight = Read.result(target.weight + source.weight)
          source.values.each_with_index { |sum, part| target.values[part] = Read.result(target.values[part] + sum) }
          source.squares.each_with_index do |row, part|
            mine = target.squares[part]
            (part...@parts.size).each { |other_part| mine[other_part] = Read.result(mine[other_part] + row[other_part]) }
          end
        end
      end

      private

      # [credited, present]: the arms the row credits above 0, [[arm index, weight]] in the arms' order, and its values
      # that are not 0, [[part index, value]] in the parts' order.
      def read_row(row)
        raise ArgumentError, "a row must be a Hash { credit:, values: }, got #{row.class}" unless row.is_a?(Hash)

        fields = Read.fields(row, %w[credit values], %w[credit values], "a row")
        credited = Read.keyed(fields["credit"], @arm_index, "a row's credit").filter_map do |arm, weight|
          weight = Read.weight(weight, "the credit of arm #{@arms[arm].inspect}")
          [ arm, weight ] if weight.positive?
        end
        values = fields["values"]
        unless values.is_a?(Array) && values.size == @parts.size
          raise ArgumentError, "a row's values must be an Array of #{@parts.size} numbers, one per part"
        end

        present = values.each_with_index.filter_map do |value, part|
          value = Read.number(value, "the value of part #{@parts[part].inspect}")
          [ part, value ] unless value.zero?
        end
        [ credited.sort_by(&:first), present ]
      end

      # Refuses a row one of whose products leaves a Float's range, or that would take a sum past Float::MAX, while no
      # sum has moved: `add` and `fold` then take the same products.
      def check_fold(credited, present)
        credited.each do |arm, weight|
          Read.result(@weights[arm] + weight)
          present.each { |part, value| Read.result(@values[arm][part] + Read.times(weight, value)) }
        end
        credited.each_with_index do |(first, first_weight), position|
          credited[position..].each do |second, second_weight|
            check_pair(@pairs[[ first, second ]], Read.times(first_weight, second_weight), present)
          end
        end
      end

      def check_pair(pair, product, present)
        Read.result((pair ? pair.weight : 0.0) + product)
        present.each_with_index do |(part, value), position|
          once = Read.times(product, value)
          Read.result((pair ? pair.values[part] : 0.0) + once)
          present[position..].each do |other, other_value|
            Read.result((pair ? pair.squares[part][other] : 0.0) + Read.times(once, other_value))
          end
        end
      end

      def pair(first, second)
        @pairs[[ first, second ]] ||= Pair.new(0.0, Array.new(@parts.size, 0.0), Array.new(@parts.size) { Array.new(@parts.size, 0.0) })
      end

      def fold(pair, product, present)
        pair.weight += product
        present.each_with_index do |(part, value), position|
          pair.values[part] += product * value
          squares = pair.squares[part]
          present[position..].each { |other, other_value| squares[other] += product * value * other_value }
        end
      end

      # [sum_i w_ai * w_bi * (x_ip - r_ap) * (x_iq - r_bq), from the pair's sums, exactly 0.0 within the rounding of its
      # four terms over `rows` rows; that bound]. Terms too large to sum make the bound an infinity, which `estimate`
      # refuses, so an entry they cleared is never answered.
      def entry(pair, first_rate, second_rate, part, other, rows)
        square = part <= other ? pair.squares[part][other] : pair.squares[other][part]
        first = Read.times(first_rate, pair.values[other])
        second = Read.times(second_rate, pair.values[part])
        both = Read.times(Read.times(first_rate, second_rate), pair.weight)
        value = square - first - second + both
        bound = (CANCELLATION + rows) * Float::EPSILON * (square.abs + first.abs + second.abs + both.abs)
        [ value.abs <= bound ? 0.0 : value, bound ]
      end

      # A (arm, part) of variance 0 holds no covariance: what its row still holds is rounding, and a row of rounding
      # under a zero pivot is what Cholesky refuses.
      def clear_constant(covariance)
        covariance.each_index do |cell|
          next unless covariance[cell][cell].zero?

          covariance.each_index do |other|
            covariance[cell][other] = 0.0
            covariance[other][cell] = 0.0
          end
        end
      end

      # What an arm's sums say of its rows: Kish's size rounded up, nil where the size leaves a Float's range.
      def implied_rows(arm)
        weight = @weights[arm]
        return 0.0 unless weight.positive?

        size = weight * weight / @pairs[[ arm, arm ]].weight
        size.ceil.to_f if size.finite?
      end

      # { part => sum } of the sums that are not 0, from part `from` on.
      def sparse(sums, from = 0)
        (from...@parts.size).each_with_object({}) do |part, kept|
          kept[@parts[part].to_s] = sums[part] unless sums[part].zero?
        end
      end

      def restore_h(sums, pairs)
        weights = Array.new(@arms.size, 0.0)
        values = Array.new(@arms.size) { Array.new(@parts.size, 0.0) }
        rows = {}
        Read.keyed(sums, @arm_index, "the accumulator's sums").each do |arm, entry|
          label = "the sums of arm #{@arms[arm].inspect}"
          fields = Read.fields(entry, %w[w x rows], %w[w x], label)
          weights[arm] = Read.weight(fields["w"], "#{label}: w")
          rows[arm] = Read.weight(fields["rows"], "#{label}: rows") if fields.key?("rows")
          read_parts(fields["x"], "#{label}: x").each { |part, value| values[arm][part] = value }
        end

        restored = {}
        Read.keyed(pairs, @arm_index, "the accumulator's pairs").each do |first, inner|
          Read.keyed(inner, @arm_index, "the pairs of arm #{@arms[first].inspect}").each do |second, entry|
            label = "the pair of arms #{@arms[first].inspect} and #{@arms[second].inspect}"
            raise ArgumentError, "#{label} must be named first by the earlier arm" if second < first

            restored[[ first, second ]] = read_pair(entry, label)
          end
        end
        restore(weights, values, restored, rows)
      end

      def read_pair(entry, label)
        fields = Read.fields(entry, %w[w x xx], %w[w x xx], label)
        pair = Pair.new(Read.weight(fields["w"], "#{label}: w"), Array.new(@parts.size, 0.0),
                        Array.new(@parts.size) { Array.new(@parts.size, 0.0) })
        read_parts(fields["x"], "#{label}: x").each { |part, value| pair.values[part] = value }
        Read.keyed(fields["xx"], @part_index, "#{label}: xx").each do |part, row|
          read_parts(row, "#{label}: xx of part #{@parts[part].inspect}").each do |other, value|
            raise ArgumentError, "#{label}: xx must name each product first by the earlier part" if other < part

            pair.squares[part][other] = value
          end
        end
        pair
      end

      def read_parts(hash, label)
        Read.keyed(hash, @part_index, label).map { |part, value| [ part, Read.number(value, "#{label} of part #{@parts[part].inspect}") ] }
      end

      def restore_sums(sums)
        weights = Array.new(@arms.size, 0.0)
        values = Array.new(@arms.size) { Array.new(@parts.size, 0.0) }
        pairs = {}
        Read.keyed(sums, @arm_index, "sums").each do |arm, entry|
          label = "the sums of arm #{@arms[arm].inspect}"
          fields = Read.fields(entry, %w[w w2 a b c], %w[w w2], label)
          weight = Read.weight(fields["w"], "#{label}: w")
          squared = Read.weight(fields["w2"], "#{label}: w2")
          if weight.zero?
            given = [ squared ]
            given.concat(read_parts(fields["a"], "#{label}: a").map(&:last)) if fields.key?("a")
            given.concat(read_parts(fields["b"], "#{label}: b").map(&:last)) if fields.key?("b")
            given.concat(read_products(fields["c"], label).values) if fields.key?("c")
            raise ArgumentError, "#{label}: w is 0, so every other sum must be 0" unless given.all?(&:zero?)

            next
          end
          raise ArgumentError, "#{label}: w2 must be above 0 where w is" unless squared.positive?

          missing = %w[a b c] - fields.keys
          raise ArgumentError, "#{label} has no #{missing.join(', ')}" unless missing.empty?

          weights[arm] = weight
          values[arm] = complete_parts(fields["a"], "#{label}: a")
          pairs[[ arm, arm ]] = Pair.new(squared, complete_parts(fields["b"], "#{label}: b"), triangle(fields["c"], label))
        end
        restore(weights, values, pairs)
      end

      def complete_parts(hash, label)
        given = read_parts(hash, label).to_h
        @parts.each_index.map do |part|
          given.fetch(part) { raise ArgumentError, "#{label} has no sum for part #{@parts[part].inspect}" }
        end
      end

      # { [part, part] => sum } of c, its keys read as pairs of part names; an ordered pair at most once.
      def read_products(hash, label)
        raise ArgumentError, "#{label}: c must be a Hash of [part, part] => sum, got #{hash.class}" unless hash.is_a?(Hash)

        hash.each_with_object({}) do |(key, value), read|
          unless key.is_a?(Array) && key.size == 2 && key.all? { |name| (name.is_a?(Symbol) || name.is_a?(String)) && @part_index.key?(name.to_s) }
            raise ArgumentError, "#{label}: c's key #{key.inspect} must be a pair of the parts' names"
          end

          ordered = key.map { |name| @part_index[name.to_s] }
          raise ArgumentError, "#{label}: c names #{key.inspect} twice" if read.key?(ordered)

          read[ordered] = Read.number(value, "#{label}: c of #{key.inspect}")
        end
      end

      # The upper triangle of c (part <= other in the parts' order), each pair from either order.
      def triangle(hash, label)
        products = read_products(hash, label)
        squares = Array.new(@parts.size) { Array.new(@parts.size, 0.0) }
        @parts.each_index do |part|
          (part...@parts.size).each do |other|
            forward = products[[ part, other ]]
            backward = products[[ other, part ]]
            names = [ @parts[part], @parts[other] ]
            raise ArgumentError, "#{label}: c has no sum for the parts #{names.inspect}" if forward.nil? && backward.nil?
            if forward && backward && (forward - backward).abs > MIRROR * [ forward.abs, backward.abs ].max
              raise ArgumentError, "#{label}: c's two orders of the parts #{names.inspect} differ (#{forward}, #{backward})"
            end

            squares[part][other] = forward || backward
          end
        end
        squares
      end

      # Takes the sums once they hold together: an arm with credit has its own pair, of a weight above 0; an arm
      # without has no sum and no pair; a pair of weight 0 has no sum.
      def restore(weights, values, pairs, rows = {})
        pairs.each do |(first, second), pair|
          unless weights[first].positive? && weights[second].positive?
            raise ArgumentError, "the pair of arms #{@arms[first].inspect} and #{@arms[second].inspect} needs credit to both"
          end
          if pair.weight.zero? && !(pair.values.all?(&:zero?) && pair.squares.flatten.all?(&:zero?))
            raise ArgumentError, "the pair of arms #{@arms[first].inspect} and #{@arms[second].inspect} has sums beside a weight of 0"
          end
        end
        weights.each_with_index do |weight, arm|
          if weight.positive?
            own = pairs[[ arm, arm ]]
            raise ArgumentError, "arm #{@arms[arm].inspect} has credit but no sum of its squared credit" unless own&.weight&.positive?
          elsif !values[arm].all?(&:zero?)
            raise ArgumentError, "arm #{@arms[arm].inspect} has sums but no credit"
          end
        end

        @weights = weights
        @values = values
        @pairs = pairs
        @rows = weights.each_index.map { |arm| rows.fetch(arm) { implied_rows(arm) || 0.0 } }
      end
    end
  end
end
