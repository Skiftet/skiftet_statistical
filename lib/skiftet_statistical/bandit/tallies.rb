# frozen_string_literal: true

require_relative "../numbers"
require_relative "../inputs"
require_relative "checks"

module SkiftetStatistical
  class Bandit
    # How one tally is read and how it becomes one posterior pair, a rule per kind, each written once. Private to the
    # bandit: the posterior set, Prior and Decay read every tally here (the posterior set an amounts prior), every list
    # is read here (Decay's rows, Prior's others and levels, the posterior set's lists of names), the posterior set and
    # Prior.pooled_new add yes/no evidence to a prior here, Prior.pooled_amount and the amounts rule take the moments
    # here, and Prior and the amounts rule shrink a mean here, so each is the same arithmetic on every path. The
    # moments, the amounts rule's gap and the shrunk mean are formed exactly from the Floats they are given (`exact`,
    # `exact_sum`) and rounded once (`rounded`), so no cancellation and no underflow on the way changes an answer; the
    # posterior set's Normal draws are compared by the same exact arithmetic, never rounded (`exact_key`).
    # Which weighted sums count, the yes/no rule and the effective views are 0.3.0's WeightedPosterior.count,
    # WeightedPosterior.beta and WeightedPosterior.effective_exposures, called on the tally's checked Floats, never
    # written again here. The rules check nothing and bound no pair: the posterior set bounds a pair above its
    # LARGEST_TOTAL once it is built (never here).
    module Tallies
      # THE sums that may be below 0, declared once: in a tally of amounts the plain and the weighted sum of the
      # amounts (`outcomes`, `weighted_outcomes`); every other sum, and every sum of a yes/no tally, is >= 0. Every
      # reader of a tally asks `read` with `amounts:`, never a list of its own.
      SIGNED_SUMS = %i[outcomes weighted_outcomes].freeze
      # A Float's significant bits (53), and the power of 2 of the smallest Float above 0 (2**-1074, a subnormal).
      MANTISSA_BITS = Float::MANT_DIG
      SMALLEST_POWER = Float::MIN_EXP - Float::MANT_DIG
      # The exact numbers 0 and 1, as `exact` answers them.
      ZERO = [ 0, 0 ].freeze
      ONE = [ 1, 0 ].freeze
      # The smallest power of 2 `exact` answers for a product of at most two finite Floats, and so `exact_sum` for two
      # such terms: each factor's mantissa is read as an Integer of MANTISSA_BITS bits, so the smallest Float above 0,
      # 2**-1074, reads 2**52 * 2**-1126, and two factors read at least 2**-2252.
      PAIR_FLOOR = 2 * (SMALLEST_POWER - MANTISSA_BITS + 1)

      module_function

      # A tally's sums by `keys` (Symbol or String keys; a nil tally holds none), read by the bandit's one reading of a
      # caller's Hash by Symbol names (Checks.named: from its pairs, so a default the Hash holds is never read and a Hash
      # that compares its keys by identity reads as a plain one), each by the library's one number reader
      # (Numbers.read, the input domain), nil when absent or nil. Every sum is >= 0, save, with `amounts: true`, those
      # in SIGNED_SUMS, which may be any number in the domain. `label` names the tally in the error words.
      def read(tally, label, keys, amounts: false)
        tally = {} if nil.equal?(tally)
        raise ArgumentError, "#{label} must be a Hash or nil (got #{Inputs.class_name(tally)})" unless Hash === tally

        named = Checks.named(tally, keys, label)
        keys.to_h do |key|
          found = named.fetch(key, nil)
          name = "#{label}'s #{key}"
          sum =
            if nil.equal?(found) then nil
            elsif amounts && SIGNED_SUMS.include?(key) then Numbers.read(found, name)
            else Checks.nonnegative(found, name)
            end
          [ key, sum ]
        end
      end

      # THE reading of a list, written once for every list the bandit reads (Decay's rows, Prior.pooled_new's others,
      # Prior.chain's levels, and the posterior set's lists of item names: `may_lead`, `pinned`'s names, `among`): a
      # list is any Enumerable but a Hash (an Array, a Set, an Enumerator, a lazy list), read once, by one pass of its
      # each into a frozen Array (one call into the caller's object, Inputs.asked), so what its map, its to_a or a
      # second pass would answer is never read. Anything else, a Hash or an object that answers each but is no
      # Enumerable included, is an ArgumentError, "<label> must be a list of <what> (got <value>)".
      def list(given, label, what)
        unless Enumerable === given && !(Hash === given)
          raise ArgumentError, "#{label} must be a list of #{what} (got #{Inputs.shown(given)})"
        end

        read = []
        Inputs.asked(given, "#{label}'s each") { given.each { |element| read << element } }
        read.freeze
      end

      # An amounts prior as the amounts rule reads it: a Prior::Amount, its three numbers (asked by the library's one
      # call into a caller's object, Inputs.asked, as a subclass may answer them) read again by the Amount's own checks
      # (Prior.amount), so an Amount that never ran them (made by allocate, or by a subclass that skips them) is
      # refused as any number that is not one is. Anything else is an ArgumentError naming `label`.
      def amount_prior(given, label)
        unless Prior::Amount === given
          raise ArgumentError, "#{label} must be a Prior::Amount (there is no scale-free amounts prior; got " \
                               "#{Inputs.shown(given)})"
        end

        mean, variance, strength = Inputs.asked(given, "#{label}'s numbers") do
          [ given.mean, given.variance, given.strength ]
        end
        Prior.amount(mean: mean, variance: variance, strength: strength)
      end

      # Which weighted sums a tally counts, as [weighted, squared]: weighted when weighted_views is above 0 and every
      # weighted sum in `sums` is given (else the plain sums are judged); the squared sum only beside them, and only
      # above 0. WeightedPosterior.count, 0.3.0's rule, judges it on these sums (its weighted conversions given when
      # every sum in `sums` is), never written again here.
      def counted(weighted_views, sums, weighted_views_squared)
        count = WeightedPosterior.count(
          { weighted_exposures: weighted_views, weighted_conversions: sums.none?(&:nil?) ? 0.0 : nil,
            weighted_exposures_squared: weighted_views_squared }
        )
        [ !count[:weighted_exposures].nil?, !count[:weighted_exposures_squared].nil? ]
      end

      # The views a posterior is built on: with a squared sum, the Kish size (sum w)^2 / sum w^2, never above the views;
      # without one, the views as given. WeightedPosterior.effective_exposures, 0.3.0's rule, on these sums.
      def effective_views(views, weighted_views, weighted_views_squared)
        WeightedPosterior.effective_exposures(
          { exposures: views, weighted_exposures: weighted_views, weighted_exposures_squared: weighted_views_squared }
        )
      end

      # THE yes/no rule: [prior alpha + c, prior beta + n - c], n the effective views and c the outcomes clamped to
      # [0, n], or, when weighted_views is given, the weighted outcomes rescaled to n (weighted_outcomes * n /
      # weighted_views) and clamped to [0, n]. A weighted sum the tally does not count is passed as nil (the default).
      # WeightedPosterior.beta, 0.3.0's rule, on these sums and this prior.
      def yes_no(views:, outcomes:, prior:, weighted_views: nil, weighted_outcomes: nil, weighted_views_squared: nil)
        prior_alpha, prior_beta = prior
        WeightedPosterior.beta(
          { exposures: views, conversions: outcomes, weighted_exposures: weighted_views,
            weighted_conversions: weighted_outcomes, weighted_exposures_squared: weighted_views_squared },
          prior_alpha: prior_alpha, prior_beta: prior_beta
        )
      end

      # THE amounts rule: the mean per view is Normal(mu_n, sd_n) under a prior of k views at mean mu0 and per-view
      # variance v0, with n the effective views, m the mean amount a view and s^2 its variance (from the weighted sums
      # when weighted_views is given, else the plain ones, `moments`; at n = 0, m and s^2 read 0):
      # mu_n = (k mu0 + n m) / (k + n), the one shrunk mean (`shrunk`), and
      # sd_n^2 = [k v0 + n s^2 + k n (m - mu0)^2 / (k + n)] / (k + n)^2, its root taken by `spread`, which forms no
      # product of numbers of the input domain's size at all (Numbers::LARGEST). With m = sum / base (base the views
      # the moments divide by), n m is n sum / base, so mu_n is (k mu0 base + n sum) / ((k + n) base) and the gap
      # m - mu0 is (sum - mu0 base) / base: each formed exactly from the Floats and rounded once, as s^2 is.
      def amounts(views:, outcomes:, outcome_squares:, weighted_views:, weighted_outcomes:, weighted_outcome_squares:,
                  weighted_views_squared:, prior:)
        n = effective_views(views, weighted_views, weighted_views_squared)
        strength = prior.strength
        if n.zero?
          return [ shrunk(exact(strength, prior.mean), exact(strength), ZERO, ZERO),
                   spread(strength, prior.variance, n, 0.0, 0.0) ]
        end

        sum, squares, base =
          if weighted_views then [ weighted_outcomes, weighted_outcome_squares, weighted_views ]
          else [ outcomes, outcome_squares, views ]
          end
        _mean, variance = moments(sum, squares, base)
        gap = rounded(exact_sum(exact(sum), exact(-prior.mean, base)), exact(base))
        [ shrunk(exact(strength, prior.mean, base), exact(strength, base), exact(n, sum), exact(n, base)),
          spread(strength, prior.variance, n, variance, gap) ]
      end

      # THE amounts rule's standard deviation, sd_n = sqrt(a + b + c) with T = k + n and gap = m - mu0:
      # a = k v0 / T^2, b = n s^2 / T^2 and c = k n gap gap / T^3 (the rule's three terms, each over (k + n)^2).
      # Each term is a product of numbers of the domain's size divided by T two or three times, which, formed as it
      # reads, could leave a Float either way: k n gap gap reaches 4 * 10**400 (k = n = 10**100, gap = 2 * 10**100),
      # and a share such as k / T underflows when k is far below n (k = 10**-300 at n = 10**100 reads 0, though c
      # there is 10**-200 * gap gap), as does n / T when n is far below k, and k v0 / T^2 itself when the answer is
      # small (k = 10**100 at n = 0 and v0 = 10**-300: the term is 10**-400, the answer 10**-200). So no term is formed
      # as a Float at all: `scaled` splits every factor and the divisor by Math.frexp into a mantissa in [0.5, 1) and a
      # power of 2, multiplies and divides the mantissas only (a's and b's in (1/4, 4), c's in (1/16, 8): no mantissa
      # product leaves a Float either way) and adds the powers as Integers. The terms are then added on the largest
      # power, made even (each term below 8, their sum in [1/32, 24)), and the root is the root of that sum times 2 to
      # half that power. The answer is the one number rounded into a Float's range: below the smallest Float (k = v0 =
      # 5 * 10**-324 at n = 10**100, an exact sd near 10**-424) it reads 0, which the set refuses as a standard
      # deviation that is not above 0; above the largest it reads Infinity, which only a mean amount or a variance that
      # left a Float (a quotient by a small input) can make and the check of the answer refuses; a finite sd beyond
      # the input domain (k = n = 5 * 10**-324 at v0 = 10**100: about 10**211) is refused by the set's reading of its
      # pair.
      # Every other in-domain tally reads its exact sd (the rule on the Floats it is given, s^2 and the gap exact)
      # within 1e-15 relative: s^2 and the gap are each formed exactly and rounded once (`moments`, `amounts`), T
      # rounds once, and the mantissa arithmetic of a term rounds once a step, so a term is within 5 (a), 6 (b) and 11
      # (c) roundings of its exact value; the terms are >= 0, so their sum (2 roundings more) cancels nothing, and the
      # root halves that and rounds once: within 7.5 roundings, 2**-53 each, 8.4e-16. An sd below the smallest normal
      # Float (2**-1022, about 2.2 * 10**-308) reads on the subnormal grid as well, within half its spacing (2**-1075)
      # more. An infinity in s^2 or the gap (a quotient by a small input that left a Float) rides through to the
      # answer, as it is.
      def spread(strength, prior_variance, views, variance, gap)
        total = strength + views
        terms = [ scaled([ strength, prior_variance ], total, 2), scaled([ views, variance ], total, 2),
                  scaled([ strength, views, gap, gap ], total, 3) ].compact
        power = terms.map(&:last).max
        power += 1 if power.odd?
        sum = terms.sum { |mantissa, own| Math.ldexp(mantissa, own - power) }
        Math.ldexp(Math.sqrt(sum), power / 2)
      end

      # [mantissa, power], the product of `factors` divided `times` times by `divisor` (above 0) as mantissa * 2**power,
      # every number split by Math.frexp into a mantissa in [0.5, 1) and a power of 2, the mantissas multiplied and
      # divided and the powers added as Integers, so no step leaves a Float's range; nil when a factor is 0 (the term
      # is 0).
      def scaled(factors, divisor, times)
        mantissa = 1.0
        power = 0
        factors.each do |factor|
          return nil if factor.zero?

          fraction, exponent = Math.frexp(factor)
          mantissa *= fraction
          power += exponent
        end
        fraction, exponent = Math.frexp(divisor)
        times.times do
          mantissa /= fraction
          power -= exponent
        end
        [ mantissa, power ]
      end

      # THE shrunk mean, written once (Prior's pooled means and the amounts rule's posterior mean): (prior outcomes +
      # outcomes) / (prior views + views), its four numbers exact (`exact`: a strength times a mean is passed as the
      # exact product), the prior views and views summing above 0, else an ArgumentError. The sums and the quotient are
      # formed exactly and rounded once (`rounded`), so no step underflows a mean a Float can hold, and when both views
      # are above 0 the answer, a convex combination of the two means it weighs, is held between them as they round
      # (prior outcomes / prior views and outcomes / views, each rounded once): it never rounds past either.
      def shrunk(prior_outcomes, prior_views, outcomes, views)
        total = exact_sum(prior_views, views)
        unless total.first.positive?
          raise ArgumentError, "strength + views must be above 0 (got #{rounded(total, ONE)})"
        end

        mean = rounded(exact_sum(prior_outcomes, outcomes), total)
        return mean unless prior_views.first.positive? && views.first.positive?

        mean.clamp(*[ rounded(prior_outcomes, prior_views), rounded(outcomes, views) ].minmax)
      end

      # THE moments of amounts: [m, s^2] = [sum / views, (squares views - sum sum) / views^2], views above 0: m one
      # Float quotient (one rounding), s^2 formed exactly from the Floats (`exact`, `exact_sum`) and rounded once
      # (`rounded`), so no cancellation between squares / views and m^2 changes it; a residual below 0 (squares that
      # no amounts could sum to) reads 0.
      def moments(sum, squares, views)
        residual = exact_sum(exact(squares, views), exact(-sum, sum))
        [ sum / views, residual.first.negative? ? 0.0 : rounded(residual, exact(views, views)) ]
      end

      # The product of finite Floats, exactly, as [integer, power], the number integer * 2**power: each factor split by
      # Math.frexp into a mantissa in [0.5, 1), scaled to an Integer of at most 53 bits, and a power of 2, the Integers
      # multiplied and the powers added, so no step rounds, over- or underflows. One factor is the Float itself.
      def exact(*factors)
        integer = 1
        power = 0
        factors.each do |factor|
          fraction, exponent = Math.frexp(factor)
          integer *= Math.ldexp(fraction, MANTISSA_BITS).to_i
          power += exponent - MANTISSA_BITS
        end
        [ integer, power ]
      end

      # The sum of two exact numbers (`exact`), exactly: the Integer of the larger power shifted onto the smaller power,
      # and added (a term that is 0 is the other).
      def exact_sum(first, second)
        return second if first.first.zero?
        return first if second.first.zero?

        (left, left_power), (right, right_power) = first, second
        return [ left + (right << (right_power - left_power)), left_power ] if left_power < right_power

        [ (left << (left_power - right_power)) + right, right_power ]
      end

      # An exact number (`exact`, `exact_sum`) of terms of at most two Floats each, as one Integer: the number times
      # 2**-PAIR_FLOOR, so any two such numbers compare exactly as their Integers (Integer#<=>), with no rounding and
      # no fraction to reduce (the posterior set's Normal draw keys).
      def exact_key(number)
        integer, power = number
        integer << (power - PAIR_FLOOR)
      end

      # The quotient of two exact numbers (`exact`; the denominator above 0) rounded once to the nearest Float, ties
      # to the even one, as a Float's own division rounds: its leading bit found by one comparison, its 53 leading bits
      # (fewer below the smallest normal Float, on the subnormal grid down to 2**-1074) taken by one Integer division
      # and rounded by its remainder; a quotient beyond the largest Float reads an infinity (Math.ldexp). (Ruby's own
      # Rational#to_f does not round so: it can miss the nearest Float by one.)
      def rounded(numerator, denominator)
        top, top_power = numerator
        return 0.0 if top.zero?

        bottom, bottom_power = denominator
        size = top.abs
        power = top_power - bottom_power
        # size / bottom lies in [2**(lead - 1), 2**(lead + 1)), and in [2**lead, 2**(lead + 1)) once lead is lowered
        # where it lies below 2**lead.
        lead = size.bit_length - bottom.bit_length
        lead -= 1 if (lead.negative? ? size << -lead : size) < (lead.negative? ? bottom : bottom << lead)
        shift = [ lead + 1 - MANTISSA_BITS, SMALLEST_POWER - power ].max
        divisor = shift.negative? ? bottom : bottom << shift
        quotient, remainder = (shift.negative? ? size << -shift : size).divmod(divisor)
        half = (remainder << 1) <=> divisor
        quotient += 1 if half.positive? || (half.zero? && quotient.odd?)
        Math.ldexp(top.negative? ? -quotient.to_f : quotient.to_f, shift + power)
      end
    end
    private_constant :Tallies
  end
end
