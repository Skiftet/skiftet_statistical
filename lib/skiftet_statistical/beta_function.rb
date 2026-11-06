# frozen_string_literal: true

module SkiftetStatistical
  # The regularized incomplete beta function I_x(alpha, beta), which is the Beta(alpha, beta) distribution
  # function, and its inverse, the Beta quantile. An exact credible interval on a conversion rate is two quantiles
  # of its Beta posterior; computed this way it does not move when a Monte Carlo seed does. Beside them, the log of
  # the Beta density, the shortest interval holding a given mass (the highest-density interval, which holds the
  # mode), and the intervals of means of non-negative values read as shares of their visits.
  #
  #   SkiftetStatistical::BetaFunction.regularized(0.5, 2.0, 2.0)   # => 0.5
  #   SkiftetStatistical::BetaFunction.quantile(0.975, 41.0, 361.0) # => the 97.5 % bound of Beta(41, 361)
  #   SkiftetStatistical::BetaFunction.log_density(0.25, 2.0, 3.0)  # => 0.5232481437645479 (log 27/16)
  #   SkiftetStatistical::BetaFunction.shortest_interval(0.95, 1.0, 2.0) # => [0.0, 0.776393202250...] (1 - sqrt(0.05))
  #   SkiftetStatistical::BetaFunction.mean_intervals({ "a" => { visits: 300, mean: 0.02, standard_error: 0.008 },
  #                                                     "b" => { visits: 300, mean: 0.0, standard_error: 0.0 } },
  #                                                   mass: 0.95)
  #   # => { "a" => [lower, upper] around 0.02, "b" => [0.0, upper] }
  #
  # The arithmetic and its order are part of the contract: with parameters in the hundreds or more the lgamma
  # terms of the front factor reach 1e3-1e5, so a reordered sum can move the result by 1e-12 to 1e-10.
  #
  # Accuracy (spec/beta_function_spec.rb): for parameters up to a few hundred, `regularized` is within 5e-12
  # relative of an exact reference (measured worst 2.4e-13). For parameters from 1 000 to 50 000 each, with n = alpha
  # + beta, the front factor's lgamma terms reach about n log n, whose roundings bound the answer to 4 n log n machine
  # epsilons, relative (about 1e-9 at n = 100 000; measured worst 0.9 of it against an exact binomial reference), and
  # the identities I_x(a, b) + I_(1-x)(b, a) = 1 and I_(1/2)(a, a) = 1/2 hold within that bound. `quantile` bisects to
  # a Float's spacing, so regularized(quantile(p)) is p within the same bound, and its 2.5 % and 97.5 % quantiles lie
  # within 0.001 sd of the normal approximation with its first skewness term. Past a total of about 3 * 10**8 the
  # continued fraction's fixed bound binds (Bandit::Posteriors#summary states what that does to an interval). The same
  # inputs answer the same Float on one platform; another math library's lgamma can move the last bits.
  module BetaFunction
    # Lentz's floor for a denominator that would otherwise be zero.
    TINY = 1.0e-30
    # A fraction step this close to 1 is the last one taken.
    EPSILON = 1.0e-14
    # Numerical Recipes' bound for betacf. Near the mean the fraction takes 116 steps for Beta(10 001, 40 001) and
    # 244 for Beta(100 001, 400 001) (measured), so it binds only past a few hundred thousand observations; there a
    # different bound gives different digits, which makes it part of the contract.
    MAX_ITERATIONS = 300
    # Halvings of [0, 1] in the quantile: 80 take the bracket below a Float's spacing anywhere in (0, 1).
    QUANTILE_STEPS = 80
    # Halvings in the shortest interval, of [0, mode] for its lower end and of [mode, 1] for the point beyond the mode
    # of equal density: as in the quantile, enough to take each bracket below a Float's spacing.
    INTERVAL_STEPS = 80

    class << self
      # I_x(alpha, beta) for alpha, beta > 0: 0.0 for x <= 0, 1.0 for x >= 1. Refuses, as an ArgumentError: an
      # alpha or beta whose to_f is not a finite number above 0, and an x whose to_f is NaN.
      def regularized(x, alpha, beta)
        x = number(x, "x")
        incomplete(x, *parameters(alpha, beta))
      end

      # The x where I_x(alpha, beta) = probability, by bisection: 0.0 for probability <= 0, 1.0 for >= 1. Refuses, as
      # an ArgumentError: an alpha or beta whose to_f is not a finite number above 0, and a probability whose to_f is
      # NaN.
      def quantile(probability, alpha, beta)
        probability = number(probability, "probability")
        alpha, beta = parameters(alpha, beta)
        return 0.0 if probability <= 0.0
        return 1.0 if probability >= 1.0

        low = 0.0
        high = 1.0
        QUANTILE_STEPS.times do
          mid = (low + high) / 2.0
          if incomplete(mid, alpha, beta) < probability
            low = mid
          else
            high = mid
          end
        end
        (low + high) / 2.0
      end

      # The log of the Beta(alpha, beta) density at x, for alpha, beta > 0:
      #   lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta) + (alpha - 1) log x + (beta - 1) log(1 - x)
      # and -Float::INFINITY for x outside (0, 1). x, alpha and beta must be finite real numbers.
      def log_density(x, alpha, beta)
        x = finite(x, "x")
        alpha = finite(alpha, "alpha")
        beta = finite(beta, "beta")
        unless alpha.positive? && beta.positive?
          raise ArgumentError, "alpha and beta must be finite numbers > 0, got #{alpha}, #{beta}"
        end
        return -Float::INFINITY unless x > 0.0 && x < 1.0

        Math.lgamma(alpha + beta)[0] - Math.lgamma(alpha)[0] - Math.lgamma(beta)[0] +
          ((alpha - 1.0) * Math.log(x)) + ((beta - 1.0) * Math.log(1.0 - x))
      end

      # The shortest interval holding `mass` of Beta(alpha, beta), as [lower, upper], for a mass strictly between 0
      # and 1 and alpha, beta > 0.
      #
      # * ONE PEAK INSIDE (alpha > 1 and beta > 1): the highest-density interval. Its two ends have one density, the
      #   lower end below the mode (alpha - 1) / (alpha + beta - 2) and the upper end above it, so it holds the mode.
      # * A DENSITY THAT FALLS FROM 0 (alpha <= 1 <= beta, not both 1): [0, quantile(mass)]; one that rises to 1
      #   (beta <= 1 <= alpha, not both 1): [quantile(1 - mass), 1]. Again the highest-density interval, holding the
      #   mode on its boundary.
      # * A FLAT DENSITY (alpha = beta = 1), where every interval of the mass is as short: the central one,
      #   [(1 - mass) / 2, (1 + mass) / 2].
      # * A U-SHAPED DENSITY (alpha < 1 and beta < 1), whose highest-density region is two pieces: the shorter of
      #   [0, quantile(mass)] and [quantile(1 - mass), 1], the first on a tie.
      #
      # Under the flat prior Beta(1, 1), the posterior of c successes in n trials is Beta(1 + c, 1 + n - c), whose mode
      # is the observed share c / n: its shortest interval always holds that share, which the equal-tailed interval
      # misses at c = 0 and c = n (it leaves 0, or 1, out). For alpha, beta >= 1 the answer holds the mode for any
      # parameters, whole or not.
      #
      # By bisection, no draw: on the side of the peak nearer 0 (a peak above 1/2 read on Beta(beta, alpha) and
      # mirrored), the upper end halves [mode, 1] until the mass between it and the point below the mode of the same
      # density is `mass`, each such point found by halving [0, mode] on the log density. Each end is the bracket's
      # side that does not shrink it, so the answer holds `mass` to within a Float's last bits (a mirrored answer may
      # fall that much short); unrounded.
      #
      # Takes: `mass` a finite real number strictly between 0 and 1 (a String is not one); `alpha` and `beta` read by
      # to_f, as `regularized` reads them. Answers [lower, upper], two Floats with 0.0 <= lower <= upper <= 1.0.
      # Guarantees (exact: no draw, so the same inputs answer the same two Floats on one platform; the mass goes
      # through `regularized`, whose lgamma another math library can move in the last bits):
      # - the interval holds `mass` within `regularized`'s accuracy, short of it by a Float's last bits at most;
      # - with one peak inside, its ends have one density and it is no wider than the equal-tailed interval of the
      #   same mass; the one-sided, flat and U-shaped cases are the closed forms above;
      # - for alpha, beta >= 1 (not both 1) it holds the mode, so under the flat prior the observed share, 0 and 1
      #   included, for whole and fractional counts alike.
      # Refuses, as an ArgumentError: a mass that is not a finite real number strictly between 0 and 1, then an alpha
      # or beta whose to_f is not a finite number above 0.
      def shortest_interval(mass, alpha, beta)
        mass = interval_mass(mass)
        alpha, beta = parameters(alpha, beta)

        if alpha > 1.0 && beta > 1.0
          peaked_interval(mass, alpha, beta)
        elsif alpha == 1.0 && beta == 1.0
          [ (1.0 - mass) / 2.0, (1.0 + mass) / 2.0 ]
        elsif alpha <= 1.0 && beta >= 1.0
          [ 0.0, quantile(mass, alpha, beta) ]
        elsif alpha >= 1.0 && beta <= 1.0
          [ quantile(1.0 - mass, alpha, beta), 1.0 ]
        else
          upper = quantile(mass, alpha, beta)
          lower = quantile(1.0 - mass, alpha, beta)
          upper <= 1.0 - lower ? [ 0.0, upper ] : [ lower, 1.0 ]
        end
      end

      # Each item's `mass` interval of a MEAN OF NON-NEGATIVE VALUES (a value per visit that is never below 0), read
      # as a share of its visits: the shortest interval (.shortest_interval) of the flat-prior Beta posterior of the
      # item's effective count of events, scaled by the value of one such event.
      #
      #   items  { key => { visits:, mean:, standard_error: } } (Symbol or String field names): `visits` the visits
      #          the mean is over, `mean` the mean value per visit, `standard_error` its standard error (the sandwich
      #          of the mean: the spread of the values per visit over the visits, as a Hájek or plain mean gives it).
      #   mass   strictly between 0 and 1.
      #
      # With m the mean, n the visits and s the standard error, the visits' second moment is q = n * s^2 + m^2, one
      # event is worth u = q / m and the item holds k = n * m / u = n * m^2 / q of them, never more than n. The
      # interval is u times the shortest interval of Beta(1 + k, 1 + n - k), whose mode k / n times u is m: the mean
      # always lies inside its interval. In the normal limit the interval's variance, u^2 * (k / n) * (1 - k / n) / n,
      # is s^2 exactly, so a large item's interval is the mean -/+ its normal half-width; a small one's is skewed and
      # never below 0. For a value that is 0 or 1 per visit (c of n visits at 1) q / n is c / n, so u is 1 and k is c:
      # the interval is the share's own, Beta(1 + c, 1 + n - c)'s shortest interval.
      #
      # AN ITEM WITH A MEAN OF 0 has seen no event, so it has no scale of its own: it takes the set's, the value of one
      # event over every item with a mean above 0 (the sum of n * q over the sum of n * m: their values' summed squares
      # over their summed values), and k = 0, so its interval runs from 0 to that value times quantile(mass) of
      # Beta(1, 1 + n), shrinking as its visits grow. With no item above 0 in the set, it has no interval (nil).
      #
      # Takes: `items` a Hash of key => Hash of fields, the keys any objects, answered as given; each figure, where
      # given, a finite real number (a String, a numeric one included, is not one), visits and the mean fractional or
      # whole; `mass` a finite real number strictly between 0 and 1.
      #
      # Answers { key => [lower, upper] } in the items' order, each end clamped to the mean against a last-bit rounding;
      # nil for an item without data, a figure missing (nil: its other figures are then not read) or no visits (0), and
      # for an item with a mean of 0 in a set with no item above 0.
      #
      # Guarantees (exact: no draw, so the same items answer the same Floats on one platform, through
      # .shortest_interval's arithmetic): every item's mean lies inside its interval and no end is below 0; an item
      # with a mean above 0 answers from its own visits, mean and standard error alone, so it answers alike in any set
      # that holds it; an item with a mean of 0 reads the set's items above 0 for its value of one event, and its
      # interval narrows as its visits grow; a value of 0 or 1 per visit gives the share's own interval; in the normal
      # limit the interval's width is the normal one's.
      #
      # Refuses, as an ArgumentError: a mass that is not a finite real number strictly between 0 and 1; items that are
      # not a Hash of Hashes; a visit count, mean or standard error that is given beside the other two but is not a
      # finite real number, visits or a standard error below 0, and a mean below 0; and an item whose arithmetic leaves
      # a Float's range ("item ... leaves a Float's range"): a value of one event (its own, or for a mean of 0 the
      # set's) that is not a finite number above 0, which only extreme figures give (a mean whose square underflows, a
      # tiny mean beside a huge spread, visits times a second moment past 1e308), or visits past about 2**53 that all
      # hold one value.
      def mean_intervals(items, mass:)
        mass = interval_mass(mass)
        raise ArgumentError, "items must be a Hash of key => fields, got #{items.class}" unless items.is_a?(Hash)

        read = items.to_h { |key, fields| [ key, mean_item(key, fields) ] }
        evented = read.values.compact.select { |item| item[:mean].positive? }
        unit = evented.empty? ? nil : set_unit(evented)

        read.to_h do |key, item|
          next [ key, nil ] if item.nil?
          next [ key, nil ] if item[:mean].zero? && unit.nil?

          [ key, scaled_share(mass, key, item, unit) ]
        end
      end

      private

      # A mass strictly between 0 and 1, as a Float.
      def interval_mass(mass)
        mass = finite(mass, "mass")
        raise ArgumentError, "mass must be strictly between 0 and 1, got #{mass}" unless mass > 0.0 && mass < 1.0

        mass
      end

      # The highest-density interval of a Beta with one peak inside (0, 1) (alpha, beta > 1), worked out on the side
      # where the peak is nearer 0: a peak above 1/2 is read on Beta(beta, alpha) and mirrored, x for 1 - x. The mass
      # between an upper end u in [mode, 1] and the point below the mode of the same density rises as u does, from 0
      # at the mode to 1 at u = 1, so halving keeps the side whose mass is at least `mass`. A lower end too near 0 for
      # the halving to tell from 0 (a peak just inside 0, as for alpha just above 1, whose density climbs from 0 within
      # far less than a Float's spacing at the mode) is answered as 0: the mass it leaves out is below that spacing.
      def peaked_interval(mass, alpha, beta)
        if alpha - 1.0 > beta - 1.0
          lower, upper = peaked_interval(mass, beta, alpha)
          return [ 1.0 - upper, 1.0 - lower ]
        end

        mode = (alpha - 1.0) / (alpha + beta - 2.0)
        low = mode
        high = 1.0
        INTERVAL_STEPS.times do
          upper = (low + high) / 2.0
          lower = level_point(upper, mode, alpha, beta)
          if incomplete(upper, alpha, beta) - incomplete(lower, alpha, beta) >= mass
            high = upper
          else
            low = upper
          end
        end
        [ level_point(high, mode, alpha, beta), high ]
      end

      # The point at or below the mode whose density equals the density at `upper` (at most it, so the interval
      # [point, upper] holds no less): 0.0 where that density is 0, or where the point is nearer 0 than the halving
      # of [0, mode] can tell.
      def level_point(upper, mode, alpha, beta)
        target = kernel(upper, alpha, beta)
        return 0.0 if target == -Float::INFINITY

        low = 0.0
        high = mode
        INTERVAL_STEPS.times do
          mid = (low + high) / 2.0
          if kernel(mid, alpha, beta) > target
            high = mid
          else
            low = mid
          end
        end
        low
      end

      # The log of the Beta density without its constant, (alpha - 1) log x + (beta - 1) log(1 - x); -Infinity at and
      # outside the ends.
      def kernel(x, alpha, beta)
        return -Float::INFINITY unless x > 0.0 && x < 1.0

        ((alpha - 1.0) * Math.log(x)) + ((beta - 1.0) * Math.log(1.0 - x))
      end

      # One item of .mean_intervals, read: { visits:, mean:, second: } (second the visits' second moment), or nil
      # without data.
      def mean_item(key, fields)
        raise ArgumentError, "item #{key.inspect} must be a Hash of fields, got #{fields.class}" unless fields.is_a?(Hash)

        visits, mean, error = %i[visits mean standard_error].map { |name| fields.fetch(name) { fields[name.to_s] } }
        return nil if visits.nil? || mean.nil? || error.nil?

        visits = at_least_zero(visits, "item #{key.inspect}'s visits")
        mean = at_least_zero(mean, "item #{key.inspect}'s mean")
        error = at_least_zero(error, "item #{key.inspect}'s standard error")
        return nil if visits.zero?

        { visits: visits, mean: mean, second: (visits * error * error) + (mean * mean) }
      end

      def at_least_zero(value, name)
        number = finite(value, name)
        raise ArgumentError, "#{name} must be at least 0, got #{value.inspect}" if number.negative?

        number
      end

      # The value of one event over the items with a mean above 0: their summed n * q over their summed n * m.
      def set_unit(evented)
        evented.sum { |item| item[:visits] * item[:second] } / evented.sum { |item| item[:visits] * item[:mean] }
      end

      # An item's interval: its own event value and count where its mean is above 0, else the set's event value and
      # no event. Where the arithmetic has left a Float's range on the way (an event value that is not a finite number
      # above 0, or visits so many that 1 + n - k rounds to 0), the item is refused rather than answered.
      def scaled_share(mass, key, item, set_unit)
        mean = item[:mean]
        visits = item[:visits]
        unit, events =
          if mean.positive?
            own = event_value(item[:second] / mean, key, "its own")
            [ own, [ visits * mean / own, visits ].min ]
          else
            [ event_value(set_unit, key, "the set's"), 0.0 ]
          end
        rest = 1.0 + visits - events
        unless rest.positive?
          raise ArgumentError, "item #{key.inspect}'s interval leaves a Float's range (1 + visits - events is #{rest}): " \
                               "rescale the values"
        end

        lower, upper = shortest_interval(mass, 1.0 + events, rest)
        [ [ unit * lower, mean ].min, [ unit * upper, mean ].max ]
      end

      # The value of one event, a finite number above 0, else an ArgumentError naming the item.
      def event_value(value, key, whose)
        return value if value.finite? && value.positive?

        raise ArgumentError, "item #{key.inspect}'s interval leaves a Float's range (#{whose} value of one event is " \
                             "#{value}): rescale the values"
      end

      # A finite real number, as the newer modules read one (a NaN, an infinity, a Complex and a non-number are
      # refused alike), and finite as a Float too (an Integer beyond a Float's range is not).
      def finite(value, name)
        number = value.to_f if value.is_a?(Numeric) && value.real? && value.finite?
        return number if number&.finite?

        raise ArgumentError, "#{name} must be a finite number, got #{value.inspect}"
      end

      def number(value, name)
        value = value.to_f
        raise ArgumentError, "#{name} must be a number, got #{value}" if value.nan?

        value
      end

      def parameters(alpha, beta)
        alpha = alpha.to_f
        beta = beta.to_f
        unless alpha.finite? && alpha.positive? && beta.finite? && beta.positive?
          raise ArgumentError, "alpha and beta must be finite numbers > 0, got #{alpha}, #{beta}"
        end

        [ alpha, beta ]
      end

      # The continued fraction converges fast below x = (alpha + 1) / (alpha + beta + 2); above it the function
      # is read through the symmetry I_x(a, b) = 1 - I_(1-x)(b, a).
      def incomplete(x, alpha, beta)
        return 0.0 if x <= 0.0
        return 1.0 if x >= 1.0

        front = Math.exp(
          Math.lgamma(alpha + beta)[0] - Math.lgamma(alpha)[0] - Math.lgamma(beta)[0] +
          (alpha * Math.log(x)) + (beta * Math.log(1.0 - x))
        )

        if x < (alpha + 1.0) / (alpha + beta + 2.0)
          front * continued_fraction(x, alpha, beta) / alpha
        else
          1.0 - (front * continued_fraction(1.0 - x, beta, alpha) / beta)
        end
      end

      # Numerical Recipes' betacf, evaluated with Lentz's method.
      def continued_fraction(x, alpha, beta)
        qab = alpha + beta
        qap = alpha + 1.0
        qam = alpha - 1.0

        c = 1.0
        d = 1.0 - (qab * x / qap)
        d = TINY if d.abs < TINY
        d = 1.0 / d
        h = d

        (1..MAX_ITERATIONS).each do |m|
          m2 = 2 * m

          numerator = m * (beta - m) * x / ((qam + m2) * (alpha + m2))
          d = 1.0 / guard(1.0 + (numerator * d))
          c = guard(1.0 + (numerator / c))
          h *= d * c

          numerator = -(alpha + m) * (qab + m) * x / ((alpha + m2) * (qap + m2))
          d = 1.0 / guard(1.0 + (numerator * d))
          c = guard(1.0 + (numerator / c))
          delta = d * c
          h *= delta

          break if (delta - 1.0).abs < EPSILON
        end

        h
      end

      def guard(value)
        value.abs < TINY ? TINY : value
      end
    end
  end
end
