# frozen_string_literal: true

module SkiftetStatistical
  # Hierarchical Poisson rates: how far a node's rate stands from its parent's, from the node's count s and the count E
  # its parent leads us to expect there. The node's ratio theta = rate / expected rate has a Gamma prior with mean 1 and
  # strength k (Clayton and Kaldor 1987), so its posterior is Gamma(k + s, k + E). Under a two-groups prior (Efron 2008)
  # the node is null with probability 1 - h (its count Poisson(E), its ratio 1) and else drawn from that Gamma slab, so
  # its count is negative binomial NB(s; k, k / (k + E)). Beside the node rule: the leave-one-out share that sets E, the
  # level fit of k (and h) by grid marginal maximum likelihood, the leaf fit that estimates each parent's null rate
  # jointly with k and h, the smoothing of a fitted value across runs, Marshall's (1991) moment estimate and the
  # boundary likelihood-ratio test (Self and Liang 1987).
  #
  # Counts are real numbers >= 0 throughout, so a caller may count by share: a node that stands in its parent only by a
  # share w has w multiplied into its count and total by the caller (there is no weight keyword). The probability mass
  # functions are taken in logs and through log-gamma, s! being Gamma(s + 1), with no rounding.
  #
  #   SkiftetStatistical::HierarchicalRates.node(count: 6.0, expected: 2.5, strength: 8.0, threshold: 1.25,
  #                                            parent_tail: 0.0, parent_ratio: 1.0)
  #   # => { ratio: 1.3333333333333333 (14 / 10.5), hot: 1.0, tail: 0.5593048238168208 (Q(14, 13.125)) }
  #
  # The arithmetic and its order are part of the contract: an app pins these outputs at 1e-9 relative (they go through
  # Math.lgamma), so a reordered formula is a breaking change even when it is "the same".
  module HierarchicalRates
    # The strengths a level is fitted on: 2**-2 to 2**12 in quarter powers of 2, 57 points.
    GRID = Array.new(57) { |i| 2.0**(-2 + (i / 4.0)) }.freeze
    # A strength's log-likelihood within this of the maximum lies in the 95 % profile interval: half the 95 % point
    # of chi-squared(1).
    PROFILE_DROP = 1.92
    # A pair whose parent has fewer children with data than this carries nothing about its level: the fit and both
    # checks leave it out.
    MINIMUM_SIBLINGS = 2
    # The leaf fit takes at most this many steps on a parent's null rate ...
    LEAF_STEPS = 50
    # ... and stops at the first step that moves the rate by at most this share of itself.
    LEAF_CHANGE = 1e-10

    class << self
      # The parent's share with the node's own count and total taken out, smoothed toward a prior share:
      # (parent_count - node_count + prior_strength * prior_share) / (parent_total - node_total + prior_strength).
      def leave_one_out_share(parent_count:, parent_total:, node_count:, node_total:, prior_share:, prior_strength:)
        parent_count = nonnegative(parent_count, "parent_count")
        parent_total = nonnegative(parent_total, "parent_total")
        node_count = nonnegative(node_count, "node_count")
        node_total = nonnegative(node_total, "node_total")
        prior_share = share(prior_share, "prior_share")
        prior_strength = nonnegative(prior_strength, "prior_strength")

        denominator = parent_total - node_total + prior_strength
        unless denominator.positive?
          raise ArgumentError, "parent_total - node_total + prior_strength must be > 0, got #{denominator}"
        end

        (parent_count - node_count + (prior_strength * prior_share)) / denominator
      end

      # One node: count s, expected count E, strength k, hot share h, a threshold L on the ratio to the reference
      # (theta = parent_ratio * node ratio), and the parent's own P(theta > L) and ratio.
      #   hot   = h * NB(s; k, k / (k + E)) / [h * NB + (1 - h) * Poisson(s; E)], 1.0 at h = 1
      #   ratio = (1 - hot) + hot * (k + s) / (k + E)
      #   tail  = (1 - hot) * parent_tail + hot * Q(k + s, (k + E) * L / parent_ratio), Q = Gamma.regularized_upper
      # A node with nothing expected (E = 0) may only have a count of 0, and then answers its prior.
      def node(count:, expected:, strength:, hot_share: 1.0, threshold:, parent_tail:, parent_ratio:)
        s = nonnegative(count, "count")
        e = nonnegative(expected, "expected")
        k = positive(strength, "strength")
        h = share(hot_share, "hot_share")
        limit = nonnegative(threshold, "threshold")
        parent_tail = share(parent_tail, "parent_tail")
        parent_ratio = positive(parent_ratio, "parent_ratio")
        observable(s, e)

        hot = hot_probability(s, e, k, h)
        ratio = (1.0 - hot) + (hot * (k + s) / (k + e))
        tail = ((1.0 - hot) * parent_tail) + (hot * Gamma.regularized_upper(k + s, (k + e) * limit / parent_ratio))
        { ratio: ratio, hot: hot, tail: tail }
      end

      # The level fit: the marginal log-likelihood sum log[h * NB(s; k, k / (k + E)) + (1 - h) * Poisson(s; E)] over
      # pairs [{ count:, expected:, siblings: }] (siblings: the children of the pair's parent that hold data, the pair's
      # own node included), maximised over GRID and over hot_grid (h = 1 without one). A tie goes to the larger
      # strength (the nodes then differ less), then to the hot share listed first. interval holds the grid strengths
      # within PROFILE_DROP of the maximum at the fitted h; profile is [strength, log-likelihood] at every grid point
      # at the fitted h. at_upper_bound: the strength is GRID's top, so the level's nodes inherit their parent.
      def fit_strength(pairs:, hot_grid: nil)
        level = level_terms(pairs)
        hots = hot_grid.nil? ? [ 1.0 ] : hot_shares(hot_grid)

        profiles = hots.map { [] }
        GRID.each do |k|
          terms = slab_terms(level, k)
          hots.each_with_index { |h, j| profiles[j] << log_likelihood(terms, h) }
        end

        best_ll, best_k, best_j = best_point(profiles)
        profile = profiles[best_j]
        inside = GRID.each_index.select { |i| profile[i] >= best_ll - PROFILE_DROP }
        {
          strength: GRID[best_k],
          hot_share: hots[best_j],
          log_likelihood: best_ll,
          at_upper_bound: best_k == GRID.size - 1,
          interval: [ GRID[inside.min], GRID[inside.max] ],
          profile: GRID.each_with_index.map { |k, i| [ k, profile[i] ] }
        }
      end

      # The leaf level's two-groups fit, each parent's NULL rate estimated jointly with k and h instead of read from a
      # leave-one-out share. With few leaves to a parent (about five), a hot leaf lifts its siblings' leave-one-out
      # share, so they read as cold, and a fit on those expected counts reads the hot share several times too high.
      #
      # parents: [[{ count:, total: }]], each parent's leaves, count a real number >= 0 and total the leaf's exposure,
      # > 0; a leaf expects total * rate, rate its parent's null rate. At a strength k (grid: rising strengths, GRID
      # by default) and a hot share h (hot_shares: each in [0, 1), since the rate is read from the leaves that may be
      # null), every parent's rate starts at its pooled rate, sum count / sum total, and each step makes it
      #   rate = sum w * count / sum w * total, w = the leaf's posterior probability of being null at the rate before,
      # the null part of the mixture whose slab part is a node's hot. A parent stops at the first step that moves its
      # rate by at most LEAF_CHANGE of itself (converged) or after LEAF_STEPS steps; parents share only k and h, so
      # each is stepped on its own. A step whose sums give no rate, or a rate of 0 beside a count above 0 (which no
      # rate of 0 explains), is not taken: the parent keeps its rate, not converged.
      #
      # The fit maximises fit_strength's marginal log-likelihood, over the leaves at their parents' rates, on grid
      # times hot_shares; a tie goes to the larger strength, then to the hot share listed first. A parent with fewer
      # than MINIMUM_SIBLINGS leaves is its own leaf's rate and carries nothing about the level: it is left out of
      # the fit and answered at the fitted k and h. The answer, converged, rates and hot in the order given:
      #   { strength:, hot_share:, log_likelihood:, at_upper_bound: (the strength is grid's last), converged: [whether
      #     the parent's rate settled at the fitted point], rates: [rate], hot: [[posterior probability that the leaf
      #     is hot]] }
      # A leaf's hot is node's hot at count: count, expected: total * rate, strength: and hot_share:, bit for bit. A
      # parent whose one leaf is near the line between null and hot can take hundreds of steps to settle: it answers
      # its rate after LEAF_STEPS steps, and converged says so.
      def fit_leaves(parents:, hot_shares:, grid: GRID)
        groups = leaf_groups(parents)
        strengths = strength_grid(grid)
        hots = leaf_hot_shares(hot_shares)
        kept = groups.select { |leaves| leaves.size >= MINIMUM_SIBLINGS }
        raise ArgumentError, "no parent with #{MINIMUM_SIBLINGS} or more leaves" if kept.empty?

        profiles = hots.map { [] }
        strengths.each do |k|
          level = with_strength(kept, k)
          hots.each_with_index { |h, j| profiles[j] << leaf_level(level, k, h)[:log_likelihood] }
        end

        best_ll, best_k, best_j = best_point(profiles)
        k = strengths[best_k]
        h = hots[best_j]
        fitted = leaf_level(with_strength(groups, k), k, h)
        {
          strength: k,
          hot_share: h,
          log_likelihood: best_ll,
          at_upper_bound: best_k == strengths.size - 1,
          converged: fitted[:converged],
          rates: fitted[:rates],
          hot: groups.zip(fitted[:rates]).map do |leaves, rate|
            leaves.map { |s, t| hot_probability(s, t * rate, k, h) }
          end
        }
      end

      # exp((1 - weight) * log previous + weight * log fitted), a moving average in logs; fitted when there is no
      # previous value.
      def smooth_strength(previous:, fitted:, weight: 0.2)
        fitted = positive(fitted, "fitted")
        return fitted if previous.nil?

        previous = positive(previous, "previous")
        weight = share(weight, "weight")
        Math.exp(((1.0 - weight) * Math.log(previous)) + (weight * Math.log(fitted)))
      end

      # Marshall's moment estimate of the ratios' variance over the fit's pairs: A = sum E * (s / E - m)**2 / sum E -
      # m / mean(E), m = sum s / sum E, floored at 0; strength 1 / A, infinite at 0.
      def marshall(pairs:)
        level = kept_pairs(pairs)
        total = level.sum { |_s, e| e }
        m = level.sum { |s, _e| s } / total
        spread = level.sum { |s, e| e * (((s / e) - m)**2) } / total
        variance = spread - (m / (total / level.size))
        variance = 0.0 unless variance.positive?
        { variance: variance, strength: variance.zero? ? Float::INFINITY : 1.0 / variance }
      end

      # Does the level matter at all? LR = 2 * (l(k, h) - l(infinite)), l the fit's log-likelihood on the fit's
      # pairs and l(infinite) the all-Poisson one; p = 0.5 * chi_squared_survival(LR, 1) on the boundary, 1.0 at
      # LR <= 0. The statistic is answered as computed, negative included.
      def boundary_test(pairs:, strength:, hot_share: 1.0)
        level = level_terms(pairs)
        k = positive(strength, "strength")
        h = share(hot_share, "hot_share")

        statistic = 2.0 * (log_likelihood(slab_terms(level, k), h) - level.sum { |term| term[3] })
        p_value = statistic > 0.0 ? 0.5 * Significance.chi_squared_survival(statistic, 1) : 1.0
        { statistic: statistic, p_value: p_value }
      end

      private

      # The posterior probability that the node is drawn from the slab: the slab's part of the same two-part mixture
      # whose log is a pair's term in the level's likelihood (log_likelihood).
      def hot_probability(s, e, k, h)
        return 1.0 if h == 1.0
        return 0.0 if h.zero?

        log_hot, log_rest = log_weights(h)
        log_factorial = Math.lgamma(s + 1.0)[0]
        slab = log_hot + log_slab(s, e, k, Math.lgamma(k)[0], log_factorial)
        null = log_rest + log_null(s, e, log_factorial)
        1.0 / (1.0 + Math.exp(null - slab))
      end

      # sum over pairs of log[h * NB + (1 - h) * Poisson], each term mixed in logs.
      def log_likelihood(terms, h)
        return terms.sum { |slab, _null| slab } if h == 1.0
        return terms.sum { |_slab, null| null } if h.zero?

        log_hot, log_rest = log_weights(h)
        terms.sum { |slab, null| log_sum(log_hot + slab, log_rest + null) }
      end

      # [log h, log(1 - h)] for 0 < h < 1: the two parts' weights.
      def log_weights(h)
        [ Math.log(h), Math.log(1.0 - h) ]
      end

      # log(e**a + e**b), from the larger.
      def log_sum(a, b)
        if a > b
          a + Math.log(1.0 + Math.exp(b - a))
        else
          b + Math.log(1.0 + Math.exp(a - b))
        end
      end

      # log NB(s; k, k / (k + E)) = lgamma(s + k) - lgamma(k) - lgamma(s + 1) + k log(k / (k + E)) + s log(E / (k + E)).
      def log_slab(s, e, k, log_gamma_k, log_factorial)
        log_slab_from(Math.lgamma(s + k)[0], s, e, k, log_gamma_k, log_factorial)
      end

      # log_slab with lgamma(s + k) given, in the same order.
      def log_slab_from(log_gamma_sk, s, e, k, log_gamma_k, log_factorial)
        log_gamma_sk - log_gamma_k - log_factorial + (k * Math.log(k / (k + e))) + x_log_y(s, e / (k + e))
      end

      # log Poisson(s; E) = s log E - E - lgamma(s + 1).
      def log_null(s, e, log_factorial)
        x_log_y(s, e) - e - log_factorial
      end

      # x log y, 0 at x = 0 (a count of 0 against nothing expected).
      def x_log_y(x, y)
        x.zero? ? 0.0 : x * Math.log(y)
      end

      # [log NB, log Poisson] of every pair at strength k.
      def slab_terms(level, k)
        log_gamma_k = Math.lgamma(k)[0]
        level.map { |s, e, log_factorial, null| [ log_slab(s, e, k, log_gamma_k, log_factorial), null ] }
      end

      # The kept pairs with what does not depend on k: [s, E, lgamma(s + 1), log Poisson(s; E)].
      def level_terms(pairs)
        kept_pairs(pairs).map do |s, e|
          log_factorial = Math.lgamma(s + 1.0)[0]
          [ s, e, log_factorial, log_null(s, e, log_factorial) ]
        end
      end

      # Every pair checked, the lone ones (fewer than MINIMUM_SIBLINGS children with data) left out: [[s, E]].
      def kept_pairs(pairs)
        kept = pairs.filter_map do |pair|
          siblings = pair.fetch(:siblings) { raise ArgumentError, "each pair needs siblings:, got #{pair.inspect}" }
          unless siblings.is_a?(Integer) && siblings >= 0
            raise ArgumentError, "siblings must be an Integer >= 0, got #{siblings.inspect}"
          end

          s = nonnegative(pair.fetch(:count) { raise ArgumentError, "each pair needs count:" }, "count")
          e = positive(pair.fetch(:expected) { raise ArgumentError, "each pair needs expected:" }, "expected")
          [ s, e ] if siblings >= MINIMUM_SIBLINGS
        end
        raise ArgumentError, "no pair whose parent has #{MINIMUM_SIBLINGS} or more children with data" if kept.empty?

        kept
      end

      # The largest log-likelihood, ties to the larger strength, then to the hot share listed first:
      # [log-likelihood, grid index, hot index].
      def best_point(profiles)
        best = nil
        profiles.each_with_index do |profile, j|
          profile.each_with_index do |ll, i|
            best = [ ll, i, j ] if best.nil? || ll > best[0] || (ll == best[0] && i > best[1])
          end
        end
        best
      end

      def hot_shares(hot_grid, name = "hot_grid")
        hots = hot_grid.map { |h| share(h, "each #{name} value") }
        raise ArgumentError, "#{name} must hold at least one hot share" if hots.empty?

        hots
      end

      # The leaf fit's hot shares, each below 1: the null rate is read from the leaves that may be null.
      def leaf_hot_shares(values)
        raise ArgumentError, "hot_shares must be an Array, got #{values.inspect}" unless values.is_a?(Array)

        hots = hot_shares(values, "hot_shares")
        raise ArgumentError, "each hot_shares value must be below 1, got #{values.inspect}" if hots.include?(1.0)

        hots
      end

      def strength_grid(grid)
        unless grid.is_a?(Array) && !grid.empty?
          raise ArgumentError, "grid must be an Array of at least one strength, got #{grid.inspect}"
        end

        strengths = grid.map { |k| positive(k, "each grid strength") }
        raise ArgumentError, "grid must rise strictly, got #{grid.inspect}" unless strengths.each_cons(2).all? { |a, b| a < b }

        strengths
      end

      # Every parent's leaves checked: [[[s, t, lgamma(s + 1)]]].
      def leaf_groups(parents)
        raise ArgumentError, "parents must be an Array of parents, got #{parents.inspect}" unless parents.is_a?(Array)

        parents.map do |leaves|
          unless leaves.is_a?(Array) && !leaves.empty?
            raise ArgumentError, "each parent must be an Array of at least one leaf, got #{leaves.inspect}"
          end

          leaves.map do |leaf|
            raise ArgumentError, "each leaf must be a Hash of count: and total:, got #{leaf.inspect}" unless leaf.is_a?(Hash)

            s = nonnegative(leaf.fetch(:count) { raise ArgumentError, "each leaf needs count:" }, "count")
            t = positive(leaf.fetch(:total) { raise ArgumentError, "each leaf needs total:" }, "total")
            [ s, t, Math.lgamma(s + 1.0)[0] ]
          end
        end
      end

      # The parents at strength k, each [pooled rate, any count above 0, leaves], a leaf [s, t, lgamma(s + 1),
      # lgamma(s + k), lgamma(s + k) - lgamma(k) + k log k].
      def with_strength(groups, k)
        base = (k * Math.log(k)) - Math.lgamma(k)[0]
        groups.map do |leaves|
          parent = leaves.map do |s, t, log_factorial|
            log_gamma_sk = Math.lgamma(s + k)[0]
            [ s, t, log_factorial, log_gamma_sk, log_gamma_sk + base ]
          end
          [ pooled_rate(leaves), leaves.any? { |s, _t| s.positive? }, parent ]
        end
      end

      # Every parent's null rate at strength k and hot share h, whether it settled, and the log-likelihood of their
      # leaves at those rates (fit_strength's terms, in the same order).
      def leaf_level(level, k, h)
        log_gamma_k = Math.lgamma(k)[0]
        odds = h.zero? ? nil : Math.log(h) - Math.log(1.0 - h)
        converged = []
        terms = []
        rates = level.map do |pooled, counted, leaves|
          rate, settled = odds.nil? || !counted || leaves.size == 1 ? [ pooled, true ] : null_rate(pooled, leaves, k, odds)
          converged << settled
          leaves.each do |s, t, log_factorial, log_gamma_sk|
            e = t * rate
            terms << [ log_slab_from(log_gamma_sk, s, e, k, log_gamma_k, log_factorial), log_null(s, e, log_factorial) ]
          end
          rate
        end
        { rates: rates, converged: converged, log_likelihood: log_likelihood(terms, h) }
      end

      # One parent's null rate (fit_leaves) from its pooled rate, at log odds log(h / (1 - h)) with 0 < h < 1 and a
      # count above 0: [rate, converged]. A leaf's weight is its posterior probability of being null, 1 / (1 +
      # exp(d)), d the log odds of slab to null, log(h / (1 - h)) + lgamma(s + k) - lgamma(k) + k log k + E - (k + s)
      # log(k + E), the two parts' lgamma(s + 1) and s log E cancelling. (At h = 0 every weight is 1, with no count
      # every rate is 0, and a lone leaf's weight cancels, so in each case the pooled rate is its own next step.)
      def null_rate(rate, leaves, k, odds)
        LEAF_STEPS.times do
          count = 0.0
          total = 0.0
          leaves.each do |s, t, _log_factorial, _log_gamma_sk, base|
            e = t * rate
            w = 1.0 / (1.0 + Math.exp(odds + base + e - ((k + s) * Math.log(k + e))))
            count += w * s
            total += w * t
          end
          following = count / total
          return [ rate, false ] unless following.finite? && following.positive?

          settled = (following - rate).abs <= LEAF_CHANGE * rate
          rate = following
          return [ rate, true ] if settled
        end
        [ rate, false ]
      end

      # sum s / sum t over a parent's leaves, summed in order: every weight 1.
      def pooled_rate(leaves)
        count = 0.0
        total = 0.0
        leaves.each do |s, t|
          count += s
          total += t
        end
        count / total
      end

      def observable(s, e)
        return unless e.zero? && s.positive?

        raise ArgumentError, "a count above 0 needs an expected count above 0, got count #{s} and expected 0"
      end

      # A finite real number, as the other modules read one (a Complex is not real, whatever its imaginary part), and
      # finite as a Float too (an Integer or Rational beyond a Float's range is not).
      def real(value, name)
        number = value.to_f if value.is_a?(Numeric) && value.real? && value.finite?
        return number if number&.finite?

        raise ArgumentError, "#{name} must be a finite number, got #{value.inspect}"
      end

      def nonnegative(value, name)
        value = real(value, name)
        raise ArgumentError, "#{name} must be >= 0, got #{value}" if value.negative?

        value
      end

      def positive(value, name)
        value = real(value, name)
        raise ArgumentError, "#{name} must be > 0, got #{value}" unless value.positive?

        value
      end

      def share(value, name)
        value = real(value, name)
        raise ArgumentError, "#{name} must lie in [0, 1], got #{value}" unless value.between?(0.0, 1.0)

        value
      end
    end
  end
end
