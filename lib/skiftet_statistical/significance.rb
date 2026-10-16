# frozen_string_literal: true

module SkiftetStatistical
  # Frequentist significance testing for A/B experiments. Consolidates the
  # two-proportion z-test, Welch's t-test and the normal CDF that were previously
  # re-implemented (inconsistently) across mej.la's AbTestAnalyzer and skram.la's
  # CRM::AbTestAnalysis. One correct, exact (erf-based) normal CDF — no polynomial
  # approximations. Since 0.2 also the normal quantile, the chi-squared survival
  # and the sample-ratio check (f1's copies, one home).
  module Significance
    # normal_quantile bisects [-40, 40]: 0.5 * erfc(40 / sqrt(2)) underflows to 0.0,
    # so every p a Float can hold strictly inside (0, 1) has its z in there.
    NORMAL_QUANTILE_LIMIT = 40.0
    # 60 halvings leave a bracket of 80 / 2**60 (about 7e-17).
    NORMAL_QUANTILE_STEPS = 60
    # Under this many observations a sample-ratio check is not run: too few to tell
    # a broken split from chance.
    SAMPLE_RATIO_MINIMUM_TOTAL = 200

    module_function

    # Standard normal cumulative distribution Phi(z), exact via erf.
    def normal_cdf(z)
      0.5 * (1.0 + Math.erf(z / Math.sqrt(2.0)))
    end

    # Two-tailed p-value for a z (or normal-approx t) statistic. Clamped to [0, 1].
    def two_tailed_p_value(z)
      (2.0 * (1.0 - normal_cdf(z.abs))).clamp(0.0, 1.0)
    end

    # Two-proportion z-test with a pooled standard error, two-tailed. Pass the
    # successes and trials for each group. Returns a Result, or nil when the test
    # is undefined (an empty group, or zero pooled variance). The z sign follows
    # b - a, so a positive z means group B converts higher.
    def two_proportion_z_test(successes_a, trials_a, successes_b, trials_b)
      return nil if trials_a <= 0 || trials_b <= 0

      p_a = successes_a.to_f / trials_a
      p_b = successes_b.to_f / trials_b
      p_pool = (successes_a + successes_b).to_f / (trials_a + trials_b)
      se = Math.sqrt(p_pool * (1.0 - p_pool) * ((1.0 / trials_a) + (1.0 / trials_b)))
      return nil if se.zero?

      z = (p_b - p_a) / se
      Result.new(statistic: z, p_value: two_tailed_p_value(z))
    end

    # Welch's t-test (normal approximation) for two means given their sample
    # variances and sizes. Suitable for revenue-per-visitor style metrics. Returns
    # a Result, or nil when undefined (n < 2 or zero combined variance).
    def welch_t_test(mean_a, variance_a, n_a, mean_b, variance_b, n_b)
      return nil if n_a < 2 || n_b < 2

      denom = (variance_a.to_f / n_a) + (variance_b.to_f / n_b)
      return nil if denom <= 0

      t = (mean_b - mean_a) / Math.sqrt(denom)
      Result.new(statistic: t, p_value: two_tailed_p_value(t))
    end

    # The z where the standard normal distribution function is p (the probit), by
    # bisection on erfc. -Infinity at p = 0 and Infinity at p = 1.
    def normal_quantile(p)
      p = p.to_f
      raise ArgumentError, "p must be between 0 and 1, got #{p}" unless p.between?(0.0, 1.0)
      return -Float::INFINITY if p.zero?
      return Float::INFINITY if p == 1.0

      low = -NORMAL_QUANTILE_LIMIT
      high = NORMAL_QUANTILE_LIMIT
      NORMAL_QUANTILE_STEPS.times do
        middle = (low + high) / 2
        0.5 * Math.erfc(-middle / Math.sqrt(2.0)) < p ? low = middle : high = middle
      end
      (low + high) / 2
    end

    # P(X >= statistic) for X ~ chi-squared(degrees_of_freedom): the regularized
    # upper incomplete gamma Q(dof / 2, statistic / 2). 1.0 for a statistic <= 0.
    def chi_squared_survival(statistic, degrees_of_freedom)
      return 1.0 if statistic <= 0.0

      Gamma.regularized_upper(degrees_of_freedom / 2.0, statistic / 2.0)
    end

    # The sample-ratio mismatch check: does each arm's observed count fit the count
    # its share of the traffic leads us to expect? Pearson's chi-squared over the
    # arms expected to get traffic, on k - 1 degrees of freedom. A small p means the
    # split does not do what its weights say (a broken arm, a cache, a bot).
    #
    #   Significance.sample_ratio_p_value(observed: { control: 700, a: 300 },
    #                                     expected: { control: 500.0, a: 500.0 }) # => ~1.1e-36
    #
    # The arms are observed's keys (expected is read by the same keys, missing = 0).
    # nil (not tested) under minimum_total observations or with fewer than two arms
    # expected to get traffic; 0.0 when an arm expected to get nothing got some,
    # a mismatch no count explains.
    def sample_ratio_p_value(observed:, expected:, minimum_total: SAMPLE_RATIO_MINIMUM_TOTAL)
      total = observed.values.sum
      return nil if total < minimum_total

      cells = observed.keys.map { |arm| [ observed.fetch(arm).to_f, expected.fetch(arm, 0.0).to_f ] }
      return 0.0 if cells.any? { |seen, wanted| wanted <= 0.0 && seen.positive? }

      cells = cells.select { |_seen, wanted| wanted.positive? }
      return nil if cells.size < 2

      statistic = cells.sum { |seen, wanted| ((seen - wanted)**2) / wanted }
      chi_squared_survival(statistic, cells.size - 1)
    end

    # The outcome of a significance test: the test statistic and its two-tailed
    # p-value, with convenience predicates for the usual confidence levels.
    Result = Struct.new(:statistic, :p_value, keyword_init: true) do
      def significant?(alpha = 0.05)
        p_value < alpha
      end

      def significant_90? = significant?(0.10)
      def significant_95? = significant?(0.05)
      def significant_99? = significant?(0.01)

      # Certainty = 1 - p, the complement of the p-value.
      def confidence
        1.0 - p_value
      end
    end
  end
end
