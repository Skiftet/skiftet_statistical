# frozen_string_literal: true

module SkiftetStatistical
  # Frequentist significance testing for A/B experiments. Consolidates the
  # two-proportion z-test, Welch's t-test and the normal CDF that were previously
  # re-implemented (inconsistently) across mej.la's AbTestAnalyzer and skram.la's
  # CRM::AbTestAnalysis. One correct, exact (erf-based) normal CDF — no polynomial
  # approximations.
  module Significance
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
