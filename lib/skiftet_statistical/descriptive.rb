# frozen_string_literal: true

module SkiftetStatistical
  # Descriptive statistics over a collection of numbers — mean, variance,
  # standard deviation, and interpolated percentiles. Consolidates the ad-hoc
  # mean/variance (skram.la's revenue-per-visitor stats) and percentile logic
  # (ekonomidata.nu's income distributions) scattered across the workspace.
  module Descriptive
    module_function

    # Arithmetic mean (0.0 for an empty collection).
    def mean(values)
      return 0.0 if values.empty?

      values.sum.to_f / values.length
    end

    # Variance. `sample: true` (default) divides by n-1 (Bessel's correction);
    # `sample: false` divides by n (population variance). 0.0 for n < 2.
    def variance(values, sample: true)
      n = values.length
      return 0.0 if n < 2

      m = mean(values)
      ss = values.sum { |v| (v - m)**2 }
      ss / (sample ? (n - 1) : n).to_f
    end

    def standard_deviation(values, sample: true)
      Math.sqrt(variance(values, sample: sample))
    end

    # Linear-interpolation percentile, `p` in [0, 100]. nil for an empty
    # collection. percentile(values, 50) == median.
    def percentile(values, p)
      return nil if values.empty?

      sorted = values.sort
      return sorted.first.to_f if sorted.length == 1

      rank = (p.clamp(0, 100) / 100.0) * (sorted.length - 1)
      lower = rank.floor
      upper = rank.ceil
      return sorted[lower].to_f if lower == upper

      weight = rank - lower
      (sorted[lower] * (1.0 - weight)) + (sorted[upper] * weight)
    end

    def median(values)
      percentile(values, 50)
    end
  end
end
