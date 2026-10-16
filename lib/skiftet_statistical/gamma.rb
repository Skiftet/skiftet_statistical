# frozen_string_literal: true

module SkiftetStatistical
  # The regularized incomplete gamma functions: P(a, x), the Gamma(a, 1) distribution function at x, and
  # Q(a, x) = 1 - P(a, x). The chi-squared survival (Significance.chi_squared_survival) and an exact Poisson
  # interval or p-value are both read off them.
  #
  # Below x = a + 1 the power series converges fast and gives P; at and above it the continued fraction
  # (modified Lentz) converges fast and gives Q. Each function reads the other half as one minus it
  # (Numerical Recipes, gammp / gammq).
  #
  #   SkiftetStatistical::Gamma.regularized_lower(1.0, 2.0) # => 0.8646647167633873 (1 - e**-2)
  #   SkiftetStatistical::Gamma.regularized_upper(1.0, 2.0) # => 0.1353352832366127 (e**-2)
  #
  # The arithmetic and its order are part of the contract: an app that pins outputs at 1e-12 compares them with
  # this exact sequence of operations, so a reordered formula is a breaking change even when it is "the same".
  module Gamma
    # A series term or a fraction step this small next to the total is the last one taken.
    EPSILON = 1e-15
    # A bound, not a tuning knob: near x = a the series takes 7 323 terms at a = 1e6 and 62 463 at a = 1e8 (the
    # fraction fewer), measured. A lower bound gives different digits for large a.
    MAX_TERMS = 100_000
    # Lentz's floor for a denominator that would otherwise be zero.
    TINY = 1e-300

    class << self
      # P(a, x) for a > 0. 0.0 for x <= 0.
      def regularized_lower(a, x)
        a, x = arguments(a, x)
        return 0.0 if x <= 0.0
        return 1.0 if x.infinite?

        x < a + 1.0 ? series(a, x) : 1.0 - fraction(a, x)
      end

      # Q(a, x) = 1 - P(a, x) for a > 0. 1.0 for x <= 0.
      def regularized_upper(a, x)
        a, x = arguments(a, x)
        return 1.0 if x <= 0.0
        return 0.0 if x.infinite?

        x < a + 1.0 ? 1.0 - series(a, x) : fraction(a, x)
      end

      private

      def arguments(a, x)
        a = a.to_f
        x = x.to_f
        raise ArgumentError, "a must be a finite number > 0, got #{a}" unless a.finite? && a.positive?
        raise ArgumentError, "x must be a number, got #{x}" if x.nan?

        [ a, x ]
      end

      # x**a * e**-x / Gamma(a), in logs so it neither overflows nor underflows on its way.
      def prefix(a, x)
        Math.exp((a * Math.log(x)) - x - Math.lgamma(a).first)
      end

      # P(a, x) by its power series; converges fast for x < a + 1.
      def series(a, x)
        term = 1.0 / a
        sum = term
        denominator = a
        MAX_TERMS.times do
          denominator += 1.0
          term *= x / denominator
          sum += term
          break if term.abs < sum.abs * EPSILON
        end
        sum * prefix(a, x)
      end

      # Q(a, x) by its continued fraction, modified Lentz; converges fast for x >= a + 1.
      def fraction(a, x)
        b = x + 1.0 - a
        c = 1.0 / TINY
        d = 1.0 / b
        h = d
        (1..MAX_TERMS).each do |i|
          an = -i * (i - a)
          b += 2.0
          d = (an * d) + b
          d = TINY if d.abs < TINY
          c = b + (an / c)
          c = TINY if c.abs < TINY
          d = 1.0 / d
          delta = d * c
          h *= delta
          break if (delta - 1.0).abs < EPSILON
        end
        h * prefix(a, x)
      end
    end
  end
end
