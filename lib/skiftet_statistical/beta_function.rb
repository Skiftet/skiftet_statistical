# frozen_string_literal: true

module SkiftetStatistical
  # The regularized incomplete beta function I_x(alpha, beta), which is the Beta(alpha, beta) distribution
  # function, and its inverse, the Beta quantile. An exact credible interval on a conversion rate is two quantiles
  # of its Beta posterior; computed this way it does not move when a Monte Carlo seed does.
  #
  #   SkiftetStatistical::BetaFunction.regularized(0.5, 2.0, 2.0)   # => 0.5
  #   SkiftetStatistical::BetaFunction.quantile(0.975, 41.0, 361.0) # => the 97.5 % bound of Beta(41, 361)
  #
  # The arithmetic and its order are part of the contract: with parameters in the hundreds or more the lgamma
  # terms of the front factor reach 1e3-1e5, so a reordered sum can move the result by 1e-12 to 1e-10.
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

    class << self
      # I_x(alpha, beta) for alpha, beta > 0: 0.0 for x <= 0, 1.0 for x >= 1.
      def regularized(x, alpha, beta)
        x = number(x, "x")
        incomplete(x, *parameters(alpha, beta))
      end

      # The x where I_x(alpha, beta) = probability, by bisection: 0.0 for probability <= 0, 1.0 for >= 1.
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

      private

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
