# frozen_string_literal: true

module SkiftetStatistical
  # The n-point Gauss-Legendre rule on [-1, 1]: n nodes x_i and weights w_i with sum w_i * f(x_i) = the integral of f
  # over [-1, 1] for every polynomial f of degree up to 2n - 1. The nodes are the roots of the Legendre polynomial P_n,
  # found by Newton's method from the cosine first guess cos(pi * (i - 1/4) / (n + 1/2)); each weight is
  # 2 / ((1 - x**2) * P_n'(x)**2). For a panel [a, b], x maps to (a + b) / 2 + (b - a) / 2 * x and w to (b - a) / 2 * w.
  #
  #   SkiftetStatistical::GaussLegendre.nodes(2)
  #   # => [[-0.5773502691896257, 1.0], [0.5773502691896257, 1.0]] (x = -+1 / sqrt(3), w = 1)
  #
  # The rule is computed once per n and kept: every later call answers the same frozen pairs. The positive roots are
  # found by Newton's method and mirrored, so the rule is exactly symmetric, and for odd n the middle node is exactly 0
  # (P_n is odd). The pairs come in ascending order of x.
  module GaussLegendre
    # Newton's method stops once a step is this small (the roots lie in (-1, 1), so the bound is absolute); from there
    # one more step could not move a node by a Float's spacing.
    NEWTON_TOLERANCE = 1e-15
    # A bound, not a tuning knob: from the cosine guess Newton takes 2 to 5 steps for every root at every n from 1 to
    # 300 and at 500, 1 000, 2 000 and 4 000 (measured).
    NEWTON_STEPS = 100

    @rules = {}
    @lock = Mutex.new

    class << self
      # [[x, w]] for the n-point rule on [-1, 1], ascending in x; n an Integer >= 1.
      def nodes(n)
        raise ArgumentError, "n must be an Integer >= 1, got #{n.inspect}" unless n.is_a?(Integer) && n >= 1

        @lock.synchronize { @rules[n] ||= build(n) }
      end

      private

      def build(n)
        half = n / 2
        positive = (1..half).map { |i| root(n, i) }
        middle = n.odd? ? [ [ 0.0, weight(n, 0.0) ] ] : []
        lower = positive.map { |x, w| [ -x, w ].freeze }
        (lower + middle + positive.reverse).map(&:freeze).freeze
      end

      # The i-th largest root of P_n and its weight.
      def root(n, i)
        x = Math.cos(Math::PI * (i - 0.25) / (n + 0.5))
        NEWTON_STEPS.times do
          value, derivative = legendre(n, x)
          step = value / derivative
          x -= step
          break if step.abs <= NEWTON_TOLERANCE
        end
        [ x, weight(n, x) ]
      end

      def weight(n, x)
        _value, derivative = legendre(n, x)
        2.0 / ((1.0 - (x * x)) * derivative * derivative)
      end

      # P_n(x) by the three-term recurrence j P_j = (2j - 1) x P_(j-1) - (j - 1) P_(j-2), and its derivative
      # n (x P_n - P_(n-1)) / (x**2 - 1).
      def legendre(n, x)
        previous = 1.0
        current = x
        (2..n).each do |j|
          previous, current = current, ((((2 * j) - 1) * x * current) - ((j - 1) * previous)) / j
        end
        [ current, n * ((x * current) - previous) / ((x * x) - 1.0) ]
      end
    end
  end
end
