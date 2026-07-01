# frozen_string_literal: true

module SkiftetStatistical
  module Policies
    # epsilon-greedy: exploit the best-mean arm with probability (1 - epsilon),
    # explore a uniformly random arm with probability epsilon. Every arm is
    # pulled once first so none is starved by a zero initial mean.
    class EpsilonGreedy < Base
      attr_reader :epsilon

      def initialize(epsilon: 0.1, rng: Random.new)
        super()
        raise ArgumentError, "epsilon must be in [0, 1]" unless (0.0..1.0).cover?(epsilon)

        @epsilon = Float(epsilon)
        @rng = rng
      end

      def choose(arms)
        ensure_arms!(arms)

        fresh = unpulled(arms)
        return fresh[@rng.rand(fresh.length)] unless fresh.empty?

        if @rng.rand < @epsilon
          arms[@rng.rand(arms.length)]
        else
          pick_max(arms.map { |a| [ a, a.mean ] }, @rng)
        end
      end

      def to_h
        super.merge(epsilon: @epsilon)
      end
    end
  end
end
