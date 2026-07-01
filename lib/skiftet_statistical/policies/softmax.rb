# frozen_string_literal: true

module SkiftetStatistical
  module Policies
    # Softmax / Boltzmann exploration: pick arm i with probability proportional
    # to exp(mean_i / temperature). Low temperature => near-greedy; high
    # temperature => near-uniform exploration.
    class Softmax < Base
      attr_reader :temperature

      def initialize(temperature: 0.1, rng: Random.new)
        super()
        raise ArgumentError, "temperature must be > 0" unless temperature.positive?

        @temperature = Float(temperature)
        @rng = rng
      end

      def choose(arms)
        ensure_arms!(arms)

        # Shift by the max mean for numerical stability (exp can overflow).
        max_mean = arms.map(&:mean).max
        weights = arms.map { |a| Math.exp((a.mean - max_mean) / @temperature) }
        total = weights.sum
        target = @rng.rand * total

        cumulative = 0.0
        arms.each_with_index do |arm, i|
          cumulative += weights[i]
          return arm if cumulative >= target
        end
        arms.last
      end

      def to_h
        super.merge(temperature: @temperature)
      end
    end
  end
end
