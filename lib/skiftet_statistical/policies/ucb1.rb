# frozen_string_literal: true

module SkiftetStatistical
  module Policies
    # UCB1: deterministic optimism under uncertainty. Pull the arm maximising
    # mean + sqrt(c * ln(total_pulls) / arm_pulls). Each arm is pulled once first
    # (the confidence bound is undefined at zero pulls). Larger `c` explores more;
    # c = 2.0 is the classic Auer et al. value.
    class UCB1 < Base
      attr_reader :c

      def initialize(c: 2.0, rng: Random.new)
        super()
        raise ArgumentError, "c must be > 0" unless c.positive?

        @c = Float(c)
        @rng = rng
      end

      def choose(arms)
        ensure_arms!(arms)

        fresh = unpulled(arms)
        return fresh[@rng.rand(fresh.length)] unless fresh.empty?

        ln_total = Math.log(arms.sum(&:pulls))
        scored = arms.map do |arm|
          bonus = Math.sqrt(@c * ln_total / arm.pulls)
          [ arm, arm.mean + bonus ]
        end
        pick_max(scored, @rng)
      end

      def to_h
        super.merge(c: @c)
      end
    end
  end
end
