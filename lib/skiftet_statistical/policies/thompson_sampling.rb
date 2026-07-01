# frozen_string_literal: true

module SkiftetStatistical
  module Policies
    # Thompson Sampling (Beta-Bernoulli). For each arm draw theta ~ Beta(alpha0 +
    # successes, beta0 + failures) and pull the arm with the highest draw. It
    # balances exploration and exploitation automatically: under-sampled arms
    # have wide posteriors and get tried often, while the best arm is chosen more
    # and more as evidence accrues. With no data every arm is Beta(1, 1) =
    # uniform, so the opening pulls are pure (random) exploration.
    class ThompsonSampling < Base
      attr_reader :prior_alpha, :prior_beta

      def initialize(prior_alpha: 1.0, prior_beta: 1.0, rng: Random.new)
        super()
        @prior_alpha = Float(prior_alpha)
        @prior_beta = Float(prior_beta)
        @sampler = Sampler.new(rng)
        @rng = rng
      end

      def choose(arms)
        ensure_arms!(arms)

        scored = arms.map do |arm|
          theta = @sampler.beta(@prior_alpha + arm.successes, @prior_beta + arm.failures)
          [ arm, theta ]
        end
        pick_max(scored, @rng)
      end

      def to_h
        super.merge(prior_alpha: @prior_alpha, prior_beta: @prior_beta)
      end
    end
  end
end
