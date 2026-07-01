# frozen_string_literal: true

module SkiftetStatistical
  # One option ("arm") the bandit can choose, tracking online reward statistics.
  #
  # Rewards are expected in [0.0, 1.0] for the Bernoulli/Beta policies (Thompson
  # Sampling, UCB1, Epsilon-Greedy treat the mean as a success rate). A binary
  # 0/1 reward is the common case ("did this share convert?"), but any value in
  # [0, 1] works (the summed rewards act as fractional successes).
  class Arm
    attr_reader :name, :pulls, :reward_sum, :reward_square_sum

    def initialize(name, pulls: 0, reward_sum: 0.0, reward_square_sum: 0.0)
      raise ArgumentError, "arm name cannot be nil" if name.nil?

      @name = name
      @pulls = Integer(pulls)
      @reward_sum = Float(reward_sum)
      @reward_square_sum = Float(reward_square_sum)
    end

    # Record one observed reward for this arm. Returns self for chaining.
    def update(reward)
      r = Float(reward)
      @pulls += 1
      @reward_sum += r
      @reward_square_sum += r * r
      self
    end

    # Empirical mean reward (0.0 when never pulled).
    def mean
      return 0.0 if @pulls.zero?

      @reward_sum / @pulls
    end
    alias rate mean

    # Population variance of observed rewards (0.0 with fewer than two pulls).
    def variance
      return 0.0 if @pulls < 2

      m = mean
      [ (@reward_square_sum / @pulls) - (m * m), 0.0 ].max
    end

    # Beta-Bernoulli view: summed rewards are "successes", the remaining pulls
    # "failures". With [0, 1] rewards these can be fractional — Beta handles that.
    def successes
      @reward_sum
    end

    def failures
      [ @pulls - @reward_sum, 0.0 ].max
    end

    def pulled?
      @pulls.positive?
    end

    def to_h
      {
        name: @name,
        pulls: @pulls,
        reward_sum: @reward_sum,
        reward_square_sum: @reward_square_sum
      }
    end

    def self.from_h(hash)
      h = hash.transform_keys(&:to_sym)
      new(
        h.fetch(:name),
        pulls: h.fetch(:pulls, 0),
        reward_sum: h.fetch(:reward_sum, 0.0),
        reward_square_sum: h.fetch(:reward_square_sum, 0.0),
      )
    end
  end
end
