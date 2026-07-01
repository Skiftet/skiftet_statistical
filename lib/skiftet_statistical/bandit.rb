# frozen_string_literal: true

module SkiftetStatistical
  # The bandit: a named set of arms plus a selection policy. Ask it which arm to
  # play (`#select`), observe a reward, and tell it (`#record`). All state lives
  # in the arms, so a bandit serialises to a plain Hash and back for persistence.
  #
  #   bandit = SkiftetStatistical::Bandit.new(
  #     arms: %w[facebook whatsapp x],
  #     policy: SkiftetStatistical::Policies::ThompsonSampling.new,
  #   )
  #   choice = bandit.select          # => "whatsapp"
  #   bandit.record("whatsapp", 1)    # a conversion
  #   bandit.best_arm                 # => highest empirical mean
  class Bandit
    attr_reader :policy

    def initialize(arms: [], policy: nil)
      @arms = {}
      Array(arms).each { |a| add_arm(a) }
      @policy = policy || Policies::ThompsonSampling.new
    end

    # Add an arm by name (String/Symbol) or an existing Arm. Idempotent — an
    # already-known name is left untouched. Returns the arm.
    def add_arm(arm)
      a = arm.is_a?(Arm) ? arm : Arm.new(arm)
      @arms[a.name] ||= a
    end

    def arm(name)
      @arms.fetch(name) { raise Error, "unknown arm: #{name.inspect}" }
    end

    def arms
      @arms.values
    end

    def arm_names
      @arms.keys
    end

    # Choose an arm to play. Returns the arm's name.
    def select
      raise Error, "bandit has no arms" if @arms.empty?

      @policy.choose(@arms.values).name
    end

    # Record an observed reward for the named arm. Returns self for chaining.
    def record(name, reward)
      arm(name).update(reward)
      self
    end

    # The arm with the highest empirical mean — the current exploitation choice.
    def best_arm
      return nil if @arms.empty?

      @arms.values.max_by(&:mean)&.name
    end

    # Per-arm summary: { name => { pulls:, mean:, reward_sum: } }.
    def stats
      @arms.transform_values do |a|
        { pulls: a.pulls, mean: a.mean, reward_sum: a.reward_sum }
      end
    end

    def to_h
      { arms: @arms.values.map(&:to_h), policy: @policy.to_h }
    end

    # Rebuild a bandit's ARM STATE from a hash produced by #to_h. The policy is
    # not reconstructed from its serialised config (policies carry an RNG); pass
    # the policy instance you want to run with.
    def self.from_h(hash, policy: nil)
      h = hash.transform_keys(&:to_sym)
      arms = Array(h[:arms]).map { |ah| Arm.from_h(ah) }
      new(arms: arms, policy: policy)
    end
  end
end
