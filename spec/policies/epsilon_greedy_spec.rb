# frozen_string_literal: true

RSpec.describe SkiftetStatistical::Policies::EpsilonGreedy do
  def arm(name, pulls:, reward_sum:)
    SkiftetStatistical::Arm.new(name, pulls: pulls, reward_sum: reward_sum)
  end

  it "validates the epsilon range" do
    expect { described_class.new(epsilon: 1.5) }.to raise_error(ArgumentError)
  end

  it "explores unpulled arms first" do
    arms = [arm("a", pulls: 3, reward_sum: 3), SkiftetStatistical::Arm.new("b")]
    policy = described_class.new(epsilon: 0.0, rng: Random.new(1))
    expect(policy.choose(arms).name).to eq("b")
  end

  it "with epsilon 0 exploits the best mean" do
    arms = [arm("a", pulls: 10, reward_sum: 9), arm("b", pulls: 10, reward_sum: 2)]
    policy = described_class.new(epsilon: 0.0, rng: Random.new(1))
    expect(policy.choose(arms).name).to eq("a")
  end

  it "mostly exploits with a small epsilon" do
    arms = [arm("a", pulls: 100, reward_sum: 90), arm("b", pulls: 100, reward_sum: 10)]
    policy = described_class.new(epsilon: 0.1, rng: Random.new(42))
    picks = Array.new(1000) { policy.choose(arms).name }
    expect(picks.count("a")).to be > 800
  end
end
