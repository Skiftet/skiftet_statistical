# frozen_string_literal: true

RSpec.describe SkiftetStatistical::Policies::Softmax do
  def arm(name, pulls:, reward_sum:)
    SkiftetStatistical::Arm.new(name, pulls: pulls, reward_sum: reward_sum)
  end

  it "validates the temperature" do
    expect { described_class.new(temperature: 0) }.to raise_error(ArgumentError)
  end

  it "is near-greedy at low temperature" do
    arms = [arm("a", pulls: 50, reward_sum: 45), arm("b", pulls: 50, reward_sum: 5)]
    policy = described_class.new(temperature: 0.05, rng: Random.new(2))
    picks = Array.new(500) { policy.choose(arms).name }
    expect(picks.count("a")).to be > 450
  end

  it "explores broadly at high temperature" do
    arms = [arm("a", pulls: 50, reward_sum: 45), arm("b", pulls: 50, reward_sum: 5)]
    policy = described_class.new(temperature: 100.0, rng: Random.new(2))
    picks = Array.new(1000) { policy.choose(arms).name }
    expect(picks.count("b")).to be > 300
  end
end
