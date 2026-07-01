# frozen_string_literal: true

RSpec.describe SkiftetStatistical::Policies::UCB1 do
  def arm(name, pulls:, reward_sum:)
    SkiftetStatistical::Arm.new(name, pulls: pulls, reward_sum: reward_sum)
  end

  it "validates c" do
    expect { described_class.new(c: 0) }.to raise_error(ArgumentError)
  end

  it "pulls each arm once first" do
    arms = [arm("a", pulls: 5, reward_sum: 5), SkiftetStatistical::Arm.new("b")]
    policy = described_class.new(rng: Random.new(1))
    expect(policy.choose(arms).name).to eq("b")
  end

  it "favours an uncertain arm via its confidence bonus" do
    # b has a slightly lower mean but far fewer pulls -> a wider bound.
    arms = [arm("a", pulls: 1000, reward_sum: 600), arm("b", pulls: 10, reward_sum: 5)]
    policy = described_class.new(c: 2.0, rng: Random.new(1))
    expect(policy.choose(arms).name).to eq("b")
  end

  it "prefers the clearly-better arm when pulls are comparable" do
    arms = [arm("a", pulls: 100, reward_sum: 90), arm("b", pulls: 100, reward_sum: 20)]
    policy = described_class.new(c: 2.0, rng: Random.new(1))
    expect(policy.choose(arms).name).to eq("a")
  end
end
