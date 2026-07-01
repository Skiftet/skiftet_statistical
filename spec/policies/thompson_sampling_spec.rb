# frozen_string_literal: true

RSpec.describe SkiftetStatistical::Policies::ThompsonSampling do
  def arm(name, pulls:, reward_sum:)
    SkiftetStatistical::Arm.new(name, pulls: pulls, reward_sum: reward_sum)
  end

  it "is deterministic for a fixed seed" do
    arms = [arm("a", pulls: 10, reward_sum: 7), arm("b", pulls: 10, reward_sum: 3)]
    p1 = described_class.new(rng: Random.new(99))
    p2 = described_class.new(rng: Random.new(99))
    expect(p1.choose(arms).name).to eq(p2.choose(arms).name)
  end

  it "favours the higher-converting arm" do
    arms = [arm("a", pulls: 200, reward_sum: 160), arm("b", pulls: 200, reward_sum: 40)]
    policy = described_class.new(rng: Random.new(7))
    picks = Array.new(500) { policy.choose(arms).name }
    expect(picks.count("a")).to be > picks.count("b")
  end

  it "still explores the worse arm sometimes" do
    arms = [arm("a", pulls: 30, reward_sum: 20), arm("b", pulls: 30, reward_sum: 12)]
    policy = described_class.new(rng: Random.new(7))
    picks = Array.new(500) { policy.choose(arms).name }
    expect(picks.count("b")).to be > 0
  end

  it "cold start (no data) spreads across all arms" do
    arms = %w[a b c].map { |n| SkiftetStatistical::Arm.new(n) }
    policy = described_class.new(rng: Random.new(3))
    picks = Array.new(300) { policy.choose(arms).name }
    expect(picks.uniq).to contain_exactly("a", "b", "c")
  end
end
