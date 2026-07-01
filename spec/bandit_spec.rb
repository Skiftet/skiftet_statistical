# frozen_string_literal: true

RSpec.describe SkiftetStatistical::Bandit do
  it "defaults to Thompson Sampling" do
    bandit = described_class.new(arms: %w[a b])
    expect(bandit.policy).to be_a(SkiftetStatistical::Policies::ThompsonSampling)
  end

  it "adds arms idempotently" do
    bandit = described_class.new(arms: %w[a b])
    bandit.add_arm("a")
    bandit.add_arm("c")
    expect(bandit.arm_names).to contain_exactly("a", "b", "c")
  end

  it "selects an existing arm name" do
    bandit = described_class.new(arms: %w[a b c])
    expect(bandit.arm_names).to include(bandit.select)
  end

  it "records rewards and reports best_arm" do
    bandit = described_class.new(arms: %w[a b])
    5.times { bandit.record("a", 1) }
    5.times { bandit.record("b", 0) }
    expect(bandit.best_arm).to eq("a")
  end

  it "raises on an unknown arm" do
    bandit = described_class.new(arms: %w[a])
    expect { bandit.record("z", 1) }.to raise_error(SkiftetStatistical::Error)
  end

  it "raises when selecting with no arms" do
    expect { described_class.new.select }.to raise_error(SkiftetStatistical::Error)
  end

  it "round-trips arm state through to_h/from_h" do
    bandit = described_class.new(arms: %w[a b])
    3.times { bandit.record("a", 1) }
    bandit.record("b", 0)

    restored = described_class.from_h(bandit.to_h, policy: SkiftetStatistical::Policies::UCB1.new)
    expect(restored.arm("a").pulls).to eq(3)
    expect(restored.arm("a").reward_sum).to eq(3.0)
    expect(restored.best_arm).to eq("a")
    expect(restored.policy).to be_a(SkiftetStatistical::Policies::UCB1)
  end

  it "exposes per-arm stats" do
    bandit = described_class.new(arms: %w[a])
    bandit.record("a", 1)
    expect(bandit.stats["a"]).to include(pulls: 1, reward_sum: 1.0)
  end

  it "accepts pre-built Arm objects" do
    arm = SkiftetStatistical::Arm.new("seed", pulls: 4, reward_sum: 3)
    bandit = described_class.new(arms: [arm])
    expect(bandit.arm("seed").pulls).to eq(4)
  end
end
