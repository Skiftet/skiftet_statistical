# frozen_string_literal: true

RSpec.describe SkiftetStatistical::Arm do
  it "starts empty" do
    arm = described_class.new("a")
    expect(arm.pulls).to eq(0)
    expect(arm.mean).to eq(0.0)
    expect(arm).not_to be_pulled
  end

  it "tracks pulls and mean" do
    arm = described_class.new("a")
    arm.update(1).update(0).update(1)
    expect(arm.pulls).to eq(3)
    expect(arm.reward_sum).to eq(2.0)
    expect(arm.mean).to be_within(1e-9).of(2.0 / 3.0)
    expect(arm).to be_pulled
  end

  it "computes successes/failures for the Beta-Bernoulli view" do
    arm = described_class.new("a")
    arm.update(1).update(0).update(0)
    expect(arm.successes).to eq(1.0)
    expect(arm.failures).to eq(2.0)
  end

  it "computes variance" do
    arm = described_class.new("a")
    [1, 0, 1, 0].each { |r| arm.update(r) }
    expect(arm.variance).to be_within(1e-9).of(0.25)
  end

  it "round-trips through to_h/from_h" do
    arm = described_class.new("a")
    arm.update(1).update(0)
    restored = described_class.from_h(arm.to_h)
    expect(restored.name).to eq("a")
    expect(restored.pulls).to eq(2)
    expect(restored.reward_sum).to eq(1.0)
    expect(restored.mean).to eq(arm.mean)
  end

  it "from_h accepts string keys" do
    restored = described_class.from_h("name" => "x", "pulls" => 4, "reward_sum" => 3.0)
    expect(restored.name).to eq("x")
    expect(restored.pulls).to eq(4)
  end

  it "rejects a nil name" do
    expect { described_class.new(nil) }.to raise_error(ArgumentError)
  end
end
