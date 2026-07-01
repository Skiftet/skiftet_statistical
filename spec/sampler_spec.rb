# frozen_string_literal: true

RSpec.describe SkiftetStatistical::Sampler do
  subject(:sampler) { described_class.new(Random.new(12_345)) }

  it "is deterministic for a fixed seed" do
    a = described_class.new(Random.new(7)).beta(2, 5)
    b = described_class.new(Random.new(7)).beta(2, 5)
    expect(a).to eq(b)
  end

  it "draws gamma > 0 across shapes (including shape < 1)" do
    [0.3, 1.0, 2.5, 50.0].each do |shape|
      100.times { expect(sampler.gamma(shape)).to be > 0.0 }
    end
  end

  it "draws beta within [0, 1]" do
    200.times do
      v = sampler.beta(2.0, 3.0)
      expect(v).to be_between(0.0, 1.0).inclusive
    end
  end

  it "Beta(1, 1) averages ~0.5 (uniform)" do
    n = 4000
    mean = Array.new(n) { sampler.beta(1.0, 1.0) }.sum / n
    expect(mean).to be_within(0.03).of(0.5)
  end

  it "Beta(a, b) mean approximates a / (a + b)" do
    n = 5000
    mean = Array.new(n) { sampler.beta(8.0, 2.0) }.sum / n
    expect(mean).to be_within(0.03).of(0.8)
  end

  it "rejects a non-positive shape" do
    expect { sampler.gamma(0) }.to raise_error(ArgumentError)
  end
end
