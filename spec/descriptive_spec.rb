# frozen_string_literal: true

RSpec.describe SkiftetStatistical::Descriptive do
  describe ".mean" do
    it "averages the values" do
      expect(described_class.mean([1, 2, 3, 4])).to eq(2.5)
    end

    it "is 0.0 for an empty collection" do
      expect(described_class.mean([])).to eq(0.0)
    end
  end

  describe ".variance" do
    it "uses Bessel's correction by default (sample)" do
      expect(described_class.variance([1, 2, 3, 4, 5])).to be_within(1e-9).of(2.5)
    end

    it "divides by n for population variance" do
      expect(described_class.variance([1, 2, 3, 4, 5], sample: false)).to be_within(1e-9).of(2.0)
    end

    it "is 0.0 for fewer than two values" do
      expect(described_class.variance([7])).to eq(0.0)
    end
  end

  describe ".standard_deviation" do
    it "is the root of the variance" do
      expect(described_class.standard_deviation([1, 2, 3, 4, 5])).to be_within(1e-9).of(Math.sqrt(2.5))
    end
  end

  describe ".percentile" do
    let(:values) { (1..10).to_a }

    it "interpolates the median" do
      expect(described_class.percentile(values, 50)).to be_within(1e-9).of(5.5)
    end

    it "returns the extremes at 0 and 100" do
      expect(described_class.percentile(values, 0)).to eq(1.0)
      expect(described_class.percentile(values, 100)).to eq(10.0)
    end

    it "is nil for an empty collection" do
      expect(described_class.percentile([], 50)).to be_nil
    end
  end

  describe ".median" do
    it "is the 50th percentile" do
      expect(described_class.median([3, 1, 2])).to be_within(1e-9).of(2.0)
    end
  end
end
