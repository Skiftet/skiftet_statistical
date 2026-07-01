# frozen_string_literal: true

RSpec.describe SkiftetStatistical::Significance do
  describe ".normal_cdf" do
    it "is 0.5 at 0" do
      expect(described_class.normal_cdf(0)).to eq(0.5)
    end

    it "matches known quantiles" do
      expect(described_class.normal_cdf(1.96)).to be_within(0.001).of(0.975)
      expect(described_class.normal_cdf(-1.96)).to be_within(0.001).of(0.025)
    end
  end

  describe ".two_tailed_p_value" do
    it "is ~0.05 at z = 1.96" do
      expect(described_class.two_tailed_p_value(1.96)).to be_within(0.001).of(0.05)
    end

    it "is 1.0 at z = 0 and symmetric in sign" do
      expect(described_class.two_tailed_p_value(0)).to eq(1.0)
      expect(described_class.two_tailed_p_value(-2.5)).to eq(described_class.two_tailed_p_value(2.5))
    end
  end

  describe ".two_proportion_z_test" do
    it "finds a clear difference highly significant" do
      result = described_class.two_proportion_z_test(90, 100, 10, 100)
      expect(result.statistic).to be < 0          # B (10/100) lower than A (90/100)
      expect(result.p_value).to be < 0.001
      expect(result).to be_significant_99
    end

    it "finds equal rates not significant" do
      result = described_class.two_proportion_z_test(50, 100, 50, 100)
      expect(result.p_value).to eq(1.0)
      expect(result).not_to be_significant
    end

    it "returns nil for an empty group" do
      expect(described_class.two_proportion_z_test(0, 0, 5, 100)).to be_nil
    end

    it "returns nil when pooled variance is zero" do
      expect(described_class.two_proportion_z_test(0, 100, 0, 100)).to be_nil
    end
  end

  describe ".welch_t_test" do
    it "finds a clear mean difference significant" do
      result = described_class.welch_t_test(100.0, 25.0, 500, 120.0, 25.0, 500)
      expect(result.statistic).to be > 0
      expect(result).to be_significant_95
    end

    it "finds equal means not significant" do
      result = described_class.welch_t_test(100.0, 25.0, 500, 100.0, 25.0, 500)
      expect(result.statistic).to eq(0.0)
      expect(result).not_to be_significant
    end

    it "returns nil with too few samples" do
      expect(described_class.welch_t_test(1.0, 1.0, 1, 2.0, 1.0, 5)).to be_nil
    end
  end

  describe SkiftetStatistical::Significance::Result do
    it "exposes confidence and level predicates" do
      result = described_class.new(statistic: 2.0, p_value: 0.04)
      expect(result.confidence).to be_within(1e-9).of(0.96)
      expect(result).to be_significant_95
      expect(result).not_to be_significant_99
    end
  end
end
