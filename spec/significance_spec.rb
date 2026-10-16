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

  describe ".normal_quantile" do
    it "matches known quantiles" do
      { 0.975 => 1.959963984540054, 0.025 => -1.959963984540054, 0.95 => 1.6448536269514722,
        0.999 => 3.090232306167813, 1e-10 => -6.361340902404056 }.each do |p, z|
        expect(described_class.normal_quantile(p)).to be_within(1e-12 * z.abs).of(z)
      end
      expect(described_class.normal_quantile(0.5)).to be_within(1e-15).of(0.0)
    end

    it "inverts the normal distribution function, deep into both tails" do
      [ 1e-300, 1e-12, 1e-8, 0.001, 0.3, 0.5, 0.9, 0.999999 ].each do |p|
        z = described_class.normal_quantile(p)
        # erfc keeps the lower tail's precision, where 0.5 * (1 + erf) cancels.
        expect(0.5 * Math.erfc(-z / Math.sqrt(2.0))).to be_within(1e-12 * p).of(p)
      end
    end

    it "is -Infinity at 0 and Infinity at 1" do
      expect(described_class.normal_quantile(0)).to eq(-Float::INFINITY)
      expect(described_class.normal_quantile(1)).to eq(Float::INFINITY)
    end

    it "refuses a p outside [0, 1]" do
      [ -0.1, 1.1, Float::NAN ].each do |p|
        expect { described_class.normal_quantile(p) }.to raise_error(ArgumentError)
      end
    end
  end

  describe ".chi_squared_survival" do
    it "has the closed forms at 1 and 2 degrees of freedom" do
      [ 0.01, 0.5, 1.0, 2.0, 3.9, 4.0, 10.0, 50.0, 100.0 ].each do |statistic|
        expect(described_class.chi_squared_survival(statistic, 2))
          .to be_within(1e-12 * Math.exp(-statistic / 2.0)).of(Math.exp(-statistic / 2.0))
        one = Math.erfc(Math.sqrt(statistic / 2.0))
        expect(described_class.chi_squared_survival(statistic, 1)).to be_within(1e-12 * one).of(one)
      end
    end

    it "is 0.05 at the 95 % critical values and 0.01 at the 99 % one" do
      { [ 3.841458820694124, 1 ] => 0.05, [ 5.991464547107979, 2 ] => 0.05, [ 11.070497693516351, 5 ] => 0.05,
        [ 6.6348966010212145, 1 ] => 0.01 }.each do |(statistic, dof), p|
        expect(described_class.chi_squared_survival(statistic, dof)).to be_within(1e-12 * p).of(p)
      end
    end

    it "is 1 for a statistic of 0 or below" do
      expect(described_class.chi_squared_survival(0.0, 3)).to eq(1.0)
      expect(described_class.chi_squared_survival(-1.0, 3)).to eq(1.0)
    end
  end

  describe ".sample_ratio_p_value" do
    it "tests the counts against the split on k - 1 degrees of freedom" do
      # Statistic 2 * 200**2 / 500 = 160 on 1 degree of freedom: Q(1/2, 80) = erfc(sqrt(80)).
      two = described_class.sample_ratio_p_value(observed: { control: 700, a: 300 },
                                                 expected: { control: 500.0, a: 500.0 })
      expect(two).to be_within(1e-12 * Math.erfc(Math.sqrt(80.0))).of(Math.erfc(Math.sqrt(80.0)))
      # Statistic (50**2 + 50**2 + 0) / 500 = 10 on 2 degrees of freedom: e**-5.
      three = described_class.sample_ratio_p_value(observed: { control: 550, a: 450, b: 500 },
                                                   expected: { control: 500.0, a: 500.0, b: 500.0 })
      expect(three).to be_within(1e-12 * Math.exp(-5.0)).of(Math.exp(-5.0))
    end

    it "is 1 when the counts are the split" do
      expect(described_class.sample_ratio_p_value(observed: { control: 100, a: 100 },
                                                  expected: { control: 100.0, a: 100.0 })).to eq(1.0)
    end

    it "does not test under minimum_total observations (200 unless given)" do
      fixture = { observed: { control: 90, a: 109 }, expected: { control: 99.5, a: 99.5 } }
      expect(described_class.sample_ratio_p_value(**fixture)).to be_nil
      expect(described_class.sample_ratio_p_value(**fixture, minimum_total: 199)).to be_a(Float)
    end

    it "is 0 when an arm expected to get nothing got some, whether it is expected 0 or not listed" do
      expect(described_class.sample_ratio_p_value(observed: { control: 150, a: 50 },
                                                  expected: { control: 200.0, a: 0.0 })).to eq(0.0)
      expect(described_class.sample_ratio_p_value(observed: { control: 150, a: 50, b: 10 },
                                                  expected: { control: 150.0, a: 60.0 })).to eq(0.0)
    end

    it "does not test fewer than two arms expected to get traffic" do
      expect(described_class.sample_ratio_p_value(observed: { control: 200, a: 0 },
                                                  expected: { control: 200.0, a: 0.0 })).to be_nil
    end

    it "reads expected by observed's keys, strings as well as symbols" do
      symbols = described_class.sample_ratio_p_value(observed: { control: 5_300, a: 4_700 },
                                                     expected: { control: 5_000.0, a: 5_000.0 })
      strings = described_class.sample_ratio_p_value(observed: { "control" => 5_300, "a" => 4_700 },
                                                     expected: { "control" => 5_000.0, "a" => 5_000.0 })
      expect(strings).to eq(symbols)
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
