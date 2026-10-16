# frozen_string_literal: true

RSpec.describe SkiftetStatistical::BetaFunction do
  # Relative tolerance against an independent reference. Measured worst on macOS: 2.4e-13 (Beta(41, 361)). The
  # rest is headroom for a libm whose lgamma rounds differently in the last bit.
  def tolerance = 5e-12

  # I_x(a, b) for whole a and b is a binomial tail: sum_{j=a}^{n} C(n, j) x**j (1 - x)**(n - j), n = a + b - 1.
  # Exact in Rationals (x is the Float's exact value).
  def binomial_tail(x, a, b)
    x = Rational(x)
    n = a + b - 1
    (a..n).sum { |j| binomial(n, j) * (x**j) * ((1 - x)**(n - j)) }.to_f
  end

  def binomial(n, k)
    (1..k).reduce(1) { |product, i| product * (n - k + i) / i }
  end

  def relative(actual, expected)
    (actual - expected).abs / expected.abs
  end

  describe ".regularized" do
    it "matches the binomial tail for whole parameters, on both sides of the fraction's switch" do
      [ [ 1, 1 ], [ 2, 3 ], [ 3, 2 ], [ 5, 5 ], [ 1, 20 ], [ 20, 1 ], [ 7, 13 ], [ 31, 71 ], [ 41, 361 ],
        [ 100, 100 ], [ 200, 50 ] ].each do |a, b|
        # Dyadic xs: exact Floats with small denominators, so the Rational powers stay cheap.
        [ 0.0009765625, 0.0078125, 0.0625, 0.1875, 0.3125, 0.5, 0.6875, 0.875, 0.984375 ].each do |x|
          expected = binomial_tail(x, a, b)
          next if expected < 1e-300 # subnormal: no relative precision to compare

          actual = described_class.regularized(x, a.to_f, b.to_f)
          expect(relative(actual, expected)).to be < tolerance, "I_#{x}(#{a}, #{b})"
        end
      end
    end

    it "has the closed forms at a or b = 1 and at a = b = 1/2" do
      [ 0.01, 0.2, 0.5, 0.8, 0.99 ].each do |x|
        [ 0.5, 2.0, 7.0 ].each do |b|
          expect(relative(described_class.regularized(x, 1.0, b), 1.0 - ((1.0 - x)**b))).to be < tolerance
          expect(relative(described_class.regularized(x, b, 1.0), x**b)).to be < tolerance
        end
        arcsine = 2.0 / Math::PI * Math.asin(Math.sqrt(x))
        expect(relative(described_class.regularized(x, 0.5, 0.5), arcsine)).to be < tolerance
      end
    end

    it "is 1/2 at x = 1/2 when alpha = beta" do
      [ 1.0, 2.0, 7.0, 30.0, 100.0 ].each do |a|
        expect(described_class.regularized(0.5, a, a)).to be_within(1e-12).of(0.5)
      end
    end

    it "has I_x(a, b) = 1 - I_(1-x)(b, a)" do
      rng = Random.new(4)
      500.times do
        a = 0.5 + (rng.rand * 300.0)
        b = 0.5 + (rng.rand * 300.0)
        x = rng.rand
        sum = described_class.regularized(x, a, b) + described_class.regularized(1.0 - x, b, a)
        expect(sum).to be_within(1e-12).of(1.0)
      end
    end

    it "is 0 at and below x = 0 and 1 at and above x = 1" do
      expect(described_class.regularized(0.0, 2.0, 3.0)).to eq(0.0)
      expect(described_class.regularized(-0.5, 2.0, 3.0)).to eq(0.0)
      expect(described_class.regularized(1.0, 2.0, 3.0)).to eq(1.0)
      expect(described_class.regularized(1.5, 2.0, 3.0)).to eq(1.0)
    end
  end

  describe ".quantile" do
    it "has the closed forms at a or b = 1 and at a = b = 1/2" do
      [ 0.001, 0.025, 0.3, 0.5, 0.9, 0.975, 0.999 ].each do |p|
        expect(described_class.quantile(p, 1.0, 7.0)).to be_within(1e-12).of(1.0 - ((1.0 - p)**(1.0 / 7.0)))
        expect(described_class.quantile(p, 3.0, 1.0)).to be_within(1e-12).of(p**(1.0 / 3.0))
        expect(described_class.quantile(p, 0.5, 0.5)).to be_within(1e-12).of(Math.sin(Math::PI * p / 2.0)**2)
      end
    end

    it "inverts .regularized" do
      [ 0.001, 0.025, 0.3, 0.5, 0.9, 0.975, 0.999 ].each do |p|
        [ [ 2.0, 5.0 ], [ 41.0, 361.0 ], [ 300.0, 600.0 ], [ 0.7, 3.2 ] ].each do |a, b|
          x = described_class.quantile(p, a, b)
          expect(described_class.regularized(x, a, b)).to be_within(1e-12).of(p)
        end
      end
    end

    it "is 0 at and below p = 0 and 1 at and above p = 1" do
      expect(described_class.quantile(0.0, 2.0, 3.0)).to eq(0.0)
      expect(described_class.quantile(-0.1, 2.0, 3.0)).to eq(0.0)
      expect(described_class.quantile(1.0, 2.0, 3.0)).to eq(1.0)
      expect(described_class.quantile(1.1, 2.0, 3.0)).to eq(1.0)
    end
  end

  it "gives whole-number arguments the same answer as their Floats" do
    expect(described_class.regularized(0.3, 4, 9)).to eq(described_class.regularized(0.3, 4.0, 9.0))
    expect(described_class.quantile(0.975, 41, 361)).to eq(described_class.quantile(0.975, 41.0, 361.0))
  end

  it "refuses parameters that are not finite numbers above 0, and an x or p that is not a number" do
    [ [ 0.0, 1.0 ], [ 1.0, -2.0 ], [ Float::NAN, 1.0 ], [ 1.0, Float::INFINITY ] ].each do |a, b|
      expect { described_class.regularized(0.5, a, b) }.to raise_error(ArgumentError)
      expect { described_class.quantile(0.5, a, b) }.to raise_error(ArgumentError)
    end
    expect { described_class.regularized(Float::NAN, 1.0, 1.0) }.to raise_error(ArgumentError)
    expect { described_class.quantile(Float::NAN, 1.0, 1.0) }.to raise_error(ArgumentError)
  end
end
