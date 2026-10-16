# frozen_string_literal: true

RSpec.describe SkiftetStatistical::Gamma do
  # Relative tolerance against an independent reference. Measured worst on macOS: 2e-13 (a = 200). The rest is
  # headroom for a libm whose lgamma or exp rounds differently in the last bit.
  def tolerance = 5e-12

  # Q(n, x) = e**-x * sum_{k < n} x**k / k! for a whole n (the Poisson tail), the sum exact in Rationals: a
  # reference independent of the series and fraction under test, good to a few ulps.
  def upper_for_whole(n, x)
    term = Rational(1)
    sum = Rational(1)
    (1...n).each do |k|
      term = term * Rational(x) / k
      sum += term
    end
    Math.exp(-x) * sum.to_f
  end

  # P(n, x) = e**-x * sum_{k >= n} x**k / k!, summed until a term no longer counts.
  def lower_for_whole(n, x)
    term = (1..n).reduce(Rational(1)) { |product, k| product * Rational(x) / k }
    sum = Rational(0)
    k = n
    loop do
      sum += term
      k += 1
      term = term * Rational(x) / k
      break if term < sum * Rational(1, 10**20)
    end
    Math.exp(-x) * sum.to_f
  end

  def relative(actual, expected)
    (actual - expected).abs / expected.abs
  end

  # Whole shapes, and xs on both sides of x = a + 1 for each, so both the series and the fraction run. The xs
  # are dyadic (exact Floats with small denominators), so the Rational sums stay cheap.
  def whole_points
    [ 1, 2, 3, 5, 10, 30, 100, 200 ].product(
      [ 0.0078125, 0.5, 1.0, 1.875, 2.0, 2.5, 5.0, 9.0, 10.0, 11.5, 25.0, 31.0, 60.0, 110.0, 180.0, 199.0, 201.0,
        250.0 ]
    )
  end

  describe ".regularized_upper" do
    it "matches the Poisson tail sum for a whole a, on both sides of x = a + 1" do
      whole_points.each do |a, x|
        expected = upper_for_whole(a, x)
        next if expected < 1e-300 # subnormal: no relative precision to compare

        expect(relative(described_class.regularized_upper(a.to_f, x), expected)).to be < tolerance, "Q(#{a}, #{x})"
      end
    end

    it "is erfc(sqrt(x)) at a = 1/2" do
      [ 0.001, 0.1, 0.5, 1.0, 1.4, 1.5, 2.0, 5.0, 20.0, 100.0 ].each do |x|
        expect(relative(described_class.regularized_upper(0.5, x), Math.erfc(Math.sqrt(x)))).to be < tolerance
      end
    end

    it "is 1 at and below x = 0, and 0 at x = Infinity" do
      expect(described_class.regularized_upper(2.0, 0.0)).to eq(1.0)
      expect(described_class.regularized_upper(2.0, -3.0)).to eq(1.0)
      expect(described_class.regularized_upper(2.0, Float::INFINITY)).to eq(0.0)
    end
  end

  describe ".regularized_lower" do
    it "matches the Poisson sum for a whole a, on both sides of x = a + 1" do
      whole_points.each do |a, x|
        expected = lower_for_whole(a, x)
        next if expected < 1e-300

        expect(relative(described_class.regularized_lower(a.to_f, x), expected)).to be < tolerance, "P(#{a}, #{x})"
      end
    end

    it "is erf(sqrt(x)) at a = 1/2" do
      [ 0.001, 0.1, 0.5, 1.0, 1.4, 1.5, 2.0, 5.0, 20.0 ].each do |x|
        expect(relative(described_class.regularized_lower(0.5, x), Math.erf(Math.sqrt(x)))).to be < tolerance
      end
    end

    it "is 1 - e**-x at a = 1" do
      [ 0.25, 1.0, 1.99, 2.0, 7.5, 30.0 ].each do |x|
        expect(relative(described_class.regularized_lower(1.0, x), 1.0 - Math.exp(-x))).to be < tolerance
      end
    end

    it "is near 1/2 + 1/(3 sqrt(2 pi a)) at x = a for a large a, where the series runs thousands of terms" do
      a = 1_000_000.0
      expect(described_class.regularized_lower(a, a)).to be_within(1e-9).of(0.5 + (1.0 / (3.0 * Math.sqrt(2.0 * Math::PI * a))))
    end

    it "is 0 at and below x = 0, and 1 at x = Infinity" do
      expect(described_class.regularized_lower(2.0, 0.0)).to eq(0.0)
      expect(described_class.regularized_lower(2.0, -3.0)).to eq(0.0)
      expect(described_class.regularized_lower(2.0, Float::INFINITY)).to eq(1.0)
    end
  end

  it "has P + Q = 1" do
    rng = Random.new(3)
    2_000.times do
      a = 0.5 + (rng.rand * 300.0)
      x = rng.rand * 600.0
      sum = described_class.regularized_lower(a, x) + described_class.regularized_upper(a, x)
      expect(sum).to be_within(1e-15).of(1.0)
    end
  end

  it "gives a whole-number argument the same answer as its Float" do
    expect(described_class.regularized_lower(3, 2)).to eq(described_class.regularized_lower(3.0, 2.0))
    expect(described_class.regularized_upper(3, 7)).to eq(described_class.regularized_upper(3.0, 7.0))
  end

  it "refuses a shape that is not a finite number above 0, and an x that is not a number" do
    [ 0.0, -1.0, Float::NAN, Float::INFINITY ].each do |a|
      expect { described_class.regularized_lower(a, 1.0) }.to raise_error(ArgumentError)
      expect { described_class.regularized_upper(a, 1.0) }.to raise_error(ArgumentError)
    end
    expect { described_class.regularized_lower(1.0, Float::NAN) }.to raise_error(ArgumentError)
    expect { described_class.regularized_upper(1.0, Float::NAN) }.to raise_error(ArgumentError)
  end
end
