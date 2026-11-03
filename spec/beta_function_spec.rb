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

  # Both parameters from 1 000 to 50 000, where the examples above (parameters up to a few hundred) do not reach. Here
  # the front factor's lgamma terms each reach about n log n (n = alpha + beta), so their roundings bound the answer to
  # about n log n machine epsilons, relative; the bound below is four times that (measured worst on macOS: 0.9 of it,
  # against the exact reference). At n = 100 000 it is about 1e-9.
  describe "at large parameters (each 1 000 to 50 000)" do
    def large_tolerance(alpha, beta)
      n = alpha + beta
      4.0 * n * Math.log(n) * Float::EPSILON
    end

    # Pairs of whole parameters: skewed both ways, near equal, and both ends of the range.
    def large_pairs
      [ [ 1_000, 1_500 ], [ 2_500, 7_500 ], [ 6_000, 1_200 ], [ 9_000, 31_000 ], [ 20_000, 30_000 ],
        [ 1_000, 50_000 ], [ 50_000, 2_000 ] ]
    end

    # Seeded draws of non-whole parameters across the range, each with a uniform.
    def drawn_pairs(count)
      rng = Random.new(11)
      Array.new(count) { [ 1_000.0 + (rng.rand * 49_000.0), 1_000.0 + (rng.rand * 49_000.0), rng.rand ] }
    end

    # n! / (k! (n - k)!) by two product trees, exact.
    def choose(n, k)
      k = [ k, n - k ].min
      product(n - k + 1, n) / product(1, k)
    end

    def product(from, to)
      return 1 if from > to
      return (from..to).reduce(:*) if to - from < 16

      middle = (from + to) / 2
      product(from, middle) * product(middle + 1, to)
    end

    # I_x(a, b) for whole a and b at the dyadic x = numerator / 2**bits, independent of the function under test:
    # P(J >= a) for J ~ Binomial(a + b - 1, x), each term an exact Integer over 2**(bits (a + b - 1)). The sum starts
    # at the term next to j = a on the side away from the mode and stops once a term is below 2**-80 of the sum (the
    # rest falls geometrically, so it moves no Float digit): below the mode it is the upper tail summed upward from
    # j = a, above it 1 less the lower tail summed downward from j = a - 1.
    def exact_tail(a, b, numerator, bits)
      n = a + b - 1
      rest = (1 << bits) - numerator
      upward = numerator * n < a * (1 << bits)
      j = upward ? a : a - 1
      term = choose(n, j) * (numerator**j) * (rest**(n - j))
      sum = 0
      loop do
        sum += term
        break if upward ? j == n : j.zero?

        if upward
          term = term * (n - j) * numerator / ((j + 1) * rest)
          j += 1
        else
          term = term * j * rest / ((n - j + 1) * numerator)
          j -= 1
        end
        break if (term << 80) < sum
      end
      tail = top_bits(sum, n * bits)
      upward ? tail : 1.0 - tail
    end

    # sum * 2**-shift as a Float, from the sum's top 64 bits.
    def top_bits(sum, shift)
      dropped = [ sum.bit_length - 64, 0 ].max
      Math.ldexp((sum >> dropped).to_f, dropped - shift)
    end

    it "matches the exact binomial tail for whole parameters, at a point on either side of the mean" do
      large_pairs.each do |a, b|
        mean = a.fdiv(a + b)
        sd = Math.sqrt(a * b / (((a + b)**2) * (a + b + 1.0)))
        [ -2.0, 1.5 ].each do |z|
          numerator = ((mean + (z * sd)) * 1024).round
          expected = exact_tail(a, b, numerator, 10)
          actual = described_class.regularized(numerator / 1024.0, a.to_f, b.to_f)
          expect(relative(actual, expected)).to be < large_tolerance(a, b), "I_#{numerator}/1024(#{a}, #{b})"
        end
      end
    end

    it "has I_x(a, b) + I_(1-x)(b, a) = 1 across the bulk" do
      drawn_pairs(100).each do |a, b, u|
        mean = a / (a + b)
        sd = Math.sqrt(a * b / (((a + b)**2) * (a + b + 1.0)))
        x = mean + (((8.0 * u) - 4.0) * sd)
        sum = described_class.regularized(x, a, b) + described_class.regularized(1.0 - x, b, a)
        expect(sum).to be_within(large_tolerance(a, b)).of(1.0), "x = #{x}, a = #{a}, b = #{b}"
      end
    end

    it "is 1/2 at x = 1/2 when alpha = beta, alpha up to 50 000" do
      drawn_pairs(100).each do |a, _b, _u|
        expect(described_class.regularized(0.5, a, a)).to be_within(large_tolerance(a, a)).of(0.5), "a = #{a}"
      end
    end

    it "has a quantile that .regularized maps back to its probability" do
      drawn_pairs(40).each do |a, b, _u|
        [ 0.001, 0.025, 0.5, 0.975, 0.999 ].each do |p|
          x = described_class.quantile(p, a, b)
          expect(described_class.regularized(x, a, b)).to be_within(large_tolerance(a, b)).of(p), "#{p}, #{a}, #{b}"
        end
      end
    end

    # The normal approximation with its first correction (Cornish-Fisher): the q quantile of Beta(a, b) is about
    # mean + sd * (z + skewness * (z**2 - 1) / 6), z the standard normal quantile at q. The terms left out are of
    # order skewness**2 and the excess kurtosis, below 1e-3 sd over this range (measured worst: 1.8e-4 sd).
    it "puts the 2.5 % and 97.5 % quantiles at the corrected normal approximation, within 0.001 sd" do
      z = SkiftetStatistical::Significance.normal_quantile(0.975)
      drawn_pairs(40).each do |a, b, _u|
        n = a + b
        mean = a / n
        sd = Math.sqrt(a * b / ((n**2) * (n + 1.0)))
        skewness = 2.0 * (b - a) * Math.sqrt(n + 1.0) / ((n + 2.0) * Math.sqrt(a * b))
        shift = skewness * ((z**2) - 1.0) / 6.0
        lower = described_class.quantile(0.025, a, b)
        upper = described_class.quantile(0.975, a, b)

        expect(lower).to be_within(0.001 * sd).of(mean + ((-z + shift) * sd)), "a = #{a}, b = #{b}"
        expect(upper).to be_within(0.001 * sd).of(mean + ((z + shift) * sd)), "a = #{a}, b = #{b}"
      end
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
