# frozen_string_literal: true

# Acceptance for SL10: the exact probability that each of several Betas is the largest, the seedless pick it
# describes, the greedy pick, and BetaFunction.log_density. The reference integral is this file's own: the same
# integrand, written here, by composite Gauss-Legendre at 200 nodes a panel (nodes found here by Newton's method) with
# tails at 1e-14, on panels broken at every member's quantiles. Closed forms come from the Beta(a, 1) family, whose
# distribution function is x**a, so P(member c is largest) = a_c / sum of a. None of it calls BetaArgmax,
# GaussLegendre or log_density; it uses 0.2.0's BetaFunction.regularized and .quantile, which SL10 leaves unchanged.
RSpec.describe "SkiftetStatistical::BetaArgmax" do
  def argmax = SkiftetStatistical::BetaArgmax
  def beta_function = SkiftetStatistical::BetaFunction

  # Gauss-Legendre nodes and weights on [-1, 1] by Newton's method on P_n, from the cosine first guess.
  def reference_nodes(n)
    (1..n).map do |i|
      x = Math.cos(Math::PI * (i - 0.25) / (n + 0.5))
      derivative = 1.0
      60.times do
        value, derivative = legendre(n, x)
        step = value / derivative
        x -= step
        break if step.abs < 1e-16
      end
      _value, derivative = legendre(n, x)
      [ x, 2.0 / ((1.0 - (x * x)) * derivative * derivative) ]
    end
  end

  # P_n(x) by the three-term recurrence and its derivative n (x P_n - P_(n-1)) / (x**2 - 1).
  def legendre(n, x)
    previous = 1.0
    current = x
    (2..n).each { |j| previous, current = current, ((((2 * j) - 1) * x * current) - ((j - 1) * previous)) / j }
    previous = 1.0 if n == 1
    [ current, n * ((x * current) - previous) / ((x * x) - 1.0) ]
  end

  def reference_density(x, alpha, beta)
    Math.exp(
      Math.lgamma(alpha + beta)[0] - Math.lgamma(alpha)[0] - Math.lgamma(beta)[0] +
      ((alpha - 1.0) * Math.log(x)) + ((beta - 1.0) * Math.log(1.0 - x))
    )
  end

  # SL10's integral, by the reference rule: normalised to sum 1, then mixed with epsilon / k.
  def reference_probabilities(params, epsilon: 0.0, tail: 1e-14, nodes_per_panel: 200)
    nodes = reference_nodes(nodes_per_panel)
    levels = [ tail, 1e-4, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0 - 1e-4, 1.0 - tail ]
    low = params.map { |alpha, beta| beta_function.quantile(tail, alpha, beta) }.min
    high = params.map { |alpha, beta| beta_function.quantile(1.0 - tail, alpha, beta) }.max
    breaks = params.flat_map { |alpha, beta| levels.map { |level| beta_function.quantile(level, alpha, beta) } }
    points = (breaks + [ low, high ]).select { |x| x.between?(low, high) }.uniq.sort

    sums = Array.new(params.size, 0.0)
    points.each_cons(2) do |left, right|
      half = (right - left) / 2.0
      middle = (right + left) / 2.0
      nodes.each do |node, weight|
        x = middle + (half * node)
        next unless x.positive? && x < 1.0

        cdfs = params.map { |alpha, beta| beta_function.regularized(x, alpha, beta) }
        params.each_with_index do |(alpha, beta), c|
          others = 1.0
          cdfs.each_with_index { |cdf, j| others *= cdf unless j == c }
          sums[c] += weight * half * reference_density(x, alpha, beta) * others
        end
      end
    end
    total = sums.sum
    sums.map { |sum| (epsilon / params.size) + ((1.0 - epsilon) * sum / total) }
  end

  def expect_close(actual, expected, tolerance)
    expect(actual.size).to eq(expected.size)
    actual.zip(expected).each_with_index do |(a, e), i|
      expect(a).to be_within(tolerance).of(e), "member #{i}: #{a} vs #{e}"
    end
  end

  describe "the constants" do
    it "SL10 names the tail, the nine panel breaks and the nodes a panel" do
      expect(argmax::TAIL).to eq(1e-10)
      expect(argmax::NODES).to eq(20)
      expect_close(argmax::BREAKS, [ 1e-10, 1e-4, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0 - 1e-4, 1.0 - 1e-10 ], 1e-16)
    end
  end

  describe ".probabilities" do
    # Shapes as callers hold them: flat and sharp (alpha + beta up to 2 * 10**5), near-ties, a small alpha.
    def fixtures
      [
        [ [ 2.0, 8.0 ], [ 3.0, 7.0 ], [ 1.5, 30.0 ] ],
        [ [ 41.0, 361.0 ], [ 52.0, 350.0 ], [ 1.5, 30.0 ] ],
        [ [ 20_000.0, 180_000.0 ], [ 20_200.0, 179_800.0 ] ],
        [ [ 20_000.0, 180_000.0 ], [ 1.0, 9.0 ], [ 101.0, 899.0 ] ],
        [ [ 1_000.0, 99_000.0 ], [ 1_050.0, 98_950.0 ], [ 990.0, 99_010.0 ], [ 12.0, 988.0 ], [ 0.6, 50.0 ] ],
        [ [ 100_000.0, 100_000.0 ], [ 100_300.0, 99_700.0 ] ],
        [ [ 0.03, 29.97 ], [ 0.3, 29.7 ], [ 1.2, 28.8 ] ],
        [ [ 0.9, 29.1 ], [ 3.0, 27.0 ], [ 1.5, 40.0 ] ]
      ]
    end

    it "SL10 is within 1e-8 absolute of the reference integral (200 nodes a panel, tails 1e-14) on every fixture" do
      fixtures.each do |params|
        actual = argmax.probabilities(params)
        expect_close(actual, reference_probabilities(params), 1e-8)
        expect(actual.sum).to be_within(1e-12).of(1.0)
      end
    end

    it "SL10 mixes in epsilon / k, every member keeping at least that much" do
      params = [ [ 41.0, 361.0 ], [ 52.0, 350.0 ], [ 1.5, 30.0 ] ]
      actual = argmax.probabilities(params, epsilon: 0.1)
      expect_close(actual, reference_probabilities(params, epsilon: 0.1), 1e-8)
      expect(actual.sum).to be_within(1e-12).of(1.0)
      expect(actual.min).to be >= (0.1 / 3) - 1e-12
    end

    it "SL10 matches the closed forms of the Beta(a, 1) family" do
      # Beta(2, 1) against Beta(1, 1): integral of 2x * x = 2/3. Three: a_c / (1 + 2 + 3).
      expect_close(argmax.probabilities([ [ 1.0, 1.0 ], [ 2.0, 1.0 ] ]), [ 1.0 / 3, 2.0 / 3 ], 1e-9)
      expect_close(argmax.probabilities([ [ 1.0, 1.0 ], [ 2.0, 1.0 ], [ 3.0, 1.0 ] ]), [ 1.0 / 6, 2.0 / 6, 3.0 / 6 ], 1e-9)
      # epsilon 0.1: 0.05 + 0.9 * (1/3, 2/3) = (0.35, 0.65)
      expect_close(argmax.probabilities([ [ 1.0, 1.0 ], [ 2.0, 1.0 ] ], epsilon: 0.1), [ 0.35, 0.65 ], 1e-9)
    end

    it "SL10 gives equal members equal shares" do
      expect_close(argmax.probabilities([ [ 3.0, 3.0 ] ] * 3), [ 1.0 / 3 ] * 3, 1e-9)
      expect_close(argmax.probabilities([ [ 41.0, 361.0 ] ] * 2), [ 0.5, 0.5 ], 1e-9)
      expect_close(argmax.probabilities([ [ 1.0, 1.0 ], [ 1.0, 1.0 ] ], epsilon: 0.1), [ 0.5, 0.5 ], 1e-9)
    end

    it "SL10 answers [1.0] for one member" do
      expect(argmax.probabilities([ [ 41.0, 361.0 ] ])).to eq([ 1.0 ])
      expect(argmax.probabilities([ [ 2.0, 5.0 ] ], epsilon: 0.3)).to eq([ 1.0 ])
    end

    it "SL10 takes up to 20 members and refuses more" do
      twenty = (1..20).map { |a| [ a.to_f, 1.0 ] }
      expect_close(argmax.probabilities(twenty), (1..20).map { |a| a / 210.0 }, 1e-8)
      expect { argmax.probabilities(twenty + [ [ 1.0, 1.0 ] ]) }.to raise_error(ArgumentError)
    end
  end

  describe ".pick" do
    def params = [ [ 2.0, 8.0 ], [ 8.0, 2.0 ], [ 5.0, 5.0 ] ]

    it "SL10 explores below epsilon: the member with the smallest priority" do
      pick = argmax.pick(params, explore: 0.05, priorities: [ 0.7, 0.2, 0.9 ], uniforms: [ 0.5, 0.5, 0.5 ], epsilon: 0.1)
      expect(pick).to eq(1)
      pick = argmax.pick(params, explore: 0.05, priorities: [ 0.1, 0.2, 0.9 ], uniforms: [ 0.5, 0.5, 0.99 ], epsilon: 0.1)
      expect(pick).to eq(0)
    end

    it "SL10 exploits at and above epsilon: the largest quantile at each member's own uniform" do
      # Medians 0.1796, 0.8204, 0.5 (0.2.0's quantile): the second member.
      expect(argmax.pick(params, explore: 0.1, priorities: [ 0.0, 0.5, 0.9 ], uniforms: [ 0.5, 0.5, 0.5 ],
                         epsilon: 0.1)).to eq(1)
      # The uniforms reorder them: Q(1 - 1e-7; 2, 8) = 0.8975 beats Q(1e-7; 8, 2) = 0.1025 and Q(0.5; 5, 5) = 0.5.
      quantiles = [ beta_function.quantile(0.9999999, 2.0, 8.0), beta_function.quantile(0.0000001, 8.0, 2.0),
                    beta_function.quantile(0.5, 5.0, 5.0) ]
      expect(quantiles.index(quantiles.max)).to eq(0)
      expect(argmax.pick(params, explore: 0.7, priorities: [ 0.9, 0.0, 0.5 ], uniforms: [ 0.9999999, 0.0000001, 0.5 ],
                         epsilon: 0.1)).to eq(0)
    end

    it "SL10 never explores at epsilon 0" do
      # Exploring would take the first member (priority 0.0); the quantiles take the second.
      expect(argmax.pick(params, explore: 0.0, priorities: [ 0.0, 0.9, 0.5 ], uniforms: [ 0.5, 0.5, 0.5 ],
                         epsilon: 0.0)).to eq(1)
    end

    it "SL10 breaks ties to the lower index" do
      same = [ [ 3.0, 7.0 ] ] * 3
      expect(argmax.pick(same, explore: 0.5, priorities: [ 0.3, 0.1, 0.2 ], uniforms: [ 0.4, 0.4, 0.4 ], epsilon: 0.1)).to eq(0)
      expect(argmax.pick(params, explore: 0.01, priorities: [ 0.6, 0.3, 0.3 ], uniforms: [ 0.5, 0.5, 0.5 ],
                         epsilon: 0.1)).to eq(1)
    end
  end

  describe ".greedy" do
    it "SL10 picks the largest mean" do
      # Means 0.25, 0.75, 0.5.
      expect(argmax.greedy([ [ 1.0, 3.0 ], [ 3.0, 1.0 ], [ 2.0, 2.0 ] ])).to eq(1)
      # Means 0.10, 0.1005, 0.102: the sharpest is not the largest.
      expect(argmax.greedy([ [ 20_000.0, 180_000.0 ], [ 201.0, 1_799.0 ], [ 10.2, 89.8 ] ])).to eq(2)
    end

    it "SL10 breaks ties to the lower index" do
      # Means 1/3, 1/3 (the same Float), 1/4.
      expect(argmax.greedy([ [ 1.0, 2.0 ], [ 2.0, 4.0 ], [ 3.0, 9.0 ] ])).to eq(0)
      expect(argmax.greedy([ [ 1.0, 3.0 ], [ 2.0, 2.0 ], [ 4.0, 4.0 ] ])).to eq(1)
    end
  end
end

RSpec.describe SkiftetStatistical::BetaFunction do
  describe ".log_density" do
    it "SL10 gives the log of the Beta density at hand-checked points" do
      # Beta(2, 3): 12 x (1 - x)**2, at 1/4: 12 * 1/4 * 9/16 = 27/16.
      expect(described_class.log_density(0.25, 2.0, 3.0)).to be_within(1e-12).of(Math.log(27.0 / 16.0))
      # Beta(1, 1) is flat: log 1.
      expect(described_class.log_density(0.3, 1.0, 1.0)).to be_within(1e-12).of(0.0)
      # Beta(1/2, 1/2): 1 / (pi sqrt(x (1 - x))), at 1/2: 2 / pi.
      expect(described_class.log_density(0.5, 0.5, 0.5)).to be_within(1e-12).of(Math.log(2.0 / Math::PI))
    end

    it "SL10 holds for large parameters, against the exact binomial coefficient" do
      # Beta(41, 361): 401! / (40! 360!) x**40 (1 - x)**360 = 401 * C(400, 40) x**40 (1 - x)**360.
      binomial = (1..40).reduce(1) { |product, i| product * (360 + i) / i }
      [ 0.05, 0.1, 0.13 ].each do |x|
        expected = Math.log(401 * binomial) + (40 * Math.log(x)) + (360 * Math.log(1.0 - x))
        expect(described_class.log_density(x, 41.0, 361.0)).to be_within(1e-9).of(expected), "x = #{x}"
      end
    end

    it "SL10 is minus infinity outside (0, 1)" do
      [ -0.5, 0.0, 1.0, 1.5 ].each do |x|
        expect(described_class.log_density(x, 2.0, 3.0)).to eq(-Float::INFINITY), "x = #{x}"
      end
    end
  end
end
