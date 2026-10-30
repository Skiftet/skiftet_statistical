# frozen_string_literal: true

# Acceptance for SL11: the prequential log score, the gain of each depth over the one before in bits, and the
# cluster-robust interval of a mean from per-cluster totals. Every expected value is worked out by hand below. The
# probabilities are powers of 2, so every gain is a whole number of bits.
RSpec.describe "SkiftetStatistical::Prequential" do
  def prequential = SkiftetStatistical::Prequential

  # The two-sided normal points, to 16 digits: Phi^-1(0.975) and Phi^-1(0.95).
  def z95 = 1.959963984540054
  def z90 = 1.6448536269514722

  # Four events in three clusters, each with the probability three depths gave its outcome.
  #   cluster 1: a 1/4, b 1/2, c 1/2   -> b over a: +1 bit, c over b:  0
  #   cluster 1: a 1/2, b 1/2, c 1     -> b over a:  0,     c over b: +1
  #   cluster 2: a 1/2, b 1/4, c 1/8   -> b over a: -1,     c over b: -1
  #   cluster 3: a 1/8, b 1/2, c 1/4   -> b over a: +2,     c over b: -1
  def rows
    [
      [ 1, 0.25, 0.5, 0.5 ], [ 1, 0.5, 0.5, 1.0 ], [ 2, 0.5, 0.25, 0.125 ], [ 3, 0.125, 0.5, 0.25 ]
    ].map do |cluster, a, b, c|
      { cluster: cluster, log_probs: { a: Math.log(a), b: Math.log(b), c: Math.log(c) } }
    end
  end

  describe ".gains" do
    it "SL11 gives each depth after the first its mean gain in bits with a cluster-robust standard error" do
      gains = prequential.gains(rows, depths: %i[a b c])
      expect(gains.keys).to eq(%i[b c])

      # b: gains (1, 0, -1, 2), cluster totals (1, -1, 2) over counts (2, 1, 1); mean 2 / 4 = 0.5.
      #    residuals s_g - mean * n_g = (0, -1.5, 1.5); sum of squares 4.5; se = sqrt(3/2 * 4.5) / 4 = 3 sqrt(3) / 8.
      expect(gains[:b][:bits]).to be_within(1e-12).of(0.5)
      expect(gains[:b][:se]).to be_within(1e-12).of(3.0 * Math.sqrt(3.0) / 8.0)
      expect(gains[:b][:events]).to eq(4)
      expect(gains[:b][:clusters]).to eq(3)

      # c: gains (0, 1, -1, -1), cluster totals (1, -1, -1); mean -1 / 4.
      #    residuals (1.5, -0.75, -0.75); sum of squares 3.375; se = sqrt(3/2 * 3.375) / 4 = 2.25 / 4 = 0.5625.
      expect(gains[:c][:bits]).to be_within(1e-12).of(-0.25)
      expect(gains[:c][:se]).to be_within(1e-12).of(0.5625)
      expect(gains[:c][:events]).to eq(4)
      expect(gains[:c][:clusters]).to eq(3)
    end

    it "SL11 measures each depth against the one before it in the given order" do
      # c over a directly: gains (1, 1, -2, 1); cluster totals (2, -2, 1); mean 1 / 4.
      # residuals (2 - 0.5, -2 - 0.25, 1 - 0.25) = (1.5, -2.25, 0.75); sum of squares 7.875; se = sqrt(11.8125) / 4.
      gains = prequential.gains(rows, depths: %i[a c])
      expect(gains.keys).to eq(%i[c])
      expect(gains[:c][:bits]).to be_within(1e-12).of(0.25)
      expect(gains[:c][:se]).to be_within(1e-12).of(Math.sqrt(1.5 * 7.875) / 4.0)
    end
  end

  describe ".score" do
    it "SL11 is the mean of -log2 p at one depth" do
      # a: -log2 of (1/4, 1/2, 1/2, 1/8) = (2, 1, 1, 3), mean 1.75; c: (1, 0, 3, 2), mean 1.5.
      expect(prequential.score(rows, depth: :a)).to be_within(1e-12).of(1.75)
      expect(prequential.score(rows, depth: :c)).to be_within(1e-12).of(1.5)
      expect(prequential.score([ { cluster: 1, log_probs: { d: Math.log(0.5) } } ], depth: :d)).to be_within(1e-12).of(1.0)
    end
  end

  describe ".clustered_mean" do
    it "SL11 gives the mean of per-cluster totals and its normal interval at the level" do
      sums = { x: 3.0, y: -1.0, z: 2.0, w: 0.0 }
      counts = { x: 4, y: 2, z: 3, w: 1 }
      # mean 4 / 10 = 0.4; residuals (3 - 1.6, -1 - 0.8, 2 - 1.2, 0 - 0.4) = (1.4, -1.8, 0.8, -0.4);
      # sum of squares 6.0; se = sqrt(4/3 * 6) / 10 = sqrt(8) / 10.
      se = Math.sqrt(8.0) / 10.0
      result = prequential.clustered_mean(sums: sums, counts: counts)

      expect(result.keys).to contain_exactly(:mean, :lower, :upper, :n, :clusters)
      expect(result[:mean]).to be_within(1e-12).of(0.4)
      expect(result[:lower]).to be_within(1e-12).of(0.4 - (z95 * se))
      expect(result[:upper]).to be_within(1e-12).of(0.4 + (z95 * se))
      expect(result[:n]).to eq(10)
      expect(result[:clusters]).to eq(4)

      ninety = prequential.clustered_mean(sums: sums, counts: counts, level: 0.90)
      expect(ninety[:lower]).to be_within(1e-12).of(0.4 - (z90 * se))
      expect(ninety[:upper]).to be_within(1e-12).of(0.4 + (z90 * se))
    end

    it "SL11 gives no bounds under two clusters" do
      result = prequential.clustered_mean(sums: { only: 2.0 }, counts: { only: 4 })
      expect(result[:mean]).to be_within(1e-12).of(0.5)
      expect(result[:lower]).to be_nil
      expect(result[:upper]).to be_nil
      expect(result[:n]).to eq(4)
      expect(result[:clusters]).to eq(1)
    end

    it "SL11 equals gains' mean and standard error on the same rows' cluster totals" do
      gains = prequential.gains(rows, depths: %i[a b c])
      { b: :a, c: :b }.each do |depth, before|
        sums = Hash.new(0.0)
        counts = Hash.new(0)
        rows.each do |row|
          sums[row[:cluster]] += (row[:log_probs][depth] - row[:log_probs][before]) / Math.log(2.0)
          counts[row[:cluster]] += 1
        end

        result = prequential.clustered_mean(sums: sums, counts: counts)
        expect(result[:mean]).to be_within(1e-12).of(gains[depth][:bits]), "depth #{depth}"
        expect(result[:upper] - result[:mean]).to be_within(1e-12).of(z95 * gains[depth][:se]), "depth #{depth}"
        expect(result[:mean] - result[:lower]).to be_within(1e-12).of(z95 * gains[depth][:se]), "depth #{depth}"
        expect(result[:n]).to eq(gains[depth][:events])
        expect(result[:clusters]).to eq(gains[depth][:clusters])
      end
    end
  end
end
