# frozen_string_literal: true

require "json"

# SL5: shared-credit marginal IPS. Each row is { credit: { arm => w >= 0 }, values: [x_1 .. x_P] }. Per arm and part the
# Hajek mean r_ap = sum_i w_ai * x_ip / W_a (W_a = sum_i w_ai); the covariance is sum_i psi_ap(i) * psi_bq(i) over the
# influence values psi_ap(i) = w_ai * (x_ip - r_ap) / W_a, laid out arm-major as LinearValue reads it: index
# arm_index * P + part_index. Kish n per arm = W_a^2 / sum_i w_ai^2. Every expected number below is worked out by hand in
# the comment beside it.
RSpec.describe "SkiftetStatistical::SharedCredit" do
  def shared_credit
    SkiftetStatistical::SharedCredit
  end

  # Every number within 1e-12 of the hand value (relative above 1), nil where nil is expected, the same Hash keys and
  # Array sizes.
  def mismatches(actual, expected, path = "")
    case expected
    when Hash
      same = actual.is_a?(Hash) && actual.keys.sort_by(&:inspect) == expected.keys.sort_by(&:inspect)
      return [ "#{path}: keys #{actual.inspect} are not #{expected.keys.inspect}" ] unless same

      expected.flat_map { |key, value| mismatches(actual[key], value, "#{path}[#{key.inspect}]") }
    when Array
      same = actual.is_a?(Array) && actual.size == expected.size
      return [ "#{path}: #{actual.inspect} is not #{expected.size} long" ] unless same

      expected.each_with_index.flat_map { |value, index| mismatches(actual[index], value, "#{path}[#{index}]") }
    when nil
      actual.nil? ? [] : [ "#{path}: #{actual.inspect} is not nil" ]
    else
      close = actual.is_a?(Numeric) && (actual - expected).abs <= 1e-12 * [ 1.0, expected.abs ].max
      close ? [] : [ "#{path}: #{actual.inspect} is not #{expected.inspect}" ]
    end
  end

  # Three visits, two arms, two parts; the middle visit credits both arms.
  let(:shared_rows) do
    [
      { credit: { a: 2.0 }, values: [ 1.0, 0.0 ] },
      { credit: { a: 1.0, b: 1.0 }, values: [ 0.0, 2.0 ] },
      { credit: { b: 4.0 }, values: [ 1.0, 1.0 ] }
    ]
  end

  # By hand, for shared_rows:
  #   W_a = 2 + 1 = 3, r_a = [(2 * 1 + 1 * 0) / 3, (2 * 0 + 1 * 2) / 3] = [2/3, 2/3]
  #   W_b = 1 + 4 = 5, r_b = [(1 * 0 + 4 * 1) / 5, (1 * 2 + 4 * 1) / 5] = [4/5, 6/5]
  #   psi, visit by visit, as [(a,p), (a,q), (b,p), (b,q)]:
  #     visit 1: [2 (1 - 2/3) / 3, 2 (0 - 2/3) / 3, 0, 0] = [2/9, -4/9, 0, 0]
  #     visit 2: [(0 - 2/3) / 3, (2 - 2/3) / 3, (0 - 4/5) / 5, (2 - 6/5) / 5] = [-2/9, 4/9, -4/25, 4/25]
  #     visit 3: [0, 0, 4 (1 - 4/5) / 5, 4 (1 - 6/5) / 5] = [0, 0, 4/25, -4/25]
  #   covariance = sum of the products: e.g. (a,p)(a,p) = 4/81 + 4/81 = 8/81, (a,p)(b,p) = (-2/9)(-4/25) = 8/225,
  #   (b,p)(b,q) = (-4/25)(4/25) + (4/25)(-4/25) = -32/625.
  #   Kish n: a 3^2 / (4 + 1) = 9/5, b 5^2 / (1 + 16) = 25/17.
  let(:shared_estimate) do
    {
      means: { a: [ 2.0 / 3, 2.0 / 3 ], b: [ 4.0 / 5, 6.0 / 5 ] },
      covariance: [
        [ 8.0 / 81, -16.0 / 81, 8.0 / 225, -8.0 / 225 ],
        [ -16.0 / 81, 32.0 / 81, -16.0 / 225, 16.0 / 225 ],
        [ 8.0 / 225, -16.0 / 225, 32.0 / 625, -32.0 / 625 ],
        [ -8.0 / 225, 16.0 / 225, -32.0 / 625, 32.0 / 625 ]
      ],
      weight_sums: { a: 3.0, b: 5.0 },
      effective: { a: 9.0 / 5, b: 25.0 / 17 }
    }
  end

  # An ordinary test: every visit credits only its drawn arm, with 1 / p (shares 0.5, 0.25 and 0.5).
  let(:single_rows) do
    [
      { credit: { a: 2.0 }, values: [ 1.0, 0.0 ] },
      { credit: { a: 4.0 }, values: [ 0.0, 2.0 ] },
      { credit: { b: 2.0 }, values: [ 1.0, 3.0 ] }
    ]
  end

  # Their per-arm Hajek sums, by hand: w = sum 1/p, w2 = sum 1/p^2, a = sum x/p, b = sum x/p^2, c = sum x_p x_q / p^2.
  #   a: w = 2 + 4 = 6, w2 = 4 + 16 = 20, a = {p: 1 * 2 + 0 * 4 = 2, q: 0 * 2 + 2 * 4 = 8},
  #      b = {p: 1 * 4 + 0 * 16 = 4, q: 0 * 4 + 2 * 16 = 32}, c = {pp: 1 * 4 = 4, pq: 0, qq: 4 * 16 = 64}
  #   b: w = 2, w2 = 4, a = {p: 2, q: 6}, b = {p: 4, q: 12}, c = {pp: 4, pq: 12, qq: 36}
  # Every ordered pair of parts is given.
  let(:single_sums) do
    {
      a: { w: 6.0, w2: 20.0, a: { p: 2.0, q: 8.0 }, b: { p: 4.0, q: 32.0 },
           c: { [ :p, :p ] => 4.0, [ :p, :q ] => 0.0, [ :q, :p ] => 0.0, [ :q, :q ] => 64.0 } },
      b: { w: 2.0, w2: 4.0, a: { p: 2.0, q: 6.0 }, b: { p: 4.0, q: 12.0 },
           c: { [ :p, :p ] => 4.0, [ :p, :q ] => 12.0, [ :q, :p ] => 12.0, [ :q, :q ] => 36.0 } }
    }
  end

  # By hand: r_a = [2/6, 8/6] = [1/3, 4/3], r_b = [2/2, 6/2] = [1, 3].
  #   Within a, cov(r_p, r_q) = (c_pq - r_p b_q - r_q b_p + r_p r_q w2) / w^2:
  #     pp: (4 - 2 (1/3) 4 + (1/9) 20) / 36 = (32/9) / 36 = 8/81
  #     pq: (0 - (1/3) 32 - (4/3) 4 + (4/9) 20) / 36 = (-64/9) / 36 = -16/81
  #     qq: (64 - 2 (4/3) 32 + (16/9) 20) / 36 = (128/9) / 36 = 32/81
  #   (The same from the rows: psi_a = [2 (1 - 1/3) / 6, 2 (0 - 4/3) / 6] = [2/9, -4/9] and [-2/9, 4/9].)
  #   Within b, one visit: (4 - 8 + 4) / 4 = 0, (12 - 12 - 12 + 12) / 4 = 0, (36 - 72 + 36) / 4 = 0. Across arms 0.
  #   Kish n: a 36 / 20 = 9/5, b 4 / 4 = 1.
  let(:single_estimate) do
    {
      means: { a: [ 1.0 / 3, 4.0 / 3 ], b: [ 1.0, 3.0 ] },
      covariance: [
        [ 8.0 / 81, -16.0 / 81, 0.0, 0.0 ],
        [ -16.0 / 81, 32.0 / 81, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0 ]
      ],
      weight_sums: { a: 6.0, b: 2.0 },
      effective: { a: 9.0 / 5, b: 1.0 }
    }
  end

  # An Enumerable whose rows may be read once only.
  let(:once_class) do
    Class.new do
      include Enumerable

      def initialize(rows)
        @rows = rows
        @passes = 0
      end

      def each(&block)
        @passes += 1
        raise "the rows were read #{@passes} times" if @passes > 1

        @rows.each(&block)
      end
    end
  end

  describe ".estimate" do
    it "SL5 gives the hand-computed Hajek means, influence covariance, weight sums and Kish n on three shared rows" do
      estimate = shared_credit.estimate(shared_rows, arms: %i[a b], parts: %i[p q])

      expect(estimate.keys).to contain_exactly(:means, :covariance, :weight_sums, :effective)
      expect(mismatches(estimate, shared_estimate)).to eq([])
    end

    it "SL5 gives the hand-computed estimate of an ordinary test, each visit crediting its drawn arm" do
      expect(mismatches(shared_credit.estimate(single_rows, arms: %i[a b], parts: %i[p q]), single_estimate)).to eq([])
    end

    it "SL5 reads any Enumerable once" do
      estimate = shared_credit.estimate(once_class.new(shared_rows), arms: %i[a b], parts: %i[p q])

      expect(mismatches(estimate, shared_estimate)).to eq([])
    end

    it "SL5 gives an arm with no credit nil means and zero covariance rows and columns" do
      # c is never credited: its means are nil, rows and columns 4 and 5 (arm-major) are 0, and a and b are as before.
      estimate = shared_credit.estimate(shared_rows, arms: %i[a b c], parts: %i[p q])
      covariance = estimate[:covariance]

      expect(estimate[:means]).to include(c: nil)
      expect(mismatches(estimate[:means].slice(:a, :b), shared_estimate[:means])).to eq([])
      expect(covariance.size).to eq(6)
      expect(covariance).to all(have_attributes(size: 6))
      expect(mismatches(covariance.first(4).map { |row| row.first(4) }, shared_estimate[:covariance])).to eq([])
      (0...6).each do |i|
        [ 4, 5 ].each do |j|
          expect(covariance[i][j]).to eq(0.0)
          expect(covariance[j][i]).to eq(0.0)
        end
      end
      expect(estimate[:weight_sums][:c]).to be_zero
    end

    it "SL5 gives nil means for an arm a single row leaves out (one arm credited, one value)" do
      # One visit credits a with 2: r_a = 2 * 1 / 2 = 1, psi = 2 (1 - 1) / 2 = 0, Kish n 2^2 / 4 = 1; b has no credit.
      estimate = shared_credit.estimate([ { credit: { a: 2.0 }, values: [ 1.0 ] } ], arms: %i[a b], parts: %i[p])

      expect(estimate.dig(:means, :b)).to be_nil
      expect(mismatches(estimate[:means][:a], [ 1.0 ])).to eq([])
      expect(mismatches(estimate[:covariance], [ [ 0.0, 0.0 ], [ 0.0, 0.0 ] ])).to eq([])
      expect(mismatches(estimate[:effective][:a], 1.0)).to eq([])
    end

    it "SL5 gives the same estimate whether a zero credit is written or left out" do
      with_zeros = [
        { credit: { a: 2.0, b: 0.0 }, values: [ 1.0, 0.0 ] },
        { credit: { a: 1.0, b: 1.0 }, values: [ 0.0, 2.0 ] },
        { credit: { a: 0.0, b: 4.0 }, values: [ 1.0, 1.0 ] }
      ]

      expect(mismatches(shared_credit.estimate(with_zeros, arms: %i[a b], parts: %i[p q]), shared_estimate)).to eq([])
    end
  end

  describe ".from_sums" do
    it "SL5 gives the hand-computed estimate from per-arm Hajek sums" do
      estimate = shared_credit.from_sums(single_sums, arms: %i[a b], parts: %i[p q])

      expect(estimate.keys).to contain_exactly(:means, :covariance, :weight_sums, :effective)
      expect(mismatches(estimate, single_estimate)).to eq([])
    end

    it "SL5 equals estimate on the same three rows, each crediting one arm, within 1e-12" do
      from_sums = shared_credit.from_sums(single_sums, arms: %i[a b], parts: %i[p q])
      from_rows = shared_credit.estimate(single_rows, arms: %i[a b], parts: %i[p q])

      expect(mismatches(from_sums, from_rows)).to eq([])
    end
  end

  describe ".accumulator" do
    def round_trip(accumulator)
      shared_credit::Accumulator.from_h(JSON.parse(JSON.generate(accumulator.to_h)))
    end

    it "SL5 builds estimate on the accumulator: all rows added give estimate's answer exactly" do
      accumulator = shared_credit.accumulator(arms: %i[a b], parts: %i[p q])
      shared_rows.each { |row| accumulator.add(row) }

      expect(accumulator).to be_a(shared_credit::Accumulator)
      expect(accumulator.estimate).to eq(shared_credit.estimate(shared_rows, arms: %i[a b], parts: %i[p q]))
    end

    it "SL5 folded in two halves, through to_h and from_h, and merged, equals one estimate of the rows" do
      first = shared_credit.accumulator(arms: %i[a b], parts: %i[p q])
      first.add(shared_rows[0])
      second = shared_credit.accumulator(arms: %i[a b], parts: %i[p q])
      shared_rows.drop(1).each { |row| second.add(row) }

      merged = round_trip(first).merge(round_trip(second))

      expect(mismatches(merged.estimate, shared_estimate)).to eq([])
      expect(mismatches(merged.estimate, shared_credit.estimate(shared_rows, arms: %i[a b], parts: %i[p q]))).to eq([])
    end

    it "SL5 answers a JSON-safe to_h: a JSON round trip gives the same Hash" do
      accumulator = shared_credit.accumulator(arms: %i[a b], parts: %i[p q])
      shared_rows.each { |row| accumulator.add(row) }
      hash = accumulator.to_h

      expect(JSON.parse(JSON.generate(hash))).to eq(hash)
    end
  end

  describe ".linear and .contrast" do
    # At fixed weights {p: 1, q: 2}, by hand from shared_estimate:
    #   V_a = 2/3 + 2 (2/3) = 2, V_b = 4/5 + 2 (6/5) = 16/5.
    #   Var V_a = 8/81 + 4 (-16/81) + 4 (32/81) = 72/81 = 8/9; Var V_b = 32/625 + 4 (-32/625) + 4 (32/625) = 32/625;
    #   Cov = 8/225 + 2 (-8/225) + 2 (-16/225) + 4 (16/225) = 24/225 = 8/75.
    #   (By the influence values: V's psi is [-2/3, 2/3, 0] for a and [0, 4/25, -4/25] for b, visit by visit.)
    #   a - b: mean 2 - 16/5 = -6/5, variance 8/9 + 32/625 - 2 (8/75) = (5000 + 288 - 1200) / 5625 = 4088/5625.
    let(:weights) { { p: 1.0, q: 2.0 } }

    it "SL5 gives each arm's value and the K x K covariance at fixed weights" do
      estimate = shared_credit.estimate(shared_rows, arms: %i[a b], parts: %i[p q])
      linear = shared_credit.linear(estimate, weights: weights)

      expect(linear.keys).to contain_exactly(:means, :covariance)
      expect(mismatches(linear, { means: { a: 2.0, b: 16.0 / 5 },
                                  covariance: [ [ 8.0 / 9, 8.0 / 75 ], [ 8.0 / 75, 32.0 / 625 ] ] })).to eq([])
    end

    it "SL5 gives one difference with its variance" do
      estimate = shared_credit.estimate(shared_rows, arms: %i[a b], parts: %i[p q])
      contrast = shared_credit.contrast(estimate, :a, :b, weights: weights)

      expect(contrast.keys).to contain_exactly(:mean, :variance)
      expect(mismatches(contrast, { mean: -6.0 / 5, variance: 4088.0 / 5625 })).to eq([])
    end

    it "SL5 gives a value and its variance from from_sums as from estimate" do
      # From single_estimate at {p: 1, q: 2}: V_a = 1/3 + 8/3 = 3, V_b = 1 + 6 = 7;
      # Var V_a = 8/81 + 4 (-16/81) + 4 (32/81) = 8/9, b's block and the cross terms 0. a - b: -4 with variance 8/9.
      estimate = shared_credit.from_sums(single_sums, arms: %i[a b], parts: %i[p q])

      expect(mismatches(shared_credit.linear(estimate, weights: weights),
                        { means: { a: 3.0, b: 7.0 }, covariance: [ [ 8.0 / 9, 0.0 ], [ 0.0, 0.0 ] ] })).to eq([])
      expect(mismatches(shared_credit.contrast(estimate, :a, :b, weights: weights),
                        { mean: -4.0, variance: 8.0 / 9 })).to eq([])
    end
  end
end
