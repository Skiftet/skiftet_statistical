# frozen_string_literal: true

# SL3: the linear value of several rates with uncertain weights. THE DRAW ORDER IS THE CONTRACT, per draw:
#   1. each parameter in the given order: `fixed` takes nothing, `uniform: [lo, hi]` one `rng.rand` (lo + (hi - lo) * u),
#      `normal: [mean, sd]` one Sampler#gaussian (mean + sd * g), then its `clip: [lo, hi]`;
#   2. the part weights from the caller's `weights` callable, given the drawn values by name;
#   3. one joint normal draw of every (arm, part) rate, z = means + L * n, n being K * P Sampler#gaussian values in
#      arm-major order and L the Cholesky factor of the covariance (SL4);
#   4. each arm's value V_a = sum over parts p of w_p * z_(a, p).
# The expected rows below come from an independent re-implementation of that order (#reference_rows), on the 0.2.0
# Sampler over the same RNG and with the factor L written out by hand (as in cholesky_spec.rb).
RSpec.describe "SkiftetStatistical::LinearValue" do
  let(:arms) { %i[a b] }
  let(:parts) { %i[p q] }
  let(:means) { { a: [ 1.0, 2.0 ], b: [ 1.5, 1.0 ] } }
  # Arm-major: (a, p), (a, q), (b, p), (b, q). This is L * L^T for the hand L of #hand_lower.
  let(:covariance) do
    [ [ 4.0, 2.0, 1.0, 0.0 ], [ 2.0, 5.0, 0.5, 1.0 ], [ 1.0, 0.5, 1.25, 0.5 ], [ 0.0, 1.0, 0.5, 1.5 ] ]
  end
  let(:hand_lower) do
    [ [ 2.0, 0.0, 0.0, 0.0 ], [ 1.0, 2.0, 0.0, 0.0 ], [ 0.5, 0.0, 1.0, 0.0 ], [ 0.0, 0.5, 0.5, 1.0 ] ]
  end
  let(:parameters) do
    { base: { fixed: 2.0 }, scale: { uniform: [ 1.0, 3.0 ] }, tilt: { normal: [ 0.5, 0.2 ], clip: [ 0.0, 0.4 ] } }
  end
  let(:weights) { ->(drawn) { { p: drawn.fetch(:base), q: drawn.fetch(:scale) * drawn.fetch(:tilt) } } }

  def linear_value
    SkiftetStatistical::LinearValue
  end

  def inputs(**overrides)
    { arms: arms, parts: parts, means: means, covariance: covariance, parameters: parameters, weights: weights }
      .merge(overrides)
  end

  # The contract's draw order, written out: the parameters in `order`, the weights, then K * P gaussians.
  def reference_rows(seed:, draws:, lower:, order: %i[base scale tilt], flat_means: [ 1.0, 2.0, 1.5, 1.0 ])
    rng = Random.new(seed)
    sampler = SkiftetStatistical::Sampler.new(rng)
    Array.new(draws) do
      drawn = { base: 2.0 }
      order.each do |name|
        case name
        when :scale then drawn[:scale] = 1.0 + ((3.0 - 1.0) * rng.rand)
        when :tilt then drawn[:tilt] = (0.5 + (0.2 * sampler.gaussian)).clamp(0.0, 0.4)
        end
      end
      weight = [ drawn[:base], drawn[:scale] * drawn[:tilt] ]
      normals = Array.new(4) { sampler.gaussian }
      z = (0...4).map { |i| flat_means[i] + (0..i).sum { |j| lower[i][j] * normals[j] } }
      [ (weight[0] * z[0]) + (weight[1] * z[1]), (weight[0] * z[2]) + (weight[1] * z[3]) ]
    end
  end

  def close?(actual, expected, tolerance = 1e-12)
    actual.is_a?(Numeric) && (actual - expected).abs <= tolerance * [ 1.0, expected.abs ].max
  end

  def expect_rows(actual, expected)
    expect(actual.size).to eq(expected.size)
    expected.each_with_index do |row, draw|
      expect(actual[draw].size).to eq(row.size)
      row.each_with_index do |value, arm|
        expect(close?(actual[draw][arm], value)).to be(true), "draw #{draw}, arm #{arm}: #{actual[draw][arm]} vs #{value}"
      end
    end
  end

  describe ".each_draw" do
    it "SL3 draws each parameter in order, then the weights, then K * P gaussians in arm-major order, per draw" do
      [ 1, 7, 20_261_030 ].each do |seed|
        rows = []
        linear_value.each_draw(**inputs, draws: 200, rng: Random.new(seed)) { |row| rows << row }

        expect_rows(rows, reference_rows(seed: seed, draws: 200, lower: hand_lower))
      end
    end

    it "SL3 takes the parameters in the order given" do
      swapped = parameters.slice(:base, :tilt, :scale)
      rows = linear_value.each_draw(**inputs(parameters: swapped), draws: 100, rng: Random.new(3)).to_a

      expect_rows(rows, reference_rows(seed: 3, draws: 100, lower: hand_lower, order: %i[base tilt scale]))
      expect(rows).not_to eq(linear_value.each_draw(**inputs, draws: 100, rng: Random.new(3)).to_a)
    end

    it "SL3 moves every row with one extra random number taken before the draws" do
      rng = Random.new(5)
      rng.rand
      rows = linear_value.each_draw(**inputs, draws: 20, rng: rng).to_a
      reference = reference_rows(seed: 5, draws: 20, lower: hand_lower)

      expect(rows.zip(reference).count { |row, hand| close?(row[0], hand[0]) }).to be < 20
    end

    it "SL3 is an Enumerator of the same rows without a block, one Array of K values per draw in arms order" do
      rows = linear_value.each_draw(**inputs, draws: 30, rng: Random.new(11))

      expect(rows).to be_a(Enumerator)
      expect(rows.to_a.size).to eq(30)
      expect(rows.to_a).to all(be_an(Array).and(have_attributes(size: 2)))
      expect_rows(rows.to_a, reference_rows(seed: 11, draws: 30, lower: hand_lower))
    end

    it "SL3 draws a part with no variance at its mean and still takes its normal" do
      # (a, q) has no variance. By hand (SL4): L00 = 2; pivot of column 1 is 0, so column 1 is all zeros; L20 = 0.5,
      # L22 = sqrt(1.25 - 0.25) = 1; L30 = 0, L32 = 0.5 / 1, L33 = sqrt(1.5 - 0.25) = sqrt(1.25). z_(a, q) = 2.0 in
      # every draw, while its gaussian n_1 is still drawn, so n_2 and n_3 are the same values as with variance.
      zeroed = [ [ 4.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 1.0, 0.0, 1.25, 0.5 ], [ 0.0, 0.0, 0.5, 1.5 ] ]
      lower = [ [ 2.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.5, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.5, Math.sqrt(1.25) ] ]
      rows = linear_value.each_draw(**inputs(covariance: zeroed), draws: 100, rng: Random.new(13)).to_a

      expect_rows(rows, reference_rows(seed: 13, draws: 100, lower: lower))
    end

    it "SL3 draws a lone part with no variance exactly at its mean, taking two random numbers a draw" do
      # One arm, one part, covariance [[0]], weight fixed at 3: V = 3 * 2.0 = 6.0 in every draw. Each draw takes one
      # Sampler#gaussian (two rng.rand), so five draws leave the RNG ten numbers on.
      rng = Random.new(17)
      rows = linear_value.each_draw(arms: %i[a], parts: %i[p], means: { a: [ 2.0 ] }, covariance: [ [ 0.0 ] ],
                                    parameters: { w: { fixed: 3.0 } }, weights: ->(drawn) { { p: drawn.fetch(:w) } },
                                    draws: 5, rng: rng).to_a
      fresh = Random.new(17)
      10.times { fresh.rand }

      expect(rows).to eq([ [ 6.0 ] ] * 5)
      expect(rng.rand).to eq(fresh.rand)
    end

    it "SL3 factors the covariance once per call" do
      allow(SkiftetStatistical::Cholesky).to receive(:factor).and_call_original
      linear_value.each_draw(**inputs, draws: 50, rng: Random.new(1)).to_a

      expect(SkiftetStatistical::Cholesky).to have_received(:factor).once
    end
  end

  describe ".summary" do
    # P(best) with a tie split equally, expected loss = mean(best - V_a), mean = the draws' mean, each read by hand
    # from the each_draw rows of the same seed.
    def hand_summary(rows)
      count = rows.size.to_f
      wins = [ 0.0, 0.0 ]
      losses = [ 0.0, 0.0 ]
      rows.each do |row|
        best = row.max
        leaders = row.each_index.select { |index| row[index] == best }
        leaders.each { |index| wins[index] += 1.0 / leaders.size }
        row.each_with_index { |value, index| losses[index] += best - value }
      end
      (0...2).map do |index|
        { p_best: wins[index] / count, expected_loss: losses[index] / count,
          mean: rows.sum { |row| row[index] } / count }
      end
    end

    it "SL3 is built on each_draw: the same seed gives the rows' P(best), expected loss and mean" do
      [ 2, 9 ].each do |seed|
        rows = linear_value.each_draw(**inputs, draws: 400, rng: Random.new(seed)).to_a
        summary = linear_value.summary(**inputs, draws: 400, rng: Random.new(seed))

        expect(summary.keys).to eq(arms)
        arms.each_with_index do |arm, index|
          expected = hand_summary(rows)[index]
          expect(summary[arm].keys).to contain_exactly(:point, :mean, :lower, :upper, :p_best, :expected_loss)
          expected.each do |key, value|
            expect(close?(summary[arm][key], value)).to be(true), "seed #{seed}, #{arm} #{key}: #{summary[arm][key]}"
          end
        end
        expect(summary.values.sum { |arm| arm[:p_best] }).to be_within(1e-12).of(1.0)
      end
    end

    it "SL3 reads the credible_mass interval of each arm's drawn values (equal tails, 0.95 by default)" do
      rows = linear_value.each_draw(**inputs, draws: 400, rng: Random.new(4)).to_a
      wide = linear_value.summary(**inputs, draws: 400, rng: Random.new(4))
      narrow = linear_value.summary(**inputs, draws: 400, rng: Random.new(4), credible_mass: 0.5)

      # Any quantile convention lands between the order statistics around position tail * (n - 1).
      { wide => 0.95, narrow => 0.5 }.each do |summary, mass|
        tail = (1.0 - mass) / 2.0
        arms.each_with_index do |arm, index|
          sorted = rows.map { |row| row[index] }.sort
          [ [ :lower, tail ], [ :upper, 1.0 - tail ] ].each do |key, probability|
            position = probability * (sorted.size - 1)
            low = sorted[position.floor - 1]
            high = sorted[position.ceil + 1]
            expect(summary[arm][key]).to be_between(low, high), "#{arm} #{key} at mass #{mass}"
          end
        end
      end
      expect(narrow[:a][:upper] - narrow[:a][:lower]).to be < wide[:a][:upper] - wide[:a][:lower]
    end

    it "SL3 values each arm at every parameter's centre for its point" do
      # Centres: base the fixed 2.0; scale the uniform's midpoint (1 + 3) / 2 = 2.0; tilt the normal's mean 0.5, clipped
      # to [0, 0.4]: 0.4. Weights: p = 2.0, q = 2.0 * 0.4 = 0.8.
      # point_a = 2.0 * 1.0 + 0.8 * 2.0 = 3.6; point_b = 2.0 * 1.5 + 0.8 * 1.0 = 3.8.
      summary = linear_value.summary(**inputs, draws: 50, rng: Random.new(1))

      expect(summary[:a][:point]).to be_within(1e-12).of(3.6)
      expect(summary[:b][:point]).to be_within(1e-12).of(3.8)
    end

    it "SL3 splits every tie equally: two equal arms with no variance each have P(best) 0.5 and no loss" do
      # Every draw: V_a = V_b = 1.0 * 0.3, a tie split 0.5 / 0.5; best - V = 0.
      summary = linear_value.summary(arms: %i[a b], parts: %i[p], means: { a: [ 0.3 ], b: [ 0.3 ] },
                                     covariance: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ] ], parameters: { w: { fixed: 1.0 } },
                                     weights: ->(drawn) { { p: drawn.fetch(:w) } }, draws: 1_000, rng: Random.new(1))

      %i[a b].each do |arm|
        expect(summary[arm][:p_best]).to eq(0.5)
        expect(summary[arm][:expected_loss]).to eq(0.0)
        expect(summary[arm][:mean]).to be_within(1e-12).of(0.3)
        expect(summary[arm][:point]).to be_within(1e-12).of(0.3)
      end
    end

    it "SL3 splits a two-way tie and charges the third arm its shortfall" do
      # V = 2 * (1.0, 1.0, 0.5) in every draw: a and b tie for best (0.5 each), c never best and 2 * 0.5 = 1.0 short.
      summary = linear_value.summary(arms: %i[a b c], parts: %i[p], means: { a: [ 1.0 ], b: [ 1.0 ], c: [ 0.5 ] },
                                     covariance: Array.new(3) { [ 0.0, 0.0, 0.0 ] }, parameters: { w: { fixed: 2.0 } },
                                     weights: ->(drawn) { { p: drawn.fetch(:w) } }, draws: 10, rng: Random.new(1))

      expect(summary.transform_values { |arm| arm[:p_best] }).to eq(a: 0.5, b: 0.5, c: 0.0)
      expect(summary.transform_values { |arm| arm[:expected_loss] }).to eq(a: 0.0, b: 0.0, c: 1.0)
    end

    it "SL3 factors the covariance once per call" do
      allow(SkiftetStatistical::Cholesky).to receive(:factor).and_call_original
      linear_value.summary(**inputs, draws: 50, rng: Random.new(1))

      expect(SkiftetStatistical::Cholesky).to have_received(:factor).once
    end

    it "SL3 takes up to 7 arms and 12 parts" do
      seven = (1..7).map { |n| :"arm#{n}" }
      twelve = (1..12).map { |n| :"part#{n}" }
      size = 7 * 12
      identity = Array.new(size) { |i| Array.new(size) { |j| i == j ? 1.0 : 0.0 } }
      summary = linear_value.summary(arms: seven, parts: twelve, means: seven.to_h { |arm| [ arm, [ 0.0 ] * 12 ] },
                                     covariance: identity, parameters: { w: { fixed: 1.0 } },
                                     weights: ->(_drawn) { twelve.to_h { |part| [ part, 1.0 ] } }, draws: 3,
                                     rng: Random.new(1))

      expect(summary.keys).to eq(seven)
    end
  end

  describe "refusals" do
    def one_part(arms, covariance)
      { arms: arms, parts: %i[p], means: arms.to_h { |arm| [ arm, [ 0.0 ] ] }, covariance: covariance,
        parameters: { w: { fixed: 1.0 } }, weights: ->(_drawn) { { p: 1.0 } } }
    end

    def identity(size)
      Array.new(size) { |i| Array.new(size) { |j| i == j ? 1.0 : 0.0 } }
    end

    it "SL3 raises ArgumentError on more than 7 arms" do
      eight = (1..8).map { |n| :"arm#{n}" }

      expect { linear_value.summary(**one_part(eight, identity(8)), draws: 2, rng: Random.new(1)) }
        .to raise_error(ArgumentError)
    end

    it "SL3 raises ArgumentError on more than 12 parts" do
      thirteen = (1..13).map { |n| :"part#{n}" }

      expect {
        linear_value.summary(arms: %i[a], parts: thirteen, means: { a: [ 0.0 ] * 13 }, covariance: identity(13),
                             parameters: { w: { fixed: 1.0 } }, weights: ->(_drawn) { thirteen.to_h { |part| [ part, 1.0 ] } },
                             draws: 2, rng: Random.new(1))
      }.to raise_error(ArgumentError)
    end

    it "SL3 raises ArgumentError on a covariance that is not square" do
      expect { linear_value.summary(**inputs(covariance: Array.new(4) { [ 1.0, 0.0, 0.0 ] }), draws: 2, rng: Random.new(1)) }
        .to raise_error(ArgumentError)
    end

    it "SL3 raises ArgumentError on a covariance asymmetric beyond 1e-12 relative, and takes one within it" do
      # 0.5 against 0.5 + 5e-10: 1e-9 relative, refused. 0.5 against 0.5 * (1 + 1e-14): within 1e-12, taken.
      asymmetric = [ [ 1.0, 0.5 ], [ 0.5 + 5e-10, 1.0 ] ]
      within = [ [ 1.0, 0.5 ], [ 0.5 * (1 + 1e-14), 1.0 ] ]

      expect { linear_value.summary(**one_part(%i[a b], asymmetric), draws: 2, rng: Random.new(1)) }
        .to raise_error(ArgumentError)
      expect(linear_value.summary(**one_part(%i[a b], within), draws: 2, rng: Random.new(1)).keys).to eq(%i[a b])
    end

    it "SL3 raises ArgumentError when the weights callable leaves out a part" do
      expect {
        linear_value.summary(**inputs(weights: ->(_drawn) { { p: 1.0 } }), draws: 2, rng: Random.new(1))
      }.to raise_error(ArgumentError)
    end

    it "SL3 raises ArgumentError on a covariance that is not positive semidefinite" do
      expect { linear_value.summary(**one_part(%i[a b], [ [ 1.0, 2.0 ], [ 2.0, 1.0 ] ]), draws: 2, rng: Random.new(1)) }
        .to raise_error(ArgumentError)
    end
  end
end
