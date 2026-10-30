# frozen_string_literal: true

# SL15: the pins of LinearValue's and Cholesky's outputs on an invented fixture, and the readings where SL3 and SL4 are
# silent. Every pinned value was checked first by an independent reference that writes the draw order out: the
# parameters by their Float formulas over the 0.2.0 Sampler and the same RNG, then every product and sum of the joint
# normal draw and of the value in exact Rationals, over a Cholesky factor of its own in Rationals with 60-digit square
# roots, the interval by Descriptive's percentile rule over the exact values. This file ran red against those digits
# before the module existed. A pinned value is never edited.
#
# One tolerance rule (SL15) for every Float this file compares with a pinned or hand-worked value: 1e-12 relative, and
# a pinned 0.0, 1.0 or infinity exactly. SL15 gives outputs through lgamma 1e-9; none of these does (the draws are
# Sampler's and the factor plain arithmetic and square roots), so none takes it. A reconstruction L * L^T of a matrix
# is an invariant, not a pin: it is held to the rounding the factor gives a pivot and a residual under a zero pivot,
# 1e-9 times the trace (a matrix of lower rank drops its rounding there: up to 6e-12 of the trace in the seeded ones).
RSpec.describe "SkiftetStatistical::LinearValue pins" do
  def linear_value
    SkiftetStatistical::LinearValue
  end

  def cholesky
    SkiftetStatistical::Cholesky
  end

  def relative
    1e-12
  end

  def expect_pin(actual, pinned, label)
    expect(actual).to be_a(Float), "#{label}: #{actual.inspect} is not a Float"
    if pinned.zero? || pinned == 1.0 || pinned.infinite?
      expect(actual).to eq(pinned), "#{label}: #{actual} is not exactly #{pinned}"
    else
      error = (actual - pinned).abs / pinned.abs
      expect(error).to be <= relative, "#{label}: #{actual} is #{error} from #{pinned}"
    end
  end

  # Three arms of two parts. Arm-major order: (a, p), (a, q), (b, p), (b, q), (c, p), (c, q). Arms a and b are
  # correlated with each other (one visit may count for both); c is independent of both, so its rows of the factor
  # hold zeros under a's and b's columns.
  let(:arms) { %i[a b c] }
  let(:parts) { %i[p q] }
  let(:means) { { a: [ 0.12, 0.031 ], b: [ 0.105, 0.044 ], c: [ 0.09, 0.052 ] } }
  let(:covariance) do
    [
      [ 0.0004, 0.00006, 0.0002, 0.00001, 0.0, 0.0 ],
      [ 0.00006, 0.0001, 0.00002, 0.00004, 0.0, 0.0 ],
      [ 0.0002, 0.00002, 0.0005, -0.00005, 0.0, 0.0 ],
      [ 0.00001, 0.00004, -0.00005, 0.00012, 0.0, 0.0 ],
      [ 0.0, 0.0, 0.0, 0.0, 0.0006, 0.00003 ],
      [ 0.0, 0.0, 0.0, 0.0, 0.00003, 0.00015 ]
    ]
  end
  let(:parameters) do
    { base: { fixed: 1.5 }, worth: { uniform: [ 2.0, 6.0 ] }, lift: { normal: [ 0.5, 0.2 ], clip: [ 0.0, 1.0 ] } }
  end
  let(:weights) { ->(drawn) { { p: drawn.fetch(:base), q: drawn.fetch(:worth) * drawn.fetch(:lift) } } }

  def inputs(**overrides)
    { arms: arms, parts: parts, means: means, covariance: covariance, parameters: parameters, weights: weights }
      .merge(overrides)
  end

  describe "pins" do
    it "SL15 pins the Cholesky factor of the fixture's covariance" do
      pinned = [
        [ 0.02 ],
        [ 0.0030000000000000001, 0.0095393920141694562 ],
        [ 0.01, -0.0010482848367219181, 0.01997250857807048 ],
        [ 0.00050000000000000001, 0.0040358966213793861, -0.0025419556372089703, 0.0098488578017961043 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.024494897427831779 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0012247448713915891, 0.01218605760695394 ]
      ]
      lower = cholesky.factor(covariance)

      expect(lower.size).to eq(6)
      lower.each_with_index do |row, i|
        expect(row.size).to eq(6)
        row.each_with_index do |value, j|
          expect_pin(value, j <= i ? pinned[i][j] : 0.0, "L[#{i}][#{j}]")
        end
      end
    end

    it "SL15 pins the first five rows of each_draw on seed 20261030, and the random numbers they took" do
      pinned = [
        [ 0.2796153321829562, 0.2189726397405605, 0.21371998019046903 ],
        [ 0.25051998611673365, 0.16359268431253932, 0.28486521261043746 ],
        [ 0.22213489176290857, 0.21125116583998493, 0.12510892479283284 ],
        [ 0.19899438021040167, 0.16757060985568994, 0.24941129405514487 ],
        [ 0.23333554905590631, 0.26368354006585354, 0.22499424728747175 ]
      ]
      rng = Random.new(20_261_030)
      rows = []
      result = linear_value.each_draw(**inputs, draws: 5, rng: rng) { |row| rows << row }

      expect(result).to be_nil
      expect(rows.size).to eq(5)
      rows.each_with_index do |row, draw|
        expect(row.size).to eq(3)
        row.each_with_index { |value, arm| expect_pin(value, pinned[draw][arm], "draw #{draw}, arm #{arm}") }
      end
      # Per draw: one rng.rand (worth), one gaussian (lift), six gaussians (the rates); a gaussian takes two. 5 * 15.
      fresh = Random.new(20_261_030)
      75.times { fresh.rand }
      following = rng.rand
      expect(following).to eq(fresh.rand)
      expect_pin(following, 0.49790573891844858, "the next random number")
    end

    it "SL15 pins summary on seed 4242 with 2,000 draws at the default credible mass" do
      pinned = {
        a: { point: 0.24199999999999999, mean: 0.24067854433040264, lower: 0.14730560406449483,
             upper: 0.35765057397478728, p_best: 0.31850000000000001, expected_loss: 0.035513101078313278 },
        b: { point: 0.2455, mean: 0.24568128879184817, lower: 0.14809545169397306,
             upper: 0.37195923065103526, p_best: 0.316, expected_loss: 0.030510356616867733 },
        c: { point: 0.23899999999999999, mean: 0.24139462315818586, lower: 0.11952047664985094,
             upper: 0.39660996392022779, p_best: 0.36549999999999999, expected_loss: 0.034797022250530045 }
      }
      summary = linear_value.summary(**inputs, draws: 2_000, rng: Random.new(4242))

      expect(summary.keys).to eq(arms)
      pinned.each do |arm, values|
        expect(summary[arm].keys).to contain_exactly(*values.keys)
        values.each { |key, value| expect_pin(summary[arm][key], value, "#{arm} #{key}") }
      end
    end

    it "SL15 pins summary on seed 4242 at credible_mass 0.8: the same draws, a narrower interval" do
      pinned = {
        a: { lower: 0.17983899174511342, upper: 0.30555913561240866, p_best: 0.31850000000000001,
             expected_loss: 0.035513101078313278, mean: 0.24067854433040264 },
        b: { lower: 0.1755892564343266, upper: 0.32145349141013674, p_best: 0.316,
             expected_loss: 0.030510356616867733, mean: 0.24568128879184817 },
        c: { lower: 0.15371008314575563, upper: 0.33572438082123507, p_best: 0.36549999999999999,
             expected_loss: 0.034797022250530045, mean: 0.24139462315818586 }
      }
      summary = linear_value.summary(**inputs, draws: 2_000, rng: Random.new(4242), credible_mass: 0.8)

      pinned.each do |arm, values|
        values.each { |key, value| expect_pin(summary[arm][key], value, "#{arm} #{key}") }
      end
    end
  end

  describe "readings" do
    def rows_of(**overrides)
      linear_value.each_draw(**inputs(**overrides), draws: 40, rng: Random.new(8)).to_a
    end

    # A copy of the matrix with each [i, j, value] entry set.
    def changed(matrix, *entries)
      copy = matrix.map(&:dup)
      entries.each { |i, j, value| copy[i][j] = value }
      copy
    end

    # The call is refused, and the RNG it was handed has taken nothing.
    def expect_refused_untouched(label, &call)
      rng = Random.new(31)
      expect { call.call(rng) }.to raise_error(ArgumentError), label
      expect(rng.rand).to eq(Random.new(31).rand), "#{label}: the RNG moved before the refusal"
    end

    it "SL3 answers the arms as given, Strings too, and reads means and weights by Symbol or String" do
      by_string = linear_value.each_draw(arms: %w[a b c], parts: %w[p q], means: means, covariance: covariance,
                                         parameters: parameters,
                                         weights: ->(drawn) { { "p" => drawn.fetch(:base), q: drawn.fetch(:worth) * drawn.fetch(:lift) } },
                                         draws: 40, rng: Random.new(8)).to_a
      summary = linear_value.summary(**inputs(arms: %w[a b c]), draws: 10, rng: Random.new(8))

      expect(by_string).to eq(rows_of)
      expect(summary.keys).to eq(%w[a b c])
    end

    it "SL3 without a block draws every row at once: the Enumerator replays them, sized, the RNG left as a block leaves it" do
      lazy_rng = Random.new(12)
      block_rng = Random.new(12)
      rows = linear_value.each_draw(**inputs, draws: 25, rng: lazy_rng)
      streamed = []
      linear_value.each_draw(**inputs, draws: 25, rng: block_rng) { |row| streamed << row }

      expect(rows.size).to eq(25)
      expect(rows.to_a).to eq(streamed)
      expect(rows.to_a).to eq(streamed)
      expect(lazy_rng.rand).to eq(block_rng.rand)
    end

    it "SL3 takes 0 draws in each_draw (no row, no random number) and refuses 0 draws in summary" do
      rng = Random.new(5)

      expect(linear_value.each_draw(**inputs, draws: 0, rng: rng).to_a).to eq([])
      expect(rng.rand).to eq(Random.new(5).rand)
      expect_refused_untouched("summary at 0 draws") { |r| linear_value.summary(**inputs, draws: 0, rng: r) }
    end

    it "SL3 refuses a draw count that is not an Integer of at least 0, before any random number" do
      [ 10.0, -1, nil, "10" ].each do |draws|
        expect_refused_untouched("draws #{draws.inspect}") { |r| linear_value.each_draw(**inputs, draws: draws, rng: r).to_a }
        expect_refused_untouched("summary draws #{draws.inspect}") { |r| linear_value.summary(**inputs, draws: draws, rng: r) }
      end
    end

    it "SL3 refuses a credible_mass outside (0, 1), before any random number" do
      [ 0.0, 1.0, 1.5, -0.2, Float::NAN, "0.9", nil ].each do |mass|
        expect_refused_untouched("credible_mass #{mass.inspect}") do |r|
          linear_value.summary(**inputs, draws: 10, rng: r, credible_mass: mass)
        end
      end
    end

    it "SL3 refuses every malformed input before any random number" do
      eight = (1..8).map { |n| :"arm#{n}" }
      cases = {
        "8 arms" => inputs(arms: eight, means: eight.to_h { |arm| [ arm, [ 0.0, 0.0 ] ] },
                           covariance: Array.new(16) { |i| Array.new(16) { |j| i == j ? 1.0 : 0.0 } }),
        "no arm" => inputs(arms: [], covariance: []),
        "no part" => inputs(parts: [], means: { a: [], b: [], c: [] }, covariance: [], weights: ->(_drawn) { {} }),
        "arms not an Array" => inputs(arms: :a),
        "an arm twice, as a Symbol and a String" => inputs(arms: [ :a, :b, "a" ]),
        "an arm that is not a Symbol or a String" => inputs(arms: [ :a, :b, 3 ]),
        "a part twice" => inputs(parts: %i[p p]),
        "means not a Hash" => inputs(means: [ [ 0.1, 0.1 ] ] * 3),
        "means without an arm" => inputs(means: means.slice(:a, :b)),
        "means of the wrong length" => inputs(means: means.merge(c: [ 0.09 ])),
        "a mean that is not finite" => inputs(means: means.merge(c: [ 0.09, Float::NAN ])),
        "a mean that is not a number" => inputs(means: means.merge(c: [ 0.09, "0.05" ])),
        "a covariance of the wrong size" => inputs(covariance: covariance.first(5).map { |row| row.first(5) }),
        "a covariance that is not an Array" => inputs(covariance: { 0 => covariance }),
        "a covariance row that is too short" => inputs(covariance: covariance.each_with_index.map { |row, i| i == 3 ? row.first(5) : row }),
        "an asymmetric covariance" => inputs(covariance: changed(covariance, [ 1, 0, 0.00007 ])),
        "a covariance that is not positive semidefinite" => inputs(covariance: changed(covariance, [ 4, 5, 0.001 ], [ 5, 4, 0.001 ])),
        "weights that cannot be called" => inputs(weights: { p: 1.0, q: 1.0 }),
        "weights that answer no Hash" => inputs(weights: ->(_drawn) { [ 1.0, 1.0 ] }),
        "weights that leave out a part at the centres" => inputs(weights: ->(drawn) { { p: drawn.fetch(:base) } })
      }

      cases.each do |label, call_inputs|
        expect_refused_untouched(label) { |r| linear_value.each_draw(**call_inputs, draws: 3, rng: r).to_a }
        expect_refused_untouched("summary: #{label}") { |r| linear_value.summary(**call_inputs, draws: 3, rng: r) }
      end
    end

    it "SL3 refuses a malformed parameter before any random number" do
      cases = {
        "not a Hash" => 1.5,
        "no kind" => { clip: [ 0.0, 1.0 ] },
        "two kinds" => { fixed: 1.0, uniform: [ 0.0, 1.0 ] },
        "an unknown key" => { fixed: 1.0, scale: 2.0 },
        "a fixed value that is not finite" => { fixed: Float::INFINITY },
        "a fixed value that is not a number" => { fixed: "1.0" },
        "a uniform of one bound" => { uniform: [ 1.0 ] },
        "a uniform whose low end is above its high end" => { uniform: [ 3.0, 1.0 ] },
        "a uniform bound that is not finite" => { uniform: [ 0.0, Float::INFINITY ] },
        "a normal with a negative spread" => { normal: [ 0.5, -0.1 ] },
        "a normal that is not a pair" => { normal: 0.5 },
        "a clip whose low end is above its high end" => { normal: [ 0.5, 0.1 ], clip: [ 1.0, 0.0 ] },
        "a clip bound that is not a number" => { normal: [ 0.5, 0.1 ], clip: [ 0.0, nil ] },
        "a clip bound that is NaN" => { normal: [ 0.5, 0.1 ], clip: [ Float::NAN, 1.0 ] }
      }

      expect_refused_untouched("parameters not a Hash") do |r|
        linear_value.each_draw(**inputs(parameters: [ [ :base, { fixed: 1.5 } ] ]), draws: 3, rng: r).to_a
      end
      cases.each do |label, spec|
        expect_refused_untouched(label) do |r|
          linear_value.each_draw(**inputs(parameters: parameters.merge(lift: spec)), draws: 3, rng: r).to_a
        end
      end
    end

    it "SL3 refuses a weight that is not a finite number, at the centres or in a draw" do
      [ nil, Float::NAN, Float::INFINITY, "1.0" ].each do |bad|
        expect_refused_untouched("centre weight #{bad.inspect}") do |r|
          linear_value.summary(**inputs(weights: ->(_drawn) { { p: 1.0, q: bad } }), draws: 3, rng: r)
        end
      end
      # Fine at the centres (worth 4.0), not once a draw's worth passes 5.
      dropping = ->(drawn) { drawn.fetch(:worth) > 5.0 ? { p: 1.0 } : { p: 1.0, q: 1.0 } }
      not_finite = ->(drawn) { { p: 1.0, q: drawn.fetch(:worth) > 5.0 ? Float::NAN : 1.0 } }

      expect { linear_value.summary(**inputs(weights: dropping), draws: 200, rng: Random.new(2)) }
        .to raise_error(ArgumentError)
      expect { linear_value.each_draw(**inputs(weights: not_finite), draws: 200, rng: Random.new(2)).to_a }
        .to raise_error(ArgumentError)
    end

    it "SL3 leaves out a weight named for no part" do
      extra = ->(drawn) { weights.call(drawn).merge(r: 99.0, "s" => nil) }

      expect(rows_of(weights: extra)).to eq(rows_of)
    end

    it "SL3 takes a clip on any kind of parameter, an infinite bound too" do
      # base fixed 1.5 clipped to [0, 1] is 1.0 in every draw and at the centre; worth uniform on [2, 6] clipped to
      # [0, 3] is 3.0 at its centre (4.0); lift normal clipped to [0, infinity) never goes negative.
      seen = []
      recording = lambda do |drawn|
        seen << drawn
        { p: drawn.fetch(:base), q: drawn.fetch(:worth) * drawn.fetch(:lift) }
      end
      clipped = { base: { fixed: 1.5, clip: [ 0.0, 1.0 ] }, worth: { uniform: [ 2.0, 6.0 ], clip: [ 0.0, 3.0 ] },
                  lift: { normal: [ 0.0, 1.0 ], clip: [ 0.0, Float::INFINITY ] } }
      summary = linear_value.summary(**inputs(parameters: clipped, weights: recording), draws: 300, rng: Random.new(6))

      expect(seen.map { |drawn| drawn.fetch(:base) }.uniq).to eq([ 1.0 ])
      expect(seen.map { |drawn| drawn.fetch(:worth) }.max).to eq(3.0)
      expect(seen.map { |drawn| drawn.fetch(:lift) }.min).to eq(0.0)
      # The point: p weighted 1.0, q weighted 3.0 * 0.0 = 0.
      expect(summary[:a][:point]).to be_within(1e-15).of(0.12)
    end

    it "SL3 hands the weights callable the drawn values under the parameters' own names" do
      named = { "base" => { fixed: 1.5 }, worth: { "uniform" => [ 2.0, 6.0 ] }, lift: { normal: [ 0.5, 0.2 ], "clip" => [ 0.0, 1.0 ] } }
      by_name = ->(drawn) { { p: drawn.fetch("base"), q: drawn.fetch(:worth) * drawn.fetch(:lift) } }

      expect(rows_of(parameters: named, weights: by_name)).to eq(rows_of)
    end

    it "SL4 refuses a matrix asymmetric beyond 1e-12 of its pair's scale, and takes rounding noise beside zero" do
      # The scale of entries (i, j) and (j, i) is the larger of their sizes and sqrt(|a_ii| * |a_jj|). The factor reads
      # the lower triangle.
      expect(cholesky.factor([ [ 1.0, 1e-17 ], [ -1e-17, 1.0 ] ])).to eq([ [ 1.0, 0.0 ], [ -1e-17, 1.0 ] ])
      expect { cholesky.factor([ [ 1.0, 2e-12 ], [ 0.0, 1.0 ] ]) }.to raise_error(ArgumentError)
      # Scale sqrt(4e6 * 1e-6) = 2: a gap of 1e-7 is refused although the matrix is positive definite.
      expect { cholesky.factor([ [ 4e6, 1.0 ], [ 1.0 + 1e-7, 1e-6 ] ]) }.to raise_error(ArgumentError)
    end

    it "SL4 refuses a matrix that is not square, not an Array of rows, or holds an entry that is no finite number" do
      [
        [ [ 1.0, 0.0 ] ],
        [ [ 1.0 ], [ 0.0, 1.0 ] ],
        [ [ 1.0, nil ], [ nil, 1.0 ] ],
        [ [ Float::NAN ] ],
        [ [ Float::INFINITY ] ],
        [ [ 1.0, "0" ], [ "0", 1.0 ] ],
        [ 1.0 ],
        { 0 => [ 1.0 ] },
        nil
      ].each do |matrix|
        expect { cholesky.factor(matrix) }.to raise_error(ArgumentError), matrix.inspect
      end
    end

    it "SL4 factors an empty matrix to no rows and answers Floats for Integer and Rational entries" do
      expect(cholesky.factor([])).to eq([])
      lower = cholesky.factor([ [ 4, 2 ], [ Rational(2), 5 ] ])

      expect(lower).to eq([ [ 2.0, 0.0 ], [ 1.0, 2.0 ] ])
      expect(lower.flatten).to all(be_a(Float))
    end

    it "SL4 refuses a real number no Float holds, and a complex number with no imaginary part" do
      [ [ [ 10**400 ] ], [ [ Rational(10**400, 3) ] ], [ [ Complex(1, 0) ] ], [ [ 1.0, Complex(0, 0) ], [ 0.0, 1.0 ] ] ]
        .each do |matrix|
          expect { cholesky.factor(matrix) }.to raise_error(ArgumentError, /finite number/), matrix.inspect
        end
    end
  end

  # A pivot set to 0 drops its whole column from L. Under it, row i's residual r_ij = a_ij - sum over k < j of
  # L_ik * L_jk may only be what that pivot holds, sqrt(pivot * q_ij) with q_ij row i's own pivot at column j, plus the
  # rounding a pivot is given (1e-9 times the trace); a larger one is a part with no variance that covaries, which no
  # positive semidefinite matrix has.
  describe "zero pivots" do
    def product(lower)
      lower.map { |row| lower.map { |other| row.zip(other).sum { |x, y| x * y } } }
    end

    def expect_rebuilt(lower, matrix, label)
      rounding = 1e-9 * matrix.each_index.sum { |i| matrix[i][i] }
      error = product(lower).zip(matrix).flat_map { |row, given| row.zip(given).map { |x, y| (x - y).abs } }.max
      expect(error).to be <= rounding, "#{label}: L * L^T is #{error} from the matrix"
    end

    def expect_factor(matrix, pinned, label)
      lower = cholesky.factor(matrix)
      expect(lower.size).to eq(pinned.size)
      lower.each_with_index do |row, i|
        expect(row.size).to eq(pinned.size)
        row.each_with_index { |value, j| expect_pin(value, pinned[i][j], "#{label} L[#{i}][#{j}]") }
      end
      lower
    end

    it "SL4 refuses a zero pivot with a residual under it, as not positive semidefinite" do
      {
        # A part with no variance that covaries: the first pivot is 0 and the residual under it is the entry itself.
        "a zero diagonal entry beside 1.0" => [ [ 0.0, 1.0 ], [ 1.0, 1.0 ] ],
        "a zero diagonal entry beside 0.6" => [ [ 0.0, 0.6 ], [ 0.6, 0.4 ] ],
        "two zero diagonal entries that covary with a third part" =>
          [ [ 0.0, 0.0, 0.5 ], [ 0.0, 0.0, 0.3 ], [ 0.5, 0.3, 0.4 ] ],
        "zero diagonal entries throughout but one" =>
          [ [ 0.0, 0.0, -0.3, 0.5 ], [ 0.0, 0.0, 0.6, 0.0 ], [ -0.3, 0.6, 0.0, 0.1 ], [ 0.5, 0.0, 0.1, 0.0 ] ],
        # L00 = 1, L10 = L20 = 1: the pivot of column 1 is 1 - 1 = 0, and row 2's residual under it 2 - 1 * 1 = 1.
        "a pivot that the first part's elimination brings to 0" =>
          [ [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0, 2.0 ], [ 1.0, 2.0, 5.0 ] ],
        # The pivot 1e-13 is below 1e-12 * 1 and holds sqrt(1e-13 * 1) = 3.16e-7: a residual of 4e-7 is more.
        "a pivot under the zero threshold with more residual than it holds" => [ [ 1e-13, 4e-7 ], [ 4e-7, 1.0 ] ],
        # The pivot of column 1 is (1 - 1e-14) - 1, within rounding of 0; row 2's residual 1e-6 is not rounding.
        "a pivot that rounding took below 0 with a residual of 1e-6 under it" =>
          [ [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0 - 1e-14, 1.0 + 1e-6 ], [ 1.0, 1.0 + 1e-6, 2.0 ] ],
        # Every diagonal entry is 0, so the rounding is 0 too: any covariance is refused.
        "an all-zero diagonal with the smallest covariance" => [ [ 0.0, 1e-300 ], [ 1e-300, 0.0 ] ]
      }.each do |label, matrix|
        expect { cholesky.factor(matrix) }
          .to raise_error(ArgumentError, /not positive semidefinite \(residual .* under the zero pivot/), label
      end
    end

    it "SL4 still factors a positive semidefinite matrix of lower rank, whose zero pivots carry only rounding" do
      # A part with no variance and no covariance (row and column 1 and 3 zero): L00 = 2, L20 = 2 / 2 = 1,
      # L22 = sqrt(5 - 1) = 2, and the zero rows of L.
      zero_rows = [ [ 4.0, 0.0, 2.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 2.0, 0.0, 5.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
      lower = expect_factor(zero_rows, [ [ 2.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 1.0, 0.0, 2.0, 0.0 ],
                                         [ 0.0, 0.0, 0.0, 0.0 ] ], "zero rows")
      expect(product(lower)).to eq(zero_rows)

      # The Gram matrix of the rows (1, 2), (2, 4), (1, 1): the second is twice the first, so column 1's pivot
      # 20 - (10 / sqrt 5)^2 and row 2's residual 6 - (3 / sqrt 5) * (10 / sqrt 5) are 0 but for rounding.
      # L00 = sqrt 5, L10 = 2 sqrt 5, L20 = 3 / sqrt 5, L22 = sqrt(2 - 9 / 5) = sqrt 0.2.
      whole = [ [ 5.0, 10.0, 3.0 ], [ 10.0, 20.0, 6.0 ], [ 3.0, 6.0, 2.0 ] ]
      lower = expect_factor(whole, [ [ Math.sqrt(5.0), 0.0, 0.0 ], [ 2.0 * Math.sqrt(5.0), 0.0, 0.0 ],
                                     [ 3.0 / Math.sqrt(5.0), 0.0, Math.sqrt(0.2) ] ], "whole numbers")
      expect_rebuilt(lower, whole, "whole numbers")

      # The Gram matrix of the rows (0.1, 0.3), (0.2, 0.6), (0.7, 0.1), written in decimals no Float holds exactly:
      # L00 = sqrt 0.1, L10 = 2 sqrt 0.1, L20 = sqrt 0.1, L22 = sqrt(0.5 - 0.1) = sqrt 0.4.
      decimals = [ [ 0.1, 0.2, 0.1 ], [ 0.2, 0.4, 0.2 ], [ 0.1, 0.2, 0.5 ] ]
      lower = expect_factor(decimals, [ [ Math.sqrt(0.1), 0.0, 0.0 ], [ 2.0 * Math.sqrt(0.1), 0.0, 0.0 ],
                                        [ Math.sqrt(0.1), 0.0, Math.sqrt(0.4) ] ], "decimals")
      expect_rebuilt(lower, decimals, "decimals")

      # Column 1's pivot (1 - 1e-14) - 1 is about -1e-14, within rounding; the residual under it is 1 - 1 * 1 = 0.
      rounded = [ [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0 - 1e-14, 1.0 ], [ 1.0, 1.0, 2.0 ] ]
      lower = expect_factor(rounded, [ [ 1.0, 0.0, 0.0 ], [ 1.0, 0.0, 0.0 ], [ 1.0, 0.0, 1.0 ] ], "rounded")
      expect_rebuilt(lower, rounded, "rounded")
    end

    it "SL4 factors a positive definite matrix whose pivot falls under the zero threshold, its residual within it" do
      # det = 1e-13 - 9e-14 > 0. The pivot 1e-13 is below 1e-12 * 1, so it goes to 0 with its column (the part is
      # drawn at its mean); the residual 3e-7 is within the sqrt(1e-13 * 1) = 3.16e-7 that pivot holds, so the
      # matrix is not refused. L11 = sqrt(1 - 0) = 1.
      expect_factor([ [ 1e-13, 3e-7 ], [ 3e-7, 1.0 ] ], [ [ 0.0, 0.0 ], [ 0.0, 1.0 ] ], "small pivot")
    end

    it "SL4 factors seeded Gram matrices of lower rank, and refuses each once a part with no variance covaries" do
      rng = Random.new(20_261_030)
      40.times do |trial|
        size = 2 + rng.rand(11)
        rank = 1 + rng.rand(size - 1)
        # Row `none` of the generating rows is all zero, so the matrix's row and column `none` are exactly zero.
        none = rng.rand(size)
        rows = Array.new(size) { |i| Array.new(rank) { i == none ? 0.0 : rng.rand - 0.5 } }
        matrix = Array.new(size) { |i| Array.new(size) { |j| rows[i].zip(rows[j]).sum { |x, y| x * y } } }
        matrix = Array.new(size) { |i| Array.new(size) { |j| i <= j ? matrix[i][j] : matrix[j][i] } }
        expect_rebuilt(cholesky.factor(matrix), matrix, "trial #{trial}")

        # Give the part with no variance a covariance with the part of the largest variance, half that variance. Before
        # that part, the residual under the zero pivot is the covariance itself; after it, the zero part's pivot falls
        # to at most -0.25 times that variance, far below -1e-9 times a trace of at most 12 such variances.
        other = (0...size).max_by { |i| matrix[i][i] }
        covaried = matrix.map(&:dup)
        covaried[none][other] = covaried[other][none] = 0.5 * matrix[other][other]
        expect { cholesky.factor(covaried) }
          .to raise_error(ArgumentError, /not positive semidefinite/), "trial #{trial} size #{size} rank #{rank}"
      end
    end
  end
end
