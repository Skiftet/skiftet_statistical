# frozen_string_literal: true

# SL15: the pins of Prequential's outputs, and the readings where SL11 is silent. Every pinned value was checked by hand
# first: an independent reference at 250 significant digits (exact fractions and the exact binary value of every Float
# input, its own Machin pi, a Taylor series of erf with Newton for the normal quantile, closed forms of the gamma
# function at whole numbers and half-integers) gave the digits below, and this file ran red against them before the
# module existed. A pinned value is never edited.
#
# One tolerance rule (SL15) for every Float this file compares with a pinned value. The module itself only adds,
# divides and takes square roots of log probabilities, so an output on plain inputs is compared at 1e-12 relative. The
# log probabilities a caller scores usually come through Math.lgamma, whose last bit differs between math libraries, so
# the block whose inputs come through it is compared at 1e-9 relative, LGAMMA_RELATIVE. A pinned 0.0 or 1.0 is compared
# exactly; a size must stay equal.
RSpec.describe "SkiftetStatistical::Prequential pins" do
  def prequential = SkiftetStatistical::Prequential

  def lgamma_relative = 1e-9

  def arithmetic_relative = 1e-12

  def log2 = Math.log(2.0)

  def expect_pin(actual, pinned, relative, label)
    expect(actual).to be_a(Float), "#{label}: #{actual.inspect} is not a Float"
    if pinned.zero? || pinned == 1.0
      expect(actual).to eq(pinned), "#{label}: #{actual} is not exactly #{pinned}"
    else
      error = (actual - pinned).abs / pinned.abs
      expect(error).to be <= relative, "#{label}: #{actual} is #{error} from #{pinned}"
    end
  end

  def row(cluster, logs) = { cluster: cluster, log_probs: logs }

  # Seven events in four clusters, whose keys are of three kinds: the probability three depths gave each outcome.
  def mixed_rows
    [
      [ [ 7, 1 ], 0.3, 0.45, 0.6 ], [ [ 7, 1 ], 0.2, 0.15, 0.35 ], [ [ 7, 2 ], 0.55, 0.4, 0.25 ],
      [ "x", 0.1, 0.35, 0.4 ], [ "x", 0.65, 0.7, 0.9 ], [ "x", 0.05, 0.1, 0.02 ], [ :y, 0.45, 0.6, 0.75 ]
    ].map { |cluster, a, b, c| row(cluster, { a: Math.log(a), b: Math.log(b), c: Math.log(c) }) }
  end

  # Five events whose log probabilities are exact binary fractions, so no library's log stands between input and pin.
  def literal_rows
    [ [ 1, -1.25, -0.75 ], [ 1, -0.5, -0.625 ], [ 2, -2.75, -1.0 ], [ 3, -0.375, -0.875 ], [ 3, -1.125, -0.25 ] ]
      .map { |cluster, a, b| row(cluster, { a: a, b: b }) }
  end

  # log NB(s; k, k / (k + e)) by Math.lgamma, the negative binomial's mass at s with mean e and strength k.
  def log_negative_binomial(s, k, e)
    Math.lgamma(k + s).first - Math.lgamma(k).first - Math.lgamma(s + 1.0).first +
      (k * Math.log(k / (k + e))) + (s * Math.log(e / (k + e)))
  end

  # Six events in four clusters, scored by three strengths of one predictive: (cluster, s, e).
  def lgamma_rows
    [ [ :u, 3.0, 2.0 ], [ :u, 0.0, 1.5 ], [ :v, 7.0, 4.0 ], [ :w, 1.0, 2.5 ], [ :w, 12.0, 5.0 ], [ :z, 2.0, 0.75 ] ]
      .map do |cluster, s, e|
        row(cluster, { wide: log_negative_binomial(s, 0.5, e), middle: log_negative_binomial(s, 2.5, e),
                       narrow: log_negative_binomial(s, 8.5, e) })
      end
  end

  def whole_sums = { p: 12.5, q: -3.25, r: 7.75, s: 0.625, t: 21.0 }

  def whole_counts = { p: 40, q: 12, r: 31, s: 9, t: 58 }

  describe ".score" do
    it "SL15 pins the mean of -log2 p on rows of Math.log" do
      { a: 2.0483911178956683, b: 1.6141338310302434, c: 1.6834805206259922 }.each do |depth, pinned|
        expect_pin(prequential.score(mixed_rows, depth: depth), pinned, arithmetic_relative, "depth #{depth}")
      end
    end

    it "SL15 pins the mean of -log2 p on exact binary log probabilities" do
      { a: 1.7312340490667561, b: 1.0098865286222744 }.each do |depth, pinned|
        expect_pin(prequential.score(literal_rows, depth: depth), pinned, arithmetic_relative, "depth #{depth}")
      end
    end

    it "SL15 pins a score through log-gamma at the wider tolerance" do
      { wide: 3.6588139075256243, middle: 3.3055473877910779, narrow: 3.3969360648608496 }.each do |depth, pinned|
        expect_pin(prequential.score(lgamma_rows, depth: depth), pinned, lgamma_relative, "depth #{depth}")
      end
    end

    it "SL15 pins a probability of one half at exactly 1.0 bit and a certain outcome at exactly 0.0" do
      expect_pin(prequential.score([ row(1, { d: -log2 }) ], depth: :d), 1.0, arithmetic_relative, "one half")
      zero = prequential.score([ row(1, { d: 0.0 }), row(2, { d: 0 }) ], depth: :d)
      expect_pin(zero, 0.0, arithmetic_relative, "certain")
      expect(1.0 / zero).to eq(Float::INFINITY), "a score of nothing is +0.0, not -0.0"
    end
  end

  describe ".gains" do
    it "SL15 pins each depth's bits and standard error on rows of Math.log" do
      gains = prequential.gains(mixed_rows, depths: %i[a b c])

      expect_pin(gains[:b][:bits], 0.43425728686542493, arithmetic_relative, "b bits")
      expect_pin(gains[:b][:se], 0.32509562423182962, arithmetic_relative, "b se")
      expect_pin(gains[:c][:bits], -0.069346689595748809, arithmetic_relative, "c bits")
      expect_pin(gains[:c][:se], 0.40767191800904196, arithmetic_relative, "c se")
      expect(gains[:b].values_at(:events, :clusters)).to eq([ 7, 4 ])
      expect(gains[:c].values_at(:events, :clusters)).to eq([ 7, 4 ])
    end

    it "SL15 pins the gain on exact binary log probabilities" do
      gains = prequential.gains(literal_rows, depths: %i[a b])

      expect_pin(gains[:b][:bits], 0.72134752044448170, arithmetic_relative, "bits")
      expect_pin(gains[:b][:se], 0.54101064033336128, arithmetic_relative, "se")
      expect(gains[:b].values_at(:events, :clusters)).to eq([ 5, 3 ])
    end

    it "SL15 pins a gain through log-gamma at the wider tolerance" do
      gains = prequential.gains(lgamma_rows, depths: %i[wide middle narrow])

      expect_pin(gains[:middle][:bits], 0.35326651973454643, lgamma_relative, "middle bits")
      expect_pin(gains[:middle][:se], 0.14011247154984367, lgamma_relative, "middle se")
      expect_pin(gains[:narrow][:bits], -0.091388677069771736, lgamma_relative, "narrow bits")
      expect_pin(gains[:narrow][:se], 0.13895248590719386, lgamma_relative, "narrow se")
      expect(gains[:narrow].values_at(:events, :clusters)).to eq([ 6, 4 ])
    end

    it "SL15 pins depths that say the same at exactly 0.0, standard error included" do
      rows = [ row(1, { a: -0.5, b: -0.5 }), row(2, { a: -1.5, b: -1.5 }), row(3, { a: -0.25, b: -0.25 }) ]
      gains = prequential.gains(rows, depths: %i[a b])

      expect_pin(gains[:b][:bits], 0.0, arithmetic_relative, "bits")
      expect_pin(gains[:b][:se], 0.0, arithmetic_relative, "se")
    end

    it "SL11 widens the standard error when events in a cluster move together" do
      up = { before: -2.0 * log2, after: -log2 }
      down = { before: -log2, after: -2.0 * log2 }

      # Gains (+1, +1, -1, -1). One cluster each: totals (1, 1, -1, -1), mean 0, se = sqrt(4/3 * 4) / 4 = sqrt(1/3).
      apart = prequential.gains([ row(1, up), row(2, up), row(3, down), row(4, down) ], depths: %i[before after])
      expect(apart[:after][:se]).to be_within(1e-12).of(Math.sqrt(1.0 / 3.0))

      # Together: totals (2, -2), se = sqrt(2 * 8) / 4 = 1, nearly twice as wide on the same four gains.
      together = prequential.gains([ row(1, up), row(1, up), row(2, down), row(2, down) ], depths: %i[before after])
      expect(together[:after][:se]).to be_within(1e-12).of(1.0)

      # Cancelling inside each cluster: totals (0, 0), so the clusters agree and no spread is left.
      mixed = prequential.gains([ row(1, up), row(1, down), row(2, up), row(2, down) ], depths: %i[before after])
      expect(mixed[:after][:se]).to be_within(1e-12).of(0.0)
    end

    it "SL11 gives no standard error under two clusters, the mean kept" do
      gains = prequential.gains([ row(1, { a: -2.0, b: -1.0 }), row(1, { a: -1.0, b: -1.0 }) ], depths: %i[a b])

      expect(gains[:b][:bits]).to be_within(1e-12).of(0.5 / log2)
      expect(gains[:b][:se]).to be_nil
      expect(gains[:b].values_at(:events, :clusters)).to eq([ 2, 1 ])
    end

    it "SL11 answers every depth with nil figures on no rows" do
      expect(prequential.gains([], depths: %i[a b c])).to eq(
        b: { bits: nil, se: nil, events: 0, clusters: 0 }, c: { bits: nil, se: nil, events: 0, clusters: 0 }
      )
    end

    it "SL11 reads a one-shot Enumerable once" do
      reads = 0
      once = Enumerator.new do |yielder|
        reads += 1
        literal_rows.each { |item| yielder << item }
      end

      expect(prequential.gains(once.lazy, depths: %i[a b]).dig(:b, :events)).to eq(5)
      expect(reads).to eq(1)
    end

    it "SL11 answers no depth for one depth, its rows still read" do
      expect(prequential.gains(literal_rows, depths: %i[a])).to eq({})
      expect { prequential.gains([ row(1, { a: Float::NAN }) ], depths: %i[a]) }.to raise_error(ArgumentError)
    end

    it "SL11 reads depths and row fields by Symbol or String and answers the depths as given" do
      strings = literal_rows.map do |item|
        { "cluster" => item[:cluster], "log_probs" => item[:log_probs].transform_keys(&:to_s) }
      end
      plain = prequential.gains(literal_rows, depths: %i[a b])[:b]

      expect(prequential.gains(strings, depths: %i[a b])[:b]).to eq(plain)
      expect(prequential.gains(literal_rows, depths: %w[a b])).to eq("b" => plain)
      expect(prequential.score(strings, depth: :a)).to eq(prequential.score(literal_rows, depth: "a"))
    end

    it "SL11 leaves a depth it was not asked for unread and carries a row's other fields" do
      rows = literal_rows.map { |item| item.merge(note: "kept", log_probs: item[:log_probs].merge(other: nil)) }

      expect(prequential.gains(rows, depths: %i[a b])).to eq(prequential.gains(literal_rows, depths: %i[a b]))
      expect(prequential.score(rows, depth: :a)).to eq(prequential.score(literal_rows, depth: :a))
    end

    it "SL11 refuses a row that does not hold what it scores, naming no result" do
      good = { a: -1.0, b: -1.0 }
      [
        [ row(1, good), :not_a_row ],
        [ row(1, good), { log_probs: good } ],
        [ row(1, good), row(nil, good) ],
        [ row(1, good), { cluster: 2 } ],
        [ row(1, good), row(2, :not_a_hash) ],
        [ row(1, good), row(2, { a: -1.0 }) ],
        [ row(1, good), { cluster: 2, "cluster" => 3, log_probs: good } ],
        [ row(1, good), row(2, { a: -1.0, b: -1.0, "b" => -2.0 }) ]
      ].each do |rows|
        expect { prequential.gains(rows, depths: %i[a b]) }.to raise_error(ArgumentError), rows.inspect
      end
    end

    it "SL11 refuses depths that are no list of names, each once" do
      [ nil, :a, [], [ :a, :a ], [ :a, "a" ], [ :a, 1 ], [ :a, nil ] ].each do |depths|
        expect { prequential.gains(literal_rows, depths: depths) }.to raise_error(ArgumentError), depths.inspect
      end
    end

    it "SL11 refuses rows that are no Enumerable" do
      [ nil, 3, "rows" ].each do |rows|
        expect { prequential.gains(rows, depths: %i[a b]) }.to raise_error(ArgumentError), rows.inspect
        expect { prequential.score(rows, depth: :a) }.to raise_error(ArgumentError), rows.inspect
      end
    end
  end

  describe "a log probability" do
    def scored(value) = [ row(1, { a: value, b: -1.0 }) ]

    it "SL11 refuses what is no finite number, a probability of zero's -Infinity among them" do
      bad = [ Float::NAN, Float::INFINITY, -Float::INFINITY, nil, "-1.0", Complex(-1.0, 0.0), -(10**400),
              Rational(-(10**400), 1) ]
      bad.each do |value|
        expect { prequential.score(scored(value), depth: :a) }.to raise_error(ArgumentError), value.inspect
        expect { prequential.gains(scored(value), depths: %i[a b]) }.to raise_error(ArgumentError), value.inspect
      end
    end

    it "SL11 refuses one above 0, which is no probability's, and takes exactly 0 as a certain outcome" do
      expect { prequential.score(scored(0.5), depth: :a) }.to raise_error(ArgumentError, /at most 0/)
      expect { prequential.gains(scored(1e-300), depths: %i[a b]) }.to raise_error(ArgumentError, /at most 0/)
      expect(prequential.score(scored(0.0), depth: :a)).to eq(0.0)
      expect(prequential.score(scored(-0.0), depth: :a)).to eq(0.0)
    end

    it "SL11 reads an Integer and a Rational as the Float they equal" do
      rows = [ row(1, { a: -1, b: Rational(-1, 2) }), row(2, { a: -2, b: Rational(-3, 4) }) ]
      floats = [ row(1, { a: -1.0, b: -0.5 }), row(2, { a: -2.0, b: -0.75 }) ]

      expect(prequential.gains(rows, depths: %i[a b])).to eq(prequential.gains(floats, depths: %i[a b]))
      expect(prequential.score(rows, depth: :b)).to eq(prequential.score(floats, depth: :b))
    end

    it "SL11 refuses a total beyond a Float's range instead of answering NaN or infinity" do
      rows = [ row(1, { a: -1e308, b: -1.0 }), row(1, { a: -1e308, b: -1.0 }), row(2, { a: -1e308, b: -1.0 }) ]

      expect { prequential.score(rows, depth: :a) }.to raise_error(ArgumentError)
      expect { prequential.gains(rows, depths: %i[a b]) }.to raise_error(ArgumentError)

      # The gains cancel, so the mean is a plain 0.0 and only the squares leave a Float's range.
      cancelling = [ row(1, { a: -1e200, b: 0.0 }), row(2, { a: 0.0, b: -1e200 }) ]
      expect { prequential.gains(cancelling, depths: %i[a b]) }.to raise_error(ArgumentError)
    end
  end

  describe ".score on what it is given" do
    it "SL11 is nil on no rows" do
      expect(prequential.score([], depth: :a)).to be_nil
    end

    it "SL11 reads a one-shot Enumerable once and reads no cluster" do
      reads = 0
      once = Enumerator.new do |yielder|
        reads += 1
        yielder << { log_probs: { a: -log2 } }
      end

      expect(prequential.score(once.lazy, depth: :a)).to eq(1.0)
      expect(reads).to eq(1)
    end

    it "SL11 refuses a depth that is no name" do
      [ nil, 1, [ :a ] ].each do |depth|
        expect { prequential.score(literal_rows, depth: depth) }.to raise_error(ArgumentError), depth.inspect
      end
    end

    it "SL11 refuses a row without the depth" do
      expect { prequential.score(literal_rows, depth: :z) }.to raise_error(ArgumentError)
    end
  end

  describe ".clustered_mean" do
    {
      0.5 => [ 0.21122149427590513, 0.30377850572409487 ],
      0.8 => [ 0.16956939550299029, 0.34543060449700971 ],
      0.9 => [ 0.14464229094208977, 0.37035770905791023 ],
      0.95 => [ 0.12302174521378873, 0.39197825478621127 ],
      0.99 => [ 0.080765615031325966, 0.43423438496867403 ],
      0.999 => [ 0.031728355383658344, 0.48327164461634166 ]
    }.each do |level, (lower, upper)|
      it "SL15 pins the bounds at level #{level}" do
        result = prequential.clustered_mean(sums: whole_sums, counts: whole_counts, level: level)

        expect_pin(result[:mean], 0.2575, arithmetic_relative, "mean")
        expect_pin(result[:lower], lower, arithmetic_relative, "lower")
        expect_pin(result[:upper], upper, arithmetic_relative, "upper")
        expect(result[:n]).to eql(150)
        expect(result[:clusters]).to eql(5)
      end
    end

    it "SL15 pins real-valued counts, whose summed count stays real" do
      result = prequential.clustered_mean(sums: { p: 3.75, q: -1.5, r: 2.125 }, counts: { p: 4.5, q: 2.25, r: 3.75 })

      expect_pin(result[:mean], 0.41666666666666667, arithmetic_relative, "mean")
      expect_pin(result[:lower], -0.29804042516465324, arithmetic_relative, "lower")
      expect_pin(result[:upper], 1.1313737584979866, arithmetic_relative, "upper")
      expect(result[:n]).to eql(10.5)
    end

    it "SL15 pins one large cluster among small ones, where the event counts weigh the residuals" do
      result = prequential.clustered_mean(sums: { big: 300.0, m: 1.0, o: -2.0, v: 0.5 },
                                          counts: { big: 1000, m: 3, o: 2, v: 1 })

      expect_pin(result[:mean], 0.29771371769383698, arithmetic_relative, "mean")
      expect_pin(result[:lower], 0.28991553411069581, arithmetic_relative, "lower")
      expect_pin(result[:upper], 0.30551190127697815, arithmetic_relative, "upper")
      expect(result[:n]).to eql(1006)
    end

    it "SL15 pins the widest and the narrowest level a Float can hold, both finite" do
      sums = { p: 2.0, q: -1.0, r: 0.5 }
      counts = { p: 2, q: 2, r: 1 }

      widest = prequential.clustered_mean(sums: sums, counts: counts, level: 1.0 - (2.0**-53))
      expect_pin(widest[:lower], -4.0374676719386022, arithmetic_relative, "widest lower")
      expect_pin(widest[:upper], 4.6374676719386022, arithmetic_relative, "widest upper")

      narrowest = prequential.clustered_mean(sums: sums, counts: counts, level: 1e-6)
      expect_pin(narrowest[:lower], 0.29999934443163971, arithmetic_relative, "narrowest lower")
      expect_pin(narrowest[:upper], 0.30000065556836029, arithmetic_relative, "narrowest upper")
    end

    it "SL15 pins clusters that all hold the mean at exact bounds" do
      zero = prequential.clustered_mean(sums: { a: 0.0, b: 0.0 }, counts: { a: 1, b: 3 })
      expect_pin(zero[:mean], 0.0, arithmetic_relative, "zero mean")
      expect_pin(zero[:lower], 0.0, arithmetic_relative, "zero lower")
      expect_pin(zero[:upper], 0.0, arithmetic_relative, "zero upper")

      one = prequential.clustered_mean(sums: { a: 3.0, b: 2.0 }, counts: { a: 3, b: 2 })
      expect_pin(one[:mean], 1.0, arithmetic_relative, "one mean")
      expect_pin(one[:lower], 1.0, arithmetic_relative, "one lower")
      expect_pin(one[:upper], 1.0, arithmetic_relative, "one upper")
    end

    it "SL11 pairs a cluster's total and count by key, never by position" do
      reversed = whole_counts.to_a.reverse.to_h

      expect(prequential.clustered_mean(sums: whole_sums, counts: reversed))
        .to eq(prequential.clustered_mean(sums: whole_sums, counts: whole_counts))
    end

    it "SL11 reads an Integer and a Rational as the Float they equal, keeping the summed count as given" do
      exact = prequential.clustered_mean(sums: { a: 3, b: Rational(-1, 2) }, counts: { a: 2, b: 2 })
      float = prequential.clustered_mean(sums: { a: 3.0, b: -0.5 }, counts: { a: 2, b: 2 })

      expect(exact).to eq(float)
      expect(exact[:n]).to eql(4)
      expect(prequential.clustered_mean(sums: { a: 1.0 }, counts: { a: Rational(3, 2) })[:n]).to eql(Rational(3, 2))
    end

    it "SL11 answers no mean and no bounds with no cluster at all" do
      expect(prequential.clustered_mean(sums: {}, counts: {}))
        .to eq(mean: nil, lower: nil, upper: nil, n: 0, clusters: 0)
    end

    it "SL11 takes cluster keys of any kind as plain Hash keys, nil inside a pair among them" do
      keys = [ [ 1, "2026-01-01" ], [ nil, "2026-01-01" ], 7, "s", :s ]
      sums = keys.zip([ 1.0, 2.0, 3.0, 4.0, 5.0 ]).to_h
      counts = keys.zip([ 1, 1, 1, 1, 1 ]).to_h
      result = prequential.clustered_mean(sums: sums, counts: counts)

      expect(result[:mean]).to be_within(1e-12).of(3.0)
      expect(result[:clusters]).to eq(5)
    end

    it "SL11 refuses a nil cluster, as a row's nil cluster is refused" do
      expect { prequential.clustered_mean(sums: { nil => 1.0, a: 1.0 }, counts: { nil => 1, a: 1 }) }
        .to raise_error(ArgumentError, /nil/)
    end

    it "SL11 refuses a level that is no number strictly between 0 and 1, under two clusters too" do
      bad = [ 0, 0.0, 1, 1.0, 1.5, -0.1, Float::NAN, Float::INFINITY, nil, "0.95", Complex(0.5, 0.0), 10**400 ]
      bad.each do |level|
        expect { prequential.clustered_mean(sums: whole_sums, counts: whole_counts, level: level) }
          .to raise_error(ArgumentError), level.inspect
        expect { prequential.clustered_mean(sums: { a: 1.0 }, counts: { a: 1 }, level: level) }
          .to raise_error(ArgumentError), level.inspect
      end
    end

    it "SL11 takes a level as any real number" do
      expect(prequential.clustered_mean(sums: whole_sums, counts: whole_counts, level: Rational(1, 2)))
        .to eq(prequential.clustered_mean(sums: whole_sums, counts: whole_counts, level: 0.5))
    end

    it "SL11 refuses sums and counts that are no Hash or not on one key set" do
      [
        [ nil, {} ], [ {}, nil ], [ [ 1.0 ], [ 1 ] ], [ { a: 1.0 }, {} ], [ {}, { a: 1 } ], [ { a: 1.0 }, { b: 1 } ],
        [ { a: 1.0, b: 1.0 }, { a: 1 } ], [ { a: 1.0 }, { a: 1, b: 1 } ], [ { a: 1.0 }, { "a" => 1 } ]
      ].each do |sums, counts|
        expect { prequential.clustered_mean(sums: sums, counts: counts) }
          .to raise_error(ArgumentError), [ sums, counts ].inspect
      end
    end

    it "SL11 refuses a total or a count that is no finite number, and a count not above 0" do
      bad_numbers = [ Float::NAN, Float::INFINITY, -Float::INFINITY, nil, "1", Complex(1.0, 0.0), 10**400 ]
      bad_numbers.each do |value|
        expect { prequential.clustered_mean(sums: { a: value }, counts: { a: 1 }) }
          .to raise_error(ArgumentError), value.inspect
        expect { prequential.clustered_mean(sums: { a: 1.0 }, counts: { a: value }) }
          .to raise_error(ArgumentError), value.inspect
      end
      [ 0, 0.0, -1, -0.5 ].each do |count|
        expect { prequential.clustered_mean(sums: { a: 0.0 }, counts: { a: count }) }
          .to raise_error(ArgumentError, /above 0/), count.inspect
      end
    end

    it "SL11 refuses totals beyond a Float's range instead of answering NaN or infinity" do
      [
        [ { a: 1e308, b: 1e308 }, { a: 1, b: 1 } ],
        [ { a: 1e308, b: -1e308 }, { a: 1e-300, b: 1e-300 } ],
        [ { a: 1.0, b: 1.0 }, { a: 1e308, b: 1e308 } ],
        [ { a: 1.0, b: 1.0 }, { a: 10**308, b: 10**308 } ],
        [ { a: 1e308, b: 1e308 }, { a: 0.25, b: 1e-300 } ],
        [ { a: 1e308 }, { a: 1e-300 } ]
      ].each do |sums, counts|
        expect { prequential.clustered_mean(sums: sums, counts: counts) }
          .to raise_error(ArgumentError), [ sums, counts ].inspect
      end
    end
  end
end
