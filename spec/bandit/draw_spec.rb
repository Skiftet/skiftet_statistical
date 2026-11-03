# frozen_string_literal: true

# The randomness helpers: a text becomes a uniform, a list of parts becomes a uniform or a seed, and a uniform picks
# a key by weight, answering the key and its share. Every literal below was worked outside the library: the first 8
# bytes of SHA256 of the text (the parts joined by NUL, nil read as "", a number by its decimal text), big-endian,
# shifted right by 11 and divided by 2**53; the weighted draw by a running sum written out by hand.
RSpec.describe "SkiftetStatistical::Bandit::Draw" do
  def draw = SkiftetStatistical::Bandit::Draw

  describe "keyed uniforms and seeds" do
    it "LO1 uniform is the first 8 bytes of SHA256(text), big-endian, shifted right 11, over 2**53" do
      expect(draw.uniform("t")).to eq(0.8895498694938413)
    end

    it "LO1 uniform_for is the uniform of the parts joined by NUL" do
      expect(draw.uniform_for("t", "explore")).to eq(0.856909028643178)
      expect(draw.uniform_for("t", "priority", "a")).to eq(0.7111955205766066)
      expect(draw.uniform_for("page", "draw", "item-3")).to eq(0.7042127664619353)
      expect(draw.uniform_for("t", "explore")).to eq(draw.uniform("t\u0000explore"))
      expect(draw.uniform_for("t")).to eq(draw.uniform("t"))
    end

    it "LO1 uniform_for reads nil as an empty part and a number by its decimal text" do
      expect(draw.uniform_for("t", nil, "a")).to eq(0.2535285453689655)
      expect(draw.uniform_for("t", nil, "a")).to eq(draw.uniform("t\u0000\u0000a"))
      expect(draw.uniform_for("t", 7)).to eq(0.8440543858434072)
      expect(draw.uniform_for("t", 7)).to eq(draw.uniform_for("t", "7"))
    end

    it "LO1 seed_for is the same 8 bytes read as one Integer below 2**64" do
      expect(draw.seed_for("t", "explore")).to eq(15_807_181_545_831_752_473)
      expect(draw.seed_for("tok", 12, nil)).to eq(3_669_298_782_440_433_199)
      expect(draw.seed_for("k")).to eq(9_391_345_706_673_393_910)
      expect(draw.seed_for("k")).to be_a(Integer)
    end

    it "LO1 a part list's uniform is its seed's top 53 bits over 2**53, in [0, 1)" do
      part_lists = [ %w[t explore], [ "tok", 12, nil ], [ "k" ], [ "page", "draw", "item-3" ], [ "", "x" ], [ "z" * 300 ] ]

      part_lists.each do |parts|
        uniform = draw.uniform_for(*parts)
        seed = draw.seed_for(*parts)

        expect(seed).to be_between(0, (2**64) - 1)
        expect(uniform).to eq((seed >> 11) / (2.0**53))
        expect(uniform).to be >= 0.0
        expect(uniform).to be < 1.0
        expect(uniform * (2**53)).to eq((uniform * (2**53)).floor)
      end
    end

    it "LO1 the NUL between parts keeps part lists apart" do
      expect(draw.uniform_for("ab", "c")).not_to eq(draw.uniform_for("a", "bc"))
      expect(draw.seed_for("ab", "c")).not_to eq(draw.seed_for("a", "bc"))
    end

    it "LO1 a part holding a NUL is refused, as an ArgumentError" do
      expect(draw.uniform_for("a", "b")).to be_a(Float)
      expect { draw.uniform_for("a\u0000b") }.to raise_error(ArgumentError)
      expect { draw.uniform_for("t", "x\u0000") }.to raise_error(ArgumentError)
      expect { draw.seed_for("t\u0000") }.to raise_error(ArgumentError)
      expect { draw.seed_for("t", "\u0000", "a") }.to raise_error(ArgumentError)
    end
  end

  describe "the draw by weight" do
    it "LO1 from_weights answers the first key whose running sum passes uniform * total" do
      weights = { "a" => 1, "b" => 3 }

      expect(draw.from_weights(weights, 0.0)).to eq("a")
      expect(draw.from_weights(weights, 0.2)).to eq("a")   # 0.8 < 1
      expect(draw.from_weights(weights, 0.3)).to eq("b")   # 1.2 > 1
      expect(draw.from_weights(weights, 0.999)).to eq("b")

      halves = { "a" => 0.5, "b" => 0.25, "c" => 0.25 }
      expect(draw.from_weights(halves, 0.49)).to eq("a")
      expect(draw.from_weights(halves, 0.51)).to eq("b")
      expect(draw.from_weights(halves, 0.76)).to eq("c")
    end

    it "LO1 a weight that is not a finite positive number counts 0: zero, nil, NaN, infinite or negative" do
      weights = { "zero" => 0.0, "none" => nil, "nan" => Float::NAN, "infinite" => Float::INFINITY,
                  "negative" => -2.0, "a" => 1.0, "b" => 1.0 }

      expect(draw.from_weights(weights, 0.0)).to eq("a")
      expect(draw.from_weights(weights, 0.49)).to eq("a")
      expect(draw.from_weights(weights, 0.51)).to eq("b")
      expect(draw.from_weights(weights, 1.0)).to eq("b")

      rng = Random.new(77)
      answers = Array.new(1_000) { draw.from_weights(weights, rng.rand) }
      expect(answers.uniq).to contain_exactly("a", "b")
    end

    it "LO1 weights that all count 0 count as equal" do
      weights = { "a" => 0, "b" => 0.0, "c" => nil }

      expect(draw.from_weights(weights, 0.2)).to eq("a")   # 0.6 of 3
      expect(draw.from_weights(weights, 0.5)).to eq("b")   # 1.5 of 3
      expect(draw.from_weights(weights, 0.8)).to eq("c")   # 2.4 of 3
      expect(draw.from_weights({ "a" => -1.0, "b" => Float::NAN }, 0.7)).to eq("b")
    end

    it "LO1 answers nil for an empty Hash, and refuses weights that are not a Hash or a uniform outside [0, 1]" do
      expect(draw.from_weights({}, 0.5)).to be_nil
      expect(draw.weighted({}, 0.5)).to eq([ nil, nil ])
      expect { draw.from_weights([ [ "a", 1 ] ], 0.5) }.to raise_error(ArgumentError, /weights must be a Hash/)
      [ -0.1, 1.1, Float::NAN, nil, "0.5" ].each do |uniform|
        expect { draw.from_weights({ "a" => 1, "b" => 3 }, uniform) }.to raise_error(ArgumentError), uniform.inspect
      end
      expect(draw.from_weights({ "a" => 1, "b" => 3 }, 0.0)).to eq("a")
      expect(draw.from_weights({ "a" => 1, "b" => 3 }, 1.0)).to eq("b")
    end

    it "LO1 a uniform of 1.0 answers the last positive key" do
      expect(draw.from_weights({ "a" => 1.0, "b" => 2.0, "c" => 0.0 }, 1.0)).to eq("b")
      expect(draw.from_weights({ "a" => 0.1, "b" => 0.2, "c" => 0.3 }, 1.0)).to eq("c")
      expect(draw.from_weights({ "a" => 0.0, "b" => nil }, 1.0)).to eq("b")
    end

    it "LO1 keys are answered as given, whatever hashable objects they are, and never read" do
      unread = Class.new do
        def to_s = raise("a key was read")
        def to_str = raise("a key was read")
        def name = raise("a key was read")
      end.new
      pair = [ "x", 1 ]
      weights = { unread => 1.0, sym: 1.0, 42 => 1.0, pair => 1.0 }

      expect(draw.from_weights(weights, 0.1)).to equal(unread)
      expect(draw.from_weights(weights, 0.3)).to eq(:sym)
      expect(draw.from_weights(weights, 0.6)).to eq(42)
      expect(draw.from_weights(weights, 0.9)).to equal(pair)
      expect(draw.weighted(weights, 0.1).first).to equal(unread)
    end

    it "LO1 from_weights over 10^5 uniforms draws each key within 4 standard errors of weight / total" do
      weights = { "a" => 1.0, "b" => 2.0, "c" => 3.0, "d" => 0.0, "e" => 4.0 }
      draws = 100_000
      rng = Random.new(20_261_101)
      counts = Hash.new(0)
      draws.times { counts[draw.from_weights(weights, rng.rand)] += 1 }

      expect(counts["d"]).to eq(0)
      weights.each do |key, weight|
        share = weight / 10.0
        standard_error = Math.sqrt(draws * share * (1.0 - share))
        expect(counts[key]).to be_within(4.0 * standard_error).of(draws * share)
      end
    end

    it "LO1 weighted answers the key and its share: the counted weight over the counted total" do
      expect(draw.weighted({ "a" => 1, "b" => 3 }, 0.5)).to eq([ "b", 0.75 ])

      mixed = { "a" => nil, "b" => 2.0, "c" => -5, "d" => 6.0 }
      expect(draw.weighted(mixed, 0.1)).to eq([ "b", 0.25 ])
      expect(draw.weighted(mixed, 0.9)).to eq([ "d", 0.75 ])

      expect(draw.weighted({ "a" => nil, "b" => -1.0 }, 0.2)).to eq([ "a", 0.5 ])
      expect(draw.weighted({ "a" => nil, "b" => -1.0 }, 0.7)).to eq([ "b", 0.5 ])

      key, share = draw.weighted({ "a" => 0, "b" => 0, "c" => 0 }, 0.5)
      expect(key).to eq("b")
      expect(share).to be_within(1e-15).of(1.0 / 3.0)

      key, share = draw.weighted({ "a" => 0.333333, "b" => 0.333333, "c" => 0.333334 }, 0.9)
      expect(key).to eq("c")
      expect(share).to be_within(1e-12).of(0.333334)
    end

    it "LO1 from_weights is weighted's key, and the share sums to 1 over the keys (one rule)" do
      weights = { "a" => 0.2, "b" => nil, "c" => 1.3, "d" => -0.4, "e" => 0.5, "f" => Float::NAN, "g" => 2.0 }
      counted = { "a" => 0.2, "c" => 1.3, "e" => 0.5, "g" => 2.0 }
      total = 4.0
      rng = Random.new(31)
      seen = {}

      2_000.times do
        uniform = rng.rand
        key, share = draw.weighted(weights, uniform)

        expect(draw.from_weights(weights, uniform)).to eq(key)
        expect(share).to be_within(1e-15).of(counted.fetch(key) / total)
        seen[key] = share
      end
      expect(seen.keys).to contain_exactly("a", "c", "e", "g")
      expect(seen.values.sum).to be_within(1e-12).of(1.0)
    end
  end

  describe "the L3 barrier, round 3: the arithmetic of 0.1's Softmax policy" do
    # 0.1's Softmax draw written out: the total by Array#sum, the target uniform * total, a running sum in input order
    # and the first index at or above the target, else the last index.
    def reference(weights, uniform)
      total = weights.sum
      target = uniform * total
      cumulative = 0.0
      weights.each_with_index do |weight, index|
        cumulative += weight
        return index if cumulative >= target
      end
      weights.size - 1
    end

    def keyed(weights) = weights.each_index.to_h { |index| [ index, weights[index] ] }

    # The uniforms at each running sum over the total, and the Floats beside them.
    def edges(weights)
      total = weights.sum
      running = 0.0
      weights.flat_map do |weight|
        running += weight
        edge = running / total
        [ edge, edge.prev_float, edge.next_float ]
      end.select { |uniform| uniform >= 0.0 && uniform < 1.0 }
    end

    it "the L3 barrier, round 3: answers the key whose running sum is at or above the target" do
      expect(draw.from_weights({ "a" => 1, "b" => 3 }, 0.25)).to eq("a")
      expect(draw.from_weights({ "a" => 0.5, "b" => 0.25, "c" => 0.25 }, 0.5)).to eq("a")
      expect(draw.from_weights({ "a" => 0.5, "b" => 0.25, "c" => 0.25 }, 0.75)).to eq("b")
      expect(draw.weighted({ "a" => 1, "b" => 3 }, 0.25)).to eq([ "a", 0.25 ])
    end

    it "the L3 barrier, round 3: totals the weights by Array#sum, so a long run of small weights moves the target" do
      weights = [ 1.0 ] + Array.new(1_000) { |index| 1.0e-16 * (1.0 + (index * 1.0e-6)) }
      plain = weights.inject(0.0) { |sum, weight| sum + weight }
      expect(weights.sum).to be > plain

      [ 1.0 - 1.0e-14, 0.99999999999995, 1.0.prev_float ].each do |uniform|
        expect(uniform * weights.sum).to be > plain
        expect(draw.from_weights(keyed(weights), uniform)).to eq(1_000)
        expect(reference(weights, uniform)).to eq(1_000)
      end
      expect(draw.weighted(keyed(weights), 0.5)).to eq([ 0, 1.0 / weights.sum ])
    end

    it "the L3 barrier, round 3: draws as the arithmetic written out on random weights, at and beside every edge" do
      rng = Random.new(20_261_103)
      checked = 0
      2_000.times do
        weights = Array.new(rng.rand(1..9)) { Math.exp(-rng.rand * rng.rand(1..60)) }
        weights[rng.rand(1...weights.size)] = 0.0 if weights.size > 2 && rng.rand(3).zero? && rng.rand(2).zero?
        weights[-1] = weights[-1].nonzero? || 0.5
        (edges(weights) + Array.new(4) { rng.rand }).each do |uniform|
          expect(draw.from_weights(keyed(weights), uniform)).to eq(reference(weights, uniform)), "#{weights} at #{uniform}"
          checked += 1
        end
      end
      expect(checked).to be > 20_000
    end
  end
end
