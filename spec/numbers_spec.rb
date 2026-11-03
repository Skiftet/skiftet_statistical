# frozen_string_literal: true

# The library's one number reader, SkiftetStatistical::Numbers: a number is a real Numeric whose to_f answers a finite
# Float of magnitude at most Numbers::LARGEST (10**100), the input domain. Everything else is refused alike, as an
# ArgumentError, a String included (a numeric one too: "0.5" and "0x14" are not numbers here), and so is a Numeric
# that answers no to_f, or whose to_f answers no finite Float or raises. The code new in 0.4.0 reads its numbers
# through it (the L3 barrier's sixth round: `read` refuses, and its block is a reader's own rule for what is not a
# number); 0.3.0's modules keep their own copies as 0.3.0 wrote them (the L3 barrier's fourth round), which refuse
# these inputs with the same words, so each refuses alike; the priors and the decay, which once read numbers with
# Kernel#Float, refuse Strings like the rest.
RSpec.describe SkiftetStatistical::Numbers do
  def prior = SkiftetStatistical::Bandit::Prior
  def decay = SkiftetStatistical::Bandit::Decay

  # Inputs that are not numbers by the rule.
  def refused
    [ "0.5", "0x14", "1_0", "", " 2 ", nil, :one, true, [ 1.0 ], { value: 1.0 }, Object.new,
      Float::NAN, Float::INFINITY, -Float::INFINITY, Complex(0.5, 1), Complex(0.5, 0),
      10**400, -(10**400), Rational(10**400, 3) ]
  end

  # Numerics the rule cannot read: one with no to_f, one whose to_f answers a String, nil or an infinity, and one
  # whose to_f raises.
  def odd_numerics
    [ Class.new(Numeric).new, Class.new(Numeric) { define_method(:to_f) { "x" } }.new,
      Class.new(Numeric) { define_method(:to_f) { nil } }.new,
      Class.new(Numeric) { define_method(:to_f) { Float::INFINITY } }.new,
      Class.new(Numeric) { define_method(:to_f) { raise "no" } }.new ]
  end

  describe ".read" do
    it "answers a number in the input domain as its Float" do
      expect(described_class.read(3, "a count")).to eql(3.0)
      expect(described_class.read(0, "a count")).to eql(0.0)
      expect(described_class.read(-2.5, "a count")).to eql(-2.5)
      expect(described_class.read(Rational(1, 4), "a count")).to eql(0.25)
      expect(described_class.read(1e100, "a count")).to eql(1e100)
      expect(described_class.read(-(10**100), "a count")).to eql(-1e100)
      expect(described_class.read(Float::MIN / 4, "a count")).to eql(Float::MIN / 4)
      expect(described_class::LARGEST).to eql(1e100)
    end

    it "refuses anything else as an ArgumentError naming the label and the value" do
      refused.each do |value|
        expect { described_class.read(value, "a weight") }
          .to raise_error(ArgumentError, "a weight must be a finite number, got #{value.inspect}"), value.inspect
      end
      odd_numerics.each do |value|
        expect { described_class.read(value, "a weight") }
          .to raise_error(ArgumentError, /\Aa weight must be a finite number, got /), "read #{value.class}"
      end
    end

    it "refuses a number beyond LARGEST either side of 0, with the input domain's words" do
      words = /\Aa weight must lie within the input domain, at most 1\.0e\+100 either side of 0/
      [ 1e100.next_float, -1e100.prev_float, 1e200, -1e308, Float::MAX, 10**101, Rational(10**101, 1) ].each do |value|
        expect { described_class.read(value, "a weight") }.to raise_error(ArgumentError, words), value.inspect
      end
    end

    it "answers its block's answer for what is not a number, and never waives the domain" do
      (refused + odd_numerics).each do |value|
        expect(described_class.read(value, "a weight") { |given| [ :fallback, given ] }).to eq([ :fallback, value ])
      end
      expect(described_class.read(0.5, "a weight") { :fallback }).to eql(0.5)
      expect { described_class.read(1e200, "a weight") { :fallback } }.to raise_error(ArgumentError, /input domain/)
    end
  end

  describe ".pair" do
    it "reads two numbers of a pair, each by .read, then by the reader's own bound" do
      expect(described_class.pair([ 2, Rational(1, 2) ], "a prior", %w[alpha beta])).to eql([ 2.0, 0.5 ])
      bounded = described_class.pair([ 1, 2 ], "a prior", %w[alpha beta]) { |*given| given }
      expect(bounded).to eq([ [ 1.0, "a prior's alpha", 0 ], [ 2.0, "a prior's beta", 1 ] ])
    end

    it "refuses a value that is not a pair of two numbers in the domain, naming the pair" do
      [ nil, 1.0, [ 1.0 ], [ 1.0, 2.0, 3.0 ], { a: 1 }, "12" ].each do |given|
        expect { described_class.pair(given, "a prior", %w[alpha beta]) }
          .to raise_error(ArgumentError, "a prior must be a pair [alpha, beta] (got #{given.inspect})")
      end
      expect { described_class.pair([ 1.0, "2" ], "a prior", %w[alpha beta]) }
        .to raise_error(ArgumentError, "a prior's beta must be a finite number, got \"2\"")
      expect { described_class.pair([ 1e101, 1.0 ], "a prior", %w[alpha beta]) }
        .to raise_error(ArgumentError, /\Aa prior's alpha must lie within the input domain/)
    end
  end

  describe "the one check, read by every module" do
    it "is the words of the refusal everywhere: BetaArgmax, NormalArgmax, BetaFunction and HierarchicalRates alike" do
      message = /must be a finite number, got "2"/
      expect { SkiftetStatistical::BetaArgmax.position(0.5, "2", 2.0) }.to raise_error(ArgumentError, message)
      expect { SkiftetStatistical::BetaArgmax.distribution("2", 2.0, 2.0) }.to raise_error(ArgumentError, message)
      expect { SkiftetStatistical::Bandit::NormalArgmax.probabilities([ [ "2", 1.0 ], [ 0.0, 1.0 ] ]) }
        .to raise_error(ArgumentError, message)
      expect { SkiftetStatistical::BetaFunction.log_density("2", 2.0, 2.0) }.to raise_error(ArgumentError, message)
      expect do
        SkiftetStatistical::HierarchicalRates.leave_one_out_share(parent_count: "2", parent_total: 10.0, node_count: 1.0,
                                                                  node_total: 2.0, prior_share: 0.1, prior_strength: 5.0)
      end.to raise_error(ArgumentError, message)
    end

    it "makes the priors refuse a String, a numeric one included, and a Complex, as an ArgumentError" do
      expect(prior.given(mean: 0.5, strength: 20)).to eq([ 10.0, 10.0 ])
      expect(prior.given(mean: Rational(1, 2), strength: 20)).to eq([ 10.0, 10.0 ])
      [
        -> { prior.given(mean: "0.5", strength: 20) },
        -> { prior.given(mean: 0.5, strength: "0x14") },
        -> { prior.given(mean: Complex(0.5, 1), strength: 20) },
        -> { prior.borrowed(group: { views: "10", outcomes: 2 }) },
        -> { prior.borrowed(group: { views: 10, outcomes: 2 }, own: { views: 1, outcomes: "1" }) },
        -> { prior.pooled(group: { views: 10, outcomes: 2 }, strength: "20") },
        -> { prior.pooled(group: { views: 10, outcomes: 2 }, strength: 20.0, base: [ "1", 1.0 ]) },
        -> { prior.pooled_new(others: [ { group: { views: "10", outcomes: 2 } } ]) },
        -> { prior.pooled_new(others: [], weight: "2") },
        -> { prior.pooled_mean(views: 10, outcomes: "2", prior_mean: 0.1, strength: 20.0) },
        -> { prior.pooled_mean(views: 10, outcomes: 2, prior_mean: "0.1", strength: 20.0) },
        -> { prior.tilts({ "a" => "2", "b" => 1.0 }, clip: [ 0.5, 2.0 ]) },
        -> { prior.tilts({ "a" => 2.0 }, clip: [ "0.5", 2.0 ]) },
        -> { prior.chain(levels: [ { views: 10, outcomes: 1, strength: "20" } ], root_mean: 0.1) },
        -> { prior.chain(levels: [ { views: 10, outcomes: 1, strength: 20.0 } ], root_mean: "0.1") },
        -> { prior.chain(levels: [ { views: 10, outcomes: 1, strength: 20.0 } ], root_mean: 0.1, tilt: "1") },
        -> { prior.amount(mean: "4", variance: 2.0, strength: 20.0) },
        -> { prior.amount(mean: 4.0, variance: "2", strength: 20.0) },
        -> { prior.amount(mean: 4.0, variance: 2.0, strength: Complex(20, 0)) },
        -> { prior.pooled_amount(group: { views: 10, outcomes: "40", outcome_squares: 400 }, strength: 20.0) }
      ].each_with_index do |call, i|
        expect(&call).to raise_error(ArgumentError), "prior call #{i}"
      end
    end

    it "makes the decay refuse a String, a numeric one included, as an ArgumentError" do
      expect(decay.weight(72, half_life: 72, cap: 336)).to eq(0.5)
      [
        -> { decay.weight("72", half_life: 72) },
        -> { decay.weight(72, half_life: "72") },
        -> { decay.weight(72, half_life: 72, cap: "336") },
        -> { decay.fold([ { age: "0", views: 1, outcomes: 1 } ], half_life: 72) },
        -> { decay.fold([ { age: 0, views: "1", outcomes: 1 } ], half_life: 72) },
        -> { decay.fold([ { age: 0, views: 1, outcomes: "1" } ], half_life: 72) },
        -> { decay.fold([ { age: 0, views: 1, outcomes: 1, outcome_squares: "1" } ], half_life: 72) }
      ].each_with_index do |call, i|
        expect(&call).to raise_error(ArgumentError), "decay call #{i}"
      end
    end

    it "makes the decay refuse a NaN, an infinity and negative views or outcome squares, in a row past the cap too" do
      expect(decay.fold([ { age: 0, views: 2, outcomes: -3.5 } ], half_life: 72))
        .to eq(views: 2.0, outcomes: -3.5, outcome_squares: 0.0)
      [ { views: Float::NAN }, { views: Float::INFINITY }, { views: -5 }, { outcomes: Float::NAN },
        { outcomes: -Float::INFINITY }, { outcome_squares: Float::INFINITY }, { outcome_squares: -1.0 } ].each do |bad|
        [ 0, 400 ].each do |age|
          row = { age: age, views: 1, outcomes: 1 }.merge(bad)
          expect { decay.fold([ { age: 0, views: 1, outcomes: 1 }, row ], half_life: 72, cap: 336) }
            .to raise_error(ArgumentError), "#{bad.inspect} at age #{age}"
        end
      end
    end

    it "makes the decay refuse an age, half-life or cap that is not finite (nil is no cap)" do
      expect(decay.weight(1_000, half_life: 72, cap: nil)).to be > 0.0
      [
        -> { decay.weight(Float::NAN, half_life: 72) },
        -> { decay.weight(Float::INFINITY, half_life: 72) },
        -> { decay.weight(1, half_life: Float::NAN) },
        -> { decay.weight(1, half_life: Float::INFINITY) },
        -> { decay.weight(1, half_life: 72, cap: Float::NAN) },
        -> { decay.weight(1, half_life: 72, cap: Float::INFINITY) }
      ].each_with_index do |call, i|
        expect(&call).to raise_error(ArgumentError), "decay call #{i}"
      end
    end
  end
end
