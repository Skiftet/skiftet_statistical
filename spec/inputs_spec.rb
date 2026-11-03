# frozen_string_literal: true

require "bigdecimal"
require "set"

# What the library does with what a caller hands in (K3 as restated at the L3 barrier's fifth round): 0.3.0's files
# stay as 0.3.0 wrote them, the two old policies (Policies::ThompsonSampling and Policies::Softmax) included, so every
# 0.3.0 public method answers as 0.3.0 for every input, invalid input included: the same value or the same error, and
# the same rng position after it. New callers draw through the posterior set (Bandit::Posteriors). The code new in
# 0.4.0 raises ArgumentError on invalid input, before it draws anything (an rng's answer at the draw that read it).
RSpec.describe "the library's reading of what a caller hands in" do
  def policies = SkiftetStatistical::Policies
  def arm(name, **counts) = SkiftetStatistical::Arm.new(name, **counts)

  # An rng that answers its scripted uniforms in turn.
  def scripted(*uniforms)
    Class.new do
      define_method(:initialize) { @at = -1 }
      define_method(:rand) { uniforms[@at += 1] }
    end.new
  end

  # An rng whose rand answers 0.5 and whose rand(n) answers `index`.
  def indexing(index)
    Class.new do
      define_method(:rand) { |size = nil| size.nil? ? 0.5 : index }
    end.new
  end

  # A Random that counts the values it hands out.
  def counting(seed)
    Class.new(Random) do
      def calls = (@calls ||= 0)

      def rand(*args)
        @calls = calls + 1
        super
      end
    end.new(seed)
  end

  # An rng that answers rand only through method_missing, so respond_to?(:rand) is false; it forwards to Random.new(1).
  def forwarding
    Class.new do
      define_method(:initialize) { @inner = Random.new(1) }
      define_method(:method_missing) { |name, *args| name == :rand ? @inner.rand(*args) : super(name, *args) }
      define_method(:respond_to_missing?) { |*| false }
    end.new
  end

  def no_inspect
    name = Object.new
    name.singleton_class.send(:undef_method, :inspect)
    name
  end

  def overflowed
    arm("big", pulls: 1, reward_sum: 1.0e308).update(1.0e308)
  end

  # 0.1's Softmax pick written out: weights exp((mean - max) / t), their Array#sum, the first arm whose running sum is
  # at or above uniform * total, else the last arm.
  def softmax_reference(means, temperature, uniform)
    max = means.max
    weights = means.map { |mean| Math.exp((mean - max) / temperature) }
    target = uniform * weights.sum
    cumulative = 0.0
    weights.each_with_index do |weight, index|
      cumulative += weight
      return index if cumulative >= target
    end
    weights.size - 1
  end

  def softmax_pick(arms, temperature, uniform)
    chosen = policies::Softmax.new(temperature: temperature, rng: scripted(uniform)).choose(arms)
    arms.index { |candidate| candidate.equal?(chosen) }
  end

  describe "Softmax, as 0.3.0 wrote it" do
    it "the L3 barrier, round 3: picks as 0.1's arithmetic on uniforms between two running-sum edges" do
      arms = [ 0.57, 0.17, 0.78 ].each_with_index.map { |mean, index| arm("a#{index}", pulls: 100, reward_sum: mean * 100) }
      means = arms.map(&:mean)
      expect(softmax_pick(arms, 0.1, 0.10887926105359592)).to eq(softmax_reference(means, 0.1, 0.10887926105359592))
      expect(softmax_pick(arms, 0.1, 0.1108734542815258)).to eq(softmax_reference(means, 0.1, 0.1108734542815258))
    end

    it "the L3 barrier, round 3: picks as 0.1's arithmetic on random arms, at and beside every edge" do
      rng = Random.new(20_261_104)
      checked = 0
      600.times do
        arms = Array.new(rng.rand(1..9)) do |index|
          pulls = rng.rand(1..500)
          arm("a#{index}", pulls: pulls, reward_sum: rng.rand * pulls)
        end
        temperature = Math.exp((rng.rand * 8) - 6)
        means = arms.map(&:mean)
        weights = means.map { |mean| Math.exp((mean - means.max) / temperature) }
        next if weights.first.zero? || weights.last.zero?

        total = weights.sum
        running = 0.0
        uniforms = weights.flat_map { |weight| [ (running += weight) / total ] }.flat_map { |edge| [ edge, edge.prev_float, edge.next_float ] }
        (uniforms.select { |uniform| uniform >= 0.0 && uniform < 1.0 } + [ 0.0, rng.rand ]).each do |uniform|
          expect(softmax_pick(arms, temperature, uniform)).to eq(softmax_reference(means, temperature, uniform))
          checked += 1
        end
      end
      expect(checked).to be > 5_000
    end

    it "the L3 barrier, round 5: picks the first arm at a uniform of 0.0, and the last arm when the running sum ends " \
       "below the target, whatever their weights, as 0.1's arithmetic does" do
      arms = [ arm("low", pulls: 1, reward_sum: 0.0), arm("high", pulls: 1, reward_sum: 1.0) ]
      expect(softmax_reference(arms.map(&:mean), 1.0e-3, 0.0)).to eq(0)
      expect(softmax_pick(arms, 1.0e-3, 0.0)).to eq(0)

      many = [ arm("lead", pulls: 1, reward_sum: 0.0) ] +
             Array.new(1_000) { |index| arm("m#{index}", pulls: 1, reward_sum: -36.8 - (index * 1.0e-6)) } +
             [ arm("last", pulls: 1, reward_sum: -1.0e6) ]
      uniform = 1.0 - 1.0e-14
      expect(softmax_reference(many.map(&:mean), 1.0, uniform)).to eq(1_001)
      expect(softmax_pick(many, 1.0, uniform)).to eq(1_001)
    end
  end

  describe "the two old policies, as 0.3.0 wrote them, on any input" do
    def plain_arm(name, successes, failures)
      Class.new(BasicObject) do
        define_method(:name) { name }
        define_method(:successes) { successes }
        define_method(:failures) { failures }
        define_method(:mean) { successes / (successes + failures) }
      end.new
    end

    # An rng on a fixed cycle of uniforms whose rand(n) answers n - 1, counting its calls.
    def cycling(*uniforms)
      Class.new do
        attr_reader :calls

        define_method(:initialize) { @calls = 0 }
        define_method(:rand) do |size = nil|
          @calls += 1
          size.nil? ? uniforms[(@calls - 1) % uniforms.size] : size - 1
        end
      end.new
    end

    it "the L3 barrier, round 5: draw through an rng that answers rand only through method_missing, as through Random" do
      arms = [ arm("a", pulls: 4, reward_sum: 2.0), arm("b", pulls: 4, reward_sum: 1.0) ]
      [ policies::ThompsonSampling, policies::Softmax ].each do |policy|
        expect(policy.new(rng: forwarding).choose(arms)).to equal(policy.new(rng: Random.new(1)).choose(arms)), policy.name
      end
      expect(policies::ThompsonSampling.new(rng: BasicObject.new)).to be_a(policies::ThompsonSampling)
      expect { policies::ThompsonSampling.new(rng: BasicObject.new).choose(arms) }.to raise_error(NoMethodError)
      expect { policies::Softmax.new(rng: nil).choose(arms) }.to raise_error(NoMethodError)
    end

    it "the L3 barrier, round 5: read an rng's answers as they come, outside [0, 1) included" do
      arms = [ arm("a", pulls: 4, reward_sum: 2.0), arm("b", pulls: 4, reward_sum: 1.0) ]
      { 1.5 => 1, -0.25 => 0, Float::NAN => 1, 0 => 0 }.each do |answer, index|
        expect(policies::Softmax.new(rng: scripted(answer)).choose(arms)).to equal(arms[index]), answer.inspect
      end
      expect { policies::Softmax.new(rng: scripted(nil)).choose(arms) }.to raise_error(NoMethodError)
      expect { policies::ThompsonSampling.new(rng: scripted(*Array.new(12, 1.5))).choose(arms) }.to raise_error(Math::DomainError)
    end

    it "the L3 barrier, round 5: Thompson Sampling takes a tie's rand(n) answer as it comes" do
      # Priors of 1e-300 make every draw 0.5 (both Gamma draws underflow), so the three fresh arms tie.
      fresh = [ arm("a"), arm("b"), arm("c") ]
      tied = ->(index) { policies::ThompsonSampling.new(prior_alpha: 1e-300, prior_beta: 1e-300, rng: indexing(index)).choose(fresh) }
      expect(tied.call(3)).to be_nil
      expect(tied.call(-1)).to equal(fresh[2])
      expect(tied.call(0.5)).to equal(fresh[0])
      expect { tied.call(nil) }.to raise_error(TypeError)
    end

    it "the L3 barrier, round 5: meet an arm whose mean overflowed as 0.3.0 did" do
      pair = [ arm("a", pulls: 2, reward_sum: 1.0), overflowed ]
      expect(policies::Softmax.new(rng: Random.new(5)).choose(pair)).to equal(pair[1])
      expect { policies::ThompsonSampling.new(rng: Random.new(5)).choose(pair) }.to raise_error(ArgumentError)
    end

    it "the L3 barrier, round 5: read their settings as 0.3.0 did" do
      expect(policies::Softmax.new(temperature: Float::INFINITY).temperature).to eq(Float::INFINITY)
      expect { policies::Softmax.new(temperature: "0.5") }.to raise_error(NoMethodError)
      expect { policies::Softmax.new(temperature: BasicObject.new) }.to raise_error(NoMethodError)
      expect(policies::ThompsonSampling.new(prior_alpha: Float::INFINITY).prior_alpha).to eq(Float::INFINITY)
      expect(policies::ThompsonSampling.new(prior_beta: "1e400").prior_beta).to eq(Float::INFINITY)
      expect { policies::ThompsonSampling.new(prior_alpha: nil) }.to raise_error(TypeError)
    end

    it "the L3 barrier, round 5: answer the fourth round's refute cases as 0.3.0 did" do
      three = [ arm("a", pulls: 10, reward_sum: 5.0), arm("b", pulls: 10, reward_sum: 5.0), arm("c", pulls: 3, reward_sum: 1.0) ]
      [ Rational(1, 10**400), BigDecimal("1e-400") ].each do |temperature|
        rng = Random.new(0)
        expect(policies::Softmax.new(temperature: temperature, rng: rng).choose(three)).to equal(three[2])
        expect(rng.rand).to eq(0.7151893663724195)
      end

      counts = Struct.new(:name, :successes, :failures)
      { [ 0.5, 0.1, 0.01 ] => "a", [ 0.3, 0.7, 0.2 ] => "b", [ 0.9, 0.25, 0.05 ] => "a" }.each do |cycle, name|
        rng = cycling(*cycle)
        chosen = policies::ThompsonSampling.new(rng: rng).choose([ counts.new("a", BigDecimal("1"), BigDecimal("2")), counts.new("b", 1.0, 2.0) ])
        expect([ chosen.name, rng.calls ]).to eq([ name, 12 ]), cycle.inspect
      end
      rng = Random.new(0)
      tiny = [ counts.new("a", BigDecimal("1e-400"), 3.0), counts.new("b", 2.0, 3.0) ]
      expect([ policies::ThompsonSampling.new(prior_alpha: 0.0, rng: rng).choose(tiny).name, rng.rand ]).to eq([ "b", 0.925596638292661 ])

      plain = [ plain_arm("a", 3.0, 7.0), plain_arm("b", 5.0, 5.0), plain_arm("c", 1.0, 1.0) ]
      { policies::ThompsonSampling => [ "c", 0.14038693859523377 ], policies::Softmax => [ "b", 0.7203244934421581 ] }.each do |policy, answer|
        rng = Random.new(1)
        expect([ policy.new(rng: rng).choose(plain).name, rng.rand ]).to eq(answer), policy.name
      end
      { policies::ThompsonSampling => 0.5680445610939323, policies::Softmax => 0.7151893663724195 }.each do |policy, uniform|
        rng = Random.new(0)
        set = Set[arm("a", pulls: 10, reward_sum: 5.0), arm("b", pulls: 10, reward_sum: 2.0)]
        expect([ policy.new(rng: rng).choose(set).name, rng.rand ]).to eq([ "a", uniform ]), policy.name
      end

      rng = Random.new(0)
      expect { policies::ThompsonSampling.new(rng: rng).choose([ arm("a", pulls: 4, reward_sum: 1.0), Object.new ]) }
        .to raise_error(NoMethodError)
      expect(rng.rand).to eq(0.4375872112626925)
    end
  end

  describe "every other 0.3.0 method, invalid input included, as 0.3.0 wrote it" do
    it "the L3 barrier, round 4: draws through an rng that answers rand only through method_missing" do
      arms = [ arm("a", pulls: 4, reward_sum: 2.0), arm("b", pulls: 4, reward_sum: 1.0) ]
      expect(SkiftetStatistical::Sampler.new(forwarding).beta(2.0, 3.0))
        .to eql(SkiftetStatistical::Sampler.new(Random.new(1)).beta(2.0, 3.0))
      expect(policies::EpsilonGreedy.new(rng: forwarding).choose(arms))
        .to equal(policies::EpsilonGreedy.new(rng: Random.new(1)).choose(arms))
      expect(policies::UCB1.new(rng: forwarding).choose(arms)).to equal(policies::UCB1.new(rng: Random.new(1)).choose(arms))
    end

    it "the L3 barrier, round 4: reads an rng's answers as they come, outside [0, 1) included" do
      expect(SkiftetStatistical::Sampler.new(scripted(0.5, 1.5)).gaussian)
        .to eql(Math.sqrt(-2.0 * Math.log(0.5)) * Math.cos(2.0 * Math::PI * 1.5))
      expect { SkiftetStatistical::Sampler.new(scripted(nil, nil)).gaussian }.to raise_error(NoMethodError)
    end

    it "the L3 barrier, round 4: lets Epsilon-Greedy and UCB1 pick an arm whose mean overflowed" do
      fair = arm("a", pulls: 2, reward_sum: 1.0)
      expect(policies::EpsilonGreedy.new(epsilon: 0.0, rng: Random.new(5)).choose([ fair, overflowed ]).name).to eq("big")
      expect(policies::UCB1.new(rng: Random.new(5)).choose([ fair, overflowed ]).name).to eq("big")
    end

    it "the L3 barrier, round 4: keeps 0.1's readings and their limits as they were" do
      expect { arm("a", pulls: nil) }.to raise_error(TypeError)
      expect(arm("a").update(Float::NAN).reward_sum).to be_nan
      expect(arm("a", reward_sum: "1e400").reward_sum).to eq(Float::INFINITY)
      expect(SkiftetStatistical::Sampler.new(Random.new(1)).beta(2.0, Float::INFINITY)).to eql(0.0)
      expect(SkiftetStatistical::JointBeta.each_draw([ [ 2.0, Float::INFINITY ] ], draws: 0, rng: Random.new(1)).to_a).to eq([])
    end

    it "the L3 barrier, round 4: meets a name that answers no hash or inspect as 0.3.0 did" do
      expect { arm(BasicObject.new) }.to raise_error(NoMethodError)
      expect(arm(no_inspect)).to be_a(SkiftetStatistical::Arm)
      [ BasicObject.new, no_inspect ].each do |name|
        expect { SkiftetStatistical::Bandit.new(arms: [ "a" ]).record(name, 1) }.to raise_error(NoMethodError)
        expect do
          SkiftetStatistical::Bandit::Allocator.call(arms: [ :a, name ], tallies: {}, status: "running", winner_arm: nil,
                                                     rng: Random.new(1), draws: 10)
        end.to raise_error(NoMethodError)
      end
    end

    it "the L3 barrier, round 4: answers the third round's refute cases as 0.3.0 did" do
      argmax = SkiftetStatistical::BetaArgmax
      # A uniform at an end of the 2**-53 grid, beside a member near 1: pick compares 0.3.0's own position.
      expect(argmax.pick([ [ 100.0, 0.5 ], [ 2.0, 2.0 ] ], explore: 0.5, priorities: [ 0.1, 0.2 ],
                                                            uniforms: [ 2.0**-53, 0.7944 ], epsilon: 0.0)).to eq(0)
      expect(argmax.pick([ [ 0.5, 100.0 ], [ 2.0, 2.0 ] ], explore: 0.5, priorities: [ 0.1, 0.2 ],
                                                            uniforms: [ 1.0 - 2.0**-53, 0.7944 ], epsilon: 0.0)).to eq(1)

      # Previous weights keyed by a name without to_sym, whose share 0.2.0's reading ignores.
      allocator = SkiftetStatistical::Bandit::Allocator
      tallies = { a: { exposures: 400, conversions: 40 }, b: { exposures: 400, conversions: 60 } }
      answer = allocator.call(arms: %i[a b], tallies: tallies, status: "running", winner_arm: nil, rng: Random.new(7),
                              draws: 200, previous_weights: { 5 => "x", a: 0.5, b: 0.5 })
      expect(answer.values_at(:weights, :phase)).to eq([ { a: 0.05, b: 0.95 }, :won ])
      entry = ->(mean, p_best, loss) { { mean: mean, lower: mean - 0.02, upper: mean + 0.02, p_best: p_best, expected_loss: loss } }
      summarized = allocator.from_summary(arms: %i[a b], summary: { a: entry.call(0.1, 0.4, 0.01), b: entry.call(0.11, 0.6, 0.005) },
                                          exposures: { a: 500, b: 500 }, status: "running", winner_arm: nil,
                                          previous_weights: { 5 => "x", a: 0.5, b: 0.5 })
      expect(summarized[:weights]).to eq(a: 0.4, b: 0.6)
      expect { allocator.call(arms: [ 1, 2 ], tallies: {}, status: "running", winner_arm: nil, rng: Random.new(7), draws: 20) }
        .to raise_error(NoMethodError)
      expect { allocator.call(arms: %i[a b], tallies: {}, status: "won", winner_arm: 5, rng: Random.new(7), draws: 20) }
        .to raise_error(NoMethodError)

      # Count Hashes without :conversions.
      weighted = SkiftetStatistical::Bandit::WeightedPosterior
      expect(weighted.effective_exposures({ exposures: 10 })).to eq(10)
      expect(weighted.beta({ exposures: 10, weighted_exposures: 5.0, weighted_conversions: 1.0 })).to eq([ 3.0, 9.0 ])
      expect(weighted.effective_exposures({ exposures: 10, weighted_exposures: 4.0, weighted_exposures_squared: 8.0 })).to eql(2.0)

      # A null-object name, and a decorator that answers is_a?(Arm).
      none = Class.new { define_method(:nil?) { true } }.new
      expect { arm(none) }.to raise_error(ArgumentError, "arm name cannot be nil")
      inner = arm("x")
      decorated = Class.new do
        define_method(:name) { inner.name }
        define_method(:is_a?) { |kind| kind == SkiftetStatistical::Arm || super(kind) }
      end.new
      bandit = SkiftetStatistical::Bandit.new(arms: [])
      expect(bandit.add_arm(decorated)).to equal(decorated)
      expect(bandit.arm_names).to eq([ "x" ])
    end
  end

  describe "the kept 0.1 readings" do
    it "the L3 barrier, round 3: still read what Kernel#Integer and Kernel#Float turn into a finite number" do
      expect(arm("a", pulls: "3", reward_sum: "1.5").to_h).to eq(name: "a", pulls: 3, reward_sum: 1.5, reward_square_sum: 0.0)
      expect(arm("a").update("0.5").reward_sum).to eq(0.5)
      expect(policies::ThompsonSampling.new(prior_alpha: "2").prior_alpha).to eq(2.0)
    end
  end

  describe "code new in 0.4.0" do
    def set = SkiftetStatistical::Bandit::Posteriors.beta({ "a" => [ 2.0, 3.0 ], "b" => [ 4.0, 1.0 ] })

    it "the L3 barrier, round 4: the pick odds refuse a draws: or seed: that is not an Integer within a Float's range" do
      [ Float::NAN, -5, 0, 2.5, "100", nil, BasicObject.new, 10**400 ].each do |draws|
        expect { set.pick_probabilities(draws: draws) }.to raise_error(ArgumentError, /draws must be an Integer/)
      end
      [ "x", nil, 2.5, 10**400, BasicObject.new ].each do |seed|
        expect { set.pick_probabilities(seed: seed) }.to raise_error(ArgumentError, /seed must be an Integer/)
      end
      expect(set.pick_probabilities(draws: 50_000, seed: -3)).to eq(set.pick_probabilities)
    end

    it "the L3 barrier, round 4: a pass's seed and a rank's top are read alike, an Integer within a Float's range" do
      hold = Class.new { define_method(:satisfied_for?) { |_leader| true } }.new
      expect { SkiftetStatistical::Bandit::Pass.call(status: "running", hold: hold, seed: 10**400) { { leader: nil } } }
        .to raise_error(ArgumentError, /a seed must be an Integer within a Float's range/)
      expect { set.rank(rng: Random.new(1), top: 10**400) }
        .to raise_error(ArgumentError, /top must be an Integer of at least 1 within a Float's range/)
      expect(SkiftetStatistical::Bandit::Pass.call(status: "running", hold: hold, seed: -(2**62)) { { leader: nil } })
        .to eq(leader: nil, ready: false)
    end

    it "the L3 barrier, round 4: a set refuses a name that is no name, and a credible mass outside (0, 1), by one wording each" do
      expect { set.rank(rng: Random.new(1), may_lead: [ BasicObject.new ]) }.to raise_error(ArgumentError, /is not an item/)
      expect { set.summary(among: [ no_inspect ]) }.to raise_error(ArgumentError, /is not an item/)
      [ 0.0, 1.0, Float::NAN, "0.5", nil, BasicObject.new ].each do |mass|
        expect { set.summary(credible_mass: mass) }.to raise_error(ArgumentError, /credible_mass must be a number above 0 and below 1/)
      end
    end

    it "the L3 barrier, round 4: the set's RNG draws check every answer as it is drawn, through 0.3.0's Sampler" do
      expect { set.values(rng: scripted(0.5, 1.5, 0.5, 0.5)) }.to raise_error(ArgumentError, /must answer a Float in \[0, 1\)/)
      expect { set.values(rng: forwarding) }.to raise_error(ArgumentError, /must answer rand/)
      expect(set.values(rng: Random.new(4))).to eq(SkiftetStatistical::Bandit::Posteriors.beta(set.parameters).values(rng: Random.new(4)))
    end

    # A part whose to_s answers `value`.
    def text_answering(value)
      part = Object.new
      part.define_singleton_method(:to_s) { value }
      part
    end

    # A name whose hash answers `value`.
    def hash_answering(value)
      name = Object.new
      name.define_singleton_method(:hash) { value }
      name.define_singleton_method(:inspect) { "odd" }
      name
    end

    # A Hash keyed by identity, so a key whose hash answers no Integer can be put in it.
    def identity(pairs)
      pairs.each_with_object({}.compare_by_identity) { |(key, value), hash| hash[key] = value }
    end

    it "the L3 barrier, round 5: a key part whose to_s answers no String is refused, before any draw" do
      draw = SkiftetStatistical::Bandit::Draw
      [ text_answering(42), text_answering(nil), text_answering(:s), BasicObject.new ].each do |part|
        [ -> { draw.uniform(part) }, -> { draw.uniform_for("t", part) }, -> { draw.seed_for(part) },
          -> { set.rank(key: [ "t", part ]) }, -> { set.pick(key: [ part ], explore: 0.5) } ].each do |call|
          expect { call.call }.to raise_error(ArgumentError, /a part must answer to_s with a String/)
        end
      end
      expect(draw.uniform_for("t", 7, nil)).to eq(draw.uniform_for("t", "7", ""))
    end

    it "the L3 barrier, round 5: a name whose hash answers no Integer, or that answers no eql?, is not an item" do
      no_eql = Class.new(BasicObject) do
        define_method(:hash) { "a".hash }
        define_method(:inspect) { "no-eql" }
      end.new
      [ hash_answering("x"), hash_answering(nil), hash_answering(2.5), no_eql ].each do |name|
        [ -> { set.pick(rng: Random.new(1), may_lead: [ name ]) }, -> { set.rank(rng: Random.new(1), pinned: { 0 => name }) },
          -> { set.summary(among: [ name ]) }, -> { set.greedy(may_lead: [ name ]) }, -> { set.pick_probabilities(may_lead: [ name ]) },
          -> { SkiftetStatistical::Bandit::Posteriors.beta({ "a" => [ 1.0, 1.0 ] }, views: identity([ [ name, 1.0 ] ])) },
          -> { SkiftetStatistical::Bandit::Posteriors.yes_no({ "a" => {} }, priors: identity([ [ "a", [ 1.0, 1.0 ] ], [ name, [ 1.0, 1.0 ] ] ])) } ]
          .each { |call| expect { call.call }.to raise_error(ArgumentError, /is not an item/) }
      end
    end

    it "the L3 barrier, round 5: the mixture refuses a branch or an item that a Hash cannot look up" do
      posteriors = SkiftetStatistical::Bandit::Posteriors
      [ hash_answering("x"), BasicObject.new ].each do |key|
        [ -> { posteriors.mixture(shares: identity([ [ key, 1.0 ] ]), probabilities: identity([ [ key, { "a" => 1.0 } ] ])) },
          -> { posteriors.mixture(shares: { x: 1.0 }, probabilities: { x: identity([ [ key, 1.0 ] ]) }) },
          -> { posteriors.mixture(shares: {}, probabilities: identity([ [ key, {} ] ])) } ]
          .each { |call| expect { call.call }.to raise_error(ArgumentError, /cannot be looked up/) }
      end
    end

    it "the L3 barrier, round 5: a pinned position is read by the bandit's one reading of a whole number" do
      [ 2, -1, 1.0, Rational(1, 1), 10**400, "0", nil ].each do |position|
        expect { set.rank(rng: Random.new(1), pinned: { position => "a" }) }
          .to raise_error(ArgumentError, /pinned: a position must be an Integer of at least 0 and below 2 within a Float's range/)
      end
      expect(set.rank(rng: Random.new(1), pinned: { 1 => "a" }).items).to eq(%w[b a])
    end

    it "the L3 barrier, round 5: NormalArgmax.expected_maximum refuses what probabilities refuses, by one guard" do
      normal = SkiftetStatistical::Bandit::NormalArgmax
      [ [ [ 0.0, 5e-324 ], [ 0.0, 5e-324 ] ], [ [ 0.0, 1e-320 ], [ 1e-321, 1e-320 ] ] ].each do |members|
        [ -> { normal.probabilities(members) }, -> { normal.expected_maximum(members) } ].each do |call|
          expect { call.call }.to raise_error(ArgumentError, /beyond what the rule can integrate/)
        end
      end
      expect(normal.probabilities([ [ 1e100, 1.0 ], [ -1e100, 1.0 ] ])).to eq([ 1.0, 0.0 ])
      expect(normal.expected_maximum([ [ 1e100, 1.0 ], [ -1e100, 1.0 ] ])).to eq(1e100)
    end

    # The L3 barrier's sixth round: the round 5 inputs that overflowed (parameters near a Float's largest) are out of
    # the input domain, refused at the one number reader before anything is computed.
    it "the L3 barrier, round 6: the round 5 overflow inputs are refused at the reader, as out of the input domain" do
      prior = SkiftetStatistical::Bandit::Prior
      big = { views: 1e308, outcomes: 1e308 }
      high = [ [ 1.7e308, 2e307 ], [ 1.7e308, 2e307 ] ]
      [ -> { prior.pooled_new(others: [ { group: big }, { group: big } ]) },
        -> { prior.pooled(group: big, strength: 1e308, base: [ 1e308, 1e308 ]) },
        -> { prior.pooled_mean(views: 1e308, outcomes: 1e308, prior_mean: 1e308, strength: 1e308) },
        -> { prior.pooled_amount(group: { views: 1e-300, outcomes: 1e300, outcome_squares: 1e300 }, strength: 1.0) },
        -> { SkiftetStatistical::Bandit::Decay.fold([ { age: 0, views: 1e308 }, { age: 0, views: 1e308 } ], half_life: 1.0) },
        -> { SkiftetStatistical::Bandit::NormalArgmax.probabilities([ [ 1e308, 1.0 ], [ -1e308, 1.0 ] ]) },
        -> { SkiftetStatistical::Bandit::NormalArgmax.expected_maximum(high) },
        -> { SkiftetStatistical::Bandit::Posteriors.normal({ "a" => high[0], "b" => high[1] }) } ].each do |call|
        expect { call.call }.to raise_error(ArgumentError, /must lie within the input domain, at most 1\.0e\+100 either side of 0/)
      end
    end

    it "the L3 barrier, round 5: Prior and the set refuse an answer that leaves a Float, naming the inputs" do
      prior = SkiftetStatistical::Bandit::Prior
      {
        "pooled_mean's views, outcomes, prior_mean and strength" =>
          -> { prior.pooled_mean(views: 0, outcomes: 1e100, prior_mean: 1e100, strength: 1e-300) },
        "pooled_amount's group and own" =>
          -> { prior.pooled_amount(group: { views: 1e-300, outcomes: 1e100, outcome_squares: 1e100 }, strength: 1.0) },
        "item \"a\"'s tally and prior" => lambda do
          SkiftetStatistical::Bandit::Posteriors.amounts({ "a" => { views: 1e-300, outcomes: 1e100, outcome_squares: 1e100 } },
                                                         priors: prior.amount(mean: 1.0, variance: 1.0, strength: 1.0))
        end
      }.each do |inputs, call|
        expect { call.call }.to raise_error(ArgumentError, /\A#{Regexp.escape(inputs)} overflow a Float's range/)
      end
    end

    it "the L3 barrier, round 5: which weighted sums a tally counts is 0.3.0's WeightedPosterior.count" do
      weighted = SkiftetStatistical::Bandit::WeightedPosterior
      allow(weighted).to receive(:count).and_call_original
      tally = { views: 10.0, outcomes: 2.0, weighted_views: 20.0, weighted_outcomes: 3.0, weighted_views_squared: 50.0 }
      parameters = SkiftetStatistical::Bandit::Posteriors.yes_no({ "a" => tally }).parameters
      expect(weighted).to have_received(:count).once
      count = weighted.count({ exposures: 10, conversions: 2, weighted_exposures: 20.0, weighted_conversions: 3.0,
                               weighted_exposures_squared: 50.0 })
      expect(parameters["a"]).to eq(weighted.beta(count))
    end

    # The L3 barrier's sixth round: one input domain, read by one number reader; one reading of a pair; rows and
    # Amounts read by the one tally reader; a Normal quantile on the open grid; greedy the order rule's first largest.
    def bandit = SkiftetStatistical::Bandit

    # Numerics the one reader cannot read: no to_f, a to_f answering a String, nil or an infinity, or raising.
    def odd_numerics
      [ Class.new(Numeric).new, Class.new(Numeric) { define_method(:to_f) { "x" } }.new,
        Class.new(Numeric) { define_method(:to_f) { nil } }.new,
        Class.new(Numeric) { define_method(:to_f) { -Float::INFINITY } }.new,
        Class.new(Numeric) { define_method(:to_f) { raise "no" } }.new ]
    end

    # One call of each new method that reads a number, given that number.
    def number_readers
      prior = bandit::Prior
      {
        "Decay.weight" => ->(v) { bandit::Decay.weight(v, half_life: 1.0) },
        "Decay.fold" => ->(v) { bandit::Decay.fold([ { age: 0, views: v } ], half_life: 1.0) },
        "Posteriors.beta" => ->(v) { bandit::Posteriors.beta({ "a" => [ v, 1.0 ] }) },
        "Posteriors.normal" => ->(v) { bandit::Posteriors.normal({ "a" => [ v, 1.0 ] }) },
        "Posteriors.yes_no" => ->(v) { bandit::Posteriors.yes_no({ "a" => { views: v } }) },
        "Posteriors.amounts" => lambda do |v|
          bandit::Posteriors.amounts({ "a" => { outcomes: v } }, priors: prior.amount(mean: 1.0, variance: 1.0, strength: 1.0))
        end,
        "Posteriors.mixture" => ->(v) { bandit::Posteriors.mixture(shares: { x: v }, probabilities: { x: {} }) },
        "rank min_views" => ->(v) { set.rank(rng: Random.new(1), min_views: v) },
        "rank priority" => lambda do |v|
          set.rank(uniforms: { explore: 0.5, priority: { "a" => v, "b" => 0.1 }, draw: { "a" => 0.1, "b" => 0.2 } })
        end,
        "Prior.given" => ->(v) { prior.given(mean: 0.5, strength: v) },
        "Prior.pooled_mean" => ->(v) { prior.pooled_mean(views: 1.0, outcomes: v, prior_mean: 0.5, strength: 1.0) },
        "Prior.tilts" => ->(v) { prior.tilts({ a: v }, clip: [ 0.5, 2.0 ]) },
        "Draw.open_uniform" => ->(v) { bandit::Draw.open_uniform(v) },
        "NormalArgmax.probabilities" => ->(v) { bandit::NormalArgmax.probabilities([ [ v, 1.0 ], [ 0.0, 1.0 ] ]) },
        "BetaArgmax.position" => ->(v) { SkiftetStatistical::BetaArgmax.position(0.5, v, 2.0) },
        "BetaArgmax.distribution" => ->(v) { SkiftetStatistical::BetaArgmax.distribution(v, 2.0, 2.0) }
      }
    end

    it "the L3 barrier, round 6: every new method refuses a Numeric the one reader cannot read, as an ArgumentError" do
      number_readers.each do |name, call|
        odd_numerics.each do |value|
          expect { call.call(value) }.to raise_error(ArgumentError, /must be a finite number, got/), "#{name} #{value.class}"
        end
      end
      # A weight that is not a number counts 0 (the weighted draw's own rule), so every key counts 1.
      expect(bandit::Draw.weighted({ "a" => odd_numerics.first, "b" => odd_numerics.last }, 0.75)).to eq([ "b", 0.5 ])
    end

    it "the L3 barrier, round 6: every new method refuses a number beyond the input domain where it reads it" do
      words = /must lie within the input domain, at most 1\.0e\+100 either side of 0/
      number_readers.each do |name, call|
        [ 1e100.next_float, -1e200, 10**150 ].each do |value|
          expect { call.call(value) }.to raise_error(ArgumentError, words), "#{name} #{value.inspect}"
        end
      end
      expect { bandit::Draw.weighted({ "a" => 1.0, "b" => 1e101 }, 0.5) }.to raise_error(ArgumentError, words)
      hold = Class.new { define_method(:satisfied_for?) { |_leader| true } }.new
      expect { bandit::Pass.call(status: "running", hold: hold, seed: 10**101) { {} } }.to raise_error(ArgumentError, words)
      expect { set.pick_probabilities(draws: 10**101) }.to raise_error(ArgumentError, words)
      # At the bound itself every reader answers, and no sum or product of such inputs leaves a Float.
      expect(bandit::Prior.pooled_mean(views: 1e100, outcomes: 1e100, prior_mean: 1e100, strength: 1e100))
        .to be_within(1e85).of(5e99)
      expect(bandit::Posteriors.mixture(shares: { x: 1e100, y: 1e100 }, probabilities: { x: { "a" => 1e100 }, y: { "a" => 1e100 } }))
        .to eq("a" => 2e200)
    end

    it "the L3 barrier, round 6: Decay.fold reads its rows once through the one tally reader" do
      odd = Object.new
      odd.define_singleton_method(:map) { |*| 42 }
      [ odd, Object.new, 5, "rows", { age: 0, views: 1 }, BasicObject.new ].each do |rows|
        expect { bandit::Decay.fold(rows, half_life: 1.0) }.to raise_error(ArgumentError, /rows must be a list of tallies/)
      end
      [ [ 5 ], [ "x" ] ].each do |rows|
        expect { bandit::Decay.fold(rows, half_life: 1.0) }.to raise_error(ArgumentError, /rows\[0\] must be a Hash or nil/)
      end
      expect { bandit::Decay.fold([ nil ], half_life: 1.0) }.to raise_error(ArgumentError, /rows\[0\] needs an age/)
      once = Enumerator.new { |rows| rows << { age: 0, views: 2, outcomes: -1 } }.lazy
      expect(bandit::Decay.fold(once, half_life: 1.0)).to eq(views: 2.0, outcomes: -1.0, outcome_squares: 0.0)
      # A list whose own map answers something else is read by its each, never by its map.
      listed = Class.new do
        include Enumerable
        define_method(:each) { |&row| row.call({ age: 0, views: 3 }) }
        define_method(:map) { |*| 42 }
      end.new
      expect(bandit::Decay.fold(listed, half_life: 1.0)).to eq(views: 3.0, outcomes: 0.0, outcome_squares: 0.0)
    end

    it "the L3 barrier, round 6: an Amount that skipped its checks is refused, as an ArgumentError" do
      skipping = Class.new(bandit::Prior::Amount) { define_method(:initialize) { |**| nil } }
      tally = { "a" => { views: 1, outcomes: 1, outcome_squares: 1 } }
      [ bandit::Prior::Amount.allocate, skipping.new ].each do |prior|
        expect { bandit::Posteriors.amounts(tally, priors: prior) }.to raise_error(ArgumentError, /mean must be a finite number/)
        expect { bandit::Posteriors.amounts(tally, priors: { "a" => prior }) }.to raise_error(ArgumentError, /must be a finite number/)
      end
      expect { bandit::Posteriors.amounts(tally, priors: [ 1.0, 1.0 ]) }.to raise_error(ArgumentError, /must be a Prior::Amount/)
    end

    it "the L3 barrier, round 6: which amount sum may be below 0 is declared once, for every reader of a tally" do
      expect(bandit.send(:const_get, :Tallies)::SIGNED_SUMS).to eq(%i[outcomes weighted_outcomes])
      expect(bandit::Decay.fold([ { age: 0, views: 1, outcomes: -2 } ], half_life: 1.0)[:outcomes]).to eq(-2.0)
      expect(bandit::Prior.pooled_amount(group: { views: 4, outcomes: -8, outcome_squares: 20 }, strength: 2.0).mean).to eq(-2.0)
      weighted = { views: 4, outcomes: -8, outcome_squares: 20, weighted_views: 4, weighted_outcomes: -8, weighted_outcome_squares: 20 }
      amount = bandit::Prior.amount(mean: 1.0, variance: 1.0, strength: 1.0)
      expect(bandit::Posteriors.amounts({ "a" => weighted }, priors: amount).parameters["a"][0]).to be < 0.0
      [ -> { bandit::Decay.fold([ { age: 0, views: -1 } ], half_life: 1.0) },
        -> { bandit::Decay.fold([ { age: 0, outcome_squares: -1 } ], half_life: 1.0) },
        -> { bandit::Posteriors.yes_no({ "a" => { views: 4, outcomes: -1 } }) },
        -> { bandit::Prior.borrowed(group: { views: 4, outcomes: -1 }) } ].each do |call|
        expect(&call).to raise_error(ArgumentError, /must be >= 0/)
      end
    end

    it "the L3 barrier, round 6: a Normal quantile draw reads its uniform on the open grid; a Beta draw reads it as it is" do
      low = 2.0**-53
      expect(bandit::Draw.open_uniform(0.0)).to eq(low)
      expect(bandit::Draw.open_uniform(1.0)).to eq(1.0 - low)
      expect(bandit::Draw.open_uniform(0.25)).to eq(0.25)
      [ -0.1, 1.5, nil, "0.5" ].each { |bad| expect { bandit::Draw.open_uniform(bad) }.to raise_error(ArgumentError) }
      quantile = ->(u) { SkiftetStatistical::Significance.normal_quantile(u) }
      normal = bandit::Posteriors.normal({ "a" => [ 2.0, 3.0 ], "b" => [ -1.0, 0.5 ] })
      given = { explore: 0.5, priority: { "a" => 0.1, "b" => 0.2 }, draw: { "a" => 0.0, "b" => 1.0 } }
      expect(normal.rank(uniforms: given).draws).to eq("a" => 2.0 + (3.0 * quantile.call(low)), "b" => -1.0 + (0.5 * quantile.call(1.0 - low)))
      expect(normal.pick(uniforms: given).draws.values).to all(be_finite)
      summary = normal.summary(credible_mass: 1.0 - low)["a"]
      expect(summary[:lower]).to eq(2.0 - (-quantile.call(low) * 3.0))
      expect(summary[:upper]).to eq(2.0 + (-quantile.call(low) * 3.0))
      allow(bandit::Draw).to receive(:open_uniform).and_call_original
      normal.rank(key: [ "t", 1 ])
      expect(bandit::Draw).to have_received(:open_uniform).twice
      beta = bandit::Posteriors.beta({ "a" => [ 2.0, 3.0 ], "b" => [ 4.0, 1.0 ] })
      expect(beta.rank(uniforms: given).draws).to eq("a" => 0.0, "b" => 1.0)
    end

    it "the L3 barrier, round 6: a pair is read by one reader, with one wording, wherever one is read" do
      [ nil, [ 1.0 ], [ 1.0, 2.0, 3.0 ], "ab" ].each do |pair|
        [ -> { bandit::Posteriors.beta({ "a" => pair }) }, -> { bandit::Posteriors.normal({ "a" => pair }) },
          -> { bandit::Posteriors.yes_no({ "a" => {} }, priors: { "a" => pair }) },
          -> { bandit::Prior.pooled(group: {}, strength: 1.0, base: pair) },
          -> { bandit::Prior.tilts({ a: 1.0 }, clip: pair) },
          -> { bandit::NormalArgmax.probabilities([ [ 0.0, 1.0 ], pair ]) },
          -> { SkiftetStatistical::BetaArgmax.expected_maximum([ [ 1.0, 1.0 ], pair ]) } ].each do |call|
          expect(&call).to raise_error(ArgumentError, /must be a pair \[\w+( \w+)?, \w+( \w+)?\] \(got /)
        end
      end
    end

    it "the L3 barrier, round 6: greedy is the order rule's first largest, BetaArgmax.greedy's answer on Betas" do
      expect(bandit::Posteriors.beta({ "a" => [ 1.0, 1.0 ], "b" => [ 2.0, 2.0 ], "c" => [ 1.0, 3.0 ] }).greedy).to eq("a")
      expect(bandit::Posteriors.normal({ "a" => [ 1.0, 1.0 ], "b" => [ 1.0, 9.0 ] }).greedy).to eq("a")
      expect(bandit::Posteriors.beta({}).greedy).to be_nil
      rng = Random.new(66)
      200.times do
        pairs = Array.new(1 + rng.rand(20)) { [ rng.rand(1..9).to_f, rng.rand(1..9).to_f ] }
        set = bandit::Posteriors.beta(pairs.each_with_index.to_h { |pair, i| [ "i#{i}", pair ] })
        expect(set.greedy).to eq("i#{SkiftetStatistical::BetaArgmax.greedy(pairs)}")
      end
    end

    # The L3 barrier's seventh round (01 D34). The amounts rule's spread is formed from its two shares, k / (k + n) and
    # n / (k + n), taken first, so no product multiplies more than two numbers of the input domain's size. Each exact
    # pair below is the amounts rule (01 GB3) in Rational arithmetic on the same Floats: mu_n = (k mu0 + n m) / (k + n),
    # sd_n^2 = [k v0 + n s^2 + k n (m - mu0)^2 / (k + n)] / (k + n)^2.
    def exact_amounts(views, outcomes, squares, mean, variance, strength)
      k = strength.to_r
      n = views.to_r
      mu0 = mean.to_r
      m = n.zero? ? 0r : outcomes.to_r / n
      s2 = n.zero? ? 0r : [ (squares.to_r / n) - (m * m), 0r ].max
      total = k + n
      spread = ((k * variance.to_r) + (n * s2) + (k * n * ((m - mu0)**2) / total)) / (total * total)
      [ ((k * mu0) + (n * m)) / total, Math.sqrt(spread.to_f) ]
    end

    def expect_amounts(views, outcomes, squares, mean, variance, strength)
      prior = bandit::Prior.amount(mean: mean, variance: variance, strength: strength)
      pair = bandit::Posteriors.amounts({ "a" => { views: views, outcomes: outcomes, outcome_squares: squares } },
                                        priors: prior).parameters["a"]
      exact_mean, exact_sd = exact_amounts(views, outcomes, squares, mean, variance, strength)
      label = [ views, outcomes, squares, mean, variance, strength ].inspect
      expect((pair[0] - exact_mean).abs).to be <= 1e-12 * [ exact_mean.abs, 1.0 ].max, "#{label}: mean #{pair[0]}"
      expect((pair[1] - exact_sd).abs).to be <= 1e-12 * exact_sd, "#{label}: sd #{pair[1]} vs #{exact_sd}"
    end

    it "the L3 barrier, round 7: the amounts rule builds in-domain tallies at the bound, its spread from its shares" do
      # The round 6 review's case: k = n = |mu0| = 1e100 at a mean of 0, refused before as an overflow (the answer
      # read [5e99, Infinity]); its exact pair is [5e99, 3.5355339059327376e49].
      prior = bandit::Prior.amount(mean: 1e100, variance: 1.0, strength: 1e100)
      pair = bandit::Posteriors.amounts({ "a" => { views: 1e100, outcomes: 0, outcome_squares: 0 } }, priors: prior)
                               .parameters["a"]
      expect(pair[0]).to eq(5e99)
      expect(pair[1]).to be_within(1e-12 * 3.5355339059327376e49).of(3.5355339059327376e49)
      expect_amounts(1e100, 0, 0, 1e100, 1.0, 1e100)
      expect_amounts(1e54, 0, 0, 1e100, 1.0, 1e55)
      expect_amounts(1e90, 0, 0, 1e90, 1.0, 1e90)
      expect_amounts(1e10, 1e100, 1e100, -1e100, 1.0, 1e100)
      expect_amounts(1e100, -1e100, 1e100, 1e100, 1e100, 1e100)
      rng = Random.new(707)
      300.times do
        magnitude = -> { 10.0**(rng.rand * 100) }
        views = magnitude.call
        outcomes = ((rng.rand < 0.5 ? -1 : 1) * [ magnitude.call, 1e100 / views ].min * views).clamp(-1e100, 1e100)
        squares = [ outcomes * outcomes / views * (1.0 + rng.rand), 1e100 ].min
        expect_amounts(views, outcomes, squares, (rng.rand < 0.5 ? -1 : 1) * magnitude.call, 10.0**(rng.rand * 100),
                       magnitude.call)
      end
    end

    it "the L3 barrier, round 7: the amounts posterior's mean is the one shrunk mean, the one Prior.pooled_mean asks" do
      prior = bandit::Prior.amount(mean: 4.0, variance: 2_000.0, strength: 20.0)
      # The last three are tallies whose mean a restated form, k / (k + n) * mu0 + n / (k + n) * m, rounds otherwise.
      [ [ 100.0, 600.0, 90_000.0 ], [ 4.0, 20.0, 90.0 ], [ 0.0, 0.0, 0.0 ], [ 3.0, -7.5, 30.0 ], [ 190.0, -86.56, 1e4 ],
        [ 401.0, 1458.89, 1e5 ], [ 412.0, 208.28, 1e4 ] ].each do |views, outcomes, squares|
        set = bandit::Posteriors.amounts({ "a" => { views: views, outcomes: outcomes, outcome_squares: squares } },
                                         priors: prior)
        mean = views.zero? ? 0.0 : outcomes / views
        expect(set.parameters["a"][0])
          .to eq(bandit::Prior.pooled_mean(views: views, outcomes: views * mean, prior_mean: 4.0, strength: 20.0))
      end
    end

    # NormalArgmax computes each member's probability in its own standardised variable. The references are the
    # two-member closed form P(the second is the larger) = Phi((m2 - m1) / sqrt(s1**2 + s2**2)), computed apart at 40
    # digits in BigDecimal by Phi's Taylor series: for [[1e10, 1e30], [0, 1e-10]],
    # 0.4999999999999999999960105771959856732206 (the Float 0.5); for [[1, 1000], [0, 1e-14]],
    # 0.4996010577860889374154096775556730587514. Round 6 answered [1.0, 0.0] and 0.49919 for these. The stated bound
    # for two members is 2 k (k + 1) TAIL + 2e-12, 1.202e-9.
    it "the L3 barrier, round 7: NormalArgmax gives a narrow member its own integral (the round 6 review's two cases)" do
      bound = (12 * SkiftetStatistical::BetaArgmax::TAIL) + 2e-12
      wide = bandit::NormalArgmax.probabilities([ [ 1e10, 1e30 ], [ 0.0, 1e-10 ] ])
      expect(wide[1]).to be_within(bound).of(0.4999999999999999999960105771959856732206)
      expect(wide[0]).to be_within(bound).of(0.5)
      near = bandit::NormalArgmax.probabilities([ [ 1.0, 1000.0 ], [ 0.0, 1e-14 ] ])
      expect(near[1]).to be_within(bound).of(0.4996010577860889374154096775556730587514)
      expect(near.sum).to be_within(1e-15).of(1.0)
      set = bandit::Posteriors.normal({ "wide" => [ 1e10, 1e30 ], "narrow" => [ 0.0, 1e-10 ] })
      expect(set.pick_probabilities[:heads]["narrow"]).to be_within(bound).of(0.5)
      expect(set.summary["narrow"][:p_best]).to be_within(bound).of(0.5)
    end

    it "the L3 barrier, round 7: NormalArgmax meets two members' closed form within its bound over width ratios to 1e40" do
      bound = (12 * SkiftetStatistical::BetaArgmax::TAIL) + 2e-12
      closed = ->(d) { 0.5 * Math.erfc(-d / Math.sqrt(2.0)) }
      rng = Random.new(1040)
      400.times do
        base = 10.0**((rng.rand * 80) - 40)
        deviations = [ base, base * (10.0**((rng.rand * 80) - 40)) ]
        wide = deviations.max
        means = Array.new(2) { (rng.rand - 0.5) * 6 * wide * (10.0**((rng.rand * 2) - 2)) }
        members = means.zip(deviations)
        answer = bandit::NormalArgmax.probabilities(members)
        exact = closed.call((means[1] - means[0]) / Math.hypot(*deviations))
        expect(answer[1]).to be_within(bound).of(exact), members.inspect
      end
    end

    it "the L3 barrier, round 7: a caller's method that raises is an ArgumentError naming the input (one guarded call)" do
      bad_to_s = Class.new { def to_s = raise("to_s"); def inspect = "#<bad to_s>" }.new
      bad_hash = Class.new { def hash = raise("hash"); def inspect = "#<bad hash>" }.new
      bad_eql = Class.new { def hash = "a".hash; def eql?(_other) = raise("eql"); def inspect = "#<bad eql>" }.new
      bad_rand = Class.new { def rand(*) = raise("rand") }.new
      bad_each = Class.new { include Enumerable; def each = raise("each") }.new
      hold = Class.new { def satisfied_for?(_leader) = true }.new
      {
        /a part's to_s raised RuntimeError/ => [ -> { bandit::Draw.uniform_for(bad_to_s) }, -> { bandit::Draw.uniform(bad_to_s) },
                                                 -> { set.rank(key: [ bad_to_s ]) } ],
        /a name's hash raised RuntimeError/ => [ -> { set.summary(among: [ bad_hash ]) },
                                                 -> { set.pick(key: "k", may_lead: [ bad_hash ]) },
                                                 -> { set.rank(key: "k", pinned: { 0 => bad_hash }) } ],
        /among: a name's lookup raised RuntimeError/ => [ -> { set.summary(among: [ bad_eql ]) } ],
        /a status's to_s raised RuntimeError/ => [ -> { bandit::Pass.call(status: bad_to_s, hold: hold, seed: 1) { {} } } ],
        /rng.rand raised RuntimeError/ => [ -> { set.rank(rng: bad_rand) }, -> { set.values(rng: bad_rand) } ],
        /each raised RuntimeError/ => [ -> { set.pick(key: "k", may_lead: bad_each) },
                                        -> { bandit::Decay.fold(bad_each, half_life: 1.0) },
                                        -> { bandit::Prior.chain(levels: bad_each, root_mean: 0.1) } ]
      }.each do |words, calls|
        calls.each { |call| expect(&call).to raise_error(ArgumentError, words) }
      end
      yielded = false
      expect { bandit::Pass.call(status: bad_to_s, hold: hold, seed: 1) { yielded = true } }.to raise_error(ArgumentError)
      expect(yielded).to be(false)
      # Numbers.read reads a number whose real? raises as no number, by the same guarded call.
      bad_real = Class.new(Numeric) { def real? = raise("real"); def to_f = 1.0 }.new
      expect { bandit::Decay.weight(bad_real, half_life: 1.0) }.to raise_error(ArgumentError, /an age must be a finite number/)
    end

    it "the L3 barrier, round 7: a tally, views or uniforms Hash is read by fetch, so a default it holds is never read" do
      raising = Hash.new { raise "default" }
      expect(bandit::Posteriors.yes_no({ "a" => raising }).parameters).to eq("a" => [ 1.0, 1.0 ])
      expect(bandit::Posteriors.yes_no({ "a" => Hash.new(5) }).parameters).to eq("a" => [ 1.0, 1.0 ])
      expect(bandit::Decay.fold([ Hash.new(3).merge(age: 0) ], half_life: 1.0))
        .to eq(views: 0.0, outcomes: 0.0, outcome_squares: 0.0)
      expect(bandit::Posteriors.beta({ "a" => [ 1.0, 1.0 ] }, views: Hash.new(7)).views).to eq("a" => 0.0)
      expect { set.rank(uniforms: raising) }.to raise_error(ArgumentError, /uniforms' explore must be a finite number/)
      expect { bandit::Posteriors.mixture(shares: { x: 1.0 }, probabilities: Hash.new({})) }
        .to raise_error(ArgumentError, /probabilities must name a Hash of item => probability for the branch :x/)
    end

    it "the L3 barrier, round 7: a caller's Hash, Array or String is read by its contents, never by a subclass's methods" do
      hash_class = Class.new(Hash) do
        %i[each each_pair map to_a key? fetch [] each_key keys values transform_values merge empty? size].each do |name|
          define_method(name) { |*| raise "Hash##{name}" }
        end
      end
      array_class = Class.new(Array) do
        %i[each_with_index map to_a size [] first last empty?].each { |name| define_method(name) { |*| raise "Array##{name}" } }
      end
      string_class = Class.new(String) do
        %i[b encoding valid_encoding? match? include? frozen? dup].each { |name| define_method(name) { |*| raise "String##{name}" } }
      end
      odd = lambda do |pairs|
        hash = hash_class.new
        pairs.each { |key, value| Hash.instance_method(:store).bind_call(hash, key, value) }
        hash
      end
      list = lambda do |*values|
        array = array_class.new
        values.each { |value| Array.instance_method(:push).bind_call(array, value) }
        array
      end
      expect(bandit::Posteriors.yes_no(odd.call("a" => odd.call(views: 4, outcomes: 1)), priors: odd.call("a" => [ 1, 1 ]))
                                       .parameters).to eq("a" => [ 2.0, 4.0 ])
      expect(bandit::Posteriors.beta({ "a" => list.call(1.0, 2.0) }, views: odd.call("a" => 3)).views).to eq("a" => 3.0)
      expect(set.rank(key: "k", pinned: odd.call(0 => "b")).items).to eq(%w[b a])
      expect(bandit::Posteriors.mixture(shares: odd.call(x: 1.0), probabilities: odd.call(x: odd.call("a" => 1.0))))
        .to eq("a" => 1.0)
      expect(bandit::Draw.weighted(odd.call("a" => 1.0, "b" => 3.0), 0.5)).to eq([ "b", 0.75 ])
      expect(bandit::Prior.tilts(odd.call(a: 2.0, b: 0.5), clip: list.call(0.1, 10.0))).to eq(a: 2.0, b: 0.5)
      expect(bandit::NormalArgmax.probabilities(list.call([ 0.0, 1.0 ], [ 0.0, 1.0 ]))).to eq([ 0.5, 0.5 ])
      expect(SkiftetStatistical::BetaArgmax.expected_maximum(list.call([ 1.0, 1.0 ], [ 1.0, 1.0 ]))).to be_within(1e-8).of(2.0 / 3)
      named = bandit::Posteriors.beta({ string_class.new("a") => [ 1.0, 2.0 ] })
      expect(named.items).to eq(%w[a])
      expect(named.items.first.class).to eq(String)
      hold = Class.new { def satisfied_for?(_leader) = true }.new
      expect(bandit::Pass.call(status: "running", hold: hold, seed: 1) { odd.call(leader: "a") }).to eq(leader: "a", ready: true)
    end

    it "the L3 barrier, round 7: one list reader: an Enumerator and an Array read alike, a non-Enumerable each refused alike" do
      levels = [ { views: 1_000, outcomes: 120, strength: 20.0 }, { views: 200, outcomes: 30, strength: 30.0 } ]
      rows = [ { age: 0, views: 10, outcomes: 3 }, { age: 72, views: 20, outcomes: 4 } ]
      others = [ { group: { views: 10, outcomes: 2 } }, { group: { views: 30, outcomes: 9 } } ]
      expect(bandit::Prior.chain(levels: levels.each, root_mean: 0.1)).to eq(bandit::Prior.chain(levels: levels, root_mean: 0.1))
      expect(bandit::Decay.fold(rows.each, half_life: 72)).to eq(bandit::Decay.fold(rows, half_life: 72))
      expect(bandit::Prior.pooled_new(others: others.each)).to eq(bandit::Prior.pooled_new(others: others))
      expect(set.pick(key: "k", may_lead: %w[b].each).item).to eq(set.pick(key: "k", may_lead: %w[b]).item)
      each_only = ->(values) { Class.new { define_method(:each) { |&block| values.each(&block) } }.new }
      [ -> { bandit::Prior.chain(levels: each_only.call(levels), root_mean: 0.1) },
        -> { bandit::Decay.fold(each_only.call(rows), half_life: 72) },
        -> { bandit::Prior.pooled_new(others: each_only.call(others)) },
        -> { set.pick(key: "k", may_lead: each_only.call(%w[b])) },
        -> { bandit::Prior.pooled_new(others: { group: { views: 10, outcomes: 2 } }) },
        -> { bandit::Prior.chain(levels: levels.first, root_mean: 0.1) } ].each do |call|
        expect(&call).to raise_error(ArgumentError, /must be a list of /)
      end
    end

    # The L3 barrier's eighth round (01 D35). The amounts rule's standard deviation is the root of its three terms, each
    # formed as a mantissa and a power of 2 apart (Math.frexp), so no share underflows and no product of numbers of the
    # domain's size is formed. Each reference below is the rule (01 GB3) in Rational arithmetic on the same Floats, its
    # root taken by Integer.sqrt on the scaled Rational (so a variance below the smallest Float has its root too).
    def exact_root(rational)
      return 0.0 if rational.zero?

      product = rational.numerator * rational.denominator
      shift = [ 0, 80 - (product.bit_length / 2) ].max
      Rational(Integer.sqrt(product << (2 * shift)), rational.denominator << shift).to_f
    end

    # [mean, sd, the mean's terms' size] of the rule on a plain tally, exactly.
    def exact_rule(views, outcomes, squares, mean, variance, strength)
      k = strength.to_r
      n = views.to_r
      mu0 = mean.to_r
      m = n.zero? ? 0r : outcomes.to_r / n
      s2 = n.zero? ? 0r : [ (squares.to_r / n) - (m * m), 0r ].max
      total = k + n
      spread = ((k * variance.to_r) + (n * s2) + (k * n * (m - mu0) * (m - mu0) / total)) / (total * total)
      [ ((k * mu0) + (n * m)) / total, exact_root(spread), ((k * mu0).abs + (n * m).abs) / total ]
    end

    # Builds the set when the exact pair lies in the input domain, with each number within 1e-12 of the exact one (the
    # mean within 1e-12 of its terms' size, the conditioning of a sum), and refuses it otherwise. An exact sd below the
    # smallest normal Float reads within a few of the smallest Float steps, or 0, which the set refuses.
    def expect_rule(views, outcomes, squares, mean, variance, strength)
      label = [ views, outcomes, squares, mean, variance, strength ].inspect
      exact_mean, exact_sd, size = exact_rule(views, outcomes, squares, mean, variance, strength)
      build = lambda do
        bandit::Posteriors.amounts({ "a" => { views: views, outcomes: outcomes, outcome_squares: squares } },
                                   priors: bandit::Prior.amount(mean: mean, variance: variance, strength: strength))
                          .parameters["a"]
      end
      if exact_mean.abs > 1e100 || exact_sd > 1e100
        expect(&build).to raise_error(ArgumentError), label
        return :refused
      end
      if exact_sd < Float::MIN
        pair = begin
          build.call
        rescue ArgumentError => e
          expect(e.message).to match(/standard deviation must be above 0 \(got 0\.0\)/), label
          return :below
        end
        expect((pair[1] - exact_sd).abs).to be <= Math.ldexp(1.0, -1070), "#{label}: sd #{pair[1]} vs #{exact_sd}"
        return :below
      end
      pair = build.call
      expect((pair[0] - exact_mean).abs).to be <= 1e-12 * size, "#{label}: mean #{pair[0]} vs #{exact_mean.to_f}"
      expect((pair[1] - exact_sd).abs).to be <= 1e-12 * exact_sd, "#{label}: sd #{pair[1]} vs #{exact_sd}"
      :built
    end

    it "the L3 barrier, round 8: the amounts spread keeps the item's evidence when n is far below k (the review's first case)" do
      # m = 1e150, s^2 = 1e300: sd^2 = [k v0 + n s^2 + k n (m - mu0)^2 / (k + n)] / (k + n)^2 = 2e-150; round 7 read 1e-90.
      expect(expect_rule(1e-250, 1e-100, 2e50, 0.0, 1e-80, 1e100)).to eq(:built)
      pair = bandit::Posteriors.amounts({ "a" => { views: 1e-250, outcomes: 1e-100, outcome_squares: 2e50 } },
                                        priors: bandit::Prior.amount(mean: 0.0, variance: 1e-80, strength: 1e100))
                               .parameters["a"]
      expect(pair[1]).to be_within(1e-12 * Math.sqrt(2.0) * 1e-75).of(Math.sqrt(2.0) * 1e-75)
      expect(expect_rule(5e-324, 1e-170, 2.1e-17, 0.0, 1e-30, 3.0)).to eq(:built)
      expect(expect_rule(1e-300, 1e-160, 2e-20, 0.0, 1e-40, 1e10)).to eq(:built)
      expect(expect_rule(1e-200, 1e-95, 2e10, 1.0, 1e-90, 1e100)).to eq(:built)
    end

    it "the L3 barrier, round 8: the amounts spread is above 0 when k is far below n (the review's second case)" do
      # Every amount 1, a prior of strength 1e-300 at -1e100: sd^2 = 1e-300 exactly, so sd = 1e-150; round 7 refused it.
      pair = bandit::Posteriors.amounts({ "a" => { views: 1e100, outcomes: 1e100, outcome_squares: 1e100 } },
                                        priors: bandit::Prior.amount(mean: -1e100, variance: 1e100, strength: 1e-300))
                               .parameters["a"]
      expect(pair[0]).to eq(1.0)
      expect(pair[1]).to be_within(1e-12 * 1e-150).of(1e-150)
      expect(expect_rule(1e100, 1e100, 1e100, -1e100, 1e100, 1e-300)).to eq(:built)
      # A variance whose root is a Float though the variance is not: k = 1e100 at n = 0 and v0 = 1e-300 reads 1e-200.
      expect(expect_rule(0.0, 0.0, 0.0, 0.0, 1e-300, 1e100)).to eq(:built)
      # Strength and variance at the smallest Float: sd = sqrt(v0 / k) = 1.
      expect(expect_rule(0.0, 0.0, 0.0, 0.0, 5e-324, 5e-324)).to eq(:built)
      # An exact sd beyond the input domain is refused, never answered.
      expect(expect_rule(0.0, 0.0, 0.0, 0.0, 1e100, 5e-324)).to eq(:refused)
    end

    # The views are a power of 2 and the amounts a multiple of them, so the mean amount the rule divides out is exact
    # and the difference of two means rounds once: what is measured is the rule's arithmetic on its moments, at any
    # ratio of strength to views (a mean amount that rounds is a conditioning of the moments, not of the spread).
    it "the L3 barrier, round 8: the amounts rule meets its exact pair over ratios k / n from 1e-200 to 1e200 at the bound" do
      rng = Random.new(808)
      edge = -> { rng.rand < 0.25 }
      magnitude = -> { edge.call ? 1e100 : 10.0**((rng.rand * 200) - 100) }
      signed = -> { (rng.rand < 0.5 ? -1 : 1) * magnitude.call }
      counts = Hash.new(0)
      ratios = []
      2_000.times do
        large = edge.call ? 1e100 : 10.0**(100 - (rng.rand * 4))
        ratio = 10.0**(-rng.rand * 200)
        power = Math.frexp(large * ratio).last - 1
        views = Math.ldexp(1.0, rng.rand < 0.5 ? Math.frexp(large).last - 1 : power)
        strength = views >= large * ratio * 2 ? large * ratio : large
        mean_amount = signed.call
        mean_amount = Math.ldexp(mean_amount, -1) while (mean_amount * views).abs > 1e100
        mean_amount = Math.ldexp(mean_amount, 1) while (mean_amount * views).abs < 1e-300
        outcomes = mean_amount * views
        squares = outcomes * mean_amount * (1.01 + rng.rand)
        squares = 0.0 if squares > 1e100
        ratios << (strength / views)
        counts[expect_rule(views, outcomes, squares, signed.call, magnitude.call, strength)] += 1
      end
      expect(counts[:built]).to be > 1_000
      expect(ratios.min).to be < 1e-190
      expect(ratios.max).to be > 1e190
    end

    it "the L3 barrier, round 8: the amounts rule meets its exact pair past the bound's ratios, k / n down to 1e-400" do
      rng = Random.new(8080)
      counts = Hash.new(0)
      2_000.times do
        large = 10.0**(100 - (rng.rand * 2))
        small = large * (10.0**(-rng.rand * 400))
        small = 5e-324 if small < 5e-324
        views = Math.ldexp(1.0, Math.frexp(rng.rand < 0.5 ? large : small).last - 1)
        strength = views > small * 2 ? small : large
        mean_amount = (rng.rand < 0.5 ? -1 : 1) * (10.0**((rng.rand * 200) - 100))
        mean_amount = Math.ldexp(mean_amount, -1) while (mean_amount * views).abs > 1e100
        mean_amount = Math.ldexp(mean_amount, 1) while (mean_amount * views).abs < 1e-300
        outcomes = mean_amount * views
        squares = outcomes * mean_amount * (1.01 + rng.rand)
        squares = 0.0 if squares > 1e100
        counts[expect_rule(views, outcomes, squares, (rng.rand < 0.5 ? -1 : 1) * (10.0**((rng.rand * 200) - 100)),
                           10.0**((rng.rand * 200) - 100), strength)] += 1
      end
      expect(counts[:built]).to be > 500
    end

    it "the L3 barrier, round 8: a caller's String is read by its bytes alone, never by its own methods" do
      # A frozen String whose own methods raise; a Hash key keeps its hash and eql? (a Hash literal calls them), a name
      # in a list raises on those too.
      loud = lambda do |text, keyed: true|
        string = +text
        names = %i[b encoding == inspect to_s frozen? dup valid_encoding? include? match? bytes force_encoding]
        (keyed ? names : names + %i[hash eql?]).each do |name|
          string.define_singleton_method(name) { |*| raise "String##{name}" }
        end
        string.freeze
      end
      # The round 7 review's case: a frozen plain String name with singleton methods, now copied by String's own b.
      name = loud.call("a")
      named = bandit::Posteriors.beta({ name => [ 1.0, 2.0 ], "b" => [ 2.0, 1.0 ] })
      expect(named.items).to eq(%w[a b])
      expect(named.items.first).not_to equal(name)
      expect(bandit::Posteriors.yes_no({ name => { views: 1, outcomes: 0 } }).parameters).to eq("a" => [ 1.0, 2.0 ])
      # A name in a list, a pinned Hash or a builder's Hash is read the same way.
      other = loud.call("b", keyed: false)
      expect(named.greedy(may_lead: [ other ])).to eq("b")
      expect(named.summary(among: [ other ]).keys).to eq(%w[a b])
      expect(named.rank(key: "k", pinned: { 0 => other }).items.first).to eq("b")
      expect(bandit::Posteriors.beta({ "a" => [ 1.0, 2.0 ] }, views: { loud.call("a") => 3 }).views).to eq("a" => 3.0)
      expect(bandit::Posteriors.yes_no({ "a" => {} }, priors: { loud.call("a") => [ 2, 2 ] }).parameters).to eq("a" => [ 2.0, 2.0 ])
      # A key part's to_s answer, and a key that is such a String, draw as the plain String does.
      subclass = Class.new(String) { %i[b encoding == include?].each { |m| define_method(m) { |*| raise "String##{m}" } } }
      part = Object.new
      part.define_singleton_method(:to_s) { subclass.new("t") }
      expect(bandit::Draw.uniform_for(part, "explore")).to eq(bandit::Draw.uniform_for("t", "explore"))
      expect(bandit::Draw.uniform(part)).to eq(bandit::Draw.uniform("t"))
      expect(named.rank(key: loud.call("k", keyed: false)).items).to eq(named.rank(key: "k").items)
      expect(named.rank(key: [ loud.call("k", keyed: false), 7 ]).items).to eq(named.rank(key: [ "k", "7" ]).items)
    end

    it "the L3 barrier, round 8: Pass reads its status's to_s answer by its bytes, never by its own ==" do
      status = lambda do |text, equal|
        answer = Class.new(String) { define_method(:==) { |_other| equal.nil? ? raise("== raised") : equal } }.new(text)
        holder = Object.new
        holder.define_singleton_method(:to_s) { answer }
        holder
      end
      yes = Class.new { def satisfied_for?(_leader) = true }.new
      no = Class.new { def satisfied_for?(_leader) = false }.new
      # The round 7 review's cases: an == that raises, and an == that answers true for a status that reads "paused".
      expect(bandit::Pass.call(status: status.call("running", nil), hold: no, seed: 1) { { leader: nil } })
        .to eq(leader: nil, ready: false)
      expect(bandit::Pass.call(status: status.call("running", nil), hold: yes, seed: 1) { { leader: "a" } })
        .to eq(leader: "a", ready: true)
      expect(bandit::Pass.call(status: status.call("paused", true), hold: yes, seed: 1) { |held, _rng| { leader: "a", held: held } })
        .to eq(leader: "a", held: false, ready: false)
      expect(bandit::Pass.call(status: status.call("running", false), hold: yes, seed: 1) { { leader: "a" } })
        .to eq(leader: "a", ready: true)
      frozen = +"running"
      frozen.define_singleton_method(:==) { |_other| false }
      frozen.freeze
      expect(bandit::Pass.call(status: frozen, hold: yes, seed: 1) { { leader: "a" } }).to eq(leader: "a", ready: true)
    end

    it "the L3 barrier, round 8: a refusal names a class by Module#name, so building it never raises" do
      opaque = Class.new do
        def self.to_s = raise("class to_s")
        def self.name = raise("class name")
        def self.inspect = raise("class inspect")
        def inspect = raise("inspect")
      end
      stub_const("OpaqueValue", opaque)
      error = Class.new(StandardError) do
        def self.to_s = raise("error class to_s")
        def self.name = raise("error class name")
      end
      stub_const("OpaqueError", error)
      part = Object.new
      part.define_singleton_method(:to_s) { raise OpaqueError, "part" }
      rng = Object.new
      rng.define_singleton_method(:rand) { |*| raise OpaqueError, "rng" }
      silent = Class.new(StandardError) { def message = raise("message") }
      quiet = Object.new
      quiet.define_singleton_method(:to_s) { raise silent }
      unfinished = Class.new(Numeric) { def real? = true; def finite? = true; def to_f = raise(NotImplementedError, "to_f") }.new
      {
        /item "a"'s tally must be a Hash or nil \(got OpaqueValue\)/ => -> { bandit::Posteriors.yes_no({ "a" => OpaqueValue.new }) },
        /a part's to_s raised OpaqueError: part \(got / => -> { set.rank(key: [ part ]) },
        /rng.rand raised OpaqueError: rng/ => -> { set.values(rng: rng) },
        /a part's to_s raised an anonymous class \(got / => -> { bandit::Draw.uniform(quiet) },
        /must be a pair \[alpha, beta\] \(got an object of OpaqueValue\)/ => -> { bandit::Posteriors.beta({ "a" => OpaqueValue.new }) },
        /weights must be a Hash of key => weight \(got OpaqueValue\)/ => -> { bandit::Draw.weighted(OpaqueValue.new, 0.5) },
        /views must be a finite number/ => -> { bandit::Posteriors.yes_no({ "a" => { views: unfinished } }) }
      }.each do |words, call|
        expect(&call).to raise_error(ArgumentError, words)
      end
    end

    it "the L3 barrier, round 8: a refusal's words never meet in incompatible encodings" do
      wide = Object.new
      wide.define_singleton_method(:inspect) { "é".encode("UTF-16LE") }
      latin = Class.new(StandardError) { def message = "\xE9".dup.force_encoding("ISO-8859-1") }
      part = Object.new
      part.define_singleton_method(:to_s) { raise latin }
      [ -> { bandit::Posteriors.beta({ "é" => wide }) },
        -> { bandit::Posteriors.yes_no({ "é" => { views: wide } }) },
        -> { bandit::Draw.uniform_for(part) },
        -> { bandit::Posteriors.beta({ "é".encode("ISO-8859-1") => [ 1.0, wide ] }) } ].each do |call|
        expect(&call).to raise_error(ArgumentError) { |error| expect(error.message).to be_valid_encoding }
      end
    end

    # The L3 barrier's ninth round (01 D36). A Hash keyed by item names (a builder's views or priors, the given
    # uniforms' priority and draw) is read by its pairs, each key mapped to its item by the one lookup of a name, into a
    # Hash keyed by the set's own Strings; no value is looked up in the caller's Hash by a key of the library's, so a
    # Hash that compares its keys by identity reads exactly as a plain one.
    it "the L3 barrier, round 9: a Hash compared by identity, keyed by the item Hash's own Strings, reads as a plain one" do
      pairs = { +"a" => [ 2.0, 8.0 ], +"b" => [ 8.0, 2.0 ], +"c" => [ 5.0, 5.0 ] }
      own = pairs.keys
      identity = ->(keys, values) { keys.zip(values).each_with_object({}.compare_by_identity) { |(k, v), h| h[k] = v } }
      plain = ->(values) { %w[a b c].zip(values).to_h }
      readings = lambda do |hash|
        views = hash.call([ 10.0, 20.0, 0.0 ])
        set = bandit::Posteriors.beta(pairs, views: views)
        amount = ->(mean) { bandit::Prior.amount(mean: mean, variance: 2.0, strength: 3.0) }
        tallies = { "a" => { views: 4, outcomes: 1 }, "b" => { views: 6, outcomes: 5 }, "c" => nil }
        uniforms = ->(explore) { { explore: explore, priority: hash.call([ 0.6, 0.2, 0.9 ]), draw: hash.call([ 0.99, 0.5, 0.1 ]) } }
        [ set.views, set.pick(rng: Random.new(1), min_views: 5.0).to_h,
          bandit::Posteriors.yes_no(tallies, priors: hash.call([ [ 1, 1 ], [ 2, 3 ], [ 4, 1 ] ])).parameters,
          bandit::Posteriors.amounts({ "a" => { views: 2, outcomes: 3, outcome_squares: 5 }, "b" => nil, "c" => nil },
                                     priors: hash.call([ amount.call(1.0), amount.call(2.0), amount.call(-1.0) ])).parameters,
          *[ 0.1, 0.4, 0.9 ].flat_map do |explore|
            [ set.rank(uniforms: uniforms.call(explore), shuffle: 0.25, explore: 0.25).to_h,
              set.pick(uniforms: uniforms.call(explore), shuffle: 0.25, explore: 0.25, min_views: 5.0).to_h ]
          end ]
      end
      expected = readings.call(plain)
      expect(expected[0]).to eq("a" => 10.0, "b" => 20.0, "c" => 0.0)
      expect(expected[1][:item]).not_to be_nil
      expect(readings.call(->(values) { identity.call(own, values) })).to eq(expected)
      # Keyed by other Strings of the same bytes (as the frozen literals of this file are), it reads the same too.
      expect(readings.call(->(values) { identity.call(%w[a b c].map { |name| +name }, values) })).to eq(expected)
      # Two keys that name one item, which an identity Hash can hold, are refused, never read as either.
      twice = {}.compare_by_identity
      twice[+"a"] = 1.0
      twice[+"a"] = 2.0
      expect(twice.size).to eq(2)
      set = bandit::Posteriors.beta(pairs)
      [ -> { bandit::Posteriors.beta({ "a" => [ 1.0, 1.0 ] }, views: twice) },
        -> { bandit::Posteriors.yes_no({ "a" => nil }, priors: twice.transform_values { [ 1, 1 ] }) },
        -> { set.rank(uniforms: { explore: 0.9, priority: twice.merge(own[1] => 0.5, own[2] => 0.5), draw: {} }) },
        -> { set.rank(uniforms: { explore: 0.9, priority: { "a" => 0.1, "b" => 0.2, "c" => 0.3 },
                                  draw: twice.merge(own[1] => 0.5, own[2] => 0.5) }) } ].each do |call|
        expect(&call).to raise_error(ArgumentError, /names the item "a" twice/)
      end
      # The given uniforms still read no entry for a name that is not an item; a builder's Hash still refuses one.
      strangers = { explore: 0.9, priority: { "a" => 0.6, "b" => 0.2, "c" => 0.9, "z" => "no number" },
                    draw: { "a" => 0.99, "b" => 0.5, "c" => 0.1, "z" => 2.0 }.compare_by_identity }
      expect(set.rank(uniforms: strangers).to_h)
        .to eq(set.rank(uniforms: { explore: 0.9, priority: { "a" => 0.6, "b" => 0.2, "c" => 0.9 },
                                    draw: { "a" => 0.99, "b" => 0.5, "c" => 0.1 } }).to_h)
      expect { bandit::Posteriors.beta(pairs, views: { "z" => 1.0 }.compare_by_identity) }
        .to raise_error(ArgumentError, /views: "z" is not an item/)
    end

    # The moments and the amounts rule's gap are formed exactly from the Floats (an Integer and a power of 2 each) and
    # rounded once: s^2 = (q n - o^2) / n^2 and m - mu0 = (o - mu0 n) / n, so the cancellation between two close sums
    # cannot change them. The references are the rule (01 GB3) in Rational arithmetic on the same Floats.
    it "the L3 barrier, round 9: the moments and the gap are exact, so no cancellation changes the amounts rule" do
      # The round 8 review's three cases: the amounts 1e8 and 1e8 + 2 (s^2 = 1; round 8 read 2), the amounts 1e8,
      # 1e8 + 2 and 1e8 + 4 (s^2 = 8/3; round 8 read 2), and a gap of 1/3 less its Float (round 8 read 2.5e-151).
      cases = [ [ 2.0, 200_000_002.0, 20_000_000_400_000_004.0, 1e8, 1e-300, 1e-300 ],
                [ 3.0, 300_000_006.0, 30_000_001_200_000_020.0, 1e8, 1e-300, 1e-300 ],
                [ 3.0, 1.0, 1.0 / 3, 1.0 / 3, 1e-300, 1.0 ] ]
      cases.each do |views, outcomes, squares, mean, variance, strength|
        label = [ views, outcomes, squares, mean, variance, strength ].inspect
        exact_mean, exact_sd, = exact_rule(views, outcomes, squares, mean, variance, strength)
        pair = bandit::Posteriors.amounts({ "a" => { views: views, outcomes: outcomes, outcome_squares: squares } },
                                          priors: bandit::Prior.amount(mean: mean, variance: variance, strength: strength))
                                 .parameters["a"]
        expect((pair[0] - exact_mean).abs).to be <= 1e-12 * exact_mean.abs, label
        expect((pair[1] - exact_sd).abs).to be <= 1e-12 * exact_sd, "#{label}: sd #{pair[1]} vs #{exact_sd}"
        expect(expect_rule(views, outcomes, squares, mean, variance, strength)).to eq(:built)
      end
      expect(exact_rule(*cases[2])[1]).to be_within(1e-21).of(4.006e-18)
      # The moments themselves, as Prior.pooled_amount takes them: s^2 = 1 and 8/3 (the Float nearest it).
      expect(bandit::Prior.pooled_amount(group: { views: 2.0, outcomes: 200_000_002.0, outcome_squares: 20_000_000_400_000_004.0 },
                                         strength: 1.0).variance).to eq(1.0)
      expect(bandit::Prior.pooled_amount(group: { views: 3.0, outcomes: 300_000_006.0, outcome_squares: 30_000_001_200_000_020.0 },
                                         strength: 1.0).variance).to eq(8.0 / 3)
      # Weighted sums cancel alike: the weighted amounts 1e8 and 1e8 + 2 at weights 1 (s^2 = 1), on a Kish size of 2.
      weighted = { views: 2.0, outcomes: 0.0, outcome_squares: 0.0, weighted_views: 2.0, weighted_outcomes: 200_000_002.0,
                   weighted_outcome_squares: 20_000_000_400_000_004.0, weighted_views_squared: 2.0 }
      pair = bandit::Posteriors.amounts({ "a" => weighted }, priors: bandit::Prior.amount(mean: 1e8, variance: 1e-300, strength: 1e-300))
                               .parameters["a"]
      expect(pair[1]).to be_within(1e-12 * Math.sqrt(0.5)).of(exact_rule(*cases[0])[1])
      # Random tallies of close amounts, every one at a cancellation of 10 to 15 digits, plain: each within 1e-12.
      rng = Random.new(909)
      300.times do
        views = (2 + rng.rand(6)).to_f
        base = 10.0**(rng.rand * 12)
        amounts = Array.new(views.to_i) { base + rng.rand(4) }
        outcomes = amounts.sum.to_f
        squares = amounts.sum { |amount| amount * amount }.to_f
        expect(expect_rule(views, outcomes, squares, base * (1 + (rng.rand * 1e-12)), 10.0**(-rng.rand * 300),
                           10.0**(-rng.rand * 300))).to eq(:built)
      end
    end

    # The one shrunk mean (Tallies.shrunk) is formed exactly from its numbers, strength * prior mean included, and
    # rounded once to the nearest Float, so it neither underflows a mean a Float can hold nor rounds past the two means
    # it weighs; with both views above 0 it is held between them as they round.
    it "the L3 barrier, round 9: the shrunk mean is exact, rounded once, and inside the two means it weighs" do
      # The round 8 review's 1e-200 case: k mu0 = 1e-400 underflowed, so both means read 0.0.
      expect(bandit::Prior.pooled_mean(views: 0, outcomes: 0, prior_mean: 1e-200, strength: 1e-200)).to eq(1e-200)
      tiny = bandit::Prior.amount(mean: 1e-200, variance: 1.0, strength: 1e-200)
      expect(bandit::Posteriors.amounts({ "a" => nil }, priors: tiny).parameters["a"][0]).to eq(1e-200)
      expect(bandit::Prior.pooled_mean(views: 1e-200, outcomes: 1e-300, prior_mean: 1e-100, strength: 1e-300))
        .to eq((((1e-300.to_r * 1e-100.to_r) + 1e-300.to_r) / (1e-300.to_r + 1e-200.to_r)).round(400).to_f)
      # The review's near-bound draw: its exact mean is 1e100, which round 8 read as 1.0000000000000002e100 and refused.
      near = { "a" => { views: 0.02184099981450727, outcomes: 2.184099981450725e+98, outcome_squares: 0 } }
      bound = bandit::Prior.amount(mean: 1e100, variance: 1.0, strength: 1.2143359318046165)
      expect(bandit::Posteriors.amounts(near, priors: bound).parameters["a"][0]).to eq(1e100)
      # Its search: every draw whose exact mean is at most 1e100 builds, its mean within 1e-15 of the exact one and
      # never past the larger of the two means (round 8 refused 1 039 of 20 000).
      rng = Random.new(919)
      built = 0
      2_000.times do
        views = 10.0**((rng.rand * 2) - 2)
        strength = 10.0**((rng.rand * 4) - 2)
        outcomes = views * 1e100 * (1 - (rng.rand * 1e-15))
        next if outcomes.abs > 1e100

        exact = ((strength.to_r * 1e100.to_r) + outcomes.to_r) / (strength.to_r + views.to_r)
        next if exact > 1e100.to_r

        prior = bandit::Prior.amount(mean: 1e100, variance: 1.0, strength: strength)
        mean = bandit::Posteriors.amounts({ "a" => { views: views, outcomes: outcomes, outcome_squares: 0 } }, priors: prior)
                                 .parameters["a"][0]
        expect((mean.to_r - exact).abs).to be <= exact * 1e-15
        expect(mean).to be <= 1e100
        expect(mean).to be >= outcomes / views
        built += 1
      end
      expect(built).to be > 1_000
      # Rounded once, ties to even, subnormals included: with no prior views the mean is one quotient, which a Float's
      # own division rounds once, and with no views it is the prior mean to the bit.
      rng = Random.new(929)
      float = -> { (rng.rand < 0.5 ? -1 : 1) * (rng.rand < 0.1 ? rng.rand * 2.0**-1022 : 10.0**((rng.rand * 200) - 100)) }
      2_000.times do
        outcomes = float.call
        views = float.call.abs
        if views.positive? && (outcomes / views).finite?
          expect(bandit::Prior.pooled_mean(views: views, outcomes: outcomes, prior_mean: 0.0, strength: 0.0))
            .to eq(outcomes / views)
        end
        prior_mean = float.call
        strength = float.call.abs
        expect(bandit::Prior.pooled_mean(views: 0, outcomes: 0, prior_mean: prior_mean, strength: strength)).to eq(prior_mean)
      end
      # A quotient on the subnormal grid can be a tie, which goes to the even neighbour: 3 and 5 of the smallest Float
      # halved read 2 of it, as a Float's own division reads them.
      smallest = 2.0**-1074
      [ 3, 5, 7, 2**52 + 1 ].each do |count|
        expect(bandit::Prior.pooled_mean(views: 2.0, outcomes: count * smallest, prior_mean: 0.0, strength: 0.0))
          .to eq((count * smallest) / 2.0)
      end
      expect(bandit::Prior.pooled_mean(views: 2.0, outcomes: 3 * smallest, prior_mean: 0.0, strength: 0.0)).to eq(2 * smallest)
      # Prior.pooled shrinks by the same mean, its prior views (alpha + beta) exact: a base of 1 and 2**-53 borrowing
      # nothing reads the mean 1 / (1 + 2**-53), the Float below 1, where alpha + beta as a Float (1.0) would read 1.0.
      expect(bandit::Prior.pooled(group: { views: 0, outcomes: 0 }, strength: 1.0, base: [ 1.0, 2.0**-53 ]))
        .to eq([ 1.0 - (2.0**-53), 2.0**-53 ])
      expect(bandit::Prior.pooled(group: { views: 0, outcomes: 0 }, strength: 1.0, base: [ 1e-300, 1e-200 ]))
        .to eq([ 1e-300, 1e-200 ])
    end

    # A Hash read by the library's Symbol names (a tally's sums, the given uniforms' parts, an other's group and own,
    # a level's strength) is read by its pairs too (Checks.named), a String key by its copy, so one that compares its
    # keys by identity, keyed by String names, reads as a plain one.
    it "the L3 barrier, round 9: a Hash read by Symbol names reads by its pairs, one compared by identity as a plain one" do
      identity = ->(hash) { hash.each_with_object({}.compare_by_identity) { |(key, value), copy| copy[+key.to_s] = value } }
      tally = { views: 10, outcomes: 3 }
      amounts = { views: 4, outcomes: 6, outcome_squares: 11, weighted_views: 5, weighted_outcomes: 7,
                  weighted_outcome_squares: 12, weighted_views_squared: 6 }
      prior = bandit::Prior.amount(mean: 1.0, variance: 2.0, strength: 3.0)
      expect(bandit::Posteriors.yes_no({ "a" => identity.call(tally) }).parameters).to eq("a" => [ 4.0, 8.0 ])
      expect(bandit::Posteriors.amounts({ "a" => identity.call(amounts) }, priors: prior).parameters)
        .to eq(bandit::Posteriors.amounts({ "a" => amounts }, priors: prior).parameters)
      set = bandit::Posteriors.beta({ "a" => [ 2.0, 8.0 ], "b" => [ 8.0, 2.0 ] })
      uniforms = { explore: 0.1, priority: { "a" => 0.3, "b" => 0.2 }, draw: { "a" => 0.9, "b" => 0.1 } }
      expect(set.rank(uniforms: identity.call(uniforms), shuffle: 0.5).to_h).to eq(set.rank(uniforms: uniforms, shuffle: 0.5).to_h)
      others = [ { group: { views: 20, outcomes: 5 }, own: { views: 4, outcomes: 1 } } ]
      expect(bandit::Prior.pooled_new(others: others.map(&identity))).to eq(bandit::Prior.pooled_new(others: others))
      levels = [ { views: 100, outcomes: 10, strength: 20.0 }, { views: 30, outcomes: 4, strength: 10.0 } ]
      expect(bandit::Prior.chain(levels: levels.map(&identity), root_mean: 0.1))
        .to eq(bandit::Prior.chain(levels: levels, root_mean: 0.1))
      expect(bandit::Decay.fold([ identity.call(age: 0, views: 2, outcomes: 1) ], half_life: 1.0))
        .to eq(bandit::Decay.fold([ { age: 0, views: 2, outcomes: 1 } ], half_life: 1.0))
      # The Symbol key is read before its String, as before; two Strings of one name are refused, never read as either.
      expect(bandit::Posteriors.yes_no({ "a" => { views: 10, "views" => 99, outcomes: 3 } }).parameters).to eq("a" => [ 4.0, 8.0 ])
      twice = {}.compare_by_identity
      twice[+"views"] = 10
      twice[+"views"] = 3
      expect { bandit::Posteriors.yes_no({ "a" => twice }) }.to raise_error(ArgumentError, /item "a"'s tally names views twice/)
    end
  end
end
