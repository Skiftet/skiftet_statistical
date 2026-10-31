# frozen_string_literal: true

# SL15: the pins of Allocator.from_summary (SL2), the one phase rule fed by statistics computed elsewhere. It takes no
# random input, so every pin is a hand-worked value: each fixture's shares are worked out in exact Rationals in its
# comment (SL1's lift and step written out again outside the library) and pinned as the nearest Float. Config
# defaults unless a fixture says otherwise: warm-up 300, floor 0.10, max_step 0.15, win at P(best) >= 0.95 with
# expected loss <= 0.0025, won split 0.95 / 0.05, result_changed below P(best) 0.5. This file ran red before
# `from_summary` existed. A pinned value is never edited.
#
# One tolerance rule (SL15) for every Float compared with a pin: 1e-12 relative, and a pinned 0.0, 1.0 or infinity
# exactly (none of this goes through lgamma). nil, Integers, Symbols, Booleans and every Hash's key order: exact.
#
# Beside the pins: the probe (SL2's "the two cannot drift" on 0.2.0's pinned fixture), a sweep of random calls in
# which both entry points must give the same decision or the same refusal, and the readings 17 leaves to this task
# (no summary yet, key and number readings, the summary's own checks).
module AllocatorFromSummaryPins
  RELATIVE = 1e-12

  FLOORS = { control: 0.05, a: 0.02, b: 0.02 }.freeze

  def self.entry(p_best, expected_loss, mean, lower, upper)
    { p_best: p_best, expected_loss: expected_loss, mean: mean, lower: lower, upper: upper }
  end

  def self.stat(exposures, p_best, expected_loss, mean, lower, upper)
    { exposures: exposures, mean: mean, lower: lower, upper: upper, p_best: p_best, expected_loss: expected_loss }
  end

  def self.empty_stat(exposures)
    stat(exposures, nil, nil, nil, nil, nil)
  end

  FIXTURES = {
    # A cell's pass (no warm-up), its losses in the caller's units, keys as Strings and Symbols. The target is the
    # P(best) 1/10, 3/5, 3/10, above every floor. From 3/10, 2/5, 3/10 the largest move is 1/5 > 0.15, lambda 3/4:
    # control 3/10 - 3/20 = 3/20, a 2/5 + 3/20 = 11/20, b 3/10. No arm reaches P(best) 0.95: no leader. b has no
    # interval and a missing exposure count (read as 0).
    value_shifting: {
      arms: %i[control a b],
      summary: { "control" => entry(0.1, 0.4, 1.2, 0.9, 1.5),
                 a: { "p_best" => 0.6, "expected_loss" => 0.05, "mean" => 1.6, "lower" => 1.3, "upper" => 1.9 },
                 "b" => entry(0.3, 0.2, 1.4, nil, nil) },
      exposures: { control: 120, "a" => 340 },
      status: "running", winner_arm: nil, previous_weights: { "control" => 0.3, a: 0.4, b: 0.3 },
      hold_satisfied: false, config: SkiftetStatistical::Bandit::Allocator::Config.new(warmup_exposures: 0),
      arm_floors: FLOORS
    },
    # A test that never ends (hold_satisfied: false), win_loss read in the caller's units (0.05): a leads (P(best)
    # 0.96, loss 0.03) and is reported, never crowned. The target: control 0.01 < 0.05 is pinned; a and b share 19/20
    # over 99/100: a 152/165, b 19/660 (>= 0.02). From 1/10, 3/5, 3/10 a's move 53/165 is the largest, lambda
    # 99/212: control 1/10 - (1/20)(99/212) = 65/848, a 3/5 + 3/20 = 3/4, b 3/10 - (179/660)(99/212) = 147/848.
    never_ends_leader: {
      arms: %i[control a b],
      summary: { control: entry(0.01, 0.9, 0.2, 0.1, 0.3), a: entry(0.96, 0.03, 1.1, 0.9, 1.3),
                 b: entry(0.03, 0.6, 0.5, 0.3, 0.7) },
      exposures: { control: 50, a: 60, b: 40 },
      status: "running", winner_arm: nil, previous_weights: { control: 0.1, a: 0.6, b: 0.3 }, hold_satisfied: false,
      config: SkiftetStatistical::Bandit::Allocator::Config.new(warmup_exposures: 0, win_loss: 0.05),
      arm_floors: FLOORS
    },
    # No summary yet: warm-up holds although every arm is past 300. Equal targets 1/3 put control under its floor
    # 1/2, so it is pinned there and a and b share 1/2: 1/4 each.
    no_summary_warmup: {
      arms: %i[control a b], summary: nil, exposures: { control: 1_000, a: 1_000, b: 1_000 },
      status: "running", winner_arm: nil, previous_weights: { control: 0.2, a: 0.5, b: 0.3 },
      arm_floors: { control: 0.5 }
    },
    # No summary yet, paused: the pause comes before the held warm-up, and its previous weights hold every floor.
    no_summary_paused: {
      arms: %i[control a], summary: nil, exposures: { control: 1_000, a: 1_000 },
      status: "paused", winner_arm: nil, previous_weights: { control: 0.3, a: 0.7 }
    },
    # No summary yet, a stored win (a String winner): the win comes before the held warm-up, and with no P(best) to
    # read the result has not changed. The won split 1/40, 1/40, 19/20 pins control at 1/20; a and b share 19/20 over
    # 39/40: a 19/780, b 361/390.
    no_summary_won: {
      arms: %i[control a b], summary: nil, exposures: { control: 10, a: 10, b: 10 },
      status: "won", winner_arm: "b", previous_weights: nil, arm_floors: FLOORS
    },
    # A lone free arm beside a fixed share: its P(best) among the free arms is 1 and its loss 0, so past warm-up it
    # leads and is crowned; the won split 1 into 1 - 1/5: a 4/5, control 1/5. The fixed arm is absent from the
    # summary (it is never read) and its statistics are empty.
    lone_free_won: {
      arms: %i[control a], summary: { a: entry(1.0, 0.0, 0.3, 0.2, 0.4) }, exposures: { control: 400, a: 400 },
      status: "running", winner_arm: nil, previous_weights: { control: 0.2, a: 0.8 },
      fixed_shares: { control: 0.2 }
    },
    # A stored win whose winner's P(best) has fallen to 0.4 < 0.5: result_changed. Undeclared, 0.2.0's won split.
    stored_win_changed: {
      arms: %i[control a], summary: { control: entry(0.6, 0.0, 0.12, 0.1, 0.14), a: entry(0.4, 0.01, 0.11, 0.09, 0.13) },
      exposures: { control: 2_000, a: 2_000 }, status: "won", winner_arm: :a,
      previous_weights: { control: 0.05, a: 0.95 }
    }
  }.freeze

  OUTPUTS = {
    value_shifting: {
      weights: { control: 0.15, a: 0.55, b: 0.3 }, phase: :shifting,
      stats: { control: stat(120, 0.1, 0.4, 1.2, 0.9, 1.5), a: stat(340, 0.6, 0.05, 1.6, 1.3, 1.9),
               b: stat(0, 0.3, 0.2, 1.4, nil, nil) },
      winner: nil, result_changed: false, leader: nil
    },
    never_ends_leader: {
      weights: { control: 65.0 / 848, a: 0.75, b: 147.0 / 848 }, phase: :shifting,
      stats: { control: stat(50, 0.01, 0.9, 0.2, 0.1, 0.3), a: stat(60, 0.96, 0.03, 1.1, 0.9, 1.3),
               b: stat(40, 0.03, 0.6, 0.5, 0.3, 0.7) },
      winner: nil, result_changed: false, leader: :a
    },
    no_summary_warmup: {
      weights: { control: 0.5, a: 0.25, b: 0.25 }, phase: :warmup,
      stats: { control: empty_stat(1_000), a: empty_stat(1_000), b: empty_stat(1_000) },
      winner: nil, result_changed: false, leader: nil
    },
    no_summary_paused: {
      weights: { control: 0.3, a: 0.7 }, phase: :paused,
      stats: { control: empty_stat(1_000), a: empty_stat(1_000) },
      winner: nil, result_changed: false, leader: nil
    },
    no_summary_won: {
      weights: { control: 0.05, a: 19.0 / 780, b: 361.0 / 390 }, phase: :won,
      stats: { control: empty_stat(10), a: empty_stat(10), b: empty_stat(10) },
      winner: :b, result_changed: false, leader: nil
    },
    lone_free_won: {
      weights: { control: 0.2, a: 0.8 }, phase: :won,
      stats: { control: empty_stat(400), a: stat(400, 1.0, 0.0, 0.3, 0.2, 0.4) },
      winner: :a, result_changed: false, leader: :a
    },
    stored_win_changed: {
      weights: { control: 0.05, a: 0.95 }, phase: :won,
      stats: { control: stat(2_000, 0.6, 0.0, 0.12, 0.1, 0.14), a: stat(2_000, 0.4, 0.01, 0.11, 0.09, 0.13) },
      winner: :a, result_changed: true, leader: nil
    }
  }.freeze

  # 0.2.0's pinned two-arm fixture (spec/bandit/allocator_spec.rb, :shifting), copied unchanged.
  SHIFTING_020 = {
    arms: %i[control a], tallies: { control: { exposures: 1_200, conversions: 120 },
                                    a: { exposures: 1_250, conversions: 150 } },
    status: "running", winner_arm: nil, previous_weights: { control: 0.5, a: 0.5 }, seed: 102
  }.freeze

  module_function

  # The one rule: a Float within RELATIVE of the pin, relative to it; a pinned 0.0, 1.0 or infinity exactly.
  def close?(actual, expected)
    return false unless actual.is_a?(Float)
    return actual == expected if expected.zero? || expected == 1.0 || expected.infinite?

    (actual - expected).abs <= RELATIVE * expected.abs
  end

  # Floats by #close?; every other value of the same class and equal; Hash keys in the same order.
  def mismatches(actual, expected, path = "")
    case expected
    when Float
      close?(actual, expected) ? [] : [ "#{path}: #{actual.inspect} is not #{expected.inspect}" ]
    when Hash
      unless actual.is_a?(Hash) && actual.keys == expected.keys
        return [ "#{path}: keys #{actual.inspect} are not #{expected.keys.inspect}" ]
      end

      expected.flat_map { |key, value| mismatches(actual[key], value, "#{path}[#{key.inspect}]") }
    else
      actual.instance_of?(expected.class) && actual == expected ? [] : [ "#{path}: #{actual.inspect} is not #{expected.inspect}" ]
    end
  end

  # `call`'s own Beta statistics as a summary: the phase rule reads P(best) and expected loss; the mean is the
  # interval's midpoint (the rule never reads it).
  def beta_summary(stats)
    stats.transform_values do |arm|
      { p_best: arm[:p_best], expected_loss: arm[:expected_loss], mean: (arm[:lower] + arm[:upper]) / 2.0,
        lower: arm[:lower], upper: arm[:upper] }
    end
  end

  ARMS = %i[control a b c d e].freeze

  # One random call of either entry point: 1-6 arms; tallies before and past warm-up; every status; a stored winner
  # that is an arm (Symbol or String), not an arm, or nil; previous weights absent, ordinary, under a floor, or naming
  # other arms; either hold; three settings; no shares, floors alone, or a fixed share with floors. Some of them are
  # refused (a fixed stored winner, a free mass above 1), and both entry points must refuse those alike.
  def random_fixture(rng)
    arms = ARMS.first(rng.rand(1..6))
    {
      arms: arms,
      tallies: arms.to_h do |arm|
        exposures = rng.rand(150..1_500)
        [ arm, { exposures: exposures, conversions: (exposures * rng.rand * 0.4).floor } ]
      end,
      status: %w[running running running paused won][rng.rand(5)],
      winner_arm: random_winner(arms, rng), previous_weights: random_previous(arms, rng),
      hold_satisfied: rng.rand < 0.5, config: random_config(rng), seed: rng.rand(1_000_000)
    }.merge(random_shares(arms, rng))
  end

  def random_winner(arms, rng)
    case rng.rand(4)
    when 0 then nil
    when 1 then :z
    when 2 then arms[rng.rand(arms.size)].to_s
    else arms[rng.rand(arms.size)]
    end
  end

  def random_previous(arms, rng)
    raw = arms.to_h { |arm| [ arm, rng.rand + 0.01 ] }
    raw[arms.first] = 0.001 if rng.rand < 0.25
    total = raw.values.sum
    case rng.rand(4)
    when 0 then nil
    when 1 then raw.transform_values { |share| share / total }.merge(z: 0.0)
    else raw.transform_values { |share| share / total }
    end
  end

  def random_config(rng)
    config = SkiftetStatistical::Bandit::Allocator::Config
    [ config.new, config.new(warmup_exposures: 0), config.new(floor: 0.05) ][rng.rand(3)]
  end

  def random_shares(arms, rng)
    floors = arms.select { rng.rand < 0.5 }.to_h { |arm| [ arm, (0.01 + (rng.rand * 0.11)).round(3) ] }
    case rng.rand(3)
    when 0 then {}
    when 1 then { arm_floors: floors }
    else
      return { arm_floors: floors } if arms.size < 2

      fixed = arms[rng.rand(arms.size)]
      { fixed_shares: { fixed => (0.05 + (rng.rand * 0.35)).round(3) }, arm_floors: floors.except(fixed) }
    end
  end

  # Each arm's summary entry for this P(best), with one expected loss and no mean or interval.
  def p_best_summary(p_best)
    p_best.transform_values { |share| entry(share, 0.01, nil, nil, nil) }
  end

  # P(best) over `arms`, the first above 0 and each of the others 0 half the time (so it is pinned at its floor).
  def random_p_best(arms, rng)
    raw = arms.each_with_index.to_h { |arm, index| [ arm, index.positive? && rng.rand < 0.5 ? 0.0 : rng.rand + 0.01 ] }
    total = raw.values.sum

    raw.transform_values { |share| share / total }
  end
end

RSpec.describe SkiftetStatistical::Bandit::Allocator, ".from_summary pins" do
  def from_summary(fixture, **options)
    described_class.from_summary(**{ previous_weights: nil }.merge(fixture), **options)
  end

  def call_fixture(fixture)
    described_class.call(**fixture.except(:seed), rng: Random.new(fixture[:seed]), draws: 100)
  end

  def decision(result)
    result.slice(:weights, :phase, :winner, :result_changed, :leader)
  end

  # The refusal or nil.
  def refusal
    yield
    nil
  rescue ArgumentError => e
    [ e.class, e.message ]
  end

  describe "pinned results" do
    AllocatorFromSummaryPins::FIXTURES.each do |name, fixture|
      it "SL15 SL2 gives the pinned result for #{name}" do
        expect(AllocatorFromSummaryPins.mismatches(from_summary(fixture), AllocatorFromSummaryPins::OUTPUTS.fetch(name)))
          .to eq([])
      end
    end

    it "SL2 every pinned result sums to 1, holds each fixed share exactly and each free arm at or above its floor" do
      AllocatorFromSummaryPins::FIXTURES.each do |name, fixture|
        result = from_summary(fixture)
        weights = result[:weights]
        fixed = fixture[:fixed_shares] || {}
        floors = fixture[:arm_floors] || {}

        expect((weights.values.sum - 1.0).abs).to be <= AllocatorFromSummaryPins::RELATIVE, name.to_s
        fixed.each { |arm, share| expect(weights.fetch(arm)).to eq(share), "#{name} #{arm}" }
        # Undeclared, a win is 0.2.0's won split, whose won_rest may sit under Config#floor (SL14).
        next if fixed.empty? && floors.empty? && result[:phase] == :won

        (fixture[:arms] - fixed.keys).each do |arm|
          expect(weights.fetch(arm)).to be >= floors.fetch(arm, 0.10), "#{name} #{arm}"
        end
      end
    end

    it "SL2 takes no random input: the same call answers the same result" do
      AllocatorFromSummaryPins::FIXTURES.each_value do |fixture|
        expect(Marshal.dump(from_summary(fixture))).to eq(Marshal.dump(from_summary(fixture)))
      end
    end
  end

  describe "the probe" do
    it "SL2 gives call's weights for call's own Beta summary on 0.2.0's two-arm fixture" do
      fixture = AllocatorFromSummaryPins::SHIFTING_020
      called = described_class.call(**fixture.except(:seed), rng: Random.new(fixture[:seed]), draws: 500)
      result = described_class.from_summary(arms: fixture[:arms],
                                            summary: AllocatorFromSummaryPins.beta_summary(called[:stats]),
                                            exposures: called[:stats].transform_values { |arm| arm[:exposures] },
                                            status: fixture[:status], winner_arm: fixture[:winner_arm],
                                            previous_weights: fixture[:previous_weights])

      expect(result.fetch(:weights) == called.fetch(:weights)).to be(true)
    end
  end

  describe "one rule, two entry points" do
    it "SL2 answers call's exact decision, or call's own refusal, on 400 random calls" do
      rng = Random.new(2_026)
      phases = Hash.new(0)
      refused = 0

      400.times do |index|
        fixture = AllocatorFromSummaryPins.random_fixture(rng)
        exposures = fixture[:tallies].transform_values { |tally| tally[:exposures] }
        inputs = fixture.except(:tallies, :seed).merge(exposures: exposures)
        called = nil
        call_refusal = refusal { called = call_fixture(fixture) }

        if call_refusal
          refused += 1
          expect(refusal { described_class.from_summary(**inputs, summary: nil) }).to eq(call_refusal), "##{index}"
          next
        end

        result = described_class.from_summary(**inputs, summary: AllocatorFromSummaryPins.beta_summary(called[:stats]))
        phases[called[:phase]] += 1
        phases[:leader] += 1 if called[:leader]
        expect(decision(result)).to eq(decision(called)), "##{index} #{fixture.inspect}"
        expect(result[:stats].transform_values { |arm| arm.values_at(:exposures, :p_best, :expected_loss) })
          .to eq(called[:stats].transform_values { |arm| arm.values_at(:exposures, :p_best, :expected_loss) }), "##{index}"
      end

      expect(phases.values_at(:paused, :won, :warmup, :shifting, :leader)).to all(be >= 20)
      expect(refused).to be_between(1, 80)
    end

    it "SL2 refuses what call refuses, with call's own error and message" do
      base = { arms: %i[control a b], status: "running", winner_arm: nil }
      cases = [
        { arm_floors: { c: 0.02 } }, { fixed_shares: { "c" => 0.2 } }, { arm_floors: { control: 0.9 } },
        { fixed_shares: { control: 0.5 }, arm_floors: { a: 0.6 } },
        { arm_floors: { control: 0.05 }, fixed_shares: { control: 0.2 } }, { arm_floors: { control: 0 } },
        { fixed_shares: { control: 1.0 } }, { arm_floors: { control: "0.1" } }, { arm_floors: [ 0.1 ] },
        { status: "won", winner_arm: "control", fixed_shares: { control: 0.2 } },
        { fixed_shares: { control: 0.3, a: 0.3, b: 0.4 } }, { config: :defaults }, { arms: [] },
        { arms: (1..11).map { |index| :"arm#{index}" } }
      ]

      cases.each do |options|
        inputs = base.merge(options)
        called = refusal { described_class.call(**inputs, tallies: {}, rng: Random.new(1), draws: 10) }

        expect(called).not_to be_nil, options.inspect
        expect(refusal { described_class.from_summary(**inputs, summary: nil, exposures: {}) }).to eq(called),
                                                                                               options.inspect
      end
    end
  end

  describe "the step's rounding against a floor" do
    let(:config_class) { described_class::Config }

    # The step reaches its target (lambda 1) when the previous shares sit near it, and there a share stepping down onto
    # its floor can round under it.
    def near_pass(rng)
      floor = [ 0.02, 0.04, 0.05, 0.08, 0.10 ][rng.rand(5)]
      config = config_class.new(floor: floor, warmup_exposures: 0)
      arms = AllocatorFromSummaryPins::ARMS.first(rng.rand(2..5))
      p_best = AllocatorFromSummaryPins.random_p_best(arms, rng)
      target = described_class.lift(p_best, config: config)
      previous = described_class.lift(target.transform_values { |share| share + (rng.rand * 0.1) }, config: config)
      inputs = { arms: arms, summary: AllocatorFromSummaryPins.p_best_summary(p_best), exposures: {}, status: "running",
                 winner_arm: nil, previous_weights: previous, hold_satisfied: false, config: config }

      [ floor, inputs ]
    end

    it "SL1 SL14 a declared floor differs from Config#floor's own step by the clamp alone, on 6 000 random passes" do
      rng = Random.new(2_040)
      rounded = 0

      6_000.times do |index|
        floor, inputs = near_pass(rng)
        plain = described_class.from_summary(**inputs)[:weights]
        declared = described_class.from_summary(**inputs, arm_floors: { inputs[:arms].first => floor })[:weights]
        clamped = plain.transform_values { |share| share < floor ? floor : share }

        expect(Marshal.dump(declared)).to eq(Marshal.dump(clamped)), "##{index} #{inputs.inspect}"
        expect(declared.values.all? { |share| share >= floor }).to be(true), "##{index} #{inputs.inspect}"
        rounded += plain.count { |_arm, share| share < floor }
      end

      expect(rounded).to be >= 50
    end

    it "SL1 holds a fixed share exactly and every free arm at or above its floor, strictly, pass after pass" do
      # Each pass reads the last one's weights as its previous weights, so a share that left its floor comes back to it
      # from a new place; none is ever under, and the sum never drifts past 1e-12.
      rng = Random.new(2_041)
      free = %i[control a b]
      floors = { control: 0.05, a: 0.02, b: 0.02 }
      config = config_class.new(warmup_exposures: 0)
      weights = { control: 0.3, a: 0.3, b: 0.2, c: 0.2 }
      at_floor = 0

      6_000.times do |pass|
        p_best = AllocatorFromSummaryPins.random_p_best(free.shuffle(random: rng), rng)
        weights = described_class.from_summary(arms: free + [ :c ], summary: AllocatorFromSummaryPins.p_best_summary(p_best),
                                               exposures: {}, status: "running", winner_arm: nil, previous_weights: weights,
                                               hold_satisfied: false, config: config, arm_floors: floors,
                                               fixed_shares: { c: 0.2 })[:weights]

        expect(weights[:c]).to eql(0.2), "pass #{pass}"
        expect((weights.values.sum - 1.0).abs).to be <= AllocatorFromSummaryPins::RELATIVE, "pass #{pass}"
        floors.each { |arm, floor| expect(weights[arm] >= floor).to be(true), "pass #{pass} #{arm} #{weights[arm]}" }
        at_floor += floors.count { |arm, floor| weights[arm] == floor }
      end

      expect(at_floor).to be >= 300
    end
  end

  describe "no summary yet" do
    let(:past_warmup) { { control: 5_000, a: 5_000 } }

    def without_summary(**options)
      described_class.from_summary(arms: %i[control a], summary: nil, exposures: past_warmup, status: "running",
                                   winner_arm: nil, **options)
    end

    it "SL2 holds warm-up whatever the exposures say: equal weights undeclared" do
      result = without_summary(previous_weights: { control: 0.2, a: 0.8 })

      expect(result).to include(weights: { control: 0.5, a: 0.5 }, phase: :warmup, winner: nil, leader: nil)
    end

    it "SL2 holds warm-up with no warm-up configured: the split over the free arms through the lift" do
      config = described_class::Config.new(warmup_exposures: 0)
      fixed = without_summary(config: config, fixed_shares: { control: 0.3 })
      floored = without_summary(config: config, arm_floors: { a: 0.6 })

      expect(fixed).to include(weights: { control: 0.3, a: 0.7 }, phase: :warmup)
      expect(floored).to include(weights: { control: 0.4, a: 0.6 }, phase: :warmup)
    end

    it "SL2 never crowns or reports a leader with no summary, whatever the hold" do
      [ true, false ].each do |hold|
        expect(without_summary(hold_satisfied: hold)).to include(phase: :warmup, winner: nil, leader: nil)
      end
    end

    it "SL2 makes call's checks with no summary" do
      expect { without_summary(arm_floors: { z: 0.1 }) }.to raise_error(ArgumentError)
      expect { without_summary(status: "won", winner_arm: :control, fixed_shares: { control: 0.2 }) }
        .to raise_error(ArgumentError)
      expect { without_summary(fixed_shares: { control: 0.5, a: 0.5 }) }
        .to raise_error(described_class::FixedAllocation)
      expect(without_summary(arm_floors: { a: 0.6 })[:weights]).to eq(control: 0.4, a: 0.6)
    end
  end

  describe "readings" do
    def summary_of(control, a)
      { control: AllocatorFromSummaryPins.entry(*control), a: AllocatorFromSummaryPins.entry(*a) }
    end

    def decide(summary, exposures: { control: 1_000, a: 1_000 }, **options)
      described_class.from_summary(arms: %i[control a], summary: summary, exposures: exposures, status: "running",
                                   winner_arm: nil, previous_weights: { control: 0.5, a: 0.5 }, **options)
    end

    let(:plain) { summary_of([ 0.3, 0.02, 0.1, 0.09, 0.11 ], [ 0.7, 0.005, 0.12, 0.1, 0.14 ]) }

    it "SL2 answers every arm by its Symbol, whatever the keys of arms, summary, exposures and previous weights" do
      strings = described_class.from_summary(arms: %w[control a], summary: plain.transform_keys(&:to_s),
                                             exposures: { "control" => 1_000, "a" => 1_000 }, status: "running",
                                             winner_arm: nil, previous_weights: { "control" => 0.5, "a" => 0.5 })

      expect(strings).to eq(decide(plain))
      expect(strings[:weights].keys).to eq(%i[control a])
      expect(strings[:stats].keys).to eq(%i[control a])
    end

    it "SL2 reads Integer and Rational statistics as Floats" do
      exact = { control: { p_best: Rational(3, 10), expected_loss: Rational(1, 50), mean: 0, lower: 0, upper: 1 },
                a: { p_best: Rational(7, 10), expected_loss: Rational(1, 200), mean: 1, lower: 0, upper: 1 } }
      result = decide(exact)

      # The pins' rule compares classes too (0 == 0.0 and Rational(3, 10) == 0.3 in Ruby, so eq would not).
      expect(AllocatorFromSummaryPins.mismatches(result[:stats][:control],
                                                 { exposures: 1_000, mean: 0.0, lower: 0.0, upper: 1.0, p_best: 0.3,
                                                   expected_loss: 0.02 })).to eq([])
      expect(AllocatorFromSummaryPins.mismatches(result[:stats][:a][:p_best], 0.7)).to eq([])
      expect(AllocatorFromSummaryPins.mismatches(result[:weights], { control: 0.35, a: 0.65 })).to eq([])
    end

    it "SL2 reads an arm missing from the exposures as never exposed, as call reads a missing tally" do
      expect(decide(plain, exposures: { control: 1_000 })).to include(phase: :warmup, weights: { control: 0.5, a: 0.5 })
      expect(decide(plain, exposures: { control: 1_000, a: nil })).to include(phase: :warmup)
    end

    it "SL2 never reads a fixed arm's statistics: absent, nil or any P(best), its stats carry nil P(best) and loss" do
      free = { a: AllocatorFromSummaryPins.entry(1.0, 0.0, 0.12, 0.1, 0.14) }
      results = [ free, free.merge(control: nil),
                  free.merge(control: AllocatorFromSummaryPins.entry(0.99, -1.0, nil, nil, nil)),
                  free.merge(control: { p_best: "any", expected_loss: Float::NAN }) ].map do |summary|
        decide(summary, fixed_shares: { control: 0.25 })
      end

      results.each do |result|
        expect(result).to include(weights: { control: 0.25, a: 0.75 }, phase: :won, winner: :a, leader: :a)
        expect(result[:stats][:control]).to include(p_best: nil, expected_loss: nil)
      end
    end

    it "SL2 keeps a fixed arm's interval when it is given" do
      summary = { control: AllocatorFromSummaryPins.entry(nil, nil, 0.1, 0.08, 0.12),
                  a: AllocatorFromSummaryPins.entry(0.5, 0.01, 0.12, 0.1, 0.14) }

      expect(decide(summary, fixed_shares: { control: 0.25 })[:stats][:control])
        .to eq(exposures: 1_000, mean: 0.1, lower: 0.08, upper: 0.12, p_best: nil, expected_loss: nil)
    end

    it "SL2 shares the free mass equally when every free arm's P(best) is 0" do
      # Undeclared, as 0.2.0's lift: a target summing to 0 shares the mass equally, 1/2 each; from 1/2 nothing moves.
      result = decide(summary_of([ 0.0, 0.1, 0.1, 0.09, 0.11 ], [ 0.0, 0.1, 0.12, 0.1, 0.14 ]))

      expect(result).to include(phase: :shifting, weights: { control: 0.5, a: 0.5 }, leader: nil)
    end
  end

  describe "the summary's checks" do
    let(:entry) { AllocatorFromSummaryPins.entry(0.5, 0.01, 0.12, 0.1, 0.14) }

    def decide(summary, exposures: { control: 1_000, a: 1_000 }, **options)
      described_class.from_summary(arms: %i[control a], summary: summary, exposures: exposures, status: "running",
                                   winner_arm: nil, **options)
    end

    # A refusal, labelled, beside the valid call it alters (so it never passes for a missing name).
    def expect_refused(label, &block)
      aggregate_failures(label) { expect(&block).to raise_error(ArgumentError) }
    end

    it "SL2 refuses a summary that is not a Hash of every free arm's statistics" do
      expect(decide({ control: entry, a: entry })[:phase]).to eq(:shifting)

      [ [], "summary", 0.5 ].each { |summary| expect_refused(summary.inspect) { decide(summary) } }
      expect_refused("empty") { decide({}) }
      expect_refused("a free arm absent") { decide({ control: entry }) }
      expect_refused("a free arm nil") { decide({ control: entry, a: nil }) }
      expect_refused("a free arm not a Hash") { decide({ control: entry, a: 0.5 }) }
      expect_refused("a fixed arm not a Hash") { decide({ control: [ 0.5 ], a: entry }, fixed_shares: { control: 0.2 }) }
    end

    it "SL2 refuses a summary key that is no arm of the call, or an arm named twice" do
      expect_refused("not an arm") { decide({ control: entry, a: entry, z: entry }) }
      expect_refused("not a name") { decide({ control: entry, a: entry, 1 => entry }) }
      expect_refused("twice") { decide({ control: entry, a: entry, "a" => entry }) }
    end

    it "SL2 refuses a free arm's P(best) outside [0, 1] or its expected loss below 0, by the one number check" do
      [ nil, Float::NAN, Float::INFINITY, 1.5, -0.1, "0.5", Complex(0.5, 0) ].each do |p_best|
        expect_refused("p_best #{p_best.inspect}") { decide({ control: entry, a: entry.merge(p_best: p_best) }) }
      end
      [ nil, -0.01, -Float::INFINITY, Float::NAN, "0" ].each do |loss|
        expect_refused("loss #{loss.inspect}") { decide({ control: entry, a: entry.merge(expected_loss: loss) }) }
      end
      # The bounds themselves are numbers the rule reads: a at P(best) 1 with loss 0 is crowned.
      expect(decide({ control: entry.merge(p_best: 0, expected_loss: 0.5), a: entry.merge(p_best: 1, expected_loss: 0) }))
        .to include(phase: :won, winner: :a)
    end

    it "SL2 refuses a mean or interval bound that is not a number (nil allowed)" do
      %i[mean lower upper].each do |key|
        [ "0.1", Float::NAN, Float::INFINITY, [ 0.1 ] ].each do |value|
          expect_refused("#{key} #{value.inspect}") { decide({ control: entry, a: entry.merge(key => value) }) }
        end
        expect(decide({ control: entry, a: entry.merge(key => nil) })[:stats][:a][key]).to be_nil
      end
    end

    it "SL2 refuses exposures that are not a Hash" do
      [ nil, [], 1_000 ].each do |exposures|
        expect_refused("exposures #{exposures.inspect}") { decide({ control: entry, a: entry }, exposures: exposures) }
      end
    end

    it "SL15 refuses a statistic that is finite as a number but not as a Float, and reads each bound on the Float" do
      # 10**400 is a finite Integer whose Float is an infinity. A Rational just above 1 has the Float 1.0, inside
      # [0, 1], so it reads as P(best) 1.0 (and a at P(best) 1 with loss 0 is crowned).
      %i[p_best expected_loss mean lower upper].each do |key|
        expect_refused("#{key} 10**400") { decide({ control: entry, a: entry.merge(key => 10**400) }) }
      end
      just_above = Rational((10**400) + 1, 10**400)
      result = decide({ control: entry.merge(p_best: 0, expected_loss: 0.5),
                        a: entry.merge(p_best: just_above, expected_loss: 0) })

      expect(result).to include(phase: :won, winner: :a)
      expect(AllocatorFromSummaryPins.mismatches(result[:stats][:a][:p_best], 1.0)).to eq([])
    end

    it "SL2 reads the exposures by call's Integer(), and refuses a NaN or an infinity with an ArgumentError" do
      # call reads a tally's exposures by Integer(): 1 000.9 is 1 000 and "1000" is 1 000. Integer() raises
      # FloatDomainError (a RangeError) on a NaN or an infinity; from_summary refuses those with an ArgumentError.
      called = described_class.call(arms: %i[control a], tallies: { control: { exposures: 1_000.9 }, a: { exposures: "1000" } },
                                    status: "running", winner_arm: nil, rng: Random.new(1), draws: 10)
      read = decide({ control: entry, a: entry }, exposures: { control: 1_000.9, "a" => "1000" })

      expect(read[:stats].transform_values { |arm| arm[:exposures] }).to eq(control: 1_000, a: 1_000)
      expect(read[:stats].transform_values { |arm| arm[:exposures] })
        .to eq(called[:stats].transform_values { |arm| arm[:exposures] })
      [ Float::NAN, Float::INFINITY, -Float::INFINITY ].each do |count|
        expect { decide({ control: entry, a: entry }, exposures: { control: count, a: 1_000 }) }
          .to raise_error(ArgumentError, /exposures: control/), count.inspect
      end
    end

    it "SL15 refuses with an ArgumentError every exposure Integer() cannot read cleanly, never a TypeError or a RangeError" do
      # Integer() answers an Array, a Hash, a Symbol or nothing it knows with a TypeError, a Complex with an imaginary
      # part with a RangeError and a word with its own ArgumentError. It reads 10**400, a finite Integer, without a word,
      # and the one number check refuses that: its Float is an infinity.
      [ [], {}, :many, Object.new, "many", Complex(1_000, 1), Complex(1_000, 0.0), Float::NAN, Float::INFINITY,
        -Float::INFINITY, 10**400, -(10**400), Rational(10**400, 1), "1#{'0' * 400}" ].each do |count|
        [ { a: count }, { "a" => count } ].each do |exposures|
          expect { decide({ control: entry, a: entry }, exposures: exposures) }
            .to raise_error(ArgumentError, /exposures: a/), exposures.inspect
          expect { decide(nil, exposures: exposures) }.to raise_error(ArgumentError, /exposures: a/), exposures.inspect
        end
      end
    end

    it "SL2 keeps reading every exposure Integer() reads cleanly, a Float's range included" do
      read = decide({ control: entry, a: entry }, exposures: { control: Complex(1_000, 0), a: Rational(2_001, 2) })
      large = decide({ control: entry, a: entry }, exposures: { control: 10**300, a: " 1000 " })

      expect(read[:stats].transform_values { |arm| arm[:exposures] }).to eq(control: 1_000, a: 1_000)
      expect(large[:stats].transform_values { |arm| arm[:exposures] }).to eq(control: 10**300, a: 1_000)
    end
  end
end
