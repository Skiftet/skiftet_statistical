# frozen_string_literal: true

module BanditAllocatorPins
  DRAWS = 500

  # Seeded outputs at 500 draws, as f1's Experiments::Allocator gave them before the rule moved here (f1's own pins
  # of the same rule are spec/services/statistics_golden_draws_spec.rb). Floats compare at 1e-9 relative, so a
  # libm that rounds a last digit differently still passes while an added, dropped or reordered draw (which moves
  # every later draw, JointBeta) does not; every other value, every Hash's key order and every Array's order exact.
  FIXTURES = {
    shifting: { arms: %i[control a], tallies: { control: { exposures: 1_200, conversions: 120 },
                                               a: { exposures: 1_250, conversions: 150 } },
                status: "running", winner_arm: nil, previous_weights: { control: 0.5, a: 0.5 }, seed: 102 },
    leader_held: { arms: %i[control a], tallies: { control: { exposures: 5_000, conversions: 500 },
                                                  a: { exposures: 5_000, conversions: 650 } },
                   status: "running", winner_arm: nil, previous_weights: { control: 0.5, a: 0.5 },
                   hold_satisfied: false, seed: 110 },
    kish: { arms: %i[control a],
            tallies: { control: { exposures: 800, conversions: 80, weighted_exposures: 1_600.0,
                                  weighted_conversions: 150.0, weighted_exposures_squared: 4_000.0 },
                       a: { exposures: 800, conversions: 96, weighted_exposures: 1_450.0,
                            weighted_conversions: 140.0, weighted_exposures_squared: 2_000.0 } },
            status: "running", winner_arm: nil, previous_weights: { control: 0.5, a: 0.5 }, seed: 112 },
    guardrails: { arms: %i[control a],
                  tallies: { control: { exposures: 600, conversions: 60, guardrails: { unsubscribe: 3 } },
                             a: { exposures: 650, conversions: 70,
                                  guardrails: { "unsubscribe" => { conversions: 5 } } } },
                  status: "running", winner_arm: nil, previous_weights: { control: 0.5, a: 0.5 }, seed: 113 },
    four_arms_below_floor: { arms: %i[control a b c],
                             tallies: { control: { exposures: 700, conversions: 70 },
                                        a: { exposures: 700, conversions: 84 },
                                        b: { exposures: 700, conversions: 56 },
                                        c: { exposures: 700, conversions: 63 } },
                             status: "running", winner_arm: nil,
                             previous_weights: { control: 0.05, a: 0.8, b: 0.1, c: 0.05 }, seed: 115 },
    won_result_changed: { arms: %i[control a], tallies: { control: { exposures: 2_000, conversions: 260 },
                                                         a: { exposures: 2_000, conversions: 200 } },
                          status: "won", winner_arm: :a, previous_weights: { control: 0.05, a: 0.95 }, seed: 105 }
  }.freeze

  OUTPUTS = {
    shifting: {
      weights: { control: 0.35, a: 0.65 }, phase: :shifting,
      stats: {
        control: { exposures: 1200, conversions: 120, rate: 0.1, weighted_rate: 0.1, lower: 0.08454080459361754,
          upper: 0.1193895447520584, p_best: 0.058, expected_loss: 0.020645077600422623, guardrails: {} },
        a: { exposures: 1250, conversions: 150, rate: 0.12, weighted_rate: 0.12, lower: 0.10425589449876138,
          upper: 0.14233505559738507, p_best: 0.942, expected_loss: 0.0003593718797802569, guardrails: {} }
      },
      winner: nil, result_changed: false, leader: nil
    },
    leader_held: {
      weights: { control: 0.35, a: 0.65 }, phase: :shifting,
      stats: {
        control: { exposures: 5000, conversions: 500, rate: 0.1, weighted_rate: 0.1, lower: 0.09238748087691571,
          upper: 0.1083341504218604, p_best: 0.0, expected_loss: 0.030244599547538317, guardrails: {} },
        a: { exposures: 5000, conversions: 650, rate: 0.13, weighted_rate: 0.13, lower: 0.12129672026490251,
          upper: 0.140284159819571, p_best: 1.0, expected_loss: 0.0, guardrails: {} }
      },
      winner: nil, result_changed: false, leader: :a
    },
    kish: {
      weights: { control: 0.432, a: 0.568 }, phase: :shifting,
      stats: {
        control: { exposures: 800, conversions: 80, rate: 0.1, weighted_rate: 0.09375, lower: 0.0743510212526234,
          upper: 0.11619120745151382, p_best: 0.432, expected_loss: 0.007550943225422474, guardrails: {} },
        a: { exposures: 800, conversions: 96, rate: 0.12, weighted_rate: 0.09655172413793103, lower: 0.0778833870290523,
          upper: 0.11695655600967411, p_best: 0.568, expected_loss: 0.004845089376744599, guardrails: {} }
      },
      winner: nil, result_changed: false, leader: nil
    },
    guardrails: {
      weights: { control: 0.3500000000000001, a: 0.65 }, phase: :shifting,
      stats: {
        control: { exposures: 600, conversions: 60, rate: 0.1, weighted_rate: 0.1, lower: 0.07942169747005404,
          upper: 0.1254897263120441, p_best: 0.334, expected_loss: 0.010853475183882896, guardrails: { "unsubscribe" => { conversions: 3, rate: 0.005 } } },
        a: { exposures: 650, conversions: 70, rate: 0.1076923076923077, weighted_rate: 0.1076923076923077, lower: 0.08696515793348407,
          upper: 0.13371458807856304, p_best: 0.666, expected_loss: 0.0035401884907117166, guardrails: { "unsubscribe" => { conversions: 5, rate: 0.007692307692307693 } } }
      },
      winner: nil, result_changed: false, leader: nil
    },
    four_arms_below_floor: {
      weights: { control: 0.1, a: 0.7000000000000001, b: 0.1, c: 0.1 }, phase: :shifting,
      stats: {
        control: { exposures: 700, conversions: 70, rate: 0.1, weighted_rate: 0.1, lower: 0.0800483079392037,
          upper: 0.12396906936943496, p_best: 0.106, expected_loss: 0.021160469976213944, guardrails: {} },
        a: { exposures: 700, conversions: 84, rate: 0.12, weighted_rate: 0.12, lower: 0.09834866525284755,
          upper: 0.14428288799612296, p_best: 0.864, expected_loss: 0.0011760997337069777, guardrails: {} },
        b: { exposures: 700, conversions: 56, rate: 0.08, weighted_rate: 0.08, lower: 0.06227943520151638,
          upper: 0.10149204293243044, p_best: 0.004, expected_loss: 0.041094221086365686, guardrails: {} },
        c: { exposures: 700, conversions: 63, rate: 0.09, weighted_rate: 0.09, lower: 0.06953106455711311,
          upper: 0.1168115510874947, p_best: 0.026, expected_loss: 0.030271470176411804, guardrails: {} }
      },
      winner: nil, result_changed: false, leader: nil
    },
    won_result_changed: {
      weights: { control: 0.05, a: 0.95 }, phase: :won,
      stats: {
        control: { exposures: 2000, conversions: 260, rate: 0.13, weighted_rate: 0.13, lower: 0.11578393229577505,
          upper: 0.14501430239159782, p_best: 0.998, expected_loss: 3.6943227535577993e-06, guardrails: {} },
        a: { exposures: 2000, conversions: 200, rate: 0.1, weighted_rate: 0.1, lower: 0.08773775424858009,
          upper: 0.11400957728083824, p_best: 0.002, expected_loss: 0.02982661404357294, guardrails: {} }
      },
      winner: :a, result_changed: true, leader: nil
    }
  }.freeze

  module_function

  # Floats within 1e-9 relative of the pin (a pinned 0.0 stays 0.0); every other value of the same class and equal;
  # Hash keys in the same order.
  def mismatches(actual, expected, path = "")
    case expected
    when Float
      close = actual.is_a?(Float) && (expected.zero? ? actual.zero? : (actual - expected).abs <= 1e-9 * expected.abs)
      close ? [] : [ "#{path}: #{actual.inspect} is not #{expected.inspect}" ]
    when Hash
      return [ "#{path}: keys #{actual.inspect} are not #{expected.keys.inspect}" ] unless actual.is_a?(Hash) && actual.keys == expected.keys

      expected.flat_map { |key, value| mismatches(actual[key], value, "#{path}[#{key.inspect}]") }
    else
      actual.instance_of?(expected.class) && actual == expected ? [] : [ "#{path}: #{actual.inspect} is not #{expected.inspect}" ]
    end
  end
end

RSpec.describe SkiftetStatistical::Bandit::Allocator do
  def tally(exposures, conversions, **extra)
    { exposures: exposures, conversions: conversions, **extra }
  end

  def allocate(arms:, tallies:, status: "running", winner_arm: nil, previous_weights: nil, seed: 20_261_003,
               draws: 20_000, **options)
    described_class.call(arms: arms, tallies: tallies, status: status, winner_arm: winner_arm,
                         previous_weights: previous_weights, rng: Random.new(seed), draws: draws, **options)
  end

  def expect_valid_weights(weights, arms)
    expect(weights.keys).to match_array(arms)
    expect(weights.values.sum).to be_within(1e-9).of(1.0)
  end

  # An RNG that always answers 0.5: every arm with the same posterior then draws the same value (a tie).
  let(:constant_rng) { Class.new { def rand(*) = 0.5 }.new }

  describe "pinned draws" do
    BanditAllocatorPins::FIXTURES.each do |name, fixture|
      it "gives the pinned result for #{name}" do
        result = described_class.call(**fixture.except(:seed), rng: Random.new(fixture[:seed]),
                                      draws: BanditAllocatorPins::DRAWS)

        expect(BanditAllocatorPins.mismatches(result, BanditAllocatorPins::OUTPUTS.fetch(name))).to eq([])
      end
    end

    it "moves every pin with one extra draw before the joint draws" do
      fixture = BanditAllocatorPins::FIXTURES.fetch(:shifting)
      rng = Random.new(fixture[:seed])
      rng.rand
      result = described_class.call(**fixture.except(:seed), rng: rng, draws: BanditAllocatorPins::DRAWS)

      expect(BanditAllocatorPins.mismatches(result, BanditAllocatorPins::OUTPUTS.fetch(:shifting))).not_to eq([])
    end
  end

  describe "Config" do
    let(:config_class) { described_class::Config }

    it "defaults to f1's settings" do
      expect(config_class.new.to_h).to eq(
        warmup_exposures: 300, floor: 0.10, max_step: 0.15, win_p_best: 0.95, win_loss: 0.0025, won_share: 0.95,
        won_rest: 0.05, result_changed_below: 0.5, credible_mass: 0.95, prior_alpha: 1.0, prior_beta: 1.0
      )
      expect(config_class.new.max_arms).to eq(10)
    end

    it "takes any setting by name, and equals another with the same settings" do
      config = config_class.new(floor: 0.2, warmup_exposures: 100)

      expect(config.floor).to eq(0.2)
      expect(config.max_arms).to eq(5)
      expect(config.max_step).to eq(0.15)
      expect(config).to eq(config_class.new(warmup_exposures: 100, floor: 0.2))
      expect(config).not_to eq(config_class.new)
      expect(config).to be_frozen
    end

    it "refuses an unknown setting, a setting that is not a finite number, and a floor outside (0, 1]" do
      expect { config_class.new(flor: 0.1) }.to raise_error(ArgumentError, /flor/)
      expect { config_class.new(max_step: "0.15") }.to raise_error(ArgumentError, /max_step/)
      expect { config_class.new(win_loss: Float::NAN) }.to raise_error(ArgumentError, /win_loss/)
      expect { config_class.new(floor: 0) }.to raise_error(ArgumentError, /floor/)
      expect { config_class.new(floor: 1.5) }.to raise_error(ArgumentError, /floor/)
    end

    it "is refused when the config is not a Config" do
      expect { allocate(arms: %i[control a], tallies: {}, config: { floor: 0.1 }) }.to raise_error(ArgumentError, /Config/)
    end

    it "decides by its settings: a later warm-up, a higher floor, a smaller step" do
      tallies = { control: tally(1_000, 500), a: tally(1_000, 533) }
      previous = { control: 0.5, a: 0.5 }

      warm = allocate(arms: %i[control a], tallies: tallies, config: config_class.new(warmup_exposures: 1_001))
      floored = allocate(arms: %i[control a], tallies: tallies, previous_weights: { control: 0.25, a: 0.75 },
                         config: config_class.new(floor: 0.25))
      small = allocate(arms: %i[control a], tallies: tallies, previous_weights: previous,
                       config: config_class.new(max_step: 0.05))

      expect(warm[:phase]).to eq(:warmup)
      expect(floored[:weights]).to eq(control: 0.25, a: 0.75)
      expect(small[:weights][:control]).to be_within(1e-9).of(0.45)
      expect(small[:weights][:a]).to be_within(1e-9).of(0.55)
    end

    it "builds the posterior on its prior" do
      tallies = { control: tally(300, 0), a: tally(300, 0) }
      flat = allocate(arms: %i[control a], tallies: tallies, previous_weights: { control: 0.5, a: 0.5 })
      heavy = allocate(arms: %i[control a], tallies: tallies, previous_weights: { control: 0.5, a: 0.5 },
                       config: config_class.new(prior_alpha: 50.0, prior_beta: 50.0))

      # Beta(1, 301): 97.5 % quantile 0.012; Beta(50, 350): mean 0.125, 2.5 % quantile 0.094.
      expect(flat[:stats][:control][:upper]).to be < 0.02
      expect(heavy[:stats][:control][:lower]).to be > 0.08
    end

    it "crowns with won_share and won_rest, and judges the win with win_p_best and win_loss" do
      tallies = { control: tally(1_000, 500), a: tally(1_000, 533) } # P(best) of a ~ 0.93
      config = config_class.new(win_p_best: 0.9, win_loss: 0.01, won_share: 0.8, won_rest: 0.2)
      result = allocate(arms: %i[control a], tallies: tallies, previous_weights: { control: 0.5, a: 0.5 },
                        config: config)

      expect(result).to include(phase: :won, winner: :a, leader: :a)
      expect(result[:weights]).to eq(control: 0.2, a: 0.8)
    end

    it "flags result_changed below result_changed_below" do
      tallies = { control: tally(1_000, 500), a: tally(1_000, 533) } # P(best) of a ~ 0.93
      at_half = allocate(arms: %i[control a], tallies: tallies, status: "won", winner_arm: :a)
      higher = allocate(arms: %i[control a], tallies: tallies, status: "won", winner_arm: :a,
                        config: config_class.new(result_changed_below: 0.99))

      expect(at_half[:result_changed]).to be(false)
      expect(higher[:result_changed]).to be(true)
    end

    it "takes the interval's mass from credible_mass" do
      tallies = { control: tally(1_000, 500), a: tally(1_000, 500) }
      wide = allocate(arms: %i[control a], tallies: tallies)
      narrow = allocate(arms: %i[control a], tallies: tallies, config: config_class.new(credible_mass: 0.5))

      expect(narrow[:stats][:control][:upper] - narrow[:stats][:control][:lower])
        .to be < (wide[:stats][:control][:upper] - wide[:stats][:control][:lower]) / 2
    end

    it "crowns at P(best) equal to win_p_best and expected loss equal to win_loss (both bounds inclusive)" do
      # Every draw ties: P(best) 0.5 each, expected loss 0.0; the first arm in order wins the tie-break.
      config = config_class.new(win_p_best: 0.5, win_loss: 0.0)
      result = described_class.call(arms: %i[control a], tallies: { control: tally(400, 100), a: tally(400, 100) },
                                    status: "running", winner_arm: nil, rng: constant_rng, draws: 10, config: config)

      expect(result).to include(phase: :won, winner: :control, leader: :control)
    end
  end

  describe ".won_weights" do
    it "gives the winner won_share and shares won_rest equally, keyed by the arms as given" do
      expect(described_class.won_weights(%i[control a], :a)).to eq(control: 0.05, a: 0.95)
      expect(described_class.won_weights(%w[control a b c], "b"))
        .to eq("control" => 0.05 / 3, "a" => 0.05 / 3, "b" => 0.95, "c" => 0.05 / 3)
    end

    it "gives a single arm everything" do
      expect(described_class.won_weights(%i[control], :control)).to eq(control: 1.0)
    end

    it "takes a config" do
      config = described_class::Config.new(won_share: 0.9, won_rest: 0.1)

      expect(described_class.won_weights(%i[control a], :a, config: config)).to eq(control: 0.1, a: 0.9)
    end
  end

  describe "a fixed split" do
    it "is refused before anything is computed, as FixedAllocation (an ArgumentError)" do
      [ "fixed", :fixed ].each do |allocation|
        expect { allocate(arms: [], tallies: nil, allocation: allocation) }
          .to raise_error(described_class::FixedAllocation, /never moves or crowns/)
      end
      expect(described_class::FixedAllocation.ancestors).to include(ArgumentError)
    end
  end

  describe "purity" do
    let(:arms) { %i[control a] }
    let(:tallies) { { control: tally(400, 120), a: tally(400, 140) } }

    it "gives the same output for the same input and the same seed" do
      first = allocate(arms: arms, tallies: tallies, previous_weights: { control: 0.5, a: 0.5 }, seed: 7)
      second = allocate(arms: arms, tallies: tallies, previous_weights: { control: 0.5, a: 0.5 }, seed: 7)

      expect(first).to eq(second)
    end

    it "does not change its input" do
      frozen = tallies.transform_values { |t| t.dup.freeze }.freeze

      expect {
        allocate(arms: arms.dup.freeze, tallies: frozen, previous_weights: { control: 0.5, a: 0.5 }.freeze)
      }.not_to raise_error
    end
  end

  describe "per-arm statistics" do
    it "returns exposures, conversions, both rates, the interval, P(best), expected loss and guardrails per arm" do
      result = allocate(arms: %i[control a], tallies: { control: tally(1_000, 500), a: tally(1_000, 520) },
                        previous_weights: { control: 0.5, a: 0.5 })

      expect(result.keys).to eq(%i[weights phase stats winner result_changed leader])
      %i[control a].each do |arm|
        expect(result[:stats][arm].keys)
          .to eq(%i[exposures conversions rate weighted_rate lower upper p_best expected_loss guardrails])
      end
      expect(result[:stats][:a]).to include(exposures: 1_000, conversions: 520)
    end

    it "reports the raw rate, and nil at zero exposures" do
      result = allocate(arms: %i[control a], tallies: { control: tally(200, 50), a: tally(0, 0) })

      expect(result[:stats][:control][:rate]).to be_within(1e-12).of(0.25)
      expect(result[:stats][:a][:rate]).to be_nil
    end

    it "reads the 95 % equal-tailed interval of Beta(1 + c, 1 + n - c)" do
      # Beta(501, 501): 2.5 % quantile 0.4691, 97.5 % quantile 0.5309.
      result = allocate(arms: %i[control a], tallies: { control: tally(1_000, 500), a: tally(1_000, 500) },
                        previous_weights: { control: 0.5, a: 0.5 })

      expect(result[:stats][:control][:lower]).to be_within(0.003).of(0.4691)
      expect(result[:stats][:control][:upper]).to be_within(0.003).of(0.5309)
    end

    it "shares P(best) over the arms and splits it evenly between identical arms" do
      arms = %i[control a b c]
      result = allocate(arms: arms, tallies: arms.to_h { |arm| [ arm, tally(2_000, 600) ] },
                        previous_weights: arms.to_h { |arm| [ arm, 0.25 ] })

      expect(arms.sum { |arm| result[:stats][arm][:p_best] }).to be_within(1e-9).of(1.0)
      arms.each { |arm| expect(result[:stats][arm][:p_best]).to be_within(0.03).of(0.25) }
    end

    it "splits a tied draw between the tied arms" do
      result = described_class.call(arms: %i[control a], tallies: { control: tally(400, 100), a: tally(400, 100) },
                                    status: "running", winner_arm: nil, rng: constant_rng, draws: 10)

      expect(result[:stats][:control][:p_best]).to eq(0.5)
      expect(result[:stats][:a][:p_best]).to eq(0.5)
    end

    it "reads expected loss as the mean shortfall from the best arm of each draw" do
      result = allocate(arms: %i[control a], tallies: { control: tally(1_000, 350), a: tally(1_000, 600) },
                        previous_weights: { control: 0.5, a: 0.5 })

      expect(result[:stats][:a][:p_best]).to be > 0.999
      expect(result[:stats][:a][:expected_loss]).to be < 0.0005
      expect(result[:stats][:control][:expected_loss]).to be_within(0.01).of(0.25)
    end

    it "reports each guardrail goal's conversions and raw rate, for display only" do
      tallies = {
        control: tally(400, 100, guardrails: { "moved_on" => 200 }),
        a: tally(400, 100, guardrails: { "moved_on" => { conversions: 300 }, unsubscribe: { "conversions" => nil } })
      }
      with = allocate(arms: %i[control a], tallies: tallies, previous_weights: { control: 0.5, a: 0.5 }, draws: 4_000)
      without = allocate(arms: %i[control a], tallies: tallies.transform_values { |t| t.except(:guardrails) },
                         previous_weights: { control: 0.5, a: 0.5 }, draws: 4_000)

      expect(with[:stats][:control][:guardrails]).to eq("moved_on" => { conversions: 200, rate: 0.5 })
      expect(with[:stats][:a][:guardrails])
        .to eq("moved_on" => { conversions: 300, rate: 0.75 }, "unsubscribe" => { conversions: 0, rate: 0.0 })
      expect(with[:weights]).to eq(without[:weights])
      expect(without[:stats][:a][:guardrails]).to eq({})
    end
  end

  describe "rows read back from JSON" do
    it "accepts String arm keys and String tally keys, and answers with Symbol arms" do
      result = allocate(arms: %w[control a],
                        tallies: { "control" => { "exposures" => 1_000, "conversions" => 500 },
                                   "a" => { "exposures" => 1_000, "conversions" => 533 } },
                        previous_weights: { "control" => 0.5, "a" => 0.5 })

      expect(result[:weights].keys).to eq(%i[control a])
      expect(result[:stats][:a][:conversions]).to eq(533)
      expect(result[:weights][:control]).to be_within(1e-9).of(0.35)
    end

    it "keeps a stored winner given as a String" do
      result = allocate(arms: %i[control a], status: "won", winner_arm: "a",
                        tallies: { control: tally(400, 100), a: tally(400, 140) })

      expect(result).to include(phase: :won, winner: :a)
      expect(result[:weights]).to eq(control: 0.05, a: 0.95)
    end

    it "treats a missing tally as no exposures" do
      result = allocate(arms: %i[control a], tallies: { control: tally(500, 50) })

      expect(result[:phase]).to eq(:warmup)
      expect(result[:stats][:a]).to include(exposures: 0, conversions: 0, rate: nil, weighted_rate: nil)
    end
  end

  describe "previous weights" do
    let(:tallies) { { control: tally(1_000, 500), a: tally(1_000, 533) } }

    it "starts from equal weights when the previous row names other arms, or there is none" do
      [ { control: 0.2, b: 0.8 }, nil ].each do |previous|
        result = allocate(arms: %i[control a], tallies: tallies, previous_weights: previous)

        expect(result[:weights][:control]).to be_within(1e-9).of(0.35)
        expect(result[:weights][:a]).to be_within(1e-9).of(0.65)
      end
    end

    def paused_weights(previous)
      allocate(arms: %i[control a], tallies: {}, status: "paused", previous_weights: previous, draws: 200)[:weights]
    end

    it "keeps numeric previous weights" do
      expect(paused_weights("control" => 0.9, "a" => 0.1)).to eq(control: 0.9, a: 0.1)
    end

    it "ignores previous weights that are not a Hash of finite numbers naming exactly the arms" do
      [
        { "control" => { "x" => 1 }, "a" => 0.5 }, { "control" => "0.9", "a" => "0.1" }, { "control" => nil, "a" => 1.0 },
        { control: Float::INFINITY, a: 0.5 }, [ 0.9, 0.1 ], "control"
      ].each do |previous|
        expect(paused_weights(previous)).to eq(control: 0.5, a: 0.5), previous.inspect
      end
    end

    it "lifts a stored row below the floor onto it before stepping" do
      arms = %i[control a b c]
      result = allocate(arms: arms, tallies: arms.to_h { |arm| [ arm, tally(1_000, 500) ] },
                        previous_weights: { control: 0.95, a: 0.05 / 3, b: 0.05 / 3, c: 0.05 / 3 }, draws: 4_000)

      expect(result[:phase]).to eq(:shifting)
      expect(result[:weights].values).to all(be >= 0.10)
      expect(result[:weights].values.sum).to be_within(1e-9).of(1.0)
    end
  end

  describe "paused" do
    it "returns the previous weights unchanged, a winner's 95/5 too" do
      [ { control: 0.3, a: 0.7 }, { control: 0.05, a: 0.95 } ].each do |previous|
        result = allocate(arms: %i[control a], status: "paused", winner_arm: :a, previous_weights: previous,
                          tallies: { control: tally(1_000, 100), a: tally(1_000, 600) })

        expect(result).to include(phase: :paused, winner: nil, leader: nil)
        expect(result[:weights]).to eq(previous)
      end
    end

    it "gives equal weights with no previous row" do
      result = allocate(arms: %i[control a b c], status: "paused", tallies: { control: tally(10, 1) })

      expect(result[:phase]).to eq(:paused)
      expect(result[:weights].values).to all(be_within(1e-12).of(0.25))
    end
  end

  describe "warm-up" do
    it "splits equally while one arm has 299 exposures, however far ahead another arm is" do
      result = allocate(arms: %i[control a], tallies: { control: tally(299, 30), a: tally(5_000, 2_500) })

      expect(result).to include(phase: :warmup, winner: nil, leader: nil)
      expect(result[:weights]).to eq(control: 0.5, a: 0.5)
    end

    it "splits four arms 25 % each" do
      arms = %i[control a b c]
      result = allocate(arms: arms, tallies: { control: tally(400, 40), a: tally(400, 200), b: tally(299, 10),
                                               c: tally(400, 100) })

      expect(result[:phase]).to eq(:warmup)
      expect(result[:weights].values).to all(eq(0.25))
    end

    it "ends when every arm has warmup_exposures" do
      result = allocate(arms: %i[control a], tallies: { control: tally(300, 90), a: tally(300, 92) },
                        previous_weights: { control: 0.5, a: 0.5 })

      expect(result[:phase]).to eq(:shifting)
    end
  end

  describe "the floor" do
    it "raises a losing arm to exactly the floor when one arm dominates without winning" do
      # P(best) of a = 0.93: the target after the floor is exactly 0.10 / 0.90, and from 0.15 / 0.85 the step
      # limit does not bind.
      result = allocate(arms: %i[control a], tallies: { control: tally(1_000, 500), a: tally(1_000, 533) },
                        previous_weights: { control: 0.15, a: 0.85 })

      expect(result).to include(phase: :shifting, winner: nil)
      expect(result[:weights][:control]).to be_within(1e-9).of(0.10)
      expect(result[:weights][:a]).to be_within(1e-9).of(0.90)
    end

    it "keeps a hopeless third arm at the floor and scales the two leaders down" do
      arms = %i[control a b]
      result = allocate(arms: arms, tallies: { control: tally(1_000, 500), a: tally(1_000, 500), b: tally(1_000, 100) },
                        previous_weights: { control: 0.45, a: 0.45, b: 0.10 })

      expect_valid_weights(result[:weights], arms)
      expect(result[:weights][:b]).to be_within(1e-9).of(0.10)
      expect(result[:weights][:control]).to be_within(0.01).of(0.45)
    end

    it "puts all ten arms at exactly 10 % when one arm takes nearly every draw" do
      arms = (0..9).map { |index| :"arm#{index}" }
      tallies = arms.to_h { |arm| [ arm, tally(1_000, 300) ] }.merge(arm0: tally(1_000, 330))

      result = allocate(arms: arms, tallies: tallies, previous_weights: arms.to_h { |arm| [ arm, 0.1 ] }, draws: 4_000)

      expect(result[:phase]).to eq(:shifting)
      expect(result[:weights].values.sum).to be_within(1e-12).of(1.0)
      expect(result[:weights].values).to all(be_within(1e-12).of(0.1))
    end

    it "never puts an arm under the floor before a win, nor moves one more than max_step, pass after pass" do
      arms = %i[control a b c]
      tallies = { control: tally(1_000, 300), a: tally(1_000, 330), b: tally(1_000, 280), c: tally(1_000, 250) }
      weights = arms.to_h { |arm| [ arm, 0.25 ] }

      12.times do |pass|
        result = allocate(arms: arms, tallies: tallies, previous_weights: weights, seed: pass, draws: 4_000)
        break if result[:phase] == :won

        expect_valid_weights(result[:weights], arms)
        arms.each do |arm|
          expect(result[:weights][arm]).to be >= 0.10 - 1e-9
          expect((result[:weights][arm] - weights[arm]).abs).to be <= 0.15 + 1e-9
        end
        weights = result[:weights]
      end
    end

    it "refuses more arms than can all hold the floor" do
      arms = (1..11).map { |n| :"arm#{n}" }

      expect { allocate(arms: arms, tallies: {}) }.to raise_error(ArgumentError, /at most 10 arms/)
    end
  end

  describe "the step limit" do
    it "moves four arms from 0.25 each toward [0.70, 0.10, 0.10, 0.10] to exactly [0.40, 0.20, 0.20, 0.20]" do
      arms = %i[control a b c]
      tallies = { control: tally(1_000, 500), a: tally(1_000, 540), b: tally(1_000, 500), c: tally(1_000, 500) }
      result = allocate(arms: arms, tallies: tallies, previous_weights: arms.to_h { |arm| [ arm, 0.25 ] })

      expect(result[:phase]).to eq(:shifting)
      expect(result[:weights][:a]).to be_within(1e-9).of(0.40)
      %i[control b c].each { |arm| expect(result[:weights][arm]).to be_within(1e-9).of(0.20) }
    end

    it "takes the convex step lambda = max_step / max |target - prev| with two arms" do
      result = allocate(arms: %i[control a], tallies: { control: tally(1_000, 500), a: tally(1_000, 533) },
                        previous_weights: { control: 0.5, a: 0.5 })

      expect(result[:weights][:control]).to be_within(1e-9).of(0.35)
      expect(result[:weights][:a]).to be_within(1e-9).of(0.65)
    end
  end

  describe "weighted counts" do
    it "does not crown an arm whose raw lead comes from a surge it was favoured in" do
      tallies = {
        control: tally(1_000, 200, weighted_exposures: 2_000.0, weighted_conversions: 400.0),
        a: tally(1_000, 300, weighted_exposures: 2_000.0, weighted_conversions: 400.0)
      }
      result = allocate(arms: %i[control a], tallies: tallies, previous_weights: { control: 0.5, a: 0.5 })

      expect(result).to include(phase: :shifting, winner: nil)
      expect(result[:stats][:a][:p_best]).to be_within(0.05).of(0.5)
      expect(result[:stats][:a][:rate]).to be_within(1e-12).of(0.30)
      expect(result[:stats][:a][:weighted_rate]).to be_within(1e-12).of(0.20)
    end

    it "crowns the arm the weighted counts favour although the raw rates are equal" do
      tallies = {
        control: tally(1_000, 250, weighted_exposures: 2_000.0, weighted_conversions: 400.0),
        a: tally(1_000, 250, weighted_exposures: 2_000.0, weighted_conversions: 700.0)
      }
      result = allocate(arms: %i[control a], tallies: tallies, previous_weights: { control: 0.5, a: 0.5 })

      expect(result).to include(phase: :won, winner: :a)
    end

    it "rescales the weighted counts to the raw exposure count before the posterior" do
      raw = { control: tally(1_000, 200), a: tally(1_000, 215) }
      weighted = {
        control: tally(1_000, 200, weighted_exposures: 10_000.0, weighted_conversions: 2_000.0),
        a: tally(1_000, 215, weighted_exposures: 10_000.0, weighted_conversions: 2_150.0)
      }
      from_raw = allocate(arms: %i[control a], tallies: raw, previous_weights: { control: 0.5, a: 0.5 })
      from_weighted = allocate(arms: %i[control a], tallies: weighted, previous_weights: { control: 0.5, a: 0.5 })

      expect(from_weighted[:stats][:a][:p_best]).to be < 0.9
      expect(from_weighted[:stats][:a][:p_best]).to be_within(0.02).of(from_raw[:stats][:a][:p_best])
    end

    it "judges on the raw counts when the weighted exposures are zero" do
      result = allocate(arms: %i[control a], draws: 4_000,
                        tallies: { control: tally(0, 0, weighted_exposures: 0.0, weighted_conversions: 0.0),
                                   a: tally(10, 5) })

      expect(result[:stats][:control][:weighted_rate]).to be_nil
      expect(result[:stats][:a][:weighted_rate]).to eq(0.5)
    end

    it "holds an arm drawn under mixed shares to less evidence than its raw count (Kish n)" do
      mixed = tally(1_000, 300, weighted_exposures: 6_000.0, weighted_conversions: 1_800.0)
      steady = tally(1_000, 300, weighted_exposures: 2_000.0, weighted_conversions: 600.0)
      width = ->(result) { result[:stats][:control][:upper] - result[:stats][:control][:lower] }

      raw = allocate(arms: %i[control a], tallies: { control: mixed, a: steady }, draws: 4_000)
      kish = allocate(arms: %i[control a], draws: 4_000,
                      tallies: { control: mixed.merge(weighted_exposures_squared: 52_000.0), a: steady })
      even = allocate(arms: %i[control a], draws: 4_000,
                      tallies: { control: steady.merge(weighted_exposures_squared: 4_000.0), a: steady })
      plain = allocate(arms: %i[control a], draws: 4_000, tallies: { control: steady, a: steady })

      expect(width.call(kish)).to be > width.call(raw) * 1.1
      expect(kish[:stats][:control][:weighted_rate]).to eq(raw[:stats][:control][:weighted_rate])
      expect(width.call(even)).to be_within(1e-12).of(width.call(plain))
    end
  end

  # f1's surge trap (its allocator_simulation_oracle_spec "a traffic surge during a shift"): two arms that convert
  # equally; the allocation favours one 90/10 while the base rate doubles. Raw counts read the surge as the arm's
  # lead; counts weighted by 1 / the share each visit was drawn under do not.
  describe "a traffic surge during a shift" do
    let(:raw) { { control: tally(1_500, 200), a: tally(5_500, 1_000) } }
    let(:weighted) do
      {
        control: raw[:control].merge(weighted_exposures: (1_000 / 0.5) + (500 / 0.1),
                                     weighted_conversions: (100 / 0.5) + (100 / 0.1)),
        a: raw[:a].merge(weighted_exposures: (1_000 / 0.5) + (4_500 / 0.9),
                         weighted_conversions: (100 / 0.5) + (900 / 0.9))
      }
    end

    it "does not crown the favoured arm when the counts are weighted" do
      result = allocate(arms: %i[control a], tallies: weighted, previous_weights: { control: 0.1, a: 0.9 }, seed: 42)

      expect(result).to include(phase: :shifting, winner: nil)
      expect(result[:stats][:a][:p_best]).to be_within(0.1).of(0.5)
      expect(result[:stats][:a][:weighted_rate]).to be_within(1e-9).of(result[:stats][:control][:weighted_rate])
    end

    it "would have crowned it on raw counts (the scenario really is a trap)" do
      result = allocate(arms: %i[control a], tallies: raw, previous_weights: { control: 0.1, a: 0.9 }, seed: 42)

      expect(result[:winner]).to eq(:a)
    end
  end

  describe "the automatic win" do
    it "crowns a clear winner once every arm has warmup_exposures, and jumps to the won split at once" do
      result = allocate(arms: %i[control a], tallies: { control: tally(300, 100), a: tally(300, 200) },
                        previous_weights: { control: 0.5, a: 0.5 })

      expect(result).to include(phase: :won, winner: :a, leader: :a, result_changed: false)
      expect(result[:weights]).to eq(control: 0.05, a: 0.95)
    end

    it "splits won_rest equally among three losing arms" do
      arms = %i[control a b c]
      tallies = { control: tally(400, 100), a: tally(400, 100), b: tally(400, 240), c: tally(400, 100) }
      result = allocate(arms: arms, tallies: tallies, previous_weights: arms.to_h { |arm| [ arm, 0.25 ] })

      expect(result[:winner]).to eq(:b)
      expect(result[:weights][:b]).to eq(0.95)
      %i[control a c].each { |arm| expect(result[:weights][arm]).to eq(0.05 / 3) }
    end

    it "does not crown a coin flip at 5 000 exposures each" do
      result = allocate(arms: %i[control a], tallies: { control: tally(5_000, 2_500), a: tally(5_000, 2_500) },
                        previous_weights: { control: 0.5, a: 0.5 })

      expect(result).to include(phase: :shifting, winner: nil, leader: nil)
    end

    it "does not crown an arm at P(best) 0.94, and does at 0.98" do
      arms = %i[control a b c]
      even = arms.to_h { |arm| [ arm, 0.25 ] }
      short = allocate(arms: arms, previous_weights: even, tallies: { control: tally(1_000, 500), a: tally(1_000, 545),
                                                                      b: tally(1_000, 500), c: tally(1_000, 500) })
      clear = allocate(arms: arms, previous_weights: even, tallies: { control: tally(1_000, 500), a: tally(1_000, 555),
                                                                      b: tally(1_000, 500), c: tally(1_000, 500) })

      expect(short).to include(phase: :shifting, winner: nil)
      expect(clear).to include(phase: :won, winner: :a)
    end

    it "does not crown a likely-best arm whose expected loss is above win_loss" do
      tallies = { control: tally(1_000, 500), a: tally(1_000, 555) }
      result = allocate(arms: %i[control a], tallies: tallies, previous_weights: { control: 0.5, a: 0.5 },
                        config: described_class::Config.new(win_loss: 0.0))

      expect(result[:stats][:a][:p_best]).to be >= 0.95
      expect(result[:stats][:a][:expected_loss]).to be > 0.0
      expect(result).to include(phase: :shifting, winner: nil, leader: nil)
    end

    it "never crowns a single arm" do
      result = allocate(arms: %i[control], tallies: { control: tally(1_000, 500) })

      expect(result).to include(winner: nil, leader: nil)
      expect(result[:weights]).to eq(control: 1.0)
    end
  end

  describe "the hold (hold_satisfied:)" do
    let(:clear) { { control: tally(400, 100), a: tally(400, 250) } }

    def allocate_held(hold_satisfied:, tallies: clear, status: "running", winner_arm: nil)
      allocate(arms: %i[control a], tallies: tallies, status: status, winner_arm: winner_arm, seed: 5, draws: 4_000,
               previous_weights: { control: 0.5, a: 0.5 }, hold_satisfied: hold_satisfied)
    end

    it "wins with the hold satisfied, and names the winner as the leader" do
      result = allocate_held(hold_satisfied: true)

      expect(result).to include(phase: :won, winner: :a, leader: :a)
      expect(result[:weights]).to eq(control: 0.05, a: 0.95)
    end

    it "keeps the step weights while the hold runs, and names the leader" do
      result = allocate_held(hold_satisfied: false)

      expect(result).to include(phase: :shifting, winner: nil, leader: :a)
      expect(result[:weights][:a]).to be_within(1e-9).of(0.65)
    end

    it "gives the same statistics whatever the hold says, for the same seed" do
      expect(allocate_held(hold_satisfied: false)[:stats]).to eq(allocate_held(hold_satisfied: true)[:stats])
    end

    it "is satisfied by default" do
      result = described_class.call(arms: %i[control a], tallies: clear, status: "running", winner_arm: nil,
                                    previous_weights: { control: 0.5, a: 0.5 }, rng: Random.new(5), draws: 4_000)

      expect(result).to include(phase: :won, winner: :a)
    end

    it "names no leader in warm-up, while paused or after the win" do
      warm = { control: tally(299, 10), a: tally(400, 250) }

      expect(allocate_held(hold_satisfied: false, tallies: warm)).to include(phase: :warmup, leader: nil)
      expect(allocate_held(hold_satisfied: false, status: "paused")).to include(phase: :paused, leader: nil)
      expect(allocate_held(hold_satisfied: false, status: "won", winner_arm: "a")).to include(phase: :won, leader: nil)
    end
  end

  describe "after the win" do
    it "keeps the won split and never un-wins, flagging result_changed when the winner falls below P(best) 0.5" do
      result = allocate(arms: %i[control a], status: "won", winner_arm: :a,
                        tallies: { control: tally(3_000, 1_500), a: tally(3_000, 1_200) })

      expect(result).to include(phase: :won, winner: :a, result_changed: true)
      expect(result[:weights]).to eq(control: 0.05, a: 0.95)
      expect(result[:stats][:a][:p_best]).to be < 0.5
    end

    it "does not flag result_changed while the winner is still the better arm" do
      result = allocate(arms: %i[control a], status: "won", winner_arm: :a,
                        tallies: { control: tally(3_000, 1_200), a: tally(3_000, 1_500) })

      expect(result).to include(phase: :won, result_changed: false)
    end

    it "is not a win when the stored winner is not one of the arms" do
      result = allocate(arms: %i[control a], status: "won", winner_arm: :b,
                        tallies: { control: tally(1_000, 100), a: tally(1_000, 110) })

      expect(result[:phase]).to eq(:shifting)
      expect(result[:winner]).to be_nil
    end
  end

  describe "draws" do
    it "refuses a draw count below one" do
      expect { allocate(arms: %i[control a], tallies: {}, draws: 0) }.to raise_error(ArgumentError, /draws/)
    end

    it "requires a draw count" do
      expect {
        described_class.call(arms: %i[control a], tallies: {}, status: "running", winner_arm: nil, rng: Random.new(1))
      }.to raise_error(ArgumentError, /draws/)
    end

    it "takes a whole number in any form Integer() reads" do
      expect(allocate(arms: %i[control a], tallies: {}, draws: "50", seed: 3))
        .to eq(allocate(arms: %i[control a], tallies: {}, draws: 50, seed: 3))
    end
  end

  it "refuses no arms" do
    expect { allocate(arms: [], tallies: {}) }.to raise_error(ArgumentError, /at least one arm/)
  end
end
