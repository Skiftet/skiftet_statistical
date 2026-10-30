# frozen_string_literal: true

# SL2: one phase rule for any posterior. `from_summary` runs the same paused / won / warm-up / automatic-win / shifting
# rule as `call`, with SL1's shares and floors, on statistics computed elsewhere. Two checks:
# - on every 0.2.0 fixture (and one per phase beside them), `from_summary` fed `call`'s own Beta statistics gives
#   `call`'s weights, phase, winner, result_changed and leader exactly;
# - on summaries written by hand, the rule's answer worked out by hand from the 0.2.0 rule (Config defaults: warm-up
#   300, floor 0.10, max_step 0.15, win at P(best) >= 0.95 with expected loss <= 0.0025, won split 0.95 / 0.05,
#   result_changed below P(best) 0.5).
module FromSummaryFixtures
  DRAWS = 500

  # The inputs of the 0.2.0 pinned fixtures (spec/bandit/allocator_spec.rb), copied unchanged.
  PINNED = {
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

  # One fixture per phase the pinned ones leave out.
  PHASES = {
    paused_with_previous: { arms: %i[control a], tallies: { control: { exposures: 1_000, conversions: 100 },
                                                           a: { exposures: 1_000, conversions: 160 } },
                            status: "paused", winner_arm: :a, previous_weights: { control: 0.3, a: 0.7 }, seed: 1 },
    paused_without_previous: { arms: %i[control a b], tallies: { control: { exposures: 10, conversions: 1 } },
                               status: "paused", winner_arm: nil, previous_weights: nil, seed: 2 },
    warmup: { arms: %i[control a], tallies: { control: { exposures: 299, conversions: 30 },
                                             a: { exposures: 5_000, conversions: 2_500 } },
              status: "running", winner_arm: nil, previous_weights: { control: 0.5, a: 0.5 }, seed: 3 },
    automatic_win: { arms: %i[control a], tallies: { control: { exposures: 300, conversions: 100 },
                                                    a: { exposures: 300, conversions: 200 } },
                     status: "running", winner_arm: nil, previous_weights: { control: 0.5, a: 0.5 }, seed: 4 },
    won_holding: { arms: %i[control a], tallies: { control: { exposures: 3_000, conversions: 1_200 },
                                                  a: { exposures: 3_000, conversions: 1_500 } },
                   status: "won", winner_arm: "a", previous_weights: { control: 0.05, a: 0.95 }, seed: 5 },
    stored_winner_not_an_arm: { arms: %i[control a], tallies: { control: { exposures: 1_000, conversions: 100 },
                                                               a: { exposures: 1_000, conversions: 110 } },
                                status: "won", winner_arm: :b, previous_weights: { control: 0.5, a: 0.5 }, seed: 6 },
    single_arm: { arms: %i[control], tallies: { control: { exposures: 1_000, conversions: 500 } },
                  status: "running", winner_arm: nil, previous_weights: nil, seed: 7 },
    no_warmup_config: { arms: %i[control a], tallies: {}, status: "running", winner_arm: nil,
                        previous_weights: { control: 0.25, a: 0.75 },
                        config: SkiftetStatistical::Bandit::Allocator::Config.new(warmup_exposures: 0, floor: 0.2),
                        seed: 8 }
  }.freeze

  ALL = PINNED.merge(PHASES).freeze
end

RSpec.describe SkiftetStatistical::Bandit::Allocator, ".from_summary" do
  def call_on(fixture, **options)
    described_class.call(**fixture.except(:seed), rng: Random.new(fixture[:seed]), draws: FromSummaryFixtures::DRAWS,
                         **options)
  end

  # `call`'s own Beta statistics as a summary. The phase rule reads P(best) and expected loss; `mean` is any value
  # inside the interval (its midpoint here), as the rule never reads it.
  def beta_summary(stats)
    stats.transform_values do |arm|
      { p_best: arm[:p_best], expected_loss: arm[:expected_loss], mean: (arm[:lower] + arm[:upper]) / 2.0,
        lower: arm[:lower], upper: arm[:upper] }
    end
  end

  def from_call(fixture, called, **options)
    described_class.from_summary(arms: fixture[:arms], summary: beta_summary(called[:stats]),
                                 exposures: called[:stats].transform_values { |arm| arm[:exposures] },
                                 status: fixture[:status], winner_arm: fixture[:winner_arm],
                                 previous_weights: fixture[:previous_weights],
                                 **fixture.slice(:hold_satisfied, :config), **options)
  end

  def decision(result)
    result.slice(:weights, :phase, :winner, :result_changed, :leader)
  end

  def floors_for(arms)
    arms.to_h { |arm| [ arm, arm == :control ? 0.05 : 0.02 ] }
  end

  describe "the same rule as call" do
    FromSummaryFixtures::ALL.each do |name, fixture|
      it "SL2 gives call's weights, phase, winner, result_changed and leader for call's own summary: #{name}" do
        called = call_on(fixture)

        expect(decision(from_call(fixture, called))).to eq(decision(called))
      end

      it "SL2 gives call's decision with the per-arm floors of SL1: #{name}" do
        floors = floors_for(fixture[:arms])
        called = call_on(fixture, arm_floors: floors)

        expect(decision(from_call(fixture, called, arm_floors: floors))).to eq(decision(called))
      end

      next if fixture[:arms].size < 2

      it "SL2 gives call's decision with a fixed share of SL1: #{name}" do
        called = call_on(fixture, fixed_shares: { control: 0.2 })

        expect(decision(from_call(fixture, called, fixed_shares: { control: 0.2 }))).to eq(decision(called))
      end
    end
  end

  describe "hand-worked summaries" do
    def arm_summary(p_best, expected_loss)
      { p_best: p_best, expected_loss: expected_loss, mean: 0.11, lower: 0.09, upper: 0.13 }
    end

    def decide(summary:, exposures: nil, status: "running", winner_arm: nil, previous_weights: nil, **options)
      exposures ||= summary.keys.to_h { |arm| [ arm, 1_000 ] }
      described_class.from_summary(arms: summary.keys, summary: summary, exposures: exposures, status: status,
                                   winner_arm: winner_arm, previous_weights: previous_weights, **options)
    end

    def expect_weights(actual, expected)
      expect(actual.keys).to eq(expected.keys)
      expected.each { |arm, share| expect(actual[arm]).to be_within(1e-12).of(share), arm.inspect }
    end

    let(:even) { { control: 0.5, a: 0.5 } }

    it "SL2 answers call's result shape" do
      result = decide(summary: { control: arm_summary(0.3, 0.02), a: arm_summary(0.7, 0.005) }, previous_weights: even)

      expect(result.keys).to eq(%i[weights phase stats winner result_changed leader])
    end

    it "SL2 shifts by the step limit toward P(best)" do
      # Target 0.3 / 0.7 (both above the floor 0.10); the largest move 0.2 is above max_step 0.15, so lambda = 0.75:
      # control 0.5 + 0.75 * (0.3 - 0.5) = 0.35, a 0.65.
      result = decide(summary: { control: arm_summary(0.3, 0.02), a: arm_summary(0.7, 0.005) }, previous_weights: even)

      expect(result).to include(phase: :shifting, winner: nil, result_changed: false, leader: nil)
      expect_weights(result[:weights], { control: 0.35, a: 0.65 })
    end

    it "SL2 moves four arms from 0.25 each toward 0.70 / 0.10 / 0.10 / 0.10 to exactly 0.40 / 0.20 / 0.20 / 0.20" do
      # Largest move 0.45, lambda = 0.15 / 0.45 = 1/3: 0.25 + 0.45 / 3 = 0.40, 0.25 - 0.15 / 3 = 0.20.
      summary = { control: arm_summary(0.70, 0.001), a: arm_summary(0.10, 0.03), b: arm_summary(0.10, 0.03),
                  c: arm_summary(0.10, 0.03) }
      result = decide(summary: summary, previous_weights: summary.keys.to_h { |arm| [ arm, 0.25 ] })

      expect(result[:phase]).to eq(:shifting)
      expect_weights(result[:weights], { control: 0.40, a: 0.20, b: 0.20, c: 0.20 })
    end

    it "SL2 crowns at P(best) >= 0.95 with expected loss <= 0.0025, and jumps to the won split" do
      result = decide(summary: { control: arm_summary(0.04, 0.03), a: arm_summary(0.96, 0.002) }, previous_weights: even)

      expect(result).to include(phase: :won, winner: :a, leader: :a, result_changed: false)
      expect(result[:weights]).to eq(control: 0.05, a: 0.95)
    end

    it "SL2 does not crown a likely-best arm whose expected loss is above 0.0025" do
      # Target 0.03 -> floor 0.10, a 0.90; from 0.5 / 0.5 the largest move 0.4 gives lambda 0.375: 0.35 / 0.65.
      result = decide(summary: { control: arm_summary(0.03, 0.03), a: arm_summary(0.97, 0.003) }, previous_weights: even)

      expect(result).to include(phase: :shifting, winner: nil, leader: nil)
      expect_weights(result[:weights], { control: 0.35, a: 0.65 })
    end

    it "SL2 keeps a test that never ends uncrowned pass after pass, the leader still reported" do
      # hold_satisfied: false at every pass. Target 0.04 -> floor: 0.10 / 0.90. Pass 1 from 0.5: lambda 0.15 / 0.4,
      # 0.35 / 0.65; pass 2: lambda 0.15 / 0.25, 0.20 / 0.80; pass 3: the move 0.10 <= 0.15, 0.10 / 0.90; pass 4 stays.
      summary = { control: arm_summary(0.04, 0.03), a: arm_summary(0.96, 0.002) }
      weights = even
      [ [ 0.35, 0.65 ], [ 0.20, 0.80 ], [ 0.10, 0.90 ], [ 0.10, 0.90 ] ].each do |control, a|
        result = decide(summary: summary, previous_weights: weights, hold_satisfied: false)

        expect(result).to include(phase: :shifting, winner: nil, leader: :a)
        expect_weights(result[:weights], { control: control, a: a })
        weights = result[:weights]
      end
    end

    it "SL2 splits equally in warm-up, read from the exposures" do
      result = decide(summary: { control: arm_summary(0.01, 0.05), a: arm_summary(0.99, 0.0) },
                      exposures: { control: 299, a: 5_000 }, previous_weights: even)

      expect(result).to include(phase: :warmup, winner: nil, leader: nil)
      expect(result[:weights]).to eq(control: 0.5, a: 0.5)
    end

    it "SL2 takes a per-call warm-up of 0 as Config.new(warmup_exposures: 0)" do
      # No exposures, but no warm-up: shifting from 0.5 / 0.5 toward 0.3 / 0.7, as above: 0.35 / 0.65.
      result = decide(summary: { control: arm_summary(0.3, 0.02), a: arm_summary(0.7, 0.005) },
                      exposures: { control: 0, a: 0 }, previous_weights: even,
                      config: described_class::Config.new(warmup_exposures: 0))

      expect(result[:phase]).to eq(:shifting)
      expect_weights(result[:weights], { control: 0.35, a: 0.65 })
    end

    it "SL2 keeps the previous weights while paused, and splits equally without them" do
      summary = { control: arm_summary(0.01, 0.05), a: arm_summary(0.99, 0.0) }

      expect(decide(summary: summary, status: "paused", previous_weights: { control: 0.3, a: 0.7 }))
        .to include(phase: :paused, weights: { control: 0.3, a: 0.7 }, winner: nil, leader: nil)
      expect(decide(summary: summary, status: "paused")).to include(phase: :paused, weights: { control: 0.5, a: 0.5 })
    end

    it "SL2 keeps a stored win, flagging result_changed when the winner's P(best) falls below 0.5" do
      fallen = decide(summary: { control: arm_summary(0.6, 0.0), a: arm_summary(0.4, 0.01) }, status: "won",
                      winner_arm: :a, previous_weights: { control: 0.05, a: 0.95 })
      holding = decide(summary: { control: arm_summary(0.4, 0.01), a: arm_summary(0.6, 0.0) }, status: "won",
                       winner_arm: :a, previous_weights: { control: 0.05, a: 0.95 })

      expect(fallen).to include(phase: :won, winner: :a, result_changed: true, weights: { control: 0.05, a: 0.95 })
      expect(holding).to include(phase: :won, winner: :a, result_changed: false)
    end

    it "SL2 holds each arm's own floor while shifting" do
      # Floors control 0.05, a 0.02, b 0.02. Target: control 0 < 0.05 and b 0.01 < 0.02 are set to their floors, a
      # takes the rest, 1 - 0.05 - 0.02 = 0.93. No previous share is under its floor, and the largest move from
      # 0.06 / 0.90 / 0.04 is 0.03 <= 0.15, so the step reaches the target: 0.05 / 0.93 / 0.02.
      summary = { control: arm_summary(0.0, 0.04), a: arm_summary(0.99, 0.0001), b: arm_summary(0.01, 0.03) }
      result = decide(summary: summary, previous_weights: { control: 0.06, a: 0.90, b: 0.04 }, hold_satisfied: false,
                      arm_floors: { control: 0.05, a: 0.02, b: 0.02 })

      expect(result).to include(phase: :shifting, winner: nil, leader: :a)
      expect_weights(result[:weights], { control: 0.05, a: 0.93, b: 0.02 })
    end

    it "SL2 holds a fixed share in every phase (two arms: the other arm takes the rest)" do
      leading = { control: arm_summary(0.04, 0.03), a: arm_summary(0.96, 0.002) }
      fixed = { fixed_shares: { control: 0.2 } }
      results = [
        decide(summary: leading, exposures: { control: 10, a: 10 }, previous_weights: even, **fixed),
        decide(summary: { control: arm_summary(0.3, 0.02), a: arm_summary(0.7, 0.005) },
               previous_weights: { control: 0.2, a: 0.8 }, **fixed),
        decide(summary: leading, previous_weights: { control: 0.2, a: 0.8 }, hold_satisfied: false, **fixed),
        decide(summary: leading, previous_weights: { control: 0.2, a: 0.8 }, **fixed),
        decide(summary: leading, status: "won", winner_arm: :a, previous_weights: { control: 0.2, a: 0.8 }, **fixed),
        decide(summary: leading, status: "paused", **fixed)
      ]

      results.each { |result| expect_weights(result[:weights], { control: 0.2, a: 0.8 }) }
    end
  end
end
