# frozen_string_literal: true

# SL1: per-arm shares on Allocator.call. `fixed_shares: nil, arm_floors: nil` keep every 0.2.0 output bit for bit
# (SL14). An arm named in `arm_floors` holds its own floor in place of Config#floor; an arm named in `fixed_shares`
# holds exactly its share; both in every phase. Every expected share is worked out by hand in its comment from the 0.2.0
# rule (lib/skiftet_statistical/bandit/allocator.rb): a target share below an arm's floor is set to that floor and the
# other arms are scaled in proportion to share the rest; the convex step from the previous weights moves no arm more
# than max_step (0.15). Where only one arm is left free, every proportional rule gives it the same rest.
module ArmSharesFixtures
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

  # The pinned fixtures whose phase is :shifting (the won split is not floored by the scalar floor).
  SHIFTING = %i[shifting leader_held kish guardrails four_arms_below_floor].freeze
end

RSpec.describe SkiftetStatistical::Bandit::Allocator, "per-arm shares" do
  def tally(exposures, conversions)
    { exposures: exposures, conversions: conversions }
  end

  def allocate(arms:, tallies:, status: "running", winner_arm: nil, previous_weights: nil, seed: 20_261_030,
               draws: 2_000, **options)
    described_class.call(arms: arms, tallies: tallies, status: status, winner_arm: winner_arm,
                         previous_weights: previous_weights, rng: Random.new(seed), draws: draws, **options)
  end

  def pinned(name, **options)
    fixture = ArmSharesFixtures::PINNED.fetch(name)
    described_class.call(**fixture.except(:seed), rng: Random.new(fixture[:seed]), draws: ArmSharesFixtures::DRAWS,
                         **options)
  end

  def expect_weights(actual, expected)
    expect(actual.keys).to eq(expected.keys)
    expected.each { |arm, share| expect(actual[arm]).to be_within(1e-12).of(share), arm.inspect }
  end

  def expect_floors(weights, floors)
    expect(weights.values.sum).to be_within(1e-12).of(1.0)
    floors.each { |arm, floor| expect(weights.fetch(arm)).to be >= floor - 1e-12, "#{arm} #{weights[arm]} < #{floor}" }
  end

  describe "nil keywords" do
    it "SL1 SL14 with fixed_shares: nil and arm_floors: nil, every 0.2.0 fixture gives its 0.2.0 output bit for bit" do
      ArmSharesFixtures::PINNED.each_key do |name|
        expect(pinned(name, fixed_shares: nil, arm_floors: nil)).to eq(pinned(name)), name.to_s
      end
    end
  end

  describe "arm_floors:" do
    it "SL1 floors equal to Config#floor for every arm give the scalar floor's result on every shifting fixture" do
      ArmSharesFixtures::SHIFTING.each do |name|
        arms = ArmSharesFixtures::PINNED.fetch(name)[:arms]
        floored = pinned(name, arm_floors: arms.to_h { |arm| [ arm, 0.10 ] })
        plain = pinned(name)

        expect(floored.except(:weights)).to eq(plain.except(:weights)), name.to_s
        expect_weights(floored[:weights], plain[:weights])
      end
    end

    it "SL1 an arm's own floor replaces Config#floor for that arm" do
      # control 350 of 1 000 against a's 600 of 1 000: the means are 0.25 apart, about 11.6 standard deviations, so
      # control's P(best) is 0 in 20 000 draws and a leads (P(best) 1, expected loss 0); the hold keeps it uncrowned.
      # Floored at 0.05, control's target 0 is set to 0.05 and a takes the rest, 0.95; from 0.10 / 0.90 the largest
      # move is 0.05 <= 0.15, so the step reaches 0.05 / 0.95. Under Config#floor 0.10 the same call stays 0.10 / 0.90.
      tallies = { control: tally(1_000, 350), a: tally(1_000, 600) }
      previous = { control: 0.10, a: 0.90 }
      floored = allocate(arms: %i[control a], tallies: tallies, previous_weights: previous, hold_satisfied: false,
                         draws: 20_000, arm_floors: { control: 0.05, a: 0.02 })
      plain = allocate(arms: %i[control a], tallies: tallies, previous_weights: previous, hold_satisfied: false,
                       draws: 20_000)

      expect(floored).to include(phase: :shifting, winner: nil, leader: :a)
      expect_weights(floored[:weights], { control: 0.05, a: 0.95 })
      expect_weights(plain[:weights], { control: 0.10, a: 0.90 })
    end

    context "with a reference arm floored at 0.05 beside four arms at 0.02" do
      let(:arms) { %i[control a b c d] }
      let(:floors) { { control: 0.05, a: 0.02, b: 0.02, c: 0.02, d: 0.02 } }
      # a converts at 0.20 and the others at 0.10 of 1 000 each: 0.1 apart, about 6.3 standard deviations, so a takes
      # every draw (P(best) 1, expected loss 0) and the others' P(best) is 0, under every floor.
      let(:tallies) do
        { control: tally(1_000, 100), a: tally(1_000, 200), b: tally(1_000, 100), c: tally(1_000, 100),
          d: tally(1_000, 100) }
      end
      # The target: control 0.05, b, c and d 0.02, a the rest, 1 - 0.05 - 3 * 0.02 = 0.89.
      let(:settled) { { control: 0.05, a: 0.89, b: 0.02, c: 0.02, d: 0.02 } }

      it "SL1 keeps every floor while shifting, pass after pass, and settles at 0.05 / 0.89 / 0.02 / 0.02 / 0.02" do
        # From 0.2 each, a's move to 0.89 is the largest every pass (every arm's distance shrinks by the same factor),
        # so a moves 0.15 a pass: 0.35, 0.50, 0.65, 0.80, then 0.09 <= 0.15 reaches the target on pass 5; pass 6 stays.
        weights = arms.to_h { |arm| [ arm, 0.2 ] }
        6.times do |pass|
          result = allocate(arms: arms, tallies: tallies, previous_weights: weights, hold_satisfied: false, seed: pass,
                            arm_floors: floors)

          expect(result).to include(phase: :shifting, winner: nil, leader: :a)
          expect_floors(result[:weights], floors)
          arms.each { |arm| expect((result[:weights][arm] - weights[arm]).abs).to be <= 0.15 + 1e-12 }
          weights = result[:weights]
        end

        expect_weights(weights, settled)
      end

      it "SL1 keeps every floor in warm-up, the automatic win, a stored win and a pause" do
        # Warm-up: 0.2 each, above every floor. A win: the won split gives the four others 0.05 / 4 = 0.0125, under
        # each floor, so each is set to its floor and a takes the rest: the settled shares. A pause keeps its weights.
        warm = allocate(arms: arms, tallies: arms.to_h { |arm| [ arm, tally(200, 20) ] }, arm_floors: floors)
        crowned = allocate(arms: arms, tallies: tallies, previous_weights: settled, arm_floors: floors)
        stored = allocate(arms: arms, tallies: tallies, status: "won", winner_arm: :a, previous_weights: settled,
                          arm_floors: floors)
        paused = allocate(arms: arms, tallies: tallies, status: "paused", previous_weights: settled, arm_floors: floors)

        expect(warm[:phase]).to eq(:warmup)
        expect_weights(warm[:weights], arms.to_h { |arm| [ arm, 0.2 ] })
        expect(crowned).to include(phase: :won, winner: :a, leader: :a)
        expect_weights(crowned[:weights], settled)
        expect(stored).to include(phase: :won, winner: :a)
        expect_weights(stored[:weights], settled)
        expect(paused[:phase]).to eq(:paused)
        expect_weights(paused[:weights], settled)
      end
    end

    it "SL1 lifts the equal split onto a floor above it, in warm-up and in a pause without previous weights" do
      # The equal split 1/3 leaves control under its 0.5: control is set to 0.5, and a and b, equal, share the other
      # 0.5: 0.25 each.
      floors = { control: 0.5, a: 0.02, b: 0.02 }
      warm = allocate(arms: %i[control a b], tallies: {}, arm_floors: floors)
      paused = allocate(arms: %i[control a b], tallies: {}, status: "paused", arm_floors: floors)

      expect(warm[:phase]).to eq(:warmup)
      expect(paused[:phase]).to eq(:paused)
      [ warm, paused ].each { |result| expect_weights(result[:weights], { control: 0.5, a: 0.25, b: 0.25 }) }
    end
  end

  describe "fixed_shares:" do
    it "SL1 holds a fixed share exactly in every phase, two arms (the other takes the rest, 1 - 0.2 = 0.8)" do
      fixed = { fixed_shares: { control: 0.2 } }
      held = { control: 0.2, a: 0.8 }
      results = {
        warmup: allocate(arms: %i[control a], tallies: { control: tally(100, 10), a: tally(100, 20) }, **fixed),
        shifting: allocate(arms: %i[control a], tallies: { control: tally(1_000, 500), a: tally(1_000, 533) },
                           previous_weights: held, **fixed),
        held_leader: allocate(arms: %i[control a], tallies: { control: tally(1_000, 350), a: tally(1_000, 600) },
                              previous_weights: held, hold_satisfied: false, **fixed),
        automatic_win: allocate(arms: %i[control a], tallies: { control: tally(300, 100), a: tally(300, 200) },
                                previous_weights: held, **fixed),
        stored_win: allocate(arms: %i[control a], tallies: { control: tally(400, 100), a: tally(400, 140) },
                             status: "won", winner_arm: :a, previous_weights: held, **fixed),
        paused_with_previous: allocate(arms: %i[control a], tallies: {}, status: "paused", previous_weights: held,
                                       **fixed),
        paused_without_previous: allocate(arms: %i[control a], tallies: {}, status: "paused", **fixed)
      }

      results.each do |phase, result|
        expect(result[:weights].keys).to eq(%i[control a]), phase.to_s
        held.each { |arm, share| expect(result[:weights][arm]).to be_within(1e-12).of(share), "#{phase} #{arm}" }
      end
    end

    it "SL1 splits the rest equally around a fixed share in warm-up and in a pause without previous weights" do
      # Three arms, control fixed at 0.2: a and b, equal, share 0.8: 0.4 each.
      fixed = { fixed_shares: { control: 0.2 } }
      warm = allocate(arms: %i[control a b], tallies: {}, **fixed)
      paused = allocate(arms: %i[control a b], tallies: {}, status: "paused", **fixed)

      expect(warm[:phase]).to eq(:warmup)
      [ warm, paused ].each { |result| expect_weights(result[:weights], { control: 0.2, a: 0.4, b: 0.4 }) }
    end

    it "SL1 holds a fixed share exactly while shifting, the free arms at least Config#floor and within max_step" do
      arms = %i[control a b]
      tallies = { control: tally(1_000, 300), a: tally(1_000, 330), b: tally(1_000, 280) }
      weights = { control: 0.2, a: 0.4, b: 0.4 }

      10.times do |pass|
        result = allocate(arms: arms, tallies: tallies, previous_weights: weights, hold_satisfied: false, seed: pass,
                          fixed_shares: { control: 0.2 })

        expect(result[:phase]).to eq(:shifting)
        expect(result[:weights][:control]).to be_within(1e-12).of(0.2)
        expect(result[:weights].values.sum).to be_within(1e-12).of(1.0)
        %i[a b].each do |arm|
          expect(result[:weights][arm]).to be >= 0.10 - 1e-12
          expect((result[:weights][arm] - weights[arm]).abs).to be <= 0.15 + 1e-12
        end
        weights = result[:weights]
      end
    end
  end
end
