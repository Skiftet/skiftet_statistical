# frozen_string_literal: true

# SL1: per-arm shares on Allocator.call. `fixed_shares: nil, arm_floors: nil` keep every 0.2.0 output bit for bit
# (SL14). An arm named in `arm_floors` holds its own floor in place of Config#floor; an arm named in `fixed_shares`
# holds exactly its share; both in every phase. Every expected share is worked out by hand in its comment from the 0.2.0
# rule (lib/skiftet_statistical/bandit/allocator.rb): a target share below an arm's floor is set to that floor and the
# other arms are scaled in proportion to share the rest; the convex step from the previous weights moves no arm more
# than max_step (0.15). Where only one arm is left free, every proportional rule gives it the same rest.
#
# SL1's lift, used in every comment below: F is the sum of the fixed shares, f_a a free arm's floor (its own, else
# Config#floor 0.10). Over the unpinned free arms U, m = (1 - F) - (the pinned arms' floors) and s_a = m * t_a / (sum
# over U of t); every arm with s_a < f_a is pinned at f_a and the rest is shared again, until none is below.
# P(best) with fixed shares is taken among the free arms in each of 0.2.0's joint draws (every arm still drawn).
#
# A refusal is checked beside a call that answers, so it fails now on the missing keyword and later only on the check.
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

  # A refusal, labelled, with RSpec's own message kept (so a missing name or keyword still shows as itself).
  def expect_refused(label, error = ArgumentError, &call)
    aggregate_failures(label) { expect(&call).to raise_error(error) }
  end

  # Three arms with no tallies: warm-up, whose target is the equal split over the free arms, through the lift.
  def warm(**options)
    allocate(arms: %i[control a b], tallies: {}, **options)
  end

  # Each arm's P(best) and expected loss among `free` alone, from 0.2.0's joint draws of EVERY arm on `seed` (the
  # allocator's Beta posteriors, prior 1 / 1, JointBeta's draw order): per draw the best free value wins (a tie split
  # equally) and each free arm loses the best free value less its own.
  def among_free(tallies, free, seed:, draws:)
    posteriors = tallies.values.map do |tally|
      SkiftetStatistical::Bandit::WeightedPosterior.beta(SkiftetStatistical::Bandit::WeightedPosterior.count(tally))
    end
    wins = Hash.new(0.0)
    losses = Hash.new(0.0)
    SkiftetStatistical::JointBeta.each_draw(posteriors, draws: draws, rng: Random.new(seed)) do |row|
      values = tallies.keys.zip(row).to_h
      best = free.map { |arm| values[arm] }.max
      leaders = free.select { |arm| values[arm] == best }
      leaders.each { |arm| wins[arm] += 1.0 / leaders.size }
      free.each { |arm| losses[arm] += best - values[arm] }
    end
    free.to_h { |arm| [ arm, { p_best: wins[arm] / draws, expected_loss: losses[arm] / draws } ] }
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

    it "SL1 lifts warm-up's equal split onto the one floor named, the arms it does not name keeping Config#floor" do
      # arm_floors names control alone: a and b keep 0.10. The equal 1/3 leaves control under 0.5, so control is
      # pinned there; a and b share m = 1 - 0.5 = 0.5 equally, 0.25 each, above their 0.10.
      result = warm(arm_floors: { control: 0.5 })

      expect(result[:phase]).to eq(:warmup)
      expect_weights(result[:weights], { control: 0.5, a: 0.25, b: 0.25 })
    end

    it "SL1 keeps Config#floor for an arm arm_floors does not name, while shifting" do
      # control 100, a 300 and b 100 of 1 000: a is 0.2 above both, about 11.5 standard deviations, so a takes every
      # draw (P(best) 1, expected loss 0; the hold keeps it uncrowned) and control and b have P(best) 0.
      # Floors { control: 0.05 }: the target 0 / 1 / 0 pins control at 0.05 and b, not named, at Config#floor 0.10;
      # a takes m = 1 - 0.05 - 0.10 = 0.85. From 0.10 / 0.80 / 0.10 the largest move is 0.05 <= 0.15: 0.05 / 0.85 /
      # 0.10. Floors { control: 0.05, b: 0.02 }: b is pinned at its own 0.02 and a takes 0.93; the largest move from
      # 0.10 / 0.80 / 0.10 is a's 0.13 <= 0.15: 0.05 / 0.93 / 0.02.
      tallies = { control: tally(1_000, 100), a: tally(1_000, 300), b: tally(1_000, 100) }
      previous = { control: 0.10, a: 0.80, b: 0.10 }
      control_only = allocate(arms: %i[control a b], tallies: tallies, previous_weights: previous,
                              hold_satisfied: false, arm_floors: { control: 0.05 })
      control_and_b = allocate(arms: %i[control a b], tallies: tallies, previous_weights: previous,
                               hold_satisfied: false, arm_floors: { control: 0.05, b: 0.02 })

      [ control_only, control_and_b ].each { |result| expect(result).to include(phase: :shifting, leader: :a) }
      expect_weights(control_only[:weights], { control: 0.05, a: 0.85, b: 0.10 })
      expect_weights(control_and_b[:weights], { control: 0.05, a: 0.93, b: 0.02 })
    end
  end

  describe "a pause whose previous weights break a floor or a fixed share" do
    it "SL1 lifts the previous weights onto the floors, as warm-up's split is" do
      # control's floor 0.05, a and b keep 0.10. Previous 0.02 / 0.49 / 0.49 leaves control under 0.05: control is
      # pinned there and a and b share m = 0.95 in proportion to 0.49 : 0.49, 0.95 * 0.49 / 0.98 = 0.475 each.
      paused = allocate(arms: %i[control a b], tallies: {}, status: "paused",
                        previous_weights: { control: 0.02, a: 0.49, b: 0.49 }, arm_floors: { control: 0.05 })

      expect(paused[:phase]).to eq(:paused)
      expect_weights(paused[:weights], { control: 0.05, a: 0.475, b: 0.475 })
    end

    it "SL1 lifts the free arms' previous shares when a free arm is under its floor or the fixed share is not held" do
      # control fixed at 0.2, so the free arms a and b share m = 0.8.
      # - Previous 0.2 / 0.75 / 0.05: b is under Config#floor 0.10. 0.75 / 0.05 sum to 0.8, so they scale to
      #   themselves; b is pinned at 0.10 and a takes 0.8 - 0.10 = 0.7.
      # - Previous 0.3 / 0.3 / 0.4: control is not at its 0.2. a and b scale into 0.8 in proportion 3 : 4, 0.8 * 3 / 7
      #   = 12/35 and 0.8 * 4 / 7 = 16/35, both above 0.10.
      # - Previous 0.2 / 0.5 / 0.3 holds the share and both floors and is kept.
      cases = {
        { control: 0.2, a: 0.75, b: 0.05 } => { control: 0.2, a: 0.7, b: 0.1 },
        { control: 0.3, a: 0.3, b: 0.4 } => { control: 0.2, a: 12.0 / 35, b: 16.0 / 35 },
        { control: 0.2, a: 0.5, b: 0.3 } => { control: 0.2, a: 0.5, b: 0.3 }
      }

      cases.each do |previous, expected|
        paused = allocate(arms: %i[control a b], tallies: {}, status: "paused", previous_weights: previous,
                          fixed_shares: { control: 0.2 })

        expect(paused[:phase]).to eq(:paused)
        expect_weights(paused[:weights], expected)
      end
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

    it "SL1 gives a fixed arm no floor: a fixed share under Config#floor holds" do
      # control fixed at 0.05, under 0.10: the free arms a and b share m = 0.95 equally in warm-up, 0.475 each.
      expect_weights(warm(fixed_shares: { control: 0.05 })[:weights], { control: 0.05, a: 0.475, b: 0.475 })
    end

    context "with a fixed arm that would win" do
      # a converts at 0.6 of 1 000, control at 0.1 and b at 0.2. Without fixed shares a takes every draw and is
      # crowned (0.2.0). With a fixed at 0.2, P(best) is taken among control and b: b is 0.1 above control, about 6.3
      # standard deviations, so b takes every free draw (P(best) 1, expected loss 0) and control has P(best) 0.
      let(:arms) { %i[control a b] }
      let(:tallies) { { control: tally(1_000, 100), a: tally(1_000, 600), b: tally(1_000, 200) } }
      let(:previous) { { control: 0.3, a: 0.2, b: 0.5 } }

      it "SL1 holds the fixed arm's share and crowns the leading free arm, never the fixed one" do
        # The won split over the free arms: b 0.95, control 0.05. Into m = 0.8: 0.76 / 0.04; control is under 0.10,
        # pinned there, and b takes 0.8 - 0.10 = 0.7.
        plain = allocate(arms: arms, tallies: tallies, previous_weights: previous)
        fixed = allocate(arms: arms, tallies: tallies, previous_weights: previous, fixed_shares: { a: 0.2 })

        expect(plain).to include(phase: :won, winner: :a, leader: :a)
        expect(fixed).to include(phase: :won, winner: :b, leader: :b)
        expect_weights(fixed[:weights], { control: 0.10, a: 0.2, b: 0.7 })
      end

      it "SL1 holds the fixed arm's share while shifting, the leader a free arm" do
        # hold_satisfied: false. The target: P(best) 0 / 1 over control and b, into m = 0.8: control pinned at 0.10,
        # b 0.7. From control 0.3 and b 0.5 the largest move is 0.2 > 0.15, lambda = 0.75: control 0.3 - 0.15 = 0.15,
        # b 0.5 + 0.15 = 0.65.
        fixed = allocate(arms: arms, tallies: tallies, previous_weights: previous, hold_satisfied: false,
                         fixed_shares: { a: 0.2 })

        expect(fixed).to include(phase: :shifting, winner: nil, leader: :b)
        expect_weights(fixed[:weights], { control: 0.15, a: 0.2, b: 0.65 })
      end

      it "SL1 gives the fixed arm nil P(best) and expected loss, and every other statistic as 0.2.0's" do
        # The joint draws keep 0.2.0's order over every arm, so on one seed the counts, rates and intervals are the
        # unfixed call's; P(best) and expected loss of the free arms are taken among the free arms.
        plain = allocate(arms: arms, tallies: tallies, previous_weights: previous, hold_satisfied: false)
        fixed = allocate(arms: arms, tallies: tallies, previous_weights: previous, hold_satisfied: false,
                         fixed_shares: { a: 0.2 })
        reference = among_free(tallies, %i[control b], seed: 20_261_030, draws: 2_000)

        expect(fixed[:stats][:a]).to include(p_best: nil, expected_loss: nil)
        arms.each do |arm|
          expect(fixed[:stats][arm].except(:p_best, :expected_loss))
            .to eq(plain[:stats][arm].except(:p_best, :expected_loss)), arm.inspect
        end
        expect(fixed[:stats][:b]).to include(p_best: 1.0, expected_loss: 0.0)
        expect(fixed[:stats][:control][:p_best]).to eq(0.0)
        expect(fixed[:stats][:control][:expected_loss]).to be_within(1e-12).of(reference[:control][:expected_loss])
      end
    end

    context "with three free arms beside a fixed share" do
      let(:arms) { %i[control a b c] }

      it "SL1 shares 1 - F by P(best) among the free arms" do
        # control, fixed at 0.2, converts at 0.6 and so takes every draw of all four arms: over every arm the free arms'
        # P(best) would be 0 / 0 / 0, and the lift's equal split 0.8 / 3 each. Among the free arms, b (0.2) is about
        # 6.3 standard deviations above a and c (0.1): P(best) 0 / 1 / 0. Into m = 0.8: a and c are pinned at 0.10 and
        # b takes 0.6. hold_satisfied: false, so b, the leader, steps: from 0.3 / 0.2 / 0.3 the largest move is b's
        # 0.4 > 0.15, lambda = 0.375: a 0.3 - 0.075 = 0.225, b 0.2 + 0.15 = 0.35, c 0.225. Crowned instead (the
        # default hold), the won split over the free arms 0.025 / 0.95 / 0.025 is lifted the same way: 0.1 / 0.6 / 0.1.
        tallies = { control: tally(1_000, 600), a: tally(1_000, 100), b: tally(1_000, 200), c: tally(1_000, 100) }
        previous = { control: 0.2, a: 0.3, b: 0.2, c: 0.3 }
        shifting = allocate(arms: arms, tallies: tallies, previous_weights: previous, hold_satisfied: false,
                            fixed_shares: { control: 0.2 })
        crowned = allocate(arms: arms, tallies: tallies, previous_weights: previous, fixed_shares: { control: 0.2 })

        expect(shifting).to include(phase: :shifting, winner: nil, leader: :b)
        expect(shifting[:stats].transform_values { |arm| arm[:p_best] }).to eq(control: nil, a: 0.0, b: 1.0, c: 0.0)
        expect_weights(shifting[:weights], { control: 0.2, a: 0.225, b: 0.35, c: 0.225 })
        expect(crowned).to include(phase: :won, winner: :b, leader: :b)
        expect_weights(crowned[:weights], { control: 0.2, a: 0.1, b: 0.6, c: 0.1 })
      end

      it "SL1 takes each free arm's P(best) and expected loss among the free arms from every arm's joint draws" do
        # Close free arms (0.300 / 0.310 / 0.305) and a fixed arm far above them (0.6): every P(best) over all four
        # arms would be 0 for the free arms. Among the free arms they sum to 1, and each equals the count of draws it
        # led among a, b and c in 0.2.0's joint draws of all four arms on the same seed.
        tallies = { control: tally(1_000, 600), a: tally(1_000, 300), b: tally(1_000, 310), c: tally(1_000, 305) }
        reference = among_free(tallies, %i[a b c], seed: 20_261_030, draws: 2_000)
        result = allocate(arms: arms, tallies: tallies, previous_weights: { control: 0.2, a: 0.3, b: 0.3, c: 0.2 },
                          fixed_shares: { control: 0.2 })

        expect(result[:phase]).to eq(:shifting)
        expect(result[:stats][:control]).to include(p_best: nil, expected_loss: nil)
        expect(%i[a b c].sum { |arm| result[:stats][arm][:p_best] }).to be_within(1e-12).of(1.0)
        %i[a b c].each do |arm|
          expect(result[:stats][arm][:p_best]).to be_within(1e-12).of(reference[arm][:p_best]), arm.inspect
          expect(result[:stats][arm][:expected_loss]).to be_within(1e-12).of(reference[arm][:expected_loss]), arm.inspect
        end
        expect(result[:weights][:control]).to be_within(1e-12).of(0.2)
        expect_floors(result[:weights], { a: 0.10, b: 0.10, c: 0.10 })
      end
    end
  end

  describe "the checks" do
    it "SL1 refuses an arm_floors or fixed_shares key that is not an arm, Symbol or String, before any draw" do
      # Named as arms (a String reads as its Symbol), they answer: warm-up's equal split over the free arms.
      expect_weights(warm(arm_floors: { "b" => 0.02 })[:weights], { control: 1.0 / 3, a: 1.0 / 3, b: 1.0 / 3 })
      expect_weights(warm(fixed_shares: { "control" => 0.2 })[:weights], { control: 0.2, a: 0.4, b: 0.4 })

      [ { arm_floors: { c: 0.02 } }, { arm_floors: { "c" => 0.02 } }, { fixed_shares: { c: 0.2 } },
        { fixed_shares: { "c" => 0.2 } }, { arm_floors: { b: 0.02, c: 0.02 } } ].each do |options|
        rng = Random.new(7)

        expect_refused(options.inspect) do
          described_class.call(arms: %i[control a b], tallies: {}, status: "running", winner_arm: nil, rng: rng,
                               draws: 2_000, **options)
        end
        expect(rng.rand).to eq(Random.new(7).rand), "#{options.inspect} drew before refusing"
      end
    end

    it "SL1 refuses floors and fixed shares summing above 1, an arm not named counting Config#floor" do
      # At exactly 1 they answer. control's floor 0.8 + a's and b's 0.10 = 1: the equal 1/3 leaves control under 0.8,
      # pinned there, and a and b share 0.2, 0.1 each, on their floors. Fixed 0.5 + 0.4 + b's floor 0.10 = 1: b, the
      # one free arm, takes the rest, 0.1.
      expect_weights(warm(arm_floors: { control: 0.8 })[:weights], { control: 0.8, a: 0.1, b: 0.1 })
      expect_weights(warm(fixed_shares: { control: 0.5, a: 0.4 })[:weights], { control: 0.5, a: 0.4, b: 0.1 })

      # Above 1: 0.9 + 2 * 0.10 = 1.1; 0.5 + 0.6 + 0.10 = 1.2; 0.5 + 0.45 + 0.10 = 1.05; 0.2 + 0.5 + 0.5 = 1.2.
      [ { arm_floors: { control: 0.9 } }, { fixed_shares: { control: 0.5 }, arm_floors: { a: 0.6 } },
        { fixed_shares: { control: 0.5, a: 0.45 } }, { fixed_shares: { control: 0.2 }, arm_floors: { a: 0.5, b: 0.5 } } ]
        .each { |options| expect_refused(options.inspect) { warm(**options) } }
    end

    it "SL1 refuses one arm in both keywords" do
      # A floor on a beside control's fixed share answers: a and b share 0.8 equally, 0.4 each, above 0.05 and 0.10.
      expect_weights(warm(arm_floors: { a: 0.05 }, fixed_shares: { control: 0.2 })[:weights],
                     { control: 0.2, a: 0.4, b: 0.4 })

      [ { arm_floors: { control: 0.05 }, fixed_shares: { control: 0.2 } },
        { arm_floors: { "control" => 0.05 }, fixed_shares: { control: 0.2 } } ]
        .each { |options| expect_refused(options.inspect) { warm(**options) } }
    end

    it "SL1 refuses a floor that is not a finite number in (0, 1] and a fixed share that is not one in (0, 1)" do
      expect_weights(warm(arm_floors: { control: 0.5 })[:weights], { control: 0.5, a: 0.25, b: 0.25 })
      expect_weights(warm(fixed_shares: { control: 0.2 })[:weights], { control: 0.2, a: 0.4, b: 0.4 })

      [ 0.0, -0.05, 1.5, Float::NAN, Float::INFINITY, "0.05", nil ].each do |floor|
        expect_refused(floor.inspect) { warm(arm_floors: { control: floor }) }
      end
      [ 0.0, 1.0, -0.2, Float::NAN, "0.2", nil ].each do |share|
        expect_refused(share.inspect) { warm(fixed_shares: { control: share }) }
      end
    end

    it "SL1 refuses a stored win of a fixed arm, and lifts a free arm's stored win over the free arms" do
      # control fixed at 0.2 and a won: the won split over the free arms is a 0.95, b 0.05; into m = 0.8 that is
      # 0.76 / 0.04, b is pinned at its floor 0.10 and a takes 0.7.
      tallies = { control: tally(400, 100), a: tally(400, 140), b: tally(400, 100) }
      won = allocate(arms: %i[control a b], tallies: tallies, status: "won", winner_arm: :a,
                     fixed_shares: { control: 0.2 })

      expect(won).to include(phase: :won, winner: :a)
      expect_weights(won[:weights], { control: 0.2, a: 0.7, b: 0.1 })
      [ :control, "control" ].each do |winner|
        expect_refused(winner.inspect) do
          allocate(arms: %i[control a b], tallies: tallies, status: "won", winner_arm: winner,
                   fixed_shares: { control: 0.2 })
        end
      end
    end

    it "SL1 refuses every arm fixed as FixedAllocation" do
      # One arm fixed of two answers: the other takes the rest, 0.5.
      expect_weights(allocate(arms: %i[control a], tallies: {}, fixed_shares: { control: 0.5 })[:weights],
                     { control: 0.5, a: 0.5 })

      { %i[control] => { control: 0.5 }, %i[control a] => { control: 0.5, a: 0.5 },
        %i[control a b] => { control: 0.3, a: 0.3, b: 0.4 } }.each do |arms, shares|
        expect_refused(arms.inspect, described_class::FixedAllocation) do
          allocate(arms: arms, tallies: {}, fixed_shares: shares)
        end
      end
    end
  end
end
