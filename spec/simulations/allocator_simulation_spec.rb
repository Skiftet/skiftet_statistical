# frozen_string_literal: true

# The automatic win rule under repeated looks, simulated (moved from f1's
# spec/services/experiments/allocator_simulation_oracle_spec.rb with the same seeds, runs, passes, draws and
# bounds). Tagged `simulation: true`: excluded unless SIMULATIONS=1 (spec/spec_helper.rb); CI runs them in its
# simulations job. Measured results: README.md, "Bandit simulations: measured results".
#
# Each run replays an allocation job's arithmetic in memory: every pass, visits are drawn by the weights of the
# latest allocation, each exposure counts 1 / the weight it was drawn under (inverse-propensity counts), and the
# allocator is called with the previous weights, as a job does every 15 minutes. Draw count 400: fewer draws than
# the 20 000 a production job uses make P(best) noisier near the 0.95 gate, which can only add false wins, so a
# pass at 400 draws is no weaker a claim than one at 20 000.
module AllocatorSimulation
  module_function

  def blank_tally
    { exposures: 0, conversions: 0, weighted_exposures: 0.0, weighted_conversions: 0.0 }
  end

  def pick(weights, arms, roll)
    cumulative = 0.0
    arms.find { |arm| (cumulative += weights[arm]) > roll } || arms.last
  end

  # One simulated experiment. Returns the crowned arm, or nil.
  def run_winner(arms:, rates:, visits_per_pass:, passes:, seed:, draws:)
    traffic = Random.new(seed)
    decisions = Random.new(seed + 1_000_000)
    tallies = arms.to_h { |arm| [ arm, blank_tally ] }
    weights = arms.to_h { |arm| [ arm, 1.0 / arms.size ] }
    previous = nil

    passes.times do
      visits_per_pass.times do
        arm = pick(weights, arms, traffic.rand)
        tally = tallies[arm]
        tally[:exposures] += 1
        tally[:weighted_exposures] += 1.0 / weights[arm]
        next unless traffic.rand < rates.fetch(arm)

        tally[:conversions] += 1
        tally[:weighted_conversions] += 1.0 / weights[arm]
      end

      result = SkiftetStatistical::Bandit::Allocator.call(
        arms: arms, tallies: tallies.transform_values(&:dup), status: "running", winner_arm: nil,
        previous_weights: previous, rng: decisions, draws: draws
      )
      return (result[:winner] || weights.max_by { |_arm, weight| weight }.first) if result[:winner] || result[:phase] == :won

      weights = result[:weights]
      previous = weights
    end
    nil
  end
end

RSpec.describe "Bandit::Allocator simulation", simulation: true do
  # The won rule checks P(best) on every pass, so with equal arms it crowns one in some runs. That costs nothing in
  # conversions (the crowned arm is as good as the rest, and its expected loss is at most win_loss by
  # construction), so the bound that matters is a cap on noise wins (< 10 %) plus the next example: a truly worse
  # arm is never crowned.
  it "crowns an arm in fewer than 10 % of 200 seeded runs when four arms convert equally, over 400 passes" do
    # 4 arms at a true 30 % each, 40 exposed visits per 15-minute pass (warm-up ends near pass 30, then ~370 looks
    # up to ~4 000 visits per arm).
    arms = %i[control a b c]
    rates = arms.to_h { |arm| [ arm, 0.30 ] }

    wins = Array.new(200) do |run|
      AllocatorSimulation.run_winner(arms: arms, rates: rates, visits_per_pass: 40, passes: 400,
                                     seed: 20_261_003 + run, draws: 400)
    end.count { |winner| !winner.nil? }

    expect(wins).to be < 20, "false wins in #{wins} of 200 runs (#{(wins / 2.0).round(1)} %)"
  end

  it "never crowns an arm that converts 5 points worse, in 100 seeded runs over 400 passes" do
    arms = %i[control a b worse]
    rates = { control: 0.30, a: 0.30, b: 0.30, worse: 0.25 }

    winners = Array.new(100) do |run|
      AllocatorSimulation.run_winner(arms: arms, rates: rates, visits_per_pass: 40, passes: 400,
                                     seed: 30_261_003 + run, draws: 400)
    end

    expect(winners.count(:worse)).to eq(0), "the worse arm won #{winners.count(:worse)} of 100 runs"
  end
end
