# frozen_string_literal: true

# The hold on an automatic win, simulated (moved from f1's spec/services/experiments/win_hold_simulation_spec.rb
# with the same seeds, runs, passes, draws and bounds): a win counts only after the same arm met the win rule at
# every 15-minute pass for 24 hours. Tagged `simulation: true`: excluded unless SIMULATIONS=1
# (spec/spec_helper.rb); CI runs it in its simulations job. Measured results, with and without the hold:
# README.md, "Bandit simulations: measured results".
#
# Each run replays an allocation job's arithmetic in memory, as allocator_simulation_spec.rb does: each pass
# draws 40 visits by the current weights, each exposure counts 1 / the weight it was drawn under (with the sum of
# the squares, for the Kish n), the allocator gets the previous weights, and the hold is Bandit::WinHold on a clock
# that moves 15 minutes per pass, with the previous pass one tick back as its last_pass_at (with no last pass the
# hold counts as interrupted). The decision step is f1's ExperimentAllocationJob#decide: the allocator with
# hold_satisfied: hold.satisfied?, and the same seed again without the hold when the arm that wins now is not the
# arm that held.
module WinHoldSimulation
  module_function

  PASS_SECONDS = 900

  def blank_tally
    { exposures: 0, conversions: 0, weighted_exposures: 0.0, weighted_conversions: 0.0, weighted_exposures_squared: 0.0 }
  end

  def pick(weights, arms, roll)
    cumulative = 0.0
    arms.find { |arm| (cumulative += weights[arm]) > roll } || arms.last
  end

  # One simulated experiment. Returns the crowned arm, or nil. `hold: false` measures the allocator alone
  # (hold_satisfied: true at every pass); `hold_seconds` is the hold's length.
  def crowned_arm(arms:, rates:, visits_per_pass:, passes:, seed:, draws:, hold: true,
                  hold_seconds: SkiftetStatistical::Bandit::WinHold::HOLD_SECONDS)
    traffic = Random.new(seed)
    decisions = Random.new(seed + 1_000_000)
    tallies = arms.to_h { |arm| [ arm, blank_tally ] }
    weights = arms.to_h { |arm| [ arm, 1.0 / arms.size ] }
    previous = nil
    state = { leading_arm: nil, leading_since: nil }
    clock = Time.utc(2026, 10, 4)
    last_pass_at = nil

    passes.times do
      visits_per_pass.times do
        arm = pick(weights, arms, traffic.rand)
        tally = tallies[arm]
        inverse = 1.0 / weights[arm]
        tally[:exposures] += 1
        tally[:weighted_exposures] += inverse
        tally[:weighted_exposures_squared] += inverse * inverse
        next unless traffic.rand < rates.fetch(arm)

        tally[:conversions] += 1
        tally[:weighted_conversions] += inverse
      end

      win_hold = SkiftetStatistical::Bandit::WinHold.new(**state, now: clock, last_pass_at: last_pass_at,
                                                         hold_seconds: hold_seconds)
      seed_for_pass = decisions.rand(2**62)
      inputs = { arms: arms, tallies: tallies.transform_values(&:dup), status: "running", winner_arm: nil,
                 previous_weights: previous, draws: draws }
      satisfied = hold ? win_hold.satisfied? : true
      result = SkiftetStatistical::Bandit::Allocator.call(**inputs, rng: Random.new(seed_for_pass),
                                                          hold_satisfied: satisfied)
      if hold && result[:phase] == :won && !win_hold.satisfied_for?(result[:winner])
        result = SkiftetStatistical::Bandit::Allocator.call(**inputs, rng: Random.new(seed_for_pass), hold_satisfied: false)
      end
      return result[:winner] if result[:phase] == :won

      state = win_hold.after_pass(result[:leader])
      last_pass_at = clock
      clock += PASS_SECONDS
      weights = result[:weights]
      previous = weights
    end
    nil
  end
end

RSpec.describe "Bandit::WinHold simulation", simulation: true do
  # Runs at 1 000 draws. It asserts both sides: the worse arm at most 6 / 200, and the better arm at least
  # 20 / 200 (a hold that never releases crowns nothing and would pass the first bound alone).
  it "crowns an arm 1 point worse (10 % against 11 %) in at most 3 % of 200 runs over 400 passes" do
    arms = %i[control worse]
    rates = { control: 0.11, worse: 0.10 }

    crowned = Array.new(200) do |run|
      WinHoldSimulation.crowned_arm(arms: arms, rates: rates, visits_per_pass: 40, passes: 400,
                                    seed: 40_261_003 + run, draws: 1_000)
    end

    worse = crowned.count(:worse)
    better = crowned.count(:control)
    expect(worse).to be <= 6,
                     "the worse arm was crowned in #{worse} of 200 runs (#{(worse / 2.0).round(1)} %); " \
                     "the better arm in #{better}"
    expect(better).to be >= 20,
                      "the better arm was crowned in only #{better} of 200 runs; the worse arm in #{worse}"
  end
end
