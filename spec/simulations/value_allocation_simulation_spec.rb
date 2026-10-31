# frozen_string_literal: true

# The allocation on a linear value of several parts, simulated on generated shapes (SL13 (a)-(d) and (f)). Tagged
# `simulation: true`: excluded unless SIMULATIONS=1 (spec/spec_helper.rb). Measured results: README.md, "Bandit
# simulations: measured results"; each example also prints its figures (RSpec's reporter message, so they reach the
# JSON formatter's "messages" too).
#
# Each run replays a caller's allocation passes in memory, as allocator_simulation_spec.rb does: every pass draws its
# visits by the latest weights, and each visit is an ordinary test's row, crediting only the arm it was drawn for with
# 1 / the share it was drawn under. The caller keeps the per-arm Hajek sums of those rows (w = sum 1/p, w2 = sum 1/p^2,
# a = sum x/p, b = sum x/p^2, c = sum x_p * x_q / p^2) and at each pass turns them into the arms' joint normal
# (SharedCredit.from_sums), draws the value with its uncertain weights (LinearValue.summary, P(best) taken among the free
# arms, SL1) and hands that summary to the one phase rule (Allocator.from_summary, SL2). A pass where a free arm has no
# credit yet takes the warm-up split (`summary: nil`), never a value read from nil means.
#
# Every shape is built here from round generic values: a binary part, a mostly-zero count part (1, 2 or 3 when not 0)
# and a rare binary part, each with a probability per arm.
module ValueAllocationSimulation
  module_function

  ALLOCATOR = SkiftetStatistical::Bandit::Allocator

  # Which kind of value each part takes: a binary part is 1 with its probability; a count part is 1, 2 or 3 (equally
  # often) with its probability, else 0. A count part's mean is twice its probability.
  KINDS = { rate: :binary, count: :count, rare: :binary }.freeze

  # A shape: the arms (in order), the parts, each arm's probability per part, the weights' parameters and callable
  # (LinearValue's), the allocator's settings and declared shares, and the joint draws per pass.
  Shape = Struct.new(:arms, :parts, :profile, :parameters, :weights, :config, :arm_floors, :fixed_shares, :draws,
                     keyword_init: true) do
    def free
      arms - (fixed_shares || {}).keys
    end

    # The floor each free arm keeps (its own, else Config#floor) and each fixed arm's share.
    def bounds
      arms.to_h do |arm|
        fixed = (fixed_shares || {})[arm]
        [ arm, fixed ? [ :fixed, fixed ] : [ :floor, (arm_floors || {}).fetch(arm, config.floor) ] ]
      end
    end
  end

  # The per-arm Hajek sums of an ordinary test, as SharedCredit.from_sums reads them. A visit is exposed when it is
  # drawn and its values are added when they arrive, at once or later; a value of 0 adds nothing.
  class Sums
    attr_reader :exposures

    def initialize(arms, parts)
      @parts = parts
      @sums = arms.to_h { |arm| [ arm, blank ] }
      @exposures = arms.to_h { |arm| [ arm, 0 ] }
    end

    def expose(arm, credit)
      sums = @sums.fetch(arm)
      @exposures[arm] += 1
      sums[:w] += credit
      sums[:w2] += credit * credit
    end

    def convert(arm, credit, values)
      sums = @sums.fetch(arm)
      squared = credit * credit
      @parts.each_with_index do |part, index|
        value = values[index]
        next if value.zero?

        sums[:a][part] += credit * value
        sums[:b][part] += squared * value
        (index...@parts.size).each do |other|
          product = squared * value * values[other]
          sums[:c][[ part, @parts[other] ]] += product
          sums[:c][[ @parts[other], part ]] += product unless other == index
        end
      end
    end

    # The sums of `arms` that some visit has credited (an arm no visit has credited is left out: it has no credit).
    def of(arms)
      arms.select { |arm| @sums.fetch(arm)[:w].positive? }.to_h { |arm| [ arm, @sums.fetch(arm) ] }
    end

    private

    def blank
      { w: 0.0, w2: 0.0, a: @parts.to_h { |part| [ part, 0.0 ] }, b: @parts.to_h { |part| [ part, 0.0 ] },
        c: @parts.product(@parts).to_h { |pair| [ pair, 0.0 ] } }
    end
  end

  # The yes/no tally of the binary part `rate`, as Allocator.call reads it, with the inverse-propensity sums.
  class Tallies
    def initialize(arms)
      @tallies = arms.to_h do |arm|
        [ arm, { exposures: 0, conversions: 0, weighted_exposures: 0.0, weighted_conversions: 0.0,
                 weighted_exposures_squared: 0.0 } ]
      end
    end

    def add(arm, credit, converted)
      tally = @tallies.fetch(arm)
      tally[:exposures] += 1
      tally[:weighted_exposures] += credit
      tally[:weighted_exposures_squared] += credit * credit
      return unless converted

      tally[:conversions] += 1
      tally[:weighted_conversions] += credit
    end

    def to_h
      @tallies.transform_values(&:dup)
    end
  end

  def pick(weights, arms, roll)
    cumulative = 0.0
    arms.find { |arm| (cumulative += weights[arm]) > roll } || arms.last
  end

  # One visit's values, part by part (two uniforms a count part, one a binary part), each probability times `factor`.
  def draw_values(shape, arm, rng, factor = 1.0)
    shape.parts.map do |part|
      probability = shape.profile.fetch(arm).fetch(part) * factor
      if KINDS.fetch(part) == :binary
        rng.rand < probability ? 1.0 : 0.0
      else
        hit = rng.rand < probability
        size = 1 + rng.rand(3)
        hit ? size.to_f : 0.0
      end
    end
  end

  # The true mean of each part for an arm.
  def true_means(shape, arm)
    shape.parts.map do |part|
      probability = shape.profile.fetch(arm).fetch(part)
      KINDS.fetch(part) == :count ? 2.0 * probability : probability
    end
  end

  # The estimate of the free arms, from their sums alone (no covariance is sliced), and LinearValue's summary of it; nil
  # while a free arm has no credit (SL2: the warm-up split, without the value model).
  def value_summary(shape, sums, rng, draws: shape.draws, arms: shape.free)
    estimate = SkiftetStatistical::SharedCredit.from_sums(sums.of(arms), arms: arms, parts: shape.parts)
    return nil if estimate[:means].value?(nil)

    SkiftetStatistical::LinearValue.summary(arms: arms, parts: shape.parts, means: estimate[:means],
                                            covariance: estimate[:covariance], parameters: shape.parameters,
                                            weights: shape.weights, draws: draws, rng: rng,
                                            rounding: estimate.rounding)
  end

  # One allocation pass on the value model: the summary above through the one phase rule.
  def value_pass(shape, sums, rng:, previous:, status: "running", winner_arm: nil, hold_satisfied: false)
    ALLOCATOR.from_summary(arms: shape.arms, summary: value_summary(shape, sums, rng), exposures: sums.exposures,
                           status: status, winner_arm: winner_arm, previous_weights: previous,
                           hold_satisfied: hold_satisfied, config: shape.config, fixed_shares: shape.fixed_shares,
                           arm_floors: shape.arm_floors)
  end

  # One allocation pass on the yes/no tallies of the part `rate` (Allocator.call's own Beta draws).
  def tally_pass(shape, tallies, rng:, previous:, status: "running", winner_arm: nil, hold_satisfied: false)
    ALLOCATOR.call(arms: shape.arms, tallies: tallies.to_h, status: status, winner_arm: winner_arm, rng: rng,
                   previous_weights: previous, draws: shape.draws, hold_satisfied: hold_satisfied,
                   config: shape.config, fixed_shares: shape.fixed_shares, arm_floors: shape.arm_floors)
  end

  # Every way a pass's weights can break the shape's bounds: a free arm under its floor, a fixed arm off its share, or
  # weights that do not sum to 1 (to 1e-12). An empty Array when none.
  def breaches(shape, weights)
    found = shape.bounds.filter_map do |arm, (kind, bound)|
      share = weights.fetch(arm)
      if kind == :fixed
        "#{arm} at #{share}, not its fixed #{bound}" unless share == bound
      elsif share < bound
        "#{arm} at #{share}, under its floor #{bound}"
      end
    end
    total = weights.values.sum
    found << "the shares sum to #{total}" unless (total - 1.0).abs <= 1e-12
    found
  end

  # SL13 (a)'s walk through every phase, by pass (1-based): warm-up then shifting under "running" with nothing crowned
  # (to pass 150); a pause (151-170) whose stored split, handed in at its first pass, breaks every floor, as a split
  # stored under older settings would; a stretch (171-300) where the automatic win may crown, its win then stored and
  # passed back as status "won"; and the test reopened under "running" (301-400).
  BROKEN_SPLIT_AT = 151

  def stage(pass)
    case pass
    when 151..170 then :paused
    when 171..300 then :may_win
    else :running
    end
  end

  # One run of SL13 (a)'s walk. `kind` :value runs the value model, :tally Allocator.call on the part `rate`. Answers
  # every pass's [phase, weights, leader].
  def walk(shape, kind:, seed:, passes: 400, visits_per_pass: 40)
    traffic = Random.new(seed)
    decisions = Random.new(seed + 1_000_000)
    sums = Sums.new(shape.arms, shape.parts)
    tallies = Tallies.new(shape.arms)
    weights = shape.arms.to_h { |arm| [ arm, 1.0 / shape.arms.size ] }
    previous = nil
    stored_winner = nil
    rate = shape.parts.index(:rate)
    broken = shape.arms.each_with_index.to_h { |arm, index| [ arm, index == shape.arms.size - 1 ? 0.96 : 0.01 ] }
    total = broken.values.sum
    broken.transform_values! { |share| share / total }

    (1..passes).map do |pass|
      visits_per_pass.times do
        arm = pick(weights, shape.arms, traffic.rand)
        credit = 1.0 / weights[arm]
        values = draw_values(shape, arm, traffic)
        sums.expose(arm, credit)
        sums.convert(arm, credit, values)
        tallies.add(arm, credit, values[rate] == 1.0)
      end

      previous = broken if pass == BROKEN_SPLIT_AT
      current = stage(pass)
      stored_winner = nil unless current == :may_win
      status = { paused: "paused", may_win: stored_winner ? "won" : "running", running: "running" }.fetch(current)
      inputs = { rng: decisions, previous: previous, status: status, winner_arm: stored_winner,
                 hold_satisfied: current == :may_win }
      result = kind == :value ? value_pass(shape, sums, **inputs) : tally_pass(shape, tallies, **inputs)
      stored_winner = result[:winner] if current == :may_win && result[:phase] == :won
      weights = result[:weights]
      previous = weights
      [ result[:phase], weights, result[:leader] ]
    end
  end

  def report(text)
    RSpec.configuration.reporter.message(text)
  end
end

RSpec.describe "Value allocation simulation", simulation: true do
  sim = ValueAllocationSimulation
  config = SkiftetStatistical::Bandit::Allocator::Config

  # A binary part worth one, and a count part whose worth is uncertain, uniform on [0.25, 0.75].
  rate_and_count = {
    parameters: { base: { fixed: 1.0 }, worth: { uniform: [ 0.25, 0.75 ] } },
    weights: ->(drawn) { { rate: drawn.fetch(:base), count: drawn.fetch(:worth) } }
  }

  it "SL13(a) keeps a floored reference arm (0.05) and four arms at 0.02 at or above their floors in every phase, " \
     "over 400 passes, on the value model and on Allocator.call" do
    # 40 visits a pass; one arm (d) is clearly the best, so the others are pushed onto their floors and the automatic
    # win can crown it in the walk's third stretch. Each run walks warm-up, shifting, the pause on a stored split that
    # breaks every floor, the automatic win and its stored win, and the reopened test (ValueAllocationSimulation.walk).
    floors = { reference: 0.05, a: 0.02, b: 0.02, c: 0.02, d: 0.02 }
    profile = { reference: { rate: 0.10, count: 0.05 }, a: { rate: 0.10, count: 0.05 },
                b: { rate: 0.12, count: 0.05 }, c: { rate: 0.08, count: 0.05 }, d: { rate: 0.20, count: 0.10 } }
    shape = sim::Shape.new(arms: floors.keys, parts: %i[rate count], profile: profile, config: config.new,
                           arm_floors: floors, draws: 1_000, **rate_and_count)
    # The same arms with one arm held at a fixed share: its share exactly in every phase, P(best) among the free arms.
    fixed = sim::Shape.new(arms: floors.keys, parts: %i[rate count], profile: profile, config: config.new,
                           arm_floors: floors.except(:c), fixed_shares: { c: 0.30 }, draws: 1_000,
                           **rate_and_count)

    runs = [ [ shape, :value, 3 ], [ fixed, :value, 1 ], [ shape, :tally, 2 ], [ fixed, :tally, 1 ] ]
    seen = Hash.new { |hash, key| hash[key] = Hash.new(0) }
    breaches = []
    floor_visits = Hash.new(0)
    automatic_wins = Hash.new(0)
    runs.each_with_index do |(run_shape, kind, count), group|
      count.times do |run|
        label = "#{run_shape.fixed_shares ? 'fixed' : 'floored'} #{kind} run #{run}"
        walked = sim.walk(run_shape, kind: kind, seed: 1_301_000 + (100 * group) + run)
        walked.each_with_index do |(phase, weights, leader), at|
          seen[label][phase] += 1
          automatic_wins[label] += 1 if phase == :won && leader
          sim.breaches(run_shape, weights).each { |text| breaches << "#{label}, pass #{at + 1} (#{phase}): #{text}" }
          run_shape.bounds.each do |arm, (bound_kind, bound)|
            floor_visits[label] += 1 if bound_kind == :floor && weights.fetch(arm) == bound
          end
        end
      end
    end

    sim.report("SL13(a) measured: phases per run #{seen.transform_values(&:to_h)}; automatic wins " \
               "#{automatic_wins.to_h}; arm-passes held exactly at a floor #{floor_visits.to_h}; breaches #{breaches.size}")
    expect(breaches).to be_empty, breaches.first(10).join("\n")
    seen.each do |label, phases|
      expect(phases.keys).to include(:warmup, :shifting, :paused, :won), "#{label} saw only #{phases.keys}"
    end
    expect(automatic_wins.size).to eq(seen.size), "an automatic win crowned in only #{automatic_wins.keys}"
    expect(floor_visits.values).to all(be_positive)
  end

  it "SL13(b) crowns nothing over 2,000 passes at hold_satisfied: false, while the win rule names a leader, on the " \
     "value model and on Allocator.call" do
    # A test that never ends: every pass at hold_satisfied: false, 40 visits a pass, and one arm (b) at twice the
    # others' rate, so the win rule holds for it at almost every pass and nothing may crown it.
    floors = { reference: 0.05, a: 0.02, b: 0.02 }
    profile = { reference: { rate: 0.10, count: 0.05 }, a: { rate: 0.10, count: 0.05 },
                b: { rate: 0.20, count: 0.05 } }
    shape = sim::Shape.new(arms: floors.keys, parts: %i[rate count], profile: profile, config: config.new,
                           arm_floors: floors, draws: 500, **rate_and_count)

    figures = {}
    %i[value tally].each_with_index do |kind, index|
      traffic = Random.new(1_302_000 + index)
      decisions = Random.new(1_302_500 + index)
      sums = sim::Sums.new(shape.arms, shape.parts)
      tallies = sim::Tallies.new(shape.arms)
      weights = shape.arms.to_h { |arm| [ arm, 1.0 / shape.arms.size ] }
      previous = nil
      crowned = []
      led = 0
      breaches = []
      2_000.times do |pass|
        40.times do
          arm = sim.pick(weights, shape.arms, traffic.rand)
          credit = 1.0 / weights[arm]
          values = sim.draw_values(shape, arm, traffic)
          sums.expose(arm, credit)
          sums.convert(arm, credit, values)
          tallies.add(arm, credit, values.first == 1.0)
        end
        inputs = { rng: decisions, previous: previous, hold_satisfied: false }
        result = kind == :value ? sim.value_pass(shape, sums, **inputs) : sim.tally_pass(shape, tallies, **inputs)
        crowned << pass + 1 if result[:phase] == :won || !result[:winner].nil? || result[:result_changed]
        led += 1 if result[:leader] == :b
        breaches.concat(sim.breaches(shape, result[:weights]))
        weights = result[:weights]
        previous = weights
      end
      figures[kind] = { crowned: crowned.size, led_by_b: led, final: weights.transform_values { |share| share.round(4) } }

      expect(crowned).to be_empty, "#{kind}: crowned at passes #{crowned.first(5)}"
      expect(breaches).to be_empty, "#{kind}: #{breaches.first(5)}"
      expect(led).to be >= 1_800, "#{kind}: the better arm was named the leader in only #{led} of 2,000 passes"
    end
    sim.report("SL13(b) measured: #{figures}")
  end

  it "SL13(c) estimates each arm's value without bias beyond 2 standard errors over 200 runs when every visit's " \
     "values arrive up to 23 passes late" do
    # Each visit is exposed when drawn and its values arrive 0 to 23 passes later (uniform), so every pass allocates on
    # sums that lack the values still on their way. 60 passes of 40 visits, then the last values arrive: the final
    # estimate (LinearValue's point) against the truth. The in-flight estimate at pass 60, before those arrive, is
    # reported beside it.
    floors = { reference: 0.05, a: 0.02, b: 0.02 }
    profile = { reference: { rate: 0.10, count: 0.05 }, a: { rate: 0.12, count: 0.05 },
                b: { rate: 0.14, count: 0.06 } }
    shape = sim::Shape.new(arms: floors.keys, parts: %i[rate count], profile: profile, config: config.new,
                           arm_floors: floors, draws: 300, **rate_and_count)
    window = 24
    passes = 60
    # The value at the parameters' centre (base 1.0, worth 0.5), as LinearValue's point reads it.
    truth = shape.arms.to_h { |arm| [ arm, sim.true_means(shape, arm).zip([ 1.0, 0.5 ]).sum { |m, w| m * w } ] }

    final = []
    in_flight = []
    200.times do |run|
      traffic = Random.new(1_303_000 + run)
      decisions = Random.new(1_303_500 + run)
      sums = sim::Sums.new(shape.arms, shape.parts)
      pending = Hash.new { |hash, pass| hash[pass] = [] }
      weights = shape.arms.to_h { |arm| [ arm, 1.0 / shape.arms.size ] }
      previous = nil
      (1..passes).each do |pass|
        pending.delete(pass)&.each { |arm, credit, values| sums.convert(arm, credit, values) }
        40.times do
          arm = sim.pick(weights, shape.arms, traffic.rand)
          credit = 1.0 / weights[arm]
          values = sim.draw_values(shape, arm, traffic)
          delay = traffic.rand(window)
          sums.expose(arm, credit)
          delay.zero? ? sums.convert(arm, credit, values) : pending[pass + delay] << [ arm, credit, values ]
        end
        result = sim.value_pass(shape, sums, rng: decisions, previous: previous)
        weights = result[:weights]
        previous = weights
      end
      point = ->(summary) { summary.transform_values { |stats| stats[:point] } }
      in_flight << point.call(sim.value_summary(shape, sums, Random.new(run), draws: 1))
      pending.each_value { |arrivals| arrivals.each { |arm, credit, values| sums.convert(arm, credit, values) } }
      final << point.call(sim.value_summary(shape, sums, Random.new(run), draws: 1))
    end

    stats = ->(points, arm) do
      errors = points.map { |values| values.fetch(arm) - truth.fetch(arm) }
      [ SkiftetStatistical::Descriptive.mean(errors),
        SkiftetStatistical::Descriptive.standard_deviation(errors) / Math.sqrt(errors.size) ]
    end
    figures = shape.arms.to_h do |arm|
      bias, se = stats.call(final, arm)
      flight_bias, flight_se = stats.call(in_flight, arm)
      [ arm, { truth: truth[arm].round(4), bias: bias.round(5), se: se.round(5), z: (bias / se).round(2),
               in_flight_bias: flight_bias.round(5), in_flight_z: (flight_bias / flight_se).round(1) } ]
    end
    sim.report("SL13(c) measured: #{figures}")
    figures.each do |arm, figure|
      expect(figure[:z].abs).to be <= 2.0, "#{arm}: bias #{figure[:bias]} is #{figure[:z]} standard errors"
    end
  end

  # Mean P(best) of each arm over every pass from the drift's first on, and over runs (a pass with no summary has no
  # P(best) and is left out). With `paused`, those passes run under status "paused" on `uneven`, stored at the pause's
  # first pass. With `shadow`, every such pass is also scored on unweighted sums (credit 1 a visit), never used to
  # allocate: what the same visits say when the 1 / p weighting is left out.
  def drift_runs(shape, runs:, seed:, drift:, paused: nil, uneven: nil, shadow: false, passes: 120)
    sim = ValueAllocationSimulation
    totals = Hash.new(0.0)
    unweighted_totals = Hash.new(0.0)
    count = 0
    runs.times do |run|
      traffic = Random.new(seed + run)
      decisions = Random.new(seed + 500 + run)
      shadow_rng = Random.new(seed + 900 + run)
      sums = sim::Sums.new(shape.arms, shape.parts)
      unweighted = sim::Sums.new(shape.arms, shape.parts)
      weights = shape.arms.to_h { |arm| [ arm, 1.0 / shape.arms.size ] }
      previous = nil
      (1..passes).each do |pass|
        factor = drift.cover?(pass) ? 3.0 : 1.0
        40.times do
          arm = sim.pick(weights, shape.arms, traffic.rand)
          credit = 1.0 / weights[arm]
          values = sim.draw_values(shape, arm, traffic, factor)
          sums.expose(arm, credit)
          sums.convert(arm, credit, values)
          next unless shadow

          unweighted.expose(arm, 1.0)
          unweighted.convert(arm, 1.0, values)
        end
        previous = uneven if paused && pass == paused.first
        status = paused&.cover?(pass) ? "paused" : "running"
        result = sim.value_pass(shape, sums, rng: decisions, previous: previous, status: status)
        if pass >= drift.first && result[:stats].values.all? { |stats| stats[:p_best] }
          count += 1
          result[:stats].each { |arm, stats| totals[arm] += stats[:p_best] }
          if shadow
            sim.value_summary(shape, unweighted, shadow_rng).each do |arm, stats|
              unweighted_totals[arm] += stats[:p_best]
            end
          end
        end
        weights = result[:weights]
        previous = weights
      end
    end
    [ totals.transform_values { |sum| sum / count }, unweighted_totals.transform_values { |sum| sum / count }, count ]
  end

  drift_shape = lambda do
    floors = { reference: 0.05, a: 0.02, b: 0.02, c: 0.02 }
    profile = floors.keys.to_h { |arm| [ arm, { rate: 0.10, count: 0.05 } ] }
    sim::Shape.new(arms: floors.keys, parts: %i[rate count], profile: profile, config: config.new,
                   arm_floors: floors, draws: 500, **rate_and_count)
  end

  it "SL13(d) keeps equal arms' mean P(best) within 0.05 of 1/K when every rate triples for 20 passes" do
    # Four equal arms, 40 visits a pass; warm-up ends near pass 30 and every probability is three times its own over
    # passes 41-60. Mean over passes 41-120 of 30 runs.
    means, _unweighted, passes = drift_runs(drift_shape.call, runs: 30, seed: 1_304_000, drift: 41..60)

    sim.report("SL13(d) measured: mean P(best) #{means.transform_values { |mean| mean.round(4) }} over #{passes} passes")
    means.each { |arm, mean| expect((mean - 0.25).abs).to be <= 0.05, "#{arm}'s mean P(best) is #{mean}" }
  end

  it "SL13(d) credits the drift to no arm when the split is held uneven through it, where unweighted sums would" do
    # The same shape, paused through the drift on a stored uneven split (a holds 0.65), then running again. Each pass
    # weighs every arm's visits equally once each is counted 1 / p, so the arm that held the traffic through the drift
    # gains nothing from it; unweighted sums, scored beside, crown it (the shape is sharp enough to see the drift).
    # The arm with the least traffic through the drift (the reference arm, 1 / 0.05 a visit) has the widest
    # posterior, so its P(best) reads above 1/K: that is its spread, not the drift, and no bound here reads it.
    means, unweighted, passes = drift_runs(drift_shape.call, runs: 20, seed: 1_305_000, drift: 41..60, paused: 41..60,
                                           uneven: { reference: 0.05, a: 0.65, b: 0.15, c: 0.15 }, shadow: true)

    round = ->(values) { values.transform_values { |mean| mean.round(4) } }
    sim.report("SL13(d) measured, held uneven: mean P(best) #{round.call(means)} over #{passes} passes; " \
               "on unweighted sums #{round.call(unweighted)}")
    expect(means[:a]).to be <= 0.30, "the arm that held the traffic has a mean P(best) of #{means[:a]}"
    expect(unweighted[:a]).to be > 0.5, "unweighted sums gave it only #{unweighted[:a]}: the shape does not see the drift"
  end

  it "SL13(f) covers each arm's value and each difference with SL3's 95 % intervals in at least 93 % of 400 runs, " \
     "on a mostly-zero reward with rare counts of 1-3 and a rare binary part, the true weights drawn per run" do
    # Each run: 6,000 visits on a fixed uneven split (0.2, 0.4, 0.4; each credited 1 / its share), a count part that is
    # 0 in 95 % or more of visits and a binary part at 1.0-1.5 %, the parts' worths uniform on their ranges in the
    # model and the true worths drawn inside those ranges per run. An arm's interval is LinearValue.summary's; a
    # difference's is the same equal-tailed 95 % of its joint draws.
    arms = %i[reference a b]
    profile = { reference: { count: 0.04, rare: 0.010 }, a: { count: 0.05, rare: 0.010 },
                b: { count: 0.04, rare: 0.015 } }
    ranges = { count_worth: [ 0.5, 1.5 ], rare_worth: [ 2.0, 6.0 ] }
    shape = sim::Shape.new(arms: arms, parts: %i[count rare], profile: profile, config: config.new,
                           parameters: ranges.transform_values { |range| { uniform: range } },
                           weights: ->(drawn) { { count: drawn.fetch(:count_worth), rare: drawn.fetch(:rare_worth) } },
                           draws: 4_000)
    split = { reference: 0.2, a: 0.4, b: 0.4 }
    pairs = arms.combination(2).to_a
    covered = Hash.new(0)
    runs = 400

    runs.times do |run|
      traffic = Random.new(1_306_000 + run)
      true_weights = ranges.values.map { |low, high| low + ((high - low) * traffic.rand) }
      truth = arms.to_h { |arm| [ arm, sim.true_means(shape, arm).zip(true_weights).sum { |m, w| m * w } ] }
      sums = sim::Sums.new(arms, shape.parts)
      6_000.times do
        arm = sim.pick(split, arms, traffic.rand)
        credit = 1.0 / split[arm]
        sums.expose(arm, credit)
        sums.convert(arm, credit, sim.draw_values(shape, arm, traffic))
      end

      estimate = SkiftetStatistical::SharedCredit.from_sums(sums.of(arms), arms: arms, parts: shape.parts)
      inputs = { arms: arms, parts: shape.parts, means: estimate[:means], covariance: estimate[:covariance],
                 parameters: shape.parameters, weights: shape.weights, draws: shape.draws,
                 rounding: estimate.rounding }
      summary = SkiftetStatistical::LinearValue.summary(**inputs, rng: Random.new(1_306_500 + run))
      arms.each do |arm|
        covered[arm] += 1 if summary[arm][:lower] <= truth[arm] && truth[arm] <= summary[arm][:upper]
      end
      # The same seed gives the rows the summary drew (SL3), so each difference is read off those joint draws.
      rows = SkiftetStatistical::LinearValue.each_draw(**inputs, rng: Random.new(1_306_500 + run)).to_a
      pairs.each do |first, second|
        i = arms.index(first)
        j = arms.index(second)
        differences = rows.map { |row| row[i] - row[j] }
        lower = SkiftetStatistical::Descriptive.percentile(differences, 2.5)
        upper = SkiftetStatistical::Descriptive.percentile(differences, 97.5)
        exact = truth[first] - truth[second]
        covered[[ first, second ]] += 1 if lower <= exact && exact <= upper
      end
    end

    sim.report("SL13(f) measured: covered of #{runs} #{covered.to_h}")
    (arms + pairs).each do |key|
      expect(covered[key]).to be >= 372, "#{key.inspect} covered in #{covered[key]} of #{runs} runs"
    end
  end
end
