# frozen_string_literal: true

# The allocation across two layers, simulated on generated shapes (SL13 (e) and (j)). Tagged `simulation: true`:
# excluded unless SIMULATIONS=1 (spec/spec_helper.rb). Measured results: README.md, "Bandit simulations: measured
# results"; each example also prints its figures (RSpec's reporter message, so they reach the JSON formatter's
# "messages" too).
#
# (e) Many cells under one pooled model: each cell's arms are estimated from its own visits (SharedCredit.from_sums on
# an ordinary test's per-arm sums), the arms' contrasts against the reference arm enter the pooled fit through
# SharedCredit.contrast_for_fit (one per-visit variance pooled over the two arms, an arm of a Kish's size under 20 left
# out, a nil left out), are pooled across the cells (AdditiveContrasts.fit over (arm, part) pairs), each cell's level
# comes from all its visits (AdditiveContrasts.baseline, one per part), the two are put back together as the cell's
# estimate (AdditiveContrasts.pooled_estimate), and the cell is allocated on that estimate (LinearValue.summary, then
# Allocator.from_summary): the pooling reaches the decision. The variances run as designed: a weekly fit by moments,
# each variance then held at 0 unless its test of no variation (Fit#variation_tests) rejects at a family-wise 5 % (0.05
# divided by the number of tests that week), and every pass until the next weekly fit holds them fixed (fit's
# `variances:`); before the first weekly fit nothing has rejected, so each is held at 0. The bound is read against a
# noise floor: the same simulation, seeds and runs with every cell's level equal. Every pass fitting the variances by
# moments instead is measured and reported beside it, with no bound.
#
# (j) Two options above an inner layer that picks one item among each option's candidates. The inner layer's pick
# probabilities are exact (BetaArgmax.probabilities, an epsilon of 0.1 over Beta states), each visit's shown item is
# drawn by them (one uniform against those odds: the distribution of the Beta draw they describe, to which SL13 (i)
# holds BetaArgmax.pick), and each option is scored by the value of its greedy policy (BetaArgmax.greedy at the visit): a visit
# credits every option whose greedy pick it showed, with 1 / the visit's probability of showing that item (the sum over
# options of the option's share times its pick probability), through SharedCredit. The same visits scored as they ran
# (each visit credits its drawn option with 1 / its share) are reported beside.
#
# Every shape is built here from round generic values.
module TwoLayerSimulation
  module_function

  ALLOCATOR = SkiftetStatistical::Bandit::Allocator
  CONTRASTS = SkiftetStatistical::AdditiveContrasts

  # SL13 (e)'s weekly fit: the passes from one weekly fit of the variances to the next, and the family-wise level of
  # the week's tests of no variation: each test keeps its variance's moment at FAMILY_LEVEL divided by the number of
  # tests that week (Bonferroni).
  WEEK = 7
  FAMILY_LEVEL = 0.05

  # The smallest Kish's size at which an arm's contrast enters the fit (SharedCredit.contrast_for_fit's
  # `minimum_effective:`, stated here as the ruling states it).
  KISH_MINIMUM = 20

  # What a weekly fit holds until the next: each depth's, grouping's and arm's variance by moments where its test of no
  # variation rejects at the week's level (FAMILY_LEVEL / the number of tests), else 0.0. Answers the variances to hand
  # every pass's fit, how many were tested, how many kept their moment, and the level each test was read at.
  def weekly_variances(fit)
    tested = fit[:variances].sum { |_name, by_arm| by_arm.size }
    level = FAMILY_LEVEL / tested
    kept = 0
    held = fit[:variances].to_h do |name, by_arm|
      [ name, by_arm.to_h do |arm, variance|
        rejects = fit.variation_tests.dig(name, arm, :p_value) < level
        kept += 1 if rejects
        [ arm, rejects ? variance : 0.0 ]
      end ]
    end
    [ held, tested, kept, level ]
  end

  # A binary part is 1 with its probability; a count part is 1, 2 or 3 (equally often) with its probability, else 0.
  def binary(rng, probability)
    rng.rand < probability ? 1.0 : 0.0
  end

  def count(rng, probability)
    hit = rng.rand < probability
    size = 1 + rng.rand(3)
    hit ? size.to_f : 0.0
  end

  def pick(weights, keys, roll)
    cumulative = 0.0
    keys.find { |key| (cumulative += weights[key]) > roll } || keys.last
  end

  # The per-arm Hajek sums of an ordinary test, as SharedCredit.from_sums reads them.
  class Sums
    attr_reader :exposures

    def initialize(arms, parts)
      @parts = parts
      @sums = arms.to_h do |arm|
        [ arm, { w: 0.0, w2: 0.0, a: parts.to_h { |part| [ part, 0.0 ] }, b: parts.to_h { |part| [ part, 0.0 ] },
                 c: parts.product(parts).to_h { |pair| [ pair, 0.0 ] } } ]
      end
      @exposures = arms.to_h { |arm| [ arm, 0 ] }
    end

    def add(arm, credit, values)
      sums = @sums.fetch(arm)
      squared = credit * credit
      @exposures[arm] += 1
      sums[:w] += credit
      sums[:w2] += squared
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

    def estimate(arms, parts)
      credited = arms.select { |arm| @sums.fetch(arm)[:w].positive? }.to_h { |arm| [ arm, @sums.fetch(arm) ] }
      SkiftetStatistical::SharedCredit.from_sums(credited, arms: arms, parts: parts)
    end
  end

  # AdditiveContrasts.baseline's rows: the sums over the visits of one (cell, arm, covariate), per part.
  class BaselineRows
    def initialize(parts)
      @parts = parts
      @rows = {}
    end

    def add(cell, arm, covariate, credit, values)
      row = (@rows[[ cell, arm, covariate ]] ||= { w: 0.0, q: 0.0, s: Array.new(@parts.size, 0.0),
                                                   b: Array.new(@parts.size, 0.0), c: Array.new(@parts.size, 0.0) })
      squared = credit * credit
      row[:w] += credit
      row[:q] += squared
      values.each_with_index do |value, index|
        row[:s][index] += credit * value
        row[:b][index] += squared * value * value
        row[:c][index] += squared * value
      end
    end

    # One part's rows, each arm named by its (arm, part) pair.
    def of(part)
      index = @parts.index(part)
      @rows.map do |(cell, arm, covariate), row|
        { cell: cell, arm: CONTRASTS.pair(arm, part), covariate: covariate, weight: row[:w], value: row[:s][index],
          weight_squared: row[:q], value_squared: row[:b][index], cross: row[:c][index] }
      end
    end
  end

  # A cell's contrasts against the reference arm, per (arm, part) pair, from its own estimate, each through
  # SharedCredit.contrast_for_fit at that part's unit weight (one per-visit variance pooled over the two arms, each
  # weighing as its Kish's size). A pair it answers nil for (either arm with no credit, or a Kish's size under
  # KISH_MINIMUM) is left out, and so is one whose variance is 0, which `fit` takes as nil (SL9). Answers the contrasts
  # and how many pairs were left out.
  def contrasts(estimate, reference, others, parts)
    left_out = 0
    kept = others.each_with_object({}) do |arm, out|
      parts.each_index do |index|
        unit = Array.new(parts.size, 0.0)
        unit[index] = 1.0
        contrast = SkiftetStatistical::SharedCredit.contrast_for_fit(estimate, arm, reference, weights: unit,
                                                                     minimum_effective: KISH_MINIMUM)
        if contrast && contrast[:variance].positive?
          out[CONTRASTS.pair(arm, parts[index])] = [ contrast[:mean], contrast[:variance] ]
        else
          left_out += 1
        end
      end
    end
    [ kept, left_out ]
  end

  # Spearman's rank correlation (average ranks for ties).
  def rank_correlation(first, second)
    ranks = lambda do |values|
      order = values.each_with_index.sort_by(&:first)
      out = Array.new(values.size)
      order.chunk_while { |(a, _), (b, _)| a == b }.inject(0) do |start, group|
        rank = start + ((group.size - 1) / 2.0)
        group.each { |_, index| out[index] = rank }
        start + group.size
      end
      out
    end
    x = ranks.call(first)
    y = ranks.call(second)
    mean_x = x.sum / x.size
    mean_y = y.sum / y.size
    covariance = x.zip(y).sum { |a, b| (a - mean_x) * (b - mean_y) }
    covariance / Math.sqrt(x.sum { |a| (a - mean_x)**2 } * y.sum { |b| (b - mean_y)**2 })
  end

  def report(text)
    RSpec.configuration.reporter.message(text)
  end
end

RSpec.describe "Two-layer allocation simulation", simulation: true do
  sim = TwoLayerSimulation
  config = SkiftetStatistical::Bandit::Allocator::Config

  # SL13 (e)'s shape and runs. 40 cells in 8 groups of 5 (each cell its own node under its group), crossed with 4 bands.
  # A cell's level is its factor (0.5 to 2.45 in steps of 0.05, in an order that follows neither group nor band) times
  # its covariate mix: each visit's covariate k1, k2 or k3 multiplies its probabilities by 0.5, 1.0 or 1.5, in
  # proportions that differ by cell. Every arm of a cell has the same values: no effect differs from the reference arm's,
  # in any cell. Each cell starts at equal shares, takes 30 visits a pass by its own weights, and is allocated at every
  # pass on its pooled estimate (warm-up 0, nothing crowned). `variances` :weekly runs them as designed (13 BM17): the
  # last pass of every WEEK passes runs the weekly fit by moments, TwoLayerSimulation.weekly_variances keeps what it
  # holds at the week's family-wise level, and every pass's fit holds those fixed (17 SL9's `variances:`) until the
  # next; before the first weekly fit no test has rejected, so every variance is held at 0, and no pass fits them by
  # moments. :every_pass fits them by moments at every pass. `levels` :equal is the noise floor: every cell's factor is
  # set so that its level is the mean of the 40 levels, its covariate mix, seeds, runs and passes unchanged, so all that
  # moves between the two is how the levels differ. Answers { means: each cell's mean reference share over its pooled
  # passes, in cell order; levels: the cells' levels as the shape varies them, whichever `levels` ran; pooled_passes:;
  # weekly: the weekly fits' figures (fits, variances tested, variances kept at their moment, the largest kept, the
  # levels the tests were read at); offered: the (cell, pair) contrasts computed over all passes; left_out: those left
  # out of the fit }.
  def pooled_cells(variances:, levels: :varied, runs: 3, passes: 50, seed: 1_307_000)
    sim = TwoLayerSimulation
    arms = %i[reference a b]
    others = arms - [ :reference ]
    parts = %i[rate count]
    floors = { reference: 0.05, a: 0.02, b: 0.02 }
    parameters = { base: { fixed: 1.0 }, worth: { uniform: [ 0.25, 0.75 ] } }
    weights = ->(drawn) { { rate: drawn.fetch(:base), count: drawn.fetch(:worth) } }
    pairs = others.product(parts).map { |arm, part| sim::CONTRASTS.pair(arm, part) }
    cell_config = SkiftetStatistical::Bandit::Allocator::Config.new(warmup_exposures: 0)
    cells = Array.new(40) { |index| "c#{index}" }
    factor = cells.each_with_index.to_h { |cell, index| [ cell, 0.5 + (0.05 * ((index * 13) % 40)) ] }
    multipliers = { k1: 0.5, k2: 1.0, k3: 1.5 }
    mix = cells.each_with_index.to_h do |cell, index|
      raw = [ 1.0 + (index % 3), 2.0, 1.0 + ((index / 3) % 3) ]
      [ cell, multipliers.keys.zip(raw.map { |share| share / raw.sum }).to_h ]
    end
    # A cell's level per unit of factor: its mix's mean multiplier times the value of one visit at factor 1.
    per_factor = cells.to_h do |cell|
      [ cell, mix[cell].sum { |key, share| share * multipliers[key] * ((0.10 * 1.0) + (2 * 0.05 * 0.5)) } ]
    end
    level = cells.map { |cell| factor[cell] * per_factor[cell] }
    raise ArgumentError, "levels must be :varied or :equal" unless %i[varied equal].include?(levels)

    if levels == :equal
      common = level.sum / level.size
      factor = cells.to_h { |cell| [ cell, common / per_factor[cell] ] }
    end
    tree = cells.each_with_index.to_h { |cell, index| [ cell, [ :root, "g#{index / 5}", cell ] ] }
    bands = cells.each_with_index.to_h { |cell, index| [ cell, "b#{index % 4}" ] }
    raise ArgumentError, "variances must be :weekly or :every_pass" unless %i[weekly every_pass].include?(variances)

    zero = [ "tree:1", "tree:2", "band" ].to_h { |name| [ name, pairs.to_h { |pair| [ pair, 0.0 ] } ] }

    share_sums = Hash.new(0.0)
    share_counts = Hash.new(0)
    weekly = { fits: 0, tested: 0, kept: 0, largest_kept: 0.0, test_levels: [] }
    offered = 0
    left_out = 0
    runs.times do |run|
      traffic = Random.new(seed + run)
      decisions = Random.new(seed + 500 + run)
      sums = cells.to_h { |cell| [ cell, sim::Sums.new(arms, parts) ] }
      rows = sim::BaselineRows.new(parts)
      split = cells.to_h { |cell| [ cell, arms.to_h { |arm| [ arm, 1.0 / arms.size ] } ] }
      previous = cells.to_h { |cell| [ cell, nil ] }
      held = variances == :weekly ? zero : nil

      passes.times do |pass|
        cells.each do |cell|
          30.times do
            arm = sim.pick(split[cell], arms, traffic.rand)
            credit = 1.0 / split[cell][arm]
            covariate = sim.pick(mix[cell], multipliers.keys, traffic.rand)
            scale = factor[cell] * multipliers[covariate]
            values = [ sim.binary(traffic, 0.10 * scale), sim.count(traffic, 0.05 * scale) ]
            sums[cell].add(arm, credit, values)
            rows.add(cell, arm, covariate, credit, values)
          end
        end

        own = cells.to_h { |cell| [ cell, sums[cell].estimate(arms, parts) ] }
        fit_cells = cells.map do |cell|
          kept_contrasts, left = sim.contrasts(own[cell], :reference, others, parts)
          offered += pairs.size
          left_out += left
          { id: cell, contrasts: kept_contrasts, tree: tree[cell], groups: { band: bands[cell] } }
        end
        named = fit_cells.flat_map { |cell| cell[:contrasts].keys }.uniq
        fit = nil
        if (pairs - named).empty?
          if variances == :weekly && ((pass + 1) % sim::WEEK).zero?
            held, tested, kept, test_level = sim.weekly_variances(sim::CONTRASTS.fit(cells: fit_cells, arms: pairs))
            weekly[:fits] += 1
            weekly[:tested] += tested
            weekly[:kept] += kept
            weekly[:largest_kept] = [ weekly[:largest_kept], *held.values.flat_map(&:values) ].max
            weekly[:test_levels] |= [ test_level ]
          end
          fit = sim::CONTRASTS.fit(cells: fit_cells, arms: pairs, variances: held)
        end
        baselines = fit && parts.to_h { |part| [ part, sim::CONTRASTS.baseline(rows: rows.of(part), effects: fit[:cells]) ] }

        cells.each do |cell|
          summary = nil
          if fit && !own[cell][:means].value?(nil)
            pooled = sim::CONTRASTS.pooled_estimate(own: own[cell], effects: fit[:cells], baselines: baselines, cell: cell)
            summary = SkiftetStatistical::LinearValue.summary(
              arms: arms, parts: parts, means: pooled[:means], covariance: pooled[:covariance], parameters: parameters,
              weights: weights, draws: 500, rng: decisions, rounding: pooled.rounding
            )
          end
          result = sim::ALLOCATOR.from_summary(arms: arms, summary: summary, exposures: sums[cell].exposures,
                                               status: "running", winner_arm: nil, previous_weights: previous[cell],
                                               hold_satisfied: false, config: cell_config, arm_floors: floors)
          split[cell] = result[:weights]
          previous[cell] = result[:weights]
          next unless summary

          share_sums[cell] += result[:weights][:reference]
          share_counts[cell] += 1
        end
      end
    end

    { means: cells.map { |cell| share_counts[cell].positive? ? share_sums[cell] / share_counts[cell] : nil },
      levels: level, pooled_passes: share_counts.values.sum, weekly: weekly, offered: offered, left_out: left_out }
  end

  # The figures of one pooled_cells answer, as the reports print them.
  def shares_text(answer)
    means = answer[:means]
    sd = SkiftetStatistical::Descriptive.standard_deviation(means)
    "#{means.min.round(4)} to #{means.max.round(4)} (spread #{(means.max - means.min).round(4)}, standard deviation " \
      "#{sd.round(4)}), over #{answer[:pooled_passes]} cell passes allocated on a pooled estimate, " \
      "#{answer[:left_out]} of #{answer[:offered]} cell contrasts left out of the fit"
  end

  def weekly_text(weekly)
    "#{weekly[:fits]} weekly fits, each test read at #{weekly[:test_levels].map { |level| level.round(6) }.join(', ')}, " \
      "#{weekly[:kept]} of #{weekly[:tested]} variances kept at their moment (the largest #{weekly[:largest_kept].round(6)})"
  end

  it "SL13(e) keeps the reference arm's mean share across 40 cells whose baselines differ, with zero effect " \
     "difference, at a rank correlation with the level within 0.3 and a spread at most 1.5 times the noise floor's " \
     "(the same runs with every level equal), each cell allocated on its pooled estimate, the contrasts entering the " \
     "fit by contrast_for_fit, the variances fitted weekly with a family-wise test of no variation and held fixed between" do
    # As designed: a weekly fit by moments every 7 passes (7 a run), each variance held at 0 unless its test of no
    # variation rejects at 0.05 over the week's number of tests, and every pass until the next weekly fit holding them
    # fixed. Where nothing rejects, each cell's effects are the pooled root's and only its level
    # (AdditiveContrasts.baseline) is its own; a variance kept at its moment lets each cell keep that share of its own
    # noise in its effects until the next weekly fit. The orchestrator restated the bound on 2026-10-31 so that it
    # tests leakage of the level into the allocation, not Monte Carlo noise: the rank correlation within 0.3, and the
    # spread at most 1.5 times the spread of the same simulation (seeds, runs, passes) with every cell's level equal. 17
    # SL13 (e)'s fixed 0.03 is reported beside it.
    varied = pooled_cells(variances: :weekly)
    floor = pooled_cells(variances: :weekly, levels: :equal)
    means = varied[:means]
    spread = means.max - means.min
    floor_spread = floor[:means].max - floor[:means].min
    ratio = spread / floor_spread
    correlation = sim.rank_correlation(means, varied[:levels])
    floor_correlation = sim.rank_correlation(floor[:means], varied[:levels])

    sim.report("SL13(e) measured, contrasts by contrast_for_fit (Kish's minimum #{sim::KISH_MINIMUM}), variances " \
               "fitted weekly (every #{sim::WEEK} passes) by moments, held at 0 unless the test of no variation " \
               "rejects at a family-wise #{sim::FAMILY_LEVEL}, and held fixed between: reference arm's mean share " \
               "#{shares_text(varied)}; rank correlation with the level #{correlation.round(3)}; " \
               "#{weekly_text(varied[:weekly])}. Noise floor (every level equal, same seeds and runs): " \
               "#{shares_text(floor)}; its rank correlation with the levels the varied run has " \
               "#{floor_correlation.round(3)}; #{weekly_text(floor[:weekly])}. Spread over the noise floor's " \
               "#{ratio.round(3)} (bound 1.5); spread against the fixed 0.03: #{spread.round(4)}")
    expect(means).to all(be_a(Float))
    expect(floor[:means]).to all(be_a(Float))
    expect(varied[:weekly][:fits]).to eq(21), "#{varied[:weekly][:fits]} weekly fits over the 3 runs, not 7 a run"
    expect(floor[:weekly][:fits]).to eq(21), "#{floor[:weekly][:fits]} weekly fits over the noise floor's 3 runs"
    aggregate_failures do
      expect(correlation.abs).to be <= 0.3, "its rank correlation with the cells' level is #{correlation}"
      expect(spread).to be <= 1.5 * floor_spread,
                        "the reference arm's mean share spans #{spread} across the cells, #{ratio} times the noise " \
                        "floor's #{floor_spread}"
    end
  end

  it "SL13(e) reports, with no bound, the reference arm's mean share across the same 40 cells when every pass fits " \
     "the variances by moments" do
    # The same shape and seeds, the contrasts by contrast_for_fit, with every pass fitting the variances by moments: at
    # no true variation a term's fitted variance comes out above 0 in about half the fits, and a cell then keeps that
    # share of its own noise in its effects, which its traffic follows. Measured for the README beside the weekly fit.
    answer = pooled_cells(variances: :every_pass)
    correlation = sim.rank_correlation(answer[:means], answer[:levels])

    sim.report("SL13(e) measured, contrasts by contrast_for_fit, variances fitted by moments at every pass (no " \
               "bound): reference arm's mean share #{shares_text(answer)}; rank correlation with the level " \
               "#{correlation.round(3)}")
    expect(answer[:means]).to all(be_a(Float))
  end

  it "SL13(j) scores two options of identical value by greedy shared credit, one with cold candidates: their mean " \
     "P(best) lies in 0.40-0.60 over 100 runs and the cold option ends at its floor in at most 10 % of them" do
    options = %i[warm cold]
    parts = %i[item rest]
    floors = { warm: 0.02, cold: 0.02 }
    parameters = { item_worth: { uniform: [ 0.5, 1.5 ] }, rest_worth: { fixed: 1.0 } }
    weights = ->(drawn) { { item: drawn.fetch(:item_worth), rest: drawn.fetch(:rest_worth) } }
    epsilon = 0.1
    rates = [ 0.30, 0.20, 0.10 ]
    # The warm option's three candidates have 1,000 showings each before the run, at their true rates, so a run's own
    # showings move them little: their state is held, its probabilities computed once. The cold option's three
    # candidates start at Beta(1, 1) and learn from their showings at each pass; their best (0.30, as the warm option's
    # best) sits first, second or third by run, so the greedy pick before any showing (ties go to the first) is right
    # in a third of the runs. 30 passes of 80 visits, the traffic split by the greedy score's allocation (default
    # settings, warm-up included, nothing crowned). P(best) is averaged over the passes where both options have credit
    # (a pass without takes the warm-up split and has no P(best), SL2) and over the runs. The score as they ran drives a
    # second allocation beside it, from the same visits, that moves no traffic.
    warm_state = rates.map { |rate| [ 1.0 + (1_000 * rate), 1.0 + (1_000 * (1.0 - rate)) ] }
    warm_odds = SkiftetStatistical::BetaArgmax.probabilities(warm_state, epsilon: epsilon)
    warm_greedy = SkiftetStatistical::BetaArgmax.greedy(warm_state)

    runs = 100
    passes = 30
    scored = { greedy: Hash.new(0.0), as_ran: Hash.new(0.0) }
    scored_passes = { greedy: 0, as_ran: 0 }
    at_floor = { greedy: 0, as_ran: 0 }
    runs.times do |run|
      traffic = Random.new(1_308_000 + run)
      decisions = { greedy: Random.new(1_308_500 + run), as_ran: Random.new(1_308_900 + run) }
      cold_rates = rates.rotate(-(run % 3))
      cold_state = Array.new(3) { [ 1.0, 1.0 ] }
      candidates = { warm: rates.map.with_index { |rate, index| [ "w#{index}", rate ] },
                     cold: cold_rates.map.with_index { |rate, index| [ "k#{index}", rate ] } }
      credits = { greedy: SkiftetStatistical::SharedCredit.accumulator(arms: options, parts: parts),
                  as_ran: SkiftetStatistical::SharedCredit.accumulator(arms: options, parts: parts) }
      exposures = options.to_h { |option| [ option, 0 ] }
      split = options.to_h { |option| [ option, 0.5 ] }
      chains = { greedy: { previous: nil, weights: split }, as_ran: { previous: nil, weights: split } }

      passes.times do
        odds = { warm: warm_odds, cold: SkiftetStatistical::BetaArgmax.probabilities(cold_state, epsilon: epsilon) }
        greedy = { warm: warm_greedy, cold: SkiftetStatistical::BetaArgmax.greedy(cold_state) }
        showings = Array.new(3) { [ 0.0, 0.0 ] }
        split = chains[:greedy][:weights]
        80.times do
          option = sim.pick(split, options, traffic.rand)
          index = sim.pick(odds[option].each_with_index.to_h { |q, at| [ at, q ] }, [ 0, 1, 2 ], traffic.rand)
          item, rate = candidates[option][index]
          values = [ sim.binary(traffic, rate), sim.count(traffic, 0.05) ]
          exposures[option] += 1
          showings[index][0] += values[0] if option == :cold
          showings[index][1] += 1.0 - values[0] if option == :cold

          # The visit's probability of showing this item, over every option that has it among its candidates.
          marginal = options.sum do |other|
            at = candidates[other].index { |name, _| name == item }
            at ? split[other] * odds[other][at] : 0.0
          end
          credit = options.select { |other| candidates[other][greedy[other]].first == item }
                          .to_h { |other| [ other, 1.0 / marginal ] }
          credits[:greedy].add({ credit: credit, values: values }) unless credit.empty?
          credits[:as_ran].add({ credit: { option => 1.0 / split[option] }, values: values })
        end
        showings.each_with_index do |(taps, misses), index|
          cold_state[index] = [ cold_state[index][0] + taps, cold_state[index][1] + misses ]
        end

        chains.each do |score, chain|
          estimate = credits[score].estimate
          summary = nil
          unless estimate[:means].value?(nil)
            summary = SkiftetStatistical::LinearValue.summary(
              arms: options, parts: parts, means: estimate[:means], covariance: estimate[:covariance],
              parameters: parameters, weights: weights, draws: 500, rng: decisions[score], rounding: estimate.rounding
            )
          end
          result = sim::ALLOCATOR.from_summary(arms: options, summary: summary, exposures: exposures, status: "running",
                                               winner_arm: nil, previous_weights: chain[:previous],
                                               hold_satisfied: false, config: config.new, arm_floors: floors)
          chain[:weights] = result[:weights]
          chain[:previous] = result[:weights]
          next unless summary

          scored_passes[score] += 1
          options.each { |option| scored[score][option] += summary[option][:p_best] }
        end
      end
      chains.each { |score, chain| at_floor[score] += 1 if chain[:weights][:cold] == floors[:cold] }
    end

    means = scored.to_h { |score, sums| [ score, sums.transform_values { |sum| (sum / scored_passes[score]).round(4) } ] }
    sim.report("SL13(j) measured: mean P(best) by greedy shared credit #{means[:greedy]} over " \
               "#{scored_passes[:greedy]} passes, the cold option at its floor at the end of #{at_floor[:greedy]} of " \
               "#{runs} runs; the same runs scored as they ran #{means[:as_ran]} over #{scored_passes[:as_ran]} " \
               "passes, an allocation on that score ending the cold option at its floor in #{at_floor[:as_ran]}")
    means[:greedy].each do |option, mean|
      expect(mean).to be_between(0.40, 0.60), "#{option}'s mean P(best) by greedy shared credit is #{mean}"
    end
    expect(at_floor[:greedy]).to be <= 10, "the cold option ended at its floor in #{at_floor[:greedy]} of #{runs} runs"
  end
end
