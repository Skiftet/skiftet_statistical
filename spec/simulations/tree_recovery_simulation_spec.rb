# frozen_string_literal: true

# HierarchicalRates on generated five-level trees (SL13 (g)). Tagged `simulation: true`: excluded unless SIMULATIONS=1
# (spec/spec_helper.rb). Every figure the examples measure is printed through the reporter (the JSON formatter keeps it
# under "messages") for the README's measured-results section.
#
# THE TREE, from round generic values only: 1, 20, 200, 800 and 4,000 nodes a level. Parents with a single child: 1 at
# level 1, 10 at level 2 and 40 at level 3 (the root has 20 children); every other parent has two or more. Near-whole
# parents: each parent at levels 1-3 with two or more children is, with probability 0.05, one whose drawn child holds
# 95 % of its exposure (moved within the parent, whose total stays). A leaf's exposure is lognormal (sigma 0.5). Split
# leaves: 2 % of the leaves count in their own level-3 parent by a share w drawn from [0.5, 0.95] and in another
# level-3 node by 1 - w, so the counts above the leaves are real numbers (SL6).
#
# THE TRUTH, one law for every item, which is what gives each level one true strength: at levels 1-3 each (node, item)
# has a multiplier drawn from Gamma(k, rate k), mean 1, k = 8, 8 and 16; at the leaves a two-groups law, the multiplier
# 1 with probability 0.98 and else drawn from Gamma(0.5, rate 0.5), so the hot share is 0.02 and the slab strength 0.5.
# A leaf's expected count of an item is its exposure times the multipliers on its path, scaled to the item's volume.
# Nine items, three of each kind, the kind being how deep an item's counts reach: national (200,000 a half, every level
# well filled), regional (20,000, about 5 a leaf) and very local (2,000, mostly 0 or 1 a leaf, seen below level 2
# mostly where a hotspot leaf lifts it). Beside them every node holds the rest of the pool, 2,000,000 a half summed over
# items too small to fit, whose multipliers average out to 1. Each count is Poisson, drawn twice: half A is the data
# and half B is held out.
#
# THE FIT is the one a caller makes level by level. For a node v at level l with parent p, w its share in p (1 except a
# split leaf in its own parent) and q(c) = (n(c) + 1) / (N + 2) the item's national share, the pairs at l are every v
# with S(p) > w S(v) and every item with s(p, c) > 0 and E > 0, E = S(v) * leave_one_out_share(parent_count: s(p, c),
# parent_total: S(p), node_count: w s(v, c), node_total: w S(v), prior_share: q(c), prior_strength: 16), siblings the
# children of p holding data; the pairs of single-child parents are handed in too. Levels 1-3 are fitted on those
# pairs and bounded on them.
#
# THE LEAVES are bounded on HierarchicalRates.fit_leaves, the leaf fit that estimates each parent's null rate jointly
# with the leaves' slab strength and hot share instead of reading it from a leave-one-out share: a leave-one-out parent
# of about five leaves cannot tell a null leaf from a hotspot beside it, since a hotspot lifts its siblings' parent
# share, so they read as cold, and the parent's own sampling noise reads as overdispersion. The leaf fit's parents are
# every (level-3 node p, item c) whose leaves hold data, a leaf v counted by its share in p as { count: w s(v, c),
# total: w S(v) }; a parent with a single such leaf is handed in too. Its hot shares are HOT_GRID's below 1 (the null
# rate is read from the leaves that may be null) and its strengths every other point of GRID, 2^-2 to 2^12 in half
# powers of 2: a step of a factor 1.41, inside the bound's factor of 2, at half the time of the whole grid. It reads the
# hot share at or below the true one, since a hot leaf whose multiplier is drawn near 1 reads as null (Gamma(0.5, rate
# 0.5) puts about a quarter of its draws between 0.5 and 1.5). The same leaves fitted on the leave-one-out pairs
# (fit_strength) are measured and reported, not bounded: they read the hot share 0.05 to 0.1 for a true 0.02 and the
# slab strength up to about 2.4 times the true one.
#
# HELD OUT: at every level, the half-A prediction of an item's share at a node is the leave-one-out share times a
# ratio, either shrunken by HierarchicalRates.node at the level's fitted strength (at the leaves with the leave-one-out
# fit's hot share) or raw, s / E. Each node's predictions over the nine items are normalised and mixed with 10^-6 of a
# uniform over them, the same for both, so a raw ratio of 0 for an item the node held out costs a finite amount (a mix
# as large as 1 % lifts an item of a 0.2 % share by half and decides the comparison by itself); the log score is the sum
# over half B's counts of the log of their item's predicted share at their node. At level 1, where a node holds
# thousands of an item's counts, shrinking moves a share by a few parts in 10^5, so the gain there is a few nats either
# way and is reported; the sum over the levels, and each level below the first, must favour the shrunken ratio.
module TreeRecoverySimulation
  module_function

  SIZES = [ 1, 20, 200, 800, 4_000 ].freeze
  SINGLE_CHILD_PARENTS = [ 0, 1, 10, 40 ].freeze
  NEAR_WHOLE_RATE = 0.05
  NEAR_WHOLE_SHARE = 0.95
  SPLIT_RATE = 0.02
  SPLIT_SHARE = [ 0.5, 0.95 ].freeze
  EXPOSURE_SIGMA = 0.5
  STRENGTHS = [ 8.0, 8.0, 16.0 ].freeze
  HOT_SHARE = 0.02
  SLAB_STRENGTH = 0.5
  KINDS = { national: 200_000.0, regional: 20_000.0, very_local: 2_000.0 }.freeze
  ITEMS_A_KIND = 3
  REST_OF_POOL = 2_000_000.0
  PRIOR_STRENGTH = 16.0
  HOT_GRID = [ 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005 ].freeze
  LEAF_HOT_SHARES = HOT_GRID.reject { |h| h == 1.0 }.freeze
  LEAF_GRID = SkiftetStatistical::HierarchicalRates::GRID.each_slice(2).map(&:first).freeze
  UNIFORM_MIX = 1e-6
  RUNS = 20
  SEED = 20_261_131
  # Both of the leaves' fits are refitted without the single-child parents in this many runs (they are the slow ones).
  LEAF_SINGLE_CHILD_RUNS = 2

  def volumes
    KINDS.values.flat_map { |volume| Array.new(ITEMS_A_KIND, volume) }
  end

  # Poisson(mean): Knuth's product of uniforms below 10, else Hormann's transformed rejection (PTRS, 1993).
  def poisson(rng, mean)
    return 0 unless mean.positive?
    return knuth(rng, mean) if mean < 10.0

    slam = Math.sqrt(mean)
    log_mean = Math.log(mean)
    b = 0.931 + (2.53 * slam)
    a = -0.059 + (0.02483 * b)
    inverse_alpha = 1.1239 + (1.1328 / (b - 3.4))
    v_r = 0.9277 - (3.6224 / (b - 2.0))
    loop do
      u = rng.rand - 0.5
      v = rng.rand
      us = 0.5 - u.abs
      k = ((((2.0 * a / us) + b) * u) + mean + 0.43).floor
      return k if us >= 0.07 && v <= v_r
      next if k.negative? || (us < 0.013 && v > us)

      accept = -mean + (k * log_mean) - Math.lgamma(k + 1.0)[0]
      return k if Math.log(v * inverse_alpha / ((a / (us * us)) + b)) <= accept
    end
  end

  def knuth(rng, mean)
    limit = Math.exp(-mean)
    count = 0
    product = rng.rand
    while product > limit
      count += 1
      product *= rng.rand
    end
    count
  end

  # Child counts of `parents` parents summing to `total`: `singles` of them have one child, the rest at least two.
  def partition(rng, parents, total, singles)
    counts = Array.new(parents, 2)
    single = (0...parents).to_a.shuffle(random: rng).first(singles)
    single.each { |parent| counts[parent] = 1 }
    others = (0...parents).to_a - single
    (total - counts.sum).times { counts[others[rng.rand(others.size)]] += 1 }
    counts
  end

  # One run's tree, multipliers and both halves' leaf counts.
  def generate(seed)
    rng = Random.new(seed)
    sampler = SkiftetStatistical::Sampler.new(rng)
    parents = [ [ nil ] ]
    (1...SIZES.size).each do |level|
      counts = partition(rng, SIZES[level - 1], SIZES[level], SINGLE_CHILD_PARENTS[level - 1])
      parents << counts.each_with_index.flat_map { |n, parent| Array.new(n, parent) }
    end
    leaves = SIZES.last
    paths = Array.new(leaves) do |leaf|
      path = [ leaf ]
      (SIZES.size - 1).downto(2) { |level| path.unshift(parents[level][path.first]) }
      path
    end
    exposure = Array.new(leaves) { Math.exp(EXPOSURE_SIGMA * sampler.gaussian) }
    near_whole(rng, parents, paths, exposure)

    items = volumes
    multipliers = (1..3).map do |level|
      Array.new(SIZES[level]) { Array.new(items.size) { sampler.gamma(STRENGTHS[level - 1]) / STRENGTHS[level - 1] } }
    end
    leaf_multipliers = Array.new(leaves) do
      Array.new(items.size) { rng.rand < HOT_SHARE ? sampler.gamma(SLAB_STRENGTH) / SLAB_STRENGTH : 1.0 }
    end
    rates = Array.new(leaves) do |leaf|
      path = paths[leaf]
      Array.new(items.size) do |c|
        exposure[leaf] * multipliers[0][path[0]][c] * multipliers[1][path[1]][c] * multipliers[2][path[2]][c] *
          leaf_multipliers[leaf][c]
      end
    end
    scales = Array.new(items.size) { |c| items[c] / rates.sum { |row| row[c] } }
    expected = rates.map { |row| row.each_with_index.map { |rate, c| rate * scales[c] } }
    total_exposure = exposure.sum
    rest = exposure.map { |x| REST_OF_POOL * x / total_exposure }
    halves = Array.new(2) do
      expected.each_with_index.map { |row, leaf| row.map { |mean| poisson(rng, mean) } << poisson(rng, rest[leaf]) }
    end

    splits = {}
    (0...leaves).to_a.shuffle(random: rng).first((SPLIT_RATE * leaves).round).each do |leaf|
      other = rng.rand(SIZES[3] - 1)
      other += 1 if other >= parents[4][leaf]
      splits[leaf] = [ SPLIT_SHARE[0] + ((SPLIT_SHARE[1] - SPLIT_SHARE[0]) * rng.rand), other ]
    end
    { parents: parents, halves: halves, splits: splits, items: items.size }
  end

  # Each near-whole parent's drawn child takes NEAR_WHOLE_SHARE of the parent's exposure, the other children the rest,
  # each subtree scaled as a whole; parents are read from the top, so a deeper parent moves what it holds by then.
  def near_whole(rng, parents, paths, exposure)
    (2...SIZES.size).each do |level|
      children = Hash.new { |hash, key| hash[key] = [] }
      parents[level].each_with_index { |parent, child| children[parent] << child }
      children.each_value do |list|
        next if list.size < 2 || rng.rand >= NEAR_WHOLE_RATE

        big = list[rng.rand(list.size)]
        members = list.to_h { |child| [ child, true ] }
        totals = Hash.new(0.0)
        paths.each_with_index { |path, leaf| totals[path[level - 1]] += exposure[leaf] if members[path[level - 1]] }
        whole = totals.values.sum
        rest = whole - totals[big]
        paths.each_with_index do |path, leaf|
          node = path[level - 1]
          next unless members[node]

          share = node == big ? NEAR_WHOLE_SHARE * whole / totals[big] : (1.0 - NEAR_WHOLE_SHARE) * whole / rest
          exposure[leaf] *= share
        end
      end
    end
  end

  # Counts per level, node and column (the items, then the rest of the pool), split leaves counted by share above.
  def aggregate(run, leaf_counts)
    parents = run[:parents]
    columns = run[:items] + 1
    levels = SIZES.map { |size| Array.new(size) { Array.new(columns, 0.0) } }
    leaf_counts.each_with_index do |row, leaf|
      levels[4][leaf] = row.map(&:to_f)
      share, other = run[:splits][leaf]
      home = parents[4][leaf]
      row.each_with_index do |count, column|
        if share
          levels[3][home][column] += share * count
          levels[3][other][column] += (1.0 - share) * count
        else
          levels[3][home][column] += count
        end
      end
    end
    3.downto(1) do |level|
      levels[level].each_with_index do |row, node|
        above = levels[level - 1][parents[level][node]]
        row.each_with_index { |count, column| above[column] += count }
      end
    end
    levels
  end

  # Every (node, item) at a level with its leave-one-out share, E, count and siblings. `pair` marks the caller's pairs
  # (S(p) > w S(v), s(p, c) > 0, E > 0); `single` marks the pairs of nodes whose parent has fewer than two children
  # holding data, which the caller hands in too.
  def cells(run, levels, level)
    parents = run[:parents]
    items = run[:items]
    totals = levels.map { |rows| rows.map(&:sum) }
    root = levels[0][0]
    national = root.sum
    shares = Array.new(items) { |c| (root[c] + 1.0) / (national + 2.0) }
    holding = Hash.new(0)
    parents[level].each_with_index { |parent, node| holding[parent] += 1 if totals[level][node].positive? }

    out = []
    parents[level].each_with_index do |parent, node|
      share = level == 4 && run[:splits][node] ? run[:splits][node][0] : 1.0
      parent_total = totals[level - 1][parent]
      node_total = totals[level][node]
      items.times do |c|
        parent_count = levels[level - 1][parent][c]
        loo = SkiftetStatistical::HierarchicalRates.leave_one_out_share(
          parent_count: parent_count, parent_total: parent_total, node_count: share * levels[level][node][c],
          node_total: share * node_total, prior_share: shares[c], prior_strength: PRIOR_STRENGTH
        )
        expected = node_total * loo
        counted = parent_count.positive? && expected.positive?
        out << { node: node, item: c, count: levels[level][node][c], expected: expected, share: loo,
                 siblings: holding[parent], pair: counted && parent_total > share * node_total,
                 single: counted && holding[parent] < SkiftetStatistical::HierarchicalRates::MINIMUM_SIBLINGS }
      end
    end
    out
  end

  def pairs_of(list)
    list.map { |cell| { count: cell[:count], expected: cell[:expected], siblings: cell[:siblings] } }
  end

  # The leaf fit's parents: for every level-3 node and item, the node's leaves holding data, each counted by its share
  # in the node (a split leaf by w, every other by 1): [[{ count:, total: }]], parents with no such leaf left out.
  def leaf_parents(run, levels)
    children = Array.new(SIZES[3]) { [] }
    run[:parents][4].each_with_index { |parent, leaf| children[parent] << leaf }
    children.flat_map do |leaves|
      held = leaves.filter_map do |leaf|
        total = levels[4][leaf].sum
        [ leaf, run[:splits][leaf] ? run[:splits][leaf][0] : 1.0, total ] if total.positive?
      end
      next [] if held.empty?

      Array.new(run[:items]) do |c|
        held.map { |leaf, share, total| { count: share * levels[4][leaf][c], total: share * total } }
      end
    end
  end

  def fit_leaves(parents)
    SkiftetStatistical::HierarchicalRates.fit_leaves(parents: parents, hot_shares: LEAF_HOT_SHARES, grid: LEAF_GRID)
  end

  # The leaf fit refitted without the parents of a single leaf: every answer the same, the kept parents' own included.
  def leaf_fit_unchanged(parents, fit)
    minimum = SkiftetStatistical::HierarchicalRates::MINIMUM_SIBLINGS
    kept = parents.each_index.select { |index| parents[index].size >= minimum }
    without = fit_leaves(parents.values_at(*kept))
    same = without.except(:converged, :rates, :hot) == fit.except(:converged, :rates, :hot) &&
           %i[converged rates hot].all? { |key| fit[key].values_at(*kept) == without[key] }
    { same: same, single: parents.size - kept.size }
  end

  # The half-B log score of one level's predictions, shrunken or raw, over the nine items.
  def held_out_score(cells, held_out, items, ratio)
    cells.group_by { |cell| cell[:node] }.sum do |node, list|
      predicted = Array.new(items, 0.0)
      list.each { |cell| predicted[cell[:item]] = cell[:share] * ratio.call(cell) }
      total = predicted.sum
      shares = predicted.map do |value|
        normalised = total.positive? ? value / total : 1.0 / items
        ((1.0 - UNIFORM_MIX) * normalised) + (UNIFORM_MIX / items)
      end
      (0...items).sum { |c| held_out[node][c] * Math.log(shares[c]) }
    end
  end

  def shrunken_ratio(cell, strength, hot_share)
    SkiftetStatistical::HierarchicalRates.node(count: cell[:count], expected: cell[:expected], strength: strength,
                                               hot_share: hot_share, threshold: 1.25, parent_tail: 0.0,
                                               parent_ratio: 1.0)[:ratio]
  end

  def raw_ratio(cell)
    cell[:expected].positive? ? cell[:count] / cell[:expected] : 0.0
  end

  # Every run, computed once for the file.
  def runs
    @runs ||= Array.new(RUNS) { |index| measure(index) }
  end

  def measure(index)
    run = generate(SEED + index)
    data = aggregate(run, run[:halves][0])
    held = aggregate(run, run[:halves][1])
    result = { levels: {}, single_child: {}, held_out: {} }

    (1..4).each do |level|
      list = cells(run, data, level)
      paired = list.select { |cell| cell[:pair] || cell[:single] }
      hot_grid = level == 4 ? HOT_GRID : nil
      fit = SkiftetStatistical::HierarchicalRates.fit_strength(pairs: pairs_of(paired), hot_grid: hot_grid)
      entry = { fit: fit, single: paired.count { |cell| cell[:single] } }
      if (2..3).cover?(level) || (level == 4 && index < LEAF_SINGLE_CHILD_RUNS)
        kept = pairs_of(paired.reject { |cell| cell[:single] })
        without = SkiftetStatistical::HierarchicalRates.fit_strength(pairs: kept, hot_grid: hot_grid)
        result[:single_child][level] = { same: without == fit, single: entry[:single] }
      end
      if level == 4
        parents = leaf_parents(run, data)
        entry[:leaves] = fit_leaves(parents)
        entry[:leaf_parents] = parents.size
        result[:single_child_leaves] = leaf_fit_unchanged(parents, entry[:leaves]) if index < LEAF_SINGLE_CHILD_RUNS
      end
      result[:levels][level] = entry

      shrunken = ->(cell) { shrunken_ratio(cell, fit[:strength], fit[:hot_share]) }
      result[:held_out][level] = held_out_score(list, held[level], run[:items], shrunken) -
                                 held_out_score(list, held[level], run[:items], method(:raw_ratio))
    end
    result
  end

  def within_factor_two(value, truth)
    value.between?(truth / 2.0, truth * 2.0)
  end

  # "value xcount, ..." over the runs, for the report.
  def tally(values)
    values.tally.sort.map { |value, count| "#{value.round(3)} x#{count}" }.join(", ")
  end

  def report(text)
    RSpec.configuration.reporter.message("[tree simulation] #{text}")
  end
end

RSpec.describe "HierarchicalRates tree simulation", simulation: true do
  it "SL13(g) fits each level's strength within a factor of 2 of the true one, and the leaves' hot share within " \
     "0.05, in at least 90 % of 20 runs" do
    runs = TreeRecoverySimulation.runs
    needed = (0.9 * runs.size).ceil
    shortfalls = []

    (1..3).each do |level|
      truth = TreeRecoverySimulation::STRENGTHS[level - 1]
      fitted = runs.map { |run| run[:levels][level][:fit][:strength] }
      inside = fitted.count { |value| TreeRecoverySimulation.within_factor_two(value, truth) }
      TreeRecoverySimulation.report(format("level %d strength (true %.1f): within a factor of 2 in %d of %d runs; " \
                                           "fitted %s", level, truth, inside, runs.size,
                                           TreeRecoverySimulation.tally(fitted)))
      shortfalls << "level #{level}: #{inside} of #{runs.size}" if inside < needed
    end

    leaves = runs.map { |run| run[:levels][4][:leaves] }
    slabs = leaves.map { |fit| fit[:strength] }
    hots = leaves.map { |fit| fit[:hot_share] }
    slab_inside = slabs.count do |value|
      TreeRecoverySimulation.within_factor_two(value, TreeRecoverySimulation::SLAB_STRENGTH)
    end
    hot_inside = hots.count { |value| (value - TreeRecoverySimulation::HOT_SHARE).abs <= 0.05 }
    unsettled = leaves.map { |fit| fit[:converged].count(false) }
    TreeRecoverySimulation.report(format("leaves on the leaf fit (true slab %.2f, hot share %.2f): slab within a " \
                                         "factor of 2 in %d of %d runs, hot share within 0.05 in %d; slab %s; hot " \
                                         "share %s; %d parents a run, %d to %d of them not settled after %d steps",
                                         TreeRecoverySimulation::SLAB_STRENGTH, TreeRecoverySimulation::HOT_SHARE,
                                         slab_inside, runs.size, hot_inside, TreeRecoverySimulation.tally(slabs),
                                         TreeRecoverySimulation.tally(hots), runs.first[:levels][4][:leaf_parents],
                                         unsettled.min, unsettled.max,
                                         SkiftetStatistical::HierarchicalRates::LEAF_STEPS))
    loo = runs.map { |run| run[:levels][4][:fit] }
    TreeRecoverySimulation.report(format("leaves on the leave-one-out E (measured, not bounded): slab %s; hot share %s",
                                         TreeRecoverySimulation.tally(loo.map { |fit| fit[:strength] }),
                                         TreeRecoverySimulation.tally(loo.map { |fit| fit[:hot_share] })))
    shortfalls << "leaf slab: #{slab_inside} of #{runs.size}" if slab_inside < needed
    shortfalls << "leaf hot share: #{hot_inside} of #{runs.size}" if hot_inside < needed
    expect(shortfalls).to be_empty, shortfalls.join("\n")
  end

  it "SL13(g) leaves every fit unchanged by the single-child parents' pairs, at every level that has them, and " \
     "the leaf fit unchanged by the parents of a single leaf" do
    runs = TreeRecoverySimulation.runs
    checks = runs.flat_map { |run| run[:single_child].map { |level, check| [ level, check ] } }
    by_level = checks.group_by(&:first)
    by_level.each do |level, list|
      singles = list.map { |_, check| check[:single] }
      TreeRecoverySimulation.report(format("level %d: %d runs refitted without the single-child parents' pairs, " \
                                           "%d to %d such pairs a run, every fit identical: %s", level, list.size,
                                           singles.min, singles.max, list.all? { |_, check| check[:same] }))
    end
    leaf_checks = runs.filter_map { |run| run[:single_child_leaves] }
    TreeRecoverySimulation.report(format("leaf fit: %d runs refitted without the parents of a single leaf, %d to %d " \
                                         "such parents a run, every answer identical: %s", leaf_checks.size,
                                         leaf_checks.map { |check| check[:single] }.min,
                                         leaf_checks.map { |check| check[:single] }.max,
                                         leaf_checks.all? { |check| check[:same] }))
    expect(by_level.keys.sort).to eq([ 2, 3, 4 ])
    expect(checks.map { |_, check| check[:single] }).to all(be_positive)
    expect(checks.reject { |_, check| check[:same] }).to be_empty
    expect(leaf_checks.size).to eq(TreeRecoverySimulation::LEAF_SINGLE_CHILD_RUNS)
    expect(leaf_checks.map { |check| check[:single] }).to all(be_positive)
    expect(leaf_checks.reject { |check| check[:same] }).to be_empty
  end

  it "SL13(g) gives the shrunken ratio a better held-out log score than the raw ratio in every run, summed over the " \
     "four levels and at each level below the first" do
    runs = TreeRecoverySimulation.runs
    (1..4).each do |level|
      gains = runs.map { |run| run[:held_out][level] }
      TreeRecoverySimulation.report(format("level %d held-out log score, shrunken less raw: better in %d of %d runs; " \
                                           "min %.1f, median %.1f, max %.1f nats", level, gains.count(&:positive?),
                                           runs.size, gains.min, gains.sort[gains.size / 2], gains.max))
    end
    losses = runs.each_with_index.flat_map do |run, index|
      total = run[:held_out].values.sum
      below = run[:held_out].select { |level, gain| level > 1 && !gain.positive? }
      (total.positive? ? [] : [ "run #{index} summed over the levels: #{total}" ]) +
        below.map { |level, gain| "run #{index} level #{level}: #{gain}" }
    end
    expect(losses).to be_empty, losses.join("\n")
  end
end
