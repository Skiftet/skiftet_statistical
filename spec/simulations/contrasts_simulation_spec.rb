# frozen_string_literal: true

# AdditiveContrasts.fit on generated cells whose variances are known (SL13 (h)), at the default 5 sweeps. Tagged
# `simulation: true`: excluded unless SIMULATIONS=1 (spec/spec_helper.rb). Every figure the examples measure is printed
# through the reporter (the JSON formatter keeps it under "messages") for the README's measured-results section.
#
# THE SHAPE, from round generic values only. One root; 400 nodes under it ("tree:1"); 6 leaves under each node
# ("tree:2", 2,400 leaves); 2 cells in each leaf, 4,800 cells. Two crossed groupings: "a" with 300 keys and "b" with
# 150, each cell's keys drawn uniformly, so the cells of a leaf differ in their crossed keys. A cell's sampling
# variance is drawn uniformly from [0.4, 1.4], so the cells' mean sampling variance is about 0.9. For each arm, the
# true contrast of a cell is
#
#   delta(cell) = root + u_tree1(node) + u_tree2(leaf) + u_a(key) + u_b(key),   each u ~ N(0, tau^2 of its term)
#
# and its estimate is delta(cell) plus N(0, its sampling variance). Three arms, fitted in one call:
#   x: every term at tau^2 = 4;
#   y: every term at 0;
#   z: mixed, "tree:1" and "a" at 0 beside "tree:2" at 4 and "b" at 8, so a term at 0 sits beside large ones: a cell
#      that did not add its crossed keys, or groupings that did not net each other out, would show here.
#
# "THE MEAN SAMPLING VARIANCE" is the mean of the sampling variances handed to `fit` (the known variances). A term at 4
# or more is also at least 4 times its groups' own mean sampling variance, the term the moment step subtracts (sum
# (K - 1) / scale over the groups' precisions, a node's precision carrying the true variance of the level under it):
# the example checks both before it measures, so the bound is tested inside the regime it names.
#
# Measured beside this shape and kept out of it (the handoff and the README carry the figures): with ONE cell a leaf, a
# leaf's deviation and its cell's noise cannot be told apart within the leaf, and at 5 sweeps the crossed grouping at
# 0 beside "b" at 8 reads about 0.6 to 1.2.
module ContrastsSimulation
  module_function

  ROOT_NODES = 400
  LEAVES_PER_NODE = 6
  CELLS_PER_LEAF = 2
  A_KEYS = 300
  B_KEYS = 150
  VARIANCE_RANGE = [ 0.4, 1.4 ].freeze
  TERMS = [ "tree:1", "tree:2", "a", "b" ].freeze
  TRUTH = {
    x: { "tree:1" => 4.0, "tree:2" => 4.0, "a" => 4.0, "b" => 4.0 },
    y: { "tree:1" => 0.0, "tree:2" => 0.0, "a" => 0.0, "b" => 0.0 },
    z: { "tree:1" => 0.0, "tree:2" => 4.0, "a" => 0.0, "b" => 8.0 }
  }.freeze
  ROOT_VALUE = { x: 0.3, y: -0.2, z: 0.1 }.freeze
  RUNS = 100
  SEED = 20_261_101

  # The K - 1 check's tree: 1,000 nodes of 2 leaves, one cell a leaf, no crossed grouping.
  PAIR_NODES = 1_000
  PAIR_TRUTH = { "tree:1" => 2.0, "tree:2" => 1.0 }.freeze
  PAIR_VARIANCE_RANGE = [ 0.5, 1.5 ].freeze
  PAIR_SEED = 20_261_102

  def normal(sampler, variance)
    variance.zero? ? 0.0 : Math.sqrt(variance) * sampler.gaussian
  end

  def uniform(rng, range)
    range[0] + ((range[1] - range[0]) * rng.rand)
  end

  # One run: { cells:, places:, variances: }, each place { node:, leaf:, a:, b: } beside its cell.
  def shape(seed)
    sampler = SkiftetStatistical::Sampler.new(Random.new(seed))
    rng = sampler.rng
    leaves = ROOT_NODES * LEAVES_PER_NODE
    places = Array.new(leaves * CELLS_PER_LEAF) do |index|
      leaf = index / CELLS_PER_LEAF
      { node: leaf / LEAVES_PER_NODE, leaf: leaf, a: rng.rand(A_KEYS), b: rng.rand(B_KEYS) }
    end
    variances = Array.new(places.size) { uniform(rng, VARIANCE_RANGE) }
    effects = TRUTH.to_h do |arm, truth|
      [ arm, {
        node: Array.new(ROOT_NODES) { normal(sampler, truth["tree:1"]) },
        leaf: Array.new(leaves) { normal(sampler, truth["tree:2"]) },
        a: Array.new(A_KEYS) { normal(sampler, truth["a"]) },
        b: Array.new(B_KEYS) { normal(sampler, truth["b"]) }
      } ]
    end
    cells = places.each_with_index.map do |place, index|
      contrasts = TRUTH.keys.to_h do |arm|
        u = effects[arm]
        delta = ROOT_VALUE[arm] + u[:node][place[:node]] + u[:leaf][place[:leaf]] + u[:a][place[:a]] + u[:b][place[:b]]
        [ arm, [ delta + normal(sampler, variances[index]), variances[index] ] ]
      end
      { id: index, contrasts: contrasts, tree: [ :root, "n#{place[:node]}", "l#{place[:leaf]}" ],
        groups: { "a" => place[:a], "b" => place[:b] } }
    end
    { cells: cells, places: places, variances: variances }
  end

  # Each term's groups' mean sampling variance under one arm's true variances: sum (K - 1) / scale, scale = sum over
  # sibling sets of (sum P - sum P^2 / sum P), P a group's precision.
  def group_sampling_variances(run, truth)
    precisions = run[:variances].map { |variance| 1.0 / variance }
    leaf = Array.new(ROOT_NODES * LEAVES_PER_NODE, 0.0)
    a = Array.new(A_KEYS, 0.0)
    b = Array.new(B_KEYS, 0.0)
    run[:places].each_with_index do |place, index|
      leaf[place[:leaf]] += precisions[index]
      a[place[:a]] += precisions[index]
      b[place[:b]] += precisions[index]
    end
    node = Array.new(ROOT_NODES, 0.0)
    leaf.each_with_index { |p, at| node[at / LEAVES_PER_NODE] += p / (1.0 + (truth["tree:2"] * p)) }
    sets = { "tree:1" => [ node ], "tree:2" => leaf.each_slice(LEAVES_PER_NODE).to_a,
             "a" => [ a.reject(&:zero?) ], "b" => [ b.reject(&:zero?) ] }
    sets.transform_values do |groups|
      scale = groups.sum { |set| set.sum - (set.sum { |p| p * p } / set.sum) }
      groups.sum { |set| set.size - 1 } / scale
    end
  end

  # Every run's fitted variances, the cells' mean sampling variance and the groups' ones, computed once for the file.
  def runs
    @runs ||= Array.new(RUNS) do |index|
      run = shape(SEED + index)
      answer = SkiftetStatistical::AdditiveContrasts.fit(cells: run[:cells], arms: TRUTH.keys)
      { fitted: answer[:variances], cell_mean: run[:variances].sum / run[:variances].size,
        groups: TRUTH.to_h { |arm, truth| [ arm, group_sampling_variances(run, truth) ] } }
    end
  end

  # One run of the K - 1 tree: its fitted { "tree:1" => Float, "tree:2" => Float }.
  def pair_run(seed)
    sampler = SkiftetStatistical::Sampler.new(Random.new(seed))
    nodes = Array.new(PAIR_NODES) { normal(sampler, PAIR_TRUTH["tree:1"]) }
    cells = Array.new(PAIR_NODES * 2) do |leaf|
      variance = uniform(sampler.rng, PAIR_VARIANCE_RANGE)
      estimate = 0.2 + nodes[leaf / 2] + normal(sampler, PAIR_TRUTH["tree:2"]) + normal(sampler, variance)
      { id: leaf, contrasts: { a: [ estimate, variance ] }, tree: [ :root, "n#{leaf / 2}", "l#{leaf}" ], groups: {} }
    end
    SkiftetStatistical::AdditiveContrasts.fit(cells: cells, arms: [ :a ])[:variances].transform_values { |v| v[:a] }
  end

  def report(text)
    RSpec.configuration.reporter.message("[contrasts simulation] #{text}")
  end
end

RSpec.describe "AdditiveContrasts simulation", simulation: true do
  it "SL13(h) recovers every variance of at least 4 times the mean sampling variance within a factor of 2, and " \
     "every variance of 0 under a quarter of it, in each of 100 seeded runs at the default 5 sweeps" do
    failures = []
    ContrastsSimulation.runs.each_with_index do |run, index|
      ContrastsSimulation::TRUTH.each do |arm, truth|
        ContrastsSimulation::TERMS.each do |term|
          value = run[:fitted].fetch(term).fetch(arm)
          if truth[term].zero?
            bound = run[:cell_mean] / 4.0
            failures << "run #{index} #{arm} #{term}: #{value} not under #{bound}" unless value < bound
          else
            premise = 4.0 * [ run[:cell_mean], run[:groups][arm][term] ].max
            raise "the shape leaves its regime: #{arm} #{term} #{truth[term]} < #{premise}" if truth[term] < premise

            unless value.between?(truth[term] / 2.0, truth[term] * 2.0)
              failures << "run #{index} #{arm} #{term}: #{value} not within a factor of 2 of #{truth[term]}"
            end
          end
        end
      end
    end

    ContrastsSimulation::TRUTH.each do |arm, truth|
      ContrastsSimulation::TERMS.each do |term|
        values = ContrastsSimulation.runs.map { |run| run[:fitted][term][arm] }
        groups = ContrastsSimulation.runs.map { |run| run[:groups][arm][term] }
        cells = ContrastsSimulation.runs.map { |run| run[:cell_mean] }
        ContrastsSimulation.report(format("%s %-6s truth %.1f: mean %.4f, min %.4f, max %.4f (cells' mean sampling " \
                                          "variance %.4f, the groups' %.4f)", arm, term, truth[term],
                                          values.sum / values.size, values.min, values.max, cells.sum / cells.size,
                                          groups.sum / groups.size))
      end
    end
    expect(failures).to be_empty, failures.first(10).join("\n")
  end

  it "SL13(h) reads each level's variance without bias where K - 1 matters most (two leaves a node): the mean of 100 " \
     "seeded runs within 5 % of the truth" do
    fits = Array.new(ContrastsSimulation::RUNS) do |index|
      ContrastsSimulation.pair_run(ContrastsSimulation::PAIR_SEED + index)
    end

    off = ContrastsSimulation::PAIR_TRUTH.filter_map do |term, truth|
      values = fits.map { |fit| fit.fetch(term) }
      mean = values.sum / values.size
      ContrastsSimulation.report(format("two leaves a node, %s truth %.1f: mean %.4f, min %.4f, max %.4f", term, truth,
                                        mean, values.min, values.max))
      "#{term}: mean #{mean} against #{truth}" unless (mean / truth).between?(0.95, 1.05)
    end
    expect(off).to be_empty, off.join("\n")
  end
end
