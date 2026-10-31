# frozen_string_literal: true

# SL15: the pins of HierarchicalRates' outputs on invented counts. Every pinned value was checked by hand first: an
# independent reference at 60 significant digits (its own log-gamma by Stirling's series after an upward shift, its own
# power series for the regularized incomplete gamma, exact fractions for the outputs of plain arithmetic, and the exact
# binary value of every Float input) gave the digits below, and this file ran red against them before the module
# existed. A pinned value is never edited.
#
# One tolerance rule (SL15) for every Float this file compares with a pinned value. An output that goes through
# Math.lgamma (a node's hot share, ratio and tail, a log-likelihood, a profile point, the boundary statistic and its
# p-value) is compared at 1e-9 relative, LGAMMA_RELATIVE: lgamma's last bit can differ between math libraries, and at a
# strength of 4096 one ulp of lgamma is about 1e-12 of a log-likelihood. An output of plain arithmetic, log and exp
# (the leave-one-out share, Marshall's variance, the smoothed strength) and a grid point are compared at 1e-12
# relative. A pinned 0.0, 1.0 or infinity is compared exactly, whichever the output; a Boolean and a size must stay
# equal.
#
# The fit fixtures were chosen so that no rounding can move a choice: the best grid point beats the next by at least
# 2e-4 in log-likelihood and every profile point sits at least 0.025 from the 1.92 cut.
RSpec.describe "SkiftetStatistical::HierarchicalRates pins" do
  def rates = SkiftetStatistical::HierarchicalRates

  def lgamma_relative = 1e-9

  def arithmetic_relative = 1e-12

  def expect_pin(actual, pinned, relative, label)
    expect(actual).to be_a(Float), "#{label}: #{actual.inspect} is not a Float"
    if pinned.zero? || pinned == 1.0 || pinned.infinite?
      expect(actual).to eq(pinned), "#{label}: #{actual} is not exactly #{pinned}"
    else
      error = (actual - pinned).abs / pinned.abs
      expect(error).to be <= relative, "#{label}: #{actual} is #{error} from #{pinned}"
    end
  end

  def pairs(list) = list.map { |count, expected, siblings| { count: count, expected: expected, siblings: siblings } }

  # (count, expected, siblings). The last pair's parent has one child with data, so the fit and both checks leave it out.
  def mixed_level = [ [ 3.0, 5.2, 3 ], [ 11.4, 4.1, 3 ], [ 0.0, 1.8, 3 ], [ 7.75, 6.3, 3 ], [ 2.2, 2.9, 4 ],
                      [ 16.0, 7.7, 4 ], [ 1.0, 3.6, 4 ], [ 5.5, 2.4, 4 ], [ 9.0, 9.9, 2 ], [ 0.5, 1.2, 2 ],
                      [ 40.0, 3.0, 1 ] ]

  # Twenty small nodes, two of them far above what is expected.
  def two_spots = [ [ 1.0, 1.4 ], [ 2.0, 1.1 ], [ 0.0, 0.9 ], [ 1.5, 1.6 ], [ 3.0, 2.2 ], [ 0.0, 1.3 ], [ 2.0, 1.7 ],
                    [ 1.0, 0.8 ], [ 0.5, 1.2 ], [ 2.5, 2.0 ], [ 1.0, 1.5 ], [ 9.0, 1.3 ], [ 0.0, 1.0 ], [ 2.0, 2.4 ],
                    [ 11.5, 1.9 ], [ 1.0, 1.1 ], [ 3.0, 2.6 ], [ 0.75, 0.7 ], [ 1.0, 1.8 ], [ 2.0, 1.5 ] ]
                  .map { |count, expected| [ count, expected, 3 ] }

  # Counts at (or a little above) what is expected: less spread than Poisson.
  def even_level = [ [ 4.0, 4.0, 2 ], [ 6.0, 6.0, 2 ], [ 2.5, 2.5, 3 ], [ 7.2, 7.0, 3 ] ]

  def hot_grid = [ 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005 ]

  describe ".leave_one_out_share" do
    it "SL15 pins the share of whole counts: (120 - 18 + 16 * 0.125) / (950 - 75 + 16) = 104 / 891" do
      share = rates.leave_one_out_share(parent_count: 120.0, parent_total: 950.0, node_count: 18.0, node_total: 75.0,
                                        prior_share: 0.125, prior_strength: 16.0)
      expect_pin(share, 0.11672278338945006, arithmetic_relative, "whole")
    end

    it "SL15 pins the share of a node standing in its parent by a share of 0.35, multiplied in by the caller" do
      share = rates.leave_one_out_share(parent_count: 33.6, parent_total: 412.8, node_count: 0.35 * 6.0,
                                        node_total: 0.35 * 61.0, prior_share: 0.0625, prior_strength: 16.0)
      expect_pin(share, 0.079764388268499204, arithmetic_relative, "by share")
    end
  end

  describe ".node" do
    {
      "a whole count at the default hot share" =>
        [ { count: 9.0, expected: 4.0, strength: 16.0, threshold: 1.25, parent_tail: 0.0, parent_ratio: 1.1 },
          { ratio: 1.25, hot: 1.0, tail: 0.65610369595691418 } ],
      "a real count under a weak hot share, the parent's tail mixed in" =>
        [ { count: 3.7, expected: 1.9, strength: 2.0**1.5, hot_share: 0.1, threshold: 1.25, parent_tail: 1.0,
            parent_ratio: 1.4 },
          { ratio: 1.0338730021218191, hot: 0.088981123349663677, tail: 0.98369883586769639 } ],
      "a node with no count and nothing expected, which keeps its prior" =>
        [ { count: 0.0, expected: 0.0, strength: 8.0, hot_share: 0.2, threshold: 1.25, parent_tail: 1.0,
            parent_ratio: 0.9 },
          { ratio: 1.0, hot: 0.2, tail: 0.82723202808615033 } ],
      "the grid's top strength, where the slab is nearly the null" =>
        [ { count: 40.0, expected: 12.5, strength: 4096.0, hot_share: 0.02, threshold: 1.25, parent_tail: 0.0,
            parent_ratio: 1.25 },
          { ratio: 1.0001457678747861, hot: 0.021777720493047708, tail: 0.014460131098342161 } ],
      "a count far above what is expected under a weak strength: decisive" =>
        [ { count: 25.5, expected: 6.0, strength: 0.5, hot_share: 0.05, threshold: 1.25, parent_tail: 0.0,
            parent_ratio: 0.8 },
          { ratio: 3.9999687899133682, hot: 0.99998959663778941, tail: 0.99996676843177607 } ],
      "a count far below what is expected, which the wide slab also explains better than the null" =>
        [ { count: 1.0, expected: 7.5, strength: 1.0, hot_share: 0.3, threshold: 1.25, parent_tail: 1.0,
            parent_ratio: 1.6 },
          { ratio: 0.30051469205819533, hot: 0.91471155653928303, tail: 0.094417507577147316 } ]
    }.each do |label, (arguments, pinned)|
      it "SL15 pins #{label}" do
        node = rates.node(**arguments)
        expect(node.keys).to eq(%i[ratio hot tail])
        pinned.each { |key, value| expect_pin(node.fetch(key), value, lgamma_relative, "#{label} #{key}") }
      end
    end
  end

  describe ".fit_strength" do
    def expect_fit(fit, pinned)
      expect(fit.keys).to eq(%i[strength hot_share log_likelihood at_upper_bound interval profile])
      expect_pin(fit[:strength], pinned[:strength], arithmetic_relative, "strength")
      expect(fit[:hot_share]).to eq(pinned[:hot_share])
      expect_pin(fit[:log_likelihood], pinned[:log_likelihood], lgamma_relative, "log_likelihood")
      expect(fit[:at_upper_bound]).to be(pinned[:at_upper_bound])
      expect(fit[:interval].size).to eq(2)
      fit[:interval].zip(pinned[:interval]).each { |actual, value| expect_pin(actual, value, arithmetic_relative, "interval") }
      expect(fit[:profile].size).to eq(57)
      pinned.fetch(:profile, {}).each do |index, (strength, value)|
        expect_pin(fit[:profile][index][0], strength, arithmetic_relative, "profile #{index} strength")
        expect_pin(fit[:profile][index][1], value, lgamma_relative, "profile #{index}")
      end
    end

    it "SL15 pins a level fitted at hot share 1, one lone pair left out" do
      expect_fit(rates.fit_strength(pairs: pairs(mixed_level)),
                 strength: 2.8284271247461903, hot_share: 1.0, log_likelihood: -24.969696154912956,
                 at_upper_bound: false, interval: [ 0.7071067811865476, 22.627416997969522 ],
                 profile: { 0 => [ 0.25, -30.984864320512724 ], 14 => [ 2.8284271247461903, -24.969696154912956 ],
                            28 => [ 32.0, -27.100980718974363 ], 42 => [ 362.03867196751236, -28.107929053650265 ],
                            56 => [ 4096.0, -28.223197957033208 ] })
    end

    it "SL15 pins a level with two hot spots, its hot share fitted on an eight-point grid" do
      expect_fit(rates.fit_strength(pairs: pairs(two_spots), hot_grid: hot_grid),
                 strength: 0.3535533905932738, hot_share: 0.2, log_likelihood: -36.763946127545699,
                 at_upper_bound: false, interval: [ 0.25, 1.189207115002721 ],
                 profile: { 0 => [ 0.25, -36.817358738905613 ], 2 => [ 0.3535533905932738, -36.763946127545699 ],
                            14 => [ 2.8284271247461903, -41.025980071404542 ], 28 => [ 32.0, -46.451998947872220 ],
                            42 => [ 362.03867196751236, -46.888618323917142 ],
                            56 => [ 4096.0, -46.918293622649128 ] })
    end

    it "SL15 pins a level whose nodes do not differ, fitted at the grid's top" do
      expect_fit(rates.fit_strength(pairs: pairs(even_level), hot_grid: [ 0.5, 1.0 ]),
                 strength: 4096.0, hot_share: 0.5, log_likelihood: -6.7933925689908438, at_upper_bound: true,
                 interval: [ 0.42044820762685725, 4096.0 ])
    end
  end

  describe ".smooth_strength" do
    it "SL15 pins a smoothed strength at the default weight" do
      expect_pin(rates.smooth_strength(previous: 2.0**3.5, fitted: 3.0), 8.6757741695815893, arithmetic_relative,
                 "default weight")
    end

    it "SL15 pins a smoothed hot share at another weight" do
      expect_pin(rates.smooth_strength(previous: 0.2, fitted: 0.05, weight: 0.35), 0.12311444133449164,
                 arithmetic_relative, "weight 0.35")
    end
  end

  describe ".marshall" do
    it "SL15 pins Marshall's variance beside the fit, on the pairs the fit keeps" do
      result = rates.marshall(pairs: pairs(mixed_level))
      expect(result.keys).to eq(%i[variance strength])
      expect_pin(result[:variance], 0.36029167334494817, arithmetic_relative, "variance")
      expect_pin(result[:strength], 1.0 / 0.36029167334494817, arithmetic_relative, "strength")
    end

    it "SL15 pins Marshall's variance on twenty small nodes" do
      expect_pin(rates.marshall(pairs: pairs(two_spots))[:variance], 2.0947126035013032, arithmetic_relative,
                 "variance")
    end

    it "SL15 pins a floored variance and its infinite strength" do
      result = rates.marshall(pairs: pairs(even_level))
      expect_pin(result[:variance], 0.0, arithmetic_relative, "variance")
      expect_pin(result[:strength], Float::INFINITY, arithmetic_relative, "strength")
    end
  end

  describe ".boundary_test" do
    it "SL15 pins the test at the fitted strength, the lone pair left out" do
      result = rates.boundary_test(pairs: pairs(mixed_level), strength: 2.0**1.5)
      expect(result.keys).to eq(%i[statistic p_value])
      expect_pin(result[:statistic], 6.5299418202114834, lgamma_relative, "statistic")
      expect_pin(result[:p_value], 0.0053036713220987509, lgamma_relative, "p_value")
    end

    it "SL15 pins the test at a fitted strength and hot share" do
      result = rates.boundary_test(pairs: pairs(two_spots), strength: 2.0**-1.5, hot_share: 0.2)
      expect_pin(result[:statistic], 20.314283990954331, lgamma_relative, "statistic")
      expect_pin(result[:p_value], 3.2854104225778056e-06, lgamma_relative, "p_value")
    end

    it "SL15 pins a negative statistic and its p-value of 1" do
      result = rates.boundary_test(pairs: pairs(even_level), strength: 2.0)
      expect_pin(result[:statistic], -5.0445463328868941, lgamma_relative, "statistic")
      expect_pin(result[:p_value], 1.0, lgamma_relative, "p_value")
    end
  end

  # The leaf fit (SL13 (g), decided at the L1 barrier review on 2026-10-31): each parent's null rate is fitted with the
  # leaves' strength and hot share, the leaves weighted by their posterior probability of being null, instead of read
  # from a leave-one-out share. Every pinned digit below is the reference's, not the module's: the same independent
  # 60-digit arithmetic as above, its fixed points iterated to a change of 1e-45 (the parent that does not settle is
  # pinned at the reference's own 50th step). The module stops at a change of 1e-10 of the rate, which leaves it within
  # 3e-11 of the fixed point here: at every pinned fixed point each step shrinks the change to at most 0.22 of the one
  # before. Each grid choice was read off the reference's whole grid (in doubles) and checked in 60 digits at its best
  # two points: the best beats the next by at least 0.04 in log-likelihood.
  describe ".fit_leaves" do
    def leaf_parents(list) = list.map { |leaves| leaves.map { |count, total| { count: count, total: total } } }

    def expect_leaves(fit, pinned)
      expect(fit.keys).to eq(%i[strength hot_share log_likelihood at_upper_bound converged rates hot])
      expect_pin(fit[:strength], pinned[:strength], arithmetic_relative, "strength")
      expect(fit[:hot_share]).to eq(pinned[:hot_share])
      expect_pin(fit[:log_likelihood], pinned[:log_likelihood], lgamma_relative, "log_likelihood")
      expect(fit[:at_upper_bound]).to be(pinned[:at_upper_bound])
      expect(fit[:converged]).to eq(Array.new(fit[:rates].size, true))
      pinned.fetch(:rates).each do |parent, rate|
        expect_pin(fit[:rates][parent], rate, pinned.fetch(:rate_relative, lgamma_relative), "rate #{parent}")
      end
      pinned.fetch(:hot).each do |(parent, leaf), hot|
        expect_pin(fit[:hot][parent][leaf], hot, lgamma_relative, "hot #{parent}/#{leaf}")
      end
    end

    # (count, total) per leaf. Four parents of three to five leaves, a leaf well above its siblings in two of them,
    # real counts (2.1, 7.5), and a parent with one leaf, which the fit leaves out.
    def small_level = [ [ [ 3.0, 30.0 ], [ 2.0, 25.0 ], [ 14.0, 40.0 ], [ 4.0, 35.0 ] ],
                        [ [ 6.0, 50.0 ], [ 2.1, 20.0 ], [ 5.0, 45.0 ] ],
                        [ [ 0.0, 12.0 ], [ 1.0, 15.0 ], [ 7.5, 10.0 ], [ 1.0, 18.0 ], [ 2.0, 20.0 ] ],
                        [ [ 10.0, 80.0 ], [ 12.0, 90.0 ], [ 9.0, 70.0 ] ],
                        [ [ 4.0, 9.0 ] ] ]

    def small_grid = [ 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0 ]

    def small_hot_shares = [ 0.4, 0.2, 0.1, 0.05 ]

    it "SL15 pins a level at hot share 0: the pooled rates, every posterior 0.0, the all-Poisson likelihood, and a " \
       "flat profile that ties to the larger strength" do
      fit = rates.fit_leaves(parents: leaf_parents([ [ [ 2.0, 4.0 ], [ 6.0, 4.0 ] ], [ [ 0.0, 1.5 ], [ 1.0, 2.5 ] ] ]),
                             grid: [ 2.0, 8.0 ], hot_shares: [ 0.0 ])
      # Rates 8 / 8 and 1 / 4; expected 4, 4, 0.375 and 0.625; the log-likelihood is
      # (2 + 6) ln 4 - 8 - ln 2! - ln 6! - 0.375 + ln 0.625 - 0.625 = 8 ln 4 - 9 - ln 1440 + ln 0.625.
      expect_leaves(fit, strength: 8.0, hot_share: 0.0, log_likelihood: -5.6520471328566568, at_upper_bound: true,
                         rates: { 0 => 1.0, 1 => 0.25 }, rate_relative: arithmetic_relative, hot: {})
      expect(fit[:hot]).to eq([ [ 0.0, 0.0 ], [ 0.0, 0.0 ] ])
    end

    it "SL15 pins one parent's null rate at a given strength and hot share: the fixed point, a real count among its " \
       "leaves" do
      fit = rates.fit_leaves(parents: leaf_parents([ [ [ 1.0, 2.0 ], [ 12.0, 2.5 ], [ 2.4, 3.0 ], [ 3.0, 2.0 ] ] ]),
                             grid: [ 2.0 ], hot_shares: [ 0.2 ])
      expect_leaves(fit, strength: 2.0, hot_share: 0.2, log_likelihood: -12.573901991407499, at_upper_bound: true,
                         rates: { 0 => 0.93969181692155423 },
                         hot: { [ 0, 0 ] => 0.18324755633581008, [ 0, 1 ] => 0.98689295126009857,
                                [ 0, 2 ] => 0.14378747590663737, [ 0, 3 ] => 0.15174961629578557 })
    end

    it "SL15 pins a small level's fit on a 7 x 4 grid, the parent with one leaf answered beside it" do
      fit = rates.fit_leaves(parents: leaf_parents(small_level), grid: small_grid, hot_shares: small_hot_shares)
      expect_leaves(fit, strength: 0.125, hot_share: 0.1, log_likelihood: -35.302021023226786, at_upper_bound: false,
                         rates: { 0 => 0.17314397873277912, 1 => 0.11392509525597098, 2 => 0.063431630645454901,
                                  3 => 0.12916781442696132 },
                         hot: { [ 0, 0 ] => 0.024276360118616055, [ 0, 2 ] => 0.094426670540951427,
                                [ 1, 1 ] => 0.017318660407721436, [ 2, 0 ] => 0.15698020261875109,
                                [ 2, 2 ] => 0.99761460712245154, [ 3, 2 ] => 0.0083350187728200333,
                                [ 4, 0 ] => 0.012480799433646678 })
      # The lone parent's rate is its leaf's own, 4 / 9, as plain arithmetic.
      expect_pin(fit[:rates][4], 4.0 / 9.0, arithmetic_relative, "lone rate")
      expect(fit[:hot].map(&:size)).to eq([ 4, 3, 5, 3, 1 ])
    end

    it "SL15 pins a parent that has not settled after LEAF_STEPS steps: its rate after the 50th, not converged" do
      # Its leaf of 12 sits near the line between null and hot, so each step moves its weight a little: the 50th step
      # still moves the rate by 4.2e-9 of itself, 42 times LEAF_CHANGE, and the reference settles only after 288 steps,
      # 1.2e-8 below the rate pinned here (the reference's own 50th step).
      leaves = [ [ 5.0, 900.0 ], [ 1.0, 750.0 ], [ 12.0, 680.0 ], [ 1.0, 660.0 ], [ 2.0, 310.0 ] ]
      fit = rates.fit_leaves(parents: leaf_parents([ leaves ]), grid: [ 0.25 ], hot_shares: [ 0.02 ])
      expect(fit[:converged]).to eq([ false ])
      expect_pin(fit[:rates][0], 0.0057431689633961154, lgamma_relative, "rate")
      expect_pin(fit[:hot][0][2], 0.27802779791898702, lgamma_relative, "hot")
      expect_pin(fit[:hot][0][0], 0.0034426010923924453, lgamma_relative, "hot")
      expect_pin(fit[:log_likelihood], -15.611826209123217, lgamma_relative, "log_likelihood")
    end

    # 50 parents of five leaves, generated from round values: parent p's null rate is 0.05, 0.1, 0.15 or 0.2 (p mod 4)
    # and its leaves' totals 100 to 500 (rotated); in every fifth parent one leaf, a different one each time, is hot,
    # its rate five times its parent's, so 10 of the 250 leaves are hot: a hot share of 0.04. Each count is Poisson, by
    # Knuth's product of uniforms from Random.new(4).
    def generated_rates = [ 0.05, 0.1, 0.15, 0.2 ]

    def hot_leaf(parent) = (parent % 5).zero? ? (parent / 5) % 5 : nil

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

    def generated_level
      rng = Random.new(4)
      Array.new(50) do |parent|
        Array.new(5) do |leaf|
          total = 100.0 * (1 + ((leaf + parent) % 5))
          mean = total * generated_rates[parent % 4] * (leaf == hot_leaf(parent) ? 5.0 : 1.0)
          { count: knuth(rng, mean).to_f, total: total }
        end
      end
    end

    def leaf_hot_grid = [ 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005 ]

    it "SL15 pins the generated level where 1 of 5 leaves is hot in every fifth parent, on the whole grid" do
      level = generated_level
      # The fixture itself: a change in how it is drawn shows here first.
      expect(level.flatten.sum { |leaf| leaf[:count] }).to eq(10_567.0)

      fit = rates.fit_leaves(parents: level, hot_shares: leaf_hot_grid)
      expect_leaves(fit, strength: 2.0**-1.5, hot_share: 0.05, log_likelihood: -820.35019070232997,
                         at_upper_bound: false,
                         rates: { 0 => 0.059298634209110246, 1 => 0.099355011869473139, 5 => 0.11002593023986243 },
                         hot: { [ 0, 0 ] => 0.99993618941295859, [ 0, 1 ] => 0.011947629600024736,
                                [ 1, 4 ] => 0.014897763600165977, [ 5, 2 ] => 0.019447171285825137 })
    end

    it "SL13(g) reads the generated level's hot leaves and null rates, where a leave-one-out fit overstates the hot " \
       "share" do
      level = generated_level
      fit = rates.fit_leaves(parents: level, hot_shares: leaf_hot_grid)
      level.each_index do |parent|
        expect(fit[:rates][parent] / generated_rates[parent % 4]).to be_between(1 / 1.3, 1.3), "rate #{parent}"
        fit[:hot][parent].each_with_index do |hot, leaf|
          if leaf == hot_leaf(parent)
            expect(hot).to be > 0.99, "hot leaf #{parent}/#{leaf}: #{hot}"
          else
            expect(hot).to be < 0.2, "null leaf #{parent}/#{leaf}: #{hot}"
          end
        end
      end
      # The grid's two hot shares around the true 0.04.
      expect(fit[:hot_share]).to be_between(0.02, 0.05)

      # The same leaves on leave-one-out expected counts: each hot leaf lifts its four siblings' share, so they read
      # cold, and the fit reads a hot share of 0.2, five times the true one (its best beats every other hot share's by
      # more than 10 in log-likelihood).
      pairs = level.flat_map do |leaves|
        count = leaves.sum { |leaf| leaf[:count] }
        total = leaves.sum { |leaf| leaf[:total] }
        leaves.map do |leaf|
          share = rates.leave_one_out_share(parent_count: count, parent_total: total, node_count: leaf[:count],
                                            node_total: leaf[:total], prior_share: 0.0, prior_strength: 0.0)
          { count: leaf[:count], expected: leaf[:total] * share, siblings: leaves.size }
        end
      end
      expect(rates.fit_strength(pairs: pairs, hot_grid: leaf_hot_grid)[:hot_share]).to eq(0.2)
    end

    it "SL6 answers each leaf's posterior as node's hot at its count and expected total * rate, bit for bit" do
      fit = rates.fit_leaves(parents: leaf_parents(small_level), grid: small_grid, hot_shares: small_hot_shares)
      small_level.each_with_index do |leaves, parent|
        leaves.each_with_index do |(count, total), leaf|
          node = rates.node(count: count, expected: total * fit[:rates][parent], strength: fit[:strength],
                            hot_share: fit[:hot_share], threshold: 1.25, parent_tail: 0.0, parent_ratio: 1.0)
          expect(fit[:hot][parent][leaf]).to eq(node[:hot])
        end
      end
    end

    it "SL7 maximises fit_strength's likelihood: at the fitted rates the two read the same log-likelihood, bit for bit" do
      level = generated_level
      fit = rates.fit_leaves(parents: level, hot_shares: leaf_hot_grid)
      pairs = level.each_with_index.flat_map do |leaves, parent|
        leaves.map do |leaf|
          { count: leaf[:count], expected: leaf[:total] * fit[:rates][parent], siblings: leaves.size }
        end
      end
      profile = rates.fit_strength(pairs: pairs, hot_grid: [ fit[:hot_share] ])[:profile]
      expect(profile[rates::GRID.index(fit[:strength])][1]).to eq(fit[:log_likelihood])
    end

    it "SL7 leaves a parent with one leaf out of the fit: without it every other answer is the same, bit for bit" do
      with = rates.fit_leaves(parents: leaf_parents(small_level), grid: small_grid, hot_shares: small_hot_shares)
      without = rates.fit_leaves(parents: leaf_parents(small_level[0, 4]), grid: small_grid,
                                 hot_shares: small_hot_shares)
      expect(with.except(:converged, :rates, :hot)).to eq(without.except(:converged, :rates, :hot))
      %i[converged rates hot].each { |key| expect(with[key][0, 4]).to eq(without[key]) }
      expect(with[:rates][4]).to eq(4.0 / 9.0)
      expect(with[:converged][4]).to be(true)
    end

    it "SL6 gives a parent with no count a null rate of 0 and its leaves the hot share itself" do
      fit = rates.fit_leaves(parents: leaf_parents([ [ [ 0.0, 3.0 ], [ 0.0, 1.0 ] ], [ [ 1.0, 1.0 ], [ 2.0, 1.0 ] ] ]),
                             grid: [ 1.0 ], hot_shares: [ 0.3 ])
      expect(fit[:rates][0]).to eq(0.0)
      fit[:hot][0].each { |hot| expect(hot).to be_within(1e-15).of(0.3) }
      expect(fit[:converged]).to eq([ true, true ])
    end

    it "SL6 takes whole numbers given as Integers as the same real numbers" do
      integers = [ [ [ 3, 30 ], [ 2, 25 ], [ 14, 40 ] ], [ [ 6, 50 ], [ 5, 45 ] ] ]
      floats = integers.map { |leaves| leaves.map { |pair| pair.map(&:to_f) } }
      expect(rates.fit_leaves(parents: leaf_parents(integers), grid: [ 1, 4 ], hot_shares: [ 0.1 ]))
        .to eq(rates.fit_leaves(parents: leaf_parents(floats), grid: [ 1.0, 4.0 ], hot_shares: [ 0.1 ]))
    end

    it "SL6 does not take a step that no rate can explain: the parent keeps its rate, finite, and is not converged" do
      # The first step weighs the leaf with no count so far above the other that the rate would fall to 0, which no
      # count of 10,000 can come from.
      fit = rates.fit_leaves(parents: leaf_parents([ [ [ 0.0, 1e6 ], [ 10_000.0, 1.0 ] ] ]), grid: [ 0.25 ],
                             hot_shares: [ 0.5 ])
      expect(fit[:rates]).to eq([ 10_000.0 / 1_000_001.0 ])
      expect(fit[:converged]).to eq([ false ])
      expect(fit[:log_likelihood]).to be_finite
      expect(fit[:hot].flatten).to all(be_between(0.0, 1.0))
    end

    it "SL7 refuses a leaf fit outside its domain" do
      good = leaf_parents([ [ [ 1.0, 2.0 ], [ 3.0, 4.0 ] ] ])
      [
        [ { parents: good, hot_shares: [ 1.0 ] }, /below 1/ ],
        [ { parents: good, hot_shares: [ 0.2, 1.0 ] }, /below 1/ ],
        [ { parents: good, hot_shares: [] }, /hot_shares/ ],
        [ { parents: good, hot_shares: nil }, /hot_shares/ ],
        [ { parents: good, hot_shares: [ 1.5 ] }, /hot_shares/ ],
        [ { parents: good, hot_shares: [ 0.2 ], grid: [] }, /grid/ ],
        [ { parents: good, hot_shares: [ 0.2 ], grid: [ 2.0, 1.0 ] }, /grid must rise/ ],
        [ { parents: good, hot_shares: [ 0.2 ], grid: [ 1.0, 1.0 ] }, /grid must rise/ ],
        [ { parents: good, hot_shares: [ 0.2 ], grid: [ 0.0, 1.0 ] }, /grid strength/ ],
        [ { parents: [], hot_shares: [ 0.2 ] }, /no parent/ ],
        [ { parents: leaf_parents([ [ [ 1.0, 2.0 ] ] ]), hot_shares: [ 0.2 ] }, /no parent/ ],
        [ { parents: nil, hot_shares: [ 0.2 ] }, /parents/ ],
        [ { parents: [ [] ], hot_shares: [ 0.2 ] }, /leaf/ ],
        [ { parents: [ [ { count: 1.0 } ] ], hot_shares: [ 0.2 ] }, /total/ ],
        [ { parents: [ [ { total: 1.0 } ] ], hot_shares: [ 0.2 ] }, /count/ ],
        [ { parents: [ [ [ 1.0, 2.0 ] ] ], hot_shares: [ 0.2 ] }, /Hash/ ],
        [ { parents: leaf_parents([ [ [ 1.0, 0.0 ], [ 3.0, 4.0 ] ] ]), hot_shares: [ 0.2 ] }, /total/ ],
        [ { parents: leaf_parents([ [ [ -1.0, 2.0 ], [ 3.0, 4.0 ] ] ]), hot_shares: [ 0.2 ] }, /count/ ],
        [ { parents: leaf_parents([ [ [ Float::NAN, 2.0 ], [ 3.0, 4.0 ] ] ]), hot_shares: [ 0.2 ] }, /count/ ]
      ].each do |arguments, message|
        expect { rates.fit_leaves(**arguments) }.to raise_error(ArgumentError, message), "accepted #{arguments}"
      end
    end
  end

  # The edges of the contract that no pin above reaches.
  describe "edges" do
    def node_arguments = { count: 6.0, expected: 2.5, strength: 8.0, hot_share: 0.5, threshold: 1.25, parent_tail: 0.0,
                           parent_ratio: 1.0 }

    it "SL6 takes whole numbers given as Integers as the same real numbers" do
      expect(rates.node(**node_arguments, count: 6, strength: 8)).to eq(rates.node(**node_arguments))
    end

    it "SL6 takes no weight keyword: a caller multiplies a node's share into its count and total" do
      expect { rates.node(**node_arguments, weight: 0.5) }.to raise_error(ArgumentError)
      expect do
        rates.leave_one_out_share(parent_count: 4.0, parent_total: 9.0, node_count: 1.0, node_total: 2.0, prior_share: 0.5,
                                  prior_strength: 1.0, weight: 0.5)
      end.to raise_error(ArgumentError)
    end

    it "SL6 refuses a count above 0 where nothing is expected" do
      expect { rates.node(**node_arguments, expected: 0.0) }.to raise_error(ArgumentError, /expected/)
    end

    it "SL6 refuses inputs outside the rule's domain" do
      [
        { count: -0.5 }, { count: Float::NAN }, { count: "6" }, { count: nil }, { expected: -1.0 },
        { expected: Float::INFINITY }, { strength: 0.0 }, { strength: Float::INFINITY }, { hot_share: 1.5 },
        { hot_share: -0.1 }, { threshold: -1.0 }, { parent_tail: 1.2 }, { parent_ratio: 0.0 }
      ].each do |change|
        expect { rates.node(**node_arguments, **change) }.to raise_error(ArgumentError), "accepted #{change}"
      end
      expect do
        rates.leave_one_out_share(parent_count: 1.0, parent_total: 2.0, node_count: 1.0, node_total: 2.0, prior_share: 0.5,
                                  prior_strength: 0.0)
      end.to raise_error(ArgumentError, /total/)
    end

    it "SL6 reads a number as the other modules do: a complex number or one no Float holds is refused" do
      # A Complex is not real even with no imaginary part; Integers and Rationals beyond a Float's range are not finite.
      [ Complex(6, 0), Complex(6, 1), Complex(0.5, 0), 10**400, Rational(10**400, 3) ].each do |value|
        expect { rates.node(**node_arguments, count: value) }
          .to raise_error(ArgumentError, /count must be a finite number/), "accepted #{value.inspect}"
        expect { rates.smooth_strength(previous: 2.0, fitted: value) }
          .to raise_error(ArgumentError, /fitted must be a finite number/), "accepted #{value.inspect}"
        expect do
          rates.leave_one_out_share(parent_count: 4.0, parent_total: 9.0, node_count: 1.0, node_total: 2.0,
                                    prior_share: value, prior_strength: 1.0)
        end.to raise_error(ArgumentError, /prior_share must be a finite number/), "accepted #{value.inspect}"
        expect { rates.fit_strength(pairs: [ { count: 1.0, expected: value, siblings: 2 } ]) }
          .to raise_error(ArgumentError, /expected must be a finite number/), "accepted #{value.inspect}"
      end
      # A Rational within range is read as its Float, as an Integer is.
      expect(rates.node(**node_arguments, count: Rational(6), strength: Rational(16, 2))).to eq(rates.node(**node_arguments))
    end

    it "SL6 answers 0.0 for the hot share, and the parent's tail, when the hot share is 0" do
      node = rates.node(**node_arguments, hot_share: 0.0, parent_tail: 1.0)
      expect(node).to eq(ratio: 1.0, hot: 0.0, tail: 1.0)
    end

    it "SL7 refuses a fit with no pair left after the lone pairs" do
      expect { rates.fit_strength(pairs: []) }.to raise_error(ArgumentError, /pair/)
      expect { rates.fit_strength(pairs: pairs([ [ 4.0, 2.0, 1 ], [ 1.0, 3.0, 0 ] ])) }.to raise_error(ArgumentError, /pair/)
      expect { rates.marshall(pairs: pairs([ [ 4.0, 2.0, 1 ] ])) }.to raise_error(ArgumentError, /pair/)
      expect { rates.boundary_test(pairs: [], strength: 2.0) }.to raise_error(ArgumentError, /pair/)
    end

    it "SL7 refuses a pair without its siblings, an expected count of 0 or an empty hot grid" do
      expect { rates.fit_strength(pairs: [ { count: 1.0, expected: 2.0 } ]) }.to raise_error(ArgumentError, /siblings/)
      expect { rates.fit_strength(pairs: [ { count: 1.0, expected: 2.0, siblings: 2.5 } ]) }
        .to raise_error(ArgumentError, /siblings/)
      expect { rates.marshall(pairs: pairs([ [ 0.0, 0.0, 3 ], [ 1.0, 2.0, 3 ] ])) }.to raise_error(ArgumentError, /expected/)
      expect { rates.fit_strength(pairs: pairs(even_level), hot_grid: []) }.to raise_error(ArgumentError, /hot_grid/)
      expect { rates.fit_strength(pairs: pairs(even_level), hot_grid: [ 1.0, 2.0 ]) }.to raise_error(ArgumentError)
    end

    it "SL7 reads a flat likelihood as nodes that do not differ: a tie goes to the larger strength" do
      # At a hot share of 0 every node is null, so every strength gives the same log-likelihood, bit for bit.
      fit = rates.fit_strength(pairs: pairs(mixed_level), hot_grid: [ 0.0 ])
      expect(fit[:strength]).to eq(rates::GRID.last)
      expect(fit[:at_upper_bound]).to be(true)
      expect(fit[:interval]).to eq([ rates::GRID.first, rates::GRID.last ])
      expect(fit[:profile].map(&:last).uniq.size).to eq(1)
    end

    it "SL8 reads the fit's own likelihood: no strength on the grid gives a larger statistic than the fitted one" do
      fit = rates.fit_strength(pairs: pairs(two_spots), hot_grid: hot_grid)
      fitted = rates.boundary_test(pairs: pairs(two_spots), strength: fit[:strength], hot_share: fit[:hot_share])
      others = hot_grid.product(rates::GRID).map do |hot, strength|
        rates.boundary_test(pairs: pairs(two_spots), strength: strength, hot_share: hot)[:statistic]
      end
      expect(others.max).to eq(fitted[:statistic])
      # The fit's profile and the test share one likelihood: the statistic moves by twice the profile's difference.
      lower = rates.boundary_test(pairs: pairs(two_spots), strength: rates::GRID[0], hot_share: fit[:hot_share])
      gap = fit[:profile][2][1] - fit[:profile][0][1]
      expect(fitted[:statistic] - lower[:statistic]).to be_within(1e-12).of(2.0 * gap)
    end

    it "SL8 leaves out the fit's lone pairs in both checks" do
      kept = mixed_level.reject { |_count, _expected, siblings| siblings < 2 }
      expect(rates.marshall(pairs: pairs(mixed_level))).to eq(rates.marshall(pairs: pairs(kept)))
      expect(rates.boundary_test(pairs: pairs(mixed_level), strength: 4.0))
        .to eq(rates.boundary_test(pairs: pairs(kept), strength: 4.0))
    end
  end
end
