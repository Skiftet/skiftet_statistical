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
