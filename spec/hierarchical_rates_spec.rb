# frozen_string_literal: true

# Acceptance for SL6-SL8: a node's Gamma-Poisson estimate under a two-groups prior, the level fit on a grid and the
# checks beside it. Every expected value below is worked out by hand (in the comments) or by the reference arithmetic
# in this file: exact Rational sums, finite products, Math, and 0.2.0's Gamma.regularized_upper where SL6 names it as
# the tail's Q. None of it calls HierarchicalRates. Counts are real numbers >= 0 throughout (SL6).
RSpec.describe "SkiftetStatistical::HierarchicalRates" do
  def rates = SkiftetStatistical::HierarchicalRates

  def relative(actual, expected)
    (actual - expected).abs / expected.abs
  end

  # Q(n, x) = e**-x * sum_{j < n} x**j / j! for a whole n (the Poisson tail), summed exactly in Rationals.
  def upper_for_whole(n, x)
    term = Rational(1)
    sum = Rational(1)
    (1...n).each do |j|
      term = term * Rational(x) / j
      sum += term
    end
    Math.exp(-x) * sum.to_f
  end

  # log NB(s; k, k / (k + E)) and log Poisson(s; E), written from SL6's formulas: s! is Gamma(s + 1), no rounding.
  def log_nb(count, expected, strength)
    Math.lgamma(count + strength)[0] - Math.lgamma(strength)[0] - Math.lgamma(count + 1.0)[0] +
      (strength * Math.log(strength / (strength + expected))) + (count * Math.log(expected / (strength + expected)))
  end

  def log_poisson(count, expected)
    (count * Math.log(expected)) - expected - Math.lgamma(count + 1.0)[0]
  end

  # SL7's marginal log-likelihood: sum over pairs of log[h * NB + (1 - h) * Poisson], mixed in logs.
  def log_likelihood(list, strength, hot)
    list.sum do |count, expected|
      slab = log_nb(count.to_f, expected.to_f, strength)
      next slab if hot == 1.0

      slab += Math.log(hot)
      null = Math.log(1.0 - hot) + log_poisson(count.to_f, expected.to_f)
      top = [ slab, null ].max
      top + Math.log(Math.exp(slab - top) + Math.exp(null - top))
    end
  end

  def poisson_log_likelihood(list)
    list.sum { |count, expected| log_poisson(count.to_f, expected.to_f) }
  end

  # The 57 strengths of SL7, 2**-2 to 2**12 in quarter powers of 2.
  def grid = (0..56).map { |i| 2.0**(-2 + (i / 4.0)) }

  # siblings: how many children of the pair's parent hold data, the pair's own node included. Three keeps a pair in.
  def pairs(list, siblings: 3)
    list.map { |count, expected| { count: count.to_f, expected: expected.to_f, siblings: siblings } }
  end

  # (count, expected) fixtures, invented. Their grid maxima were found by the reference above over the whole grid; the
  # best and second-best grid points differ by at least 3e-4 in log-likelihood and every profile point sits at least
  # 0.04 from the 1.92 cut, so no rounding difference can move an argmax or an interval end.
  def spread_out = [ [ 1, 4.0 ], [ 12, 6.0 ], [ 0, 2.5 ], [ 15, 8.0 ], [ 5, 5.0 ], [ 7, 3.0 ], [ 2, 7.0 ], [ 4, 4.5 ] ]
  def two_hot_spots = [ 3, 2, 4, 3, 1, 5, 2, 3, 4, 16, 3, 19 ].map { |count| [ count, 3.0 ] }
  def all_as_expected = [ [ 2, 2.0 ], [ 3, 3.0 ], [ 4, 4.0 ], [ 5, 5.0 ] ]
  def by_share = [ [ 1.5, 4.0 ], [ 8.25, 3.5 ], [ 0.0, 2.25 ], [ 11.75, 5.0 ], [ 4.4, 4.4 ], [ 0.6, 3.1 ] ]

  describe ".leave_one_out_share" do
    it "SL6 takes the node's own count and total out of its parent and smooths toward the prior share" do
      share = rates.leave_one_out_share(parent_count: 40.5, parent_total: 300.0, node_count: 4.5, node_total: 60.0,
                                        prior_share: 0.1, prior_strength: 50.0)
      # (40.5 - 4.5 + 50 * 0.1) / (300 - 60 + 50) = 41 / 290
      expect(relative(share, 41.0 / 290.0)).to be < 1e-12
    end

    it "SL6 takes a by-share count as the real number it is, with no rounding" do
      # A node standing in its parent by a share of 1/4, of a count of 3 and a total of 10: the caller multiplied it in.
      share = rates.leave_one_out_share(parent_count: 7.25, parent_total: 19.5, node_count: 0.75, node_total: 2.5,
                                        prior_share: 0.2, prior_strength: 4.0)
      # (7.25 - 0.75 + 4 * 0.2) / (19.5 - 2.5 + 4) = 7.3 / 21 = 73 / 210
      expect(relative(share, 73.0 / 210.0)).to be < 1e-12
    end
  end

  describe ".node" do
    it "SL6 gives count 6, expected 2.5, strength 8 a ratio of 14 / 10.5 and a hot share of exactly 1 by default" do
      node = rates.node(count: 6.0, expected: 2.5, strength: 8.0, threshold: 1.25, parent_tail: 0.0, parent_ratio: 1.0)

      expect(node.keys).to contain_exactly(:ratio, :hot, :tail)
      expect(node[:hot]).to eq(1.0)
      # (k + s) / (k + E) = 14 / 10.5
      expect(relative(node[:ratio], 14.0 / 10.5)).to be < 1e-12
      # tail = Q(k + s, (k + E) * L / parent_ratio) = Q(14, 10.5 * 1.25) = Q(14, 13.125)
      expect(relative(node[:tail], upper_for_whole(14, 13.125))).to be < 1e-9
    end

    it "SL6 reads the tail at the threshold over the parent's ratio, and ignores the parent's tail when hot is 1" do
      node = rates.node(count: 6.0, expected: 2.5, strength: 8.0, threshold: 1.25, parent_tail: 1.0, parent_ratio: 1.25)
      # Q(14, 10.5 * 1.25 / 1.25) = Q(14, 10.5); (1 - hot) * parent_tail = 0
      expect(relative(node[:tail], upper_for_whole(14, 10.5))).to be < 1e-9
      expect(relative(node[:ratio], 14.0 / 10.5)).to be < 1e-12
    end

    it "SL6 weighs the slab against the null by the hot share" do
      node = rates.node(count: 6.0, expected: 2.5, strength: 8.0, hot_share: 0.5, threshold: 1.25, parent_tail: 1.0,
                        parent_ratio: 1.0)
      # NB(6; 8, 8 / 10.5) = C(13, 6) * (16 / 21)**8 * (5 / 21)**6, exactly; Poisson(6; 2.5) = 2.5**6 e**-2.5 / 720.
      slab = (Rational(1716) * (Rational(16, 21)**8) * (Rational(5, 21)**6)).to_f
      null = (2.5**6) * Math.exp(-2.5) / 720.0
      hot = 0.5 * slab / ((0.5 * slab) + (0.5 * null)) # 0.56052209651155...
      tail = ((1.0 - hot) * 1.0) + (hot * upper_for_whole(14, 13.125))

      expect(relative(node[:hot], hot)).to be < 1e-9
      expect(relative(node[:ratio], (1.0 - hot) + (hot * 14.0 / 10.5))).to be < 1e-9
      expect(relative(node[:tail], tail)).to be < 1e-9
    end

    it "SL6 takes a real count of 2.4 through log-gamma, to its hand value" do
      node = rates.node(count: 2.4, expected: 2.5, strength: 8.0, hot_share: 0.3, threshold: 1.25, parent_tail: 0.0,
                        parent_ratio: 1.0)
      # NB(2.4; 8, p) = Gamma(10.4) / (Gamma(8) Gamma(3.4)) p**8 (1 - p)**2.4, and Gamma(10.4) / Gamma(3.4) is the
      # finite product 3.4 * 4.4 * ... * 9.4. Poisson(2.4; 2.5) = 2.5**2.4 e**-2.5 / Gamma(3.4), with
      # Gamma(3.4) = 2.4 * 1.4 * 0.4 * Gamma(0.4) and Gamma(0.4) = 2.218159543757688 (tables: 2.2181595437576882...).
      rising = [ 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4 ].reduce(:*)
      slab = rising / 5040.0 * ((8.0 / 10.5)**8) * ((2.5 / 10.5)**2.4)
      null = (2.5**2.4) * Math.exp(-2.5) / (2.4 * 1.4 * 0.4 * 2.218159543757688)
      hot = 0.3 * slab / ((0.3 * slab) + (0.7 * null)) # 0.27281178909858...

      expect(relative(node[:hot], hot)).to be < 1e-9
      expect(relative(node[:ratio], (1.0 - hot) + (hot * 10.4 / 10.5))).to be < 1e-9
      # parent_tail 0: the tail is hot * Q(10.4, 13.125), Q being 0.2.0's Gamma.regularized_upper (SL6)
      tail = hot * SkiftetStatistical::Gamma.regularized_upper(10.4, 13.125)
      expect(relative(node[:tail], tail)).to be < 1e-9
    end

    it "SL6 takes a count of 0" do
      node = rates.node(count: 0.0, expected: 2.5, strength: 8.0, hot_share: 0.5, threshold: 1.25, parent_tail: 1.0,
                        parent_ratio: 1.0)
      # NB(0; 8, p) = p**8 = (16 / 21)**8; Poisson(0; 2.5) = e**-2.5
      slab = (Rational(16, 21)**8).to_f
      hot = slab / (slab + Math.exp(-2.5)) # 0.58042791665...
      tail = (1.0 - hot) + (hot * upper_for_whole(8, 13.125))

      expect(relative(node[:hot], hot)).to be < 1e-9
      expect(relative(node[:ratio], (1.0 - hot) + (hot * 8.0 / 10.5))).to be < 1e-9
      expect(relative(node[:tail], tail)).to be < 1e-9
    end

    it "SL6 moves a two-groups leaf from a weak to a decisive hot share as its count rises" do
      hots = [ 2.0, 3.0, 4.5, 6.0, 7.25, 9.0, 10.5, 12.0 ].map do |count|
        rates.node(count: count, expected: 2.0, strength: 0.5, hot_share: 0.05, threshold: 1.25, parent_tail: 0.0,
                   parent_ratio: 1.0).fetch(:hot)
      end
      # Whole counts by finite products: NB(s; 1/2, 1/5) = prod_{j < s}(j + 1/2) / s! * (1/5)**(1/2) * (4/5)**s.
      #   count 2, the expected count: 0.0204437696074..., below the prior hot share of 0.05;
      #   count 12: 0.9955806603078..., decisive.
      expect(relative(hots.first, 0.020443769607400786)).to be < 1e-9
      expect(relative(hots.last, 0.9955806603078936)).to be < 1e-9
      expect(hots.first).to be < 0.05
      expect(hots.last).to be > 0.99
      # Above the expected count each step multiplies the odds by more than one, so the hot share rises throughout.
      expect(hots.drop(1).each_cons(2).all? { |lower, higher| higher > lower }).to be(true)
    end
  end

  describe "::GRID" do
    it "SL7 holds the 57 strengths from 2**-2 to 2**12 in quarter powers of 2" do
      expect(rates::GRID.size).to eq(57)
      expect(rates::GRID.first).to eq(0.25)
      expect(rates::GRID.last).to eq(4096.0)
      rates::GRID.zip(grid).each { |actual, expected| expect(relative(actual, expected)).to be < 1e-12 }
    end
  end

  describe ".fit_strength" do
    def expect_profile(fit, list, hot)
      expect(fit[:profile].size).to eq(57)
      fit[:profile].zip(grid).each do |(strength, value), expected_strength|
        expect(relative(strength, expected_strength)).to be < 1e-12
        expect(relative(value, log_likelihood(list, expected_strength, hot))).to be < 1e-9
      end
    end

    it "SL7 maximises the marginal log-likelihood on the grid, at hot share 1 without a hot grid" do
      fit = rates.fit_strength(pairs: pairs(spread_out))

      expect(fit.keys).to contain_exactly(:strength, :hot_share, :log_likelihood, :at_upper_bound, :interval, :profile)
      expect(relative(fit[:strength], 2.0)).to be < 1e-12 # grid point 12, 2**1
      expect(fit[:hot_share]).to eq(1.0)
      expect(relative(fit[:log_likelihood], log_likelihood(spread_out, 2.0, 1.0))).to be < 1e-9 # -21.3406570453...
      expect(fit[:at_upper_bound]).to be(false)
      expect_profile(fit, spread_out, 1.0)
    end

    it "SL7 holds the profile-likelihood interval: the strengths within 1.92 of the maximum, the fit inside it" do
      fit = rates.fit_strength(pairs: pairs(spread_out))
      lower, upper = fit[:interval]
      # Grid points 4 to 23 lie within 1.92 of the maximum (the reference profile), 2**-1 and 2**3.75.
      expect(relative(lower, 0.5)).to be < 1e-12
      expect(relative(upper, 2.0**3.75)).to be < 1e-12
      expect(lower).to be <= fit[:strength]
      expect(fit[:strength]).to be <= upper
    end

    it "SL7 fits the hot share on hot_grid, and reads the interval at the fitted hot share" do
      fit = rates.fit_strength(pairs: pairs(two_hot_spots), hot_grid: [ 1.0, 0.5, 0.2, 0.1, 0.05 ])

      expect(fit[:hot_share]).to eq(0.2)
      expect(relative(fit[:strength], 2.0**-1.5)).to be < 1e-12 # grid point 2
      expect(relative(fit[:log_likelihood], log_likelihood(two_hot_spots, 2.0**-1.5, 0.2))).to be < 1e-9
      expect(fit[:at_upper_bound]).to be(false)
      # At h = 0.2, grid points 0 to 9 lie within 1.92 of the maximum: 2**-2 and 2**0.25.
      lower, upper = fit[:interval]
      expect(relative(lower, 0.25)).to be < 1e-12
      expect(relative(upper, 2.0**0.25)).to be < 1e-12
      expect(lower).to be <= fit[:strength]
      expect(fit[:strength]).to be <= upper
    end

    it "SL7 says when a level's nodes do not differ: the strength at the grid's top" do
      fit = rates.fit_strength(pairs: pairs(all_as_expected))

      expect(fit[:strength]).to eq(rates::GRID.last)
      expect(fit[:at_upper_bound]).to be(true)
      lower, upper = fit[:interval]
      expect(relative(lower, 2.0**1.25)).to be < 1e-12 # grid point 13
      expect(upper).to eq(rates::GRID.last)
      expect_profile(fit, all_as_expected, 1.0)
    end

    it "SL7 takes real-valued counts and expected counts" do
      fit = rates.fit_strength(pairs: pairs(by_share))

      expect(relative(fit[:strength], 2.0**0.5)).to be < 1e-12 # grid point 10
      expect(relative(fit[:log_likelihood], log_likelihood(by_share, 2.0**0.5, 1.0))).to be < 1e-9
      lower, upper = fit[:interval]
      expect(relative(lower, 2.0**-1.75)).to be < 1e-12
      expect(relative(upper, 2.0**3.5)).to be < 1e-12
      expect_profile(fit, by_share, 1.0)
    end

    it "SL7 leaves out a pair whose parent has fewer than two children with data" do
      base = rates.fit_strength(pairs: pairs(spread_out))
      lone = rates.fit_strength(pairs: pairs(spread_out) + pairs([ [ 30, 2.0 ] ], siblings: 1))

      expect(lone[:strength]).to eq(base[:strength])
      expect(lone[:hot_share]).to eq(base[:hot_share])
      expect(lone[:at_upper_bound]).to eq(base[:at_upper_bound])
      expect(lone[:interval]).to eq(base[:interval])
      expect(lone[:log_likelihood]).to be_within(1e-12).of(base[:log_likelihood])

      # Kept, the same pair moves the fit (the reference puts its maximum at 2**-1.25), so leaving it out is visible.
      kept = rates.fit_strength(pairs: pairs(spread_out) + pairs([ [ 30, 2.0 ] ], siblings: 2))
      expect(relative(kept[:strength], 2.0**-1.25)).to be < 1e-12
    end
  end

  describe ".smooth_strength" do
    it "SL7 moves the strength a fifth of the way in logs by default" do
      # exp(0.8 * log 16 + 0.2 * log 1) = 16**0.8 = 2**3.2
      expect(relative(rates.smooth_strength(previous: 16.0, fitted: 1.0), 2.0**3.2)).to be < 1e-12
    end

    it "SL7 takes another weight" do
      # exp(0.5 * log 2 + 0.5 * log 8) = 4
      expect(relative(rates.smooth_strength(previous: 2.0, fitted: 8.0, weight: 0.5), 4.0)).to be < 1e-12
    end

    it "SL7 answers the fitted strength when there is no previous one" do
      expect(rates.smooth_strength(previous: nil, fitted: 5.5)).to eq(5.5)
    end
  end

  describe ".marshall" do
    it "SL8 gives Marshall's moment estimate on hand-computed pairs" do
      result = rates.marshall(pairs: pairs([ [ 2, 4.0 ], [ 9, 4.0 ], [ 1, 2.0 ], [ 8, 6.0 ] ]))
      # m = 20 / 16 = 5/4; sum E (s/E - m)**2 = 9/4 + 4 + 9/8 + 1/24 = 89/12; A = 89/192 - (5/4) / 4 = 29/192
      expect(result.keys).to contain_exactly(:variance, :strength)
      expect(relative(result[:variance], 29.0 / 192.0)).to be < 1e-12
      expect(relative(result[:strength], 192.0 / 29.0)).to be < 1e-12
    end

    it "SL8 takes real-valued counts" do
      list = [ [ 1.5, 4.0 ], [ 8.25, 3.5 ], [ 0.0, 2.25 ], [ 11.75, 5.0 ] ]
      exact = list.map { |count, expected| [ Rational(count), Rational(expected) ] }
      total = exact.sum { |_count, expected| expected }
      m = exact.sum { |count, _expected| count } / total
      spread = exact.sum { |count, expected| expected * (((count / expected) - m)**2) } / total
      variance = (spread - (m / (total / exact.size))).to_f # 86333 / 121835 = 0.70860590142...

      result = rates.marshall(pairs: pairs(list))
      expect(relative(result[:variance], variance)).to be < 1e-12
      expect(relative(result[:strength], 1.0 / variance)).to be < 1e-12
    end

    it "SL8 floors the variance at 0, where the strength is infinite" do
      # Every count as expected: the spread is 0 and A = 0 - m / mean(E) = -1 / 3.5, floored.
      result = rates.marshall(pairs: pairs(all_as_expected))
      expect(result[:variance]).to eq(0.0)
      expect(result[:strength]).to eq(Float::INFINITY)
    end
  end

  describe ".boundary_test" do
    # chi-squared(1) survival is erfc(sqrt(x / 2)): independent of 0.2.0's chi_squared_survival.
    def half_chi_squared_one(statistic) = 0.5 * Math.erfc(Math.sqrt(statistic / 2.0))

    it "SL8 compares the fitted level with the all-Poisson one" do
      result = rates.boundary_test(pairs: pairs(spread_out), strength: 2.0)
      statistic = 2.0 * (log_likelihood(spread_out, 2.0, 1.0) - poisson_log_likelihood(spread_out)) # 8.01009272757...

      expect(result.keys).to contain_exactly(:statistic, :p_value)
      expect(relative(result[:statistic], statistic)).to be < 1e-9
      # 0.0023258677836...; at this statistic the p-value's relative error is about 4.5 times the statistic's
      # (d log p / d log LR = -(LR + 1) / 2 near the tail), hence 1e-8 here and 1e-6 where LR is 44.
      expect(relative(result[:p_value], half_chi_squared_one(statistic))).to be < 1e-8
    end

    it "SL8 takes the hot share" do
      result = rates.boundary_test(pairs: pairs(two_hot_spots), strength: 2.0**-1.5, hot_share: 0.2)
      statistic = 2.0 * (log_likelihood(two_hot_spots, 2.0**-1.5, 0.2) - poisson_log_likelihood(two_hot_spots)) # 43.877...

      expect(relative(result[:statistic], statistic)).to be < 1e-9
      expect(relative(result[:p_value], half_chi_squared_one(statistic))).to be < 1e-6 # 1.748e-11
    end

    it "SL8 answers a p-value of 1 where the level is no better than all-Poisson" do
      result = rates.boundary_test(pairs: pairs(all_as_expected), strength: 1.0)
      statistic = 2.0 * (log_likelihood(all_as_expected, 1.0, 1.0) - poisson_log_likelihood(all_as_expected)) # -6.3766...

      expect(statistic).to be < 0.0
      expect(relative(result[:statistic], statistic)).to be < 1e-9
      expect(result[:p_value]).to eq(1.0)
    end
  end
end
