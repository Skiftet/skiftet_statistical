# frozen_string_literal: true

# SL9: the pooling reaches the decision. Pooling runs per (arm, part): `fit`'s `arms:` name the pairs, one `.baseline`
# runs per part, and `.pooled_estimate` answers a cell's SharedCredit-shaped estimate, theta(a | cell) = mu(cell) +
# delta(a | cell), the reference arm's delta 0 and the level mu one quantity every arm with credit shares. Per entries
# (a, p) and (b, q):
#
#   means[a][p]             = mu_p(cell) + delta_ap(cell)
#   cov[(a, p), (b, q)]     = Cov(mu_p, mu_q) + [a == b] * Cov(delta_ap, delta_aq)
#   Cov(mu_p, mu_q)         = rho_ref(p, q) * sd(mu_p) * sd(mu_q), Var(mu_p) at p == q
#   Cov(delta_ap, delta_aq) = rho_a(p, q) * sd(delta_ap) * sd(delta_aq), Var(delta_ap) at p == q
#
# rho_ref and rho_a are `own`'s correlations between the parts of the reference arm and of arm a, 0 beside an own
# variance of 0; nothing between two arms is read from `own`. Beside each variance `fit` answers its test of no
# variation (#variation_tests): Q, the moment's spread sum P * (estimate - mean)^2, on its degrees, and p = P(chi-squared
# >= Q). Every expected value below is worked by hand from these formulas alone, in Rationals (a p value by a closed
# chi-squared tail: exp(-Q / 2) on 2 degrees, erfc(sqrt(Q / 2)) on 1), in the comment above it, and compared to 1e-12.
RSpec.describe "SkiftetStatistical::AdditiveContrasts pins" do
  def contrasts = SkiftetStatistical::AdditiveContrasts

  def shared_credit = SkiftetStatistical::SharedCredit

  def rational(numerator, denominator = 1) = Rational(numerator, denominator)

  def node(id, by_pair) = { id: id, contrasts: by_pair, tree: [ :root, id ], groups: {} }

  # An estimate as SharedCredit answers one: the four keys, its parts and its rounding.
  def estimate(parts, means, covariance, rounding = 0.0)
    shared_credit::Estimate.new(parts, {
      means: means.transform_values { |row| row&.map(&:to_f) },
      covariance: covariance.map { |row| row.map(&:to_f) },
      weight_sums: means.transform_values { |row| row.nil? ? 0.0 : 12.0 },
      effective: means.transform_values { |row| row.nil? ? 0.0 : 9.0 }
    }, rounding)
  end

  def expect_close(actual, expected, label)
    expect(actual).to be_within(1e-12).of(expected.to_f), "#{label}: #{actual} is not #{expected}"
  end

  def expect_means(answer, expected)
    expect(answer[:means].keys).to eq(expected.keys)
    expected.each do |arm, row|
      row.each_with_index { |value, part| expect_close(answer[:means][arm][part], value, "the mean of #{arm}, #{part}") }
    end
  end

  def expect_covariance(answer, expected)
    expect(answer[:covariance].size).to eq(expected.size)
    expected.each_with_index do |row, i|
      row.each_with_index { |value, j| expect_close(answer[:covariance][i][j], value, "covariance (#{i}, #{j})") }
    end
  end

  def correlation(rows, first, second)
    rows[first][second] / Math.sqrt(rows[first][first] * rows[second][second])
  end

  # Two arms, ref the reference, a the other; parts p and q; entries (ref, p), (ref, q), (a, p), (a, q).
  #
  # c1's own sds are 1/5, 1/10, 3/10, 1/5, its correlations r(0, 1) = r(2, 3) = 1/2, r(0, 2) = r(1, 3) = 3/4 and
  # r(0, 3) = r(1, 2) = 3/8, of full rank: with A = [[1, 1/2], [1/2, 1]] and B = [[3/4, 3/8], [3/8, 3/4]] its eigenvalues
  # are those of A + B (21/8, 7/8) and A - B (3/8, 1/8). Each entry is r * sd * sd: (0, 1) 1/100, (0, 2) 9/200, (0, 3)
  # 3/200, (1, 2) 9/800, (1, 3) 3/200, (2, 3) 3/100.
  # Its contrasts of a against ref: 8/5 - 1 = 3/5 at variance 9/100 + 1/25 - 2 * 9/200 = 1/25, and 4/5 - 1/2 = 3/10 at
  # variance 1/25 + 1/100 - 2 * 3/200 = 1/50.
  def own_one(rounding = 0.0)
    estimate(%i[p q], { ref: [ rational(1), rational(1, 2) ], a: [ rational(8, 5), rational(4, 5) ] },
             [ [ rational(1, 25), rational(1, 100), rational(9, 200), rational(3, 200) ],
               [ rational(1, 100), rational(1, 100), rational(9, 800), rational(3, 200) ],
               [ rational(9, 200), rational(9, 800), rational(9, 100), rational(3, 100) ],
               [ rational(3, 200), rational(3, 200), rational(3, 100), rational(1, 25) ] ], rounding)
  end

  # c2's own sds are 3/25, 1/10, 4/25, 1/10, no correlation across arms, 1/2 across the parts of an arm: (0, 1) 1/2 *
  # 3/25 * 1/10 = 3/500, (2, 3) 1/2 * 4/25 * 1/10 = 1/125. Its contrasts: 11/5 - 2 = 1/5 at variance 16/625 + 9/625 =
  # 1/25, and 13/10 - 1 = 3/10 at variance 1/100 + 1/100 = 1/50.
  def own_two
    estimate(%i[p q], { ref: [ rational(2), rational(1) ], a: [ rational(11, 5), rational(13, 10) ] },
             [ [ rational(9, 625), rational(3, 500), 0, 0 ],
               [ rational(3, 500), rational(1, 100), 0, 0 ],
               [ 0, 0, rational(16, 625), rational(1, 125) ],
               [ 0, 0, rational(1, 125), rational(1, 100) ] ])
  end

  def pair_p = contrasts.pair(:a, :p)

  def pair_q = contrasts.pair(:a, :q)

  # Those contrasts over a one-level tree, c1 and c2 its two nodes.
  def two_cells
    [ node("c1", { pair_p => [ 0.6, 0.04 ], pair_q => [ 0.3, 0.02 ] }),
      node("c2", { pair_p => [ 0.2, 0.04 ], pair_q => [ 0.3, 0.02 ] }) ]
  end

  # Effects handed for c1 and c2, each delta's sd a round number: c1's (a, p) [1/2, 9/100] (sd 3/10) and (a, q) [3/10,
  # 1/25] (sd 1/5); c2's (a, p) [3/10, 1/25] (sd 1/5) and (a, q) [3/10, 1/100] (sd 1/10).
  def round_effects
    { "c1" => { pair_p => [ 0.5, 0.09 ], pair_q => [ 0.3, 0.04 ] },
      "c2" => { pair_p => [ 0.3, 0.04 ], pair_q => [ 0.3, 0.01 ] } }
  end

  # A standard normal from two uniforms (Box and Muller).
  def standard_normal(rng) = Math.sqrt(-2.0 * Math.log(1.0 - rng.rand)) * Math.cos(2.0 * Math::PI * rng.rand)

  describe ".pair" do
    it "SL9 names a pair by its arm and its part, the name fit's arms: and an answer's effects carry" do
      expect(contrasts.pair(:a, :p)).to eq("a|p")
      expect(contrasts.pair("a", "p")).to eq("a|p")
    end

    it "SL9 refuses a name that could not be told apart again" do
      # "a|b" with "c" and "a" with "b|c" would both read "a|b|c".
      expect { contrasts.pair("a|b", "c") }.to raise_error(ArgumentError)
      expect { contrasts.pair("a", "b|c") }.to raise_error(ArgumentError)
      expect { contrasts.pair(1, "c") }.to raise_error(ArgumentError)
      expect { contrasts.pair("a", nil) }.to raise_error(ArgumentError)
    end
  end

  describe ".fit over pairs" do
    it "SL9 answers the two cells' worked posteriors" do
      # a|p: precision P = 25 in each cell, estimates 3/5 and 1/5, their mean 2/5. Spread 25 * (1/5)^2 * 2 = 2, one degree
      # of freedom, scale 50 - 1250 / 50 = 25: tau^2 = (2 - 1) / 25 = 1/25.
      # Up: scale 1 + 25 / 25 = 2, so the nodes send precision 25/2 with sums 15/2 and 5/2, the root holds 25 and 10.
      # Down, c1: the rest is 25/2 with sum 5/2, scale 1 + 1/2 = 3/2, so the rest gives 25/3 and 5/3. The node holds
      # 25 + 25/3 = 100/3 and 15 + 5/3 = 50/3: mean 1/2, variance 3/100. c2: 5 + (15/2) / (3/2) = 10, mean 3/10.
      # a|q: P = 50 and equal estimates, spread 0: tau^2 = 0. The rest is 50 with sum 15, the node 100 and 30: mean 3/10,
      # variance 1/100.
      answer = contrasts.fit(cells: two_cells, arms: [ pair_p, pair_q ])

      expect_close(answer.dig(:variances, "tree:1", pair_p), rational(1, 25), "tau^2 of a|p")
      expect(answer.dig(:variances, "tree:1", pair_q)).to eq(0.0)
      { "c1" => rational(1, 2), "c2" => rational(3, 10) }.each do |id, mean|
        expect_close(answer.dig(:cells, id, pair_p, 0), mean, "#{id} a|p mean")
        expect_close(answer.dig(:cells, id, pair_p, 1), rational(3, 100), "#{id} a|p variance")
        expect_close(answer.dig(:cells, id, pair_q, 0), rational(3, 10), "#{id} a|q mean")
        expect_close(answer.dig(:cells, id, pair_q, 1), rational(1, 100), "#{id} a|q variance")
      end
    end

    it "SL9 reads a nil contrast as a left-out one, and still answers that cell from its node" do
      # What a caller hands for a contrast with a nil mean or a variance of 0.0. c3 adds nothing for a|q, the same fit as
      # with the key absent; its answer is the root's, the prior its node gives it.
      nil_contrast = two_cells + [ node("c3", { pair_p => [ 0.4, 0.04 ], pair_q => nil }) ]
      absent = two_cells + [ node("c3", { pair_p => [ 0.4, 0.04 ] }) ]

      expect(contrasts.fit(cells: nil_contrast, arms: [ pair_p, pair_q ]))
        .to eq(contrasts.fit(cells: absent, arms: [ pair_p, pair_q ]))
      expect_close(contrasts.fit(cells: nil_contrast, arms: [ pair_q ]).dig(:cells, "c3", pair_q, 0), rational(3, 10), "c3")
    end

    it "SL9 leaves the nil to the caller: a nil mean, a variance of 0.0 and an arm with no contrast are refused" do
      expect { contrasts.fit(cells: [ node("c1", { pair_p => [ nil, 0.04 ] }) ], arms: [ pair_p ]) }
        .to raise_error(ArgumentError)
      expect { contrasts.fit(cells: [ node("c1", { pair_p => [ 0.1, 0.0 ] }) ], arms: [ pair_p ]) }
        .to raise_error(ArgumentError)
      expect { contrasts.fit(cells: [ node("c1", { pair_p => nil }) ], arms: [ pair_p ]) }.to raise_error(ArgumentError)
    end
  end

  describe ".fit's test of no variation" do
    def expect_test(answer, name, arm, statistic, degrees, p_value)
      test = answer.variation_tests.dig(name, arm)
      expect(test.keys).to eq(%i[statistic degrees p_value])
      expect_close(test[:statistic], statistic, "#{name} #{arm}: Q")
      expect(test[:degrees]).to eq(degrees)
      expect_close(test[:p_value], p_value, "#{name} #{arm}: p")
    end

    # Four cells under one leaf n, with variance 1 each, crossed with g: x holds the estimates 0 and 2, y holds 3 and 3.
    def crossed
      [ [ 1, 0.0, "x" ], [ 2, 2.0, "x" ], [ 3, 3.0, "y" ], [ 4, 3.0, "y" ] ].map do |id, estimate, key|
        { id: id, contrasts: { a: [ estimate, 1.0 ] }, tree: %i[root n], groups: { "g" => key } }
      end
    end

    it "SL9 answers beside each variance Q, its degrees and Q's chi-squared tail, and keeps its three keys" do
      # a|p: the spread is 25 * (1/5)^2 * 2 = 2 (the moment of the worked posteriors above) on 2 - 1 = 1 degree, so
      # p = P(chi-squared(1) >= 2) = P(|Z| >= sqrt(2)) = erfc(1). a|q: equal estimates, spread 0, p = 1.
      answer = contrasts.fit(cells: two_cells, arms: [ pair_p, pair_q ])

      expect(answer.keys).to eq(%i[variances effects cells])
      expect(answer.variation_tests.keys).to eq([ "tree:1" ])
      expect(answer.variation_tests["tree:1"].keys).to eq([ pair_p, pair_q ])
      expect_test(answer, "tree:1", pair_p, 2, 1, Math.erfc(1.0))
      expect_test(answer, "tree:1", pair_q, 0, 1, 1)
    end

    it "SL9 tests each depth on its nodes' own sampling variances, the depth under them carried" do
      # m1 > l1 (0), l2 (4); m2 > l3 (1), l4 (1); variance 1 each.
      # tree:2: around 2, 1 * 2^2 + 1 * 2^2 = 8; around 1, 0. Q = 8 on 1 + 1 = 2 degrees, p = exp(-8/2) = exp(-4); scale
      # (2 - 2/2) * 2 = 2, so tau^2 = (8 - 2) / 2 = 3.
      # Each leaf then sends precision 1 / (1 + 3 * 1) = 1/4 with sums 0, 1, 1/4, 1/4: m1 holds 1/2 and 1 (estimate 2),
      # m2 1/2 and 1/2 (estimate 1). tree:1: around 3/2, 1/2 * (1/2)^2 * 2 = 1/4 on 1 degree, p = P(|Z| >= 1/2) =
      # erfc(sqrt(1/8)); scale 1 - (1/4 + 1/4) / 1 = 1/2, tau^2 = max(0, (1/4 - 1) / (1/2)) = 0.
      cells = [ [ :m1, :l1, 0.0 ], [ :m1, :l2, 4.0 ], [ :m2, :l3, 1.0 ], [ :m2, :l4, 1.0 ] ].map do |middle, leaf, estimate|
        { id: leaf, contrasts: { a: [ estimate, 1.0 ] }, tree: [ :root, middle, leaf ], groups: {} }
      end
      answer = contrasts.fit(cells: cells, arms: %i[a])

      expect_test(answer, "tree:2", :a, 8, 2, Math.exp(-4.0))
      expect_test(answer, "tree:1", :a, rational(1, 4), 1, Math.erfc(Math.sqrt(0.125)))
      expect_close(answer.dig(:variances, "tree:2", :a), 3, "tree:2's variance")
      expect(answer.dig(:variances, "tree:1", :a)).to eq(0.0)
    end

    it "SL9 tests a crossed grouping on its keys, and a level of one node on no degrees at p 1" do
      # The tree reads the weighted mean 8/4 = 2, so g's keys hold x: -2 + 0 = -2 over precision 2 (estimate -1), y: 1 +
      # 1 = 2 over 2 (estimate 1). Around 0: 2 * 1 + 2 * 1 = 4 on 1 degree, p = P(|Z| >= 2) = erfc(sqrt(2)); scale 4 - 8/4
      # = 2, tau^2 = (4 - 1) / 2 = 3/2. The keys' means -3/4 and 3/4 cancel in the leaf, which stays at 2, so every sweep
      # reads the same Q. tree:1 holds n alone: no spread, no degrees, p 1.
      answer = contrasts.fit(cells: crossed, arms: %i[a])

      expect_test(answer, "g", :a, 4, 1, Math.erfc(Math.sqrt(2.0)))
      expect_test(answer, "tree:1", :a, 0, 0, 1)
      expect_close(answer.dig(:variances, "g", :a), rational(3, 2), "g's variance")
    end

    it "SL9 tests a variance held in variances: on the data all the same, and none it names that the cells do not" do
      # g held at 0: its keys' means are 0, the leaf reads 8/4 = 2 as before, so g's keys hold the same -2 and 2: Q = 4.
      # "other" is no grouping of these cells: it stays in variances, untested.
      answer = contrasts.fit(cells: crossed, arms: %i[a], variances: { "g" => { a: 0.0 }, "other" => { a: 1.0 } })

      expect(answer.dig(:variances, "g", :a)).to eq(0.0)
      expect(answer.dig(:variances, "other", :a)).to eq(1.0)
      expect_test(answer, "g", :a, 4, 1, Math.erfc(Math.sqrt(2.0)))
      expect(answer.variation_tests.keys).to contain_exactly("tree:1", "g")
    end

    it "SL9 rejects at 5 % about one fit in twenty where nothing varies, whatever the sampling variances" do
      # Twenty nodes at truth 0, sampling variances 1/2 to 2: Q is chi-squared on 19 degrees, so p is uniform and about
      # 5 % of 400 fits (20, sd about 4.4) fall below 0.05. With a true tau^2 of 1 most fits reject.
      rng = Random.new(31)
      shares = [ 0.0, 1.0 ].map do |tau_squared|
        rejected = Array.new(400) do
          cells = Array.new(20) do |index|
            variance = 0.5 + (1.5 * rng.rand)
            truth = Math.sqrt(tau_squared) * standard_normal(rng)
            node("n#{index}", { a: [ truth + (Math.sqrt(variance) * standard_normal(rng)), variance ] })
          end
          contrasts.fit(cells: cells, arms: %i[a]).variation_tests.dig("tree:1", :a, :p_value) < 0.05
        end
        rejected.count(true) / 400.0
      end

      expect(shares.first).to be_between(0.02, 0.09)
      expect(shares.last).to be > 0.5
    end
  end

  describe ".baseline with an arm left out of fit's arms:" do
    # Rows of one visit: c1 ref k1, c1 a k2, c1 b k1, c2 ref k2; b left out of `fit`, so named at [0.0, 0.0].
    let(:rows) do
      [ { cell: "c1", arm: :ref, covariate: "k1", weight: 2.0, value: 2.0 },
        { cell: "c1", arm: :a, covariate: "k2", weight: 1.0, value: 2.0 },
        { cell: "c1", arm: :b, covariate: "k1", weight: 1.0, value: 1.0 },
        { cell: "c2", arm: :ref, covariate: "k2", weight: 1.0, value: 0.0 } ]
    end

    it "SL9 keeps that arm's visits, each read at effect 0" do
      # e(c1, a) = 1, e(c1, b) = 0, ref 0. S' = 2, 2 - 1 = 1, 1, 0 over W = 2, 1, 1, 1: m = 4/5. k1 holds S' 3 over W 3,
      # m_k1 = 1, s_k1 = -1/5; k2 holds S' 1 over W 2, m_k2 = 1/2, s_k2 = 3/10.
      # o: c1 ref -1/5, c1 a 3/10 - 1 = -7/10, c1 b -1/5, c2 ref 3/10. Moved values: 4/5, 13/10, 4/5, and 3/10.
      # c1: W_c 4, mu = (2 * 4/5 + 13/10 + 4/5) / 4 = 37/40. Variance (4 * (-1/8)^2 + (3/8)^2 + (-1/8)^2) / 16 = (14/64) /
      # 16 = 7/512. c2: W_c 1, mu = 3/10, one visit, variance 0.
      effects = { "c1" => { a: [ 1.0, 0.25 ], b: [ 0.0, 0.0 ] }, "c2" => { a: [ 0.5, 0.25 ], b: [ 0.0, 0.0 ] } }
      answer = contrasts.baseline(rows: rows, effects: effects)

      expect(answer.keys).to contain_exactly("c1", "c2")
      expect_close(answer["c1"][0], rational(37, 40), "c1 mean")
      expect_close(answer["c1"][1], rational(7, 512), "c1 variance")
      expect_close(answer["c2"][0], rational(3, 10), "c2 mean")
      expect_close(answer["c2"][1], 0, "c2 variance")
    end

    it "SL9 keeps the raise for an arm left out with no such name" do
      effects = { "c1" => { a: [ 1.0, 0.25 ] }, "c2" => { a: [ 0.5, 0.25 ] } }

      expect { contrasts.baseline(rows: rows, effects: effects) }.to raise_error(ArgumentError)
    end
  end

  describe ".pooled_estimate" do
    let(:effects) { contrasts.fit(cells: two_cells, arms: [ pair_p, pair_q ])[:cells] }

    # The levels' sds: p's 1/10 (variance 1/100) and q's 3/40 (variance 9/1600), in both cells.
    let(:baselines) do
      { p: { "c1" => [ 1.1, 0.01 ], "c2" => [ 1.9, 0.01 ] }, q: { "c1" => [ 0.6, 9.0 / 1600 ], "c2" => [ 1.2, 9.0 / 1600 ] } }
    end

    def pooled(own, cell, effects_given: effects, baselines_given: baselines)
      contrasts.pooled_estimate(own: own, effects: effects_given, baselines: baselines_given, cell: cell)
    end

    it "SL9 answers the worked estimate of c1: two arms, two parts, the level shared by both" do
      # Means: ref mu = 11/10, 3/5. a mu + delta = 11/10 + 1/2 = 8/5, 3/5 + 3/10 = 9/10.
      # own_one's correlation between parts is 1/2 in each arm; its correlations across the arms (3/4, 3/8) are not read.
      # Cov(mu_p, mu_q) = 1/2 * 1/10 * 3/40 = 3/800. Cov(delta_ap, delta_aq) = 1/2 * 3/10 * 1/5 = 3/100.
      # (0, 0) 1/100, (1, 1) 9/1600, (2, 2) 1/100 + 9/100 = 1/10, (3, 3) 9/1600 + 64/1600 = 73/1600.
      # (0, 2) Var(mu_p) = 1/100, (1, 3) Var(mu_q) = 9/1600, (0, 1) = (0, 3) = (1, 2) = 3/800, (2, 3) 3/800 + 3/100 =
      # 27/800. The contrast of a and ref at p: 1/10 + 1/100 - 2 * 1/100 = 9/100, delta_ap's variance alone.
      answer = pooled(own_one, "c1", effects_given: round_effects)

      expect_means(answer, { ref: [ rational(11, 10), rational(3, 5) ], a: [ rational(8, 5), rational(9, 10) ] })
      expect_covariance(answer, [ [ rational(1, 100), rational(3, 800), rational(1, 100), rational(3, 800) ],
                                  [ rational(3, 800), rational(9, 1600), rational(3, 800), rational(9, 1600) ],
                                  [ rational(1, 100), rational(3, 800), rational(1, 10), rational(27, 800) ],
                                  [ rational(3, 800), rational(9, 1600), rational(27, 800), rational(73, 1600) ] ])
      expect_close(shared_credit.contrast(answer, :a, :ref, weights: [ 1.0, 0.0 ])[:variance], rational(9, 100),
                   "the contrast's variance")
    end

    it "SL9 answers the worked estimate of c2, whose own arms do not covary and whose pooled ones do, by the level" do
      # Means: ref 19/10, 6/5. a 19/10 + 3/10 = 11/5, 6/5 + 3/10 = 3/2.
      # own_two's correlation between parts: ref 3/500 / (3/25 * 1/10) = 1/2, a 1/125 / (4/25 * 1/10) = 1/2.
      # Cov(mu_p, mu_q) = 1/2 * 1/10 * 3/40 = 3/800. Cov(delta_ap, delta_aq) = 1/2 * 1/5 * 1/10 = 1/100.
      # (0, 0) 1/100, (1, 1) 9/1600, (2, 2) 1/100 + 1/25 = 1/20, (3, 3) 9/1600 + 16/1600 = 1/64.
      # (0, 2) 1/100, (1, 3) 9/1600, (0, 1) = (0, 3) = (1, 2) = 3/800, (2, 3) 3/800 + 1/100 = 11/800.
      answer = pooled(own_two, "c2", effects_given: round_effects)

      expect_means(answer, { ref: [ rational(19, 10), rational(6, 5) ], a: [ rational(11, 5), rational(3, 2) ] })
      expect_covariance(answer, [ [ rational(1, 100), rational(3, 800), rational(1, 100), rational(3, 800) ],
                                  [ rational(3, 800), rational(9, 1600), rational(3, 800), rational(9, 1600) ],
                                  [ rational(1, 100), rational(3, 800), rational(1, 20), rational(11, 800) ],
                                  [ rational(3, 800), rational(9, 1600), rational(11, 800), rational(1, 64) ] ])
    end

    it "SL9 reads nothing between two arms from own: own_one and own_two, alike within each arm, answer alike" do
      # Their correlations between parts are 1/2 in every arm; across the arms own_one's are 3/4 and 3/8, own_two's 0.
      # Only the levels, the effects and those within-arm correlations reach the answer, so c1's is the same on either.
      one = pooled(own_one, "c1", effects_given: round_effects)
      two = pooled(own_two, "c1", effects_given: round_effects)

      expect_means(two, one[:means])
      expect_covariance(two, one[:covariance])
    end

    it "SL9 lets no share follow the level: an arm's P(best) is the same at a level's variance of 1/10000 and of 4" do
      # One part, no shared credit. The entries are [[v, v], [v, v + 1/100]], so a draw of a - ref is delta's alone,
      # N(1/10, 1/100): P(best) of a is Phi(1), about 0.8413, whatever v; the same draws give the same share.
      own = estimate(%i[p], { ref: [ 1.0 ], a: [ 1.1 ] }, [ [ 0.04, 0.0 ], [ 0.0, 0.04 ] ])
      shares = [ 1e-4, 4.0 ].map do |variance|
        answer = pooled(own, "c1", effects_given: { "c1" => { contrasts.pair(:a, :p) => [ 0.1, 0.01 ] } },
                                   baselines_given: { p: { "c1" => [ 1.0, variance ] } })
        SkiftetStatistical::LinearValue.summary(
          arms: %i[ref a], parts: %i[p], means: answer[:means], covariance: answer[:covariance],
          parameters: { base: { fixed: 1.0 } }, weights: ->(drawn) { { p: drawn.fetch(:base) } }, draws: 2_000,
          rng: Random.new(5), rounding: answer.rounding
        )[:a][:p_best]
      end

      expect_close(shares.last, shares.first, "a's P(best) at a level's variance of 4")
      expect(shares.first).to be_within(0.03).of(SkiftetStatistical::Significance.normal_cdf(1.0))
    end

    it "SL9 runs the three doors end to end: fit over the pairs, one baseline per part, the estimate" do
      # Rows of one visit, one covariate value, so every o is -e. Part p, e(c1) = 1/2, e(c2) = 3/10, W 2 each.
      # c1: ref 1, a 8/5 moved to 11/10: mu = 21/20, each visit 1/20 off: (2 * 4 * 1/400) / 16 = 1/800.
      # c2: ref 2, a 11/5 moved to 19/10: mu = 39/20, likewise 1/800.
      # Part q, e = 3/10 in both. c1: ref 2/5, a 1 moved to 7/10: mu = 11/20, each visit 3/20 off: (2 * 4 * 9/400) / 16 =
      # 9/800. c2: ref 4/5, a 8/5 moved to 13/10: mu = 21/20, each visit 1/4 off: (2 * 4 * 1/16) / 16 = 1/32.
      # c1's estimate: means ref 21/20, 11/20; a 21/20 + 1/2 = 31/20, 11/20 + 3/10 = 17/20. Variances 1/800, 9/800,
      # 1/800 + 3/100 = 1/32, 9/800 + 1/100 = 17/800. Across the arms the level alone: (0, 2) Var(mu_p) = 1/800, (1, 3)
      # Var(mu_q) = 9/800, (0, 1) = (0, 3) = (1, 2) = Cov(mu_p, mu_q) = 1/2 * sqrt(1/800) * sqrt(9/800) = 3/1600. Within a:
      # (2, 3) 3/1600 + 1/2 * sqrt(3/100) * sqrt(1/100) = 3/1600 + sqrt(3)/200.
      rows = lambda do |name, by_cell|
        by_cell.flat_map do |cell, (first, second)|
          [ { cell: cell, arm: :ref, covariate: "k", weight: 2.0, value: 2.0 * first },
            { cell: cell, arm: name, covariate: "k", weight: 2.0, value: 2.0 * second } ]
        end
      end
      levels = {
        p: contrasts.baseline(rows: rows.call(pair_p, { "c1" => [ 1.0, 1.6 ], "c2" => [ 2.0, 2.2 ] }), effects: effects),
        q: contrasts.baseline(rows: rows.call(pair_q, { "c1" => [ 0.4, 1.0 ], "c2" => [ 0.8, 1.6 ] }), effects: effects)
      }
      answer = pooled(own_one, "c1", baselines_given: levels)

      expect_close(levels[:p]["c2"][0], rational(39, 20), "p's level in c2")
      expect_close(levels[:q]["c2"][1], rational(1, 32), "q's variance in c2")
      expect_means(answer, { ref: [ rational(21, 20), rational(11, 20) ], a: [ rational(31, 20), rational(17, 20) ] })
      { [ 0, 0 ] => rational(1, 800), [ 1, 1 ] => rational(9, 800), [ 2, 2 ] => rational(1, 32),
        [ 3, 3 ] => rational(17, 800), [ 0, 2 ] => rational(1, 800), [ 1, 3 ] => rational(9, 800),
        [ 0, 1 ] => rational(3, 1600), [ 0, 3 ] => rational(3, 1600), [ 1, 2 ] => rational(3, 1600),
        [ 2, 3 ] => rational(3, 1600) + (Math.sqrt(3.0) / 200) }.each do |(i, j), value|
        expect_close(answer[:covariance][i][j], value, "covariance (#{i}, #{j})")
        expect(answer[:covariance][j][i]).to eq(answer[:covariance][i][j])
      end
    end

    it "SL9 answers the shape of a SharedCredit estimate: its four keys, its parts and its rounding" do
      answer = pooled(own_one, "c1")

      expect(answer).to be_a(shared_credit::Estimate)
      expect(answer.keys).to eq(%i[means covariance weight_sums effective])
      expect(answer.parts).to eq(%i[p q])
      expect(answer.rounding).to be_a(Float)
      expect(answer[:weight_sums]).to eq({ ref: 12.0, a: 12.0 })
      expect(answer[:effective]).to eq({ ref: 9.0, a: 9.0 })
      expect(answer[:covariance]).to eq(answer[:covariance].transpose)
      expect(shared_credit.linear(answer, weights: { p: 1.0, q: 2.0 })[:means].keys).to eq(%i[ref a])
    end

    it "SL9 leaves own as it was" do
      own = own_one
      before = Marshal.load(Marshal.dump(own.to_h))
      pooled(own, "c1")

      expect(own.to_h).to eq(before)
    end

    it "SL9 cancels the level from every contrast of two arms, whatever its variance and own's correlation of the two" do
      # a and b share most of their rows, so own holds them strongly correlated; nothing of that is read. One part:
      # every entry between two arms is Var(mu) and each diagonal Var(mu) + Var(delta), so a contrast of two arms is
      # their deltas' alone: a - b 9/100 + 16/100 = 1/4 at mean 1/10 - 3/10 = -1/5, a - ref 9/100, b - ref 16/100, at a
      # level's variance of 1/25 and of 9/4 alike.
      rows = [ { credit: { ref: 2.0 }, values: [ 1.0 ] }, { credit: { ref: 2.0 }, values: [ 3.0 ] },
               { credit: { ref: 2.0 }, values: [ 0.0 ] } ]
      rows += [ 2.0, 0.0, 1.0, 4.0, 0.0, 3.0 ].map { |value| { credit: { a: 2.0, b: 2.0 }, values: [ value ] } }
      rows += [ { credit: { a: 2.0 }, values: [ 1.0 ] }, { credit: { b: 2.0 }, values: [ 2.0 ] } ]
      own = shared_credit.estimate(rows, arms: %i[ref a b], parts: %i[p])
      pairs = { "c1" => { contrasts.pair(:a, :p) => [ 0.1, 0.09 ], contrasts.pair(:b, :p) => [ 0.3, 0.16 ] } }

      expect(correlation(own[:covariance], 1, 2)).to be > 0.5
      [ rational(1, 25), rational(9, 4) ].each do |variance|
        answer = pooled(own, "c1", effects_given: pairs, baselines_given: { p: { "c1" => [ 2.0, variance.to_f ] } })
        gap = shared_credit.contrast(answer, :a, :b, weights: [ 1.0 ])

        expect_close(gap[:mean], rational(-1, 5), "the contrast's mean at #{variance}")
        expect_close(gap[:variance], rational(1, 4), "a - b at #{variance}")
        expect_close(shared_credit.contrast(answer, :a, :ref, weights: [ 1.0 ])[:variance], rational(9, 100), "a - ref")
        expect_close(shared_credit.contrast(answer, :b, :ref, weights: [ 1.0 ])[:variance], rational(16, 100), "b - ref")
      end
    end

    it "SL9 answers a covariance LinearValue takes, at full rank" do
      # c1's own correlation between parts is of full rank in each arm, and the pooled covariance is J (x) L, the level's
      # block over both arms, plus a's block: x' * cov * x = (x_ref + x_a)' L (x_ref + x_a) + x_a' A x_a, which is 0
      # only at x_a = 0 and then x_ref = 0, so every eigenvalue stays above 0.
      answer = pooled(own_one, "c1")
      means = answer[:means]
      summary = SkiftetStatistical::LinearValue.summary(
        arms: means.keys, parts: answer.parts, means: means, covariance: answer[:covariance],
        parameters: { base: { fixed: 1.0 } }, weights: ->(drawn) { { p: drawn.fetch(:base), q: 2.0 } },
        draws: 2_000, rng: Random.new(7), rounding: answer.rounding
      )

      # ref 11/10 + 2 * 3/5 = 23/10; a 8/5 + 2 * 9/10 = 17/5.
      expect_close(summary[:ref][:point], rational(23, 10), "ref's value")
      expect_close(summary[:a][:point], rational(17, 5), "a's value")
    end

    it "SL9 answers a covariance LinearValue takes where its own is of lower rank, on the answer's rounding" do
      # Fewer rows than entries, so own's blocks are of lower rank and the level's and each arm's block with them. Each
      # trial's sds move by different factors, the case that scales each row's rounding differently.
      rng = Random.new(11)
      40.times do |trial|
        rows = Array.new(3) do
          { credit: { ref: 1.0 + rng.rand, a: rng.rand(2) * 3.0, b: 1.0 + (9.0 * rng.rand) },
            values: [ rng.rand(4).to_f, rng.rand, rng.rand(3) * 0.1 ] }
        end
        rows << { credit: { a: 2.0 }, values: [ 1.0, rng.rand, 0.0 ] }
        own = shared_credit.estimate(rows, arms: %i[ref a b], parts: %i[p q r])
        names = %i[a b].product(%i[p q r]).to_h { |arm, part| [ contrasts.pair(arm, part), [ rng.rand, 1e-4 + rng.rand ] ] }
        levels = %i[p q r].to_h { |part| [ part, { "c1" => [ rng.rand, 1e-4 + (0.1 * rng.rand) ] } ] }
        answer = pooled(own, "c1", effects_given: { "c1" => names }, baselines_given: levels)

        drawn = SkiftetStatistical::LinearValue.each_draw(
          arms: %i[ref a b], parts: %i[p q r], means: answer[:means], covariance: answer[:covariance],
          parameters: {}, weights: ->(_drawn) { { p: 1.0, q: 1.0, r: 1.0 } }, draws: 5, rng: Random.new(3),
          rounding: answer.rounding
        )
        expect(drawn.count).to eq(5), "trial #{trial}"
      end
    end

    it "SL9 takes, on the answer's rounding alone, what an own covariance of a large level leaves" do
      # Every own entry is a small difference of large terms, so below its rank it is all rounding, and that rounding
      # moves with the sds, here by factors up to 10 either way. Some trials are refused at a rounding of 0: those are
      # the ones the answer's rounding has to carry.
      arms = %i[ref a b c]
      parts = %i[p q r s t u]
      moved = lambda { |rng, variance| [ rng.rand, ((variance.positive? ? variance : 1e-6) * (10.0**(2.0 * rng.rand))) + 1e-300 ] }
      draw = lambda do |answer, rounding|
        SkiftetStatistical::LinearValue.each_draw(
          arms: arms, parts: parts, means: answer[:means], covariance: answer[:covariance],
          parameters: { one: { fixed: 1.0 } }, weights: ->(drawn) { parts.to_h { |part| [ part, drawn.fetch(:one) ] } },
          draws: 3, rng: Random.new(3), rounding: rounding
        ).to_a
      end
      decided = 0
      60.times do |trial|
        rng = Random.new(trial + 1)
        level = 10.0**(1 + rng.rand(7))
        rows = Array.new(10) do
          drawn = arms.select { rng.rand < 0.4 }
          { credit: (drawn.empty? ? [ arms.sample(random: rng) ] : drawn).to_h { |arm| [ arm, 1.0 + (6.0 * rng.rand) ] },
            values: parts.map { (rng.rand < 0.3 ? 1 + rng.rand(3) : 0) * 1e-3 + level } }
        end
        rows += arms.map { |arm| { credit: { arm => 1.0 + rng.rand }, values: parts.map { level + rng.rand(2) } } }
        own = shared_credit.estimate(rows, arms: arms, parts: parts)
        variances = own[:covariance].each_index.map { |entry| own[:covariance][entry][entry] }
        names = arms.drop(1).product(parts).to_h do |arm, part|
          [ contrasts.pair(arm, part), moved.call(rng, variances[(arms.index(arm) * parts.size) + parts.index(part)]) ]
        end
        levels = parts.each_with_index.to_h { |part, at| [ part, { "c1" => moved.call(rng, variances[at]) } ] }
        answer = pooled(own, "c1", effects_given: { "c1" => names }, baselines_given: levels)

        begin
          draw.call(answer, 0.0)
        rescue ArgumentError
          decided += 1
        end
        expect(draw.call(answer, answer.rounding).size).to eq(3), "trial #{trial}"
      end
      expect(decided).to be > 5
    end

    it "SL9 brings a big cell's means to its own, and its covariance as near as the variances go to 0" do
      # A cell of sampling variance v beside two of 1/25: as v falls, delta nears its contrast 1/2 and every own number
      # is what the pooled one is built of. The covariance keeps within 3 own variances: Var(mu) is the reference arm's
      # v / 2 and Var(delta) about v, so a's pooled variance tends to 3 / 2 * v, three times its own v / 2.
      gaps = [ 1e-2, 1e-4, 1e-6, 1e-8 ].map do |variance|
        cells = [ node("big", { pair_p => [ 0.5, variance ] }), node("s1", { pair_p => [ 0.1, 0.04 ] }),
                  node("s2", { pair_p => [ 0.3, 0.04 ] }) ]
        fitted = contrasts.fit(cells: cells, arms: [ pair_p ])[:cells]
        own = estimate(%i[p], { ref: [ 1.0 ], a: [ 1.5 ] }, [ [ variance / 2, 0.0 ], [ 0.0, variance / 2 ] ])
        answer = pooled(own, "big", effects_given: fitted, baselines_given: { p: { "big" => [ 1.0, variance / 2 ] } })

        expect_close(answer[:means][:ref][0], 1.0, "the reference's mean")
        expect_close(answer[:covariance][0][0], variance / 2, "the reference's variance")
        expect(answer[:covariance][1][1]).to be_between(variance / 2, 1.5 * variance + 1e-12)
        (answer[:means][:a][0] - 1.5).abs
      end

      expect(gaps).to eq(gaps.sort.reverse)
      expect(gaps.last).to be < 1e-6
    end

    it "SL9 shrinks a small cell's delta all the way to its parent's when the fitted variance is 0" do
      # Precisions 10000 and 25, estimates 1/10 and 1/5: the root is (1000 + 5) / 10025 = 201/2005. The spread, about
      # 0.2494, is below the one degree of freedom, so tau^2 = 0 and every node sits at the root. The small cell's own
      # contrast 1/5 is gone: its delta is 201/2005, its mean 1 + 201/2005 = 2206/2005.
      cells = [ node("big", { pair_p => [ 0.1, 0.0001 ] }), node("small", { pair_p => [ 0.2, 0.04 ] }) ]
      fitted = contrasts.fit(cells: cells, arms: [ pair_p ])
      own = estimate(%i[p], { ref: [ 1.0 ], a: [ 1.2 ] }, [ [ 0.02, 0.0 ], [ 0.0, 0.02 ] ])
      answer = pooled(own, "small", effects_given: fitted[:cells], baselines_given: { p: { "small" => [ 1.0, 0.02 ] } })

      expect(fitted.dig(:variances, "tree:1", pair_p)).to eq(0.0)
      expect_close(fitted.dig(:cells, "small", pair_p, 0), rational(201, 2005), "the small cell's delta")
      expect_close(fitted.dig(:cells, "big", pair_p, 0), rational(201, 2005), "the big cell's delta")
      expect_means(answer, { ref: [ 1 ], a: [ rational(2206, 2005) ] })
    end

    it "SL9 keeps nil means, zero rows and columns and own's sums for an arm with no credit" do
      own = estimate(%i[p], { ref: [ 1.0 ], a: nil, b: [ 2.0 ] }, [ [ 0.04, 0.0, 0.012 ], [ 0.0, 0.0, 0.0 ],
                                                                    [ 0.012, 0.0, 0.09 ] ])
      name = contrasts.pair(:b, :p)
      answer = pooled(own, "c1", effects_given: { "c1" => { name => [ 0.5, 0.03 ] } },
                                 baselines_given: { p: { "c1" => [ 1.0, 0.01 ] } })

      # One part, so between the two arms with credit the level alone: (0, 2) Var(mu) = 1/100, own's 0.012 not read;
      # b's diagonal 1/100 + 3/100 = 1/25.
      expect(answer[:means][:a]).to be_nil
      expect(answer[:weight_sums][:a]).to eq(0.0)
      expect(answer[:effective][:a]).to eq(0.0)
      expect_means(answer.merge(means: answer[:means].compact), { ref: [ 1 ], b: [ rational(3, 2) ] })
      expect_covariance(answer, [ [ rational(1, 100), 0, rational(1, 100) ], [ 0, 0, 0 ],
                                  [ rational(1, 100), 0, rational(1, 25) ] ])
    end

    it "SL9 answers the worked one-part example" do
      # One part: (0, 1) is Cov(mu, mu) = Var(mu) = 1/100, and own's correlation of 1/5 between the arms is not read;
      # a's variance 1/100 + 3/100 = 1/25. Means 1 and 1 + 1/2.
      own = estimate(%i[p], { ref: [ 1.0 ], a: [ 2.0 ] }, [ [ 0.04, 0.012 ], [ 0.012, 0.09 ] ])
      answer = pooled(own, "c1", effects_given: { "c1" => { contrasts.pair(:a, :p) => [ 0.5, 0.03 ] } },
                                 baselines_given: { p: { "c1" => [ 1.0, 0.01 ] } })

      expect_means(answer, { ref: [ 1 ], a: [ rational(3, 2) ] })
      expect_covariance(answer, [ [ rational(1, 100), rational(1, 100) ], [ rational(1, 100), rational(1, 25) ] ])
    end

    it "SL9 gives an arm's delta correlation 0 beside its own variance of 0, its pooled variance on the diagonal" do
      # a's part q never varied in this cell, so own's row and column of it are 0 and its contrast was left out of fit.
      # The fit answers the cell from its node all the same: its diagonal is 1/100 + 3/100 = 1/25. rho_a(p, q) is 0, so
      # (2, 3) is the level's alone; the level keeps ref's 0.01 / (0.2 * 0.1) = 1/2: Cov(mu_p, mu_q) = 1/2 * 1/10 * 1/10
      # = 1/200, which (0, 1), (0, 3), (1, 2) and (2, 3) hold; (0, 2) and (1, 3) hold Var(mu_p) = Var(mu_q) = 1/100.
      own = estimate(%i[p q], { ref: [ 1.0, 0.5 ], a: [ 2.0, 0.5 ] },
                     [ [ 0.04, 0.01, 0.012, 0.0 ], [ 0.01, 0.01, 0.0, 0.0 ], [ 0.012, 0.0, 0.09, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0 ] ])
      names = { "c1" => { contrasts.pair(:a, :p) => [ 0.5, 0.03 ], contrasts.pair(:a, :q) => [ 0.1, 0.03 ] } }
      levels = { p: { "c1" => [ 1.0, 0.01 ] }, q: { "c1" => [ 0.4, 0.01 ] } }
      answer = pooled(own, "c1", effects_given: names, baselines_given: levels)

      expect_covariance(answer, [ [ rational(1, 100), rational(1, 200), rational(1, 100), rational(1, 200) ],
                                  [ rational(1, 200), rational(1, 100), rational(1, 200), rational(1, 100) ],
                                  [ rational(1, 100), rational(1, 200), rational(1, 25), rational(1, 200) ],
                                  [ rational(1, 200), rational(1, 100), rational(1, 200), rational(1, 25) ] ])
    end

    it "SL9 reads correlation 0 beside an own variance of 0, whatever the entry says" do
      # No variance has no correlation to keep: ref's q and a's p have own variance 0 beside an entry of 0.01, so
      # Cov(mu_p, mu_q) and Cov(delta_ap, delta_aq) are 0, not some share of the sds. What is left: (0, 0) 1/100,
      # (1, 1) 1/25, (2, 2) 1/100 + 3/100 = 1/25, (3, 3) 1/25 + 1/20 = 9/100, (0, 2) 1/100, (1, 3) 1/25.
      own = estimate(%i[p q], { ref: [ 1.0, 0.5 ], a: [ 2.0, 0.5 ] },
                     [ [ 0.04, 0.01, 0.0, 0.0 ], [ 0.01, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.01 ],
                       [ 0.0, 0.0, 0.01, 0.04 ] ])
      names = { "c1" => { contrasts.pair(:a, :p) => [ 0.5, 0.03 ], contrasts.pair(:a, :q) => [ 0.2, 0.05 ] } }
      levels = { p: { "c1" => [ 1.0, 0.01 ] }, q: { "c1" => [ 0.5, 0.04 ] } }
      answer = pooled(own, "c1", effects_given: names, baselines_given: levels)

      expect_covariance(answer, [ [ rational(1, 100), 0, rational(1, 100), 0 ], [ 0, rational(1, 25), 0, rational(1, 25) ],
                                  [ rational(1, 100), 0, rational(1, 25), 0 ], [ 0, rational(1, 25), 0, rational(9, 100) ] ])
    end

    it "SL9 reads a correlation that rounding takes past 1 as 1" do
      # Own sds 1 and 1 in each arm with an entry a little above 1 between the parts: the exact correlation is within 1,
      # so Cov(mu_p, mu_q) = 1/10 * 1/5 = 1/50 and Cov(delta_ap, delta_aq) = 3/10 * 2/5 = 3/25. (0, 0) 1/100, (1, 1)
      # 1/25, (2, 2) 1/100 + 9/100 = 1/10, (3, 3) 1/25 + 4/25 = 1/5, (0, 2) 1/100, (1, 3) 1/25, (0, 1) = (0, 3) = (1, 2)
      # = 1/50, (2, 3) 1/50 + 3/25 = 7/50.
      past = 1.0 + 1e-9
      own = estimate(%i[p q], { ref: [ 1.0, 0.5 ], a: [ 2.0, 0.5 ] },
                     [ [ 1.0, past, 0.0, 0.0 ], [ past, 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, past ], [ 0.0, 0.0, past, 1.0 ] ])
      names = { "c1" => { contrasts.pair(:a, :p) => [ 0.5, 0.09 ], contrasts.pair(:a, :q) => [ 0.2, 0.16 ] } }
      levels = { p: { "c1" => [ 1.0, 0.01 ] }, q: { "c1" => [ 0.5, 0.04 ] } }
      answer = pooled(own, "c1", effects_given: names, baselines_given: levels)

      expect_covariance(answer, [ [ rational(1, 100), rational(1, 50), rational(1, 100), rational(1, 50) ],
                                  [ rational(1, 50), rational(1, 25), rational(1, 50), rational(1, 25) ],
                                  [ rational(1, 100), rational(1, 50), rational(1, 10), rational(7, 50) ],
                                  [ rational(1, 50), rational(1, 25), rational(7, 50), rational(1, 5) ] ])
    end

    it "SL9 reads an arm left out of fit's arms: at effect [0.0, 0.0], beside the reference arm" do
      # b is named, so ref alone is the reference; b's delta is 0 with variance 0, so its theta is the level: its mean
      # mu, its variance Var(mu) and every entry beside it the level's, Var(mu) = 1/100, as ref's.
      own = estimate(%i[p], { ref: [ 1.0 ], a: [ 2.0 ], b: [ 3.0 ] },
                     [ [ 0.04, 0.0, 0.0 ], [ 0.0, 0.09, 0.0 ], [ 0.0, 0.0, 0.16 ] ])
      names = { "c1" => { contrasts.pair(:a, :p) => [ 0.5, 0.03 ], contrasts.pair(:b, :p) => [ 0.0, 0.0 ] } }
      answer = pooled(own, "c1", effects_given: names, baselines_given: { p: { "c1" => [ 1.0, 0.01 ] } })

      expect_means(answer, { ref: [ 1 ], a: [ rational(3, 2) ], b: [ 1 ] })
      expect_covariance(answer, [ [ rational(1, 100), rational(1, 100), rational(1, 100) ],
                                  [ rational(1, 100), rational(1, 25), rational(1, 100) ],
                                  [ rational(1, 100), rational(1, 100), rational(1, 100) ] ])
    end

    it "SL9 carries own's rounding to the answer's scale, the level's block and the effects' each" do
      # own_one's bound 1e-10, on c1's baselines and round effects (a|p sd 1/5, a|q sd 1/10). The level's block moves
      # ref's sds 1/5 to 1/10 (by 1/2) and 1/10 to 3/40 (by 3/4); a's block moves a's 3/10 to 1/5 (by 2/3) and 1/5 to
      # 1/10 (by 1/2). An entry holds one term of each block, so its bound is 1e-10 * ((3/4)^2 + (2/3)^2) = 1e-10 *
      # (81 + 64) / 144 = 1e-10 * 145/144. The rounding of the entries, a few units of 1/20, is far below that.
      effects = { "c1" => { pair_p => [ 0.5, 0.04 ], pair_q => [ 0.3, 0.01 ] } }
      answer = pooled(own_one(1e-10), "c1", effects_given: effects)

      expect(answer.rounding).to be_within(1e-14).of(1e-10 * rational(145, 144))
      expect(answer.rounding).to be >= 1e-10 * rational(145, 144)
    end

    it "SL9 carries none of own's rounding where one part reads no entry of own's covariance" do
      # One part: every entry is Var(mu) or Var(mu) + Var(delta), and own's bound of 1e-10 reaches none of them.
      own = estimate(%i[p], { ref: [ 1.0 ], a: [ 2.0 ] }, [ [ 0.04, 0.012 ], [ 0.012, 0.09 ] ], 1e-10)
      answer = pooled(own, "c1", effects_given: { "c1" => { contrasts.pair(:a, :p) => [ 0.5, 0.03 ] } },
                                 baselines_given: { p: { "c1" => [ 1.0, 0.01 ] } })

      expect(answer.rounding).to be_between(0.0, 1e-16).exclusive
    end

    it "SL9 gives the rounding of its own entries where own has none" do
      own = estimate(%i[p], { ref: [ 1.0 ], a: [ 2.0 ] }, [ [ 0.04, 0.012 ], [ 0.012, 0.09 ] ])
      answer = pooled(own, "c1", effects_given: { "c1" => { contrasts.pair(:a, :p) => [ 0.5, 0.03 ] } },
                                 baselines_given: { p: { "c1" => [ 1.0, 0.01 ] } })

      expect(answer.rounding).to be_between(0.0, 1e-16).exclusive
    end

    it "SL9 reads names by Symbol or String, as the module does" do
      answer = contrasts.pooled_estimate(
        own: own_one, effects: effects.transform_values { |by_pair| by_pair.transform_keys(&:to_sym) },
        baselines: baselines.transform_keys(&:to_s), cell: "c1"
      )

      expect_means(answer, { ref: [ rational(11, 10), rational(3, 5) ], a: [ rational(8, 5), rational(9, 10) ] })
    end

    it "SL9 needs no baseline where no arm has credit" do
      own = estimate(%i[p], { ref: nil, a: nil }, [ [ 0.0, 0.0 ], [ 0.0, 0.0 ] ])
      answer = pooled(own, "c1", effects_given: {}, baselines_given: {})

      expect(answer[:means]).to eq({ ref: nil, a: nil })
      expect(answer[:covariance]).to eq([ [ 0.0, 0.0 ], [ 0.0, 0.0 ] ])
      expect(answer.rounding).to eql(0.0)
    end

    describe "its refusals, each an ArgumentError" do
      it "SL9 refuses an estimate read back from JSON, which names no parts" do
        expect { pooled(own_one.to_h, "c1") }.to raise_error(ArgumentError)
        expect { pooled(nil, "c1") }.to raise_error(ArgumentError)
      end

      it "SL9 refuses a covariance of the wrong shape" do
        own = estimate(%i[p q], { ref: [ 1.0, 0.5 ], a: [ 2.0, 0.5 ] }, [ [ 0.04, 0.0 ], [ 0.0, 0.04 ] ])

        expect { pooled(own, "c1") }.to raise_error(ArgumentError)
      end

      it "SL9 refuses effects that are not cells of arms" do
        expect { pooled(own_one, "c1", effects_given: nil) }.to raise_error(ArgumentError)
        expect { pooled(own_one, "c1", effects_given: { "c1" => [] }) }.to raise_error(ArgumentError)
      end

      it "SL9 refuses an arm with credit and no effect in the cell, once some cell names it" do
        # a is named in c2 and has no effect in c1, and with p named but not q, so a is no reference arm.
        expect { pooled(own_one, "c1", effects_given: { "c2" => effects["c2"] }) }.to raise_error(ArgumentError)
        one_part = { "c1" => { pair_p => [ 0.5, 0.03 ] } }
        expect { pooled(own_one, "c1", effects_given: one_part) }.to raise_error(ArgumentError)
      end

      it "SL9 refuses two credited arms that no cell names, as .baseline does" do
        expect { pooled(own_one, "c1", effects_given: {}) }.to raise_error(ArgumentError)
      end

      it "SL9 refuses a part with no baseline, a cell with no baseline, and a baseline that is no pair" do
        expect { pooled(own_one, "c1", baselines_given: baselines.slice(:p)) }.to raise_error(ArgumentError)
        expect { pooled(own_one, "c9", effects_given: { "c9" => effects["c1"] }) }.to raise_error(ArgumentError)
        expect { pooled(own_one, "c1", baselines_given: nil) }.to raise_error(ArgumentError)
        expect { pooled(own_one, "c1", baselines_given: baselines.merge(p: { "c1" => 1.0 })) }.to raise_error(ArgumentError)
      end

      it "SL9 refuses a variance below 0 and a number that is not one" do
        below = baselines.merge(p: { "c1" => [ 1.1, -0.01 ] })
        expect { pooled(own_one, "c1", baselines_given: below) }.to raise_error(ArgumentError)
        nan = { "c1" => effects["c1"].merge(pair_p => [ Float::NAN, 0.03 ]) }
        expect { pooled(own_one, "c1", effects_given: nan) }.to raise_error(ArgumentError)
        minus = { "c1" => effects["c1"].merge(pair_p => [ 0.5, -0.03 ]) }
        expect { pooled(own_one, "c1", effects_given: minus) }.to raise_error(ArgumentError)
      end
    end
  end
end
