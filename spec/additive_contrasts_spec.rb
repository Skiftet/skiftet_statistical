# frozen_string_literal: true

# SL9: additive normal contrasts with crossed groupings. Each cell holds every arm's contrast against the reference arm
# (estimate, sampling variance), its tree path [root, ..., leaf] and its other groupings { name => key }. The model:
# delta(a | cell) = delta_root(a) + u_tree(a) + sum over groupings g of u_g(a), the tree nested (a node's deviation
# N(0, tau^2 of its depth), "tree:1" the first level under the root), every other grouping crossed (N(0, tau^2_g)).
# Variances by moments on the contrasts, tau^2 = max(0, weighted spread of the group estimates - their mean sampling
# variance), per depth, grouping and arm; normal-normal posteriors; `variances:` holds the variances it names fixed.
# The checks below hold whatever the moment's weights are: each is worked out by hand in its comment.
RSpec.describe "SkiftetStatistical::AdditiveContrasts" do
  def fit(**arguments)
    SkiftetStatistical::AdditiveContrasts.fit(**arguments)
  end

  def cell(id, contrasts, tree, groups = {})
    { id: id, contrasts: contrasts, tree: tree, groups: groups }
  end

  def cell_mean(answer, id, arm = :a)
    answer.dig(:cells, id, arm, 0)
  end

  # Five cells on a two-level tree (m1 > l1, l2; m2 > l3, l4, l5) with one crossed grouping "g", two arms.
  let(:nested) do
    [
      cell(1, { a: [ 0.10, 0.004 ], b: [ -0.02, 0.003 ] }, %i[root m1 l1], { "g" => "x" }),
      cell(2, { a: [ 0.30, 0.006 ], b: [ 0.05, 0.004 ] }, %i[root m1 l2], { "g" => "y" }),
      cell(3, { a: [ -0.20, 0.005 ], b: [ 0.01, 0.002 ] }, %i[root m2 l3], { "g" => "x" }),
      cell(4, { a: [ 0.05, 0.003 ], b: [ 0.12, 0.005 ] }, %i[root m2 l4], { "g" => "y" }),
      cell(5, { a: [ 0.25, 0.004 ], b: [ -0.06, 0.006 ] }, %i[root m2 l5], { "g" => "x" })
    ]
  end

  # One level under the root, no other grouping, equal sampling variances 1.
  def one_level(estimates)
    estimates.each_with_index.map { |estimate, index| cell(index + 1, { a: [ estimate, 1.0 ] }, [ :root, :"n#{index + 1}" ]) }
  end

  describe ".fit" do
    it "SL9 fits no variance between two cells whose contrasts are equal, and leaves both at that contrast" do
      # The spread of two equal estimates is 0, below their mean sampling variance: tau^2 = max(0, 0 - v) = 0.
      # Every estimate is 0.3, so every weighted mean of them, and every posterior, is 0.3.
      answer = fit(cells: [ cell(1, { a: [ 0.3, 0.01 ] }, %i[root x]), cell(2, { a: [ 0.3, 0.02 ] }, %i[root y]) ],
                   arms: %i[a])

      expect(answer.dig(:variances, "tree:1", :a)).to eq(0.0)
      expect(cell_mean(answer, 1)).to be_within(1e-12).of(0.3)
      expect(cell_mean(answer, 2)).to be_within(1e-12).of(0.3)
    end

    it "SL9 answers variances per depth and grouping, effects per key and posteriors per cell, each per arm" do
      [ 1, 5 ].each do |sweeps|
        answer = fit(cells: nested, arms: %i[a b], sweeps: sweeps)

        expect(answer.keys).to contain_exactly(:variances, :effects, :cells)
        expect(answer[:variances].keys).to include("tree:1", "tree:2", "g")
        %w[tree:1 tree:2 g].each do |name|
          expect(answer[:variances][name].keys).to contain_exactly(:a, :b)
          expect(answer[:variances][name].values).to all(be_a(Float).and(be >= 0.0))
        end
        expect(answer[:effects].keys).to include("tree:1", "tree:2", "g")
        expect(answer[:effects]["tree:1"].keys).to contain_exactly(:m1, :m2)
        expect(answer[:effects]["tree:2"].keys).to contain_exactly(:l1, :l2, :l3, :l4, :l5)
        expect(answer[:effects]["g"].keys).to contain_exactly("x", "y")
        answer[:effects].each_value do |by_key|
          by_key.each_value do |by_arm|
            expect(by_arm.keys).to contain_exactly(:a, :b)
            by_arm.each_value { |pair| expect(pair).to match([ a_kind_of(Float), a_value >= 0.0 ]) }
          end
        end
        expect(answer[:cells].keys).to contain_exactly(1, 2, 3, 4, 5)
        answer[:cells].each_value do |by_arm|
          expect(by_arm.keys).to contain_exactly(:a, :b)
          by_arm.each_value { |pair| expect(pair).to match([ a_kind_of(Float), a_value > 0.0 ]) }
        end
      end
    end

    it "SL9 fits every variance at 0 and every cell at the common contrast when all contrasts are equal" do
      # Equal estimates (0.2) with unequal sampling variances: every group's spread is 0, so every tau^2 is 0; the
      # root and every posterior are weighted means of 0.2 alone.
      equal = nested.map do |row|
        row.merge(contrasts: row[:contrasts].transform_values { |(_estimate, variance)| [ 0.2, variance ] })
      end
      answer = fit(cells: equal, arms: %i[a b])

      answer[:variances].each do |name, by_arm|
        by_arm.each { |arm, variance| expect(variance).to eq(0.0), "#{name} #{arm}" }
      end
      equal.each do |row|
        %i[a b].each { |arm| expect(cell_mean(answer, row[:id], arm)).to be_within(1e-12).of(0.2) }
      end
    end

    it "SL9 shrinks a one-level tree's nodes part of the way toward the root, the plain normal shrinkage" do
      # Estimates 0, 1 and 5, sampling variance 1 each: the root is their mean 2.0 under any weighting by variance.
      # Their spread about it, 14 / 3 (7 with n - 1), is above the sampling variance 1, so tau^2 > 0 and each node's
      # posterior mean lies strictly between its estimate and 2.0; its posterior variance is below its own 1.
      answer = fit(cells: one_level([ 0.0, 1.0, 5.0 ]), arms: %i[a])

      expect(answer.dig(:variances, "tree:1", :a)).to be > 0.0
      { 1 => 0.0, 2 => 1.0, 3 => 5.0 }.each do |id, estimate|
        low, high = [ estimate, 2.0 ].minmax
        expect(cell_mean(answer, id)).to be > low
        expect(cell_mean(answer, id)).to be < high
        expect(answer.dig(:cells, id, :a, 1)).to be_between(0.0, 1.0).exclusive
      end
      expect(cell_mean(answer, 3) - cell_mean(answer, 1)).to be < 5.0
    end

    it "SL9 shrinks a symmetric one-level tree symmetrically about its root" do
      # Estimates -3, 0 and 3 with variance 1: the root is 0 by symmetry, the middle node stays at 0 and the outer two
      # are pulled the same way in: mean(n1) = -mean(n3), strictly between -3 and 0.
      answer = fit(cells: one_level([ -3.0, 0.0, 3.0 ]), arms: %i[a])

      expect(cell_mean(answer, 2)).to be_within(1e-12).of(0.0)
      expect(cell_mean(answer, 1)).to be_within(1e-12).of(-cell_mean(answer, 3))
      expect(cell_mean(answer, 1)).to be_between(-3.0, 0.0).exclusive
    end

    it "SL9 fits each arm on its own contrasts" do
      # No term joins two arms, so b's answer is the answer of a fit of b's contrasts alone.
      both = fit(cells: nested, arms: %i[a b])
      alone = fit(cells: nested.map { |row| row.merge(contrasts: row[:contrasts].slice(:b)) }, arms: %i[b])

      both[:cells].each do |id, by_arm|
        expect(by_arm[:b][0]).to be_within(1e-12).of(alone[:cells][id][:b][0])
        expect(by_arm[:b][1]).to be_within(1e-12).of(alone[:cells][id][:b][1])
      end
      both[:variances].each do |name, by_arm|
        expect(by_arm[:b]).to be_within(1e-12).of(alone[:variances][name][:b])
      end
    end
  end

  describe ".fit with variances:" do
    it "SL9 keeps an earlier fit's variances: the answer's variances equal them" do
      earlier = fit(cells: nested, arms: %i[a b])
      later_cells = nested.map do |row|
        row.merge(contrasts: row[:contrasts].transform_values { |(estimate, variance)| [ -estimate + 0.05, variance ] })
      end
      later = fit(cells: later_cells, arms: %i[a b], variances: earlier[:variances])

      expect(later[:variances]).to eq(earlier[:variances])
    end

    it "SL9 takes no moment step for a named variance, and fits by moments the ones it does not name" do
      # All contrasts equal, so moments give 0 everywhere; "tree:1" for a is named at 0.25 and stays 0.25, while
      # "tree:1" for b and every other variance are fitted: 0. With every estimate equal, each cell stays at it.
      equal = nested.map do |row|
        row.merge(contrasts: row[:contrasts].transform_values { |(_estimate, variance)| [ 0.2, variance ] })
      end
      answer = fit(cells: equal, arms: %i[a b], variances: { "tree:1" => { a: 0.25 } })

      expect(answer.dig(:variances, "tree:1", :a)).to eq(0.25)
      expect(answer.dig(:variances, "tree:1", :b)).to eq(0.0)
      expect(answer.dig(:variances, "tree:2", :a)).to eq(0.0)
      expect(answer.dig(:variances, "g", :a)).to eq(0.0)
      equal.each { |row| expect(cell_mean(answer, row[:id])).to be_within(1e-12).of(0.2) }
    end

    it "SL9 still fits the effects under fixed variances: a wide one leaves each node near its estimate" do
      # tau^2 = 1e6 against a sampling variance of 1: each node keeps all but about 1e-6 of its distance from the root,
      # so it sits within 1e-4 of its estimate. Fixed at 0, every node is pulled further in: the spread of the
      # posterior means is smaller than under 1e6.
      cells = one_level([ 0.0, 1.0, 5.0 ])
      wide = fit(cells: cells, arms: %i[a], variances: { "tree:1" => { a: 1e6 } })
      none = fit(cells: cells, arms: %i[a], variances: { "tree:1" => { a: 0.0 } })

      expect(wide.dig(:variances, "tree:1", :a)).to eq(1e6)
      expect(none.dig(:variances, "tree:1", :a)).to eq(0.0)
      { 1 => 0.0, 2 => 1.0, 3 => 5.0 }.each do |id, estimate|
        expect(cell_mean(wide, id)).to be_within(1e-4).of(estimate)
      end
      spread = ->(answer) { cell_mean(answer, 3) - cell_mean(answer, 1) }
      expect(spread.call(none)).to be < spread.call(wide)
    end
  end

  # SL9's baseline, every expected value worked by hand in Rationals from these formulas alone:
  # - a row sums the visits i of one (cell, arm, covariate), weights w_i > 0 and values x_i: weight W = sum w_i,
  #   value S = sum w_i * x_i, weight_squared Q = sum w_i^2, value_squared B = sum w_i^2 * x_i^2, cross C = sum w_i^2 *
  #   x_i. All three of Q, B and C nil is one visit (x = S / W): Q = W^2, B = S^2, C = W * S;
  # - e(c, a) is the mean of effects[c][a]; the reference arm, the one arm of the rows no cell of effects names, has
  #   e = 0; S' = S - e * W;
  # - pooled over every row: m = sum S' / sum W; per covariate value k, m_k = (sum over k's rows of S') / (sum over k's
  #   rows of W) and s_k = m - m_k; a row's offset o = s_k - e(c, a);
  # - per cell: mu_c = (sum over c's rows of (S + o * W)) / W_c, W_c = sum over c's rows of W; its variance (sum over
  #   c's rows of (B + 2 * d * C + d^2 * Q)) / W_c^2, d = o - mu_c.
  describe ".baseline" do
    def baseline(**arguments)
      SkiftetStatistical::AdditiveContrasts.baseline(**arguments)
    end

    # A refusal, labelled, with RSpec's own message kept (so a missing name still shows as itself).
    def expect_refused(label, &call)
      aggregate_failures(label) { expect(&call).to raise_error(ArgumentError) }
    end

    def expect_baseline(answer, expected)
      expect(answer.keys).to contain_exactly(*expected.keys)
      expected.each do |id, (mean, variance)|
        expect(answer[id][0]).to be_within(1e-12).of(mean.to_f), "#{id} mean"
        expect(answer[id][1]).to be_within(1e-12).of(variance.to_f), "#{id} variance"
      end
    end

    # Three one-visit rows: two cells, the arms :ref and :a, the covariate values k1 and k2.
    let(:visit_rows) do
      [
        { cell: "c1", arm: :ref, covariate: "k1", weight: 2.0, value: 2.0 },
        { cell: "c1", arm: :a, covariate: "k2", weight: 1.0, value: 2.0 },
        { cell: "c2", arm: :ref, covariate: "k2", weight: 1.0, value: 0.0 }
      ]
    end
    let(:visit_effects) { { "c1" => { a: [ 1.0, 0.25 ] }, "c2" => { a: [ 0.5, 0.25 ] } } }

    # Four rows of sums: two cells, both arms in each, both covariate values in each. Visits (w, x):
    #   c1 ref k1: (1, 0), (1, 1) -> W 2, S 1, Q 2, B 1, C 1
    #   c1 a   k2: (2, 3), one visit -> W 2, S 6, nil sums read as Q 4, B 36, C 12
    #   c2 ref k2: (1, 2), (2, 3) -> W 3, S 8, Q 5, B 40, C 14
    #   c2 a   k1: (1, 0), (2, 2) -> W 3, S 4, Q 5, B 16, C 8
    let(:sum_rows) do
      [
        { cell: "c1", arm: :ref, covariate: "k1", weight: 2.0, value: 1.0, weight_squared: 2.0, value_squared: 1.0,
          cross: 1.0 },
        { cell: "c1", arm: :a, covariate: "k2", weight: 2.0, value: 6.0 },
        { cell: "c2", arm: :ref, covariate: "k2", weight: 3.0, value: 8.0, weight_squared: 5.0, value_squared: 40.0,
          cross: 14.0 },
        { cell: "c2", arm: :a, covariate: "k1", weight: 3.0, value: 4.0, weight_squared: 5.0, value_squared: 16.0,
          cross: 8.0 }
      ]
    end
    let(:sum_effects) { { "c1" => { a: [ 0.5, 0.04 ] }, "c2" => { a: [ -1.0, 0.09 ] } } }

    it "SL9 gives each cell its effect- and covariate-adjusted mean and variance on one-visit rows" do
      # Rows (one visit each): c1 ref k1 W 2, S 2 (Q 4, B 4, C 4); c1 a k2 W 1, S 2 (Q 1, B 4, C 2); c2 ref k2 W 1,
      # S 0 (Q 1, B 0, C 0). :ref is named by no cell, so it is the reference arm (e = 0); e(c1, a) = 1.
      # S' = 2, 2 - 1 * 1 = 1, 0. m = (2 + 1 + 0) / (2 + 1 + 1) = 3/4; m_k1 = 2/2 = 1, s_k1 = -1/4; m_k2 = (1 + 0) /
      # (1 + 1) = 1/2, s_k2 = 1/4. o: c1 ref -1/4, c1 a 1/4 - 1 = -3/4, c2 ref 1/4.
      # c1: W_c 3, mu = (2 - 1/4 * 2 + 2 - 3/4 * 1) / 3 = (11/4) / 3 = 11/12. d: -1/4 - 11/12 = -7/6 and -3/4 - 11/12 =
      # -5/3; terms 4 + 2 * (-7/6) * 4 + (49/36) * 4 = 1/9 and 4 + 2 * (-5/3) * 2 + 25/9 = 1/9; variance (2/9) / 9 =
      # 2/81.
      # c2: W_c 1, mu = (0 + 1/4 * 1) / 1 = 1/4; d = 0 and B = 0: variance 0.
      expect_baseline(baseline(rows: visit_rows, effects: visit_effects),
                      { "c1" => [ Rational(11, 12), Rational(2, 81) ], "c2" => [ Rational(1, 4), 0 ] })
    end

    it "SL9 reads a row of sums for several visits through its squared sums" do
      # c1's ref row replaced by two visits of weight 1 and values 1/2 and 3/2: W 2, S 2 as before, Q 2, B 1/4 + 9/4 =
      # 5/2, C 1/2 + 3/2 = 2. W and S are unchanged, so m, every m_k, every o and both means are; c1 ref's term is now
      # 5/2 + 2 * (-7/6) * 2 + (49/36) * 2 = (45 - 84 + 49) / 18 = 5/9, and c1's variance (5/9 + 1/9) / 9 = 2/27.
      rows = [ visit_rows[0].merge(weight_squared: 2.0, value_squared: 2.5, cross: 2.0), *visit_rows.drop(1) ]

      expect_baseline(baseline(rows: rows, effects: visit_effects),
                      { "c1" => [ Rational(11, 12), Rational(2, 27) ], "c2" => [ Rational(1, 4), 0 ] })
    end

    it "SL9 pools the covariate's group means over every cell, both arms in each cell, both values in each" do
      # e(c1, a) = 1/2, e(c2, a) = -1, :ref 0. S' = S - e * W: c1 ref 1, c1 a 6 - 1/2 * 2 = 5, c2 ref 8, c2 a 4 + 1 *
      # 3 = 7. m = (1 + 5 + 8 + 7) / (2 + 2 + 3 + 3) = 21/10.
      # k1 (c1 ref, c2 a): m_k1 = (1 + 7) / (2 + 3) = 8/5, s_k1 = 21/10 - 16/10 = 1/2. k2 (c1 a, c2 ref): m_k2 = (5 +
      # 8) / (2 + 3) = 13/5, s_k2 = 21/10 - 26/10 = -1/2. o = s_k - e: c1 ref 1/2, c1 a -1/2 - 1/2 = -1, c2 ref -1/2,
      # c2 a 1/2 + 1 = 3/2.
      # c1: W_c 4, mu = (1 + 1/2 * 2 + 6 - 1 * 2) / 4 = 3/2. d: 1/2 - 3/2 = -1 and -1 - 3/2 = -5/2; terms 1 + 2 * (-1)
      # * 1 + 1 * 2 = 1 and 36 + 2 * (-5/2) * 12 + (25/4) * 4 = 1; variance 2 / 16 = 1/8.
      # c2: W_c 6, mu = (8 - 1/2 * 3 + 4 + 3/2 * 3) / 6 = 15/6 = 5/2. d: -1/2 - 5/2 = -3 and 3/2 - 5/2 = -1; terms 40
      # + 2 * (-3) * 14 + 9 * 5 = 1 and 16 + 2 * (-1) * 8 + 1 * 5 = 5; variance 6 / 36 = 1/6.
      # A cell of effects with no row (c3) is left out, and the effects' variances are never read.
      effects = sum_effects.merge("c3" => { a: [ 2.0, 0.01 ] })
      wider = effects.transform_values { |by_arm| by_arm.transform_values { |(mean, _variance)| [ mean, 9.0 ] } }
      expected = { "c1" => [ Rational(3, 2), Rational(1, 8) ], "c2" => [ Rational(5, 2), Rational(1, 6) ] }

      expect_baseline(baseline(rows: sum_rows, effects: effects), expected)
      expect_baseline(baseline(rows: sum_rows, effects: wider), expected)
    end

    it "SL9 reads a row whose three squared sums are nil as one visit" do
      # c1 a's nil row is (w 2, x 3): Q = W^2 = 4, B = S^2 = 36, C = W * S = 12, so the row given those sums answers
      # the same. Two visits of the same W and S, (1, 2) and (1, 4) (Q 2, B 4 + 16 = 20, C 2 + 4 = 6), move nothing
      # but c1 a's term: 20 + 2 * (-5/2) * 6 + (25/4) * 2 = 5/2, c1's variance (1 + 5/2) / 16 = 7/32.
      explicit = [ sum_rows[0], sum_rows[1].merge(weight_squared: 4.0, value_squared: 36.0, cross: 12.0),
                   *sum_rows.drop(2) ]
      two_visits = [ sum_rows[0], sum_rows[1].merge(weight_squared: 2.0, value_squared: 20.0, cross: 6.0),
                     *sum_rows.drop(2) ]
      c2 = [ Rational(5, 2), Rational(1, 6) ]

      expect_baseline(baseline(rows: explicit, effects: sum_effects), { "c1" => [ Rational(3, 2), Rational(1, 8) ],
                                                                        "c2" => c2 })
      expect_baseline(baseline(rows: two_visits, effects: sum_effects), { "c1" => [ Rational(3, 2), Rational(7, 32) ],
                                                                          "c2" => c2 })
    end

    it "SL9 refuses a row whose arm has no effect for its cell, and two arms that no cell names" do
      # a is named in c2, so :ref is the reference arm and c1's a row needs e(c1, a): missing, with or without c1.
      expect { baseline(rows: sum_rows, effects: { "c2" => sum_effects["c2"] }) }.to raise_error(ArgumentError)
      expect { baseline(rows: sum_rows, effects: { "c1" => {}, "c2" => sum_effects["c2"] }) }
        .to raise_error(ArgumentError)
      # A row of a third arm :b that no cell names: :ref and :b would both be the reference arm.
      third = sum_rows + [ { cell: "c2", arm: :b, covariate: "k1", weight: 1.0, value: 1.0 } ]
      expect { baseline(rows: third, effects: sum_effects) }.to raise_error(ArgumentError)
    end

    it "SL9 refuses a row with some but not all three squared sums, and a weight that is not above 0" do
      partial = [ { weight_squared: 4.0 }, { value_squared: 36.0, cross: 12.0 }, { weight_squared: 4.0, cross: 12.0 } ]
      partial.each do |sums|
        rows = [ sum_rows[0], sum_rows[1].merge(sums), *sum_rows.drop(2) ]
        expect_refused(sums.inspect) { baseline(rows: rows, effects: sum_effects) }
      end
      [ 0.0, -2.0 ].each do |weight|
        rows = [ sum_rows[0], sum_rows[1].merge(weight: weight), *sum_rows.drop(2) ]
        expect_refused(weight.inspect) { baseline(rows: rows, effects: sum_effects) }
      end
    end
  end
end
