# frozen_string_literal: true

# The priors the bandit reads: flat, given, an amount prior, the leave-one-out borrow, the pooled prior of a measured
# item and of a new one, a pooled mean, the tilts, the chain of nested levels and the pooled amount prior. Beta priors
# are [alpha, beta]. Every literal was worked outside the library from the formulas, step by step.
RSpec.describe "SkiftetStatistical::Bandit::Prior" do
  def prior = SkiftetStatistical::Bandit::Prior

  def pair_close(actual, expected, tolerance = 1e-12)
    expect(actual.size).to eq(2)
    expect(actual[0]).to be_within(tolerance).of(expected[0])
    expect(actual[1]).to be_within(tolerance).of(expected[1])
  end

  describe "the plain priors" do
    it "LO1 flat is Beta(1, 1), one frozen pair" do
      expect(prior.flat).to eq([ 1.0, 1.0 ])
      expect(prior::FLAT).to eq([ 1.0, 1.0 ])
      expect(prior::FLAT).to be_frozen
      expect(prior.flat).to be_frozen
    end

    it "LO1 given carries a mean at a strength" do
      expect(prior.given(mean: 0.25, strength: 8.0)).to eq([ 2.0, 6.0 ])
      expect(prior.given(mean: 0.5, strength: 2.0)).to eq([ 1.0, 1.0 ])
    end

    it "LO1 amount is a frozen keyword Struct of mean, variance and strength" do
      amount = prior.amount(mean: 4.0, variance: 2_000.0, strength: 20.0)

      expect(prior::Amount.ancestors).to include(Struct)
      expect(prior::Amount.members).to eq(%i[mean variance strength])
      expect(prior::Amount.keyword_init?).to be(true)
      expect(amount).to be_a(prior::Amount)
      expect(amount).to be_frozen
      expect([ amount.mean, amount.variance, amount.strength ]).to eq([ 4.0, 2_000.0, 20.0 ])
    end
  end

  describe ".borrowed, the one leave-one-out rule" do
    it "LO1 is the group less the item's own evidence, in Floats" do
      borrowed = prior.borrowed(group: { views: 100, outcomes: 30 }, own: { views: 40, outcomes: 10 })

      expect(borrowed).to eq(views: 60.0, outcomes: 20.0)
      expect(borrowed.values).to all(be_a(Float))
      expect(prior.borrowed(group: { views: 100, outcomes: 30 })).to eq(views: 100.0, outcomes: 30.0)
      expect(prior.borrowed(group: { views: 12.5, outcomes: 4.25 }, own: { views: 2.5, outcomes: 1.0 }))
        .to eq(views: 10.0, outcomes: 3.25)
    end

    it "LO1 clamps: views at least 0, outcomes within [0, views]" do
      expect(prior.borrowed(group: { views: 10, outcomes: 3 }, own: { views: 12, outcomes: 1 }))
        .to eq(views: 0.0, outcomes: 0.0)
      expect(prior.borrowed(group: { views: 10, outcomes: 3 }, own: { views: 4, outcomes: 3.5 }))
        .to eq(views: 6.0, outcomes: 0.0)
      expect(prior.borrowed(group: { views: 10, outcomes: 8 }, own: { views: 5, outcomes: 1 }))
        .to eq(views: 5.0, outcomes: 5.0)
    end
  end

  describe ".pooled, a measured item's prior" do
    it "LO1 with no group views is the base: [1.0, 1.0] by default" do
      expect(prior.pooled(group: { views: 0, outcomes: 0 }, strength: 20.0)).to eq([ 1.0, 1.0 ])
      expect(prior.pooled(group: { views: 30, outcomes: 9 }, own: { views: 30, outcomes: 9 }, strength: 20.0))
        .to eq([ 1.0, 1.0 ])
      pair_close(prior.pooled(group: { views: 30, outcomes: 9 }, own: { views: 30, outcomes: 9 }, strength: 20.0,
                              base: [ 2.0, 3.0 ]), [ 2.0, 3.0 ])
    end

    it "LO1 below the strength is the group's posterior, the group less the item's own" do
      pair_close(prior.pooled(group: { views: 12, outcomes: 3 }, strength: 20.0), [ 4.0, 10.0 ])
      pair_close(prior.pooled(group: { views: 50, outcomes: 10 }, own: { views: 40, outcomes: 9 }, strength: 20.0),
                 [ 2.0, 10.0 ])
      pair_close(prior.pooled(group: { views: 20, outcomes: 5 }, strength: 20.0), [ 6.0, 16.0 ])
    end

    it "LO1 above the strength keeps the borrowed mean (base_a + y) / (base_a + base_b + n) at base + strength" do
      # n = 800, y = 50: mean 51 / 802 at weight 2 + 20
      above = prior.pooled(group: { views: 1_000, outcomes: 100 }, own: { views: 200, outcomes: 50 }, strength: 20.0)
      pair_close(above, [ 1.3990024937655858, 20.600997506234414 ])
      expect(above.sum).to be_within(1e-12).of(22.0)
      expect(above[0] / above.sum).to be_within(1e-15).of(51.0 / 802.0)

      # base (3, 7), n = 500, y = 60: mean 63 / 510 at weight 10 + 40
      based = prior.pooled(group: { views: 500, outcomes: 60 }, strength: 40.0, base: [ 3.0, 7.0 ])
      pair_close(based, [ 6.176470588235294, 43.8235294117647 ])
    end
  end

  describe ".pooled_new, a new item's prior" do
    it "LO1 with no others and no group is Beta(1, 1)" do
      expect(prior.pooled_new(others: [])).to eq([ 1.0, 1.0 ])
    end

    it "LO1 is the others' pooled borrowed rate (1 + Y) / (2 + N) at its weight, plus the item's own borrowed evidence" do
      others = [
        { group: { views: 400, outcomes: 40 }, own: { views: 100, outcomes: 10 } },
        { group: { views: 600, outcomes: 90 }, own: nil }
      ]
      # N = 300 + 600, Y = 30 + 90: 2 * (121 / 902, 781 / 902), plus the item's (20 views, 5 outcomes)
      pair_close(prior.pooled_new(others: others, group: { views: 30, outcomes: 6 }, own: { views: 10, outcomes: 1 }),
                 [ 5.2682926829268295, 16.73170731707317 ])

      # N = 80, Y = 8 at weight 10: 10 * (9 / 82, 73 / 82)
      pair_close(prior.pooled_new(others: [ { group: { views: 90, outcomes: 9 }, own: { views: 10, outcomes: 1 } } ],
                                  weight: 10.0),
                 [ 1.0975609756097562, 8.902439024390244 ])
    end

    it "LO1 borrows each other item before pooling: an other with nothing borrowed adds nothing" do
      pair_close(prior.pooled_new(others: [ { group: { views: 100, outcomes: 50 }, own: { views: 100, outcomes: 50 } } ]),
                 [ 1.0, 1.0 ], 1e-15)
    end
  end

  describe ".pooled_mean" do
    it "LO1 is (strength * prior_mean + outcomes) / (strength + views)" do
      expect(prior.pooled_mean(views: 200, outcomes: 30, prior_mean: 0.1, strength: 50.0)).to be_within(1e-15).of(0.14)
      expect(prior.pooled_mean(views: 0, outcomes: 0, prior_mean: 0.07, strength: 50.0)).to be_within(1e-15).of(0.07)
    end
  end

  describe ".tilts" do
    it "LO1 divides each multiplier by the geometric mean of them all, then clips" do
      tilts = prior.tilts({ "a" => 4.0, "b" => 1.0, "c" => 0.25 }, clip: [ 0.5, 2.0 ])
      expect(tilts.keys).to contain_exactly("a", "b", "c")
      expect(tilts["a"]).to be_within(1e-12).of(2.0)   # 4 / 1, clipped
      expect(tilts["b"]).to be_within(1e-12).of(1.0)
      expect(tilts["c"]).to be_within(1e-12).of(0.5)   # 0.25 / 1, clipped

      # the geometric mean of 1, 2, 3, 6 is sqrt(6)
      tilts = prior.tilts({ "x" => 1.0, "y" => 2.0, "z" => 3.0, "w" => 6.0 }, clip: [ 0.5, 2.0 ])
      expect(tilts["x"]).to be_within(1e-12).of(0.5)   # 1 / sqrt(6) = 0.408..., clipped
      expect(tilts["y"]).to be_within(1e-12).of(0.8164965809277261)
      expect(tilts["z"]).to be_within(1e-12).of(1.2247448713915892)
      expect(tilts["w"]).to be_within(1e-12).of(2.0)   # 6 / sqrt(6) = 2.449..., clipped

      tilts = prior.tilts({ "p" => 2.0, "q" => 8.0 }, clip: [ 0.1, 10.0 ])
      expect(tilts["p"]).to be_within(1e-12).of(0.5)
      expect(tilts["q"]).to be_within(1e-12).of(2.0)

      tilts = prior.tilts({ "p" => 3.5, "q" => 3.5, "r" => 3.5 }, clip: [ 0.5, 2.0 ])
      expect(tilts.values).to all(be_within(1e-12).of(1.0))
    end
  end

  describe ".chain, the nested levels' prior" do
    let(:top) { { views: 1_000, outcomes: 120, strength: 20.0 } }
    let(:middle) { { views: 200, outcomes: 30, strength: 30.0 } }
    let(:leaf) { { views: 40, outcomes: 8, strength: 30.0 } }

    it "LO1 three levels equal the hand-built fixture within 1e-12" do
      # p0 = (20 * 0.1 + (120 - 30)) / (20 + (1 000 - 200)) = 92 / 820
      # p1 = (30 * p0 + (30 - 8)) / (30 + (200 - 40))
      # m = 1.2 * p1 = 0.16020539152759947; the leaf answers [30 m, 30 (1 - m)]; its own (40, 8) is the item's evidence
      pair_close(prior.chain(levels: [ top, middle, leaf ], root_mean: 0.1, tilt: 1.2),
                 [ 4.806161745827984, 25.193838254172015 ])
    end

    it "LO1 two levels and one level" do
      # p0 = 92 / 820; m = 0.8 * p0
      pair_close(prior.chain(levels: [ top, middle ], root_mean: 0.1, tilt: 0.8),
                 [ 2.6926829268292685, 27.307317073170733 ])
      # tilt 1.0 and clip [0.001, 0.999] by default: m = p0
      pair_close(prior.chain(levels: [ top, middle ], root_mean: 0.1), [ 3.3658536585365852, 26.634146341463413 ])
      # one level: m = 1.5 * root_mean at the level's strength
      pair_close(prior.chain(levels: [ top ], root_mean: 0.1, tilt: 1.5), [ 3.0, 17.0 ])
    end

    it "LO1 clips the leaf's mean" do
      high = [ { views: 100, outcomes: 70, strength: 20.0 }, { views: 20, outcomes: 15, strength: 30.0 } ]
      # p0 = (12 + 55) / (20 + 80) = 0.67; 2 * 0.67 clipped to 0.999
      pair_close(prior.chain(levels: high, root_mean: 0.6, tilt: 2.0), [ 29.97, 0.030000000000000027 ])
      # 0.5 * 0.0015 clipped to 0.001
      pair_close(prior.chain(levels: [ top ], root_mean: 0.0015, tilt: 0.5), [ 0.02, 19.98 ])
      pair_close(prior.chain(levels: [ top ], root_mean: 0.1, tilt: 1.0, clip: [ 0.2, 0.3 ]), [ 4.0, 16.0 ])
    end

    it "LO1 borrows through the clamp: a child summed slightly above its parent borrows nothing" do
      levels = [ { views: 50.0, outcomes: 5.0, strength: 20.0 }, { views: 50.000001, outcomes: 5.0000001, strength: 30.0 } ]

      # p0 = (20 * 0.2 + 0) / (20 + 0); unclamped it would read [6.00000015..., 23.99999985...]
      pair_close(prior.chain(levels: levels, root_mean: 0.2), [ 6.0, 24.0 ])
    end

    it "LO1 its level means equal HierarchicalRates.leave_one_out_share where neither refuses nor clamps" do
      share = SkiftetStatistical::HierarchicalRates.method(:leave_one_out_share)
      p0 = share.call(parent_count: 120.0, parent_total: 1_000.0, node_count: 30.0, node_total: 200.0,
                      prior_share: 0.1, prior_strength: 20.0)
      p1 = share.call(parent_count: 30.0, parent_total: 200.0, node_count: 8.0, node_total: 40.0,
                      prior_share: p0, prior_strength: 30.0)
      two = prior.chain(levels: [ top, middle ], root_mean: 0.1)
      three = prior.chain(levels: [ top, middle, leaf ], root_mean: 0.1)
      expect(two[0] / two.sum).to be_within(1e-12).of(p0)
      expect(three[0] / three.sum).to be_within(1e-12).of(p1)

      rng = Random.new(5_150)
      200.times do
        leaf_views = rng.rand(1.0..50.0)
        leaf_outcomes = rng.rand * leaf_views
        middle_views = leaf_views + rng.rand(1.0..200.0)
        middle_outcomes = leaf_outcomes + (rng.rand * (middle_views - leaf_views))
        top_views = middle_views + rng.rand(1.0..900.0)
        top_outcomes = middle_outcomes + (rng.rand * (top_views - middle_views))
        root = rng.rand(0.01..0.3)
        strengths = Array.new(3) { rng.rand(5.0..40.0) }
        levels = [ [ top_views, top_outcomes ], [ middle_views, middle_outcomes ], [ leaf_views, leaf_outcomes ] ]
                 .each_with_index.map { |(v, o), i| { views: v, outcomes: o, strength: strengths[i] } }

        p0 = share.call(parent_count: top_outcomes, parent_total: top_views, node_count: middle_outcomes,
                        node_total: middle_views, prior_share: root, prior_strength: strengths[0])
        p1 = share.call(parent_count: middle_outcomes, parent_total: middle_views, node_count: leaf_outcomes,
                        node_total: leaf_views, prior_share: p0, prior_strength: strengths[1])
        wide = [ 1e-9, 1.0 - 1e-9 ]
        two = prior.chain(levels: levels.first(2), root_mean: root, clip: wide)
        three = prior.chain(levels: levels, root_mean: root, clip: wide)

        expect(two[0] / two.sum).to be_within(1e-12).of(p0)
        expect(two.sum).to be_within(1e-12).of(strengths[1])
        expect(three[0] / three.sum).to be_within(1e-12).of(p1)
        expect(three.sum).to be_within(1e-12).of(strengths[2])
      end
    end
  end

  describe ".pooled_amount" do
    it "LO1 is the borrowed mean and variance (outcome_squares / views - mean^2) at min(strength, n)" do
      amount = prior.pooled_amount(group: { views: 10, outcomes: 40, outcome_squares: 400 }, strength: 20.0)
      expect(amount).to be_a(prior::Amount)
      expect(amount).to be_frozen
      expect(amount.mean).to be_within(1e-12).of(4.0)
      expect(amount.variance).to be_within(1e-12).of(24.0)    # 40 - 16
      expect(amount.strength).to be_within(1e-12).of(10.0)    # min(20, 10)

      # borrowed: 20 views, 120 in amounts, 1 600 in squares
      amount = prior.pooled_amount(group: { views: 30, outcomes: 150, outcome_squares: 2_000 },
                                   own: { views: 10, outcomes: 30, outcome_squares: 400 }, strength: 5.0)
      expect(amount.mean).to be_within(1e-12).of(6.0)
      expect(amount.variance).to be_within(1e-12).of(44.0)    # 80 - 36
      expect(amount.strength).to be_within(1e-12).of(5.0)
    end

    it "LO1 refuses a variance of 0 and nothing borrowed, as ArgumentErrors" do
      expect(prior.pooled_amount(group: { views: 10, outcomes: 40, outcome_squares: 400 }, strength: 20.0))
        .to be_a(prior::Amount)
      # 250 / 10 - 5^2 = 0
      expect { prior.pooled_amount(group: { views: 10, outcomes: 50, outcome_squares: 250 }, strength: 20.0) }
        .to raise_error(ArgumentError)
      # 200 / 10 - 5^2 = -5, a residual below 0 that reads 0
      expect { prior.pooled_amount(group: { views: 10, outcomes: 50, outcome_squares: 200 }, strength: 20.0) }
        .to raise_error(ArgumentError)
      group = { views: 10, outcomes: 40, outcome_squares: 400 }
      expect { prior.pooled_amount(group: group, own: group, strength: 20.0) }.to raise_error(ArgumentError)
      expect { prior.pooled_amount(group: { views: 0, outcomes: 0, outcome_squares: 0 }, strength: 20.0) }
        .to raise_error(ArgumentError)
    end
  end
end
