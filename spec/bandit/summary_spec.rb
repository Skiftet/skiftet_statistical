# frozen_string_literal: true

# LO3: the acceptance of the posterior set's summaries (01 GB12, GB18), written from the contract before their
# implementers: `summary` (LC2) and `order_summary` (LD1); each example's name says which. Until then each fails on a
# missing constant or method.
#
# `summary(credible_mass:, among:)` answers, per item, its views, its mean and its exact equal-tailed interval on the
# pair as given (BetaFunction.quantile, or mean + sd * the Normal's quantile), and, among `among` (nil: every item),
# P(best) and the expected loss E[max] - mean on the scaled pairs, a loss that rounds below 0 answered as 0; its shape
# is Allocator.from_summary's. `among: []` is a means-only summary: views, mean and interval for every item, for a set
# of any size, with no P(best) and no loss computed. P(best) and the loss refuse a parameter below
# BetaArgmax::SMALLEST_PARAMETER, as the exact odds do (01 GB6). The references, none from the code under test:
# - p_best: BetaArgmax.probabilities (0.3.0, held exact to 1e-8 by its pins), within 1e-12; for Normals Simpson's rule
#   on 400 000 intervals over 14 standard deviations (error under 2e-13), within 1e-8;
# - E[max] of Betas: exact, worked in fractions (whole members' distribution functions are polynomials; one member of
#   any parameters beside them is a sum of its moments E[X**m] = product over r < m of (a + r) / (a + b + r)), each also
#   matched by a trapezoid sum to 1e-11; for Normals Simpson's rule; the loss within 1e-8;
# - the Normal's quantiles 1.9599639845400534 (0.975) and 1.6448536269514715 (0.95), from an inverse distribution
#   function outside this library, within 1e-9 relative;
# - the means-only intervals: Beta(a, 1) and Beta(1, b) members, whose quantiles have the closed forms q**(1 / a) and
#   1 - (1 - q)**(1 / b), within 1e-12 relative (checked against 50-digit decimals and 0.3.0's BetaFunction.quantile,
#   at most 2.7e-14 apart); at a = b = 0.5 they are exact: 1/1600, 1521/1600, 79/1600 and 1599/1600.
# `order_summary(order)`: the posterior P(X_s1 > ... > X_sK), that the shown order is the best one, and P(item at
# position) under the posterior alone, by hand at K = 3: Beta(a, 1) members race as Plackett-Luce with weights a
# (-log X is exponential with rate a); Beta(1, b) members as the reverse race (-log(1 - X) is exponential with rate b,
# so the last place goes first, with probability b over the sum); Normals from an independent Stieltjes sum on 4 000
# quantile levels an item (every order of two members within 5e-9 of the closed form Phi(d)). Tolerance 1e-4, the odds'
# own (01 Decisions).
module SummaryOracle
  THREE = { "a" => [ 2.0, 8.0 ], "b" => [ 3.0, 7.0 ], "c" => [ 5.0, 5.0 ] }.freeze
  THREE_VIEWS = { "a" => 120.0, "b" => 340.0, "c" => 560.0 }.freeze
  # Exact E[max] of THREE, and of its first two members.
  THREE_MAXIMUM = 0.5229418134250547
  FIRST_TWO_MAXIMUM = 0.33359133126934987
  # name, pairs, exact E[max]
  BETAS = {
    "four Betas, an alpha of 0.3 among them" => [
      { "a" => [ 0.3, 20.0 ], "b" => [ 1.0, 20.0 ], "c" => [ 2.0, 30.0 ], "d" => [ 5.0, 90.0 ] }, 0.08903251985698629
    ],
    "five Betas, an alpha of 0.02 among them" => [
      { "a" => [ 0.02, 19.98 ], "b" => [ 3.0, 60.0 ], "c" => [ 4.0, 70.0 ], "d" => [ 6.0, 95.0 ], "e" => [ 2.0, 45.0 ] },
      0.08066474297473783
    ],
    "six Betas, Beta(0.5, 4.5) among them" => [
      { "a" => [ 12.0, 88.0 ], "b" => [ 15.0, 85.0 ], "c" => [ 9.0, 91.0 ], "d" => [ 20.0, 180.0 ], "e" => [ 3.0, 17.0 ],
        "f" => [ 0.5, 4.5 ] },
      0.21353379000310863
    ]
  }.freeze
  NORMALS = { "a" => [ 5.0, 0.4 ], "b" => [ 5.3, 0.5 ], "c" => [ 4.8, 0.3 ], "d" => [ 5.1, 0.8 ] }.freeze
  NORMALS_LARGEST = { "a" => 0.16748793058298564, "b" => 0.43470920285171183, "c" => 0.05189145526742245,
                      "d" => 0.34591141129780234 }.freeze
  NORMALS_MAXIMUM = 5.6581335636537595
  Z975 = 1.9599639845400534
  Z95 = 1.6448536269514715

  # Beta(1, 1), Beta(2, 1), Beta(3, 1): Plackett-Luce with weights 1, 2 and 3.
  POWERS = { "a" => [ 1.0, 1.0 ], "b" => [ 2.0, 1.0 ], "c" => [ 3.0, 1.0 ] }.freeze
  POWERS_ORDERS = {
    %w[c b a] => 1 / 3r, %w[c a b] => 1 / 6r, %w[b c a] => 1 / 4r, %w[b a c] => 1 / 12r, %w[a c b] => 1 / 10r,
    %w[a b c] => 1 / 15r
  }.freeze
  POWERS_POSITIONS = { "a" => [ 1 / 6r, 1 / 4r, 7 / 12r ], "b" => [ 1 / 3r, 2 / 5r, 4 / 15r ],
                       "c" => [ 1 / 2r, 7 / 20r, 3 / 20r ] }.freeze
  # Beta(1, 0.5), Beta(1, 1), Beta(1, 2): the reverse race with rates 0.5, 1 and 2.
  REVERSE = { "x" => [ 1.0, 0.5 ], "y" => [ 1.0, 1.0 ], "z" => [ 1.0, 2.0 ] }.freeze
  REVERSE_ORDERS = {
    %w[x y z] => 8 / 21r, %w[x z y] => 8 / 35r, %w[y x z] => 4 / 21r, %w[y z x] => 2 / 21r, %w[z x y] => 2 / 35r,
    %w[z y x] => 1 / 21r
  }.freeze
  REVERSE_POSITIONS = { "x" => [ 64 / 105r, 26 / 105r, 1 / 7r ], "y" => [ 2 / 7r, 3 / 7r, 2 / 7r ],
                        "z" => [ 11 / 105r, 34 / 105r, 4 / 7r ] }.freeze
  # Three Normals, from the independent sum.
  NORMAL3 = { "a" => [ 5.0, 0.4 ], "b" => [ 5.3, 0.5 ], "c" => [ 4.8, 0.3 ] }.freeze
  NORMAL3_ORDERS = {
    %w[a b c] => 0.17862818, %w[a c b] => 0.09493225, %w[b a c] => 0.38186132, %w[b c a] => 0.24392425,
    %w[c a b] => 0.04614550, %w[c b a] => 0.05450851
  }.freeze
  NORMAL3_POSITIONS = { "a" => [ 0.27356043, 0.42800682, 0.29843276 ], "b" => [ 0.62578556, 0.23313669, 0.14107775 ],
                        "c" => [ 0.10065401, 0.33885650, 0.56048949 ] }.freeze
end

RSpec.describe "SkiftetStatistical::Bandit::Posteriors summaries" do
  def posteriors = SkiftetStatistical::Bandit::Posteriors
  def argmax = SkiftetStatistical::BetaArgmax
  def quantile(level, alpha, beta) = SkiftetStatistical::BetaFunction.quantile(level, alpha, beta)
  def oracle = SummaryOracle

  def expect_relative(actual, expected, tolerance, label)
    expect(actual).to be_a(Float), "#{label}: #{actual.inspect} is not a Float"
    expect((actual - expected).abs).to be <= tolerance * expected.abs, "#{label}: #{actual} is not #{expected}"
  end

  describe "summary" do
    it "LO3 (LC2) answers every item its views, mean, interval, p_best and expected loss, the shape from_summary reads" do
      answer = posteriors.beta(oracle::THREE, views: oracle::THREE_VIEWS).summary
      expect(answer.keys).to match_array(oracle::THREE.keys)
      answer.each_value do |entry|
        expect(entry.keys).to contain_exactly(:views, :mean, :lower, :upper, :p_best, :expected_loss)
      end
    end

    it "LO3 (LC2) gives each item its views, its mean and its exact equal-tailed interval on the pair as given" do
      answer = posteriors.beta(oracle::THREE, views: oracle::THREE_VIEWS).summary(credible_mass: 0.95)
      oracle::THREE.each do |item, (alpha, beta)|
        expect(answer[item][:views]).to eq(oracle::THREE_VIEWS[item])
        expect(answer[item][:mean]).to be_within(1e-15).of(alpha / (alpha + beta))
        expect_relative(answer[item][:lower], quantile(0.025, alpha, beta), 1e-12, "#{item} lower")
        expect_relative(answer[item][:upper], quantile(0.975, alpha, beta), 1e-12, "#{item} upper")
      end
    end

    it "LO3 (LC2) reads credible_mass: 0.9 at the 0.05 and 0.95 quantiles, and 0.95 by default" do
      set = posteriors.beta(oracle::THREE)
      narrow = set.summary(credible_mass: 0.9)
      oracle::THREE.each do |item, (alpha, beta)|
        expect_relative(narrow[item][:lower], quantile(0.05, alpha, beta), 1e-12, "#{item} lower")
        expect_relative(narrow[item][:upper], quantile(0.95, alpha, beta), 1e-12, "#{item} upper")
      end
      expect(set.summary).to eq(set.summary(credible_mass: 0.95))
    end

    it "LO3 (LC2) answers p_best as BetaArgmax.probabilities within 1e-12 and the loss E[max] - mean within 1e-8 of " \
       "the exact integral" do
      { "three whole members" => [ oracle::THREE, oracle::THREE_MAXIMUM ] }.merge(oracle::BETAS)
                                                                       .each do |label, (pairs, maximum)|
        answer = posteriors.beta(pairs).summary
        largest = argmax.probabilities(pairs.values)
        pairs.each_with_index do |(item, (alpha, beta)), index|
          expect(answer[item][:p_best]).to be_within(1e-12).of(largest[index]), "#{label}, #{item}"
          expect(answer[item][:expected_loss]).to be_within(1e-8).of(maximum - (alpha / (alpha + beta))),
                                                  "#{label}, #{item}"
        end
      end
    end

    it "LO3 (LC2) takes p_best and the loss among the items named, and gives the others neither" do
      answer = posteriors.beta(oracle::THREE).summary(among: %w[a b])
      largest = argmax.probabilities([ oracle::THREE["a"], oracle::THREE["b"] ])
      expect(answer["a"][:p_best]).to be_within(1e-12).of(largest[0])
      expect(answer["b"][:p_best]).to be_within(1e-12).of(largest[1])
      expect(answer["a"][:expected_loss]).to be_within(1e-8).of(oracle::FIRST_TWO_MAXIMUM - 0.2)
      expect(answer["b"][:expected_loss]).to be_within(1e-8).of(oracle::FIRST_TWO_MAXIMUM - 0.3)
      expect(answer["c"][:p_best]).to be_nil
      expect(answer["c"][:expected_loss]).to be_nil
      expect(answer["c"][:mean]).to be_within(1e-15).of(0.5)
      expect_relative(answer["c"][:lower], quantile(0.025, 5.0, 5.0), 1e-12, "c lower")
    end

    it "LO3 (LC2) refuses among naming an item the set does not hold" do
      set = posteriors.beta(oracle::THREE)
      expect { set.summary(among: %w[a zz]) }.to raise_error(ArgumentError)
    end

    it "LO3 (LC2) reads the interval on the pair as given above LARGEST_TOTAL, and p_best and the loss on the scaled " \
       "pair" do
      given = { "a" => [ 1_500_000.0, 2_500_000.0 ], "b" => [ 1_500_400.0, 2_499_600.0 ] }
      scaled = [ [ 375_000.0, 625_000.0 ], [ 375_100.0, 624_900.0 ] ]
      answer = posteriors.beta(given).summary
      given.each_with_index do |(item, (alpha, beta)), index|
        expect(answer[item][:mean]).to be_within(1e-15).of(alpha / (alpha + beta))
        expect_relative(answer[item][:lower], quantile(0.025, alpha, beta), 1e-12, "#{item} lower")
        expect_relative(answer[item][:upper], quantile(0.975, alpha, beta), 1e-12, "#{item} upper")
        wide = quantile(0.975, *scaled[index]) - quantile(0.025, *scaled[index])
        expect(answer[item][:upper] - answer[item][:lower]).to be < 0.6 * wide, "#{item}: the interval was scaled"
        expect(answer[item][:p_best]).to be_within(1e-12).of(argmax.probabilities(scaled)[index])
        expect(answer[item][:expected_loss]).to be_within(1e-12).of(
          argmax.expected_maximum(scaled) - (alpha / (alpha + beta))
        )
      end
    end

    it "LO3 (LC2) answers a Normal set: the mean, mean -+ z * sd, p_best and the loss against independent integrals" do
      set = posteriors.normal(oracle::NORMALS)
      answer = set.summary
      narrow = set.summary(credible_mass: 0.9)
      direct = SkiftetStatistical::Bandit::NormalArgmax.probabilities(oracle::NORMALS.values)
      oracle::NORMALS.each_with_index do |(item, (mean, sd)), index|
        expect(answer[item][:mean]).to eq(mean)
        expect_relative(answer[item][:lower], mean - (oracle::Z975 * sd), 1e-9, "#{item} lower")
        expect_relative(answer[item][:upper], mean + (oracle::Z975 * sd), 1e-9, "#{item} upper")
        expect_relative(narrow[item][:lower], mean - (oracle::Z95 * sd), 1e-9, "#{item} lower at 0.9")
        expect(answer[item][:p_best]).to be_within(1e-8).of(oracle::NORMALS_LARGEST[item])
        expect(answer[item][:p_best]).to be_within(1e-12).of(direct[index])
        expect(answer[item][:expected_loss]).to be_within(1e-8).of(oracle::NORMALS_MAXIMUM - mean)
      end
    end

    it "LO3 (LC2) gives one item a p_best of 1.0 and no loss, Beta or Normal" do
      beta = posteriors.beta({ "only" => [ 3.0, 7.0 ] }).summary["only"]
      expect(beta[:p_best]).to be_within(1e-12).of(1.0)
      expect(beta[:expected_loss]).to be_between(0.0, 1e-8)
      normal = posteriors.normal({ "only" => [ 12.5, 2.0 ] }).summary(credible_mass: 0.95)["only"]
      expect(normal[:p_best]).to be_within(1e-12).of(1.0)
      expect(normal[:expected_loss]).to be_between(0.0, 1e-8)
      expect_relative(normal[:lower], 12.5 - (oracle::Z975 * 2.0), 1e-9, "lower")
      expect_relative(normal[:upper], 12.5 + (oracle::Z975 * 2.0), 1e-9, "upper")
    end

    # A leader far ahead has an exact loss below P(the other item is larger): 1.9e-85 for the Betas (E[(1 - A)**99] for
    # A ~ Beta(900, 100), worked in fractions) and 64 standard deviations out for the Normals. So its E[max] - mean may
    # round to either side of 0; the answer is never below 0.
    it "LO3 (LC2) never answers an expected loss below 0, so Allocator.from_summary takes a leader far ahead" do
      [ posteriors.beta({ "ahead" => [ 900.0, 100.0 ], "behind" => [ 1.0, 99.0 ] }),
        posteriors.normal({ "ahead" => [ 10.0, 0.1 ], "behind" => [ 1.0, 0.1 ] }) ].each do |set|
        summary = set.summary
        expect(summary["ahead"][:expected_loss]).to be_between(0.0, 1e-8), set.kind.inspect
        expect(summary["behind"][:expected_loss]).to be > 0.0, set.kind.inspect
        result = SkiftetStatistical::Bandit::Allocator.from_summary(
          arms: %w[ahead behind], summary: summary, exposures: { "ahead" => 1_000, "behind" => 1_000 },
          status: "running", winner_arm: nil, previous_weights: nil, hold_satisfied: false
        )
        expect(result[:stats].fetch(:ahead)[:expected_loss]).to eq(summary["ahead"][:expected_loss]), set.kind.inspect
      end
    end

    # 60 items, more than EXACT_PICK_MAX: Beta(a, 1) for the even ones and Beta(1, a) for the odd ones, a = 0.5 up to
    # 15.0 in steps of 0.5, then 60 Normals.
    it "LO3 (LC2) answers a means-only summary with among: [], views, mean and interval for 60 items, no p_best or loss" do
      pairs = (0...60).to_h do |i|
        a = 0.5 + (0.5 * (i / 2))
        [ "item-#{i}", i.even? ? [ a, 1.0 ] : [ 1.0, a ] ]
      end
      views = pairs.keys.each_with_index.to_h { |item, i| [ item, 10.0 * i ] }
      answer = posteriors.beta(pairs, views: views).summary(among: [])
      expect(answer.keys).to eq(pairs.keys)
      pairs.each do |item, (alpha, beta)|
        if beta == 1.0
          lower = 0.025**(1.0 / alpha)
          upper = 0.975**(1.0 / alpha)
        else
          lower = 1.0 - (0.975**(1.0 / beta))
          upper = 1.0 - (0.025**(1.0 / beta))
        end
        entry = answer.fetch(item)
        expect(entry[:views]).to eq(views[item]), item
        expect(entry[:mean]).to be_within(1e-15).of(alpha / (alpha + beta)), item
        expect_relative(entry[:lower], lower, 1e-12, "#{item} lower")
        expect_relative(entry[:upper], upper, 1e-12, "#{item} upper")
        expect(entry[:p_best]).to be_nil, item
        expect(entry[:expected_loss]).to be_nil, item
      end
      expect_relative(answer["item-0"][:lower], 1 / 1600r, 1e-12, "Beta(0.5, 1) lower")
      expect_relative(answer["item-0"][:upper], 1521 / 1600r, 1e-12, "Beta(0.5, 1) upper")
      expect_relative(answer["item-1"][:lower], 79 / 1600r, 1e-12, "Beta(1, 0.5) lower")
      expect_relative(answer["item-1"][:upper], 1599 / 1600r, 1e-12, "Beta(1, 0.5) upper")

      normals = (0...60).to_h { |i| [ "item-#{i}", [ 1.0 + (0.1 * i), 0.5 + (0.01 * i) ] ] }
      normal = posteriors.normal(normals).summary(among: [])
      normals.each do |item, (mean, sd)|
        expect(normal[item][:mean]).to eq(mean), item
        expect_relative(normal[item][:lower], mean - (oracle::Z975 * sd), 1e-9, "#{item} lower")
        expect_relative(normal[item][:upper], mean + (oracle::Z975 * sd), 1e-9, "#{item} upper")
        expect(normal[item][:p_best]).to be_nil, item
        expect(normal[item][:expected_loss]).to be_nil, item
      end
    end

    it "LO3 (LC2) refuses p_best and the loss over a parameter below BetaArgmax::SMALLEST_PARAMETER (01 GB6)" do
      [ { "a" => [ 2.0, 3.0 ], "b" => [ 1e-9, 1.0 ] }, { "a" => [ 2.0, 3.0 ], "b" => [ 1.0, 5e-7 ] } ].each do |pairs|
        set = posteriors.beta(pairs)
        set.method(:summary)
        expect { set.summary }.to raise_error(ArgumentError), pairs.inspect
        expect { set.summary(among: %w[a b]) }.to raise_error(ArgumentError), pairs.inspect
      end
    end

    # 01 GB6's scope, ruled at the L3 barrier (2026-11-01): summary refuses such a parameter of any item on every path,
    # whatever among names, the means-only one included, whose interval BetaFunction.quantile cannot read there.
    it "the L3 barrier: refuses a parameter below BetaArgmax::SMALLEST_PARAMETER on every path, means-only included" do
      [ { "a" => [ 2.0, 3.0 ], "b" => [ 1e-9, 1.0 ] }, { "a" => [ 2.0, 3.0 ], "b" => [ 1.0, 5e-7 ] },
        { "a" => [ 2.0, 3.0 ], "b" => [ 1e-300, 1.0 ] } ].each do |pairs|
        set = posteriors.beta(pairs)
        [ [], %w[a], %w[b] ].each do |among|
          expect { set.summary(among: among) }.to raise_error(ArgumentError), "#{pairs.inspect} among #{among.inspect}"
        end
      end
      edge = posteriors.beta({ "a" => [ 2.0, 3.0 ], "b" => [ 1e-6, 1.0 ] }).summary(among: [])
      expect(edge["b"][:mean]).to be_within(1e-21).of(1e-6 / (1.0 + 1e-6))
      expect(edge["b"][:p_best]).to be_nil
    end

    # 01 GB6 and D16 as ruled at the L3 barrier's second round (2026-11-01): each computation refuses the pairs it
    # reads. The interval reads the pair as given, so a pair whose given parameters are all at least SMALLEST_PARAMETER
    # is answered there even when the scaling (to a total of 1e6) takes one below it: [1, 2e12] scales to about
    # [5e-7, 1e6]. P(best) and the loss read the scaled pairs, so they refuse it, over every item whenever among names
    # one, as the keyed draws and the pick odds do. [1, 9e11] scales to about [1.1e-6, 1e6] and is taken everywhere.
    it "the L3 barrier, round 2: the interval reads the pair as given, P(best) and the loss the scaled pairs" do
      tail = (1.0 - 0.95) / 2.0
      set = posteriors.beta({ "a" => [ 2.0, 3.0 ], "b" => [ 1.0, 2e12 ] })
      expect(set.parameters["b"].min).to be < argmax::SMALLEST_PARAMETER

      means_only = set.summary(among: [])
      expect(means_only["b"][:mean]).to eq(1.0 / (1.0 + 2e12))
      expect(means_only["b"][:lower]).to eq(quantile(tail, 1.0, 2e12))
      expect(means_only["b"][:upper]).to eq(quantile(1.0 - tail, 1.0, 2e12))
      expect(means_only["a"][:lower]).to eq(quantile(tail, 2.0, 3.0))
      expect(means_only["a"][:upper]).to eq(quantile(1.0 - tail, 2.0, 3.0))
      means_only.each_value do |entry|
        expect(entry[:p_best]).to be_nil
        expect(entry[:expected_loss]).to be_nil
      end

      [ nil, %w[a], %w[b], %w[a b] ].each do |among|
        expect { set.summary(among: among) }
          .to raise_error(ArgumentError, /P\(best\) and the expected loss on the scaled pairs/), among.inspect
      end
      expect { set.pick(key: "k") }.to raise_error(ArgumentError, /quantile draw on the scaled pairs/)
      expect { set.pick_probabilities }.to raise_error(ArgumentError, /exact pick odds on the scaled pairs/)
      expect(set.pick(rng: Random.new(1)).head).to eq("a").or eq("b")

      taken = posteriors.beta({ "a" => [ 2.0, 3.0 ], "b" => [ 1.0, 9e11 ] })
      expect(taken.parameters["b"].min).to be >= argmax::SMALLEST_PARAMETER
      answer = taken.summary
      expect(answer["b"][:lower]).to eq(quantile(tail, 1.0, 9e11))
      expect(answer["b"][:upper]).to eq(quantile(1.0 - tail, 1.0, 9e11))
      expect(answer["a"][:p_best] + answer["b"][:p_best]).to be_within(1e-12).of(1.0)
    end

    it "the L3 barrier, round 2: each refusal's words name what it computes, on the pairs it reads" do
      given_below = posteriors.beta({ "a" => [ 2.0, 3.0 ], "b" => [ 1e-9, 1.0 ] })
      expect { given_below.summary(among: []) }
        .to raise_error(ArgumentError, /"b".*\[1\.0e-09, 1\.0\].*the interval, BetaFunction\.quantile on the pair as given/)

      scaled_below = posteriors.beta({ "a" => [ 2.0, 3.0 ], "b" => [ 1.0, 2e12 ] })
      expect { scaled_below.summary(among: %w[a]) }.to raise_error(ArgumentError) do |error|
        expect(error.message).to include("\"b\"", scaled_below.parameters["b"].inspect)
        expect(error.message).to include("P(best) and the expected loss on the scaled pairs")
        expect(error.message).not_to include("interval")
      end
    end

    it "LO3 (LC2) is taken by Allocator.from_summary as it is, String keys and all, statistics unchanged" do
      set = posteriors.beta(oracle::THREE, views: oracle::THREE_VIEWS.transform_values { 1_000.0 })
      summary = set.summary
      result = SkiftetStatistical::Bandit::Allocator.from_summary(
        arms: %w[a b c], summary: summary, exposures: summary.transform_values { |entry| entry[:views].to_i },
        status: "running", winner_arm: nil, previous_weights: nil, hold_satisfied: false
      )
      expect(result[:phase]).to eq(:shifting)
      oracle::THREE.each_key do |item|
        stats = result[:stats].fetch(item.to_sym)
        %i[mean lower upper p_best expected_loss].each do |key|
          expect(stats[key]).to eq(summary[item][key]), "#{item} #{key}"
        end
        expect(stats[:exposures]).to eq(1_000)
      end
    end
  end

  describe "order_summary" do
    it "LO3 (LD1) answers by hand at K = 3 among Beta(a, 1) members: P(c > b > a) = 1/3, and every rank's probability" do
      set = posteriors.beta(oracle::POWERS)
      oracle::POWERS_ORDERS.each do |order, exact|
        answer = set.order_summary(order)
        expect(answer.keys).to contain_exactly(:probability, :positions)
        expect(answer[:probability]).to be_within(1e-4).of(exact.to_f), order.inspect
      end
      positions = set.order_summary(%w[c b a])[:positions]
      expect(positions.keys).to match_array(oracle::POWERS_POSITIONS.keys)
      oracle::POWERS_POSITIONS.each do |item, row|
        expect(positions[item].size).to eq(3)
        row.each_with_index { |p, slot| expect(positions[item][slot]).to be_within(1e-4).of(p.to_f), "#{item} at #{slot}" }
      end
    end

    it "LO3 (LD1) answers by hand at K = 3 among Beta(1, b) members, a beta of 0.5 among them" do
      set = posteriors.beta(oracle::REVERSE)
      oracle::REVERSE_ORDERS.each do |order, exact|
        expect(set.order_summary(order)[:probability]).to be_within(1e-4).of(exact.to_f), order.inspect
      end
      positions = set.order_summary(%w[x y z])[:positions]
      oracle::REVERSE_POSITIONS.each do |item, row|
        row.each_with_index { |p, slot| expect(positions[item][slot]).to be_within(1e-4).of(p.to_f), "#{item} at #{slot}" }
      end
    end

    it "LO3 (LD1) answers three Normals within 1e-4 of the independent sum, its orders summing to 1" do
      set = posteriors.normal(oracle::NORMAL3)
      total = oracle::NORMAL3_ORDERS.sum do |order, reference|
        answer = set.order_summary(order)
        expect(answer[:probability]).to be_within(1e-4).of(reference), order.inspect
        answer[:probability]
      end
      expect(total).to be_within(1e-4).of(1.0)
      positions = set.order_summary(%w[b a c])[:positions]
      oracle::NORMAL3_POSITIONS.each do |item, row|
        row.each_with_index { |p, slot| expect(positions[item][slot]).to be_within(1e-4).of(p), "#{item} at #{slot}" }
      end
      3.times { |slot| expect(positions.values.sum { |row| row[slot] }).to be_within(1e-4).of(1.0), "position #{slot}" }
    end
  end
end
