# frozen_string_literal: true

# Building a posterior set: yes/no tallies become Beta pairs, amount tallies Normal pairs for the mean per view, and
# pairs may be given as they are. The set holds its items (input order, the tie order), its kind, its parameters
# ({ item => pair }, a Beta pair above LARGEST_TOTAL scaled to it with its mean kept) and its views ({ item => Float }).
# Every literal was worked outside the library from the formulas, step by step.
RSpec.describe "SkiftetStatistical::Bandit::Posteriors" do
  def posteriors = SkiftetStatistical::Bandit::Posteriors
  def prior = SkiftetStatistical::Bandit::Prior
  def decay = SkiftetStatistical::Bandit::Decay
  def weighted_posterior = SkiftetStatistical::Bandit::WeightedPosterior

  def pair_close(actual, expected, tolerance = 1e-12)
    expect(actual.size).to eq(2)
    expect(actual[0]).to be_within(tolerance).of(expected[0])
    expect(actual[1]).to be_within(tolerance).of(expected[1])
  end

  # One item name held twice, as two distinct but equal Strings.
  def repeated(value)
    items = {}.compare_by_identity
    items[String.new("a")] = value
    items[String.new("a")] = value
    items
  end

  let(:amount_prior) { prior.amount(mean: 4.0, variance: 2_000.0, strength: 20.0) }

  it "LO1 the three limits: EXACT_PICK_MAX 20, EXACT_ORDER_MAX 10, LARGEST_TOTAL 1e6" do
    expect(posteriors::EXACT_PICK_MAX).to eq(20)
    expect(posteriors::EXACT_ORDER_MAX).to eq(10)
    expect(posteriors::LARGEST_TOTAL).to eq(1_000_000.0)
  end

  describe ".yes_no" do
    it "LO1 is Beta(a0 + c, b0 + n - c) on the raw counts, flat by default" do
      set = posteriors.yes_no({ "a" => { views: 40, outcomes: 8 }, "b" => { views: 10.5, outcomes: 3.25 } })

      expect(set.kind).to eq(:beta)
      expect(set.items).to eq(%w[a b])
      expect(set.parameters).to eq("a" => [ 9.0, 33.0 ], "b" => [ 4.25, 8.25 ])
      expect(set.parameters.keys).to eq(set.items)
      expect(set.views).to eq("a" => 40.0, "b" => 10.5)
      expect(set.views.values).to all(be_a(Float))
      expect(posteriors.yes_no({ "a" => { views: 40, outcomes: 8 } }, priors: prior.flat).parameters)
        .to eq("a" => [ 9.0, 33.0 ])
    end

    it "LO1 reads Symbol and String keys alike and clamps outcomes to the views" do
      expect(posteriors.yes_no({ "a" => { "views" => 40, "outcomes" => 8 } }).parameters).to eq("a" => [ 9.0, 33.0 ])
      expect(posteriors.yes_no({ "a" => { views: 10, outcomes: 12 } }).parameters).to eq("a" => [ 11.0, 1.0 ])
    end

    it "LO1 takes one prior pair for every item, or one per item" do
      tallies = { "a" => { views: 40, outcomes: 8 }, "b" => { views: 10, outcomes: 1 } }

      expect(posteriors.yes_no(tallies, priors: [ 2.0, 3.0 ]).parameters)
        .to eq("a" => [ 10.0, 35.0 ], "b" => [ 3.0, 12.0 ])
      expect(posteriors.yes_no(tallies, priors: { "a" => [ 2.0, 3.0 ], "b" => [ 0.5, 0.5 ] }).parameters)
        .to eq("a" => [ 10.0, 35.0 ], "b" => [ 1.5, 9.5 ])
    end

    it "LO1 rescales weighted outcomes to the views without the squared sum" do
      tally = { views: 800, outcomes: 80, weighted_views: 1_600.0, weighted_outcomes: 150.0 }

      # c = 150 * 800 / 1 600 = 75
      pair_close(posteriors.yes_no({ "a" => tally }).parameters["a"], [ 76.0, 726.0 ])
    end

    it "LO1 rescales to the Kish size only with the squared sum, never above the views" do
      kish = { views: 800, outcomes: 80, weighted_views: 1_600.0, weighted_outcomes: 150.0,
               weighted_views_squared: 4_000.0 }
      capped = { views: 100, outcomes: 10, weighted_views: 120.0, weighted_outcomes: 12.0,
                 weighted_views_squared: 130.0 }

      # n = 1 600^2 / 4 000 = 640, c = 150 * 640 / 1 600 = 60; the set's views stay the tally's views
      pair_close(posteriors.yes_no({ "a" => kish }).parameters["a"], [ 61.0, 581.0 ])
      expect(posteriors.yes_no({ "a" => kish }).views).to eq("a" => 800.0)
      # n = 120^2 / 130 = 110.8, above the 100 views, so n = 100 and c = 12 * 100 / 120 = 10
      pair_close(posteriors.yes_no({ "a" => capped }).parameters["a"], [ 11.0, 91.0 ])
    end

    it "LO1 judges the raw counts when the weighted views are 0 or the weighted outcomes absent, and clamps c" do
      zero = { views: 800, outcomes: 96, weighted_views: 0.0, weighted_outcomes: 0.0 }
      half = { views: 800, outcomes: 96, weighted_views: 1_450.0 }
      over = { views: 100, outcomes: 10, weighted_views: 50.0, weighted_outcomes: 80.0 }

      pair_close(posteriors.yes_no({ "a" => zero }).parameters["a"], [ 97.0, 705.0 ])
      pair_close(posteriors.yes_no({ "a" => half }).parameters["a"], [ 97.0, 705.0 ])
      # c = 80 * 100 / 50 = 160, clamped to the 100 views
      pair_close(posteriors.yes_no({ "a" => over }).parameters["a"], [ 101.0, 1.0 ])
    end

    it "LO1 equals WeightedPosterior.beta on 500 random integer tallies, weighted and not (one rule)" do
      rng = Random.new(4_001)
      shapes = %i[plain weighted kish zero_weight no_weighted_outcomes]
      priors = [ [ 1.0, 1.0 ], [ 0.5, 0.5 ], [ 2.0, 3.0 ] ]

      500.times do |i|
        exposures = rng.rand(0..2_000)
        tally = { exposures: exposures, conversions: rng.rand(0..(exposures + 5)) }
        case shapes[i % shapes.size]
        when :weighted
          tally[:weighted_exposures] = rng.rand(1..((exposures * 8) + 1))
          tally[:weighted_conversions] = rng.rand(0..3_000)
        when :kish
          weighted_exposures = rng.rand(1..((exposures * 8) + 1))
          tally[:weighted_exposures] = weighted_exposures
          tally[:weighted_conversions] = rng.rand(0..3_000)
          tally[:weighted_exposures_squared] = rng.rand(weighted_exposures..(weighted_exposures * 40))
        when :zero_weight
          tally[:weighted_exposures] = 0
          tally[:weighted_conversions] = rng.rand(0..100)
        when :no_weighted_outcomes
          tally[:weighted_exposures] = rng.rand(1..6_000)
        end
        prior_alpha, prior_beta = priors[i % priors.size]
        expected = weighted_posterior.beta(weighted_posterior.count(tally), prior_alpha: prior_alpha,
                                                                           prior_beta: prior_beta)
        item = {
          views: tally[:exposures], outcomes: tally[:conversions], weighted_views: tally[:weighted_exposures],
          weighted_outcomes: tally[:weighted_conversions], weighted_views_squared: tally[:weighted_exposures_squared]
        }.compact

        actual = posteriors.yes_no({ "a" => item }, priors: [ prior_alpha, prior_beta ]).parameters["a"]
        expect(actual).to eq(expected), "tally #{tally.inspect}, prior #{[ prior_alpha, prior_beta ].inspect}"
      end
    end
  end

  describe ".amounts" do
    it "LO1 is Normal(mu_n, sd_n) for the mean per view under k views at mu0 and v0" do
      set = posteriors.amounts({ "a" => { views: 100, outcomes: 600, outcome_squares: 90_000 } }, priors: amount_prior)

      # m = 6, s^2 = 900 - 36 = 864: mu_n = (20 * 4 + 100 * 6) / 120,
      # sd_n^2 = [20 * 2 000 + 100 * 864 + 20 * 100 * (6 - 4)^2 / 120] / 120^2
      expect(set.kind).to eq(:normal)
      expect(set.items).to eq(%w[a])
      expect(set.views).to eq("a" => 100.0)
      pair_close(set.parameters["a"], [ 5.666666666666667, 2.9635126804870278 ])
    end

    it "LO1 at no views answers (mu0, sqrt(v0 / k))" do
      set = posteriors.amounts({ "z" => { views: 0, outcomes: 0, outcome_squares: 0 } }, priors: amount_prior)

      pair_close(set.parameters["z"], [ 4.0, 10.0 ])
    end

    it "LO1 reads the unweighted variance as outcome_squares / views - m^2, a residual below 0 as 0" do
      centred = prior.amount(mean: 6.0, variance: 2_000.0, strength: 20.0)
      set = posteriors.amounts({ "a" => { views: 100, outcomes: 600, outcome_squares: 90_000 } }, priors: centred)
      # mu0 = m, so sd_n^2 = (20 * 2 000 + 100 * 864) / 120^2; the n - 1 reading would give 2.9729...
      pair_close(set.parameters["a"], [ 6.0, 2.96273147243853 ])

      # m = 5, 90 / 4 - 25 < 0 reads 0: sd_n^2 = [40 000 + 0 + 20 * 4 * (5 - 4)^2 / 24] / 24^2
      set = posteriors.amounts({ "a" => { views: 4, outcomes: 20, outcome_squares: 90 } }, priors: amount_prior)
      pair_close(set.parameters["a"], [ 4.166666666666667, 8.33368054832206 ])
    end

    it "LO1 weighted: m and s^2 from the weighted sums, n the Kish size only with the squared sum" do
      tally = { views: 100, outcomes: 600, outcome_squares: 90_000, weighted_views: 250.0, weighted_outcomes: 1_600.0,
                weighted_outcome_squares: 230_000.0 }

      # m = 6.4, s^2 = 920 - 40.96 = 879.04; n = 250^2 / 900 = 69.4 with the squared sum, else the 100 views
      kish = posteriors.amounts({ "a" => tally.merge(weighted_views_squared: 900.0) }, priors: amount_prior)
      pair_close(kish.parameters["a"], [ 5.863354037267081, 3.5554532731159467 ])
      plain = posteriors.amounts({ "a" => tally }, priors: amount_prior)
      pair_close(plain.parameters["a"], [ 6.0, 2.9814239699997196 ])
    end

    it "LO1 takes one amount prior or one per item, and refuses flat or none" do
      tallies = { "a" => { views: 100, outcomes: 600, outcome_squares: 90_000 },
                  "b" => { views: 50, outcomes: 100, outcome_squares: 1_000 } }
      set = posteriors.amounts(tallies, priors: { "a" => amount_prior,
                                                  "b" => prior.amount(mean: 1.0, variance: 9.0, strength: 4.0) })

      pair_close(set.parameters["a"], [ 5.666666666666667, 2.9635126804870278 ])
      pair_close(set.parameters["b"], [ 1.9259259259259258, 0.5366229897186185 ])
      expect { posteriors.amounts(tallies, priors: prior.flat) }.to raise_error(ArgumentError)
      expect { posteriors.amounts(tallies, priors: nil) }.to raise_error(ArgumentError)
    end
  end

  describe ".beta and .normal" do
    it "LO1 take pairs as they are, with views when given" do
      beta = posteriors.beta({ "a" => [ 2.0, 3.0 ], "b" => [ 0.5, 9.5 ] }, views: { "a" => 4.0, "b" => 9.0 })
      normal = posteriors.normal({ "a" => [ 5.0, 2.0 ], "b" => [ 0.5, 0.25 ], "c" => [ 2_000_000.0, 3.0 ] })

      expect(beta.kind).to eq(:beta)
      expect(beta.items).to eq(%w[a b])
      expect(beta.parameters).to eq("a" => [ 2.0, 3.0 ], "b" => [ 0.5, 9.5 ])
      expect(beta.views).to eq("a" => 4.0, "b" => 9.0)
      expect(normal.kind).to eq(:normal)
      expect(normal.parameters).to eq("a" => [ 5.0, 2.0 ], "b" => [ 0.5, 0.25 ], "c" => [ 2_000_000.0, 3.0 ])
    end
  end

  describe "the item rules" do
    it "LO1 items keep their input order, the tie order" do
      tallies = { "z" => { views: 1, outcomes: 0 }, "a" => { views: 2, outcomes: 1 }, "m" => { views: 3, outcomes: 1 } }

      expect(posteriors.yes_no(tallies).items).to eq(%w[z a m])
      expect(posteriors.yes_no(tallies).parameters.keys).to eq(%w[z a m])
      expect(posteriors.beta({ "q" => [ 1.0, 1.0 ], "b" => [ 2.0, 2.0 ] }).items).to eq(%w[q b])
    end

    it "LO1 a Symbol, blank or repeated item is an ArgumentError, whichever builder" do
      tally = { views: 4, outcomes: 1 }
      amount_tally = { views: 4, outcomes: 8, outcome_squares: 20 }
      builders = {
        yes_no: ->(items) { posteriors.yes_no(items) },
        amounts: ->(items) { posteriors.amounts(items, priors: amount_prior) },
        beta: ->(items) { posteriors.beta(items) },
        normal: ->(items) { posteriors.normal(items) }
      }
      values = { yes_no: tally, amounts: amount_tally, beta: [ 2.0, 3.0 ], normal: [ 1.0, 0.5 ] }
      builders.each { |name, build| expect(build.call({ "a" => values.fetch(name) }).items).to eq(%w[a]) }

      builders.each do |name, build|
        value = values.fetch(name)
        expect { build.call({ a: value }) }.to raise_error(ArgumentError), "#{name}: a Symbol"
        expect { build.call({ "b" => value, a: value }) }.to raise_error(ArgumentError), "#{name}: a Symbol beside a String"
        expect { build.call({ "" => value }) }.to raise_error(ArgumentError), "#{name}: an empty name"
        expect { build.call({ "  " => value }) }.to raise_error(ArgumentError), "#{name}: a blank name"
        expect { build.call(repeated(value)) }.to raise_error(ArgumentError), "#{name}: a repeated name"
      end
    end

    # A keyed draw joins its key's parts and the item by NUL bytes (Draw.uniform_for), which refuses a part holding
    # one; so an item name holding a NUL byte is refused when the set is built, never mid-draw.
    it "LO1 (LB1) an item name holding a NUL byte is an ArgumentError when the set is built, before any draw" do
      tally = { views: 4, outcomes: 1 }
      amount_tally = { views: 4, outcomes: 8, outcome_squares: 20 }
      builders = {
        yes_no: ->(items) { posteriors.yes_no(items) },
        amounts: ->(items) { posteriors.amounts(items, priors: amount_prior) },
        beta: ->(items) { posteriors.beta(items) },
        normal: ->(items) { posteriors.normal(items) }
      }
      values = { yes_no: tally, amounts: amount_tally, beta: [ 2.0, 3.0 ], normal: [ 1.0, 0.5 ] }
      expect(SkiftetStatistical::Bandit::Draw).not_to receive(:uniform_for)
      expect(SkiftetStatistical::Bandit::Draw).not_to receive(:uniform)

      builders.each do |name, build|
        value = values.fetch(name)
        expect(build.call({ "a" => value, "a b" => value }).items).to eq([ "a", "a b" ])
        [ "a\u0000b", "a\u0000", "\u0000a", "a\x00b".b ].each do |held|
          expect { build.call({ held => value }) }.to raise_error(ArgumentError), "#{name}: #{held.inspect}"
          expect { build.call({ "a" => value, held => value }) }
            .to raise_error(ArgumentError), "#{name}: #{held.inspect} beside a good name"
        end
      end
    end

    # A keyed draw reads an item by its bytes (Draw.uniform_for), so two names holding the same bytes in different
    # encodings would draw the same uniforms; the second is refused when the set is built (the L2 barrier,
    # 2026-11-01). Names of one text in encodings with different bytes stay two items.
    it "LO1 (LB1) a name holding the same bytes as an earlier one in another encoding is an ArgumentError when built" do
      tally = { views: 4, outcomes: 1 }
      amount_tally = { views: 4, outcomes: 8, outcome_squares: 20 }
      builders = {
        yes_no: ->(items) { posteriors.yes_no(items) },
        amounts: ->(items) { posteriors.amounts(items, priors: amount_prior) },
        beta: ->(items) { posteriors.beta(items) },
        normal: ->(items) { posteriors.normal(items) }
      }
      values = { yes_no: tally, amounts: amount_tally, beta: [ 2.0, 3.0 ], normal: [ 1.0, 0.5 ] }
      text = "é"
      same_bytes = [ text.b, text.dup.force_encoding(Encoding::Windows_1252) ]
      other_bytes = text.encode(Encoding::ISO_8859_1)
      expect(SkiftetStatistical::Bandit::Draw).not_to receive(:uniform_for)
      expect(SkiftetStatistical::Bandit::Draw).not_to receive(:uniform)

      builders.each do |name, build|
        value = values.fetch(name)
        expect(build.call({ text => value, "e" => value }).items).to eq([ text, "e" ])
        expect(build.call({ text => value, other_bytes => value }).items.map(&:bytes))
          .to eq([ text.bytes, other_bytes.bytes ]), "#{name}: one text, two byte strings"
        same_bytes.each do |held|
          expect { build.call({ text => value, held => value }) }
            .to raise_error(ArgumentError, /same bytes/), "#{name}: #{held.encoding} after UTF-8"
          expect { build.call({ held => value, "e" => value, text => value }) }
            .to raise_error(ArgumentError, /same bytes/), "#{name}: UTF-8 after #{held.encoding}"
        end
        expect { build.call({ other_bytes => value, other_bytes.b => value }) }
          .to raise_error(ArgumentError, /same bytes/), "#{name}: binary after ISO-8859-1"
      end
    end

    it "LO1 a parameter not finite or not above 0 is an ArgumentError when the set is built" do
      good = [ 2.0, 3.0 ]
      expect(posteriors.beta({ "a" => good }).items).to eq(%w[a])
      [ [ 0.0, 1.0 ], [ 1.0, -2.0 ], [ Float::NAN, 1.0 ], [ 1.0, Float::INFINITY ] ].each do |bad|
        expect { posteriors.beta({ "a" => good, "b" => bad }) }.to raise_error(ArgumentError), "beta #{bad.inspect}"
      end
      [ [ 1.0, 0.0 ], [ 1.0, -0.5 ], [ 1.0, Float::NAN ], [ 1.0, Float::INFINITY ], [ Float::NAN, 1.0 ],
        [ Float::INFINITY, 1.0 ] ].each do |bad|
        expect { posteriors.normal({ "a" => [ 1.0, 0.5 ], "b" => bad }) }.to raise_error(ArgumentError),
                                                                            "normal #{bad.inspect}"
      end
      zero = { views: 0, outcomes: 0 }
      expect { posteriors.yes_no({ "a" => zero }, priors: [ 0.0, 1.0 ]) }.to raise_error(ArgumentError)
      expect { posteriors.yes_no({ "a" => zero }, priors: [ Float::NAN, 1.0 ]) }.to raise_error(ArgumentError)
      expect { posteriors.yes_no({ "a" => { views: Float::NAN, outcomes: 1 } }) }.to raise_error(ArgumentError)
      expect { posteriors.yes_no({ "a" => { views: Float::INFINITY, outcomes: 1 } }) }.to raise_error(ArgumentError)
      expect { posteriors.amounts({ "a" => { views: Float::NAN, outcomes: 1, outcome_squares: 1 } }, priors: amount_prior) }
        .to raise_error(ArgumentError)
    end

    it "LO1 a parameter above 0 but below the quantile draws' smallest still builds: the draws refuse it, not the set" do
      set = posteriors.beta({ "a" => [ 1e-9, 1.0 ], "b" => [ 2.0, 3.0 ] })

      expect(set.parameters["a"]).to eq([ 1e-9, 1.0 ])
    end

    it "LO1 60 items build a set (no cap), and so does none" do
      tallies = (0...60).to_h { |i| [ "item-#{i}", { views: i, outcomes: i / 3 } ] }
      set = posteriors.yes_no(tallies)

      expect(set.items).to eq(tallies.keys)
      expect(set.parameters.size).to eq(60)
      expect(set.parameters["item-59"]).to eq([ 20.0, 41.0 ])
      expect(posteriors.beta((0...60).to_h { |i| [ "item-#{i}", [ 1.0 + i, 2.0 ] ] }).items.size).to eq(60)

      [ posteriors.yes_no({}), posteriors.beta({}), posteriors.normal({}), posteriors.amounts({}, priors: amount_prior) ]
        .each do |empty|
        expect(empty.items).to eq([])
        expect(empty.parameters).to eq({})
        expect(empty.views).to eq({})
        expect(empty.values(rng: Random.new(1))).to eq([])
      end
    end
  end

  describe "the largest total" do
    it "LO1 a Beta pair above LARGEST_TOTAL is scaled to it in parameters with its mean kept; one at it is not" do
      set = posteriors.beta({ "big" => [ 300_000.0, 1_700_000.0 ], "edge" => [ 250_000.0, 750_000.0 ],
                              "small" => [ 3.0, 7.0 ] })
      big = set.parameters["big"]

      expect(big[0]).to be_within(150_000.0 * 1e-12).of(150_000.0)
      expect(big[1]).to be_within(850_000.0 * 1e-12).of(850_000.0)
      expect(big.sum).to be_within(1e-6).of(1_000_000.0)
      expect(big[0] / big.sum).to be_within(1e-15).of(0.15)
      expect(set.parameters["edge"]).to eq([ 250_000.0, 750_000.0 ])
      expect(set.parameters["small"]).to eq([ 3.0, 7.0 ])

      # flat + (600 000 of 3 000 000) = (600 001, 2 400 001), times 1e6 / 3 000 002
      tallied = posteriors.yes_no({ "a" => { views: 3_000_000, outcomes: 600_000 } }).parameters["a"]
      expect(tallied[0]).to be_within(200_000.0 * 1e-12).of(200_000.19999986666)
      expect(tallied[1]).to be_within(800_000.0 * 1e-12).of(799_999.8000001333)
      expect(tallied[0] / tallied.sum).to be_within(1e-15).of(600_001.0 / 3_000_002.0)
    end

    # 01 GB6 (the L2 barrier, 2026-11-01; restated at the L3 barrier's sixth round, 01 D33): every parameter lies in
    # the input domain (Numbers::LARGEST, 1e100), so a pair at that bound builds and scales to LARGEST_TOTAL with its
    # mean kept, and a pair whose total would overflow a Float is refused at the reader; values and the RNG draws read
    # the scaled pairs, one Sampler Beta each, as the keyed draws and the odds do, and greedy judges their means (01 D2
    # as amended at the L4 barrier's third round, 2026-11-02; D38). Re-pinned then: drawn on the pairs as given, this
    # read [0.5, 0.28966520931877265, 0.6666666666666666]; computed apart from the library, 0.3.0's Sampler on
    # Random.new(17) draws Beta(500 000, 500 000), Beta(6, 6) and Beta(666 666.6666666666, 333 333.3333333333) (1e100
    # and 5e99 times 5e5 / 7.5e99) as [0.49926845108823614, 0.28966520931877265, 0.6671093188663838], its next rand
    # 0.4835286243100021.
    it "LO1 (LB1) a Beta pair at the input domain's bound builds, scaled in parameters, drawn on them" do
      [ [ 1e308 + 3, 1e308 + 5 ], [ Float::MAX, Float::MAX / 2 ], [ 1.0, 1e100.next_float ] ].each do |beyond|
        expect { posteriors.beta({ "a" => beyond }) }.to raise_error(ArgumentError, /must lie within the input domain/)
      end
      pairs = { "0" => [ 1e100, 1e100 ], "1" => [ 6.0, 6.0 ], "2" => [ 1e100, 5e99 ] }
      set = posteriors.beta(pairs)
      rng = Random.new(17)
      reference = Random.new(17)
      sampler = SkiftetStatistical::Sampler.new(reference)

      expect(set.parameters["0"]).to eq([ 500_000.0, 500_000.0 ])
      expect(set.parameters["1"]).to eq([ 6.0, 6.0 ])
      alpha, beta = set.parameters["2"]
      expect(alpha + beta).to be <= 1_000_000.0
      expect(alpha + beta).to be_within(1e-6).of(1_000_000.0)
      expect(alpha / (alpha + beta)).to be_within(1e-15).of(2.0 / 3.0)
      scaled = [ [ 500_000.0, 500_000.0 ], [ 6.0, 6.0 ], [ 666_666.6666666666, 333_333.3333333333 ] ]
      expect(set.values(rng: rng)).to eq(scaled.map { |first, second| sampler.beta(first, second) })
      expect(rng.rand).to eq(reference.rand)
      expect(set.greedy).to eq("2")
      expect(set.greedy(may_lead: %w[0 1])).to eq("0")
    end

    # 01 GB6 (the L2 barrier, 2026-11-01): a scaled pair never sums above LARGEST_TOTAL, so a set built from another
    # set's parameters keeps them bit for bit. Scaled by its products alone, [14 840 026, 9 386 062] sums one rounding
    # above 1e6, as many pairs above the limit do (the fixture checks it holds over a thousand); the larger then reads
    # LARGEST_TOTAL less the smaller, a sum of exactly 1e6. A parameter the scale rounds to 0 reads the smallest Float above 0.
    it "LO1 (LB1) a set built from another set's parameters keeps them bit for bit, none summing above LARGEST_TOTAL" do
      rng = Random.new(8_675)
      pairs = (0...20_000).to_h do |i|
        total = 10.0**rng.rand(6.0..(i.even? ? 8.0 : 100.0))
        share = rng.rand(0.001..0.999)
        [ "p#{i}", [ total * share, total * (1.0 - share) ] ]
      end
      pairs["over"] = [ 14_840_026.0, 9_386_062.0 ]
      pairs["tiny"] = [ 1e-300, 1e100 ]
      pairs["bound"] = [ 1e100, 1e100 ]
      by_products = pairs.count do |_, (first, second)|
        total = first + second
        total.finite? && total > 1e6 && (first * (1e6 / total)) + (second * (1e6 / total)) > 1e6
      end
      set = posteriors.beta(pairs)
      rebuilt = posteriors.beta(set.parameters)

      expect(by_products).to be > 1_000
      bits = ->(pair) { pair.pack("G*") }
      expect(set.parameters.reject { |item, pair| bits.call(rebuilt.parameters[item]) == bits.call(pair) }.keys).to eq([])
      expect(set.parameters.reject { |_, (alpha, beta)| alpha.positive? && beta.positive? && alpha + beta <= 1e6 }.keys)
        .to eq([])
      over = set.parameters["over"]
      expect(over[0] + over[1]).to eq(1_000_000.0)
      expect(over[0] / 1e6).to be_within(1e-15).of(14_840_026.0 / 24_226_088.0)
      expect(set.parameters["tiny"][0]).to eq(0.0.next_float)
      expect(set.parameters["tiny"][1]).to be_within(1e-9).of(1_000_000.0)
      expect(set.parameters["bound"]).to eq([ 500_000.0, 500_000.0 ])
    end

    # Re-pinned at the L4 barrier's third round (2026-11-02; 01 D2 as amended): values draws on the scaled pairs, big's
    # [150 000, 850 000] (half of [300 000, 1 700 000]); on the pair as given it read [0.1504967908954545,
    # 0.25491432517311646]; computed apart from the library, 0.3.0's Sampler on Random.new(11) draws Beta(150 000,
    # 850 000) and Beta(3, 7) as [0.1507027596528838, 0.25491432517311646], its next rand 0.6323340138234961.
    it "LO1 values draws one Sampler Beta per item, in order, on the scaled pairs, and nothing else from the stream" do
      set = posteriors.beta({ "big" => [ 300_000.0, 1_700_000.0 ], "small" => [ 3.0, 7.0 ] })
      rng = Random.new(11)
      reference = Random.new(11)
      sampler = SkiftetStatistical::Sampler.new(reference)

      expect(set.values(rng: rng)).to eq([ sampler.beta(150_000.0, 850_000.0), sampler.beta(3.0, 7.0) ])
      expect(rng.rand).to eq(reference.rand)
    end

    # 01 GB6: values draws by Sampler and takes any parameter above 0; only the quantile draws and the exact odds stop
    # at BetaArgmax::SMALLEST_PARAMETER. At 1e-300 a Gamma draw underflows to 0, so a, b and c draw exactly 0.5, 1.0
    # and 0.0 (the old policy's ties run at a prior of 1e-300).
    it "LO1 (LB1) values draws parameters below BetaArgmax::SMALLEST_PARAMETER from an RNG, 1e-300 included" do
      pairs = { "a" => [ 1e-300, 1e-300 ], "b" => [ 3.0, 1e-300 ], "c" => [ 1e-300, 5.0 ], "d" => [ 1e-9, 1.0 ] }
      set = posteriors.beta(pairs)
      rng = Random.new(13)
      reference = Random.new(13)
      sampler = SkiftetStatistical::Sampler.new(reference)

      values = set.values(rng: rng)
      expect(values).to eq(pairs.values.map { |alpha, beta| sampler.beta(alpha, beta) })
      expect(values.first(3)).to eq([ 0.5, 1.0, 0.0 ])
      expect(rng.rand).to eq(reference.rand)
    end

    it "LO1 values on Normals is mean + sd * one Sampler gaussian per item, in order" do
      set = posteriors.normal({ "a" => [ 5.0, 2.0 ], "b" => [ 0.5, 0.25 ] })
      rng = Random.new(5)
      reference = Random.new(5)
      sampler = SkiftetStatistical::Sampler.new(reference)
      expected = [ 5.0 + (2.0 * sampler.gaussian), 0.5 + (0.25 * sampler.gaussian) ]

      values = set.values(rng: rng)
      expect(values.size).to eq(2)
      expect(values[0]).to be_within(1e-12).of(expected[0])
      expect(values[1]).to be_within(1e-12).of(expected[1])
      expect(rng.rand).to eq(reference.rand)
    end

    it "LO1 (green at LC2) summary's quantiles read a scaled pair as given" do
      set = posteriors.beta({ "big" => [ 300_000.0, 1_700_000.0 ], "small" => [ 3.0, 7.0 ] })
      row = set.summary(credible_mass: 0.95)["big"]
      quantile = SkiftetStatistical::BetaFunction.method(:quantile)

      # as given these read 0.149505... and 0.150495...; on the scaled pair the lower one would read 0.149300...
      expect(row[:lower]).to be_within(1e-10).of(quantile.call(0.025, 300_000.0, 1_700_000.0))
      expect(row[:upper]).to be_within(1e-10).of(quantile.call(0.975, 300_000.0, 1_700_000.0))
      expect(row[:mean]).to be_within(1e-15).of(0.15)
    end
  end

  describe "#greedy" do
    it "LO1 is the contender with the largest mean, ties to the earlier item" do
      set = posteriors.beta({ "a" => [ 2.0, 8.0 ], "b" => [ 3.0, 3.0 ], "c" => [ 6.0, 6.0 ], "d" => [ 1.0, 9.0 ] })

      expect(set.greedy).to eq("b")
      expect(set.greedy(may_lead: %w[a c d])).to eq("c")
      expect(set.greedy(may_lead: %w[a d])).to eq("a")
      expect(posteriors.normal({ "a" => [ 1.0, 5.0 ], "b" => [ 2.0, 0.1 ] }).greedy).to eq("b")
    end
  end

  describe "the composed priors" do
    it "LO1 a gated row's priors: pooled for the measured items, pooled_new for a new one, then the row's own evidence" do
      groups = { "x" => { views: 900, outcomes: 90 }, "y" => { views: 15, outcomes: 3 }, "z" => { views: 8, outcomes: 2 } }
      own = { "x" => { views: 60, outcomes: 9 }, "y" => { views: 5, outcomes: 1 }, "z" => { views: 3, outcomes: 1 } }
      priors = {
        "x" => prior.pooled(group: groups["x"], own: own["x"], strength: 20.0),
        "y" => prior.pooled(group: groups["y"], own: own["y"], strength: 20.0),
        "z" => prior.pooled_new(others: %w[x y].map { |item| { group: groups[item], own: own[item] } },
                                group: groups["z"], own: own["z"], weight: 2.0)
      }
      set = posteriors.yes_no(own, priors: priors)

      # x: n = 840 above the strength, mean 82 / 842 at 22; y: n = 10 below it, (1 + 2, 1 + 8);
      # z: the others' (850, 83) pooled to 84 / 852 at weight 2, plus z's own borrowed (5, 1)
      pair_close(set.parameters["x"], [ 11.14251781472684, 70.85748218527316 ])
      pair_close(set.parameters["y"], [ 4.0, 13.0 ])
      pair_close(set.parameters["z"], [ 2.197183098591549, 7.802816901408452 ])
    end

    it "LO1 a two-layer prior: decayed sums, nested levels, tilts and the chain, then the leaf's own evidence" do
      rows = ->(list) { list.map { |age, views, outcomes| { age: age, views: views, outcomes: outcomes } } }
      fold = ->(list) { decay.fold(rows.call(list), half_life: 72, cap: 336) }
      a_leaf = [ [ 2, 30, 4.5 ], [ 80, 10, 2.0 ], [ 400, 500, 400 ] ]
      a_middle = [ [ 5, 120, 15 ], [ 150, 40, 6 ] ]
      a_top = [ [ 1, 600, 50 ], [ 300, 200, 30 ] ]
      b_middle = [ [ 3, 50, 9 ], [ 40, 25, 2.5 ] ]
      b_top = [ [ 12, 400, 30 ], [ 200, 100, 20 ], [ 340, 900, 800 ] ]
      c_top = [ [ 0, 80, 6 ], [ 72, 40, 4 ] ]
      level = ->(sums, strength) { { views: sums[:views], outcomes: sums[:outcomes], strength: strength } }
      evidence = ->(sums) { { views: sums[:views], outcomes: sums[:outcomes] } }

      every = fold.call(a_leaf + a_middle + a_top + b_middle + b_top + c_top)
      root_mean = prior.pooled_mean(views: every[:views], outcomes: every[:outcomes], prior_mean: 0.05, strength: 50.0)
      tilts = prior.tilts({ "a" => 1.6, "b" => 0.9, "c" => 0.5 }, clip: [ 0.5, 2.0 ])
      a_levels = [ level.call(fold.call(a_leaf + a_middle + a_top), 20.0), level.call(fold.call(a_leaf + a_middle), 30.0),
                   level.call(fold.call(a_leaf), 30.0) ]
      b_levels = [ level.call(fold.call(b_middle + b_top), 20.0), level.call(fold.call(b_middle), 30.0) ]
      c_levels = [ level.call(fold.call(c_top), 20.0) ]
      priors = {
        "a" => prior.chain(levels: a_levels, root_mean: root_mean, tilt: tilts["a"]),
        "b" => prior.chain(levels: b_levels, root_mean: root_mean, tilt: tilts["b"]),
        "c" => prior.chain(levels: c_levels, root_mean: root_mean, tilt: tilts["c"])
      }
      set = posteriors.yes_no({ "a" => evidence.call(fold.call(a_leaf)), "b" => evidence.call(fold.call(b_middle)),
                                "c" => evidence.call(fold.call(c_top)) }, priors: priors)

      # root mean 0.0910008546773131; tilts 1.78515..., 1.00414..., 0.55786...; each item's prior from its deepest
      # level, plus that level's own decayed sums (the 400-hour leaf row and the 340-hour top row weigh 0)
      expect(set.items).to eq(%w[a b c])
      pair_close(set.parameters["a"], [ 11.696255273175705, 52.361020918328194 ], 1e-10)
      pair_close(set.parameters["b"], [ 12.869221197592477, 82.71725086226753 ], 1e-10)
      pair_close(set.parameters["c"], [ 9.015316176777208, 110.9846838232228 ], 1e-10)
    end
  end
end
