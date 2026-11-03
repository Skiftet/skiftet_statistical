# frozen_string_literal: true

# L7 (the W3 barrier, 2026-11-03): the means-only reader. `Posteriors#means` answers { item => mean }, every item in
# item order, frozen: the mean `summary` answers as :mean, read by summary's own reader on the pair as given, so the two
# agree bit for bit; it works out no interval, so it calls no quantile, and it draws nothing. The reference is summary
# itself (`summary(among: [])`, which works out the same mean beside its interval), compared by the Floats' bytes, over
# Beta and Normal sets of 1 to 20 items whose pairs run from the smallest the interval reads
# (BetaArgmax::SMALLEST_PARAMETER, 1e-6, for a Beta; the smallest Float above 0 for a Normal's standard deviation) to
# the largest the set takes (Numbers::LARGEST, 1e100, a parameter, so a Beta total of 2e100).
RSpec.describe "SkiftetStatistical::Bandit::Posteriors#means" do
  def posteriors = SkiftetStatistical::Bandit::Posteriors
  def smallest = SkiftetStatistical::BetaArgmax::SMALLEST_PARAMETER
  def largest = SkiftetStatistical::Numbers::LARGEST

  # The magnitudes a parameter is drawn around, from the smallest the interval reads to the domain's bound.
  def beta_magnitudes = [ smallest, 1e-3, 0.5, 1.0, 10.0, 1e3, 1e6, 1e9, 1e20, 1e50, 1e99, largest ]
  def normal_magnitudes = [ 5e-324, 1e-300, 1e-6, 0.5, 1.0, 1e3, 1e9, 1e50, largest ]

  def beta_parameter(rng)
    (beta_magnitudes[rng.rand(beta_magnitudes.size)] * (0.5 + rng.rand)).clamp(smallest, largest)
  end

  def normal_size(rng)
    [ normal_magnitudes[rng.rand(normal_magnitudes.size)] * (0.5 + rng.rand), largest ].min
  end

  def normal_mean(rng)
    return [ 0.0, -0.0 ][rng.rand(2)] if rng.rand < 0.05

    (rng.rand < 0.4 ? -1.0 : 1.0) * normal_size(rng)
  end

  # Each item's pair: a drawn one, or, at the first items, the ends of the range.
  def beta_pairs(k, rng)
    edges = [ [ smallest, smallest ], [ largest, largest ], [ smallest, largest ], [ largest, smallest ],
              [ 1_500_000.0, 2_500_000.0 ], [ 1.0, 2e12 ] ]
    (0...k).to_h do |i|
      [ "item-#{i}", rng.rand < 0.25 ? edges[rng.rand(edges.size)] : [ beta_parameter(rng), beta_parameter(rng) ] ]
    end
  end

  def normal_pairs(k, rng)
    edges = [ [ largest, largest ], [ -largest, 5e-324 ], [ -0.0, 1.0 ], [ 0.0, largest ], [ 1e-300, 1e-300 ] ]
    (0...k).to_h do |i|
      [ "item-#{i}", rng.rand < 0.25 ? edges[rng.rand(edges.size)] : [ normal_mean(rng), normal_size(rng) ] ]
    end
  end

  def bits(number) = [ number ].pack("G")

  def expect_bit_for_bit(set, label)
    means = set.means
    summary = set.summary(among: [])
    expect(means.keys).to eq(set.items), label
    expect(means.keys).to eq(summary.keys), label
    means.each do |item, mean|
      expect(mean).to be_a(Float), "#{label}, #{item}"
      expect(bits(mean)).to eq(bits(summary.fetch(item)[:mean])),
                            "#{label}, #{item}: #{mean} is not #{summary[item][:mean]}"
    end
  end

  it "L7 answers summary(among: [])'s mean bit for bit over Beta and Normal sets of 1 to 20 items, tiny pairs to the " \
     "largest total" do
    rng = Random.new(20_261_103)
    (1..20).each do |k|
      3.times do |round|
        beta = posteriors.beta(beta_pairs(k, rng), views: (0...k).to_h { |i| [ "item-#{i}", 10.0 * i ] })
        expect_bit_for_bit(beta, "Beta, #{k} items, round #{round}")
        expect_bit_for_bit(posteriors.normal(normal_pairs(k, rng)), "Normal, #{k} items, round #{round}")
      end
    end

    ends = { "tiny" => [ smallest, smallest ], "largest" => [ largest, largest ],
             "tiny over largest" => [ smallest, largest ], "largest over tiny" => [ largest, smallest ] }
    expect_bit_for_bit(posteriors.beta(ends), "the Beta ends")
    expect(posteriors.beta(ends).means).to eq({ "tiny" => 0.5, "largest" => 0.5,
                                                "tiny over largest" => smallest / (smallest + largest),
                                                "largest over tiny" => largest / (largest + smallest) })
    normal_ends = { "largest" => [ largest, largest ], "most negative" => [ -largest, 5e-324 ],
                    "negative zero" => [ -0.0, 1.0 ] }
    expect_bit_for_bit(posteriors.normal(normal_ends), "the Normal ends")
    expect(bits(posteriors.normal(normal_ends).means["negative zero"])).to eq(bits(-0.0))
  end

  it "L7 answers the same mean as every summary, among: nil and a named list included" do
    pairs = { "a" => [ 2.0, 8.0 ], "b" => [ 3.0, 7.0 ], "c" => [ 5.0, 5.0 ] }
    set = posteriors.beta(pairs)
    [ nil, %w[a], %w[b c] ].each do |among|
      summary = set.summary(among: among)
      set.means.each { |item, mean| expect(bits(mean)).to eq(bits(summary[item][:mean])), "#{among.inspect}, #{item}" }
    end
    normal = posteriors.normal({ "a" => [ 5.0, 0.4 ], "b" => [ 5.3, 0.5 ] })
    expect(normal.means).to eq(normal.summary.transform_values { |entry| entry[:mean] })
  end

  it "L7 reads the pair as given above LARGEST_TOTAL, as summary does, never the scaled pair" do
    given = { "a" => [ 1_500_000.0, 2_500_000.0 ], "b" => [ 1.0, 2e12 ], "c" => [ 1e99, 3e99 ] }
    set = posteriors.beta(given)
    expect(set.parameters["b"]).not_to eq(given["b"])
    scaled_alpha, scaled_beta = set.parameters["c"]
    expect(scaled_alpha / (scaled_alpha + scaled_beta)).not_to eq(0.25), "c's scaled mean is a rounding off its own"
    expect(set.means["c"]).to eq(0.25)
    given.each do |item, (alpha, beta)|
      expect(bits(set.means[item])).to eq(bits(alpha / (alpha + beta))), item
    end
  end

  it "L7 answers a frozen Hash of Floats keyed by the set's own items, in item order, and {} for no items" do
    set = posteriors.beta({ "b" => [ 3.0, 7.0 ], "a" => [ 2.0, 8.0 ] })
    means = set.means
    expect(means).to eq({ "b" => 0.3, "a" => 0.2 })
    expect(means.keys).to eq(%w[b a])
    expect(means).to be_frozen
    means.keys.zip(set.items).each { |key, item| expect(key).to equal(item) }
    expect { means["c"] = 0.5 }.to raise_error(FrozenError)
    expect(set.means).to eq(means)
    expect(posteriors.beta({}).means).to eq({})
    expect(posteriors.beta({}).means).to be_frozen
    expect(posteriors.normal({}).means).to eq({})
    amounts = posteriors.amounts({ "x" => { views: 10, outcomes: 30.0, outcome_squares: 120.0 } },
                                 priors: SkiftetStatistical::Bandit::Prior.amount(mean: 2.0, variance: 4.0,
                                                                                  strength: 5.0))
    expect(amounts.means).to eq({ "x" => amounts.parameters["x"].first })
    yes_no = posteriors.yes_no({ "x" => { views: 40, outcomes: 8 } })
    expect(yes_no.means).to eq({ "x" => 9.0 / 42.0 })
  end

  # summary refuses a Beta parameter below SMALLEST_PARAMETER for its interval on every path, the means-only one
  # included (01 GB6, the L3 barrier); that refusal is the interval's, and means works out no interval, so it reads the
  # pair's mean. The mean itself has no refusal beyond the one check of an answer, which every set as built passes.
  it "L7 refuses only what summary refuses for the mean: a pair below SMALLEST_PARAMETER is read, its mean its own" do
    set = posteriors.beta({ "a" => [ 2.0, 3.0 ], "b" => [ 1e-300, 1.0 ], "c" => [ 1.0, 5e-7 ] })
    expect { set.summary(among: []) }.to raise_error(ArgumentError, /the interval/)
    expect(set.means).to eq({ "a" => 0.4, "b" => 1e-300, "c" => 1.0 / (1.0 + 5e-7) })
    expect(posteriors.beta({ "a" => [ 5e-324, 1e100 ] }).means).to eq({ "a" => 0.0 })
    expect { set.means(among: []) }.to raise_error(ArgumentError)
  end

  # The cost: means works out no interval, so it calls no quantile at all, where summary(among: []) calls
  # BetaFunction.quantile twice an item; it draws nothing either. Its CPU time, measured on 60 Betas (the means-only
  # summary's own example), lies far below the means-only summary's.
  it "L7 timing: costs no quantile (BetaFunction.quantile calls: 0) and no draw, far below summary(among: [])" do
    pairs = (0...60).to_h do |i|
      a = 0.5 + (0.5 * (i / 2))
      [ "item-#{i}", i.even? ? [ a, 1.0 ] : [ 1.0, a ] ]
    end
    beta = posteriors.beta(pairs)
    normal = posteriors.normal((0...60).to_h { |i| [ "item-#{i}", [ 1.0 + (0.1 * i), 0.5 + (0.01 * i) ] ] })

    cpu = -> { Process.clock_gettime(Process::CLOCK_PROCESS_CPUTIME_ID) }
    timed = lambda do |runs, &call|
      Array.new(runs) do
        start = cpu.call
        call.call
        cpu.call - start
      end.min
    end
    5.times { beta.means }
    means_time = timed.call(5) { beta.means }
    summary_time = timed.call(2) { beta.summary(among: []) }
    expect(means_time).to be < summary_time / 10.0, "means #{means_time * 1e3} ms, summary(among: []) " \
                                                    "#{summary_time * 1e3} ms"

    library = SkiftetStatistical
    spied = [ [ library::BetaFunction, :quantile ], [ library::Significance, :normal_quantile ],
              [ library::BetaArgmax, :position ], [ library::BetaArgmax, :probabilities ],
              [ library::BetaArgmax, :expected_maximum ], [ library::Bandit::NormalArgmax, :probabilities ],
              [ library::Bandit::NormalArgmax, :expected_maximum ], [ library::Bandit::Draw, :uniform ],
              [ library::Bandit::Draw, :uniform_for ], [ library::Bandit::Draw, :open_uniform ],
              [ library::Sampler, :new ], [ Random, :new ], [ Random, :rand ], [ Random, :new_seed ] ]
    spied.each { |owner, name| allow(owner).to receive(name).and_call_original }

    expect(beta.means.size).to eq(60)
    expect(normal.means.size).to eq(60)
    spied.each { |owner, name| expect(owner).not_to have_received(name), "#{owner}.#{name}" }

    beta.summary(among: [])
    expect(library::BetaFunction).to have_received(:quantile).exactly(120).times
  end
end
