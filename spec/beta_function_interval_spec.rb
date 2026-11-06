# frozen_string_literal: true

# BetaFunction.shortest_interval and BetaFunction.mean_intervals: the shortest interval holding a given mass of a Beta
# (the highest-density interval wherever the density has one peak or falls or rises monotonically, so it holds the
# mode), and the intervals of means of non-negative values read as scaled shares of their visits. Under the flat prior
# the mode of Beta(1 + c, 1 + n - c) is the observed share c / n, so a share always lies inside its interval; the
# equal-tailed interval leaves 0 out at c = 0 (and 1 out at c = n), which is what these methods exist to avoid. Every
# expected number is a closed form, the module's own distribution function or quantile, or a brute-force search over
# the lower tail's probability that shares no code with the method.
RSpec.describe "SkiftetStatistical::BetaFunction intervals" do
  def beta_function = SkiftetStatistical::BetaFunction

  def interval(mass, alpha, beta) = beta_function.shortest_interval(mass, alpha, beta)

  def mass_of(lower, upper, alpha, beta)
    beta_function.regularized(upper, alpha, beta) - beta_function.regularized(lower, alpha, beta)
  end

  def equal_tailed(mass, alpha, beta)
    tail = (1.0 - mass) / 2.0
    [ beta_function.quantile(tail, alpha, beta), beta_function.quantile(1.0 - tail, alpha, beta) ]
  end

  # The shortest width over a grid of lower-tail probabilities p in [0, 1 - mass]: [Q(p), Q(p + mass)].
  def brute_force_width(mass, alpha, beta, steps: 400)
    (0..steps).map do |i|
      p = (1.0 - mass) * i / steps
      beta_function.quantile([ p + mass, 1.0 ].min, alpha, beta) - beta_function.quantile(p, alpha, beta)
    end.min
  end

  # A deterministic spread of parameters at or above 1: whole and fractional, just above 1, skewed both ways, large.
  def peaked
    [ [ 2.0, 2.0 ], [ 1.5, 7.25 ], [ 7.25, 1.5 ], [ 1.02, 400.0 ], [ 400.0, 1.02 ], [ 1.0001, 3.0 ], [ 3.0, 1.0001 ],
      [ 41.0, 261.0 ], [ 3.0, 341.0 ], [ 2.0, 10_000.0 ], [ 12.5, 87.5 ], [ 345.678, 12.345 ], [ 1.07, 2.3 ],
      [ 40_001.0, 1_160_001.0 ], [ 500.0, 500.0 ] ]
  end

  def masses = [ 0.95, 0.9, 0.99, 0.5, 0.2 ]

  describe ".shortest_interval" do
    it "has the closed forms of a density that falls from 0 or rises to 1: Beta(1, b) and Beta(a, 1)" do
      [ 2.0, 3.0, 17.0, 301.0 ].product(masses).each do |b, mass|
        lower, upper = interval(mass, 1.0, b)
        expect(lower).to eq(0.0)
        expect(upper).to be_within(1e-12).of(1.0 - ((1.0 - mass)**(1.0 / b))), [ b, mass ].inspect
        lower, upper = interval(mass, b, 1.0)
        expect(upper).to eq(1.0)
        expect(lower).to be_within(1e-12).of((1.0 - mass)**(1.0 / b)), [ b, mass ].inspect
      end
    end

    it "is 0 to 1 - sqrt(0.05) for one visit and no event at 95 %, where the equal-tailed interval leaves 0 out" do
      expect(interval(0.95, 1.0, 2.0).first).to eq(0.0)
      expect(interval(0.95, 1.0, 2.0).last).to be_within(1e-12).of(1.0 - Math.sqrt(0.05))
      expect(equal_tailed(0.95, 1.0, 2.0).first).to be > 0.01
    end

    it "is the central interval of a flat density, and the equal-tailed one of a symmetric peak" do
      masses.each do |mass|
        expect(interval(mass, 1.0, 1.0)).to eq([ (1.0 - mass) / 2.0, (1.0 + mass) / 2.0 ])
        [ 2.0, 9.5, 500.0 ].each do |a|
          expect(interval(mass, a, a).zip(equal_tailed(mass, a, a)).map { |x, y| (x - y).abs }.max).to be < 1e-9
        end
      end
    end

    it "holds the mode, holds the mass, has one density at both ends and is no wider than any interval of the mass" do
      peaked.product(masses).each do |(alpha, beta), mass|
        lower, upper = interval(mass, alpha, beta)
        mode = (alpha - 1.0) / (alpha + beta - 2.0)
        label = [ alpha, beta, mass ].inspect

        expect(lower).to be <= mode, label
        expect(upper).to be >= mode, label
        expect(mass_of(lower, upper, alpha, beta)).to be_between(mass - 1e-12, mass + 1e-9), label
        if lower.positive? && upper < 1.0
          expect(beta_function.log_density(lower, alpha, beta))
            .to be_within(1e-6).of(beta_function.log_density(upper, alpha, beta)), label
        end
        tails = equal_tailed(mass, alpha, beta)
        expect(upper - lower).to be <= (tails.last - tails.first) + 1e-12, label
        next if alpha + beta > 10_000 # the grid is too coarse to see the peak of the largest pairs

        expect(upper - lower).to be <= brute_force_width(mass, alpha, beta) + 1e-9, label
      end
    end

    it "holds the observed share of every count under the flat prior, 0 and n included, where equal tails miss" do
      missed = []
      (1..60).each do |visits|
        (0..visits).each do |events|
          share = events.fdiv(visits)
          lower, upper = interval(0.95, 1.0 + events, 1.0 + visits - events)
          expect(lower).to be <= share, [ visits, events ].inspect
          expect(upper).to be >= share, [ visits, events ].inspect
          tails = equal_tailed(0.95, 1.0 + events, 1.0 + visits - events)
          missed << [ visits, events ] unless tails.first <= share && share <= tails.last
        end
      end
      # The equal-tailed interval leaves the share out exactly at no event and at every visit an event.
      expect(missed).to eq((1..60).flat_map { |visits| [ [ visits, 0 ], [ visits, visits ] ] })
    end

    it "holds the mode of fractional (weighted) counts just above an empty count, which equal tails leave out" do
      [ 0.005, 0.01, 0.02, 0.05, 0.3 ].product([ 50.0, 400.0, 5_000.0 ]).each do |events, visits|
        alpha = 1.0 + events
        beta = 1.0 + visits - events
        mode = events / visits
        lower, upper = interval(0.95, alpha, beta)
        expect(lower).to be <= mode, [ events, visits ].inspect
        expect(upper).to be >= mode, [ events, visits ].inspect
      end
      expect(equal_tailed(0.95, 1.01, 400.0).first).to be > 0.01 / 400.0
    end

    it "keeps to the boundary of a density that falls or rises with a fractional parameter below 1" do
      lower, upper = interval(0.9, 0.5, 3.0)
      expect([ lower, upper ]).to eq([ 0.0, beta_function.quantile(0.9, 0.5, 3.0) ])
      lower, upper = interval(0.9, 3.0, 0.5)
      expect([ lower, upper ]).to eq([ beta_function.quantile(0.1, 3.0, 0.5), 1.0 ])
    end

    it "takes the shorter one-sided interval of a U-shaped density, the lower one on a tie" do
      expect(interval(0.9, 0.5, 0.5)).to eq([ 0.0, beta_function.quantile(0.9, 0.5, 0.5) ])
      lower, upper = interval(0.9, 0.3, 0.8)
      from_zero = beta_function.quantile(0.9, 0.3, 0.8)
      to_one = 1.0 - beta_function.quantile(0.1, 0.3, 0.8)
      expect(upper - lower).to be_within(1e-15).of([ from_zero, to_one ].min)
      expect(lower.zero? || upper == 1.0).to be(true)
    end

    it "answers the same every time: no draw" do
      expect(interval(0.95, 41.0, 261.0)).to eq(interval(0.95, 41.0, 261.0))
    end

    it "refuses a mass not strictly between 0 and 1 and parameters not above 0, with an ArgumentError" do
      [ [ 0.0, 2.0, 2.0 ], [ 1.0, 2.0, 2.0 ], [ 1.5, 2.0, 2.0 ], [ -0.1, 2.0, 2.0 ], [ Float::NAN, 2.0, 2.0 ],
        [ nil, 2.0, 2.0 ], [ "0.95", 2.0, 2.0 ], [ 0.95, 0.0, 2.0 ], [ 0.95, 2.0, -1.0 ], [ 0.95, Float::NAN, 2.0 ],
        [ 0.95, 2.0, Float::INFINITY ] ].each do |mass, alpha, beta|
        expect { interval(mass, alpha, beta) }.to raise_error(ArgumentError), [ mass, alpha, beta ].inspect
      end
    end
  end

  describe ".mean_intervals" do
    def intervals(items, mass: 0.95) = beta_function.mean_intervals(items, mass: mass)

    # A mean of n values with its plain sandwich standard error (the population spread over n, as a Hájek mean of
    # equal weights gives it).
    def item_of(values)
      n = values.size.to_f
      mean = values.sum / n
      { visits: values.size, mean: mean, standard_error: Math.sqrt(values.sum { |v| (v - mean)**2 } / n) / Math.sqrt(n) }
    end

    it "is the share's own interval for a value of 0 or 1 per visit: Beta(1 + c, 1 + n - c)'s shortest interval" do
      [ [ 1, 1 ], [ 3, 2 ], [ 300, 40 ], [ 293, 25 ], [ 342, 2 ], [ 12, 12 ], [ 5_000, 37 ] ].each do |visits, events|
        values = Array.new(events, 1.0) + Array.new(visits - events, 0.0)
        answer = intervals({ "x" => item_of(values) }).fetch("x")
        expected = interval(0.95, 1.0 + events, 1.0 + visits - events)
        expect(answer.zip(expected).map { |a, e| (a - e).abs }.max).to be < 1e-9, [ visits, events ].inspect
      end
    end

    it "holds every item's mean, never goes below 0, and gives an item of no event an interval from 0" do
      rng = Random.new(20_261_105)
      200.times do |round|
        items = (1..4).to_h do |index|
          visits = [ 1, 2, 3, 5, 30, 301, 2_000 ].sample(random: rng)
          values = Array.new(visits) do
            rng.rand < [ 0.0, 0.01, 0.2, 0.6 ].sample(random: rng) ? [ 0.4, 1.0, 1.0, 2.6, 3.0 ].sample(random: rng) : 0.0
          end
          [ "v#{index}", item_of(values) ]
        end
        any_event = items.values.any? { |item| item[:mean].positive? }
        intervals(items).each do |key, bounds|
          item = items.fetch(key)
          label = [ round, key, item ].inspect
          if item[:mean].zero? && !any_event
            expect(bounds).to be_nil, label
            next
          end
          lower, upper = bounds
          expect(lower).to be >= 0.0, label
          expect(lower).to be <= item[:mean], label
          expect(upper).to be >= item[:mean], label
          expect(upper).to be > lower, label if item[:mean].zero?
        end
      end
    end

    it "gives a mean of 0 the set's value of one event: 0 to it times quantile(mass) of Beta(1, 1 + n)" do
      seen = item_of([ 2.0, 1.0 ] + Array.new(340, 0.0))
      unseen = { visits: 300, mean: 0.0, standard_error: 0.0 }
      second = (seen[:visits] * seen[:standard_error]**2) + (seen[:mean]**2)
      unit = (seen[:visits] * second) / (seen[:visits] * seen[:mean]) # summed squares over summed values: 5 / 3
      expect(unit).to be_within(1e-12).of(5.0 / 3.0)

      lower, upper = intervals({ seen: seen, unseen: unseen }).fetch(:unseen)
      expect(lower).to eq(0.0)
      expect(upper).to be_within(1e-12).of(unit * (1.0 - (0.05**(1.0 / 301.0))))
      # More visits of no event, a narrower interval.
      wider = intervals({ seen: seen, unseen: unseen.merge(visits: 30) }).fetch(:unseen)
      expect(wider.last).to be > upper
    end

    it "has no interval for any mean of 0 while no item of the set has an event" do
      expect(intervals({ "a" => { visits: 300, mean: 0.0, standard_error: 0.0 },
                         "b" => { visits: 1, mean: 0.0, standard_error: 0.0 } })).to eq("a" => nil, "b" => nil)
    end

    it "matches the normal interval in the large-sample limit and is skewed above the mean at a handful of events" do
      values = Array.new(400, 1.0) + Array.new(150, 3.0) + Array.new(19_450, 0.0)
      item = item_of(values)
      lower, upper = intervals({ big: item }).fetch(:big)
      half = 1.959964 * item[:standard_error]
      # About 413 effective events: the width is the normal one's, the ends shifted up by the posterior's slight skew.
      expect(upper - lower).to be_within(0.01 * 2 * half).of(2 * half)
      expect(lower).to be_within(0.05 * half).of(item[:mean] - half)
      expect(upper).to be_within(0.05 * half).of(item[:mean] + half)

      small = item_of([ 1.0, 1.0 ] + Array.new(340, 0.0))
      lower, upper = intervals({ small: small }).fetch(:small)
      expect(upper - small[:mean]).to be > small[:mean] - lower
      expect(lower).to be > 0.0
    end

    it "keeps an item whose visits all hold one value above 0 inside: up to its mean, from (1 - mass)^(1 / (n + 1))" do
      lower, upper = intervals({ flat: { visits: 9, mean: 0.15, standard_error: 0.0 } }).fetch(:flat)
      expect(upper).to eq(0.15)
      expect(lower).to be_within(1e-12).of(0.15 * (0.05**(1.0 / 10.0)))
    end

    it "has no interval without data, keeps the items' keys and order, and reads Symbol or String field names" do
      answer = intervals({ "a" => { "visits" => 300, "mean" => 0.02, "standard_error" => 0.008 },
                           b: { visits: 0, mean: 0.0, standard_error: 0.0 }, "c" => { visits: 300, mean: nil },
                           "d" => { visits: 300, mean: 0.02, standard_error: nil }, "e" => { visits: nil, mean: 0.0 } })
      expect(answer.keys).to eq([ "a", :b, "c", "d", "e" ])
      expect(answer.values_at(:b, "c", "d", "e")).to eq([ nil, nil, nil, nil ])
      lower, upper = answer.fetch("a")
      expect(lower).to be < 0.02
      expect(upper).to be > 0.02
    end

    it "answers an item with a mean above 0 from its own figures alone, the same in any set and on every call" do
      own = { visits: 342, mean: 0.0058, standard_error: 0.0041 }
      alone = intervals({ "a" => own }).fetch("a")
      beside = intervals({ "z" => { visits: 300, mean: 0.0, standard_error: 0.0 }, "a" => own,
                           "b" => { visits: 12, mean: 2.5, standard_error: 1.1 } }).fetch("a")

      expect(beside).to eq(alone)
      expect(intervals({ "a" => own }).fetch("a")).to eq(alone)
    end

    it "refuses an item whose arithmetic leaves a Float's range, naming the item" do
      # A mean whose square underflows: no value of one event above 0.
      expect { intervals({ "a" => { visits: 9, mean: 1e-200, standard_error: 0.0 } }) }
        .to raise_error(ArgumentError, /item "a"'s interval leaves a Float's range \(its own value of one event is 0.0\)/)
      # A set whose value of one event underflows to 0 / 0, read by its item of no event (the other item is answered).
      tiny = { visits: 1e-200, mean: 1e-200, standard_error: 1.0 }
      expect(intervals({ "a" => tiny }).fetch("a")).to eq([ 1e-200, 0.975 ])
      expect { intervals({ "a" => tiny, "z" => { visits: 5, mean: 0.0, standard_error: 0.0 } }) }
        .to raise_error(ArgumentError, /item "z"'s interval leaves a Float's range \(the set's value of one event is NaN\)/)
      # Visits so many that 1 + visits - events rounds to 0.
      expect { intervals({ "a" => { visits: 1e20, mean: 0.5, standard_error: 0.0 } }) }
        .to raise_error(ArgumentError, /item "a"'s interval leaves a Float's range \(1 \+ visits - events is 0.0\)/)
    end

    it "refuses malformed input with an ArgumentError" do
      good = { visits: 300, mean: 0.02, standard_error: 0.008 }
      [ [ [ good ], 0.95 ], [ { "a" => [ 1, 2 ] }, 0.95 ], [ { "a" => good }, 1.0 ], [ { "a" => good }, nil ],
        [ { "a" => good.merge(mean: -0.01) }, 0.95 ], [ { "a" => good.merge(standard_error: -0.1) }, 0.95 ],
        [ { "a" => good.merge(visits: -3) }, 0.95 ], [ { "a" => good.merge(mean: Float::NAN) }, 0.95 ],
        [ { "a" => good.merge(visits: Float::INFINITY) }, 0.95 ], [ { "a" => good.merge(mean: "0.02") }, 0.95 ] ]
        .each do |items, mass|
          expect { intervals(items, mass: mass) }.to raise_error(ArgumentError), [ items, mass ].inspect
        end
    end
  end
end
