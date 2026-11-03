# frozen_string_literal: true

# LD2: the order rule at small-site traffic (01 GB15). A list of several items is ordered for every visit by the
# posterior set's `rank`, and the best item is wanted on top. Tagged `simulation: true`: excluded unless SIMULATIONS=1
# (spec/spec_helper.rb). Every figure is printed through RSpec's reporter (the JSON formatter keeps it under
# "messages") for the README's measured-results section.
#
# THE MODEL. A list of K items (K = 3, 5 or 10), each with its own true rate, fixed for the run and dealt to the items
# in a random order a run. Every visit is shown the whole list in one order and sees its first `depth` positions; the
# depth is drawn once a visit, independent of the order, from a reach curve (the share of visits that see position l):
# (1, 0.6, 0.35) at K = 3, (1, 0.70, 0.50, 0.35, 0.25) at K = 5 and 0.8**l at K = 10. Each item a visit sees yields an
# outcome at the item's rate: yes/no (1 at the rate, else 0), or amounts (at the rate an amount, log-normal with mean
# 200 and sigma 0.9, capped at 2 000; else 0).
#
# THE POLICY, the caller's whole part. Views count only where an item was seen, so an item low in the order is not
# penalised for the visits that never reached it. Every hour the caller builds the posterior set from its tallies:
# `Posteriors.yes_no` on the flat prior, or `Posteriors.amounts` under Prior.amount(mean: 4.0, variance: 2_000.0,
# strength: 20.0). Every visit of the hour is shown that set's `rank(rng:, shuffle: 0.10)`: the order by one Thompson
# draw of every item, and a uniformly random order on a 0.10 share of the visits. 28 days of 24 hours; a day's visits
# spread evenly over its hours (the visits an hour rounded, so 300 a day is 13 an hour). One seeded Random a run feeds
# the model and the rule alike.
#
# THE MEASURES, over 50 seeded runs a cell (run r of the cell at index i of CELLS, both from 0, on Random.new(SEED +
# 50 i + r)). Lost:the expected outcomes the shown orders missed against the best order
# (the items by true rate, the highest on top), summed over every visit of every run, as a share of the best order's;
# the same for a fixed order, one random order a run shown to every visit. An order's expected outcomes a visit are
# the sum over its positions of the share of visits that see the position times the rate of the item there (for
# amounts times the mean amount too, the same factor for every item, so it leaves the shares as they are). The
# posterior-mean order on day 7 and on day 28: the items by the means of `summary(among: [])`, ties to the earlier
# item, on the set built from the tallies at the end of that day; whether its first item is the best one, and whether
# it is the best order exactly. The shuffled share: how often `rank` answered the :shuffled branch, against 0.10.
#
# THE MIXED CELL: two kinds of visits, half shallow (reach (1, 0.5, 0.3, 0.15, 0.1), every rate halved), half deep
# (reach (1, 0.9, 0.7, 0.55, 0.4), every rate times 1.5). The deep visits both see the lower positions and yield more,
# so an item shown low reads a rate above the one it would show on top: the known weakness of counting views only
# where seen. An order's expected outcomes a visit weigh each position by both kinds (the share of each kind times its
# reach there times its rate factor).
#
# THE BOUNDS, each cell's own, asserted on the posterior-mean order and the lost share. Every cell: lost at most half
# the fixed order's lost, and the shuffled share within 4 standard errors of 0.10. Close rates at 300 / 1 000 / 3 000
# visits a day: best on top on day 28 in at least 75 / 90 / 90 % of runs, the exact order on day 28 in at least
# 35 / 75 / 90 %, lost at most 4.5 / 3.0 / 2.0 %. Clear rates: best on top on day 7 in at least 90 %. K = 3 and
# K = 10: best on top on day 28 in at least 90 %. Amounts: best on top on day 28 in at least 85 / 95 / 95 %, lost at
# most 6 / 4 / 3 %. Mixed visits at 300 / 1 000: best on top on day 28 in at least 75 / 90 %.
#
# Each cell also prints its reference: the same model run 100 times a cell in another language and generator, the
# figures the bounds were set from with margin (fixed lost, lost, best on top on day 7 and day 28, exact order on day
# 28). Fifty runs and another generator read near them, not equal to them.
module OrderSimulation
  module_function

  RUNS = 50
  DAYS = 28
  HOURS = 24
  SHUFFLE = 0.10
  STANDARD_ERRORS = 4.0
  SEED = 20_261_102
  ITEMS = %w[a b c d e f g h i j].freeze
  # The days, 1-based, on which the posterior-mean order is read.
  READ_ON = [ 7, 28 ].freeze

  AMOUNT_PRIOR = SkiftetStatistical::Bandit::Prior.amount(mean: 4.0, variance: 2_000.0, strength: 20.0)
  AMOUNT_MEAN = 200.0
  AMOUNT_SIGMA = 0.9
  AMOUNT_CAP = 2_000.0
  # The log-normal's location: its mean is AMOUNT_MEAN before the cap.
  AMOUNT_LOCATION = Math.log(AMOUNT_MEAN) - (AMOUNT_SIGMA * AMOUNT_SIGMA / 2.0)

  REACH3 = [ 1.0, 0.6, 0.35 ].freeze
  REACH5 = [ 1.0, 0.70, 0.50, 0.35, 0.25 ].freeze
  REACH10 = Array.new(10) { |position| 0.8**position }.freeze
  CLOSE = [ 0.05, 0.045, 0.04, 0.035, 0.03 ].freeze
  CLEAR = [ 0.08, 0.06, 0.05, 0.04, 0.03 ].freeze
  THREE = [ 0.05, 0.04, 0.03 ].freeze
  TEN = [ 0.06, 0.055, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015 ].freeze
  AMOUNT_RATES = [ 0.04, 0.03, 0.025, 0.02, 0.015 ].freeze

  # A kind of visit: its share of the visits, its reach curve and the factor on every rate.
  Visitors = Struct.new(:weight, :reach, :factor, keyword_init: true)

  ALIKE3 = [ Visitors.new(weight: 1.0, reach: REACH3, factor: 1.0) ].freeze
  ALIKE5 = [ Visitors.new(weight: 1.0, reach: REACH5, factor: 1.0) ].freeze
  ALIKE10 = [ Visitors.new(weight: 1.0, reach: REACH10, factor: 1.0) ].freeze
  MIXED = [ Visitors.new(weight: 0.5, reach: [ 1.0, 0.5, 0.3, 0.15, 0.1 ].freeze, factor: 0.5),
            Visitors.new(weight: 0.5, reach: [ 1.0, 0.9, 0.7, 0.55, 0.4 ].freeze, factor: 1.5) ].freeze

  # A cell: its words, the outcome kind (:yes_no or :amounts), the true rates, the visits a day, the kinds of visit,
  # its own bounds (top_day7, top_day28 and exact_day28 the least share of runs, lost the largest lost share) and the
  # reference figures.
  Cell = Struct.new(:label, :kind, :rates, :per_day, :visitors, :bounds, :reference, keyword_init: true)

  CELLS = [
    Cell.new(label: "yes/no, five items, close rates (5, 4.5, 4, 3.5, 3 %), 300 visits a day", kind: :yes_no,
             rates: CLOSE, per_day: 300, visitors: ALIKE5,
             bounds: { top_day28: 0.75, exact_day28: 0.35, lost: 0.045 },
             reference: "fixed lost 8.3 %, lost 2.8 %, top day 7 69 %, top day 28 90 %, exact day 28 54 %"),
    Cell.new(label: "yes/no, five items, close rates (5, 4.5, 4, 3.5, 3 %), 1 000 visits a day", kind: :yes_no,
             rates: CLOSE, per_day: 1_000, visitors: ALIKE5,
             bounds: { top_day28: 0.90, exact_day28: 0.75, lost: 0.030 },
             reference: "fixed lost 7.6 %, lost 1.6 %, top day 7 91 %, top day 28 99 %, exact day 28 91 %"),
    Cell.new(label: "yes/no, five items, close rates (5, 4.5, 4, 3.5, 3 %), 3 000 visits a day", kind: :yes_no,
             rates: CLOSE, per_day: 3_000, visitors: ALIKE5,
             bounds: { top_day28: 0.90, exact_day28: 0.90, lost: 0.020 },
             reference: "fixed lost 7.7 %, lost 1.1 %, top day 7 99 %, top day 28 100 %, exact day 28 100 %"),
    Cell.new(label: "yes/no, five items, clear rates (8, 6, 5, 4, 3 %), 300 visits a day", kind: :yes_no,
             rates: CLEAR, per_day: 300, visitors: ALIKE5, bounds: { top_day7: 0.90 },
             reference: "fixed lost 12.8 %, lost 2.4 %, top day 7 100 %, top day 28 100 %, exact day 28 97 %"),
    Cell.new(label: "yes/no, five items, clear rates (8, 6, 5, 4, 3 %), 1 000 visits a day", kind: :yes_no,
             rates: CLEAR, per_day: 1_000, visitors: ALIKE5, bounds: { top_day7: 0.90 },
             reference: "fixed lost 12.8 %, lost 1.7 %, top day 7 100 %, top day 28 100 %, exact day 28 100 %"),
    Cell.new(label: "yes/no, five items, clear rates (8, 6, 5, 4, 3 %), 3 000 visits a day", kind: :yes_no,
             rates: CLEAR, per_day: 3_000, visitors: ALIKE5, bounds: { top_day7: 0.90 },
             reference: "fixed lost 13.6 %, lost 1.5 %, top day 7 100 %, top day 28 100 %, exact day 28 100 %"),
    Cell.new(label: "yes/no, three items (5, 4, 3 %), 1 000 visits a day", kind: :yes_no,
             rates: THREE, per_day: 1_000, visitors: ALIKE3, bounds: { top_day28: 0.90 },
             reference: "fixed lost 7.0 %, lost 1.1 %, top day 7 100 %, top day 28 100 %, exact day 28 100 %"),
    Cell.new(label: "yes/no, ten items (6 % down to 1.5 %), 3 000 visits a day", kind: :yes_no,
             rates: TEN, per_day: 3_000, visitors: ALIKE10, bounds: { top_day28: 0.90 },
             reference: "fixed lost 18.6 %, lost 2.2 %, top day 7 98 %, top day 28 100 %, exact day 28 100 %"),
    Cell.new(label: "amounts, five items (4, 3, 2.5, 2, 1.5 %), 300 visits a day", kind: :amounts,
             rates: AMOUNT_RATES, per_day: 300, visitors: ALIKE5, bounds: { top_day28: 0.85, lost: 0.06 },
             reference: "fixed lost 13.1 %, lost 4.3 %, top day 7 88 %, top day 28 98 %, exact day 28 53 %"),
    Cell.new(label: "amounts, five items (4, 3, 2.5, 2, 1.5 %), 1 000 visits a day", kind: :amounts,
             rates: AMOUNT_RATES, per_day: 1_000, visitors: ALIKE5, bounds: { top_day28: 0.95, lost: 0.04 },
             reference: "fixed lost 12.4 %, lost 2.6 %, top day 7 98 %, top day 28 100 %, exact day 28 88 %"),
    Cell.new(label: "amounts, five items (4, 3, 2.5, 2, 1.5 %), 3 000 visits a day", kind: :amounts,
             rates: AMOUNT_RATES, per_day: 3_000, visitors: ALIKE5, bounds: { top_day28: 0.95, lost: 0.03 },
             reference: "fixed lost 12.9 %, lost 1.8 %, top day 7 100 %, top day 28 100 %, exact day 28 98 %"),
    Cell.new(label: "yes/no, five items, close rates, mixed visits (half shallow, half deep), 300 visits a day",
             kind: :yes_no, rates: CLOSE, per_day: 300, visitors: MIXED, bounds: { top_day28: 0.75 },
             reference: "fixed lost 6.4 %, lost 2.4 %, top day 28 90 %, exact day 28 58 %"),
    Cell.new(label: "yes/no, five items, close rates, mixed visits (half shallow, half deep), 1 000 visits a day",
             kind: :yes_no, rates: CLOSE, per_day: 1_000, visitors: MIXED, bounds: { top_day28: 0.90 },
             reference: "fixed lost 6.2 %, lost 1.5 %, top day 28 99 %, exact day 28 95 %")
  ].freeze

  # The share of visits that see each position, both kinds of visit weighed, times each kind's rate factor: an order's
  # expected outcomes a visit are the sum over positions of this times the rate of the item there.
  def worth(visitors, size)
    Array.new(size) do |position|
      visitors.sum { |kind| kind.weight * kind.reach.fetch(position) * kind.factor }
    end
  end

  def expected(order, rate, worth)
    total = 0.0
    order.each_with_index { |item, position| total += worth[position] * rate.fetch(item) }
    total
  end

  # The kind of the next visit: the first whose running weight is above one uniform (no draw for a single kind).
  def visitor(visitors, rng)
    return visitors.first if visitors.size == 1

    uniform = rng.rand
    running = 0.0
    visitors.each do |kind|
      running += kind.weight
      return kind if uniform < running
    end
    visitors.last
  end

  # How many positions a visit sees: every position whose reach is above one uniform, from the top.
  def depth(reach, rng)
    uniform = rng.rand
    seen = 0
    seen += 1 while seen < reach.size && uniform < reach[seen]
    seen
  end

  def amount(sampler)
    [ Math.exp(AMOUNT_LOCATION + (AMOUNT_SIGMA * sampler.gaussian)), AMOUNT_CAP ].min
  end

  def blank_tallies(kind, items)
    items.to_h do |item|
      [ item, kind == :amounts ? { views: 0.0, outcomes: 0.0, outcome_squares: 0.0 } : { views: 0.0, outcomes: 0.0 } ]
    end
  end

  def posteriors(kind, tallies)
    set = SkiftetStatistical::Bandit::Posteriors
    kind == :amounts ? set.amounts(tallies, priors: AMOUNT_PRIOR) : set.yes_no(tallies)
  end

  # The items by their posterior means, the largest first, ties to the earlier item.
  def mean_order(set)
    means = set.summary(among: []).transform_values { |row| row.fetch(:mean) }
    set.items.each_with_index.sort_by { |item, index| [ -means.fetch(item), index ] }.map(&:first)
  end

  # One run of a cell: { lost:, fixed_lost:, best_value:, visits:, shuffled:, top: { day => bool }, exact: { day =>
  # bool } }, lost and fixed_lost the summed expected outcomes missed, best_value the best order's a visit.
  def run(cell, seed)
    rng = Random.new(seed)
    sampler = SkiftetStatistical::Sampler.new(rng)
    items = ITEMS.first(cell.rates.size)
    rate = items.zip(cell.rates.shuffle(random: rng)).to_h
    fixed = items.shuffle(random: rng)
    best = items.each_with_index.sort_by { |item, index| [ -rate.fetch(item), index ] }.map(&:first)
    weights = worth(cell.visitors, items.size)
    best_value = expected(best, rate, weights)
    per_hour = (cell.per_day / HOURS.to_f).round
    visits = DAYS * HOURS * per_hour
    tallies = blank_tallies(cell.kind, items)
    amounts = cell.kind == :amounts
    lost = 0.0
    shuffled = 0
    top = {}
    exact = {}

    (1..DAYS).each do |day|
      HOURS.times do
        set = posteriors(cell.kind, tallies)
        per_hour.times do
          order = set.rank(rng: rng, shuffle: SHUFFLE)
          shuffled += 1 if order.branch == :shuffled
          shown = order.items
          lost += best_value - expected(shown, rate, weights)
          kind = visitor(cell.visitors, rng)
          depth(kind.reach, rng).times do |position|
            tally = tallies.fetch(shown[position])
            tally[:views] += 1.0
            next unless rng.rand < rate.fetch(shown[position]) * kind.factor

            outcome = amounts ? amount(sampler) : 1.0
            tally[:outcomes] += outcome
            tally[:outcome_squares] += outcome * outcome if amounts
          end
        end
      end
      next unless READ_ON.include?(day)

      order = mean_order(posteriors(cell.kind, tallies))
      top[day] = order.first == best.first
      exact[day] = order == best
    end

    { lost: lost, fixed_lost: (best_value - expected(fixed, rate, weights)) * visits, best_value: best_value,
      visits: visits, shuffled: shuffled, top: top, exact: exact }
  end

  # A cell's figures over its runs: the lost shares pooled over every visit of every run, the shares of runs.
  def measure(cell, index)
    runs = Array.new(RUNS) { |number| run(cell, SEED + (index * RUNS) + number) }
    best = runs.sum { |figures| figures[:best_value] * figures[:visits] }
    visits = runs.sum { |figures| figures[:visits] }
    share = ->(name, day) { runs.count { |figures| figures.fetch(name).fetch(day) }.fdiv(RUNS) }
    { lost: runs.sum { |figures| figures[:lost] } / best,
      fixed_lost: runs.sum { |figures| figures[:fixed_lost] } / best,
      top_day7: share.call(:top, 7), top_day28: share.call(:top, 28), exact_day28: share.call(:exact, 28),
      shuffled: runs.sum { |figures| figures[:shuffled] }.fdiv(visits), visits: visits }
  end

  # Every bound the cell holds that its figures break, in words.
  def failures(cell, figures)
    broken = []
    if figures[:lost] > figures[:fixed_lost] / 2.0
      broken << format("lost %.2f %% is above %.2f %%, half the fixed order's %.2f %%", 100 * figures[:lost],
                       50 * figures[:fixed_lost], 100 * figures[:fixed_lost])
    end
    error = Math.sqrt(SHUFFLE * (1.0 - SHUFFLE) / figures[:visits])
    if (figures[:shuffled] - SHUFFLE).abs > STANDARD_ERRORS * error
      broken << format("the shuffled share %.5f is %.1f standard errors from %.2f", figures[:shuffled],
                       (figures[:shuffled] - SHUFFLE).abs / error, SHUFFLE)
    end
    cell.bounds.each do |name, bound|
      if name == :lost
        broken << format("lost %.2f %% is above %.1f %%", 100 * figures[:lost], 100 * bound) if figures[:lost] > bound
      elsif figures.fetch(name) < bound
        broken << format("%s %.0f %% is below %.0f %%", name, 100 * figures.fetch(name), 100 * bound)
      end
    end
    broken
  end

  def report(text)
    RSpec.configuration.reporter.message("[order simulation] #{text}")
  end
end

RSpec.describe "The order rule at small-site traffic", simulation: true do
  def sim = OrderSimulation

  OrderSimulation::CELLS.each_with_index do |cell, index|
    it "LD2 finds the best order: #{cell.label} (rank(rng:, shuffle: 0.10), 50 runs of 28 days)" do
      started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      figures = sim.measure(cell, index)
      elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
      sim.report(format("%s: fixed lost %.1f %%, lost %.1f %%, best on top day 7 %.0f %%, day 28 %.0f %%, exact " \
                        "order day 28 %.0f %%, shuffled %.4f of %d visits, %.1f s (reference, 100 runs: %s)",
                        cell.label, 100 * figures[:fixed_lost], 100 * figures[:lost], 100 * figures[:top_day7],
                        100 * figures[:top_day28], 100 * figures[:exact_day28], figures[:shuffled], figures[:visits],
                        elapsed, cell.reference))
      broken = sim.failures(cell, figures)
      expect(broken).to be_empty, "#{cell.label}: #{broken.join('; ')}"
    end
  end
end
