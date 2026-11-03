# frozen_string_literal: true

# LO3: the Monte Carlo cross-checks of the posterior set's odds and summaries (01 GB11, GB12), written before their
# implementers: the rank rule's visits against its exact odds (LD1), and the exact P(best) and expected loss against
# plain joint draws (LC2, on LA4's NormalArgmax and BetaArgmax.expected_maximum). Tagged `simulation: true`: excluded
# unless SIMULATIONS=1 (spec/spec_helper.rb). Every figure is printed through the reporter for the README's
# measured-results section.
#
# THE VISITS: 2 * 10**5 calls of `rank(rng:)` a fixture on one seeded Random. Every head and every position is checked
# against its exact value (pick_probabilities, positions) within 4 standard errors, sqrt(p * (1 - p) / n), and the
# orders by one chi-square test against order_probability: each order with n * p >= 5 its own bin, the rest pooled
# into one, at a p-value above 0.001 (an order below 3e-7 could not pass a 4-standard-error test by itself). An order
# the exact odds give no probability must never be seen.
#
# THE JOINT DRAWS: 10**6 draws a fixture, one Sampler#beta (or mean + sd * Sampler#gaussian) an item, in order; the
# frequency with which each item is the largest against summary's p_best (and NormalArgmax.probabilities), and the mean
# of max - X_i against its expected loss, each within 4 standard errors of the draws. The Normal fixtures are amounts
# near 5 a view, where E[max] must be read over the Normals' own range, not as 1 - an integral over [0, 1].
module OddsSimulation
  module_function

  VISITS = 200_000
  JOINT_DRAWS = 1_000_000
  STANDARD_ERRORS = 4.0
  P_VALUE = 0.001
  RANK_SEED = 20_261_131
  JOINT_SEED = 20_261_141

  BETA4 = { "a" => [ 0.3, 20.0 ], "b" => [ 1.0, 20.0 ], "c" => [ 2.0, 30.0 ], "d" => [ 5.0, 90.0 ] }.freeze
  BETA5 = { "a" => [ 0.02, 19.98 ], "b" => [ 3.0, 60.0 ], "c" => [ 4.0, 70.0 ], "d" => [ 6.0, 95.0 ],
            "e" => [ 2.0, 45.0 ] }.freeze
  NORMAL4 = { "a" => [ 5.0, 0.4 ], "b" => [ 5.3, 0.5 ], "c" => [ 4.8, 0.3 ], "d" => [ 5.1, 0.8 ] }.freeze
  NORMAL5 = { "a" => [ 1.0, 1.0 ], "b" => [ 1.5, 0.5 ], "c" => [ 0.5, 2.0 ], "d" => [ 1.2, 0.2 ],
              "e" => [ 0.8, 0.7 ] }.freeze
  VIEWS4 = { "a" => 50.0, "b" => 400.0, "c" => 400.0, "d" => 400.0 }.freeze
  LOSSES = { "a" => [ 12.0, 88.0 ], "b" => [ 15.0, 85.0 ], "c" => [ 9.0, 91.0 ], "d" => [ 3.0, 17.0 ] }.freeze

  CASES = [
    { label: "five Betas, an alpha of 0.02 among them, shuffle 0.1 and explore 0.1", kind: :beta, pairs: BETA5,
      rule: { shuffle: 0.1, explore: 0.1 } },
    { label: "four Betas, an alpha of 0.3 among them, explore 0.2 among items of at least 100 views", kind: :beta,
      pairs: BETA4, views: VIEWS4, rule: { explore: 0.2, min_views: 100.0 } },
    { label: "four Normals near 5 a view, shuffle 0.05", kind: :normal, pairs: NORMAL4, rule: { shuffle: 0.05 } },
    { label: "four Betas, c pinned at position 2, shuffle 0.1", kind: :beta, pairs: BETA4,
      rule: { shuffle: 0.1, pinned: { 2 => "c" } } },
    { label: "five Normals of mixed widths, explore 0.1, the top 2", kind: :normal, pairs: NORMAL5,
      rule: { explore: 0.1, top: 2 } }
  ].freeze

  def build(kind, pairs, views = nil)
    posteriors = SkiftetStatistical::Bandit::Posteriors
    kind == :beta ? posteriors.beta(pairs, views: views) : posteriors.normal(pairs, views: views)
  end

  # Every order the rule can answer: the free items' permutations (all of them, or the first `top`), each pinned item
  # at its own position.
  def orders(items, rule)
    pinned = rule.fetch(:pinned, {})
    free = items - pinned.values
    slots = (0...items.size).reject { |slot| pinned.key?(slot) }
    free.permutation(rule[:top] || free.size).map do |ranks|
      next ranks if pinned.empty?

      full = Array.new(items.size)
      pinned.each { |slot, item| full[slot] = item }
      ranks.each_with_index { |item, rank| full[slots[rank]] = item }
      full
    end
  end

  # |frequency - p| in standard errors; 0.0 where p is 0 or 1 and the frequency equals it, infinite where it does not.
  def gap(count, p, draws)
    frequency = count.to_f / draws
    sd = Math.sqrt(p * (1.0 - p) / draws)
    return (frequency - p).abs.zero? ? 0.0 : Float::INFINITY if sd.zero? || !sd.finite?

    (frequency - p).abs / sd
  end

  # Pearson's chi-square over the orders: each order with draws * p >= 5 its own bin, the rest pooled into one.
  def chi_square(observed, expected, draws)
    own, pooled = expected.partition { |_order, p| draws * p >= 5.0 }
    bins = own.map { |order, p| [ observed.fetch(order, 0), draws * p ] }
    pooled_expected = pooled.sum { |_order, p| draws * p }
    bins << [ pooled.sum { |order, _p| observed.fetch(order, 0) }, pooled_expected ] if pooled_expected.positive?
    statistic = bins.sum { |seen, wanted| ((seen - wanted)**2) / wanted }
    [ statistic, bins.size - 1, SkiftetStatistical::Significance.chi_squared_survival(statistic, bins.size - 1) ]
  end

  # Joint draws: how often each item is the largest, and the mean and standard error of max - X_i.
  def joint(values_of, items, draws)
    wins = Array.new(items, 0)
    sums = Array.new(items, 0.0)
    squares = Array.new(items, 0.0)
    draws.times do
      values = values_of.call
      best = values.max
      wins[values.index(best)] += 1
      values.each_with_index do |value, index|
        loss = best - value
        sums[index] += loss
        squares[index] += loss * loss
      end
    end
    losses = sums.zip(squares).map do |sum, square|
      mean = sum / draws
      [ mean, Math.sqrt([ (square / draws) - (mean * mean), 0.0 ].max / draws) ]
    end
    [ wins, losses ]
  end

  def report(text)
    RSpec.configuration.reporter.message("[odds simulation] #{text}")
  end
end

RSpec.describe "Posteriors odds simulation", simulation: true do
  def sim = OddsSimulation

  it "LO3 (LD1) gives rank(rng:)'s heads and positions within 4 standard errors of their exact odds, and its orders " \
     "a chi-square p-value above 0.001, over 2 * 10**5 visits a fixture" do
    failures = []
    sim::CASES.each_with_index do |fixture, index|
      set = sim.build(fixture[:kind], fixture[:pairs], fixture[:views])
      rule = fixture[:rule]
      rng = Random.new(sim::RANK_SEED + index)
      heads = Hash.new(0)
      slots = Hash.new(0)
      seen = Hash.new(0)
      sim::VISITS.times do
        items = set.rank(rng: rng, **rule).items
        heads[items.first] += 1
        items.each_with_index { |item, slot| slots[[ item, slot ]] += 1 }
        seen[items] += 1
      end

      gaps = []
      unless rule.key?(:pinned)
        set.pick_probabilities(**rule.except(:top))[:heads].each do |item, p|
          gaps << [ "head #{item}", sim.gap(heads[item], p, sim::VISITS) ]
        end
      end
      unless rule.key?(:top)
        set.positions(**rule)[:probabilities].each do |item, row|
          row.each_with_index { |p, slot| gaps << [ "#{item} at #{slot}", sim.gap(slots[[ item, slot ]], p, sim::VISITS) ] }
        end
      end
      expected = sim.orders(fixture[:pairs].keys, rule).to_h { |order| [ order, set.order_probability(order, **rule)[:probability] ] }
      unexpected = seen.keys.reject { |order| expected.fetch(order, 0.0).positive? }
      statistic, freedom, p_value = sim.chi_square(seen, expected, sim::VISITS)
      worst = gaps.max_by(&:last)
      sim.report(format("%s: largest gap %.2f standard errors (%s); orders chi-square %.1f on %d degrees, p %.4f",
                        fixture[:label], worst.last, worst.first, statistic, freedom, p_value))

      gaps.each { |name, z| failures << "#{fixture[:label]}: #{name} at #{z.round(2)}" if z > sim::STANDARD_ERRORS }
      failures << "#{fixture[:label]}: orders p-value #{p_value}" unless p_value > sim::P_VALUE
      failures << "#{fixture[:label]}: orders with no probability seen: #{unexpected.inspect}" unless unexpected.empty?
    end
    expect(failures).to be_empty, failures.join("\n")
  end

  it "LO3 (LC2) gives a Beta set's p_best and expected loss within 4 standard errors of 10**6 joint draws" do
    failures = []
    [ [ "four Betas of 100 to 200 views", sim::LOSSES ], [ "four Betas, an alpha of 0.3 among them", sim::BETA4 ] ]
      .each_with_index do |(label, pairs), index|
      summary = sim.build(:beta, pairs).summary
      sampler = SkiftetStatistical::Sampler.new(Random.new(sim::JOINT_SEED + index))
      wins, losses = sim.joint(-> { pairs.values.map { |alpha, beta| sampler.beta(alpha, beta) } }, pairs.size,
                               sim::JOINT_DRAWS)
      pairs.keys.each_with_index do |item, at|
        p_best = sim.gap(wins[at], summary[item][:p_best], sim::JOINT_DRAWS)
        mean, error = losses[at]
        loss = (mean - summary[item][:expected_loss]).abs / error
        sim.report(format("%s, %s: p_best %.2f, expected loss %.2f standard errors (%.3e against %.3e)", label, item,
                          p_best, loss, summary[item][:expected_loss], mean))
        failures << "#{label} #{item}: p_best at #{p_best.round(2)}" if p_best > sim::STANDARD_ERRORS
        failures << "#{label} #{item}: expected loss at #{loss.round(2)}" if loss > sim::STANDARD_ERRORS
      end
    end
    expect(failures).to be_empty, failures.join("\n")
  end

  it "LO3 (LA4, LC2) gives a Normal set's p_best, NormalArgmax.probabilities and expected loss within 4 standard " \
     "errors of 10**6 joint draws, amounts near 5 a view" do
    failures = []
    [ [ "four Normals near 5 a view", sim::NORMAL4 ], [ "five Normals of mixed widths", sim::NORMAL5 ] ]
      .each_with_index do |(label, pairs), index|
      summary = sim.build(:normal, pairs).summary
      direct = SkiftetStatistical::Bandit::NormalArgmax.probabilities(pairs.values)
      sampler = SkiftetStatistical::Sampler.new(Random.new(sim::JOINT_SEED + 10 + index))
      wins, losses = sim.joint(-> { pairs.values.map { |mean, sd| mean + (sd * sampler.gaussian) } }, pairs.size,
                               sim::JOINT_DRAWS)
      pairs.keys.each_with_index do |item, at|
        p_best = sim.gap(wins[at], summary[item][:p_best], sim::JOINT_DRAWS)
        largest = sim.gap(wins[at], direct[at], sim::JOINT_DRAWS)
        mean, error = losses[at]
        loss = (mean - summary[item][:expected_loss]).abs / error
        sim.report(format("%s, %s: p_best %.2f, NormalArgmax %.2f, expected loss %.2f standard errors", label, item,
                          p_best, largest, loss))
        failures << "#{label} #{item}: p_best at #{p_best.round(2)}" if p_best > sim::STANDARD_ERRORS
        failures << "#{label} #{item}: NormalArgmax at #{largest.round(2)}" if largest > sim::STANDARD_ERRORS
        failures << "#{label} #{item}: expected loss at #{loss.round(2)}" if loss > sim::STANDARD_ERRORS
      end
    end
    expect(failures).to be_empty, failures.join("\n")
  end
end
