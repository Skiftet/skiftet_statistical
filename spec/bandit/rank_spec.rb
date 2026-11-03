# frozen_string_literal: true

require "digest"
require "json"
require "open3"
require "rbconfig"

# LO2: the acceptance of Posteriors#rank, the one order rule, under each of its three sources (a key, given uniforms,
# an RNG). Written from the rule as stated before it was built; nothing here reads the code under test to make an
# expected value. The references:
# - keyed uniforms: SHA256 of the parts joined by NUL, the first 8 bytes big-endian, >> 11, / 2**53, written here and
#   checked against the stated values for "t" (the first example);
# - Beta draws at chosen uniforms: whole-parameter Betas, whose distribution function is a finite binomial sum,
#   inverted by bisection in exact Rationals by hand; Beta(1, b) by its closed form 1 - (1 - u)**(1 / b);
#   the bulk keyed checks read 0.2.0's BetaFunction.quantile, which this release leaves unchanged;
# - Normal draws: mean + sd * Significance.normal_quantile(u), the stated rule (hand values by Newton on erfc);
# - RNG draws: a replay on a fresh Random of the same seed through Sampler, in the stated order: one branch value
#   when shuffle + explore > 0, then one priority per free item on the shuffled or explored branch, then one draw per
#   free item where the branch draws.
# A keyed or given-uniform draw is compared at 1e-9 relative (it goes through Math.lgamma), a replayed RNG draw at
# 1e-12; orders, leaders, branches and what the stream gave exactly. The rule says nothing of what draws holds on the
# shuffled branch, which orders by priority alone, so no example reads it there.
RSpec.describe "SkiftetStatistical::Bandit::Posteriors#rank" do
  def posteriors = SkiftetStatistical::Bandit::Posteriors
  def beta_function = SkiftetStatistical::BetaFunction
  def significance = SkiftetStatistical::Significance
  def sampler_for(rng) = SkiftetStatistical::Sampler.new(rng)

  def keyed(*parts)
    text = parts.map { |part| part.nil? ? "" : part.to_s }.join("\0")
    (Digest::SHA256.digest(text).byteslice(0, 8).unpack1("Q>") >> 11) / (2.0**53)
  end

  def keyed_uniforms(key, items)
    parts = Array(key)
    {
      explore: keyed(*parts, "explore"),
      priority: items.to_h { |item| [ item, keyed(*parts, "priority", item) ] },
      draw: items.to_h { |item| [ item, keyed(*parts, "draw", item) ] }
    }
  end

  # A Random whose every rand call is recorded, so a test can say what the rule took from the stream.
  def recording(seed)
    Class.new(Random) do
      def calls = (@calls ||= [])

      def rand(*args)
        value = super
        calls << value
        value
      end
    end.new(seed)
  end

  def relative(actual, expected)
    return (actual - expected).abs if expected.zero?

    (actual - expected).abs / expected.abs
  end

  def expect_draws(actual, expected, tolerance, label)
    expect(actual.keys.sort).to eq(expected.keys.sort), "#{label}: drawn items #{actual.keys.inspect}"
    expected.each do |item, value|
      expect(relative(actual.fetch(item), value)).to be <= tolerance,
                                                     "#{label}: #{item} drew #{actual.fetch(item)}, not #{value}"
    end
  end

  # The stated rule, written out from the requirement, for the bulk keyed and replayed checks:
  # contenders are the free items in may_lead (all free items when none is), eligible the contenders with at least
  # min_views; c < shuffle shuffles (head the contender of smallest priority, the rest by priority); c < shuffle +
  # explore with an eligible item explores (head the eligible item of smallest priority, the rest by draw); else
  # Thompson (head the contender of largest draw, the rest by draw); ties to the earlier item; pins hold their slots
  # and the free order fills the others in order; the leader is the free head when it has min_views.
  def rule_order(items:, views:, rule:, branch_value:, priorities:, draws:)
    pinned = rule.fetch(:pinned, {})
    free = items - pinned.values
    may_lead = rule[:may_lead]
    contenders = may_lead.nil? ? free : free.select { |item| may_lead.include?(item) }
    contenders = free if contenders.empty?
    min_views = rule.fetch(:min_views, 0.0)
    eligible = contenders.select { |item| views.fetch(item) >= min_views }
    shuffle = rule.fetch(:shuffle, 0.0)
    explore = rule.fetch(:explore, 0.0)
    by_priority = ->(list) { list.each_with_index.sort_by { |item, i| [ priorities.fetch(item), i ] }.map(&:first) }
    by_draw = ->(list) { list.each_with_index.sort_by { |item, i| [ -draws.fetch(item), i ] }.map(&:first) }

    if branch_value && branch_value < shuffle
      branch = :shuffled
      head = by_priority.call(contenders).first
      rest = by_priority.call(free - [ head ])
    elsif branch_value && branch_value < shuffle + explore && !eligible.empty?
      branch = :explored
      head = by_priority.call(eligible).first
      rest = by_draw.call(free - [ head ])
    else
      branch = :thompson
      head = by_draw.call(contenders).first
      rest = by_draw.call(free - [ head ])
    end

    full = Array.new(items.size)
    pinned.each { |position, item| full[position] = item }
    (0...items.size).reject { |position| pinned.key?(position) }.zip([ head, *rest ]).each { |position, item| full[position] = item }
    { items: full.first(rule[:top] || items.size), leader: views.fetch(head) >= min_views ? head : nil, branch: branch }
  end

  def pairs_of(set)
    parameters = set.parameters
    set.items.to_h { |item| [ item, parameters.is_a?(Hash) ? parameters.fetch(item) : parameters.fetch(set.items.index(item)) ] }
  end

  def keyed_beta_draws(set, key)
    pairs_of(set).to_h { |item, (alpha, beta)| [ item, beta_function.quantile(keyed(*Array(key), "draw", item), alpha, beta) ] }
  end

  def answer(order) = { items: order.items, leader: order.leader, branch: order.branch }

  # What the rng source must take, replayed on a fresh Random of the same seed: the branch value, the priorities and
  # the draws (Sampler#beta, or mean + sd * Sampler#gaussian), in that order, and the stream's next value after them.
  def replay(seed, free, pairs, branch_value:, priorities:, draws:, kind: :beta)
    rng = Random.new(seed)
    sampler = sampler_for(rng)
    value = branch_value ? rng.rand : nil
    priority = priorities ? free.to_h { |item| [ item, rng.rand ] } : {}
    drawn = if draws
              free.to_h do |item|
                first, second = pairs.fetch(item)
                [ item, kind == :beta ? sampler.beta(first, second) : first + (second * sampler.gaussian) ]
              end
    else
              {}
    end
    { value: value, priority: priority, draws: drawn, next: rng.rand }
  end

  # How many rand calls that replay makes.
  def calls_in(seed, free, pairs, **options)
    counter = recording(seed)
    sampler = sampler_for(counter)
    counter.rand if options[:branch_value]
    free.each { counter.rand } if options[:priorities]
    free.each { |item| options[:kind] == :normal ? sampler.gaussian : sampler.beta(*pairs.fetch(item)) } if options[:draws]
    counter.calls.size
  end

  # Four items with whole parameters, drawn at chosen uniforms. Their draws, by hand:
  #   a Beta(2, 8) at 0.99 = 0.5440336922480249    b Beta(5, 5) at 0.5 = 0.5
  #   c Beta(8, 2) at 0.1  = 0.6316376362100744    d Beta(3, 3) at 0.7 = 0.6101816494690415
  # so by draw c > d > a > b, while by mean c (0.8) > b = d (0.5) > a (0.2). Priorities: b 0.2 < d 0.4 < a 0.6 < c 0.9.
  let(:four_pairs) { { "a" => [ 2.0, 8.0 ], "b" => [ 5.0, 5.0 ], "c" => [ 8.0, 2.0 ], "d" => [ 3.0, 3.0 ] } }
  let(:four_views) { { "a" => 40.0, "b" => 10.0, "c" => 50.0, "d" => 0.0 } }
  let(:four) { posteriors.beta(four_pairs, views: four_views) }
  let(:four_draws) do
    { "a" => 0.5440336922480249, "b" => 0.5, "c" => 0.6316376362100744, "d" => 0.6101816494690415 }
  end

  def chosen(explore)
    {
      explore: explore,
      priority: { "a" => 0.6, "b" => 0.2, "c" => 0.9, "d" => 0.4 },
      draw: { "a" => 0.99, "b" => 0.5, "c" => 0.1, "d" => 0.7 }
    }
  end

  let(:split) { { shuffle: 0.25, explore: 0.25 } }

  it "LO2 builds keyed uniforms as the randomness helpers state them (the reference this file uses)" do
    expect((Digest::SHA256.digest("t").byteslice(0, 8).unpack1("Q>") >> 11) / (2.0**53)).to eq(0.8895498694938413)
    expect(keyed("t", "explore")).to eq(0.856909028643178)
    expect(keyed("t", "priority", "a")).to eq(0.7111955205766066)
  end

  describe "LO2 each branch on chosen uniforms" do
    it "LO2 shuffles below shuffle: the contender of smallest priority leads, the rest follow by priority" do
      expect(answer(four.rank(uniforms: chosen(0.1), **split))).to eq(items: %w[b d a c], leader: "b", branch: :shuffled)
      expect(answer(four.rank(uniforms: chosen(0.1), **split, may_lead: %w[a c])))
        .to eq(items: %w[a b d c], leader: "a", branch: :shuffled)
    end

    it "LO2 explores below shuffle + explore: the eligible item of smallest priority leads, the rest follow by draw" do
      explored = four.rank(uniforms: chosen(0.3), **split)

      expect(answer(explored)).to eq(items: %w[b c d a], leader: "b", branch: :explored)
      expect_draws(explored.draws, four_draws, 1e-9, "explored")
      expect(answer(four.rank(uniforms: chosen(0.3), **split, min_views: 20.0)))
        .to eq(items: %w[a c d b], leader: "a", branch: :explored)
    end

    it "LO2 an item barred from leading is never the explored head, even with the smallest priority" do
      expect(answer(four.rank(uniforms: chosen(0.3), **split, may_lead: %w[a c d])))
        .to eq(items: %w[d c a b], leader: "d", branch: :explored)
      expect(answer(four.rank(uniforms: chosen(0.3), **split, may_lead: %w[b d], min_views: 5.0)))
        .to eq(items: %w[b c d a], leader: "b", branch: :explored)
    end

    it "LO2 falls to Thompson when the explore branch has no eligible item" do
      expect(answer(four.rank(uniforms: chosen(0.3), **split, may_lead: %w[d], min_views: 5.0)))
        .to eq(items: %w[d c a b], leader: nil, branch: :thompson)
    end

    it "LO2 takes the contender of largest draw at or above shuffle + explore, the rest by draw" do
      thompson = four.rank(uniforms: chosen(0.7), **split)

      expect(answer(thompson)).to eq(items: %w[c d a b], leader: "c", branch: :thompson)
      expect_draws(thompson.draws, four_draws, 1e-9, "thompson")
      expect(answer(four.rank(uniforms: chosen(0.7), **split, may_lead: %w[a b])))
        .to eq(items: %w[a c d b], leader: "a", branch: :thompson)
    end

    it "LO2 lets every item lead when every item is barred from leading" do
      expect(answer(four.rank(uniforms: chosen(0.7), **split, may_lead: [])))
        .to eq(items: %w[c d a b], leader: "c", branch: :thompson)
    end

    it "LO2 min_views limits only the exploration and the leader, never the Thompson or shuffled head" do
      expect(answer(four.rank(uniforms: chosen(0.7), **split, may_lead: %w[a b], min_views: 50.0)))
        .to eq(items: %w[a c d b], leader: nil, branch: :thompson)
      expect(answer(four.rank(uniforms: chosen(0.7), **split, min_views: 60.0)))
        .to eq(items: %w[c d a b], leader: nil, branch: :thompson)
      expect(answer(four.rank(uniforms: chosen(0.1), **split, min_views: 20.0)))
        .to eq(items: %w[b d a c], leader: nil, branch: :shuffled)
    end

    it "LO2 reads a min_views below 0 as none, as 0 reads it: every contender may be explored and every head leads" do
      # d has no views at all, so at 0 and below it is eligible and leads, and above 0 it is not.
      [ 0.1, 0.3, 0.7 ].each do |value|
        none = four.rank(uniforms: chosen(value), **split, may_lead: %w[d], min_views: 0.0)

        expect(four.rank(uniforms: chosen(value), **split, may_lead: %w[d], min_views: -1.0)).to eq(none), value.to_s
        expect(four.rank(uniforms: chosen(value), **split, may_lead: %w[d], min_views: -1e100)).to eq(none), value.to_s
        expect(none.leader).to eq("d"), value.to_s
      end
      expect(four.rank(uniforms: chosen(0.3), **split, may_lead: %w[d], min_views: 0.0).branch).to eq(:explored)
      expect(answer(four.rank(uniforms: chosen(0.3), **split, may_lead: %w[d], min_views: 1e-9)))
        .to eq(items: %w[d c a b], leader: nil, branch: :thompson)
    end

    it "LO2 compares the branch value strictly: at shuffle it explores, at shuffle + explore it takes Thompson" do
      expect(four.rank(uniforms: chosen(0.25), **split).branch).to eq(:explored)
      expect(four.rank(uniforms: chosen(0.5), **split).branch).to eq(:thompson)
      expect(four.rank(uniforms: chosen(0.0)).branch).to eq(:thompson)
      expect(four.rank(uniforms: chosen(0.999), shuffle: 1.0).branch).to eq(:shuffled)
      expect(four.rank(uniforms: chosen(0.999), explore: 1.0).branch).to eq(:explored)
    end

    it "LO2 accepts shuffle + explore of exactly 1" do
      expect(four.rank(uniforms: chosen(0.9), shuffle: 0.75, explore: 0.25).branch).to eq(:explored)
    end
  end

  describe "LO2 draws are posterior values under every source" do
    it "LO2 answers a Normal draw as mean + sd * the normal quantile, below 0 when it falls there" do
      normal = posteriors.normal({ "n1" => [ 4.0, 1.0 ], "n2" => [ 5.0, 2.0 ], "n3" => [ 3.0, 0.5 ], "n4" => [ 0.5, 2.0 ] },
                                 views: { "n1" => 9.0, "n2" => 9.0, "n3" => 9.0, "n4" => 9.0 })
      uniforms = { explore: 0.7, priority: { "n1" => 0.1, "n2" => 0.2, "n3" => 0.3, "n4" => 0.4 },
                   draw: { "n1" => 0.9, "n2" => 0.2, "n3" => 0.6, "n4" => 0.05 } }
      order = normal.rank(uniforms: uniforms)

      expect(answer(order)).to eq(items: %w[n1 n2 n3 n4], leader: "n1", branch: :thompson)
      expect_draws(order.draws, { "n1" => 5.281551565544601, "n2" => 3.316757532854172, "n3" => 3.1266735515679,
                                  "n4" => -2.789707253902945 }, 1e-9, "normal")
      by_rule = { "n1" => 4.0 + (1.0 * significance.normal_quantile(0.9)),
                  "n2" => 5.0 + (2.0 * significance.normal_quantile(0.2)),
                  "n3" => 3.0 + (0.5 * significance.normal_quantile(0.6)),
                  "n4" => 0.5 + (2.0 * significance.normal_quantile(0.05)) }
      expect_draws(order.draws, by_rule, 1e-12, "normal by rule")
    end

    it "LO2 keeps every Beta draw in [0, 1] under every source, the extreme parameters included" do
      pairs = { "p" => [ 1.0, 0.05 ], "q" => [ 0.05, 1.0 ], "r" => [ 400.0, 600.0 ], "s" => [ 2.0, 2.0 ] }
      set = posteriors.beta(pairs, views: pairs.keys.to_h { |item| [ item, 5.0 ] })
      orders = (0...200).flat_map do |i|
        [ set.rank(key: [ "range", "k#{i}" ], explore: 0.5), set.rank(rng: Random.new(i), explore: 0.5) ]
      end
      orders << set.rank(uniforms: { explore: 0.9, priority: pairs.transform_values { 0.5 },
                                     draw: { "p" => 0.0, "q" => 0.999999, "r" => 0.5, "s" => 0.95 } })
      values = orders.flat_map { |order| order.draws.values }

      expect(values.size).to eq((400 * 4) + 4)
      expect(values).to all(be_a(Float))
      expect(values).to all(be_between(0.0, 1.0))
    end

    it "LO2 answers the keyed draws as the Beta quantiles at the keyed draw uniforms" do
      [ [ "visit", "k1" ], "visit-7", [ "visit", "k2", "x" ] ].each do |key|
        order = four.rank(key: key, explore: 0.1)

        expect_draws(order.draws, keyed_beta_draws(four, key), 1e-9, key.inspect)
      end
    end
  end

  describe "LO2 ties and comparison" do
    it "LO2 breaks a tie of draws to the earlier item, whichever the input order" do
      pairs = { "x" => [ 3.0, 3.0 ], "y" => [ 3.0, 3.0 ], "z" => [ 3.0, 3.0 ] }
      uniforms = { explore: 0.9, priority: { "x" => 0.5, "y" => 0.5, "z" => 0.5 }, draw: { "x" => 0.4, "y" => 0.4, "z" => 0.3 } }
      forward = posteriors.beta(pairs, views: pairs.transform_values { 1.0 })
      backward = posteriors.beta(pairs.slice("y", "x", "z"), views: pairs.transform_values { 1.0 })

      expect(forward.rank(uniforms: uniforms).items).to eq(%w[x y z])
      expect(backward.rank(uniforms: uniforms).items).to eq(%w[y x z])
      expect_draws(forward.rank(uniforms: uniforms).draws,
                   { "x" => 0.44625409748647316, "y" => 0.44625409748647316, "z" => 0.38981835053095837 }, 1e-9, "ties")
    end

    it "LO2 breaks a tie of priorities to the earlier item on the shuffled and explored branches" do
      pairs = { "x" => [ 3.0, 3.0 ], "y" => [ 3.0, 3.0 ], "z" => [ 3.0, 3.0 ] }
      uniforms = { explore: 0.1, priority: { "x" => 0.5, "y" => 0.5, "z" => 0.5 }, draw: { "x" => 0.3, "y" => 0.4, "z" => 0.6 } }
      forward = posteriors.beta(pairs, views: pairs.transform_values { 1.0 })
      backward = posteriors.beta(pairs.slice("z", "y", "x"), views: pairs.transform_values { 1.0 })

      expect(answer(forward.rank(uniforms: uniforms, shuffle: 0.5))).to eq(items: %w[x y z], leader: "x", branch: :shuffled)
      expect(answer(backward.rank(uniforms: uniforms, shuffle: 0.5))).to eq(items: %w[z y x], leader: "z", branch: :shuffled)
      expect(answer(forward.rank(uniforms: uniforms, explore: 0.5))).to eq(items: %w[x z y], leader: "x", branch: :explored)
      expect(answer(backward.rank(uniforms: uniforms, explore: 0.5))).to eq(items: %w[z y x], leader: "z", branch: :explored)
    end

    it "LO2 breaks a tie of Normal draws to the earlier item" do
      pairs = { "p" => [ 2.0, 1.0 ], "q" => [ 2.0, 1.0 ] }
      uniforms = { explore: 0.9, priority: { "p" => 0.5, "q" => 0.5 }, draw: { "p" => 0.3, "q" => 0.3 } }

      expect(posteriors.normal(pairs, views: pairs.transform_values { 1.0 }).rank(uniforms: uniforms).items).to eq(%w[p q])
      expect(posteriors.normal(pairs.slice("q", "p"), views: pairs.transform_values { 1.0 }).rank(uniforms: uniforms).items)
        .to eq(%w[q p])
    end

    it "LO2 compares Beta draws by position, so two draws that both round to 1.0 do not tie" do
      # Beta(1, 0.05): 1 - x = (1 - u)**20, so 1e-20 at 0.9 and 9.5e-27 at 0.95; both x read 1.0 as Floats.
      pairs = { "x" => [ 1.0, 0.05 ], "y" => [ 1.0, 0.05 ] }
      set = posteriors.beta(pairs, views: pairs.transform_values { 1.0 })
      order = set.rank(uniforms: { explore: 0.9, priority: { "x" => 0.5, "y" => 0.5 }, draw: { "x" => 0.9, "y" => 0.95 } })

      expect(answer(order)).to eq(items: %w[y x], leader: "y", branch: :thompson)
      expect(order.draws).to eq("x" => 1.0, "y" => 1.0)
    end
  end

  describe "LO2 pins and top" do
    it "LO2 keeps a pinned item at its position, drawing nothing for it, and leads with the free head" do
      thompson = four.rank(uniforms: chosen(0.7), **split, pinned: { 0 => "b" })

      expect(answer(thompson)).to eq(items: %w[b c d a], leader: "c", branch: :thompson)
      expect_draws(thompson.draws, four_draws.except("b"), 1e-9, "pinned b")
      expect(answer(four.rank(uniforms: chosen(0.1), **split, pinned: { 1 => "c" })))
        .to eq(items: %w[b c d a], leader: "b", branch: :shuffled)
    end

    it "LO2 fills the free slots in order around several pins" do
      explored = four.rank(uniforms: chosen(0.3), **split, pinned: { 0 => "d", 3 => "b" })

      expect(answer(explored)).to eq(items: %w[d a c b], leader: "a", branch: :explored)
      expect_draws(explored.draws, four_draws.slice("a", "c"), 1e-9, "two pins")
    end

    it "LO2 counts only free items as contenders" do
      expect(answer(four.rank(uniforms: chosen(0.7), **split, pinned: { 0 => "c" }, may_lead: %w[c b])))
        .to eq(items: %w[c b d a], leader: "b", branch: :thompson)
    end

    it "LO2 answers the first top items of the whole order and keeps its leader" do
      expect(answer(four.rank(uniforms: chosen(0.7), **split, top: 2))).to eq(items: %w[c d], leader: "c", branch: :thompson)
      expect(four.rank(uniforms: chosen(0.7), top: 1).items).to eq(%w[c])
      expect(four.rank(uniforms: chosen(0.7), top: 4).items).to eq(%w[c d a b])
      expect(four.rank(uniforms: chosen(0.7), top: 9).items).to eq(%w[c d a b])
      expect(answer(four.rank(uniforms: chosen(0.7), **split, pinned: { 3 => "c" }, top: 2)))
        .to eq(items: %w[d a], leader: "d", branch: :thompson)
      expect(four.rank(uniforms: chosen(0.7), top: 2).draws.keys.sort).to eq(%w[a b c d])
    end
  end

  describe "LO2 the key source" do
    it "LO2 draws a key's uniforms as the helpers state, for a String key and an Array of parts alike" do
      [ "visit-7", [ "visit", "k3" ], [ "visit", "k4", "x" ] ].each do |key|
        [ split, split.merge(may_lead: %w[a b c], min_views: 5.0), split.merge(pinned: { 2 => "a" }, top: 3) ].each do |rule|
          expect(four.rank(key: key, **rule)).to eq(four.rank(uniforms: keyed_uniforms(key, four_pairs.keys), **rule)),
                                                 "#{key.inspect} #{rule.inspect}"
        end
      end
    end

    let(:six_pairs) do
      { "a" => [ 3.0, 12.0 ], "b" => [ 4.0, 12.0 ], "c" => [ 5.0, 13.0 ], "d" => [ 2.0, 5.0 ], "e" => [ 6.0, 18.0 ],
        "f" => [ 9.0, 30.0 ] }
    end
    let(:six_views) { six_pairs.keys.to_h { |item| [ item, 50.0 ] } }
    let(:keys) { (0...1000).map { |i| [ "visit", "k#{i}" ] } }

    it "LO2 an added item never changes the others' relative order over 1 000 keys at shuffle 0.1" do
      before = posteriors.beta(six_pairs, views: six_views)
      after = posteriors.beta(six_pairs.merge("g" => [ 3.0, 9.0 ]), views: six_views.merge("g" => 0.0))
      normal_pairs = { "a" => [ 4.0, 1.0 ], "b" => [ 4.5, 1.5 ], "c" => [ 3.5, 0.8 ], "d" => [ 5.0, 2.5 ],
                       "e" => [ 4.2, 0.6 ], "f" => [ 3.9, 1.1 ] }
      normal_before = posteriors.normal(normal_pairs, views: six_views)
      normal_after = posteriors.normal(normal_pairs.merge("g" => [ 4.4, 1.2 ]), views: six_views.merge("g" => 0.0))
      branches = Hash.new(0)
      moved = []

      keys.each do |key|
        [ [ before, after ], [ normal_before, normal_after ] ].each do |small, large|
          old = small.rank(key: key, shuffle: 0.1)
          new = large.rank(key: key, shuffle: 0.1)
          branches[old.branch] += 1 if small.equal?(before)
          moved << [ key, old.items, new.items ] unless new.items - [ "g" ] == old.items && new.branch == old.branch
        end
      end

      expect(moved).to be_empty, "orders moved: #{moved.first(3).inspect}"
      # By hand: 85 of these keys have an explore uniform below 0.1.
      expect(branches).to eq(shuffled: 85, thompson: 915)
    end

    # The rule's contenders are the free items may_lead names, so an item it does not name takes no part in choosing
    # the head or the branch, and the rest follow by their own keyed draws (or priorities), which no other item moves.
    # Here the added item has the most views and the best posterior, so it would lead most keys were it named.
    it "LO2 an added item absent from may_lead never changes the others' order, the leader or the branch over " \
       "1 000 keys, with explore and min_views set" do
      views = { "a" => 50.0, "b" => 10.0, "c" => 50.0, "d" => 5.0, "e" => 50.0, "f" => 30.0 }
      normal_pairs = { "a" => [ 4.0, 1.0 ], "b" => [ 4.5, 1.5 ], "c" => [ 3.5, 0.8 ], "d" => [ 5.0, 2.5 ],
                       "e" => [ 4.2, 0.6 ], "f" => [ 3.9, 1.1 ] }
      sets = [
        [ posteriors.beta(six_pairs, views: views), posteriors.beta(six_pairs.merge("g" => [ 40.0, 10.0 ]),
                                                                    views: views.merge("g" => 500.0)) ],
        [ posteriors.normal(normal_pairs, views: views), posteriors.normal(normal_pairs.merge("g" => [ 9.0, 0.5 ]),
                                                                          views: views.merge("g" => 500.0)) ]
      ]
      rules = [ { explore: 0.2, may_lead: %w[a b c d e], min_views: 20.0 },
                { shuffle: 0.1, explore: 0.2, may_lead: %w[a b d], min_views: 20.0 } ]
      branches = Hash.new(0)
      second = 0
      moved = []

      keys.each do |key|
        sets.each do |small, large|
          rules.each do |rule|
            old = small.rank(key: key, **rule)
            new = large.rank(key: key, **rule)
            branches[old.branch] += 1
            second += 1 if new.items[1] == "g"
            next if new.items - [ "g" ] == old.items && new.leader == old.leader && new.branch == old.branch

            moved << [ key, rule, answer(old), answer(new) ]
          end
        end
      end

      expect(moved).to be_empty, "orders moved: #{moved.first(3).inspect}"
      # Every branch ran, and the added item was often the first item after the head, where it would have led.
      expect(branches.keys).to contain_exactly(:shuffled, :explored, :thompson)
      expect(branches.values).to all(be > 100)
      expect(second).to be > 1_000
    end

    it "LO2 gives the same orders in a fresh process" do
      set = posteriors.beta(six_pairs, views: six_views)
      here = keys.map { |key| order = set.rank(key: key, shuffle: 0.1); [ order.items, order.branch.to_s ] }
      # The child loads what this process has loaded from the library, wired or not.
      lib = File.expand_path("../../lib", __dir__)
      roots = [ lib, File.realpath(lib) ].uniq.map { |root| "#{root}/" }
      features = $LOADED_FEATURES.select { |path| roots.any? { |root| path.start_with?(root) } }
      script = <<~'RUBY'
        require "json"
        input = JSON.parse($stdin.read)
        $LOAD_PATH.unshift(input.fetch("lib"))
        require "skiftet_statistical"
        input.fetch("features").each { |path| require path }
        set = SkiftetStatistical::Bandit::Posteriors.beta(input.fetch("pairs"), views: input.fetch("views"))
        orders = input.fetch("keys").map do |key|
          order = set.rank(key: key, shuffle: 0.1)
          [ order.items, order.branch.to_s ]
        end
        $stdout.write(JSON.generate(orders))
      RUBY
      input = JSON.generate("lib" => lib, "features" => features, "pairs" => six_pairs, "views" => six_views, "keys" => keys)
      output, errors, status = Open3.capture3(RbConfig.ruby, "-e", script, stdin_data: input)

      expect(status).to be_success, errors
      expect(JSON.parse(output)).to eq(here)
    end

    it "LO2 on the explored branch an added item of smaller priority takes the head, the old head back at its draw rank" do
      before = posteriors.beta(six_pairs, views: six_views)
      after = posteriors.beta(six_pairs.merge("g" => [ 3.0, 9.0 ]), views: six_views.merge("g" => 0.0))
      explored = 0
      taken = 0

      keys.each do |key|
        old = before.rank(key: key, explore: 0.1)
        new = after.rank(key: key, explore: 0.1)
        next unless old.branch == :explored

        explored += 1
        expect(new.branch).to eq(:explored)
        uniforms = keyed_uniforms(key, after.items)
        expect(answer(new)).to eq(rule_order(items: after.items, views: after.views, rule: { explore: 0.1 }, branch_value: uniforms[:explore],
                                             priorities: uniforms[:priority], draws: keyed_beta_draws(after, key))), key.inspect
        if uniforms[:priority]["g"] < uniforms[:priority][old.items.first]
          taken += 1
          by_draw = keyed_beta_draws(before, key).sort_by { |item, draw| [ -draw, before.items.index(item) ] }.map(&:first)
          expect(new.items).to eq([ "g", *by_draw ]), key.inspect
        else
          expect(new.items.first).to eq(old.items.first)
          expect(new.items - [ "g" ]).to eq(old.items)
        end
      end

      # By hand: 85 keys explore, and on 9 of them the added item's priority is below every other item's.
      expect([ explored, taken ]).to eq([ 85, 9 ])
    end
  end

  describe "LO2 the rng source" do
    def expect_consumption(order, rng, seed, free, pairs, **options)
      expected = replay(seed, free, pairs, **options)
      expect(rng.calls.size).to eq(calls_in(seed, free, pairs, **options)), "rand calls"
      expect(rng.rand).to eq(expected[:next]), "the stream's next value after the rule"
      expect_draws(order.draws, expected[:draws], 1e-12, "replayed draws") unless expected[:draws].empty?
      expected
    end

    it "LO2 on the Thompson branch takes one branch value, then one Sampler draw per free item in input order" do
      expect(Random.new(3).rand).to be >= 0.5
      rng = recording(3)
      order = four.rank(rng: rng, **split)

      expect(answer(order)).to eq(items: %w[c b d a], leader: "c", branch: :thompson)
      expect_consumption(order, rng, 3, %w[a b c d], four_pairs, branch_value: true, priorities: false, draws: true)
    end

    it "LO2 on the explored branch takes the branch value, a priority per free item, then a draw per free item" do
      expect(Random.new(1).rand).to be_between(0.25, 0.5).exclusive
      rng = recording(1)
      order = four.rank(rng: rng, **split)

      expect(answer(order)).to eq(items: %w[b c a d], leader: "b", branch: :explored)
      expect_consumption(order, rng, 1, %w[a b c d], four_pairs, branch_value: true, priorities: true, draws: true)
    end

    it "LO2 takes a priority for every free item on the explored branch, eligible or not" do
      rng = recording(1)
      order = four.rank(rng: rng, **split, min_views: 20.0)

      # Replayed: priorities a 0.720, b 0.000114, c 0.302, d 0.147; only a and c have 20 views, so c leads; draws
      # b 0.685, a 0.399, d 0.323.
      expect(answer(order)).to eq(items: %w[c b a d], leader: "c", branch: :explored)
      expect_consumption(order, rng, 1, %w[a b c d], four_pairs, branch_value: true, priorities: true, draws: true)
    end

    it "LO2 on the shuffled branch takes the branch value and a priority per free item, and draws nothing" do
      expect(Random.new(5).rand).to be < 0.25
      rng = recording(5)
      order = four.rank(rng: rng, **split)

      expect(answer(order)).to eq(items: %w[b d a c], leader: "b", branch: :shuffled)
      expect_consumption(order, rng, 5, %w[a b c d], four_pairs, branch_value: true, priorities: true, draws: false)
    end

    it "LO2 takes no branch value when shuffle + explore is 0" do
      rng = recording(3)
      order = four.rank(rng: rng)

      expect(answer(order)).to eq(items: %w[c b d a], leader: "c", branch: :thompson)
      expect_consumption(order, rng, 3, %w[a b c d], four_pairs, branch_value: false, priorities: false, draws: true)
    end

    it "LO2 takes no priorities when the explore branch has no eligible item and falls to Thompson" do
      rng = recording(5)
      order = four.rank(rng: rng, explore: 1.0, min_views: 100.0)

      expect(answer(order)).to eq(items: %w[c b d a], leader: nil, branch: :thompson)
      expect_consumption(order, rng, 5, %w[a b c d], four_pairs, branch_value: true, priorities: false, draws: true)
    end

    it "LO2 takes nothing for a pinned item" do
      rng = recording(9)
      order = four.rank(rng: rng, explore: 1.0, pinned: { 0 => "b" })

      expect(answer(order)).to eq(items: %w[b d c a], leader: "d", branch: :explored)
      expect_consumption(order, rng, 9, %w[a c d], four_pairs, branch_value: true, priorities: true, draws: true)
      expect(order.draws).not_to have_key("b")
    end

    it "LO2 draws a Normal item as mean + sd * a Sampler Gaussian" do
      pairs = { "n1" => [ 4.0, 1.0 ], "n2" => [ 5.0, 2.0 ], "n3" => [ 3.0, 0.5 ] }
      normal = posteriors.normal(pairs, views: pairs.transform_values { 0.0 })
      rng = recording(21)
      order = normal.rank(rng: rng)

      expect(answer(order)).to eq(items: %w[n2 n3 n1], leader: "n2", branch: :thompson)
      expect_consumption(order, rng, 21, %w[n1 n2 n3], pairs, branch_value: false, priorities: false, draws: true, kind: :normal)
    end

    # 01 GB6: the rng source draws by Sampler on the pairs as given and takes any parameter above 0; only the quantile
    # draws and the exact odds stop at BetaArgmax::SMALLEST_PARAMETER. At 1e-300 a Gamma draw underflows to 0, so a, b,
    # c and d draw exactly 0.5, 1.0, 0.0 and 1.0 (the old policy's ties run at a prior of 1e-300).
    it "LO2 (LC1) draws from an RNG over parameters below BetaArgmax::SMALLEST_PARAMETER, 1e-300 included" do
      pairs = { "a" => [ 1e-300, 1e-300 ], "b" => [ 3.0, 1e-300 ], "c" => [ 1e-300, 5.0 ], "d" => [ 4.0, 1e-300 ],
                "e" => [ 1e-9, 1.0 ], "f" => [ 2.0, 3.0 ] }
      views = pairs.transform_values { 5.0 }
      set = posteriors.beta(pairs, views: views)
      set.method(:rank)
      (1..20).each do |seed|
        order = set.rank(rng: Random.new(seed))
        replayed = replay(seed, pairs.keys, pairs, branch_value: false, priorities: false, draws: true)[:draws]

        expect(replayed.values_at("a", "b", "c", "d")).to eq([ 0.5, 1.0, 0.0, 1.0 ])
        expect_draws(order.draws, replayed, 1e-12, "seed #{seed}")
        expect(answer(order)).to eq(rule_order(items: pairs.keys, views: views, rule: {}, branch_value: nil,
                                               priorities: {}, draws: replayed)), "seed #{seed}"
        expect(set.rank(rng: Random.new(seed), shuffle: 0.2, explore: 0.3).items).to match_array(pairs.keys)
      end
    end
  end

  describe "LO2 the bandit shapes in generic words" do
    it "LO2 a gated row: pooled priors, a new item barred from leading, min_views and explore over keys" do
      prior = SkiftetStatistical::Bandit::Prior
      groups = { "e1" => [ 2400.0, 150.0 ], "e2" => [ 1800.0, 140.0 ], "e3" => [ 900.0, 50.0 ], "e4" => [ 3100.0, 170.0 ],
                 "e5" => [ 600.0, 45.0 ], "n1" => [ 12.0, 2.0 ] }
      owns = { "e1" => [ 120.0, 9.0 ], "e2" => [ 45.0, 4.0 ], "e3" => [ 20.0, 3.0 ], "e4" => [ 300.0, 18.0 ],
               "e5" => [ 8.0, 0.0 ], "n1" => [ 5.0, 1.0 ] }
      tally = ->((views, outcomes)) { { views: views, outcomes: outcomes } }
      established = %w[e1 e2 e3 e4 e5]
      priors = established.to_h do |item|
        [ item, prior.pooled(group: tally.call(groups[item]), own: tally.call(owns[item]), strength: 20.0) ]
      end
      priors["n1"] = prior.pooled_new(others: established.map { |item| { group: tally.call(groups[item]), own: tally.call(owns[item]) } },
                                      group: tally.call(groups["n1"]), own: tally.call(owns["n1"]))
      set = posteriors.yes_no(owns.transform_values { |pair| tally.call(pair) }, priors: priors)
      rule = { may_lead: established, min_views: 30.0, explore: 0.1 }
      heads = Hash.new(0)
      explored_heads = Hash.new(0)

      (0...1000).each do |i|
        key = [ "row", "k#{i}" ]
        order = set.rank(key: key, **rule)
        heads[order.items.first] += 1
        explored_heads[order.items.first] += 1 if order.branch == :explored
        expect(order.leader).to eq(%w[e3 e5].include?(order.items.first) ? nil : order.items.first)
        next unless i < 300

        uniforms = keyed_uniforms(key, set.items)
        expect(answer(order)).to eq(rule_order(items: set.items, views: set.views, rule: rule, branch_value: uniforms[:explore],
                                               priorities: uniforms[:priority], draws: keyed_beta_draws(set, key))), key.inspect
      end

      expect(heads).not_to have_key("n1")
      expect(explored_heads.keys - %w[e1 e2 e4]).to be_empty
      # By hand: 104 of these keys have an explore uniform below 0.1, and each has an eligible item to explore.
      expect(explored_heads.values.sum).to eq(104)
    end

    it "LO2 a list with a random head: explore 0.1 over every item, no minimum, the rest by draw, from the visit's RNG" do
      tallies = { "k1" => [ 0.0, 0.0 ], "k2" => [ 12.0, 3.0 ], "k3" => [ 40.0, 2.0 ], "k4" => [ 7.0, 1.0 ],
                  "k5" => [ 90.0, 12.0 ], "k6" => [ 3.0, 0.0 ], "k7" => [ 25.0, 4.0 ], "k8" => [ 60.0, 5.0 ] }
      set = posteriors.yes_no(tallies.transform_values { |(views, outcomes)| { views: views, outcomes: outcomes } },
                              priors: SkiftetStatistical::Bandit::Prior.flat)
      pairs = pairs_of(set)
      explored = Hash.new(0)

      (1..4000).each do |seed|
        rng = recording(seed)
        order = set.rank(rng: rng, explore: 0.1)
        explored[order.items.first] += 1 if order.branch == :explored
        next unless seed <= 300

        branch_value = Random.new(seed).rand
        expected = replay(seed, set.items, pairs, branch_value: true, priorities: branch_value < 0.1, draws: true)
        expect(answer(order)).to eq(rule_order(items: set.items, views: set.views, rule: { explore: 0.1 }, branch_value: expected[:value],
                                               priorities: expected[:priority], draws: expected[:draws])), "seed #{seed}"
        expect_draws(order.draws, expected[:draws], 1e-12, "seed #{seed}")
        expect(rng.calls.size).to eq(calls_in(seed, set.items, pairs, branch_value: true, priorities: branch_value < 0.1, draws: true))
      end

      total = explored.values.sum
      expect(total).to be_within(4 * Math.sqrt(4000 * 0.1 * 0.9)).of(400)
      expect(explored.keys.sort).to eq(tallies.keys.sort)
      explored.each_value { |count| expect(count).to be_within(4 * Math.sqrt(total * (1.0 / 8) * (7.0 / 8))).of(total / 8.0) }
    end
  end

  describe "LO2 what an Order is and carries" do
    let(:order_class) { SkiftetStatistical::Bandit::Order }

    it "LO2 is a frozen keyword Struct of items, leader, branch and draws" do
      order = four.rank(uniforms: chosen(0.7), explore: 0.1)

      expect(order_class.ancestors).to include(Struct)
      expect(order_class.members).to eq(%i[items leader branch draws])
      expect(order_class.keyword_init?).to be(true)
      expect(order).to be_a(order_class)
      expect(order).to be_frozen
    end

    it "LO2 carries its posterior set, its rule and a cache in instance variables set before the freeze" do
      order = four.rank(uniforms: chosen(0.7), explore: 0.1)
      carried = order.instance_variables.map { |name| order.instance_variable_get(name) }

      expect(carried.size).to be >= 3
      expect(carried.any? { |value| value.equal?(four) }).to be(true)
    end

    it "LO2 compares only the four members" do
      order = four.rank(uniforms: chosen(0.7), explore: 0.1)
      same_members = posteriors.beta(four_pairs, views: four_views).rank(uniforms: chosen(0.7))

      expect(order).to eq(order_class.new(items: order.items, leader: order.leader, branch: order.branch, draws: order.draws))
      expect(same_members).to eq(order)
      expect(four.rank(uniforms: chosen(0.3), **split)).not_to eq(order)
    end
  end

  describe "LO2 any number of items" do
    let(:many_pairs) { (0...60).to_h { |i| [ format("i%02d", i), [ 1.0 + (i % 7), 3.0 + (i % 5) ] ] } }
    let(:many_views) { many_pairs.keys.each_with_index.to_h { |item, i| [ item, i.to_f ] } }
    let(:many) { posteriors.beta(many_pairs, views: many_views) }

    it "LO2 ranks 60 items under every source" do
      key = [ "visit", "many" ]
      keyed_order = many.rank(key: key, explore: 0.2, min_views: 10.0)
      uniforms = keyed_uniforms(key, many.items)
      expect(answer(keyed_order)).to eq(rule_order(items: many.items, views: many_views, rule: { explore: 0.2, min_views: 10.0 },
                                                   branch_value: uniforms[:explore], priorities: uniforms[:priority],
                                                   draws: keyed_beta_draws(many, key)))

      rng = recording(8)
      rng_order = many.rank(rng: rng)
      expected = replay(8, many.items, many_pairs, branch_value: false, priorities: false, draws: true)
      expect(answer(rng_order)).to eq(rule_order(items: many.items, views: many_views, rule: {}, branch_value: nil, priorities: {},
                                                 draws: expected[:draws]))
      expect(rng_order.draws.size).to eq(60)

      given = Random.new(12)
      chosen_uniforms = { explore: 0.95, priority: many.items.to_h { |item| [ item, given.rand ] },
                          draw: many.items.to_h { |item| [ item, given.rand ] } }
      draws = many_pairs.to_h { |item, (alpha, beta)| [ item, beta_function.quantile(chosen_uniforms[:draw][item], alpha, beta) ] }
      expect(answer(many.rank(uniforms: chosen_uniforms)))
        .to eq(rule_order(items: many.items, views: many_views, rule: {}, branch_value: 0.95, priorities: chosen_uniforms[:priority],
                          draws: draws))
      expect(keyed_order.items.sort).to eq(many_pairs.keys.sort)
    end

    it "LO2 answers no items for an empty set" do
      empty = posteriors.beta({})

      [ { key: [ "visit", "k1" ] }, { uniforms: { explore: 0.5, priority: {}, draw: {} } }, { rng: Random.new(1) } ].each do |source|
        order = empty.rank(**source, explore: 0.1)

        expect(order.items).to eq([]), source.keys.inspect
        expect(order.leader).to be_nil
      end
    end
  end

  describe "LO2 refusals, each before any draw" do
    # The set and its method are read first, so a missing one fails as itself, outside the matcher.
    def refusing
      four.method(:rank)
      four
    end

    def expect_refused(**arguments)
      set = refusing
      rng = recording(4)
      expect { set.rank(rng: rng, **arguments) }.to raise_error(ArgumentError), arguments.inspect
      expect(rng.calls).to be_empty, "#{arguments.inspect} drew before refusing"
    end

    it "LO2 refuses a pinned position outside the items" do
      expect_refused(pinned: { 4 => "a" })
      expect_refused(pinned: { -1 => "a" })
    end

    it "LO2 refuses a pinned name that is not an item" do
      expect_refused(pinned: { 0 => "z" })
    end

    it "LO2 refuses one name pinned twice" do
      expect_refused(pinned: { 0 => "a", 2 => "a" })
    end

    it "LO2 refuses a may_lead name that is not an item" do
      expect_refused(may_lead: %w[a z])
    end

    it "LO2 refuses top below 1" do
      expect_refused(top: 0)
      expect_refused(top: -1)
    end

    it "LO2 refuses shuffle or explore outside [0, 1], or summing above 1" do
      expect_refused(shuffle: -0.1)
      expect_refused(shuffle: 1.1)
      expect_refused(explore: -0.5)
      expect_refused(explore: 1.5)
      expect_refused(shuffle: 0.6, explore: 0.5)
    end

    # 01 GB6: a key or given uniforms draw by quantile, which refuses a parameter below BetaArgmax::SMALLEST_PARAMETER
    # (1e-6, alpha or beta) whatever branch the uniforms choose, before any draw: the item drawn first, "a", never
    # reaches BetaArgmax.position, and the shuffled branch, which draws nothing, refuses too. A parameter at
    # SMALLEST_PARAMETER is taken.
    it "LO2 (LC1) refuses under a key or given uniforms a parameter below BetaArgmax::SMALLEST_PARAMETER, before any draw" do
      sets = [ { "a" => [ 2.0, 3.0 ], "b" => [ 1e-9, 1.0 ] }, { "a" => [ 2.0, 3.0 ], "b" => [ 1.0, 5e-7 ] } ].map do |pairs|
        posteriors.beta(pairs, views: pairs.transform_values { 5.0 })
      end
      sets.first.method(:rank)
      edge = posteriors.beta({ "a" => [ 2.0, 3.0 ], "b" => [ 1e-6, 1.0 ], "c" => [ 1.0, 1e-6 ] })
      expect(edge.rank(key: "k").items).to match_array(%w[a b c])
      thompson = { explore: 0.9, priority: { "a" => 0.3, "b" => 0.6 }, draw: { "a" => 0.4, "b" => 0.5 } }
      shuffled = thompson.merge(explore: 0.1)

      expect(SkiftetStatistical::BetaArgmax).not_to receive(:position)
      sets.each do |set|
        label = set.parameters.inspect
        expect { set.rank(key: "k") }.to raise_error(ArgumentError), label
        expect { set.rank(key: [ "k", 2 ], shuffle: 0.5, explore: 0.5) }.to raise_error(ArgumentError), label
        expect { set.rank(uniforms: thompson) }.to raise_error(ArgumentError), label
        expect { set.rank(uniforms: shuffled, shuffle: 0.5) }.to raise_error(ArgumentError), label
      end
    end

    it "LO2 refuses no source and two sources" do
      set = refusing
      rng = recording(4)
      uniforms = keyed_uniforms("k", four_pairs.keys)

      expect { set.rank }.to raise_error(ArgumentError)
      expect { set.rank(explore: 0.1) }.to raise_error(ArgumentError)
      expect { set.rank(key: "k", uniforms: uniforms) }.to raise_error(ArgumentError)
      expect { set.rank(key: "k", rng: rng) }.to raise_error(ArgumentError)
      expect { set.rank(uniforms: uniforms, rng: rng) }.to raise_error(ArgumentError)
      expect { set.rank(key: "k", uniforms: uniforms, rng: rng) }.to raise_error(ArgumentError)
      expect(rng.calls).to be_empty
    end
  end

  # 01 D20 as ruled at the L3 barrier's second round (2026-11-01): the set reads a list of item names in one place,
  # once, and words a name that is not an item in one place, so a list that answers differently on a second pass is
  # never read twice, and a stranger is an ArgumentError wherever it is named.
  describe "the L3 barrier, round 2: one reading of item names" do
    # A list that yields `first` on its first pass and `later` on every pass after, counting its passes.
    def shifting(first, later)
      Class.new do
        include Enumerable

        attr_reader :passes

        define_method(:initialize) { @passes = 0 }
        define_method(:each) do |&block|
          @passes += 1
          (@passes == 1 ? first : later).each(&block)
          self
        end
      end.new
    end

    it "the L3 barrier, round 2: reads a may_lead or among list once, so a stranger is always an ArgumentError" do
      calls = {
        "rank" => ->(list) { four.rank(uniforms: chosen(0.7), explore: 0.5, may_lead: list) },
        "pick" => ->(list) { four.pick(uniforms: chosen(0.7), explore: 0.5, may_lead: list) },
        "pick_probabilities" => ->(list) { four.pick_probabilities(explore: 0.5, may_lead: list) },
        "greedy" => ->(list) { four.greedy(may_lead: list) }
      }
      calls.each do |name, call|
        later_stranger = shifting(%w[d a], %w[d z])
        expect(call.call(later_stranger)).to eq(call.call(%w[d a])), name
        expect(later_stranger.passes).to eq(1), name

        first_stranger = shifting(%w[d z], %w[d a])
        expect { call.call(first_stranger) }.to raise_error(ArgumentError, 'may_lead: "z" is not an item'), name
        expect(first_stranger.passes).to eq(1), name
      end

      later_stranger = shifting(%w[d a], %w[d z])
      expect(four.summary(among: later_stranger)).to eq(four.summary(among: %w[d a]))
      expect(later_stranger.passes).to eq(1)
      first_stranger = shifting(%w[d z], %w[d a])
      expect { four.summary(among: first_stranger) }.to raise_error(ArgumentError, 'among: "z" is not an item')
      expect(first_stranger.passes).to eq(1)
    end

    it "the L3 barrier, round 2: words a name that is not an item alike wherever it is named" do
      expect { four.rank(uniforms: chosen(0.7), pinned: { 1 => "z" }) }
        .to raise_error(ArgumentError, 'pinned: "z" is not an item')
      expect { four.rank(uniforms: chosen(0.7), may_lead: %w[a z]) }
        .to raise_error(ArgumentError, 'may_lead: "z" is not an item')
      expect { four.summary(among: %w[z]) }.to raise_error(ArgumentError, 'among: "z" is not an item')
      expect { posteriors.beta(four_pairs, views: { "a" => 1.0, "z" => 1.0 }) }
        .to raise_error(ArgumentError, 'views: "z" is not an item')
      expect { posteriors.yes_no({ "a" => {} }, priors: { "a" => [ 1.0, 1.0 ], "z" => [ 1.0, 1.0 ] }) }
        .to raise_error(ArgumentError, 'priors: "z" is not an item')
      expect { four.rank(uniforms: chosen(0.7), pinned: { 0 => "a", 2 => +"a" }) }
        .to raise_error(ArgumentError, 'pinned: "a" is pinned twice')

      order = four.rank(uniforms: chosen(0.7), pinned: { 1 => +"a", 3 => +"b" })
      expect(order.items.values_at(1, 3)).to eq(%w[a b])
      expect(order.items[1]).to equal(four.items[0])
      expect(order.items[3]).to equal(four.items[1])
    end
  end

  describe "the L3 barrier, round 3: a list is read once, and only the library's own Array is kept" do
    # A list counting the passes made over it.
    def counted(names)
      Class.new do
        include Enumerable

        attr_reader :passes

        define_method(:initialize) { @passes = 0 }
        define_method(:each) do |&block|
          @passes += 1
          names.each(&block)
          self
        end
      end.new
    end

    def rule_of(answer) = answer.instance_variable_get(:@rule)

    it "the L3 barrier, round 3: reads a lazy list once, and the rule keeps a frozen Array of the set's own items" do
      { "pick" => ->(list) { four.pick(uniforms: chosen(0.7), explore: 0.5, may_lead: list) },
        "rank" => ->(list) { four.rank(uniforms: chosen(0.7), explore: 0.5, may_lead: list) } }.each do |name, call|
        source = counted(%w[d a])
        lazy = source.lazy
        answer = call.call(lazy)
        expect(answer).to eq(call.call(%w[d a])), name
        expect(source.passes).to eq(1), name

        kept = rule_of(answer)[:may_lead]
        expect(kept).to be_an_instance_of(Array), name
        expect(kept).to be_frozen, name
        expect(kept).to eq(%w[d a]), name
        expect(kept.map(&:object_id)).to eq(four.items.values_at(3, 0).map(&:object_id)), name
        expect(source.passes).to eq(1), name

        given = %w[d a]
        expect(rule_of(call.call(given))[:may_lead]).not_to equal(given), name
      end

      by_each = Struct.new(:names) do
        include Enumerable

        def each(&block) = names.each(&block)
      end.new(%w[d a])
      expect(by_each.to_a).to eq([ %w[d a] ])
      expect(rule_of(four.pick(uniforms: chosen(0.7), explore: 0.5, may_lead: by_each))[:may_lead]).to eq(%w[d a])

      source = counted(%w[d a])
      expect(four.greedy(may_lead: source.lazy)).to eq(four.greedy(may_lead: %w[d a]))
      expect(source.passes).to eq(1)
      source = counted(%w[d a])
      expect(four.summary(among: source.lazy)).to eq(four.summary(among: %w[d a]))
      expect(source.passes).to eq(1)
      source = counted(%w[d a])
      expect(four.pick_probabilities(explore: 0.5, may_lead: source.lazy))
        .to eq(four.pick_probabilities(explore: 0.5, may_lead: %w[d a]))
      expect(source.passes).to eq(1)
    end

    it "the L3 barrier, round 3: a name that answers no hash or no inspect is an ArgumentError, never a NoMethodError" do
      no_inspect = Object.new
      no_inspect.singleton_class.send(:undef_method, :inspect)
      no_hash = Object.new
      no_hash.singleton_class.send(:undef_method, :hash)
      [ BasicObject.new, no_inspect, no_hash ].each do |name|
        expect { four.rank(uniforms: chosen(0.7), may_lead: [ "a", name ]) }.to raise_error(ArgumentError, /may_lead: .* is not an item/)
        expect { four.pick(uniforms: chosen(0.7), may_lead: [ name ].lazy) }.to raise_error(ArgumentError, /may_lead: .* is not an item/)
        expect { four.rank(uniforms: chosen(0.7), pinned: { 0 => name }) }.to raise_error(ArgumentError, /pinned: .* is not an item/)
        expect { four.summary(among: [ name ]) }.to raise_error(ArgumentError, /among: .* is not an item/)
        expect { four.greedy(may_lead: [ name ]) }.to raise_error(ArgumentError, /may_lead: .* is not an item/)
      end
      expect { posteriors.beta(four_pairs, views: { no_inspect => 1.0 }) }.to raise_error(ArgumentError, /views: .* is not an item/)
    end
  end
end
