# frozen_string_literal: true

require "digest"

# LO2: the acceptance of Posteriors#pick, the head of the one order rule (rank with top: 1 and no pins), under each of
# its three sources. Written from the rule as stated before it was built; nothing here reads the code under test to
# make an expected value. The references:
# - with a key or uniforms and no gate, exclusion or shuffle, BetaArgmax.pick on the same uniforms. It compares the
#   public BetaArgmax.position, one rule with rank's, so it answers as 0.3.0's did except where two members' positions
#   lie within position's stated bound of each other (near-ties);
# - keyed uniforms: SHA256 of the parts joined by NUL, the first 8 bytes big-endian, >> 11, / 2**53, written here;
# - Beta draws for the larger checks: 0.2.0's BetaFunction.quantile, unchanged; RNG draws: a replay on a fresh Random
#   of the same seed through Sampler.
# The head, the item, the branch and the draws are compared with rank's under the same source and rule, since pick is
# rank's head; the draws exactly, the rest exactly.
RSpec.describe "SkiftetStatistical::Bandit::Posteriors#pick" do
  def posteriors = SkiftetStatistical::Bandit::Posteriors
  def argmax = SkiftetStatistical::BetaArgmax
  def beta_function = SkiftetStatistical::BetaFunction
  def prior = SkiftetStatistical::Bandit::Prior

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

  def pairs_of(set)
    parameters = set.parameters
    set.items.map { |item| parameters.is_a?(Hash) ? parameters.fetch(item) : parameters.fetch(set.items.index(item)) }
  end

  # 0.3.0's pick on the uniforms a key or a uniforms Hash gives, as an item.
  def argmax_pick(set, uniforms, epsilon)
    index = argmax.pick(pairs_of(set), explore: uniforms[:explore], priorities: set.items.map { |item| uniforms[:priority][item] },
                                       uniforms: set.items.map { |item| uniforms[:draw][item] }, epsilon: epsilon)
    set.items[index]
  end

  def first_largest(values) = values.each_with_index.max_by { |value, i| [ value, -i ] }.last
  def first_smallest(values) = values.each_with_index.min_by { |value, i| [ value, i ] }.last

  let(:four_pairs) { { "a" => [ 2.0, 8.0 ], "b" => [ 5.0, 5.0 ], "c" => [ 8.0, 2.0 ], "d" => [ 3.0, 3.0 ] } }
  let(:four_views) { { "a" => 40.0, "b" => 10.0, "c" => 50.0, "d" => 0.0 } }
  let(:four) { posteriors.beta(four_pairs, views: four_views) }
  let(:split) { { shuffle: 0.25, explore: 0.25 } }

  # By hand (rank_spec.rb holds the same fixture): the draws at these uniforms are a 0.5440336922480249, b 0.5,
  # c 0.6316376362100744, d 0.6101816494690415; the priorities b < d < a < c.
  def chosen(explore)
    {
      explore: explore,
      priority: { "a" => 0.6, "b" => 0.2, "c" => 0.9, "d" => 0.4 },
      draw: { "a" => 0.99, "b" => 0.5, "c" => 0.1, "d" => 0.7 }
    }
  end

  def answer(choice) = { item: choice.item, head: choice.head, branch: choice.branch }

  it "LO2 equals BetaArgmax.pick on the same keyed uniforms over 1 000 keys at explore 0.1" do
    pairs = { "a" => [ 3.0, 12.0 ], "b" => [ 4.0, 12.0 ], "c" => [ 5.0, 13.0 ], "d" => [ 2.0, 5.0 ], "e" => [ 6.0, 18.0 ] }
    set = posteriors.beta(pairs, views: pairs.transform_values { 50.0 })
    branches = Hash.new(0)
    differ = []

    (0...1000).each do |i|
      key = [ "visit", "k#{i}" ]
      choice = set.pick(key: key, explore: 0.1)
      uniforms = keyed_uniforms(key, set.items)
      expected = argmax_pick(set, uniforms, 0.1)
      branch = uniforms[:explore] < 0.1 ? :explored : :thompson
      branches[choice.branch] += 1
      differ << [ key, answer(choice), expected, branch ] unless answer(choice) == { item: expected, head: expected, branch: branch }
    end

    expect(differ).to be_empty, "picks differ: #{differ.first(3).inspect}"
    # By hand: 85 of these keys have an explore uniform below 0.1.
    expect(branches).to eq(explored: 85, thompson: 915)
  end

  it "LO2 a two-layer pick on given uniforms over a chain prior equals BetaArgmax.pick on those uniforms" do
    top = { views: 5000.0, outcomes: 260.0, strength: 40.0 }
    groups = { "g1" => { views: 1200.0, outcomes: 70.0, strength: 20.0 }, "g2" => { views: 900.0, outcomes: 40.0, strength: 20.0 } }
    leaves = { "m1" => [ "g1", 300.0, 21.0 ], "m2" => [ "g1", 150.0, 6.0 ], "m3" => [ "g2", 200.0, 9.0 ], "m4" => [ "g2", 40.0, 3.0 ] }
    priors = leaves.to_h do |item, (group, views, outcomes)|
      [ item, prior.chain(levels: [ top, groups[group], { views: views, outcomes: outcomes, strength: 10.0 } ], root_mean: 0.05) ]
    end
    set = posteriors.yes_no(leaves.transform_values { |(_, views, outcomes)| { views: views, outcomes: outcomes } }, priors: priors)
    given = Random.new(31)

    200.times do |i|
      uniforms = { explore: given.rand, priority: set.items.to_h { |item| [ item, given.rand ] },
                   draw: set.items.to_h { |item| [ item, given.rand ] } }
      choice = set.pick(uniforms: uniforms, explore: 0.2)
      expected = argmax_pick(set, uniforms, 0.2)

      expect(answer(choice)).to eq(item: expected, head: expected, branch: uniforms[:explore] < 0.2 ? :explored : :thompson), "draw #{i}"
    end
  end

  it "LO2 one pick: a flat prior, no exploration, a key read from the caller's stream" do
    tallies = { "101" => [ 140.0, 9.0 ], "205" => [ 80.0, 7.0 ], "317" => [ 0.0, 0.0 ], "422" => [ 33.0, 1.0 ],
                "530" => [ 210.0, 15.0 ], "648" => [ 12.0, 2.0 ] }
    set = posteriors.yes_no(tallies.transform_values { |(views, outcomes)| { views: views, outcomes: outcomes } }, priors: prior.flat)
    stream = Random.new(77)
    picked = Hash.new(0)

    300.times do
      value = stream.rand
      choice = set.pick(key: [ value ])
      expected = argmax_pick(set, keyed_uniforms([ value ], set.items), 0.0)
      picked[choice.item] += 1

      expect(answer(choice)).to eq(item: expected, head: expected, branch: :thompson), "key #{value}"
    end
    expect(picked.size).to be > 1
  end

  describe "LO2 pick is rank's head" do
    let(:rules) do
      [ {}, split, split.merge(may_lead: %w[a b], min_views: 20.0), { explore: 1.0, min_views: 45.0 },
        { shuffle: 1.0, may_lead: %w[c d] }, { explore: 0.5, may_lead: %w[d], min_views: 5.0 } ]
    end

    def expect_head(choice, order, label)
      expect(choice.head).to eq(order.items.first), "#{label}: head"
      expect(choice.item).to eq(order.leader), "#{label}: item"
      expect(choice.branch).to eq(order.branch), "#{label}: branch"
      expect(choice.draws).to eq(order.draws), "#{label}: draws"
    end

    it "LO2 under a key" do
      rules.each do |rule|
        (0...50).each do |i|
          key = [ "visit", "k#{i}" ]
          expect_head(four.pick(key: key, **rule), four.rank(key: key, **rule), "#{key.inspect} #{rule.inspect}")
        end
      end
    end

    it "LO2 under given uniforms" do
      rules.each do |rule|
        [ 0.1, 0.3, 0.7 ].each do |explore|
          expect_head(four.pick(uniforms: chosen(explore), **rule), four.rank(uniforms: chosen(explore), **rule),
                      "#{explore} #{rule.inspect}")
        end
      end
    end

    it "LO2 under an RNG, taking from the stream exactly what rank takes" do
      rules.each do |rule|
        (1..30).each do |seed|
          picking = recording(seed)
          ranking = recording(seed)
          choice = four.pick(rng: picking, **rule)
          order = four.rank(rng: ranking, **rule)

          expect_head(choice, order, "seed #{seed} #{rule.inspect}")
          expect(picking.calls).to eq(ranking.calls), "seed #{seed} #{rule.inspect}: what the stream gave"
          expect(picking.rand).to eq(ranking.rand)
        end
      end
    end

    it "LO2 on chosen uniforms, by hand" do
      expect(answer(four.pick(uniforms: chosen(0.7), **split))).to eq(item: "c", head: "c", branch: :thompson)
      expect(answer(four.pick(uniforms: chosen(0.3), **split, min_views: 20.0))).to eq(item: "a", head: "a", branch: :explored)
      expect(answer(four.pick(uniforms: chosen(0.1), **split))).to eq(item: "b", head: "b", branch: :shuffled)
      expect(answer(four.pick(uniforms: chosen(0.7), may_lead: %w[a b]))).to eq(item: "a", head: "a", branch: :thompson)
    end

    it "LO2 answers item nil when the head has fewer than min_views, and still the head" do
      expect(answer(four.pick(uniforms: chosen(0.7), **split, min_views: 60.0))).to eq(item: nil, head: "c", branch: :thompson)
      expect(answer(four.pick(uniforms: chosen(0.1), **split, min_views: 20.0))).to eq(item: nil, head: "b", branch: :shuffled)
      expect(answer(four.pick(uniforms: chosen(0.3), **split, may_lead: %w[d], min_views: 5.0)))
        .to eq(item: nil, head: "d", branch: :thompson)
    end
  end

  describe "LO2 what a Choice is and carries" do
    let(:choice_class) { SkiftetStatistical::Bandit::Choice }

    it "LO2 is a frozen keyword Struct of item, head, branch and draws" do
      choice = four.pick(uniforms: chosen(0.7), explore: 0.1)

      expect(choice_class.ancestors).to include(Struct)
      expect(choice_class.members).to eq(%i[item head branch draws])
      expect(choice_class.keyword_init?).to be(true)
      expect(choice).to be_a(choice_class)
      expect(choice).to be_frozen
    end

    it "LO2 carries its posterior set, its rule and a cache in instance variables set before the freeze" do
      choice = four.pick(uniforms: chosen(0.7), explore: 0.1)
      carried = choice.instance_variables.map { |name| choice.instance_variable_get(name) }

      expect(carried.size).to be >= 3
      expect(carried.any? { |value| value.equal?(four) }).to be(true)
    end

    it "LO2 compares only the four members" do
      choice = four.pick(uniforms: chosen(0.7), explore: 0.1)
      same_members = posteriors.beta(four_pairs, views: four_views).pick(uniforms: chosen(0.7))

      expect(choice).to eq(choice_class.new(item: choice.item, head: choice.head, branch: choice.branch, draws: choice.draws))
      expect(same_members).to eq(choice)
      expect(four.pick(uniforms: chosen(0.3), **split)).not_to eq(choice)
    end
  end

  describe "LO2 any number of items" do
    let(:many_pairs) { (0...60).to_h { |i| [ format("i%02d", i), [ 1.0 + (i % 7), 3.0 + (i % 5) ] ] } }
    let(:many_views) { many_pairs.keys.each_with_index.to_h { |item, i| [ item, i.to_f ] } }
    let(:many) { posteriors.beta(many_pairs, views: many_views) }

    it "LO2 picks among 60 items under every source" do
      key = [ "visit", "many" ]
      keyed_draws = many_pairs.map { |item, (alpha, beta)| beta_function.quantile(keyed(*key, "draw", item), alpha, beta) }
      expect(keyed(*key, "explore")).to be >= 0.1
      expect(many.pick(key: key, explore: 0.1).head).to eq(many.items[first_largest(keyed_draws)])

      rng = recording(8)
      replay = Random.new(8)
      sampler = SkiftetStatistical::Sampler.new(replay)
      replayed = many_pairs.values.map { |alpha, beta| sampler.beta(alpha, beta) }
      choice = many.pick(rng: rng)
      expect(choice.head).to eq(many.items[first_largest(replayed)])
      expect(choice.draws.size).to eq(60)

      given = Random.new(12)
      uniforms = { explore: 0.01, priority: many.items.to_h { |item| [ item, given.rand ] },
                   draw: many.items.to_h { |item| [ item, given.rand ] } }
      eligible = many.items.select { |item| many_views[item] >= 30.0 }
      explored = many.pick(uniforms: uniforms, explore: 0.05, min_views: 30.0)
      expect(answer(explored)).to eq(item: eligible[first_smallest(eligible.map { |item| uniforms[:priority][item] })],
                                     head: eligible[first_smallest(eligible.map { |item| uniforms[:priority][item] })],
                                     branch: :explored)
    end

    it "LO2 answers item nil, head nil and no draws for an empty set" do
      empty = posteriors.beta({})

      [ { key: [ "visit", "k1" ] }, { uniforms: { explore: 0.5, priority: {}, draw: {} } }, { rng: Random.new(1) } ].each do |source|
        choice = empty.pick(**source, explore: 0.1)

        expect(choice.item).to be_nil, source.keys.inspect
        expect(choice.head).to be_nil
        expect(choice.draws.to_h).to eq({})
      end
    end
  end

  describe "LO2 refusals, each before any draw" do
    # The set and its method are read first, so a missing one fails as itself, outside the matcher.
    def refusing
      four.method(:pick)
      four
    end

    def expect_refused(**arguments)
      set = refusing
      rng = recording(4)
      expect { set.pick(rng: rng, **arguments) }.to raise_error(ArgumentError), arguments.inspect
      expect(rng.calls).to be_empty, "#{arguments.inspect} drew before refusing"
    end

    it "LO2 refuses a may_lead name that is not an item" do
      expect_refused(may_lead: %w[a z])
    end

    it "LO2 refuses shuffle or explore outside [0, 1], or summing above 1" do
      expect_refused(shuffle: -0.1)
      expect_refused(explore: 1.5)
      expect_refused(shuffle: 0.6, explore: 0.5)
    end

    it "LO2 refuses no source and two sources" do
      set = refusing
      rng = recording(4)
      uniforms = keyed_uniforms("k", four_pairs.keys)

      expect { set.pick }.to raise_error(ArgumentError)
      expect { set.pick(key: "k", uniforms: uniforms) }.to raise_error(ArgumentError)
      expect { set.pick(key: "k", rng: rng) }.to raise_error(ArgumentError)
      expect { set.pick(uniforms: uniforms, rng: rng) }.to raise_error(ArgumentError)
      expect(rng.calls).to be_empty
    end

    # 01 GB6: a key or given uniforms draw by quantile, which refuses a parameter below BetaArgmax::SMALLEST_PARAMETER
    # (1e-6, alpha or beta) whatever branch the uniforms choose, before any draw: the item drawn first, "a", never
    # reaches BetaArgmax.position, and the shuffled branch, which draws nothing, refuses too.
    it "LO2 (LC1) refuses under a key or given uniforms a parameter below BetaArgmax::SMALLEST_PARAMETER, before any draw" do
      sets = [ { "a" => [ 2.0, 3.0 ], "b" => [ 1e-9, 1.0 ] }, { "a" => [ 2.0, 3.0 ], "b" => [ 1.0, 5e-7 ] } ].map do |pairs|
        posteriors.beta(pairs, views: pairs.transform_values { 5.0 })
      end
      sets.first.method(:pick)
      thompson = { explore: 0.9, priority: { "a" => 0.3, "b" => 0.6 }, draw: { "a" => 0.4, "b" => 0.5 } }
      shuffled = thompson.merge(explore: 0.1)

      expect(argmax).not_to receive(:position)
      sets.each do |set|
        label = set.parameters.inspect
        expect { set.pick(key: "k") }.to raise_error(ArgumentError), label
        expect { set.pick(key: [ "k", 2 ], shuffle: 0.5, explore: 0.5) }.to raise_error(ArgumentError), label
        expect { set.pick(uniforms: thompson) }.to raise_error(ArgumentError), label
        expect { set.pick(uniforms: shuffled, shuffle: 0.5) }.to raise_error(ArgumentError), label
      end
    end

    # The other side of GB6: a parameter at SMALLEST_PARAMETER is taken by the quantile draws, and the rng source draws
    # by Sampler, which takes any parameter above 0. At 1e-300 a Gamma draw underflows to 0, so "b" draws exactly 1.0
    # on every seed and heads every pick.
    it "LO2 (LC1) picks under a key at SMALLEST_PARAMETER, and from an RNG below it, 1e-300 included" do
      edge = posteriors.beta({ "a" => [ 2.0, 3.0 ], "b" => [ 1e-6, 1.0 ], "c" => [ 1.0, 1e-6 ] })
      expect(%w[a b c]).to include(edge.pick(key: "k").head)

      pairs = { "a" => [ 1e-300, 1e-300 ], "b" => [ 3.0, 1e-300 ], "c" => [ 1e-9, 1.0 ], "d" => [ 2.0, 3.0 ] }
      set = posteriors.beta(pairs, views: pairs.transform_values { 5.0 })
      (1..20).each do |seed|
        choice = set.pick(rng: Random.new(seed))
        order = set.rank(rng: Random.new(seed))

        expect(answer(choice)).to eq(item: "b", head: "b", branch: :thompson), "seed #{seed}"
        expect(choice.draws).to eq(order.draws), "seed #{seed}"
        expect(choice.draws["b"]).to eq(1.0), "seed #{seed}"
      end
    end
  end
end
