# frozen_string_literal: true

# GB9, the order rule written once: Order#repin(pinned), an order drawn from a key or given uniforms re-pinned on its
# own draws. The claim: it answers the very Order its set's `rank` answers on the same source with `pinned` under the
# held rule (shuffle 0, explore 0, no may_lead, min_views 0, top as the first order's), bit for bit (`==`, items,
# leader, branch, every draw's bits, the carried rule and what it keeps for the next re-pin), whatever rule drew the
# first order, and it draws by quantile only an item the first order did not draw (one it pinned, or every item of a
# shuffled order), counted at BetaArgmax.position (Betas) and Significance.normal_quantile (Normals). The reference is
# `rank` itself, called on the same source: the re-pin is held to the order rule, not to a second copy of it.
RSpec.describe "SkiftetStatistical::Bandit::Order#repin" do
  def posteriors = SkiftetStatistical::Bandit::Posteriors
  def bits(value) = value.is_a?(Float) ? [ value ].pack("G") : value
  def carried(order, name) = order.instance_variable_get(name)

  # A set of `size` items of `kind`, its parameters and views drawn from `gen`.
  def random_set(gen, kind, size)
    items = (0...size).map { |index| "i#{index}" }
    views = items.to_h { |item| [ item, gen.rand(0..40).to_f ] }
    if kind == :beta
      posteriors.beta(items.to_h { |item| [ item, [ 0.5 + (gen.rand * 30.0), 0.5 + (gen.rand * 120.0) ] ] }, views: views)
    else
      posteriors.normal(items.to_h { |item| [ item, [ (gen.rand * 4.0) - 2.0, 0.05 + gen.rand ] ] }, views: views)
    end
  end

  # A first order's rule, every keyword drawn: shuffle and explore (summing to at most 1), may_lead (nil, or a list of
  # up to three items), min_views, pinned (none, one or two positions) and top (nil, or 1 to two past the size).
  def random_rule(gen, items)
    shuffle = [ 0.0, 0.0, 0.1, 0.3, 0.5 ].sample(random: gen)
    explore = [ 0.0, 0.1, 0.3, 0.5 ].sample(random: gen)
    pinned = {}
    gen.rand(0..2).times { pinned[gen.rand(items.size)] = (items - pinned.values).sample(random: gen) }
    { shuffle: shuffle, explore: explore,
      may_lead: gen.rand < 0.5 ? nil : items.sample(gen.rand(0..3), random: gen),
      min_views: [ 0.0, 10.0, 30.0 ].sample(random: gen),
      pinned: pinned.compact,
      top: gen.rand < 0.5 ? nil : gen.rand(1..(items.size + 2)) }
  end

  # A source: a key (an Array of parts or a String), or given uniforms naming every item.
  def random_source(gen, kind, index, items)
    return { key: gen.rand < 0.7 ? [ "visit", "k#{index}" ] : "k#{index}" } if kind == :key

    { uniforms: { explore: gen.rand, priority: items.to_h { |item| [ item, gen.rand ] },
                  draw: items.to_h { |item| [ item, gen.rand ] } } }
  end

  def one_pin_maps(items) = (0...items.size).to_a.product(items).map { |position, item| { position => item } }

  def two_pin_maps(items)
    (0...items.size).to_a.combination(2).flat_map do |first, second|
      items.permutation(2).map { |one, other| { first => one, second => other } }
    end
  end

  # nil when the re-pin and `rank` answer alike, bit for bit; else what differs.
  def difference(actual, expected)
    return "==" unless actual == expected
    return "items" unless actual.items == expected.items && actual.leader == expected.leader
    return "branch" unless actual.branch == expected.branch
    return "draw items" unless actual.draws.keys == expected.draws.keys
    return "draw bits" unless actual.draws.values.map { |value| bits(value) } == expected.draws.values.map { |value| bits(value) }
    return "rule" unless carried(actual, :@rule) == carried(expected, :@rule)
    return "kept source" unless carried(actual, :@rank_source) == carried(expected, :@rank_source)
    return "kept keys" unless carried(actual, :@rank_keys).keys == carried(expected, :@rank_keys).keys &&
                              carried(actual, :@rank_keys).values.map { |value| bits(value) } ==
                              carried(expected, :@rank_keys).values.map { |value| bits(value) }

    nil
  end

  # The re-pin of `first` with `pinned` against `rank` on the same source under the held rule; [label, what differs]
  # appended to `failures` when they differ.
  def compare(set, source, first, pinned, failures, label)
    expected = set.rank(**source, pinned: pinned, top: carried(first, :@rule)[:top])
    actual = first.repin(pinned)
    what = difference(actual, expected)
    failures << "#{label} #{pinned.inspect}: #{what}" if what
  end

  def branches_of(orders) = orders.map(&:branch).tally

  it "GB9 answers rank on the same key or uniforms under the held rule, bit for bit, over 2 000 keys and 2 000 " \
     "given-uniforms sources on Beta and Normal sets of 1 to 15 items, every first rule, pins of one and two positions" do
    gen = Random.new(20_261_102)
    failures = []
    firsts = []
    compared = 0
    %i[key uniforms].each do |source_kind|
      2_000.times do |index|
        set = random_set(gen, index.even? ? :beta : :normal, gen.rand(1..15))
        items = set.items
        source = random_source(gen, source_kind, index, items)
        first = set.rank(**source, **random_rule(gen, items))
        firsts << first
        # Up to three pins of one position (the first order's own head held at position 0 among them) and up to three
        # of two.
        maps = [ { 0 => first.items.first } ]
        maps += (1...items.size).to_a.sample(2, random: gen).map { |position| { position => items.sample(random: gen) } }
        maps += Array.new([ items.size - 1, 3 ].min) do
          (0...items.size).to_a.sample(2, random: gen).zip(items.sample(2, random: gen)).to_h
        end
        maps.each do |pinned|
          compare(set, source, first, pinned, failures, "#{source_kind} #{index} #{set.kind} #{items.size}")
          compared += 1
        end
      end
    end

    expect(failures.first(5)).to eq([])
    expect(compared).to be > 18_000
    expect(branches_of(firsts).keys).to contain_exactly(:thompson, :explored, :shuffled)
  end

  it "GB9 answers rank for every pinned map of one and of two positions on sets of 1 to 6 items" do
    gen = Random.new(11)
    failures = []
    compared = 0
    %i[beta normal].each do |kind|
      %i[key uniforms].each do |source_kind|
        (1..6).each do |size|
          4.times do |index|
            set = random_set(gen, kind, size)
            source = random_source(gen, source_kind, index, set.items)
            first = set.rank(**source, **random_rule(gen, set.items))
            (one_pin_maps(set.items) + two_pin_maps(set.items)).each do |pinned|
              compare(set, source, first, pinned, failures, "#{kind} #{source_kind} #{size} #{index}")
              compared += 1
            end
          end
        end
      end
    end

    expect(failures.first(5)).to eq([])
    expect(compared).to eq(4 * 4 * [ 1, 6, 27, 88, 225, 486 ].sum)
  end

  it "GB9 answers rank with no pins, and a re-pin of a re-pin answers rank with the last pins" do
    gen = Random.new(5)
    failures = []
    200.times do |index|
      set = random_set(gen, index.even? ? :beta : :normal, gen.rand(2..10))
      source = random_source(gen, index % 4 < 2 ? :key : :uniforms, index, set.items)
      first = set.rank(**source, **random_rule(gen, set.items))
      top = carried(first, :@rule)[:top]
      none = difference(first.repin({}), set.rank(**source, top: top))
      failures << "#{index} none: #{none}" if none
      none = difference(first.repin(nil), set.rank(**source, top: top))
      failures << "#{index} nil: #{none}" if none
      once = { 0 => set.items.last }
      twice = { 1 => set.items.first }
      chained = difference(first.repin(once).repin(twice), set.rank(**source, pinned: twice, top: top))
      failures << "#{index} chained: #{chained}" if chained
    end

    expect(failures).to eq([])
  end

  describe "GB9 no second draw" do
    def counting(target, method)
      count = [ 0 ]
      allow(target).to receive(method).and_wrap_original do |original, *args|
        count[0] += 1
        original.call(*args)
      end
      count
    end

    it "GB9 draws by quantile only the items the first order did not draw, counted at BetaArgmax.position and " \
       "Significance.normal_quantile" do
      beta_calls = counting(SkiftetStatistical::BetaArgmax, :position)
      normal_calls = counting(SkiftetStatistical::Significance, :normal_quantile)
      gen = Random.new(3)
      seen = Hash.new(0)
      mismatches = []
      400.times do |index|
        kind = index.even? ? :beta : :normal
        set = random_set(gen, kind, gen.rand(1..13))
        items = set.items
        source = random_source(gen, index % 4 < 2 ? :key : :uniforms, index, items)
        first = set.rank(**source, **random_rule(gen, items))
        maps = [ { 0 => items.sample(random: gen) } ]
        maps << { 0 => items.first, items.size - 1 => items.last } if items.size > 1
        maps.each do |pinned|
          # A Beta draw is counted at its position (whose first guess reads the normal quantile); a Normal draw at
          # its normal quantile, and a Normal set computes no Beta position at all.
          before = [ beta_calls[0], normal_calls[0] ]
          order = first.repin(pinned)
          drawn_now = (items - pinned.values) - first.draws.keys
          used = kind == :beta ? beta_calls[0] - before[0] : normal_calls[0] - before[1]
          other = kind == :beta ? 0 : beta_calls[0] - before[0]
          mismatches << [ index, first.branch, pinned, used, drawn_now.size ] unless used == drawn_now.size && other.zero?
          expect(order.draws.keys).to eq(items - pinned.values)
          seen[[ first.branch, drawn_now.empty? ? :none : :some ]] += 1
        end
      end

      expect(mismatches.first(5)).to eq([])
      expect(seen.keys).to include([ :thompson, :none ], [ :thompson, :some ], [ :explored, :none ], [ :shuffled, :some ])
    end

    it "GB9 holds a head: the re-pin of a keyed order with its own head first draws nothing at all" do
      set = posteriors.beta((0...13).to_h { |index| [ "i#{index}", [ 2.0 + index, 40.0 - index ] ] })
      beta_calls = counting(SkiftetStatistical::BetaArgmax, :position)
      first = set.rank(key: [ "visit", "k1" ])
      drawn_first = beta_calls[0]
      head = first.items.first
      held = first.repin({ 0 => head })

      expect(drawn_first).to eq(13)
      expect(beta_calls[0]).to eq(13)
      expect(held.items).to eq(first.items)
      expect(held.leader).to eq(first.items[1])
      expect(held.draws).to eq(first.draws.reject { |item, _| item == head })
      expect(held).to eq(set.rank(key: [ "visit", "k1" ], pinned: { 0 => head }))
    end

    it "GB9 draws every free item now when the first order was shuffled, and none of its pinned ones" do
      set = posteriors.beta({ "a" => [ 2.0, 8.0 ], "b" => [ 8.0, 2.0 ], "c" => [ 3.0, 3.0 ], "d" => [ 1.0, 1.0 ] })
      beta_calls = counting(SkiftetStatistical::BetaArgmax, :position)
      uniforms = { explore: 0.05, priority: { "a" => 0.4, "b" => 0.3, "c" => 0.2, "d" => 0.1 },
                   draw: { "a" => 0.9, "b" => 0.1, "c" => 0.5, "d" => 0.5 } }
      first = set.rank(uniforms: uniforms, shuffle: 0.1)

      expect(first.branch).to eq(:shuffled)
      expect(beta_calls[0]).to eq(0)
      order = first.repin({ 1 => "a" })

      expect(beta_calls[0]).to eq(3)
      expect(order).to eq(set.rank(uniforms: uniforms, pinned: { 1 => "a" }))
    end
  end

  describe "GB9 what a re-pin answers and carries" do
    let(:set) { posteriors.beta({ "a" => [ 2.0, 8.0 ], "b" => [ 8.0, 2.0 ], "c" => [ 3.0, 3.0 ], "d" => [ 1.0, 1.0 ] }) }

    it "GB9 answers a frozen Order under the held rule, with the first order's top, and its probability as rank's" do
      first = set.rank(key: "k1", explore: 0.3, shuffle: 0.2, may_lead: %w[a b], min_views: 5, pinned: { 2 => "d" },
                       top: 3)
      order = first.repin({ 0 => "c" })
      expected = set.rank(key: "k1", pinned: { 0 => "c" }, top: 3)

      expect(order).to be_frozen
      expect(order).to be_a(SkiftetStatistical::Bandit::Order)
      expect(order.items.size).to eq(3)
      expect(order.branch).to eq(:thompson)
      expect(carried(order, :@rule)).to eq(shuffle: 0.0, explore: 0.0, may_lead: nil, min_views: 0.0,
                                            pinned: { 0 => "c" }, top: 3)
      expect(order.probability).to eq(expected.probability)
      expect(carried(first, :@rule)[:pinned]).to eq(2 => "d")
    end

    it "GB9 keeps the source as the set read it: a key or uniforms the caller changes afterwards moves no re-pin" do
      key = [ "visit", "k2" ]
      by_key = set.rank(key: key, pinned: { 0 => "a" })
      key << "later"
      priority = { "a" => 0.1, "b" => 0.2, "c" => 0.3, "d" => 0.4 }
      draw = { "a" => 0.2, "b" => 0.4, "c" => 0.6, "d" => 0.8 }
      uniforms = { explore: 0.5, priority: priority, draw: draw }
      by_uniforms = set.rank(uniforms: uniforms, pinned: { 0 => "a" })
      draw["a"] = 0.99

      expect(by_key.repin({ 0 => "b" })).to eq(set.rank(key: [ "visit", "k2" ], pinned: { 0 => "b" }))
      original = { explore: 0.5, priority: { "a" => 0.1, "b" => 0.2, "c" => 0.3, "d" => 0.4 },
                   draw: { "a" => 0.2, "b" => 0.4, "c" => 0.6, "d" => 0.8 } }
      repinned = by_uniforms.repin({ 0 => "b" })
      expect(repinned).to eq(set.rank(uniforms: original, pinned: { 0 => "b" }))
      expect(repinned.draws.fetch("a")).not_to eq(set.rank(uniforms: uniforms, pinned: { 0 => "b" }).draws.fetch("a"))
    end

    # A part whose to_s answers the text the caller holds, counting each time it is asked.
    def held_text(text)
      Class.new do
        attr_accessor :text
        attr_reader :asked

        def initialize(text)
          @text = text
          @asked = 0
        end

        def to_s
          @asked += 1
          @text
        end
      end.new(text)
    end

    # A number in a container the caller holds: a Numeric whose to_f answers the value it holds now, counting each time
    # it is asked.
    def held_number(value)
      Class.new(Numeric) do
        attr_accessor :value
        attr_reader :asked

        def initialize(value)
          super()
          @value = value
          @asked = 0
        end

        def real? = true
        def finite? = true

        def to_f
          @asked += 1
          @value
        end
      end.new(value)
    end

    it "GB9 keeps each key part as the key source reads it, its bytes, frozen, whose uniforms are the part's own, bit " \
       "for bit, for String, Symbol, Integer and Float parts, nil and an Array key" do
      draw = SkiftetStatistical::Bandit::Draw
      latin = "é".encode("ISO-8859-1")
      uniforms = lambda do |parts|
        [ draw.uniform_for(*parts, "explore"),
          *set.items.flat_map { |item| [ draw.uniform_for(*parts, "priority", item), draw.uniform_for(*parts, "draw", item) ] } ]
      end
      [ "k1", [ "visit", "k1" ], [ :visit, 7, 0.1 ], [ 2.5e-300, -0.0, 2**70, 1e22 ], [ latin, "é", nil, [ 1, "a" ] ],
        [ :a, "b", 3, 4.25 ] ].each do |key|
        order = set.rank(key: key, explore: 0.3, shuffle: 0.2)
        origin, parts = carried(order, :@rank_source)
        given = key.is_a?(Array) ? key : [ key ]

        expect(origin).to eq(:key)
        expect(parts).to be_frozen
        expect(parts.map(&:class).uniq).to eq([ String ])
        expect(parts).to all(be_frozen)
        expect(parts.map(&:bytes)).to eq(given.map { |part| part.is_a?(String) ? part.bytes : part.to_s.bytes })
        expect(uniforms.call(parts).map { |value| bits(value) }).to eq(uniforms.call(given).map { |value| bits(value) })
        parts.zip(given) { |kept, part| expect(bits(draw.uniform_for(kept))).to eq(bits(draw.uniform_for(part))) }
        expect(draw.seed_for(*parts)).to eq(draw.seed_for(*given))
        expect(difference(order.repin({ 0 => "c" }), set.rank(key: key, pinned: { 0 => "c" }))).to be_nil
      end
      expect(carried(set.rank(key: [ latin ]), :@rank_source).last).not_to eq(carried(set.rank(key: [ "é" ]), :@rank_source).last)
    end

    it "GB9 reads no key part again: a String part changed in place, or a part whose to_s answers otherwise, after rank " \
       "moves no re-pin, which answers the rank of the original key pinned, bit for bit" do
      part = +"k4"
      key = [ "visit", part ]
      first = set.rank(key: key, explore: 0.3, pinned: { 0 => "a" })
      part << "-later"
      repinned = first.repin({ 0 => "b" })

      expect(difference(repinned, set.rank(key: [ "visit", "k4" ], pinned: { 0 => "b" }))).to be_nil
      expect(repinned.draws.fetch("a")).not_to eq(set.rank(key: key, pinned: { 0 => "b" }).draws.fetch("a"))

      text = +"k5"
      by_text = set.rank(key: text, shuffle: 1.0, pinned: { 1 => "c" })
      text.replace("other")
      expect(by_text.branch).to eq(:shuffled)
      expect(difference(by_text.repin({ 1 => "d" }), set.rank(key: "k5", pinned: { 1 => "d" }))).to be_nil
      expect(by_text.repin({ 1 => "d" })).not_to eq(set.rank(key: text, pinned: { 1 => "d" }))

      holder = held_text("k6")
      by_holder = set.rank(key: [ "visit", holder, :x ], pinned: { 0 => "d" })
      asked = holder.asked
      holder.text = "k7"
      expect(difference(by_holder.repin({ 0 => "a" }), set.rank(key: [ "visit", "k6", :x ], pinned: { 0 => "a" }))).to be_nil
      expect(asked).to eq(1)
      expect(holder.asked).to eq(asked)
    end

    it "GB9 reads no given number again: a number in a container the caller changes after rank moves no re-pin, which " \
       "answers the rank of the original uniforms pinned, bit for bit" do
      # Thompson: "a" is pinned, so its entries are not drawn by the first order and are drawn now by the re-pin.
      pinned_priority = held_number(0.1)
      pinned_draw = held_number(0.2)
      priority = { "a" => pinned_priority, "b" => 0.2, "c" => 0.3, "d" => 0.4 }
      draw = { "a" => pinned_draw, "b" => 0.7, "c" => 0.6, "d" => 0.8 }
      uniforms = { explore: 0.5, priority: priority, draw: draw }
      first = set.rank(uniforms: uniforms, pinned: { 0 => "a" })
      asked = [ pinned_priority.asked, pinned_draw.asked ]
      pinned_draw.value = 0.95
      pinned_priority.value = 1e200
      draw["c"] = 0.01
      priority.delete("d")
      original = { explore: 0.5, priority: { "a" => 0.1, "b" => 0.2, "c" => 0.3, "d" => 0.4 },
                   draw: { "a" => 0.2, "b" => 0.7, "c" => 0.6, "d" => 0.8 } }
      repinned = first.repin({ 0 => "b" })

      expect(first.branch).to eq(:thompson)
      expect(difference(repinned, set.rank(uniforms: original, pinned: { 0 => "b" }))).to be_nil
      expect([ pinned_priority.asked, pinned_draw.asked ]).to eq(asked)
      expect { set.rank(uniforms: uniforms, pinned: { 0 => "b" }) }
        .to raise_error(ArgumentError, /priority of the item "a" must lie within the input domain/)

      # Shuffled: the first order drew nothing, so the re-pin draws every free item now, each by its number as read.
      held = { "a" => held_number(0.9), "b" => held_number(0.1), "c" => held_number(0.5), "d" => held_number(0.3) }
      shuffled_uniforms = { explore: 0.05, priority: { "a" => 0.4, "b" => 0.3, "c" => 0.2, "d" => 0.1 }, draw: held }
      shuffled = set.rank(uniforms: shuffled_uniforms, shuffle: 0.1, pinned: { 3 => "d" })
      asked = held.transform_values(&:asked)
      held.each_value { |number| number.value = 1.0 - number.value }
      fixed = { explore: 0.05, priority: { "a" => 0.4, "b" => 0.3, "c" => 0.2, "d" => 0.1 },
                draw: { "a" => 0.9, "b" => 0.1, "c" => 0.5, "d" => 0.3 } }

      repinned = shuffled.repin({ 1 => "a" })

      expect(shuffled.branch).to eq(:shuffled)
      expect(difference(repinned, set.rank(uniforms: fixed, pinned: { 1 => "a" }))).to be_nil
      expect(held.transform_values(&:asked)).to eq(asked)
      expect(repinned.items).not_to eq(set.rank(uniforms: shuffled_uniforms, pinned: { 1 => "a" }).items)
    end

    it "GB9 reads a pinned item's given entry as it read at rank time: one refused then stays refused in rank's words " \
       "after the caller mends it, and rank itself takes it" do
      [ [ held_number(1.5), ->(entries, number) { number.value = 0.5 } ],
        [ 1.5, ->(entries, _number) { entries["c"] = 0.5 } ],
        [ nil, ->(entries, _number) { entries["c"] = 0.5 } ] ].each do |number, mend|
        draw = { "a" => 0.5, "b" => 0.5, "c" => number, "d" => 0.5 }
        draw.delete("c") if number.nil?
        uniforms = { explore: 0.5, priority: { "a" => 0.1, "b" => 0.2, "c" => 0.3, "d" => 0.4 }, draw: draw }
        first = set.rank(uniforms: uniforms, pinned: { 0 => "c" })
        words = refusal_words { set.rank(uniforms: uniforms, pinned: { 0 => "a" }) }
        mend.call(draw, number)

        expect(words).to match(/uniforms' draw of the item "c" (must lie in \[0, 1\], got 1.5|is not given)/)
        expect(refusal_words { set.rank(uniforms: uniforms, pinned: { 0 => "a" }) }).to be_nil
        expect(refusal_words { first.repin({ 0 => "a" }) }).to eq(words)
        expect(first.repin({ 1 => "c" })).to eq(set.rank(uniforms: uniforms, pinned: { 1 => "c" }))
      end
    end

    def refusal_words
      yield
      nil
    rescue ArgumentError => e
      e.message
    end

    it "GB9 answers an empty set and a set whose items are all pinned as rank does" do
      empty = posteriors.beta({})

      expect(empty.rank(key: "k").repin({})).to eq(empty.rank(key: "k"))
      all = set.rank(key: "k3").repin({ 0 => "d", 1 => "c", 2 => "b", 3 => "a" })
      expect(all).to eq(set.rank(key: "k3", pinned: { 0 => "d", 1 => "c", 2 => "b", 3 => "a" }))
      expect(all.items).to eq(%w[d c b a])
      expect(all.leader).to be_nil
      expect(all.draws).to eq({})
    end
  end

  describe "GB9 refusals, as rank refuses" do
    let(:set) { posteriors.beta({ "a" => [ 2.0, 8.0 ], "b" => [ 8.0, 2.0 ], "c" => [ 3.0, 3.0 ] }) }

    def refusal
      yield
      nil
    rescue ArgumentError => e
      e.message
    end

    it "GB9 refuses an order drawn from an RNG: a re-pin needs a key or given uniforms" do
      order = set.rank(rng: Random.new(1), explore: 0.1)

      expect { order.repin({ 0 => "a" }) }.to raise_error(ArgumentError, /re-pin needs an order drawn from a key or given uniforms/)
    end

    it "GB9 refuses an Order no posterior set made" do
      order = SkiftetStatistical::Bandit::Order.new(items: %w[a], leader: "a", branch: :thompson, draws: { "a" => 0.5 })

      expect { order.repin({}) }.to raise_error(ArgumentError, /re-pinned only when a posterior set made it/)
    end

    it "GB9 refuses a malformed pinned map with rank's own words" do
      first = set.rank(key: "k", top: 2)
      [ [ 1 ], "a", { 3 => "a" }, { -1 => "a" }, { 0.5 => "a" }, { "0" => "a" }, { 0 => "z" }, { 0 => "a", 1 => "a" },
        { 0 => :a } ].each do |pinned|
        words = refusal { set.rank(key: "k", pinned: pinned, top: 2) }

        expect(words).not_to be_nil, pinned.inspect
        expect(refusal { first.repin(pinned) }).to eq(words), pinned.inspect
      end
    end

    it "GB9 refuses, as rank does, a Beta parameter below the smallest of an item the first order pinned" do
      narrow = posteriors.beta({ "a" => [ 2.0, 8.0 ], "b" => [ 1e-7, 2.0 ], "c" => [ 3.0, 3.0 ] })
      first = narrow.rank(key: "k", pinned: { 0 => "b" })
      words = refusal { narrow.rank(key: "k", pinned: { 0 => "a" }) }

      expect(words).to match(/must be at least/)
      expect(refusal { first.repin({ 0 => "a" }) }).to eq(words)
      expect(first.repin({ 1 => "b" })).to eq(narrow.rank(key: "k", pinned: { 1 => "b" }))
    end

    it "GB9 refuses, as rank does, a given uniform missing or malformed for an item the first order pinned" do
      missing = { explore: 0.5, priority: { "a" => 0.1, "b" => 0.2 }, draw: { "a" => 0.5, "b" => 0.5 } }
      malformed = { explore: 0.5, priority: { "a" => 0.1, "b" => 0.2, "c" => 0.3 },
                    draw: { "a" => 0.5, "b" => 0.5, "c" => 1.5 } }
      [ missing, malformed ].each do |uniforms|
        first = set.rank(uniforms: uniforms, pinned: { 0 => "c" })
        words = refusal { set.rank(uniforms: uniforms, pinned: { 0 => "a" }) }

        expect(words).not_to be_nil
        expect(refusal { first.repin({ 0 => "a" }) }).to eq(words)
        expect(first.repin({ 1 => "c" })).to eq(set.rank(uniforms: uniforms, pinned: { 1 => "c" }))
      end
    end
  end
end
