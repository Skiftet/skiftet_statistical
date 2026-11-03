# frozen_string_literal: true

require_relative "../numbers"
require_relative "../inputs"
require_relative "../significance"
require_relative "../beta_argmax"
require_relative "checks"
require_relative "draw"
require_relative "posteriors"

module SkiftetStatistical
  class Bandit
    # A pick: `item` (the leader: the head when it has at least `min_views` views, else nil), `head` (the first free
    # item, nil when there is none), `branch` (:thompson, :explored or :shuffled) and `draws` ({ item => posterior
    # value } for every free item the branch drew; none on the shuffled branch). Frozen. It carries, in instance
    # variables set before the freeze, the posterior set that made it (@posteriors), the rule it was made under
    # (@rule: { shuffle:, explore:, may_lead:, min_views: }, the keywords of `pick` and of `pick_probabilities`) and an
    # empty Hash made before the freeze (@cache), so its odds can be computed under that rule once and kept. `==`
    # compares the four members only.
    Choice = Struct.new(:item, :head, :branch, :draws, keyword_init: true)

    # An order: `items` (the first `top` items of the whole order, pinned items at their positions), `leader` (the
    # first free item when it has at least `min_views` views, else nil), `branch` and `draws` (as a Choice's). Frozen,
    # carrying @posteriors, @rule ({ shuffle:, explore:, may_lead:, min_views:, pinned:, top: }, the keywords of `rank`
    # and of `order_probability`) and @cache as a Choice does, and, drawn from a key or given uniforms, what `repin`
    # reads: @rank_source (the source as `rank` read it, library data that holds no caller's object: [:key, the key's
    # parts as the one reading of a keyed draw's parts read them, each its bytes, a frozen String], or [:uniforms,
    # { item => the given priority }, { item => the given draw uniform }], two Hashes the set built, keyed by its own
    # item Strings, each holding for every item the number `rank` read, or, for an item it pinned whose entry is
    # missing or malformed, the words of that refusal; all frozen) and @rank_keys ({ item => the key the rule compared
    # its draw by }, frozen, for every item it drew). `==` compares the four members only.
    Order = Struct.new(:items, :leader, :branch, :draws, keyword_init: true)

    # THE ORDER RULE, written once: `rank` orders a posterior set's items and `pick` is its head (`rank` with top 1
    # and no pins), so the head, its exploration share and the order by draw are one rule under every source.
    #
    #   set = SkiftetStatistical::Bandit::Posteriors.beta({ "a" => [ 2.0, 8.0 ], "b" => [ 8.0, 2.0 ] })
    #   set.rank(key: [ "visit", "k1" ], explore: 0.1).items   # => an order of "a" and "b", the same for that key
    #   set.pick(rng: Random.new(1)).item                      # => the head of one Thompson draw
    #
    # The rule: `shuffle` and `explore` (each in [0, 1], summing to at most 1), `may_lead` (the items that may lead;
    # nil, or a list naming none of the free items, lets every free item lead), `min_views` (0), `pinned`
    # ({ position => item }, 0-based) and `top` (nil: every item). Pinned items keep their positions and draw nothing;
    # the free items fill the other positions in order. The contenders are the free items `may_lead` names, the
    # eligible the contenders with at least `min_views` views. With the branch value c: c < shuffle is :shuffled (the
    # contender of smallest priority leads, the other free items follow by priority); c < shuffle + explore with an
    # eligible item is :explored (the eligible item of smallest priority leads, the others follow by draw); else
    # :thompson (the contender of largest draw leads, the others follow by draw). Ties go to the earlier item. The
    # leader is the free head when it has at least `min_views` views, else nil.
    #
    # Exactly one source of randomness:
    # - `key:` (a String, or an Array of parts): the branch value is Draw.uniform_for(*key, "explore"), an item's
    #   priority uniform_for(*key, "priority", item) and its draw uniform uniform_for(*key, "draw", item), so no
    #   item's uniforms depend on another item, on the set or on its order;
    # - `uniforms:` ({ explore: Float, priority: { item => Float }, draw: { item => Float } }): the same three given,
    #   explore and every free item's draw uniform in [0, 1] and every free item's priority a finite number (entries
    #   for other names are not read);
    # - `rng:` (anything that answers rand, each answer a Float in [0, 1)): one branch value when shuffle + explore is
    #   above 0, then one priority per free item, in item order, on the shuffled or explored branch, then one Sampler
    #   draw per free item, in item order, where the branch draws (Sampler#beta, or mean + sd * Sampler#gaussian), and
    #   nothing else from the stream.
    # Every source draws on the scaled pairs (`parameters`), the one pair the odds read too. A key or given uniforms
    # draw by quantile: a Beta item at BetaArgmax.position(uniform, alpha, beta), compared by that position so that two
    # draws never tie by rounding, its draw the value there (exp(p + JOINT) for a position p below 0, else 1 -
    # exp(JOINT - p), the position's two halves, `rank_beta_half`); a Normal item at mean + sd * z, z the standard
    # quantile Significance.normal_quantile(u) of its uniform u on the open grid (Draw.open_uniform, the one reading of
    # a quantile draw's uniform: clamped to [2**-53, 1 - 2**-53], so the quantile is never infinite). An RNG draws a
    # Beta item by Sampler#beta, compared by its value, and a Normal item at z = Sampler#gaussian. Every Normal draw,
    # whatever its source, is compared by the exact sign of (mean_i - mean_j) + (sd_i z_i - sd_j z_j), its mean + sd
    # * z formed exactly (the set's one Normal draw, `normal_draw`), so two draws never tie or misorder by rounding,
    # even where a standard deviation lies below a rounding of its mean; the value it answers is that sum rounded to a
    # Float. Every source draws exactly from the posteriors, so one set of odds covers all three. `draws` answers
    # posterior values, never positions or exact keys, and every one passes the bandit's one check of an answer
    # (Checks.finite_answer).
    #
    # Refused, as ArgumentErrors before any draw and before anything is taken from an RNG: a rule outside the bounds
    # above, a pinned position that is not an Integer in 0...items.size, a pinned name that is not an item, one name
    # pinned twice, a `may_lead` name that is not an item, `top` that is not an Integer of at least 1, no source or
    # more than one, a malformed `uniforms` Hash, and under a key or given uniforms, whatever branch they choose, a
    # Beta parameter of a free item below BetaArgmax::SMALLEST_PARAMETER (the set's one refusal; a parameter at it is
    # taken, and the RNG source takes any parameter above 0). An empty set, or one whose items are all pinned, ranks
    # with no head and no leader.
    #
    # THE RE-PIN (`Order#repin(pinned)`): an order drawn from a key or given uniforms answers the order its set's rule
    # answers on the same source with `pinned` under the held rule (shuffle 0, explore 0, no `may_lead`, `min_views`
    # 0, `top` as the order's own), so the Thompson order of the items `pinned` leaves free, with the pinned items at
    # their positions, the very Order `rank` answers there, bit for bit, `==` and every draw. It is the order rule's
    # own code fed the order's own draws: a free item the order drew keeps its draw (the key it was compared by and its
    # value), and only an item it did not draw (one it pinned, or every item of a shuffled order) is drawn now, by
    # quantile from the kept source: only what `rank` read (each key part's bytes, each given number as read), never a
    # caller's object, so a key part or a given number the caller changes after `rank` moves no re-pin, which answers
    # `rank` on the source as it was. So a caller holding a head asks once more and pays no second draw:
    #
    #   order = set.rank(key: [ "visit", "k1" ], explore: 0.1)
    #   order.repin({ 0 => "b" }) == set.rank(key: [ "visit", "k1" ], pinned: { 0 => "b" })   # => true
    #
    # Refused as `rank` refuses on that source (a malformed `pinned`, a Beta parameter below the smallest of an item
    # now free, a given uniform of an item now free missing or malformed at rank time, in the words `rank` gives), and,
    # as an ArgumentError, an order drawn from an RNG (a re-pin needs a key or given uniforms) or one no posterior set
    # made.
    #
    # For the files that reopen this class, private: `rank_rule(...)` (the rule's keywords read and checked once,
    # answered as the frozen Hash an Order carries; the pick odds read their rule through it, with no pins and no top),
    # `rank_free(rule)` (the free items under a rule, which the order rule and the order odds both read),
    # `rank_free_slots(pinned)` (the free-slot rule: the slots the free ranks take, in order, which the order rule's
    # whole order and the order and position odds read), `rank_beta_half(position)` (a Beta position's two halves,
    # which the quantile draws' values and the order odds' distribution functions read) and `rank_ordered(rule, key,
    # uniforms, rng)` (the order rule itself, its first answer the whole order; the odds' Monte Carlo visits it).
    class Posteriors
      # What a key or given uniforms compute, in the words of GB6's refusal.
      RANK_QUANTILE_DRAW = "a quantile draw on the scaled pairs (an RNG draw takes any above 0)"
      private_constant :RANK_QUANTILE_DRAW

      # The posterior set's items in the order the rule draws: an Order.
      #
      # Takes (the class comment's rule): exactly one of `key:` (a String, or an Array of parts: nil, a String, or
      # anything whose to_s answers a String, none holding a NUL byte), `uniforms:` or `rng:` (answering rand with
      # Floats in [0, 1)); `shuffle` and `explore` in [0, 1] summing to at most 1; `may_lead` nil or a list of item
      # names; `min_views` a number in the input domain (one below 0 asks for none, as 0 does); `pinned` a Hash of
      # Integer position (0-based) => item name; `top` nil or an Integer of at least 1.
      # Answers: a frozen Order (`items`, `leader`, `branch`, `draws`), and writes nothing to the set.
      # Guarantees:
      # - A key draws the same Order from the same set in every process: its uniforms are Draw.uniform_for of the
      #   key's parts (SHA256, identical on every machine and Ruby version), and its draws their Beta or Normal
      #   quantiles on `parameters`, whose last bits rest on the math library's lgamma (the pins compare a draw at
      #   1e-9 relative, and an order, a leader and a branch exactly).
      # - An item's uniforms are the key's and its own name's alone: with no pin, an added item never changes the
      #   others' relative order, and an added item that `may_lead` does not name changes neither the others' order,
      #   the leader nor the branch. `may_lead`, `min_views`, `shuffle` and `explore` choose the head, the branch and
      #   the leader; the items after the head follow by draw (by priority on the shuffled branch) whatever they are.
      # - An RNG gives the same Order for the same stream: the rule takes from it exactly one branch value (when
      #   shuffle + explore is above 0), one priority per free item on the shuffled or explored branch, then one
      #   Sampler draw per free item where the branch draws, in item order, and nothing else.
      # - Every source draws exactly from the posteriors, so the order and pick odds describe each of them.
      # Refuses: the class comment's list, each an ArgumentError before any draw and before anything is taken from an
      # RNG; an RNG's answer that is not a Float in [0, 1), or a raise from its rand, is an ArgumentError at that answer.
      def rank(key: nil, uniforms: nil, rng: nil, shuffle: 0.0, explore: 0.0, may_lead: nil, min_views: 0.0,
               pinned: {}, top: nil)
        rule = rank_rule(shuffle: shuffle, explore: explore, may_lead: may_lead, min_views: min_views, pinned: pinned,
                         top: top)
        rank_answered(rule, *rank_ordered(rule, key, uniforms, rng))
      end

      # The head of the order rule (`rank` with top 1 and no pins): a Choice.
      #
      # Takes `rank`'s source and rule, without `pinned` and `top`. Answers a frozen Choice: `item` (the leader: the
      # head when it has at least `min_views` views, else nil), `head`, `branch` and `draws` (every free item's draw,
      # none on the shuffled branch). Guarantees `rank`'s: under every source it is the head of the Order `rank`
      # answers there, an RNG giving up exactly the values `rank` takes. Refuses what `rank` refuses.
      def pick(key: nil, uniforms: nil, rng: nil, shuffle: 0.0, explore: 0.0, may_lead: nil, min_views: 0.0)
        rule = rank_rule(shuffle: shuffle, explore: explore, may_lead: may_lead, min_views: min_views, pinned: {},
                         top: nil)
        _full, head, branch, draws = rank_ordered(rule, key, uniforms, rng)
        Checks.finite_answer(draws.values, "the parameters and the draw's source")
        choice = Choice.new(item: rank_leader(head, rule), head: head, branch: branch, draws: draws)
        rank_carrying(choice, rule.slice(:shuffle, :explore, :may_lead, :min_views).freeze)
      end

      private

      # The rule's keywords, each read and checked once: { shuffle:, explore:, may_lead:, min_views:, pinned:, top: },
      # frozen, its names the set's own item Strings. shuffle and explore by the bandit's one [0, 1] bound
      # (Checks.unit), summing to at most 1; may_lead nil or a list of items; min_views any finite number (views are
      # >= 0, so one below 0 asks for none); pinned nil or a Hash of distinct Integer positions in 0...items.size to
      # distinct items; top nil or an Integer of at least 1 within a Float's range (the bandit's one reading of a whole
      # number, Checks.integer). The pick odds read their rule here too.
      def rank_rule(shuffle:, explore:, may_lead:, min_views:, pinned:, top:)
        shuffle = Checks.unit(shuffle, "shuffle")
        explore = Checks.unit(explore, "explore")
        if shuffle + explore > 1.0
          raise ArgumentError, "shuffle and explore must sum to at most 1 (got #{shuffle} and #{explore})"
        end
        Checks.integer(top, "top", minimum: 1) unless nil.equal?(top)

        { shuffle: shuffle, explore: explore, may_lead: rank_may_lead(may_lead),
          min_views: Numbers.read(min_views, "min_views"), pinned: rank_pinned(pinned), top: top }.freeze
      end

      # nil, or the names `may_lead` gives as the set's own Strings, in its order, frozen: the items the set's one list
      # reading (`listed_items`) read, `may_lead` read once, a name that is not an item and a `may_lead` that is not a
      # list refused there.
      def rank_may_lead(may_lead)
        return nil if nil.equal?(may_lead)

        listed_items(may_lead, "may_lead", @items, @items).last
      end

      # { position => item }, frozen: distinct Integer positions inside the items (each read by the bandit's one reading
      # of a whole number, Checks.integer, in 0...items.size), each naming a distinct item. `pinned` is read once; its
      # names are the items the set's one list reading (`listed_items`) read, a name that is not an item refused there.
      def rank_pinned(pinned)
        return {}.freeze if nil.equal?(pinned)
        raise ArgumentError, "pinned must be a Hash of position => item (got #{Inputs.shown(pinned)})" unless Hash === pinned

        entries = Inputs.pairs(pinned)
        entries.each { |position, _name| Checks.integer(position, "pinned: a position", minimum: 0, below: @items.size) }
        _chosen, items = listed_items(entries.map(&:last), "pinned", @items, [])
        seen = {}
        items.each do |item|
          raise ArgumentError, "pinned: #{Inputs.shown(item)} is pinned twice" if seen.key?(item)

          seen[item] = true
        end
        entries.map(&:first).zip(items).to_h.freeze
      end

      # THE free items under the rule, written once (the order rule's draws and the order and position odds read
      # them here): every item but the pinned ones, in item order.
      def rank_free(rule)
        pinned = rule[:pinned]
        pinned.empty? ? @items : @items - pinned.values
      end

      # THE Order `rank` and the re-pin answer, written once: from the rule's whole order, free head, branch and draws
      # (`rank_ordered`), the first `top` items and the leader, its draws through the bandit's one check of an answer,
      # carrying the set, the rule and what a re-pin keeps (`rank_carrying`), frozen.
      def rank_answered(rule, full, head, branch, draws, kept)
        Checks.finite_answer(draws.values, "the parameters and the draw's source")
        top = rule[:top]
        order = Order.new(items: (nil.equal?(top) ? full : full.first([ top, full.size ].min)).freeze,
                          leader: rank_leader(head, rule),
                          branch: branch, draws: draws)
        rank_carrying(order, rule, kept)
      end

      # [the whole order, the free head, the branch, the draws, what a re-pin keeps] under the rule, from exactly one
      # source, every draw on the scaled pairs (`parameters`): a key, given uniforms or an RNG, or, for a re-pin
      # (`rank_repinned`), `kept`, an order's own [source, keys, draws]. What a re-pin keeps is [the source as read,
      # { item => the key its draw was compared by }] from a key or given uniforms, nil from an RNG.
      def rank_ordered(rule, key, uniforms, rng, kept = nil)
        source = nil.equal?(kept) ? rank_source(key, uniforms, rng) : :kept
        free = rank_free(rule)
        leading = contenders(rule[:may_lead], free)
        eligible = leading.select { |item| eligible?(item, rule[:min_views]) }
        head, rest, branch, draws, keys, read = case source
        when :rng then rank_by_rng(rng, rule, free, leading, eligible)
        when :key then rank_by_key(key, rule, free, leading, eligible)
        when :uniforms then rank_by_uniforms(uniforms, rule, free, leading, eligible)
        else rank_by_kept(kept, rule, free, leading, eligible)
        end
        free_order = head.nil? ? [] : [ head, *rest ]
        [ rank_slotted(rule[:pinned], free_order), head, branch, draws,
          nil.equal?(read) ? nil : [ read, (keys || {}).freeze ].freeze ]
      end

      # :key, :uniforms or :rng, the one source given (nil is not given).
      def rank_source(key, uniforms, rng)
        given = { key: key, uniforms: uniforms, rng: rng }.reject { |_, value| nil.equal?(value) }
        return given.keys.first if given.size == 1

        raise ArgumentError, "give exactly one source of randomness, key:, uniforms: or rng: (got " \
                             "#{given.empty? ? 'none' : given.keys.join(' and ')})"
      end

      # The RNG source: the branch value (when shuffle + explore is above 0), a priority per free item where the
      # branch reads priorities, a Sampler draw per free item where it draws (`sampler_draw`, on the scaled pairs), in
      # that order, each compared by its key and answered as its value.
      def rank_by_rng(rng, rule, free, leading, eligible)
        sampler = checked_sampler(rng)
        value = (rule[:shuffle] + rule[:explore]).positive? ? Inputs.uniform(rng) : nil
        branch = rank_branch(value, rule, eligible)
        priorities = branch == :thompson ? nil : free.to_h { |item| [ item, Inputs.uniform(rng) ] }
        return rank_decided(branch, free, leading, eligible, priorities, nil, nil) if branch == :shuffled

        drawn = free.to_h { |item| [ item, sampler_draw(sampler, item) ] }
        rank_decided(branch, free, leading, eligible, priorities, drawn.transform_values(&:first),
                     drawn.transform_values(&:last))
      end

      # The key source: the key's parts read once, by the bandit's one reading of a keyed draw's parts (KeyParts.read:
      # each part's bytes, a frozen String, a part that cannot be read refused in its turn, as Draw.uniform_for refuses
      # it), and every uniform drawn from what it read, the branch value first. A part read so reads again as its own
      # bytes, so each uniform is the one Draw.uniform_for answers on the caller's parts, bit for bit. Kept for a
      # re-pin: [:key, the parts as read], library Strings, never a caller's part, so a part changed after `rank` (a
      # String changed in place, an object whose to_s answers otherwise) moves no re-pin.
      def rank_by_key(key, rule, free, leading, eligible)
        unless String === key || Array === key
          raise ArgumentError, "key must be a String or an Array of parts (got #{Inputs.shown(key)})"
        end

        refuse_below_smallest_parameter(free, computing: RANK_QUANTILE_DRAW)
        parts = KeyParts.read(*(Array === key ? Inputs.elements(key) : [ key ]))
        branch = rank_branch(Draw.uniform_for(*parts, "explore"), rule, eligible)
        priorities = branch == :thompson ? nil : rank_keyed(parts, "priority", free)
        uniforms = branch == :shuffled ? nil : rank_keyed(parts, "draw", free)
        [ *rank_by_quantile(branch, free, leading, eligible, priorities, uniforms), [ :key, parts ].freeze ]
      end

      # THE keyed uniform of each item, written once (the key source's priorities and draw uniforms, and a re-pin's
      # draw uniforms): { item => Draw.uniform_for(*parts, part, item) }, in the items' order.
      def rank_keyed(parts, part, items)
        items.to_h { |item| [ item, Draw.uniform_for(*parts, part, item) ] }
      end

      # The given-uniforms source: explore, and every free item's priority and draw uniform, read and checked before
      # anything is drawn. Kept for a re-pin: [:uniforms, the priorities, the draw uniforms], each as `rank` read it
      # (`rank_kept_given`), never a caller's number.
      def rank_by_uniforms(uniforms, rule, free, leading, eligible)
        unless Hash === uniforms
          raise ArgumentError, "uniforms must be a Hash of explore:, priority: and draw: (got #{Inputs.shown(uniforms)})"
        end

        refuse_below_smallest_parameter(free, computing: RANK_QUANTILE_DRAW)
        explore = Checks.unit(Checks.fetch(uniforms, :explore, "uniforms"), "uniforms' explore")
        named_priorities = rank_named(uniforms, :priority)
        priorities = rank_read(named_priorities, :priority, free)
        named_draws = rank_named(uniforms, :draw)
        draw_uniforms = rank_read(named_draws, :draw, free)
        branch = rank_branch(explore, rule, eligible)
        [ *rank_by_quantile(branch, free, leading, eligible, branch == :thompson ? nil : priorities,
                            branch == :shuffled ? nil : draw_uniforms),
          [ :uniforms, rank_kept_given(named_priorities, :priority, priorities),
            rank_kept_given(named_draws, :draw, draw_uniforms) ].freeze ]
      end

      # The uniforms' entry `part` (a Hash naming items) read by the set's one reading of a caller's Hash keyed by item
      # names (`item_values`; a key that names no item is not read): { item => the number as given }, the Hash it
      # builds, keyed by the set's own Strings, frozen.
      def rank_named(uniforms, part)
        entries = Checks.fetch(uniforms, part, "uniforms")
        unless Hash === entries
          raise ArgumentError, "uniforms' #{part} must be a Hash of item => number (got #{Inputs.shown(entries)})"
        end

        item_values(entries, @items, @item_index, "uniforms' #{part}", strangers: :skip).freeze
      end

      # The given uniforms' entry `part` read for each of `items` (`rank_given`): { item => number }; the first one
      # refused is an ArgumentError.
      def rank_read(named, part, items)
        items.to_h { |item| [ item, rank_given(named, part, item) ] }
      end

      # THE reading of a given uniform, written once (the given-uniforms source and what it keeps for a re-pin): the
      # item's number in `named` (`rank_named`'s Hash), a priority by the library's one number reader (Numbers.read), a
      # draw uniform by the bandit's one [0, 1] bound (Checks.unit); one that is not given is an ArgumentError.
      def rank_given(named, part, item)
        label = "uniforms' #{part} of the item #{Inputs.shown(item)}"
        raise ArgumentError, "#{label} is not given" unless named.key?(item)

        given = named.fetch(item)
        part == :priority ? Numbers.read(given, label) : Checks.unit(given, label)
      end

      # What a re-pin keeps of the given uniforms' entry `part`: { item => the number `rank` read, or the words of a
      # refusal }, for every item, keyed by the set's own Strings, frozen: library data, never a caller's number, so a
      # number changed after `rank` (a caller's Numeric whose to_f answers otherwise, a Hash it changes) moves no
      # re-pin. A free item's is the number `rank` read (`read`). An item `rank` did not read (one the order pinned)
      # keeps a frozen copy that reads exactly as its entry read at rank time, not its number alone. Why: `rank` reads
      # no pinned item's entry, so reading it at rank time and refusing it there would make `rank` refuse a source it
      # takes. So its entry is read at rank time by the one reading (`rank_given`) and its number kept, or, when that
      # reading refuses, the refusal's words in place of the raise; a re-pin answers that number or raises those words
      # (`rank_kept_read`), as `rank` on the source as it was answers or refuses.
      def rank_kept_given(named, part, read)
        @items.to_h { |item| [ item, read.key?(item) ? read.fetch(item) : rank_given_or_words(named, part, item) ] }
          .freeze
      end

      # The item's given number (`rank_given`), or, refused, the refusal's words, frozen.
      def rank_given_or_words(named, part, item)
        rank_given(named, part, item)
      rescue ArgumentError => e
        e.message.dup.freeze
      end

      # A re-pin's reading of what the given-uniforms source kept (`rank_kept_given`'s Hash) for each of `items`:
      # { item => the number `rank` read }; one kept as a refusal's words is that refusal, the first an ArgumentError
      # in the same words.
      def rank_kept_read(kept, items)
        items.to_h do |item|
          given = kept.fetch(item)
          raise ArgumentError, given if String === given

          [ item, given ]
        end
      end

      # A re-pin's draws (`rank_repinned`): the order rule fed an order's own kept [source, keys, draws] under the
      # re-pin's rule, which takes no branch value (shuffle and explore are 0), so the branch is Thompson. A free item
      # the order drew keeps its draw (the key it was compared by and its value); only the others are drawn now, by
      # quantile (`rank_quantile_draw`), from the kept source, which holds only what `rank` read, never a caller's
      # object: the keyed uniforms of the parts as read (`rank_keyed`), or the given priority and draw uniform as read
      # (`rank_kept_read`). Refused as `rank` refuses on that source, before anything is drawn: a Beta parameter below
      # the smallest over the free items, and a given uniform of an item now free that was missing or malformed at rank
      # time, in the words `rank` gave. Kept for the next re-pin: the same source.
      def rank_by_kept(kept, rule, free, leading, eligible)
        source, keys, values = kept
        refuse_below_smallest_parameter(free, computing: RANK_QUANTILE_DRAW)
        uniforms = rank_kept_uniforms(source, free.reject { |item| keys.key?(item) })
        drawn_keys = {}
        drawn_values = {}
        free.each do |item|
          if keys.key?(item)
            drawn_keys[item] = keys.fetch(item)
            drawn_values[item] = values.fetch(item)
          else
            drawn_keys[item], drawn_values[item] = rank_quantile_draw(item, uniforms.fetch(item))
          end
        end
        [ *rank_decided(rank_branch(nil, rule, eligible), free, leading, eligible, nil, drawn_keys, drawn_values),
          source ]
      end

      # { item => draw uniform } for the items a re-pin draws, from the kept source: keyed on the parts as read
      # (`rank_keyed`), or given (`rank_kept_read`: each item's priority, then each item's draw uniform, in the order
      # the given-uniforms source reads them).
      def rank_kept_uniforms(source, items)
        origin, first, second = source
        return rank_keyed(first, "draw", items) if origin == :key

        rank_kept_read(first, items)
        rank_kept_read(second, items)
      end

      # The quantile draws of a key or given uniforms, on the scaled pairs (`rank_quantile_draw`), each compared by its
      # key and answered as its value.
      def rank_by_quantile(branch, free, leading, eligible, priorities, uniforms)
        return rank_decided(branch, free, leading, eligible, priorities, nil, nil) if uniforms.nil?

        keys = {}
        values = {}
        uniforms.each { |item, uniform| keys[item], values[item] = rank_quantile_draw(item, uniform) }
        rank_decided(branch, free, leading, eligible, priorities, keys, values)
      end

      # THE quantile draw of an item at its uniform, written once (a key's, given uniforms' and a re-pin's), on its
      # scaled pair: [the key the rule compares it by, the value it answers]. A Beta item at its position at the uniform
      # (BetaArgmax.position), answered as the value there (the position's half, `rank_beta_half`); a Normal item the
      # set's one Normal draw (`normal_draw`) at the normal quantile of its uniform on the open grid
      # (Draw.open_uniform), compared by its exact key.
      def rank_quantile_draw(item, uniform)
        first, second = @parameters.fetch(item)
        unless @kind == :beta
          return normal_draw(first, second, Significance.normal_quantile(Draw.open_uniform(uniform)))
        end

        position = BetaArgmax.position(uniform, first, second)
        lower, log_end = rank_beta_half(position)
        [ position, lower ? Math.exp(log_end) : 1.0 - Math.exp(log_end) ]
      end

      # THE two halves of a Beta draw's position p (BetaArgmax.position's log scale), written once (the quantile
      # draws' values and the order odds' distribution functions read them): [true, log x] below 0, where x = exp(p +
      # JOINT), else [false, log (1 - x)], where 1 - x = exp(JOINT - p), so each half is read from its own end and no
      # draw near either end rounds into the end itself.
      def rank_beta_half(position)
        position.negative? ? [ true, position + BetaArgmax::JOINT ] : [ false, BetaArgmax::JOINT - position ]
      end

      # The branch at value c (nil when the rule takes none): c < shuffle shuffles; c < shuffle + explore explores
      # when an item is eligible; else Thompson.
      def rank_branch(value, rule, eligible)
        return :thompson if nil.equal?(value)
        return :shuffled if value < rule[:shuffle]
        return :explored if value < rule[:shuffle] + rule[:explore] && !eligible.empty?

        :thompson
      end

      # [head, the other free items in order, branch, draws, keys]: the head by the branch, the others by priority on
      # the shuffled branch and by draw (`keys`, largest first) on the others; draws the values, frozen, none on the
      # shuffled branch; keys as given (nil on the shuffled branch).
      def rank_decided(branch, free, leading, eligible, priorities, keys, values)
        head = case branch
        when :shuffled then rank_first(leading, priorities, :<)
        when :explored then rank_first(eligible, priorities, :<)
        else rank_first(leading, keys, :>)
        end
        rest = head.nil? ? [] : free - [ head ]
        rest = branch == :shuffled ? rank_sorted(rest, priorities, 1) : rank_sorted(rest, keys, -1)
        [ head, rest, branch, (values || {}).freeze, keys ]
      end

      # The first item of `list` (in item order) whose number in `numbers` no later item beats by `beats`.
      def rank_first(list, numbers, beats)
        best = nil
        best_number = nil
        list.each do |item|
          number = numbers.fetch(item)
          next unless best.nil? || number.public_send(beats, best_number)

          best = item
          best_number = number
        end
        best
      end

      # `list` by its numbers, smallest first (direction 1) or largest first (-1), ties to the earlier item.
      def rank_sorted(list, numbers, direction)
        list.sort do |left, right|
          by_number = (numbers.fetch(left) <=> numbers.fetch(right)).to_i * direction
          by_number.zero? ? @item_index.fetch(left) <=> @item_index.fetch(right) : by_number
        end
      end

      # The whole order: pinned items at their positions, the free order filling the free slots in order
      # (`rank_free_slots`).
      def rank_slotted(pinned, free_order)
        return free_order if pinned.empty?

        full = Array.new(@items.size)
        pinned.each { |position, item| full[position] = item }
        rank_free_slots(pinned).zip(free_order) { |position, item| full[position] = item }
        full
      end

      # THE free-slot rule, written once (the order rule's whole order, the order odds' reading of an order and the
      # position odds' table read it): the slots the free ranks take, in order, every slot no item is pinned to, so
      # pinned items keep their positions and each free rank takes the next free slot.
      def rank_free_slots(pinned)
        (0...@items.size).reject { |slot| pinned.key?(slot) }
      end

      # The free head when it has at least min_views views, else nil.
      def rank_leader(head, rule)
        return nil if head.nil?

        eligible?(head, rule[:min_views]) ? head : nil
      end

      # The answer, carrying this set, its rule and an empty cache in instance variables, and, for an Order drawn from
      # a key or given uniforms, what a re-pin reads (`kept`: @rank_source and @rank_keys), then frozen.
      def rank_carrying(answer, rule, kept = nil)
        answer.instance_variable_set(:@posteriors, self)
        answer.instance_variable_set(:@rule, rule)
        answer.instance_variable_set(:@cache, {})
        unless nil.equal?(kept)
          answer.instance_variable_set(:@rank_source, kept.first)
          answer.instance_variable_set(:@rank_keys, kept.last)
        end
        answer.freeze
      end
    end

    # The re-pin's one hand-off from an Order to the set that made it. The set and the Order both include this module,
    # so an Order may call its one protected method on its set (Ruby's protected: a caller whose self includes the
    # module that defines the method), and the set's own private order rule answers: no visibility bypass, and nothing
    # but the set's code reads its private methods.
    module RankRepin
      protected

      # On the set: the Order its rule answers under the held rule (shuffle 0, explore 0, no may_lead, min_views 0,
      # `top` as the order's own rule's) with `pinned`, fed the order's own kept source, keys and draws
      # (`rank_by_kept`); `pinned` read and refused as `rank` reads it (`rank_rule`).
      def rank_repinned(rule, source, keys, draws, pinned)
        held = rank_rule(shuffle: 0.0, explore: 0.0, may_lead: nil, min_views: 0.0, pinned: pinned, top: rule[:top])
        rank_answered(held, *rank_ordered(held, nil, nil, nil, [ source, keys, draws ]))
      end
    end
    private_constant :RankRepin

    class Posteriors
      include RankRepin
    end

    # An order re-pinned on its own draws (THE RE-PIN, above).
    class Order
      include RankRepin

      # The Order this order's set answers on this order's source with `pinned` ({ position => item }, 0-based)
      # under the held rule (shuffle 0, explore 0, no may_lead, min_views 0, top as this order's): `rank(key: <its
      # key>, pinned: pinned, top: <its top>)`, or the same with its given uniforms, bit for bit, computed on the draws
      # this order made (an item it did not draw is drawn now). An ArgumentError for an order drawn from an RNG, one no
      # posterior set made, and whatever `rank` refuses there.
      #
      # Guarantees: the answer is `rank`'s on the same set, source and pins, `==` and every draw; an item this order
      # drew keeps its draw and is not drawn again, so re-pinning an order's own head first draws nothing at all; the
      # source is the one `rank` read (each key part's bytes, each given number), so a caller's object changed after
      # `rank` moves no re-pin. Neither the order nor its set is changed.
      def repin(pinned)
        unless Posteriors === @posteriors && Hash === @rule
          raise ArgumentError, "a #{Inputs.class_name(self)} is re-pinned only when a posterior set made it (it " \
                               "carries the set, the rule and the draws)"
        end
        unless Array === @rank_source && Hash === @rank_keys
          raise ArgumentError, "a re-pin needs an order drawn from a key or given uniforms (this one was drawn " \
                               "from an RNG, whose draws cannot be read again)"
        end

        @posteriors.rank_repinned(@rule, @rank_source, @rank_keys, draws, pinned)
      end
    end
  end
end
