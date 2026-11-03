# frozen_string_literal: true

require_relative "../numbers"
require_relative "../inputs"
require_relative "../sampler"
require_relative "../beta_argmax"
require_relative "checks"
require_relative "tallies"
require_relative "prior"
require_relative "draw"
require_relative "normal_argmax"

module SkiftetStatistical
  class Bandit
    # A posterior set: items (non-blank Strings, in input order, which is the tie order everywhere), each with a
    # posterior of one kind and its views. Yes/no tallies become Beta pairs [alpha, beta] (`yes_no`), amount tallies
    # Normal pairs [mean, standard deviation] for the mean per view (`amounts`), and pairs may be given as they are
    # (`beta`, `normal`). Any number of items, none included; only the exact odds have limits (EXACT_PICK_MAX,
    # EXACT_ORDER_MAX).
    #
    #   posteriors = SkiftetStatistical::Bandit::Posteriors
    #   set = posteriors.yes_no({ "a" => { views: 40, outcomes: 8 }, "b" => { views: 10, outcomes: 1 } })
    #   set.parameters                    # => { "a" => [9.0, 33.0], "b" => [2.0, 10.0] }
    #   set.greedy                        # => "a", the largest posterior mean
    #   set.values(rng: Random.new(1))    # => one posterior draw per item, in order
    #
    # A tally is a Hash with Symbol or String keys, read by the bandit's one tally reader (Tallies.read), each sum by
    # the library's one number reader (SkiftetStatistical::Numbers: a finite real number of magnitude at most
    # Numbers::LARGEST; a String, a numeric one included, is not a number); a nil tally or a missing sum reads 0,
    # whatever default the Hash holds (Checks.named reads a tally by its pairs, never by a default), and a tally that
    # compares its keys by identity reads as a plain one.
    # Yes/no: `views` and `outcomes` (>= 0, fractional allowed, outcomes clamped to the views), optionally
    # `weighted_views` (sum 1/p), `weighted_outcomes` (sum y/p) and `weighted_views_squared` (sum 1/p^2), each >= 0. The
    # tally is weighted when weighted_views is above 0 and weighted_outcomes is given: the weighted outcomes are
    # rescaled to n, the Kish size (sum w)^2 / sum w^2 (never above the views) when the squared sum is above 0, else the
    # views. Amounts: `views`, `outcomes` (the sum of amounts, any sign) and `outcome_squares`, optionally the weighted
    # sums and `weighted_outcome_squares` (sum x^2/p), weighted only when all three weighted sums are given and
    # weighted_views is above 0. A set's `views` are each tally's own views; the Kish size never replaces them.
    #
    # Priors: `yes_no` takes nil (Beta(1, 1)), one pair [alpha, beta] (each a number >= 0) for every item, or a Hash
    # naming each item's pair; `amounts` takes one Prior::Amount or a Hash naming each item's (there is no flat amounts
    # prior, so nil and a pair are refused), each Amount's numbers read again by its own checks (Tallies.amount_prior).
    # A Hash of priors or views naming a name that is not an item is refused; a Hash of priors must name every item; an
    # item missing from `views` reads 0. Such a Hash, and the given uniforms' priority and draw, is read by its pairs,
    # each key mapped to its item (`item_values`), so one that compares its keys by identity reads as a plain one, and
    # two keys naming one item are refused.
    #
    # Limits, checked when the set is built: every pair is read by the library's one reading of a pair (Numbers.pair),
    # so every parameter, given or computed from a tally, is a number in the input domain (Numbers::LARGEST either side
    # of 0); a Beta pair's two parameters are above 0; a Normal's standard deviation is above 0 (its mean any number in
    # the domain). A pair a tally and a prior compute passes the bandit's one check of an answer first
    # (Checks.finite_answer, naming the item's tally and prior). A Beta pair whose total is above LARGEST_TOTAL is
    # scaled down to it, its mean kept, in `parameters`, the one pair every draw and every odds call reads: the
    # quantile draws and the RNG draws (`values` and the RNG source of `rank` and `pick`), every odds call (exact or by
    # its stated Monte Carlo), `summary`'s P(best) and loss, and `greedy`; the pairs as given feed `summary`'s means and
    # intervals alone. A scaled pair never sums above LARGEST_TOTAL, so a set built from another set's `parameters`
    # keeps them bit for bit. A parameter above 0 but below BetaArgmax::SMALLEST_PARAMETER builds: each computation
    # that cannot read it refuses it on the pairs it reads (the quantile draws and the odds, exact or counted, on
    # `parameters`, `summary`'s interval on the pair as given), and the RNG draws take it. An item name that is not a
    # String (a Symbol, say), is blank, is given twice, holds the same bytes as an earlier name (in another encoding)
    # or holds a NUL byte (the byte a keyed draw joins its parts with) is an ArgumentError, as is anything else a
    # builder cannot read: a keyed draw reads a name by its bytes, so every item's bytes are its own.
    #
    # For the files that reopen this class, private: `given_parameters` ({ item => the pair as given }),
    # `mean_as_given(item)` (the posterior mean on the pair as given), `sampler_draw(sampler, item)` (one draw of the
    # item from a Sampler on its scaled pair, as [the key the order rule compares it by, the value it answers]),
    # `normal_draw(mean, deviation, z)` (THE Normal draw at a standard deviate z, [its exact key, its value], for every
    # source), `mean_of(pair)` (the posterior mean of a pair: alpha / (alpha + beta), or the Normal's mean),
    # `contenders(may_lead, free)` (the items that may lead), `listed_items(names, label, free, none)` (the one reading
    # of a list of item names, each list read once: [the items of `free` it names, or `none`; the items it read], which
    # `contenders`, the order rule's `may_lead` and `pinned`, and `summary` share), `indexed_item(index, name, label)`
    # (the one refusal of a name that is not an item, on the one lookup `item_position(index, name, label)`, both mixed
    # into the class and its instances, as is `item_values(given, items, index, label, strangers:)`, the one reading of
    # a caller's Hash keyed by item names into a Hash keyed by the set's own Strings), `eligible?(item, min_views)` (the
    # one eligibility test), `largest_draw(among, maximum:)` (the one exact P(largest), and E[max], by kind on the
    # scaled pairs), `checked_sampler(rng)` and `refuse_below_smallest_parameter(among, pairs:, computing:)` (GB6's one
    # refusal, before anything is computed, over `among` on the pairs the computation reads, `parameters` unless told,
    # its words naming what is computed), and the @item_index ivar ({ item => index }).
    class Posteriors
      # THE reading of item names a caller gives, written once for both readers of them: a Hash keyed by item names
      # (`item_values`: a builder's priors or views, before the set is built, and the given uniforms' priority and
      # draw) and a set's list of names (`listed_items`: the rule's `may_lead` and `pinned`, `summary`'s `among`), each
      # name looked up among the items by one lookup (`item_position`) and refused by one refusal (`indexed_item`).
      # Mixed into the class and its instances alike, so neither reaches into the other's private methods.
      module ItemNames
        private

        # index[name] when the Hash `index`, keyed by the items, holds the name; else an ArgumentError, "<label>:
        # <name> is not an item". The one refusal of a name that is not an item, on the one lookup (`item_position`).
        def indexed_item(index, name, label)
          found = item_position(index, name, label)
          return found unless found.nil?

          raise ArgumentError, "#{label}: #{Inputs.shown(name)} is not an item"
        end

        # index[name] when the Hash `index`, keyed by the items, holds the name; else nil. THE lookup of a caller's
        # name among the items, written once. A String name is read by the library's one reading of a caller's String
        # (Inputs.string) and its copy looked up, so none of its own methods is called. Any other name is looked up
        # as given when it can be (the library's one reading, Inputs.name?: it answers hash, eql? and inspect, its hash
        # an Integer; a BasicObject, say, cannot, and the items are Strings, so no such name is one), the lookup calling
        # its hash and eql?, one call into the caller's object (the library's one such call, Inputs.asked): a name whose
        # hash or eql? raises is an ArgumentError naming `label`.
        def item_position(index, name, label)
          copy = Inputs.string(name)
          return index[copy] unless nil.equal?(copy)

          Inputs.name?(name, label) ? Inputs.asked(name, "#{label}: a name's lookup") { index[name] } : nil
        end

        # THE reading of a caller's Hash keyed by item names, written once (a builder's priors and views, the given
        # uniforms' priority and draw): { item => value }, a Hash the library builds, keyed by the set's own item
        # Strings (`items`, at the positions `index` holds), from the caller's pairs (Inputs.pairs: Hash's own to_a),
        # each key mapped to its item by the one lookup (`item_position`). Every value is then looked up in this Hash,
        # never in the caller's by a key of the library's, so a Hash that compares its keys by identity
        # (compare_by_identity) reads exactly as a plain one, and a key is read by one reading wherever it is checked
        # and looked up. A key that names no item is refused by the one refusal (`indexed_item`), or, with `strangers:
        # :skip`, not read; two keys that name one item (a Hash that compares by identity, or a key whose own hash
        # differs from its String's, can hold both) are an ArgumentError, "<label> names the item <item> twice".
        def item_values(given, items, index, label, strangers: :refuse)
          read = {}
          Inputs.pairs(given).each do |name, value|
            position = strangers == :skip ? item_position(index, name, label) : indexed_item(index, name, label)
            next if position.nil?

            item = items[position]
            raise ArgumentError, "#{label} names the item #{item.inspect} twice" if read.key?(item)

            read[item] = value
          end
          read
        end
      end
      private_constant :ItemNames
      include ItemNames
      extend ItemNames

      # The most contenders whose pick odds are computed exactly.
      EXACT_PICK_MAX = 20
      # The most free positions whose order and position odds are computed exactly.
      EXACT_ORDER_MAX = 10
      # A Beta pair whose total is above this is scaled down to it, its mean kept, when the set is built.
      LARGEST_TOTAL = 1_000_000.0

      YES_NO_SUMS = %i[views outcomes weighted_views weighted_outcomes weighted_views_squared].freeze
      AMOUNT_SUMS = %i[views outcomes outcome_squares weighted_views weighted_outcomes weighted_outcome_squares
                       weighted_views_squared].freeze
      # The names of a pair's two numbers, by kind.
      PAIR_NAMES = { beta: %w[alpha beta].freeze, normal: [ "mean", "standard deviation" ].freeze }.freeze
      # Half of LARGEST_TOTAL, against which half a pair's total is judged (half of each parameter, summed).
      HALF_LARGEST_TOTAL = LARGEST_TOTAL * 0.5
      # The smallest Float above 0: a scaled parameter never reads less, so a scaled pair stays above 0.
      SMALLEST_SCALED_PARAMETER = 0.0.next_float
      private_constant :YES_NO_SUMS, :AMOUNT_SUMS, :PAIR_NAMES, :HALF_LARGEST_TOTAL, :SMALLEST_SCALED_PARAMETER

      # items: [String], in input order; kind: :beta or :normal; parameters: { item => pair }, a Beta pair above
      # LARGEST_TOTAL scaled to it; views: { item => Float }. All frozen.
      attr_reader :items, :kind, :parameters, :views

      # THE CONTRACT OF A SET, whichever builder made it (README, "Interface contract"; each guarantee's examples are
      # indexed in spec/interface_contract_spec.rb):
      # - A set is a value: `items`, `kind`, `parameters` and `views` are frozen, and no reader (`rank`, `pick`,
      #   `Order#repin`, `summary`, `means`, `values`, `greedy`) writes to the set, so one set may be built once and read
      #   from many requests and threads. (The order and position odds keep a memo on a set its caller has not frozen;
      #   a frozen set computes them each time.)
      # - An item's pair is its own tally's and prior's (or its own given pair) alone: no other item moves it.
      # - The pairs are plain arithmetic, formed as stated and rounded once where the class comment says so, so the same
      #   inputs build the same pairs bit for bit in every process, on every machine and Ruby version.
      class << self
        # Beta posteriors from yes/no tallies, { item => tally }, under `priors` (nil, one pair, or { item => pair }).
        #
        # Takes: `items`, a Hash of item name (a non-blank String, in the tie order) => tally ({ views:, outcomes: },
        # Symbol or String keys, numbers >= 0 and at most Numbers::LARGEST, fractional allowed, optionally the weighted
        # sums); `priors` nil (Beta(1, 1)), one pair [alpha, beta] (each >= 0) or { item => pair } naming every item.
        # Answers: a :beta set whose `parameters` are { item => [alpha + c, beta + n - c] }, c the outcomes clamped to
        # [0, views] (or the weighted outcomes rescaled to the Kish size), a pair above LARGEST_TOTAL scaled to it.
        # Refuses, as an ArgumentError when the set is built: what the class comment lists (a name that is not a
        # usable String, a count below 0 or not a number, a Hash of priors naming a name that is not an item or leaving
        # one out, a pair that is not above 0 once the tally is added).
        def yes_no(items, priors: nil)
          entries = named_entries(items, "items")
          prior_of = prior_per_item(nil.equal?(priors) ? Prior.flat : priors, entries, "priors") do |given, label|
            prior_pair(given, label)
          end
          views = {}
          pairs = entries.to_h do |item, tally|
            label = "item #{item.inspect}'s tally"
            sums = Tallies.read(tally, label, YES_NO_SUMS)
            weighted, squared = Tallies.counted(sums[:weighted_views], [ sums[:weighted_outcomes] ],
                                                sums[:weighted_views_squared])
            views[item] = sums[:views] || 0.0
            pair = Tallies.yes_no(views: views[item], outcomes: sums[:outcomes] || 0.0,
                                  weighted_views: weighted ? sums[:weighted_views] : nil,
                                  weighted_outcomes: weighted ? sums[:weighted_outcomes] : nil,
                                  weighted_views_squared: squared ? sums[:weighted_views_squared] : nil,
                                  prior: prior_of.fetch(item))
            [ item, Checks.finite_answer(pair, "item #{item.inspect}'s tally and prior") ]
          end
          new(:beta, pairs, views)
        end

        # Normal posteriors for the mean per view from amount tallies, { item => tally }, under `priors` (one
        # Prior::Amount, or { item => Prior::Amount }).
        #
        # Takes: tallies { views:, outcomes:, outcome_squares: } (views and outcome_squares >= 0, outcomes the sum of
        # amounts, any sign), optionally the weighted sums. Answers: a :normal set whose `parameters` are { item =>
        # [mean, standard deviation] } of the mean per view. Refuses, as an ArgumentError: as `yes_no` does, and a prior
        # that is not a Prior::Amount (there is no flat amounts prior, so nil and a pair are refused).
        def amounts(items, priors:)
          entries = named_entries(items, "items")
          prior_of = prior_per_item(priors, entries, "priors") { |given, label| Tallies.amount_prior(given, label) }
          views = {}
          pairs = entries.to_h do |item, tally|
            sums = Tallies.read(tally, "item #{item.inspect}'s tally", AMOUNT_SUMS, amounts: true)
            weighted, squared = Tallies.counted(sums[:weighted_views],
                                                [ sums[:weighted_outcomes], sums[:weighted_outcome_squares] ],
                                                sums[:weighted_views_squared])
            views[item] = sums[:views] || 0.0
            pair = Tallies.amounts(views: views[item], outcomes: sums[:outcomes] || 0.0,
                                   outcome_squares: sums[:outcome_squares] || 0.0,
                                   weighted_views: weighted ? sums[:weighted_views] : nil,
                                   weighted_outcomes: weighted ? sums[:weighted_outcomes] : nil,
                                   weighted_outcome_squares: weighted ? sums[:weighted_outcome_squares] : nil,
                                   weighted_views_squared: squared ? sums[:weighted_views_squared] : nil,
                                   prior: prior_of.fetch(item))
            [ item, Checks.finite_answer(pair, "item #{item.inspect}'s tally and prior") ]
          end
          new(:normal, pairs, views)
        end

        # Beta posteriors given as pairs, { item => [alpha, beta] }, with views { item => Float } when given.
        #
        # Takes: `pairs`, a Hash of item name => [alpha, beta], each a finite number above 0 within the input domain;
        # `views` nil (every item 0) or a Hash of item name => views >= 0 (an item it leaves out reads 0). Answers: a
        # :beta set holding the pairs as given (`summary` reads them so), scaled in `parameters` only above
        # LARGEST_TOTAL. Refuses, as an ArgumentError: the names `yes_no` refuses, a parameter that is not a number
        # above 0, and views that name a name that is not an item or are below 0.
        def beta(pairs, views: nil)
          from_pairs(:beta, pairs, views)
        end

        # Normal posteriors given as pairs, { item => [mean, standard_deviation] }, with views when given.
        def normal(pairs, views: nil)
          from_pairs(:normal, pairs, views)
        end

        private

        def from_pairs(kind, pairs, views)
          entries = named_entries(pairs, "pairs")
          read = entries.to_h { |item, pair| [ item, pair_numbers(kind, pair, "item #{item.inspect}") ] }
          new(kind, read, given_views(views, entries))
        end

        # [[item, value], ...] in input order, each item name checked: a String, holding no NUL byte, not blank, not
        # given twice, and holding bytes no earlier name holds (a keyed draw reads a name by its bytes, so two names of
        # one byte string in different encodings would draw alike). Answers frozen names.
        def named_entries(entries, label)
          unless Hash === entries
            raise ArgumentError, "#{label} must be a Hash of item => value (got #{Inputs.class_name(entries)})"
          end

          by_bytes = {}
          Inputs.pairs(entries).map do |name, value|
            item = checked_item_name(name)
            earlier = by_bytes[item.b]
            refuse_same_bytes(earlier, item) unless earlier.nil?

            by_bytes[item.b] = item
            [ item, value ]
          end
        end

        def refuse_same_bytes(earlier, item)
          raise ArgumentError, "the item #{item.inspect} is given twice" if earlier == item

          raise ArgumentError, "the items #{earlier.inspect} (#{earlier.encoding}) and #{item.inspect} " \
                               "(#{item.encoding}) hold the same bytes, which a keyed draw cannot tell apart"
        end

        # The name as a frozen plain String, read by the library's one reading of a caller's String (Inputs.string: a
        # copy by String's own b and encoding, so none of the name's own methods is called, a subclass's or a singleton
        # method's), when it holds no NUL byte and is not blank.
        def checked_item_name(name)
          copy = Inputs.string(name)
          raise ArgumentError, "an item name must be a String (got #{Inputs.shown(name)})" if nil.equal?(copy)

          if copy.b.include?(Draw::SEPARATOR)
            raise ArgumentError, "an item name must not hold a NUL byte, which a keyed draw joins its parts with " \
                                 "(got #{copy.inspect})"
          end
          raise ArgumentError, "an item name must not be blank (got #{copy.inspect})" if blank_name?(copy)

          copy
        end

        # Empty or white space only (Unicode white space in a readable text, ASCII white space in any other bytes).
        def blank_name?(name)
          text = name.encoding.ascii_compatible? && name.valid_encoding? ? name : name.b
          text.match?(/\A[[:space:]]*\z/)
        end

        # { item => prior }: one prior for every item (read once), or a Hash naming every item's and nothing else, read
        # by the one reading of a caller's Hash keyed by item names (`item_values`).
        def prior_per_item(given, entries, label)
          items = entries.map(&:first)
          unless Hash === given
            shared = yield(given, label)
            return items.to_h { |item| [ item, shared ] }
          end

          named = item_values(given, items, items.each_with_index.to_h, label)
          missing = items.find { |item| !named.key?(item) }
          raise ArgumentError, "#{label} names no prior for the item #{missing.inspect}" unless missing.nil?

          items.to_h { |item| [ item, yield(named.fetch(item), "#{label}[#{item.inspect}]") ] }
        end

        # A yes/no prior, by the library's one reading of a pair (Numbers.pair): two numbers, each >= 0 (a 0 is taken
        # here, as WeightedPosterior.beta takes it; Prior's own pairs are above 0).
        def prior_pair(given, label)
          Numbers.pair(given, label, PAIR_NAMES.fetch(:beta)) { |number, name| Checks.nonnegative(number, name) }
        end

        # A pair's two numbers as Floats, by the library's one reading of a pair (Numbers.pair), named by kind.
        def pair_numbers(kind, pair, label)
          Numbers.pair(pair, label, PAIR_NAMES.fetch(kind))
        end

        # { item => Float }: nil reads 0 for every item; a Hash names only items, each views >= 0, an item it does not
        # name reading 0 (read by the one reading of a caller's Hash keyed by item names, `item_values`, from its pairs,
        # so a default the Hash holds is never read).
        def given_views(views, entries)
          items = entries.map(&:first)
          return items.to_h { |item| [ item, 0.0 ] } if nil.equal?(views)
          raise ArgumentError, "views must be a Hash of item => views (got #{Inputs.class_name(views)})" unless Hash === views

          named = item_values(views, items, items.each_with_index.to_h, "views")
          items.to_h do |item|
            given = named.fetch(item, nil)
            next [ item, 0.0 ] if nil.equal?(given)

            [ item, Checks.nonnegative(given, "item #{item.inspect}'s views") ]
          end
        end
      end
      private_class_method :new

      # kind :beta or :normal; given { item => [Float, Float] } with checked names, in input order; views
      # { item => Float }. Each pair is checked here, once, and a Beta pair above LARGEST_TOTAL scaled.
      def initialize(kind, given, views)
        @kind = kind
        @given_parameters = given.to_h { |item, pair| [ item, checked_pair(item, pair).freeze ] }.freeze
        @parameters = @given_parameters.transform_values { |pair| scaled_to_largest_total(pair) }.freeze
        @items = @given_parameters.keys.freeze
        @views = @items.to_h { |item| [ item, views.fetch(item) ] }.freeze
        @item_index = @items.each_with_index.to_h.freeze
        Checks.finite_answer(@given_parameters.values.flatten + @parameters.values.flatten + @views.values,
                             "the parameters and views")
      end

      # One posterior draw per item, in item order, on the scaled pairs (`parameters`, the pairs every draw and every
      # odds call read): Sampler#beta, or mean + sd * Sampler#gaussian, from `rng` and nothing else from it. Takes any
      # parameter above 0.
      def values(rng:)
        sampler = checked_sampler(rng)
        drawn = @items.map { |item| sampler_draw(sampler, item).last }
        Checks.finite_answer(drawn, "the parameters and the rng's answers")
      end

      # The contender with the largest posterior mean (ties to the earlier item); nil for an empty set. `may_lead`
      # names the items that may lead (nil: all; when it names no item, all). The order rule's one choice of a first
      # largest (`rank_first`, ranking.rb) on the means of the scaled pairs (`parameters`, the pairs the keyed draws
      # and every odds call read, so a pick, its odds and the greedy item read one pair), so for Betas it is
      # BetaArgmax.greedy's rule on `parameters`: the first largest alpha / (alpha + beta), by the same quotient and
      # the same strict comparison, for a set of any size. Above LARGEST_TOTAL the scaled mean may differ from the
      # mean as given by a rounding, so two means within a rounding may order differently on the two pairs.
      def greedy(may_lead: nil)
        candidates = contenders(may_lead)
        rank_first(candidates, candidates.to_h { |item| [ item, mean_of(@parameters.fetch(item)) ] }, :>)
      end

      private

      # { item => pair as given }, frozen; `parameters` differs from it only for a Beta pair above LARGEST_TOTAL.
      attr_reader :given_parameters

      # The posterior mean on the pair as given (`mean_of`).
      def mean_as_given(item)
        mean_of(@given_parameters.fetch(item))
      end

      # THE posterior mean of a pair, written once: alpha / (alpha + beta) (BetaArgmax.greedy's quotient; both
      # parameters lie in the input domain, so the total never overflows), or the Normal's mean.
      def mean_of(pair)
        first, second = pair
        @kind == :beta ? first / (first + second) : first
      end

      # THE draw of an item from `sampler`, written once for every RNG draw (`values`, the order rule's RNG source and
      # the odds' Monte Carlo), on its scaled pair (`parameters`), as [the key the order rule compares it by, the value
      # it answers]: a Beta Sampler#beta, its key its value; a Normal mean + sd * Sampler#gaussian (`normal_draw`).
      def sampler_draw(sampler, item)
        first, second = @parameters.fetch(item)
        return normal_draw(first, second, sampler.gaussian) unless @kind == :beta

        value = sampler.beta(first, second)
        [ value, value ]
      end

      # THE Normal draw at the standard deviate z, written once for every source (a quantile draw's z is the standard
      # quantile of its uniform, an RNG draw's Sampler#gaussian): [its key, its value]. The value is mean + sd * z
      # rounded to a Float, the value a draw answers; the key is that sum exact, formed by the bandit's one exact
      # arithmetic on Floats (Tallies.exact, Tallies.exact_sum: a Float is an exact multiple of a power of 2) and held
      # as one Integer on a common power of 2 (Tallies.exact_key), so two draws compare by the exact sign of (mean_i -
      # mean_j) + (sd_i z_i - sd_j z_j), never by their rounded values: where a standard deviation lies at or below a
      # rounding of its mean, the values of two draws round together or out of order, while their keys order them as
      # the posteriors do.
      def normal_draw(mean, deviation, z)
        exact = Tallies.exact_sum(Tallies.exact(mean), Tallies.exact(deviation, z))
        [ Tallies.exact_key(exact), mean + (deviation * z) ]
      end

      # The items of `free` (in their order) that may lead: those `may_lead` names (nil names all); when it names none
      # of them, all of `free`. A `may_lead` name that is not an item is an ArgumentError.
      def contenders(may_lead, free = @items)
        return free if nil.equal?(may_lead)

        listed_items(may_lead, "may_lead", free, free).first
      end

      # THE reading of a list of item names, written once (`contenders`, the order rule's `may_lead` and `pinned`, and
      # `summary`'s `among` read through it). Answers [chosen, read]: chosen the items of `free` (in their order) that
      # the list names, read as a set (a name given twice counts once), or `none` when it names none of them, the one
      # thing its readers differ in; read the items it read, as the set's own Strings in the list's order (a name given
      # twice listed twice), in a frozen Array of the set's own. The list is read by the bandit's one list reader
      # (Tallies.list: any Enumerable but a Hash, read once, by one pass of its each, so a list that answers
      # differently on a second pass, a lazy list or a list whose to_a answers something else is read by that one
      # pass), and only that Array is read and kept, never the caller's object. A `names` that is not a list, or a
      # name that is not an item (the one refusal, `indexed_item`), is an ArgumentError naming `label`.
      def listed_items(names, label, free, none)
        read = Tallies.list(names, label, "items").map { |name| @items[indexed_item(@item_index, name, label)] }.freeze
        named = read.to_h { |item| [ item, true ] }
        chosen = free.select { |item| named.key?(item) }
        [ chosen.empty? ? none : chosen, read ]
      end

      # THE eligibility test, written once for the order rule and the pick odds: the item has at least `min_views` views
      # (it may take the explored head, and a head that has them leads).
      def eligible?(item, min_views)
        @views.fetch(item) >= min_views
      end

      # THE P(largest) by kind, written once for the pick odds, `summary` and the order odds: for the items of `among`
      # (1 to EXACT_PICK_MAX, in its order) on the scaled pairs (`parameters`), [P(the item's draw is the largest) for
      # each, and with `maximum: true` E[the largest draw], else nil], by BetaArgmax or NormalArgmax as the set's kind
      # is :beta or :normal, each by quadrature within its rule's stated bound (BetaArgmax's TAIL a member,
      # NormalArgmax's 2 k (k + 1) TAIL + 2e-12).
      def largest_draw(among, maximum: false)
        pairs = among.map { |item| @parameters.fetch(item) }
        argmax = @kind == :beta ? BetaArgmax : NormalArgmax
        [ argmax.probabilities(pairs), maximum ? argmax.expected_maximum(pairs) : nil ]
      end

      # A Sampler (0.3.0's, as written) on `rng`, which must answer rand (the library's one reading of an rng,
      # Inputs.rng: Kernel#respond_to?, so an object that answers rand only through method_missing is refused), wrapped
      # by Inputs.checked so that each of its answers is checked, as a Float in [0, 1), when it is drawn.
      def checked_sampler(rng)
        Sampler.new(Inputs.checked(rng))
      end

      # THE refusal of 01 GB6, written once for every computation that cannot read a Beta parameter below
      # BetaArgmax::SMALLEST_PARAMETER, each refusing on the pairs it reads (`pairs`: `parameters` unless told), its
      # words naming what it computes (`computing`): the keyed and given-uniform draws of `rank` and `pick` (whatever
      # branch the uniforms choose, the shuffled one included) and every exact probability (the pick odds, the order
      # and position odds, `summary`'s P(best) and loss) on the scaled pairs, over the free items (every item but the
      # pinned ones, whichever of them may lead or are named, so one set is refused or taken alike by every such call
      # under one pinning; for `summary`, every item whenever `among` names one); and `summary`'s interval on the
      # pairs as given, over every item, on every path. An ArgumentError, before anything is computed, when an item of
      # `among` holds a parameter below it there; a parameter at it is taken. A Normal set has no such bound, and the
      # RNG draws and `values` never ask.
      def refuse_below_smallest_parameter(among = @items, pairs: @parameters,
                                          computing: "a quantile draw or an exact probability on the scaled pairs")
        return unless @kind == :beta

        smallest = BetaArgmax::SMALLEST_PARAMETER
        among.each do |item|
          pair = pairs.fetch(item)
          next if pair.min >= smallest

          raise ArgumentError, "item #{item.inspect}'s Beta parameters #{pair.inspect} must be at least #{smallest} " \
                               "for #{computing}"
        end
      end

      # A pair as given or computed, checked by the library's one reading of a pair (Numbers.pair): numbers in the input
      # domain, as Floats; a Beta's both above 0; a Normal's standard deviation above 0.
      def checked_pair(item, pair)
        Numbers.pair(pair, "item #{item.inspect}", PAIR_NAMES.fetch(@kind)) do |number, name, index|
          @kind == :beta || index == 1 ? Checks.positive(number, name) : number
        end
      end

      # A Beta pair whose total is above LARGEST_TOTAL, scaled down to it with its mean kept; every other pair as it
      # is. Half the total (half of each parameter, summed) is judged against half of LARGEST_TOTAL, and the scale is
      # LARGEST_TOTAL / total to the bit. The
      # scaled pair never sums above LARGEST_TOTAL, so a set built from these parameters keeps them as they are: when
      # the two products round to a sum above it, the larger becomes LARGEST_TOTAL less the smaller (a sum of exactly
      # LARGEST_TOTAL; the mean moves by at most a rounding). A product that rounds to 0 reads the smallest Float above 0.
      def scaled_to_largest_total(pair)
        return pair unless @kind == :beta

        alpha, beta = pair
        half_total = (alpha * 0.5) + (beta * 0.5)
        return pair unless half_total > HALF_LARGEST_TOTAL

        scale = HALF_LARGEST_TOTAL / half_total
        alpha = [ alpha * scale, SMALLEST_SCALED_PARAMETER ].max
        beta = [ beta * scale, SMALLEST_SCALED_PARAMETER ].max
        return [ alpha, beta ].freeze unless alpha + beta > LARGEST_TOTAL

        (alpha < beta ? [ alpha, LARGEST_TOTAL - alpha ] : [ LARGEST_TOTAL - beta, beta ]).freeze
      end
    end
  end
end
