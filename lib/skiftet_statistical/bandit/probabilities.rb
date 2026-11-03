# frozen_string_literal: true

require_relative "../numbers"
require_relative "../inputs"
require_relative "../beta_argmax"
require_relative "../significance"
require_relative "checks"
require_relative "draw"
require_relative "normal_argmax"
require_relative "posteriors"
require_relative "ranking"

module SkiftetStatistical
  class Bandit
    # The posterior set's odds: the probability of each head a pick answers, of each order and of each item at each
    # position that `rank` answers, all under the rule that made the draw, and the probability that a two-layer draw
    # shows an item.
    #
    #   set = SkiftetStatistical::Bandit::Posteriors.beta({ "a" => [ 1.0, 1.0 ], "b" => [ 2.0, 1.0 ] })
    #   set.pick_probabilities[:heads]                 # => { "a" => 1/3, "b" => 2/3 }
    #   set.pick_probabilities(explore: 0.1)[:heads]   # => { "a" => 0.05 + 0.9 * 1/3, "b" => 0.05 + 0.9 * 2/3 }
    #   set.order_probability(%w[b a])                 # => { probability: 2/3, method: :exact, standard_error: 0.0 }
    #   set.positions[:probabilities]                  # => { "a" => [1/3, 2/3], "b" => [2/3, 1/3] }
    #   set.order_summary(%w[b a])                     # => { probability: 2/3, positions: { "a" => [...], ... } }
    #   set.rank(key: "k1").probability                # => the probability of the order that key drew
    #   SkiftetStatistical::Bandit::Posteriors.mixture(
    #     shares: { "x" => 0.25, "y" => 0.75 },
    #     probabilities: { "x" => { "a" => 0.5, "b" => 0.5 }, "y" => { "a" => 0.2, "c" => 0.8 } }
    #   )                                              # => { "a" => 0.275, "b" => 0.125, "c" => 0.6 }
    #
    # THE PICK (`pick_probabilities`), under the rule `pick` draws by: `shuffle` and `explore` (each in [0, 1], summing
    # to at most 1), `may_lead` (nil: every item) and `min_views` (0). The contenders C are the items `may_lead` names
    # (every item when it is nil or names none), the eligible E the contenders with at least `min_views` views. The
    # head is item c with probability
    #
    #   shuffle / |C| [c in C] + explore / |E| [c in E, E not empty] + (1 - shuffle - explore [E not empty]) P(c)
    #
    # with P(c) the probability that c's draw is the largest among C's: BetaArgmax.probabilities or
    # NormalArgmax.probabilities on the scaled pairs (`parameters`), through the set's one P(largest) by kind, up to
    # EXACT_PICK_MAX contenders, each by its quadrature within its stated bound, not to the last digit: BetaArgmax's
    # within TAIL (1e-10) a member, and NormalArgmax's, each member in its own standardised variable, within
    # 2 k (k + 1) TAIL + 2e-12 of the exact probability for k contenders (1.2e-9 for two, 8.4e-8 for 20). Above
    # EXACT_PICK_MAX contenders P(c) is the stated Monte Carlo below, and the rest of the formula stands as it is. When
    # no contender has `min_views` views the exploration share goes back to the draw, as the pick does. Every source of
    # the pick (a key, given uniforms, an RNG) draws exactly from the posteriors on the scaled pairs, and compares two
    # draws as the posteriors order them (a Beta by its position or its value, a Normal by the exact sign of its mean
    # + sd * z against the other's, ranking.rb), so these are, within that bound, the odds of a pick from any of them.
    # The answer:
    # - `heads`: { item => P(head = item) } for every item, in item order (0.0 for an item that may not lead);
    # - `leaders`: the pick's leader is its head when the head has at least `min_views` views, else nil, so each item
    #   with at least `min_views` views gets its head's probability and nil gets the heads of the items below it
    #   (0.0 when there are none); an empty set answers no heads and a leader nil with probability 1;
    # - `method`: :exact (the quadrature, not a Monte Carlo), and `standard_error`: 0.0; or, above EXACT_PICK_MAX
    #   contenders with a share left to the draw, :monte_carlo and the largest standard error of a head or a leader.
    #
    # THE ORDERS (`order_probability(order)`) and THE POSITIONS (`positions`), under the rule `rank` draws by: the
    # pick's four keywords and `pinned` ({ position => item }) and `top`. Pinned items keep their positions; the n free
    # items fill the other positions in order, each free rank on the next free slot. `order` is a list of distinct
    # items, as `rank` answers them (pinned items at their positions): its probability is that of the rule's whole
    # order beginning with it, so for an order of `top` items (every item when top is nil) it is the probability that
    # `rank` answers it, and `top` itself changes no odds. An order whose pinned slot holds another item, or that puts
    # a pinned item on a free slot, has probability 0; the empty order 1. With s1, ..., sm the free items the order
    # names, in its order, its probability is
    #
    #   shuffle / (n (n - 1) ... (n - m + 1))
    #   + explore [E not empty] [s1 in E] / |E| * T(s2, ..., sm among the free items less s1)
    #   + (1 - shuffle - explore [E not empty]) * T(s1, ..., sm among the free items),
    #
    # T(s1, ..., sm among S) = P(X_s1 > ... > X_sm > every other item of S), and `positions` answers P(item at slot)
    # on the same parts: the shuffled part 1/n at every free rank; the explored part 1/|E| at rank 0 for an eligible
    # item, and at rank l >= 1 the sum over h in E, h != i, of 1/|E| * P(i at rank l - 1 by draw among the free items
    # less h); the drawn part P(i at rank l by draw among the free items). A pinned item is 1 at its own slot.
    #
    # Each is exact, by a quadrature, when no free item is barred from leading (`may_lead` names every free item, or
    # none) and there are at most EXACT_ORDER_MAX free items; otherwise it is the stated Monte Carlo below. A rank by
    # draw at the head is P(largest), the set's one P(largest) by kind (above); every other is a Stieltjes sum against
    # exact distribution increments, each level of a chain and each item's position sums integrated over its own
    # item's points, in a coordinate where that item's own points keep their digits (on the scaled pairs). Each item's
    # points are -infinity, +infinity and the free items' quantiles at the ORDER_LEVELS - 1 evenly spaced levels
    # inside (0, 1) (for Normals also at 0 and 1, below), sorted, each once.
    # - Betas share one grid, in the position of BetaArgmax.position (the key ranking.rb compares their draws by:
    #   log(2 x) up to 1/2, then -log(2 (1 - x))), where a Beta's points keep their digits. Every item's distribution
    #   function is read there by BetaArgmax.distribution (the very function its P(largest) and its draws read), each
    #   half from its own end as BetaArgmax reads its halves, and below 2**-1023 from that end, where a Float holds
    #   fewer digits of x and then none (and where a parameter near SMALLEST_PARAMETER holds nearly all of its mass),
    #   by its leading term, a power of x scaled from BetaArgmax.distribution at 2**-1023.
    # - Normals have no such shared coordinate within the input domain: x itself rounds a narrow item's points
    #   together at a large mean, x less the largest mean rounds two narrow items near 0 together beside a far item,
    #   and a scale by a standard deviation sends a wide item's points past a Float's range. So each item has its own
    #   points in its own standardised variable z = (x - mean) / sd, as NormalArgmax.probabilities integrates each
    #   member: every free item's quantiles, at the levels inside (0, 1) and at 0 and 1 read as a quantile draw's
    #   uniform is (Draw.open_uniform, 2**-53 within the ends), carried into z as ((the other's mean - the item's mean)
    #   + the other's sd * the standard quantile) / the item's sd (the means' difference formed first; a point beyond a
    #   Float's range is left out), and beside each item's carried points the second Float below the lowest and above
    #   the highest, so that an item whose points round together in z rises wholly between two points. The ends keep
    #   an item's own mass below its first quantile on a finite point, where the other items' F hold their values, not
    #   the 0 every F reads at -infinity. Every item's distribution function is read at the item's points as
    #   Phi(((the item's mean - the other's mean) + the item's sd * z) / the other's sd).
    # So the chain, each level on its own item's points,
    #
    #   H(x) = the product of the other items' F (1 when none); then, for k = m down to 1,
    #   H_k(x_j) = H_k(x_(j-1)) + (F_sk(x_j) - F_sk(x_(j-1))) * (H_(k+1)(x_(j-1)) + H_(k+1)(x_j)) / 2
    #
    # gives T = H_1 at the top (an order's last item, when no other is left, is the rest, and a chain of one item is
    # its P(largest)). H_(k+1), held on s_(k+1)'s points, is read at s_k's: on the Betas' one grid as it stands; for
    # Normals at each of s_k's points mapped into s_(k+1)'s z (the means' difference formed first, then plus s_k's sd
    # times z, over s_(k+1)'s sd), linearly in F_s(k+1) between the two of s_(k+1)'s points around it, which keeps
    # it monotone and exact where the point is one of s_(k+1)'s (H rises between two points by the increment of F
    # times a mean of what it integrates, so where F does not move H does not either). P(i at rank l by draw) sums each
    # increment of F_i on i's points times the mean, at the interval's two ends, of the z**l coefficient of the
    # product over j != i of (F_j + z (1 - F_j)) (the explored part the same over the items less h, summed over h).
    # Each sum is taken on the points and on their coarse part (the infinities, the quantiles at every other level and,
    # for Normals, the points beside each item's carried points) and extrapolated, fine + (fine - coarse) / 3
    # (Richardson), which takes out the trapezoid's error that falls as the square of the spacing. Each item's mass is
    # split evenly between its own quantiles, so no item is sampled coarsely, a parameter below 1 included, where a
    # trapezoid of the density reads up to 7e-3 off: at 400 levels an item the orders and positions read within 4e-6
    # of the closed forms of Beta(a, 1) and Beta(1, b) items, which race as Plackett-Luce (800 random sets of three and
    # four, a and b from 1e-6 to 10), three identical items every order 1/n! and every position 1/n within 1e-10 at
    # any scale, two narrow Normals beside a far one (deviations 1e-12 to 1e-7, near 0 or 1e9, the far one 1e6 ahead
    # with a deviation of 1 or 1e10) within 1.2e-6 of their closed forms, random Normal sets of three to five (means
    # up to 5e7 either side of 0, deviations 1e-6 to 1e6) with their positions' rows and columns and their orders by
    # head within 3e-6 of 1 and of the pick's odds, and all of them within 1e-4 of the exact figures (the library's
    # stated tolerance), answered as Floats in [0, 1] (`odds_bounded`, the odds' one clamp).
    #
    # THE MONTE CARLO, stated, wherever an odds call has no exact rule (`draws` and `seed` read first, by the bandit's
    # one reading of a whole number, Checks.integer: draws at least 1, seed any Integer within a Float's range):
    # `draws` draws from the RNG source on Random.new(seed), so the same call answers the same figures, each on the
    # scaled pairs (`parameters`), the one pair the exact odds and every draw (`rank`, `pick` and `values`, from any
    # source) read. For the pick, P(c is C's largest draw) is the share of those draws in which c's draw (one Sampler
    # draw per contender, in item order, the set's one RNG draw) is the largest by the order rule's own choice of a
    # first largest, compared by the key the rule compares (a Normal's exact mean + sd * z); the standard error of a
    # head is its drawn share times sqrt(q (1 - q) / draws). For orders and positions, the rule itself visits `draws`
    # times (the order rule's own ordering, as `rank(rng:)` orders), and each figure is the share of visits that answer
    # it, its standard error sqrt(p (1 - p) / draws). The answer's
    # `method` is :monte_carlo and its `standard_error` the largest of its figures'. A Beta parameter far below 1 draws
    # 0.0 in a share of Sampler's draws (its Gamma draw underflows: about a tenth of them at 0.003), where two such
    # draws tie and the tie goes to the earlier item, so below about 0.003 the Monte Carlo reads that tie rule as well
    # as the posterior. It refuses what the exact odds refuse (below), at every size.
    #
    # THE POSTERIOR ORDER (`order_summary(order)`): the posterior alone, no rule: P(X_s1 > ... > X_sm > every other
    # item), that the order is the best one (T above, over every item), and every item's P(at each position) by draw,
    # exact as above, for a set of at most EXACT_ORDER_MAX items.
    #
    # Choice#probability and Order#probability answer the odds of the head or the order they hold under the rule they
    # were made by (the set, the rule and the cache each carries): `pick_probabilities` of the head (1.0 for no head,
    # an empty set) and `order_probability` of the items, at the default draws and seed where those are a Monte Carlo;
    # computed on the first call and kept.
    #
    # Refused, as ArgumentErrors before anything is computed: `draws` that is not an Integer of at least 1, or `seed`
    # that is not an Integer, within a Float's range; a rule `rank` or `pick` refuses (read by the order rule's one
    # reader); an `order` that is not a list of items or that names an item twice; `order_summary` of more than
    # EXACT_ORDER_MAX items; and, by the set's one refusal, a Beta parameter below BetaArgmax::SMALLEST_PARAMETER on the
    # scaled pairs of any free item (one that may not lead included; for the pick every item), the scope a keyed draw
    # of the same set and rule refuses, at every size: the exact odds and the stated Monte Carlo alike (the pick at any
    # number of contenders, the orders and positions with or without items barred from leading, the posterior order);
    # a parameter at it is taken.
    #
    # THE MIXTURE (`Posteriors.mixture`): a first layer draws branch b with probability shares[b], then a second layer
    # shows item c with probability probabilities[b][c]. Item c is shown with probability
    #
    #   sum over b of shares[b] * probabilities[b][c]
    #
    # (an item a branch does not name counts 0 there), answered as { item => Float } in the order the items are first
    # named, branch by branch. Branches and items may be any keys a Hash can look up (the library's one test of a name,
    # Inputs.name?), read as given (Symbols and Strings are different keys). Every branch of `shares` must have its
    # probabilities, a Hash (empty for a branch that shows nothing), and `probabilities` may name no branch that
    # `shares` does not; every share and probability is a number >= 0 (by the library's one number reader, in the input
    # domain, Numbers::LARGEST), so no product or sum leaves a Float. An ArgumentError otherwise. Every answer passes
    # the bandit's one check of an answer (Checks.finite_answer).
    class Posteriors
      # The quantile levels an item, evenly spaced, at which the free items split the order odds' grid.
      ORDER_LEVELS = 400
      # Half a Float's smallest normal, 2**-1023: at and below it a Beta's distribution function is its leading term.
      ORDER_FLOOR = 2.0**-1023
      ORDER_LOG_FLOOR = Math.log(ORDER_FLOOR)
      # What the odds compute, exact or by their stated Monte Carlo, and the posterior order, in the words of GB6's
      # refusal: one refusal at every size, over the items a keyed draw of the same set and rule refuses.
      ODDS_PICK_EXACT = "the exact pick odds on the scaled pairs"
      ODDS_PICK_COUNTED = "the pick odds' stated Monte Carlo, which refuses what the exact pick odds on the scaled " \
                          "pairs refuse"
      ORDER_EXACT = "the exact order and position odds on the scaled pairs"
      ORDER_COUNTED = "the order and position odds' stated Monte Carlo, which refuses what the exact order and " \
                      "position odds on the scaled pairs refuse"
      ORDER_POSTERIOR = "the posterior order and positions on the scaled pairs"
      private_constant :ORDER_LEVELS, :ORDER_FLOOR, :ORDER_LOG_FLOOR, :ODDS_PICK_EXACT, :ODDS_PICK_COUNTED,
                       :ORDER_EXACT, :ORDER_COUNTED, :ORDER_POSTERIOR

      class << self
        # The probability that a two-layer draw shows each item: { item => sum over branches b of shares[b] *
        # probabilities[b][item] }.
        def mixture(shares:, probabilities:)
          odds_refuse_unpaired(shares, probabilities)
          shown = {}
          Inputs.pairs(shares).each do |branch, given_share|
            share = Checks.nonnegative(given_share, "shares[#{Inputs.shown(branch)}]")
            label = "probabilities[#{Inputs.shown(branch)}]"
            Inputs.pairs(Checks.lookup(probabilities, branch, "probabilities")).each do |item, given|
              odds_key(item, label)
              probability = Checks.nonnegative(given, "#{label}[#{Inputs.shown(item)}]")
              sum = Checks.lookup(shown, item, label, 0.0) + (share * probability)
              Inputs.asked(item, "#{label}: an item's hash") { shown[item] = sum }
            end
          end
          Checks.finite_answer(shown.values, "the shares and probabilities", shown)
        end

        private

        # Both Hashes, every branch of `shares` with a Hash of probabilities, and no branch in `probabilities` that
        # `shares` does not name, each branch looked up by the bandit's one lookup in a caller's Hash (Checks.lookup: a
        # default the Hash holds is never read, and the branch's hash and eql? are asked by Inputs.asked).
        def odds_refuse_unpaired(shares, probabilities)
          raise ArgumentError, "shares must be a Hash of branch => share (got #{Inputs.class_name(shares)})" unless Hash === shares
          unless Hash === probabilities
            raise ArgumentError, "probabilities must be a Hash of branch => { item => probability } " \
                                 "(got #{Inputs.class_name(probabilities)})"
          end

          branches = Inputs.pairs(shares).map(&:first)
          named = Inputs.pairs(probabilities).map(&:first)
          branches.each { |branch| odds_key(branch, "shares") }
          named.each { |branch| odds_key(branch, "probabilities") }
          branches.each do |branch|
            found = Checks.lookup(probabilities, branch, "probabilities")
            next if Hash === found

            raise ArgumentError, "probabilities must name a Hash of item => probability for the branch " \
                                 "#{Inputs.shown(branch)} (got #{Inputs.shown(found)})"
          end
          stranger = named.find { |branch| !Checks.holds?(shares, branch, "shares") }
          raise ArgumentError, "probabilities names the branch #{Inputs.shown(stranger)}, which has no share" unless stranger.nil?
        end

        # A branch or an item as given, when a Hash can look it up (the library's one test of a name, Inputs.name?:
        # it answers hash, with an Integer, eql? and inspect, a hash that raises refused there); else an ArgumentError
        # naming `label`.
        def odds_key(key, label)
          return key if Inputs.name?(key, label)

          raise ArgumentError, "#{label}: #{Inputs.shown(key)} cannot be looked up (a key must answer hash, with an " \
                               "Integer, eql? and inspect)"
        end
      end

      # The odds of the pick under its rule (above): { heads:, leaders:, method:, standard_error: }, exact up to
      # EXACT_PICK_MAX contenders, the stated Monte Carlo of P(largest) above.
      def pick_probabilities(draws: 20_000, seed: 0, shuffle: 0.0, explore: 0.0, may_lead: nil, min_views: 0.0)
        Checks.integer(draws, "draws", minimum: 1)
        Checks.integer(seed, "seed")
        rule = rank_rule(shuffle: shuffle, explore: explore, may_lead: may_lead, min_views: min_views, pinned: {},
                         top: nil)
        candidates = contenders(rule[:may_lead])
        exact = candidates.size <= EXACT_PICK_MAX
        refuse_below_smallest_parameter(computing: exact ? ODDS_PICK_EXACT : ODDS_PICK_COUNTED)
        return { heads: {}, leaders: { nil => 1.0 }, method: :exact, standard_error: 0.0 } if candidates.empty?

        shares = odds_shares(candidates, rule)
        drawn = shares.last
        counted = !exact && drawn.positive?
        largest = if counted
          order_largest_counted(candidates, draws, seed)
        elsif drawn.positive?
          largest_draw(candidates).first
        else
          Array.new(candidates.size, 0.0)
        end
        heads = odds_heads(candidates, rule, shares, largest)
        leaders = odds_leaders(heads, rule[:min_views])
        error = counted ? odds_error(candidates, rule, drawn, largest, draws) : 0.0
        Checks.finite_answer(heads.values + leaders.values << error, "the parameters and the rule",
                             { heads: heads, leaders: leaders, method: counted ? :monte_carlo : :exact,
                               standard_error: error })
      end

      # The probability that the rule's whole order begins with `order` (above): { probability:, method:,
      # standard_error: }, exact without items barred from leading up to EXACT_ORDER_MAX free items, else the stated
      # Monte Carlo.
      def order_probability(order, draws: 20_000, seed: 0, shuffle: 0.0, explore: 0.0, may_lead: nil, min_views: 0.0,
                            pinned: {}, top: nil)
        Checks.integer(draws, "draws", minimum: 1)
        Checks.integer(seed, "seed")
        rule = rank_rule(shuffle: shuffle, explore: explore, may_lead: may_lead, min_views: min_views, pinned: pinned,
                         top: top)
        named = order_named(order, "order")
        free = rank_free(rule)
        exact = order_exact?(rule, free)
        refuse_below_smallest_parameter(free, computing: exact ? ORDER_EXACT : ORDER_COUNTED)
        if exact
          sequence = order_sequence(named, rule[:pinned])
          probability = nil.equal?(sequence) ? 0.0 : order_exact(sequence, free, rule)
          answer = { probability: probability, method: :exact, standard_error: 0.0 }
        else
          probability, error = order_counted(named, rule, draws, seed)
          answer = { probability: probability, method: :monte_carlo, standard_error: error }
        end
        Checks.finite_answer(answer.values_at(:probability, :standard_error), "the parameters and the rule", answer)
      end

      # P(item at each slot) under the rule (above): { probabilities: { item => [Float, one a slot] }, method:,
      # standard_error: }, every item in item order, exact as the orders are, else the stated Monte Carlo. `top` only
      # cuts what `rank` answers, so the table is the whole order's.
      def positions(draws: 20_000, seed: 0, shuffle: 0.0, explore: 0.0, may_lead: nil, min_views: 0.0, pinned: {},
                    top: nil)
        Checks.integer(draws, "draws", minimum: 1)
        Checks.integer(seed, "seed")
        rule = rank_rule(shuffle: shuffle, explore: explore, may_lead: may_lead, min_views: min_views, pinned: pinned,
                         top: top)
        free = rank_free(rule)
        exact = order_exact?(rule, free)
        refuse_below_smallest_parameter(free, computing: exact ? ORDER_EXACT : ORDER_COUNTED)
        if exact
          table = order_slotted(order_ranks(free, rule), free, rule[:pinned])
          error = 0.0
          method = :exact
        else
          table, error = order_counted_positions(rule, draws, seed)
          method = :monte_carlo
        end
        Checks.finite_answer(table.values.flatten << error, "the parameters and the rule",
                             { probabilities: table, method: method, standard_error: error })
      end

      # The posterior alone (above): { probability: P(X_s1 > ... > X_sm > every other item), positions: { item =>
      # [P(at each position by draw)] } }, exact, for a set of at most EXACT_ORDER_MAX items.
      def order_summary(order)
        named = order_named(order, "order")
        if @items.size > EXACT_ORDER_MAX
          raise ArgumentError, "order_summary computes the posterior order exactly among at most #{EXACT_ORDER_MAX} " \
                               "items (the set holds #{@items.size})"
        end

        refuse_below_smallest_parameter(computing: ORDER_POSTERIOR)
        rule = rank_rule(shuffle: 0.0, explore: 0.0, may_lead: nil, min_views: 0.0, pinned: {}, top: nil)
        probability = order_exact(named, @items, rule)
        table = order_slotted(order_ranks(@items, rule), @items, rule[:pinned])
        Checks.finite_answer(table.values.flatten << probability, "the parameters",
                             { probability: probability, positions: table })
      end

      private

      # [eligible ({ item => true }, in item order), the explored share, the drawn share] for the contenders under the
      # rule: the explored share is explore when an item is eligible, else 0 (it goes back to the draw); the drawn
      # share what is left, never below 0. The pick's heads and the orders' parts read their shares here.
      def odds_shares(candidates, rule)
        eligible = candidates.select { |item| eligible?(item, rule[:min_views]) }.to_h { |item| [ item, true ] }
        explored = eligible.empty? ? 0.0 : rule[:explore]
        [ eligible, explored, [ 1.0 - rule[:shuffle] - explored, 0.0 ].max ]
      end

      # { item => P(head = item) } for every item: the shuffled share spread over the contenders, the explored share
      # over the eligible, the drawn share by `largest` (P(largest draw) of each contender, in their order), held to
      # [0, 1] (`odds_bounded`).
      def odds_heads(candidates, rule, shares, largest)
        eligible, explored, drawn = shares
        heads = @items.to_h { |item| [ item, 0.0 ] }
        candidates.each_with_index do |item, index|
          head = (rule[:shuffle] / candidates.size) + (eligible.key?(item) ? explored / eligible.size : 0.0) +
                 (drawn * largest[index])
          heads[item] = odds_bounded(head)
        end
        heads
      end

      # THE clamp of the odds, written once: a probability held to [0, 1], so a sum that rounds below 0 reads 0 and
      # one that rounds above 1 reads 1 (the pick's heads, the orders' chains and parts, the positions' cells).
      def odds_bounded(probability)
        probability.clamp(0.0, 1.0)
      end

      # { item => P(leader = item) } for each item with at least min_views views, then nil => the heads of the rest.
      def odds_leaders(heads, min_views)
        leaders = {}
        without = 0.0
        heads.each do |item, probability|
          if eligible?(item, min_views)
            leaders[item] = probability
          else
            without += probability
          end
        end
        leaders[nil] = without
        leaders
      end

      # The largest standard error of a head or a leader whose P(largest) part is the Monte Carlo's: the drawn share
      # times sqrt(q (1 - q) / draws), q a contender's share of the draws, or for the leader nil the share summed over
      # the contenders below min_views.
      def odds_error(candidates, rule, drawn, largest, draws)
        without = 0.0
        errors = candidates.each_with_index.map do |item, index|
          without += largest[index] unless eligible?(item, rule[:min_views])
          order_binomial_error(largest[index], draws)
        end
        drawn * (errors << order_binomial_error(without, draws)).max
      end

      # The Monte Carlo of P(largest) among the contenders: over `draws` draws from Random.new(seed), one Sampler draw
      # per contender in item order (the set's one RNG draw, `sampler_draw`, on the scaled pairs the exact pick odds
      # read), the share in which each contender's draw is the largest by the order rule's own choice of a first
      # largest (`rank_first`, ties to the earlier item) on the keys the rule compares.
      def order_largest_counted(candidates, draws, seed)
        sampler = checked_sampler(Random.new(seed))
        keys = candidates.to_h { |item| [ item, 0.0 ] }
        counts = candidates.to_h { |item| [ item, 0 ] }
        draws.times do
          candidates.each { |item| keys[item] = sampler_draw(sampler, item).first }
          counts[rank_first(candidates, keys, :>)] += 1
        end
        candidates.map { |item| counts.fetch(item).fdiv(draws) }
      end

      # sqrt(p (1 - p) / draws), a share's standard error over `draws` draws.
      def order_binomial_error(share, draws)
        Math.sqrt([ share * (1.0 - share), 0.0 ].max / draws)
      end

      # The items a caller's order names, read by the set's one reading of a list of item names (`listed_items`), as
      # the set's own Strings in the order's order, frozen; an item named twice is an ArgumentError.
      def order_named(order, label)
        _chosen, named = listed_items(order, label, @items, [])
        seen = {}
        named.each do |item|
          raise ArgumentError, "#{label}: #{Inputs.shown(item)} is named twice" if seen.key?(item)

          seen[item] = true
        end
        named
      end

      # Whether the orders and positions are exact under the rule: no free item barred from leading, and at most
      # EXACT_ORDER_MAX free items.
      def order_exact?(rule, free)
        free.size <= EXACT_ORDER_MAX && contenders(rule[:may_lead], free).size == free.size
      end

      # The free items `named` holds, in its order (each free rank in turn, on the free slots of the free-slot rule,
      # `rank_free_slots`), or nil when the order cannot be the rule's: a pinned slot naming another item, or a pinned
      # item on a free slot.
      def order_sequence(named, pinned)
        return named if pinned.empty?
        return nil if pinned.any? { |slot, item| slot < named.size && named[slot] != item }

        held = pinned.values
        sequence = rank_free_slots(pinned).take_while { |slot| slot < named.size }.map { |slot| named[slot] }
        sequence.any? { |item| held.include?(item) } ? nil : sequence
      end

      # P(the free ranks begin with `sequence`) under the rule, exact (no item barred from leading): the shuffled,
      # explored and drawn parts by the shares the pick odds read (`odds_shares`).
      def order_exact(sequence, free, rule)
        return 1.0 if sequence.empty?

        grid = order_grid(free)
        eligible, explored, drawn = odds_shares(free, rule)
        size = free.size
        probability = 0.0
        if rule[:shuffle].positive?
          arrangements = ((size - sequence.size + 1)..size).reduce(1.0) { |product, count| product * count }
          probability += rule[:shuffle] / arrangements
        end
        if explored.positive? && eligible.key?(sequence.first)
          probability += explored / eligible.size * order_chain(sequence.drop(1), free - sequence.first(1), grid)
        end
        probability += drawn * order_chain(sequence, free, grid) if drawn.positive?
        odds_bounded(probability)
      end

      # T(sequence among `among`) = P(X_s1 > ... > X_sm > every other item of `among`), on the grid: 1 for no item; an
      # order's last item, when no other is left, is the rest; a chain of one item is its P(largest) (the set's one
      # P(largest) by kind); longer chains are the Stieltjes sum on both levels of the grid, each level on its own
      # item's points, reading the level below where it is held (`order_read`), extrapolated.
      def order_chain(sequence, among, grid)
        rest = among - sequence
        if rest.empty?
          rest = sequence.last(1)
          sequence = sequence[0...-1]
        end
        return 1.0 if sequence.empty?
        return order_largest(among)[among.index(sequence.first)] if sequence.size == 1

        fine, coarse = grid.map do |frames|
          deepest = frames.fetch(sequence.last)
          below = order_product(rest.map { |item| deepest.fetch(item) })
          (sequence.size - 1).downto(1) do |rank|
            item = sequence[rank]
            held = order_step(frames.fetch(item).fetch(item), below)
            below = order_read(frames, item, held, sequence[rank - 1])
          end
          head = sequence.first
          order_total(frames.fetch(head).fetch(head), below)
        end
        odds_bounded(order_extrapolated(fine, coarse))
      end

      # H of `item`, held at the item's own points (`held`), read at the points of `reader`: as it stands where both
      # share one grid (Betas); else at each of the reader's points mapped into the item's own z, where the grid holds
      # the item's distribution function F at the reader's points (Phi(((the reader's mean - the item's mean) + the
      # reader's sd * z) / the item's sd), the means' difference formed first), linearly in F between the two of the
      # item's own points around it (`order_interpolated`).
      def order_read(frames, item, held, reader)
        own = frames.fetch(item)
        at = frames.fetch(reader)
        return held if own.equal?(at)

        order_interpolated(own.fetch(item), held, at.fetch(item))
      end

      # `held` (H at the points where F reads `levels`, both in the points' order, F nondecreasing) read at each of
      # `wanted` (F at the points read), linearly in F between the two points around it: the first point whose F is at
      # least the one wanted, and the one before it. Monotone, and exact at a point whose F is one of `levels`: between
      # two points H moves by the increment of F times a mean of what it integrates, so where F does not move H does not
      # either, and any point of a run of equal F reads one H.
      def order_interpolated(levels, held, wanted)
        size = levels.size
        index = 0
        wanted.map do |level|
          index -= 1 while index.positive? && levels[index - 1] >= level
          index += 1 while index < size - 1 && levels[index] < level
          next held[index] if index.zero? || levels[index] <= level

          low = levels[index - 1]
          held[index - 1] + ((held[index] - held[index - 1]) * ((level - low) / (levels[index] - low)))
        end
      end

      # The Stieltjes sum extrapolated from the grid (`fine`) and its every other level (`coarse`): fine + (fine -
      # coarse) / 3, which takes out the trapezoid's error that falls as the square of the spacing (Richardson).
      def order_extrapolated(fine, coarse)
        fine + ((fine - coarse) / 3.0)
      end

      # The pointwise product of the columns (each item's distribution function on the grid).
      def order_product(columns)
        return columns.first if columns.size == 1

        Array.new(columns.first.size) do |point|
          columns.inject(1.0) { |product, column| product * column[point] }
        end
      end

      # THE trapezoid's Stieltjes increment, written once (`order_step`, `order_total` and `order_accumulate` sum it):
      # the integrator's increment over an interval times the mean of the integrand at the interval's two ends.
      def order_trapezoid(increment, before, now)
        increment * (before + now) * 0.5
      end

      # One Stieltjes step: H_k at every grid point, from its item's distribution function `own` and H_(k+1) `below`,
      # 0 at the first point.
      def order_step(own, below)
        size = own.size
        step = Array.new(size, 0.0)
        point = 1
        while point < size
          step[point] = step[point - 1] + order_trapezoid(own[point] - own[point - 1], below[point - 1], below[point])
          point += 1
        end
        step
      end

      # The last Stieltjes step's value at the top of the grid.
      def order_total(own, below)
        order_step(own, below).last
      end

      # { item => [P(item at free rank l), l = 0 ... n - 1] } for the free items under the rule, exact (no item barred
      # from leading): the shuffled part 1/n at every rank, the explored and drawn parts by rank (above).
      def order_ranks(free, rule)
        return {} if free.empty?
        return { free.first => [ 1.0 ] } if free.size == 1

        size = free.size
        eligible, explored, drawn = odds_shares(free, rule)
        grid = order_grid(free)
        largest = order_largest(free)
        sums = order_coefficient_sums(free, grid, eligible, explored.positive?)
        free.each_with_index.to_h do |item, index|
          by_draw, by_explore = sums.fetch(item)
          row = Array.new(size) do |rank|
            probability = rule[:shuffle] / size
            probability += drawn * (rank.zero? ? largest[index] : by_draw[rank]) if drawn.positive?
            if explored.positive?
              probability += explored * order_explored(item, rank, free, eligible, by_explore) / eligible.size
            end
            odds_bounded(probability)
          end
          [ item, row ]
        end
      end

      # |E| times the explored part's P(item at `rank`): 1 at rank 0 for an eligible item; at rank 1 the sum over the
      # eligible h != item of P(item is the largest among the free items less h), the set's one P(largest); deeper, the
      # Stieltjes sum of the coefficient over those h.
      def order_explored(item, rank, free, eligible, by_explore)
        return eligible.key?(item) ? 1.0 : 0.0 if rank.zero?
        return by_explore[rank - 1] unless rank == 1

        eligible.each_key.sum do |head|
          next 0.0 if head == item

          among = free - [ head ]
          order_largest(among)[among.index(item)]
        end
      end

      # { item => [the Stieltjes sums by draw, one a rank, and by explore, one a rank less one] }. At every one of the
      # item's own points, the coefficients of the product over the other free items j of (F_j + z (1 - F_j)) (by draw)
      # and, when `explored`, of the sum over the eligible h of that product without h (by explore); each increment of
      # the item's own F times the mean of the coefficients at its two ends; on both levels of the grid, extrapolated
      # (`order_extrapolated`).
      def order_coefficient_sums(free, grid, eligible, explored)
        free.to_h do |item|
          others = free.reject { |other| other == item }
          flags = others.map { |other| eligible.key?(other) }
          fine, coarse = grid.map do |frames|
            columns = frames.fetch(item)
            order_item_sums(columns.fetch(item), others.map { |other| columns.fetch(other) }, flags, explored)
          end
          sums = fine.zip(coarse).map do |by_fine, by_coarse|
            by_fine.zip(by_coarse).map { |sum, half| order_extrapolated(sum, half) }
          end
          [ item, sums ]
        end
      end

      # One item's sums (`order_coefficient_sums`): `own` its distribution function on its points, `columns` the other
      # items' there, `flags` whether each other is eligible. The products are built factor by factor in place, highest
      # coefficient first, the explored sum before the product it reads (B' = B (f + c z) + [eligible] A,
      # A' = A (f + c z)).
      def order_item_sums(own, columns, flags, explored)
        size = columns.size + 1
        product = Array.new(size, 0.0)
        product_before = Array.new(size, 0.0)
        without = Array.new(size, 0.0)
        without_before = Array.new(size, 0.0)
        by_draw = Array.new(size, 0.0)
        by_explore = Array.new(size, 0.0)
        own.each_index do |point|
          product.fill(0.0)
          product[0] = 1.0
          without.fill(0.0) if explored
          columns.each_with_index do |column, factor|
            below = column[point]
            above = 1.0 - below
            order_add_factor(without, product, factor, below, above, flags[factor]) if explored
            order_multiply(product, factor, below, above)
          end
          if point.positive?
            increment = own[point] - own[point - 1]
            unless increment.zero?
              order_accumulate(by_draw, increment, product_before, product)
              order_accumulate(by_explore, increment, without_before, without) if explored
            end
          end
          product, product_before = product_before, product
          without, without_before = without_before, without
        end
        [ by_draw, by_explore ]
      end

      # `coefficients` (degree `factor`, held in 0..factor) times (below + above z), in place.
      def order_multiply(coefficients, factor, below, above)
        coefficients[factor + 1] = above * coefficients[factor]
        degree = factor
        while degree.positive?
          coefficients[degree] = (below * coefficients[degree]) + (above * coefficients[degree - 1])
          degree -= 1
        end
        coefficients[0] *= below
      end

      # The explored sum B (degree factor - 1) times (below + above z), plus the product A as it stands when the
      # factor's item is eligible, in place.
      def order_add_factor(sum, product, factor, below, above, eligible)
        degree = factor
        while degree >= 0
          value = below * sum[degree]
          value += above * sum[degree - 1] if degree.positive?
          value += product[degree] if eligible
          sum[degree] = value
          degree -= 1
        end
      end

      # Each coefficient's sum plus its trapezoid increment (`order_trapezoid`) over the interval.
      def order_accumulate(sums, increment, before, now)
        index = 0
        size = sums.size
        while index < size
          sums[index] += order_trapezoid(increment, before[index], now[index])
          index += 1
        end
      end

      # The rank table placed on the slots: each free rank on the next free slot (the free-slot rule,
      # `rank_free_slots`), a pinned item 1 at its own slot, every item in item order, a row a slot.
      def order_slotted(ranks, free, pinned)
        slots = rank_free_slots(pinned)
        table = @items.to_h { |item| [ item, Array.new(@items.size, 0.0) ] }
        pinned.each { |slot, item| table.fetch(item)[slot] = 1.0 }
        free.each { |item| ranks.fetch(item).each_with_index { |probability, rank| table.fetch(item)[slots[rank]] = probability } }
        table
      end

      # The grid of the free items' order odds, built once a set and kept, on two levels: [fine, coarse], each { item
      # => { free item => its distribution function F at each of the item's points } }, the coarse level a part of the
      # fine one's points: the infinities, the quantiles at every other level (2/ORDER_LEVELS, 4/ORDER_LEVELS, ...)
      # and, for Normals, the levels 0 and 1 and the points beside each item's carried points. Betas share one grid
      # (`order_beta_grid`), so every item names one Hash; Normals have each item's own (`order_normal_grid`). F is 0
      # at -infinity and 1 at +infinity. The free items' P(largest) is read first, the one guard of what the integrals
      # can read.
      def order_grid(free)
        order_memo(:grid, free) do
          order_largest(free)
          @kind == :beta ? order_beta_grid(free) : order_normal_grid(free)
        end
      end

      # The Betas' one grid, in the position the rule compares their draws by (BetaArgmax.position's log scale,
      # ranking.rb): -infinity, +infinity and every free item's position at the ORDER_LEVELS - 1 levels inside (0, 1)
      # on its scaled pair (where a draw at that uniform compares), each item's F read there
      # (`order_beta_distribution`).
      def order_beta_grid(free)
        tagged = [ [ -Float::INFINITY, true ], [ Float::INFINITY, true ] ]
        free.each do |item|
          alpha, beta = @parameters.fetch(item)
          (1...ORDER_LEVELS).each do |level|
            tagged << [ BetaArgmax.position(level.fdiv(ORDER_LEVELS), alpha, beta), level.even? ]
          end
        end
        points, coarse = order_merged(tagged)
        fine = free.to_h do |item|
          alpha, beta = @parameters.fetch(item)
          [ item, points.map { |position| order_beta_distribution(position, alpha, beta) }.freeze ]
        end.freeze
        thinned = fine.transform_values { |column| column.values_at(*coarse).freeze }.freeze
        [ free.to_h { |item| [ item, fine ] }.freeze, free.to_h { |item| [ item, thinned ] }.freeze ].freeze
      end

      # Each Normal's own points, in its own standardised variable z = (x - mean) / sd on its scaled pair (as
      # NormalArgmax.probabilities integrates each member): -infinity, +infinity, and every free item's quantile at
      # each level 0, 1/ORDER_LEVELS, ..., 1 (the standard quantile at the level read as a quantile draw's uniform is,
      # Draw.open_uniform, so the levels 0 and 1 read the open grid's ends, 2**-53 within them) carried into z, ((the
      # other's mean - the item's mean) + the other's sd * the standard quantile) / the item's sd, the means'
      # difference formed first, where it is finite; beside each item's carried points the second Float below the
      # lowest and above the highest, so an item whose points round together in z rises wholly between two points. An
      # item's own ends keep its mass between -infinity and its first quantile at 2**-53, where the other items' F
      # reads 0 by the formula whatever their scale. Every free item's F is read at those points
      # (`order_normal_column`).
      def order_normal_grid(free)
        scores = (0..ORDER_LEVELS).map do |level|
          Significance.normal_quantile(Draw.open_uniform(level.fdiv(ORDER_LEVELS)))
        end
        own = free.map do |item|
          mean, deviation = @parameters.fetch(item)
          tagged = [ [ -Float::INFINITY, true ], [ Float::INFINITY, true ] ]
          free.each { |other| order_carried(tagged, scores, mean, deviation, @parameters.fetch(other)) }
          points, coarse = order_merged(tagged)
          columns = free.to_h do |other|
            [ other, order_normal_column(points, mean, deviation, @parameters.fetch(other)).freeze ]
          end
          [ columns.freeze, columns.transform_values { |column| column.values_at(*coarse).freeze }.freeze ]
        end
        [ free.zip(own.map(&:first)).to_h.freeze, free.zip(own.map(&:last)).to_h.freeze ].freeze
      end

      # The other item's quantiles carried into the item's z (`order_normal_grid`) by the one map between two Normals'
      # standardised variables (Standardised.carried, the means' difference formed first), each tagged coarse at every
      # other level (the ends included), and the two points beside them, tagged coarse, added to `tagged`.
      def order_carried(tagged, scores, mean, deviation, (other_mean, other_deviation))
        lowest = Float::INFINITY
        highest = -Float::INFINITY
        scores.each_with_index do |score, index|
          z = Standardised.carried(other_mean, other_deviation, score, mean, deviation)
          next unless z.finite?

          tagged << [ z, index.even? ]
          lowest = z if z < lowest
          highest = z if z > highest
        end
        return unless lowest.finite?

        tagged << [ lowest.prev_float.prev_float, true ] << [ highest.next_float.next_float, true ]
      end

      # The other item's F at each of the item's points z: Phi(((the item's mean - the other's mean) + the item's sd *
      # z) / the other's sd), the item's z carried into the other's by the one map (Standardised.carried, the means'
      # difference formed first), so a finite numerator over a standard deviation reads 0 or 1 where the quotient
      # leaves a Float, never a NaN (0 at -infinity, 1 at +infinity).
      def order_normal_column(points, mean, deviation, (other_mean, other_deviation))
        points.map do |z|
          Significance.normal_cdf(Standardised.carried(mean, deviation, z, other_mean, other_deviation))
        end
      end

      # [the points, sorted, each once; the indices of the coarse ones] from [point, coarse?] pairs: a point given more
      # than once is coarse when any of its givings is.
      def order_merged(tagged)
        points = []
        coarse = []
        tagged.sort_by(&:first).each do |point, half|
          if points.empty? || points.last != point
            points << point
            coarse << half
          elsif half
            coarse[-1] = true
          end
        end
        [ points.freeze, coarse.each_index.select { |index| coarse[index] }.freeze ]
      end

      # Beta(alpha, beta)'s distribution function at the position p (x = exp(p + JOINT) below 0, else 1 -
      # exp(JOINT - p): the position's two halves, `rank_beta_half`, as the quantile draws read them), read in each
      # half from its own end, as BetaArgmax reads its two halves: below 0 the item's own at log x = p + JOINT, from 0
      # up 1 less the reflected item's (beta, alpha) at log (1 - x) = JOINT - p (`order_beta_end`). So no mass near
      # either end rounds into the end itself.
      def order_beta_distribution(position, alpha, beta)
        lower, log_end = rank_beta_half(position)
        return order_beta_end(log_end, alpha, beta) if lower

        1.0 - order_beta_end(log_end, beta, alpha)
      end

      # I_w(p, q) at log w = v (v at most JOINT, -infinity included): BetaArgmax.distribution at w = exp(v) where w is
      # at least ORDER_FLOOR, the function BetaArgmax integrates and positions on. Below it, where a Float holds fewer
      # digits of w and then none, the distribution function is its leading term w**p / (p * B(p, q)) to a Float's
      # precision (the next term is about q * w of it, below 1e-301 within the scaled pairs' total), a power of w:
      # BetaArgmax.distribution at ORDER_FLOOR times exp(p * (v - log ORDER_FLOOR)), a factor at most 1.
      def order_beta_end(v, p, q)
        return BetaArgmax.distribution(Math.exp(v), p, q) if v >= ORDER_LOG_FLOOR

        BetaArgmax.distribution(ORDER_FLOOR, p, q) * Math.exp(p * (v - ORDER_LOG_FLOOR))
      end

      # P(largest) of each item of `among` (in its order), the set's one P(largest) by kind, kept once read.
      def order_largest(among)
        order_memo(:largest, among) { largest_draw(among).first.freeze }
      end

      # The value kept on this set under `what` for the items `among` (a frozen copy of the list), computed by the
      # block once (a set the caller froze keeps nothing and computes each time).
      def order_memo(what, among)
        return yield if frozen?

        kept = (@order_kept ||= {})
        key = [ what, among.frozen? ? among : among.dup.freeze ].freeze
        kept.fetch(key) { kept[key] = yield }
      end

      # The rule's visits from the RNG source: `draws` whole orders (the order rule's own ordering, as `rank(rng:)`
      # orders, pinned items at their slots, its Sampler draws on the scaled pairs, `parameters`, the pairs the exact
      # order odds read, compared as the rule compares them) drawn on Random.new(seed), each yielded.
      def order_visits(rule, draws, seed)
        rng = Random.new(seed)
        draws.times { yield rank_ordered(rule, nil, nil, rng).first }
      end

      # [the share of the rule's visits whose whole order begins with `named`, its standard error].
      def order_counted(named, rule, draws, seed)
        length = named.size
        hits = 0
        order_visits(rule, draws, seed) { |whole| hits += 1 if whole.first(length) == named }
        share = hits.fdiv(draws)
        [ share, order_binomial_error(share, draws) ]
      end

      # [{ item => [the share of visits with the item at each slot] }, the largest standard error of those shares].
      def order_counted_positions(rule, draws, seed)
        size = @items.size
        counts = @items.to_h { |item| [ item, Array.new(size, 0) ] }
        order_visits(rule, draws, seed) do |whole|
          whole.each_with_index { |item, slot| counts.fetch(item)[slot] += 1 }
        end
        error = 0.0
        table = counts.transform_values do |row|
          row.map do |count|
            share = count.fdiv(draws)
            error = [ error, order_binomial_error(share, draws) ].max
            share
          end
        end
        [ table, error ]
      end
    end

    # What a Choice or an Order carries from the set that made it (ranking.rb sets it before the freeze): [the set,
    # the rule, the cache]; an ArgumentError for one made by hand, which carries none.
    module OrderCarried
      module_function

      def read(answer)
        posteriors = answer.instance_variable_get(:@posteriors)
        rule = answer.instance_variable_get(:@rule)
        cache = answer.instance_variable_get(:@cache)
        return [ posteriors, rule, cache ] if Posteriors === posteriors && Hash === rule && Hash === cache

        raise ArgumentError, "a #{Inputs.class_name(answer)} answers its probability only when a posterior set made " \
                             "it (it carries the set, the rule and the cache)"
      end
    end
    private_constant :OrderCarried

    # A pick's odds under the rule that made it (probabilities.rb).
    class Choice
      # P(head = this head) under the rule the pick was made by: its set's `pick_probabilities` of the head (at the
      # default draws and seed where they are a Monte Carlo), 1.0 for no head (an empty set). Computed once and kept
      # in the cache it carries.
      def probability
        posteriors, rule, cache = OrderCarried.read(self)
        cache.fetch(:probability) do
          cache[:probability] = head.nil? ? 1.0 : posteriors.pick_probabilities(**rule)[:heads].fetch(head)
        end
      end
    end

    # An order's odds under the rule that made it (probabilities.rb).
    class Order
      # P(the rule's order begins with these items) under the rule the order was made by: its set's
      # `order_probability` of its items (at the default draws and seed where they are a Monte Carlo). Computed once
      # and kept in the cache it carries.
      def probability
        posteriors, rule, cache = OrderCarried.read(self)
        cache.fetch(:probability) do
          cache[:probability] = posteriors.order_probability(items, **rule)[:probability]
        end
      end
    end
  end
end
