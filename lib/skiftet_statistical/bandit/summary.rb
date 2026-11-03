# frozen_string_literal: true

require_relative "../numbers"
require_relative "../inputs"
require_relative "../beta_function"
require_relative "../beta_argmax"
require_relative "../significance"
require_relative "checks"
require_relative "draw"
require_relative "normal_argmax"
require_relative "posteriors"

module SkiftetStatistical
  class Bandit
    # A posterior set's summary: per item, the figures a pass reads, taken from the set itself, so the figures a caller
    # shows and the allocation it hands them to read one posterior set. The answer is Allocator.from_summary's
    # `summary:` as it is (String keys and all), so a pass on exact figures is one call more.
    #
    #   set = SkiftetStatistical::Bandit::Posteriors.beta({ "a" => [ 2.0, 8.0 ], "b" => [ 3.0, 7.0 ] },
    #                                                     views: { "a" => 120.0, "b" => 340.0 })
    #   summary = set.summary(credible_mass: 0.95)
    #   # => { "a" => { views: 120.0, mean: 0.2, lower: 0.0281..., upper: 0.4825..., p_best: 0.2882...,
    #   #               expected_loss: 0.1335... },
    #   #      "b" => { views: 340.0, mean: 0.3, lower: 0.0748..., upper: 0.6000..., p_best: 0.7117...,
    #   #               expected_loss: 0.0335... } }
    #   SkiftetStatistical::Bandit::Allocator.from_summary(arms: %w[a b], summary: summary,
    #                                                      exposures: { "a" => 120, "b" => 340 }, status: "running",
    #                                                      winner_arm: nil, hold_satisfied: false)
    #
    # Every item, in item order, gets:
    # - views: the set's views of the item;
    # - mean, lower and upper: the posterior mean and the equal-tailed `credible_mass` interval, read on the pair as
    #   given, so the interval is the posterior's own, never the scaled pair's, a Beta pair above LARGEST_TOTAL
    #   included. For a Beta the bounds are BetaFunction.quantile (0.2.0's, kept as it is) at (1 - credible_mass) / 2
    #   and at 1 less that, accurate to about 2e-10 up to totals (alpha + beta) near 10**8. Its continued fraction
    #   stops at a fixed number of steps, so beyond a total of about 3 * 10**8 one bound can fall on or past the mean
    #   (measured, at a mean of 1/2: the lower bound lies 0.0025 standard deviations above the mean at 4 * 10**8 and
    #   0.013 at 10**9; at 10**15 the interval spans +1.71 to +3.03 standard deviations from a mean of 1/2 and -3.48 to
    #   +3.48 from a mean of 1/10, where -1.96 to +1.96 is right; at 10**20 both bounds sit 963 to 980 standard
    #   deviations above the mean). For a Normal they are the mean less and plus its standard deviation times the
    #   Normal's quantile at 1 less that tail (Significance.normal_quantile), the same distance on each side, the tail
    #   read on the open grid as every quantile draw reads its uniform (Draw.open_uniform: at least 2**-53, which
    #   moves only the credible mass 1 - 2**-53, whose tail is 2**-54);
    # - p_best and expected_loss, for an item `among` names: P(the item is the largest) and E[the largest] less the
    #   item's mean (the mean answered, on the pair as given), among the items named only, the largest read on the
    #   scaled pairs (`parameters`) by BetaArgmax.probabilities and BetaArgmax.expected_maximum, or NormalArgmax's,
    #   through the set's one P(largest) by kind, which the exact odds call too. A loss that rounds below
    #   0 (a leader far ahead, whose exact loss is below a rounding of its mean) is answered as 0, so
    #   Allocator.from_summary, which refuses a loss below 0, takes every answer. nil for an item `among` does not name.
    #
    # `among`: nil names every item; [] names none, a means-only summary, which computes no P(best) and no loss, for a
    # set of any size; otherwise a list of items, read as the items it names (a name given twice counts once), at most
    # EXACT_PICK_MAX of them. P(best) and the loss are computed in item order, whatever the list's order.
    #
    # Refused, as an ArgumentError before anything is computed: a credible_mass that is not a number above 0 and below
    # 1; an `among` that is not a list, or names something that is not an item; more than EXACT_PICK_MAX items named
    # (nil names every item, so a larger set passes [] or names its items); and, by the set's one refusal, a Beta
    # parameter below BetaArgmax::SMALLEST_PARAMETER on the pairs each computation reads: on the pair as given, of any
    # item, on every path, the means-only summary included, for the interval (BetaFunction.quantile cannot read such a
    # pair: Beta(1e-300, 1) answered both bounds 4.1e-25 around a mean of 1e-300); and in `parameters`, of any item
    # whenever `among` names one, for P(best) and the loss. A pair above LARGEST_TOTAL can be taken by the one and
    # refused by the other: [1, 2 * 10**12] scales to about [5e-7, 10**6], so its means-only summary answers the
    # interval BetaFunction.quantile gives on the pair as given, and a summary naming any item refuses it. A set with
    # no items answers {}. Every figure of the answer passes the bandit's one check of an answer (Checks.finite_answer).
    #
    # `means` answers the mean alone, { item => mean }, as summary answers it (`set.means # => { "a" => 0.2, "b" =>
    # 0.3 }`), with no interval and no quantile (below).
    class Posteriors
      # What each of summary's computations reads, in the words of GB6's refusal.
      SUMMARY_INTERVAL = "the interval, BetaFunction.quantile on the pair as given"
      SUMMARY_BEST = "P(best) and the expected loss on the scaled pairs"
      private_constant :SUMMARY_INTERVAL, :SUMMARY_BEST

      # { item => { views:, mean:, lower:, upper:, p_best:, expected_loss: } }, every item in item order; p_best and
      # expected_loss nil for an item `among` does not name (see above).
      #
      # Takes: `credible_mass` a number above 0 and below 1; `among` nil (every item), [] (none: means and intervals
      # only, for a set of any size) or a list of at most EXACT_PICK_MAX item names. Guarantees:
      # - Exact, never drawn: no key, uniform or RNG is read, so the same set answers the same summary bit for bit on
      #   every call, whatever the process's random state.
      # - An item's views, mean and interval are its own pair's alone (the pair as given), so the item answers them
      #   alike in any set that holds it, a set of its own included; P(best) and the loss read the items `among` names
      #   and no other.
      # - The interval's ends are BetaFunction.quantile's (that function's stated accuracy), or the Normal's mean -+ z
      #   sd; P(best) is BetaArgmax.probabilities' (or NormalArgmax's) quadrature, and the loss is never below 0.
      # Refuses the list above, each an ArgumentError before anything is computed.
      def summary(credible_mass: 0.95, among: nil)
        tail = summary_tail(credible_mass)
        named = summary_named(among)
        refuse_below_smallest_parameter(pairs: given_parameters, computing: SUMMARY_INTERVAL)
        refuse_below_smallest_parameter(computing: SUMMARY_BEST) unless named.empty?

        best = summary_best(named)
        answer = summary_intervals(tail).to_h do |item, (lower, upper)|
          p_best, expected_loss = best[item]
          [ item, { views: @views.fetch(item), mean: mean_as_given(item), lower: lower, upper: upper, p_best: p_best,
                    expected_loss: expected_loss } ]
        end
        Checks.finite_answer(answer.values.flat_map(&:values), "the parameters and the credible mass", answer)
      end

      # { item => mean }, every item in item order, frozen: the posterior mean `summary` answers as :mean, read by
      # summary's own reader on the pair as given (`mean_as_given`), never a second formula, so the two agree bit for
      # bit, a Beta pair above LARGEST_TOTAL included. The means-only reader: it works out no interval, no P(best) and
      # no loss, and draws nothing, so it calls no quantile (BetaFunction.quantile, Significance.normal_quantile),
      # reads no key, uniform or RNG and holds no randomness, for a set of any size; a caller that needs only the means
      # pays for no interval (summary(among: []) works out two quantile bisections an item). Its refusals are
      # summary's for the mean: the bandit's one check of an answer (Checks.finite_answer), which every set as built
      # passes (a Beta mean lies in [0, 1], a Normal's in the input domain). The refusal of a parameter below
      # BetaArgmax::SMALLEST_PARAMETER is the interval's, not the mean's, so such a pair is read here, its mean the
      # pair's own (Beta(1e-300, 1): 1e-300). A set with no items answers {}.
      def means
        answer = @items.to_h { |item| [ item, mean_as_given(item) ] }.freeze
        Checks.finite_answer(answer.values, "the parameters", answer)
      end

      private

      # (1 - credible_mass) / 2, the probability the interval leaves out at each end; credible_mass lies above 0 and
      # below 1 (the bandit's one such bound, Checks.open_unit).
      def summary_tail(credible_mass)
        (1.0 - Checks.open_unit(credible_mass, "credible_mass")) / 2.0
      end

      # The items `among` names, in item order (the set's one list reading): every item for nil, none for a list naming
      # none. At most EXACT_PICK_MAX.
      def summary_named(among)
        named = nil.equal?(among) ? @items : listed_items(among, "among", @items, []).first
        return named if named.size <= EXACT_PICK_MAX

        raise ArgumentError, "summary computes P(best) and the loss exactly among at most #{EXACT_PICK_MAX} items " \
                             "(#{named.size} named); name them in among, or pass among: [] for means and intervals only"
      end

      # { item => [p_best, expected_loss] } among `named`, on the scaled pairs (the set's one P(largest) by kind), the
      # loss E[max] less the mean as given, never below 0; {} for none.
      def summary_best(named)
        return {} if named.empty?

        largest, maximum = largest_draw(named, maximum: true)
        named.each_with_index.to_h do |item, index|
          loss = maximum - mean_as_given(item)
          [ item, [ largest[index], loss.positive? ? loss : 0.0 ] ]
        end
      end

      # { item => [lower, upper] } in item order: the equal-tailed interval leaving `tail` out at each end, on the
      # pair as given.
      def summary_intervals(tail)
        if @kind == :beta
          return given_parameters.transform_values do |alpha, beta|
            [ BetaFunction.quantile(tail, alpha, beta), BetaFunction.quantile(1.0 - tail, alpha, beta) ]
          end
        end

        score = -Significance.normal_quantile(Draw.open_uniform(tail))
        given_parameters.transform_values do |mean, deviation|
          [ mean - (score * deviation), mean + (score * deviation) ]
        end
      end
    end
  end
end
