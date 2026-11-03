# frozen_string_literal: true

require_relative "../numbers"
require_relative "../inputs"
require_relative "checks"
require_relative "tallies"

module SkiftetStatistical
  class Bandit
    # Every prior the bandit reads, built in one place from a group's and an item's own evidence. A yes/no prior is a
    # Beta pair [alpha, beta]; an amounts prior is a Prior::Amount (a mean per view, a variance per view and a strength
    # in views; there is no flat amounts prior, because an amount has a scale). Pure: no randomness, no state.
    #
    # Three rules are written once here, and every other prior is built from them:
    # - a mean held at a strength: [strength * mean, strength * (1 - mean)] (`given`);
    # - a mean shrunk toward a prior mean: (strength * prior_mean + outcomes) / (strength + views) (`pooled_mean`, on
    #   the bandit's one shrunk mean, Tallies.shrunk, which the amounts posterior's mean asks too);
    # - the leave-one-out rule: what a group holds once the item's own evidence is taken out (`borrowed`). Views
    #   borrowed are at least 0 and yes/no outcomes borrowed lie in [0, views], because decayed nested sums may round a
    #   child slightly above its parent. `pooled_amount` takes out the same differences but bounds no amount by views.
    #   (HierarchicalRates.leave_one_out_share is a separate, older rule: it refuses negative inputs and never clamps.)
    #
    #   prior = SkiftetStatistical::Bandit::Prior
    #   prior.pooled(group: { views: 1_000, outcomes: 100 }, own: { views: 200, outcomes: 50 }, strength: 20.0)
    #   # => [1.399..., 20.600...]: the borrowed mean (1 + 50) / (2 + 800) held at 2 + 20 views
    #   prior.chain(levels: [ { views: 1_000, outcomes: 120, strength: 20.0 },
    #                         { views: 200, outcomes: 30, strength: 30.0 } ], root_mean: 0.1)
    #   # => [3.365..., 26.634...]: the top level's mean, the lower level taken out, held at the lower level's 30 views
    #
    # A tally is a Hash with Symbol or String keys: `views` and `outcomes`, and for amounts `outcome_squares`; other
    # keys are ignored, and a nil tally or a missing number reads 0 (a missing key reads 0 whatever default the Hash
    # holds: a tally is read by its pairs, never by a default, Checks.named, so one that compares its keys by identity
    # reads as a plain one). Every number is read by the library's one number
    # reader (SkiftetStatistical::Numbers: a finite real number of magnitude at most Numbers::LARGEST; a String, a
    # numeric one included, is not a number), and a pair by its one reading of a pair (Numbers.pair); views, yes/no
    # outcomes, outcome_squares, strengths and weights are >= 0 (an amounts strength above 0; an amount may be below
    # 0). Anything else is an ArgumentError. Answers are Floats, finite ones: every answer passes the bandit's one check
    # of an answer (Checks.finite_answer), which refuses one that left a Float's range (a quotient by a small input;
    # no sum over fewer than 10**208 in-domain inputs and no product the bandit forms of them overflows: Numbers),
    # naming the inputs. A tally is read, yes/no evidence is added to a prior, the moments of amounts are taken and a
    # mean is shrunk by the bandit's tally rules, the ones the posterior set asks; a list (`others`, `levels`) is read
    # by its one list reader (Tallies.list: any Enumerable but a Hash, read once by its each).
    #
    # Each module checks its own bounds: a Beta pair Prior takes (`base`) has both parameters above 0, while the priors
    # Posteriors.yes_no and WeightedPosterior.beta take may hold a 0 (their tallies' evidence is added to it).
    module Prior
      # Beta(1, 1).
      FLAT = [ 1.0, 1.0 ].freeze

      # An amounts prior: k = strength views at mean mu0 = mean, per-view variance v0 = variance. Frozen; the mean is a
      # finite number, the variance and the strength finite and above 0.
      Amount = Struct.new(:mean, :variance, :strength, keyword_init: true) do
        def initialize(mean:, variance:, strength:)
          super(mean: Numbers.read(mean, "mean"),
                variance: Checks.positive(variance, "variance (there is no scale-free amounts prior)"),
                strength: Checks.positive(strength, "strength"))
          freeze
        end
      end

      # The leaf's mean in `chain` is clamped to this by default.
      MEAN_CLIP = [ 0.001, 0.999 ].freeze
      private_constant :MEAN_CLIP

      YES_NO = %i[views outcomes].freeze
      AMOUNTS = %i[views outcomes outcome_squares].freeze
      private_constant :YES_NO, :AMOUNTS

      class << self
        # Beta(1, 1), the frozen FLAT pair: the same frozen Array on every call.
        def flat = FLAT

        # A mean in [0, 1] (the bandit's one [0, 1] bound, Checks.unit) held at a strength >= 0: [strength * mean,
        # strength * (1 - mean)].
        def given(mean:, strength:)
          Checks.finite_answer(centred(Checks.unit(mean, "mean"), Checks.nonnegative(strength, "strength")),
                               "given's mean and strength")
        end

        # An amounts prior of `strength` views at `mean`, per-view variance `variance`.
        def amount(mean:, variance:, strength:)
          prior = Amount.new(mean: mean, variance: variance, strength: strength)
          Checks.finite_answer(prior.to_a, "amount's mean, variance and strength", prior)
        end

        # THE leave-one-out rule: { views:, outcomes: }, the group less the item's own (own nil takes out nothing),
        # views at least 0 and outcomes within [0, views].
        #
        # Takes two yes/no tallies (nil reads as no evidence). Answers Floats. Guarantees: the one rule `pooled` and
        # `pooled_new` borrow by, so a caller never nets an item out by hand; an own tally larger than its group
        # borrows nothing rather than a negative count. Refuses, as an ArgumentError: a tally that is not a Hash or
        # nil, a count below 0, a count that is not a number (a String included) or beyond the input domain.
        def borrowed(group:, own: nil)
          taken = left_out(tally(group, "group"), tally(own, "own"))
          answer = { views: taken[:views], outcomes: taken[:outcomes].clamp(0.0, taken[:views]) }
          Checks.finite_answer(answer.values, "borrowed's group and own", answer)
        end

        # A measured item's prior, from the group's evidence with the item's own taken out (n views and y outcomes
        # borrowed): the mean (base_alpha + y) / (base_alpha + base_beta + n) held at base_alpha + base_beta +
        # min(strength, n). With nothing borrowed it is `base`; while n is at most the strength it is the group's
        # posterior from `base`; above it the borrowed mean stays and its weight stops growing, so a large group never
        # outweighs the item's own evidence for long.
        #
        # Takes: `group` and `own` yes/no tallies (as `borrowed`), `strength` a number >= 0 (views), `base` a pair
        # above 0. Answers [alpha, beta], plain arithmetic, the mean formed exactly and rounded once, so the same
        # inputs answer the same pair bit for bit everywhere. Refuses, as an ArgumentError: a `base` that is not a pair
        # above 0, then a `strength` below 0 or not a number ("strength must be >= 0 (got -1.0)"), whatever the tallies
        # hold, a nil group included; then whatever `borrowed` refuses.
        def pooled(group:, own: nil, strength:, base: FLAT)
          base_alpha, base_beta = pair(base, "base")
          strength = Checks.nonnegative(strength, "strength")
          taken = borrowed(group: group, own: own)
          prior_views = Tallies.exact_sum(Tallies.exact(base_alpha), Tallies.exact(base_beta))
          mean = Tallies.shrunk(Tallies.exact(base_alpha), prior_views, Tallies.exact(taken[:outcomes]),
                                Tallies.exact(taken[:views]))

          Checks.finite_answer(centred(mean, base_alpha + base_beta + [ strength, taken[:views] ].min),
                               "pooled's group, own, strength and base")
        end

        # A new item's prior: the others' borrowed evidence pooled (N views, Y outcomes, each other item's own taken
        # out of its group) into the rate (1 + Y) / (2 + N), held at `weight`, plus the new item's own borrowed
        # evidence (`group` less `own`; a nil group adds nothing), added by the bandit's one yes/no rule. With no
        # others and no group it is Beta(1, 1) at the default weight of 2. `others` is a list, read by the bandit's one
        # list reader (Tallies.list), each a Hash of group: and own:.
        #
        # Answers [alpha, beta], plain arithmetic, the pooled rate formed exactly and rounded once. Refuses, as an
        # ArgumentError: a `weight` below 0 or not a number, `others` that is not a list of Hashes, and whatever
        # `borrowed` refuses of any tally.
        def pooled_new(others:, group: nil, own: nil, weight: 2.0)
          weight = Checks.nonnegative(weight, "weight")
          others = Tallies.list(others, "others", "Hashes of group: and own:")

          views = 0.0
          outcomes = 0.0
          others.each_with_index do |other, index|
            raise ArgumentError, "each of others must be a Hash of group: and own: (got #{Inputs.class_name(other)})" unless Hash === other

            label = "others[#{index}]"
            taken = borrowed(group: Checks.fetch(other, :group, label), own: Checks.fetch(other, :own, label))
            views += taken[:views]
            outcomes += taken[:outcomes]
          end
          own_taken = borrowed(group: group, own: own)

          rate = Tallies.shrunk(Tallies.exact(FLAT[0]), Tallies.exact(FLAT.sum), Tallies.exact(outcomes),
                                Tallies.exact(views))
          Checks.finite_answer(Tallies.yes_no(views: own_taken[:views], outcomes: own_taken[:outcomes],
                                              prior: centred(rate, weight)),
                               "pooled_new's others, group, own and weight")
        end

        # (strength * prior_mean + outcomes) / (strength + views); strength + views must be above 0. Any finite
        # outcomes and prior mean, so it shrinks a mean amount as well as a rate. The bandit's one shrunk mean
        # (Tallies.shrunk), formed exactly, strength * prior_mean included, and rounded once.
        def pooled_mean(views:, outcomes:, prior_mean:, strength:)
          strength = Checks.nonnegative(strength, "strength")
          prior_outcomes = Tallies.exact(strength, Numbers.read(prior_mean, "prior_mean"))
          outcomes = Tallies.exact(Numbers.read(outcomes, "outcomes"))
          mean = Tallies.shrunk(prior_outcomes, Tallies.exact(strength), outcomes,
                                Tallies.exact(Checks.nonnegative(views, "views")))
          Checks.finite_answer([ mean ], "pooled_mean's views, outcomes, prior_mean and strength").first
        end

        # Each multiplier divided by the geometric mean of all of them, then clamped to clip = [low, high]
        # (0 <= low <= high): { key => tilt }, the keys as given. Multipliers are finite and above 0. The division
        # centres the tilts on 1, so they reorder items without raising or lowering them all.
        def tilts(multipliers, clip:)
          raise ArgumentError, "multipliers must be a Hash (got #{Inputs.class_name(multipliers)})" unless Hash === multipliers

          low, high = clip_bounds(clip, unit: false)
          return {} if Inputs.pairs(multipliers).empty?

          values = Inputs.with_values(multipliers) { |multiplier| Checks.positive(multiplier, "a multiplier") }
          center = Math.exp(values.values.sum { |value| Math.log(value) } / values.size)
          answer = values.transform_values { |value| (value / center).clamp(low, high) }
          Checks.finite_answer(answer.values, "tilts' multipliers and clip", answer)
        end

        # The prior of an item at the bottom of nested levels. `levels` lists them top first (a list, read by the
        # bandit's one list reader, Tallies.list), each { views:, outcomes:, strength: } with nested sums (each level's
        # sums include the level below's). Above the leaf, level j's mean
        # p_j = (s_j * p_(j-1) + y) / (s_j + n), n and y the level's sums with the next level taken out (`borrowed`),
        # starting from p_(-1) = root_mean; the leaf answers [s_d * m, s_d * (1 - m)] with m = p_(d-1) * tilt clamped
        # to clip (within [0, 1]). The leaf's own sums only leave the level above; they are the item's evidence, which
        # the posterior adds. One level answers root_mean * tilt, clamped, at that level's strength.
        def chain(levels:, root_mean:, tilt: 1.0, clip: MEAN_CLIP)
          levels = chain_levels(levels)
          mean = Checks.unit(root_mean, "root_mean")
          tilt = Checks.nonnegative(tilt, "tilt")
          low, high = clip_bounds(clip, unit: true)
          levels.each_cons(2) do |level, child|
            taken = borrowed(group: level, own: child)
            mean = pooled_mean(views: taken[:views], outcomes: taken[:outcomes], prior_mean: mean,
                               strength: level[:strength])
          end

          Checks.finite_answer(centred((mean * tilt).clamp(low, high), levels.last[:strength]),
                               "chain's levels, root_mean, tilt and clip")
        end

        # A measured item's amounts prior: the borrowed mean m = outcomes / n and variance outcome_squares / n - m^2
        # (n the views borrowed, a residual below 0 read as 0) at a strength of min(strength, n). Refused, as an
        # ArgumentError, when nothing is borrowed or the variance is 0 (a group with no amounts yet): the caller then
        # states its own scale with `amount`.
        def pooled_amount(group:, own: nil, strength:)
          strength = Checks.positive(strength, "strength")
          taken = left_out(tally(group, "group", amounts: true), tally(own, "own", amounts: true))
          views = taken[:views]
          raise ArgumentError, "nothing to borrow: no views are left once the item's own are taken out" unless views.positive?

          mean, variance = Checks.finite_answer(Tallies.moments(taken[:outcomes], taken[:outcome_squares], views),
                                                "pooled_amount's group and own")
          unless variance.positive?
            raise ArgumentError, "the borrowed amounts have a variance of 0; state a scale with Prior.amount instead"
          end

          amount(mean: mean, variance: variance, strength: [ strength, views ].min)
        end

        private

        # The one centring: a mean held at a weight.
        def centred(mean, weight)
          [ weight * mean, weight * (1.0 - mean) ]
        end

        # Every sum of the group less the item's own, views at least 0.
        def left_out(group, own)
          taken = group.to_h { |key, value| [ key, value - own.fetch(key) ] }
          taken.merge(views: [ taken[:views], 0.0 ].max)
        end

        # A tally's sums as Floats, read by the bandit's one tally reader (Tallies.read), a missing sum read as 0:
        # views and outcome_squares >= 0, yes/no outcomes >= 0, a sum of amounts any number in the domain.
        def tally(given, name, amounts: false)
          sums = Tallies.read(given, name, amounts ? AMOUNTS : YES_NO, amounts: amounts)
          sums.transform_values { |sum| sum || 0.0 }
        end

        # The levels, read once by the bandit's one list reader (Tallies.list), at least one, each a Hash with a
        # strength.
        def chain_levels(given)
          levels = Tallies.list(given, "levels", "Hashes of views:, outcomes: and strength:, top first")
          if levels.empty?
            raise ArgumentError, "levels must be a non-empty list, top first (got #{Inputs.shown(given)})"
          end

          levels.each_with_index.map do |level, index|
            raise ArgumentError, "level #{index} must be a Hash (got #{Inputs.class_name(level)})" unless Hash === level

            strength = Checks.fetch(level, :strength, "level #{index}")
            raise ArgumentError, "level #{index} needs a strength" if nil.equal?(strength)

            tally(level, "level #{index}").merge(strength: Checks.nonnegative(strength, "level #{index}'s strength"))
          end
        end

        # A Beta pair Prior takes, by the library's one reading of a pair (Numbers.pair): both parameters above 0
        # (Prior's own bound; Posteriors.yes_no's priors may hold a 0).
        def pair(given, name)
          Numbers.pair(given, name, %w[alpha beta]) { |number, label| Checks.positive(number, label) }
        end

        # [low, high] by the library's one reading of a pair (Numbers.pair), with 0 <= low <= high, each bound >= 0,
        # or with `unit` in [0, 1] (the bandit's one [0, 1] bound, Checks.unit).
        def clip_bounds(given, unit:)
          low, high = Numbers.pair(given, "clip", %w[low high]) do |number, label|
            unit ? Checks.unit(number, label) : Checks.nonnegative(number, label)
          end
          raise ArgumentError, "clip must have low <= high (got #{Inputs.shown(given)})" if low > high

          [ low, high ]
        end
      end
    end
  end
end
