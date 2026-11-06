# frozen_string_literal: true

# THE INTERFACE CONTRACT, INDEXED. For each public method an application calls, the guarantees its doc comment and the
# README's "Interface contract" section state (what it takes, what it answers, what holds of the answer, what it
# refuses), each with the library examples that back it, so a caller can trust the method without pinning it. The
# index is data: the examples below check that every method is public, that every cited example is still in its file,
# and that the README lists every method, so a guarantee never loses its backing quietly. A guarantee added to a doc
# comment or the README is added here with its example, or with a new example in the file it belongs to.
#
# A citation is [spec file, words from the first line of the example's or group's description]. "Every Ruby" means the
# CI matrix (Ruby 3.1 to 3.4): the pins are literal values recorded once and never edited, so a pin passing on every
# Ruby is the backing for "the same answer on every Ruby version".
module InterfaceContract
  POSTERIORS = "spec/bandit/posteriors_spec.rb"
  RANK = "spec/bandit/rank_spec.rb"
  PICK = "spec/bandit/pick_spec.rb"
  REPIN = "spec/bandit/repin_spec.rb"
  SUMMARY = "spec/bandit/summary_spec.rb"
  MEANS = "spec/bandit/means_spec.rb"
  PRIOR = "spec/bandit/prior_spec.rb"
  DRAW = "spec/bandit/draw_spec.rb"
  DECAY = "spec/bandit/decay_spec.rb"
  ALLOCATOR = "spec/bandit/allocator_spec.rb"
  SHARES_PINS = "spec/pins/allocator_shares_pins_spec.rb"
  WIN_HOLD = "spec/bandit/win_hold_spec.rb"
  WEIGHTED = "spec/bandit/weighted_posterior_spec.rb"
  JOINT = "spec/joint_beta_spec.rb"
  BETA = "spec/beta_function_spec.rb"
  GAMMA = "spec/gamma_spec.rb"
  SIGNIFICANCE = "spec/significance_spec.rb"
  INPUTS = "spec/inputs_spec.rb"
  PINS = "spec/pins/generic_bandit_pins_spec.rb"
  PROBABILITIES = "spec/bandit/probabilities_spec.rb"
  INTERVALS = "spec/beta_function_interval_spec.rb"
  SHARED_PRIOR = "spec/shared_credit_prior_spec.rb"

  SET_IS_A_VALUE = [ POSTERIORS, "LO1 is frozen data that no reader changes" ].freeze
  NEGATIVE_COUNT = [ INPUTS, "which amount sum may be below 0 is declared once, for every reader of a tally" ].freeze
  BEYOND_DOMAIN = [ INPUTS, "every new method refuses a number beyond the input domain where it reads it" ].freeze

  # method => [[guarantee, [citation, ...]], ...]
  INDEX = {
    "Bandit::Posteriors.yes_no" => [
      [ "answers Beta(a0 + c, b0 + n - c), c the outcomes clamped to [0, views], flat by default",
        [ [ POSTERIORS, "LO1 is Beta(a0 + c, b0 + n - c) on the raw counts, flat by default" ],
          [ POSTERIORS, "LO1 reads Symbol and String keys alike and clamps outcomes to the views" ] ] ],
      [ "takes one prior pair for every item, or one an item",
        [ [ POSTERIORS, "LO1 takes one prior pair for every item, or one per item" ] ] ],
      [ "rescales weighted outcomes to the Kish size, never above the views",
        [ [ POSTERIORS, "LO1 rescales to the Kish size only with the squared sum, never above the views" ],
          [ POSTERIORS, "LO1 judges the raw counts when the weighted views are 0" ] ] ],
      [ "reads a tally by WeightedPosterior's rule (one rule)",
        [ [ POSTERIORS, "LO1 equals WeightedPosterior.beta on 500 random integer tallies" ] ] ],
      [ "keeps the input order, the tie order; any number of items, none included",
        [ [ POSTERIORS, "LO1 items keep their input order, the tie order" ],
          [ POSTERIORS, "LO1 60 items build a set (no cap), and so does none" ] ] ],
      [ "scales a pair above LARGEST_TOTAL in parameters, its mean kept; a set built from parameters keeps them",
        [ [ POSTERIORS, "LO1 a Beta pair above LARGEST_TOTAL is scaled to it in parameters" ],
          [ POSTERIORS, "LO1 (LB1) a set built from another set's parameters keeps them bit for bit" ] ] ],
      [ "is a value: frozen, and no reader writes to it", [ SET_IS_A_VALUE ] ],
      [ "refuses a name that is not a usable String",
        [ [ POSTERIORS, "LO1 a Symbol, blank or repeated item is an ArgumentError, whichever builder" ],
          [ POSTERIORS, "LO1 (LB1) an item name holding a NUL byte is an ArgumentError" ],
          [ POSTERIORS, "LO1 (LB1) a name holding the same bytes as an earlier one in another encoding" ] ] ],
      [ "refuses a count below 0, one that is not a number, one beyond the input domain",
        [ NEGATIVE_COUNT, BEYOND_DOMAIN,
          [ INPUTS, "every new method refuses a Numeric the one reader cannot read, as an ArgumentError" ] ] ],
      [ "refuses a Hash of priors that names a stranger",
        [ [ INPUTS, "a Hash compared by identity, keyed by the item Hash's own Strings, reads as a plain one" ],
          [ RANK, "the L3 barrier, round 2: words a name that is not an item alike wherever it is named" ] ] ]
    ],
    "Bandit::Posteriors.beta" => [
      [ "holds the pairs as given, with views when given",
        [ [ POSTERIORS, "LO1 take pairs as they are, with views when given" ] ] ],
      [ "refuses a parameter that is not a finite number above 0",
        [ [ POSTERIORS, "LO1 a parameter not finite or not above 0 is an ArgumentError when the set is built" ] ] ],
      [ "is a value: frozen, and no reader writes to it", [ SET_IS_A_VALUE ] ]
    ],
    "Bandit::Posteriors.amounts" => [
      [ "answers Normal(mean, sd) for the mean per view under a Prior::Amount",
        [ [ POSTERIORS, "LO1 is Normal(mu_n, sd_n) for the mean per view under k views at mu0 and v0" ] ] ],
      [ "refuses a flat prior or none", [ [ POSTERIORS, "LO1 takes one amount prior or one per item, and refuses flat" ] ] ]
    ],
    "Bandit::Posteriors#parameters" => [
      [ "is the one pair every draw and every odds call reads, scaled above LARGEST_TOTAL",
        [ [ POSTERIORS, "LO1 a Beta pair above LARGEST_TOTAL is scaled to it in parameters" ],
          [ POSTERIORS, "LO1 (LB1) a Beta pair at the input domain's bound builds, scaled in parameters" ] ] ],
      [ "is frozen and the same after every reader", [ SET_IS_A_VALUE ] ]
    ],
    "Bandit::Posteriors#rank" => [
      [ "orders by the stated branches: shuffled, explored, Thompson",
        [ [ RANK, "LO2 shuffles below shuffle: the contender of smallest priority leads" ],
          [ RANK, "LO2 explores below shuffle + explore: the eligible item of smallest priority leads" ],
          [ RANK, "LO2 takes the contender of largest draw at or above shuffle + explore" ],
          [ RANK, "LO2 compares the branch value strictly" ] ] ],
      [ "may_lead and min_views choose the head and the leader only",
        [ [ RANK, "LO2 an item barred from leading is never the explored head" ],
          [ RANK, "LO2 falls to Thompson when the explore branch has no eligible item" ],
          [ RANK, "LO2 min_views limits only the exploration and the leader" ],
          [ RANK, "LO2 reads a min_views below 0 as none" ] ] ],
      [ "a key's uniforms are Draw.uniform_for of its parts, so a key draws the same Order in every process",
        [ [ RANK, "LO2 draws a key's uniforms as the helpers state" ],
          [ RANK, "LO2 gives the same orders in a fresh process" ],
          [ RANK, "LO2 answers the keyed draws as the Beta quantiles at the keyed draw uniforms" ] ] ],
      [ "the same Order on every Ruby: orders, leaders and branches exact, draws within 1e-9",
        [ [ PINS, "answers as the rule's first build did" ] ] ],
      [ "an added item moves no other item's order; one may_lead leaves out moves neither order, leader nor branch",
        [ [ RANK, "LO2 an added item never changes the others' relative order over 1 000 keys at shuffle 0.1" ],
          [ RANK, "LO2 an added item absent from may_lead never changes the others' order, the leader or the branch" ] ] ],
      [ "an RNG gives up exactly a branch value, the priorities its branch reads and one draw a free item",
        [ [ RANK, "LO2 on the Thompson branch takes one branch value, then one Sampler draw per free item" ],
          [ RANK, "LO2 on the explored branch takes the branch value, a priority per free item" ],
          [ RANK, "LO2 on the shuffled branch takes the branch value and a priority per free item" ],
          [ RANK, "LO2 takes no branch value when shuffle + explore is 0" ],
          [ RANK, "LO2 takes nothing for a pinned item" ] ] ],
      [ "every source draws exactly from the posteriors, so the odds describe each",
        [ [ PICK, "LO2 equals BetaArgmax.pick on the same keyed uniforms over 1 000 keys at explore 0.1" ],
          [ PROBABILITIES, "LO3 (LC4) answers the heads under explore 0.1 as BetaArgmax.probabilities" ] ] ],
      [ "ties go to the earlier item", [ [ RANK, "LO2 breaks a tie of draws to the earlier item" ] ] ],
      [ "pinned items keep their positions and draw nothing; top answers the first items",
        [ [ RANK, "LO2 keeps a pinned item at its position, drawing nothing for it" ],
          [ RANK, "LO2 answers the first top items of the whole order and keeps its leader" ] ] ],
      [ "answers a frozen Order; any number of items",
        [ [ RANK, "LO2 is a frozen keyword Struct of items, leader, branch and draws" ],
          [ RANK, "LO2 ranks 60 items under every source" ], [ RANK, "LO2 answers no items for an empty set" ] ] ],
      [ "the gated row and the list with a random head, in generic words",
        [ [ RANK, "LO2 a gated row: pooled priors, a new item barred from leading, min_views and explore over keys" ],
          [ RANK, "LO2 a list with a random head: explore 0.1 over every item" ] ] ],
      [ "refuses before any draw: the rule, the names, the sources, a parameter below the smallest",
        [ [ RANK, "LO2 refuses a pinned position outside the items" ], [ RANK, "LO2 refuses one name pinned twice" ],
          [ RANK, "LO2 refuses top below 1" ], [ RANK, "LO2 refuses shuffle or explore outside [0, 1], or summing above 1" ],
          [ RANK, "LO2 refuses no source and two sources" ],
          [ RANK, "LO2 (LC1) refuses under a key or given uniforms a parameter below BetaArgmax::SMALLEST_PARAMETER" ],
          [ INPUTS, "a key part whose to_s answers no String is refused, before any draw" ] ] ],
      [ "an RNG's answer outside [0, 1), or a raise from its rand, is an ArgumentError at that answer",
        [ [ INPUTS, "the set's RNG draws check every answer as it is drawn" ],
          [ INPUTS, "a caller's method that raises is an ArgumentError naming the input" ] ] ],
      [ "writes nothing to the set", [ SET_IS_A_VALUE ] ]
    ],
    "Bandit::Posteriors#pick" => [
      [ "is rank's head under every source, an RNG giving up what rank takes",
        [ [ PICK, "LO2 pick is rank's head" ], [ PICK, "LO2 under a key" ],
          [ PICK, "LO2 under an RNG, taking from the stream exactly what rank takes" ] ] ],
      [ "item is the head with min_views, else nil",
        [ [ PICK, "LO2 answers item nil when the head has fewer than min_views, and still the head" ] ] ],
      [ "one pick from a key read from the caller's stream",
        [ [ PICK, "LO2 one pick: a flat prior, no exploration, a key read from the caller's stream" ] ] ],
      [ "answers a frozen Choice; nil for an empty set",
        [ [ PICK, "LO2 is a frozen keyword Struct of item, head, branch and draws" ],
          [ PICK, "LO2 answers item nil, head nil and no draws for an empty set" ],
          [ PICK, "LO2 picks among 60 items under every source" ] ] ],
      [ "refuses what rank refuses",
        [ [ PICK, "LO2 refuses no source and two sources" ], [ PICK, "LO2 refuses a may_lead name that is not an item" ],
          [ PICK, "LO2 refuses shuffle or explore outside [0, 1], or summing above 1" ] ] ]
    ],
    "Bandit::Order#repin" => [
      [ "answers rank on the same source with the pins under the held rule, bit for bit",
        [ [ REPIN, "GB9 answers rank on the same key or uniforms under the held rule, bit for bit" ],
          [ REPIN, "GB9 answers rank for every pinned map of one and of two positions" ],
          [ REPIN, "GB9 answers a frozen Order under the held rule, with the first order's top" ] ] ],
      [ "draws only what the order did not draw; a held head draws nothing",
        [ [ REPIN, "GB9 draws by quantile only the items the first order did not draw" ],
          [ REPIN, "GB9 holds a head: the re-pin of a keyed order with its own head first draws nothing at all" ] ] ],
      [ "reads the source as rank read it, never the caller's objects again",
        [ [ REPIN, "GB9 keeps the source as the set read it" ], [ REPIN, "GB9 reads no key part again" ] ] ],
      [ "refuses an order drawn from an RNG, one no set made, and what rank refuses",
        [ [ REPIN, "GB9 refuses an order drawn from an RNG" ], [ REPIN, "GB9 refuses an Order no posterior set made" ],
          [ REPIN, "GB9 refuses a malformed pinned map with rank's own words" ] ] ]
    ],
    "Bandit::Posteriors#summary" => [
      [ "answers views, mean, interval, p_best and expected loss per item, in from_summary's shape",
        [ [ SUMMARY, "LO3 (LC2) answers every item its views, mean, interval, p_best and expected loss" ],
          [ SUMMARY, "LO3 (LC2) is taken by Allocator.from_summary as it is" ] ] ],
      [ "the interval is the exact equal-tailed quantiles on the pair as given, at credible_mass",
        [ [ SUMMARY, "LO3 (LC2) gives each item its views, its mean and its exact equal-tailed interval" ],
          [ SUMMARY, "LO3 (LC2) reads credible_mass: 0.9 at the 0.05 and 0.95 quantiles" ],
          [ SUMMARY, "LO3 (LC2) reads the interval on the pair as given above LARGEST_TOTAL" ] ] ],
      [ "exact: the same answer under any seed; an item's mean and interval its own pair's in any set",
        [ [ SUMMARY, "LO3 is exact and per item" ] ] ],
      [ "p_best and the loss among the items named only, p_best within 1e-12 of BetaArgmax, the loss never below 0",
        [ [ SUMMARY, "LO3 (LC2) takes p_best and the loss among the items named, and gives the others neither" ],
          [ SUMMARY, "LO3 (LC2) answers p_best as BetaArgmax.probabilities within 1e-12" ],
          [ SUMMARY, "LO3 (LC2) never answers an expected loss below 0" ] ] ],
      [ "among: [] is means and intervals only, for any number of items",
        [ [ SUMMARY, "LO3 (LC2) answers a means-only summary with among: [], views, mean and interval for 60 items" ] ] ],
      [ "refuses a stranger in among, a credible mass outside (0, 1), a parameter below the smallest",
        [ [ SUMMARY, "LO3 (LC2) refuses among naming an item the set does not hold" ],
          [ INPUTS, "a set refuses a name that is no name, and a credible mass outside (0, 1)" ],
          [ SUMMARY, "the L3 barrier: refuses a parameter below BetaArgmax::SMALLEST_PARAMETER on every path" ] ] ]
    ],
    "Bandit::Posteriors#means" => [
      [ "answers summary's mean bit for bit, a frozen Hash in item order",
        [ [ MEANS, "L7 answers summary(among: [])'s mean bit for bit" ],
          [ MEANS, "L7 answers a frozen Hash of Floats keyed by the set's own items" ],
          [ MEANS, "L7 reads the pair as given above LARGEST_TOTAL" ] ] ],
      [ "computes no quantile and reads no randomness", [ [ MEANS, "L7 timing: costs no quantile" ] ] ]
    ],
    "Bandit::Prior.flat" => [
      [ "is Beta(1, 1), one frozen pair", [ [ PRIOR, "LO1 flat is Beta(1, 1), one frozen pair" ] ] ]
    ],
    "Bandit::Prior.borrowed" => [
      [ "is the group less the item's own, views at least 0 and outcomes within [0, views]",
        [ [ PRIOR, "LO1 is the group less the item's own evidence, in Floats" ],
          [ PRIOR, "LO1 clamps: views at least 0, outcomes within [0, views]" ] ] ],
      [ "refuses a negative count and a number beyond the input domain", [ NEGATIVE_COUNT, BEYOND_DOMAIN ] ]
    ],
    "Bandit::Prior.pooled" => [
      [ "is the base with nothing borrowed, the group's posterior below the strength, the borrowed mean above it",
        [ [ PRIOR, "LO1 with no group views is the base" ], [ PRIOR, "LO1 below the strength is the group's posterior" ],
          [ PRIOR, "LO1 above the strength keeps the borrowed mean" ] ] ],
      [ "composes with pooled_new and the posterior set as stated",
        [ [ POSTERIORS, "LO1 a gated row's priors: pooled for the measured items, pooled_new for a new one" ] ] ],
      [ "refuses a negative strength, with or without a group", [ [ PRIOR, "LO1 refuses a negative strength" ] ] ]
    ],
    "Bandit::Prior.pooled_new" => [
      [ "is the others' pooled borrowed rate at its weight, plus the item's own borrowed evidence",
        [ [ PRIOR, "LO1 with no others and no group is Beta(1, 1)" ],
          [ PRIOR, "LO1 is the others' pooled borrowed rate (1 + Y) / (2 + N) at its weight" ],
          [ PRIOR, "LO1 borrows each other item before pooling" ] ] ]
    ],
    "Bandit::Draw.seed_for" => [
      [ "is the first 8 bytes of SHA256 of the parts joined by NUL: the same Integer in every process and Ruby",
        [ [ DRAW, "LO1 seed_for is the same 8 bytes read as one Integer below 2**64" ],
          [ DRAW, "LO1 uniform_for reads nil as an empty part and a number by its decimal text" ] ] ],
      [ "keeps part lists apart, and refuses a part holding a NUL byte or a to_s that answers no String",
        [ [ DRAW, "LO1 the NUL between parts keeps part lists apart" ],
          [ DRAW, "LO1 a part holding a NUL is refused, as an ArgumentError" ],
          [ INPUTS, "a key part whose to_s answers no String is refused, before any draw" ] ] ]
    ],
    "Bandit::Draw.uniform_for" => [
      [ "is the seed's top 53 bits over 2**53, in [0, 1), the same in every process and Ruby",
        [ [ DRAW, "LO1 uniform_for is the uniform of the parts joined by NUL" ],
          [ DRAW, "LO1 a part list's uniform is its seed's top 53 bits over 2**53" ] ] ]
    ],
    "Bandit::Draw.from_weights" => [
      [ "answers the first counted key whose running sum reaches uniform * total",
        [ [ DRAW, "LO1 from_weights answers the first key whose running sum passes uniform * total" ],
          [ DRAW, "LO1 a uniform of 1.0 answers the last positive key" ] ] ],
      [ "counts a weight that is not a finite number above 0 as 0, and all such as equal",
        [ [ DRAW, "LO1 a weight that is not a finite positive number counts 0" ],
          [ DRAW, "LO1 weights that all count 0 count as equal" ] ] ],
      [ "draws each key with probability weight / total", [ [ DRAW, "LO1 from_weights over 10^5 uniforms" ] ] ],
      [ "answers keys as given and never reads them",
        [ [ DRAW, "LO1 keys are answered as given, whatever hashable objects they are, and never read" ] ] ],
      [ "answers nil for no weights; refuses weights that are not a Hash, a uniform outside [0, 1], a weight beyond the domain",
        [ [ DRAW, "LO1 answers nil for an empty Hash, and refuses weights that are not a Hash" ], BEYOND_DOMAIN ] ]
    ],
    "Bandit::Decay.weight" => [
      [ "is 2 ** (-age / half_life), 0.0 past the cap", [ [ DECAY, "LO1 halves at one half-life" ],
                                                         [ DECAY, "LO1 is 0.0 beyond the cap" ] ] ],
      [ "refuses a negative age", [ [ DECAY, "LO1 refuses a negative age, as an ArgumentError" ] ] ]
    ],
    "Bandit::Decay.fold" => [
      [ "sums every sum at its row's weight, linear in the rows",
        [ [ DECAY, "LO1 sums views, outcomes and outcome squares each at its row's weight" ],
          [ DECAY, "LO1 is one formula: each row at weight's weight" ] ] ]
    ],
    "Bandit::Allocator::Config.new" => [
      [ "holds the defaults, takes any setting by name, and is a value",
        [ [ ALLOCATOR, "has the default settings and at most ten arms" ],
          [ ALLOCATOR, "takes any setting by name, and equals another with the same settings" ] ] ],
      [ "every setting reaches the decision it names",
        [ [ ALLOCATOR, "decides by its settings: a later warm-up, a higher floor, a smaller step" ],
          [ ALLOCATOR, "builds the posterior on its prior" ],
          [ ALLOCATOR, "crowns with won_share and won_rest, and judges the win with win_p_best and win_loss" ],
          [ ALLOCATOR, "flags result_changed below result_changed_below" ],
          [ ALLOCATOR, "takes the interval's mass from credible_mass" ] ] ],
      [ "refuses an unknown setting, a value that is not a finite number, a floor outside (0, 1]",
        [ [ ALLOCATOR, "refuses an unknown setting, a setting that is not a finite number" ] ] ]
    ],
    "Bandit::Allocator.call" => [
      [ "pure: the same inputs and seed give the same answer, the inputs unchanged",
        [ [ ALLOCATOR, "gives the same output for the same input and the same seed" ],
          [ ALLOCATOR, "does not change its input" ] ] ],
      [ "the same answer on every Ruby, and the stream's use fixed (one extra draw moves every pin)",
        [ [ ALLOCATOR, "gives the pinned result for" ],
          [ ALLOCATOR, "moves every pin with one extra draw before the joint draws" ] ] ],
      [ "hold_satisfied moves the crown only, never the statistics",
        [ [ ALLOCATOR, "gives the same statistics whatever the hold says, for the same seed" ],
          [ ALLOCATOR, "keeps the step weights while the hold runs, and names the leader" ] ] ],
      [ "reads String keys from stored rows and answers Symbol arms",
        [ [ ALLOCATOR, "accepts String arm keys and String tally keys, and answers with Symbol arms" ] ] ],
      [ "refuses a config that is not a Config, no arms, too many arms, a draw count below one, a fixed split",
        [ [ ALLOCATOR, "is refused when the config is not a Config" ], [ ALLOCATOR, "refuses no arms" ],
          [ ALLOCATOR, "refuses more arms than can all hold the floor" ],
          [ ALLOCATOR, "refuses a draw count below one" ],
          [ ALLOCATOR, "is refused before anything is computed, as FixedAllocation" ] ] ]
    ],
    "Bandit::Allocator.won_weights" => [
      [ "the winner won_share, the others won_rest shared equally, keys as given; one arm 1.0; the config's settings",
        [ [ ALLOCATOR, "gives the winner won_share and shares won_rest equally, keyed by the arms as given" ],
          [ ALLOCATOR, "gives a single arm everything" ], [ ALLOCATOR, "takes a config" ] ] ]
    ],
    "Bandit::Allocator.lift" => [
      [ "is 0.2.0's private with_floor, step for step, in public",
        [ [ SHARES_PINS, "SL1 answers 0.2.0's with_floor step for step, Marshal-identical" ],
          [ SHARES_PINS, "SL1 lifts a held won split onto the floor, keeps the keys as given" ] ] ],
      [ "refuses weights, a config or a share the one number check refuses",
        [ [ SHARES_PINS, "SL15 refuses weights, a config or a share the one number check refuses" ] ] ]
    ],
    "Bandit::WinHold.new" => [
      [ "holds 24 hours with a 20-minute gap by default, and reads the lengths given",
        [ [ WIN_HOLD, "holds 24 hours and allows 20 minutes between passes by default" ],
          [ WIN_HOLD, "takes other hold and gap lengths" ] ] ],
      [ "takes Times or plain numbers of seconds as its clock, and requires last_pass_at",
        [ [ WIN_HOLD, "takes plain numbers of seconds as its clock" ],
          [ WIN_HOLD, "requires last_pass_at, so a call that leaves it out cannot build a hold that never holds" ] ] ],
      [ "reads a blank arm as none and keeps any other as a String",
        [ [ WIN_HOLD, "reads a blank arm as none" ], [ WIN_HOLD, "keeps any other arm as a String" ] ] ]
    ],
    "Bandit::WinHold#satisfied?" => [
      [ "true from exactly hold_seconds after leading_since; false without a leader or when interrupted",
        [ [ WIN_HOLD, "is true from exactly hold_seconds after leading_since, not a second before" ],
          [ WIN_HOLD, "is false without a leader or without leading_since" ],
          [ WIN_HOLD, "is false when the run of passes is interrupted" ],
          [ WIN_HOLD, "is false up to and at max_gap_seconds since the last pass, and true after" ] ] ]
    ],
    "Bandit::WinHold#satisfied_for?" => [
      [ "true only for the arm that held, by its String form", [ [ WIN_HOLD, "is true only for the arm that held" ] ] ]
    ],
    "Bandit::WinHold#after_pass" => [
      [ "keeps leading_since for an unbroken run of one arm, starts over otherwise, resets with no leader",
        [ [ WIN_HOLD, "keeps leading_since while the same arm leads an unbroken run" ],
          [ WIN_HOLD, "starts over at this pass for another arm, after a gap, or without leading_since" ],
          [ WIN_HOLD, "resets with no leader" ], [ WIN_HOLD, "makes the header's example true" ] ] ]
    ],
    "Bandit::WeightedPosterior.count" => [
      [ "reads a nil tally as nobody seen, Symbol and String keys alike",
        [ [ WEIGHTED, "reads a nil tally as nobody seen" ],
          [ WEIGHTED, "reads Symbol and String keys alike, and whole numbers as Integers" ] ] ],
      [ "reads the counts by Integer(), the weighted sums by to_f, and keeps them only as stated",
        [ [ WEIGHTED, "reads the counts by Kernel#Integer" ],
          [ WEIGHTED, "keeps the weighted sums only when both are given" ],
          [ WEIGHTED, "keeps the squared sum only beside the weighted sums" ],
          [ INPUTS, "which weighted sums a tally counts is 0.3.0's WeightedPosterior.count" ] ] ]
    ],
    "Bandit::WeightedPosterior.beta" => [
      [ "is [prior_alpha + c, prior_beta + n - c], c clamped to [0, n], n the Kish size never above the raw count",
        [ [ WEIGHTED, "is Beta(1 + c, 1 + n - c) on the raw counts" ], [ WEIGHTED, "clamps the conversions" ],
          [ WEIGHTED, "rescales to the Kish n" ], [ WEIGHTED, "never counts more than the raw exposures" ],
          [ WEIGHTED, "takes the prior it is given" ] ] ],
      [ "refuses a negative exposure count", [ [ WEIGHTED, "refuses a negative exposure count" ] ] ]
    ],
    "Bandit::WeightedPosterior.effective_exposures" => [
      [ "is the raw count without a squared sum, else the Kish n never above it",
        [ [ WEIGHTED, "is the raw count (an Integer) without a squared sum" ], [ WEIGHTED, "rescales to the Kish n" ] ] ]
    ],
    "JointBeta.each_draw" => [
      [ "draws per draw, arm by arm in the given order, from the given stream",
        [ [ JOINT, "draws per draw, arm by arm in the given order" ], [ JOINT, "takes its order as given" ],
          [ JOINT, "is an Enumerator of the same rows without a block" ] ] ],
      [ "refuses a draw count that is not an Integer >= 0, and a parameter not above 0 at its first draw",
        [ [ JOINT, "refuses draws that are not a whole number >= 0" ],
          [ JOINT, "refuses a parameter that is not above 0 at its first draw" ] ] ]
    ],
    "JointBeta.summary" => [
      [ "holds the draws, the wins (a tie split) and the summed loss of the same rows; the same seed, the same answer",
        [ [ JOINT, "holds each arm's draws, its wins and its summed loss" ], [ JOINT, "splits a tie equally" ],
          [ JOINT, "gives the same answer for the same seed" ] ] ]
    ],
    "BetaFunction.regularized" => [
      [ "within 5e-12 relative of an exact reference for parameters up to a few hundred",
        [ [ BETA, "matches the binomial tail for whole parameters, on both sides of the fraction's switch" ],
          [ BETA, "has the closed forms at a or b = 1 and at a = b = 1/2" ] ] ],
      [ "within 4 n log n machine epsilons for parameters from 1 000 to 50 000, and its identities hold there",
        [ [ BETA, "matches the exact binomial tail for whole parameters, at a point on either side of the mean" ],
          [ BETA, "has I_x(a, b) + I_(1-x)(b, a) = 1 across the bulk" ],
          [ BETA, "is 1/2 at x = 1/2 when alpha = beta, alpha up to 50 000" ] ] ],
      [ "0 and 1 outside (0, 1); refuses parameters not finite above 0 and an x that is NaN",
        [ [ BETA, "is 0 at and below x = 0 and 1 at and above x = 1" ],
          [ BETA, "refuses parameters that are not finite numbers above 0" ] ] ]
    ],
    "BetaFunction.quantile" => [
      [ "inverts regularized, for small and large parameters",
        [ [ BETA, "inverts .regularized" ], [ BETA, "has a quantile that .regularized maps back to its probability" ] ] ],
      [ "its 2.5 % and 97.5 % quantiles sit at the corrected normal approximation at large parameters",
        [ [ BETA, "puts the 2.5 % and 97.5 % quantiles at the corrected normal approximation" ] ] ],
      [ "0 and 1 outside (0, 1); refuses parameters not finite above 0 and a p that is NaN",
        [ [ BETA, "is 0 at and below p = 0 and 1 at and above p = 1" ],
          [ BETA, "refuses parameters that are not finite numbers above 0" ] ] ]
    ],
    "BetaFunction.shortest_interval" => [
      [ "with one peak, the highest-density interval: one density at both ends, the mode and the mass held, no wider",
        [ [ INTERVALS, "holds the mode, holds the mass, has one density at both ends and is no wider than any interval" ],
          [ INTERVALS, "is the central interval of a flat density, and the equal-tailed one of a symmetric peak" ] ] ],
      [ "the closed forms of a density that falls from 0 or rises to 1, and of the flat density",
        [ [ INTERVALS, "has the closed forms of a density that falls from 0 or rises to 1: Beta(1, b) and Beta(a, 1)" ],
          [ INTERVALS, "keeps to the boundary of a density that falls or rises with a fractional parameter below 1" ],
          [ INTERVALS, "is the central interval of a flat density" ] ] ],
      [ "a U-shaped density: the shorter one-sided interval, the lower on a tie",
        [ [ INTERVALS, "takes the shorter one-sided interval of a U-shaped density, the lower one on a tie" ] ] ],
      [ "under the flat prior it holds the observed share of every count, 0 and n included, and of fractional counts",
        [ [ INTERVALS, "holds the observed share of every count under the flat prior, 0 and n included" ],
          [ INTERVALS, "holds the mode of fractional (weighted) counts just above an empty count" ],
          [ INTERVALS, "is 0 to 1 - sqrt(0.05) for one visit and no event at 95 %" ] ] ],
      [ "exact: no draw", [ [ INTERVALS, "answers the same every time: no draw" ] ] ],
      [ "refuses a mass not strictly between 0 and 1, then parameters not finite above 0",
        [ [ INTERVALS, "refuses a mass not strictly between 0 and 1 and parameters not above 0" ] ] ]
    ],
    "BetaFunction.mean_intervals" => [
      [ "a value of 0 or 1 per visit gives the share's own interval",
        [ [ INTERVALS, "is the share's own interval for a value of 0 or 1 per visit" ] ] ],
      [ "every mean lies inside its interval and no end is below 0",
        [ [ INTERVALS, "holds every item's mean, never goes below 0, and gives an item of no event an interval from 0" ],
          [ INTERVALS, "keeps an item whose visits all hold one value above 0 inside" ] ] ],
      [ "an item above 0 answers from its own figures alone, the same in any set and on every call",
        [ [ INTERVALS, "answers an item with a mean above 0 from its own figures alone, the same in any set" ] ] ],
      [ "a mean of 0 takes the set's value of one event and narrows as its visits grow; none without an event in the set",
        [ [ INTERVALS, "gives a mean of 0 the set's value of one event" ],
          [ INTERVALS, "has no interval for any mean of 0 while no item of the set has an event" ] ] ],
      [ "the normal interval in the large-sample limit, skewed above the mean at a handful of events",
        [ [ INTERVALS, "matches the normal interval in the large-sample limit and is skewed above the mean" ] ] ],
      [ "nil without data; the items' keys and order kept; Symbol or String field names",
        [ [ INTERVALS, "has no interval without data, keeps the items' keys and order" ] ] ],
      [ "refuses malformed input, and an item whose arithmetic leaves a Float's range, naming it",
        [ [ INTERVALS, "refuses malformed input with an ArgumentError" ],
          [ INTERVALS, "refuses an item whose arithmetic leaves a Float's range, naming the item" ] ] ]
    ],
    "SharedCredit.with_prior" => [
      [ "the rule: by hand, and against a Rational reference from rows and from per-arm sums",
        [ [ SHARED_PRIOR, "tops an arm of one visit up to a strength of 3" ], [ SHARED_PRIOR, "tops both arms up to a strength of 20" ],
          [ SHARED_PRIOR, "tops both arms of shared rows up (strength 2)" ],
          [ SHARED_PRIOR, "matches the reference on seeded shared-credit rows of three arms, one never credited" ],
          [ SHARED_PRIOR, "reads an ordinary test's estimate from per-arm sums as it reads one from rows" ] ] ],
      [ "an arm at or above the strength, and an arm with no credit, keeps its means and block bit for bit",
        [ [ SHARED_PRIOR, "tops up only the arm below the strength" ],
          [ SHARED_PRIOR, "an arm at exactly the strength is not topped up; just above its size it is" ],
          [ SHARED_PRIOR, "keeps the arms, their order, their names as given, weight sums, Kish's sizes and parts" ] ] ],
      [ "every credited arm at or above the strength: the estimate as given, every seeded draw and summary the same",
        [ [ SHARED_PRIOR, "answers the estimate's means, covariance and rounding as given when every arm is at or above" ],
          [ SHARED_PRIOR, "leaves every seeded draw and summary the same, bit for bit" ],
          [ SHARED_PRIOR, "still finds a real difference at plenty of visits" ] ] ],
      [ "in an ordinary test an arm at or above the strength draws the same values beside an arm topped up",
        [ [ SHARED_PRIOR, "in an ordinary test, an arm at or above the strength draws the same values beside an arm topped up" ] ] ],
      [ "a part every credited arm holds at one mean keeps it exactly",
        [ [ SHARED_PRIOR, "a part every arm holds at one mean keeps it exactly, with no prior spread" ] ] ],
      [ "exactly symmetric, taken by LinearValue at its rounding; read back from JSON it names no parts",
        [ [ SHARED_PRIOR, "is exactly symmetric and is drawn from" ],
          [ SHARED_PRIOR, "reads an estimate read back from JSON as it reads the Estimate, naming no parts" ] ] ],
      [ "keeps the arms, parts, weight sums and Kish's sizes; changes nothing it is given",
        [ [ SHARED_PRIOR, "keeps the arms, their order, their names as given, weight sums, Kish's sizes and parts" ],
          [ SHARED_PRIOR, "changes nothing it is given" ] ] ],
      [ "at a strength of 20 a version of one visit reads near even, none of three visits or fewer reaches 0.95",
        [ [ SHARED_PRIOR, "the walk's case: one visit against three read P(best) 0.994" ],
          [ SHARED_PRIOR, "\#{goal}: \#{handful.size} visit(s)" ],
          [ SHARED_PRIOR, "two versions of one true value read decided under the prior at most once in 50 datasets" ] ] ],
      [ "refuses a strength below 0 or not finite, a malformed estimate, an arm of size 0, a figure out of a Float's range",
        [ [ SHARED_PRIOR, "refuses a strength below 0 or that is not a finite number" ],
          [ SHARED_PRIOR, "refuses an estimate without its Kish's sizes or weight sums, or one that is not an estimate" ],
          [ SHARED_PRIOR, "refuses an arm with means but a Kish's size of 0" ],
          [ SHARED_PRIOR, "refuses an estimate whose figures leave a Float's range under the prior" ] ] ]
    ],
    "Gamma.regularized_lower" => [
      [ "within 5e-12 relative of the exact Poisson sum; P + Q = 1",
        [ [ GAMMA, "matches the Poisson sum for a whole a, on both sides of x = a + 1" ], [ GAMMA, "has P + Q = 1" ],
          [ GAMMA, "is near 1/2 + 1/(3 sqrt(2 pi a)) at x = a for a large a" ] ] ],
      [ "refuses a shape not finite above 0 and an x that is NaN",
        [ [ GAMMA, "refuses a shape that is not a finite number above 0, and an x that is not a number" ] ] ]
    ],
    "Gamma.regularized_upper" => [
      [ "within 5e-12 relative of the exact Poisson tail; P + Q = 1",
        [ [ GAMMA, "matches the Poisson tail sum for a whole a, on both sides of x = a + 1" ],
          [ GAMMA, "has P + Q = 1" ], [ GAMMA, "is erfc(sqrt(x)) at a = 1/2" ] ] ],
      [ "refuses a shape not finite above 0 and an x that is NaN",
        [ [ GAMMA, "refuses a shape that is not a finite number above 0, and an x that is not a number" ] ] ]
    ],
    "Significance.normal_quantile" => [
      [ "maps back to p within 1e-12 relative, deep into both tails; -Infinity and Infinity at 0 and 1",
        [ [ SIGNIFICANCE, "inverts the normal distribution function, deep into both tails" ],
          [ SIGNIFICANCE, "is -Infinity at 0 and Infinity at 1" ] ] ],
      [ "refuses a p outside [0, 1]", [ [ SIGNIFICANCE, "refuses a p outside [0, 1]" ] ] ]
    ],
    "Significance.chi_squared_survival" => [
      [ "is Q(dof / 2, statistic / 2), 1 at a statistic of 0 or below",
        [ [ SIGNIFICANCE, "has the closed forms at 1 and 2 degrees of freedom" ],
          [ SIGNIFICANCE, "is 0.05 at the 95 % critical values and 0.01 at the 99 % one" ],
          [ SIGNIFICANCE, "is 1 for a statistic of 0 or below" ] ] ]
    ],
    "Significance.sample_ratio_p_value" => [
      [ "is Pearson's chi-squared on k - 1 degrees of freedom over the arms expected to get traffic",
        [ [ SIGNIFICANCE, "tests the counts against the split on k - 1 degrees of freedom" ],
          [ SIGNIFICANCE, "is 1 when the counts are the split" ],
          [ SIGNIFICANCE, "reads expected by observed's keys, strings as well as symbols" ] ] ],
      [ "nil under minimum_total, read as given above or below its default, or under two arms expected",
        [ [ SIGNIFICANCE, "does not test under minimum_total observations (200 unless given)" ],
          [ SIGNIFICANCE, "reads minimum_total as given above the default too" ],
          [ SIGNIFICANCE, "does not test fewer than two arms expected to get traffic" ] ] ],
      [ "0 when an arm expected to get nothing got some",
        [ [ SIGNIFICANCE, "is 0 when an arm expected to get nothing got some" ] ] ]
    ]
  }.freeze

  module_function

  # [owner, :singleton or :instance, method name] for an index key, e.g. "Bandit::Posteriors#rank".
  def resolve(name)
    owner, separator, method = name.match(/\A(.+?)([.#])([a-z_]+[?!]?)\z/).captures
    [ owner.split("::").reduce(SkiftetStatistical) { |scope, part| scope.const_get(part, false) },
      separator == "." ? :singleton : :instance, method.to_sym ]
  end

  # The lines of `file` that begin an example or a group whose description holds `words`.
  def described(file, words)
    File.readlines(File.expand_path("../#{file}", __dir__)).select do |line|
      line.match?(/\A\s*(?:it|describe|context|specify)\s+["']/) && line.include?(words)
    end
  end
end

RSpec.describe "The interface contract" do
  def index = InterfaceContract::INDEX

  it "names only public methods of the library" do
    index.each_key do |name|
      owner, kind, method = InterfaceContract.resolve(name)
      public = kind == :singleton ? owner.singleton_class.public_method_defined?(method) : owner.public_method_defined?(method)
      expect(public).to be(true), "#{name} is not a public method"
    end
  end

  it "backs every guarantee with at least one example, and every cited example is still in its file" do
    missing = index.flat_map do |name, guarantees|
      guarantees.flat_map do |guarantee, citations|
        expect(citations).not_to be_empty, "#{name}: #{guarantee} cites no example"
        citations.select { |file, words| InterfaceContract.described(file, words).empty? }
                 .map { |file, words| "#{name}: #{file}: #{words}" }
      end
    end

    expect(missing).to be_empty, "cited examples not found:\n#{missing.join("\n")}"
  end

  it "is the list the README's interface contract states, method for method" do
    readme = File.read(File.expand_path("../README.md", __dir__))
    section = readme[/^## Interface contract\b.*?(?=^## )/m]

    expect(section).not_to be_nil
    listed = section.scan(/^#### `([^`(]+)/).flatten
    expect(listed).to match_array(index.keys)
  end
end
