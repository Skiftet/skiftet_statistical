# skiftet_statistical

A dependency-free **statistics toolkit** for Ruby: reusable statistical analysis
code, so the same z-test, percentile, sampler or bandit isn't re-implemented
(differently) in every app.

Modules:

- **`Descriptive`** — mean, variance, standard deviation, percentiles/median.
- **`Significance`** — A/B significance testing: two-proportion z-test, Welch's
  t-test, exact normal CDF and two-tailed p-values, the normal quantile, the
  chi-squared survival and the sample-ratio check.
- **`Gamma`** / **`BetaFunction`** — the regularized incomplete gamma and beta
  functions and the Beta quantile; since 0.5.0 also the shortest
  (highest-density) Beta interval, and intervals of means of non-negative
  values that always hold the mean.
- **`Sampler`** — Gamma/Beta/Gaussian random sampling, RNG-injectable.
- **`JointBeta`** — joint Monte Carlo draws over Beta posteriors (P(best),
  expected loss).
- **`Bandit`** (+ `Policies`) — multi-armed bandit (Thompson Sampling,
  epsilon-greedy, UCB1, Softmax) for online explore/exploit decisions.
- **`Bandit::Posteriors`** (with **`Bandit::Prior`**, **`Bandit::Decay`**,
  **`Bandit::Draw`**, **`Bandit::NormalArgmax`** and **`Bandit::Pass`**) — new
  in 0.4.0, the generic bandit: a set of items with their views and yes/no or
  amount posteriors, ordered and picked by one rule from a key, given uniforms
  or an RNG, with the odds of what it draws (exact for a pick among up to
  `EXACT_PICK_MAX` (20) contenders and for orders and positions over up to
  `EXACT_ORDER_MAX` (10) free items with none barred from leading, otherwise a
  seeded Monte Carlo estimate with its standard error) and the summaries an
  allocation reads.
- **`Bandit::Allocator`**, **`Bandit::WinHold`**, **`Bandit::WeightedPosterior`**
  — an A/B test's traffic allocation: Thompson sampling with a floor per arm, a
  step limit, an automatic win and a hold on it.
  Also per-arm floors and fixed shares, and one phase rule for any posterior
  (`Allocator.from_summary`).
- **`LinearValue`** (with **`Cholesky`**) — P(best), expected loss and
  intervals of a linear value of several rates when both the rates (one joint
  normal) and their weights are uncertain.
- **`SharedCredit`** — Hájek inverse-propensity means of several parts per arm,
  with one covariance over every (arm, part) pair, when one row may credit
  several arms; its sums fold once and store as JSON. Since 0.5.0 also the
  estimate under a prior pooled from its own arms, for arms of a handful of
  visits.
- **`HierarchicalRates`** — Gamma-Poisson rates on a tree: a node's ratio to
  what its parent leads us to expect, the fit of a level's strength (and of a
  two-groups hot share), the leaf fit, and checks beside them.
- **`AdditiveContrasts`** — the arms' contrasts against a reference arm pooled
  across cells along a nested tree and crossed groupings, a test of no
  variation, each cell's level from all its visits, and the pooled estimate a
  cell is allocated on.
- **`BetaArgmax`** (with **`GaussLegendre`** and `BetaFunction.log_density`) —
  the exact probability that each of several Betas is the largest, the seedless
  pick it describes, and the greedy pick; since 0.4.0 also a draw's position,
  the distribution function and E[the largest].
- **`Prequential`** — the prequential log score, and the bits per event a
  deeper predictor saves, with cluster-robust intervals.
- **`Cusum`** — a one-sided change test: its statistic, its run length, the
  threshold for a stated false-alarm rate and the smallest change it can see.

The methods an application calls, each with what it takes, answers, guarantees
and refuses, are under [Interface contract](#interface-contract); what a
version number promises about seeded answers is under [Stability](#stability).

## Install

In a Gemfile (published on rubygems.org):

```ruby
gem "skiftet_statistical", "~> 0.5.0"
```

Or build/install locally:

```sh
cd skiftet_statistical
bundle install
gem build skiftet_statistical.gemspec
```

## Stability

The version follows [semantic versioning](https://semver.org), with one rule
made explicit for a statistics library: **a seeded answer is part of the
public API.** A release that moves an answer for the same inputs, the same
key and the same RNG seed (a draw, an order, a pick, a pinned figure, a
p-value beyond its stated tolerance) is a major or a minor version, as its
CHANGELOG entry states (before 1.0, as semver allows, a minor one), and that
entry says which answers move and why. It is never a patch: a patch release
fixes documentation, adds examples, or turns a crash into the refusal the
documentation states, and moves no answer. Adding a method, or a keyword whose
default keeps every old answer, is a minor version.

The pins in `spec/pins` are literal values recorded when each behaviour was
built and never edited: a change that fails one has moved a public answer, and
ships only as the version above. The methods an application calls, with what
each guarantees, are listed under [Interface contract](#interface-contract).

## Descriptive statistics

```ruby
SkiftetStatistical::Descriptive.mean([1, 2, 3, 4])             # => 2.5
SkiftetStatistical::Descriptive.variance([1, 2, 3, 4, 5])      # => 2.5  (sample; pass sample: false for population)
SkiftetStatistical::Descriptive.standard_deviation(values)
SkiftetStatistical::Descriptive.percentile(incomes, 90)       # interpolated 90th percentile
SkiftetStatistical::Descriptive.median(values)
```

## A/B significance

```ruby
S = SkiftetStatistical::Significance

# Two-proportion z-test: did variant B convert better than A?
result = S.two_proportion_z_test(conversions_a, visitors_a, conversions_b, visitors_b)
result.statistic        # the z score (positive => B higher)
result.p_value          # two-tailed p
result.significant?(0.05)
result.significant_95?  # also _90? / _99?
result.confidence       # 1 - p

# Welch's t-test for a continuous metric (e.g. revenue per visitor):
S.welch_t_test(mean_a, var_a, n_a, mean_b, var_b, n_b)

# And the building blocks directly:
S.normal_cdf(1.96)         # => ~0.975
S.two_tailed_p_value(1.96) # => ~0.05
```

`two_proportion_z_test` / `welch_t_test` return `nil` when the test is undefined
(an empty group or zero variance), matching the existing analyzers' behaviour.

### Sample-ratio check

Does each arm's traffic fit the split its weights promise? Pearson's chi-squared
over the arms expected to get traffic, on k - 1 degrees of freedom. A small p
means the split is not doing what it says (a broken arm, a cache, a bot).

```ruby
S.sample_ratio_p_value(observed: { control: 700, a: 300 },
                       expected: { control: 500.0, a: 500.0 })   # => ~1.1e-36
S.sample_ratio_p_value(observed: ..., expected: ..., minimum_total: 500)
```

`nil` (not tested) under `minimum_total` observations (default 200) or with
fewer than two arms expected to get traffic; `0.0` when an arm expected to get
nothing got some.

## Distribution functions

Pure Ruby, Float in and out. Every function takes Integers too.

```ruby
S.normal_quantile(0.975)                  # => 1.9599639845400532 (the probit; bisection on erfc)
S.chi_squared_survival(3.841458820694124, 1) # => 0.05000000000000006: P(X >= statistic), X ~ chi-squared(1)

G = SkiftetStatistical::Gamma
G.regularized_lower(a, x)                 # P(a, x), the Gamma(a, 1) distribution function
G.regularized_upper(a, x)                 # Q(a, x) = 1 - P(a, x)

B = SkiftetStatistical::BetaFunction
B.regularized(x, alpha, beta)             # I_x(alpha, beta), the Beta(alpha, beta) distribution function
B.quantile(0.975, 41.0, 361.0)            # its inverse: an exact 95 % credible bound
B.shortest_interval(0.95, 1.0, 2.0)       # => [0.0, 0.7763...]: the shortest 95 % interval, holding the mode
```

The arithmetic and its order are part of the API: an app that pins these
outputs (at 1e-12, say) compares them with this exact sequence of
operations, so a reordered formula is a breaking change even when it is "the
same" formula.

### Intervals that hold their figure

An equal-tailed interval of a share's flat-prior posterior, Beta(1 + c, 1 + n - c),
leaves the observed share c / n out when c is 0 or n: 0 of 1 reads 1.3-84.2 %.
`shortest_interval` is the highest-density interval wherever the density has one
peak or is monotone, so it holds the mode, which under the flat prior is the share:

```ruby
B.shortest_interval(0.95, 1.0, 2.0)       # => [0.0, 0.776...]  (0 of 1 visit: 0 % inside)
B.shortest_interval(0.95, 41.0, 261.0)    # => [0.0981, 0.1748] (40 of 300; equal tails: 0.0996-0.1765)
```

`mean_intervals` does the same for means of non-negative values (a value per
visit that is never below 0), each read as a share of its visits: with the
visits' second moment q = n * s^2 + m^2 (s the mean's standard error), one event
is worth q / m and the item holds n * m^2 / q events; its interval is that worth
times the shortest interval of the events' flat-prior Beta. The mean always lies
inside, the interval never goes below 0, a large item's interval is the normal
one, and a value of 0 or 1 per visit gives the share's own interval. An item
with a mean of 0 takes the set's value of one event, so it too has an interval:

```ruby
B.mean_intervals({ "a" => { visits: 342, mean: 0.0058, standard_error: 0.0041 },
                   "b" => { visits: 300, mean: 0.0, standard_error: 0.0 } }, mass: 0.95)
# => { "a" => [0.0009, 0.0184], "b" => [0.0, 0.0099] }
```

Both are exact (no draw) and work by bisection: an interval with one peak
inside cost 0.7 to 2.0 ms of CPU with YJIT (Beta(3, 341) to
Beta(40 001, 1 160 001)), a one-sided one 0.06 ms, and `mean_intervals` of ten
items 9.9 ms, measured on one busy development machine, so a caller with many
items works them out once a pass rather than once a visit. An item whose
figures leave a Float's range in this arithmetic is refused, naming the item,
rather than answered with an infinite or empty interval; what each method
takes, answers and refuses is under [Interface contract](#interface-contract).

## Joint Beta draws

For Bayesian A/B readouts and Thompson-sampling allocators: draw every arm's
Beta posterior once per draw, jointly, so "P(arm is best)" and "expected loss"
compare the same worlds.

```ruby
posteriors = [ [ 41.0, 361.0 ], [ 52.0, 350.0 ] ]   # [alpha, beta] per arm

SkiftetStatistical::JointBeta.each_draw(posteriors, draws: 2_000, rng: Random.new(7)) do |row|
  row  # => one Float per arm, in order
end

summary = SkiftetStatistical::JointBeta.summary(posteriors, draws: 2_000, rng: Random.new(7))
summary[:samples]  # each arm's 2 000 draws
summary[:wins]     # draws each arm won (a tie split equally); / draws = P(best)
summary[:losses]   # summed (best - arm) per arm; / draws = expected loss
```

**The draw order is part of the API.** Per draw, arm by arm in the order
given, one `Sampler#beta` each. `Sampler#gamma` takes a variable number of
values from the RNG, so one arm's posterior moves every later draw: a seeded
caller gets the same rows only while this order holds.

## Quick start (multi-armed bandit)

```ruby
require "skiftet_statistical"

bandit = SkiftetStatistical.bandit(
  arms: %w[a b c d e],
  policy: SkiftetStatistical::Policies::ThompsonSampling.new,
)

choice = bandit.select          # which option to show right now, e.g. "b"
# ... show that option to the user ...
bandit.record(choice, 1)        # reward: 1 = it converted, 0 = it didn't

bandit.best_arm                 # current best by empirical mean
bandit.stats                    # { "b" => { pulls:, mean:, reward_sum: }, ... }
```

Rewards are expected in **[0.0, 1.0]** — a binary `0`/`1` (e.g. "did this view
lead to an outcome?") is the common case, but any value in that range works.

To order several items, or to pick one by a sticky key with the odds of the
pick (exact up to `EXACT_PICK_MAX` contenders), use the posterior set
(`Bandit::Posteriors`, below).

## Policies

| Policy | How it picks | Good when | Key params |
|---|---|---|---|
| `ThompsonSampling` | Sample `theta ~ Beta(successes, failures)` per arm, play the highest draw | The default. Best all-round explore/exploit balance; self-tunes | `prior_alpha`, `prior_beta` |
| `EpsilonGreedy` | Exploit the best mean with prob. `1 - epsilon`, else a random arm | You want a simple, predictable explore rate | `epsilon` (default `0.1`) |
| `UCB1` | Play `argmax(mean + sqrt(c·ln N / n))` — optimism under uncertainty | You prefer deterministic selection (no RNG in the choice) | `c` (default `2.0`) |
| `Softmax` | Play arm `i` with prob. `∝ exp(mean_i / temperature)` | You want exploration weighted by how good each arm looks | `temperature` (default `0.1`) |

```ruby
SkiftetStatistical::Policies::ThompsonSampling.new(prior_alpha: 1.0, prior_beta: 1.0)
SkiftetStatistical::Policies::EpsilonGreedy.new(epsilon: 0.1)
SkiftetStatistical::Policies::UCB1.new(c: 2.0)
SkiftetStatistical::Policies::Softmax.new(temperature: 0.1)
```

**Which to use?** When unsure, use `ThompsonSampling` — it converges fast, needs
no tuning, and explores exactly as much as the evidence warrants. Cold start (no
data) is `Beta(1,1)` on every arm, i.e. uniform random, so early plays are pure
exploration.

The four policies are kept as 0.3.0 wrote them: in 0.4.0, `ThompsonSampling`
and `Softmax` included, each answers as in 0.3.0 for every input, the same arm
and the same use of the rng. New callers draw through `Bandit::Posteriors`
instead, the posterior set, which holds one set of items and ranks, picks and
computes odds and summaries on it.

## Persistence

A bandit's state is just its arms' counters, so it round-trips through a `Hash`
(store it as JSON/JSONB, in Redis, in a column — wherever):

```ruby
saved = bandit.to_h
# => { arms: [{ name:, pulls:, reward_sum:, reward_square_sum: }, ...], policy: {...} }

restored = SkiftetStatistical::Bandit.from_h(
  saved,
  policy: SkiftetStatistical::Policies::ThompsonSampling.new,
)
```

The policy holds an RNG, so it is **not** rebuilt from the serialised config —
pass the policy instance you want to run with.

## Deterministic testing

Every stochastic policy (and the sampler) takes an `rng:`. Inject a seeded
`Random` and selection becomes reproducible:

```ruby
policy = SkiftetStatistical::Policies::ThompsonSampling.new(rng: Random.new(42))
```

## The generic bandit (`Bandit::Posteriors`)

New in 0.4.0. One set of items, each with its posterior and its views, and every
operation a bandit needs on it, each written once: the order rule (`rank`, with
`pick` its head), the odds of what that rule draws, per-item summaries for an
allocation, and plain draws. The caller decides what counts as a view (an item
was seen) and what counts as an outcome; an outcome is yes/no (a Beta posterior
an item) or an amount (a Normal posterior for the mean per view). Every random
input is injected: a key, given uniforms or an RNG.

```ruby
P = SkiftetStatistical::Bandit::Posteriors

set = P.yes_no({ "a" => { views: 40, outcomes: 8 },
                 "b" => { views: 10, outcomes: 1 },
                 "c" => { views: 25, outcomes: 4 } })
set.items       # => ["a", "b", "c"]
set.kind        # => :beta
set.parameters  # => { "a" => [9.0, 33.0], "b" => [2.0, 10.0], "c" => [5.0, 22.0] }  (Beta(1, 1) prior)
set.views       # => { "a" => 40.0, "b" => 10.0, "c" => 25.0 }
set.greedy      # => "a", the largest posterior mean

order = set.rank(key: %w[list visit-1], shuffle: 0.1)
order.items     # => ["a", "c", "b"], the same for this key in every process
order.leader    # => "a"
order.branch    # => :thompson (:shuffled for about 10 % of keys)
order.draws     # => { "a" => 0.2907..., "b" => 0.1004..., "c" => 0.1386... }

set.pick(rng: Random.new(1)).item     # => "c", the head of one Thompson draw
set.values(rng: Random.new(1))        # => [0.1721..., 0.0861..., 0.3054...], one draw an item
```

### Building a set

- `yes_no(items, priors: nil)`: `{ item => tally }`, a tally
  `{ views:, outcomes: }` (Symbol or String keys, numbers >= 0, fractional
  allowed, the outcomes clamped to the views), optionally with
  inverse-propensity sums `weighted_views` (the sum of 1 / p),
  `weighted_outcomes` (of y / p) and `weighted_views_squared` (of 1 / p^2):
  the weighted outcomes are then rescaled to the Kish size, never above the
  views, so an item that was shown more while it led is not overrated. Priors:
  nil (Beta(1, 1)), one pair `[alpha, beta]` for every item, or
  `{ item => pair }`.
- `amounts(items, priors:)`: a tally `{ views:, outcomes:, outcome_squares: }`
  (the outcomes the sum of amounts, any sign; optionally the weighted sums and
  `weighted_outcome_squares`) under one `Prior::Amount` or
  `{ item => Prior::Amount }`; there is no flat amounts prior, as an amount has
  a scale. With k prior views at mean mu0 and per-view variance v0, and n views
  of mean m and variance s^2, the mean per view is Normal with mean
  (k mu0 + n m) / (k + n) and variance
  [k v0 + n s^2 + k n (m - mu0)^2 / (k + n)] / (k + n)^2, formed exactly and
  rounded once.
- `beta(pairs, views: nil)` and `normal(pairs, views: nil)`:
  `{ item => [alpha, beta] }` or `{ item => [mean, standard_deviation] }` as
  they are, with views when given (an item missing from them reads 0).

```ruby
prior = SkiftetStatistical::Bandit::Prior.amount(mean: 4.0, variance: 2_000.0, strength: 20.0)
amounts = P.amounts({ "a" => { views: 300, outcomes: 900.0, outcome_squares: 400_000.0 },
                      "b" => { views: 200, outcomes: 500.0, outcome_squares: 300_000.0 } }, priors: prior)
amounts.parameters  # => { "a" => [3.0625, 2.0665...], "b" => [2.6363..., 2.6457...] }  ([mean, sd])

P.beta({ "a" => [3.0e6, 7.0e6] }).parameters  # => { "a" => [300000.0, 700000.0] }, scaled to LARGEST_TOTAL
```

- Items are non-blank Strings, any number of them (an empty set ranks with no
  head), in input order, which is the tie order everywhere. A Symbol, a blank
  or repeated name, a name holding a NUL byte (the byte a key joins its parts
  with) or the same bytes as an earlier name in another encoding is an
  `ArgumentError` when the set is built; so is a Hash of priors or views that
  names something that is not an item, or two keys naming one item.
- One pair everywhere. A Beta pair whose total is above `LARGEST_TOTAL` (1e6)
  is scaled down to it, its mean kept, in `parameters`: the one pair every draw
  (`rank`, `pick` and `values`, from any source), every odds call (exact or
  Monte Carlo), `summary`'s P(best) and loss and `greedy` read. The pairs as
  given feed `summary`'s means and intervals alone. A scaled pair never sums
  above `LARGEST_TOTAL`, so a set built from another set's `parameters` keeps
  them bit for bit.
- A Beta parameter above 0 but below `BetaArgmax::SMALLEST_PARAMETER` (1e-6)
  builds: the quantile draws (a key or given uniforms), every odds call and
  `summary` refuse it, and the RNG draws take it.

### The order rule (`rank`, `pick`)

```ruby
set.rank(key: "k", may_lead: %w[b c], min_views: 20)   # only b or c may lead; a leader needs 20 views
set.rank(rng: Random.new(1), pinned: { 0 => "c" }, top: 2).items       # => ["c", "a"]
set.rank(uniforms: { explore: 0.5, priority: { "a" => 0.1, "b" => 0.2, "c" => 0.3 },
                     draw: { "a" => 0.5, "b" => 0.5, "c" => 0.5 } }).items  # => ["a", "c", "b"]
set.pick(key: %w[list visit-1], explore: 0.1)           # => a Choice: item, head, branch, draws
# Order#repin: rank again with new pins under the held rule (shuffle and explore 0, no may_lead,
# min_views 0, the same top), on the order's own draws and the source as rank read it
set.rank(key: "k", explore: 0.1).repin({ 0 => "c" })   # == set.rank(key: "k", pinned: { 0 => "c" })
```

The rule: `shuffle:` and `explore:` (each in [0, 1], summing to at most 1),
`may_lead:` (the items that may lead; nil lets every free item lead),
`min_views:` (0), `pinned:` (`{ position => item }`, 0-based) and `top:` (nil:
every item). Pinned items keep their positions and draw nothing; the free items
fill the other positions in order. With the branch value c: below `shuffle` the
order is `:shuffled` (by priority); below `shuffle + explore`, when a contender
has at least `min_views` views, it is `:explored` (the eligible item of smallest
priority leads, the rest follow by draw); otherwise it is `:thompson` (the
largest draw leads, the rest follow by draw). Ties go to the earlier item.

`rank` answers an `Order`: `items` (the first `top`, pinned items at their
positions), `leader` (the free head when it has at least `min_views` views,
else nil), `branch` and `draws` (each free item's posterior value; none on the
shuffled branch). `pick` is `rank` with top 1 and no pins, answering a `Choice`:
`item` (the leader), `head`, `branch` and `draws`. Both are frozen; `==`
compares those four members.

Exactly one source of randomness:

- `key:` (a String, or an Array of parts): the branch value is
  `Draw.uniform_for(*key, "explore")`, an item's priority
  `uniform_for(*key, "priority", item)` and its draw
  `uniform_for(*key, "draw", item)`. The same key draws the same order in every
  process with nothing stored, and no item's uniforms depend on another item:
  over 1 000 keys with `shuffle: 0.1` and no exclusion or pin, an added item
  never changed the others' relative order (with an item pinned at position 1
  it did on 352 of them, as the added item takes a free slot ahead of the pin).
- `uniforms:` (`{ explore:, priority: { item => Float }, draw: { item => Float } }`):
  the same three, given.
- `rng:` (anything that answers `rand`): one branch value when
  `shuffle + explore` is above 0, one priority a free item on the shuffled or
  explored branch, then one `Sampler` draw a free item where the branch draws,
  in item order, and nothing else from the stream.

A key or given uniforms draw by quantile: a Beta item at
`BetaArgmax.position(u, alpha, beta)`, compared by that position so that two
draws never tie by rounding, and a Normal item at mean + sd times
`Significance.normal_quantile(u)`, u on the open grid (`Draw.open_uniform`). An
RNG draws by `Sampler`. Two Normal draws are compared, from every source, by the
exact sign of (mean_i - mean_j) + (sd_i z_i - sd_j z_j), each mean + sd * z
formed exactly, never rounded, so they never tie or misorder where a standard
deviation lies below a rounding of its mean; `draws` answers the rounded values.
Each source draws exactly from the posteriors, so one set of odds (below) covers
all three.

Refused, each an `ArgumentError` before any draw and before anything is taken
from an RNG: a rule outside its bounds, a pinned position or name that is not an
item, a name pinned twice, a `may_lead` name that is not an item, `top` below 1,
no source or more than one, a malformed `uniforms` Hash, and under a key or
given uniforms a Beta parameter below 1e-6. An RNG's own answer is checked as
it comes: one that is not a Float in [0, 1), or a raise from its `rand`, is an
`ArgumentError` at that answer, after the draws taken before it.

`greedy(may_lead: nil)` is the contender with the largest posterior mean on
`parameters` (`BetaArgmax.greedy`'s rule, for a set of any size); `values(rng:)`
is one posterior draw an item, in item order.

### Odds of a pick, an order and a position

The posterior set answers the probability of what its rule draws, under that
same rule (`shuffle`, `explore`, `may_lead`, `min_views`, and for orders
`pinned` and `top`):

```ruby
set = SkiftetStatistical::Bandit::Posteriors.beta({ "a" => [ 1.0, 1.0 ], "b" => [ 2.0, 1.0 ] })
set.pick_probabilities(explore: 0.1)[:heads]  # => { "a" => 0.35000000000000003, "b" => 0.6500000000000001 }
set.order_probability(%w[b a])                # => { probability: 0.6666..., method: :exact, standard_error: 0.0 }
set.positions[:probabilities]                 # => { "a" => [0.3333..., 0.6666...], "b" => [0.6666..., 0.3333...] }
set.order_summary(%w[b a])                    # => { probability: 0.6666..., positions: { "a" => [...], "b" => [...] } }
set.rank(key: "k1").probability               # => 0.6666..., the odds of the order that key drew

SkiftetStatistical::Bandit::Posteriors.mixture(
  shares: { "x" => 0.25, "y" => 0.75 },
  probabilities: { "x" => { "a" => 0.5, "b" => 0.5 }, "y" => { "a" => 0.2, "c" => 0.8 } },
)                                             # => { "a" => 0.275, "b" => 0.125, "c" => 0.6000000000000001 }
```

- `pick_probabilities` gives each item's probability of heading the pick
  (`heads`) and of being the pick's leader (`leaders`, nil for no leader);
  `order_probability(order)` the probability that the rule's whole order
  begins with `order`; `positions` each item's probability at each slot;
  `order_summary(order)` the posterior alone, with no rule: the probability
  that `order` is the best one and each item's probability at each position,
  for a set of at most `EXACT_ORDER_MAX` (10) items. `Choice#probability` and
  `Order#probability` answer the same odds for a pick or an order the set made,
  computed on the first call and kept.
- `Posteriors.mixture(shares:, probabilities:)` is the probability that a
  two-layer draw shows each item: the sum over the first layer's branches of a
  branch's share times its odds of the item.
- They are exact, by quadrature, up to `EXACT_PICK_MAX` (20) contenders for the
  pick and up to `EXACT_ORDER_MAX` (10) free items for orders and positions
  when no free item is barred from leading. The pick's P(largest) is
  `BetaArgmax.probabilities` or `NormalArgmax.probabilities`. The order odds
  sum on every free item's quantiles at 400 levels (an internal grid, not a
  public constant), each sum on points where its own item's points keep their
  digits: Betas share one grid in the position on a log scale the rule
  compares, and each Normal sums on its own points in its own standardised
  variable, the other items read there, so a
  parameter near 1e-6, a standard deviation small beside its mean, or two
  narrow items beside a far one loses no point; each sum is extrapolated from
  the points and their every other level. They read within 4e-6 of closed
  forms, inside the library's stated 1e-4.
- Otherwise each is one of two stated Monte Carlo forms, `draws:` (20 000)
  draws on `Random.new(seed:)` (seed 0), so the same call answers the same
  figures, with `method: :monte_carlo` and the largest standard error: above
  `EXACT_PICK_MAX` contenders the pick counts how often each contender's draw
  is the largest; orders and positions (an item barred from leading, or more
  than `EXACT_ORDER_MAX` free items) count the rule's own visits. Both draw on
  the scaled pairs (`parameters`), the pairs the exact odds read. A Beta
  parameter far below 1 draws 0.0 in a share of `Sampler`'s draws (about a
  tenth at 0.003), where two such draws tie and the tie goes to the earlier
  item, so below about 0.003 the Monte Carlo reads that tie rule as well.
- A Beta parameter below `BetaArgmax::SMALLEST_PARAMETER` (1e-6) is an
  `ArgumentError` for every odds call at every size, exact or Monte Carlo,
  over the items a keyed draw of the same set and rule refuses.

Measured against the rule's own visits in `odds_simulation_spec.rb` (below):
every head and position within 2.75 standard errors of its exact value over
2 * 10^5 visits a fixture.

### Summaries (`summary`)

```ruby
set = SkiftetStatistical::Bandit::Posteriors.beta({ "a" => [ 2.0, 8.0 ], "b" => [ 3.0, 7.0 ] },
                                                  views: { "a" => 120.0, "b" => 340.0 })
summary = set.summary(credible_mass: 0.95)
# => { "a" => { views: 120.0, mean: 0.2, lower: 0.0281..., upper: 0.4825..., p_best: 0.2882...,
#               expected_loss: 0.1335... },
#      "b" => { views: 340.0, mean: 0.3, lower: 0.0748..., upper: 0.6000..., p_best: 0.7117...,
#               expected_loss: 0.0335... } }
SkiftetStatistical::Bandit::Allocator.from_summary(arms: %w[a b], summary: summary,
                                                   exposures: { "a" => 120, "b" => 340 },
                                                   status: "running", winner_arm: nil)
set.summary(among: [])   # views, mean and interval only, for a set of any size
set.means                # => { "a" => 0.2, "b" => 0.3 }, the means alone: no interval, no quantile
```

`summary(credible_mass: 0.95, among: nil)` gives every item, in item order, its
views, its posterior mean and its equal-tailed interval, read on the pair as
given, and for the items `among` names (nil: every item; at most
`EXACT_PICK_MAX`) the exact P(best) and expected loss, E[the largest] less the
item's mean, among those items, on `parameters` (`BetaArgmax.probabilities` and
`.expected_maximum`, or `NormalArgmax`'s). A loss that rounds below 0 is
answered as 0. The answer is `Allocator.from_summary`'s `summary:` as it is, so
a pass on exact figures is one call more. `among: []` computes no P(best) and no
loss; `means` answers the means alone (frozen, summary's mean bit for bit) and
computes no interval, so it calls no quantile. A Beta interval is
`BetaFunction.quantile`, accurate to about 2e-10 up to totals (alpha + beta)
near 10^8; beyond about 3 * 10^8 one bound can fall on or past the mean (at a
mean of 1/2 the lower bound lies 0.0025 standard deviations above it at
4 * 10^8). A Normal interval is the mean less and plus its standard deviation
times the Normal quantile, the tail read on the open grid.

### Priors (`Bandit::Prior`)

A yes/no prior is a Beta pair `[alpha, beta]`; an amounts prior a
`Prior::Amount` (a mean per view, a variance per view and a strength in views).
Three rules are written once, and every other prior is built from them: a mean
held at a strength (`given`), a mean shrunk toward a prior mean (`pooled_mean`)
and the leave-one-out rule (`borrowed`: what a group holds once the item's own
evidence is taken out, views at least 0 and outcomes within [0, views]).

```ruby
PR = SkiftetStatistical::Bandit::Prior

PR.flat                                         # => [1.0, 1.0]
PR.given(mean: 0.1, strength: 20.0)             # => [2.0, 18.0]
PR.amount(mean: 4.0, variance: 2_000.0, strength: 20.0)
PR.borrowed(group: { views: 1_000, outcomes: 100 }, own: { views: 200, outcomes: 50 })
# => { views: 800.0, outcomes: 50.0 }
PR.pooled(group: { views: 1_000, outcomes: 100 }, own: { views: 200, outcomes: 50 }, strength: 20.0)
# => [1.399..., 20.600...]: the borrowed mean (1 + 50) / (2 + 800) held at 2 + 20 views
PR.pooled_mean(views: 40, outcomes: 8, prior_mean: 0.1, strength: 20.0)   # => 0.1666...
PR.tilts({ "a" => 2.0, "b" => 0.5, "c" => 1.0 }, clip: [ 0.5, 1.5 ])     # => { "a" => 1.5, "b" => 0.5, "c" => 1.0 }
PR.chain(levels: [ { views: 1_000, outcomes: 120, strength: 20.0 },
                   { views: 200, outcomes: 30, strength: 30.0 } ], root_mean: 0.1)
# => [3.365..., 26.634...]: the top level's mean, the lower level taken out, held at its 30 views
```

- `pooled(group:, own:, strength:, base: FLAT)`: the borrowed mean
  (base_alpha + y) / (base_alpha + base_beta + n) held at
  base_alpha + base_beta + min(strength, n), so a large group never outweighs
  the item's own evidence for long.
- `pooled_new(others:, group:, own:, weight: 2.0)`: a new item's prior, the
  others' borrowed evidence pooled into (1 + Y) / (2 + N) at `weight`, plus the
  new item's own borrowed evidence.
- `tilts(multipliers, clip:)`: each multiplier over the geometric mean of all
  of them, clamped to `clip`, so tilts reorder items without raising them all.
- `chain(levels:, root_mean:, tilt: 1.0, clip: [0.001, 0.999])`: the prior of
  an item at the bottom of nested levels, top first, each level's mean shrunk
  toward the one above with the level below taken out.
- `pooled_amount(group:, own:, strength:)`: the borrowed mean and variance of
  amounts at min(strength, n); it refuses a group with nothing left out or a
  variance of 0, and the caller then states its own scale with `amount`.

### Decay (`Bandit::Decay`)

```ruby
SkiftetStatistical::Bandit::Decay.weight(72, half_life: 72, cap: 336)   # => 0.5
SkiftetStatistical::Bandit::Decay.weight(337, half_life: 72, cap: 336)  # => 0.0
rows = [ { age: 0, views: 10, outcomes: 3 }, { age: 72, views: 20, outcomes: 4 } ]
SkiftetStatistical::Bandit::Decay.fold(rows, half_life: 72, cap: 336)
# => { views: 20.0, outcomes: 5.0, outcome_squares: 0.0 }
```

One formula, 2 ** (-age / half_life), and 0.0 for an age above the cap; age,
half-life and cap share one unit of the caller's choosing. `fold` weighs every
sum of a row (views, outcomes and outcome squares alike), so the folded tally is
what a posterior reads. The fold is linear, so a caller that sums in a database
groups its rows by age bucket and folds the buckets.

### Keyed uniforms and the draw by weight (`Bandit::Draw`)

```ruby
D = SkiftetStatistical::Bandit::Draw

D.uniform("t")                                # => 0.8895498694938413
D.uniform_for("t", "explore")                 # => 0.856909028643178
D.seed_for("t", "explore")                    # => 15807181545831752473, for Random.new
D.weighted({ "a" => 1, "b" => 3 }, 0.5)       # => ["b", 0.75]
D.from_weights({ "a" => 1, "b" => 3 }, 0.5)   # => "b"
D.open_uniform(0.0)                           # => 1.1102230246251565e-16 (2**-53)
```

A uniform is the first 8 bytes of SHA256 of the text, big-endian, shifted right
by 11 and divided by 2**53: the same text gives the same uniform in every
process, so a draw keyed by a visit, a pass or a test comes back without being
stored. `uniform_for` joins its parts by one NUL byte (nil reads "", a part
holding a NUL byte is refused); `seed_for` is those 8 bytes as an Integer.
`weighted` is the one draw by weight: a weight that is not a finite number above
0 counts 0, all 0 counts every key 1, and it answers the key with its share, the
probability that a uniform draw answers it; `from_weights` is its key. Keys are
any objects, answered as given. `open_uniform` clamps a uniform to
[2**-53, 1 - 2**-53], the one reading of a quantile draw's uniform.

### Normal argmax (`Bandit::NormalArgmax`)

```ruby
NA = SkiftetStatistical::Bandit::NormalArgmax

NA.probabilities([ [ 0.0, 1.0 ], [ 0.0, 1.0 ] ])                # => [0.5, 0.5]
NA.probabilities([ [ 5.0, 0.4 ], [ 5.3, 0.5 ] ], epsilon: 0.1)  # => [0.3377..., 0.6622...]
NA.expected_maximum([ [ 0.0, 1.0 ], [ 0.0, 1.0 ] ])             # => 0.5641... (1 / sqrt(pi))
```

`BetaArgmax`'s rule on independent Normals `[mean, standard_deviation]`, 1 to 20
members: the probability that each is the largest, mixed with epsilon / k, and
E[the largest]. Each member's probability is its own integral in its own
standardised variable, so a narrow member far from the others is weighed on its
own scale; each answer lies within 2 k (k + 1) * 1e-10 + 2e-12 of the exact
probability for k members (1.2e-9 for two, 8.4e-8 for 20).

### A pass with a hold on its win (`Bandit::Pass`)

```ruby
hold = SkiftetStatistical::Bandit::WinHold.new(leading_arm: "a", leading_since: now - 90_000, now: now,
                                               last_pass_at: now - 600)
SkiftetStatistical::Bandit::Pass.call(status: "running", hold: hold) do |hold_satisfied, rng|
  SkiftetStatistical::Bandit::Allocator.call(arms: %i[control a], tallies: tallies, status: "running",
                                             winner_arm: nil, previous_weights: weights, rng: rng,
                                             draws: 20_000, hold_satisfied: hold_satisfied)
end
# => the crown's answer (phase: :won, winner: :a, leader: :a, ...) and ready: true
```

The probe yields (false, `Random.new(seed)`), so a leader gets the step, never
the won split. The pass is ready when the status reads "running", the probe
names a leader and `hold.satisfied_for?(leader)` answers true; then, and only
when `crown:` is true (the default), the block runs again with (true, a fresh
`Random.new(seed)`), so the probe and the crown read the same draws and only the
decision changes. The answer is the last block's Hash plus `ready:`; with
`crown: false` a ready leader keeps the probe's answer and `ready: true` says it
would have been crowned. `seed:` defaults to `Random.new_seed`; a given seed
makes the pass repeatable.

### Amounts: what the Normal posterior gets wrong

Under about 30 non-zero amounts the mean of amounts is right-skewed and the
Normal posterior understates its upside; the prior's k views at v0 keep it
wide, and the shuffle or exploration share keeps the item shown. One large
amount x moves a mean of n views by x / n, so cap amounts before summing them;
`min_views` holds the lead until an item has enough views. Draws and interval
bounds may fall below 0: they are answered as computed, and the order rule
compares draws only. When `Prior.pooled_amount` refuses (nothing borrowed, or a
variance of 0), state a scale with `Prior.amount`. Measured in
`order_simulation_spec.rb` (below): on amounts at 300, 1 000 and 3 000 visits a
day the best item was on top on day 28 in every run, and the rule lost 4.3, 2.6
and 1.9 % of the best order's expected outcomes.

### Inputs

Every number the code new in 0.4.0 reads must lie in one input domain: a real
number whose Float is finite and at most `SkiftetStatistical::Numbers::LARGEST`
(1e100) either side of 0. A larger number, a NaN, an infinity or a String is an
`ArgumentError` where it is read, so no sum or product of inputs can leave a
Float on its way to an answer, and every answer is checked finite before it is
returned. A caller's object is called only where it must be (a key part's
`to_s`, a name's `hash` or `eql?`, a list's `each`, an RNG's `rand`); what it
raises there is an `ArgumentError` naming the input. A caller's String, Hash or
Array is read by its contents, never by its own methods, so a missing key reads
as absent whatever default the Hash holds. A list is any Enumerable but a Hash,
read once by its `each`.

### Costs

CPU time a call with YJIT, Ruby 3.4.9, on one development machine on
2026-11-02 while other work kept it busy (CPU time moves less with the load than
the wall clock does):

| Call | Time |
|---|---|
| `Posteriors.yes_no`, 8 items | 0.055 ms |
| `Posteriors.amounts`, 2 items | 0.034 ms |
| `rank(key:, explore: 0.1)`, 8 Beta items, totals up to 1e3 | 0.21 ms |
| the same, totals up to 1e5 | 0.54 ms |
| the same, 8 Normal items | 0.08 ms |
| `rank(rng:, shuffle: 0.1)`, 8 Beta or Normal items | 0.017 ms |
| `BetaArgmax.position`, Beta(3, 300) / Beta(41, 960) / Beta(4 000, 96 000) | 0.019 / 0.028 / 0.057 ms (0.3.0's bisection: 0.11 / 0.29 / 0.92 ms) |
| `pick_probabilities(explore: 0.1)`, 5 Betas / 5 Normals | 24 / 1.6 ms |
| `summary`, 5 Betas / 20 Betas / 5 Normals | 46 / 440 / 3.8 ms |
| `summary(among: [])`, 60 Betas | 26 ms |
| `NormalArgmax.probabilities`, 20 members | 91 ms |
| `order_probability` (exact), 5 Betas / 5 Normals | 145 / 14 ms |
| `positions` (exact), 5 / 10 Betas | 160 / 560 ms |
| `positions(explore: 0.1)` (exact), 10 Betas | 1.2 s |
| `order_probability` (Monte Carlo, 20 000 visits), 5 Betas | 200 ms |
| `pick_probabilities` (Monte Carlo, 20 000 draws), 21 Betas | 470 ms |

A draw is cheap enough for every visit; the odds and summaries belong off the
request path, once a pass (a set keeps the grids of its exact orders and
positions, so later odds calls on the same set cost less).

## Example: an ordered list, one visit at a time

```ruby
P = SkiftetStatistical::Bandit::Posteriors

# Once a pass (hourly, say): the posterior set from stored tallies, where a view
# counts only for an item the visit reached, kept as its pairs.
set = P.yes_no(tallies)                          # { "a" => { views:, outcomes: }, ... }
cache.write("list:pairs", { pairs: set.parameters, views: set.views })

# Each visit: the set rebuilt from its pairs (bit for bit) and ranked by the
# visit's key, so a visit sees the same order on every request and nothing is
# stored per visit.
kept = cache.read("list:pairs")
order = P.beta(kept[:pairs], views: kept[:views]).rank(key: [ "list", visit_id ], shuffle: 0.1)
order.items                                      # show the items in this order

# Off the request path, once a pass: each item's odds at each position under
# that rule, for an estimate that weighs a view by 1 / its probability.
set.positions(shuffle: 0.1)[:probabilities]
```

## A/B-test allocation (`Bandit::Allocator`)

The decision an A/B test's allocation job makes on every pass: how much traffic
each arm gets next, and whether an arm has won. It is the rule an app's
allocation job ran, moved here; pure (no clock, no state), with the randomness only from the injected `rng`.

```ruby
A = SkiftetStatistical::Bandit::Allocator

result = A.call(
  arms: %i[control a],
  tallies: { control: { exposures: 1_000, conversions: 500 },
             a: { exposures: 1_000, conversions: 533 } },
  status: "running", winner_arm: nil,
  previous_weights: { control: 0.5, a: 0.5 },   # the latest allocation, or nil
  rng: Random.new(7), draws: 20_000,
)
result[:weights]   # => { control: 0.35, a: 0.65 }  (next pass's split)
result[:phase]     # => :shifting  (:warmup, :shifting, :won or :paused)
result[:stats][:a] # => { exposures: 1000, conversions: 533, rate: 0.533, weighted_rate: 0.533,
                   #      lower: 0.502..., upper: 0.563..., p_best: 0.93175, expected_loss: 0.00066...,
                   #      guardrails: {} }
result[:winner]    # => nil  (the crowned arm)
result[:leader]    # => nil  (the arm meeting the win rule at this pass)
result[:result_changed] # => false  (a won test whose winner fell below P(best) 0.5)
```

The rule, in order:

1. **Paused** (`status: "paused"`): the previous weights unchanged (equal
   weights without them).
2. **Won** (`status: "won"` and `winner_arm` one of the arms): the won split,
   95 % to the winner and 5 % shared by the others. `result_changed` flags a
   winner whose P(best) has fallen below 0.5; a won test never un-wins.
3. **Warm-up**, while any arm has fewer than 300 exposures: equal weights.
4. **The automatic win**: the first arm (in the order given) with P(best)
   >= 0.95 and expected loss <= 0.0025 is the `leader`. It gets the won split
   when `hold_satisfied:` is true (the default); with false (the hold below is
   still running) it gets step 5's weights.
5. **Shifting**: the target is each arm's P(best), every arm at least 10 %
   (the others scaled down in proportion), reached by a convex step from the
   previous weights in which no arm moves more than 0.15.

P(best) and expected loss come from `draws` joint draws of the arms' Beta
posteriors (`JointBeta`); the interval is the 95 % equal-tailed one of the
arm's draws. **The draw order is part of the API**: with the same input and
seed the result is the same only while no draw is added, dropped or reordered.

A tally may carry inverse-propensity sums, `weighted_exposures` and
`weighted_conversions` (each exposure counted as 1 / the arm's share when it
was drawn) and `weighted_exposures_squared`: the posterior is then built on
the weighted conversions rescaled to the Kish effective sample size, never
above the raw count (`WeightedPosterior`), so an arm whose lead comes from a
traffic surge it was favoured in is not crowned. A tally may also carry
`guardrails: { goal => conversions }`, reported per arm with its raw rate and
never decided on. Tallies and previous weights may use Symbol or String keys
(rows read back from JSON); previous weights that do not name exactly the arms
with finite numbers read as none.

Every number above is a setting of `Allocator::Config`, with the defaults
below:

```ruby
config = A::Config.new(warmup_exposures: 100, floor: 0.05) # the rest stay
config.max_arms                                             # => 20 (1 / floor)
A.call(..., config: config)
A.won_weights(%i[control a b], :a, config: config)          # => { control: 0.025, a: 0.95, b: 0.025 }
```

Settings: `warmup_exposures` (300), `floor` (0.10), `max_step` (0.15),
`win_p_best` (0.95), `win_loss` (0.0025), `won_share` (0.95), `won_rest`
(0.05), `result_changed_below` (0.5), `credible_mass` (0.95), `prior_alpha`
and `prior_beta` (1.0). More arms than `max_arms` raise `ArgumentError`, and
`allocation: :fixed` (a split that is measured, not allocated) raises
`A::FixedAllocation`, an `ArgumentError`, before anything is computed.

### Per-arm floors and fixed shares (`arm_floors:`, `fixed_shares:`)

Both keywords are nil by default, and then nothing new is checked and every
output is the rule above, bit for bit. `arm_floors: { arm => floor }` gives an
arm its own floor, a number in (0, 1], in place of `Config#floor` (an arm it
does not name keeps `Config#floor`); `fixed_shares: { arm => share }` holds an
arm at exactly its share, a number in (0, 1), in every phase.

```ruby
warm = { control: { exposures: 120, conversions: 30 }, a: { exposures: 110, conversions: 25 },
         b: { exposures: 100, conversions: 31 } }                        # still in warm-up
inputs = { arms: %i[control a b], tallies: warm, status: "running", winner_arm: nil,
           rng: Random.new(7), draws: 2_000 }
A.call(**inputs, arm_floors: { control: 0.5 })[:weights]    # => { control: 0.5, a: 0.25, b: 0.25 }
A.call(**inputs, fixed_shares: { control: 0.2 })[:weights]  # => { control: 0.2, a: 0.4, b: 0.4 }
```

- The keys are arms of the call (Symbols or Strings); no arm in both; the fixed
  shares and the other arms' floors sum to at most 1; every arm fixed raises
  `FixedAllocation`. Each refusal is an `ArgumentError`, raised before any draw.
- In every phase each fixed arm gets exactly its share, and the free arms (the
  others, in the order given) share the rest, `1 - F` with `F` the sum of the
  fixed shares, each at least its floor.
- One lift serves every phase: a target over the free arms is scaled into the
  free mass, every arm under its floor is pinned there, and the others are
  scaled again into what is left, until none is under. With no fixed share and
  every floor `Config#floor` it is the shifting phase's floor above, public as
  `Allocator.lift`.
- Paused: the previous weights while they keep every floor and share, else the
  free arms' previous shares through the lift (a pause on
  `{ control: 0.02, a: 0.49, b: 0.49 }` with `arm_floors: { control: 0.05 }`
  answers `{ control: 0.05, a: 0.475, b: 0.475 }`). Won: the won split over the
  free arms; a fixed arm is never the won arm, so a stored win of one raises.
  Warm-up: equal targets over the free arms. Shifting: the free arms' P(best),
  reached by the step over the free arms and clamped onto their floors.
- With fixed shares, P(best) and expected loss are taken among the free arms in
  each joint draw (every arm is still drawn, in order), so a fixed arm's are
  nil. A lone free arm beside fixed arms can lead and win; with no share
  declared the 0.2.0 guard stays, and a single arm never leads.

Measured (`value_allocation_simulation_spec.rb`, below): no share below its
floor or off its fixed share in 2,800 passes through every phase, and nothing
crowned in 2,000 passes at `hold_satisfied: false` while the win rule named a
leader in 1,926 and 1,978 of them.

### Any posterior (`Allocator.from_summary`)

The same paused / won / warm-up / automatic-win / shifting rule, with the same
per-arm shares and checks, on statistics computed elsewhere: a `summary`
`{ arm => { p_best:, expected_loss:, mean:, lower:, upper: } }` and `exposures`
`{ arm => Integer }` in place of tallies, draws and rng. `call` builds its Beta
statistics and hands them to that one private rule, so the two entry points
cannot disagree on a phase: fed `call`'s own statistics, `from_summary` answers
`call`'s decision.

```ruby
A.from_summary(
  arms: %i[control a],
  summary: { control: { p_best: 0.3, expected_loss: 0.02, mean: 1.1, lower: 0.9, upper: 1.3 },
             a: { p_best: 0.7, expected_loss: 0.005, mean: 1.2, lower: 1.0, upper: 1.4 } },
  exposures: { control: 1_000, a: 1_000 }, status: "running", winner_arm: nil,
  previous_weights: { control: 0.5, a: 0.5 }, hold_satisfied: false,
)
# => { weights: { control: 0.35, a: 0.6499999999999999 }, phase: :shifting, stats: { ... },
#      winner: nil, result_changed: false, leader: nil }
```

- `summary: nil` (no posterior yet) holds warm-up whatever the exposures say; a
  pause and a stored win still come first.
- A test that never ends passes `hold_satisfied: false` at every pass: nothing
  is crowned and the leader is still named, which a caller whose own rule shows
  no leader blanks itself. A warm-up of 0 is `Config.new(warmup_exposures: 0)`.
- `expected_loss` and `Config#win_loss` are read in the caller's units (value
  per visit for a value summary). The rule never reads `mean`; `stats` carry the
  summary's figures as Floats. Keys may be Symbols or Strings; the answer names
  every arm by its Symbol.

### The lift (`Allocator.lift`)

```ruby
A.lift({ control: 0.99, a: 0.01 })      # => { control: 0.9, a: 0.1 }
A.lift({ a: 0.9, b: 0.06, c: 0.04 })    # => { a: 0.8, b: 0.1, c: 0.1 }
```

Every arm at least `Config#floor` (`config:` as for `call`), the others scaled
in proportion into what the pinned floors leave: 0.2.0's private `with_floor`,
step for step, made public (the private method stays, delegating to it).

### The hold on a win (`Bandit::WinHold`)

An automatic win counts only after the same arm met the win rule at every pass
for 24 hours. The caller keeps the state (the leading arm, when it started
leading, the time of the previous pass) and gives the clock:

```ruby
hold = SkiftetStatistical::Bandit::WinHold.new(
  leading_arm: "a", leading_since: now - 90_000, now: now, last_pass_at: now - 900,
  # hold_seconds: 86_400, max_gap_seconds: 1_200 (the defaults)
)
hold.satisfied?        # => true: the stored leader held for 24 hours or more
hold.satisfied_for?(:a)
result = A.call(..., hold_satisfied: hold.satisfied?)
state = hold.after_pass(result[:leader]) # => the leading arm and since, to store
```

A run of passes with a gap longer than `max_gap_seconds` (or no previous pass)
is interrupted: not satisfied, and a leader starts over. Call the allocator a
second time with the same seed and `hold_satisfied: false` when the arm it
crowns is not the arm that held (`satisfied_for?`); `Bandit::Pass` (above)
makes both calls in one.

## Linear value of several rates (`LinearValue`)

When an arm's worth is a weighted sum of several rates, V_a = sum over parts p
of w_p * r_(a, p), and both the rates and the weights are uncertain,
`LinearValue` draws them together: the rates from one joint normal over every
(arm, part) pair (so arms that share evidence stay correlated), the weights from
named parameters drawn afresh in every draw. P(best) and the expected loss then
hedge every uncertain weight at once instead of judging the arms at one assumed
weight.

```ruby
LV = SkiftetStatistical::LinearValue

inputs = {
  arms: %i[a b], parts: %i[p q],
  means: { a: [ 0.12, 0.031 ], b: [ 0.105, 0.044 ] },
  covariance: [ [ 4e-4, 6e-5, 2e-4, 1e-5 ], [ 6e-5, 1e-4, 2e-5, 4e-5 ],      # arm-major:
                [ 2e-4, 2e-5, 5e-4, -5e-5 ], [ 1e-5, 4e-5, -5e-5, 1.2e-4 ] ], # (a, p), (a, q), (b, p), (b, q)
  parameters: { base: { fixed: 1.5 }, worth: { uniform: [ 2.0, 6.0 ] },
                lift: { normal: [ 0.5, 0.2 ], clip: [ 0.0, 1.0 ] } },
  weights: ->(drawn) { { p: drawn.fetch(:base), q: drawn.fetch(:worth) * drawn.fetch(:lift) } }
}
LV.summary(**inputs, draws: 20_000, rng: Random.new(7))
# => { a: { point: 0.242, mean: 0.2423..., lower: 0.1516..., upper: 0.3587..., p_best: 0.47715,
#           expected_loss: 0.0188... },
#      b: { point: 0.2455, mean: 0.2461..., lower: 0.1460..., upper: 0.3736..., p_best: 0.52285,
#           expected_loss: 0.0150... } }
LV.each_draw(**inputs, draws: 3, rng: Random.new(7)).to_a
# => [[0.2233..., 0.1935...], [0.2945..., 0.3738...], [0.1634..., 0.2035...]]  (summary's first rows)
```

`point` is the value at every parameter's centre (the fixed value, the
uniform's midpoint, the normal's mean, each clipped) and the part means;
`mean`, `lower` and `upper` are the drawn values' mean and equal-tailed
`credible_mass` interval (0.95 by default); a tie splits P(best) equally.
`each_draw` yields each draw's values, the rows `summary` is built on (an
Enumerator without a block).

**The draw order is part of the API.** Per draw: each parameter in the order
given (`fixed: x` takes nothing, `uniform: [lo, hi]` one `rng.rand`,
`normal: [mean, sd]` one `Sampler#gaussian`, then its `clip: [lo, hi]` when it
has one); the part weights from the `weights` callable; one joint normal draw
of every rate, means + L * n, with n the K * P `Sampler#gaussian` values in
arm-major order and L the covariance's `Cholesky` factor; each arm's value. A
part with no variance is drawn at its mean and still takes its normal, so the
numbers taken never depend on the data.

Everything is checked before the first random number (`ArgumentError`): 1 to 7
arms and 1 to 12 parts (`MAX_ARMS`, `MAX_PARTS`; sum the parts that share one
weight), a symmetric positive semidefinite covariance, finite numbers, a weight
for every part. `draws:` is an Integer >= 0 (`summary` needs at least 1).
`rounding:` (0.0 by default) is how far each covariance entry may be off: hand
it a `SharedCredit` estimate's `#rounding`, so a covariance summed from rows is
never refused for its own rounding. An arm of one visit, or of visits that all
hold one value, has no variance and is drawn as a point; where arms may hold a
handful of visits, draw from `SharedCredit.with_prior`'s estimate (below).

Measured (`value_allocation_simulation_spec.rb`, below): the 95 % intervals of
each arm's value and of each difference covered the truth in 377 to 385 of 400
runs (bound 372), the true weights drawn inside their ranges per run;
allocating on it through `Allocator.from_summary`, values that arrived up to 23
passes late left a final bias within 1.08 standard errors, and a drift shared by
every arm moved no equal arm's mean P(best) more than 0.028 from 1 / 4.

## Cholesky factor (`Cholesky`)

```ruby
SkiftetStatistical::Cholesky.factor([ [ 4.0, 2.0 ], [ 2.0, 5.0 ] ])  # => [[2.0, 0.0], [1.0, 2.0]]
SkiftetStatistical::Cholesky.root([ [ 1.0, 2.0 ], [ 2.0, 4.0 ] ])    # => [[1.0, 0.0], [2.0, 0.0]]
```

`factor` answers the lower-triangular L with L * L^T = the matrix, rows in and
rows out, the lower triangle read, with no matrix library. The matrix must be
square, finite and symmetric (entries (i, j) and (j, i) within 1e-12 of their
scale). A pivot below 1e-12 times the largest diagonal entry is no variance and
is set to 0 with its column (a residual under it beyond rounding raises: a part
with no variance cannot covary); a pivot below -1e-9 times the trace raises:
the matrix is not positive semidefinite. `root(rows, rounding: 0.0)` is the
same arithmetic taken largest pivot first, for a matrix of lower rank whose
rounding the given order would carry into every later pivot: a square root S
with S * S^T = the matrix within its tolerance, not triangular in the given
order. `LinearValue` draws through `factor`, and through `root` where `factor`
refuses.

Measured: no simulation of its own; every `LinearValue` draw in the
simulations below goes through it.

## Shared-credit estimates (`SharedCredit`)

Each arm's mean of several parts, with one covariance over every (arm, part)
pair, when one row may count for several arms. A row is
`{ credit: { arm => w >= 0 }, values: [x_1, ..., x_P] }`: in an ordinary test
it credits only the arm drawn, with 1 / p; with shared credit it credits every
arm whose own choice is what the row shows, each with 1 over the probability
that the row showed it. One estimator serves both.

```ruby
SC = SkiftetStatistical::SharedCredit

estimate = SC.estimate(
  [ { credit: { a: 2.0 }, values: [ 1.0, 0.0 ] },
    { credit: { a: 1.0, b: 1.0 }, values: [ 0.0, 2.0 ] },
    { credit: { b: 4.0 }, values: [ 1.0, 1.0 ] } ],
  arms: %i[a b], parts: %i[p q],
)
# => { means: { a: [0.6666..., 0.6666...], b: [0.8, 1.2] },
#      covariance: [[0.0987..., -0.1975..., 0.0355..., -0.0355...], ...],  # arm-major: (a, p), (a, q), (b, p), (b, q)
#      weight_sums: { a: 3.0, b: 5.0 }, effective: { a: 1.8, b: 1.4705... } }
estimate.parts     # => [:p, :q]
estimate.rounding  # => 7.78...e-15, the largest rounding bound of an entry
SC.contrast(estimate, :b, :a, weights: { p: 1.0, q: 0.5 })
# => { mean: 0.3999999999999999, variance: 0.012800000000000138 }
```

Means are Hájek, r_ap = sum_i w_ai * x_ip / W_a; the covariance is the sandwich
sum_i psi_ap(i) * psi_bq(i) with psi_ap(i) = w_ai * (x_ip - r_ap) / W_a, so arms
that share rows covary. It is laid out arm-major, as `LinearValue` reads it,
computed on one triangle and mirrored, so it is exactly symmetric; Kish's
effective size stands beside it. An arm no row credits has nil means and zero
rows. An entry within its cancellation rounding ((8 + n) * `Float::EPSILON`
times the sum of its terms' sizes, n the larger of Kish's size and the rows
folded) answers exactly 0.0, so an arm of constant values gives zeros that
`LinearValue` accepts.

The sums are public, for a caller that keeps its rows in a store and must fold
each one once:

```ruby
acc = SC.accumulator(arms: %i[a b], parts: %i[p q])
acc.add({ credit: { a: 2.0 }, values: [ 1.0, 0.0 ] })
saved = acc.to_h            # JSON-safe: { "format" => 1, "arms" => ["a", "b"], "parts" => ["p", "q"], ... }
SC::Accumulator.from_h(saved).merge(later).estimate   # = one estimate over every row of both
```

For an ordinary test the per-arm sums are enough, `{ arm => { w:, w2:,
a: { part => }, b: { part => }, c: { [p, q] => } } }`, with w = sum 1/p,
w2 = sum 1/p^2, a = sum x/p, b = sum x/p^2 and c = sum x_p * x_q / p^2 (both
orders of each pair):

```ruby
sums = { a: { w: 4.0, w2: 8.0, a: { p: 2.0, q: 2.0 }, b: { p: 4.0, q: 4.0 },
              c: { %i[p p] => 4.0, %i[p q] => 0.0, %i[q p] => 0.0, %i[q q] => 4.0 } },
         b: { w: 8.0, w2: 32.0, a: { p: 4.0, q: 4.0 }, b: { p: 16.0, q: 16.0 },
              c: { %i[p p] => 16.0, %i[p q] => 16.0, %i[q p] => 16.0, %i[q q] => 16.0 } } }
SC.from_sums(sums, arms: %i[a b], parts: %i[p q])
# => { means: { a: [0.5, 0.5], b: [0.5, 0.5] },
#      covariance: [[0.125, -0.125, 0.0, 0.0], [-0.125, 0.125, 0.0, 0.0],
#                   [0.0, 0.0, 0.125, 0.125], [0.0, 0.0, 0.125, 0.125]],
#      weight_sums: { a: 4.0, b: 8.0 }, effective: { a: 2.0, b: 2.0 } }
# the estimate of four rows: a credited 2.0 with [1, 0] and [0, 1], b credited 4.0 with [1, 1] and [0, 0]
```

- `linear(estimate, weights:)`: each arm's value at fixed weights and the K x K
  covariance of the values. `contrast(estimate, a, b, weights:)`: one
  difference with its variance. `weights:` is an Array in the parts' order or a
  Hash by part name (every part once, in any order; refused on an estimate read
  back from JSON, which names no parts).
- `contrast_for_fit(estimate, arm, reference, weights:, minimum_effective: 20)`:
  the contrast a fit across cells takes, the same difference with one per-visit
  variance pooled over the two arms, s^2 * (1 / n_a + 1 / n_r), and nil when
  either arm has nil means or a Kish's size under `minimum_effective`. An arm
  held at a small floor reads a plug-in variance that moves with its own
  estimate (a rare part it has not seen yet reads a variance near 0), and a
  test across cells would then reject too often.

Measured (`two_layer_simulation_spec.rb`, below): scored by greedy shared
credit, two options of identical value read a mean P(best) of 0.58 and 0.42
(bound 0.40 to 0.60) where the same visits scored as they ran read 0.63 and
0.37; in the pooled cells, `contrast_for_fit` left 1,135 of 24,000 cell
contrasts out of the fit.

### An arm of a handful of visits (`with_prior`)

The sandwich of an arm of one visit, or of visits that all hold one value, is
exactly 0, so `LinearValue` draws that arm as a point: one visit at 1 against
three at 0, 1 and 0 reads P(best) 0.994, and one visit at 0 against them 0.114,
by the one visit's value alone. `with_prior(estimate, strength:)` answers the
estimate under a prior pooled from its own credited arms, every arm's evidence
topped up to `strength` visits, for drawing:

```ruby
estimate = SC.estimate(
  [ { credit: { a: 1.0 }, values: [ 1.0 ] },
    { credit: { b: 1.0 }, values: [ 0.0 ] }, { credit: { b: 1.0 }, values: [ 1.0 ] },
    { credit: { b: 1.0 }, values: [ 0.0 ] } ],
  arms: %i[a b], parts: %i[converted],
)
topped = SC.with_prior(estimate, strength: 20)
# => { means: { a: [0.525], b: [0.475] }, covariance: [[0.01246875..., 0.0], [0.0, 0.01246875]],
#      weight_sums: { a: 1.0, b: 3.0 }, effective: { a: 1.0, b: 3.0 } }
LV.summary(arms: %i[a b], parts: %i[converted], means: topped[:means], covariance: topped[:covariance],
           parameters: { w: { fixed: 1.0 } }, weights: ->(drawn) { { converted: drawn.fetch(:w) } },
           draws: 20_000, rng: Random.new(7), rounding: topped.rounding)
# => P(best) a 0.61965, b 0.38035 (0.994 and 0.006 on the estimate itself)
```

The prior's mean per part is the credited arms' mean weighted by their Kish's
sizes n_a, m0 = sum_a n_a * r_a / N; its per-visit covariance is
V0 = sum_a n_a * (n_a * cov(r_a) + g_a * g_a^T) / N with g_a = r_a - m0, each
arm's per-visit covariance and its distance from m0, so a spread between the
arms widens it. An arm of Kish's size n below the strength k takes e = k - n
visits of it, the Normal mean's rule taken jointly over its parts: means
(e * m0 + n * r_a) / k, covariance block (n / k)^2 * cov(r_a) +
(e * V0 + (e * n / k) * g_a * g_a^T) / k^2, and every covariance with another
arm scaled by n / k, as its means are.

**Where visits are plenty nothing moves.** An arm at or above the strength,
and an arm with no credit, keeps its means and its own covariance block bit for
bit; where every credited arm is at or above it, the answer holds the given
means and covariance, so every seeded draw `LinearValue` takes from it is the
draw it takes from the estimate. The answer keeps the arms, the parts, the
weight sums and Kish's sizes (the evidence's own: hand `contrast_for_fit` the
given estimate, not this one); its rounding is the given one's or, where
larger, that of an entry it changed. A strength of 0 tops nothing up. Refused
(`ArgumentError`): a strength below 0 or not a finite number, an estimate
without its weight sums or Kish's sizes, an arm with means but a Kish's size
of 0, and an estimate whose figures rise past a Float's largest value under the
prior (figures so small that the prior's products fall below Float::MIN read
as 0; no estimate `estimate` answers holds them).

Measured (`shared_credit_prior_spec.rb`), at a strength of 20: a version of
one visit read P(best) between 0.41 and 0.77 against versions of 1 to 100
visits, over values of one part, of two parts at uncertain prices and of
amounts, where the estimate itself read 0.98 or more, or 0.02 or less, in 12 of
the table's 16 rows; no version of three visits or fewer reached 0.95. Two
versions of one true value, one of 1 to 3 visits and the other of 3 to 50, read
decided (a P(best) at or above 0.95) in none of 250 seeded datasets, against
158 on the estimate itself.

## Hierarchical Poisson rates (`HierarchicalRates`)

How far a node's rate stands from its parent's, from its count s and the count
E its parent leads us to expect there. The node's ratio has a Gamma prior with
mean 1 and strength k (Clayton and Kaldor 1987), so its posterior is
Gamma(k + s, k + E); under a two-groups prior (Efron 2008) the node is null
with probability 1 - h (its count Poisson(E), its ratio 1) and otherwise drawn
from that slab.

```ruby
HR = SkiftetStatistical::HierarchicalRates

HR.leave_one_out_share(parent_count: 40.0, parent_total: 1_000.0, node_count: 6.0, node_total: 100.0,
                       prior_share: 0.04, prior_strength: 16.0)        # => 0.0378... (34.64 / 916)
HR.node(count: 6.0, expected: 2.5, strength: 8.0, threshold: 1.25, parent_tail: 0.0, parent_ratio: 1.0)
# => { ratio: 1.3333333333333333 (14 / 10.5), hot: 1.0, tail: 0.5593048238168208 (Q(14, 13.125)) }

pairs = [ { count: 6.0, expected: 2.5, siblings: 3 }, { count: 0.0, expected: 2.0, siblings: 3 },
          { count: 3.0, expected: 3.5, siblings: 3 }, { count: 12.0, expected: 4.0, siblings: 2 },
          { count: 1.0, expected: 3.0, siblings: 2 }, { count: 5.0, expected: 1.0, siblings: 1 } ]
HR.fit_strength(pairs: pairs)
# => { strength: 1.1892... (2**0.25), hot_share: 1.0, log_likelihood: -12.57..., at_upper_bound: false,
#      interval: [0.25, 9.5136...], profile: [[0.25, ...], ...] }
HR.marshall(pairs: pairs)                           # => { variance: 0.9136..., strength: 1.0945... }
HR.boundary_test(pairs: pairs, strength: 2**0.25)   # => { statistic: 7.5936..., p_value: 0.0029... }
HR.smooth_strength(previous: 8.0, fitted: 16.0)     # => 9.1895... (exp(0.8 * log 8 + 0.2 * log 16))
```

The last pair (siblings 1) is left out of the fit and of both checks: its
parent has one child with data, which carries nothing about the level.

- `node(count:, expected:, strength:, hot_share: 1.0, threshold:, parent_tail:,
  parent_ratio:)`: hot = h * NB(s; k, k / (k + E)) / [h * NB + (1 - h) *
  Poisson(s; E)] (1.0 at h = 1), ratio = (1 - hot) + hot * (k + s) / (k + E),
  tail = (1 - hot) * parent_tail + hot * Q(k + s, (k + E) * L / parent_ratio).
- `fit_strength(pairs:, hot_grid: nil)`: the marginal log-likelihood maximised
  on `GRID` (57 strengths, 2^-2 to 2^12 in quarter powers of 2) and on
  `hot_grid` (h = 1 without one); `interval` holds the strengths within 1.92
  (half the chi-squared(1) 95 % point) of the maximum, and `at_upper_bound`
  says the level's nodes inherit their parent.
- `fit_leaves(parents:, hot_shares:, grid: GRID)`: the leaf level's fit, each
  parent's null rate estimated jointly with k and h (from its pooled rate, each
  step weighting its leaves by their posterior chance of being null, at most 50
  steps) instead of read from a leave-one-out share. With about five leaves to a
  parent, a hot leaf lifts its siblings' leave-one-out share, they read as cold,
  and a fit on those expected counts reads the hot share several times too
  high. `parents` is `[[{ count:, total: }]]`, `hot_shares` each in [0, 1); the
  answer holds `strength`, `hot_share`, `log_likelihood`, `at_upper_bound`, and
  per parent `converged`, `rates` and each leaf's `hot`.
- `smooth_strength(previous:, fitted:, weight: 0.2)`, a moving average in
  logs; `marshall(pairs:)`, Marshall's (1991) moment estimate;
  `boundary_test(pairs:, strength:, hot_share: 1.0)`, the likelihood-ratio test
  on the boundary (Self and Liang 1987).

Counts are real numbers >= 0: a node that stands in its parent by a share w has
w multiplied into its count and total by the caller. The probability mass
functions are taken in logs, through log-gamma.

Measured (`tree_recovery_simulation_spec.rb`, below): on generated five-level
trees each level's fitted strength was within a factor of 2 of the true one in
20 of 20 runs, the leaves' hot share within 0.05 in 20 of 20 on `fit_leaves`
(0.05 to 0.1 for a true 0.02 on leave-one-out expected counts), and the
shrunken ratio beat the raw one on a held-out half at every level below the
first in every run.

## Additive contrasts across cells (`AdditiveContrasts`)

What differs between cells is pooled on the arms' contrasts against a reference
arm, never on the arms' own levels (Hong, Kveton, Zaheer and Ghavamzadeh 2022).
Per arm a, delta(a | cell) = delta_root(a) + u_tree(a) + the sum over groupings
g of u_g(a): the tree nested (each node's deviation N(0, tau^2 of its depth)),
every other grouping crossed (N(0, tau^2_g)).

```ruby
AC = SkiftetStatistical::AdditiveContrasts

cells = [ { id: 1, contrasts: { a: [ 0.0, 1.0 ] }, tree: %i[root x], groups: {} },  # [estimate, sampling variance]
          { id: 2, contrasts: { a: [ 1.0, 1.0 ] }, tree: %i[root y], groups: {} },
          { id: 3, contrasts: { a: [ 5.0, 1.0 ] }, tree: %i[root z], groups: {} } ]
fit = AC.fit(cells: cells, arms: %i[a])
# => { variances: { "tree:1" => { a: 6.0 } },
#      effects: { "tree:1" => { x: { a: [-1.7142..., 2.5714...] }, y: { a: [-0.8571..., 2.5714...] },
#                               z: { a: [2.5714..., 2.5714...] } } },
#      cells: { 1 => { a: [0.2857..., 0.9047...] }, 2 => { a: [1.1428..., 0.9047...] },
#               3 => { a: [4.5714..., 0.9047...] } } }
fit.variation_tests  # => { "tree:1" => { a: { statistic: 14.0, degrees: 2, p_value: 0.000911... } } }
AC.fit(cells: cells, arms: %i[a], variances: { "tree:1" => { a: 0.0 } })[:cells]
# => { 1 => { a: [2.0, 0.3333...] }, 2 => { a: [2.0, 0.3333...] }, 3 => { a: [2.0, 0.3333...] } }
```

A tree of one level with no other grouping, as here, is plain normal shrinkage
toward the root: the library's one home of it. Variances come by moments on the
contrasts, per depth, grouping and arm: tau^2 = max(0, the weighted spread of
the group estimates less their mean sampling variance), unbiased whatever the
precisions. `variances:` holds what it names fixed (an earlier fit's
`variances`), and the rest is fitted. Posteriors are normal-normal: the tree is
solved exactly for given variances, and the crossed groupings are backfitted
with it, `sweeps` times (5 by default). A cell may leave out an arm (nil); an
arm no cell has a contrast for raises.

`fit` answers an `AdditiveContrasts::Fit`, the Hash above, whose
`#variation_tests` stand beside each variance: Cochran's Q (the spread the
moment reads), its degrees and the chi-squared chance under tau^2 = 0 of a
spread at least that large (1.0 with no degrees). A caller that holds a
variance at 0 unless its test rejects hands the next fit that 0 in
`variances:`.

`baseline(rows:, effects:)` is each cell's level from all its visits, less each
arm's fitted effect and adjusted on a categorical covariate, with its Hájek
variance. A row sums one (cell, arm, covariate)'s visits: `weight` (sum w),
`value` (sum w * x) and, beyond one visit, `weight_squared`, `value_squared` and
`cross` (sum w^2, sum w^2 * x^2, sum w^2 * x). The reference arm is the one arm
of the rows that no cell of `effects` names.

```ruby
AC.baseline(
  rows: [ { cell: "c1", arm: :ref, covariate: "k1", weight: 2.0, value: 2.0 },
          { cell: "c1", arm: :a, covariate: "k2", weight: 1.0, value: 2.0 },
          { cell: "c2", arm: :ref, covariate: "k2", weight: 1.0, value: 0.0 } ],
  effects: { "c1" => { a: [ 1.0, 0.25 ] }, "c2" => { a: [ 0.5, 0.25 ] } },
)
# => { "c1" => [0.9166... (11 / 12), 0.0246... (2 / 81)], "c2" => [0.25, 0.0] }
```

Pooling per (arm, part) is what reaches a decision: `fit`'s `arms:` name the
pairs (`AC.pair(:a, :p) # => "a|p"`), one `baseline` runs per part, and
`pooled_estimate(own:, effects:, baselines:, cell:)` puts them back together as
the cell's estimate in `SharedCredit`'s shape, with `#parts` and `#rounding`,
which `LinearValue` and the allocator read as they read the cell's own. Means
are mu_p + delta_ap (delta 0 at the reference arm); the covariance is
Cov(mu_p, mu_q) + [a == b] * Cov(delta_ap, delta_aq): the cell's level is one
quantity shared by every arm, each arm's effect is its own, and each block keeps
the cell's own correlation between parts. A contrast of two arms then holds
their effects alone, so no arm's share follows its cell's level.

```ruby
# own: the cell's SharedCredit estimate, means ref [1.0] and a [2.0], covariance [[0.04, 0.012], [0.012, 0.09]]
AC.pooled_estimate(own: own, effects: { "c1" => { "a|p" => [ 0.5, 0.03 ] } },
                   baselines: { p: { "c1" => [ 1.0, 0.01 ] } }, cell: "c1")
# => { means: { ref: [1.0], a: [1.5] }, covariance: [[0.01, 0.01], [0.01, 0.04]], weight_sums: ..., effective: ... }
```

Measured (`contrasts_simulation_spec.rb` and `two_layer_simulation_spec.rb`,
below): every variance of at least 4 times the mean sampling variance came back
within a factor of 2, and every variance of 0 under a quarter of it, in each of
100 runs; across 40 cells whose levels differ with no effect difference, each
allocated on its pooled estimate, the reference arm's mean share spread 0.0066
(the equal-level noise floor's 0.0065) at a rank correlation of -0.073 with the
level.

## Beta argmax (`BetaArgmax`, `GaussLegendre`, `BetaFunction.log_density`)

Which of several independent Beta variables is the largest: the exact
probability of each, the seedless pick that probability describes, and the
greedy pick.

```ruby
BA = SkiftetStatistical::BetaArgmax

BA.probabilities([ [ 1.0, 1.0 ], [ 2.0, 1.0 ] ])                # => [0.3333..., 0.6666...] (1/3, 2/3)
BA.probabilities([ [ 1.0, 1.0 ], [ 2.0, 1.0 ] ], epsilon: 0.1)  # => [0.35..., 0.65...]
BA.pick([ [ 2.0, 8.0 ], [ 8.0, 2.0 ] ], explore: 0.5, priorities: [ 0.3, 0.6 ],
        uniforms: [ 0.5, 0.5 ], epsilon: 0.1)                      # => 1
BA.greedy([ [ 1.0, 3.0 ], [ 3.0, 1.0 ] ])                         # => 1

SkiftetStatistical::GaussLegendre.nodes(2)
# => [[-0.5773..., 1.0000...], [0.5773..., 1.0000...]]  (x = -+1 / sqrt(3), w = 1)
SkiftetStatistical::BetaFunction.log_density(0.25, 2.0, 3.0)    # => 0.5232... (log 27/16)
```

`pick` takes its uniforms from the caller, never from an RNG: below `epsilon`
(`explore < epsilon`) the member with the smallest priority, else the member
whose Beta quantile at its own uniform is the largest. Over uniform inputs
member c is picked with probability epsilon / k + (1 - epsilon) * the integral
of f_c * the product over j != c of F_j, which is what `probabilities` answers,
so a caller can log the exact probability of the pick it made. The integral is
composite Gauss-Legendre (`NODES`, 20 a panel), the panels broken at every
member's quantiles at `BREAKS` (1e-10 to 1 - 1e-10, `TAIL` = 1e-10), each half
of (0, 1) worked in the distance from its own end so that a parameter far below
1 does not round, and normalised to sum 1. It is exact to 1e-8 absolute for 1
to 20 members (`MAX_MEMBERS`) with every parameter at least 1e-6
(`SMALLEST_PARAMETER`) and alpha + beta at most 3 * 10^6 (`LARGEST_TOTAL`):
beyond those parameter limits `probabilities` and `pick` raise, while `greedy`
(the largest mean) takes any parameters; every call takes 1 to 20 members. Ties
go to the lower index.

`GaussLegendre.nodes(n)` is the n-point rule on [-1, 1], ascending and exactly
symmetric, computed once per n and kept, frozen. `BetaFunction.log_density` is
-Infinity outside (0, 1); `regularized` and `quantile` are unchanged.

Since 0.4.0 three more answers sit beside these, and `probabilities`, `pick` and
`greedy` answer as in 0.3.0 (`pick` still compares its own position):

```ruby
BA.position(0.5, 2.0, 2.0)                         # => 0.0, the draw 1/2
BA.distribution(0.3, 2.0, 8.0)                     # => 0.8039967659999996
BA.expected_maximum([ [ 1.0, 1.0 ], [ 1.0, 1.0 ] ])  # => 0.6666666666666667 (2/3)
```

- `position(uniform, alpha, beta)`: where the draw of Beta(alpha, beta) at a
  uniform lies, log(2 x) up to 1/2 and -log(2 (1 - x)) beyond, a key that
  rounds at neither end; the posterior set's quantile draws compare by it. A
  guarded Newton step on the same distribution function. Against `pick`'s own
  bisection (0.3.0's position), recorded on 30 members at 1 000 uniforms each,
  the spec bounds |position - recorded| / max(1, |recorded|), an absolute
  error where the recorded position lies within 1 of 0 and a relative one
  beyond: 1e-12 (measured at most 4.7e-13), 1e-9 where alpha + beta exceeds
  10^4 (measured 3.0e-13) and 1e-8 at the smallest parameter (measured
  9.5e-11). It takes 0.019 / 0.028 / 0.057 ms at Beta(3, 300) / Beta(41, 960) /
  Beta(4 000, 96 000) where the bisection took 0.11 / 0.29 / 0.92 ms.
- `distribution(x, alpha, beta)`: that distribution function itself, the one
  `probabilities`, `pick`, `position` and `expected_maximum` read, bit for bit.
- `expected_maximum(params)`: E[the largest member], 1 less the integral over
  (0, 1) of the product of the members' distribution functions.

Measured (`argmax_accuracy_simulation_spec.rb`, below): on 30 fixtures, sharp
ones with alpha + beta up to 2 * 10^5 among them, `pick`'s frequencies over
10^4 draws came within 2.44 standard deviations of `probabilities`, and
`probabilities` within 2.43 of 10^6 plain `Sampler#beta` draws (bound 5).

## Prequential log score (`Prequential`)

```ruby
PQ = SkiftetStatistical::Prequential

rows = [ { cluster: 1, log_probs: { shallow: Math.log(0.25), deep: Math.log(0.5) } },
         { cluster: 1, log_probs: { shallow: Math.log(0.5),  deep: Math.log(0.5) } },
         { cluster: 2, log_probs: { shallow: Math.log(0.5),  deep: Math.log(0.25) } },
         { cluster: 3, log_probs: { shallow: Math.log(0.125), deep: Math.log(0.5) } } ]
PQ.score(rows, depth: :deep)               # => 1.25 (bits per event)
PQ.gains(rows, depths: %i[shallow deep])   # => { deep: { bits: 0.49999999999999994, se: 0.6495..., events: 4, clusters: 3 } }
PQ.clustered_mean(sums: { 1 => 1.0, 2 => -1.0, 3 => 2.0 }, counts: { 1 => 2, 2 => 1, 3 => 1 })
# => { mean: 0.5, lower: -0.7730..., upper: 1.7730..., n: 4, clusters: 3 }
```

The prequential log score (Dawid 1984): how many bits an outcome cost a
predictor that gave it probability p, -log2 p, on outcomes it had not seen.
`gains` compares depths of one predictor, each deeper than the one before, on
the same events: per depth after the first, the mean gain in bits per event and
its standard error. Events of one cluster move together, so the standard error
comes from the spread between clusters, with the factor G / (G - 1) (Liang and
Zeger 1986); `clustered_mean(sums:, counts:, level: 0.95)` is the same interval
for a caller that stores per-cluster totals and counts, and `gains` is built on
it, so the two cannot differ. Under 2 clusters the standard error and the
bounds are nil. A log probability must be finite and at most 0: smooth a
probability of 0 first.

Measured: no simulation; its specs hold hand-checked values.

## One-sided change test (`Cusum`)

```ruby
CU = SkiftetStatistical::Cusum

CU.run([ 0.0, 0.0, 5.0 ], reference: 0.5, threshold: 4.0)
# => { statistic: 4.5, alarm_at: 3, path: [0.0, 0.0, 4.5] }
CU.average_run_length(shift: 0.0, reference: 0.5, threshold: 4.0)       # => 338.09...
h = CU.threshold(reference: 0.5, false_alarm_steps: 365.0)              # => 4.0743...
CU.minimum_detectable(reference: 0.5, threshold: h, horizon_steps: 30)  # => 0.4752...
```

Page's (1954) statistic over standardised values,
S_t = max(0, S_(t-1) + z_t - k), alarms at the first S_t above the threshold h
(steps count from 1; nothing resets after an alarm). Run lengths are
Siegmund's approximation, (exp(-2 theta b) + 2 theta b - 1) / (2 theta^2) with
theta = shift - k and b = h + 1.166 (b^2 at theta = 0, summed as its series
near 0). `threshold` turns a stated false-alarm rate, one alarm in
`false_alarm_steps`, into h by bisection on [0, 50]; the rate is the
approximation's, so a caller states it as about that rate.
`minimum_detectable` is the smallest shift, in standard units, whose run length
is at most `horizon_steps`; the caller scales it by its standard error.

Measured (`change_test_simulation_spec.rb`, below): a stated one false alarm in
365 steps gave 0.983 per 365 null steps, and one in 1,460 gave 0.260 per 365
(stated 0.25), each inside its 95 % binomial interval; a shift of the minimum
detectable size was caught within the horizon in 62 to 66 % of runs (bound
half).

## Interface contract

The methods an application calls, each with what it takes, what it answers,
what holds of the answer and what it refuses, so a caller can rely on the
method without pinning its outputs in its own suite. An app tests that it hands
these methods the right inputs and uses their answers; the library tests that
the answers are right. Every guarantee below is backed by a library example:
`spec/interface_contract_spec.rb` indexes them, method by method, and fails
when a cited example is gone or a method below is no longer public. If a
guarantee you need is missing here, ask for it to be stated and tested in the
library rather than pinning it in your app.

Conventions:

- **Refused** means an `ArgumentError` raised before anything is computed or
  taken from an RNG, unless the entry says otherwise.
- **Numbers** are read by the library's one number reader where the entry says
  so: a finite real number of magnitude at most 1e100 (`Numbers::LARGEST`); a
  String, a numeric one included, is not a number.
- **Determinism** comes in three kinds, and each entry says which:
  - *keyed*: SHA256 and Integer arithmetic only, so the same parts give the
    same answer in every process, on every machine and on every Ruby version;
  - *seeded*: the same RNG stream gives the same answer, the library taking
    values from it in a stated order (`Random.new(seed)` gives the same stream
    on every Ruby the CI runs, 3.1 to 3.4);
  - *exact*: no randomness at all; the same inputs give the same Float on one
    platform, and where the computation goes through `Math.lgamma` another math
    library can move the last bits (the pins compare such figures at 1e-9
    relative, and orders, heads, branches and counts exactly, on every Ruby the
    CI runs).

### Posterior sets (`Bandit::Posteriors`)

#### `Bandit::Posteriors.yes_no(items, priors: nil)`

- Takes: `items`, a Hash of item name => tally, the names non-blank Strings in
  the tie order; a tally `{ views:, outcomes: }` (Symbol or String keys,
  numbers >= 0, fractional allowed; optionally `weighted_views`,
  `weighted_outcomes` and `weighted_views_squared`); `priors` nil (Beta(1, 1)),
  one pair `[alpha, beta]` (each >= 0) or `{ item => pair }` naming every item.
- Answers: a set, `kind` `:beta`, `parameters` `{ item => [alpha + c, beta + n - c] }`
  with c the outcomes clamped to [0, views] (or the weighted outcomes rescaled
  to the Kish size, never above the views), a pair whose total is above
  `LARGEST_TOTAL` (1e6) scaled to it with its mean kept; `views` the tallies'
  views as Floats.
- Guarantees: plain arithmetic, so the same tallies build the same pairs bit
  for bit everywhere; an item's pair is its own tally's and prior's alone; a
  set built from another set's `parameters` keeps them bit for bit; the same
  rule as `WeightedPosterior.beta`; any number of items, none included. A set
  is a value: `items`, `kind`, `parameters` and `views` are frozen and no
  reader (`rank`, `pick`, `Order#repin`, `summary`, `means`, `values`,
  `greedy`) writes to it, so one set may be built once and read from many
  requests and threads (the order and position odds keep a memo on a set its
  caller has not frozen; a frozen set answers alike).
- Refuses: `items` that is not a Hash; a name that is not a String, is blank,
  is given twice, holds a NUL byte or an earlier name's bytes; a count below 0,
  not a number or beyond 1e100; a Hash of priors naming a name that is not an
  item or leaving one out; a pair not above 0 once the tally is added.

#### `Bandit::Posteriors.beta(pairs, views: nil)`

- Takes: `pairs`, a Hash of item name => `[alpha, beta]`, each a finite number
  above 0; `views` nil (every item 0) or `{ item => views >= 0 }`, an item it
  leaves out reading 0.
- Answers: a `:beta` set holding the pairs as given (scaled in `parameters`
  only above `LARGEST_TOTAL`; `summary`'s means and intervals read them as
  given). `normal(pairs, views:)` is the same for `[mean, standard deviation]`.
- Guarantees: as `yes_no`'s set: a value, frozen, read by no reader's writes.
- Refuses: the names `yes_no` refuses; a parameter that is not a finite number
  above 0; views naming a name that is not an item, or below 0.

#### `Bandit::Posteriors.amounts(items, priors:)`

- Takes: tallies `{ views:, outcomes:, outcome_squares: }` (the outcomes the
  sum of amounts, any sign) under one `Prior::Amount` or one an item.
- Answers: a `:normal` set of `[mean, standard deviation]` for the mean per
  view.
- Refuses: as `yes_no`, and a prior that is not a `Prior::Amount` (there is no
  flat amounts prior).

#### `Bandit::Posteriors#parameters`

- Answers: `{ item => pair }`, frozen: the one pair every draw (`rank`, `pick`,
  `values`, from any source), every odds call, `summary`'s P(best) and loss and
  `greedy` read.
- Guarantees: unchanged by every reader; a Beta pair above `LARGEST_TOTAL` is
  scaled to it with its mean kept, never summing above it.

#### `Bandit::Posteriors#rank(key: | uniforms: | rng:, shuffle: 0.0, explore: 0.0, may_lead: nil, min_views: 0.0, pinned: {}, top: nil)`

- Takes: exactly one source: `key:` (a String, or an Array of parts: nil, a
  String, or anything whose `to_s` answers a String, none holding a NUL byte),
  `uniforms:` or `rng:` (answering `rand` with Floats in [0, 1)); `shuffle` and
  `explore` in [0, 1] summing to at most 1; `may_lead` nil or a list of item
  names; `min_views` a number (one below 0 asks for none, as 0 does); `pinned`
  `{ Integer position => item }` (0-based); `top` nil or an Integer >= 1.
- Answers: a frozen `Order`: `items` (the first `top`, pinned items at their
  positions), `leader` (the free head when it has `min_views` views, else nil),
  `branch` (`:shuffled`, `:explored` or `:thompson`) and `draws` (each free
  item's posterior value; none on the shuffled branch).
- Guarantees:
  - A key draws the same Order from the same set in every process (*keyed*
    uniforms, `Draw.uniform_for(*key, "explore")`, `(*key, "priority", item)`
    and `(*key, "draw", item)`; the draws are their quantiles on `parameters`,
    *exact*).
  - An item's uniforms are the key's and its own name's alone: with no pin, an
    added item never changes the others' relative order, and an added item
    that `may_lead` does not name changes neither the others' order, the leader
    nor the branch. `may_lead`, `min_views`, `shuffle` and `explore` choose the
    head, the branch and the leader; the items after the head follow by draw
    (by priority on the shuffled branch).
  - From an RNG (*seeded*): exactly one branch value (when shuffle + explore is
    above 0), one priority per free item on the shuffled or explored branch,
    then one `Sampler` draw per free item where the branch draws, in item
    order, and nothing else from the stream.
  - Every source draws exactly from the posteriors, so the pick, order and
    position odds describe each of them. Ties go to the earlier item. Pinned
    items keep their positions and draw nothing. The set is not changed.
- Refuses, before any draw and before anything is taken from an RNG: a rule
  outside its bounds, a pinned position or name that is not an item, a name
  pinned twice, a `may_lead` name that is not an item, `top` below 1, no source
  or two, a malformed `uniforms` Hash, a key part holding a NUL byte or whose
  `to_s` answers no String, and under a key or given uniforms a Beta parameter
  below 1e-6. An RNG answer that is not a Float in [0, 1), or a raise from its
  `rand`, is an `ArgumentError` at that answer.

#### `Bandit::Posteriors#pick(key: | uniforms: | rng:, shuffle: 0.0, explore: 0.0, may_lead: nil, min_views: 0.0)`

- Answers: a frozen `Choice`: `item` (the leader, nil when the head has fewer
  than `min_views` views), `head`, `branch` and `draws`; nil, nil and no draws
  for an empty set.
- Guarantees: `rank`'s; under every source it is the head of the Order `rank`
  answers there, an RNG giving up exactly the values `rank` takes.
- Refuses: what `rank` refuses.

#### `Bandit::Order#repin(pinned)`

- Takes: `{ Integer position => item }`.
- Answers: the Order `rank` answers on the same set and the same key or given
  uniforms with these pins under the held rule (shuffle 0, explore 0, no
  `may_lead`, `min_views` 0, the order's own `top`), `==` and every draw.
- Guarantees: an item the order drew keeps its draw and is not drawn again, so
  re-pinning an order's own head first draws nothing at all; the source is the
  one `rank` read, so a caller's object changed after `rank` moves no re-pin;
  neither the order nor its set changes.
- Refuses: an order drawn from an RNG, one no posterior set made, and what
  `rank` refuses on that source.

#### `Bandit::Posteriors#summary(credible_mass: 0.95, among: nil)`

- Takes: `credible_mass` above 0 and below 1; `among` nil (every item), `[]`
  (none: means and intervals only, any number of items) or a list of at most
  `EXACT_PICK_MAX` (20) item names.
- Answers: `{ item => { views:, mean:, lower:, upper:, p_best:, expected_loss: } }`
  in item order, `p_best` and `expected_loss` nil for an item `among` does not
  name; the shape `Allocator.from_summary` reads.
- Guarantees (*exact*): no key, uniform or RNG is read, so the same set answers
  the same summary bit for bit on every call whatever the process's random
  state; an item's views, mean and interval are its own pair's alone (the pair
  as given, never the scaled one), so the item answers them alike in any set
  that holds it, a set of its own included; the interval's ends are
  `BetaFunction.quantile`'s at (1 - mass) / 2 and 1 less that (its accuracy
  below), or the Normal's mean -+ z sd; P(best) is within 1e-12 of
  `BetaArgmax.probabilities` (`NormalArgmax`'s for a Normal set) among the
  named items only, and the loss is never below 0.
- Refuses: a `credible_mass` outside (0, 1); an `among` that is not a list,
  names a name that is not an item or names more than 20; a Beta parameter
  below 1e-6 (on the pair as given, on every path; on the scaled pair when
  `among` names an item).

#### `Bandit::Posteriors#means`

- Answers: `{ item => mean }`, frozen, in item order: `summary`'s `:mean`, bit
  for bit.
- Guarantees (*exact*): no quantile, no draw and no randomness, for a set of
  any size.
- Refuses: nothing a built set holds.

### Priors (`Bandit::Prior`)

#### `Bandit::Prior.flat`

- Answers: `[1.0, 1.0]`, the same frozen Array on every call.

#### `Bandit::Prior.borrowed(group:, own: nil)`

- Takes: two yes/no tallies `{ views:, outcomes: }`; nil reads as no evidence.
- Answers: `{ views:, outcomes: }`, the group less the item's own, views at
  least 0 and outcomes within [0, views].
- Guarantees: the one leave-one-out rule `pooled` and `pooled_new` borrow by.
- Refuses: a tally that is not a Hash or nil; a count below 0, not a number or
  beyond 1e100.

#### `Bandit::Prior.pooled(group:, own: nil, strength:, base: Prior::FLAT)`

- Takes: two tallies, a `strength` >= 0 in views, a `base` pair above 0.
- Answers: `[alpha, beta]`: the borrowed mean (base_alpha + y) /
  (base_alpha + base_beta + n) held at base_alpha + base_beta +
  min(strength, n); the base itself when nothing is borrowed.
- Guarantees: plain arithmetic, the mean formed exactly and rounded once, so
  the same inputs answer the same pair bit for bit everywhere.
- Refuses: a `base` that is not a pair above 0, then a `strength` below 0 or
  not a number (`"strength must be >= 0 (got -1.0)"`), whatever the tallies
  hold, a nil group included; then what `borrowed` refuses.

#### `Bandit::Prior.pooled_new(others:, group: nil, own: nil, weight: 2.0)`

- Takes: `others`, a list of `{ group:, own: }`; the new item's own tallies; a
  `weight` >= 0.
- Answers: `[alpha, beta]`: the others' borrowed evidence pooled into
  (1 + Y) / (2 + N) at `weight`, plus the new item's own borrowed evidence;
  Beta(1, 1) with no others and no group.
- Refuses: a `weight` below 0, `others` that is not a list of Hashes, and what
  `borrowed` refuses of any tally.

### Keyed randomness (`Bandit::Draw`, `Bandit::Decay`)

#### `Bandit::Draw.seed_for(*parts)`

- Takes: parts, each nil (read as ""), a String (its bytes) or anything whose
  `to_s` answers a String (so 7 and "7" are one part).
- Answers: an Integer in [0, 2**64): the first 8 bytes of SHA256 of the parts'
  bytes joined by one NUL byte, big-endian, for `Random.new`.
- Guarantees (*keyed*): the same Integer in every process, on every machine
  and every Ruby version (unlike `String#hash`, which Ruby seeds per process);
  two different part lists never join to the same text.
- Refuses: a part holding a NUL byte, and a part whose `to_s` answers no
  String.

#### `Bandit::Draw.uniform_for(*parts)`

- Answers: `seed_for`'s top 53 bits over 2**53, a Float in [0, 1).
- Guarantees and refusals: `seed_for`'s.

#### `Bandit::Draw.from_weights(weights, uniform)`

- Takes: a Hash of key => weight (keys any objects) and a uniform in [0, 1].
- Answers: the first key, in input order, whose running sum of counted weights
  reaches uniform * total (the last counted key when rounding leaves the sum
  short); nil for an empty Hash.
- Guarantees: pure; a weight that is not a finite number above 0 counts 0 and
  is never answered; with no positive weight every key counts 1; a uniform
  drawn uniformly answers a key with probability its counted weight over the
  total; keys are answered as given and never read.
- Refuses: weights that are not a Hash, a uniform outside [0, 1], a weight
  beyond 1e100, and counted weights whose sum is not finite.

#### `Bandit::Decay.weight(age, half_life:, cap: nil)`

- Answers: 2 ** (-age / half_life); 0.0 above the cap (inclusive at it).
- Guarantees (*exact*): no clock of its own; 1.0 at age 0, 0.5 at one half-life.
- Refuses: a negative age, a half-life not above 0, a cap below 0.

#### `Bandit::Decay.fold(rows, half_life:, cap: nil)`

- Answers: `{ views:, outcomes:, outcome_squares: }` summed over rows, each
  row's sums at its age's weight; linear in the rows, so rows summed by age
  first fold alike.
- Refuses: a row without an age or with a negative one, and what the tally
  reader refuses.

### A/B allocation (`Bandit::Allocator`, `Bandit::WinHold`, `Bandit::WeightedPosterior`)

#### `Bandit::Allocator::Config.new(**settings)`

- Takes: any of `warmup_exposures`, `floor`, `max_step`, `win_p_best`,
  `win_loss`, `won_share`, `won_rest`, `result_changed_below`,
  `credible_mass`, `prior_alpha` and `prior_beta` by keyword; one left out is
  its `DEFAULTS` value.
- Answers: a frozen value, `==` by its settings.
- Guarantees: every setting is read by the decision it names, at every call
  handed this Config, and from nowhere else.
- Refuses: a name that is not a setting, a value that is not a finite number, a
  floor outside (0, 1].

#### `Bandit::Allocator.call(arms:, tallies:, status:, winner_arm:, rng:, draws:, previous_weights: nil, hold_satisfied: true, config: Config.new, fixed_shares: nil, arm_floors: nil)`

- Takes: 1 to `config.max_arms` arms (Symbols or Strings); tallies
  `{ arm => { exposures:, conversions: } }` (Symbol or String keys, optionally
  the weighted sums and guardrails); status `"running"`, `"paused"` or `"won"`;
  an RNG; a whole number of joint draws.
- Answers: `{ weights:, phase:, stats:, winner:, result_changed:, leader: }`,
  arms as Symbols.
- Guarantees (*seeded*): pure, no clock and no state; the same inputs and the
  same RNG stream answer the same result, the joint draws taken arm by arm in
  the given order (`JointBeta`); the inputs are not changed; `hold_satisfied`
  moves only the crown (weights, phase, winner), never `stats`; every setting
  read is the `config`'s. The pinned fixtures answer alike on every Ruby the CI
  runs.
- Refuses, before any draw: a config that is not a `Config`, no arms or more
  than `config.max_arms`, `draws` below 1 once read by `Integer()` (whose own
  error a value it cannot read raises), the per-arm share checks, and
  `allocation: :fixed` (`FixedAllocation`, an `ArgumentError`).

#### `Bandit::Allocator.won_weights(arms, winner, config: Config.new)`

- Answers: the winner `config.won_share`, every other arm
  `config.won_rest / (arms.size - 1)`, keyed by the arms as given; a single arm
  1.0.
- Guarantees: pure arithmetic; the config's two settings are the ones read.
- Refuses: nothing of its own: a winner that is not one of the arms is not
  checked (every arm then gets its `won_rest` share), and a config that is not
  a `Config` raises `NoMethodError`.

#### `Bandit::Allocator.lift(weights, config: Config.new)`

- Answers: every arm at least `config.floor`, the others scaled in proportion
  into what the pinned floors leave, keys as given.
- Guarantees: 0.2.0's private `Allocator#with_floor`, step for step, in public.
  `with_floor` itself stays private and carries no promise: call `lift`.
- Refuses: weights that are not a Hash of 1 to `max_arms` arms, a config that
  is not a `Config`, a share that is not a finite number.

#### `Bandit::WinHold.new(leading_arm:, leading_since:, now:, last_pass_at:, hold_seconds: 86_400, max_gap_seconds: 1_200)`

- Takes: any arm (a blank one reads as none, any other is kept as a String);
  clock values that are Times, numbers of seconds or anything whose difference
  compares with the two lengths; `leading_since` and `last_pass_at` may be nil.
- Guarantees: pure, no clock of its own; the two lengths given are the two
  read.
- Refuses: nothing of its own: leaving out a required keyword (`last_pass_at`
  included) is Ruby's `ArgumentError`, and a clock value that cannot be
  subtracted raises its own error when the hold is first asked.

#### `Bandit::WinHold#satisfied?`

- Answers: true from exactly `hold_seconds` after `leading_since`; false with
  no leader, no `leading_since`, or a run interrupted (no previous pass, or one
  more than `max_gap_seconds` ago; a gap of exactly `max_gap_seconds` is
  unbroken).

#### `Bandit::WinHold#satisfied_for?(arm)`

- Answers: `satisfied?` and `arm`, by its String form, is the arm that held.

#### `Bandit::WinHold#after_pass(leader)`

- Answers: `{ leading_arm:, leading_since: }`: unchanged while the same arm
  leads an unbroken run; the leader from `now` for another arm, after a gap or
  without `leading_since`; both nil with no leader.

#### `Bandit::WeightedPosterior.count(tally)`

- Takes: a Hash with Symbol or String keys, or nil (nobody seen); anything
  else raises `NoMethodError`.
- Answers: `{ exposures:, conversions:, weighted_exposures:,
  weighted_conversions:, weighted_exposures_squared: }`, the weighted sums kept
  only when both are given and the weighted exposures are above 0, the squared
  sum only beside them and above 0.
- Guarantees: 0.2.0's reading, kept as it was: the counts by `Kernel#Integer`
  (a Float truncates toward 0, a numeric String reads as its number), the
  weighted sums by `to_f`.
- Refuses: what `Integer()` cannot read raises `Integer()`'s own error: an
  `ArgumentError` for a String that is no number, a `TypeError` for an Array or
  another object, a `FloatDomainError` for NaN or an infinity.

#### `Bandit::WeightedPosterior.beta(count, prior_alpha: 1.0, prior_beta: 1.0)`

- Answers: `[prior_alpha + c, prior_beta + n - c]`, n `effective_exposures`, c
  the conversions (or the weighted conversions rescaled to n) clamped to
  [0, n].
- Guarantees: plain arithmetic; `Posteriors.yes_no` reads a tally by the same
  rule.
- Refuses: a negative exposure count, as an `ArgumentError`.

#### `Bandit::WeightedPosterior.effective_exposures(count)`

- Answers: the raw count (an Integer) without a squared sum, else the Kish n
  (sum w)^2 / sum w^2, never above the raw count.

### Joint draws, distribution functions and tests

#### `JointBeta.each_draw(posteriors, draws:, rng:)`

- Takes: a list of `[alpha, beta]` pairs, each above 0; an Integer >= 0 of
  draws; an RNG.
- Answers: one Array per draw (one Float per arm, in order), yielded, or an
  Enumerator of them without a block.
- Guarantees (*seeded*): the same pairs, draws and stream give the same rows;
  per draw, arm by arm in the given order, one `Sampler#beta` each, so adding,
  dropping or reordering an arm moves every later draw.
- Refuses: `draws` that is not an Integer >= 0 (before anything is drawn), and
  a parameter that is not above 0 at its first draw (`Sampler`'s
  `"shape must be > 0"`; with 0 draws nothing is read).

#### `JointBeta.summary(posteriors, draws:, rng:)`

- Answers: `{ samples:, wins:, losses: }` over the same rows as `each_draw`, a
  tie's win split equally between the tied arms.
- Guarantees and refusals: `each_draw`'s.

#### `BetaFunction.regularized(x, alpha, beta)`

- Answers: I_x(alpha, beta), the Beta distribution function; 0.0 for x <= 0
  and 1.0 for x >= 1.
- Guarantees (*exact*): for parameters up to a few hundred, within 5e-12
  relative of an exact binomial reference. For parameters from 1 000 to
  50 000 each (n = alpha + beta), within 4 n ln n machine epsilons, relative
  (about 1e-9 at n = 100 000), where the identities I_x(a, b) + I_(1-x)(b, a) = 1
  and I_(1/2)(a, a) = 1/2 hold to the same bound. Beyond a total of about
  3 * 10**8 the continued fraction's fixed bound binds (see `summary`).
- Refuses: alpha or beta whose `to_f` is not a finite number above 0, and an x
  whose `to_f` is NaN (each read by `to_f`, as `Gamma` reads its arguments).

#### `BetaFunction.quantile(probability, alpha, beta)`

- Answers: the x where I_x(alpha, beta) = probability, by bisection to a
  Float's spacing; 0.0 at and below 0, 1.0 at and above 1.
- Guarantees (*exact*): `regularized(quantile(p))` is p within `regularized`'s
  bound; at parameters from 1 000 to 50 000 the 2.5 % and 97.5 % quantiles lie
  within 0.001 sd of the normal approximation with its first skewness term.
- Refuses: as `regularized`, for the probability.

#### `BetaFunction.shortest_interval(mass, alpha, beta)`

- Takes: `mass`, a finite real number strictly between 0 and 1 (a String is
  not one); `alpha` and `beta`, read by `to_f` as `regularized` reads them.
- Answers: `[lower, upper]`, two Floats with 0 <= lower <= upper <= 1: with one
  peak inside (alpha, beta > 1) the highest-density interval, its two ends of
  one density; [0, quantile(mass)] for a density that falls from 0
  (alpha <= 1 <= beta, not both 1) and [quantile(1 - mass), 1] for one that
  rises to 1; [(1 - mass) / 2, (1 + mass) / 2] for the flat density; for a
  U-shaped one (alpha, beta < 1) the shorter of the two one-sided intervals,
  the lower on a tie.
- Guarantees (*exact*, by bisection, no draw; the mass goes through
  `regularized`): the interval holds `mass` within `regularized`'s accuracy,
  short of it by a Float's last bits at most; with one peak it is no wider than
  the equal-tailed interval of the mass, or than any interval of a 400-point
  grid; for alpha, beta >= 1 (not both 1) it holds the mode, so under the flat
  prior, Beta(1 + c, 1 + n - c), it holds the observed share c / n, at c = 0 and
  c = n too, where the equal-tailed interval leaves it out, and for fractional
  counts.
- Refuses: a mass that is not a finite real number strictly between 0 and 1,
  then an alpha or beta whose `to_f` is not a finite number above 0.

#### `BetaFunction.mean_intervals(items, mass:)`

- Takes: `items`, a Hash of key => `{ visits:, mean:, standard_error: }`
  (Symbol or String field names, the keys any objects): the visits a mean of
  non-negative values is over, its mean per visit and its standard error, each
  a finite real number >= 0 where given, visits fractional or whole; `mass` as
  `shortest_interval` takes it.
- Answers: `{ key => [lower, upper] }` in the items' order. With n the visits,
  m the mean, s the standard error and q = n s^2 + m^2: q / m times the shortest
  interval of Beta(1 + k, 1 + n - k), k = n m^2 / q (never above n); for a mean
  of 0, the set's value of one event (the summed n q over the summed n m of the
  items above 0) times [0, quantile(mass) of Beta(1, 1 + n)]. nil for an item
  with a figure missing (nil: its other figures are then not read) or no
  visits, and for a mean of 0 in a set with no item above 0.
- Guarantees (*exact*): every mean lies inside its interval and no end is below
  0; an item with a mean above 0 answers from its own figures alone, the same
  in any set; a mean of 0's interval narrows as its visits grow; a value of 0
  or 1 per visit gives the share's own interval (`shortest_interval` of
  Beta(1 + c, 1 + n - c), within 1e-9); in the normal limit the interval is the
  normal one (its width within 1 % at about 400 events), and with a handful of
  events it is skewed above the mean.
- Refuses: a mass `shortest_interval` refuses; then items that are not a Hash
  of Hashes, and a visit count, mean or standard error, given beside the other
  two, that is not a finite real number or is below 0, before any interval is
  worked out; and, as the item is worked out, an item whose arithmetic leaves a
  Float's range (a value of one event, its own or the set's, that is not a
  finite number above 0; visits past about 2**53 that all hold one value),
  naming the item.

#### `Gamma.regularized_lower(a, x)`

- Answers: P(a, x), the Gamma(a, 1) distribution function; 0.0 for x <= 0.
- Guarantees (*exact*): within 5e-12 relative of the exact Poisson sums for
  whole a up to 200, on both sides of x = a + 1; P + Q = 1.
- Refuses: an a whose `to_f` is not a finite number above 0, and an x whose
  `to_f` is NaN (both are read by `to_f`, so a numeric String reads as its
  number, a non-numeric one as 0.0, and an object without `to_f` raises
  `NoMethodError`).

#### `Gamma.regularized_upper(a, x)`

- Answers: Q(a, x) = 1 - P(a, x); 1.0 for x <= 0.
- Guarantees and refusals: `regularized_lower`'s.

#### `Significance.normal_quantile(p)`

- Answers: the z with Phi(z) = p; -Infinity at 0 and Infinity at 1.
- Guarantees (*exact*): z maps back to p within 1e-12 relative, from p = 1e-300
  to 1 - 1e-6.
- Refuses: a p outside [0, 1], NaN included.

#### `Significance.chi_squared_survival(statistic, degrees_of_freedom)`

- Answers: P(X >= statistic) for X ~ chi-squared(degrees_of_freedom), Q(dof / 2,
  statistic / 2); 1.0 for a statistic of 0 or below.
- Refuses: degrees of freedom not above 0 (for a statistic above 0).

#### `Significance.sample_ratio_p_value(observed:, expected:, minimum_total: 200)`

- Takes: Hashes of arm => count and arm => expected count (Symbol or String
  keys, as `observed`'s); `minimum_total`, read as given above or below its
  default.
- Answers: Pearson's chi-squared p on k - 1 degrees of freedom over the arms
  expected to get traffic; nil under `minimum_total` or with fewer than two
  such arms; 0.0 when an arm expected to get nothing got some.
- Guarantees (*exact*): `chi_squared_survival`'s accuracy.
- Refuses: nothing of its own (a non-Hash raises `NoMethodError`).

### Estimates of several parts (`SharedCredit`)

#### `SharedCredit.with_prior(estimate, strength:)`

- Takes: an estimate in `SharedCredit.estimate`'s shape (an `Estimate`, or a
  plain Hash such as one read back from JSON), read by Symbol or String, with
  its weight sums and Kish's sizes; `strength`, a finite real number >= 0, in
  visits (0 tops nothing up).
- Answers: an `Estimate` with the same arms in the same order, the parts (none
  for a plain Hash), the weight sums and Kish's sizes (the evidence's own),
  the means and covariance under the prior pooled from its own credited arms
  (the rule under
  [An arm of a handful of visits](#an-arm-of-a-handful-of-visits-with_prior)),
  and its rounding: the given one's (0.0 for a plain Hash) or, where larger,
  that of an entry it changed.
- Guarantees (*exact*: plain arithmetic in a fixed order, no draw and no `Math`
  function, so the same Floats everywhere): the rule, against a reference in
  Rationals that reads the rows, from rows and from per-arm sums alike; the
  covariance exactly symmetric wherever the given one is (as every estimate
  `estimate` answers is), and taken by `LinearValue` at the answer's rounding;
  an arm at or above the strength, and an arm with no credit, keeps its means
  and covariance block bit for bit; with every credited arm at or above it the
  answer holds the given means, covariance and rounding, so every seeded
  `LinearValue` draw and summary is the one the estimate gives; in an ordinary
  test (no row credits two arms) an arm at or above the strength draws the same
  values beside an arm topped up; a part every credited arm holds at one mean
  keeps that mean exactly; the given estimate is not changed.
  Measured at a strength of 20: a version of one visit reads P(best) between
  0.41 and 0.77, and no version of three visits or fewer reaches 0.95.
- Refuses: a strength below 0 or not a finite real number; an estimate that is
  not a Hash, or whose means, covariance, weight sums or Kish's sizes are
  missing or malformed; an arm with means but a Kish's size of 0, each before
  anything is computed; and, as it is computed, an estimate whose figures leave
  a Float's range under the prior ("the sums leave a Float's range").

## Bandit simulations: measured results

`spec/simulations` holds ten files, tagged `simulation: true` and run only
with `SIMULATIONS=1` (`bundle exec rspec spec/simulations`; CI's simulations
job runs them nightly, by hand, and on a change to the code they measure). Each
example asserts its bounds; the figures beside the bounds were measured with
the same code and seeds, and the files new in 0.3.0 and 0.4.0 print theirs
through RSpec's reporter (the JSON formatter keeps them under "messages"). Every
run is seeded, so each count below is exact for this version of the code, and
every shape is generated here from round generic values. The eight files from
0.2.0 and 0.3.0 were measured on 2026-10-31 with skiftet_statistical 0.3.0, and
again on 2026-11-02 with 0.4.0, every count the same; the two files
new in 0.4.0 were measured on 2026-11-02. Ruby 3.4.9, YJIT on, as CI runs
them.

The two simulations from 0.2.0 replay A/B tests through `Bandit::Allocator` (and
`Bandit::WinHold`) in memory: every pass, 40 visits are drawn by the current
weights, each exposure counts 1 / the weight it was drawn under, and the
allocator is called with the previous weights, as an allocation job does every
15 minutes. Their three examples assert the bounds; the other figures were measured with the
same code and seeds. Moved from the app that first ran them, with the same
seeds, runs, passes, draws and bounds; measured again here on 2026-10-15 with
skiftet_statistical 0.2.0 and Ruby 3.4.9, and every count the app had recorded
came out the same.
Measured again on 2026-10-31 with 0.3.0, the three asserted examples counted
exactly as below (15 / 200; 0 / 100, an equal arm in 21 / 100; worse 0, better
46, undecided 154 / 200). The other counts are 0.2.0's, from an allocator that
0.3.0 and 0.4.0 keep bit for bit for every call that declares no per-arm share.
In 0.4.0 these simulations draw a visit's arm by `Draw.from_weights`, and over
all 12,263,200 draws of the four files that draw by weight it answered the
draw the files had written by hand every time.

### The automatic win (`allocator_simulation_spec.rb`)

400 passes of 40 visits, 400 draws.

| Runs | Arms (true conversion rate) | Bound (asserted) | Measured |
|---|---|---|---|
| 200 (seeds 20 261 003 + run) | four arms at 30 % | an arm crowned in fewer than 20 / 200 | 15 / 200 (7.5 %) |
| 100 (seeds 30 261 003 + run) | three arms at 30 %, one at 25 % | the worse arm crowned in 0 / 100 | 0 / 100 (an equal arm in 21 / 100) |

With equal arms a crowned arm costs nothing (it is as good as the rest, and its
expected loss is at most `win_loss`), so the bound caps noise wins; a truly
worse arm is never crowned.

### The hold (`win_hold_simulation_spec.rb`)

Two arms, 11 % against 10 % (the worse arm), 400 passes of 40 visits (100
hours), 200 runs (seeds 40 261 003 + run). The decision step is an allocation
job's: the allocator with `hold_satisfied: hold.satisfied?`, and the same seed again with
`hold_satisfied: false` when the arm it crowns is not the arm that held.

- WITHOUT the hold (the allocator alone, `hold_satisfied: true` at every pass):
  the worse arm crowned in 26 / 200 (13.0 %) at 400 draws, 24 / 200 (12.0 %) at
  2 000 draws.
- The hold's code with no wait (`hold_seconds: 0`), 1 000 draws: the worse arm
  crowned in 13 / 200 (6.5 %).
- WITH the hold (24 hours): the worse arm crowned in 0 / 200 at 400, 1 000,
  2 000 and 20 000 draws; the better arm in 41-46 / 200 (43 at 400, 46 at
  1 000, 41 at 2 000, 44 at 20 000), and the rest undecided after 100 hours.
- The asserted example runs at 1 000 draws: the worse arm at most 6 / 200 and
  the better arm at least 20 / 200 (a hold that never releases would crown
  nothing and pass the first bound alone). Measured: worse 0 / 200, better
  46 / 200, undecided 154 / 200.

### A value of several parts (`value_allocation_simulation_spec.rb`)

Each run replays a caller's passes in memory: visits drawn by the latest
weights, each credited 1 / the share it was drawn under; the per-arm Hájek sums
turned into the arms' joint normal (`SharedCredit.from_sums`), the value drawn
with its uncertain weights (`LinearValue.summary`, P(best) among the free arms)
and handed to `Allocator.from_summary`. A pass where a free arm has no credit
yet takes the warm-up split (`summary: nil`). Where marked, the same shape also
runs on `Allocator.call`'s yes/no tallies of the binary part. Unless stated, a
visit's parts are a binary part worth 1 and a count part (1, 2 or 3 when not 0)
worth uniform [0.25, 0.75], 40 visits a pass.

- **(a) Floors in every phase.** A reference arm floored at 0.05 and four arms
  at 0.02 (one arm clearly best), 400 passes walking warm-up, shifting, a pause
  on a stored split that breaks every floor, the automatic win and its stored
  win, and the reopened test; the same arms with one arm fixed at 0.30 beside
  them. 7 runs: floored on the value model 3, fixed on the value model 1,
  floored on tallies 2, fixed on tallies 1. Bound (asserted): no share below its
  floor or off its fixed share in any phase; every phase and an automatic win
  in every run. Measured: 0 breaches in 2,800 passes; per run warm-up 38 to 46
  passes, shifting 204 to 212, paused 20, won 130, one automatic win each;
  arm-passes held exactly at a floor 1,411, 1,428 and 1,428 (floored, value),
  1,054 (fixed, value), 1,364 and 1,428 (floored, tallies), 1,063 (fixed,
  tallies).
- **(b) No end.** A reference arm (floor 0.05) and two arms (0.02), one at twice
  the others' rate, 2,000 passes at `hold_satisfied: false`, on the value model
  and on tallies. Bound (asserted): nothing crowned, no share below its floor, and
  the better arm named leader in at least 1,800 passes. Measured: crowned 0 and
  0; leader named in 1,926 (value) and 1,978 (tallies) passes; the final split
  0.05 / 0.02 / 0.93 in both.
- **(c) Late values.** Three arms whose values (0.15, 0.17 and 0.20 at the
  weights' centre) arrive 0 to 23 passes after the visit, uniformly, so every
  pass allocates on sums that lack the values still on their way; 60 passes,
  then the last values arrive; 200 runs. Bound (asserted): each arm's final
  estimate (`LinearValue`'s point) within 2 standard errors of the truth.
  Measured: bias -0.00054, -0.00197 and -0.00116 (standard errors 0.00166,
  0.00188, 0.00107; z -0.33, -1.05, -1.08). Before the last values arrived (at
  pass 60, reported): -0.02798, -0.03399 and -0.03958 (z -19.7, -22.3, -41.8).
- **(d) A drift shared by every arm.** Four equal arms, every probability three
  times its own over passes 41 to 60; the mean P(best) over passes 41 to 120 of
  30 runs. Bound (asserted): each within 0.05 of 1 / 4. Measured: 0.2254,
  0.2424, 0.2779 and 0.2543 over 2,400 scored passes. Held uneven: the same
  shape paused through the drift on a stored split giving one arm 0.65, 20
  runs. Bound (asserted): that arm's mean P(best) at most 0.30, while unweighted
  sums of the same visits (scored beside, never allocating) give it more than
  0.5. Measured: 0.2092 (the others 0.3129, 0.2569 and 0.2209) over 1,600
  passes; on unweighted sums 0.9601 (the others 0.0114, 0.0157, 0.0129). The
  reference arm reads above 1 / 4 for its spread: it had the least traffic
  through the drift.
- **(f) Coverage with drawn weights.** Three arms on a fixed split (0.2, 0.4,
  0.4), 6,000 visits a run, a count part that is 0 in 95 % or more of visits
  (worth uniform [0.5, 1.5]) and a binary part at 1.0 to 1.5 % (worth uniform
  [2, 6]), the true worths drawn inside those ranges per run; 4,000 joint draws;
  400 runs. Bound (asserted): each arm's 95 % interval and each difference's
  covers the truth in at least 372 of 400 (93 %). Measured: arms 378, 377 and
  383; differences 382, 385 and 383 (94.25 to 96.25 %).

### Pooled cells and two layers (`two_layer_simulation_spec.rb`)

- **(e) Cells whose levels differ.** 40 cells in 8 groups of 5, crossed with 4
  bands. Each cell's level is its factor (0.5 to 2.45 in steps of 0.05, in an
  order that follows neither group nor band) times its covariate mix: each
  visit's covariate value multiplies its probabilities by 0.5, 1.0 or 1.5, in
  proportions that differ by cell. No arm's values differ from the reference
  arm's in any cell. Three arms (the reference floored at 0.05, two at 0.02),
  30 visits a cell a pass, 50 passes, 3 runs. Each cell is allocated at every
  pass on its pooled estimate: its contrasts by `SharedCredit.contrast_for_fit`
  (Kish's minimum 20), `AdditiveContrasts.fit` over (arm, part) pairs, one
  `baseline` a part, `pooled_estimate`, `LinearValue.summary` and
  `from_summary` (warm-up 0, nothing crowned). The variances run as designed: a
  weekly fit by moments every 7 passes, each variance held at 0 unless its test
  of no variation rejects at 0.05 divided by that week's number of tests, and
  held fixed between. Bound (asserted): the reference arm's mean share across
  the 40 cells at a rank correlation with the level within plus or minus 0.3,
  and a spread (largest less smallest) at most 1.5 times the noise floor's, the
  same simulation, seeds, runs and passes with every cell's level equal.
  Measured: 0.1426 to 0.1492 (spread 0.0066, standard deviation 0.0018) over
  5,880 cell passes allocated on a pooled estimate, rank correlation -0.073;
  noise floor 0.1225 to 0.1291 (spread 0.0065, standard deviation 0.0014), so
  1.007 times; 21 weekly fits, each test read at 0.004167, 1 of 252 variances
  kept at its moment (0.004293) and none in the noise floor; 1,135 of 24,000
  cell contrasts left out of the fit (1,440 in the noise floor). Against a fixed
  0.03 the spread is 0.0066 (reported). With every pass fitting the variances by
  moments instead (reported, no bound): 0.11 to 0.2625 (spread 0.1525,
  standard deviation 0.0425), rank correlation -0.047, 1,051 contrasts left
  out; at no true variation a fitted variance comes out above 0 in about half
  the fits, and a cell then keeps that share of its own noise in its effects.
- **(j) Two layers.** Two options of identical value, each above an inner layer
  that picks one of three candidates (rates 0.30, 0.20 and 0.10). The warm
  option's candidates are held at 1,000 showings each; the cold option's start
  at Beta(1, 1) and learn from their showings, their best placed first, second
  or third by run. Each option's candidates are one posterior set
  (`Posteriors.beta`); each visit's item is drawn by the exact pick odds
  (`Posteriors#pick_probabilities`, explore 0.1) through `Draw.from_weights`,
  and each option is scored by its greedy pick (`Posteriors#greedy`) through
  `SharedCredit`: a visit credits every option whose greedy pick it showed,
  with 1 / the visit's probability of showing that item, the two layers' odds
  combined by `Posteriors.mixture`. 30 passes of 80 visits, 100 runs. Bound (asserted): both options' mean
  P(best) in 0.40 to 0.60, and the cold option at its floor at the end in at
  most 10 runs. Measured: 0.5807 and 0.4193 over 3,000 passes, the cold option
  at its floor in 2 of 100. Scored as they ran (each visit crediting its drawn
  option with 1 / its share; reported): 0.6273 and 0.3727, and an allocation on
  that score ends the cold option at its floor in 5.

### Tree recovery (`tree_recovery_simulation_spec.rb`)

Five-level generated trees, 1, 20, 200, 800 and 4,000 nodes a level, with
single-child parents (1, 10 and 40 at levels 1 to 3), parents that are one
child's near-whole (95 % of their exposure, at a rate of 0.05) and 2 % of the
leaves split between two parents, so counts above the leaves are real numbers.
One law for every item: multipliers Gamma(k, rate k) with k = 8, 8 and 16 at
levels 1 to 3, and at the leaves a two-groups law, hot share 0.02 and slab
strength 0.5. Nine items, three national (200,000 counts a half), three
regional (20,000) and three very local (2,000); each count Poisson, drawn twice:
half A is fitted, half B held out. 20 runs (seeds 20 261 131 + run).

- **Level strengths** (bound: within a factor of 2 in at least 18 of 20 runs):
  level 1 (true 8) in 20 of 20, fitted 5.657 four times, 6.727 twelve times and
  8.0 four times; level 2 (true 8) in 20 of 20, 5.657 nine times and 6.727
  eleven times; level 3 (true 16) in 20 of 20, 8.0 twice and 9.514 eighteen
  times. The upper levels read low (they absorb part of the variance below
  them), and the two level-3 runs at 8.0 sit on the bound's edge.
- **Leaves on `fit_leaves`** (bound: the slab strength within a factor of 2 and
  the hot share within 0.05, each in at least 18 of 20 runs): slab in 20 of 20
  (0.354 ten times, 0.5 nine times, 0.707 once), hot share in 20 of 20 (0.01
  nineteen times, 0.02 once); 7,200 parents a run, 0 to 7 of them not settled
  after 50 steps. The same leaves fitted on leave-one-out expected counts
  (`fit_strength`; reported, no bound): hot share 0.05 five times and 0.1
  fifteen times, slab 0.42 once, 0.5 twice, 0.595 twice, 0.841 twice, 1.0 five
  times and 1.189 eight times (up to 2.4 times the truth).
- **Single-child parents** (bound: every fit identical without their pairs):
  identical at level 2 (8 to 9 such pairs a run, 20 runs), level 3 (76 to 88,
  20 runs) and level 4 (269 to 276, 2 runs); `fit_leaves` identical without the
  360 parents of a single leaf a run (2 runs).
- **Held-out log score of the shrunken ratio less the raw one** (bound: better
  summed over the levels, and at each level below the first, in every run):
  level 2 better in 20 of 20 (min 251.3, median 478.7, max 749.0 nats), level 3
  in 20 of 20 (4,431.6; 5,364.2; 6,051.1), level 4 in 20 of 20 (46,538.5;
  48,844.4; 51,500.6). Level 1, where a node holds thousands of an item's counts
  and shrinking moves a share by parts in 10^5 (reported): better in 14 of 20
  (min -2.6, median 3.0, max 31.3).

### Contrasts with known variances (`contrasts_simulation_spec.rb`)

One root, 400 nodes under it ("tree:1"), 6 leaves a node ("tree:2"), 2 cells a
leaf (4,800 cells), crossed with grouping a (300 keys) and grouping b (150
keys), each cell's keys drawn uniformly; each cell's sampling variance uniform
on [0.4, 1.4] (mean 0.8999 as handed to `fit`). Three arms in one call at the
default 5 sweeps: x with every term at tau^2 = 4, y with every term at 0, z with
"tree:1" and a at 0 beside "tree:2" at 4 and b at 8. 100 runs (seeds
20 261 101 + run). Bound (asserted, in every run): a variance of at least 4
times the mean sampling variance (and of its groups' own) within a factor of 2;
a variance of 0 under a quarter of the mean sampling variance. Measured, the
mean (min to max) of 100 fits:

| Term | x (truth 4) | y (truth 0) | z |
|---|---|---|---|
| tree:1 | 3.9072 (2.9473 to 4.5137) | 0.0010 (to 0.0118) | truth 0: 0.0173 (to 0.1126) |
| tree:2 | 3.8028 (3.3739 to 4.2671) | 0.0049 (to 0.0387) | truth 4: 3.9761 (3.6161 to 4.3358) |
| a | 4.0323 (3.3199 to 4.6050) | 0.0012 (to 0.0088) | truth 0: 0.0006 (to 0.0111) |
| b | 3.9434 (2.8883 to 5.6817) | 0.0009 (to 0.0086) | truth 8: 7.9093 (6.0397 to 10.5169) |

The groups' own mean sampling variances: 0.7373 (x and z at "tree:1"), 0.0665
(y at "tree:1"), 0.4038 ("tree:2"), 0.0499 (a) and 0.0250 (b). Two leaves a
node, where the K - 1 in the moment matters most (bound: the mean of 100 runs
within 5 % of the truth): "tree:1" truth 2.0, mean 2.0097 (1.6855 to 2.3061);
"tree:2" truth 1.0, mean 1.0004 (0.7745 to 1.2209). Measured beside the shape
and kept out of it: with one cell a leaf, a leaf's deviation and its cell's
noise cannot be told apart, and at 5 sweeps a crossed grouping at 0 beside one
at 8 reads about 0.6 to 1.2.

### The argmax (`argmax_accuracy_simulation_spec.rb`)

30 fixtures from round values (closed forms, parameters well below 1 at either
end, U shapes, close competitors, up to 8 members, mixed widths, and sharp
members with alpha + beta from 10^3 to 2 * 10^5), each with the epsilon `pick`
explores at. `pick` takes its uniforms from one seeded Random a fixture, 10^4
draws; the reference for `probabilities` (at epsilon 0) is the plain argmax of
one `Sampler#beta` draw a member, 10^6 draws a fixture. Bound (asserted): every
member within 5 standard deviations, sqrt(p * (1 - p) / n) with p the answer of
`probabilities`. Measured, the largest gap of each fixture in standard
deviations: `pick` at most 2.44, `probabilities` at most 2.43.

| Fixture | Members `[alpha, beta]` | `pick`'s epsilon | `pick` | `probabilities` (largest absolute gap) |
|---|---|---|---|---|
| 0 | `[[1, 1], [2, 1]]` | 0.0 | 0.47 | 1.41 (6.62e-04) |
| 1 | `[[1, 1], [1, 1]]` | 0.0 | 0.80 | 0.44 (2.19e-04) |
| 2 | `[[2, 8], [8, 2]]` | 0.1 | 0.51 | 0.18 (7.55e-06) |
| 3 | `[[1, 0.05], [1, 1]]` | 0.0 | 0.57 | 0.78 (1.65e-04) |
| 4 | `[[0.05, 1], [1, 1]]` | 0.0 | 1.02 | 0.51 (1.08e-04) |
| 5 | `[[0.5, 0.5], [1, 1]]` | 0.0 | 0.74 | 0.18 (8.80e-05) |
| 6 | `[[0.2, 0.3], [0.3, 0.2]]` | 0.05 | 0.02 | 0.87 (4.14e-04) |
| 7 | `[[0.1, 0.1], [2, 2]]` | 0.0 | 1.44 | 0.61 (3.04e-04) |
| 8 | `[[3, 7], [4, 6], [5, 5]]` | 0.0 | 0.83 | 1.34 (4.08e-04) |
| 9 | `[[1, 9], [2, 8], [3, 7], [4, 6]]` | 0.05 | 1.66 | 0.35 (9.44e-05) |
| 10 | `[[2, 3], [3, 2], [5, 5], [1, 1], [4, 6]]` | 0.1 | 2.44 | 1.65 (6.99e-04) |
| 11 | `[[12, 88], [15, 85], [9, 91], [20, 180], [3, 17]]` | 0.1 | 1.00 | 0.66 (3.22e-04) |
| 12 | `[[0.5, 50], [50, 0.5]]` | 0.0 | 0.00 | 0.00 (1.24e-31) |
| 13 | `[[10, 10], [10.5, 9.5]]` | 0.0 | 1.03 | 0.65 (3.25e-04) |
| 14 | `[[1, 3], [3, 1], [2, 2]]` | 0.2 | 0.82 | 1.15 (4.77e-04) |
| 15 | `[[0.8, 0.8], [0.9, 0.7], [0.7, 0.9]]` | 0.0 | 1.29 | 0.89 (4.41e-04) |
| 16 | `[[3, 7], [3.1, 6.9], [2.9, 7.1], [3.05, 6.95]]` | 0.1 | 1.90 | 2.01 (8.67e-04) |
| 17 | `[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]` | 0.0 | 1.38 | 2.42 (9.01e-04) |
| 18 | `[[5, 1], [1, 5]]` | 0.0 | 0.59 | 0.38 (2.37e-05) |
| 19 | `[[100, 900], [1, 9]]` | 0.0 | 0.71 | 1.45 (7.06e-04) |
| 20 | `[[2, 2], [2, 2], [2, 2], [3, 3]]` | 0.3 | 1.45 | 2.29 (1.00e-03) |
| 21 | `[[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]` | 0.05 | 1.15 | 2.43 (8.03e-04) |
| 22 | `[[1, 2], [2, 1], [1.5, 1.5]]` | 0.1 | 1.24 | 1.07 (4.78e-04) |
| 23 | `[[4, 4], [4.4, 3.6]]` | 0.0 | 0.45 | 1.53 (7.54e-04) |
| 24 | `[[0.3, 3], [3, 0.3], [1, 1]]` | 0.0 | 1.08 | 1.94 (5.38e-04) |
| 25 | `[[700, 300], [7000, 3000]]` | 0.1 | 1.05 | 0.88 (4.39e-04) |
| 26 | `[[20000, 80000], [20, 80]]` | 0.0 | 0.85 | 0.03 (1.52e-05) |
| 27 | `[[50, 99950], [1, 999]]` | 0.0 | 0.55 | 1.40 (6.83e-04) |
| 28 | `[[99950, 50], [999, 1]]` | 0.0 | 0.34 | 0.31 (1.52e-04) |
| 29 | `[[60000, 140000], [60150, 139850]]` | 0.1 | 0.69 | 0.64 (2.94e-04) |

### The change test (`change_test_simulation_spec.rb`)

Two settings, each a stated false-alarm rate and a horizon. False alarms: one
null stream of 2,000 consecutive spans of 365 steps (730,000 steps, z ~ N(0, 1)
by `Sampler#gaussian`), the statistic starting again from 0 after each alarm.
Detection: 2,000 runs shifted by the smallest detectable change from their
first step, and the same change after 365 null steps. Bound (asserted): the
false alarms inside the stated rate's 95 % binomial interval, and the change
caught within the horizon in at least 1,000 of 2,000 runs, both ways.

| Reference | Stated rate | Threshold | False alarms in 730,000 steps (95 % interval) | Per 365 steps | Horizon | Smallest change | Caught from the start | Caught after 365 null steps |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 1 in 365 | 4.0744 | 1,966 (1,913 to 2,088) | 0.983 | 30 | 0.4752 | 1,239 (0.620) | 1,322 (0.661) |
| 0.25 | 1 in 1,460 | 9.3147 | 521 (457 to 544) | 0.260 | 60 | 0.3532 | 1,264 (0.632) | 1,305 (0.652) |

Measured beside it: 2,000 separate spans, each starting at 0, count 1.2 % fewer
alarms at reference 0.5 and 3.9 % fewer at 0.25, because every span's first
steps start from 0.

### The order rule at small-site traffic (`order_simulation_spec.rb`)

A list of K items (3, 5 or 10), each with its own true rate, fixed for a run and
dealt to the items in a random order a run. Every visit is shown the whole list
in one order and sees it down to a depth drawn from a reach curve, the share of
visits that see position l: (1, 0.6, 0.35) at K = 3,
(1, 0.70, 0.50, 0.35, 0.25) at K = 5 and 0.8^l at K = 10. Each item a visit
sees yields an outcome at its rate: yes/no, or an amount (log-normal, mean 200,
sigma 0.9, capped at 2 000). The caller's whole part: a view counts only where
an item was seen; every hour the set is rebuilt from the tallies
(`Posteriors.yes_no` on the flat prior, or `Posteriors.amounts` under
`Prior.amount(mean: 4.0, variance: 2_000.0, strength: 20.0)`), and every visit
of that hour is shown the set's `rank(rng:, shuffle: 0.10)`. 28 days of 24
hours, 50 seeded runs a cell (`Random.new(20 261 102 + 50 i + r)` for run r of
cell i).

Measured: lost, the expected outcomes the shown orders missed against the best
order (the items by true rate), as a share of the best order's, and the same for
a fixed order (one random order a run shown to every visit); the posterior-mean
order (the items by `summary(among: [])`'s means) on days 7 and 28, whether its
first item is the best one and whether it is the best order exactly. Mixed
visits: half shallow (reach (1, 0.5, 0.3, 0.15, 0.1), every rate halved), half
deep (reach (1, 0.9, 0.7, 0.55, 0.4), every rate times 1.5), the known weakness
of counting views only where seen, as an item shown low is seen mostly by deep
visits and reads a rate above the one it would show on top. Bound (asserted,
every cell): lost at most half the fixed order's, and the random share within 4
standard errors of 0.10 (measured 0.0990 to 0.1003); each cell's own bounds
below. The reference is the same model run 100 times a cell in another language
and generator, the figures the bounds were set from with margin (fixed lost /
lost / best on top day 7 / day 28 / exact order day 28); 50 runs and another
generator read near them, not equal to them.

| Rates (true) | Visits a day | Fixed order lost | Lost | Best on top, day 7 | Best on top, day 28 | Exact order, day 28 | Bound (asserted) | Reference (100 runs) |
|---|---|---|---|---|---|---|---|---|
| five, close: 5, 4.5, 4, 3.5, 3 % | 300 | 8.0 % | 2.4 % | 74 % | 88 % | 62 % | day 28 >= 75 %, exact >= 35 %, lost <= 4.5 % | 8.3 / 2.8 / 69 / 90 / 54 |
| five, close | 1 000 | 7.4 % | 1.6 % | 82 % | 100 % | 94 % | day 28 >= 90 %, exact >= 75 %, lost <= 3.0 % | 7.6 / 1.6 / 91 / 99 / 91 |
| five, close | 3 000 | 7.4 % | 1.1 % | 94 % | 100 % | 100 % | day 28 >= 90 %, exact >= 90 %, lost <= 2.0 % | 7.7 / 1.1 / 99 / 100 / 100 |
| five, clear: 8, 6, 5, 4, 3 % | 300 | 14.1 % | 2.5 % | 98 % | 100 % | 86 % | day 7 >= 90 % | 12.8 / 2.4 / 100 / 100 / 97 |
| five, clear | 1 000 | 14.3 % | 1.7 % | 100 % | 100 % | 100 % | day 7 >= 90 % | 12.8 / 1.7 / 100 / 100 / 100 |
| five, clear | 3 000 | 13.9 % | 1.5 % | 100 % | 100 % | 100 % | day 7 >= 90 % | 13.6 / 1.5 / 100 / 100 / 100 |
| three: 5, 4, 3 % | 1 000 | 7.7 % | 1.1 % | 98 % | 100 % | 100 % | day 28 >= 90 % | 7.0 / 1.1 / 100 / 100 / 100 |
| ten: 6 % down to 1.5 % by 0.5 | 3 000 | 19.6 % | 2.2 % | 100 % | 100 % | 100 % | day 28 >= 90 % | 18.6 / 2.2 / 98 / 100 / 100 |
| five amounts: 4, 3, 2.5, 2, 1.5 % | 300 | 13.8 % | 4.3 % | 82 % | 100 % | 68 % | day 28 >= 85 %, lost <= 6 % | 13.1 / 4.3 / 88 / 98 / 53 |
| five amounts | 1 000 | 13.7 % | 2.6 % | 92 % | 100 % | 84 % | day 28 >= 95 %, lost <= 4 % | 12.4 / 2.6 / 98 / 100 / 88 |
| five amounts | 3 000 | 12.6 % | 1.9 % | 100 % | 100 % | 100 % | day 28 >= 95 %, lost <= 3 % | 12.9 / 1.8 / 100 / 100 / 98 |
| five, close, mixed visits | 300 | 6.9 % | 2.3 % | 78 % | 94 % | 70 % | day 28 >= 75 % | 6.4 / 2.4 / - / 90 / 58 |
| five, close, mixed visits | 1 000 | 5.5 % | 1.5 % | 82 % | 98 % | 96 % | day 28 >= 90 % | 6.2 / 1.5 / - / 99 / 95 |

At 300 visits a day the order rule loses a fifth to a third of what a fixed
order loses, and the posterior-mean order finds the best item first in
most runs by day 28; at 1 000 and more it finds the best order exactly in most.
The mixed cells hold their bounds, though the deep visits flatter the items
shown low.

### The odds against the rule's visits (`odds_simulation_spec.rb`)

Visits: 2 * 10^5 calls of `rank(rng:)` a fixture on one seeded Random (seed
20 261 131); every head and every position against its exact value
(`pick_probabilities`, `positions`), and the orders by one chi-square test
against `order_probability` (each order with n p >= 5 its own bin, the rest
pooled into one). Bound (asserted): every head and position within 4 standard
errors, sqrt(p (1 - p) / n), the chi-square's p-value above 0.001, and no order
seen that the exact odds give no probability. Items a, b, c, ... in the order
their pairs are listed; slots count from 0.

| Fixture | Rule | Largest gap (standard errors) | Orders' chi-square | p |
|---|---|---|---|---|
| five Betas: [0.02, 19.98], [3, 60], [4, 70], [6, 95], [2, 45] | shuffle 0.1, explore 0.1 | 2.75 (d at slot 2) | 137.5 on 119 degrees | 0.1186 |
| four Betas: [0.3, 20], [1, 20], [2, 30], [5, 90]; views 50, 400, 400, 400 | explore 0.2 among items of at least 100 views | 1.12 (a at slot 1) | 15.8 on 23 | 0.8656 |
| four Normals near 5 a view: [5.0, 0.4], [5.3, 0.5], [4.8, 0.3], [5.1, 0.8] | shuffle 0.05 | 1.71 (c at slot 1) | 23.3 on 23 | 0.4457 |
| the four Betas above, c pinned at slot 2 | shuffle 0.1 | 1.33 (a at slot 3) | 2.3 on 5 | 0.8086 |
| five Normals of mixed widths: [1.0, 1.0], [1.5, 0.5], [0.5, 2.0], [1.2, 0.2], [0.8, 0.7] | explore 0.1, the top 2 | 1.34 (e as head) | 12.2 on 19 | 0.8781 |

Joint draws: 10^6 a fixture (seed 20 261 141), one draw an item, in order; how
often each item is the largest against `summary`'s p_best (and, for Normals,
`NormalArgmax.probabilities`, which read the same), and the mean of max - X_i
against its expected loss. Bound (asserted): each within 4 standard errors of
the draws. Measured, the gap of each item in standard errors:

| Fixture | P(best) | Expected loss |
|---|---|---|
| the four Normals near 5 a view | 1.60, 0.28, 1.57, 0.82 | 0.87, 0.74, 0.46, 1.34 |
| the five Normals of mixed widths | 0.41, 1.14, 0.79, 1.30, 0.01 | 1.20, 0.44, 0.55, 0.04, 0.39 |
| four Betas: [12, 88], [15, 85], [9, 91], [3, 17] | 1.37, 1.07, 0.37, 0.24 | 0.26, 0.18, 0.21, 0.31 |
| the four Betas with an alpha of 0.3 | 1.44, 1.16, 1.10, 1.61 | 1.20, 0.85, 1.44, 0.76 |

### Run time

On one development machine, Ruby 3.4.9 with YJIT, in seconds, both columns
measured the same way: each figure is the sum of a file's examples' own run
times as RSpec reports them, Ruby's boot and the file's load left out. 0.3.0 ran
on 2026-10-31 with the machine otherwise idle, 0.4.0 on 2026-11-02 while other
work kept it busy (a load average of 31 to 46), so the two columns differ by the
load as well as by the code.

| File | Examples | 0.3.0 | 0.4.0 |
|---|---|---|---|
| `allocator_simulation_spec.rb` | 2 | 150 | 173 |
| `win_hold_simulation_spec.rb` | 1 | 132 | 160 |
| `value_allocation_simulation_spec.rb` | 6 | 56 | 85 |
| `two_layer_simulation_spec.rb` | 3 | 68 | 96 |
| `tree_recovery_simulation_spec.rb` | 3 | 178 | 515 |
| `contrasts_simulation_spec.rb` | 2 | 17 | 19 |
| `argmax_accuracy_simulation_spec.rb` | 2 | 232 | 265 |
| `change_test_simulation_spec.rb` | 4 | 1 | 1 |
| `odds_simulation_spec.rb` | 3 | - | 33 |
| `order_simulation_spec.rb` | 13 | - | 667 |
| All | 39 | 833 (about 14 minutes) | 2,013 (about 34 minutes) |

In three other runs the order file alone took 263, 311 and 407 seconds, the
more the busier the machine. Without YJIT the eight files of 0.3.0 took about 22
minutes in earlier runs on the same machine. The budget is 25 minutes on CI's
Ruby 3.4; the job's timeout is 45 minutes (30 before 0.4.0).

A change that moves any of these counts changes the rule's behaviour; measure
again and update these figures in the same change.

## Development

```sh
bundle install
bundle exec rake spec      # run the specs
bundle exec rake rubocop   # lint
bundle exec rake           # both
SIMULATIONS=1 bundle exec rspec spec/simulations   # the bandit simulations (minutes)
RUBY_YJIT_ENABLE=1 SIMULATIONS=1 bundle exec rspec spec/simulations   # as CI runs them, in about 60 % of the time
```

## License

MIT — see [LICENSE.txt](LICENSE.txt).
