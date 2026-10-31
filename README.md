# skiftet_statistical

Skiftet's shared, dependency-free **statistics toolkit** — the workspace home for
reusable statistical analysis code, so the same z-test, percentile, or sampler
isn't re-implemented (differently) in every app.

Modules:

- **`Descriptive`** — mean, variance, standard deviation, percentiles/median.
- **`Significance`** — A/B significance testing: two-proportion z-test, Welch's
  t-test, exact normal CDF and two-tailed p-values, the normal quantile, the
  chi-squared survival and the sample-ratio check.
- **`Gamma`** / **`BetaFunction`** — the regularized incomplete gamma and beta
  functions, and the Beta quantile.
- **`Sampler`** — Gamma/Beta/Gaussian random sampling, RNG-injectable.
- **`JointBeta`** — joint Monte Carlo draws over Beta posteriors (P(best),
  expected loss).
- **`Bandit`** (+ `Policies`) — multi-armed bandit (Thompson Sampling,
  epsilon-greedy, UCB1, Softmax) for online explore/exploit decisions.
- **`Bandit::Allocator`**, **`Bandit::WinHold`**, **`Bandit::WeightedPosterior`**
  — an A/B test's traffic allocation: Thompson sampling with a floor per arm, a
  step limit, an automatic win and a hold on it (f1's rule).
  Since 0.3 also per-arm floors and fixed shares, and one phase rule for any
  posterior (`Allocator.from_summary`).
- **`LinearValue`** (with **`Cholesky`**) — P(best), expected loss and
  intervals of a linear value of several rates when both the rates (one joint
  normal) and their weights are uncertain.
- **`SharedCredit`** — Hájek inverse-propensity means of several parts per arm,
  with one covariance over every (arm, part) pair, when one row may credit
  several arms; its sums fold once and store as JSON.
- **`HierarchicalRates`** — Gamma-Poisson rates on a tree: a node's ratio to
  what its parent leads us to expect, the fit of a level's strength (and of a
  two-groups hot share), the leaf fit, and checks beside them.
- **`AdditiveContrasts`** — the arms' contrasts against a reference arm pooled
  across cells along a nested tree and crossed groupings, a test of no
  variation, each cell's level from all its visits, and the pooled estimate a
  cell is allocated on.
- **`BetaArgmax`** (with **`GaussLegendre`** and `BetaFunction.log_density`) —
  the exact probability that each of several Betas is the largest, the seedless
  pick it describes, and the greedy pick.
- **`Prequential`** — the prequential log score, and the bits per event a
  deeper predictor saves, with cluster-robust intervals.
- **`Cusum`** — a one-sided change test: its statistic, its run length, the
  threshold for a stated false-alarm rate and the smallest change it can see.

## Install

In a Gemfile (published on rubygems.org):

```ruby
gem "skiftet_statistical", "~> 0.3.0"
```

Or build/install locally:

```sh
cd skiftet_statistical
bundle install
gem build skiftet_statistical.gemspec
```

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
```

The arithmetic and its order are part of the API: an app that pins these
outputs (f1 does, at 1e-12) compares them with this exact sequence of
operations, so a reordered formula is a breaking change even when it is "the
same" formula.

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
  arms: %w[facebook whatsapp bluesky x email],
  policy: SkiftetStatistical::Policies::ThompsonSampling.new,
)

choice = bandit.select          # which channel to promote right now, e.g. "whatsapp"
# ... show that option to the user ...
bandit.record(choice, 1)        # reward: 1 = it converted, 0 = it didn't

bandit.best_arm                 # current best by empirical mean
bandit.stats                    # { "whatsapp" => { pulls:, mean:, reward_sum: }, ... }
```

Rewards are expected in **[0.0, 1.0]** — a binary `0`/`1` (e.g. "did this share
lead to a signup?") is the common case, but any value in that range works.

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

## Example: f1 share-channel bandit

The motivating use case — make the petition ShareStep's **primary** button the
channel that drives the most signups, while continuously testing the others:

```ruby
# Nightly (or per request) build a bandit from observed share -> signup data.
bandit = SkiftetStatistical::Bandit.from_h(
  Rails.cache.read("share_bandit_state") || { arms: SHARE_CHANNELS.map { { name: _1 } } },
  policy: SkiftetStatistical::Policies::ThompsonSampling.new,
)
primary = bandit.select            # the channel to feature as the primary CTA

# When a share converts:
bandit.record(channel, 1)
Rails.cache.write("share_bandit_state", bandit.to_h)
```

## A/B-test allocation (`Bandit::Allocator`)

The decision an A/B test's allocation job makes on every pass: how much traffic
each arm gets next, and whether an arm has won. It is f1's rule, moved here;
pure (no clock, no state), with the randomness only from the injected `rng`.

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

Every number above is a setting of `Allocator::Config`, with f1's values as
defaults:

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
crowns is not the arm that held (`satisfied_for?`), as f1's job does.

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
never refused for its own rounding.

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

## Bandit simulations: measured results

`spec/simulations` holds eight files, tagged `simulation: true` and run only
with `SIMULATIONS=1` (`bundle exec rspec spec/simulations`; CI's simulations
job runs them nightly, by hand, and on a change to the code they measure). Each
example asserts its bounds; the figures beside the bounds were measured with
the same code and seeds, and the six files new in 0.3.0 print theirs through
RSpec's reporter (the JSON formatter keeps them under "messages"). Every run is
seeded, so each count below is exact for this version of the code, and every
shape is generated here from round generic values. Measured on 2026-10-31 with
skiftet_statistical 0.3.0 on Ruby 3.4.9, YJIT on, as CI runs them.

The two simulations from 0.2.0 replay A/B tests through `Bandit::Allocator` (and
`Bandit::WinHold`) in memory: every pass, 40 visits are drawn by the current
weights, each exposure counts 1 / the weight it was drawn under, and the
allocator is called with the previous weights, as an allocation job does every
15 minutes. Their three examples assert the bounds; the other figures were measured with the
same code and seeds. Moved from f1 (its `allocator_simulation_oracle_spec.rb`
and `win_hold_simulation_spec.rb`) with the same seeds, runs, passes, draws and
bounds; measured again here on 2026-10-15 with skiftet_statistical 0.2.0 and
Ruby 3.4.9, and every count f1 had recorded came out the same.
Measured again on 2026-10-31 with 0.3.0, the three asserted examples counted
exactly as below (15 / 200; 0 / 100, an equal arm in 21 / 100; worse 0, better
46, undecided 154 / 200). The other counts are 0.2.0's, from an allocator that
0.3.0 keeps bit for bit for every call that declares no per-arm share.

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
hours), 200 runs (seeds 40 261 003 + run). The decision step is f1's: the
allocator with `hold_satisfied: hold.satisfied?`, and the same seed again with
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
  or third by run. Each visit's item is drawn by `BetaArgmax.probabilities`
  (epsilon 0.1), and each option is scored by its greedy pick
  (`BetaArgmax.greedy`) through `SharedCredit`: a visit credits every option
  whose greedy pick it showed, with 1 / the visit's probability of showing that
  item. 30 passes of 80 visits, 100 runs. Bound (asserted): both options' mean
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

### Run time

On one development machine, Ruby 3.4.9 with YJIT, 2026-10-31 (seconds, the sum
of each file's examples):

| File | Examples | Seconds |
|---|---|---|
| `allocator_simulation_spec.rb` | 2 | 150 |
| `win_hold_simulation_spec.rb` | 1 | 132 |
| `value_allocation_simulation_spec.rb` | 6 | 56 |
| `two_layer_simulation_spec.rb` | 3 | 68 |
| `tree_recovery_simulation_spec.rb` | 3 | 178 |
| `contrasts_simulation_spec.rb` | 2 | 17 |
| `argmax_accuracy_simulation_spec.rb` | 2 | 232 |
| `change_test_simulation_spec.rb` | 4 | 1 |
| All | 23 | 833 (about 14 minutes) |

Without YJIT the same files took about 22 minutes in earlier runs on the same
machine. The budget is 25 minutes on CI's Ruby 3.4 (the job's timeout is 30).

A change that moves any of these counts changes the rule's behaviour; measure
again and update this section in the same change.

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
