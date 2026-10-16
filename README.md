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

## Install

In a Gemfile (published on rubygems.org):

```ruby
gem "skiftet_statistical", "~> 0.2.0"
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

## Bandit simulations: measured results

`spec/simulations` replays A/B tests through `Bandit::Allocator` (and
`Bandit::WinHold`) in memory: every pass, 40 visits are drawn by the current
weights, each exposure counts 1 / the weight it was drawn under, and the
allocator is called with the previous weights, as an allocation job does every
15 minutes. Every run is seeded, so each count below is exact for this version
of the code. The three examples (`SIMULATIONS=1 bundle exec rspec
spec/simulations`) assert the bounds; the other figures were measured with the
same code and seeds. Moved from f1 (its `allocator_simulation_oracle_spec.rb`
and `win_hold_simulation_spec.rb`) with the same seeds, runs, passes, draws and
bounds; measured again here on 2026-10-15 with skiftet_statistical 0.2.0 and
Ruby 3.4.9, and every count f1 had recorded came out the same.

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

A change that moves any of these counts changes the rule's behaviour; measure
again and update this section in the same change.

## Development

```sh
bundle install
bundle exec rake spec      # run the specs
bundle exec rake rubocop   # lint
bundle exec rake           # both
SIMULATIONS=1 bundle exec rspec spec/simulations   # the bandit simulations (minutes)
```

## License

MIT — see [LICENSE.txt](LICENSE.txt).
