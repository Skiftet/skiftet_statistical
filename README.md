# skiftet_statistical

Skiftet's shared, dependency-free **statistics toolkit** — the workspace home for
reusable statistical analysis code, so the same z-test, percentile, or sampler
isn't re-implemented (differently) in every app.

Modules:

- **`Descriptive`** — mean, variance, standard deviation, percentiles/median.
- **`Significance`** — A/B significance testing: two-proportion z-test, Welch's
  t-test, exact normal CDF and two-tailed p-values.
- **`Sampler`** — Gamma/Beta/Gaussian random sampling, RNG-injectable.
- **`Bandit`** (+ `Policies`) — multi-armed bandit (Thompson Sampling,
  epsilon-greedy, UCB1, Softmax) for online explore/exploit decisions.

## Install

In a Gemfile (path dependency within the Skiftet workspace):

```ruby
gem "skiftet_statistical", path: "../skiftet_statistical"
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

## Development

```sh
bundle install
bundle exec rake spec      # run the specs
bundle exec rake rubocop   # lint
bundle exec rake           # both
```

## License

MIT — see [LICENSE.txt](LICENSE.txt).
