# Changelog

## [0.2.0] - 2026-10-16

The statistics f1 wrote more than once, and f1's A/B-test allocator, in one
place. Pure Ruby, no new dependency; each function keeps f1's arithmetic in
f1's order, so on the inputs f1 gives it (and the same seed) it answers as f1
did, bit for bit.

- `SkiftetStatistical::Gamma` — `regularized_lower(a, x)` and
  `regularized_upper(a, x)`, the regularized incomplete gamma functions P and Q
  (series below x = a + 1, Lentz continued fraction above).
- `SkiftetStatistical::BetaFunction` — `regularized(x, alpha, beta)`, the
  regularized incomplete beta function, and `quantile(p, alpha, beta)`, its
  inverse by bisection.
- `SkiftetStatistical::JointBeta` — `each_draw(posteriors, draws:, rng:)` (one
  row per draw, arms in order, an Enumerator without a block) and
  `summary(posteriors, draws:, rng:)` (`{ samples:, wins:, losses: }`, a tie
  split equally).
- `Significance.normal_quantile(p)`, `.chi_squared_survival(statistic, dof)`
  and `.sample_ratio_p_value(observed:, expected:, minimum_total: 200)`.
- `SkiftetStatistical::Bandit::Allocator` — f1's A/B-test allocator:
  `call(arms:, tallies:, status:, winner_arm:, rng:, previous_weights: nil,
  draws:, hold_satisfied: true, allocation: :adaptive, config: Config.new)`
  returns `{ weights:, phase:, stats:, winner:, result_changed:, leader: }`
  (Thompson sampling on joint Beta draws, a floor per arm, a step limit, an
  automatic win and the won split); `won_weights(arms, winner, config:)`;
  `Allocator::Config` (warmup_exposures, floor, max_step, win_p_best,
  win_loss, won_share, won_rest, result_changed_below, credible_mass,
  prior_alpha, prior_beta; f1's values by default); `Allocator::
  FixedAllocation` (an `ArgumentError`), raised for a fixed split.
- `SkiftetStatistical::Bandit::WinHold` — the hold on an automatic win:
  `new(leading_arm:, leading_since:, now:, last_pass_at:, hold_seconds:
  86_400, max_gap_seconds: 1_200)` with `satisfied?`, `interrupted?`,
  `satisfied_for?(arm)` and `after_pass(leader)`.
- `SkiftetStatistical::Bandit::WeightedPosterior` — an arm's Beta posterior
  from raw or inverse-propensity counts: `count(tally)`, `beta(count,
  prior_alpha: 1.0, prior_beta: 1.0)`, `effective_exposures(count)` (the Kish
  n, never above the raw count).
- The bandit simulations (`spec/simulations`, tagged `simulation: true`) run
  only with `SIMULATIONS=1`; CI runs them nightly, by hand, and on a push or
  pull request that changes the bandit code. Measured results: README,
  "Bandit simulations: measured results".
- Unchanged: the 0.1 API, and how `Sampler` uses its RNG (a seed gives the same
  draws as in 0.1).

## [0.1.0] - 2026-06-23

Initial release — Skiftet's shared statistics toolkit.

- `SkiftetStatistical::Descriptive` — mean, variance, standard deviation,
  percentiles and median.
- `SkiftetStatistical::Significance` — A/B significance testing: two-proportion
  z-test, Welch's t-test, exact normal CDF and two-tailed p-values (consolidates
  the duplicated significance math from mej.la and skram.la).
- `SkiftetStatistical::Bandit` — arms + a pluggable policy, with `#select`,
  `#record`, `#best_arm`, `#stats`, and Hash (de)serialisation.
- `SkiftetStatistical::Arm` — online reward statistics (pulls, mean, variance,
  Beta-Bernoulli successes/failures).
- Policies: `ThompsonSampling` (Beta-Bernoulli), `EpsilonGreedy`, `UCB1`,
  `Softmax`.
- `SkiftetStatistical::Sampler` — RNG-injectable Gamma/Beta/Gaussian sampling, so
  every stochastic policy is deterministic under test.
