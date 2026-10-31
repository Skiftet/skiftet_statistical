# Changelog

## [0.3.0] - 2026-10-31

Allocating on a value of several parts, pooling across cells and groups,
hierarchical rates, an exact Beta argmax, a prequential score and a change
test, with per-arm floors and fixed shares in the allocator and one phase rule
for any posterior. Pure Ruby, no new dependency, Ruby >= 3.1 as before. Every
0.2.0 call answers as before, bit for bit (the same signatures, defaults,
arithmetic order and draw order), and every 0.2.0 spec passes unedited.

- `Bandit::Allocator.call` gains `arm_floors:` (`{ arm => floor }`, each in
  (0, 1], in place of `Config#floor`) and `fixed_shares:` (`{ arm => share }`,
  each in (0, 1), held exactly in every phase), both nil by default. One lift
  serves every phase; the free arms share `1 - F` by P(best) taken among
  themselves; a fixed arm's `p_best` and `expected_loss` are nil, and a fixed
  arm is never the won arm. Refused with an `ArgumentError` before any draw: a
  key that is not an arm, an arm in both, floors and shares summing above 1;
  every arm fixed raises `FixedAllocation`.
- `Bandit::Allocator.from_summary(arms:, summary:, exposures:, status:,
  winner_arm:, previous_weights: nil, hold_satisfied: true, config: Config.new,
  fixed_shares: nil, arm_floors: nil)` — the same paused / won / warm-up /
  automatic-win / shifting rule on statistics computed elsewhere (`summary`
  `{ arm => { p_best:, expected_loss:, mean:, lower:, upper: } }`, or nil to
  hold warm-up), answering `call`'s shape. `call` runs on the same private rule,
  so the two cannot disagree on a phase.
- `Bandit::Allocator.lift(weights, config: Config.new)` — 0.2.0's private
  `with_floor`, step for step, made public; the private method stays as a
  delegate.
- `SkiftetStatistical::LinearValue` — `summary(arms:, parts:, means:,
  covariance:, parameters:, weights:, draws:, rng:, credible_mass: 0.95,
  rounding: 0.0)` (per arm `point`, `mean`, `lower`, `upper`, `p_best`,
  `expected_loss`) and `each_draw(...)` (each draw's values, the rows `summary`
  is built on): P(best) of a linear value of several rates, one joint normal
  over every (arm, part) pair, with weights drawn from `fixed`, `uniform` or
  clipped `normal` parameters in every draw. Up to 7 arms and 12 parts
  (`MAX_ARMS`, `MAX_PARTS`). The draw order is part of the API.
- `SkiftetStatistical::Cholesky` — `factor(rows)`, the lower-triangular factor
  (a pivot with no variance set to 0 with its column), and `root(rows,
  rounding: 0.0)`, the same arithmetic largest pivot first, for a matrix of
  lower rank. No matrix library.
- `SkiftetStatistical::SharedCredit` — `estimate(rows, arms:, parts:)`: Hájek
  means per (arm, part) and their sandwich covariance when one row may credit
  several arms (an ordinary test credits only the arm drawn, 1 / p), with
  Kish's effective size; `accumulator(arms:, parts:)` (`Accumulator#add`,
  `#merge`, `#to_h`, `.from_h`, `.from_sums`, `#estimate`, `#arms`, `#parts`), the
  sufficient sums folded once and stored as JSON; `from_sums(sums, arms:, parts:)`, the same estimate from
  per-arm sums; `linear(estimate, weights:)` and `contrast(estimate, a, b,
  weights:)` at fixed weights; `contrast_for_fit(estimate, arm, reference,
  weights:, minimum_effective: 20)`, a contrast with one per-visit variance
  pooled over its two arms, nil under the Kish minimum. An estimate
  (`SharedCredit::Estimate`) answers `#parts` and `#rounding`; a covariance
  entry within its cancellation rounding answers exactly 0.0.
- `SkiftetStatistical::HierarchicalRates` — `leave_one_out_share`, `node`
  (a node's ratio, hot probability and tail under a Gamma-Poisson or two-groups
  prior), `fit_strength` (a level's strength and hot share on `GRID`, 57
  strengths from 2^-2 to 2^12, with its profile interval), `fit_leaves` (the
  leaf level's fit, each parent's null rate estimated jointly),
  `smooth_strength`, `marshall` and `boundary_test`. Counts may be real numbers.
- `SkiftetStatistical::AdditiveContrasts` — `fit(cells:, arms:, sweeps: 5,
  variances: nil)`: the arms' contrasts against a reference arm pooled along a
  nested tree and crossed groupings, the variances by moments or held fixed,
  answering an `AdditiveContrasts::Fit` whose `#variation_tests` gives each
  variance's Cochran's Q, degrees and p; `baseline(rows:, effects:)`, each
  cell's level from all its visits, adjusted on a covariate; `pair(arm, part)`,
  the name of a pooled (arm, part); `pooled_estimate(own:, effects:, baselines:,
  cell:)`, a cell's pooled estimate in `SharedCredit`'s shape, its level shared
  between the arms and each arm's effect its own.
- `SkiftetStatistical::BetaArgmax` — `probabilities(params, epsilon: 0.0)`,
  the exact probability that each of up to 20 Betas is the largest (composite
  Gauss-Legendre, exact to 1e-8), `pick(params, explore:, priorities:,
  uniforms:, epsilon:)`, the seedless pick those probabilities describe, and
  `greedy(params)`; `TAIL`, `BREAKS`, `NODES`, `MAX_MEMBERS`,
  `SMALLEST_PARAMETER`, `LARGEST_TOTAL`.
- `SkiftetStatistical::GaussLegendre.nodes(n)` — the n-point rule on [-1, 1],
  computed once per n.
- `BetaFunction.log_density(x, alpha, beta)`; `regularized` and `quantile` are
  unchanged.
- `SkiftetStatistical::Prequential` — `score(rows, depth:)`, `gains(rows,
  depths:)` (the bits per event a deeper predictor saves, with a cluster-robust
  standard error) and `clustered_mean(sums:, counts:, level: 0.95)`.
- `SkiftetStatistical::Cusum` — `run(z, reference:, threshold:)`,
  `average_run_length(shift:, reference:, threshold:)` (Siegmund's
  approximation), `threshold(reference:, false_alarm_steps:)` and
  `minimum_detectable(reference:, threshold:, horizon_steps:)`.
- Every new module reads numbers by one check (a finite real number, finite as
  a Float too), takes `draws:` as an Integer >= 0, never converted, and takes
  every random input as an injected `rng:` or given uniforms.
- `spec/pins/` pins the new outputs (1e-12 relative, 1e-9 through log-gamma, a
  pinned 0.0, 1.0 or infinity exact).
- Six new simulations in `spec/simulations` (`SIMULATIONS=1`), each asserting
  its bounds, every figure in the README, "Bandit simulations: measured
  results". Among them: no share below its floor or off its fixed share in
  2,800 passes through every phase; nothing crowned in 2,000 passes at
  `hold_satisfied: false`; values arriving up to 23 passes late leaving a final
  bias within 1.08 standard errors; 95 % intervals covering the truth in 94.25
  to 96.25 % of 400 runs; across 40 pooled cells whose levels differ, the
  reference arm's mean share spreading 1.007 times the equal-level noise floor's
  at a rank correlation of -0.073; every level's strength within a factor of 2
  in 20 of 20 runs; the argmax within 2.44 standard deviations of 10^4 picks and
  of 10^6 Beta draws on 30 fixtures; the change test's false alarms 0.983 and
  0.260 per 365 steps against a stated 1 and 0.25. All eight simulation files
  take about 14 minutes with YJIT on one development machine; CI's simulations
  job now sets `RUBY_YJIT_ENABLE`, and its path list holds every new module and
  `spec/pins`.
- The gemspec no longer lists a contact address (`spec.email` removed).

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
