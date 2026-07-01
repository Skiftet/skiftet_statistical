# Changelog

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
