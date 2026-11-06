# Changelog

## [0.5.0] - 2026-11-06

Three methods, a minor version by the README's Stability section: one that
lets an estimate whose arms hold a handful of visits be drawn without reading
one visit as certain, and two for intervals that always hold the figure they
stand beside. No answer moves: every 0.4.0 call answers as before, for every
input, key and seed, and every 0.4.0 spec and pin passes unedited. The
documentation and examples written since 0.4.0, which move no answer either
(the Stability and Interface contract sections, the last three items), ship in
this version too. Pure Ruby, no new dependency, Ruby >= 3.1 as before.

- `SharedCredit.with_prior(estimate, strength:)`: the estimate under a prior
  pooled from its own credited arms, each arm's evidence topped up to
  `strength` visits, for `LinearValue` to draw from. Why: the sandwich of an
  arm of one visit, or of visits that all hold one value, is exactly 0, so
  `LinearValue` drew that arm as a point, and one visit at 1 against three at
  0, 1 and 0 read P(best) 0.994; at a strength of 20 it reads 0.620 (20 000
  draws at one seed). The prior's mean per part is the credited arms' mean
  weighted by their Kish's sizes; its per-visit covariance is theirs (the
  sandwich times Kish's size) plus their spread around that mean. An arm of
  Kish's size n below the strength k takes e = k - n visits of it: means
  (e * m0 + n * r) / k, covariance block
  (n / k)^2 * cov + (e * V0 + (e * n / k) * g * g^T) / k^2 (g the arm's
  distance from m0), every covariance with another arm scaled by n / k. An arm
  at or above the strength, and an arm with no credit, keeps its means and its
  covariance block bit for bit; where every credited arm is at or above it the
  answer holds the given means, covariance and rounding, so every seeded draw
  and summary is the same. The answer keeps the arms, parts, weight sums and
  Kish's sizes (the evidence's own); its rounding is the given one's or, where
  larger, that of an entry it changed. Exact: plain arithmetic, no draw.
  Refused, as an `ArgumentError`: a strength below 0 or not a finite number,
  an estimate without its weight sums or Kish's sizes, an arm with means but a
  Kish's size of 0, and an estimate whose figures rise past a Float's largest
  value under the prior (figures so small that the prior's products fall below
  Float::MIN read as 0; no estimate `estimate` answers holds them).
- `BetaFunction.shortest_interval(mass, alpha, beta)`: the shortest interval
  holding `mass` of Beta(alpha, beta). With one peak inside (0, 1) it is the
  highest-density interval, its ends of one density, so it holds the mode; a
  density that falls from 0 (alpha <= 1 <= beta) gives [0, quantile(mass)],
  one that rises to 1 gives [quantile(1 - mass), 1]; a flat density the
  central interval; a U-shaped one the shorter one-sided interval. Why: under
  the flat prior the mode of Beta(1 + c, 1 + n - c) is the observed share
  c / n, and the equal-tailed interval leaves that share out at c = 0 and
  c = n (0 of 1 visit reads 0.013 to 0.842 equal-tailed, 0.000 to 0.776
  shortest), and out of fractional (weighted) counts just above 0; the
  shortest interval holds it for every alpha, beta >= 1. By bisection on the
  side of the peak nearer 0 (a peak above 1/2 is mirrored), no draw. Refused,
  as an `ArgumentError`: a mass not strictly between 0 and 1, then a
  parameter whose `to_f` is not a finite number above 0.
- `BetaFunction.mean_intervals(items, mass:)`: each item's interval of a mean
  of non-negative values, `{ key => { visits:, mean:, standard_error: } }` in,
  `{ key => [lower, upper] }` out, read as a share of its visits: with the
  visits' second moment q = n * s^2 + m^2, one event is worth u = q / m, the
  item holds k = n * m^2 / q events, and the interval is u times the shortest
  interval of Beta(1 + k, 1 + n - k). So the mean always lies inside, the
  interval is never below 0, its variance in the normal limit is s^2 exactly,
  and a value of 0 or 1 per visit gives the share's own interval. An item with
  a mean of 0 (no event, a standard error of 0) takes the set's value of one
  event (summed squares over summed values of the items above 0) and k = 0,
  so it still has an interval, from 0, which narrows as its visits grow; with
  no item above 0 it has none. An item with a mean above 0 answers from its
  own figures alone, the same in any set. nil for an item without data (no
  visits, no mean or no standard error). Refused, as an `ArgumentError`: a
  mass not strictly between 0 and 1, items that are not a Hash of Hashes, a
  figure given beside the other two that is not a finite number, a negative
  visit count, mean or standard error, and an item whose arithmetic leaves a Float's range (a
  value of one event that is not a finite number above 0, as only extreme
  figures give it, or visits past about 2**53 that all hold one value), named
  in the refusal, rather than answered with an infinite or empty interval.
- `spec/shared_credit_prior_spec.rb`: the rule by hand and against a Rational
  reference that reads the rows, from rows and from per-arm sums; the answer
  unchanged, and every seeded draw and summary the same, where every arm is at
  or above the strength; an arm at or above it drawing the same values beside a
  topped-up one in an ordinary test; a table over visit counts and goals
  (values of one part, of two parts at uncertain prices, and of amounts): at a
  strength of 20 a version of one visit reads P(best) between 0.41 and 0.77
  (0.98 or more, or 0.02 or less, in 12 of the table's 16 rows on the estimate
  itself) and none of three visits or fewer reaches 0.95 (at most 0.90); two
  versions of one true value read decided (a P(best) at or above 0.95) in 0 of
  250 seeded datasets, against 158 on the estimate itself; the refusals.
- `spec/beta_function_interval_spec.rb`: the closed forms (Beta(1, b),
  Beta(a, 1), the flat density, a symmetric peak), and over a table of
  parameters and masses the mode held, the mass held, one density at both
  ends and no wider than the equal-tailed interval or any interval of a
  400-point grid; every count of 1 to 60 visits holds its share, where the
  equal-tailed interval misses exactly at 0 and at n; fractional counts just
  above 0; the U-shaped and monotone cases; refusals. For the means: a 0/1
  value gives the share's interval, 200 seeded sets hold every mean and never
  go below 0, the zero item's closed form and its narrowing, no interval in a
  set of zeros, the normal limit, an item whose visits all hold one value, an
  item answered alike alone and beside others, missing data, keys and field
  names, refusals, and the items whose arithmetic leaves a Float's range.
- The README's Interface contract lists the three methods, each with what it
  takes, answers, guarantees (each *exact*) and refuses, as its doc comment
  states them, and `spec/interface_contract_spec.rb` indexes the examples
  behind their 22 guarantees.
- Costs, measured as CPU time with YJIT on one development machine while other
  work kept it busy (a load average near 30): `shortest_interval` 0.06 ms for
  a one-sided interval and 0.7 to 2.0 ms with one peak inside (Beta(3, 341) to
  Beta(40 001, 1 160 001)); `mean_intervals` of ten items 9.9 ms;
  `with_prior` of four arms and two parts 0.04 ms.
- README: a Stability section (a change that moves a seeded answer for the same
  inputs is a major or minor version, as its entry here states, never a patch)
  and an Interface contract section: for each method an application calls,
  what it takes, what it answers, what it guarantees (determinism, which inputs
  move which outputs, accuracy) and what it refuses. The same contract is in
  each method's doc comment.
- `spec/interface_contract_spec.rb` indexes, method by method, the examples
  that back each stated guarantee, and fails when a cited example is gone, a
  listed method is no longer public, or the README's list differs.
- New examples: `BetaFunction.regularized` and `.quantile` at parameters from
  1 000 to 50 000 against an exact binomial reference, with the symmetry
  identity, I_(1/2)(a, a) = 1/2, the quantile's inversion and the 2.5 % and
  97.5 % quantiles at the skew-corrected normal approximation (accuracy now
  stated: 4 n ln n machine epsilons, relative, n = alpha + beta);
  `Posteriors#rank`: an added item absent from `may_lead` changes neither the
  others' order, the leader nor the branch over 1 000 keys with `explore` and
  `min_views` set, and a `min_views` below 0 reads as none; `Prior.pooled`
  refuses a negative strength with or without a group; a posterior set is a
  value no reader writes to; `summary` is exact and per item;
  `Draw.from_weights`' refusals; `WeightedPosterior`'s reading of its counts by
  `Integer()`; `JointBeta`'s refusal of a parameter not above 0;
  `sample_ratio_p_value`'s `minimum_total` above its default.

## [0.4.0] - 2026-11-03

The generic bandit: one posterior set of items, ordered and picked by one rule
from a key, given uniforms or an RNG, with the odds of what that rule draws
(exact within the sizes stated below, otherwise a seeded Monte Carlo estimate
with its standard error) and the summaries an allocation reads. The caller
hands it items, views and outcomes (yes/no or amounts) and decides what counts
as each. Pure Ruby, no new dependency, Ruby >= 3.1 as before. 0.3.0 was never
released, so its additions (its entry below) ship in this version. Every 0.3.0
module answers as 0.3.0 did, for every input, and every 0.3.0 spec and pin
passes unedited.

- `Bandit::Posteriors`, the posterior set. `yes_no(items, priors: nil)`: Beta
  posteriors from tallies `{ item => { views:, outcomes: } }`, optionally with
  inverse-propensity sums (`weighted_views`, `weighted_outcomes`,
  `weighted_views_squared`) rescaled to the Kish size; priors nil (Beta(1, 1)),
  one pair or one pair an item. `amounts(items, priors:)`: Normal posteriors
  for the mean per view from `views`, `outcomes` (the sum of amounts) and
  `outcome_squares`, under a `Prior::Amount` (or one an item). `beta(pairs,
  views: nil)` and `normal(pairs, views: nil)` take pairs as they are. A set
  answers `items`, `kind`, `parameters`, `views`, `values(rng:)` (one draw an
  item) and `greedy(may_lead: nil)` (the first largest mean). Items are
  non-blank Strings, any number of them, none included, in input order, which
  is the tie order everywhere; a Symbol, a blank or repeated name, a name
  holding a NUL byte or an earlier name's bytes is refused when the set is
  built. `EXACT_PICK_MAX` (20), `EXACT_ORDER_MAX` (10), `LARGEST_TOTAL` (1e6).
- One pair everywhere: a Beta pair above `LARGEST_TOTAL` is scaled to it, its
  mean kept, in `parameters`, the one pair every draw (`rank`, `pick` and
  `values`, from any source), every odds call (exact or Monte Carlo),
  `summary`'s P(best) and loss and `greedy` read; the pairs as given feed
  `summary`'s means and intervals alone. A scaled pair never sums above
  `LARGEST_TOTAL`, so a set built from another set's `parameters` keeps them
  bit for bit. A Beta parameter above 0 but below
  `BetaArgmax::SMALLEST_PARAMETER` (1e-6) builds; each computation that cannot
  read it (the quantile draws, the odds, `summary`) refuses it, and the RNG
  draws take it.
- The order rule, written once: `rank(key:, uniforms:, rng:, shuffle: 0.0,
  explore: 0.0, may_lead: nil, min_views: 0.0, pinned: {}, top: nil)` answers
  an `Order` (`items`, `leader`, `branch`, `draws`), and `pick` (`rank` with
  top 1 and no pins) a `Choice` (`item`, `head`, `branch`, `draws`), both
  frozen keyword Structs. The branch is `:shuffled` (by priority), `:explored`
  (an eligible item of smallest priority leads, the rest by draw) or
  `:thompson` (by draw); pinned items keep their positions and draw nothing;
  the leader is the head when it has at least `min_views` views. Exactly one
  source: a key, whose uniforms are `Draw.uniform_for(*key, "explore")` and,
  per item, `(*key, "priority", item)` and `(*key, "draw", item)`, so a key
  draws the same order in every process and no item's uniforms depend on
  another item;
  given uniforms; or an RNG (one branch value, the priorities, one `Sampler`
  draw a free item, nothing else from the stream). A key or given uniforms
  draw by quantile (`BetaArgmax.position`, or mean + sd times the Normal
  quantile on the open grid), an RNG by `Sampler`; each draws exactly from the
  posteriors, so one set of odds covers all three. Over 1 000 keys with
  `shuffle: 0.1` and no exclusion or pin, an added item never changed the
  others' relative order (with an item pinned at position 1 it did on 352).
  Refused, each an `ArgumentError` before any draw and before anything is
  taken from an RNG: a rule outside its bounds, a pinned position or name that
  is not an item, a name pinned twice, a `may_lead` name that is not an item,
  `top` below 1, no source or more than one, a malformed `uniforms` Hash, and
  under a key or given uniforms a Beta parameter below 1e-6. An RNG's own
  answer is checked as it comes: one that is not a Float in [0, 1), or a raise
  from its `rand`, is an `ArgumentError` at that answer, after the draws taken
  before it.
- `Order#repin(pinned)`: `rank(pinned:)` again on an order's own quantile draws.
- Exact Normal keys: the order rule compares two Normal draws, from every
  source, by the exact sign of (mean_i - mean_j) + (sd_i z_i - sd_j z_j), each
  mean + sd * z formed exactly, never rounded, so two draws never tie or
  misorder where a standard deviation lies below a rounding of its mean, and
  the odds are the probability of what the rule draws; `draws` answers the
  rounded values.
- The odds of what the rule draws: `pick_probabilities`, `order_probability`,
  `positions` and `order_summary`, with `Choice#probability` and
  `Order#probability`. They are exact, by quadrature, for a pick among up to
  `EXACT_PICK_MAX` (20) contenders and for orders and positions over up to
  `EXACT_ORDER_MAX` (10) free items with none barred from leading; the orders
  and positions sum on every item's quantiles at 400 levels an item (an
  internal grid, not a public constant), each sum on points where its own
  item's points keep their digits (Betas share one grid in the position on a
  log scale the rule compares; each Normal sums on its own points in its own
  standardised variable, the other items read there), extrapolated from the
  points and their every other level, within 4e-6 of closed forms. Otherwise
  they are one of two stated Monte Carlo forms on `Random.new(seed:)`: the
  pick's share of draws in which each contender is the largest, and the rule's
  own visits for orders and positions, each drawing on the scaled pairs the
  exact odds read. Both forms refuse a Beta parameter below 1e-6 at every
  size, as a keyed draw does.
- `Posteriors.mixture(shares:, probabilities:)`: the probability that a
  two-layer draw shows each item, the sum over branches of a branch's share
  times its odds.
- `summary(credible_mass: 0.95, among: nil)`: per item its views, mean and
  equal-tailed interval on the pair as given (`BetaFunction.quantile`,
  accurate to about 2e-10 up to totals near 1e8; beyond about 3e8 one bound
  can fall on or past the mean), and for the items `among` names (at most 20)
  the exact P(best) and expected loss (never below 0), the shape
  `Allocator.from_summary` takes; `among: []` is a means-only summary for a set
  of any size, and `means` answers those means alone (`{ item => mean }`,
  frozen, bit for bit summary's), with no interval and no quantile call.
- `Bandit::Prior`: `flat`, `given(mean:, strength:)`, `amount(mean:,
  variance:, strength:)` (a frozen `Prior::Amount`), `borrowed(group:, own:)`
  (the one leave-one-out rule), `pooled`, `pooled_new`, `pooled_mean`,
  `tilts(multipliers, clip:)`, `chain(levels:, root_mean:, tilt:, clip:)` and
  `pooled_amount`, which refuses a group with nothing left out or no variance
  (the caller then states its own scale with `amount`).
- `Bandit::Decay`: `weight(age, half_life:, cap: nil)`, 2 ** (-age /
  half_life) and 0.0 past the cap, and `fold(rows, half_life:, cap: nil)`,
  every sum of the rows weighted.
- `Bandit::Draw`: `uniform(text)` (the first 8 bytes of SHA256, on the 2**53
  grid), `uniform_for(*parts)`, `seed_for(*parts)`, `weighted(weights,
  uniform)` (the one draw by weight, `[key, share]`; a weight that is not a
  finite number above 0 counts 0), `from_weights` (its key) and
  `open_uniform(uniform)` (clamped to [2**-53, 1 - 2**-53], the one reading of
  a quantile draw's uniform).
- `Bandit::NormalArgmax`: `probabilities(params, epsilon: 0.0)` and
  `expected_maximum(params)`, BetaArgmax's rule on Normals. Each member is
  integrated in its own standardised variable, so a narrow member far from
  the others is weighed on its own scale; each answer lies within
  2 k (k + 1) * 1e-10 + 2e-12 of the exact probability for k members.
- `Bandit::Pass.call(status:, hold:, crown: true, seed: Random.new_seed)`: one
  allocation pass with a hold on its win. The block is yielded (false,
  `Random.new(seed)`), then, only when `crown` and the status is "running" and
  the hold is satisfied for the probe's leader, again with (true, a fresh
  `Random.new(seed)`), so the probe and the crown read the same draws; the
  answer is the last block's Hash plus `ready:`.
- `BetaArgmax.position(uniform, alpha, beta)`, `distribution(x, alpha, beta)`
  and `expected_maximum(params)`, beside `probabilities`, `pick` and `greedy`,
  which are unchanged (`pick` compares its own 0.3.0 position). `position` is a
  guarded Newton step on the same distribution function. Against 0.3.0's
  bisection, recorded on 30 members at 1 000 uniforms each, the spec bounds
  |position - recorded| / max(1, |recorded|), an absolute error where the
  recorded position lies within 1 of 0 and a relative one beyond: 1e-12
  (measured at most 4.7e-13), 1e-9 above alpha + beta 1e4 (measured 3.0e-13)
  and 1e-8 at the smallest parameter (measured 9.5e-11).
- `Policies::ThompsonSampling` and `Policies::Softmax` are kept as 0.3.0 wrote
  them, as are `EpsilonGreedy`, `UCB1`, `Sampler` and every other 0.3.0 module:
  each answers as in 0.3.0 for every input, the same value or the same error,
  and the same use of the rng. New callers draw through `Bandit::Posteriors`.
- Every number the code new in 0.4.0 reads lies in one input domain: a real
  number whose Float is finite and at most `SkiftetStatistical::Numbers::LARGEST`
  (1e100) either side of 0. Anything else, a larger number included, is an
  `ArgumentError` where it is read, and no product the new code forms
  multiplies more than two numbers of that size, so no sum or product of
  inputs leaves a Float; every answer of the new bandit is checked finite
  before it is returned, and a Normal quantile draw reads its uniform on
  [2**-53, 1 - 2**-53].
- A caller's object whose method raises where the new code calls it (a key
  part's `to_s`, a name's `hash` or `eql?`, a list's `each`, an rng's `rand`),
  with a `StandardError` or a `NotImplementedError`, is an `ArgumentError`
  naming the input; a caller's String, Hash or Array is read by its contents,
  never by its own methods, so a missing key reads as absent whatever default
  the Hash holds, and a refusal names a class as Ruby records it. A list is
  any Enumerable but a Hash, read once by its `each`.
- The moments of amounts, the amounts posterior's mean and every shrunk mean
  (`Prior.pooled_mean`, `pooled`, `pooled_new`, `chain`) are formed exactly
  from the Floats given and rounded once to the nearest Float, and the
  posterior's standard deviation from those moments, so sums that cancel and
  products that would underflow change no answer. A caller's Hash is read by
  its keys as names (items, or a tally's sums): one that compares its keys by
  identity reads as a plain one, and two keys of one name are refused.
- Two new simulations, each asserting its bounds, every figure in the README,
  "Bandit simulations: measured results". The order rule at small-site
  traffic (`order_simulation_spec.rb`, 13 cells of 50 runs, 28 days each):
  with five items at close rates (5 down to 3 %) and 300, 1 000 and 3 000
  visits a day, `rank(rng:, shuffle: 0.10)` lost 2.4, 1.6 and 1.1 % of the
  best order's expected outcomes, where a fixed random order lost 8.0, 7.4 and
  7.4 %, and put the best item on top on day 28 in 88, 100 and 100 % of runs;
  on amounts (4 down to 1.5 %) it lost 4.3, 2.6 and 1.9 % (a fixed order 12.6
  to 13.8 %); with ten items at 3 000 a day 2.2 % (19.6 %); every cell's
  random share 0.099 to 0.1003 of its visits. The odds against the rule's own
  visits (`odds_simulation_spec.rb`, 2 * 10^5 visits on five fixtures): every
  head and position within 2.75 standard errors of its exact value, the orders'
  chi-square p-values 0.12 to 0.88; P(best) and expected loss within 1.61 and
  1.44 standard errors of 10^6 joint draws.
- The simulations draw by weight through `Draw.from_weights` (the four copies
  of that draw are gone), and no count moved: 12,263,200 draws compared, 0
  different. Block (j) of the two-layer simulation draws through the posterior
  set (`pick_probabilities`, `greedy`, `from_weights`, `mixture`), its figures
  unchanged (0.5807 and 0.4193; as they ran, 0.6273 and 0.3727). Every count
  and figure of 0.3.0's entry, measured again on 0.4.0, came out the same, its
  run time aside (below).
- The ten simulation files took 2,013 seconds with YJIT on one development
  machine while other work kept it busy (a load average of 31 to 46), the
  order file 667 of them (263 to 407 in other runs), each figure the sum of
  the examples' own run times, boot and load left out, as 0.3.0's figures
  are; CI's simulations job's timeout goes from 30 to 45 minutes.
- Costs, measured as CPU time with YJIT: a keyed `rank` of 8 Beta items 0.21
  ms (totals up to 1e3) and 0.54 ms (up to 1e5), of 8 Normals 0.08 ms; an RNG
  `rank` of 8 items 0.017 ms; `BetaArgmax.position` 0.02 to 0.06 ms from
  Beta(3, 300) to Beta(4 000, 96 000), where 0.3.0's bisection took 0.11 to
  0.92 ms in the same run; exact odds and summaries 2 to 560 ms
  a call (1.2 s for the positions of 10 items with exploration), so a caller
  computes them off the request path, once a pass, never once a visit.
- Docs in generic words; the gemspec's summary and description cover the
  whole toolkit, and its metadata names the source once (`homepage_uri` gone,
  `spec.homepage` kept), so the gem builds with no warning.

## [0.3.0] - not released; its additions ship in 0.4.0

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

Statistics an app had written more than once, and its A/B-test allocator, in
one place. Pure Ruby, no new dependency; each function keeps the app's
arithmetic in the app's order, so on the inputs the app gives it (and the same
seed) it answers as the app did, bit for bit.

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
- `SkiftetStatistical::Bandit::Allocator` — the A/B-test allocator:
  `call(arms:, tallies:, status:, winner_arm:, rng:, previous_weights: nil,
  draws:, hold_satisfied: true, allocation: :adaptive, config: Config.new)`
  returns `{ weights:, phase:, stats:, winner:, result_changed:, leader: }`
  (Thompson sampling on joint Beta draws, a floor per arm, a step limit, an
  automatic win and the won split); `won_weights(arms, winner, config:)`;
  `Allocator::Config` (warmup_exposures, floor, max_step, win_p_best,
  win_loss, won_share, won_rest, result_changed_below, credible_mass,
  prior_alpha, prior_beta; the app's values by default); `Allocator::
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

Initial release — a shared statistics toolkit.

- `SkiftetStatistical::Descriptive` — mean, variance, standard deviation,
  percentiles and median.
- `SkiftetStatistical::Significance` — A/B significance testing: two-proportion
  z-test, Welch's t-test, exact normal CDF and two-tailed p-values (consolidates
  significance math that two apps had each written).
- `SkiftetStatistical::Bandit` — arms + a pluggable policy, with `#select`,
  `#record`, `#best_arm`, `#stats`, and Hash (de)serialisation.
- `SkiftetStatistical::Arm` — online reward statistics (pulls, mean, variance,
  Beta-Bernoulli successes/failures).
- Policies: `ThompsonSampling` (Beta-Bernoulli), `EpsilonGreedy`, `UCB1`,
  `Softmax`.
- `SkiftetStatistical::Sampler` — RNG-injectable Gamma/Beta/Gaussian sampling, so
  every stochastic policy is deterministic under test.
