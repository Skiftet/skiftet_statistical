# frozen_string_literal: true

require_relative "skiftet_statistical/version"
require_relative "skiftet_statistical/numbers"
require_relative "skiftet_statistical/sampler"
require_relative "skiftet_statistical/gamma"
require_relative "skiftet_statistical/beta_function"
require_relative "skiftet_statistical/joint_beta"
require_relative "skiftet_statistical/descriptive"
require_relative "skiftet_statistical/significance"
require_relative "skiftet_statistical/arm"
require_relative "skiftet_statistical/policies/base"
require_relative "skiftet_statistical/policies/epsilon_greedy"
require_relative "skiftet_statistical/policies/thompson_sampling"
require_relative "skiftet_statistical/policies/ucb1"
require_relative "skiftet_statistical/policies/softmax"
require_relative "skiftet_statistical/bandit"
require_relative "skiftet_statistical/bandit/checks"
require_relative "skiftet_statistical/bandit/tallies"
require_relative "skiftet_statistical/bandit/prior"
require_relative "skiftet_statistical/bandit/decay"
require_relative "skiftet_statistical/bandit/draw"
require_relative "skiftet_statistical/bandit/normal_argmax"
require_relative "skiftet_statistical/bandit/posteriors"
require_relative "skiftet_statistical/bandit/ranking"
require_relative "skiftet_statistical/bandit/summary"
require_relative "skiftet_statistical/bandit/probabilities"
require_relative "skiftet_statistical/bandit/pass"
require_relative "skiftet_statistical/bandit/weighted_posterior"
require_relative "skiftet_statistical/bandit/allocator"
require_relative "skiftet_statistical/bandit/win_hold"
require_relative "skiftet_statistical/cholesky"
require_relative "skiftet_statistical/linear_value"
require_relative "skiftet_statistical/hierarchical_rates"
require_relative "skiftet_statistical/shared_credit"
require_relative "skiftet_statistical/gauss_legendre"
require_relative "skiftet_statistical/beta_argmax"
require_relative "skiftet_statistical/additive_contrasts"
require_relative "skiftet_statistical/prequential"
require_relative "skiftet_statistical/cusum"

# A dependency-free statistics toolkit: reusable, app-independent statistical
# analysis code, so the same test, interval or draw is written once.
#
# Modules:
# - {Descriptive} — mean, variance, standard deviation, percentiles/median.
# - {Significance} — A/B significance testing (two-proportion z-test, Welch's
#   t-test, normal CDF / two-tailed p-values, normal quantile, chi-squared
#   survival, the sample-ratio check).
# - {Gamma} — the regularized incomplete gamma functions P(a, x) and Q(a, x).
# - {BetaFunction} — the regularized incomplete beta function and its quantile.
# - {Sampler} — Gamma/Beta/Gaussian random sampling (RNG-injectable).
# - {JointBeta} — joint Monte Carlo draws over Beta posteriors (P(best), loss).
# - {Bandit} + {Policies} — multi-armed bandit (Thompson Sampling, epsilon-greedy,
#   UCB1, Softmax) for online explore/exploit decisions, kept as 0.3.0 wrote it.
# - {Bandit::Posteriors} — the generic bandit: a set of items with their views
#   and yes/no or amount posteriors, ordered (`rank`) and picked (`pick`) by one
#   rule from a key, given uniforms or an RNG, with the odds of what it draws
#   (exact for a pick among up to EXACT_PICK_MAX contenders and for orders and
#   positions over up to EXACT_ORDER_MAX free items with none barred from
#   leading, otherwise a seeded Monte Carlo estimate with its standard error),
#   and per-item summaries; its priors
#   ({Bandit::Prior}), decay ({Bandit::Decay}) and keyed uniforms and draw by
#   weight ({Bandit::Draw}) beside it, with {Bandit::NormalArgmax} and
#   {Bandit::Pass}.
# - {Bandit::Allocator}, {Bandit::WinHold}, {Bandit::WeightedPosterior} — an A/B
#   test's Thompson-sampling allocation with a floor, a step limit, an automatic
#   win and a hold on it.
# - {LinearValue}, {Cholesky}, {SharedCredit}, {HierarchicalRates},
#   {AdditiveContrasts}, {BetaArgmax}, {GaussLegendre}, {Prequential}, {Cusum} —
#   values of several parts, pooled estimates, the exact Beta argmax, the
#   prequential score and a change test.
#
#   bandit = SkiftetStatistical.bandit(arms: %w[a b c])
#   choice = bandit.select          # which arm to play now
#   bandit.record(choice, 1)        # observed a reward (e.g. a conversion)
#   bandit.best_arm                 # current best by empirical mean
#
#   set = SkiftetStatistical::Bandit::Posteriors.yes_no({ "a" => { views: 40, outcomes: 8 },
#                                                       "b" => { views: 10, outcomes: 1 } })
#   set.rank(key: %w[list visit-1], shuffle: 0.1).items  # => the order that key draws, every time
module SkiftetStatistical
  class Error < StandardError; end

  # Convenience constructor for a multi-armed bandit.
  def self.bandit(...)
    Bandit.new(...)
  end
end
