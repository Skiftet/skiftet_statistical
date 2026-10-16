# frozen_string_literal: true

require_relative "skiftet_statistical/version"
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
require_relative "skiftet_statistical/bandit/weighted_posterior"
require_relative "skiftet_statistical/bandit/allocator"
require_relative "skiftet_statistical/bandit/win_hold"

# Skiftet's shared statistics toolkit — a home for reusable, app-independent
# statistical analysis code across the workspace.
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
#   UCB1, Softmax) for online explore/exploit decisions.
# - {Bandit::Allocator}, {Bandit::WinHold}, {Bandit::WeightedPosterior} — an A/B
#   test's Thompson-sampling allocation with a floor, a step limit, an automatic
#   win and a hold on it.
#
#   bandit = SkiftetStatistical.bandit(arms: %w[facebook whatsapp x])
#   choice = bandit.select          # which arm to play now
#   bandit.record(choice, 1)        # observed a reward (e.g. a conversion)
#   bandit.best_arm                 # current best by empirical mean
module SkiftetStatistical
  class Error < StandardError; end

  # Convenience constructor for a multi-armed bandit.
  def self.bandit(...)
    Bandit.new(...)
  end
end
