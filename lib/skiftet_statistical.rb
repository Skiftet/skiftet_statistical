# frozen_string_literal: true

require_relative "skiftet_statistical/version"
require_relative "skiftet_statistical/sampler"
require_relative "skiftet_statistical/descriptive"
require_relative "skiftet_statistical/significance"
require_relative "skiftet_statistical/arm"
require_relative "skiftet_statistical/policies/base"
require_relative "skiftet_statistical/policies/epsilon_greedy"
require_relative "skiftet_statistical/policies/thompson_sampling"
require_relative "skiftet_statistical/policies/ucb1"
require_relative "skiftet_statistical/policies/softmax"
require_relative "skiftet_statistical/bandit"

# Skiftet's shared statistics toolkit — a home for reusable, app-independent
# statistical analysis code across the workspace.
#
# Modules:
# - {Descriptive} — mean, variance, standard deviation, percentiles/median.
# - {Significance} — A/B significance testing (two-proportion z-test, Welch's
#   t-test, normal CDF / two-tailed p-values).
# - {Sampler} — Gamma/Beta/Gaussian random sampling (RNG-injectable).
# - {Bandit} + {Policies} — multi-armed bandit (Thompson Sampling, epsilon-greedy,
#   UCB1, Softmax) for online explore/exploit decisions.
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
