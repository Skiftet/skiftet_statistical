# frozen_string_literal: true

require_relative "lib/skiftet_statistical/version"

Gem::Specification.new do |spec|
  spec.name = "skiftet_statistical"
  spec.version = SkiftetStatistical::VERSION
  spec.authors = [ "Skiftet" ]

  spec.summary = "A dependency-free statistics toolkit for Ruby: significance tests, Bayesian A/B allocation " \
                 "and a Thompson-sampling bandit that picks and orders items."
  spec.description = <<~DESC.strip
    Pure Ruby statistics for deciding under uncertainty. Descriptive statistics,
    significance tests and distribution functions; Beta and Normal posteriors with
    exact P(best), expected loss and credible intervals; an A/B allocator with
    floors, fixed shares, an automatic win and a hold on it; estimators for values
    of several parts, pooled contrasts, hierarchical rates and a change test; and
    a generic Thompson-sampling bandit: a set of items with their views and
    yes/no or amount outcomes, ranked or picked by one rule from a key, given
    uniforms or an RNG, with the odds of what it draws: exact for a pick among
    up to 20 contenders (EXACT_PICK_MAX) and for orders and positions over up to
    10 free items (EXACT_ORDER_MAX) with none barred from leading, otherwise a
    seeded Monte Carlo estimate with its standard error. Every random input is
    injected, so each answer can be reproduced.
  DESC
  spec.homepage = "https://github.com/Skiftet/skiftet_statistical"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.1"

  spec.metadata["allowed_push_host"] = "https://rubygems.org"
  spec.metadata["github_repo"] = "https://github.com/Skiftet/skiftet_statistical"
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["changelog_uri"] = "#{spec.homepage}/blob/main/CHANGELOG.md"
  spec.metadata["rubygems_mfa_required"] = "true"

  spec.files = Dir["lib/**/*.rb", "README.md", "CHANGELOG.md", "LICENSE.txt"]
  spec.require_paths = [ "lib" ]
end
