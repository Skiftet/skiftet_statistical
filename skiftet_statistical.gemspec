# frozen_string_literal: true

require_relative "lib/skiftet_statistical/version"

Gem::Specification.new do |spec|
  spec.name = "skiftet_statistical"
  spec.version = SkiftetStatistical::VERSION
  spec.authors = [ "Skiftet" ]
  spec.email = [ "joel@skram.la" ]

  spec.summary = "Multi-armed bandit policies (Thompson Sampling, epsilon-greedy, UCB1, Softmax) for Ruby."
  spec.description = <<~DESC.strip
    A small, dependency-free toolkit for online decision-making under uncertainty:
    register arms, ask which to play, record rewards, and let a pluggable policy
    balance exploration and exploitation. Ships Thompson Sampling, epsilon-greedy,
    UCB1 and Softmax; state serialises to a plain Hash for persistence and every
    policy is deterministic under an injected RNG for testing.
  DESC
  spec.homepage = "https://github.com/Skiftet/skiftet_statistical"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.1"

  spec.metadata["allowed_push_host"] = "https://rubygems.org"
  spec.metadata["github_repo"] = "https://github.com/Skiftet/skiftet_statistical"
  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["changelog_uri"] = "#{spec.homepage}/blob/main/CHANGELOG.md"
  spec.metadata["rubygems_mfa_required"] = "true"

  spec.files = Dir["lib/**/*.rb", "README.md", "CHANGELOG.md", "LICENSE.txt"]
  spec.require_paths = [ "lib" ]
end
