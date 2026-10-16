# frozen_string_literal: true

require "skiftet_statistical"

RSpec.configure do |config|
  config.expect_with(:rspec) { |c| c.syntax = :expect }
  config.disable_monkey_patching!
  config.order = :random
  Kernel.srand config.seed

  # The bandit simulations (spec/simulations, tagged `simulation: true`) take minutes, so they run only with
  # SIMULATIONS=1: CI's simulations job runs them when the bandit code changes, nightly, and by hand.
  config.filter_run_excluding simulation: true unless ENV["SIMULATIONS"] == "1"
end
