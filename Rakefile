# frozen_string_literal: true

require "bundler/gem_tasks" # build / install / release (the Release workflow runs `rake release`)
require "rspec/core/rake_task"
require "rubocop/rake_task"

RSpec::Core::RakeTask.new(:spec)
RuboCop::RakeTask.new

task default: %i[spec rubocop]
