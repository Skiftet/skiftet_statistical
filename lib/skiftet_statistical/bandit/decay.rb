# frozen_string_literal: true

require_relative "../numbers"
require_relative "../inputs"
require_relative "checks"
require_relative "tallies"

module SkiftetStatistical
  class Bandit
    # How much an old view weighs: one formula, 2 ** (-age / half_life), so a view one half-life old counts half and
    # a view older than `cap` counts nothing. Age, half-life and cap share one unit of the caller's choosing (hours,
    # days, passes). Pure: no randomness, no state.
    #
    #   SkiftetStatistical::Bandit::Decay.weight(72, half_life: 72, cap: 336)  # => 0.5
    #   SkiftetStatistical::Bandit::Decay.weight(337, half_life: 72, cap: 336) # => 0.0
    #
    # `fold` sums rows that each carry an `age` at that weight: every sum is weighted (views, outcomes and
    # outcome_squares alike, never views alone), so the folded tally is what a yes/no or an amounts posterior reads.
    #
    #   rows = [{ age: 0, views: 10, outcomes: 3 }, { age: 72, views: 20, outcomes: 4 }]
    #   SkiftetStatistical::Bandit::Decay.fold(rows, half_life: 72, cap: 336)
    #   # => { views: 20.0, outcomes: 5.0, outcome_squares: 0.0 }
    #
    # The fold is linear in the rows, so a caller that sums in a database groups its records by age bucket and folds
    # the buckets: the answer is the fold of the records one by one, up to rounding.
    #
    # Rules: every number is read by the library's one number reader (SkiftetStatistical::Numbers: a finite real
    # number of magnitude at most Numbers::LARGEST; a NaN, an infinity and a String, a numeric one included, are not
    # numbers). An age is >= 0 (a negative age is an ArgumentError); the half-life is > 0; the cap is nil (no cap) or
    # >= 0, and it is inclusive (an age at the cap still weighs by the formula, an age above it weighs 0.0). The rows
    # are a list, read by the bandit's one list reader (Tallies.list: any Enumerable but a Hash, read once by its each),
    # each row a tally read by its one tally reader (Tallies.read, Symbol or String keys); its age is required, and a
    # missing or nil views, outcomes or outcome_squares reads 0. Views and outcome_squares are >= 0; outcomes may be
    # below 0, a sum of amounts (the tally reader's one list of sums that may be, Tallies::SIGNED_SUMS). Anything else
    # is an ArgumentError, in a row past the cap too, and every answer passes the bandit's one check of an answer
    # (Checks.finite_answer).
    module Decay
      # The sums `fold` answers, each weighted.
      SUMS = %i[views outcomes outcome_squares].freeze
      # What a row holds: its age and its sums.
      ROW = %i[age views outcomes outcome_squares].freeze
      private_constant :ROW

      class << self
        # The weight of a view `age` old: 2 ** (-age / half_life), 0.0 when `cap` is given and age is above it.
        def weight(age, half_life:, cap: nil)
          half_life, cap = settings(half_life, cap)
          Checks.finite_answer([ factor(Checks.nonnegative(age, "an age"), half_life, cap) ],
                               "the age, half_life and cap").first
        end

        # { views:, outcomes:, outcome_squares: } summed over `rows`, each row's numbers at its age's weight.
        def fold(rows, half_life:, cap: nil)
          half_life, cap = settings(half_life, cap)
          read = Tallies.list(rows, "rows", "tallies").each_with_index.map do |row, index|
            Tallies.read(row, "rows[#{index}]", ROW, amounts: true)
          end
          weighted = read.each_with_index.map do |row, index|
            age = row[:age]
            raise ArgumentError, "rows[#{index}] needs an age" if nil.equal?(age)

            [ factor(age, half_life, cap), row ]
          end
          sums = SUMS.to_h { |sum| [ sum, weighted.sum(0.0) { |row_weight, row| row_weight * (row[sum] || 0.0) } ] }
          Checks.finite_answer(sums.values, "the rows", sums)
        end

        private

        # The one formula, on a checked age and checked settings.
        def factor(age, half_life, cap)
          return 0.0 if cap && age > cap

          2.0**(-age / half_life)
        end

        def settings(half_life, cap)
          half_life = Checks.positive(half_life, "half_life")
          return [ half_life, nil ] if nil.equal?(cap)

          [ half_life, Checks.nonnegative(cap, "cap") ]
        end
      end
    end
  end
end
