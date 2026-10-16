# frozen_string_literal: true

module SkiftetStatistical
  # Joint Monte Carlo draws from several independent Beta posteriors, one row per draw. Arms must be drawn
  # jointly (the same draw index across arms) or "P(arm is best)" and "expected loss" compare unrelated worlds.
  #
  #   posteriors = [ [ 41.0, 361.0 ], [ 52.0, 350.0 ] ]   # [alpha, beta] per arm
  #   SkiftetStatistical::JointBeta.each_draw(posteriors, draws: 3, rng: Random.new(7)).to_a
  #   # => [[0.1035..., 0.1491...], [0.1197..., 0.1343...], [0.1074..., 0.1065...]]
  #   SkiftetStatistical::JointBeta.summary(posteriors, draws: 2_000, rng: Random.new(7))
  #   # => { samples: [[2 000 Floats], [2 000 Floats]], wins: [219.0, 1781.0], losses: [57.83..., 2.58...] }
  #
  # THE DRAW ORDER IS THE CONTRACT: per draw, arm by arm in the order given, one Sampler#beta each. Sampler#gamma
  # takes a variable number of values from the RNG, so one arm's posterior moves every later draw: a seeded
  # caller gets the same rows only while this order holds, and never with an added, dropped or reordered draw.
  module JointBeta
    class << self
      # Yields one Array per draw (one Float per arm, in order). Without a block, an Enumerator of the rows.
      # Returns nil.
      def each_draw(posteriors, draws:, rng:)
        draws = count(draws)
        return enum_for(:each_draw, posteriors, draws: draws, rng: rng) { draws } unless block_given?

        sampler = Sampler.new(rng)
        draws.times { yield posteriors.map { |alpha, beta| sampler.beta(alpha, beta) } }
        nil
      end

      # Over the same rows as each_draw: every arm's draws (samples[arm][draw]), how many draws each arm won
      # (a tie split equally between the tied arms) and its summed loss (the draw's best minus the arm's value).
      # Divide wins and losses by draws for P(best) and expected loss.
      def summary(posteriors, draws:, rng:)
        draws = count(draws)
        size = posteriors.size
        samples = Array.new(size) { Array.new(draws) }
        wins = Array.new(size, 0.0)
        losses = Array.new(size, 0.0)

        each_draw(posteriors, draws: draws, rng: rng).with_index do |row, draw|
          best = row.max
          leaders = []
          row.each_with_index do |value, index|
            samples[index][draw] = value
            losses[index] += best - value
            leaders << index if value == best
          end
          share = 1.0 / leaders.size
          leaders.each { |index| wins[index] += share }
        end

        { samples: samples, wins: wins, losses: losses }
      end

      private

      def count(draws)
        raise ArgumentError, "draws must be an Integer >= 0, got #{draws.inspect}" unless draws.is_a?(Integer) && draws >= 0

        draws
      end
    end
  end
end
