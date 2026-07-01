# frozen_string_literal: true

module SkiftetStatistical
  module Policies
    # A selection policy decides which arm to pull next from the current stats.
    # Subclasses implement `#choose(arms)`, returning the chosen Arm.
    class Base
      # Pick an arm. `arms` is a non-empty Array<Arm>; returns the chosen Arm.
      def choose(_arms)
        raise NotImplementedError, "#{self.class} must implement #choose"
      end

      # Serialisable config (so a Bandit can describe its policy).
      def to_h
        { type: self.class.name.split("::").last }
      end

      private

      def ensure_arms!(arms)
        raise Error, "no arms to choose from" if arms.nil? || arms.empty?
      end

      # Arms never pulled yet — explored first by the deterministic policies so
      # no arm is starved by an undefined/zero initial estimate.
      def unpulled(arms)
        arms.reject(&:pulled?)
      end

      # Given [[arm, score], ...] return the arm with the highest score, breaking
      # ties uniformly at random via the supplied rng.
      def pick_max(scored, rng)
        best_score = scored.map { |(_, s)| s }.max
        leaders = scored.select { |(_, s)| s == best_score }.map(&:first)
        leaders.length == 1 ? leaders.first : leaders[rng.rand(leaders.length)]
      end
    end
  end
end
