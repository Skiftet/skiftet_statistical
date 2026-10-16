# frozen_string_literal: true

module SkiftetStatistical
  class Bandit
    # A Thompson-sampling allocator's decision for one experiment: how much traffic each arm gets next, and
    # whether an arm has won. Pure: no clock, no state; the randomness comes only from the injected `rng`, so the
    # same input and the same seed give the same output.
    #
    #   SkiftetStatistical::Bandit::Allocator.call(
    #     arms: %i[control a],
    #     tallies: { control: { exposures: 1_000, conversions: 500 }, a: { exposures: 1_000, conversions: 533 } },
    #     status: "running", winner_arm: nil, previous_weights: { control: 0.5, a: 0.5 },
    #     rng: Random.new(7), draws: 20_000)
    #   # => { weights: { control: 0.35, a: 0.65 }, phase: :shifting,
    #   #      stats: { control: { ... }, a: { ... } }, winner: nil, result_changed: false, leader: nil }
    #
    # The rule, with the settings of Config:
    # - paused: the previous weights unchanged (equal weights without them);
    # - won (status "won" and `winner_arm` one of the arms): the won split, `won_share` to the winner and
    #   `won_rest` shared by the others, and `result_changed` when the winner's P(best) has fallen below
    #   `result_changed_below`;
    # - warm-up, while any arm has fewer than `warmup_exposures`: equal weights;
    # - the automatic win: the first arm (in order) with P(best) >= `win_p_best` and expected loss <= `win_loss`
    #   is the `leader`; it gets the won split when `hold_satisfied:` is true (the default). With false the
    #   leader gets the step below and the caller keeps its hold (WinHold);
    # - otherwise, shifting: the target is each arm's P(best), every arm at least `floor` (the others scaled
    #   down in proportion), reached by a convex step from the previous weights in which no arm moves more than
    #   `max_step`.
    #
    # The statistics (`stats`, per arm): exposures, conversions, the raw rate (nil at no exposures), the weighted
    # rate, the `credible_mass` interval (lower, upper) of the arm's draws, P(best) (a tie split equally) and the
    # expected loss (the mean of the draw's best minus the arm). The posteriors are WeightedPosterior's; the
    # draws are JointBeta's (`draws` joint draws, arm by arm in order). A tally MAY carry `guardrails:
    # { goal => conversions }` (or `{ goal => { conversions: n } }`): each goes into the stats with its raw rate,
    # for display only.
    #
    # Arms are Symbols in the result (Symbols or Strings in); tallies and previous weights may use either. Previous
    # weights count only when they name exactly the current arms, each with a finite number: anything else reads
    # as none. A share below the floor (a won split read back under "running") is lifted onto it before the step.
    #
    # THE DRAW ORDER IS THE CONTRACT (JointBeta): a seeded caller gets the same result only while no draw is added,
    # dropped or reordered.
    class Allocator
      # Asked to allocate a fixed split: its weights are its own, never the allocator's to move or crown.
      class FixedAllocation < ArgumentError; end

      # The allocator's settings, each a finite number. The defaults are the values f1's A/B tests run with.
      class Config
        DEFAULTS = {
          warmup_exposures: 300, floor: 0.10, max_step: 0.15, win_p_best: 0.95, win_loss: 0.0025, won_share: 0.95,
          won_rest: 0.05, result_changed_below: 0.5, credible_mass: 0.95, prior_alpha: 1.0, prior_beta: 1.0
        }.freeze

        attr_reader(*DEFAULTS.keys)

        def initialize(**settings)
          unknown = settings.keys - DEFAULTS.keys
          raise ArgumentError, "unknown allocator setting(s): #{unknown.join(', ')}" unless unknown.empty?

          DEFAULTS.merge(settings).each do |name, value|
            unless value.is_a?(Numeric) && value.finite?
              raise ArgumentError, "#{name} must be a finite number, got #{value.inspect}"
            end

            instance_variable_set(:"@#{name}", value)
          end
          raise ArgumentError, "floor must be above 0 and at most 1, got #{floor.inspect}" unless floor.positive? && floor <= 1

          freeze
        end

        # More arms than this cannot all hold the floor.
        def max_arms
          (1 / floor).round
        end

        def to_h
          DEFAULTS.keys.to_h { |name| [ name, public_send(name) ] }
        end

        def ==(other)
          other.is_a?(Config) && to_h == other.to_h
        end
        alias_method :eql?, :==

        def hash
          to_h.hash
        end
      end

      # The won split: the winner `won_share`, the other arms `won_rest` shared equally; a single arm gets
      # everything. The keys are `arms` as given (Symbols or Strings).
      def self.won_weights(arms, winner, config: Config.new)
        return { winner => 1.0 } if arms.size == 1

        rest = config.won_rest / (arms.size - 1)
        arms.to_h { |arm| [ arm, arm == winner ? config.won_share : rest ] }
      end

      def self.call(arms:, tallies:, status:, winner_arm:, rng:, previous_weights: nil, draws:, hold_satisfied: true,
                    allocation: :adaptive, config: Config.new)
        new(arms: arms, tallies: tallies, status: status, winner_arm: winner_arm, rng: rng,
            previous_weights: previous_weights, draws: draws, hold_satisfied: hold_satisfied, allocation: allocation,
            config: config).call
      end

      def initialize(arms:, tallies:, status:, winner_arm:, rng:, draws:, previous_weights: nil, hold_satisfied: true,
                     allocation: :adaptive, config: Config.new)
        if allocation.to_s == "fixed"
          raise FixedAllocation, "a fixed split's weights are its phase's: the allocator never moves or crowns them"
        end
        raise ArgumentError, "config must be a #{Config}, got #{config.inspect}" unless config.is_a?(Config)

        @config = config
        @arms = arms.map(&:to_sym)
        raise ArgumentError, "an experiment needs at least one arm" if @arms.empty?
        raise ArgumentError, "at most #{config.max_arms} arms keep every arm at the floor" if @arms.size > config.max_arms

        @counts = @arms.map { |arm| read_tally(tallies, arm) }
        @status = status.to_s
        @winner_arm = winner_arm&.to_sym
        @previous = read_weights(previous_weights)
        @rng = rng
        @draws = Integer(draws)
        raise ArgumentError, "draws must be positive" unless @draws.positive?

        @hold_satisfied = hold_satisfied ? true : false
      end

      def call
        stats = build_stats

        if @status == "paused"
          result(@previous || equal_weights, :paused, stats)
        elsif @status == "won" && @arms.include?(@winner_arm)
          changed = stats.fetch(@winner_arm)[:p_best] < @config.result_changed_below
          result(won_weights(@winner_arm), :won, stats, winner: @winner_arm, result_changed: changed)
        elsif @counts.any? { |count| count[:exposures] < @config.warmup_exposures }
          result(equal_weights, :warmup, stats)
        elsif (leader = automatic_winner(stats)) && @hold_satisfied
          result(won_weights(leader), :won, stats, winner: leader, leader: leader)
        else
          target = with_floor(@arms.to_h { |arm| [ arm, stats.fetch(arm)[:p_best] ] })
          result(step_toward(target), :shifting, stats, leader: leader)
        end
      end

      private

      def result(weights, phase, stats, winner: nil, result_changed: false, leader: nil)
        { weights: weights, phase: phase, stats: stats, winner: winner, result_changed: result_changed, leader: leader }
      end

      # The tally's numbers as WeightedPosterior reads them, plus the guardrails (display only).
      def read_tally(tallies, arm)
        tally = tallies[arm] || tallies[arm.to_s] || {}
        WeightedPosterior.count(tally).merge(guardrails: fetch_key(tally, :guardrails) || {})
      end

      def fetch_key(hash, key)
        hash.key?(key) ? hash[key] : hash[key.to_s]
      end

      # Previous weights count only when they are a Hash that names exactly the current arms, each with a finite
      # number; a value that is not a number (a String, a Hash, nil) is ignored, so a corrupt row gives none. Zero
      # and negative shares are kept: #step_toward lifts a share below the floor onto it first.
      def read_weights(weights)
        return nil unless weights.is_a?(Hash)

        normalized = weights.each_with_object({}) do |(arm, share), out|
          out[arm.to_sym] = share.to_f if share.is_a?(Numeric) && share.to_f.finite?
        end
        return nil unless normalized.keys.sort == @arms.sort

        @arms.to_h { |arm| [ arm, normalized.fetch(arm) ] }
      end

      def equal_weights
        share = 1.0 / @arms.size
        @arms.to_h { |arm| [ arm, share ] }
      end

      def won_weights(winner)
        self.class.won_weights(@arms, winner, config: @config)
      end

      # The arm that is both very likely best and cheap to be wrong about.
      def automatic_winner(stats)
        return nil if @arms.size < 2

        @arms.find do |arm|
          stats.fetch(arm)[:p_best] >= @config.win_p_best && stats.fetch(arm)[:expected_loss] <= @config.win_loss
        end
      end

      # Every arm at least the floor, the others scaled down in proportion.
      def with_floor(target)
        floor = @config.floor
        fixed = []
        loop do
          free = @arms - fixed
          mass = 1.0 - (fixed.size * floor)
          free_total = free.sum { |arm| target.fetch(arm) }
          scaled = free.to_h do |arm|
            [ arm, free_total.positive? ? mass * target.fetch(arm) / free_total : mass / free.size ]
          end
          below = scaled.select { |_arm, share| share < floor }.keys
          return @arms.to_h { |arm| [ arm, fixed.include?(arm) ? floor : scaled.fetch(arm) ] } if below.empty?

          fixed.concat(below)
        end
      end

      # A convex step from the previous weights, no arm moving more than max_step. A previous share below the
      # floor is lifted onto it first.
      def step_toward(target)
        previous = @previous || equal_weights
        previous = with_floor(previous) if previous.values.any? { |share| share < @config.floor }
        largest = @arms.map { |arm| (target.fetch(arm) - previous.fetch(arm)).abs }.max
        lambda = largest > @config.max_step ? @config.max_step / largest : 1.0

        @arms.to_h { |arm| [ arm, previous.fetch(arm) + (lambda * (target.fetch(arm) - previous.fetch(arm))) ] }
      end

      def build_stats
        posteriors = @counts.map do |count|
          WeightedPosterior.beta(count, prior_alpha: @config.prior_alpha, prior_beta: @config.prior_beta)
        end
        samples, wins, losses = JointBeta.summary(posteriors, draws: @draws, rng: @rng).values_at(:samples, :wins, :losses)
        tail = (1.0 - @config.credible_mass) / 2.0

        @arms.each_with_index.to_h do |arm, index|
          count = @counts[index]
          sorted = samples[index].sort
          [ arm, {
            exposures: count[:exposures],
            conversions: count[:conversions],
            rate: rate(count[:conversions], count[:exposures]),
            weighted_rate: weighted_rate(count),
            lower: quantile(sorted, tail),
            upper: quantile(sorted, 1.0 - tail),
            p_best: wins[index] / @draws,
            expected_loss: losses[index] / @draws,
            guardrails: guardrails(count)
          } ]
        end
      end

      def quantile(sorted, probability)
        position = probability * (sorted.size - 1)
        low = sorted[position.floor]
        high = sorted[position.ceil]
        low + ((high - low) * (position - position.floor))
      end

      def rate(conversions, exposures)
        exposures.positive? ? conversions.fdiv(exposures) : nil
      end

      def weighted_rate(count)
        return rate(count[:conversions], count[:exposures]) unless count[:weighted_exposures]

        count[:weighted_conversions] / count[:weighted_exposures]
      end

      def guardrails(count)
        count[:guardrails].to_h do |goal, value|
          conversions = Integer((value.is_a?(Hash) ? fetch_key(value, :conversions) : value) || 0)
          [ goal.to_s, { conversions: conversions, rate: rate(conversions, count[:exposures]) } ]
        end
      end
    end
  end
end
