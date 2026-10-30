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
    # PER-ARM SHARES (`fixed_shares:` and `arm_floors:`, both nil by default). With both nil (or naming no arm)
    # nothing here is checked and every output is the rule above, bit for bit. Otherwise:
    # - `arm_floors: { arm => floor }` gives an arm its own floor, a finite number in (0, 1], in place of
    #   Config#floor (an arm it does not name keeps Config#floor); `fixed_shares: { arm => share }` holds an arm at
    #   exactly its share, a finite number in (0, 1), and gives it no floor. Keys are arms of the call (Symbols or
    #   Strings); no arm in both; the fixed shares and the other arms' floors sum to at most 1. Every arm fixed is a
    #   fixed split (FixedAllocation). Each refusal is an ArgumentError raised before any draw.
    # - In EVERY phase each fixed arm gets exactly its share and the free arms (the others, in the call's order)
    #   share the rest, 1 - F (F the sum of the fixed shares), each at least its floor.
    # - The lift, one rule for every phase: a target over the free arms is scaled into the free mass; every arm
    #   under its floor is pinned there and the others are scaled again into what is left, until none is under.
    #   With no fixed share and every floor Config#floor it is the shifting phase's floor above.
    # - The free arms' previous shares hold when each is at least its floor and together they sum to 1 - F (to
    #   1e-12; 1 with no fixed share), and each fixed arm's previous share is its own (to 1e-12).
    # - paused: the previous weights while they hold, else the free arms' previous shares (equal ones without them)
    #   through the lift; won and the automatic win: the won split over the free arms (a single free arm 1.0)
    #   through the lift (a fixed arm is never the won arm: a stored win of one is refused); warm-up: equal targets
    #   over the free arms through the lift; shifting: the free arms' P(best) through the lift, reached by the step
    #   over the free arms (from previous shares lifted first when they break a floor or do not sum to 1 - F).
    # - The automatic win's leader is the first free arm by the rule above. A lone free arm (beside fixed arms, or
    #   the call's one arm) is read by the same rule: its P(best) among the free arms is 1 and its expected loss 0,
    #   so past warm-up it is the leader. Undeclared, 0.2.0's guard stays: a single arm is never the leader.
    # - With fixed shares, P(best) and expected loss are taken among the free arms in each joint draw (every arm is
    #   still drawn, in order), so a fixed arm's are nil and its other statistics are as without fixed shares.
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

      # The arms' declared shares: which arms are fixed and at what share, the free arms' floors, and the one lift
      # every phase goes through. Built (and checked) before any draw. Undeclared (both keywords nil or naming no
      # arm), every arm is free at Config#floor and each phase answers exactly as without per-arm shares.
      class Shares
        # How far a fixed arm's previous share, or the free arms' previous sum, may sit from what is declared.
        TOLERANCE = 1e-12

        attr_reader :free, :fixed

        def initialize(arms, fixed_shares:, arm_floors:, config:)
          @arms = arms
          @config = config
          fixed = read(fixed_shares, :fixed_shares, "a finite number in (0, 1)") { |share| share.positive? && share < 1 }
          floors = read(arm_floors, :arm_floors, "a finite number in (0, 1]") { |floor| floor.positive? && floor <= 1 }
          @declared = !(fixed.nil? && floors.nil?)
          fixed ||= {}
          floors ||= {}
          check(fixed, floors)

          @fixed = arms.select { |arm| fixed.key?(arm) }.to_h { |arm| [ arm, fixed.fetch(arm) ] }.freeze
          @free = (arms - @fixed.keys).freeze
          @floors = @free.to_h { |arm| [ arm, floors.fetch(arm, config.floor) ] }.freeze
          check_sum if @declared
          @free_mass = 1.0 - @fixed.values.sum
          freeze
        end

        def declared?
          @declared
        end

        def fixed?(arm)
          @fixed.key?(arm)
        end

        # Equal weights over every arm (the step's start and a pause's weights when there are no previous ones).
        def equal
          share = 1.0 / @arms.size
          @arms.to_h { |arm| [ arm, share ] }
        end

        # Undeclared: the previous weights (equal ones without them) unchanged. Declared: the same while they hold
        # (every floor and fixed share, the free arms summing to 1 - F), else the free arms' previous shares (equal
        # ones without them) through the lift, as warm-up's split is.
        def paused(previous)
          weights = previous || equal
          return weights unless @declared && !holds?(weights)

          lift(previous || equal_free)
        end

        # Equal weights; declared, equal targets over the free arms through the lift.
        def warmup
          @declared ? lift(equal_free) : equal
        end

        # The won split; declared, the won split over the free arms (a single free arm 1.0) through the lift.
        def won(winner)
          return Allocator.won_weights(@arms, winner, config: @config) unless @declared

          lift(Allocator.won_weights(@free, winner, config: @config))
        end

        # The free arms' previous shares the step starts from: lifted first when one is under its floor or,
        # declared, when they do not sum to 1 - F.
        def step_start(previous)
          breaks_free_shares?(previous) ? lift(previous) : previous
        end

        # The lift (0.2.0's floor rule over the free arms' mass): the target's free arms scaled into the free mass;
        # every arm under its floor is pinned there and the unpinned arms are scaled again into what the pinned
        # floors leave, until none is under. Each fixed arm takes its share. With no fixed share and every floor
        # Config#floor this is 0.2.0's arithmetic step for step: the pinned floors are summed per distinct floor
        # as count * floor, and a target summing to 0 or less shares the mass equally.
        def lift(target)
          pinned = []
          loop do
            unpinned = @free - pinned
            mass = @free_mass - pinned_mass(pinned)
            total = unpinned.sum { |arm| target.fetch(arm) }
            scaled = unpinned.to_h do |arm|
              [ arm, total.positive? ? mass * target.fetch(arm) / total : mass / unpinned.size ]
            end
            below = scaled.select { |arm, share| share < @floors.fetch(arm) }.keys
            if below.empty?
              return @arms.to_h do |arm|
                [ arm, @fixed.fetch(arm) { pinned.include?(arm) ? @floors.fetch(arm) : scaled.fetch(arm) } ]
              end
            end

            pinned.concat(below)
          end
        end

        private

        # nil for nil or an empty Hash (naming no arm); else the arms it names (Symbols) and their Float values.
        def read(values, name, requirement)
          return nil if values.nil? || (values.is_a?(Hash) && values.empty?)
          raise ArgumentError, "#{name} must be a Hash of arm => number, got #{values.inspect}" unless values.is_a?(Hash)

          values.each_with_object({}) do |(key, value), out|
            unless (key.is_a?(Symbol) || key.is_a?(String)) && @arms.include?(key.to_sym)
              raise ArgumentError, "#{name}: #{key.inspect} is not an arm of the call"
            end

            arm = key.to_sym
            raise ArgumentError, "#{name} names #{arm} twice" if out.key?(arm)
            unless value.is_a?(Numeric) && value.real? && value.finite? && yield(value)
              raise ArgumentError, "#{name}: #{arm} must be #{requirement}, got #{value.inspect}"
            end

            out[arm] = value.to_f
          end
        end

        def check(fixed, floors)
          both = fixed.keys & floors.keys
          raise ArgumentError, "an arm is fixed or floored, never both: #{both.join(', ')}" unless both.empty?
          return unless fixed.size == @arms.size

          raise FixedAllocation, "every arm has a fixed share: a fixed split's weights are its own, never the allocator's"
        end

        def check_sum
          total = @fixed.values.sum + @floors.values.sum
          return if total <= 1 + TOLERANCE

          raise ArgumentError, "the fixed shares and the free arms' floors sum to #{total}, above 1"
        end

        def equal_free
          @free.to_h { |arm| [ arm, 1.0 ] }
        end

        # Every floor and fixed share held, the free arms summing to 1 - F (the sum and each fixed arm to TOLERANCE).
        def holds?(weights)
          !breaks_free_shares?(weights) && @fixed.all? { |arm, share| (weights.fetch(arm) - share).abs <= TOLERANCE }
        end

        # A free arm under its floor or, declared, free arms that do not sum to 1 - F (1 with no fixed share).
        # Undeclared, only the floor is read, as 0.2.0's step reads it.
        def breaks_free_shares?(weights)
          return true if @free.any? { |arm| weights.fetch(arm) < @floors.fetch(arm) }
          return false unless @declared

          (@free.sum { |arm| weights.fetch(arm) } - @free_mass).abs > TOLERANCE
        end

        def pinned_mass(pinned)
          pinned.group_by { |arm| @floors.fetch(arm) }.sum { |floor, group| group.size * floor }
        end
      end
      private_constant :Shares

      # The won split: the winner `won_share`, the other arms `won_rest` shared equally; a single arm gets
      # everything. The keys are `arms` as given (Symbols or Strings).
      def self.won_weights(arms, winner, config: Config.new)
        return { winner => 1.0 } if arms.size == 1

        rest = config.won_rest / (arms.size - 1)
        arms.to_h { |arm| [ arm, arm == winner ? config.won_share : rest ] }
      end

      def self.call(arms:, tallies:, status:, winner_arm:, rng:, previous_weights: nil, draws:, hold_satisfied: true,
                    allocation: :adaptive, config: Config.new, fixed_shares: nil, arm_floors: nil)
        new(arms: arms, tallies: tallies, status: status, winner_arm: winner_arm, rng: rng,
            previous_weights: previous_weights, draws: draws, hold_satisfied: hold_satisfied, allocation: allocation,
            config: config, fixed_shares: fixed_shares, arm_floors: arm_floors).call
      end

      def initialize(arms:, tallies:, status:, winner_arm:, rng:, draws:, previous_weights: nil, hold_satisfied: true,
                     allocation: :adaptive, config: Config.new, fixed_shares: nil, arm_floors: nil)
        if allocation.to_s == "fixed"
          raise FixedAllocation, "a fixed split's weights are its phase's: the allocator never moves or crowns them"
        end
        raise ArgumentError, "config must be a #{Config}, got #{config.inspect}" unless config.is_a?(Config)

        @config = config
        @arms = arms.map(&:to_sym)
        raise ArgumentError, "an experiment needs at least one arm" if @arms.empty?
        raise ArgumentError, "at most #{config.max_arms} arms keep every arm at the floor" if @arms.size > config.max_arms

        @shares = Shares.new(@arms, fixed_shares: fixed_shares, arm_floors: arm_floors, config: config)
        @counts = @arms.map { |arm| read_tally(tallies, arm) }
        @status = status.to_s
        @winner_arm = winner_arm&.to_sym
        if @status == "won" && @shares.fixed?(@winner_arm)
          raise ArgumentError, "a fixed arm is never the won arm, got #{@winner_arm.inspect}"
        end

        @previous = read_weights(previous_weights)
        @rng = rng
        @draws = Integer(draws)
        raise ArgumentError, "draws must be positive" unless @draws.positive?

        @hold_satisfied = hold_satisfied ? true : false
      end

      def call
        decide(build_stats)
      end

      private

      # The phase rule on the arms' statistics: it reads each free arm's P(best) and expected loss, and the arms'
      # exposures for warm-up.
      def decide(stats)
        if @status == "paused"
          result(@shares.paused(@previous), :paused, stats)
        elsif @status == "won" && @arms.include?(@winner_arm)
          changed = stats.fetch(@winner_arm)[:p_best] < @config.result_changed_below
          result(@shares.won(@winner_arm), :won, stats, winner: @winner_arm, result_changed: changed)
        elsif @counts.any? { |count| count[:exposures] < @config.warmup_exposures }
          result(@shares.warmup, :warmup, stats)
        elsif (leader = automatic_winner(stats)) && @hold_satisfied
          result(@shares.won(leader), :won, stats, winner: leader, leader: leader)
        else
          target = @shares.lift(@shares.free.to_h { |arm| [ arm, stats.fetch(arm)[:p_best] ] })
          result(step_toward(target), :shifting, stats, leader: leader)
        end
      end

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

      # The first free arm that is both very likely best and cheap to be wrong about. Undeclared, 0.2.0's guard: a
      # single arm is compared with nothing and is never the leader. Declared, a lone free arm (beside fixed arms, or
      # the call's one arm) is read by the same rule as any free arm: its P(best) among the free arms is 1 and its
      # expected loss 0, so past warm-up it is the leader.
      def automatic_winner(stats)
        return nil if !@shares.declared? && @arms.size < 2

        @shares.free.find do |arm|
          stats.fetch(arm)[:p_best] >= @config.win_p_best && stats.fetch(arm)[:expected_loss] <= @config.win_loss
        end
      end

      # A convex step over the free arms from their previous shares, no free arm moving more than max_step; each
      # fixed arm takes its share. Previous shares that break a floor are lifted first (Shares#step_start).
      def step_toward(target)
        previous = @shares.step_start(@previous || @shares.equal)
        free = @shares.free
        largest = free.map { |arm| (target.fetch(arm) - previous.fetch(arm)).abs }.max
        lambda = largest > @config.max_step ? @config.max_step / largest : 1.0

        @arms.to_h do |arm|
          [ arm, @shares.fixed.fetch(arm) { previous.fetch(arm) + (lambda * (target.fetch(arm) - previous.fetch(arm))) } ]
        end
      end

      def build_stats
        posteriors = @counts.map do |count|
          WeightedPosterior.beta(count, prior_alpha: @config.prior_alpha, prior_beta: @config.prior_beta)
        end
        samples, wins, losses = joint_summary(posteriors).values_at(:samples, :wins, :losses)
        tail = (1.0 - @config.credible_mass) / 2.0

        @arms.each_with_index.to_h do |arm, index|
          count = @counts[index]
          sorted = samples[index].sort
          fixed = @shares.fixed?(arm)
          [ arm, {
            exposures: count[:exposures],
            conversions: count[:conversions],
            rate: rate(count[:conversions], count[:exposures]),
            weighted_rate: weighted_rate(count),
            lower: quantile(sorted, tail),
            upper: quantile(sorted, 1.0 - tail),
            p_best: fixed ? nil : wins[index] / @draws,
            expected_loss: fixed ? nil : losses[index] / @draws,
            guardrails: guardrails(count)
          } ]
        end
      end

      # JointBeta's summary of every arm's joint draws. With fixed shares the same rows (every arm drawn, in
      # order), with each draw's best taken among the free arms: a tie split equally, each free arm's loss the free
      # best less its own; a fixed arm wins and loses nothing.
      def joint_summary(posteriors)
        return JointBeta.summary(posteriors, draws: @draws, rng: @rng) if @shares.fixed.empty?

        free = @shares.free.map { |arm| @arms.index(arm) }
        samples = Array.new(posteriors.size) { Array.new(@draws) }
        wins = Array.new(posteriors.size, 0.0)
        losses = Array.new(posteriors.size, 0.0)

        JointBeta.each_draw(posteriors, draws: @draws, rng: @rng).with_index do |row, draw|
          row.each_with_index { |value, index| samples[index][draw] = value }
          best = free.map { |index| row[index] }.max
          leaders = free.select { |index| row[index] == best }
          free.each { |index| losses[index] += best - row[index] }
          share = 1.0 / leaders.size
          leaders.each { |index| wins[index] += share }
        end

        { samples: samples, wins: wins, losses: losses }
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
