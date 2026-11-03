# frozen_string_literal: true

require_relative "../numbers"
require_relative "../inputs"
require_relative "checks"

module SkiftetStatistical
  class Bandit
    # One allocation pass with a hold on its win: probe, then crown on the same draws only when the hold allows. The
    # caller's block is the allocation (Allocator.call or Allocator.from_summary, or any rule answering a Hash); the
    # pass decides how often it runs and with which flag, and hands it its randomness.
    #
    #   hold = SkiftetStatistical::Bandit::WinHold.new(leading_arm: "a", leading_since: now - 90_000, now: now,
    #                                                  last_pass_at: now - 600)
    #   SkiftetStatistical::Bandit::Pass.call(status: "running", hold: hold) do |hold_satisfied, rng|
    #     SkiftetStatistical::Bandit::Allocator.call(arms: %i[control a], tallies: tallies, status: "running",
    #                                                winner_arm: nil, previous_weights: weights, rng: rng,
    #                                                draws: 20_000, hold_satisfied: hold_satisfied)
    #   end
    #   # => the crown's answer (weights, phase: :won, stats, winner: :a, result_changed, leader: :a) and ready: true
    #
    # The rule:
    # - the probe: the block is yielded (false, Random.new(seed)), so a leader gets the step, never the won split;
    # - ready: the status reads "running" (a String by its bytes, anything else by its to_s, as the Allocator reads
    #   it, once, before the first yield: Inputs.text, never a String's own ==), the
    #   probe's Hash names a leader under :leader (nil for none; read by the bandit's one lookup in a caller's Hash,
    #   Checks.lookup, so a default the Hash holds is not read) and `hold.satisfied_for?(leader)` answers true; the
    #   hold is asked only then, once, about that leader;
    # - the crown: only when `crown` is true and the pass is ready, the block is yielded again (true, a fresh
    #   Random.new(seed)), so the probe and the crown read the same draws and only the decision changes;
    # - the answer: the last yield's Hash merged with `ready:` (true or false) by Hash's own merge, nothing else. A paused or won status
    #   takes the block's own path at the probe and is never ready. The caller computes its next hold itself (for a
    #   WinHold, `after_pass` with the answer's leader).
    #
    # With `crown: false` a ready leader keeps the probe's answer (its step toward its P(best)), and `ready:` says it
    # would have been crowned, so the caller can crown it by its own hand.
    #
    # `seed` defaults to Random.new_seed, drawn once a pass; a given seed is honoured, so a pass is repeatable. The
    # checks (ArgumentError, before the first yield): a block is given, a status that is no String answers to_s
    # without raising (the library's one call into a caller's object, Inputs.asked), the hold answers
    # `satisfied_for?` and the seed is an
    # Integer within a Float's range (the bandit's one reading of a whole number, Checks.integer, as `top` and the
    # odds' `draws` and `seed` are read); each yield must answer a Hash (ArgumentError otherwise), checked when it is
    # answered. The block and the hold's satisfied_for? are the caller's own rules: what they raise reaches the caller
    # as raised.
    module Pass
      class << self
        # The probe, then the crown when `crown` and ready: the last yield's Hash plus { ready: true | false }.
        def call(status:, hold:, crown: true, seed: Random.new_seed, &allocation)
          running = check(status, hold, seed, allocation)
          probe = run(allocation, false, seed)
          ready = ready?(running, Checks.lookup(probe, :leader, "the block's Hash"), hold)
          answer = crown && ready ? run(allocation, true, seed) : probe
          Inputs.merged(answer, { ready: ready })
        end

        private

        # Ready: a running status, a probe leader, and the hold satisfied for that leader (asked only then).
        def ready?(running, leader, hold)
          return false unless running && !nil.equal?(leader)

          hold.satisfied_for?(leader) ? true : false
        end

        # One yield, with a fresh Random.new(seed): the block's Hash.
        def run(allocation, hold_satisfied, seed)
          answer = allocation.call(hold_satisfied, Random.new(seed))
          return answer if Hash === answer

          raise ArgumentError, "a pass's block must answer a Hash (got #{Inputs.class_name(answer)})"
        end

        # The checks, before the first yield; answers whether the status reads "running": the library's one reading of
        # a value as text (Inputs.text: a String status read by its bytes and encoding alone, by Inputs.string; any
        # other by its to_s, asked by the library's one call into a caller's object, so a to_s that raises is an
        # ArgumentError, its String answer read the same way), whose copy is the String "running" (so a String's own
        # == or any other of its methods is never called).
        def check(status, hold, seed, allocation)
          raise ArgumentError, "a pass needs a block: the allocation it runs" if allocation.nil?
          unless Inputs.responds?(status, :to_s, "a status")
            raise ArgumentError, "a status must answer to_s (got #{Inputs.shown(status)})"
          end
          text = Inputs.text(status, "a status")
          unless Inputs.responds?(hold, :satisfied_for?, "a hold")
            raise ArgumentError, "a hold must answer satisfied_for? (got #{Inputs.class_name(hold)})"
          end
          Checks.integer(seed, "a seed")
          !nil.equal?(text) && text == "running"
        end
      end
    end
  end
end
