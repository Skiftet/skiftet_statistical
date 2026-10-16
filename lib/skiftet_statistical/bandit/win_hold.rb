# frozen_string_literal: true

module SkiftetStatistical
  class Bandit
    # The hold on an automatic win: the win rule must hold for the SAME arm at every allocation pass for
    # `hold_seconds` before the win counts. Pure: the caller gives the clock (Time values, or any values whose
    # difference is a number of seconds).
    #
    # The state is the leading arm, the time of the first pass in the current run of passes where that arm met the
    # rule (`leading_since`), and the time of the previous pass (`last_pass_at`):
    #
    #   hold = SkiftetStatistical::Bandit::WinHold.new(leading_arm: "a", leading_since: now - 90_000, now: now,
    #                                                  last_pass_at: now - 600)
    #   hold.satisfied?            # => true (the stored leader met the rule for 24 hours or more)
    #   hold.satisfied_for?(:a)    # => true
    #   hold.after_pass(:a)        # => { leading_arm: "a", leading_since: <unchanged> }
    #   hold.after_pass(:control)  # => { leading_arm: "control", leading_since: now }  (starts over)
    #   hold.after_pass(nil)       # => { leading_arm: nil, leading_since: nil }         (reset)
    #
    # "Every pass" is checked, not assumed: `max_gap_seconds` is the longest time between two passes that still
    # counts as an unbroken run. When `last_pass_at` is nil (no previous pass) or more than `max_gap_seconds` before
    # `now`, the hold is interrupted: it is not satisfied, and a leader starts over at this pass. `last_pass_at` is
    # required, so a call cannot leave it out by mistake and get a hold that never holds.
    #
    # A blank leading arm (nil, false, an empty or whitespace-only String, an empty Symbol) reads as none; any other
    # arm is kept as a String. This is ActiveSupport's `blank?`, in every encoding: a String that is not
    # ASCII-compatible (UTF-16, UTF-32) is matched with the pattern in its own encoding.
    class WinHold
      HOLD_SECONDS = 86_400
      MAX_GAP_SECONDS = 1_200
      BLANK = /\A[[:space:]]*\z/

      attr_reader :leading_arm, :leading_since, :now, :last_pass_at, :hold_seconds, :max_gap_seconds

      def self.blank?(value)
        case value
        when nil, false then true
        when String then value.empty? || blank_string?(value)
        else value.respond_to?(:empty?) ? (value.empty? ? true : false) : false
        end
      end

      def self.blank_string?(value)
        BLANK.match?(value)
      rescue Encoding::CompatibilityError
        Regexp.new(BLANK.source.encode(value.encoding), BLANK.options | Regexp::FIXEDENCODING).match?(value)
      end
      private_class_method :blank_string?

      def initialize(leading_arm:, leading_since:, now:, last_pass_at:, hold_seconds: HOLD_SECONDS,
                     max_gap_seconds: MAX_GAP_SECONDS)
        @leading_arm = self.class.blank?(leading_arm) ? nil : leading_arm.to_s
        @leading_since = leading_since
        @now = now
        @last_pass_at = last_pass_at
        @hold_seconds = hold_seconds
        @max_gap_seconds = max_gap_seconds
      end

      # The stored leader has met the rule at every pass since `hold_seconds` ago or longer.
      def satisfied?
        !leading_arm.nil? && !leading_since.nil? && !interrupted? && now - leading_since >= hold_seconds
      end

      # The run of passes since `leading_since` has a gap: no pass in the last `max_gap_seconds`.
      def interrupted?
        last_pass_at.nil? || now - last_pass_at > max_gap_seconds
      end

      # The hold is satisfied AND `arm` is the arm that held it.
      def satisfied_for?(arm)
        satisfied? && !arm.nil? && arm.to_s == leading_arm
      end

      # The state after a pass whose leader (the arm meeting the win rule this pass) is `leader`, or nil.
      def after_pass(leader)
        return { leading_arm: nil, leading_since: nil } if leader.nil?
        if leader.to_s == leading_arm && leading_since && !interrupted?
          return { leading_arm: leading_arm, leading_since: leading_since }
        end

        { leading_arm: leader.to_s, leading_since: now }
      end
    end
  end
end
