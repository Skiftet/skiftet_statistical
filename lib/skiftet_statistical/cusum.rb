# frozen_string_literal: true

module SkiftetStatistical
  # A one-sided change test (Page 1954): has a stream of standardised values z_1, z_2, ... moved up? The statistic
  # gathers each value's excess over a reference k and never falls below 0:
  #
  #   S_0 = 0, S_t = max(0, S_(t-1) + z_t - k), an alarm at the first t with S_t > h
  #
  # Steps count from 1, so the first alarm's step is the run length. Nothing resets after an alarm.
  #
  # How long a run lasts is Siegmund's approximation. For values of mean shift, a drift theta = shift - k and a boundary
  # b = h + 1.166:
  #
  #   average run length = (exp(-2 * theta * b) + 2 * theta * b - 1) / (2 * theta**2), b**2 at theta = 0
  #
  # threshold turns a stated false-alarm rate into h by that approximation, one alarm in false_alarm_steps while
  # nothing moves (shift 0). minimum_detectable then states the smallest change that h can see: the shift, in standard
  # units, whose run length is at most horizon_steps. The caller scales it by its standard error, and it is the caller
  # who states the rate.
  #
  #   SkiftetStatistical::Cusum.run([ 0.0, 0.0, 5.0 ], reference: 0.5, threshold: 4.0)
  #   # => { statistic: 4.5, alarm_at: 3, path: [0.0, 0.0, 4.5] }
  #   SkiftetStatistical::Cusum.average_run_length(shift: 0.0, reference: 0.5, threshold: 4.0)    # => 338.09...
  #   SkiftetStatistical::Cusum.threshold(reference: 0.5, false_alarm_steps: 365.0)               # => 4.0743...
  #   SkiftetStatistical::Cusum.minimum_detectable(reference: 0.5, threshold: 4.07, horizon_steps: 30.0) # => 0.4747...
  #
  # The run length falls from infinity to 0 as the shift rises and it rises with the threshold, so each search has one
  # answer. Near theta = 0 the formula cancels (at b = 5 it reads 0 or -55.5 at theta = 1e-9, not 25), so there the
  # same function is summed as its series. A run length past what exp or a Float can say reads as Infinity, the
  # formula's own limit. Every other number in and out is a finite Float: one past a Float's range is refused.
  module Cusum
    # Siegmund's correction to the boundary: 2 * 0.583.
    BOUNDARY_CORRECTION = 1.166
    # threshold searches [0, LARGEST_THRESHOLD].
    LARGEST_THRESHOLD = 50.0
    # Below this |2 * theta * b| the run length is summed as a series; at this size the formula has lost one digit.
    SERIES_BELOW = 0.5
    # The series stops at the term with this divisor, 2 * x**18 / 20!: under SERIES_BELOW the first term left out is
    # under 1e-24 of the sum.
    SERIES_LAST_DIVISOR = 20
    # A bound, not a tuning knob: each search ends once its two ends are neighbouring Floats.
    BISECTIONS = 200
    # Math.exp overflows past this.
    LARGEST_EXPONENT = Math.log(Float::MAX)

    class << self
      # The statistic over z, an Array of finite numbers: { statistic:, alarm_at:, path: }, path every S_t (one per
      # value, none for an empty z), statistic the last of them (S_0 = 0.0 for an empty z) and alarm_at the step of
      # the first S_t above the threshold, nil if there is none. The threshold is >= 0.
      def run(z, reference:, threshold:)
        raise ArgumentError, "z must be an Array of numbers, got #{z.class}" unless z.is_a?(Array)

        k = number(reference, "reference")
        h = level(threshold)
        statistic = 0.0
        alarm_at = nil
        path = z.map.with_index(1) do |value, step|
          statistic = gather(statistic, number(value, "each z"), k)
          alarm_at ||= step if statistic > h
          statistic
        end
        { statistic: statistic, alarm_at: alarm_at, path: path }
      end

      # Siegmund's run length for values of mean shift. The threshold is >= 0.
      def average_run_length(shift:, reference:, threshold:)
        run_length(drift(shift, reference), boundary(threshold))
      end

      # The threshold in [0, LARGEST_THRESHOLD] whose in-control run length is false_alarm_steps, a number > 0, by
      # Siegmund's approximation: the first Float whose approximate run length reaches it. The rate is that
      # approximation's, not a bound on the test. Simulated at 200,000 runs a setting over 30 to 5,000 steps, a real run
      # lasts within 1 % of the stated steps at a reference of 0.25 or below, within about 2 % at 0.5, 3 % at 0.75 and
      # 7 % at 1.0, mostly short, and a run that much shorter alarms that much more often. A run length no threshold in
      # the range gives raises ArgumentError, because answering an end would answer a rate nobody stated.
      def threshold(reference:, false_alarm_steps:)
        theta = drift(0.0, reference)
        target = positive(false_alarm_steps, "false_alarm_steps")
        lowest = run_length(theta, BOUNDARY_CORRECTION)
        highest = run_length(theta, LARGEST_THRESHOLD + BOUNDARY_CORRECTION)
        unless target.between?(lowest, highest)
          raise ArgumentError, "no threshold in [0, #{LARGEST_THRESHOLD}] has a run length of #{target} at reference " \
                               "#{reference}: they run from #{lowest} to #{highest}"
        end
        return 0.0 if target == lowest

        bisect(0.0, LARGEST_THRESHOLD) { |h| run_length(theta, h + BOUNDARY_CORRECTION) >= target }
      end

      # The smallest shift, in standard units, whose run length is at most horizon_steps, a number > 0: the first Float
      # that meets the horizon. A horizon at or past the in-control run length answers a shift at or below 0, as the
      # formula has it: noise alone alarms inside such a horizon.
      def minimum_detectable(reference:, threshold:, horizon_steps:)
        k = number(reference, "reference")
        b = boundary(threshold)
        horizon = positive(horizon_steps, "horizon_steps")

        low, high = drift_bracket(b, horizon)
        low = number(k + low, "the smallest shift")
        high = number(k + high, "the smallest shift")
        bisect(low, high) { |shift| run_length(shift - k, b) <= horizon }
      end

      private

      def gather(statistic, value, k)
        moved = statistic + value - k
        raise ArgumentError, "the statistic has left a Float's range, at #{moved}" unless moved.finite?

        moved.positive? ? moved : 0.0
      end

      # The formula, with x = 2 * theta * b. Since x / (2 * theta**2) = b / theta, the same function writes as
      # (b / theta) * (1 + (exp(-x) - 1) / x), in which nothing overflows that the answer does not.
      def run_length(theta, b)
        x = 2.0 * theta * b
        return Float::INFINITY if x < -LARGEST_EXPONENT
        return b * b * near_zero(x) if x.abs < SERIES_BELOW

        (b / theta) * (1.0 + ((Math.exp(-x) - 1.0) / x))
      end

      # The run length over b**2: 2 * (exp(-x) - 1 + x) / x**2 = the sum over n >= 0 of 2 * (-x)**n / (n + 2)!, whose
      # first term is 1 and each next the last times -x / (n + 2). Exactly 1.0 at x = 0.
      def near_zero(x)
        term = 1.0
        (3..SERIES_LAST_DIVISOR).sum(1.0) { |divisor| term *= -x / divisor }
      end

      # Two drifts around the one whose run length is horizon. Above 0 the run length lies under b / theta, so that
      # drift is an upper end. Below 0 it grows past any horizon: b >= 1.166 takes it past exp's range by 305.
      def drift_bracket(b, horizon)
        return [ 0.0, b / horizon ] if run_length(0.0, b) > horizon

        low = -1.0
        low *= 2.0 until run_length(low, b) > horizon
        [ low, 0.0 ]
      end

      # The first Float of [low, high] where the block holds, for a block that holds from one point of the range up.
      def bisect(low, high)
        BISECTIONS.times do
          middle = low + ((high - low) / 2.0)
          break if middle <= low || middle >= high

          yield(middle) ? high = middle : low = middle
        end
        high
      end

      def drift(shift, reference)
        number(number(shift, "shift") - number(reference, "reference"), "shift - reference")
      end

      def boundary(threshold)
        level(threshold) + BOUNDARY_CORRECTION
      end

      def level(threshold)
        h = number(threshold, "threshold")
        raise ArgumentError, "threshold must be >= 0, got #{threshold.inspect}" unless h >= 0.0

        h
      end

      def positive(value, name)
        float = number(value, name)
        raise ArgumentError, "#{name} must be > 0, got #{value.inspect}" unless float.positive?

        float
      end

      # A finite real number, as the other modules read one (a NaN, an infinity, a Complex and a non-number are refused
      # alike), and finite as a Float too (an Integer or Rational beyond a Float's range is not).
      def number(value, name)
        float = value.to_f if value.is_a?(Numeric) && value.real? && value.finite?
        return float if float&.finite?

        raise ArgumentError, "#{name} must be a finite number, got #{value.inspect}"
      end
    end
  end
end
