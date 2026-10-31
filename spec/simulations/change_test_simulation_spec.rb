# frozen_string_literal: true

# Cusum on generated standardised streams (SL13 (k)). Tagged `simulation: true`: excluded unless SIMULATIONS=1
# (spec/spec_helper.rb). Every figure the examples measure is printed through the reporter (the JSON formatter keeps it
# under "messages") for the README's measured-results section.
#
# Two settings from round generic values, each a stated false-alarm rate and a horizon:
#   - reference 0.5, one false alarm in 365 steps, a horizon of 30 steps;
#   - reference 0.25, one false alarm in 1,460 steps (365 split over four streams), a horizon of 60 steps.
# The threshold is Cusum.threshold's for the stated rate and the smallest change Cusum.minimum_detectable's for that
# threshold and horizon, so the simulation measures the two answers a caller states.
#
# FALSE ALARMS: one null stream of 2,000 consecutive spans of 365 steps (730,000 steps), z ~ N(0, 1) by
# Sampler#gaussian. After an alarm the statistic starts again from 0 on the steps that follow (Cusum.run on the rest),
# as a run length is defined, and nothing else restarts it, so the stream counts every alarm a test in steady use
# raises. The count must lie inside the 95 % binomial interval of 730,000 steps at the stated rate, 1 /
# false_alarm_steps a step. (Measured beside it: 2,000 separate spans each starting at 0 count 1.2 % fewer alarms at
# reference 0.5 and 3.9 % fewer at 0.25, because every span's first steps start from 0.)
#
# DETECTION: 2,000 runs of the stream shifted by the smallest detectable change from its first step, the statistic
# starting at 0 (the slowest start: a statistic already above 0 alarms no later); the change is caught when the first
# alarm comes within the horizon, in at least half the runs. Beside it, the same change after 365 null steps of a stream
# that restarts after each alarm, caught when the first alarm at or after the change comes within the horizon.
module ChangeTestSimulation
  module_function

  SETTINGS = [
    { reference: 0.5, false_alarm_steps: 365.0, horizon_steps: 30, seed: 20_261_111 },
    { reference: 0.25, false_alarm_steps: 1_460.0, horizon_steps: 60, seed: 20_261_112 }
  ].freeze
  RUNS = 2_000
  STEPS = 365
  INTERVAL_MASS = 0.95

  def stream(sampler, steps, shift)
    Array.new(steps) { shift + sampler.gaussian }
  end

  # The steps of every alarm in z, the statistic starting again from 0 after each one. Cusum.run reads a window from
  # the last alarm, doubled until it holds the next alarm or reaches the end, so a long stream is read once or twice.
  def alarms(z, reference:, threshold:, window: 1_024)
    found = []
    start = 0
    span = window
    while start < z.size
      at = SkiftetStatistical::Cusum.run(z[start, span], reference: reference, threshold: threshold)[:alarm_at]
      if at
        found << (start + at)
        start += at
        span = window
      else
        break if start + span >= z.size

        span *= 2
      end
    end
    found
  end

  # [lower, upper]: the central interval of Binomial(trials, rate) holding at least `mass`, its tails each at most
  # (1 - mass) / 2. The probabilities are summed from log-gamma terms.
  def binomial_interval(trials, rate, mass: INTERVAL_MASS)
    tail = (1.0 - mass) / 2.0
    log_rate = Math.log(rate)
    log_rest = Math.log(1.0 - rate)
    log_choose = ->(k) { Math.lgamma(trials + 1.0)[0] - Math.lgamma(k + 1.0)[0] - Math.lgamma(trials - k + 1.0)[0] }
    cumulative = 0.0
    lower = nil
    (0..trials).each do |k|
      cumulative += Math.exp(log_choose.call(k) + (k * log_rate) + ((trials - k) * log_rest))
      lower ||= k if cumulative > tail
      return [ lower, k ] if cumulative >= 1.0 - tail
    end
  end

  def report(text)
    RSpec.configuration.reporter.message("[change test simulation] #{text}")
  end
end

RSpec.describe "Cusum simulation", simulation: true do
  ChangeTestSimulation::SETTINGS.each do |setting|
    reference = setting[:reference]
    rate = 1.0 / setting[:false_alarm_steps]
    label = "reference #{reference}, one false alarm in #{setting[:false_alarm_steps].to_i} steps"

    it "SL13(k) raises false alarms on null streams inside the stated rate's 95 % binomial interval (#{label})" do
      threshold = SkiftetStatistical::Cusum.threshold(reference: reference,
                                                      false_alarm_steps: setting[:false_alarm_steps])
      sampler = SkiftetStatistical::Sampler.new(Random.new(setting[:seed]))
      steps = ChangeTestSimulation::RUNS * ChangeTestSimulation::STEPS
      z = ChangeTestSimulation.stream(sampler, steps, 0.0)
      count = ChangeTestSimulation.alarms(z, reference: reference, threshold: threshold,
                                          window: 4 * setting[:false_alarm_steps].to_i).size
      lower, upper = ChangeTestSimulation.binomial_interval(steps, rate)

      ChangeTestSimulation.report(format("%s: threshold %.4f, %d false alarms in %d null steps, %.3f per %d steps " \
                                         "(stated %.3f; 95 %% interval %d-%d)", label, threshold, count, steps,
                                         count * ChangeTestSimulation::STEPS.to_f / steps, ChangeTestSimulation::STEPS,
                                         ChangeTestSimulation::STEPS * rate, lower, upper))
      expect(count).to be_between(lower, upper), "#{count} false alarms, outside #{lower}-#{upper}"
    end

    it "SL13(k) catches a step of the minimum detectable size within the horizon in at least half the runs " \
       "(#{label}, horizon #{setting[:horizon_steps]})" do
      horizon = setting[:horizon_steps]
      threshold = SkiftetStatistical::Cusum.threshold(reference: reference,
                                                      false_alarm_steps: setting[:false_alarm_steps])
      shift = SkiftetStatistical::Cusum.minimum_detectable(reference: reference, threshold: threshold,
                                                           horizon_steps: horizon.to_f)
      sampler = SkiftetStatistical::Sampler.new(Random.new(setting[:seed] + 1_000))

      from_start = Array.new(ChangeTestSimulation::RUNS) do
        z = ChangeTestSimulation.stream(sampler, horizon, shift)
        !SkiftetStatistical::Cusum.run(z, reference: reference, threshold: threshold)[:alarm_at].nil?
      end.count(true)
      after_null = Array.new(ChangeTestSimulation::RUNS) do
        z = ChangeTestSimulation.stream(sampler, ChangeTestSimulation::STEPS, 0.0) +
            ChangeTestSimulation.stream(sampler, horizon, shift)
        first = ChangeTestSimulation.alarms(z, reference: reference, threshold: threshold)
                                    .find { |step| step > ChangeTestSimulation::STEPS }
        !first.nil?
      end.count(true)

      ChangeTestSimulation.report(format("%s: smallest detectable change %.4f within %d steps; caught from a start " \
                                         "at 0 in %d of %d runs (%.3f), after %d null steps in %d (%.3f)", label,
                                         shift, horizon, from_start, ChangeTestSimulation::RUNS,
                                         from_start.to_f / ChangeTestSimulation::RUNS, ChangeTestSimulation::STEPS,
                                         after_null, after_null.to_f / ChangeTestSimulation::RUNS))
      expect(from_start).to be >= ChangeTestSimulation::RUNS / 2
      expect(after_null).to be >= ChangeTestSimulation::RUNS / 2
    end
  end
end
