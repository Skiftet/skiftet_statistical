# frozen_string_literal: true

# Acceptance for SL12: the one-sided CUSUM S_t = max(0, S_(t-1) + z_t - k) from S_0 = 0, Siegmund's run length, the
# threshold for a stated false-alarm rate and the minimum detectable shift. Steps count from 1, so `alarm_at` is the
# run length of the first alarm, the quantity `average_run_length` approximates. The path holds every S_t, one per z,
# with no reset after an alarm; `statistic` is the last S_t. Expected values are worked out by hand, by this file's
# own copy of SL12's formula with its own bisection, and by a simulation of 20,000 runs per fixture (17 Acceptance).
RSpec.describe "SkiftetStatistical::Cusum" do
  def cusum = SkiftetStatistical::Cusum

  # SL12's formula, written here: (exp(-2 theta b) + 2 theta b - 1) / (2 theta**2), theta = shift - k, b = h + 1.166;
  # b**2 at theta = 0.
  def siegmund(shift, reference, threshold)
    theta = shift - reference
    b = threshold + 1.166
    return b * b if theta.zero?

    (Math.exp(-2.0 * theta * b) + (2.0 * theta * b) - 1.0) / (2.0 * theta * theta)
  end

  # A bisection to the last bit, for the answers the library finds by its own search.
  def solve(low, high)
    200.times do
      middle = (low + high) / 2.0
      yield(middle) ? high = middle : low = middle
    end
    (low + high) / 2.0
  end

  # The mean step of the first alarm over runs of N(shift, 1) steps, each from S_0 = 0, on 0.2.0's Box-Muller.
  def simulated_run_length(shift, reference, threshold, runs:, seed:)
    sampler = SkiftetStatistical::Sampler.new(Random.new(seed))
    total = 0
    runs.times do
      statistic = 0.0
      step = 0
      loop do
        step += 1
        statistic = [ 0.0, statistic + shift + sampler.gaussian - reference ].max
        break if statistic > threshold
      end
      total += step
    end
    total.to_f / runs
  end

  describe ".run" do
    it "SL12 accumulates excesses over the reference from 0 and alarms at the first step above the threshold" do
      result = cusum.run([ 0.0, 0.0, 5.0 ], reference: 0.5, threshold: 4.0)

      expect(result.keys).to contain_exactly(:statistic, :alarm_at, :path)
      # S = (0, 0, 4.5): 0 - 0.5 floors at 0 twice, then 0 + 5 - 0.5.
      expect(result[:path].size).to eq(3)
      result[:path].zip([ 0.0, 0.0, 4.5 ]).each { |actual, expected| expect(actual).to be_within(1e-12).of(expected) }
      expect(result[:statistic]).to be_within(1e-12).of(4.5)
      expect(result[:alarm_at]).to eq(3)
    end

    it "SL12 floors at 0, keeps the path past the alarm with no reset, and reports its last value" do
      result = cusum.run([ 1.0, 1.0, -3.0, 2.0, 2.0, 2.0 ], reference: 0.5, threshold: 2.0)
      # S = (0.5, 1.0, max(0, 1 - 3.5) = 0, 1.5, 3.0, 4.5): first above 2 at step 5.
      result[:path].zip([ 0.5, 1.0, 0.0, 1.5, 3.0, 4.5 ]).each do |actual, expected|
        expect(actual).to be_within(1e-12).of(expected)
      end
      expect(result[:path].size).to eq(6)
      expect(result[:alarm_at]).to eq(5)
      expect(result[:statistic]).to be_within(1e-12).of(4.5)
    end

    it "SL12 alarms only strictly above the threshold, and not at all when the statistic stays at or below it" do
      # S = (2, 4, 4): equal to h, never above it.
      level = cusum.run([ 2.5, 2.5, 0.5 ], reference: 0.5, threshold: 4.0)
      expect(level[:alarm_at]).to be_nil
      expect(level[:statistic]).to be_within(1e-12).of(4.0)

      # S = (2, 4, 4.25): above at step 3.
      above = cusum.run([ 2.5, 2.5, 0.75 ], reference: 0.5, threshold: 4.0)
      expect(above[:alarm_at]).to eq(3)

      # S = (0, 0.25, 0): nothing.
      quiet = cusum.run([ 0.25, 0.75, -1.0 ], reference: 0.5, threshold: 4.0)
      expect(quiet[:alarm_at]).to be_nil
      expect(quiet[:statistic]).to be_within(1e-12).of(0.0)
      quiet[:path].zip([ 0.0, 0.25, 0.0 ]).each { |actual, expected| expect(actual).to be_within(1e-12).of(expected) }
    end
  end

  describe ".average_run_length" do
    it "SL12 is Siegmund's approximation, b**2 where the shift equals the reference" do
      # theta = -0.5, b = 5.166: (e**5.166 - 5.166 - 1) / 0.5 = 338.1...
      expect(cusum.average_run_length(shift: 0.0, reference: 0.5, threshold: 4.0))
        .to be_within(1e-9).of((Math.exp(5.166) - 6.166) / 0.5)
      # theta = 0.5, b = 5.166: (e**-5.166 + 4.166) / 0.5 = 8.343...
      expect(cusum.average_run_length(shift: 1.0, reference: 0.5, threshold: 4.0))
        .to be_within(1e-12).of((Math.exp(-5.166) + 4.166) / 0.5)
      # theta = 0: (3 + 1.166)**2 = 17.355556
      expect(cusum.average_run_length(shift: 0.5, reference: 0.5, threshold: 3.0)).to be_within(1e-12).of(4.166**2)
    end

    it "SL12 is within 10 % of a 20,000-run simulation on 5 fixtures" do
      # (shift, k, h): an in-control run, theta = 0, two shifted runs and a smaller reference. Measured once by this
      # file's simulation on seed 1, the distance to Siegmund was at most 3.1 % (at shift 2: 3.222 against 3.326).
      [ [ 0.0, 0.5, 3.0 ], [ 0.5, 0.5, 3.0 ], [ 1.0, 0.5, 4.0 ], [ 2.0, 0.5, 4.0 ], [ 0.0, 0.25, 3.0 ] ]
        .each_with_index do |(shift, reference, threshold), seed|
          simulated = simulated_run_length(shift, reference, threshold, runs: 20_000, seed: seed + 1)
          approximate = cusum.average_run_length(shift: shift, reference: reference, threshold: threshold)
          expect((approximate - simulated).abs / simulated).to be < 0.10, "#{[ shift, reference, threshold ]}: " \
                                                                         "#{approximate} against #{simulated}"
        end
    end
  end

  describe ".threshold" do
    it "SL12 finds the threshold whose in-control run length is the stated number of steps" do
      # With k = 0 the in-control run length is (h + 1.166)**2, so 100 steps want h = 10 - 1.166.
      expect(cusum.threshold(reference: 0.0, false_alarm_steps: 100.0)).to be_within(1e-6).of(8.834)

      # k = 0.5, one false alarm in 365 steps: about 4.07; this file's bisection on SL12's formula gives it.
      expected = solve(0.0, 50.0) { |h| siegmund(0.0, 0.5, h) >= 365.0 }
      h = cusum.threshold(reference: 0.5, false_alarm_steps: 365.0)
      expect(h).to be_within(1e-6).of(expected)
      expect(h).to be_within(0.01).of(4.07)
      # An h within 1e-6 moves this run length by up to about 4e-4 (its slope in h is about the run length itself).
      expect(cusum.average_run_length(shift: 0.0, reference: 0.5, threshold: h)).to be_within(1e-3).of(365.0)
    end
  end

  describe ".minimum_detectable" do
    it "SL12 finds the smallest shift whose run length is at most the horizon" do
      threshold = solve(0.0, 50.0) { |h| siegmund(0.0, 0.5, h) >= 365.0 }
      expected = solve(0.0, 20.0) { |shift| siegmund(shift, 0.5, threshold) <= 30.0 } # about 0.475
      shift = cusum.minimum_detectable(reference: 0.5, threshold: threshold, horizon_steps: 30.0)

      expect(shift).to be_within(1e-6).of(expected)
      expect(cusum.average_run_length(shift: shift, reference: 0.5, threshold: threshold)).to be_within(1e-3).of(30.0)
      expect(cusum.average_run_length(shift: shift - 1e-3, reference: 0.5, threshold: threshold)).to be > 30.0
    end

    it "SL12 lands on the reference itself where the horizon is b**2" do
      # h = 3.834 gives b = 5 and a run length of 25 at theta = 0: the minimum detectable shift is k. Near theta = 0
      # the general formula loses digits to cancellation, so this one is held to 1e-5.
      expect(cusum.minimum_detectable(reference: 0.5, threshold: 3.834, horizon_steps: 25.0)).to be_within(1e-5).of(0.5)
    end
  end
end
