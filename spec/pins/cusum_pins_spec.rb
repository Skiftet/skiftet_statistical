# frozen_string_literal: true

# SL15: the pins of Cusum, then the edges of the module that SL12's acceptance leaves open. Every pinned value was
# checked by hand first and written down before the module existed, and this file ran red against them. The reference
# shares no code with the module:
# - the paths: exact fractions of the inputs' binary values, S_t = max(0, S_(t-1) + z_t - k) from S_0 = 0, each S_t
#   then rounded to the nearest Float;
# - the run lengths: SL12's formula in 200-digit decimal arithmetic, every Float input at its exact binary value and
#   1.166 as written, so that nothing cancels however small theta is;
# - the threshold and the minimum detectable shift: a bisection of that same formula to 420 halvings, from the range
#   SL12 names (the threshold) and from a bracket of its own (the shift).
#
# One tolerance rule (SL15) for every Float this file compares with a pinned value: 1e-12 relative, arithmetic_relative,
# since nothing here goes through Math.lgamma. A pinned 0.0, 1.0 or infinity is compared exactly, whichever the output;
# an alarm step and a size must stay equal. No fixture holds an S_t within rounding of its threshold, so no rounding
# can move an alarm. A pinned value is never edited.
RSpec.describe "SkiftetStatistical::Cusum pins" do
  def cusum = SkiftetStatistical::Cusum

  def arithmetic_relative = 1e-12

  def expect_pin(actual, pinned, label)
    expect(actual).to be_a(Float), "#{label}: #{actual.inspect} is not a Float"
    if pinned.zero? || pinned == 1.0 || pinned.infinite?
      expect(actual).to eq(pinned), "#{label}: #{actual} is not exactly #{pinned}"
    else
      error = (actual - pinned).abs / pinned.abs
      expect(error).to be <= arithmetic_relative, "#{label}: #{actual} is #{error} from #{pinned}"
    end
  end

  def expect_pins(actual, pinned, label)
    expect(actual.size).to eq(pinned.size), "#{label}: #{actual.size} values, not #{pinned.size}"
    actual.zip(pinned).each_with_index { |(a, p), i| expect_pin(a, p, "#{label}[#{i}]") }
  end

  def stream = [ 0.3, 1.2, -0.4, 0.9, 1.7, -2.1, 0.2, 2.4, 1.1, -0.6, 3.2, -4.0, 0.7, 0.55, 1.35, -0.25 ]

  # A NaN, an infinity either way, a Complex, numbers beyond a Float's range, and things that are no number (SL18).
  def not_numbers = [ Float::NAN, Float::INFINITY, -Float::INFINITY, Complex(1.0, 0.0), 10**400, Rational(10**400, 1),
                      "1.0", nil, :one ]

  describe ".run" do
    it "SL15 pins the path, the first alarm and the last value of one stream under three settings" do
      [
        [ 0.5, 3.0, 11, [ 0.0, 0.7, 0.0, 0.4, 1.6, 0.0, 0.0, 1.9, 2.5, 1.4, 4.1000000000000005, 0.0,
                          0.19999999999999996, 0.25, 1.1, 0.3500000000000001 ] ],
        [ 0.2, 1.6, 5, [ 0.09999999999999998, 1.0999999999999999, 0.4999999999999999, 1.2, 2.6999999999999997,
                         0.39999999999999974, 0.39999999999999974, 2.5999999999999996, 3.4999999999999996,
                         2.6999999999999997, 5.7, 1.5, 1.9999999999999998, 2.3499999999999996, 3.5, 3.05 ] ],
        [ 0.1, 6.0, 11, [ 0.19999999999999998, 1.2999999999999998, 0.7999999999999999, 1.5999999999999999,
                          3.1999999999999997, 0.9999999999999998, 1.0999999999999999, 3.3999999999999995,
                          4.3999999999999995, 3.6999999999999997, 6.8, 2.7, 3.3, 3.75, 5.0, 4.65 ] ]
      ].each do |reference, threshold, alarm_at, path|
        result = cusum.run(stream, reference: reference, threshold: threshold)

        expect_pins(result[:path], path, "path at #{reference}, #{threshold}")
        expect(result[:alarm_at]).to eq(alarm_at)
        expect_pin(result[:statistic], path.last, "statistic at #{reference}, #{threshold}")
      end
    end

    it "SL15 reads Integers and Rationals as the Floats they are" do
      integers = cusum.run([ 1, 3, -2, 4 ], reference: 1, threshold: 2)
      expect_pins(integers[:path], [ 0.0, 2.0, 0.0, 3.0 ], "whole numbers")
      expect(integers[:alarm_at]).to eq(4)

      rationals = cusum.run([ Rational(1, 2), Rational(3, 2) ], reference: Rational(1, 4), threshold: Rational(3, 2))
      expect_pins(rationals[:path], [ 0.25, 1.5 ], "fractions")
      expect(rationals[:alarm_at]).to be_nil
    end

    it "SL12 answers an empty stream with S_0: nothing moved, nothing alarmed" do
      expect(cusum.run([], reference: 0.5, threshold: 4.0)).to eq(statistic: 0.0, alarm_at: nil, path: [])
    end

    it "SL12 keeps the first alarm when the statistic falls back under the threshold and crosses it again" do
      result = cusum.run([ 5.0, -9.0, 5.0 ], reference: 0.5, threshold: 4.0)

      expect_pins(result[:path], [ 4.5, 0.0, 4.5 ], "path")
      expect(result[:alarm_at]).to eq(1)
    end

    it "SL12 alarms at any excess over a threshold of 0, and takes a reference of 0" do
      expect(cusum.run([ 0.0, 1e-9 ], reference: 0.0, threshold: 0.0)[:alarm_at]).to eq(2)
      expect(cusum.run([ 0.0, 0.0 ], reference: 0.0, threshold: 0.0)[:alarm_at]).to be_nil
    end

    it "SL12 leaves the given list as it was and answers a path of its own" do
      given = [ 1.0, 2.0 ].freeze
      result = cusum.run(given, reference: 0.5, threshold: 4.0)

      expect(given).to eq([ 1.0, 2.0 ])
      expect(result[:path]).not_to equal(given)
    end

    it "SL18 refuses a statistic that leaves a Float's range, never reading it as Infinity" do
      expect { cusum.run([ 1e308, 1e308 ], reference: 0.0, threshold: 4.0) }
        .to raise_error(ArgumentError, /Float's range/)
      expect { cusum.run([ 1e308 ], reference: -1e308, threshold: 4.0) }.to raise_error(ArgumentError, /Float's range/)
    end

    it "SL18 refuses anything that is not an Array of numbers" do
      [ nil, 1.0, "1.0", { a: 1.0 }, (1..3) ].each do |not_a_list|
        expect { cusum.run(not_a_list, reference: 0.5, threshold: 4.0) }.to raise_error(ArgumentError, /Array/)
      end
      not_numbers.each do |value|
        expect { cusum.run([ 0.0, value ], reference: 0.5, threshold: 4.0) }.to raise_error(ArgumentError, /each z/)
      end
    end
  end

  describe ".average_run_length" do
    it "SL15 pins Siegmund's approximation on both sides of the reference and at it" do
      [
        [ 0.0, 0.5, 4.0, 3.38093167156557831081e+2 ],
        [ 1.0, 0.5, 4.0, 8.34341470526348622244e+0 ],
        [ 0.5, 0.5, 3.0, 1.73555560000000000000e+1 ],
        [ 0.0, 0.25, 8.0, 7.37795013148745440954e+2 ],
        [ 2.0, 0.5, 5.0, 3.88844444650132067745e+0 ],
        [ -1.0, 0.5, 2.0, 2.96056357281158703823e+3 ],
        [ 0.75, 0.25, 0.0, 9.55221790559942016093e-1 ],
        [ 0.0, 1.0, 6.5, 2.27809297135995335090e+6 ],
        [ 3.0, 0.5, 12.0, 5.18640000000000000000e+0 ],
        [ 0.0, 0.0, 10.0, 1.24679556000000000000e+2 ]
      ].each do |shift, reference, threshold, pinned|
        actual = cusum.average_run_length(shift: shift, reference: reference, threshold: threshold)
        expect_pin(actual, pinned, "(#{shift}, #{reference}, #{threshold})")
      end
    end

    # Here the plain formula cancels: exp(-x) + x - 1 is x**2 / 2 beside terms of 1. At b = 5 it keeps five digits at
    # theta = 1e-6 and reads 0 or -55.5 at theta = 1e-9 (measured). These pins hold the function's value, not those
    # digits.
    it "SL15 pins the run length through theta = 0, where the plain formula has no digits left" do
      [
        [ 1e-12, 2.49999999999166674127e+1 ], [ -1e-12, 2.50000000000833340794e+1 ],
        [ 1e-09, 2.49999999166666676211e+1 ], [ -1e-09, 2.50000000833333342877e+1 ],
        [ 1e-06, 2.49999166668750003294e+1 ], [ -1e-06, 2.50000833335416678294e+1 ],
        [ 0.001, 2.49168745840267876936e+1 ], [ -0.001, 2.50835420840287718343e+1 ],
        [ 0.01, 2.41870901797978665147e+1 ], [ -0.01, 2.58545903782381248614e+1 ],
        [ 0.0499, 2.13126641537161771940e+1 ], [ -0.0499, 2.97334897280222123744e+1 ],
        [ 0.05, 2.13061319425266851266e+1 ], [ -0.05, 2.97442541400256306366e+1 ],
        [ 0.0501, 2.12996028336913943849e+1 ], [ -0.0501, 2.97550242043496277167e+1 ]
      ].each do |shift, pinned|
        actual = cusum.average_run_length(shift: shift, reference: 0.0, threshold: 3.834)
        expect_pin(actual, pinned, "shift #{shift}")
      end
    end

    it "SL15 pins the far ends: a sharp rise, a run length near a Float's top, and one past it as exactly infinity" do
      expect_pin(cusum.average_run_length(shift: 1000.0, reference: 0.0, threshold: 4.0), 5.16550000000000000000e-3,
                 "a sharp rise")
      expect_pin(cusum.average_run_length(shift: -60.0, reference: 0.0, threshold: 4.0), 2.34694398882067213680e+265,
                 "far below")
      expect_pin(cusum.average_run_length(shift: -120.0, reference: 0.0, threshold: 4.0), Float::INFINITY,
                 "past the top")
    end

    it "SL12 meets b**2 from both sides, down to the smallest shifts" do
      [ 1e-13, 1e-30, 1e-200, 5e-324 ].flat_map { |shift| [ shift, -shift ] }.each do |shift|
        actual = cusum.average_run_length(shift: shift, reference: 0.0, threshold: 3.834)
        expect_pin(actual, 25.0, "shift #{shift}")
      end
    end

    it "SL12 falls as the shift rises, at every step through theta = 0 and through both joins of its two forms" do
      coarse = (-300..300).map { |i| i * 1e-3 }
      fine = [ -0.05, 0.05 ].flat_map { |join| (-50..50).map { |i| join + (i * 1e-10) } }

      [ coarse, fine ].each do |shifts|
        lengths = shifts.map { |shift| cusum.average_run_length(shift: shift, reference: 0.0, threshold: 3.834) }
        expect(lengths.each_cons(2).all? { |earlier, later| later < earlier }).to be(true)
      end
    end

    it "SL12 rises with the threshold, whichever side of the reference the shift lies" do
      [ -1.0, -1e-9, 0.0, 1e-9, 1.0 ].each do |theta|
        lengths = (0..20).map { |h| cusum.average_run_length(shift: theta, reference: 0.0, threshold: h * 0.5) }
        expect(lengths.each_cons(2).all? { |earlier, later| later > earlier }).to be(true)
      end
    end

    it "SL12 answers a number or infinity at the far ends, never NaN" do
      thetas = [ -Float::MAX, -1e200, -1e-200, 0.0, 1e-200, 1e200, Float::MAX ]
      thetas.product([ 0.0, 1.0, 1e200, Float::MAX ]).each do |theta, threshold|
        length = cusum.average_run_length(shift: theta, reference: 0.0, threshold: threshold)

        expect(length.nan?).to be(false), "theta #{theta}, threshold #{threshold}"
        expect(length).to be >= 0.0
      end
    end

    it "SL12 reads Integers and Rationals as the Floats they are" do
      expect(cusum.average_run_length(shift: 1, reference: Rational(1, 2), threshold: 4))
        .to eq(cusum.average_run_length(shift: 1.0, reference: 0.5, threshold: 4.0))
    end

    it "SL18 refuses a shift and a reference whose difference leaves a Float's range" do
      expect { cusum.average_run_length(shift: 1e308, reference: -1e308, threshold: 4.0) }
        .to raise_error(ArgumentError, /shift - reference/)
    end
  end

  describe ".threshold" do
    it "SL15 pins the threshold of a stated in-control run length" do
      [
        [ 0.5, 365.0, 4.07437237956765079410e+0 ],
        [ 0.5, 3650.0, 6.34798961748908395691e+0 ],
        [ 0.5, 10950.0, 7.44370121269578414656e+0 ],
        [ 0.25, 365.0, 6.67984366702771146659e+0 ],
        [ 0.0, 100.0, 8.83400000000000000000e+0 ],
        [ 1.0, 365.0, 2.13570314515212956006e+0 ],
        [ 0.75, 730.0, 3.31412279232767228979e+0 ]
      ].each do |reference, steps, pinned|
        expect_pin(cusum.threshold(reference: reference, false_alarm_steps: steps), pinned, "(#{reference}, #{steps})")
      end
    end

    it "SL12 answers the first Float whose approximate run length reaches the stated steps" do
      [ [ 0.5, 365.0 ], [ 0.25, 1000.0 ], [ 1.0, 30.0 ], [ 0.0, 100.0 ] ].each do |reference, steps|
        h = cusum.threshold(reference: reference, false_alarm_steps: steps)
        expect(cusum.average_run_length(shift: 0.0, reference: reference, threshold: h)).to be >= steps
        expect(cusum.average_run_length(shift: 0.0, reference: reference, threshold: h.prev_float)).to be < steps
      end
    end

    # The mean step of the first alarm over runs of N(0, 1) steps, each from S_0 = 0.
    def simulated_run_length(reference, threshold, runs:, seed:)
      sampler = SkiftetStatistical::Sampler.new(Random.new(seed))
      total = 0
      runs.times do
        statistic = 0.0
        step = 0
        loop do
          step += 1
          statistic = [ 0.0, statistic + sampler.gaussian - reference ].max
          break if statistic > threshold
        end
        total += step
      end
      total.to_f / runs
    end

    # The rate is Siegmund's approximation, not a bound on the test. At 200,000 runs a setting and 100 steps a real run
    # read 99.8 at reference 0.25, 99.1 at 0.5 and 93.8 at 1.0. These 10,000 runs hold that much and no more: the
    # module's doc gives the figures, and no run is held to a stated rate.
    it "SL12 states the approximation's rate, near the test's at small references and above it at 1.0" do
      { 0.25 => 0.05, 0.5 => 0.05, 1.0 => 0.10 }.each_with_index do |(reference, within), seed|
        h = cusum.threshold(reference: reference, false_alarm_steps: 100.0)
        simulated = simulated_run_length(reference, h, runs: 10_000, seed: seed + 1)

        expect((simulated - 100.0).abs / 100.0).to be < within, "reference #{reference}: #{simulated} steps"
        expect(simulated).to be < 100.0 if reference == 1.0
      end
    end

    it "SL12 reaches both ends of [0, 50]" do
      # At k = 0.5 and h = 0 the run length is (exp(1.166) - 2.166) / 0.5 = 2.0862...
      lowest = cusum.average_run_length(shift: 0.0, reference: 0.5, threshold: 0.0)
      expect(cusum.threshold(reference: 0.5, false_alarm_steps: lowest)).to eq(0.0)

      # At k = 0 and h = 50 it is 51.166**2 = 2617.96...
      highest = cusum.average_run_length(shift: 0.0, reference: 0.0, threshold: 50.0)
      expect(cusum.threshold(reference: 0.0, false_alarm_steps: highest)).to be_within(50.0 * 1e-12).of(50.0)
    end

    it "SL12 refuses a run length no threshold in [0, 50] gives, rather than answering an end of the range" do
      expect { cusum.threshold(reference: 0.5, false_alarm_steps: 2.0) }.to raise_error(ArgumentError, /no threshold/)
      expect { cusum.threshold(reference: 0.0, false_alarm_steps: 2618.0) }
        .to raise_error(ArgumentError, /no threshold/)
      expect { cusum.threshold(reference: 400.0, false_alarm_steps: 365.0) }
        .to raise_error(ArgumentError, /no threshold/)
    end

    it "SL12 refuses a run length that is not above 0" do
      [ 0, 0.0, -1.0 ].each do |steps|
        expect { cusum.threshold(reference: 0.5, false_alarm_steps: steps) }
          .to raise_error(ArgumentError, /false_alarm_steps/)
      end
    end
  end

  describe ".minimum_detectable" do
    it "SL15 pins the smallest shift whose run length is at most the horizon" do
      [
        [ 0.5, 4.0, 30.0, 4.66980523305339933593e-1 ],
        [ 0.5, 4.0, 7.0, 1.12364888501432875924e+0 ],
        [ 0.5, 8.0, 14.0, 1.09465673445099847247e+0 ],
        [ 0.25, 6.0, 60.0, 2.18618198118396055994e-1 ],
        [ 0.0, 5.0, 20.0, 1.88781648539611521727e-1 ],
        [ 1.0, 3.0, 5.0, 1.68840579612651422773e+0 ],
        [ 0.5, 12.0, 2.0, 7.04480174591905938073e+0 ],
        [ 0.5, 4.0, 0.25, 2.10667556883396097996e+1 ],
        [ 2.5, 4.0, 30.0, 2.46698052330533993359e+0 ]
      ].each do |reference, threshold, horizon, pinned|
        actual = cusum.minimum_detectable(reference: reference, threshold: threshold, horizon_steps: horizon)
        expect_pin(actual, pinned, "(#{reference}, #{threshold}, #{horizon})")
      end
    end

    # The formula has a root at any horizon, and the words "smallest shift" name it: nothing raises it to 0. Noise alone
    # alarms inside such a horizon, and that is what a shift below 0 says.
    it "SL12 answers a shift below 0 where the horizon outlasts the in-control run length" do
      shift = cusum.minimum_detectable(reference: 0.5, threshold: 4.0, horizon_steps: 400.0)

      expect_pin(shift, -2.50480612748132851834e-2, "past the in-control run length")
      expect(cusum.average_run_length(shift: 0.0, reference: 0.5, threshold: 4.0)).to be < 400.0
    end

    it "SL12 never answers a shift the horizon does not cover" do
      [ [ 0.5, 4.0, 30.0 ], [ 0.25, 8.0, 14.0 ], [ 0.5, 4.0, 400.0 ], [ 0.0, 3.0, 2.0 ] ].each do |k, h, horizon|
        shift = cusum.minimum_detectable(reference: k, threshold: h, horizon_steps: horizon)
        expect(cusum.average_run_length(shift: shift, reference: k, threshold: h)).to be <= horizon
      end
    end

    it "SL12 lands on the reference where the horizon is b**2, to a Float's spacing" do
      b = 1.166
      shift = cusum.minimum_detectable(reference: 0.5, threshold: 0.0, horizon_steps: b * b)

      expect(shift).to be_within(0.5 * 1e-12).of(0.5)
    end

    it "SL12 moves with the reference and with nothing else" do
      near = cusum.minimum_detectable(reference: 0.5, threshold: 4.0, horizon_steps: 30.0)
      far = cusum.minimum_detectable(reference: 2.5, threshold: 4.0, horizon_steps: 30.0)

      expect(far - near).to be_within(2.0 * 1e-12).of(2.0)
    end

    it "SL12 refuses a horizon that is not above 0" do
      [ 0, 0.0, -30.0 ].each do |horizon|
        expect { cusum.minimum_detectable(reference: 0.5, threshold: 4.0, horizon_steps: horizon) }
          .to raise_error(ArgumentError, /horizon_steps/)
      end
    end

    it "SL18 refuses an answer beyond a Float's range, never reading it as Infinity" do
      expect { cusum.minimum_detectable(reference: 0.0, threshold: 1e308, horizon_steps: 1e-10) }
        .to raise_error(ArgumentError, /shift/)
    end
  end

  describe "the shared rules" do
    it "SL18 refuses what is not a finite number, in every keyword of every call" do
      not_numbers.each do |value|
        calls = {
          "reference" => [ -> { cusum.run([ 1.0 ], reference: value, threshold: 4.0) },
                           -> { cusum.average_run_length(shift: 1.0, reference: value, threshold: 4.0) },
                           -> { cusum.threshold(reference: value, false_alarm_steps: 365.0) },
                           -> { cusum.minimum_detectable(reference: value, threshold: 4.0, horizon_steps: 30.0) } ],
          "threshold" => [ -> { cusum.run([ 1.0 ], reference: 0.5, threshold: value) },
                           -> { cusum.average_run_length(shift: 1.0, reference: 0.5, threshold: value) },
                           -> { cusum.minimum_detectable(reference: 0.5, threshold: value, horizon_steps: 30.0) } ],
          "shift" => [ -> { cusum.average_run_length(shift: value, reference: 0.5, threshold: 4.0) } ],
          "false_alarm_steps" => [ -> { cusum.threshold(reference: 0.5, false_alarm_steps: value) } ],
          "horizon_steps" => [ -> { cusum.minimum_detectable(reference: 0.5, threshold: 4.0, horizon_steps: value) } ]
        }

        calls.each do |name, list|
          list.each { |call| expect(&call).to raise_error(ArgumentError, /#{name}/) }
        end
      end
    end

    it "SL12 refuses a threshold below 0, which no statistic could stay under" do
      [ -1e-9, -1.0 ].each do |threshold|
        [ -> { cusum.run([ 1.0 ], reference: 0.5, threshold: threshold) },
          -> { cusum.average_run_length(shift: 1.0, reference: 0.5, threshold: threshold) },
          -> { cusum.minimum_detectable(reference: 0.5, threshold: threshold, horizon_steps: 30.0) } ]
          .each { |call| expect(&call).to raise_error(ArgumentError, /threshold/) }
      end
    end

    it "SL18 answers from its inputs alone and leaves the global Random where it found it" do
      answers = lambda do
        [ cusum.threshold(reference: 0.5, false_alarm_steps: 365.0),
          cusum.minimum_detectable(reference: 0.5, threshold: 4.0, horizon_steps: 30.0),
          cusum.run(stream, reference: 0.5, threshold: 3.0) ]
      end

      seed = Random.srand(1)
      first = answers.call
      after_the_calls = Random.rand
      Random.srand(1)
      with_no_calls = Random.rand
      Random.srand(seed)

      expect(after_the_calls).to eq(with_no_calls)
      expect(answers.call).to eq(first)
    end
  end
end
