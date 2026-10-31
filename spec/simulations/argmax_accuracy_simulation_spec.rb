# frozen_string_literal: true

# BetaArgmax on 30 fixtures (SL13 (i)): `.pick`'s frequencies over 10^4 seeded draws a fixture, and `.probabilities`
# against 10^6 plain Sampler#beta draws a fixture, each member within 5 standard deviations, sqrt(p * (1 - p) / n) with
# p the answer of `.probabilities`. Tagged `simulation: true`: excluded unless SIMULATIONS=1 (spec/spec_helper.rb).
# Every figure the examples measure is printed through the reporter (the JSON formatter keeps it under "messages") for
# the README's measured-results section.
#
# THE FIXTURES, from round generic values only: closed forms (equal members, Beta(1, 1) against Beta(2, 1), Beta(1,
# 0.05) beside a uniform), parameters well below 1 at either end, U shapes, close competitors, up to 8 members, mixed
# widths, and sharp members, alpha + beta from 10^3 to 2 * 10^5: beside a member ten times sharper, beside a wide
# member, near either end, and two of 2 * 10^5 a hair apart. Each carries the epsilon `.pick` explores with (0 to 0.3).
#
# `.pick` takes its uniforms from one seeded Random a fixture, in this order a draw: explore, the priorities, the
# uniforms. The Monte Carlo reference is the plain argmax of one Sampler#beta draw a member (ties to the lower index, as
# `.pick` breaks them), set against `.probabilities` at epsilon 0, so it checks the integral itself. No fixture holds two
# members whose parameter at the same end (alpha near 0, beta near 1) is 0.2 or less, because there Sampler#beta rounds
# a share of its draws to exactly 0 or 1 and two such draws would tie.
module ArgmaxAccuracySimulation
  module_function

  PICK_DRAWS = 10_000
  BETA_DRAWS = 1_000_000
  STANDARD_DEVIATIONS = 5.0
  PICK_SEED = 20_261_121
  BETA_SEED = 20_261_122

  FIXTURES = [
    # Closed forms and ordinary shapes.
    { params: [ [ 1.0, 1.0 ], [ 2.0, 1.0 ] ], epsilon: 0.0 },
    { params: [ [ 1.0, 1.0 ], [ 1.0, 1.0 ] ], epsilon: 0.0 },
    { params: [ [ 2.0, 8.0 ], [ 8.0, 2.0 ] ], epsilon: 0.1 },
    { params: [ [ 1.0, 0.05 ], [ 1.0, 1.0 ] ], epsilon: 0.0 },
    { params: [ [ 0.05, 1.0 ], [ 1.0, 1.0 ] ], epsilon: 0.0 },
    { params: [ [ 0.5, 0.5 ], [ 1.0, 1.0 ] ], epsilon: 0.0 },
    { params: [ [ 0.2, 0.3 ], [ 0.3, 0.2 ] ], epsilon: 0.05 },
    { params: [ [ 0.1, 0.1 ], [ 2.0, 2.0 ] ], epsilon: 0.0 },
    { params: [ [ 3.0, 7.0 ], [ 4.0, 6.0 ], [ 5.0, 5.0 ] ], epsilon: 0.0 },
    { params: [ [ 1.0, 9.0 ], [ 2.0, 8.0 ], [ 3.0, 7.0 ], [ 4.0, 6.0 ] ], epsilon: 0.05 },
    { params: [ [ 2.0, 3.0 ], [ 3.0, 2.0 ], [ 5.0, 5.0 ], [ 1.0, 1.0 ], [ 4.0, 6.0 ] ], epsilon: 0.1 },
    { params: [ [ 12.0, 88.0 ], [ 15.0, 85.0 ], [ 9.0, 91.0 ], [ 20.0, 180.0 ], [ 3.0, 17.0 ] ], epsilon: 0.1 },
    { params: [ [ 0.5, 50.0 ], [ 50.0, 0.5 ] ], epsilon: 0.0 },
    { params: [ [ 10.0, 10.0 ], [ 10.5, 9.5 ] ], epsilon: 0.0 },
    { params: [ [ 1.0, 3.0 ], [ 3.0, 1.0 ], [ 2.0, 2.0 ] ], epsilon: 0.2 },
    { params: [ [ 0.8, 0.8 ], [ 0.9, 0.7 ], [ 0.7, 0.9 ] ], epsilon: 0.0 },
    { params: [ [ 3.0, 7.0 ], [ 3.1, 6.9 ], [ 2.9, 7.1 ], [ 3.05, 6.95 ] ], epsilon: 0.1 },
    { params: Array.new(6) { [ 1.0, 1.0 ] }, epsilon: 0.0 },
    { params: [ [ 5.0, 1.0 ], [ 1.0, 5.0 ] ], epsilon: 0.0 },
    { params: [ [ 100.0, 900.0 ], [ 1.0, 9.0 ] ], epsilon: 0.0 },
    { params: [ [ 2.0, 2.0 ], [ 2.0, 2.0 ], [ 2.0, 2.0 ], [ 3.0, 3.0 ] ], epsilon: 0.3 },
    { params: Array.new(8) { |i| [ i + 1.0, i + 2.0 ] }, epsilon: 0.05 },
    { params: [ [ 1.0, 2.0 ], [ 2.0, 1.0 ], [ 1.5, 1.5 ] ], epsilon: 0.1 },
    { params: [ [ 4.0, 4.0 ], [ 4.4, 3.6 ] ], epsilon: 0.0 },
    { params: [ [ 0.3, 3.0 ], [ 3.0, 0.3 ], [ 1.0, 1.0 ] ], epsilon: 0.0 },
    # Sharp members, alpha + beta from 10^3 to 2 * 10^5.
    { params: [ [ 700.0, 300.0 ], [ 7_000.0, 3_000.0 ] ], epsilon: 0.1 },
    { params: [ [ 20_000.0, 80_000.0 ], [ 20.0, 80.0 ] ], epsilon: 0.0 },
    { params: [ [ 50.0, 99_950.0 ], [ 1.0, 999.0 ] ], epsilon: 0.0 },
    { params: [ [ 99_950.0, 50.0 ], [ 999.0, 1.0 ] ], epsilon: 0.0 },
    { params: [ [ 60_000.0, 140_000.0 ], [ 60_150.0, 139_850.0 ] ], epsilon: 0.1 }
  ].freeze

  # Counts of `.pick`'s answers over PICK_DRAWS draws of the fixture's uniforms.
  def pick_counts(fixture, rng)
    params = fixture[:params]
    k = params.size
    counts = Array.new(k, 0)
    PICK_DRAWS.times do
      explore = rng.rand
      priorities = Array.new(k) { rng.rand }
      uniforms = Array.new(k) { rng.rand }
      counts[SkiftetStatistical::BetaArgmax.pick(params, explore: explore, priorities: priorities, uniforms: uniforms,
                                                                epsilon: fixture[:epsilon])] += 1
    end
    counts
  end

  # Counts of the plain argmax over BETA_DRAWS joint draws, one Sampler#beta a member, ties to the lower index.
  def beta_counts(fixture, sampler)
    alphas = fixture[:params].map(&:first)
    betas = fixture[:params].map(&:last)
    k = alphas.size
    counts = Array.new(k, 0)
    BETA_DRAWS.times do
      best = 0
      top = -1.0
      index = 0
      while index < k
        draw = sampler.beta(alphas[index], betas[index])
        if draw > top
          top = draw
          best = index
        end
        index += 1
      end
      counts[best] += 1
    end
    counts
  end

  # The largest |frequency - p| / sd over the members, sd = sqrt(p * (1 - p) / draws); 0.0 where a member's p is 0 or 1
  # and its frequency equals it, infinite where it does not.
  def largest_z(counts, probabilities, draws)
    counts.each_with_index.map do |count, index|
      p = probabilities[index]
      gap = ((count.to_f / draws) - p).abs
      sd = Math.sqrt(p * (1.0 - p) / draws)
      if sd.zero?
        gap.zero? ? 0.0 : Float::INFINITY
      else
        gap / sd
      end
    end.max
  end

  def label(fixture)
    params = fixture[:params].map { |pair| pair.map { |x| x == x.round ? x.round : x } }
    "#{params.inspect} epsilon #{fixture[:epsilon]}"
  end

  def report(text)
    RSpec.configuration.reporter.message("[argmax simulation] #{text}")
  end
end

RSpec.describe "BetaArgmax simulation", simulation: true do
  it "SL13(i) gives .pick's frequencies over 10^4 seeded draws a fixture within 5 standard deviations of " \
     ".probabilities, on 30 fixtures" do
    worst = ArgmaxAccuracySimulation::FIXTURES.each_with_index.map do |fixture, index|
      probabilities = SkiftetStatistical::BetaArgmax.probabilities(fixture[:params], epsilon: fixture[:epsilon])
      counts = ArgmaxAccuracySimulation.pick_counts(fixture, Random.new(ArgmaxAccuracySimulation::PICK_SEED + index))
      z = ArgmaxAccuracySimulation.largest_z(counts, probabilities, ArgmaxAccuracySimulation::PICK_DRAWS)
      ArgmaxAccuracySimulation.report(format("pick %2d %s: largest gap %.2f standard deviations", index,
                                             ArgmaxAccuracySimulation.label(fixture), z))
      [ z, index ]
    end

    failing = worst.select { |z, _index| z > ArgmaxAccuracySimulation::STANDARD_DEVIATIONS }
    ArgmaxAccuracySimulation.report(format("pick: largest gap over the 30 fixtures %.2f standard deviations",
                                           worst.map(&:first).max))
    expect(ArgmaxAccuracySimulation::FIXTURES.size).to eq(30)
    expect(failing).to be_empty, "fixtures beyond 5 standard deviations: #{failing.inspect}"
  end

  it "SL13(i) gives .probabilities within 5 standard deviations of 10^6 plain Sampler#beta draws a fixture, on 30 " \
     "fixtures" do
    worst = ArgmaxAccuracySimulation::FIXTURES.each_with_index.map do |fixture, index|
      probabilities = SkiftetStatistical::BetaArgmax.probabilities(fixture[:params], epsilon: 0.0)
      sampler = SkiftetStatistical::Sampler.new(Random.new(ArgmaxAccuracySimulation::BETA_SEED + index))
      counts = ArgmaxAccuracySimulation.beta_counts(fixture, sampler)
      z = ArgmaxAccuracySimulation.largest_z(counts, probabilities, ArgmaxAccuracySimulation::BETA_DRAWS)
      gap = counts.each_with_index.map do |count, at|
        ((count.to_f / ArgmaxAccuracySimulation::BETA_DRAWS) - probabilities[at]).abs
      end.max
      ArgmaxAccuracySimulation.report(format("beta %2d %s: largest gap %.2f standard deviations (%.2e)", index,
                                             ArgmaxAccuracySimulation.label(fixture), z, gap))
      [ z, index ]
    end

    failing = worst.select { |z, _index| z > ArgmaxAccuracySimulation::STANDARD_DEVIATIONS }
    ArgmaxAccuracySimulation.report(format("beta: largest gap over the 30 fixtures %.2f standard deviations",
                                           worst.map(&:first).max))
    expect(ArgmaxAccuracySimulation::FIXTURES.size).to eq(30)
    expect(failing).to be_empty, "fixtures beyond 5 standard deviations: #{failing.inspect}"
  end
end
