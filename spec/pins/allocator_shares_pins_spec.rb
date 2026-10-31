# frozen_string_literal: true

# SL15: the pins of Allocator.call with `fixed_shares:` and `arm_floors:` (SL1), one fixture per phase. Every pinned
# value was checked first against an independent reference: SL1's lift and step written out again in exact Rational
# arithmetic for every share, and the statistics recomputed from JointBeta.each_draw's rows on the same seed (each
# draw's best taken among the free arms, the interval from the sorted draws); every pinned Float lay within 4e-16
# relative of it. The shares are also worked out by hand in each fixture's comment (Config defaults: floor 0.10,
# max_step 0.15, warm-up 300, won split 0.95 / 0.05). This file ran red before the keywords existed. A pinned value is
# never edited.
#
# One tolerance rule (SL15) for every Float this file compares with a pinned or hand-worked value: 1e-12 relative, and
# a pinned 0.0, 1.0 or infinity exactly. SL15 gives outputs through lgamma 1e-9; none of these does (the draws are
# Sampler's, the intervals the sorted draws), so none takes it. A math library that rounds a last digit differently
# still passes, while an added, dropped or reordered draw (which moves every later draw, JointBeta) does not. nil,
# Integers, Symbols, Booleans, every Hash's key order: exact. A sum of weights is an invariant, not a pin: it is held
# to 1e-12 relative of 1.
module AllocatorSharesPins
  DRAWS = 500

  FIXTURES = {
    # The probe: 0.2.0's two-arm pinned fixture (spec/bandit/allocator_spec.rb, :shifting) with its arms floored.
    # P(best) 0.058 / 0.942: control is pinned at its 0.5 and a takes 0.5; from 0.5 / 0.5 nothing moves. The
    # statistics are 0.2.0's pins of the same fixture (floors move no draw).
    probe: { arms: %i[control a], tallies: { control: { exposures: 1_200, conversions: 120 },
                                             a: { exposures: 1_250, conversions: 150 } },
             status: "running", winner_arm: nil, previous_weights: { control: 0.5, a: 0.5 }, seed: 102,
             arm_floors: { control: 0.5, a: 0.02 } },
    # Shifting under five floors. P(best) 0.028 / 0.522 / 0.29 / 0.008 / 0.152: control (< 0.05) and c (< 0.02) are
    # pinned; a, b and d share 0.93 in proportion over 0.964. From 0.2 each a's move 0.3036 is the largest, lambda
    # 0.15 / 0.3036: a 0.35, control 0.2 + lambda * (0.05 - 0.2).
    floors_shifting: { arms: %i[control a b c d],
                       tallies: { control: { exposures: 1_000, conversions: 100 },
                                  a: { exposures: 1_000, conversions: 118 },
                                  b: { exposures: 1_000, conversions: 112 },
                                  c: { exposures: 1_000, conversions: 95 },
                                  d: { exposures: 1_000, conversions: 108 } },
                       status: "running", winner_arm: nil,
                       previous_weights: { control: 0.2, a: 0.2, b: 0.2, c: 0.2, d: 0.2 }, hold_satisfied: false,
                       seed: 2_026, arm_floors: { control: 0.05, a: 0.02, b: 0.02, c: 0.02, d: 0.02 } },
    # A previous share under its floor is lifted before the step: control 0.03 -> 0.05, a and b 0.95 * 0.6 / 0.97
    # and 0.95 * 0.37 / 0.97. The target: P(best) 0.23 / 0.726 / 0.044, b pinned at Config#floor 0.10 (not named),
    # control and a 0.9 in proportion over 0.956. b's move is the largest (0.2624), lambda 0.15 / 0.2624.
    floors_previous_lifted: { arms: %i[control a b],
                              tallies: { control: { exposures: 900, conversions: 90 },
                                         a: { exposures: 900, conversions: 99 },
                                         b: { exposures: 900, conversions: 81 } },
                              status: "running", winner_arm: nil,
                              previous_weights: { control: 0.03, a: 0.6, b: 0.37 }, seed: 11,
                              arm_floors: { control: 0.05 } },
    # A fixed arm (control 0.25) drawn beside three free arms: P(best) among the free arms 0.176 / 0.48 / 0.344, into
    # 0.75: 0.132 / 0.36 / 0.258, every one above 0.10; the largest move 0.118 <= 0.15 reaches it.
    fixed_three_free: { arms: %i[control a b c],
                        tallies: { control: { exposures: 1_000, conversions: 300 },
                                   a: { exposures: 1_000, conversions: 300 },
                                   b: { exposures: 1_000, conversions: 310 },
                                   c: { exposures: 1_000, conversions: 305 } },
                        status: "running", winner_arm: nil,
                        previous_weights: { control: 0.25, a: 0.25, b: 0.25, c: 0.25 }, seed: 31,
                        fixed_shares: { control: 0.25 } },
    # A fixed share and floors together; the free arms' previous shares sum to 0.9, not 1 - 0.15, so they are lifted
    # (scaled into 0.85) before the step. The target: P(best) 0.096 / 0.564 / 0.34 into 0.85, above every floor.
    # control's move 0.2962 is the largest: control 0.85 * 0.4 / 0.9 - 0.15.
    fixed_and_floors_lifted: { arms: %i[control a b c],
                               tallies: { control: { exposures: 800, conversions: 80 },
                                          a: { exposures: 800, conversions: 120 },
                                          b: { exposures: 800, conversions: 92 },
                                          c: { exposures: 800, conversions: 88 } },
                               status: "running", winner_arm: nil,
                               previous_weights: { control: 0.4, a: 0.1, b: 0.3, c: 0.2 }, hold_satisfied: false,
                               seed: 47, fixed_shares: { a: 0.15 }, arm_floors: { control: 0.05, b: 0.02, c: 0.02 } },
    # The automatic win among the free arms: a leads b (P(best) 1, expected loss 0) and is crowned; the won split
    # over a and b, 0.95 / 0.05 into 0.7, is 0.665 / 0.035, b is pinned at 0.10 and a takes 0.6.
    fixed_automatic_win: { arms: %i[control a b],
                           tallies: { control: { exposures: 2_000, conversions: 200 },
                                      a: { exposures: 2_000, conversions: 300 },
                                      b: { exposures: 2_000, conversions: 200 } },
                           status: "running", winner_arm: nil, previous_weights: { control: 0.3, a: 0.35, b: 0.35 },
                           seed: 53, fixed_shares: { control: 0.3 } },
    # A stored win (a String winner and a String key) whose winner's P(best) has fallen: result_changed. The won
    # split 0.025 / 0.025 / 0.95 pins control at 0.05; a and b share 0.95 over 0.975: 19 / 780 and 361 / 390.
    floors_stored_win: { arms: %i[control a b],
                         tallies: { control: { exposures: 1_500, conversions: 150 },
                                    a: { exposures: 1_500, conversions: 180 },
                                    b: { exposures: 1_500, conversions: 150 } },
                         status: "won", winner_arm: "b", previous_weights: { control: 0.05, a: 0.05, b: 0.9 },
                         seed: 59, arm_floors: { "control" => 0.05, a: 0.02, b: 0.02 } },
    # A pause whose previous weights break a's floor 0.05: a is pinned there and b and c share 0.75 over 0.79:
    # 75 / 158 and 87 / 316; control keeps its fixed 0.2.
    fixed_paused_lifted: { arms: %i[control a b c],
                           tallies: { control: { exposures: 400, conversions: 40 },
                                      a: { exposures: 400, conversions: 44 },
                                      b: { exposures: 400, conversions: 48 },
                                      c: { exposures: 400, conversions: 36 } },
                           status: "paused", winner_arm: nil,
                           previous_weights: { control: 0.2, a: 0.01, b: 0.5, c: 0.29 }, seed: 61,
                           fixed_shares: { control: 0.2 }, arm_floors: { a: 0.05 } },
    # Warm-up: equal targets over the free arms into 0.9 are 0.3 each; control is pinned at its 0.4 and a and b share
    # 0.5: 0.25 each; c keeps its fixed 0.1.
    fixed_floors_warmup: { arms: %i[control a b c],
                           tallies: { control: { exposures: 250, conversions: 25 },
                                      a: { exposures: 260, conversions: 31 },
                                      b: { exposures: 240, conversions: 22 },
                                      c: { exposures: 255, conversions: 30 } },
                           status: "running", winner_arm: nil, previous_weights: nil, seed: 67,
                           fixed_shares: { c: 0.1 }, arm_floors: { control: 0.4 } },
    # The automatic win of a lone free arm: a is the one free arm beside control's fixed 0.2, so its P(best) among the
    # free arms is 1 and its expected loss 0; past warm-up it is the leader and, with the default hold, crowned. The
    # won split over the one free arm, 1.0 into 0.8: a 0.8, control 0.2.
    lone_free_automatic_win: { arms: %i[control a],
                               tallies: { control: { exposures: 2_000, conversions: 200 },
                                          a: { exposures: 2_000, conversions: 400 } },
                               status: "running", winner_arm: nil, previous_weights: { control: 0.2, a: 0.8 },
                               seed: 73, fixed_shares: { control: 0.2 } },
    # A stored win (a String winner and a String key) of a lone free arm: b, fixed at 0.25, converts best, but P(best)
    # is taken among the free arms, so a's is 1 and the result has not changed. a takes 1 - 0.3 - 0.25 = 0.45.
    lone_free_stored_win: { arms: %i[control a b],
                            tallies: { control: { exposures: 1_500, conversions: 150 },
                                       a: { exposures: 1_500, conversions: 165 },
                                       b: { exposures: 1_500, conversions: 180 } },
                            status: "won", winner_arm: "a", previous_weights: nil, seed: 79,
                            fixed_shares: { "control" => 0.3, b: 0.25 } },
    # One arm with a floor declared: 0.2.0's guard (a single arm is never the leader) holds only when no share is
    # declared, so this arm, P(best) 1 and expected loss 0, is the leader and is crowned; the won split is 1.0.
    single_arm_floored: { arms: %i[control], tallies: { control: { exposures: 2_000, conversions: 200 } },
                          status: "running", winner_arm: nil, previous_weights: nil, seed: 83,
                          arm_floors: { control: 1 } }
  }.freeze

  def self.arm(exposures, conversions, lower, upper, p_best, expected_loss)
    rate = conversions.fdiv(exposures)
    { exposures: exposures, conversions: conversions, rate: rate, weighted_rate: rate, lower: lower, upper: upper,
      p_best: p_best, expected_loss: expected_loss, guardrails: {} }
  end

  OUTPUTS = {
    probe: {
      weights: { control: 0.5, a: 0.5 }, phase: :shifting,
      stats: {
        control: arm(1_200, 120, 0.08454080459361754, 0.1193895447520584, 0.058, 0.020645077600422623),
        a: arm(1_250, 150, 0.10425589449876138, 0.14233505559738507, 0.942, 0.0003593718797802569)
      },
      winner: nil, result_changed: false, leader: nil
    },
    floors_shifting: {
      weights: { control: 0.12588669445773254, a: 0.35, b: 0.23941433745643406, c: 0.11106403334927904,
                 d: 0.17363493473655436 },
      phase: :shifting,
      stats: {
        control: arm(1_000, 100, 0.0827994952956707, 0.11717014734046553, 0.028, 0.022826853772236127),
        a: arm(1_000, 118, 0.09889922158639372, 0.1402415906612876, 0.522, 0.0046358590540673),
        b: arm(1_000, 112, 0.09382682395680511, 0.13320850058619013, 0.29, 0.009758136088243207),
        c: arm(1_000, 95, 0.07895860374024155, 0.113526294522479, 0.008, 0.02800169710751501),
        d: arm(1_000, 108, 0.089703256719761, 0.12813882711275779, 0.152, 0.01476265162821708)
      },
      winner: nil, result_changed: false, leader: nil
    },
    floors_previous_lifted: {
      weights: { control: 0.14520513600381424, a: 0.6424237299755672, b: 0.21237113402061855 }, phase: :shifting,
      stats: {
        control: arm(900, 90, 0.0827756851083873, 0.12107822394911073, 0.23, 0.012575318436138674),
        a: arm(900, 99, 0.09217444354323936, 0.13231441282756898, 0.726, 0.0022514805973067272),
        b: arm(900, 81, 0.07400660122378394, 0.10969386923322458, 0.044, 0.022451788543094143)
      },
      winner: nil, result_changed: false, leader: nil
    },
    fixed_three_free: {
      weights: { control: 0.25, a: 0.132, b: 0.36, c: 0.258 }, phase: :shifting,
      stats: {
        control: arm(1_000, 300, 0.2695302289184409, 0.32926339074794153, nil, nil),
        a: arm(1_000, 300, 0.2737964003845644, 0.3290925718044319, 0.176, 0.019402695530969023),
        b: arm(1_000, 310, 0.2820721968196422, 0.3407800752852517, 0.48, 0.00837782964127477),
        c: arm(1_000, 305, 0.28018217289736697, 0.33426897474943623, 0.344, 0.01253196464772875)
      },
      winner: nil, result_changed: false, leader: nil
    },
    fixed_and_floors_lifted: {
      weights: { control: 0.22777777777777777, a: 0.15, b: 0.38263180272108843, c: 0.2395904195011338 },
      phase: :shifting,
      stats: {
        control: arm(800, 80, 0.07929293941972045, 0.12124319729305455, 0.096, 0.020420457350553915),
        a: arm(800, 120, 0.1250598032743269, 0.1761875719115644, nil, nil),
        b: arm(800, 92, 0.09297758887948915, 0.14002550351751084, 0.564, 0.0047173887301399375),
        c: arm(800, 88, 0.09219628960484165, 0.1329156405606138, 0.34, 0.009462840060074536)
      },
      winner: nil, result_changed: false, leader: nil
    },
    fixed_automatic_win: {
      weights: { control: 0.3, a: 0.6, b: 0.1 }, phase: :won,
      stats: {
        control: arm(2_000, 200, 0.08792306474640532, 0.11414961759066743, nil, nil),
        a: arm(2_000, 300, 0.1360257823963408, 0.16561854464769346, 1.0, 0.0),
        b: arm(2_000, 200, 0.08803720203471144, 0.11243257840718672, 0.0, 0.049853159843100785)
      },
      winner: :a, result_changed: false, leader: :a
    },
    floors_stored_win: {
      weights: { control: 0.05, a: 0.02435897435897436, b: 0.9256410256410257 }, phase: :won,
      stats: {
        control: arm(1_500, 150, 0.08614531892287274, 0.11796445009480919, 0.054, 0.020204886626311425),
        a: arm(1_500, 180, 0.10474153579559109, 0.13768576769130914, 0.91, 0.00039708931086854693),
        b: arm(1_500, 150, 0.08685847558235271, 0.11645464584669334, 0.036, 0.019906683226222764)
      },
      winner: :b, result_changed: true, leader: nil
    },
    fixed_paused_lifted: {
      weights: { control: 0.2, a: 0.05, b: 0.47468354430379744, c: 0.2753164556962025 }, phase: :paused,
      stats: {
        control: arm(400, 40, 0.07319974741323934, 0.13194038490440366, nil, nil),
        a: arm(400, 44, 0.08262018609726686, 0.1450143952708302, 0.318, 0.014717293117159514),
        b: arm(400, 48, 0.09500217160232088, 0.15508673680518995, 0.646, 0.004908824198238499),
        c: arm(400, 36, 0.06825457701397449, 0.12217305317474983, 0.036, 0.03580906125588933)
      },
      winner: nil, result_changed: false, leader: nil
    },
    fixed_floors_warmup: {
      weights: { control: 0.4, a: 0.25, b: 0.25, c: 0.1 }, phase: :warmup,
      stats: {
        control: arm(250, 25, 0.06945801814310947, 0.14929686479089951, 0.216, 0.023339376934655078),
        a: arm(260, 31, 0.08954255795031553, 0.16173029827625635, 0.676, 0.005422628885649038),
        b: arm(240, 22, 0.06421180085928069, 0.13367954863015896, 0.108, 0.03300718850632795),
        c: arm(255, 30, 0.08420378592504334, 0.1619705235298083, nil, nil)
      },
      winner: nil, result_changed: false, leader: nil
    },
    lone_free_automatic_win: {
      weights: { control: 0.2, a: 0.8 }, phase: :won,
      stats: {
        control: arm(2_000, 200, 0.08795975909114842, 0.11402439255561231, nil, nil),
        a: arm(2_000, 400, 0.18428418891427104, 0.2186483640427088, 1.0, 0.0)
      },
      winner: :a, result_changed: false, leader: :a
    },
    lone_free_stored_win: {
      weights: { control: 0.3, a: 0.44999999999999996, b: 0.25 }, phase: :won,
      stats: {
        control: arm(1_500, 150, 0.08783580573355107, 0.11681339542565668, nil, nil),
        a: arm(1_500, 165, 0.0942934046189332, 0.1266447383766245, 1.0, 0.0),
        b: arm(1_500, 180, 0.10615085810376278, 0.137618166254353, nil, nil)
      },
      winner: :a, result_changed: false, leader: nil
    },
    single_arm_floored: {
      weights: { control: 1.0 }, phase: :won,
      stats: { control: arm(2_000, 200, 0.08730360207795172, 0.11316647802113035, 1.0, 0.0) },
      winner: :control, result_changed: false, leader: :control
    }
  }.freeze

  # The inputs of 0.2.0's pinned fixtures (spec/bandit/allocator_spec.rb), copied unchanged.
  PINNED_020 = {
    shifting: { arms: %i[control a], tallies: { control: { exposures: 1_200, conversions: 120 },
                                               a: { exposures: 1_250, conversions: 150 } },
                status: "running", winner_arm: nil, previous_weights: { control: 0.5, a: 0.5 }, seed: 102 },
    leader_held: { arms: %i[control a], tallies: { control: { exposures: 5_000, conversions: 500 },
                                                  a: { exposures: 5_000, conversions: 650 } },
                   status: "running", winner_arm: nil, previous_weights: { control: 0.5, a: 0.5 },
                   hold_satisfied: false, seed: 110 },
    kish: { arms: %i[control a],
            tallies: { control: { exposures: 800, conversions: 80, weighted_exposures: 1_600.0,
                                  weighted_conversions: 150.0, weighted_exposures_squared: 4_000.0 },
                       a: { exposures: 800, conversions: 96, weighted_exposures: 1_450.0,
                            weighted_conversions: 140.0, weighted_exposures_squared: 2_000.0 } },
            status: "running", winner_arm: nil, previous_weights: { control: 0.5, a: 0.5 }, seed: 112 },
    guardrails: { arms: %i[control a],
                  tallies: { control: { exposures: 600, conversions: 60, guardrails: { unsubscribe: 3 } },
                             a: { exposures: 650, conversions: 70,
                                  guardrails: { "unsubscribe" => { conversions: 5 } } } },
                  status: "running", winner_arm: nil, previous_weights: { control: 0.5, a: 0.5 }, seed: 113 },
    four_arms_below_floor: { arms: %i[control a b c],
                             tallies: { control: { exposures: 700, conversions: 70 },
                                        a: { exposures: 700, conversions: 84 },
                                        b: { exposures: 700, conversions: 56 },
                                        c: { exposures: 700, conversions: 63 } },
                             status: "running", winner_arm: nil,
                             previous_weights: { control: 0.05, a: 0.8, b: 0.1, c: 0.05 }, seed: 115 },
    won_result_changed: { arms: %i[control a], tallies: { control: { exposures: 2_000, conversions: 260 },
                                                         a: { exposures: 2_000, conversions: 200 } },
                          status: "won", winner_arm: :a, previous_weights: { control: 0.05, a: 0.95 }, seed: 105 }
  }.freeze

  # SL15's tolerance for every Float here (none goes through lgamma).
  RELATIVE = 1e-12

  # The floors the lift's sweep draws its Config from: the default, others whose Config#max_arms rounds up (1 / 0.15
  # is 6.67 and 1 / 0.4 is 2.5, so seven and three arms over-fill the floor) or down, a Rational and an Integer one.
  LIFT_FLOORS = [ 0.10, 0.05, 0.15, 0.2, 0.25, 0.3, 1.0 / 3, 0.4, 0.5, Rational(1, 10), 1 ].freeze

  module_function

  # 0.2.0's private Allocator#with_floor (lib/skiftet_statistical/bandit/allocator.rb at 0.2.0), copied unchanged but
  # for its two instance reads, @arms and @config.floor, which are its arguments here. The lift's reference.
  def with_floor_020(arms, floor, target)
    fixed = []
    loop do
      free = arms - fixed
      mass = 1.0 - (fixed.size * floor)
      free_total = free.sum { |arm| target.fetch(arm) }
      scaled = free.to_h do |arm|
        [ arm, free_total.positive? ? mass * target.fetch(arm) / free_total : mass / free.size ]
      end
      below = scaled.select { |_arm, share| share < floor }.keys
      return arms.to_h { |arm| [ arm, fixed.include?(arm) ? floor : scaled.fetch(arm) ] } if below.empty?

      fixed.concat(below)
    end
  end

  # 0.2.0's convex step (the end of its private Allocator#step_toward, at 0.2.0), copied unchanged but for its
  # instance reads, which are its arguments here. The step's reference.
  def step_020(arms, max_step, previous, target)
    largest = arms.map { |arm| (target.fetch(arm) - previous.fetch(arm)).abs }.max
    lambda = largest > max_step ? max_step / largest : 1.0

    arms.to_h { |arm| [ arm, previous.fetch(arm) + (lambda * (target.fetch(arm) - previous.fetch(arm))) ] }
  end

  # One random target (Symbol arms, Float shares) over 1 to Config#max_arms arms: every share 0 (the equal split), a
  # won split, shares summing to 1, or raw shares among which some are 0, tiny, negative, large or exactly the floor.
  def random_target(rng, config)
    arms = (1..rng.rand(1..config.max_arms)).map { |index| :"arm#{index}" }
    case rng.rand(5)
    when 0 then arms.to_h { |arm| [ arm, 0.0 ] }
    when 1 then arms.each_with_index.to_h { |arm, index| [ arm, index.zero? ? 0.95 : 0.05 / (arms.size - 1) ] }
    when 2
      raw = arms.to_h { |arm| [ arm, rng.rand ] }
      total = raw.values.sum
      raw.transform_values { |share| share / total }
    else arms.to_h { |arm| [ arm, random_share(rng, config) ] }
    end
  end

  def random_share(rng, config)
    case rng.rand(8)
    when 0 then 0.0
    when 1 then rng.rand * 1e-6
    when 2 then -rng.rand * 0.2
    when 3 then rng.rand * 10
    when 4 then config.floor.to_f
    else rng.rand
    end
  end

  # The one rule: a Float within RELATIVE of the pin, relative to it; a pinned 0.0, 1.0 or infinity exactly.
  def close?(actual, expected)
    return false unless actual.is_a?(Float)
    return actual == expected if expected.zero? || expected == 1.0 || expected.infinite?

    (actual - expected).abs <= RELATIVE * expected.abs
  end

  # Floats by #close?; every other value of the same class and equal; Hash keys in the same order.
  def mismatches(actual, expected, path = "")
    case expected
    when Float
      close?(actual, expected) ? [] : [ "#{path}: #{actual.inspect} is not #{expected.inspect}" ]
    when Hash
      unless actual.is_a?(Hash) && actual.keys == expected.keys
        return [ "#{path}: keys #{actual.inspect} are not #{expected.keys.inspect}" ]
      end

      expected.flat_map { |key, value| mismatches(actual[key], value, "#{path}[#{key.inspect}]") }
    else
      actual.instance_of?(expected.class) && actual == expected ? [] : [ "#{path}: #{actual.inspect} is not #{expected.inspect}" ]
    end
  end
end

RSpec.describe SkiftetStatistical::Bandit::Allocator, "per-arm share pins" do
  def call_fixture(fixture, **options)
    described_class.call(**fixture.except(:seed), rng: Random.new(fixture[:seed]), draws: AllocatorSharesPins::DRAWS,
                         **options)
  end

  def tally(exposures, conversions)
    { exposures: exposures, conversions: conversions }
  end

  # Hand-worked weights by the one tolerance rule (AllocatorSharesPins.close?), keys in the same order.
  def expect_weights(actual, expected)
    expect(AllocatorSharesPins.mismatches(actual, expected)).to eq([])
  end

  # An invariant, not a pin: the weights sum to 1 within 1e-12 relative.
  def expect_sum_one(weights, label)
    expect((weights.values.sum - 1.0).abs).to be <= AllocatorSharesPins::RELATIVE, label
  end

  describe "pinned results" do
    AllocatorSharesPins::FIXTURES.each do |name, fixture|
      it "SL15 SL1 gives the pinned result for #{name}" do
        result = call_fixture(fixture)

        expect(AllocatorSharesPins.mismatches(result, AllocatorSharesPins::OUTPUTS.fetch(name))).to eq([])
      end
    end

    it "SL1 every pinned fixture holds each fixed share exactly and each free arm at or above its floor" do
      AllocatorSharesPins::FIXTURES.each do |name, fixture|
        weights = call_fixture(fixture)[:weights]
        fixed = (fixture[:fixed_shares] || {}).transform_keys(&:to_sym)
        floors = (fixture[:arm_floors] || {}).transform_keys(&:to_sym)

        expect_sum_one(weights, name.to_s)
        fixed.each { |arm, share| expect(weights.fetch(arm)).to eq(share), "#{name} #{arm}" }
        (fixture[:arms] - fixed.keys).each do |arm|
          expect(weights.fetch(arm)).to be >= floors.fetch(arm, 0.10), "#{name} #{arm}"
        end
      end
    end

    it "SL15 moves the pins with one extra draw before the joint draws" do
      fixture = AllocatorSharesPins::FIXTURES.fetch(:fixed_three_free)
      rng = Random.new(fixture[:seed])
      rng.rand
      result = described_class.call(**fixture.except(:seed), rng: rng, draws: AllocatorSharesPins::DRAWS)

      expect(AllocatorSharesPins.mismatches(result, AllocatorSharesPins::OUTPUTS.fetch(:fixed_three_free))).not_to eq([])
    end
  end

  describe "the probe" do
    it "SL1 keeps control at its floor 0.5 or above on 0.2.0's two-arm fixture (arms control and a)" do
      fixture = AllocatorSharesPins::PINNED_020.fetch(:shifting)
      result = call_fixture(fixture, arm_floors: { control: 0.5, a: 0.02 })

      expect(result.fetch(:weights).fetch(:control) >= 0.5).to be(true)
    end
  end

  describe "a keyword that names no arm" do
    it "SL14 an empty fixed_shares or arm_floors changes no 0.2.0 output, bit for bit" do
      AllocatorSharesPins::PINNED_020.each do |name, fixture|
        plain = Marshal.dump(call_fixture(fixture))

        [ { fixed_shares: {} }, { arm_floors: {} }, { fixed_shares: {}, arm_floors: {} },
          { fixed_shares: nil, arm_floors: {} } ].each do |options|
          expect(Marshal.dump(call_fixture(fixture, **options))).to eq(plain), "#{name} #{options.inspect}"
        end
      end
    end

    it "SL14 an empty arm_floors checks nothing: seven arms at Config#floor 0.15 answer as without it" do
      # 7 * 0.15 = 1.05 is above 1, which a declared call refuses; 0.2.0 allows seven arms at that floor
      # (Config#max_arms rounds 1 / 0.15 to 7), so a keyword naming no arm must too.
      arms = %i[control a b c d e f]
      config = described_class::Config.new(floor: 0.15)
      plain = described_class.call(arms: arms, tallies: {}, status: "running", winner_arm: nil, rng: Random.new(3),
                                   draws: 100, config: config)
      empty = described_class.call(arms: arms, tallies: {}, status: "running", winner_arm: nil, rng: Random.new(3),
                                   draws: 100, config: config, arm_floors: {})

      expect(Marshal.dump(empty)).to eq(Marshal.dump(plain))
      expect do
        described_class.call(arms: arms, tallies: {}, status: "running", winner_arm: nil, rng: Random.new(3),
                             draws: 100, config: config, arm_floors: { control: 0.15 })
      end.to raise_error(ArgumentError)
    end
  end

  describe "the checks beside SL1's" do
    def warm(**options)
      described_class.call(arms: %i[control a b], tallies: {}, status: "running", winner_arm: nil,
                           rng: Random.new(7), draws: 100, **options)
    end

    it "SL1 refuses a keyword that is not a Hash and an arm named twice (once as a Symbol, once as a String)" do
      expect_sum_one(warm(arm_floors: { control: 0.05, a: 0.02 })[:weights], "floored")

      [ { arm_floors: [ [ :control, 0.05 ] ] }, { fixed_shares: 0.2 }, { arm_floors: { control: 0.05, "control" => 0.05 } },
        { fixed_shares: { a: 0.2, "a" => 0.2 } }, { arm_floors: { 1 => 0.05 } } ].each do |options|
        expect { warm(**options) }.to raise_error(ArgumentError), options.inspect
      end
    end

    it "SL1 accepts a rational or whole floor and share, read as a Float" do
      # b fixed at 1/10; the free arms control and a share 0.9 equally in warm-up, 0.45 each; control is pinned at
      # its 1/2 and a takes 0.4. A single arm floored at the Integer 1 takes everything.
      rational = warm(arm_floors: { control: Rational(1, 2) }, fixed_shares: { b: Rational(1, 10) })
      whole = described_class.call(arms: %i[control], tallies: {}, status: "running", winner_arm: nil,
                                   rng: Random.new(7), draws: 100, arm_floors: { control: 1 })

      expect_weights(rational[:weights], { control: 0.5, a: 0.4, b: 0.1 })
      expect(rational[:weights][:b]).to be_a(Float)
      expect_weights(whole[:weights], { control: 1.0 })
    end

    it "SL1 lifts free previous shares that do not sum to 1 with floors alone (no fixed share), paused and shifting" do
      # arm_floors alone: the free arms share 1. Paused, previous 0.5 / 0.5 / 0.5 (sum 1.5) are lifted, each 0.5 / 1.5;
      # a stored rounding 0.2 / 0.4 / 0.401 (sum 1.001) is lifted, each divided by 1.001. Undeclared, 0.2.0 keeps a
      # pause's previous weights as they are. Shifting from 0.5 / 0.5 / 0.5, the step starts from the lifted 1/3 each
      # toward P(best) 0.13 / 0.646 / 0.224 (every one above its floor): a's move 0.646 - 1/3 is the largest, so lambda
      # is 0.15 / (0.646 - 1/3) and a lands on 1/3 + 0.15.
      floors = { control: 0.05 }
      paused = lambda do |previous, **options|
        described_class.call(arms: %i[control a b], tallies: {}, status: "paused", winner_arm: nil, rng: Random.new(7),
                             draws: 100, previous_weights: previous, **options)[:weights]
      end
      third = Rational(1, 3)

      expect_weights(paused.call({ control: 0.5, a: 0.5, b: 0.5 }, arm_floors: floors),
                     { control: third.to_f, a: third.to_f, b: third.to_f })
      expect_weights(paused.call({ control: 0.2, a: 0.4, b: 0.401 }, arm_floors: floors),
                     { control: 0.2 / 1.001, a: 0.4 / 1.001, b: 0.401 / 1.001 })
      expect(paused.call({ control: 0.5, a: 0.5, b: 0.5 })).to eq(control: 0.5, a: 0.5, b: 0.5)

      shifting = described_class.call(arms: %i[control a b], tallies: { control: tally(1_000, 100),
                                                                       a: tally(1_000, 112), b: tally(1_000, 104) },
                                      status: "running", winner_arm: nil, rng: Random.new(7), draws: 500,
                                      previous_weights: { control: 0.5, a: 0.5, b: 0.5 }, arm_floors: floors)
      p_best = { control: Rational(65, 500), a: Rational(323, 500), b: Rational(112, 500) }
      step = Rational(15, 100) / (p_best[:a] - third)

      expect(shifting).to include(phase: :shifting, winner: nil, leader: nil)
      expect_weights(shifting[:stats].transform_values { |stats| stats[:p_best] }, p_best.transform_values(&:to_f))
      expect_weights(shifting[:weights], p_best.transform_values { |share| (third + (step * (share - third))).to_f })
      expect_sum_one(shifting[:weights], "shifting")
    end

    it "SL15 reads each floor and fixed share by the one number check, its bound read on the Float" do
      # Rational(1, 10**400) is a positive real number whose Float is 0.0, outside (0, 1] and (0, 1); 10**400 is a
      # finite Integer whose Float is an infinity; a Rational just under 1 has the Float 1.0, outside (0, 1); a Complex
      # is no real number. Each is refused, as the valid call beside them answers.
      expect_weights(warm(arm_floors: { control: 0.5 }, fixed_shares: { b: 0.2 })[:weights],
                     { control: 0.5, a: 0.3, b: 0.2 })

      [ Rational(1, 10**400), 10**400, Complex(0.05, 0) ].each do |floor|
        expect { warm(arm_floors: { control: floor }) }.to raise_error(ArgumentError), floor.inspect
      end
      [ Rational(1, 10**400), Rational((10**400) - 1, 10**400), 10**400, Complex(0.2, 0) ].each do |share|
        expect { warm(fixed_shares: { b: share }) }.to raise_error(ArgumentError), share.inspect
      end
    end

    it "SL1 refuses a stored win of a fixed arm only while the status is won" do
      # Paused, the stored winner is not read: the equal split breaks control's fixed 0.2, so the free arms' equal
      # shares go through the lift, 0.4 each.
      tallies = { control: tally(400, 100), a: tally(400, 140), b: tally(400, 100) }
      paused = described_class.call(arms: %i[control a b], tallies: tallies, status: "paused", winner_arm: :control,
                                    rng: Random.new(5), draws: 100, fixed_shares: { control: 0.2 })

      expect(paused[:phase]).to eq(:paused)
      expect_weights(paused[:weights], { control: 0.2, a: 0.4, b: 0.4 })
    end
  end

  describe "a lone free arm" do
    # 17 SL1: the automatic win's leader is the first FREE arm by the one rule, and the won split over a single free
    # arm is 1.0 through the lift. A lone free arm's P(best) among the free arms is 1 and its expected loss 0, so past
    # warm-up it is the leader. 0.2.0's guard (a single arm is never the leader) holds only while no share is declared.
    it "SL1 is the leader beside a fixed arm: crowned with the default hold, stepping as the leader under a hold" do
      # Pinned above (lone_free_automatic_win). Without the fixed share a is crowned with 0.2.0's won split
      # 0.05 / 0.95. Held (hold_satisfied: false), a is still the leader and steps from 0.2 / 0.8 to the target,
      # 0.2 / 0.8: the same weights as the win, told apart by the phase and the winner.
      fixture = AllocatorSharesPins::FIXTURES.fetch(:lone_free_automatic_win)
      plain = call_fixture(fixture.except(:fixed_shares))
      crowned = call_fixture(fixture)
      held = call_fixture(fixture, hold_satisfied: false)

      expect(plain).to include(phase: :won, winner: :a, leader: :a)
      expect_weights(plain[:weights], { control: 0.05, a: 0.95 })
      expect(crowned).to include(phase: :won, winner: :a, leader: :a)
      expect_weights(crowned[:weights], { control: 0.2, a: 0.8 })
      expect(held).to include(phase: :shifting, winner: nil, leader: :a)
      expect_weights(held[:weights], { control: 0.2, a: 0.8 })
      expect(held[:stats][:a]).to include(p_best: 1.0, expected_loss: 0.0)
    end

    it "SL1 is the won arm of a stored win, judged among the free arms, and takes the rest, 1 - F" do
      # Pinned above (lone_free_stored_win): b, fixed at 0.25, converts best, yet a's P(best) among the free arms is
      # 1, so the result has not changed; a takes 1 - 0.3 - 0.25 = 0.45.
      result = call_fixture(AllocatorSharesPins::FIXTURES.fetch(:lone_free_stored_win))

      expect(result).to include(phase: :won, winner: :a, result_changed: false, leader: nil)
      expect_weights(result[:weights], { control: 0.3, a: 0.45, b: 0.25 })
    end

    it "SL1 keeps 0.2.0's two-arm guard only while no share is declared: one arm floored is the leader" do
      # Pinned above (single_arm_floored). With no keyword, or a keyword naming no arm, the one arm is never the
      # leader and steps to 1.0 (0.2.0, bit for bit); with its floor declared it is the leader and is crowned.
      fixture = AllocatorSharesPins::FIXTURES.fetch(:single_arm_floored)
      plain = call_fixture(fixture.except(:arm_floors))
      empty = call_fixture(fixture.except(:arm_floors), arm_floors: {})

      expect(plain).to include(phase: :shifting, winner: nil, leader: nil)
      expect_weights(plain[:weights], { control: 1.0 })
      expect(Marshal.dump(empty)).to eq(Marshal.dump(plain))
      expect(call_fixture(fixture)).to include(phase: :won, winner: :control, leader: :control)
    end
  end

  describe "a free share against its floor" do
    # On these tallies P(best) is 0 / 1 whatever the seed, so a's lead is held (hold_satisfied: false) and the target is
    # control at its floor, a the rest.
    def stepped(previous, **options)
      described_class.call(arms: %i[control a], tallies: { control: tally(2_000, 100), a: tally(2_000, 400) },
                           status: "running", winner_arm: nil, rng: Random.new(7), draws: AllocatorSharesPins::DRAWS,
                           previous_weights: previous, hold_satisfied: false, **options)
    end

    # P(best) 0 / 1 / 0 with every floor 0.05: the target is 0.05 / 0.9 / 0.05, a's move from 0.45 is the largest (0.45),
    # lambda 1/3. Float arithmetic in the library's order, so these compare bit for bit: what separates the two starts
    # (under 4e-14) lies inside the 1e-12 relative of the pins above.
    def stepped_three(previous, **options)
      arms = %i[control a b]
      described_class.call(arms: arms, tallies: { control: tally(2_000, 100), a: tally(2_000, 400), b: tally(2_000, 100) },
                           status: "running", winner_arm: nil, rng: Random.new(7), draws: AllocatorSharesPins::DRAWS,
                           previous_weights: previous, hold_satisfied: false,
                           config: described_class::Config.new(floor: 0.05), **options)[:weights]
    end

    def step_from(previous)
      arms = %i[control a b]
      target = AllocatorSharesPins.with_floor_020(arms, 0.05, { control: 0.0, a: 1.0, b: 0.0 })

      AllocatorSharesPins.step_020(arms, 0.15, previous, target)
    end

    def paused(previous, **options)
      described_class.call(arms: %i[control a b], tallies: {}, status: "paused", winner_arm: nil, rng: Random.new(7),
                           draws: 100, previous_weights: previous, **options)[:weights]
    end

    it "SL1 clamps a free share the step rounds under its floor; undeclared, the step's own rounding stays (SL14)" do
      # Floor 0.05 from 0.18: the move 0.13 is under max_step, lambda 1, and 0.18 + (0.05 - 0.18) rounds to
      # 0.04999999999999999.
      previous = { control: 0.18, a: 0.82 }
      config = described_class::Config.new(floor: 0.05)
      rounded = 0.18 + (1.0 * (0.05 - 0.18))
      declared = stepped(previous, config: config, arm_floors: { control: 0.05 })
      undeclared = stepped(previous, config: config)

      expect(rounded < 0.05).to be(true)
      expect(undeclared[:weights][:control]).to eql(rounded)
      expect(declared).to include(phase: :shifting, leader: :a)
      expect(declared[:weights][:control]).to eql(0.05)
      expect(declared[:weights][:a]).to eql(undeclared[:weights][:a])
      expect_sum_one(declared[:weights], "declared")
    end

    it "SL1 clamps beside a fixed share and answers a Float for a Rational Config#floor" do
      # With a fixed share: control's floor 0.05, c fixed at 0.1; control steps from 0.18 and a takes the rest. A
      # Rational Config#floor keeps a clamped share a Float, as the step's every other share is.
      fixed = described_class.call(arms: %i[control a c],
                                   tallies: { control: tally(2_000, 100), a: tally(2_000, 400), c: tally(2_000, 100) },
                                   status: "running", winner_arm: nil, rng: Random.new(7), draws: AllocatorSharesPins::DRAWS,
                                   previous_weights: { control: 0.18, a: 0.72, c: 0.1 }, hold_satisfied: false,
                                   arm_floors: { control: 0.05 }, fixed_shares: { c: 0.1 })
      rational = stepped({ control: 0.18, a: 0.82 }, config: described_class::Config.new(floor: Rational(1, 20)),
                         arm_floors: { a: 0.05 })

      expect(fixed[:weights][:control]).to eql(0.05)
      expect(fixed[:weights][:c]).to eql(0.1)
      expect_sum_one(fixed[:weights], "fixed")
      expect(rational[:weights][:control]).to eql(0.05)
    end

    it "SL1 reads a declared floor to 1e-12 in a pause: a share within it holds, one beyond it is lifted" do
      floors = { control: 0.05 }
      within = { control: 0.05 - 1e-13, a: 0.5, b: 0.45 + 1e-13 }
      beyond = { control: 0.05 - 2e-12, a: 0.5, b: 0.45 + 2e-12 }

      expect(Marshal.dump(paused(within, arm_floors: floors))).to eq(Marshal.dump(within))
      expect(paused(beyond, arm_floors: floors)[:control]).to eql(0.05)
      expect_sum_one(paused(beyond, arm_floors: floors), "beyond")
    end

    it "SL1 starts a declared step from a share within 1e-12 of its floor as it stands, and lifts one beyond it" do
      # Within: the step starts from the shares as they are and control, which it leaves still under, is clamped.
      # Beyond: control is pinned at 0.05 first and a and b share 0.95 in proportion (with every floor 0.05 the lift is
      # 0.2.0's), then the step.
      arms = %i[control a b]
      within = { control: 0.05 - 1e-13, a: 0.45, b: 0.5 + 1e-13 }
      beyond = { control: 0.05 - 2e-12, a: 0.45, b: 0.5 + 2e-12 }
      floors = { control: 0.05 }

      expect(Marshal.dump(stepped_three(within, arm_floors: floors)))
        .to eq(Marshal.dump(step_from(within).merge(control: 0.05)))
      expect(Marshal.dump(stepped_three(beyond, arm_floors: floors)))
        .to eq(Marshal.dump(step_from(AllocatorSharesPins.with_floor_020(arms, 0.05, beyond))))
    end

    it "SL14 keeps 0.2.0's strict floor undeclared: a share 1e-13 under it is lifted before the step and a pause keeps it" do
      arms = %i[control a b]
      near = { control: 0.05 - 1e-13, a: 0.45, b: 0.5 + 1e-13 }
      lifted = step_from(AllocatorSharesPins.with_floor_020(arms, 0.05, near))

      expect(Marshal.dump(stepped_three(near))).to eq(Marshal.dump(lifted))
      expect(Marshal.dump(stepped_three(near))).not_to eq(Marshal.dump(step_from(near)))
      expect(Marshal.dump(paused(near))).to eq(Marshal.dump(near))
    end
  end

  describe "the lift in public (Allocator.lift) and 0.2.0's private with_floor" do
    # A caller that holds shares and no tallies reaches 0.2.0's floor as an allocator built with no tallies, no rng
    # and one draw, through send; with_floor now hands its target, in the call's arm order, to Allocator.lift.
    def seam(target, config, arms: target.keys)
      described_class.new(arms: arms, tallies: {}, status: "running", winner_arm: nil, rng: nil, draws: 1,
                          config: config).send(:with_floor, target)
    end

    it "SL1 answers 0.2.0's with_floor step for step, Marshal-identical, on 3 000 random targets" do
      rng = Random.new(2_030)
      seen = Hash.new(0)

      3_000.times do |index|
        floor = AllocatorSharesPins::LIFT_FLOORS[rng.rand(AllocatorSharesPins::LIFT_FLOORS.size)]
        config = described_class::Config.new(floor: floor)
        target = AllocatorSharesPins.random_target(rng, config)
        reference = AllocatorSharesPins.with_floor_020(target.keys, floor, target)
        expected = Marshal.dump(reference)
        strings = target.transform_keys(&:to_s)

        expect(Marshal.dump(described_class.lift(target, config: config))).to eq(expected), "##{index} #{target}"
        expect(Marshal.dump(seam(target, config))).to eq(expected), "##{index} with_floor #{target}"
        expect(Marshal.dump(described_class.lift(strings, config: config)))
          .to eq(Marshal.dump(AllocatorSharesPins.with_floor_020(strings.keys, floor, strings))), "##{index} Strings"
        expect(Marshal.dump(described_class.lift(target))).to eq(expected), "##{index} default" if floor.eql?(0.10)

        pinned = reference.count { |_arm, share| share == floor }
        seen[:equal_split] += 1 unless target.values.sum.positive?
        seen[:none_pinned] += 1 if pinned.zero?
        seen[:one_pinned] += 1 if pinned == 1
        seen[:several_pinned] += 1 if pinned > 1
        seen[:over_full] += 1 if target.size * floor > 1
      end

      expect(seen.values_at(:equal_split, :none_pinned, :one_pinned, :several_pinned, :over_full)).to all(be >= 50)
    end

    it "SL1 lifts a held won split onto the floor, keeps the keys as given and reads exact shares as their Floats" do
      # 0.99 / 0.01: a is pinned at 0.10 and control takes 0.9 * 0.99 / 0.99 = 0.9.
      expect_weights(described_class.lift({ control: 0.99, a: 0.01 }), { control: 0.9, a: 0.1 })
      expect_weights(seam({ control: 0.99, a: 0.01 }, described_class::Config.new), { control: 0.9, a: 0.1 })
      expect_weights(described_class.lift({ "control" => 0.99, a: 0.01 }), { "control" => 0.9, a: 0.1 })

      [ { control: 1, a: 0 }, { control: Rational(99, 100), a: Rational(1, 100) } ].each do |exact|
        floats = exact.transform_values(&:to_f)
        expect(Marshal.dump(described_class.lift(exact))).to eq(Marshal.dump(described_class.lift(floats))), exact.inspect
      end
    end

    it "SL1 with_floor reads the call's arms in their order, as 0.2.0's did: an extra key ignored, a missing one a KeyError" do
      config = described_class::Config.new
      target = { a: 0.01, control: 0.99, z: "not read" }
      lifted = seam(target, config, arms: %i[control a])

      expect(Marshal.dump(lifted))
        .to eq(Marshal.dump(AllocatorSharesPins.with_floor_020(%i[control a], config.floor, target)))
      expect(lifted.keys).to eq(%i[control a])
      expect { seam({ control: 0.5 }, config, arms: %i[control a]) }.to raise_error(KeyError)
    end

    it "SL15 refuses weights, a config or a share the one number check refuses, through lift and with_floor alike" do
      expect_weights(described_class.lift({ control: 0.5, a: 0.5 }), { control: 0.5, a: 0.5 })

      [ nil, [ [ :control, 0.5 ] ], {}, (1..11).to_h { |index| [ :"arm#{index}", 0.1 ] } ].each do |weights|
        expect { described_class.lift(weights) }.to raise_error(ArgumentError), weights.inspect
      end
      expect { described_class.lift({ control: 0.5 }, config: :defaults) }.to raise_error(ArgumentError)
      [ nil, "0.5", Float::NAN, Float::INFINITY, -Float::INFINITY, Complex(0.5, 0), 10**400, Rational(10**400, 3) ]
        .each do |share|
          target = { control: 0.5, a: share }
          expect { described_class.lift(target) }.to raise_error(ArgumentError), share.inspect
          expect { seam(target, described_class::Config.new) }.to raise_error(ArgumentError), share.inspect
        end
    end
  end
end
