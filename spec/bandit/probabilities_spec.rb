# frozen_string_literal: true

# LO3: the acceptance of the posterior set's odds (01 GB11, GB10, GB18), written from the contract before their
# implementers. Each example's name says which task turns it green: LC4 (the pick's exact odds, up to EXACT_PICK_MAX
# contenders, and Posteriors.mixture) or LD1 (every order's and position's probability, Choice#probability,
# Order#probability, and the Monte Carlo every odds call falls back to). Until then each fails on a missing constant
# or method.
#
# THE RULE (01 GB9), for the free items (the unpinned ones): contenders C are the free items that may lead (all when
# may_lead is nil), eligible E the contenders with at least min_views views. A branch value c below shuffle shuffles
# (the contender with the smallest priority leads, the rest follow by priority); below shuffle + explore, with E not
# empty, it explores (the eligible item with the smallest priority leads, the rest follow by draw); otherwise the
# contender with the largest draw leads and the rest follow by draw. Pinned items keep their positions. So, without
# items barred from leading, an order s1, ..., sm of the free items has probability
#
#   shuffle * [s1 in C] / |C| / ((n - 1) ... (n - m + 1))
#   + explore * [E not empty] * [s1 in E] / |E| * T(s2, ..., sm among the free items less s1)
#   + (1 - shuffle - explore * [E not empty]) * T(s1, ..., sm among the free items),
#
# T(order among S) = P(X_s1 > ... > X_sm > every other item of S), and the head c of a pick has probability
# shuffle / |C| [c in C] + explore / |E| [c in E, E not empty] + (1 - shuffle - explore [E not empty]) P(c is C's
# largest draw).
#
# THE REFERENCES, none of them from the code under test:
# - the pick's exact odds by that formula over BetaArgmax.probabilities (0.3.0, which its pins hold exact to 1e-8) or,
#   for Normals, Simpson's rule on 400 000 intervals (error under 2e-13), within 1e-12 of the formula and 1e-8 of the
#   integral;
# - orders and positions for K = 4, 5 and 6 from an independent program (its own distribution functions, in another
#   language): T by a Stieltjes sum against exact distribution increments on a grid of every item's quantiles at 3 000
#   evenly spaced levels, positions by summing every order the rule can answer. Checked on closed forms first: Beta(a,
#   1) members race as Plackett-Luce with weights a (-log X is exponential with rate a), every order within 6e-9;
#   Beta(1, b) members as the reverse race, within 1.4e-8; its heads within 1.4e-8 of BetaArgmax.probabilities; its
#   branch formulas within 1.3 standard errors of 600 000 simulated visits of the rule itself;
# - K = 3 by hand: Beta(1, 1), Beta(2, 1) and Beta(3, 1) race as Plackett-Luce with weights 1, 2 and 3, worked in
#   fractions; under may_lead the hand orders are P(X_a > X_b > X_c) = 1/15, P(b is the smallest) = 4/15, P(X_b > X_a >
#   X_c) = 1/12 and P(a is the smallest) = 7/12.
# Tolerances: 1e-4 for every order and position (01 Decisions), 1e-12 for the pick's odds against their formula, 4
# standard errors for a Monte Carlo answer. Every exact odds call refuses a parameter below
# BetaArgmax::SMALLEST_PARAMETER with an ArgumentError (01 GB6).
module OddsOracle
  BETA4 = { "a" => [ 0.3, 20.0 ], "b" => [ 1.0, 20.0 ], "c" => [ 2.0, 30.0 ], "d" => [ 5.0, 90.0 ] }.freeze
  BETA5 = { "a" => [ 0.02, 19.98 ], "b" => [ 3.0, 60.0 ], "c" => [ 4.0, 70.0 ], "d" => [ 6.0, 95.0 ],
            "e" => [ 2.0, 45.0 ] }.freeze
  BETA6 = { "a" => [ 12.0, 88.0 ], "b" => [ 15.0, 85.0 ], "c" => [ 9.0, 91.0 ], "d" => [ 20.0, 180.0 ],
            "e" => [ 3.0, 17.0 ], "f" => [ 0.5, 4.5 ] }.freeze
  NORMAL4 = { "a" => [ 5.0, 0.4 ], "b" => [ 5.3, 0.5 ], "c" => [ 4.8, 0.3 ], "d" => [ 5.1, 0.8 ] }.freeze
  NORMAL5 = { "a" => [ 1.0, 1.0 ], "b" => [ 1.5, 0.5 ], "c" => [ 0.5, 2.0 ], "d" => [ 1.2, 0.2 ],
              "e" => [ 0.8, 0.7 ] }.freeze
  NORMAL6 = { "a" => [ 5.0, 0.3 ], "b" => [ 5.2, 0.6 ], "c" => [ 4.9, 0.2 ], "d" => [ 5.4, 1.0 ], "e" => [ 4.6, 0.5 ],
              "f" => [ 5.1, 0.4 ] }.freeze
  VIEWS4 = { "a" => 50.0, "b" => 400.0, "c" => 400.0, "d" => 400.0 }.freeze
  # A parameter below BetaArgmax::SMALLEST_PARAMETER (1e-6): an alpha, then a beta.
  BELOW_SMALLEST = [ { "a" => [ 2.0, 3.0 ], "b" => [ 1e-9, 1.0 ], "c" => [ 3.0, 2.0 ] }.freeze,
                     { "a" => [ 2.0, 3.0 ], "b" => [ 1.0, 5e-7 ], "c" => [ 3.0, 2.0 ] }.freeze ].freeze
  # P(largest) of NORMAL4's members, by Simpson's rule.
  NORMAL4_LARGEST = [ 0.16748793058298564, 0.43470920285171183, 0.05189145526742245, 0.34591141129780234 ].freeze

  # Beta(1, 1), Beta(2, 1) and Beta(3, 1): Plackett-Luce with weights 1, 2 and 3.
  HAND = { "a" => [ 1.0, 1.0 ], "b" => [ 2.0, 1.0 ], "c" => [ 3.0, 1.0 ] }.freeze
  HAND_ORDERS = {
    %w[c b a] => 1 / 3r, %w[c a b] => 1 / 6r, %w[b c a] => 1 / 4r, %w[b a c] => 1 / 12r, %w[a c b] => 1 / 10r,
    %w[a b c] => 1 / 15r
  }.freeze
  # rule, views, P(item at position)
  HAND_POSITIONS = {
    "by draw alone" => [
      {}, nil,
      { "a" => [ 1 / 6r, 1 / 4r, 7 / 12r ], "b" => [ 1 / 3r, 2 / 5r, 4 / 15r ], "c" => [ 1 / 2r, 7 / 20r, 3 / 20r ] }
    ],
    "explore 0.3 (the explored part: 1/3 at the head, then the draw among the other two)" => [
      { explore: 0.3 }, nil,
      { "a" => [ 13 / 60r, 7 / 30r, 11 / 20r ], "b" => [ 1 / 3r, 29 / 75r, 7 / 25r ],
        "c" => [ 9 / 20r, 19 / 50r, 17 / 100r ] }
    ],
    "shuffle 0.3 (the shuffled part: 1/3 at every position)" => [
      { shuffle: 0.3 }, nil,
      { "a" => [ 13 / 60r, 11 / 40r, 61 / 120r ], "b" => [ 1 / 3r, 19 / 50r, 43 / 150r ],
        "c" => [ 9 / 20r, 69 / 200r, 41 / 200r ] }
    ],
    "shuffle 0.1 and explore 0.2" => [
      { shuffle: 0.1, explore: 0.2 }, nil,
      { "a" => [ 13 / 60r, 89 / 360r, 193 / 360r ], "b" => [ 1 / 3r, 173 / 450r, 127 / 450r ],
        "c" => [ 9 / 20r, 221 / 600r, 109 / 600r ] }
    ],
    "explore 0.3 among items of at least 100 views (b and c)" => [
      { explore: 0.3, min_views: 100.0 }, { "a" => 10.0, "b" => 500.0, "c" => 500.0 },
      { "a" => [ 7 / 60r, 21 / 80r, 149 / 240r ], "b" => [ 23 / 60r, 19 / 50r, 71 / 300r ],
        "c" => [ 1 / 2r, 143 / 400r, 57 / 400r ] }
    ],
    "a pinned at position 0, explore 0.2 (c above b: 0.2 / 2 + 0.8 * 3/5)" => [
      { explore: 0.2, pinned: { 0 => "a" } }, nil,
      { "a" => [ 1r, 0r, 0r ], "b" => [ 0r, 21 / 50r, 29 / 50r ], "c" => [ 0r, 29 / 50r, 21 / 50r ] }
    ]
  }.freeze

  # label => the set, its rule, and the reference: P(item at position) and some orders' probabilities (every order
  # where there are at most 30).
  CASES = {
    "four Betas, an alpha of 0.3 among them, by draw alone" => {
      kind: :beta, pairs: BETA4, rule: {},
      positions: {
        "a" => [ 0.04587101, 0.05959134, 0.13763406, 0.75690362 ],
        "b" => [ 0.24621612, 0.21294889, 0.38418117, 0.15665384 ],
        "c" => [ 0.40647606, 0.30517636, 0.23461907, 0.05372853 ],
        "d" => [ 0.30143683, 0.42228343, 0.24356573, 0.03271403 ]
      },
      orders: {
        %w[a b c d] => 0.00347512, %w[a b d c] => 0.00542515, %w[a c b d] => 0.00399784,
        %w[a c d b] => 0.01250486, %w[a d b c] => 0.00692634, %w[a d c b] => 0.01354170,
        %w[b a c d] => 0.00475466, %w[b a d c] => 0.00756593, %w[b c a d] => 0.00733640,
        %w[b c d a] => 0.09780961, %w[b d a c] => 0.01488253, %w[b d c a] => 0.11386699,
        %w[c a b d] => 0.00566593, %w[c a d b] => 0.01861573, %w[c b a d] => 0.00748408,
        %w[c b d a] => 0.10164445, %w[c d a b] => 0.05758743, %w[c d b a] => 0.21547844,
        %w[d a b c] => 0.00774307, %w[d a c b] => 0.01524601, %w[d b a c] => 0.01118551,
        %w[d b c a] => 0.08373457, %w[d c a b] => 0.03915810, %w[d c b a] => 0.14436955
      }
    },
    "four Betas, explore 0.1" => {
      kind: :beta, pairs: BETA4, rule: { explore: 0.1 },
      positions: {
        "a" => [ 0.06628391, 0.05856231, 0.13373192, 0.74142188 ],
        "b" => [ 0.24659451, 0.21544393, 0.37561955, 0.16234202 ],
        "c" => [ 0.39082846, 0.31277384, 0.23814693, 0.05825079 ],
        "d" => [ 0.29629315, 0.41321994, 0.25250161, 0.03798533 ]
      },
      orders: {
        %w[a b c d] => 0.00596201, %w[a b d c] => 0.00842615, %w[a c b d] => 0.00656786,
        %w[a c d b] => 0.01885903, %w[a d b c] => 0.00897344, %w[a d c b] => 0.01749542,
        %w[b a c d] => 0.00489751, %w[b a d c] => 0.00764581, %w[b c a d] => 0.00758031,
        %w[b c d a] => 0.09984165, %w[b d a c] => 0.01462071, %w[b d c a] => 0.11200852,
        %w[c a b d] => 0.00556344, %w[c a d b] => 0.01804387, %w[c b a d] => 0.00741420,
        %w[c b d a] => 0.09968510, %w[c d a b] => 0.05482205, %w[c d b a] => 0.20529980,
        %w[d a b c] => 0.00755801, %w[d a c b] => 0.01485367, %w[d b a c] => 0.01102668,
        %w[d b c a] => 0.08292981, %w[d c a b] => 0.03826797, %w[d c b a] => 0.14165701
      }
    },
    "four Betas, shuffle 0.1" => {
      kind: :beta, pairs: BETA4, rule: { shuffle: 0.1 },
      positions: {
        "a" => [ 0.06628391, 0.07863220, 0.14887066, 0.70621326 ],
        "b" => [ 0.24659451, 0.21665400, 0.37076305, 0.16598846 ],
        "c" => [ 0.39082846, 0.29965873, 0.23615716, 0.07335568 ],
        "d" => [ 0.29629315, 0.40505509, 0.24420916, 0.05444263 ]
      },
      orders: {
        %w[a b c d] => 0.00729428, %w[a b d c] => 0.00904930, %w[a c b d] => 0.00776472,
        %w[a c d b] => 0.01542104, %w[a d b c] => 0.01040037, %w[a d c b] => 0.01635420,
        %w[b a c d] => 0.00844586, %w[b a d c] => 0.01097600, %w[b c a d] => 0.01076943,
        %w[b c d a] => 0.09219532, %w[b d a c] => 0.01756095, %w[b d c a] => 0.10664696,
        %w[c a b d] => 0.00926600, %w[c a d b] => 0.02092082, %w[c b a d] => 0.01090234,
        %w[c b d a] => 0.09564667, %w[c d a b] => 0.05599536, %w[c d b a] => 0.19809726,
        %w[d a b c] => 0.01113543, %w[d a c b] => 0.01788808, %w[d b a c] => 0.01423363,
        %w[d b c a] => 0.07952778, %w[d c a b] => 0.03940896, %w[d c b a] => 0.13409927
      }
    },
    "four Betas, shuffle 0.1 and explore 0.2" => {
      kind: :beta, pairs: BETA4, rule: { shuffle: 0.1, explore: 0.2 },
      positions: {
        "a" => [ 0.10710971, 0.07657415, 0.14106638, 0.67524978 ],
        "b" => [ 0.24735129, 0.22164409, 0.35363982, 0.17736482 ],
        "c" => [ 0.35953324, 0.31485368, 0.24321289, 0.08240020 ],
        "d" => [ 0.28600578, 0.38692810, 0.26208092, 0.06498521 ]
      },
      orders: {
        %w[a b c d] => 0.01226804, %w[a b d c] => 0.01505130, %w[a c b d] => 0.01290477,
        %w[a c d b] => 0.02812939, %w[a d b c] => 0.01449458, %w[a d c b] => 0.02426163,
        %w[b a c d] => 0.00873155, %w[b a d c] => 0.01113577, %w[b c a d] => 0.01125725,
        %w[b c d a] => 0.09625939, %w[b d a c] => 0.01703730, %w[b d c a] => 0.10293002,
        %w[c a b d] => 0.00906102, %w[c a d b] => 0.01977711, %w[c b a d] => 0.01076258,
        %w[c b d a] => 0.09172796, %w[c d a b] => 0.05046460, %w[c d b a] => 0.17773998,
        %w[d a b c] => 0.01076530, %w[d a c b] => 0.01710340, %w[d b a c] => 0.01391596,
        %w[d b c a] => 0.07791825, %w[d c a b] => 0.03762870, %w[d c b a] => 0.12867418
      }
    },
    "four Betas, explore 0.1, the top 2" => {
      kind: :beta, pairs: BETA4, rule: { explore: 0.1, top: 2 },
      orders: {
        %w[a b] => 0.01438815, %w[a c] => 0.02542689, %w[a d] => 0.02646886,
        %w[b a] => 0.01254332, %w[b c] => 0.10742196, %w[b d] => 0.12662923,
        %w[c a] => 0.02360731, %w[c b] => 0.10709930, %w[c d] => 0.26012185,
        %w[d a] => 0.02241168, %w[d b] => 0.09395648, %w[d c] => 0.17992498
      }
    },
    "four Betas, b pinned at position 1, shuffle 0.05 and explore 0.1" => {
      kind: :beta, pairs: BETA4, rule: { shuffle: 0.05, explore: 0.1, pinned: { 1 => "b" } },
      positions: {
        "a" => [ 0.09946285, 0.00000000, 0.09842013, 0.80211702 ],
        "c" => [ 0.48487876, 0.00000000, 0.40928596, 0.10583528 ],
        "d" => [ 0.41565839, 0.00000000, 0.49229391, 0.09204770 ],
        "b" => [ 0.00000000, 1.00000000, 0.00000000, 0.00000000 ]
      },
      orders: {
        %w[a b c d] => 0.04723443, %w[a b d c] => 0.05222843, %w[c b a d] => 0.04481328,
        %w[c b d a] => 0.44006548, %w[d b a c] => 0.05360685, %w[d b c a] => 0.36205154
      }
    },
    "four Betas with views, explore 0.2 among items of at least 100 views" => {
      kind: :beta, pairs: BETA4, views: VIEWS4, rule: { explore: 0.2, min_views: 100.0 },
      positions: {
        "a" => [ 0.03669681, 0.06082003, 0.13640397, 0.76607922 ],
        "b" => [ 0.26363957, 0.21679117, 0.37173682, 0.14783246 ],
        "c" => [ 0.39184752, 0.31758281, 0.23795523, 0.05261446 ],
        "d" => [ 0.30781613, 0.40480601, 0.25390400, 0.03347388 ]
      },
      orders: {
        %w[a b c d] => 0.00278010, %w[a b d c] => 0.00434012, %w[a c b d] => 0.00319827,
        %w[a c d b] => 0.01000389, %w[a d b c] => 0.00554107, %w[a d c b] => 0.01083336,
        %w[b a c d] => 0.00545256, %w[b a d c] => 0.00828335, %w[b c a d] => 0.00847593,
        %w[b c d a] => 0.10974902, %w[b d a c] => 0.01517650, %w[b d c a] => 0.11650220,
        %w[c a b d] => 0.00577035, %w[c a d b] => 0.01833183, %w[c b a d] => 0.00779667,
        %w[c b d a] => 0.10319580, %w[c d a b] => 0.05405225, %w[c d b a] => 0.20270062,
        %w[d a b c] => 0.00776577, %w[d a c b] => 0.01521617, %w[d b a c] => 0.01150765,
        %w[d b c a] => 0.08717083, %w[d c a b] => 0.03939496, %w[d c b a] => 0.14676075
      }
    },
    "five Betas, an alpha of 0.02 among them, by draw alone" => {
      kind: :beta, pairs: BETA5, rule: {},
      positions: {
        "a" => [ 0.00218671, 0.00201854, 0.00302250, 0.00700490, 0.98576739 ],
        "b" => [ 0.20050332, 0.22628839, 0.28199096, 0.28728002, 0.00393734 ],
        "c" => [ 0.27401167, 0.28008749, 0.26251653, 0.18129337, 0.00209098 ],
        "d" => [ 0.35679521, 0.31975500, 0.22123580, 0.10128955, 0.00092448 ],
        "e" => [ 0.16650313, 0.17185062, 0.23123425, 0.42313219, 0.00727985 ]
      },
      orders: {
        %w[d c b e a] => 0.09289049, %w[c d b e a] => 0.08559476, %w[d b c e a] => 0.07530637,
        %w[d b a e c] => 0.00012051, %w[e a d b c] => 0.00005635, %w[b a e c d] => 0.00002501
      }
    },
    "five Betas, shuffle 0.1 and explore 0.1" => {
      kind: :beta, pairs: BETA5, rule: { shuffle: 0.1, explore: 0.1 },
      positions: {
        "a" => [ 0.04174937, 0.02183014, 0.02266002, 0.02614511, 0.88761540 ],
        "b" => [ 0.20040266, 0.22159675, 0.27044971, 0.27834046, 0.02921046 ],
        "c" => [ 0.25920934, 0.27159267, 0.25731914, 0.18641296, 0.02546593 ],
        "d" => [ 0.32543617, 0.31074271, 0.22502337, 0.11595845, 0.02283933 ],
        "e" => [ 0.17320250, 0.17423776, 0.22454780, 0.39314306, 0.03486892 ]
      },
      orders: {
        %w[d c b e a] => 0.08016789, %w[c d b e a] => 0.07514698, %w[d b c e a] => 0.06519724,
        %w[d c a e b] => 0.00104635, %w[e a d c b] => 0.00090851, %w[b a e c d] => 0.00085709
      }
    },
    "five Betas, explore 0.1, the top 2" => {
      kind: :beta, pairs: BETA5, rule: { explore: 0.1, top: 2 },
      orders: {
        %w[a b] => 0.00440577, %w[a c] => 0.00602908, %w[a d] => 0.00790347,
        %w[a e] => 0.00362971, %w[b a] => 0.00037814, %w[b c] => 0.06765434,
        %w[b d] => 0.09631654, %w[b e] => 0.03610397, %w[c a] => 0.00054189,
        %w[c b] => 0.07146045, %w[c d] => 0.14210691, %w[c e] => 0.05250125,
        %w[d a] => 0.00079977, %w[d b] => 0.10933837, %w[d c] => 0.15178966,
        %w[d e] => 0.07918788, %w[e a] => 0.00031219, %w[e b] => 0.03902099,
        %w[e c] => 0.05412835, %w[e d] => 0.07639129
      }
    },
    "six Betas by draw alone" => {
      kind: :beta, pairs: BETA6, rule: {},
      positions: {
        "a" => [ 0.10108376, 0.22654153, 0.27691083, 0.21511242, 0.13518356, 0.04516797 ],
        "b" => [ 0.32292882, 0.35813301, 0.19475506, 0.08373153, 0.03276247, 0.00768917 ],
        "c" => [ 0.01960659, 0.06626844, 0.13574352, 0.22390578, 0.35650715, 0.19796859 ],
        "d" => [ 0.01774808, 0.08699132, 0.21459133, 0.32146030, 0.26886268, 0.09034635 ],
        "e" => [ 0.34512683, 0.19358013, 0.12735528, 0.10725053, 0.13687930, 0.08980800 ],
        "f" => [ 0.19350599, 0.06848564, 0.05064404, 0.04853950, 0.06980489, 0.56901999 ]
      },
      orders: {
        %w[e b a d c f] => 0.05176277, %w[e b d a c f] => 0.02926976, %w[e b a c d f] => 0.02822048,
        %w[a e f b c d] => 0.00026006, %w[d f b e a c] => 0.00005505, %w[d c f e a b] => 0.00000269
      }
    },
    "six Betas, shuffle 0.1" => {
      kind: :beta, pairs: BETA6, rule: { shuffle: 0.1 },
      positions: {
        "a" => [ 0.10764205, 0.22055404, 0.26588641, 0.21026785, 0.13833187, 0.05731784 ],
        "b" => [ 0.30730260, 0.33898637, 0.19194622, 0.09202504, 0.04615289, 0.02358692 ],
        "c" => [ 0.03431260, 0.07630826, 0.13883583, 0.21818187, 0.33752310, 0.19483840 ],
        "d" => [ 0.03263994, 0.09495885, 0.20979886, 0.30598094, 0.25864308, 0.09797838 ],
        "e" => [ 0.32728081, 0.19088878, 0.13128642, 0.11319215, 0.13985804, 0.09749386 ],
        "f" => [ 0.19082206, 0.07830375, 0.06224630, 0.06035221, 0.07949107, 0.52878466 ]
      },
      orders: {
        %w[e b a d c f] => 0.04672538, %w[e b d a c f] => 0.02648167, %w[e b a c d f] => 0.02553733,
        %w[a e f b c d] => 0.00037294, %w[d f b e a c] => 0.00018843, %w[d c f e a b] => 0.00014131
      }
    },
    "four Normals near 5 a view, by draw alone" => {
      kind: :normal, pairs: NORMAL4, rule: {},
      positions: {
        "a" => [ 0.16748794, 0.30831029, 0.31153295, 0.21266885 ],
        "b" => [ 0.43470922, 0.30062790, 0.16327074, 0.10139218 ],
        "c" => [ 0.05189146, 0.18988747, 0.36746310, 0.39075800 ],
        "d" => [ 0.34591141, 0.20117437, 0.15773325, 0.29518100 ]
      },
      orders: {
        %w[a b c d] => 0.05423219, %w[a b d c] => 0.03241828, %w[a c b d] => 0.02678893,
        %w[a c d b] => 0.01152603, %w[a d b c] => 0.02580887, %w[a d c b] => 0.01671365,
        %w[b a c d] => 0.11549931, %w[b a d c] => 0.06829650, %w[b c a d] => 0.07014699,
        %w[b c d a] => 0.03266296, %w[b d a c] => 0.08549592, %w[b d c a] => 0.06260754,
        %w[c a b d] => 0.01289665, %w[c a d b] => 0.00554533, %w[c b a d] => 0.01561693,
        %w[c b d a] => 0.00728415, %w[c d a b] => 0.00498351, %w[c d b a] => 0.00556489,
        %w[d a b c] => 0.06616885, %w[d a c b] => 0.03990365, %w[d b a c] => 0.11256959,
        %w[d b c a] => 0.07850677, %w[d c a b] => 0.02272001, %w[d c b a] => 0.02604254
      }
    },
    "four Normals, explore 0.15" => {
      kind: :normal, pairs: NORMAL4, rule: { explore: 0.15 },
      positions: {
        "a" => [ 0.17986475, 0.29246778, 0.31129125, 0.21637625 ],
        "b" => [ 0.40700283, 0.31571204, 0.17357252, 0.10371263 ],
        "c" => [ 0.08160774, 0.17436292, 0.35414493, 0.38988444 ],
        "d" => [ 0.33152470, 0.21745729, 0.16099133, 0.29002671 ]
      },
      orders: {
        %w[a b c d] => 0.05631767, %w[a b d c] => 0.03688622, %w[a c b d] => 0.02511759,
        %w[a c d b] => 0.01083287, %w[a d b c] => 0.03255207, %w[a d c b] => 0.01815834,
        %w[b a c d] => 0.10597616, %w[b a d c] => 0.06342342, %w[b c a d] => 0.06353266,
        %w[b c d a] => 0.02965710, %w[b d a c] => 0.08407671, %w[b d c a] => 0.06033679,
        %w[c a b d] => 0.01569976, %w[c a d b] => 0.00694830, %w[c b a d] => 0.02338288,
        %w[c b d a] => 0.01324342, %w[c d a b] => 0.00925258, %w[c d b a] => 0.01308080,
        %w[d a b c] => 0.06294208, %w[d a c b] => 0.03747806, %w[d b a c] => 0.11000395,
        %w[d b c a] => 0.07587791, %w[d c a b] => 0.02104247, %w[d c b a] => 0.02418023
      }
    },
    "four Normals, a pinned at position 2, shuffle 0.1" => {
      kind: :normal, pairs: NORMAL4, rule: { shuffle: 0.1, pinned: { 2 => "a" } },
      positions: {
        "b" => [ 0.50255704, 0.34440998, 0.00000000, 0.15303299 ],
        "c" => [ 0.11451911, 0.37346250, 0.00000000, 0.51201839 ],
        "d" => [ 0.38292386, 0.28212752, 0.00000000, 0.33494863 ],
        "a" => [ 0.00000000, 0.00000000, 1.00000000, 0.00000000 ]
      },
      orders: {
        %w[b c a d] => 0.26195397, %w[b d a c] => 0.24060307, %w[c b a d] => 0.07299466,
        %w[c d a b] => 0.04152446, %w[d b a c] => 0.27141533, %w[d c a b] => 0.11150853
      }
    },
    "five Normals of mixed widths, shuffle 0.05 and explore 0.1" => {
      kind: :normal, pairs: NORMAL5, rule: { shuffle: 0.05, explore: 0.1 },
      positions: {
        "a" => [ 0.21027275, 0.16653193, 0.16416845, 0.24285414, 0.21617276 ],
        "b" => [ 0.32885111, 0.31823492, 0.20174068, 0.11100565, 0.04016768 ],
        "c" => [ 0.24962128, 0.09953881, 0.07786105, 0.12209017, 0.45088873 ],
        "d" => [ 0.10520989, 0.28426931, 0.35819101, 0.20225572, 0.05007410 ],
        "e" => [ 0.10604501, 0.13142507, 0.19803884, 0.32179435, 0.24269676 ]
      },
      orders: {
        %w[b d e a c] => 0.05580165, %w[a b d e c] => 0.05575134, %w[b d a e c] => 0.04955825,
        %w[d c b e a] => 0.00398155, %w[e c a b d] => 0.00159697, %w[d e c a b] => 0.00065611
      }
    },
    "five Normals by draw alone, the top 2" => {
      kind: :normal, pairs: NORMAL5, rule: { top: 2 },
      orders: {
        %w[a b] => 0.10541168, %w[a c] => 0.02613658, %w[a d] => 0.05355880,
        %w[a e] => 0.02697853, %w[b a] => 0.07397935, %w[b c] => 0.03908918,
        %w[b d] => 0.18060941, %w[b e] => 0.05791158, %w[c a] => 0.05791636,
        %w[c b] => 0.12354355, %w[c d] => 0.04552707, %w[c e] => 0.03139099,
        %w[d a] => 0.01445671, %w[d b] => 0.05277248, %w[d c] => 0.00619172,
        %w[d e] => 0.01506131, %w[e a] => 0.01406721, %w[e b] => 0.04023522,
        %w[e c] => 0.00774180, %w[e d] => 0.02742050
      }
    },
    "six Normals by draw alone" => {
      kind: :normal, pairs: NORMAL6, rule: {},
      positions: {
        "a" => [ 0.07363305, 0.19190176, 0.24777907, 0.23056684, 0.17699208, 0.07912728 ],
        "b" => [ 0.26366908, 0.24516227, 0.13905878, 0.11258889, 0.12411078, 0.11541028 ],
        "c" => [ 0.02281625, 0.10299288, 0.22321442, 0.30374223, 0.25579377, 0.09144052 ],
        "d" => [ 0.45613171, 0.13457165, 0.07742377, 0.06692785, 0.09280153, 0.17214356 ],
        "e" => [ 0.03475400, 0.07382798, 0.09578520, 0.12151425, 0.21138166, 0.46273699 ],
        "f" => [ 0.14899600, 0.25154354, 0.21673883, 0.16466001, 0.13892025, 0.07914144 ]
      },
      orders: {
        %w[d b f a c e] => 0.02499965, %w[d b f c a e] => 0.02124700, %w[d b a f c e] => 0.01881571,
        %w[b c d a e f] => 0.00053730, %w[c d f b a e] => 0.00022231, %w[c e d b f a] => 0.00002376
      }
    },
    "six Normals, shuffle 0.1 and explore 0.1, the top 2" => {
      kind: :normal, pairs: NORMAL6, rule: { shuffle: 0.1, explore: 0.1, top: 2 },
      orders: {
        %w[a b] => 0.02274188, %w[a c] => 0.01516947, %w[a d] => 0.02172584,
        %w[a e] => 0.00890216, %w[a f] => 0.02370041, %w[b a] => 0.05107275,
        %w[b c] => 0.02655430, %w[b d] => 0.07456067, %w[b e] => 0.02064791,
        %w[b f] => 0.07143295, %w[c a] => 0.01007497, %w[c b] => 0.01156947,
        %w[c d] => 0.01343516, %w[c e] => 0.00547936, %w[c f] => 0.01102738,
        %w[d a] => 0.07293178, %w[d b] => 0.14702992, %w[d c] => 0.03208938,
        %w[d e] => 0.03084835, %w[d f] => 0.11533925, %w[e a] => 0.01031205,
        %w[e b] => 0.01451820, %w[e c] => 0.00700520, %w[e d] => 0.01629085,
        %w[e f] => 0.01301022, %w[f a] => 0.03513097, %w[f b] => 0.04299546,
        %w[f c] => 0.02186051, %w[f d] => 0.03856530, %w[f e] => 0.01397790
      }
    }
  }.freeze

  module_function

  # Every order the rule can answer: the free items' permutations (all of them, or the first `top`), each pinned item
  # at its own position.
  def orders(items, rule)
    pinned = rule.fetch(:pinned, {})
    free = items - pinned.values
    slots = (0...items.size).reject { |slot| pinned.key?(slot) }
    free.permutation(rule[:top] || free.size).map do |ranks|
      next ranks if pinned.empty?

      full = Array.new(items.size)
      pinned.each { |slot, item| full[slot] = item }
      ranks.each_with_index { |item, rank| full[slots[rank]] = item }
      full
    end
  end

  def factorial(n) = (1..n).reduce(1, :*)
end

RSpec.describe "SkiftetStatistical::Bandit::Posteriors odds" do
  def posteriors = SkiftetStatistical::Bandit::Posteriors
  def argmax = SkiftetStatistical::BetaArgmax
  def oracle = OddsOracle

  def build(kind, pairs, views = nil)
    kind == :beta ? posteriors.beta(pairs, views: views) : posteriors.normal(pairs, views: views)
  end

  # The pick's head odds by the formula in the header. A may_lead naming no item lets every item contend, as nil does
  # (01 GB9; LO2's reference reads it so): the L3 barrier's correction of this helper, for LO3.
  def heads_by_formula(pairs, shuffle: 0.0, explore: 0.0, may_lead: nil, eligible: nil)
    contenders = may_lead.nil? || may_lead.empty? ? pairs.keys : may_lead
    eligible = contenders & (eligible || contenders)
    largest = argmax.probabilities(contenders.map { |item| pairs.fetch(item) })
    share = eligible.empty? ? 0.0 : explore
    pairs.keys.to_h do |item|
      index = contenders.index(item)
      next [ item, 0.0 ] unless index

      [ item, (shuffle / contenders.size) + (eligible.include?(item) ? share / eligible.size : 0.0) +
              ((1.0 - shuffle - share) * largest[index]) ]
    end
  end

  def expect_heads(actual, expected, tolerance)
    expect(actual.keys).to match_array(expected.keys)
    expected.each { |item, p| expect(actual.fetch(item)).to be_within(tolerance).of(p), item }
  end

  describe "pick_probabilities" do
    it "LO3 (LC4) answers heads, leaders, its method and its standard error: exact, with no standard error" do
      answer = build(:beta, oracle::BETA4).pick_probabilities(explore: 0.1)
      expect(answer.keys).to contain_exactly(:heads, :leaders, :method, :standard_error)
      expect(answer[:method]).to eq(:exact)
      expect(answer[:standard_error]).to eq(0.0)
    end

    it "LO3 (LC4) answers the heads under explore 0.1 as BetaArgmax.probabilities(epsilon: 0.1), within 1e-12" do
      [ oracle::BETA4, oracle::BETA5, oracle::BETA6 ].each do |pairs|
        answer = build(:beta, pairs).pick_probabilities(explore: 0.1)
        reference = argmax.probabilities(pairs.values, epsilon: 0.1)
        expect_heads(answer[:heads], pairs.keys.zip(reference).to_h, 1e-12)
        expect(answer[:heads].values.sum).to be_within(1e-12).of(1.0)
      end
    end

    it "LO3 (LC4) gives every head a leader when no item lacks views: leaders equal heads, none left without one" do
      answer = build(:beta, oracle::BETA5).pick_probabilities(shuffle: 0.05, explore: 0.1)
      answer[:heads].each { |item, p| expect(answer[:leaders].fetch(item)).to be_within(1e-12).of(p), item }
      expect(answer[:leaders][nil].to_f).to be_within(1e-12).of(0.0)
    end

    it "LO3 (LC4) mixes the shuffled, explored and drawn heads by the rule's shares" do
      [ oracle::BETA4, oracle::BETA6 ].each do |pairs|
        answer = build(:beta, pairs).pick_probabilities(shuffle: 0.1, explore: 0.2)
        expect_heads(answer[:heads], heads_by_formula(pairs, shuffle: 0.1, explore: 0.2), 1e-12)
      end
    end

    it "LO3 (LC4) draws the head among the items that may lead only, and gives the others no head, still exactly" do
      answer = build(:beta, oracle::BETA4).pick_probabilities(shuffle: 0.1, explore: 0.1, may_lead: %w[a b c])
      expect(answer[:method]).to eq(:exact)
      expect_heads(answer[:heads], heads_by_formula(oracle::BETA4, shuffle: 0.1, explore: 0.1, may_lead: %w[a b c]),
                   1e-12)
      expect(answer[:heads]["d"]).to be_within(1e-15).of(0.0)
    end

    it "LO3 (LC4) explores only among items with at least min_views, and a head below it has no leader" do
      set = build(:beta, oracle::BETA4, oracle::VIEWS4)
      answer = set.pick_probabilities(explore: 0.2, min_views: 100.0)
      expected = heads_by_formula(oracle::BETA4, explore: 0.2, eligible: %w[b c d])
      expect_heads(answer[:heads], expected, 1e-12)
      expect(answer[:leaders].fetch("a", 0.0)).to be_within(1e-12).of(0.0)
      expect(answer[:leaders].fetch(nil)).to be_within(1e-12).of(expected["a"])
      %w[b c d].each { |item| expect(answer[:leaders].fetch(item)).to be_within(1e-12).of(expected[item]), item }
      expect(answer[:leaders].values.sum).to be_within(1e-12).of(1.0)
    end

    it "LO3 (LC4) gives the exploration share back to the draw when no item has min_views, and names no leader" do
      set = build(:beta, oracle::BETA4, oracle::BETA4.keys.to_h { |item| [ item, 50.0 ] })
      answer = set.pick_probabilities(shuffle: 0.1, explore: 0.2, min_views: 100.0)
      expect_heads(answer[:heads], heads_by_formula(oracle::BETA4, shuffle: 0.1, explore: 0.2, eligible: []), 1e-12)
      expect(answer[:leaders].fetch(nil)).to be_within(1e-12).of(1.0)
      oracle::BETA4.each_key { |item| expect(answer[:leaders].fetch(item, 0.0)).to be_within(1e-12).of(0.0), item }
    end

    it "LO3 (LC4) answers a Normal set's heads by NormalArgmax: within 1e-8 of Simpson's rule" do
      answer = build(:normal, oracle::NORMAL4).pick_probabilities(explore: 0.1)
      expect(answer[:method]).to eq(:exact)
      reference = oracle::NORMAL4.keys.zip(oracle::NORMAL4_LARGEST).to_h { |item, p| [ item, 0.025 + (0.9 * p) ] }
      expect_heads(answer[:heads], reference, 1e-8)
      direct = SkiftetStatistical::Bandit::NormalArgmax.probabilities(oracle::NORMAL4.values, epsilon: 0.1)
      expect_heads(answer[:heads], oracle::NORMAL4.keys.zip(direct).to_h, 1e-12)
    end

    it "LO3 (LC4) answers on the scaled pairs: a pair above LARGEST_TOTAL keeps its mean at 10**6" do
      set = build(:beta, { "a" => [ 2_000_000.0, 2_000_000.0 ], "b" => [ 2_000_800.0, 1_999_200.0 ] })
      reference = argmax.probabilities([ [ 500_000.0, 500_000.0 ], [ 500_200.0, 499_800.0 ] ])
      expect_heads(set.pick_probabilities[:heads], { "a" => reference[0], "b" => reference[1] }, 1e-12)
    end

    it "LO3 (LC4) answers one item a head of 1.0, and up to EXACT_PICK_MAX contenders exactly" do
      expect_heads(build(:beta, { "only" => [ 3.0, 4.0 ] }).pick_probabilities[:heads], { "only" => 1.0 }, 1e-12)
      pairs = (1..posteriors::EXACT_PICK_MAX).to_h { |i| [ "item#{i}", [ 2.0 + (0.1 * i), 5.0 ] ] }
      answer = build(:beta, pairs).pick_probabilities(explore: 0.1)
      expect(answer[:method]).to eq(:exact)
      expect_heads(answer[:heads], pairs.keys.zip(argmax.probabilities(pairs.values, epsilon: 0.1)).to_h, 1e-12)
    end

    it "LO3 (LC4) refuses a rule the order rule refuses" do
      set = build(:beta, oracle::BETA4)
      [ { shuffle: 0.6, explore: 0.6 }, { explore: -0.1 }, { shuffle: 1.5 }, { may_lead: %w[a zz] } ].each do |rule|
        expect { set.pick_probabilities(**rule) }.to raise_error(ArgumentError), rule.inspect
      end
    end

    # 01 GB6: the exact odds refuse a parameter below BetaArgmax::SMALLEST_PARAMETER (1e-6, alpha or beta) and take one
    # at it, as BetaArgmax.probabilities does.
    it "LO3 (LC4) refuses the exact pick odds over a parameter below BetaArgmax::SMALLEST_PARAMETER, and takes one at it" do
      oracle::BELOW_SMALLEST.each do |pairs|
        set = build(:beta, pairs)
        set.method(:pick_probabilities)
        [ {}, { explore: 0.1 }, { shuffle: 0.1, explore: 0.2 } ].each do |rule|
          expect { set.pick_probabilities(**rule) }.to raise_error(ArgumentError), "#{pairs.inspect} #{rule.inspect}"
        end
      end
      edge = { "a" => [ 2.0, 3.0 ], "b" => [ 1e-6, 1.0 ], "c" => [ 1.0, 1e-6 ] }
      answer = build(:beta, edge).pick_probabilities
      expect(answer[:method]).to eq(:exact)
      expect_heads(answer[:heads], edge.keys.zip(argmax.probabilities(edge.values)).to_h, 1e-12)
    end

    # 01 GB6's scope, ruled at the L3 barrier (2026-11-01): the refusal reads the free items, whichever of them may
    # lead, so the pick odds refuse what a keyed pick of the same set and rule refuses.
    it "the L3 barrier: refuses a parameter below BetaArgmax::SMALLEST_PARAMETER on an item that may not lead, as a " \
       "keyed pick does" do
      oracle::BELOW_SMALLEST.each do |pairs|
        set = build(:beta, pairs)
        [ { may_lead: %w[a c] }, { may_lead: %w[a c], explore: 0.1 }, { may_lead: %w[c], shuffle: 0.5, explore: 0.5 } ]
          .each do |rule|
          label = "#{pairs.inspect} #{rule.inspect}"
          expect { set.pick(key: "k", **rule) }.to raise_error(ArgumentError), label
          expect { set.pick_probabilities(**rule) }.to raise_error(ArgumentError), label
        end
      end
      edge = build(:beta, { "a" => [ 2.0, 3.0 ], "b" => [ 1e-6, 1.0 ], "c" => [ 3.0, 2.0 ] })
      answer = edge.pick_probabilities(may_lead: %w[a c])
      expect_heads(answer[:heads], heads_by_formula(edge.parameters, may_lead: %w[a c]), 1e-12)
      expect(answer[:heads]["b"]).to eq(0.0)
    end

    it "the L3 barrier: reads may_lead [] as every item, as nil, in the pick odds and the helper alike" do
      set = build(:beta, oracle::BETA4)
      expect_heads(set.pick_probabilities(explore: 0.1, may_lead: [])[:heads],
                   heads_by_formula(oracle::BETA4, explore: 0.1, may_lead: []), 1e-12)
      expect_heads(heads_by_formula(oracle::BETA4, explore: 0.1, may_lead: []),
                   heads_by_formula(oracle::BETA4, explore: 0.1), 0.0)
    end

    it "LO3 (LD1) answers more than EXACT_PICK_MAX contenders by the stated Monte Carlo, within 4 standard errors" do
      count = posteriors::EXACT_PICK_MAX + 1
      set = build(:beta, (1..count).to_h { |i| [ "item#{i}", [ 2.0, 5.0 ] ] })
      answer = set.pick_probabilities
      expect(answer[:method]).to eq(:monte_carlo)
      expect(answer[:standard_error]).to be > 0.0
      answer[:heads].each { |item, p| expect((p - (1.0 / count)).abs).to be <= 4.0 * answer[:standard_error], item }
      expect(answer[:heads].values.sum).to be_within(1e-9).of(1.0)
      expect(set.pick_probabilities).to eq(answer)
      expect(set.pick_probabilities(draws: 80_000)[:standard_error]).to be_between(0.4 * answer[:standard_error],
                                                                                  0.6 * answer[:standard_error])
    end
  end

  describe "Posteriors.mixture" do
    it "LO3 (LC4) mixes two layers' odds by hand: each item's sum over the branches of share times probability" do
      answer = posteriors.mixture(shares: { "x" => 0.25, "y" => 0.75 },
                                  probabilities: { "x" => { "a" => 0.5, "b" => 0.5 }, "y" => { "a" => 0.2, "c" => 0.8 } })
      expect(answer.keys).to match_array(%w[a b c])
      expect(answer["a"]).to be_within(1e-15).of(0.275)
      expect(answer["b"]).to be_within(1e-15).of(0.125)
      expect(answer["c"]).to be_within(1e-15).of(0.6)
    end

    it "LO3 (LC4) mixes two sets' exact pick odds into probabilities that sum to 1" do
      first = build(:beta, oracle::BETA4).pick_probabilities(explore: 0.1)[:heads]
      second = build(:beta, oracle::BETA5).pick_probabilities[:heads]
      answer = posteriors.mixture(shares: { "first" => 0.4, "second" => 0.6 },
                                  probabilities: { "first" => first, "second" => second })
      (first.keys | second.keys).each do |item|
        expect(answer.fetch(item)).to be_within(1e-15).of((0.4 * first.fetch(item, 0.0)) + (0.6 * second.fetch(item, 0.0)))
      end
      expect(answer.values.sum).to be_within(1e-12).of(1.0)
    end
  end

  describe "order_probability and positions" do
    OddsOracle::CASES.each do |label, spec|
      it "LO3 (LD1) #{label}: every order exact, summing to 1, by head the pick's odds, within 1e-4 of the reference" do
        pairs = spec[:pairs]
        set = build(spec[:kind], pairs, spec[:views])
        rule = spec[:rule]
        answers = oracle.orders(pairs.keys, rule).to_h do |order|
          answer = set.order_probability(order, **rule)
          expect(answer.keys).to contain_exactly(:probability, :method, :standard_error)
          expect(answer[:method]).to eq(:exact), order.inspect
          expect(answer[:standard_error]).to eq(0.0), order.inspect
          [ order, answer[:probability] ]
        end
        expect(answers.values.sum).to be_within(1e-4).of(1.0)
        spec[:orders].each do |order, reference|
          expect(answers.fetch(order)).to be_within(1e-4).of(reference), order.inspect
        end

        unless rule.key?(:pinned)
          set.pick_probabilities(**rule.except(:top))[:heads].each do |item, p|
            expect(answers.select { |order, _| order.first == item }.values.sum).to be_within(1e-4).of(p), item
          end
        end

        shuffle = rule.fetch(:shuffle, 0.0)
        if shuffle.positive?
          free = pairs.size - rule.fetch(:pinned, {}).size
          shown = rule[:top] || free
          floor = shuffle * oracle.factorial(free - shown) / oracle.factorial(free)
          expect(answers.values.min).to be >= floor * (1.0 - 1e-12)
        end

        next unless spec[:positions]

        positions = set.positions(**rule)
        expect(positions.keys).to contain_exactly(:probabilities, :method, :standard_error)
        expect(positions[:method]).to eq(:exact)
        expect(positions[:standard_error]).to eq(0.0)
        table = positions[:probabilities]
        expect(table.keys).to match_array(pairs.keys)
        table.each do |item, row|
          expect(row.size).to eq(pairs.size), item
          expect(row.sum).to be_within(1e-4).of(1.0), item
          row.each_with_index do |p, slot|
            expect(p).to be_within(1e-4).of(spec[:positions][item][slot]), "#{item} at #{slot}"
            expect(p).to be_within(1e-4).of(answers.select { |order, _| order[slot] == item }.values.sum),
                         "#{item} at #{slot}, by the orders"
          end
        end
        pairs.size.times do |slot|
          expect(table.values.sum { |row| row[slot] }).to be_within(1e-4).of(1.0), "position #{slot}"
        end
      end
    end

    it "LO3 (LD1) answers every order by hand at K = 3" do
      set = build(:beta, oracle::HAND)
      oracle::HAND_ORDERS.each do |order, exact|
        expect(set.order_probability(order)[:probability]).to be_within(1e-4).of(exact.to_f), order.inspect
      end
    end

    OddsOracle::HAND_POSITIONS.each do |label, (rule, views, exact)|
      it "LO3 (LD1) answers positions by hand at K = 3: #{label}" do
        answer = build(:beta, oracle::HAND, views).positions(**rule)
        expect(answer[:method]).to eq(:exact)
        expect(answer[:probabilities].keys).to match_array(exact.keys)
        exact.each do |item, row|
          expect(answer[:probabilities][item].size).to eq(3)
          row.each_with_index do |p, slot|
            expect(answer[:probabilities][item][slot]).to be_within(1e-4).of(p.to_f), "#{item} at #{slot}"
          end
        end
      end
    end

    it "LO3 (LD1) answers orders under items that may not lead by the stated Monte Carlo, within 4 standard errors" do
      set = build(:beta, oracle::HAND)
      rule = { may_lead: %w[a b] }
      { %w[a b c] => 1 / 15r, %w[a c b] => 4 / 15r, %w[b a c] => 1 / 12r, %w[b c a] => 7 / 12r }.each do |order, exact|
        answer = set.order_probability(order, **rule)
        p = exact.to_f
        expect(answer[:method]).to eq(:monte_carlo), order.inspect
        expect(answer[:standard_error]).to be_between(0.5 * Math.sqrt(p * (1.0 - p) / 20_000),
                                                      2.0 * Math.sqrt(p * (1.0 - p) / 20_000)), order.inspect
        expect((answer[:probability] - p).abs).to be <= 4.0 * answer[:standard_error], order.inspect
      end
      expect(set.order_probability(%w[c a b], **rule)[:probability]).to eq(0.0)
      expect(set.order_probability(%w[a c b], **rule)).to eq(set.order_probability(%w[a c b], **rule))
      expect(set.order_probability(%w[a c b], seed: 1, **rule)).not_to eq(set.order_probability(%w[a c b], seed: 2, **rule))
      wide = set.order_probability(%w[a c b], draws: 80_000, **rule)[:standard_error]
      expect(wide).to be_between(0.4 * set.order_probability(%w[a c b], **rule)[:standard_error],
                                 0.6 * set.order_probability(%w[a c b], **rule)[:standard_error])
    end

    it "LO3 (LD1) answers positions under items that may not lead by the stated Monte Carlo, within 4 standard errors" do
      answer = build(:beta, oracle::HAND).positions(may_lead: %w[a b])
      expect(answer[:method]).to eq(:monte_carlo)
      expect(answer[:standard_error]).to be > 0.0
      { "a" => [ 1 / 3r, 1 / 12r, 7 / 12r ], "b" => [ 2 / 3r, 1 / 15r, 4 / 15r ],
        "c" => [ 0r, 17 / 20r, 3 / 20r ] }.each do |item, row|
        row.each_with_index do |p, slot|
          expect((answer[:probabilities][item][slot] - p.to_f).abs).to be <= 4.0 * answer[:standard_error],
                                                                          "#{item} at #{slot}"
        end
      end
      expect(answer[:probabilities]["c"][0]).to eq(0.0)
    end

    it "LO3 (LC4) keeps the pick's odds exact under items that may not lead" do
      answer = build(:beta, oracle::HAND).pick_probabilities(may_lead: %w[a b])
      expect(answer[:method]).to eq(:exact)
      expect_heads(answer[:heads], { "a" => 1.0 / 3, "b" => 2.0 / 3, "c" => 0.0 }, 1e-8)
    end

    it "LO3 (LD1) answers positions exactly up to EXACT_ORDER_MAX free items and by Monte Carlo above" do
      most = posteriors::EXACT_ORDER_MAX
      exact = build(:beta, (1..most).to_h { |i| [ "item#{i}", [ 2.0, 5.0 ] ] }).positions
      expect(exact[:method]).to eq(:exact)
      exact[:probabilities].each_value do |row|
        row.each { |p| expect(p).to be_within(1e-4).of(1.0 / most) }
      end
      over = build(:beta, (1..most + 1).to_h { |i| [ "item#{i}", [ 2.0, 5.0 ] ] }).positions
      expect(over[:method]).to eq(:monte_carlo)
      expect(over[:standard_error]).to be > 0.0
      over[:probabilities].each_value do |row|
        row.each { |p| expect((p - (1.0 / (most + 1))).abs).to be <= 4.0 * over[:standard_error] }
      end
    end

    # 01 GB6: every exact odds call refuses a parameter below BetaArgmax::SMALLEST_PARAMETER (1e-6, alpha or beta).
    it "LO3 (LD1) refuses exact orders and positions over a parameter below BetaArgmax::SMALLEST_PARAMETER" do
      oracle::BELOW_SMALLEST.each do |pairs|
        set = build(:beta, pairs)
        set.method(:order_probability)
        set.method(:positions)
        [ {}, { explore: 0.1 }, { shuffle: 0.1, explore: 0.2 } ].each do |rule|
          label = "#{pairs.inspect} #{rule.inspect}"
          expect { set.order_probability(%w[a b c], **rule) }.to raise_error(ArgumentError), label
          expect { set.positions(**rule) }.to raise_error(ArgumentError), label
        end
      end
    end
  end

  describe "Choice#probability and Order#probability" do
    it "LO3 (LD1) answers a Choice's probability as the pick's odds of its head, under the rule that made it" do
      [ [ :beta, oracle::BETA4, { key: "visit-1" }, { explore: 0.1 } ],
        [ :beta, oracle::BETA5, { key: %w[visit 2] }, { shuffle: 0.1, explore: 0.2 } ],
        [ :beta, oracle::BETA6, { rng: Random.new(3) }, { explore: 0.25 } ],
        [ :normal, oracle::NORMAL4, { key: "visit-4" }, { explore: 0.1 } ] ].each do |kind, pairs, source, rule|
        set = build(kind, pairs)
        choice = set.pick(**source, **rule)
        expect(choice).to be_frozen
        expected = set.pick_probabilities(**rule)[:heads].fetch(choice.head)
        expect(choice.probability).to be_a(Float)
        expect(choice.probability).to be_within(1e-12).of(expected), "#{pairs.keys.size} items, #{rule.inspect}"
        expect(choice.probability).to eq(choice.probability)
      end
    end

    it "LO3 (LD1) answers an Order's probability as order_probability of its items, under the rule that made it" do
      [ [ :beta, oracle::BETA4, { key: "visit-1" }, { shuffle: 0.05, explore: 0.1 } ],
        [ :beta, oracle::BETA4, { key: "visit-2" }, { explore: 0.1, pinned: { 1 => "b" } } ],
        [ :normal, oracle::NORMAL5, { rng: Random.new(4) }, { explore: 0.1, top: 2 } ] ].each do |kind, pairs, source, rule|
        set = build(kind, pairs)
        order = set.rank(**source, **rule)
        expect(order).to be_frozen
        expected = set.order_probability(order.items, **rule)[:probability]
        expect(order.probability).to be_a(Float)
        expect(order.probability).to be_within(1e-12).of(expected), rule.inspect
      end
    end

    it "LO3 (LD1) answers an Order made with items that may not lead by the same Monte Carlo as order_probability" do
      set = build(:beta, oracle::HAND)
      order = set.rank(rng: Random.new(6), may_lead: %w[a b])
      expect(order.probability).to be_within(1e-12).of(set.order_probability(order.items, may_lead: %w[a b])[:probability])
    end
  end
end

# The L4 barrier (2026-11-02): the order and position odds are built and integrated in a coordinate where no item's
# quantiles round together (a Beta's position; a Normal's own standardised variable since the second round, below);
# and every odds call refuses, at every size, what a keyed draw of the same set and rule refuses. The references:
# Plackett-Luce for Beta(a, 1) items (-log X is exponential with rate a) and the reverse race for Beta(1, b), worked in
# Floats here; three identical items, every order 1/n! and every position 1/n; for a narrow item beside two wide ones
# (deviations 1e-300 and 1e100), the narrow item read as a point and each order an integral of the Normal's own
# functions, by Simpson's rule here; the rule's own visits, through rank(rng:) where its Sampler draws cannot tie (a
# Beta parameter far below 1 draws 0.0 in a share of its rng draws, about a tenth at 0.003, and two such draws tie to
# the earlier item) and through rank(key:) where they can (keyed draws compare by position and never tie).
RSpec.describe "SkiftetStatistical::Bandit::Posteriors odds, the L4 barrier" do
  def posteriors = SkiftetStatistical::Bandit::Posteriors

  # P(the items finish in `order`) when item i leads the rest with probability weights[i] / their sum.
  def plackett_luce(weights, order)
    rest = order.dup
    order.reduce(1.0) do |probability, item|
      share = weights.fetch(item) / rest.sum { |other| weights.fetch(other) }
      rest.delete(item)
      probability * share
    end
  end

  # { order => probability } over every whole order of the items, by `rule_of` (an order => its probability).
  def every_order(items, &rule_of) = items.permutation.to_h { |order| [ order, rule_of.call(order) ] }

  # { item => [P(at slot 0), ...] } summed from whole orders.
  def slots_of(orders, items)
    items.to_h { |item| [ item, items.size.times.map { |slot| orders.sum { |order, p| order[slot] == item ? p : 0.0 } } ] }
  end

  # Every whole order's and every position's answer against `orders` (whole orders => probability) within `tolerance`.
  def expect_orders_and_positions(set, orders, tolerance, rule = {})
    orders.each do |order, expected|
      answer = set.order_probability(order, **rule)
      expect(answer[:method]).to eq(:exact), order.inspect
      expect(answer[:probability]).to be_within(tolerance).of(expected), "#{set.parameters.inspect} #{order.inspect}"
    end
    table = set.positions(**rule)[:probabilities]
    slots_of(orders, set.items).each do |item, row|
      row.each_with_index do |expected, slot|
        expect(table.fetch(item)[slot]).to be_within(tolerance).of(expected), "#{set.parameters.inspect} #{item} at #{slot}"
      end
    end
  end

  # `visits` orders of the rule from `source` (a lambda answering one Order): [{ whole order => share }, { item =>
  # [share at each slot] }].
  def visited(set, visits)
    orders = Hash.new(0)
    slots = set.items.to_h { |item| [ item, Array.new(set.items.size, 0) ] }
    visits.times do |visit|
      order = yield(visit).items
      orders[order] += 1
      order.each_with_index { |item, slot| slots.fetch(item)[slot] += 1 }
    end
    [ orders.transform_values { |count| count.fdiv(visits) }, slots.transform_values { |row| row.map { |c| c.fdiv(visits) } } ]
  end

  # Every whole order's and position's exact answer within 4 standard errors of the visited shares.
  def expect_within_visits(set, rule, visits, &source)
    seen_orders, seen_slots = visited(set, visits, &source)
    set.items.permutation.each do |order|
      p = set.order_probability(order, **rule)[:probability]
      error = Math.sqrt([ p * (1.0 - p), 1e-12 ].max / visits)
      expect((seen_orders.fetch(order, 0.0) - p).abs).to be <= 4.0 * error, "#{set.parameters.inspect} #{order.inspect}"
    end
    set.positions(**rule)[:probabilities].each do |item, row|
      row.each_with_index do |p, slot|
        error = Math.sqrt([ p * (1.0 - p), 1e-12 ].max / visits)
        expect((seen_slots.fetch(item)[slot] - p).abs).to be <= 4.0 * error, "#{set.parameters.inspect} #{item} at #{slot}"
      end
    end
  end

  it "the L4 barrier: answers Beta(a, 1) items at a = 0.003 and below as Plackett-Luce, every order and position " \
     "within 1e-5, down to BetaArgmax::SMALLEST_PARAMETER" do
    [ [ 0.002, 0.003, 0.005 ], [ 0.001, 0.002, 0.003 ], [ 0.003, 0.0045, 0.0075, 0.02 ], [ 1e-6, 2e-6, 3e-6 ],
      [ 1e-6, 0.003, 0.5, 2.0 ] ].each do |alphas|
      weights = alphas.each_with_index.to_h { |alpha, index| [ "i#{index}", alpha ] }
      set = posteriors.beta(weights.transform_values { |alpha| [ alpha, 1.0 ] })
      orders = every_order(weights.keys) { |order| plackett_luce(weights, order) }
      expect_orders_and_positions(set, orders, 1e-5)
      expect(weights.keys.permutation.sum { |order| set.order_probability(order)[:probability] }).to be_within(1e-5).of(1.0)
    end
  end

  it "the L4 barrier: answers Beta(1, b) items at b = 0.003 and below as the reverse race, within 1e-5, so no mass " \
     "near 1 rounds into 1" do
    [ [ 0.002, 0.003, 0.005 ], [ 1e-6, 2e-6, 3e-6 ] ].each do |betas|
      weights = betas.each_with_index.to_h { |beta, index| [ "i#{index}", beta ] }
      set = posteriors.beta(weights.transform_values { |beta| [ 1.0, beta ] })
      orders = every_order(weights.keys) { |order| plackett_luce(weights, order.reverse) }
      expect_orders_and_positions(set, orders, 1e-5)
    end
  end

  it "the L4 barrier: answers small-alpha sets within 4 standard errors of the rule's own visits through rank(rng:)" do
    [ [ { "a" => [ 0.003, 1.0 ], "b" => [ 0.02, 1.0 ], "c" => [ 0.5, 1.0 ] }, {} ],
      [ { "a" => [ 0.001, 1.0 ], "b" => [ 0.05, 2.0 ], "c" => [ 1.0, 3.0 ], "d" => [ 2e-6, 1.0 ] },
        { shuffle: 0.05, explore: 0.1 } ] ].each do |pairs, rule|
      set = posteriors.beta(pairs)
      rng = Random.new(20_261_102)
      expect_within_visits(set, rule, 20_000) { set.rank(rng: rng, **rule) }
    end
  end

  it "the L4 barrier: answers a low-mean set whose items draw 0.0 alike through rank(rng:) as its keyed visits do, " \
     "within 4 standard errors, its orders summing to 1 and by head to the pick's odds within 1e-5" do
    set = posteriors.yes_no({ "a" => { views: 0, outcomes: 0 }, "b" => { views: 20, outcomes: 0 },
                              "c" => { views: 200, outcomes: 0 }, "d" => { views: 300, outcomes: 1 } },
                            priors: [ 0.002, 19.998 ])
    expect_within_visits(set, {}, 20_000) { |visit| set.rank(key: [ "visit", visit ]) }
    orders = set.items.permutation.to_h { |order| [ order, set.order_probability(order)[:probability] ] }
    expect(orders.values.sum).to be_within(1e-5).of(1.0)
    set.pick_probabilities[:heads].each do |item, head|
      expect(orders.sum { |order, p| order.first == item ? p : 0.0 }).to be_within(1e-5).of(head), item
    end
    table = set.positions[:probabilities]
    table.each_value { |row| expect(row.sum).to be_within(1e-5).of(1.0) }
    4.times { |slot| expect(table.values.sum { |row| row[slot] }).to be_within(1e-5).of(1.0), "position #{slot}" }
  end

  it "the L4 barrier: answers three identical Normals at every scale, every order 1/6 and every position 1/3, the " \
     "sums 1, within 1e-9" do
    [ [ 1e9, 1e-7 ], [ 1e12, 1e-3 ], [ 1e100, 1e-100 ] ].each do |pair|
      set = posteriors.normal({ "a" => pair, "b" => pair, "c" => pair })
      orders = %w[a b c].permutation.to_h { |order| [ order, set.order_probability(order)[:probability] ] }
      orders.each { |order, p| expect(p).to be_within(1e-9).of(1.0 / 6), "#{pair.inspect} #{order.inspect}" }
      expect(orders.values.sum).to be_within(1e-9).of(1.0), pair.inspect
      table = set.positions[:probabilities]
      table.each do |item, row|
        row.each { |p| expect(p).to be_within(1e-9).of(1.0 / 3), "#{pair.inspect} #{item}" }
        expect(row.sum).to be_within(1e-9).of(1.0), "#{pair.inspect} #{item}"
      end
      3.times { |slot| expect(table.values.sum { |row| row[slot] }).to be_within(1e-9).of(1.0), "#{pair.inspect} #{slot}" }
      expect(set.order_summary(%w[c a b])[:probability]).to be_within(1e-9).of(1.0 / 6), pair.inspect
    end
  end

  it "the L4 barrier: answers a narrow Normal beside two wide ones (deviations 1e-300 and 1e100), the narrow one the " \
     "leader, every order and position within 1e-5 of Simpson's rule" do
    cdf = ->(z) { 0.5 * Math.erfc(-z / Math.sqrt(2.0)) }
    pdf = ->(z) { Math.exp(-0.5 * z * z) / Math.sqrt(2.0 * Math::PI) }
    simpson = lambda do |from, to, &f|
      steps = 24_000
      width = (to - from) / steps
      (0..steps).sum { |i| (i.zero? || i == steps ? 1 : (i.odd? ? 4 : 2)) * f.call(from + (i * width)) } * width / 3.0
    end
    # In units of 1e100: a a point at 0.01 (its deviation 1e-302 of the unit), b ~ N(0, 1), c ~ N(-0.5, 1).
    orders = {
      %w[a b c] => simpson.call(-12.0, 0.01) { |b| pdf.(b) * cdf.(b + 0.5) },
      %w[a c b] => simpson.call(-12.0, 0.01) { |c| pdf.(c + 0.5) * cdf.(c) },
      %w[b a c] => (1.0 - cdf.(0.01)) * cdf.(0.51),
      %w[c a b] => (1.0 - cdf.(0.51)) * cdf.(0.01),
      %w[b c a] => simpson.call(0.01, 12.0) { |c| pdf.(c + 0.5) * (1.0 - cdf.(c)) },
      %w[c b a] => simpson.call(0.01, 12.0) { |b| pdf.(b) * (1.0 - cdf.(b + 0.5)) }
    }
    expect(orders.values.sum).to be_within(1e-12).of(1.0)
    set = posteriors.normal({ "a" => [ 1e98, 1e-300 ], "b" => [ 0.0, 1e100 ], "c" => [ -5e99, 1e100 ] })
    expect_orders_and_positions(set, orders, 1e-5)
  end

  it "the L4 barrier: refuses the pick odds over a parameter below BetaArgmax::SMALLEST_PARAMETER at every size, as " \
     "pick(key:) does, and takes one at it" do
    most = posteriors::EXACT_PICK_MAX
    over = (1..most + 1).to_h { |i| [ "item#{i}", [ 2.0, 5.0 ] ] }
    barred = (1..most + 2).to_h { |i| [ "item#{i}", [ 2.0, 5.0 ] ] }
    tiny = (1..most + 1).to_h { |i| [ "item#{i}", [ 1e-9, 1.0 ] ] }
    [ [ over.merge("item3" => [ 1e-9, 1.0 ]), {} ],
      [ over.merge("item3" => [ 1e-9, 1.0 ]), { explore: 0.1, shuffle: 0.1 } ],
      [ barred.merge("item3" => [ 1e-9, 1.0 ]), { may_lead: barred.keys - [ "item3" ], explore: 0.1 } ],
      [ tiny, { shuffle: 1.0 } ] ].each do |pairs, rule|
      set = posteriors.beta(pairs)
      label = "#{pairs.size} items #{rule.except(:may_lead).inspect}"
      expect { set.pick(key: "k", **rule) }.to raise_error(ArgumentError), label
      expect { set.pick_probabilities(**rule) }.to raise_error(ArgumentError, /SMALLEST|at least/), label
      expect { set.pick(rng: Random.new(1), **rule).probability }.to raise_error(ArgumentError), label
    end
    edge = posteriors.beta(over.merge("item3" => [ 1e-6, 1.0 ]))
    expect(edge.pick_probabilities[:method]).to eq(:monte_carlo)
    expect { edge.pick(key: "k") }.not_to raise_error
  end

  it "the L4 barrier: refuses the order and position Monte Carlo over a parameter below " \
     "BetaArgmax::SMALLEST_PARAMETER, as the exact odds and rank(key:) do, and reads no pinned item" do
    hand = { "a" => [ 1.0, 1.0 ], "b" => [ 1e-9, 1.0 ], "c" => [ 3.0, 1.0 ] }
    most = posteriors::EXACT_ORDER_MAX
    over = (1..most + 1).to_h { |i| [ "item#{i}", [ 2.0, 5.0 ] ] }.merge("item2" => [ 1e-9, 1.0 ])
    [ [ hand, { may_lead: %w[a c] } ], [ hand, { may_lead: %w[a c], explore: 0.2, shuffle: 0.1 } ],
      [ over, {} ], [ over, { pinned: { 0 => "item1" } } ] ].each do |pairs, rule|
      set = posteriors.beta(pairs)
      label = "#{pairs.size} items #{rule.inspect}"
      expect { set.rank(key: "k", **rule) }.to raise_error(ArgumentError), label
      expect { set.order_probability(set.items.first(2), **rule) }.to raise_error(ArgumentError), label
      expect { set.positions(**rule) }.to raise_error(ArgumentError), label
      expect { set.rank(rng: Random.new(2), **rule).probability }.to raise_error(ArgumentError), label
    end
    pinned = posteriors.beta(hand.merge("d" => [ 2.0, 2.0 ]))
    rule = { pinned: { 0 => "b" }, may_lead: %w[a c] }
    expect(pinned.order_probability(%w[b a], **rule)[:method]).to eq(:monte_carlo)
    expect(pinned.positions(**rule)[:method]).to eq(:monte_carlo)
    expect { pinned.rank(key: "k", **rule) }.not_to raise_error
    expect(pinned.positions(pinned: { 0 => "b" })[:method]).to eq(:exact)
    edge = posteriors.beta(hand.merge("b" => [ 1e-6, 1.0 ]))
    expect(edge.order_probability(%w[a b c], may_lead: %w[a c])[:method]).to eq(:monte_carlo)
    expect(edge.positions(may_lead: %w[a c])[:method]).to eq(:monte_carlo)
  end
end

# The L4 barrier's second round (2026-11-02). No one coordinate holds every Normal set of the input domain: x itself
# rounds a narrow item's points together at a large mean, x less the largest mean rounds two narrow items near 0
# together beside a far item, and a scale by a standard deviation sends a wide item's points past a Float's range. So
# each level of a Normal chain and each item's position sums are integrated over the item's own points in its own
# standardised variable, the level below read at the mapped points. The references: closed forms where an item is far
# wider than, or far ahead of, two narrow items (its place and their order are then independent within 1e-17:
# P(X_b > X_a) = Phi(gap / (sqrt(2) sd)), P(the far item above both) = Phi(its mean / its sd)); the rule's own keyed
# and RNG visits; and the round's identities (rows and columns summing to 1, orders by head the pick's odds). The
# greedy item reads the scaled pairs, as the keyed draws and the odds do, and the odds' stated Monte Carlo draws on
# the scaled pairs.
RSpec.describe "SkiftetStatistical::Bandit::Posteriors odds, the L4 barrier's second round" do
  def posteriors = SkiftetStatistical::Bandit::Posteriors

  def normal_cdf(x) = x.negative? ? 0.5 * Math.erfc(-x / Math.sqrt(2.0)) : 1.0 - (0.5 * Math.erfc(x / Math.sqrt(2.0)))

  # { whole order => probability } for a and b narrow and close (b's mean `gap` above a's) and c far wider or far
  # ahead, its mean `ahead` above a's.
  def closed_orders(pairs, gap, ahead)
    b_above = normal_cdf(gap / (Math.sqrt(2.0) * pairs.fetch("a").last))
    c_above = normal_cdf(ahead / pairs.fetch("c").last)
    { %w[a b c] => (1.0 - b_above) * (1.0 - c_above), %w[b a c] => b_above * (1.0 - c_above),
      %w[c a b] => c_above * (1.0 - b_above), %w[c b a] => c_above * b_above, %w[a c b] => 0.0, %w[b c a] => 0.0 }
  end

  # Every whole order and position of `set` within `tolerance` of `orders` (whole order => probability; when empty,
  # the positions against the set's own orders, slot by slot), and the identities: orders summing to 1, rows and
  # columns of the positions to 1, orders by head the pick's odds.
  def expect_odds(set, orders, tolerance)
    items = set.items
    answered = items.permutation.to_h do |order|
      answer = set.order_probability(order)
      expect(answer[:method]).to eq(:exact)
      [ order, answer[:probability] ]
    end
    label = set.parameters.inspect
    orders.each { |order, p| expect(answered.fetch(order)).to be_within(tolerance).of(p), "#{label} #{order.inspect}" }
    expect(answered.values.sum).to be_within(tolerance).of(1.0), label
    table = set.positions[:probabilities]
    heads = set.pick_probabilities[:heads]
    items.each do |item|
      expect(table.fetch(item).sum).to be_within(tolerance).of(1.0), "#{label} row #{item}"
      expect(answered.sum { |order, p| order.first == item ? p : 0.0 }).to be_within(tolerance).of(heads.fetch(item)),
                                                                            "#{label} by head #{item}"
      items.size.times do |slot|
        expected = (orders.empty? ? answered : orders).sum { |order, p| order[slot] == item ? p : 0.0 }
        expect(table.fetch(item)[slot]).to be_within(tolerance).of(expected), "#{label} #{item} at #{slot}"
      end
    end
    items.size.times do |slot|
      expect(table.values.sum { |row| row[slot] }).to be_within(tolerance).of(1.0), "#{label} column #{slot}"
    end
  end

  it "the L4 barrier, round 2: answers two narrow Normals near 0, or near 1e9, beside a far one at their closed " \
     "forms within 1e-7" do
    [ [ 0.0, 2.0**-34, 1e-8, 1e10 ], [ 0.0, 2.0**-34, 1e-9, 1e10 ], [ 0.0, 2.0**-34, 1e-12, 1.0 ],
      [ 0.0, 2.0**-34, 1e-12, 1e10 ], [ 0.0, 2.0**-34, 1e-9, 1.0 ], [ 1e9, 2.0**-23, 1e-7, 1e10 ],
      [ 1e9, 2.0**-23, 1e-7, 1.0 ] ].each do |base, gap, deviation, far|
      pairs = { "a" => [ base, deviation ], "b" => [ base + gap, deviation ], "c" => [ base + 1e6, far ] }
      expect(pairs.fetch("b").first - base).to eq(gap)
      expect_odds(posteriors.normal(pairs), closed_orders(pairs, gap, 1e6), 1e-7)
    end
  end

  it "the L4 barrier, round 2: answers the first round's sets (three identical Normals at every scale, a narrow one " \
     "beside two wide ones) with rows, columns and orders by head within 1e-7" do
    [ [ 1e9, 1e-7 ], [ 1e12, 1e-3 ], [ 1e100, 1e-100 ] ].each do |pair|
      set = posteriors.normal({ "a" => pair, "b" => pair, "c" => pair })
      expect_odds(set, %w[a b c].permutation.to_h { |order| [ order, 1.0 / 6 ] }, 1e-7)
    end
    set = posteriors.normal({ "a" => [ 1e98, 1e-300 ], "b" => [ 0.0, 1e100 ], "c" => [ -5e99, 1e100 ] })
    expect_odds(set, {}, 1e-7)
  end

  it "the L4 barrier, round 2: answers two narrow Normals near 0 beside a wide one within 4 standard errors of the " \
     "rule's own keyed and RNG visits" do
    set = posteriors.normal({ "a" => [ 0.0, 1e-8 ], "b" => [ 2.0**-34, 1e-8 ], "c" => [ 1e6, 1e10 ] })
    visits = 20_000
    rng = Random.new(20_261_102)
    { keyed: ->(visit) { set.rank(key: [ "visit", visit ]) }, rng: ->(_visit) { set.rank(rng: rng) } }.each do |source, draw|
      counts = Hash.new(0)
      slots = set.items.to_h { |item| [ item, Array.new(3, 0) ] }
      visits.times do |visit|
        order = draw.call(visit).items
        counts[order] += 1
        order.each_with_index { |item, slot| slots.fetch(item)[slot] += 1 }
      end
      set.items.permutation.each do |order|
        p = set.order_probability(order)[:probability]
        error = Math.sqrt([ p * (1.0 - p), 1e-12 ].max / visits)
        expect((counts[order].fdiv(visits) - p).abs).to be <= 4.0 * error, "#{source} #{order.inspect}"
      end
      set.positions[:probabilities].each do |item, row|
        row.each_with_index do |p, slot|
          error = Math.sqrt([ p * (1.0 - p), 1e-12 ].max / visits)
          expect((slots.fetch(item)[slot].fdiv(visits) - p).abs).to be <= 4.0 * error, "#{source} #{item} at #{slot}"
        end
      end
    end
  end

  it "the L4 barrier, round 2: greedy judges the largest mean on the scaled pairs, the pairs the keyed draws and the " \
     "odds read, where the means as given order the other way" do
    first = { "101" => [ 35.67952118162001, 32.41986812014265 ], "202" => [ 2_840_925.5635073353, 2_581_380.832977532 ] }
    second = { "101" => [ 42.24029229887194, 12.945986355509204 ], "202" => [ 186_924_884.510124, 57_289_542.10948547 ] }
    mean = ->((alpha, beta)) { alpha / (alpha + beta) }
    [ [ first, "101" ], [ second, "202" ] ].each do |given, scaled_first|
      set = posteriors.beta(given)
      given_first = given.max_by { |_, pair| mean.call(pair) }.first
      expect(given_first).not_to eq(scaled_first), given.inspect
      expect(set.greedy).to eq(scaled_first), given.inspect
      expect(set.greedy).to eq(set.items[SkiftetStatistical::BetaArgmax.greedy(set.parameters.values)])
    end
    rng = Random.new(20_261_102)
    differ = 0
    2_000.times do
      share = rng.rand(0.02..0.98)
      total = rng.rand(1.0..100.0)
      scale = rng.rand(30_000.0..5_000_000.0)
      given = { "101" => [ total * share, total * (1.0 - share) ], "202" => [ total * share * scale, total * (1.0 - share) * scale ] }
      set = posteriors.beta(given)
      expect(set.greedy).to eq(set.items[SkiftetStatistical::BetaArgmax.greedy(set.parameters.values)]), given.inspect
      differ += 1 unless set.greedy == set.items[SkiftetStatistical::BetaArgmax.greedy(given.values)]
    end
    expect(differ).to be > 0
  end

  it "the L4 barrier, round 2: the odds' stated Monte Carlo draws on the scaled pairs, as the exact odds read them" do
    most = posteriors::EXACT_PICK_MAX
    pairs = (1..most).to_h { |i| [ "item#{i}", [ 500_000.0, 500_000.0 ] ] }.merge("lead" => [ 5e11, 5e11 ])
    set = posteriors.beta(pairs)
    expect(set.parameters.values.uniq).to eq([ [ 500_000.0, 500_000.0 ] ])
    heads = set.pick_probabilities
    expect(heads[:method]).to eq(:monte_carlo)
    heads[:heads].each { |item, p| expect((p - (1.0 / (most + 1))).abs).to be <= 4.0 * heads[:standard_error], item }
    count = posteriors::EXACT_ORDER_MAX + 1
    ordered = posteriors.beta((1...count).to_h { |i| [ "item#{i}", [ 500_000.0, 500_000.0 ] ] }.merge("lead" => [ 5e11, 5e11 ]))
    table = ordered.positions
    expect(table[:method]).to eq(:monte_carlo)
    table[:probabilities].each do |item, row|
      row.each_with_index { |p, slot| expect((p - (1.0 / count)).abs).to be <= 4.0 * table[:standard_error], "#{item} at #{slot}" }
    end
    expect(set.greedy).to eq("item1")
  end
end

# The L4 barrier's third round (2026-11-02; 01 D2 as amended, D38): the order rule compares Normal draws exactly, by the
# sign of (mean_i - mean_j) + (sd_i z_i - sd_j z_j), so the odds are again the probability the rule answers where a
# standard deviation lies at or below a rounding of its mean; and every draw, from any source, reads the scaled pairs.
RSpec.describe "SkiftetStatistical::Bandit::Posteriors odds, the L4 barrier's third round" do
  def posteriors = SkiftetStatistical::Bandit::Posteriors

  # The share of `visits` orders (each from the block) that answer each whole order and put each item at each slot,
  # within 4 standard errors of the set's exact order and position odds.
  def expect_visits(set, visits, label)
    counts = Hash.new(0)
    slots = set.items.to_h { |item| [ item, Array.new(set.items.size, 0) ] }
    visits.times do |visit|
      order = yield(visit).items
      counts[order] += 1
      order.each_with_index { |item, slot| slots.fetch(item)[slot] += 1 }
    end
    set.items.permutation.each do |order|
      p = set.order_probability(order)[:probability]
      error = Math.sqrt([ p * (1.0 - p), 1e-12 ].max / visits)
      expect((counts[order].fdiv(visits) - p).abs).to be <= 4.0 * error, "#{label} #{order.inspect}"
    end
    set.positions[:probabilities].each do |item, row|
      row.each_with_index do |p, slot|
        error = Math.sqrt([ p * (1.0 - p), 1e-12 ].max / visits)
        expect((slots.fetch(item)[slot].fdiv(visits) - p).abs).to be <= 4.0 * error, "#{label} #{item} at #{slot}"
      end
    end
  end

  # Two identical Normals draw in the order of their standard deviates however their values round: at a deviation
  # below a rounding of the mean, at the smallest Float above 0 (whose multiples round to 0 or to it) and at the input
  # domain's bound.
  it "the L4 barrier, round 3: orders two Normal draws that round to one Float by their exact values, under given " \
     "uniforms and an RNG, and answers the rounded values" do
    quantile = ->(uniform) { SkiftetStatistical::Significance.normal_quantile(uniform) }
    [ [ 1e9, 1e-9 ], [ 0.0, 5e-324 ], [ 1e100, 5e-324 ], [ -1e100, 1e-300 ] ].each do |mean, deviation|
      set = posteriors.normal({ "a" => [ mean, deviation ], "b" => [ mean, deviation ] })
      rng = Random.new(20_261_102)
      ties = 0
      200.times do
        first = rng.rand
        second = rng.rand
        order = set.rank(uniforms: { explore: 0.5, priority: { "a" => 0.5, "b" => 0.5 }, draw: { "a" => first, "b" => second } })
        expect(order.items).to eq(quantile.call(second) > quantile.call(first) ? %w[b a] : %w[a b]), [ mean, first, second ].inspect
        drawn = { "a" => mean + (deviation * quantile.call(first)), "b" => mean + (deviation * quantile.call(second)) }
        expect(order.draws).to eq(drawn)
        ties += 1 if drawn["a"] == drawn["b"]
      end
      expect(ties).to be > 20, mean.to_s
      (1..200).each do |seed|
        sampler = SkiftetStatistical::Sampler.new(Random.new(seed))
        first = sampler.gaussian
        second = sampler.gaussian
        order = set.rank(rng: Random.new(seed))
        expect(order.items).to eq(second > first ? %w[b a] : %w[a b]), "#{mean} seed #{seed}"
        expect(order.draws).to eq({ "a" => mean + (deviation * first), "b" => mean + (deviation * second) })
      end
    end
  end

  it "the L4 barrier, round 3: answers the refute's sets, a deviation at or below a rounding of a large mean, within " \
     "4 standard errors of the rule's own keyed and RNG visits" do
    identical = ->(pair, count) { %w[a b c].first(count).to_h { |item| [ item, pair ] } }
    sets = [ identical.call([ 1e9, 1e-9 ], 2), { "a" => [ 1e9, 1e-7 ], "b" => [ 1e9.next_float, 1e-7 ] },
             identical.call([ 1e15, 0.05 ], 3), identical.call([ 1e100, 1e-100 ], 3), identical.call([ 1e12, 1e-3 ], 3),
             identical.call([ 1e9, 1e-7 ], 3),
             { "a" => [ 1e9, 1e-7 ], "b" => [ 1e9 + (2.0**-23), 1e-7 ], "c" => [ 1e9 + 1e6, 1.0 ] } ]
    sets.each do |pairs|
      set = posteriors.normal(pairs)
      rng = Random.new(20_261_102)
      expect_visits(set, 6_000, "#{pairs.inspect} keyed") { |visit| set.rank(key: [ "visit", visit ]) }
      expect_visits(set, 6_000, "#{pairs.inspect} rng") { |_visit| set.rank(rng: rng) }
    end
  end

  it "the L4 barrier, round 3: the odds' stated Monte Carlo agrees with the exact pick odds on identical narrow " \
     "Normals at a large mean, for the positions above EXACT_ORDER_MAX and the pick above EXACT_PICK_MAX" do
    count = posteriors::EXACT_ORDER_MAX + 1
    set = posteriors.normal((1..count).to_h { |i| [ "item#{i}", [ 1e15, 0.05 ] ] })
    heads = set.pick_probabilities[:heads]
    table = set.positions
    expect(table[:method]).to eq(:monte_carlo)
    table[:probabilities].each do |item, row|
      expect(heads.fetch(item)).to be_within(1e-9).of(1.0 / count)
      expect((row.first - heads.fetch(item)).abs).to be <= 4.0 * table[:standard_error], item
    end
    leading = set.order_probability([ "item1" ])
    expect(leading[:method]).to eq(:monte_carlo)
    expect((leading[:probability] - heads.fetch("item1")).abs).to be <= 4.0 * leading[:standard_error]
    most = posteriors::EXACT_PICK_MAX + 1
    wide = posteriors.normal((1..most).to_h { |i| [ "item#{i}", [ 1e15, 0.05 ] ] }).pick_probabilities
    expect(wide[:method]).to eq(:monte_carlo)
    wide[:heads].each { |item, p| expect((p - (1.0 / most)).abs).to be <= 4.0 * wide[:standard_error], item }
  end

  it "the L4 barrier, round 3: rank, pick and values draw from an RNG on the scaled pairs, the pairs the keyed draws " \
     "and the odds read" do
    most = posteriors::EXACT_PICK_MAX
    pairs = (1..most).to_h { |i| [ "item#{i}", [ 500_000.0, 500_000.0 ] ] }.merge("lead" => [ 5e11, 5e11 ])
    set = posteriors.beta(pairs)
    expect(set.parameters.fetch("lead")).to eq([ 500_000.0, 500_000.0 ])
    reference = SkiftetStatistical::Sampler.new(Random.new(7))
    expect(set.values(rng: Random.new(7))).to eq(set.items.map { reference.beta(500_000.0, 500_000.0) })
    reference = SkiftetStatistical::Sampler.new(Random.new(8))
    expect(set.rank(rng: Random.new(8)).draws.values).to eq(set.items.map { reference.beta(500_000.0, 500_000.0) })
    rng = Random.new(20_261_102)
    visits = 6_000
    led = visits.times.count { set.pick(rng: rng).head == "lead" }
    share = 1.0 / (most + 1)
    expect((led.fdiv(visits) - share).abs).to be <= 4.0 * Math.sqrt(share * (1.0 - share) / visits)
    expect((set.pick_probabilities[:heads].fetch("lead") - share).abs).to be <= 4.0 * set.pick_probabilities[:standard_error]
  end
end
