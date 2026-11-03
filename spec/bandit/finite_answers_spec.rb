# frozen_string_literal: true

# The bandit's one check of an answer (the L3 barrier's sixth round): every public method new in 0.4.0 answers finite
# numbers for every input in the domain (SkiftetStatistical::Numbers::LARGEST, 1e100 either side of 0), or refuses it
# with an ArgumentError, never anything else. The walk draws its inputs at random over the whole domain, from the
# smallest Float above 0 to the bound, given uniforms at 0 and 1 included, and asks every public method of the
# posterior set, Prior, Decay, Draw, NormalArgmax and BetaArgmax's new methods. A refusal is allowed (a quotient by a
# small input that leaves a Float is one); a NaN or an infinity in an answer, or an error of another class, is not.
# BetaArgmax.position answers minus and plus infinity at a uniform of exactly 0 and 1 by its contract (a comparison
# key, never a draw), so it is asked inside them. A second walk (the L4 barrier) asks the odds of orders and positions
# the same way: order_probability, positions and order_summary, exact and by their stated Monte Carlo, and the
# propensities Choice#probability and Order#probability, on fewer rounds, since each exact answer integrates a grid.
RSpec.describe "the bandit's answers on inputs in the domain" do
  def bandit = SkiftetStatistical::Bandit
  def posteriors = bandit::Posteriors

  # The magnitudes the walk draws around, the smallest Float above 0 to the domain's bound.
  def magnitudes = [ 5e-324, 1e-300, 1e-100, 1e-8, 1e-6, 1e-3, 0.5, 1.0, 10.0, 1e3, 1e6, 1e9, 1e20, 1e50, 1e99, 1e100 ]

  # A non-negative number in the domain, at a random magnitude (never above LARGEST).
  def size(rng)
    [ magnitudes[rng.rand(magnitudes.size)] * (0.5 + rng.rand), 1e100 ].min
  end

  # A number in the domain of either sign.
  def signed(rng) = (rng.rand < 0.3 ? -1 : 1) * size(rng)

  # A uniform in [0, 1], its ends included.
  def uniform(rng) = [ rng.rand, 0.0, 1.0, 2.0**-53, 1.0 - (2.0**-53) ][rng.rand(5)]

  # The Floats of an answer that are not finite: in a Float, a Hash's values, an Array or a Struct's members.
  def nonfinite(answer)
    case answer
    when Float then answer.finite? ? [] : [ answer ]
    when Hash then answer.values.flat_map { |value| nonfinite(value) }
    when Array then answer.flat_map { |value| nonfinite(value) }
    when Struct then answer.to_a.flat_map { |value| nonfinite(value) }
    else []
    end
  end

  # Each public method, asked with inputs drawn from `rng` (a set of 1 to 4 items of either kind).
  def calls(rng, round)
    k = 1 + rng.rand(4)
    items = (0...k).map { |j| "i#{j}" }
    tally = -> { { views: size(rng), outcomes: size(rng) } }
    amount_tally = -> { { views: size(rng), outcomes: signed(rng), outcome_squares: size(rng) } }
    amount_prior = -> { bandit::Prior.amount(mean: signed(rng), variance: size(rng), strength: size(rng)) }
    sets = lambda do
      beta = posteriors.beta(items.to_h { |item| [ item, [ size(rng), size(rng) ] ] })
      normal = posteriors.normal(items.to_h { |item| [ item, [ signed(rng), size(rng) ] ] },
                                 views: items.to_h { |item| [ item, size(rng) ] })
      [ beta, normal ]
    end
    uniforms = -> { { explore: uniform(rng), priority: items.to_h { |item| [ item, signed(rng) ] },
                      draw: items.to_h { |item| [ item, uniform(rng) ] } } }
    mass = -> { [ 0.5, 0.95, 1.0 - (2.0**-53) ][rng.rand(3)] }
    set_calls = %i[beta normal].each_with_index.flat_map do |kind, index|
      set = -> { sets.call[index] }
      [
        [ "#{kind} values", -> { set.call.values(rng: Random.new(round)) } ],
        [ "#{kind} pick rng", -> { set.call.pick(rng: Random.new(round), explore: 0.2) } ],
        [ "#{kind} pick key", -> { set.call.pick(key: [ "k", round ], shuffle: 0.1, explore: 0.1) } ],
        [ "#{kind} rank uniforms", -> { set.call.rank(uniforms: uniforms.call, shuffle: 0.2, explore: 0.3) } ],
        [ "#{kind} rank key", -> { set.call.rank(key: "k#{round}", pinned: { 0 => "i0" }, top: 2) } ],
        [ "#{kind} rank rng", -> { set.call.rank(rng: Random.new(round), min_views: size(rng)) } ],
        [ "#{kind} greedy", -> { set.call.greedy } ],
        [ "#{kind} pick odds", -> { set.call.pick_probabilities(explore: 0.1, shuffle: 0.1) } ],
        [ "#{kind} summary", -> { set.call.summary(credible_mass: mass.call) } ],
        [ "#{kind} means-only summary", -> { set.call.summary(among: []) } ],
        [ "#{kind} means", -> { set.call.means } ],
        [ "#{kind} parameters", -> { built = set.call; [ built.parameters, built.views ] } ]
      ]
    end
    prior = bandit::Prior
    argmax = SkiftetStatistical::BetaArgmax
    weighted_tally = -> { tally.call.merge(weighted_views: size(rng), weighted_outcomes: size(rng),
                                           weighted_views_squared: size(rng)) }
    normals = -> { Array.new(1 + rng.rand(3)) { [ signed(rng), size(rng) ] } }
    betas = -> { Array.new(1 + rng.rand(3)) { [ size(rng), size(rng) ] } }
    others = -> { Array.new(rng.rand(4)) { { group: tally.call, own: tally.call } } }
    inner = -> { rng.rand.clamp(2.0**-53, 1.0 - (2.0**-53)) }
    set_calls + [
      [ "yes_no", lambda do
        tallies = items.to_h { |item| [ item, weighted_tally.call ] }
        posteriors.yes_no(tallies, priors: [ size(rng), size(rng) ]).parameters
      end ],
      [ "amounts", lambda do
        posteriors.amounts(items.to_h { |item| [ item, amount_tally.call ] }, priors: amount_prior.call).parameters
      end ],
      [ "mixture", lambda do
        posteriors.mixture(shares: { x: size(rng), y: size(rng) },
                           probabilities: { x: { a: size(rng) }, y: { a: size(rng), b: size(rng) } })
      end ],
      [ "given", -> { prior.given(mean: rng.rand, strength: size(rng)) } ],
      [ "amount", -> { amount_prior.call } ],
      [ "borrowed", -> { prior.borrowed(group: tally.call, own: tally.call) } ],
      [ "pooled", lambda do
        prior.pooled(group: tally.call, own: tally.call, strength: size(rng), base: [ size(rng), size(rng) ])
      end ],
      [ "pooled_new", lambda do
        prior.pooled_new(others: others.call, group: tally.call, own: tally.call, weight: size(rng))
      end ],
      [ "pooled_mean", lambda do
        prior.pooled_mean(views: size(rng), outcomes: signed(rng), prior_mean: signed(rng), strength: size(rng))
      end ],
      [ "pooled_amount", lambda do
        prior.pooled_amount(group: amount_tally.call, own: amount_tally.call, strength: size(rng))
      end ],
      [ "chain", lambda do
        prior.chain(levels: Array.new(1 + rng.rand(3)) { tally.call.merge(strength: size(rng)) }, root_mean: rng.rand,
                    tilt: size(rng))
      end ],
      [ "tilts", -> { prior.tilts({ a: size(rng), b: size(rng) }, clip: [ 0.0, size(rng) ]) } ],
      [ "weight", -> { bandit::Decay.weight(size(rng), half_life: size(rng), cap: size(rng)) } ],
      [ "fold", lambda do
        bandit::Decay.fold(Array.new(rng.rand(4)) { amount_tally.call.merge(age: size(rng)) }, half_life: size(rng))
      end ],
      [ "uniform", lambda do
        [ bandit::Draw.uniform("t#{round}"), bandit::Draw.uniform_for("t", round), bandit::Draw.seed_for(round) ]
      end ],
      [ "open_uniform", -> { bandit::Draw.open_uniform(uniform(rng)) } ],
      [ "weighted", -> { bandit::Draw.weighted({ a: size(rng), b: size(rng), c: signed(rng) }, uniform(rng)) } ],
      [ "normal probabilities", -> { bandit::NormalArgmax.probabilities(normals.call) } ],
      [ "normal maximum", -> { bandit::NormalArgmax.expected_maximum(normals.call) } ],
      [ "beta maximum", -> { argmax.expected_maximum(betas.call) } ],
      [ "beta distribution", -> { argmax.distribution(signed(rng), size(rng), size(rng)) } ],
      [ "beta position", -> { argmax.position(inner.call, size(rng), size(rng)) } ]
    ]
  end

  it "finds no NaN or infinity in any answer, and no refusal but an ArgumentError, over 800 random rounds" do
    rng = Random.new(20_261_102)
    answered = Hash.new(0)
    found = []
    800.times do |round|
      calls(rng, round).each do |name, call|
        answer = call.call
        answered[name] += 1
        bad = nonfinite(answer)
        found << [ name, round, bad ] unless bad.empty?
      rescue ArgumentError
        nil
      rescue StandardError, NotImplementedError => e
        found << [ name, round, e.class, e.message ]
      end
    end

    expect(found).to eq([])
    # Every method answered on many rounds: the walk is not all refusals (a Beta set below
    # BetaArgmax::SMALLEST_PARAMETER is refused by its quantile draws and odds, so those answer on the fewest).
    expect(answered.keys.size).to eq(calls(Random.new(1), 0).size)
    expect(answered.values.min).to be > 30
  end

  # The odds of orders and positions, each asked on one set of each kind a round (1 to 4 items, with views).
  def odds_calls(rng, round)
    k = 1 + rng.rand(4)
    items = (0...k).map { |j| "i#{j}" }
    views = -> { items.to_h { |item| [ item, size(rng) ] } }
    beta = posteriors.beta(items.to_h { |item| [ item, [ size(rng), size(rng) ] ] }, views: views.call)
    normal = posteriors.normal(items.to_h { |item| [ item, [ signed(rng), size(rng) ] ] }, views: views.call)
    { beta: beta, normal: normal }.flat_map do |kind, set|
      [
        [ "#{kind} order odds", -> { set.order_probability(items.first(2), shuffle: 0.1, explore: 0.1, min_views: size(rng)) } ],
        [ "#{kind} order odds pinned", -> { set.order_probability(items.first(2), pinned: { 0 => "i0" }, top: 2) } ],
        [ "#{kind} order odds counted", lambda do
          set.order_probability(items.reverse, draws: 200, seed: round, may_lead: [ "i0" ], explore: 0.2)
        end ],
        [ "#{kind} positions", -> { set.positions(shuffle: 0.1, explore: 0.2, min_views: size(rng)) } ],
        [ "#{kind} positions counted", -> { set.positions(draws: 200, seed: round, may_lead: [ "i0" ]) } ],
        [ "#{kind} order summary", -> { set.order_summary(items.reverse) } ],
        [ "#{kind} choice probability", -> { set.pick(rng: Random.new(round), explore: 0.1).probability } ],
        [ "#{kind} order probability", -> { set.rank(rng: Random.new(round), shuffle: 0.1, top: 2).probability } ]
      ]
    end
  end

  it "the L4 barrier: finds no NaN or infinity in the odds of orders and positions, and no refusal but an " \
     "ArgumentError, over 400 random rounds" do
    rng = Random.new(20_261_103)
    answered = Hash.new(0)
    found = []
    400.times do |round|
      odds_calls(rng, round).each do |name, call|
        answer = call.call
        answered[name] += 1
        bad = nonfinite(answer)
        found << [ name, round, bad ] unless bad.empty?
      rescue ArgumentError
        nil
      rescue StandardError, NotImplementedError => e
        found << [ name, round, e.class, e.message ]
      end
    end

    expect(found).to eq([])
    expect(answered.keys.size).to eq(odds_calls(Random.new(1), 0).size)
    expect(answered.values.min).to be > 10
  end
end
