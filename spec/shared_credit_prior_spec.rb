# frozen_string_literal: true

require "json"

# SharedCredit.with_prior: the estimate under a prior pooled from its own credited arms, each arm below the strength
# topped up to it. The prior's mean per part is m0 = sum_a n_a * r_a / N (n_a Kish's size, N their sum), its per-visit
# covariance V0 = sum_a n_a * (n_a * cov(r_a) + g_a * g_a^T) / N with g_a = r_a - m0; an arm of size n < k takes
# e = k - n visits of it: means (e * m0 + n * r_a) / k, block (n / k)^2 * cov(r_a) + (e * V0 + (e * n / k) * g_a g_a^T)
# / k^2, every covariance with another arm scaled by n / k. Every expected number below is worked out by hand in the
# comment beside it, or by the Rational reference (#reference), which reads the rows themselves.
RSpec.describe "SkiftetStatistical::SharedCredit.with_prior" do
  def shared_credit
    SkiftetStatistical::SharedCredit
  end

  def linear_value
    SkiftetStatistical::LinearValue
  end

  # Every number within 1e-12 of the expected (relative above 1), nil where nil is expected, the same keys and sizes.
  def mismatches(actual, expected, path = "")
    case expected
    when Hash
      same = actual.is_a?(Hash) && actual.keys.map(&:to_s).sort == expected.keys.map(&:to_s).sort
      return [ "#{path}: keys #{actual.inspect} are not #{expected.keys.inspect}" ] unless same

      expected.flat_map { |key, value| mismatches(actual[key] || actual[key.to_s], value, "#{path}[#{key.inspect}]") }
    when Array
      same = actual.is_a?(Array) && actual.size == expected.size
      return [ "#{path}: #{actual.inspect} is not #{expected.size} long" ] unless same

      expected.each_with_index.flat_map { |value, index| mismatches(actual[index], value, "#{path}[#{index}]") }
    when nil
      actual.nil? ? [] : [ "#{path}: #{actual.inspect} is not nil" ]
    else
      close = actual.is_a?(Float) && (actual - expected).abs <= 1e-12 * [ 1.0, expected.abs ].max
      close ? [] : [ "#{path}: #{actual.inspect} is not #{expected.to_f}" ]
    end
  end

  def expect_close(actual, expected)
    problems = mismatches(actual, expected)
    expect(problems).to be_empty, problems.first(8).join("\n")
  end

  # The doc comment's rule in Rationals, from the rows: Hajek means, the sandwich, Kish's sizes, the pooled prior and
  # the top-up. It shares no code with the module.
  def reference(rows, arms, parts, strength)
    k = strength.to_r
    credit = ->(row, arm) { (row[:credit][arm] || 0).to_r }
    total_credit = arms.to_h { |arm| [ arm, rows.sum(0r) { |row| credit.call(row, arm) } ] }
    credited = arms.select { |arm| total_credit[arm].positive? }
    rate = credited.to_h do |arm|
      [ arm, parts.each_index.map { |p| rows.sum(0r) { |row| credit.call(row, arm) * row[:values][p].to_r } / total_credit[arm] } ]
    end
    psi = ->(row, arm, p) { credit.call(row, arm) * (row[:values][p].to_r - rate[arm][p]) / total_credit[arm] }
    cov = ->(a, p, b, q) { rows.sum(0r) { |row| psi.call(row, a, p) * psi.call(row, b, q) } }
    kish = credited.to_h { |arm| [ arm, total_credit[arm]**2 / rows.sum(0r) { |row| credit.call(row, arm)**2 } ] }
    visits = kish.values.sum(0r)
    centre = parts.each_index.map { |p| credited.sum(0r) { |a| kish[a] * rate[a][p] } / visits }
    gap = credited.to_h { |a| [ a, parts.each_index.map { |p| rate[a][p] - centre[p] } ] }
    spread = parts.each_index.map do |p|
      parts.each_index.map do |q|
        credited.sum(0r) { |a| kish[a] * ((kish[a] * cov.call(a, p, a, q)) + (gap[a][p] * gap[a][q])) } / visits
      end
    end
    topped = ->(a) { kish[a] < k }
    share = ->(a) { topped.call(a) ? kish[a] / k : 1r }
    means = arms.to_h do |a|
      next [ a, nil ] unless credited.include?(a)

      [ a, topped.call(a) ? parts.each_index.map { |p| (((k - kish[a]) * centre[p]) + (kish[a] * rate[a][p])) / k } : rate[a] ]
    end
    cells = arms.product(parts.each_index.to_a)
    covariance = cells.map do |a, p|
      cells.map do |b, q|
        next 0r unless credited.include?(a) && credited.include?(b)

        value = share.call(a) * share.call(b) * cov.call(a, p, b, q)
        if a == b && topped.call(a)
          extra = k - kish[a]
          value += ((extra * spread[p][q]) + (extra * kish[a] / k * gap[a][p] * gap[a][q])) / (k**2)
        end
        value
      end
    end
    { means: means, covariance: covariance }
  end

  # One row a visit, each crediting its arm with `credit`; a visit's values are one number or one per part.
  def rows_of(values_by_arm, credit = 1.0)
    values_by_arm.flat_map do |arm, visits|
      visits.map { |values| { credit: { arm => credit }, values: Array(values).map(&:to_f) } }
    end
  end

  # P(best) of every arm over the estimate's linear value at `parameters` and `weights` (one part, weight 1, by
  # default), 20 000 draws at seed 7.
  def p_best(estimate, parts: %i[converted], parameters: { w: { fixed: 1.0 } }, weights: ->(drawn) { { converted: drawn.fetch(:w) } },
             draws: 20_000, seed: 7)
    summary = linear_value.summary(arms: estimate[:means].keys, parts: parts, means: estimate[:means],
                                   covariance: estimate[:covariance], parameters: parameters, weights: weights,
                                   draws: draws, rng: Random.new(seed), rounding: estimate.rounding)
    summary.transform_values { |figures| figures[:p_best] }
  end

  # One visit at 1 against three at 0, 1 and 0, one part.
  let(:walk_rows) { rows_of(a: [ 1.0 ], b: [ 0.0, 1.0, 0.0 ]) }
  let(:walk) { shared_credit.estimate(walk_rows, arms: %i[a b], parts: %i[converted]) }

  describe "the rule, by hand" do
    # By hand: n_a = 1, n_b = 3, N = 4; r_a = 1, r_b = 1/3; cov(r_a) = 0 (one visit), cov(r_b) = (1/9 + 4/9 + 1/9) / 9
    # = 2/27. m0 = (1 * 1 + 3 * 1/3) / 4 = 1/2; g_a = 1/2, g_b = -1/6; V0 = (1 * (0 + 1/4) + 3 * (3 * 2/27 + 1/36)) / 4
    # = (1/4 + 3/4) / 4 = 1/4, the variance of the four visits.
    it "tops an arm of one visit up to a strength of 3: a's 2 visits of the prior, b at 3 untouched" do
      # a: e = 2, means (2 * 1/2 + 1 * 1) / 3 = 2/3, block 0 + (2 * 1/4 + (2 * 1 / 3) * 1/4) / 9 = (1/2 + 1/6) / 9 = 2/27.
      topped = shared_credit.with_prior(walk, strength: 3)

      expect_close(topped, { means: { a: [ 2.0 / 3 ], b: [ 1.0 / 3 ] },
                             covariance: [ [ 2.0 / 27, 0.0 ], [ 0.0, 2.0 / 27 ] ],
                             weight_sums: { a: 1.0, b: 3.0 }, effective: { a: 1.0, b: 3.0 } })
      expect(topped[:means][:b]).to eq(walk[:means][:b])
      expect(topped[:covariance][1][1]).to eq(walk[:covariance][1][1])
    end

    it "tops both arms up to a strength of 20" do
      # a: e = 19, means (19 / 2 + 1) / 20 = 0.525, block (19 * 1/4 + (19 * 1 / 20) * 1/4) / 400 = 4.9875 / 400.
      # b: e = 17, means (17 / 2 + 3 * 1/3) / 20 = 0.475, block (3/20)^2 * 2/27 + (17 * 1/4 + (17 * 3 / 20) / 36) / 400
      #    = 1/600 + 4.3208333... / 400 = 4.9875 / 400 (the data is symmetric about 1/2).
      topped = shared_credit.with_prior(walk, strength: 20)

      expect_close(topped[:means], { a: [ 0.525 ], b: [ 0.475 ] })
      expect_close(topped[:covariance], [ [ 4.9875 / 400, 0.0 ], [ 0.0, 4.9875 / 400 ] ])
    end

    it "the walk's case: one visit against three read P(best) 0.994 and reads near even under the prior" do
      expect(p_best(walk)[:a]).to be >= 0.99
      expect(p_best(shared_credit.with_prior(walk, strength: 20))[:a]).to be_between(0.55, 0.7)
      expect(p_best(shared_credit.with_prior(walk, strength: 3))[:a]).to be_between(0.75, 0.85)
    end

    it "keeps the arms, their order, their names as given, weight sums, Kish's sizes and parts" do
      rows = rows_of("b" => [ 1.0 ], "a" => [ 0.0, 1.0 ])
      estimate = shared_credit.estimate(rows, arms: %w[b a c], parts: %w[converted])
      topped = shared_credit.with_prior(estimate, strength: 20)

      expect(topped).to be_a(shared_credit::Estimate)
      expect(topped.keys).to eq(%i[means covariance weight_sums effective])
      expect(topped[:means].keys).to eq(%w[b a c])
      expect(topped[:means]["c"]).to be_nil
      expect(topped[:covariance][2]).to eq([ 0.0, 0.0, 0.0 ])
      expect(topped[:covariance].map { |row| row[2] }).to eq([ 0.0, 0.0, 0.0 ])
      expect(topped[:weight_sums]).to eq(estimate[:weight_sums])
      expect(topped[:effective]).to eq(estimate[:effective])
      expect(topped.parts).to eq(%w[converted])
    end

    it "changes nothing it is given" do
      before = Marshal.load(Marshal.dump(walk.to_h))
      shared_credit.with_prior(walk, strength: 20)

      expect(walk.to_h).to eq(before)
    end

    it "a part every arm holds at one mean keeps it exactly, with no prior spread: tied arms split P(best) evenly" do
      estimate = shared_credit.estimate(rows_of(a: [ 0.55 ], b: [ 0.55 ], c: [ 0.55, 0.55 ]), arms: %i[a b c], parts: %i[converted])
      topped = shared_credit.with_prior(estimate, strength: 20)

      expect(topped[:means]).to eq(a: [ 0.55 ], b: [ 0.55 ], c: [ 0.55 ])
      expect(topped[:covariance].flatten.uniq).to eq([ 0.0 ])
      p_best(topped).each_value { |p| expect(p).to be_within(1e-12).of(1.0 / 3) }
    end
  end

  describe "against the Rational reference" do
    # Three visits, two arms, two parts; the middle visit credits both (shared_credit_spec.rb's shared_rows): Kish's
    # sizes a 9/5, b 25/17.
    let(:shared_rows) do
      [ { credit: { a: 2.0 }, values: [ 1.0, 0.0 ] },
        { credit: { a: 1.0, b: 1.0 }, values: [ 0.0, 2.0 ] },
        { credit: { b: 4.0 }, values: [ 1.0, 1.0 ] } ]
    end

    it "tops both arms of shared rows up (strength 2), the covariance between them scaled by both n / k" do
      estimate = shared_credit.estimate(shared_rows, arms: %i[a b], parts: %i[p q])

      expect_close(shared_credit.with_prior(estimate, strength: 2).slice(:means, :covariance),
                   reference(shared_rows, %i[a b], %i[p q], 2))
    end

    it "tops up only the arm below the strength (1.6: b at 25/17), a's means and block kept bit for bit" do
      estimate = shared_credit.estimate(shared_rows, arms: %i[a b], parts: %i[p q])
      topped = shared_credit.with_prior(estimate, strength: 1.6)

      expect_close(topped.slice(:means, :covariance), reference(shared_rows, %i[a b], %i[p q], 1.6))
      expect(topped[:means][:a]).to eq(estimate[:means][:a])
      expect(topped[:covariance][0..1].map { |row| row[0..1] }).to eq(estimate[:covariance][0..1].map { |row| row[0..1] })
      expect(topped[:covariance][0][2]).not_to eq(estimate[:covariance][0][2])
    end

    it "matches the reference on seeded shared-credit rows of three arms, one never credited" do
      [ 3, 11, 20_261_105 ].each do |seed|
        rng = Random.new(seed)
        rows = Array.new(14) do
          credit = { a: rng.rand < 0.5 ? 1.0 + rng.rand(4) : 0.0, b: rng.rand < 0.5 ? 1.0 + rng.rand(4) : 0.0 }
          credit[:a] = 2.0 if credit.values.all?(&:zero?)
          { credit: credit, values: [ rng.rand(3).to_f, rng.rand < 0.3 ? 1.0 : 0.0 ] }
        end
        estimate = shared_credit.estimate(rows, arms: %i[a b c], parts: %i[p q])

        [ 2, 6, 40 ].each do |strength|
          expect_close(shared_credit.with_prior(estimate, strength: strength).slice(:means, :covariance),
                       reference(rows, %i[a b c], %i[p q], strength))
        end
      end
    end

    it "reads an ordinary test's estimate from per-arm sums as it reads one from rows" do
      rows = rows_of({ a: [ [ 1, 0 ], [ 0, 2 ] ], b: [ [ 1, 3 ], [ 0, 0 ], [ 2, 1 ] ] }, 2.0)
      from_rows = shared_credit.estimate(rows, arms: %i[a b], parts: %i[p q])
      sums = {
        a: { w: 4.0, w2: 8.0, a: { p: 2.0, q: 4.0 }, b: { p: 4.0, q: 8.0 }, c: { %i[p p] => 4.0, %i[p q] => 0.0, %i[q q] => 16.0 } },
        b: { w: 6.0, w2: 12.0, a: { p: 6.0, q: 8.0 }, b: { p: 12.0, q: 16.0 },
             c: { %i[p p] => 20.0, %i[p q] => 20.0, %i[q q] => 40.0 } }
      }
      from_sums = shared_credit.from_sums(sums, arms: %i[a b], parts: %i[p q])

      expect_close(shared_credit.with_prior(from_sums, strength: 5).slice(:means, :covariance),
                   reference(rows, %i[a b], %i[p q], 5))
      expect_close(shared_credit.with_prior(from_sums, strength: 5).slice(:means, :covariance),
                   shared_credit.with_prior(from_rows, strength: 5).slice(:means, :covariance))
    end
  end

  describe "where visits are plenty" do
    # Seeded rows: three arms of an ordinary test (each row credits its drawn arm, 1 / p, p from 0.2 to 0.6), two parts,
    # 200 to 400 rows an arm, every Kish's size well above 100.
    def plenty(seed)
      rng = Random.new(seed)
      rows = %i[a b c].flat_map do |arm|
        Array.new(200 + rng.rand(200)) do
          { credit: { arm => 1.0 / (0.2 + (0.4 * rng.rand)) }, values: [ rng.rand < 0.12 ? 1.0 : 0.0, rng.rand(3).to_f ] }
        end
      end
      shared_credit.estimate(rows, arms: %i[a b c], parts: %i[p q])
    end

    let(:value_inputs) do
      { parts: %i[p q], parameters: { base: { fixed: 1.0 }, share: { normal: [ 0.4, 0.1 ], clip: [ 0.0, 1.0 ] } },
        weights: ->(drawn) { { p: drawn.fetch(:base), q: drawn.fetch(:share) } } }
    end

    def draws_of(estimate, seed)
      linear_value.each_draw(arms: estimate[:means].keys, means: estimate[:means], covariance: estimate[:covariance],
                             draws: 300, rng: Random.new(seed), rounding: estimate.rounding, **value_inputs).to_a
    end

    it "answers the estimate's means, covariance and rounding as given when every arm is at or above the strength" do
      [ 1, 7, 20_261_105 ].each do |seed|
        estimate = plenty(seed)
        expect(estimate[:effective].values.min).to be > 100

        [ 0, 20, 100 ].each do |strength|
          topped = shared_credit.with_prior(estimate, strength: strength)

          expect(topped[:means]).to eq(estimate[:means])
          expect(topped[:covariance]).to eq(estimate[:covariance])
          expect(topped.rounding).to eq(estimate.rounding)
          expect(topped.parts).to eq(estimate.parts)
        end
      end
    end

    it "leaves every seeded draw and summary the same, bit for bit" do
      [ 1, 7, 20_261_105 ].each do |seed|
        estimate = plenty(seed)
        topped = shared_credit.with_prior(estimate, strength: 20)

        expect(draws_of(topped, seed)).to eq(draws_of(estimate, seed))
        expect(linear_value.summary(arms: %i[a b c], means: topped[:means], covariance: topped[:covariance], draws: 2_000,
                                    rng: Random.new(seed), rounding: topped.rounding, **value_inputs))
          .to eq(linear_value.summary(arms: %i[a b c], means: estimate[:means], covariance: estimate[:covariance],
                                      draws: 2_000, rng: Random.new(seed), rounding: estimate.rounding, **value_inputs))
      end
    end

    it "an arm at exactly the strength is not topped up; just above its size it is" do
      expect(walk[:effective][:b]).to eq(3.0)
      expect(shared_credit.with_prior(walk, strength: 3)[:means][:b]).to eq(walk[:means][:b])
      expect(shared_credit.with_prior(walk, strength: 3.5)[:means][:b]).not_to eq(walk[:means][:b])
    end

    it "in an ordinary test, an arm at or above the strength draws the same values beside an arm topped up, either order" do
      [ 1, 7, 20_261_105 ].each do |seed|
        rng = Random.new(seed)
        big = Array.new(300) { [ rng.rand < 0.12 ? 1.0 : 0.0, rng.rand(3).to_f ] }
        [ %i[big small], %i[small big] ].each do |arms|
          estimate = shared_credit.estimate(rows_of(big: big, small: [ [ 1.0, 2.0 ], [ 0.0, 0.0 ], [ 1.0, 0.0 ] ]), arms: arms, parts: %i[p q])
          topped = shared_credit.with_prior(estimate, strength: 20)
          column = arms.index(:big)

          expect(topped[:means][:big]).to eq(estimate[:means][:big])
          expect(topped[:means][:small]).not_to eq(estimate[:means][:small])
          expect(draws_of(topped, seed).map { |row| row[column] }).to eq(draws_of(estimate, seed).map { |row| row[column] })
          expect(draws_of(topped, seed).map { |row| row[1 - column] }).not_to eq(draws_of(estimate, seed).map { |row| row[1 - column] })
        end
      end
    end

    it "still finds a real difference at plenty of visits: 600 visits at 0.10 against 600 at 0.15" do
      rng = Random.new(5)
      rows = rows_of(a: Array.new(600) { rng.rand < 0.10 ? 1.0 : 0.0 }, b: Array.new(600) { rng.rand < 0.15 ? 1.0 : 0.0 })
      estimate = shared_credit.estimate(rows, arms: %i[a b], parts: %i[converted])

      expect(p_best(shared_credit.with_prior(estimate, strength: 20))).to eq(p_best(estimate))
      expect(p_best(estimate)[:b]).to be >= 0.95
    end
  end

  # P(best) of the version with a handful of visits, by goal and visit count, at strength 20 (the Kish's size
  # contrast_for_fit leaves out by default) and without the prior. A row: [goal, the handful's visits, the other's
  # visits, whether the reading without the prior is at or beyond 0.98 either way]. Without the prior a version whose
  # visits hold one value is drawn as a point, so it reads near certain whenever the other's spread does not cover it;
  # under the prior a version of one visit reads near even (0.2 to 0.8), and no version of three visits or fewer
  # reaches the 0.95 an automatic win asks, or falls to 0.05.
  describe "a table over visit counts and goals" do
    goals = {
      # A value of one part, 0 or 1 a visit (a conversion per visit).
      one_part: { parts: %i[converted], parameters: { w: { fixed: 1.0 } }, weights: ->(drawn) { { converted: drawn.fetch(:w) } } },
      # A value of two parts at uncertain prices: a conversion at 1 and a second outcome at a price around 0.3.
      two_parts: { parts: %i[converted returned], parameters: { converted: { fixed: 1.0 }, returned: { normal: [ 0.3, 0.1 ], clip: [ 0.0, 1.0 ] } },
                   weights: ->(drawn) { drawn } },
      # A value of one part that is an amount a visit.
      amounts: { parts: %i[amount], parameters: { w: { fixed: 1.0 } }, weights: ->(drawn) { { amount: drawn.fetch(:w) } } }
    }
    zeros = ->(count) { Array.new(count, 0.0) }
    table = [
      [ :one_part, [ 1.0 ], [ 0.0, 1.0, 0.0 ], true ],                       # without: 0.9941
      [ :one_part, [ 1.0 ], [ 0.0 ], true ],                                 # 1.0
      [ :one_part, [ 0.0 ], [ 1.0, 0.0, 0.0 ], false ],                      # 0.1136
      [ :one_part, [ 1.0 ], zeros.call(27) + ([ 1.0 ] * 3), true ],          # 1.0
      [ :one_part, [ 0.0 ], zeros.call(95) + ([ 1.0 ] * 5), true ],          # 0.0101
      [ :one_part, [ 1.0, 1.0 ], [ 0.0, 1.0, 0.0 ], true ],                  # 0.9941
      [ :one_part, [ 1.0, 0.0 ], zeros.call(9) + [ 1.0 ], false ],           # 0.8660
      [ :one_part, [ 1.0, 1.0, 1.0 ], [ 0.0, 0.0, 1.0 ], true ],             # 0.9941
      [ :one_part, [ 1.0, 1.0, 1.0 ], [ 0.0, 0.0, 0.0 ], true ],             # 1.0
      [ :one_part, [ 1.0, 1.0 ], zeros.call(190) + ([ 1.0 ] * 10), true ],   # 1.0
      [ :two_parts, [ [ 1.0, 1.0 ] ], [ [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 0.0, 1.0 ] ], true ],   # 0.9997
      [ :two_parts, [ [ 0.0, 1.0 ] ], [ [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 0.0, 0.0 ] ], false ],  # 0.4540
      [ :two_parts, [ [ 1.0, 0.0 ], [ 1.0, 1.0 ] ],
        Array.new(20) { |i| [ (i % 7).zero? ? 1.0 : 0.0, (i % 5).zero? ? 1.0 : 0.0 ] }, true ], # 1.0
      [ :amounts, [ 0.55 ], [ 0.55, 0.0, 0.0 ], true ],                      # 0.9941
      [ :amounts, [ 2.5 ], [ 0.0, 1.2, 0.0, 0.4 ], true ],                   # 1.0
      [ :amounts, [ 0.0 ], [ 0.3, 0.0, 0.9, 0.0, 0.0, 0.2 ], false ]         # 0.0378
    ]

    table.each do |goal, handful, other, certain_without|
      it "#{goal}: #{handful.size} visit(s) #{handful.inspect} against #{other.size}" do
        inputs = goals.fetch(goal)
        estimate = shared_credit.estimate(rows_of(small: handful, other: other), arms: %i[small other], parts: inputs[:parts])
        without = p_best(estimate, **inputs)[:small]
        with = p_best(shared_credit.with_prior(estimate, strength: 20), **inputs)[:small]

        expect(without >= 0.98 || without <= 0.02).to be(certain_without), "without the prior: #{without}"
        expect(with).to be_between(0.2, 0.8) if handful.size == 1
        expect(with).to be < 0.95
        expect(with).to be > 0.05
      end
    end

    # Two versions of one true value (0/1 a visit, at a rate of 0.1 or 0.3): one of 1 to 3 visits, the other of 3 to
    # 50. Without the prior the pair often reads decided (one version's P(best) at or above 0.95): when this was
    # written, 23, 41, 50, 9 and 35 of the rows' 50 seeded datasets. Under the prior none did; at most one is allowed.
    it "two versions of one true value read decided under the prior at most once in 50 datasets per row" do
      decided = ->(estimate) { p_best(estimate, draws: 1_000).values.any? { |p| p >= 0.95 } }
      rows = [ [ 1, 3, 0.3 ], [ 1, 50, 0.1 ], [ 1, 50, 0.3 ], [ 2, 3, 0.3 ], [ 3, 50, 0.1 ] ]
      counts = rows.map do |small, other, rate|
        data = Random.new((small * 1_000) + other + (rate * 10).round)
        Array.new(50) do
          visits = { small: Array.new(small) { data.rand < rate ? 1.0 : 0.0 }, other: Array.new(other) { data.rand < rate ? 1.0 : 0.0 } }
          estimate = shared_credit.estimate(rows_of(visits), arms: %i[small other], parts: %i[converted])
          [ decided.call(estimate), decided.call(shared_credit.with_prior(estimate, strength: 20)) ]
        end.transpose.map { |flags| flags.count(true) }
      end

      expect(counts.map(&:first).max).to be >= 40
      expect(counts.map(&:last)).to all(be <= 1)
    end
  end

  describe "what LinearValue takes" do
    it "is exactly symmetric and is drawn from, at seeded rows whose two parts are one offset apart (lower rank)" do
      [ 2, 9, 20_261_105 ].each do |seed|
        rng = Random.new(seed)
        rows = Array.new(9) do
          value = rng.rand(4).to_f
          credit = rng.rand < 0.3 ? { a: 1.0 + rng.rand, b: 1.0 + rng.rand } : { %i[a b c].sample(random: rng) => 1.0 + rng.rand(3) }
          { credit: credit, values: [ value, value + 1.0 ] }
        end
        rows += %i[a b c].map { |arm| { credit: { arm => 1.0 }, values: [ 1.0, 2.0 ] } }
        estimate = shared_credit.estimate(rows, arms: %i[a b c], parts: %i[p q])
        topped = shared_credit.with_prior(estimate, strength: 30)
        covariance = topped[:covariance]

        expect(covariance).to eq(covariance.transpose)
        expect(topped.rounding).to be >= estimate.rounding
        summary = linear_value.summary(arms: %i[a b c], parts: %i[p q], means: topped[:means], covariance: covariance,
                                       parameters: { w: { fixed: 1.0 } }, weights: ->(drawn) { { p: drawn[:w], q: drawn[:w] } },
                                       draws: 500, rng: Random.new(seed), rounding: topped.rounding)
        expect(summary.values.sum { |figures| figures[:p_best] }).to be_within(1e-12).of(1.0)
      end
    end

    it "reads an estimate read back from JSON as it reads the Estimate, naming no parts" do
      parsed = JSON.parse(JSON.generate(walk))
      from_json = shared_credit.with_prior(parsed, strength: 20)
      from_estimate = shared_credit.with_prior(walk, strength: 20)

      expect(from_json[:means]).to eq("a" => from_estimate[:means][:a], "b" => from_estimate[:means][:b])
      expect(from_json[:covariance]).to eq(from_estimate[:covariance])
      expect(from_json.parts).to be_nil
    end
  end

  describe "refusals" do
    it "refuses a strength below 0 or that is not a finite number" do
      [ -1, -0.5, Float::NAN, Float::INFINITY, "20", nil, Complex(1, 1) ].each do |strength|
        expect { shared_credit.with_prior(walk, strength: strength) }.to raise_error(ArgumentError, /strength/)
      end
    end

    it "refuses an estimate without its Kish's sizes or weight sums, or one that is not an estimate" do
      expect { shared_credit.with_prior(walk.to_h.except(:effective), strength: 20) }
        .to raise_error(ArgumentError, /no effective sizes/)
      expect { shared_credit.with_prior(walk.to_h.except(:weight_sums), strength: 20) }
        .to raise_error(ArgumentError, /no weight sums/)
      expect { shared_credit.with_prior(walk.to_h.merge(weight_sums: [ 1.0, 3.0 ]), strength: 20) }
        .to raise_error(ArgumentError, /weight sums must be a Hash/)
      expect { shared_credit.with_prior([ walk ], strength: 20) }.to raise_error(ArgumentError, /must be a Hash/)
    end

    it "refuses an estimate whose figures leave a Float's range under the prior" do
      expect { shared_credit.with_prior(walk.to_h.merge(means: { a: [ 1e300 ], b: [ -1e300 ] }), strength: 20) }
        .to raise_error(ArgumentError, /the sums leave a Float's range/)
    end

    it "refuses an arm with means but a Kish's size of 0" do
      expect { shared_credit.with_prior(walk.to_h.merge(effective: { a: 0.0, b: 3.0 }), strength: 20) }
        .to raise_error(ArgumentError, /arm :a has means but a Kish's size of 0/)
    end
  end
end
