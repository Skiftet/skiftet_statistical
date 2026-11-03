# frozen_string_literal: true

require "zlib"

# LO3: the acceptance of 0.4.0's additions to the argmax modules (01 GB7, GB11, GB12), written from the contract before
# their implementer, LA4. Every example is red until LA4 lands: BetaArgmax.position is private in 0.3.0, and
# BetaArgmax.expected_maximum and Bandit::NormalArgmax do not exist yet.
#
# BetaArgmax.position(uniform, alpha, beta) is made public, its bisection replaced by a faster step on the same
# distribution function, and answers what 0.3.0's private position answered. THE RECORD below is 0.3.0's private
# position, read on the 4c92b05 tree through `send`, for the 30 members of the BetaArgmax pins (every member of their
# probability tables, the largest total, the smallest parameter at each end) and three sharp members, 1 000 uniforms
# each: (j + 1/2) / 995 for j = 0 to 994, then 1e-300, 1e-30, 1e-10, 1e-5 and 1 - 1e-5. Every uniform is plain
# arithmetic on exact inputs, so it is the same Float on every machine.
#
# How the record is kept: each position p goes to the scale s = p when |p| <= 1, else sign(p) * (1 + log|p|), on which
# the tolerance below is even; s is stored in steps of 1e-13 (so a position comes back within 6e-14 of its scale); the
# 995 grid values of a member are stored by their sixth differences, the five others as they are; each Integer is
# zigzag-folded and written seven bits a byte, low first, then the bytes are deflated and written in base64. Before it
# was written down, the record was decoded with the code below and checked against the positions it came from.
#
# THE TOLERANCE, one tiered bound (01 GB7), on |position - recorded| / max(1, |recorded|), since a position near 0 is
# log(2 * x) of a draw x near 1/2 and its absolute error is that draw's relative error:
# - 1e-8 for the two members at BetaArgmax::SMALLEST_PARAMETER, [1e-6, 1] and [1, 1e-6]: there the distribution
#   function is too flat to fix a position to 1e-12 (0.3.0's own positions read up to 7.8e-10 from the closed form
#   x = u**(1 / alpha), and a bisection in log x on the same distribution function lands up to 9.7e-11 from them);
# - 1e-9 (the library's LGAMMA_RELATIVE) for every other member whose alpha + beta exceeds 10**4: the distribution
#   function's constant comes from Math.lgamma, whose last bit can differ between math libraries, and a shift of 4
#   units in that bit moved these positions by up to 2.7e-11 (measured on the record);
# - 1e-12 for every other member.
#
# BetaArgmax.expected_maximum (E[max] of independent Betas, 1 - the integral of the product of their distribution
# functions) is checked against exact answers worked in fractions: whole members' distribution functions are
# polynomials; one member of any parameters beside them is a sum of its moments E[X**m] = product over r < m of
# (a + r) / (a + b + r); Beta(a, 1) members have a maximum with distribution function x**S (S the sum of the a); for
# Beta(1, b) members the integral is the sum over subsets T of (-1)**|T| / (1 + the sum of b over T). Each was also
# matched by a plain trapezoid sum to 3e-10 or better. Tolerance 1e-8, BetaArgmax's own bound.
#
# Bandit::NormalArgmax takes Normals [[mean, standard_deviation], ...] by BetaArgmax's rule. Its references: two
# members by the closed forms (P(the second is larger) = Phi(d), d = (m2 - m1) / sqrt(s1**2 + s2**2); E[max] = m1 *
# Phi(-d) + m2 * Phi(d) + sqrt(s1**2 + s2**2) * phi(d)); k equal members 1 / k each and mean + sd * e_k (e_2 = 1 /
# sqrt(pi), e_3 = 3 / (2 sqrt(pi))); the rest by Simpson's rule on 400 000 intervals over each set's range of 14
# standard deviations, whose own error is under 2e-13. Tolerance 1e-8 absolute, amounts near 5 a view and near 10**6
# included.
module ArgmaxAdditionsOracle
  # The record's members, in its order.
  MEMBERS = [
    [ 1.0, 1.0 ], [ 2.0, 1.0 ], [ 0.5, 1.0 ], [ 1.5, 1.0 ], [ 2.5, 1.0 ], [ 3.0, 3.0 ], [ 2.0, 8.0 ], [ 3.0, 7.0 ],
    [ 5.0, 5.0 ], [ 41.0, 361.0 ], [ 52.0, 350.0 ], [ 2.0, 30.0 ],
    [ 0.5, 0.5 ], [ 1.0, 0.5 ],
    [ 0.4, 0.4 ], [ 5.0, 3.0 ], [ 1.0, 0.05 ], [ 2.0, 7.0 ], [ 0.3, 0.1 ], [ 4.0, 1.0 ], [ 2.0, 2.0 ], [ 0.02, 0.02 ],
    [ 9.0, 2.0 ],
    [ 1_500_000.0, 1_500_000.0 ], [ 1_501_100.0, 1_498_900.0 ],
    [ 1e-6, 1.0 ], [ 1.0, 1e-6 ],
    [ 3.0, 300.0 ], [ 41.0, 960.0 ], [ 4_000.0, 96_000.0 ]
  ].freeze
  # The two members at the smallest parameter BetaArgmax takes (its SMALLEST_PARAMETER, 1e-6).
  AT_SMALLEST_PARAMETER = [ [ 1e-6, 1.0 ], [ 1.0, 1e-6 ] ].freeze
  GRID = 995
  EDGES = [ 1e-300, 1e-30, 1e-10, 1e-5, 1.0 - 1e-5 ].freeze
  STEP = 1e-13
  DIFFERENCES = 6

  module_function

  def uniforms
    Array.new(GRID) { |j| (j + 0.5) / GRID } + EDGES
  end

  def tolerance(alpha, beta)
    return 1e-8 if AT_SMALLEST_PARAMETER.include?([ alpha, beta ])
    return 1e-9 if alpha + beta > 1e4

    1e-12
  end

  # One Array of 1 000 recorded positions per member, in MEMBERS' order.
  def recorded
    @recorded ||= integers(Zlib::Inflate.inflate(RECORD.unpack1("m")).bytes).each_slice(GRID + EDGES.size).map do |slice|
      grid = slice.first(GRID)
      DIFFERENCES.times do
        sum = 0
        grid = grid.map { |difference| sum += difference }
      end
      (grid + slice.drop(GRID)).map { |steps| from_scale(steps * STEP) }
    end
  end

  def integers(bytes)
    out = []
    value = 0
    shift = 0
    bytes.each do |byte|
      value |= (byte & 0x7f) << shift
      if byte < 0x80
        out << ((value >> 1) ^ -(value & 1))
        value = 0
        shift = 0
      else
        shift += 7
      end
    end
    out
  end

  def from_scale(scale)
    return scale if scale.abs <= 1.0

    scale.positive? ? Math.exp(scale - 1.0) : -Math.exp(-scale - 1.0)
  end

  def seconds
    start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    yield
    Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
  end

  RECORD = <<~RECORD
    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
    trDOUD8vkukfvN6iB4C3qnwgMiKYCHitRhQ4ixWQPmLBoKeRgMh0RkTCCRQa
    C24JcxzpOtbq1hYP04Onbv8ntqX6Cqwk6QYtG2+Ejp4bXWDSO/QOo8vq0mLb
    xrOKDnsOmfwu4sawZD4NpDEca7AZUxWDhMiYASY5Be8tSSPgaA2jkk5thuKB
    GbsWI0/sj8lKmBICSCY0kn5zDUWTKAeKUo0sq1UVljSOVKNqFAvmTuyCFJXT
    2vRTlU0Qyk5XIDjEJLESwHZBko0iBxHp/FZkicNeRtKFhYepgoaBDDYIHKAU
    0YpPQcBihlCiKIHRWHlFohhJAlZkV6NUo1WNEp3QSQkhckCviUABzqsY3Xqn
    3ej1mt0R2pl2/0zjIvvMqNkRC0LeDXvl7cBv/F++9sM3f/X+17/46b9+9fcU
    4hAywVKwE+SDjZghCsEQ/kmB7SAWjoKNMB6egMXwArwHe+B6VIi2oWHUjjZS
    26gO6ibVQBfS6Uw+08oUYqeQxPVwq/gkYb+wX0yVcuQ1yrBySrtFl2HYbUw3
    D1vi7ROOVneGb1VgVWhqZEd02azYeecXHny1+u2k7+/5ZNXXd36f+e/rKaAJ
    71EJONgEtp0DGQ/AmgxYVgnPX4TlKSh+H9oyiVqPUZmF9NUkJieOvbOS69nH
    H1shFqbJGxrUzJ36Y09Mh3fY19/33LkUklIVc6L+pfQzb+anfHp13V8O5qwG
    2zKawEDDY5AV2wLvD9ejjmWXqLrLR5iUuE18+so6eee2Z4ay1uPuw0UN00fj
    J5bcIVksJ0gqSzvJZ9lAklpWkcyW/SS9ZWAqyOXmVKjLjUNdW2o2+tueTLbs
    v2RJWB93svOQ/tp/i/+f/jcKYN2zmu7zu3V70lsLUh7/KG9f/MX6/kpw4vmx
    S8v2VoPGFbeyatf+5PJEb/yF9Ya7aweWbdLtfXLo4ioYtzFz36vbtze2mJ+f
    vZDBVw6W5lDJ+WUrYHfO3dXgeWzth8c77sxq6brsLdhVaRrcvktZt/QxvzYn
    nt00lkSv3tiFdhTegPsepcK1W06Agvx8sH3o3/UTXzzJfr93dFF/4qztl8Lz
    tvnX1TjHz1oO3TWsu6B9vkLpj5WaC4XGW9z4Mbb0CLP3PH1rjErcRm3tRjXb
    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
    I7p69Tl2w5llQl15tXynIU1/ouumdaCuyb996/kzIc83pg3vhPdzLzafYQfH
    tubmMRfqCkdevJ59ePMv1sR/vnv9d6o3WvLOmNtuv3m6/z/bU3aCvNpr4MmT
    ZHji2MSBo6D2RnFyBtLUTT0cn3rqKX4xH7+pBL+5o/8/7bffzD9jrtpoqVn/
    ndXxn18jW+4kn/Fg6uP6pj46dhPZjRKyR/14147jXezFu1qLdzkX73oVbkIe
    bsoB3KQK3LQzuIk7cFP34iavxU1/gg/BQ3woKh797m7Jp4fS3yvd93LV+Ny4
    tOm3Doc/TApsqvOUHnacWW+9WG4aHNFv6tXmPlNu7ZT3ToqnbwhxF/hnV7gV
    Q2zLKra/gel/QA+W0fWDVHcdVZpMXetCp3ehlRloRz8sOAlrquG+XJiWCLsf
    gP4rILENrG0EmypAbxFIywJxaeBRLDj57do/rP1l8uebP970gx3vVb65/dWR
    RXdf6n6hflby9P6o3Ij60GeBXv9e7w33OecKR6kt0XrInGt6Zniuv6K7pX2k
    LlPjlBa5RUqTCsQRYblwjt/Mt3Cl3CC7gy1gTzO1TBozSO+lN9OJ9A2qmaqn
    xtFa6iI6iWrxbRyuQ3dhN0aSa1hBNsJsuBKmwXHQi/HkOLgCzoEacBJsnQrB
    zcVqMhmsxQqyAHxWAH6UDd5ZBj785sM/Lf7Zpwu/mv9xzGcx78QsCZ/7luUd
    z/zgGcHTYnTzHUGeUJ8u0uaymYJ1umnQZxNtBtHHzDDO0k+jVAvQqcAhBQuy
    ResBlEtgLZKI1ST02HWyUScaLVgoQhGSxFNJ8UW5tECQ7TKnNXh5aJWxi9Rq
    BAU7RAma8Z1kB3YnMIfxYVq73m4NKM7Z9OtM2AvU+/BNb8iCqHkh70cteO31
    F3487zfRf3z/x7//6M+/+c/v/1UIviQLjGWC/aAQa4cT+Oc+aMV4cg9kwiTs
    D7dhJDkKh+AEfAoPYuVwE91B+dROqhXrhm30TnoNk8HcY5ayl9lU7BAL+ad8
    u7Be3CXdlvKVO8ou7RbddX2J8aapHXvEo4509x7vrqD2kK7wp9Ouzrj/wqn5
    N19OerNwye4ftX5e/Zudfx+OBwMrQf5GsKUGnG8F+d2g6TmoK4BJR+DhQdha
    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
    myxmQCaTlCa3yQuInKC3GCZJgHU4HawEqDwqGPaGQwlBFyWbnOFiKU0HaIQi
    ZpE9BlSrgIgdVg+v0KbogFfwNgZpHFlmFCgPy1kYwUJrNtYUJMHBh40xelhN
    sY8xkpASI8dpTMDGhQWPS3NEOZinBZ8v4IoKO+I8E6WJgfGWqeZkx5valKdj
    po5/esrE6RPfeO25X7709i/f+OD3b//p0w+y0Icz0T++Bir172WgXbeiOWgH
    KNj9gGKX4Gsa3o6y8TnQr4+AZZFkjh14H76EO/EQTqNL6WV0E91AX4/kczSA
    lr3PDAEH28c+YofYVu4Ml82f4Hv5zUKvUCb2ihXSbWmfvES5rrSp27Qy/RtL
    qbHILLeetnXZNzsPu1o817ylgUvB8vCemEXx7aOzk2+P7ZywadLh57Ne+mba
    oul73m6ZcfpX+z659Hnpl2tTUWcuerAE3axAAzWorQHtOoXu30JzMvGVYrx9
    J97ejQEF2yrpHR30tmxm9X7mSjq7pJVtKuOG0vg5g3xZvzB3plRWKedeU75p
    0h4csZwYMZf02A/OdJfd8nVuDW2+HreoMvna7KeK6yZvvPp65/H3d7V91tX0
    r4ymxWh7Noj7x+fR+pGF+Oaek3hm3Qb6/uHlzI4d1eyuziFuWUaq2NtxS95Y
    dU1flr/V3nu6zn9/uHR02YOHz90/WT6jvaU5FR0cPNCEMlq31uP9h24vYbZu
    r9rI5/YXn1VPjFxq9fZev7zyhYHZ89cVom0Lqi6fpVdnn6+8qBc3p1UfmIUW
    nO89U5b30YV5mb2P2mYc62rtrT/2X4caulZm7Xmz6sGJvIYuLXvRrKbuvqiC
    E/29ZfNK0KLKsryq5mpcu+ZYTuvaUpxWmtXTdDMNDV4dSm9utC1/8r6VkYvU
    RC7YH7n4BTKO42REOWRsW8go68l408nID5B7OEbuZhhuawXc3jDc5gW43aVw
    2xvg9i/ANCyF6dgJ01IH0zMM0/R4+wb6BkzbOpi+YzCNtTCd6TCtJ5v+tbPt
    s47j72+4+vryuslXZz9VUJlcfT2uY2toxS3fgZnuxT324yPmyBHL/CYt55qy
    olLOmymt6BdmD/L30vjGMm5xK3s5na3cz9RkM3Ud9NFKuuMxru3GtTvx5WI8
    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
    64DQPLDp0MJ24f8BzkYmGQ==
  RECORD
end

RSpec.describe "SkiftetStatistical argmax additions (0.4.0)" do
  def argmax = SkiftetStatistical::BetaArgmax
  def normal_argmax = SkiftetStatistical::Bandit::NormalArgmax
  def oracle = ArgmaxAdditionsOracle

  def expect_close(actual, expected, tolerance, label)
    expect(actual).to be_a(Float), "#{label}: #{actual.inspect} is not a Float"
    expect((actual - expected).abs).to be <= tolerance, "#{label}: #{actual} is not within #{tolerance} of #{expected}"
  end

  def expect_all_close(actual, expected, tolerance, label)
    expect(actual.size).to eq(expected.size), "#{label}: #{actual.size} values, not #{expected.size}"
    actual.zip(expected).each_with_index { |(a, e), i| expect_close(a, e, tolerance, "#{label}[#{i}]") }
  end

  describe "BetaArgmax.position" do
    it "LO3 (LA4) is public" do
      expect(argmax.position(0.5, 2.0, 2.0)).to be_within(1e-15).of(0.0)
      expect(argmax.singleton_class.public_method_defined?(:position)).to be(true)
    end

    it "LO3 (LA4) answers 0.3.0's positions on the pins' members, 1 000 uniforms each, within each member's tolerance" do
      expect(oracle.recorded.size).to eq(oracle::MEMBERS.size)
      expect(oracle::MEMBERS.select { |pair| pair.min <= argmax::SMALLEST_PARAMETER }).to eq(oracle::AT_SMALLEST_PARAMETER)
      uniforms = oracle.uniforms
      oracle::MEMBERS.zip(oracle.recorded).each do |(alpha, beta), recorded|
        tolerance = oracle.tolerance(alpha, beta)
        misses = uniforms.zip(recorded).filter_map do |uniform, pinned|
          position = argmax.position(uniform, alpha, beta)
          error = (position - pinned).abs / [ 1.0, pinned.abs ].max
          [ uniform, position, pinned, error ] unless position.is_a?(Float) && error <= tolerance
        end
        expect(misses).to be_empty,
                          "Beta(#{alpha}, #{beta}): #{misses.size} of 1 000 positions beyond #{tolerance}, " \
                          "the first #{misses.first(3).inspect}"
      end
    end

    it "LO3 (LA4) keeps 0.3.0's ends: minus infinity at a uniform of 0 and infinity at 1" do
      [ [ 2.0, 8.0 ], [ 0.3, 0.1 ], [ 1_500_000.0, 1_500_000.0 ] ].each do |alpha, beta|
        expect(argmax.position(0.0, alpha, beta)).to eq(-Float::INFINITY)
        expect(argmax.position(1.0, alpha, beta)).to eq(Float::INFINITY)
      end
    end

    # 0.3.0's position bisected the same distribution function 80 times, and took as long as BetaFunction.quantile's
    # bisection (a ratio of 0.92 to 1.00, measured on these members); the new step must take at most half as long.
    # Both are timed on the same 200 uniforms, five rounds taken in turn, the fastest round of each kept.
    it "LO3 (LA4) finds a position in at most half the time of a bisecting quantile on the same uniforms" do
      uniforms = Array.new(200) { |j| (j + 0.5) / 200 }
      [ [ 3.0, 300.0 ], [ 41.0, 960.0 ], [ 4_000.0, 96_000.0 ] ].each do |alpha, beta|
        uniforms.each do |uniform|
          argmax.position(uniform, alpha, beta)
          SkiftetStatistical::BetaFunction.quantile(uniform, alpha, beta)
        end
        rounds = Array.new(5) do
          [ oracle.seconds { uniforms.each { |uniform| argmax.position(uniform, alpha, beta) } },
            oracle.seconds { uniforms.each { |uniform| SkiftetStatistical::BetaFunction.quantile(uniform, alpha, beta) } } ]
        end
        position = rounds.map(&:first).min
        quantile = rounds.map(&:last).min
        expect(position).to be <= 0.5 * quantile,
                            format("Beta(%s, %s): 200 positions took %.4f s, 200 quantiles %.4f s", alpha, beta,
                                   position, quantile)
      end
    end
  end

  describe "BetaArgmax.expected_maximum" do
    {
      "one member: its mean" => [ [ [ 2.0, 8.0 ] ], 0.2 ],
      "Beta(a, 1) members: S / (S + 1) at S = 4.5" => [ [ [ 0.5, 1.0 ], [ 1.5, 1.0 ], [ 2.5, 1.0 ] ], 9.0 / 11.0 ],
      "Beta(1, b) members, a beta of 0.05 among them" => [ [ [ 1.0, 0.05 ], [ 1.0, 1.0 ], [ 1.0, 2.0 ] ],
                                                         0.9669541354534024 ],
      "three whole members" => [ [ [ 2.0, 8.0 ], [ 3.0, 7.0 ], [ 5.0, 5.0 ] ], 0.5229418134250547 ],
      "two sharp whole members" => [ [ [ 41.0, 361.0 ], [ 52.0, 350.0 ] ], 0.13056934594076644 ],
      "an alpha of 0.3 beside three whole members" => [ [ [ 0.3, 20.0 ], [ 1.0, 20.0 ], [ 2.0, 30.0 ], [ 5.0, 90.0 ] ],
                                                        0.08903251985698629 ],
      "an alpha of 0.02 beside four whole members" => [
        [ [ 0.02, 19.98 ], [ 3.0, 60.0 ], [ 4.0, 70.0 ], [ 6.0, 95.0 ], [ 2.0, 45.0 ] ], 0.08066474297473783
      ],
      "six members, one of them Beta(0.5, 4.5)" => [
        [ [ 12.0, 88.0 ], [ 15.0, 85.0 ], [ 9.0, 91.0 ], [ 20.0, 180.0 ], [ 3.0, 17.0 ], [ 0.5, 4.5 ] ], 0.21353379000310863
      ]
    }.each do |label, (params, exact)|
      it "LO3 (LA4) answers E[max] of #{label} within 1e-8 of the exact integral" do
        expect_close(argmax.expected_maximum(params), exact, 1e-8, label)
      end
    end

    it "LO3 (LA4) answers E[max] at or above the largest mean, whatever the members' order" do
      params = [ [ 12.0, 88.0 ], [ 15.0, 85.0 ], [ 9.0, 91.0 ], [ 20.0, 180.0 ], [ 3.0, 17.0 ], [ 0.5, 4.5 ] ]
      answer = argmax.expected_maximum(params)
      expect(answer).to be >= params.map { |alpha, beta| alpha / (alpha + beta) }.max
      expect_close(argmax.expected_maximum(params.reverse), answer, 1e-12, "reversed")
      expect_close(argmax.expected_maximum(params.rotate(2)), answer, 1e-12, "rotated")
    end

    it "LO3 (LA4) refuses members that are not [alpha, beta] pairs of finite numbers > 0" do
      argmax.method(:expected_maximum)
      [ [], [ [ 1.0 ] ], [ [ 0.0, 1.0 ] ], [ [ 1.0, -2.0 ] ], [ [ Float::NAN, 1.0 ] ], [ [ 1.0, Float::INFINITY ] ],
        [ [ 2.0, 8.0 ], [ 0.0, 1.0 ] ], "a" ].each do |params|
        expect { argmax.expected_maximum(params) }.to raise_error(ArgumentError), params.inspect
      end
    end
  end

  describe "Bandit::NormalArgmax.probabilities" do
    it "LO3 (LA4) answers two members by the closed form Phi(d), one member 1.0, and equal members 1 / k" do
      expect_all_close(normal_argmax.probabilities([ [ 5.0, 0.4 ], [ 5.3, 0.5 ] ]),
                       [ 1.0 - 0.6802940735983125, 0.6802940735983125 ], 1e-8, "two")
      expect_all_close(normal_argmax.probabilities([ [ 1_000_000.0, 10.0 ], [ 1_000_005.0, 20.0 ] ]),
                       [ 1.0 - 0.5884683631209393, 0.5884683631209393 ], 1e-8, "near 10**6")
      expect_all_close(normal_argmax.probabilities([ [ 5.0, 0.4 ] ]), [ 1.0 ], 1e-12, "one")
      expect_all_close(normal_argmax.probabilities([ [ 5.0, 0.4 ] ] * 3), [ 1.0 / 3 ] * 3, 1e-8, "three equal")
      expect_all_close(normal_argmax.probabilities([ [ 5.0, 0.01 ], [ 5.0, 0.01 ] ]), [ 0.5, 0.5 ], 1e-8, "narrow")
    end

    {
      "four amounts near 5 a view" => [
        [ [ 5.0, 0.4 ], [ 5.3, 0.5 ], [ 4.8, 0.3 ], [ 5.1, 0.8 ] ],
        [ 0.16748793058298564, 0.43470920285171183, 0.05189145526742245, 0.34591141129780234 ]
      ],
      "five members of mixed widths" => [
        [ [ 1.0, 1.0 ], [ 1.5, 0.5 ], [ 0.5, 2.0 ], [ 1.2, 0.2 ], [ 0.8, 0.7 ] ],
        [ 0.21208558053310708, 0.35158951674305056, 0.258377979413205, 0.08848221258826817, 0.08946471072231704 ]
      ],
      "six members" => [
        [ [ 5.0, 0.3 ], [ 5.2, 0.6 ], [ 4.9, 0.2 ], [ 5.4, 1.0 ], [ 4.6, 0.5 ], [ 5.1, 0.4 ] ],
        [ 0.07363303906749616, 0.2636690544459897, 0.02281624434725966, 0.4561316932293029, 0.03475399085735618,
          0.14899597805252307 ]
      ]
    }.each do |label, (params, reference)|
      it "LO3 (LA4) answers #{label} within 1e-8 of Simpson's rule, summing to 1, and mixes epsilon / k in" do
        answer = normal_argmax.probabilities(params)
        expect_all_close(answer, reference, 1e-8, label)
        expect(answer.sum).to be_within(1e-12).of(1.0)
        mixed = normal_argmax.probabilities(params, epsilon: 0.2)
        expect_all_close(mixed, answer.map { |p| (0.2 / params.size) + (0.8 * p) }, 1e-12, "#{label}, epsilon 0.2")
        expect_all_close(normal_argmax.probabilities(params, epsilon: 1.0), [ 1.0 / params.size ] * params.size, 1e-12,
                         "#{label}, epsilon 1")
      end
    end

    it "LO3 (LA4) refuses a standard deviation that is not a finite number > 0, a mean that is not finite, and an epsilon " \
       "outside [0, 1]" do
      normal_argmax
      [ [], [ [ 5.0 ] ], [ [ 5.0, 0.4 ], [ 5.0, 0.0 ] ], [ [ 5.0, 0.4 ], [ 5.0, -1.0 ] ], [ [ 5.0, 0.4 ], [ Float::NAN, 1.0 ] ],
        [ [ 5.0, 0.4 ], [ 5.0, Float::INFINITY ] ] ].each do |params|
        expect { normal_argmax.probabilities(params) }.to raise_error(ArgumentError), params.inspect
        expect { normal_argmax.expected_maximum(params) }.to raise_error(ArgumentError), params.inspect
      end
      expect { normal_argmax.probabilities([ [ 5.0, 0.4 ], [ 5.3, 0.5 ] ], epsilon: 1.5) }.to raise_error(ArgumentError)
      expect { normal_argmax.probabilities([ [ 5.0, 0.4 ], [ 5.3, 0.5 ] ], epsilon: -0.1) }.to raise_error(ArgumentError)
    end
  end

  describe "Bandit::NormalArgmax.expected_maximum" do
    {
      "one member: its mean" => [ [ [ 5.0, 0.4 ] ], 5.0 ],
      "two members, by Clark's closed form" => [ [ [ 5.0, 0.4 ], [ 5.3, 0.5 ] ], 5.432982811317365 ],
      "two members near 10**6" => [ [ [ 1_000_000.0, 10.0 ], [ 1_000_005.0, 20.0 ] ], 1_000_011.6427114896 ],
      "two equal narrow members: 5 + 0.01 / sqrt(pi)" => [ [ [ 5.0, 0.01 ], [ 5.0, 0.01 ] ], 5.005641895835478 ],
      "three equal members: 5 + 0.4 * 3 / (2 sqrt(pi))" => [ [ [ 5.0, 0.4 ] ] * 3, 5.338513750128654 ],
      "four amounts near 5 a view" => [ [ [ 5.0, 0.4 ], [ 5.3, 0.5 ], [ 4.8, 0.3 ], [ 5.1, 0.8 ] ], 5.6581335636537595 ],
      "five members of mixed widths" => [ [ [ 1.0, 1.0 ], [ 1.5, 0.5 ], [ 0.5, 2.0 ], [ 1.2, 0.2 ], [ 0.8, 0.7 ] ],
                                         2.1566726841589308 ],
      "six members" => [ [ [ 5.0, 0.3 ], [ 5.2, 0.6 ], [ 4.9, 0.2 ], [ 5.4, 1.0 ], [ 4.6, 0.5 ], [ 5.1, 0.4 ] ],
                         5.888649250113639 ]
    }.each do |label, (params, reference)|
      it "LO3 (LA4) answers E[max] of #{label} within 1e-8" do
        expect_close(normal_argmax.expected_maximum(params), reference, 1e-8, label)
      end
    end
  end
end
