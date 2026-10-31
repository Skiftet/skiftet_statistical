# frozen_string_literal: true

# SL15: the pins of GaussLegendre.nodes, BetaArgmax.probabilities and BetaFunction.log_density, then the edges of the
# three modules that SL10's acceptance leaves open. Every pinned value was checked by hand first, by references that
# share no code with the modules, and was written down before any output of the modules was compared with it:
# - the nodes and weights: Newton's method on P_n in 80-digit decimal arithmetic, from the cosine first guess, the
#   middle node of an odd rule set to 0 (they agree with Abramowitz and Stegun's table 25.4 at n = 7 and n = 20);
# - the log densities: exact factorials for whole parameters and Gamma(n + 1/2) = (2n)! / (4**n n!) sqrt(pi) for
#   half-whole ones, every x taken at its exact binary value, logs and pi at 60 digits;
# - the probabilities: for whole parameters, each member's density times the others' distribution functions is a
#   polynomial with whole coefficients over B(alpha, beta), integrated exactly (Rationals) over SL10's own range
#   [min_j Q_j(1e-10), max_j Q_j(1 - 1e-10)], whose ends are 0.2.0's unchanged quantiles at their exact binary values;
#   for two members the integral over [0, 1] equals Evan Miller's closed form for P(X > Y) to every digit shown. The
#   Beta(a, 1) family has the closed form a_c / sum of a at any range, and equal members take 1 / k each;
# - a member with beta < 1 (its density unbounded at 1, the case that read 0.93617 for Beta(1, 0.05) beside a uniform,
#   exactly 1 / 1.05): one Beta(a, b) beside m uniforms is the moment E[X**m] = product over r < m of (a + r) / (a + b +
#   r); Beta(a_i, 1) members beside one Beta(c, b) have a maximum with distribution function x**s, s = sum of a_i, and
#   which of them holds it does not depend on its value, so the last wins with B(c + s, b) / B(c, b); and one member of
#   any parameters beside members of whole parameters is an exact sum of its moments (Rationals, at the Floats' exact
#   binary values), because the whole members' distribution functions and their products are polynomials;
# - Beta(1, 1), Beta(1/2, 1/2) and Beta(1, 1/2): x = sin(t)**2 turns the three integrals into closed forms in pi,
#   checked by a plain midpoint sum in t;
# - two sharp whole members at the largest total taken: Evan Miller's sum, its 1.5 million terms by their ratio;
# - sharp pairs from 2 to 3 * 10**6: the same sum in 45-digit decimal, lgamma by Stirling at 45 digits, and one
#   distribution function, I_x(10**6, 2 * 10**6) at x = 1/3's exact binary value, as its binomial tail in 50-digit
#   decimal.
#
# The rule used to be pinned as written beside a member with beta < 1, 3e-5 off the exact answer. Those pins held the
# defect, and the exact answers replace them.
#
# One tolerance rule (SL15) for every Float compared with a pinned value: what goes through Math.lgamma (the
# probabilities and the log densities) at 1e-9 relative, LGAMMA_RELATIVE; the nodes and weights, plain arithmetic, at
# 1e-12 relative. A pinned 0.0, 1.0 or infinity is compared exactly, whichever the output. The exactness of SL10 beside
# a parameter below 1 and at the limits is held at its own bound, 1e-8 absolute, EXACT_ABSOLUTE: at most TAIL of a
# member stays out of the range at each end, on purpose, so an answer sits up to 4e-9 from the exact one. A pinned value
# is never edited.
RSpec.describe "SkiftetStatistical::BetaArgmax pins" do
  def argmax = SkiftetStatistical::BetaArgmax
  def rule = SkiftetStatistical::GaussLegendre
  def beta_function = SkiftetStatistical::BetaFunction

  def lgamma_relative = 1e-9

  def arithmetic_relative = 1e-12

  def exact_absolute = 1e-8

  def expect_pin(actual, pinned, relative, label)
    expect(actual).to be_a(Float), "#{label}: #{actual.inspect} is not a Float"
    if pinned.zero? || pinned == 1.0 || pinned.infinite?
      expect(actual).to eq(pinned), "#{label}: #{actual} is not exactly #{pinned}"
    else
      error = (actual - pinned).abs / pinned.abs
      expect(error).to be <= relative, "#{label}: #{actual} is #{error} from #{pinned}"
    end
  end

  def expect_pins(actual, pinned, relative, label)
    expect(actual.size).to eq(pinned.size), "#{label}: #{actual.size} values, not #{pinned.size}"
    actual.zip(pinned).each_with_index { |(a, p), i| expect_pin(a, p, relative, "#{label}[#{i}]") }
  end

  def expect_exact(actual, exact, label)
    expect(actual.size).to eq(exact.size), "#{label}: #{actual.size} values, not #{exact.size}"
    actual.zip(exact).each_with_index do |(a, e), i|
      expect((a - e).abs).to be <= exact_absolute, "#{label}[#{i}]: #{a} is not within 1e-8 of #{e}"
    end
  end

  def expect_refused(label)
    expect { yield }.to raise_error(ArgumentError), "#{label} was not refused"
  end

  describe "GaussLegendre.nodes" do
    # [index in ascending order of x, x, w]
    def expect_rule_pins(n, pins)
      nodes = rule.nodes(n).sort_by(&:first)
      expect(nodes.size).to eq(n)
      pins.each do |index, x, w|
        expect_pin(nodes[index][0], x, arithmetic_relative, "n = #{n}, node #{index}")
        expect_pin(nodes[index][1], w, arithmetic_relative, "n = #{n}, weight #{index}")
      end
    end

    it "SL15 pins the one-point rule: 0 exactly, weight 2" do
      expect_rule_pins(1, [ [ 0, 0.0, 2.0 ] ])
    end

    it "SL15 pins the seven-point rule, its middle node exactly 0" do
      expect_rule_pins(7, [
        [ 0, -0.9491079123427585245261897, 0.1294849661688696932706114 ],
        [ 1, -0.7415311855993944398638648, 0.2797053914892766679014678 ],
        [ 2, -0.4058451513773971669066064, 0.3818300505051189449503698 ],
        [ 3, 0.0, 0.4179591836734693877551020 ],
        [ 4, 0.4058451513773971669066064, 0.3818300505051189449503698 ],
        [ 5, 0.7415311855993944398638648, 0.2797053914892766679014678 ],
        [ 6, 0.9491079123427585245261897, 0.1294849661688696932706114 ]
      ])
    end

    it "SL15 pins the twenty-point rule, the panel rule of the argmax" do
      half = [
        [ 0.0765265211334973337546404, 0.1527533871307258506980843 ],
        [ 0.2277858511416450780804962, 0.1491729864726037467878287 ],
        [ 0.3737060887154195606725482, 0.1420961093183820513292983 ],
        [ 0.5108670019508270980043641, 0.1316886384491766268984945 ],
        [ 0.6360536807265150254528367, 0.1181945319615184173123774 ],
        [ 0.7463319064601507926143051, 0.1019301198172404350367501 ],
        [ 0.8391169718222188233945291, 0.0832767415767047487247581 ],
        [ 0.9122344282513259058677524, 0.0626720483341090635695065 ],
        [ 0.9639719272779137912676661, 0.0406014298003869413310400 ],
        [ 0.9931285991850949247861224, 0.0176140071391521183118620 ]
      ]
      pins = half.reverse.each_with_index.map { |(x, w), i| [ i, -x, w ] } +
             half.each_with_index.map { |(x, w), i| [ 10 + i, x, w ] }
      expect_rule_pins(20, pins)
    end

    it "SL15 pins the outer and inner nodes of the 64- and 200-point rules" do
      expect_rule_pins(64, [
        [ 0, -0.9993050417357721394569056, 0.0017832807216964329472961 ],
        [ 1, -0.9963401167719552793469245, 0.0041470332605624676352875 ],
        [ 30, -0.0729931217877990394495429, 0.0485754674415034269347991 ],
        [ 31, -0.0243502926634244325089558, 0.0486909570091397203833654 ],
        [ 32, 0.0243502926634244325089558, 0.0486909570091397203833654 ],
        [ 63, 0.9993050417357721394569056, 0.0017832807216964329472961 ]
      ])
      expect_rule_pins(200, [
        [ 0, -0.9999280712850699770492629, 0.0001845900974712974439676 ],
        [ 98, -0.0235009500614576372897825, 0.0156644150636108721932639 ],
        [ 99, -0.0078342911423063692774083, 0.0156682617158322548075664 ],
        [ 100, 0.0078342911423063692774083, 0.0156682617158322548075664 ],
        [ 197, 0.9990687218731521808173531, 0.0006749606344793727503029 ],
        [ 198, 0.9996210312809363963281364, 0.0004296466304510570728327 ],
        [ 199, 0.9999280712850699770492629, 0.0001845900974712974439676 ]
      ])
    end
  end

  describe "BetaFunction.log_density" do
    {
      "Beta(2, 3) at 1/4, log 27/16" => [ 0.25, 2.0, 3.0, 0.5232481437645478365168 ],
      "Beta(1/2, 1/2) at 1/2, log 2/pi" => [ 0.5, 0.5, 0.5, -0.4515827052894548647262 ],
      "Beta(5/2, 7/2) at 0.3, B = 3 pi / 256" => [ 0.3, 2.5, 3.5, 0.6041887036263176577231 ],
      "Beta(1/2, 3) near 1" => [ 0.9, 0.5, 3.0, -4.6170284492967498455202 ],
      "Beta(1/2, 3) near 0, where the density is unbounded" => [ 1e-6, 0.5, 3.0, 6.8432147578435659023404 ],
      "Beta(7, 2) at 3/4" => [ 0.75, 7.0, 2.0, 0.9129648949045730498873 ],
      "Beta(41, 361) below its mean" => [ 0.05, 41.0, 361.0, -4.980530578295009339654 ],
      "Beta(41, 361) at 0.1" => [ 0.1, 41.0, 361.0, 3.2811569868035254599097 ],
      "Beta(41, 361) above its mean" => [ 0.13, 41.0, 361.0, 1.5711689622578817073601 ],
      "Beta(2001, 18001) at 0.1" => [ 0.1, 2001.0, 18_001.0, 5.2367859165100866603362 ],
      "Beta(2001, 18001) at 0.0995" => [ 0.0995, 2001.0, 18_001.0, 5.2089255200218507499183 ],
      "Beta(2001, 18001) at 0.103" => [ 0.103, 2001.0, 18_001.0, 4.254167620335509524864 ]
    }.each do |label, (x, alpha, beta, pinned)|
      it "SL15 pins #{label}" do
        expect_pin(beta_function.log_density(x, alpha, beta), pinned, lgamma_relative, label)
      end
    end

    it "SL15 pins the flat density's log at exactly 0" do
      expect_pin(beta_function.log_density(0.3, 1.0, 1.0), 0.0, lgamma_relative, "Beta(1, 1)")
    end

    it "SL15 pins minus infinity outside (0, 1), the ends included" do
      [ 0.0, 1.0, -1e-300, 1.000000000000001, 5.0 ].each do |x|
        expect_pin(beta_function.log_density(x, 0.5, 3.0), -Float::INFINITY, lgamma_relative, "x = #{x}")
      end
    end
  end

  describe "BetaArgmax.probabilities" do
    {
      "Beta(1, 1) against Beta(2, 1): 1/3, 2/3" =>
        [ [ [ 1.0, 1.0 ], [ 2.0, 1.0 ] ], 0.0, [ 1.0 / 3, 2.0 / 3 ] ],
      "the same, epsilon 0.1: 0.35, 0.65" =>
        [ [ [ 1.0, 1.0 ], [ 2.0, 1.0 ] ], 0.1, [ 0.35, 0.65 ] ],
      "a Beta(a, 1) family with an unbounded density at 0: a / 4.5" =>
        [ [ [ 0.5, 1.0 ], [ 1.5, 1.0 ], [ 2.5, 1.0 ] ], 0.0, [ 1.0 / 9, 1.0 / 3, 5.0 / 9 ] ],
      "the same, epsilon 0.3" =>
        [ [ [ 0.5, 1.0 ], [ 1.5, 1.0 ], [ 2.5, 1.0 ] ], 0.3, [ 0.1 + (0.7 / 9), 0.1 + (0.7 / 3), 0.1 + (3.5 / 9) ] ],
      "three equal members: 1/3 each" =>
        [ [ [ 3.0, 3.0 ] ] * 3, 0.0, [ 1.0 / 3 ] * 3 ],
      "three flat whole members, exact over SL10's range" =>
        [ [ [ 2.0, 8.0 ], [ 3.0, 7.0 ], [ 5.0, 5.0 ] ], 0.0,
          [ 0.0464376203553076129277, 0.1552978138811182403742, 0.7982645657635741466981 ] ],
      "the same, epsilon 0.1" =>
        [ [ [ 2.0, 8.0 ], [ 3.0, 7.0 ], [ 5.0, 5.0 ] ], 0.1,
          [ 0.0751271916531101849683, 0.1731013658263397496701, 0.7517714425205500653616 ] ],
      "two sharp whole members, exact over SL10's range" =>
        [ [ [ 41.0, 361.0 ], [ 52.0, 350.0 ] ], 0.0, [ 0.1113695548607250600482, 0.8886304451392749399518 ] ],
      "two sharp members and a flat one, exact over SL10's range" =>
        [ [ [ 41.0, 361.0 ], [ 52.0, 350.0 ], [ 2.0, 30.0 ] ], 0.0,
          [ 0.1000640936533130698252, 0.8194753117171062856737, 0.0804605946295806445012 ] ],
      "the same, epsilon 0.25" =>
        [ [ [ 41.0, 361.0 ], [ 52.0, 350.0 ], [ 2.0, 30.0 ] ], 0.25,
          [ 0.1583814035733181357022, 0.6979398171211630475886, 0.1436787793055188167092 ] ]
    }.each do |label, (params, epsilon, pinned)|
      it "SL15 pins #{label}" do
        expect_pins(argmax.probabilities(params, epsilon: epsilon), pinned, lgamma_relative, label)
      end
    end

    {
      "Beta(1/2, 1/2) against Beta(1, 1), its mean: 1/2 each" =>
        [ [ [ 0.5, 0.5 ], [ 1.0, 1.0 ] ], [ 0.5, 0.5 ] ],
      "Beta(1, 1/2) against Beta(1, 1), its mean: 2/3, 1/3" =>
        [ [ [ 1.0, 0.5 ], [ 1.0, 1.0 ] ], [ 2.0 / 3, 1.0 / 3 ] ],
      "Beta(1, 1), Beta(1/2, 1/2) and Beta(1, 1/2), closed forms in pi" =>
        [ [ [ 1.0, 1.0 ], [ 0.5, 0.5 ], [ 1.0, 0.5 ] ],
          [ 0.5 - (8.0 / (9.0 * Math::PI)), 0.5 - (2.0 / (3.0 * Math::PI)), 14.0 / (9.0 * Math::PI) ] ]
    }.each do |label, (params, pinned)|
      it "SL15 pins #{label}" do
        expect_pins(argmax.probabilities(params), pinned, lgamma_relative, label)
      end
    end

    # P(the Beta is largest) = E[X**m]; each of the uniforms takes an equal part of the rest.
    def beside_uniforms(alpha, beta, uniforms)
      first = (0...uniforms).reduce(1.0) { |product, r| product * (alpha + r) / (alpha + beta + r) }
      [ first ] + ([ (1.0 - first) / uniforms ] * uniforms)
    end

    [ 0.05, 0.1, 0.2, 0.3, 0.5 ].each do |beta|
      it "SL10 is exact for a beta of #{beta} beside uniforms: the moments E[X**m], whatever alpha" do
        [ 0.05, 0.5, 1.0, 3.0, 40.0 ].product([ 1, 3, 10, 19 ]).each do |alpha, uniforms|
          expect_exact(argmax.probabilities([ [ alpha, beta ] ] + ([ [ 1.0, 1.0 ] ] * uniforms)),
                       beside_uniforms(alpha, beta, uniforms), "Beta(#{alpha}, #{beta}) beside #{uniforms}")
        end
      end
    end

    it "SL10 reads Beta(1, 0.05) beside one uniform at 1 / 1.05 and beside ten at E[X**10]" do
      expect_exact(argmax.probabilities([ [ 1.0, 0.05 ], [ 1.0, 1.0 ] ]), [ 1.0 / 1.05, 0.05 / 1.05 ], "beside one")
      beside_ten = argmax.probabilities([ [ 1.0, 0.05 ] ] + ([ [ 1.0, 1.0 ] ] * 10))
      expect(beside_ten[0]).to be_within(1e-8).of(0.8654034437863307)
    end

    it "SL10 stays exact with both parameters of a member far below 1" do
      expect_exact(argmax.probabilities([ [ 1e-5, 1e-5 ], [ 1.0, 1.0 ], [ 1.0, 1.0 ] ]), beside_uniforms(1e-5, 1e-5, 2),
                   "1e-5")
      expect_exact(argmax.probabilities([ [ 0.001, 0.02 ], [ 1.0, 1.0 ] ]), beside_uniforms(0.001, 0.02, 1),
                   "0.001, 0.02")
    end

    it "SL10 is exact with Betas pressed against both ends: power members beside one Beta" do
      # Beta(a_i, 1) beside Beta(c, b); P(last) = B(c + s, b) / B(c, b), through lgamma, shared by no module rule.
      log_beta = ->(a, b) { Math.lgamma(a)[0] + Math.lgamma(b)[0] - Math.lgamma(a + b)[0] }
      [ [ [ 0.05 ], 0.1, 3.0 ], [ [ 0.02 ], 0.3, 0.02 ], [ [ 0.01, 0.02 ], 0.04, 5.0 ], [ [ 0.5, 0.001 ], 2.0, 0.01 ],
        [ [ 1e-4, 1e-4, 2e-4 ], 1e-4, 1e-4 ], [ [ 2.0 ], 1e-3, 1e-3 ] ].each do |powers, c, b|
        total = powers.sum
        last = Math.exp(log_beta.call(c + total, b) - log_beta.call(c, b))
        expect_exact(argmax.probabilities(powers.map { |a| [ a, 1.0 ] } + [ [ c, b ] ]),
                     powers.map { |a| a / total * (1.0 - last) } + [ last ], "powers #{powers}, Beta(#{c}, #{b})")
      end
    end

    {
      "Beta(0.4, 0.4) against Beta(5, 5), 1/2 each by symmetry and by the exact sum" =>
        [ [ [ 0.4, 0.4 ], [ 5.0, 5.0 ] ], [ 0.5, 0.5 ] ],
      "Beta(0.4, 0.4) against Beta(5, 3)" =>
        [ [ [ 0.4, 0.4 ], [ 5.0, 3.0 ] ], [ 0.4258923169993950366095, 0.5741076830006049633905 ] ],
      "Beta(1, 0.05) against Beta(2, 7) and Beta(3, 3)" =>
        [ [ [ 1.0, 0.05 ], [ 2.0, 7.0 ], [ 3.0, 3.0 ] ],
          [ 0.9608021655944737381288, 0.0029354613733994294035, 0.0362623730321268324677 ] ],
      "Beta(0.3, 0.1) against three whole members" =>
        [ [ [ 0.3, 0.1 ], [ 1.0, 1.0 ], [ 4.0, 1.0 ], [ 2.0, 2.0 ] ],
          [ 0.6281574776896611473200, 0.0672860107664463099453, 0.2691440430657852397812, 0.0354124684781073029535 ] ],
      "Beta(0.02, 0.02) against Beta(9, 2)" =>
        [ [ [ 0.02, 0.02 ], [ 9.0, 2.0 ] ], [ 0.4834082774225165446208, 0.5165917225774834553792 ] ]
    }.each do |label, (params, exact)|
      it "SL10 matches the exact sum for #{label}" do
        expect_exact(argmax.probabilities(params), exact, label)
      end
    end

    it "SL10 is exact at the smallest parameters it takes" do
      smallest = argmax::SMALLEST_PARAMETER
      expect_exact(argmax.probabilities([ [ smallest, 1.0 ], [ 3.0 * smallest, 1.0 ] ]), [ 0.25, 0.75 ], "bottom")
      expect_exact(argmax.probabilities([ [ 1.0, smallest ], [ 1.0, 3.0 * smallest ] ]), [ 0.75, 0.25 ], "top")
    end

    it "SL10 is exact at the largest total it takes, against Evan Miller's sum" do
      params = [ [ 1_500_000.0, 1_500_000.0 ], [ 1_501_100.0, 1_498_900.0 ] ]
      expect_exact(argmax.probabilities(params), [ 1.0 - 0.815446260129, 0.815446260129 ], "3 * 10**6")
    end

    # A fraction stopped at 0.2.0's 300 steps read the worst of these 3.6e-8 off. Alpha and beta of the first member,
    # of the second, and P(the second is largest) by Miller's sum.
    it "SL10 is exact on sharp pairs from 2 to 3 * 10**6, where a fraction stopped at 300 steps falls short" do
      [
      [ 100_000, 1_900_000, 100_154, 1_899_846, 0.6380203143555281521777449 ],
      [ 100_000, 1_900_000, 100_308, 1_899_692, 0.7599330341656933878078184 ],
      [ 200_000, 1_800_000, 200_212, 1_799_788, 0.6380496699013394557416222 ],
      [ 200_000, 1_800_000, 200_424, 1_799_576, 0.7600100125573684683395515 ],
      [ 400_000, 1_600_000, 400_283, 1_599_717, 0.6382193583711024485068254 ],
      [ 400_000, 1_600_000, 400_566, 1_599_434, 0.7603139317348732700071670 ],
      [ 666_667, 1_333_333, 667_000, 1_333_000, 0.6380224432088592559038882 ],
      [ 666_667, 1_333_333, 667_334, 1_332_666, 0.7603323462599828121390868 ],
      [ 800_000, 1_200_000, 800_346, 1_199_654, 0.6380015615439968538799384 ],
      [ 800_000, 1_200_000, 800_693, 1_199_307, 0.7602911220373785825306698 ],
      [ 1_000_000, 1_000_000, 1_000_354, 999_646, 0.6383305912048321878096633 ],
      [ 1_000_000, 1_000_000, 1_000_707, 999_293, 0.7602168274632393836136692 ],
      [ 125_000, 2_375_000, 125_172, 2_374_828, 0.6378891198796412924484495 ],
      [ 125_000, 2_375_000, 125_345, 2_374_655, 0.7603609214739854372882574 ],
      [ 250_000, 2_250_000, 250_237, 2_249_763, 0.6380399740426085729665466 ],
      [ 250_000, 2_250_000, 250_474, 2_249_526, 0.7599993854960098446495127 ],
      [ 500_000, 2_000_000, 500_316, 1_999_684, 0.6380521178888416202601491 ],
      [ 500_000, 2_000_000, 500_632, 1_999_368, 0.7600397371520388018343740 ],
      [ 833_333, 1_666_667, 833_706, 1_666_294, 0.6382703006423739210885590 ],
      [ 833_333, 1_666_667, 834_078, 1_665_922, 0.7601205400032160826162026 ],
      [ 1_000_000, 1_500_000, 1_000_387, 1_499_613, 0.6380568824823469911965072 ],
      [ 1_000_000, 1_500_000, 1_000_775, 1_499_225, 0.7603501842866004362778814 ],
      [ 1_250_000, 1_250_000, 1_250_395, 1_249_605, 0.6380677813530202644100463 ],
      [ 1_250_000, 1_250_000, 1_250_791, 1_249_209, 0.7603696333350567271101373 ],
      [ 150_000, 2_850_000, 150_189, 2_849_811, 0.6383022004797186561723227 ],
      [ 150_000, 2_850_000, 150_377, 2_849_623, 0.7598334049385936768220230 ],
      [ 300_000, 2_700_000, 300_260, 2_699_740, 0.6382358642793207850814232 ],
      [ 300_000, 2_700_000, 300_520, 2_699_480, 0.7603281354777698102615044 ],
      [ 600_000, 2_400_000, 600_346, 2_399_654, 0.6379920301670538647679258 ],
      [ 600_000, 2_400_000, 600_693, 2_399_307, 0.7602594222145212179226387 ],
      [ 1_000_000, 2_000_000, 1_000_408, 1_999_592, 0.6380758794912666352480225 ],
      [ 1_000_000, 2_000_000, 1_000_816, 1_999_184, 0.7600939563148162979440573 ],
      [ 1_200_000, 1_800_000, 1_200_424, 1_799_576, 0.6380768449242947641452495 ],
      [ 1_200_000, 1_800_000, 1_200_849, 1_799_151, 0.7603591851751248385843677 ],
      [ 1_500_000, 1_500_000, 1_500_433, 1_499_567, 0.6381593301544492403301546 ],
      [ 1_500_000, 1_500_000, 1_500_866, 1_499_134, 0.7602435386021605028144729 ]
      ].each do |alpha_first, beta_first, alpha_second, beta_second, second|
        params = [ [ alpha_first.to_f, beta_first.to_f ], [ alpha_second.to_f, beta_second.to_f ] ]
        expect_exact(argmax.probabilities(params), [ 1.0 - second, second ], params.inspect)
      end
    end

    it "SL15 pins one member at exactly 1.0, whatever epsilon" do
      expect_pins(argmax.probabilities([ [ 0.3, 7.0 ] ]), [ 1.0 ], lgamma_relative, "epsilon 0")
      expect_pins(argmax.probabilities([ [ 0.3, 7.0 ] ], epsilon: 1.0), [ 1.0 ], lgamma_relative, "epsilon 1")
    end
  end

  describe "edges" do
    def three = [ [ 41.0, 361.0 ], [ 52.0, 350.0 ], [ 2.0, 30.0 ] ]

    # A pick that exploits, its uniforms given.
    def exploit(params, uniforms)
      argmax.pick(params, explore: 1.0, priorities: [ 0.0 ] * params.size, uniforms: uniforms, epsilon: 0.0)
    end

    it "SL10 answers every member's share at epsilon 1, each exactly 1 / k" do
      expect(argmax.probabilities([ [ 2.0, 8.0 ], [ 8.0, 2.0 ] ], epsilon: 1.0)).to eq([ 0.5, 0.5 ])
      expect(argmax.probabilities(three, epsilon: 1.0)).to all(be_within(1e-15).of(1.0 / 3))
    end

    it "SL10 sums to 1 within 1e-12 at every size up to 20, sharp and flat members mixed" do
      shapes = [ [ 41.0, 361.0 ], [ 2.0, 8.0 ], [ 20_000.0, 180_000.0 ], [ 0.6, 50.0 ], [ 12.0, 88.0 ] ]
      [ 2, 5, 11, 20 ].each do |k|
        params = Array.new(k) { |i| shapes[i % shapes.size].map { |v| v * (1.0 + (i / 50.0)) } }
        result = argmax.probabilities(params, epsilon: 0.1)
        expect(result.sum).to be_within(1e-12).of(1.0), "k = #{k}"
        expect(result.min).to be >= (0.1 / k) - 1e-12
      end
    end

    it "SL10 reads whole numbers as Floats: Integer members answer as Float ones" do
      expect(argmax.probabilities([ [ 2, 8 ], [ 3, 7 ] ], epsilon: 0)).to eq(argmax.probabilities([ [ 2.0, 8.0 ], [ 3.0, 7.0 ] ]))
      expect(argmax.greedy([ [ 1, 3 ], [ 3, 1 ] ])).to eq(1)
      expect(argmax.pick([ [ 2, 8 ], [ 8, 2 ] ], explore: 1, priorities: [ 1, 0 ], uniforms: [ 1, 0 ], epsilon: 0)).to eq(0)
    end

    it "SL10 answers members in the order given: a reordered call reorders the answer" do
      forward = argmax.probabilities(three)
      backward = argmax.probabilities(three.reverse)
      forward.zip(backward.reverse).each { |a, b| expect(a).to be_within(1e-13).of(b) }
    end

    it "SL10 matches the seedless pick: picks over uniform inputs land where the probabilities say" do
      # 1 500 picks from a seeded source of uniforms; each member's count lies within 5 standard deviations.
      params = [ [ 2.0, 8.0 ], [ 3.0, 7.0 ], [ 5.0, 5.0 ] ]
      epsilon = 0.3
      expected = argmax.probabilities(params, epsilon: epsilon)
      rng = Random.new(20_261_030)
      draws = 1_500
      counts = Array.new(params.size, 0)
      draws.times do
        pick = argmax.pick(params, explore: rng.rand, priorities: Array.new(params.size) { rng.rand },
                                   uniforms: Array.new(params.size) { rng.rand }, epsilon: epsilon)
        counts[pick] += 1
      end
      expected.each_with_index do |p, c|
        sd = Math.sqrt(draws * p * (1.0 - p))
        expect((counts[c] - (draws * p)).abs).to be <= 5 * sd, "member #{c}: #{counts[c]} picks, #{draws * p} expected"
      end
    end

    it "SL10 matches the seedless pick beside a beta far below 1: Beta(1, 0.05) beside a uniform" do
      # 10 000 picks, within 5 standard deviations; the rule that read 0.93617 for 1 / 1.05 lies 7 away.
      params = [ [ 1.0, 0.05 ], [ 1.0, 1.0 ] ]
      expected = argmax.probabilities(params)
      rng = Random.new(20_261_030)
      draws = 10_000
      firsts = Array.new(draws) { exploit(params, [ rng.rand, rng.rand ]) }.count(0)
      sd = Math.sqrt(draws * expected[0] * expected[1])
      expect((firsts - (draws * expected[0])).abs).to be <= 5 * sd, "#{firsts} firsts, #{draws * expected[0]} expected"
    end

    it "SL10 has no artificial tie near 1: draws that round to 1.0 are told apart by their distance from 1" do
      # 1 - Q is (1 - u)**20: 1e-20 at u = 0.9, 9.5e-27 at 0.95, both 1.0 as a Float.
      params = [ [ 1.0, 0.05 ], [ 1.0, 0.05 ] ]
      [ 0.9, 0.95 ].each { |u| expect(beta_function.quantile(u, 1.0, 0.05)).to eq(1.0) }
      expect(exploit(params, [ 0.9, 0.95 ])).to eq(1)
      expect(exploit(params, [ 0.95, 0.9 ])).to eq(0)
      # Far beyond what a complement bisected on [0, 1] can say: 1e-100 and 1e-200.
      expect(exploit(params, [ 1.0 - 1e-5, 1.0 - 1e-10 ])).to eq(1)
    end

    it "SL10 has no artificial tie near 0: draws below the bisection's floor are told apart too" do
      # Q is u**20: 1e-40 and 1e-34 (0.02**20), both below 2**-81, where 0.2.0's quantile answers one floor for both.
      params = [ [ 0.05, 1.0 ], [ 0.05, 1.0 ] ]
      expect(beta_function.quantile(0.01, 0.05, 1.0)).to eq(beta_function.quantile(0.02, 0.05, 1.0))
      expect(exploit(params, [ 0.01, 0.02 ])).to eq(1)
      expect(exploit(params, [ 0.02, 0.01 ])).to eq(0)
    end

    it "SL10 orders draws on both sides of 1/2 as their quantiles are ordered" do
      params = [ [ 2.0, 8.0 ], [ 8.0, 2.0 ], [ 5.0, 5.0 ] ]
      [ [ 0.5, 0.5, 0.5 ], [ 0.99, 0.01, 0.3 ], [ 0.001, 0.2, 0.2 ], [ 0.7, 0.05, 0.9 ] ].each do |uniforms|
        quantiles = params.zip(uniforms).map { |(alpha, beta), u| beta_function.quantile(u, alpha, beta) }
        expect(exploit(params, uniforms)).to eq(quantiles.index(quantiles.max)), "uniforms #{uniforms}"
      end
    end

    it "SL10 places a uniform too small for 1 - u to hold: Q(1e-20; 2000, 2) is about 0.977, beyond a uniform's 0.9" do
      expect(exploit([ [ 2000.0, 2.0 ], [ 1.0, 1.0 ] ], [ 1e-20, 0.9 ])).to eq(0)
      expect(exploit([ [ 2000.0, 2.0 ], [ 1.0, 1.0 ] ], [ 1e-20, 0.99 ])).to eq(1)
    end

    it "SL10 lets a uniform at an end decide: Q(1) = 1 wins, Q(0) = 0 loses" do
      params = [ [ 2.0, 8.0 ], [ 8.0, 2.0 ], [ 5.0, 5.0 ] ]
      expect(argmax.pick(params, explore: 0.5, priorities: [ 0.0, 0.0, 0.0 ], uniforms: [ 1.0, 0.99, 0.99 ],
                                 epsilon: 0.1)).to eq(0)
      expect(argmax.pick(params, explore: 0.5, priorities: [ 0.0, 0.0, 0.0 ], uniforms: [ 0.5, 0.0, 0.5 ],
                                 epsilon: 0.1)).to eq(2)
    end

    it "SL10 always picks the one member of a one-member call" do
      expect(argmax.pick([ [ 2.0, 3.0 ] ], explore: 0.0, priorities: [ 0.4 ], uniforms: [ 0.7 ], epsilon: 0.5)).to eq(0)
      expect(argmax.pick([ [ 2.0, 3.0 ] ], explore: 0.9, priorities: [ 0.4 ], uniforms: [ 0.7 ], epsilon: 0.5)).to eq(0)
      expect(argmax.greedy([ [ 2.0, 3.0 ] ])).to eq(0)
    end

    it "SL10 refuses members that are not 1 to 20 [alpha, beta] pairs of finite numbers > 0, in every method" do
      bad = {
        "no members" => [],
        "not an Array" => { a: [ 1.0, 1.0 ] },
        "nil" => nil,
        "a member of one number" => [ [ 1.0 ], [ 1.0, 1.0 ] ],
        "a member of three numbers" => [ [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0 ] ],
        "a member that is a number" => [ 2.0, [ 1.0, 1.0 ] ],
        "alpha 0" => [ [ 0.0, 1.0 ], [ 1.0, 1.0 ] ],
        "a negative beta" => [ [ 1.0, -2.0 ], [ 1.0, 1.0 ] ],
        "a NaN" => [ [ Float::NAN, 1.0 ], [ 1.0, 1.0 ] ],
        "an infinity" => [ [ 1.0, Float::INFINITY ], [ 1.0, 1.0 ] ],
        "a Complex" => [ [ Complex(2.0, 0.0), 1.0 ], [ 1.0, 1.0 ] ],
        "a String" => [ [ "2", 1.0 ], [ 1.0, 1.0 ] ],
        "a nil" => [ [ nil, 1.0 ], [ 1.0, 1.0 ] ],
        "an Integer beyond a Float" => [ [ 10**400, 1.0 ], [ 1.0, 1.0 ] ],
        "21 members" => [ [ 1.0, 1.0 ] ] * 21
      }
      bad.each do |label, params|
        size = params.is_a?(Array) ? params.size : 2
        expect_refused("probabilities, #{label}") { argmax.probabilities(params) }
        expect_refused("greedy, #{label}") { argmax.greedy(params) }
        expect_refused("pick, #{label}") do
          argmax.pick(params, explore: 0.5, priorities: [ 0.5 ] * size, uniforms: [ 0.5 ] * size, epsilon: 0.1)
        end
      end
    end

    it "SL10 refuses an epsilon outside [0, 1] or not a finite number" do
      [ -0.01, 1.01, Float::NAN, Float::INFINITY, nil, "0.1", Complex(0.1, 0.0) ].each do |epsilon|
        expect_refused("probabilities, epsilon #{epsilon.inspect}") { argmax.probabilities(three, epsilon: epsilon) }
        expect_refused("pick, epsilon #{epsilon.inspect}") do
          argmax.pick(three, explore: 0.5, priorities: [ 0.5 ] * 3, uniforms: [ 0.5 ] * 3, epsilon: epsilon)
        end
      end
      expect_refused("one member, epsilon 2") { argmax.probabilities([ [ 1.0, 1.0 ] ], epsilon: 2.0) }
    end

    it "SL10 refuses a pick's inputs that are not one per member, or not numbers in range" do
      ok = { explore: 0.5, priorities: [ 0.1, 0.2, 0.3 ], uniforms: [ 0.4, 0.5, 0.6 ], epsilon: 0.1 }
      expect(argmax.pick(three, **ok)).to be_a(Integer)
      {
        "two priorities" => { priorities: [ 0.1, 0.2 ] },
        "four uniforms" => { uniforms: [ 0.4, 0.5, 0.6, 0.7 ] },
        "priorities not an Array" => { priorities: 0.1 },
        "uniforms nil" => { uniforms: nil },
        "a priority NaN" => { priorities: [ 0.1, Float::NAN, 0.3 ] },
        "a priority infinite" => { priorities: [ 0.1, -Float::INFINITY, 0.3 ] },
        "a uniform above 1" => { uniforms: [ 0.4, 1.5, 0.6 ] },
        "a uniform below 0" => { uniforms: [ -0.1, 0.5, 0.6 ] },
        "a uniform a String" => { uniforms: [ 0.4, "0.5", 0.6 ] },
        "explore above 1" => { explore: 1.5 },
        "explore below 0" => { explore: -0.5 },
        "explore NaN" => { explore: Float::NAN },
        "explore nil" => { explore: nil }
      }.each do |label, change|
        expect_refused("pick, #{label}") { argmax.pick(three, **ok.merge(change)) }
      end
    end

    it "SL10 refuses members whose densities cannot be evaluated rather than answering NaN" do
      expect_refused("parameters of 1e306") { argmax.probabilities([ [ 1e306, 1e306 ], [ 1.0, 1.0 ] ]) }
    end

    # I_(1/3)(10**6, 2 * 10**6) as its binomial tail; 0.2.0's 300 steps read 0.50005496, 7e-7 above it. A uniform's
    # quantile at 1/3 is 1/3, so beside one the sharp member's draw passes 1/3 at exactly that level.
    it "SL10 picks on the fraction run to convergence, as the probabilities read it" do
      params = [ [ 1_000_000.0, 2_000_000.0 ], [ 1.0, 1.0 ] ]
      level = 0.500054289170726815167909213425
      expect(exploit(params, [ level + 1e-8, 1.0 / 3 ])).to eq(0)
      expect(exploit(params, [ level - 1e-8, 1.0 / 3 ])).to eq(1)
    end

    # The two regions beyond the measures: a parameter below 1e-6, and alpha + beta above 3 * 10**6, the furthest the
    # exact sum was taken. Beta < 1 is not one of them.
    it "SL10 refuses in probabilities and pick exactly the two regions beyond its measures, and greedy takes them" do
      smallest = argmax::SMALLEST_PARAMETER
      total = argmax::LARGEST_TOTAL
      { "alpha below the smallest" => [ smallest / 2.0, 1.0 ], "beta below the smallest" => [ 1.0, smallest / 2.0 ],
        "a total above the largest" => [ (total / 2.0) + 1.0, total / 2.0 ],
        "a total above the largest, one sided" => [ 2.0, total ] }.each do |label, member|
        params = [ member, [ 1.0, 1.0 ] ]
        expect_refused("probabilities, #{label}") { argmax.probabilities(params) }
        expect_refused("one member, #{label}") { argmax.probabilities([ member ]) }
        expect_refused("pick, #{label}") do
          argmax.pick(params, explore: 0.5, priorities: [ 0.5, 0.5 ], uniforms: [ 0.5, 0.5 ], epsilon: 0.1)
        end
        expect_refused("an exploring pick, #{label}") do
          argmax.pick(params, explore: 0.0, priorities: [ 0.5, 0.5 ], uniforms: [ 0.5, 0.5 ], epsilon: 0.1)
        end
        expect(argmax.greedy(params)).to be_a(Integer)
      end
    end

    it "SL10 names the two limits, and takes members exactly at them" do
      expect(argmax::SMALLEST_PARAMETER).to eq(1e-6)
      expect(argmax::LARGEST_TOTAL).to eq(3_000_000.0)
      at_limits = [ [ 1e-6, 1.0 ], [ 1_500_000.0, 1_500_000.0 ] ]
      expect(argmax.probabilities(at_limits).sum).to be_within(1e-12).of(1.0)
      expect(exploit(at_limits, [ 0.5, 0.5 ])).to eq(1)
    end

    # Each of the last two members once answered: a probability of -1.9e-11, and a pick that raised Math::DomainError.
    it "SL10 refuses just below the smallest parameter, and the members that broke the answers further down" do
      refusal = /beyond what can be answered/
      just_below = 1e-6.prev_float
      expect { argmax.probabilities([ [ just_below, 1.0 ], [ 1.0, 1.0 ] ]) }.to raise_error(ArgumentError, refusal)
      expect { exploit([ [ 1.0, just_below ], [ 1.0, 1.0 ] ], [ 0.5, 0.5 ]) }.to raise_error(ArgumentError, refusal)
      negative = [ [ 56_243.769548106146, 4.329089295168302e-12 ], [ 7.284758470571884, 0.12453928227574362 ] ]
      expect { argmax.probabilities(negative) }.to raise_error(ArgumentError, refusal)
      expect { exploit([ [ 2019.7234806256356, 1.1824235353929262e-15 ], [ 1.0, 1.0 ] ], [ 1e-14, 0.5 ]) }
        .to raise_error(ArgumentError, refusal)
    end

    it "SL10 never lets a draw fall as its uniform rises, at the smallest parameters and the largest total" do
      smallest = argmax::SMALLEST_PARAMETER
      levels = [ 1e-300, 1e-100, 1e-30, 1e-16, 1e-14, 1e-10, 1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0 - 1e-6, 1.0 - 1e-10,
                 1.0 - 1e-14, 1.0 - 1e-15 ]
      [ [ smallest, smallest ], [ smallest, 1.0 ], [ 1.0, smallest ], [ 2019.7234806256356, smallest ],
        [ smallest, 2_999_999.0 ], [ 2_999_999.0, smallest ] ].each do |member|
        levels.each_cons(2) do |low, high|
          # A tie goes to the lower index, so the member holding the higher uniform keeps it unless its draw fell.
          expect(exploit([ member, member ], [ high, low ])).to eq(0), "#{member} at #{low} and #{high}"
        end
      end
    end

    it "SL10 answers inside [0, 1], summing to 1, with members pressed against both limits" do
      smallest = argmax::SMALLEST_PARAMETER
      [ [ [ 2_999_999.0, smallest ], [ smallest, 2_999_999.0 ] ],
        [ [ 56_243.769548106146, smallest ], [ 7.284758470571884, 0.12453928227574362 ] ],
        [ [ smallest, smallest ], [ 2_999_999.0, smallest ], [ 1.0, 1.0 ] ] ].each do |params|
        answer = argmax.probabilities(params)
        expect(answer).to all(be_between(0.0, 1.0)), params.inspect
        expect(answer.sum).to be_within(1e-12).of(1.0)
      end
    end

    # From 1e-6 up no integral was seen below 0 (33,794 random calls): that takes a distribution function past 0 or 1,
    # and the nearest it comes to either is 2.2e-7 (measured). So the rounding read at a beta of 4e-12 is handed in here.
    it "SL10 clamps a probability that rounds below 0 to 0 and renormalises the rest" do
      allow(argmax).to receive(:integrals).and_return([ 1.0000000000185274, -1.8527423819605714e-11 ])
      expect(argmax.probabilities(three.first(2))).to eq([ 1.0, 0.0 ])

      allow(argmax).to receive(:integrals).and_return([ 0.5, -1e-12, 0.25 ])
      expect_pins(argmax.probabilities(three), [ 2.0 / 3, 0.0, 1.0 / 3 ], arithmetic_relative, "renormalised")

      mixed = argmax.probabilities(three, epsilon: 0.3)
      expect_pins(mixed, [ 0.1 + (1.4 / 3), 0.1, 0.1 + (0.7 / 3) ], arithmetic_relative, "epsilon 0.3")
      expect(mixed.sum).to be_within(1e-15).of(1.0)
    end

    it "SL10 still refuses integrals that hold no answer: a NaN, or nothing above 0" do
      [ [ Float::NAN, 0.5 ], [ -1e-12, 0.0 ], [ -1e-12, -1e-13 ] ].each do |integrals|
        allow(argmax).to receive(:integrals).and_return(integrals)
        expect { argmax.probabilities(three.first(2)) }.to raise_error(ArgumentError, /can integrate/)
      end
    end

    it "SL18 refuses a log density's inputs that are not finite numbers, and parameters that are not > 0" do
      [ Float::NAN, Float::INFINITY, -Float::INFINITY, nil, "0.5", Complex(0.5, 0.0) ].each do |x|
        expect_refused("x #{x.inspect}") { beta_function.log_density(x, 2.0, 3.0) }
      end
      [ 0.0, -1.0, Float::NAN, Float::INFINITY, nil, "2" ].each do |value|
        expect_refused("alpha #{value.inspect}") { beta_function.log_density(0.5, value, 3.0) }
        expect_refused("beta #{value.inspect}") { beta_function.log_density(0.5, 2.0, value) }
      end
      expect(beta_function.log_density(Rational(1, 4), 2, 3)).to eq(beta_function.log_density(0.25, 2.0, 3.0))
    end

    it "SL10 refuses a rule size that is not an Integer >= 1" do
      [ 0, -3, 2.0, 2.5, "3", nil ].each do |n|
        expect_refused("n = #{n.inspect}") { rule.nodes(n) }
      end
    end

    it "SL10 answers each rule frozen and ascending in x, so no caller can change the kept rule" do
      [ 1, 2, 7, 20 ].each do |n|
        nodes = rule.nodes(n)
        expect(nodes).to be_frozen
        expect(nodes).to all(be_frozen)
        expect(nodes.map(&:first)).to eq(nodes.map(&:first).sort)
        expect(rule.nodes(n)).to equal(nodes)
      end
    end
  end
end
