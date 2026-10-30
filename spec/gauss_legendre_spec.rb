# frozen_string_literal: true

# Acceptance for SL10's GaussLegendre.nodes(n): the n-point rule on [-1, 1] as [[x, w]], checked against the closed
# forms for n <= 5, the published 20-point table, and the rule's defining property (exact for every polynomial up to
# degree 2n - 1). The order of the pairs is not part of the contract, so they are sorted by x before comparing.
RSpec.describe "SkiftetStatistical::GaussLegendre" do
  def rule = SkiftetStatistical::GaussLegendre

  def sorted(n) = rule.nodes(n).sort_by(&:first)

  def expect_rule(n, expected)
    actual = sorted(n)
    expect(actual.size).to eq(expected.size)
    actual.zip(expected).each do |(x, w), (expected_x, expected_w)|
      expect(x).to be_within(1e-14).of(expected_x), "n = #{n}: node #{expected_x}"
      expect(w).to be_within(1e-14).of(expected_w), "n = #{n}: weight #{expected_w}"
    end
  end

  it "SL10 gives the closed-form rules for one to five points" do
    expect_rule(1, [ [ 0.0, 2.0 ] ])
    third = 1.0 / Math.sqrt(3.0)
    expect_rule(2, [ [ -third, 1.0 ], [ third, 1.0 ] ])
    outer = Math.sqrt(3.0 / 5.0)
    expect_rule(3, [ [ -outer, 5.0 / 9.0 ], [ 0.0, 8.0 / 9.0 ], [ outer, 5.0 / 9.0 ] ])

    # n = 4: x = +-sqrt(3/7 -+ 2/7 sqrt(6/5)), w = (18 +- sqrt(30)) / 36
    inner = Math.sqrt((3.0 / 7.0) - (2.0 / 7.0 * Math.sqrt(6.0 / 5.0)))
    far = Math.sqrt((3.0 / 7.0) + (2.0 / 7.0 * Math.sqrt(6.0 / 5.0)))
    near_weight = (18.0 + Math.sqrt(30.0)) / 36.0
    far_weight = (18.0 - Math.sqrt(30.0)) / 36.0
    expect_rule(4, [ [ -far, far_weight ], [ -inner, near_weight ], [ inner, near_weight ], [ far, far_weight ] ])

    # n = 5: x = 0, +-1/3 sqrt(5 -+ 2 sqrt(10/7)); w = 128/225, (322 +- 13 sqrt(70)) / 900
    inner = Math.sqrt(5.0 - (2.0 * Math.sqrt(10.0 / 7.0))) / 3.0
    far = Math.sqrt(5.0 + (2.0 * Math.sqrt(10.0 / 7.0))) / 3.0
    near_weight = (322.0 + (13.0 * Math.sqrt(70.0))) / 900.0
    far_weight = (322.0 - (13.0 * Math.sqrt(70.0))) / 900.0
    expect_rule(5, [ [ -far, far_weight ], [ -inner, near_weight ], [ 0.0, 128.0 / 225.0 ], [ inner, near_weight ],
                     [ far, far_weight ] ])
  end

  it "SL10 matches the published 20-point table, the panel rule of the argmax" do
    # Abramowitz and Stegun, table 25.4, to 16 digits: the positive nodes and their weights.
    half = [
      [ 0.0765265211334973, 0.1527533871307258 ], [ 0.2277858511416451, 0.1491729864726037 ],
      [ 0.3737060887154195, 0.1420961093183820 ], [ 0.5108670019508271, 0.1316886384491766 ],
      [ 0.6360536807265150, 0.1181945319615184 ], [ 0.7463319064601508, 0.1019301198172404 ],
      [ 0.8391169718222188, 0.0832767415767048 ], [ 0.9122344282513259, 0.0626720483341091 ],
      [ 0.9639719272779138, 0.0406014298003869 ], [ 0.9931285991850949, 0.0176140071391521 ]
    ]
    expect_rule(20, half.reverse.map { |x, w| [ -x, w ] } + half)
  end

  it "SL10 integrates every polynomial up to degree 2n - 1 exactly" do
    [ 1, 2, 3, 5, 10, 20, 40, 200 ].each do |n|
      nodes = rule.nodes(n)
      expect(nodes.sum { |_x, w| w }).to be_within(1e-12).of(2.0), "n = #{n}: weights"
      (0..[ (2 * n) - 1, 79 ].min).each do |degree|
        exact = degree.odd? ? 0.0 : 2.0 / (degree + 1)
        expect(nodes.sum { |x, w| w * (x**degree) }).to be_within(1e-12).of(exact), "n = #{n}: x**#{degree}"
      end
    end
  end

  it "SL10 puts n nodes strictly inside (-1, 1), symmetric about 0, with positive weights" do
    (1..30).each do |n|
      nodes = sorted(n)
      expect(nodes.size).to eq(n)
      expect(nodes.all? { |x, w| x > -1.0 && x < 1.0 && w.positive? }).to be(true), "n = #{n}"
      nodes.zip(nodes.reverse).each do |(x, w), (mirror_x, mirror_w)|
        expect(x).to be_within(1e-14).of(-mirror_x), "n = #{n}"
        expect(w).to be_within(1e-14).of(mirror_w), "n = #{n}"
      end
    end
  end

  it "SL10 answers the same rule on every call for one n" do
    expect(rule.nodes(20)).to eq(rule.nodes(20))
    expect(rule.nodes(7)).to eq(rule.nodes(7))
  end
end
