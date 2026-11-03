# frozen_string_literal: true

# BetaArgmax.distribution(x, alpha, beta), public in 0.4.0: the Beta distribution function the module integrates,
# picks and positions on, so a caller integrating against it reads the very same function. It answers exactly what the
# private path answers (the module's distribution function read at v = log x, from the member's own constant), bit for
# bit, on the 30 members of the BetaArgmax pins (the members the position record lists: every member of the pins'
# probability tables, the largest total, the smallest parameter at each end, and three sharp members), at 995 evenly
# spaced points each and at both ends of (0, 1).
RSpec.describe "SkiftetStatistical::BetaArgmax.distribution" do
  def argmax = SkiftetStatistical::BetaArgmax

  # The pins' members, in the position record's order.
  def members
    [
      [ 1.0, 1.0 ], [ 2.0, 1.0 ], [ 0.5, 1.0 ], [ 1.5, 1.0 ], [ 2.5, 1.0 ], [ 3.0, 3.0 ], [ 2.0, 8.0 ], [ 3.0, 7.0 ],
      [ 5.0, 5.0 ], [ 41.0, 361.0 ], [ 52.0, 350.0 ], [ 2.0, 30.0 ],
      [ 0.5, 0.5 ], [ 1.0, 0.5 ],
      [ 0.4, 0.4 ], [ 5.0, 3.0 ], [ 1.0, 0.05 ], [ 2.0, 7.0 ], [ 0.3, 0.1 ], [ 4.0, 1.0 ], [ 2.0, 2.0 ], [ 0.02, 0.02 ],
      [ 9.0, 2.0 ],
      [ 1_500_000.0, 1_500_000.0 ], [ 1_501_100.0, 1_498_900.0 ],
      [ 1e-6, 1.0 ], [ 1.0, 1e-6 ],
      [ 3.0, 300.0 ], [ 41.0, 960.0 ], [ 4_000.0, 96_000.0 ]
    ]
  end

  # 995 evenly spaced points, then the ends: below a Float's smallest normal, far into each tail, and the last Floats
  # below 1/2 and below 1.
  def points
    Array.new(995) { |j| (j + 0.5) / 995 } +
      [ 1e-320, Float::MIN, 1e-300, 1e-30, 1e-10, 1e-5, 0.5.prev_float, 0.5, 1.0 - 1e-5, 1.0 - 1e-10, 1.0.prev_float ]
  end

  def private_path(x, alpha, beta)
    argmax.send(:cdf, Math.log(x), x, alpha, beta, argmax.send(:log_constant, alpha, beta))
  end

  def bits(value) = [ value ].pack("G")

  it "is public" do
    expect(argmax.singleton_class.public_method_defined?(:distribution)).to be(true)
    expect(argmax.distribution(0.5, 2.0, 2.0)).to be_within(1e-15).of(0.5)
    expect(argmax.distribution(0.25, 1.0, 1.0)).to be_within(1e-15).of(0.25)
  end

  it "answers the private path bit for bit on the pins' 30 members, 995 points each and the ends of (0, 1)" do
    expect(members.size).to eq(30)
    differ = members.flat_map do |alpha, beta|
      points.filter_map do |x|
        [ alpha, beta, x ] unless bits(argmax.distribution(x, alpha, beta)) == bits(private_path(x, alpha, beta))
      end
    end
    expect(differ).to eq([])
  end

  it "is 0.0 at or below 0 and 1.0 at or above 1" do
    members.each do |alpha, beta|
      expect(argmax.distribution(0.0, alpha, beta)).to eql(0.0)
      expect(argmax.distribution(-0.5, alpha, beta)).to eql(0.0)
      expect(argmax.distribution(1.0, alpha, beta)).to eql(1.0)
      expect(argmax.distribution(2, alpha, beta)).to eql(1.0)
    end
  end

  it "takes what position takes and refuses the rest, as an ArgumentError" do
    expect(argmax.distribution(0.5, argmax::SMALLEST_PARAMETER, 1.0)).to be_between(0.0, 1.0)
    expect(argmax.distribution(Rational(1, 2), 2, 2)).to be_within(1e-15).of(0.5)
    [
      [ "0.5", 2.0, 2.0 ], [ Float::NAN, 2.0, 2.0 ], [ Float::INFINITY, 2.0, 2.0 ], [ nil, 2.0, 2.0 ],
      [ 0.5, "2", 2.0 ], [ 0.5, 0.0, 2.0 ], [ 0.5, -1.0, 2.0 ], [ 0.5, 2.0, Float::INFINITY ],
      [ 0.5, argmax::SMALLEST_PARAMETER / 2, 1.0 ], [ 0.5, 1.0, argmax::SMALLEST_PARAMETER / 2 ],
      [ 0.5, 2_000_000.0, 1_000_001.0 ]
    ].each do |x, alpha, beta|
      expect { argmax.distribution(x, alpha, beta) }.to raise_error(ArgumentError), [ x, alpha, beta ].inspect
      next if x.is_a?(String) || x.nil? || !x.finite?

      expect { argmax.position(x, alpha, beta) }.to raise_error(ArgumentError), "position #{[ x, alpha, beta ].inspect}"
    end
  end
end
