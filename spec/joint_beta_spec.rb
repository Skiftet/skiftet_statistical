# frozen_string_literal: true

RSpec.describe SkiftetStatistical::JointBeta do
  let(:posteriors) { [ [ 41.0, 361.0 ], [ 52.0, 350.0 ], [ 1.5, 30.0 ] ] }

  # An RNG that always answers 0.5: every Sampler#beta on the same posterior then draws the same value.
  let(:constant_rng) { Class.new { def rand(*) = 0.5 }.new }

  def hand_rows(posteriors, draws, seed)
    sampler = SkiftetStatistical::Sampler.new(Random.new(seed))
    Array.new(draws) { posteriors.map { |alpha, beta| sampler.beta(alpha, beta) } }
  end

  describe ".each_draw" do
    it "draws per draw, arm by arm in the given order, on one Sampler over the given RNG" do
      (1..20).each do |seed|
        rows = []
        described_class.each_draw(posteriors, draws: 50, rng: Random.new(seed)) { |row| rows << row }
        expect(rows).to eq(hand_rows(posteriors, 50, seed)), "seed #{seed}"
      end
    end

    it "takes its order as given: reversed arms are other draws, not the same draws reversed" do
      forward = described_class.each_draw(posteriors, draws: 50, rng: Random.new(9)).to_a
      backward = described_class.each_draw(posteriors.reverse, draws: 50, rng: Random.new(9)).to_a
      expect(backward.map(&:reverse)).not_to eq(forward)
    end

    it "is an Enumerator of the same rows without a block, sized by draws" do
      rows = described_class.each_draw(posteriors, draws: 7, rng: Random.new(3))
      expect(rows).to be_a(Enumerator)
      expect(rows.size).to eq(7)
      expect(rows.to_a).to eq(hand_rows(posteriors, 7, 3))
    end

    it "returns nil with a block, and yields nothing for 0 draws" do
      yielded = 0
      expect(described_class.each_draw(posteriors, draws: 2, rng: Random.new(1)) { yielded += 1 }).to be_nil
      expect(yielded).to eq(2)
      expect(described_class.each_draw(posteriors, draws: 0, rng: Random.new(1)).to_a).to eq([])
    end

    it "refuses draws that are not a whole number >= 0" do
      [ -1, 2.0, nil, "10" ].each do |draws|
        expect { described_class.each_draw(posteriors, draws: draws, rng: Random.new(1)) }.to raise_error(ArgumentError)
      end
    end
  end

  describe ".summary" do
    it "holds each arm's draws, its wins and its summed loss over the each_draw rows" do
      rows = hand_rows(posteriors, 400, 11)
      summary = described_class.summary(posteriors, draws: 400, rng: Random.new(11))

      expect(summary.keys).to eq(%i[samples wins losses])
      expect(summary[:samples]).to eq(rows.transpose)
      winners = rows.map { |row| row.index(row.max) }
      expect(summary[:wins]).to eq((0...3).map { |arm| winners.count(arm).to_f })
      # Summed draw by draw, as the loop does (Array#sum would compensate and give other last digits).
      expect(summary[:losses]).to eq((0...3).map { |arm| rows.inject(0.0) { |total, row| total + (row.max - row[arm]) } })
    end

    it "splits a tie equally between the tied arms" do
      summary = described_class.summary([ [ 2.0, 2.0 ], [ 2.0, 2.0 ], [ 1.0, 5.0 ] ], draws: 4, rng: constant_rng)
      expect(summary[:wins]).to eq([ 2.0, 2.0, 0.0 ])
      expect(summary[:losses].first(2)).to eq([ 0.0, 0.0 ])
      expect(summary[:losses].last).to be > 0.0
      three = described_class.summary([ [ 2.0, 2.0 ] ] * 3, draws: 3, rng: constant_rng)
      expect(three[:wins]).to eq([ 1.0, 1.0, 1.0 ])
    end

    it "gives the same answer for the same seed" do
      first = described_class.summary(posteriors, draws: 100, rng: Random.new(5))
      second = described_class.summary(posteriors, draws: 100, rng: Random.new(5))
      expect(first).to eq(second)
    end
  end
end
