# frozen_string_literal: true

# How much an old view weighs: 2 ** (-age / half_life), 0.0 beyond the cap, a negative age refused; and the fold that
# sums every row's views, outcomes and outcome squares at that weight. Literals worked outside the library.
RSpec.describe "SkiftetStatistical::Bandit::Decay" do
  def decay = SkiftetStatistical::Bandit::Decay

  def row(age, views, outcomes, outcome_squares = nil)
    { age: age, views: views, outcomes: outcomes, outcome_squares: outcome_squares }.compact
  end

  describe ".weight" do
    it "LO1 halves at one half-life: weight(72, half_life: 72, cap: 336) == 0.5" do
      expect(decay.weight(72, half_life: 72, cap: 336)).to eq(0.5)
      expect(decay.weight(144, half_life: 72, cap: 336)).to be_within(1e-15).of(0.25)
    end

    it "LO1 is 2 ** (-age / half_life), 1.0 at age 0" do
      expect(decay.weight(0, half_life: 72, cap: 336)).to eq(1.0)
      expect(decay.weight(336, half_life: 72, cap: 336)).to be_within(1e-15).of(0.03937253280921478)
      expect(decay.weight(36.5, half_life: 24)).to be_within(1e-15).of(0.34848456586224985)
    end

    it "LO1 is 0.0 beyond the cap and the formula's at the cap itself" do
      expect(decay.weight(337, half_life: 72, cap: 336)).to eq(0.0)
      expect(decay.weight(336.5, half_life: 72, cap: 336)).to eq(0.0)
      expect(decay.weight(336, half_life: 72, cap: 336)).to be > 0.0
    end

    it "LO1 without a cap weighs every age by the formula" do
      expect(decay.weight(1_000, half_life: 72)).to be_within(1e-18).of(6.592161492262609e-05)
      expect(decay.weight(1_000, half_life: 72, cap: nil)).to be_within(1e-18).of(6.592161492262609e-05)
    end

    it "LO1 refuses a negative age, as an ArgumentError" do
      expect(decay.weight(1, half_life: 72, cap: 336)).to be_between(0.0, 1.0)
      expect { decay.weight(-1, half_life: 72, cap: 336) }.to raise_error(ArgumentError)
      expect { decay.weight(-0.5, half_life: 72) }.to raise_error(ArgumentError)
    end
  end

  describe ".fold" do
    it "LO1 sums views, outcomes and outcome squares each at its row's weight (ages 0, 72 and 400, cap 336)" do
      rows = [ row(0, 10, 3, 5), row(72, 20, 4, 8), row(400, 1_000, 900, 9_000) ]
      folded = decay.fold(rows, half_life: 72, cap: 336)

      # 1 * (10, 3, 5) + 0.5 * (20, 4, 8) + 0 * (1 000, 900, 9 000)
      expect(folded.keys).to contain_exactly(:views, :outcomes, :outcome_squares)
      expect(folded[:views]).to be_within(1e-12).of(20.0)
      expect(folded[:outcomes]).to be_within(1e-12).of(5.0)
      expect(folded[:outcome_squares]).to be_within(1e-12).of(9.0)
      expect(folded).to eq(decay.fold(rows.first(2), half_life: 72, cap: 336))
    end

    it "LO1 weighs rows at other ages by the formula, with no cap" do
      rows = [ row(24, 10, 3, 5), row(100, 20, 4, 8), row(500, 1_000, 900, 9_000) ]
      folded = decay.fold(rows, half_life: 48)

      expect(folded[:views]).to be_within(1e-12).of(12.522034627655135)
      expect(folded[:outcomes]).to be_within(1e-12).of(3.7236303833844095)
      expect(folded[:outcome_squares]).to be_within(1e-12).of(12.007639802726857)
    end

    it "LO1 reads a row without outcome squares (a yes/no row) as 0 there" do
      folded = decay.fold([ row(10, 6, 2), row(60.5, 9, 4) ], half_life: 72, cap: 336)

      expect(folded[:views]).to be_within(1e-12).of(10.476134335143524)
      expect(folded[:outcomes]).to be_within(1e-12).of(4.050580796045297)
      expect(folded[:outcome_squares]).to eq(0.0)
    end

    it "LO1 folds no rows to zero" do
      expect(decay.fold([], half_life: 72, cap: 336)).to eq(views: 0.0, outcomes: 0.0, outcome_squares: 0.0)
    end

    it "LO1 refuses a row with a negative age, as an ArgumentError" do
      expect(decay.fold([ row(0, 1, 1) ], half_life: 72)[:views]).to eq(1.0)
      expect { decay.fold([ row(5, 1, 1), row(-3, 2, 1) ], half_life: 72) }.to raise_error(ArgumentError)
    end

    it "LO1 is one formula: each row at weight's weight, and rows summed by age first fold alike" do
      rng = Random.new(808)
      ages = [ 0, 7.5, 30, 72, 200, 335, 336, 340 ]
      rows = Array.new(60) do
        views = rng.rand(0.0..40.0)
        outcomes = rng.rand * views
        row(ages.sample(random: rng), views, outcomes, outcomes * rng.rand(1.0..30.0))
      end
      folded = decay.fold(rows, half_life: 72, cap: 336)

      %i[views outcomes outcome_squares].each do |sum|
        by_hand = rows.sum { |r| decay.weight(r[:age], half_life: 72, cap: 336) * r[sum] }
        expect(folded[sum]).to be_within(1e-9).of(by_hand)
      end

      buckets = rows.group_by { |r| r[:age] }.map do |age, group|
        row(age, group.sum { |r| r[:views] }, group.sum { |r| r[:outcomes] }, group.sum { |r| r[:outcome_squares] })
      end
      by_bucket = decay.fold(buckets, half_life: 72, cap: 336)
      %i[views outcomes outcome_squares].each do |sum|
        expect(by_bucket[sum]).to be_within(1e-9).of(folded[sum])
      end
    end
  end
end
