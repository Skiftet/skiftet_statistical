# frozen_string_literal: true

RSpec.describe SkiftetStatistical::Bandit::WeightedPosterior do
  def tally(exposures, conversions, **extra)
    { exposures: exposures, conversions: conversions, **extra }
  end

  describe ".count" do
    it "reads a nil tally as nobody seen" do
      expect(described_class.count(nil)).to eq(exposures: 0, conversions: 0, weighted_exposures: nil,
                                               weighted_conversions: nil, weighted_exposures_squared: nil)
    end

    it "reads Symbol and String keys alike, and whole numbers as Integers" do
      expect(described_class.count("exposures" => "300", "conversions" => 27))
        .to eq(described_class.count(exposures: 300, conversions: 27))
      expect(described_class.count(exposures: 300, conversions: 27)).to include(exposures: 300, conversions: 27)
    end

    it "keeps the weighted sums only when both are given and the weighted exposures are above zero" do
      weighted = described_class.count(tally(800, 80, weighted_exposures: 1_600, weighted_conversions: 150))
      zero = described_class.count(tally(800, 96, weighted_exposures: 0.0, weighted_conversions: 0.0))
      half = described_class.count(tally(800, 96, weighted_exposures: 1_450.0))

      expect(weighted).to include(weighted_exposures: 1_600.0, weighted_conversions: 150.0, weighted_exposures_squared: nil)
      expect([ zero, half ].map { |count| count.values_at(:weighted_exposures, :weighted_conversions) }).to all(eq([ nil, nil ]))
    end

    it "keeps the squared sum only beside the weighted sums, and only above zero" do
      with = described_class.count(tally(800, 80, weighted_exposures: 1_600.0, weighted_conversions: 150.0,
                                                  weighted_exposures_squared: 4_000.0))
      zero = described_class.count(tally(800, 80, weighted_exposures: 1_600.0, weighted_conversions: 150.0,
                                                  weighted_exposures_squared: 0.0))
      alone = described_class.count(tally(800, 80, weighted_exposures_squared: 4_000.0))

      expect(with[:weighted_exposures_squared]).to eq(4_000.0)
      expect(zero[:weighted_exposures_squared]).to be_nil
      expect(alone[:weighted_exposures_squared]).to be_nil
    end
  end

  describe ".beta" do
    it "is Beta(1 + c, 1 + n - c) on the raw counts" do
      expect(described_class.beta(described_class.count(tally(400, 40)))).to eq([ 41.0, 361.0 ])
    end

    it "clamps the conversions to the exposures" do
      expect(described_class.beta(described_class.count(tally(10, 12)))).to eq([ 11.0, 1.0 ])
    end

    it "rescales the weighted conversions to the raw exposure count without a squared sum" do
      count = described_class.count(tally(800, 80, weighted_exposures: 1_600.0, weighted_conversions: 150.0))

      expect(described_class.beta(count)).to eq([ 1.0 + 75.0, 1.0 + 800 - 75.0 ])
    end

    it "rescales to the Kish n, (sum w)^2 / sum w^2, with the squared sum" do
      count = described_class.count(tally(800, 80, weighted_exposures: 1_600.0, weighted_conversions: 150.0,
                                                   weighted_exposures_squared: 4_000.0))

      expect(described_class.effective_exposures(count)).to eq(640.0)
      expect(described_class.beta(count)).to eq([ 1.0 + 60.0, 1.0 + 640.0 - 60.0 ])
    end

    it "never counts more than the raw exposures" do
      count = described_class.count(tally(800, 96, weighted_exposures: 1_450.0, weighted_conversions: 190.0,
                                                   weighted_exposures_squared: 2_000.0))

      expect((1_450.0**2) / 2_000.0).to be > 800
      expect(described_class.effective_exposures(count)).to eq(800.0)
    end

    it "clamps weighted conversions above the weighted exposures to n" do
      count = described_class.count(tally(50, 5, weighted_exposures: 40.0, weighted_conversions: 90.0))

      expect(described_class.beta(count)).to eq([ 51.0, 1.0 ])
    end

    it "takes the prior it is given" do
      count = described_class.count(tally(400, 40))

      expect(described_class.beta(count, prior_alpha: 0.5, prior_beta: 2.0)).to eq([ 40.5, 362.0 ])
    end
  end

  describe ".effective_exposures" do
    it "is the raw count (an Integer) without a squared sum" do
      expect(described_class.effective_exposures(described_class.count(tally(400, 40)))).to eq(400)
    end
  end
end
