# frozen_string_literal: true

# LO3: the acceptance of Bandit::Pass (01 GB13, GB18), written from the contract before its implementer, LB2. Until LB2
# lands every example fails on the missing constant.
#
# THE RULE: Pass.call(status:, hold:, crown: true, seed: Random.new_seed) { |hold_satisfied, rng| ... } yields the
# probe first, (false, Random.new(seed)); the pass is ready when the status is "running", the probe's Hash names a
# :leader and hold.satisfied_for?(leader) is true; only when crown and ready does it yield again, (true, a fresh
# Random.new(seed)), so the probe and the crown read the same draws; the answer is the last yielded block's Hash
# merged with ready: (true or false), and nothing else. The block here is the library's own Allocator.call, as an
# allocation pass calls it, and the hold its WinHold.
#
# THE FIXTURES, in round generic numbers: arm a converts 30 % and control 10 % at 2 000 views each, so a meets the win
# rule; a hold on a that began 25 hours ago (ripe) or 2 hours ago (unripe), the last pass 10 minutes ago; a paused
# test; a won test; and arms 10 % and 10.4 % apart at 1 000 views each, which names no leader. Each expected answer is
# the rule above worked out on the same Allocator calls (`reference`), and each phase was read on 0.3.0's Allocator
# before this was written: the ripe pass crowns a (:won), crown: false answers the probe (:shifting, leader a), the
# unripe pass and the close arms answer the probe, paused and won take the Allocator's own paths.
module PassOracle
  NOW = 1_000_000.0
  STRONG = { control: { exposures: 2_000, conversions: 200 }, a: { exposures: 2_000, conversions: 600 } }.freeze
  CLOSE = { control: { exposures: 1_000, conversions: 100 }, a: { exposures: 1_000, conversions: 104 } }.freeze
  SEED = 20_261_101

  # A hold that answers what it is told, and records what it was asked.
  class FixedHold
    attr_reader :asked

    def initialize(answer)
      @answer = answer
      @asked = []
    end

    def satisfied_for?(arm)
      @asked << arm
      @answer
    end
  end
end

RSpec.describe "SkiftetStatistical::Bandit::Pass" do
  def pass = SkiftetStatistical::Bandit::Pass
  def oracle = PassOracle

  def hold(leading_since)
    SkiftetStatistical::Bandit::WinHold.new(leading_arm: "a", leading_since: leading_since, now: oracle::NOW,
                                            last_pass_at: oracle::NOW - 600.0)
  end

  def ripe = hold(oracle::NOW - 90_000.0)
  def unripe = hold(oracle::NOW - 7_200.0)

  def fixtures
    {
      "running and ready" => { status: "running", tallies: oracle::STRONG, winner_arm: nil,
                               previous: { control: 0.3, a: 0.7 }, hold: ripe },
      "running, the hold unripe" => { status: "running", tallies: oracle::STRONG, winner_arm: nil,
                                      previous: { control: 0.5, a: 0.5 }, hold: unripe },
      "running, no leader" => { status: "running", tallies: oracle::CLOSE, winner_arm: nil,
                                previous: { control: 0.5, a: 0.5 }, hold: ripe },
      "paused" => { status: "paused", tallies: oracle::STRONG, winner_arm: nil, previous: { control: 0.7, a: 0.3 },
                    hold: ripe },
      "won" => { status: "won", tallies: oracle::STRONG, winner_arm: "control", previous: { control: 0.95, a: 0.05 },
                 hold: ripe }
    }
  end

  def allocate(fixture, hold_satisfied, rng)
    SkiftetStatistical::Bandit::Allocator.call(arms: %i[control a], tallies: fixture[:tallies], status: fixture[:status],
                                               winner_arm: fixture[:winner_arm], previous_weights: fixture[:previous],
                                               rng: rng, draws: 2_000, hold_satisfied: hold_satisfied)
  end

  # The rule, worked out on the same Allocator calls.
  def reference(fixture, crown:, seed:)
    probe = allocate(fixture, false, Random.new(seed))
    ready = fixture[:status] == "running" && !probe[:leader].nil? && fixture[:hold].satisfied_for?(probe[:leader])
    answer = crown && ready ? allocate(fixture, true, Random.new(seed)) : probe
    answer.merge(ready: ready)
  end

  # The pass over a fixture, recording each yield's flag and the first five values of its RNG (read from a copy, so
  # the Allocator still gets the RNG as it was handed over).
  def run(fixture, **options)
    yields = []
    answer = pass.call(status: fixture[:status], hold: fixture[:hold], **options) do |hold_satisfied, rng|
      yields << [ hold_satisfied, rng.dup.then { |copy| Array.new(5) { copy.rand } } ]
      allocate(fixture, hold_satisfied, rng)
    end
    [ answer, yields ]
  end

  it "LO3 (LB2) running and ready: probes, then crowns on the same seed, and answers the crown with ready: true" do
    answer, yields = run(fixtures.fetch("running and ready"), seed: oracle::SEED)
    expect(answer).to eq(reference(fixtures.fetch("running and ready"), crown: true, seed: oracle::SEED))
    expect(answer).to include(phase: :won, winner: :a, leader: :a, ready: true)
    expect(yields.map(&:first)).to eq([ false, true ])
    expect(yields[0][1]).to eq(yields[1][1])
  end

  it "LO3 (LB2) crown: false with a ripe hold answers the probe, ready, its leader stepping toward its P(best)" do
    fixture = fixtures.fetch("running and ready")
    answer, yields = run(fixture, crown: false, seed: oracle::SEED)
    expect(answer).to eq(reference(fixture, crown: false, seed: oracle::SEED))
    expect(answer).to include(phase: :shifting, winner: nil, leader: :a, ready: true)
    expect(yields.map(&:first)).to eq([ false ])
  end

  it "LO3 (LB2) crowns by default" do
    fixture = fixtures.fetch("running and ready")
    answer = pass.call(status: fixture[:status], hold: fixture[:hold], seed: oracle::SEED) do |hold_satisfied, rng|
      allocate(fixture, hold_satisfied, rng)
    end
    expect(answer).to include(phase: :won, ready: true)
  end

  {
    "running, the hold unripe" => { phase: :shifting, leader: :a },
    "running, no leader" => { phase: :shifting, leader: nil },
    "paused" => { phase: :paused },
    "won" => { phase: :won, winner: :control }
  }.each do |name, expected|
    it "LO3 (LB2) #{name}: answers the probe alone, with ready: false, in both modes" do
      fixture = fixtures.fetch(name)
      [ true, false ].each do |crown|
        answer, yields = run(fixture, crown: crown, seed: oracle::SEED)
        expect(answer).to eq(reference(fixture, crown: crown, seed: oracle::SEED)), "crown: #{crown}"
        expect(answer).to include(expected.merge(ready: false)), "crown: #{crown}"
        expect(yields.map(&:first)).to eq([ false ]), "crown: #{crown}"
      end
    end
  end

  it "LO3 (LB2) answers the block's Hash plus ready:, nothing else, from the crown's yield when it crowns" do
    block = ->(hold_satisfied, _rng) { { leader: "first", note: [ 1, 2 ], crowned: hold_satisfied } }
    crowned = pass.call(status: "running", hold: PassOracle::FixedHold.new(true), seed: 7, &block)
    expect(crowned).to eq(leader: "first", note: [ 1, 2 ], crowned: true, ready: true)
    probed = pass.call(status: "running", hold: PassOracle::FixedHold.new(true), crown: false, seed: 7, &block)
    expect(probed).to eq(leader: "first", note: [ 1, 2 ], crowned: false, ready: true)
    held = pass.call(status: "running", hold: PassOracle::FixedHold.new(false), seed: 7, &block)
    expect(held).to eq(leader: "first", note: [ 1, 2 ], crowned: false, ready: false)
  end

  it "LO3 (LB2) is ready only for a running status with a probe leader, asking the hold about that leader" do
    hold = PassOracle::FixedHold.new(true)
    answer = pass.call(status: "running", hold: hold, seed: 7) { |_satisfied, _rng| { leader: :a } }
    expect(answer[:ready]).to be(true)
    expect(hold.asked).to eq([ :a ])

    expect(pass.call(status: "running", hold: PassOracle::FixedHold.new(true), seed: 7) { { leader: nil } })
      .to eq(leader: nil, ready: false)
    expect(pass.call(status: "paused", hold: PassOracle::FixedHold.new(true), seed: 7) { { leader: :a } })
      .to eq(leader: :a, ready: false)
    expect(pass.call(status: "won", hold: PassOracle::FixedHold.new(true), seed: 7) { { leader: :a } })
      .to eq(leader: :a, ready: false)
  end

  it "LO3 (LB2) hands both yields a fresh Random.new(seed): the same sequence, a given seed honoured" do
    sequences = []
    pass.call(status: "running", hold: PassOracle::FixedHold.new(true), seed: 12_345) do |_satisfied, rng|
      sequences << Array.new(5) { rng.rand }
      { leader: :a }
    end
    expected = Random.new(12_345).then { |rng| Array.new(5) { rng.rand } }
    expect(sequences).to eq([ expected, expected ])
  end

  it "LO3 (LB2) draws its seed from Random.new_seed when none is given" do
    allow(Random).to receive(:new_seed).and_return(4_242)
    sequences = []
    pass.call(status: "running", hold: PassOracle::FixedHold.new(true)) do |_satisfied, rng|
      sequences << Array.new(3) { rng.rand }
      { leader: :a }
    end
    expected = Random.new(4_242).then { |rng| Array.new(3) { rng.rand } }
    expect(sequences).to eq([ expected, expected ])
  end
end
