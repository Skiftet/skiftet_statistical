# frozen_string_literal: true

RSpec.describe SkiftetStatistical::Bandit::WinHold do
  let(:now) { Time.utc(2026, 10, 14, 12, 0, 0) }

  def hold(leading_arm: "a", since: 90_000, last_pass: 600, **settings)
    described_class.new(leading_arm: leading_arm, leading_since: since && (now - since), now: now,
                        last_pass_at: last_pass && (now - last_pass), **settings)
  end

  # The header comment's example, as Ruby source (from `hold = ...WinHold.new(` to its closing parenthesis).
  def documented_call
    path = File.expand_path("../../lib/skiftet_statistical/bandit/win_hold.rb", __dir__)
    lines = File.readlines(path).map { |line| line.sub(/\A\s*#\s?/, "") }
    start = lines.index { |line| line.include?("hold = SkiftetStatistical::Bandit::WinHold.new(") }
    source = +""
    lines[start..].each do |line|
      source << line
      break if source.count("(") == source.count(")")
    end
    source
  end

  it "makes the header's example true: the hold, the leader and the passes after it" do
    hold = binding.eval(documented_call)

    expect(hold.satisfied?).to be(true)
    expect(hold.satisfied_for?(:a)).to be(true)
    expect(hold.after_pass(:a)).to eq(leading_arm: "a", leading_since: hold.leading_since)
    expect(hold.after_pass(:control)).to eq(leading_arm: "control", leading_since: now)
    expect(hold.after_pass(nil)).to eq(leading_arm: nil, leading_since: nil)
  end

  it "holds 24 hours and allows 20 minutes between passes by default" do
    expect(described_class::HOLD_SECONDS).to eq(86_400)
    expect(described_class::MAX_GAP_SECONDS).to eq(1_200)
    expect(hold.hold_seconds).to eq(86_400)
    expect(hold.max_gap_seconds).to eq(1_200)
  end

  describe "#satisfied?" do
    it "is true from exactly hold_seconds after leading_since, not a second before" do
      expect(hold(since: 86_400).satisfied?).to be(true)
      expect(hold(since: 86_399).satisfied?).to be(false)
    end

    it "is false without a leader or without leading_since" do
      expect(hold(leading_arm: nil).satisfied?).to be(false)
      expect(hold(since: nil).satisfied?).to be(false)
    end

    it "is false when the run of passes is interrupted" do
      expect(hold(last_pass: 1_201).satisfied?).to be(false)
      expect(hold(last_pass: nil).satisfied?).to be(false)
    end

    it "takes other hold and gap lengths" do
      expect(hold(since: 3_600, hold_seconds: 3_600).satisfied?).to be(true)
      expect(hold(since: 0, hold_seconds: 0).satisfied?).to be(true)
      expect(hold(last_pass: 1_800, max_gap_seconds: 1_800).satisfied?).to be(true)
    end
  end

  describe "#interrupted?" do
    it "is false up to and at max_gap_seconds since the last pass, and true after" do
      expect(hold(last_pass: 1_200).interrupted?).to be(false)
      expect(hold(last_pass: 1_201).interrupted?).to be(true)
    end

    it "is true with no previous pass" do
      expect(hold(last_pass: nil).interrupted?).to be(true)
    end
  end

  describe "#satisfied_for?" do
    it "is true only for the arm that held, by its String form" do
      expect(hold(leading_arm: :a).satisfied_for?("a")).to be(true)
      expect(hold.satisfied_for?(:a)).to be(true)
      expect(hold.satisfied_for?(:control)).to be(false)
      expect(hold.satisfied_for?(nil)).to be(false)
      expect(hold(since: 86_399).satisfied_for?(:a)).to be(false)
    end
  end

  describe "#after_pass" do
    it "keeps leading_since while the same arm leads an unbroken run" do
      expect(hold.after_pass(:a)).to eq(leading_arm: "a", leading_since: now - 90_000)
    end

    it "starts over at this pass for another arm, after a gap, or without leading_since" do
      expect(hold.after_pass(:control)).to eq(leading_arm: "control", leading_since: now)
      expect(hold(last_pass: 1_201).after_pass(:a)).to eq(leading_arm: "a", leading_since: now)
      expect(hold(last_pass: nil).after_pass(:a)).to eq(leading_arm: "a", leading_since: now)
      expect(hold(since: nil).after_pass(:a)).to eq(leading_arm: "a", leading_since: now)
    end

    it "resets with no leader" do
      expect(hold.after_pass(nil)).to eq(leading_arm: nil, leading_since: nil)
    end
  end

  describe "the leading arm" do
    it "reads a blank arm as none" do
      [ nil, false, "", " ", "\t\n", :"" ].each do |blank|
        expect(hold(leading_arm: blank).leading_arm).to be_nil, blank.inspect
        expect(hold(leading_arm: blank).satisfied?).to be(false), blank.inspect
      end
    end

    it "keeps any other arm as a String" do
      expect(hold(leading_arm: :a).leading_arm).to eq("a")
      expect(hold(leading_arm: 0).leading_arm).to eq("0")
    end

    # ActiveSupport's `presence` answers for a String in an encoding that is not ASCII-compatible, so the hold does too.
    wide = %w[UTF-16LE UTF-16BE UTF-32LE UTF-32BE].freeze

    it "reads a blank arm in UTF-16 or UTF-32 as none" do
      [ "", " ", "\t\n", " 　" ].product(wide).each do |text, encoding|
        expect(hold(leading_arm: text.encode(encoding)).leading_arm).to be_nil, "#{text.inspect} in #{encoding}"
      end
      expect(hold(leading_arm: "".encode("UTF-16")).leading_arm).to be_nil
    end

    it "keeps any other arm in UTF-16 or UTF-32 as it is" do
      wide.each do |encoding|
        arm = " a ".encode(encoding)

        expect(hold(leading_arm: arm).leading_arm).to eq(arm), encoding
        expect(hold(leading_arm: arm).satisfied_for?(arm)).to be(true), encoding
      end
    end

    it "raises on a non-empty arm in a dummy encoding, as ActiveSupport's presence does" do
      [ " ", "a" ].each do |text|
        expect { hold(leading_arm: text.encode("UTF-16")) }.to raise_error(RegexpError)
      end
    end
  end

  it "requires last_pass_at, so a call that leaves it out cannot build a hold that never holds" do
    expect { described_class.new(leading_arm: "a", leading_since: now - 90_000, now: now) }
      .to raise_error(ArgumentError, /last_pass_at/)
  end

  it "takes plain numbers of seconds as its clock" do
    numeric = described_class.new(leading_arm: "a", leading_since: 0, now: 86_400, last_pass_at: 85_500)

    expect(numeric.satisfied?).to be(true)
    expect(numeric.after_pass(:control)).to eq(leading_arm: "control", leading_since: 86_400)
  end
end
