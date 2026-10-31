# frozen_string_literal: true

require "json"

# SL15: the pins of SharedCredit's outputs on invented rows, and the readings where SL5 is silent. Every pinned value
# was checked first by an independent reference in exact Rationals over the exact binary value of every Float input:
# the means and the covariance by their definitions (each row's influence values psi_ap(i) = w_ai * (x_ip - r_ap) /
# W_a, their products summed over the rows), never by the sums the module keeps; from_sums' pins by SL5's sums formula
# in Rationals on the very literals given. This file ran red against those digits before the module existed. A pinned
# value is never edited.
#
# One tolerance rule (SL15) for every Float compared with a pinned value: 1e-12 relative, and a pinned 0.0, 1.0 or
# infinity exactly. Nothing here goes through lgamma, so nothing takes 1e-9.
RSpec.describe "SkiftetStatistical::SharedCredit pins" do
  def shared_credit
    SkiftetStatistical::SharedCredit
  end

  def relative
    1e-12
  end

  def expect_pin(actual, pinned, label)
    expect(actual).to be_a(Float), "#{label}: #{actual.inspect} is not a Float"
    if pinned.zero? || pinned == 1.0 || pinned.infinite?
      expect(actual).to eq(pinned), "#{label}: #{actual} is not exactly #{pinned}"
    else
      error = (actual - pinned).abs / pinned.abs
      expect(error).to be <= relative, "#{label}: #{actual} is #{error} from #{pinned}"
    end
  end

  # A pinned Hash or Array walked whole: the same keys in the same order, the same sizes, nil where nil is pinned, a
  # String or Integer equal, every Float by expect_pin.
  def expect_pins(actual, pinned, label = "")
    case pinned
    when Hash
      expect(actual).to be_a(Hash), "#{label}: #{actual.inspect} is not a Hash"
      expect(actual.keys).to eq(pinned.keys), "#{label}: keys #{actual.keys.inspect} are not #{pinned.keys.inspect}"
      pinned.each { |key, value| expect_pins(actual[key], value, "#{label}[#{key.inspect}]") }
    when Array
      expect(actual).to be_a(Array), "#{label}: #{actual.inspect} is not an Array"
      expect(actual.size).to eq(pinned.size), "#{label}: #{actual.size} entries, not #{pinned.size}"
      pinned.each_with_index { |value, index| expect_pins(actual[index], value, "#{label}[#{index}]") }
    when Float
      expect_pin(actual, pinned, label)
    else
      expect(actual).to eq(pinned), "#{label}: #{actual.inspect} is not #{pinned.inspect}"
    end
  end

  def expect_refused(label, &block)
    expect(&block).to raise_error(ArgumentError), "#{label} was not refused"
  end

  let(:arms) { %i[a b c] }
  let(:parts) { %i[p q r] }

  # Eight rows of three arms and three parts. Credit is shared: rows 1, 3, 5 and 7 count for two or three arms at once.
  # No row that credits b holds a value for part r, so (b, r) has a mean of 0 and no variance. The last row holds no
  # value at all and still weighs in every mean of b. 3.3333333333333335 is the Float nearest 10/3, taken exactly.
  let(:shared_rows) do
    [
      { credit: { a: 2.5, b: 2.5 }, values: [ 1.0, 0.0, 0.0 ] },
      { credit: { a: 1.25 }, values: [ 0.0, 2.0, 1.0 ] },
      { credit: { b: 4.0, c: 4.0 }, values: [ 1.0, 1.0, 0.0 ] },
      { credit: { c: 1.6 }, values: [ 0.0, 0.0, 3.0 ] },
      { credit: { a: 2.0, c: 5.0 }, values: [ 2.0, 0.0, 1.0 ] },
      { credit: { b: 3.3333333333333335 }, values: [ 0.0, 1.0, 0.0 ] },
      { credit: { a: 1.5, b: 1.5, c: 1.5 }, values: [ 1.0, 3.0, 0.0 ] },
      { credit: { b: 3.3333333333333335 }, values: [ 0.0, 0.0, 0.0 ] }
    ]
  end

  # An ordinary test: each row credits only its drawn arm, with 1 / p.
  let(:single_rows) do
    [
      { credit: { a: 2.0 }, values: [ 1.0, 0.0, 2.0 ] },
      { credit: { b: 1.25 }, values: [ 0.0, 1.0, 0.0 ] },
      { credit: { a: 5.0 }, values: [ 0.0, 0.0, 1.0 ] },
      { credit: { b: 4.0 }, values: [ 1.0, 2.0, 0.0 ] },
      { credit: { a: 3.3333333333333335 }, values: [ 2.0, 1.0, 0.0 ] },
      { credit: { b: 1.6 }, values: [ 0.0, 0.0, 3.0 ] }
    ]
  end

  # single_rows' per-arm sums, each exact sum rounded once to the nearest Float, c given in both orders of each pair.
  let(:single_sums) do
    {
      a: { w: 10.333333333333334, w2: 40.111111111111114,
           a: { p: 8.6666666666666679, q: 3.3333333333333335, r: 9.0 },
           b: { p: 26.222222222222225, q: 11.111111111111112, r: 33.0 },
           c: { [ :p, :p ] => 48.44444444444445, [ :p, :q ] => 22.222222222222225, [ :p, :r ] => 8.0,
                [ :q, :p ] => 22.222222222222225, [ :q, :q ] => 11.111111111111112, [ :q, :r ] => 0.0,
                [ :r, :p ] => 8.0, [ :r, :q ] => 0.0, [ :r, :r ] => 41.0 } },
      b: { w: 6.8499999999999996, w2: 20.122499999999999,
           a: { p: 4.0, q: 9.25, r: 4.8000000000000007 },
           b: { p: 16.0, q: 33.5625, r: 7.6800000000000006 },
           c: { [ :p, :p ] => 16.0, [ :p, :q ] => 32.0, [ :p, :r ] => 0.0,
                [ :q, :p ] => 32.0, [ :q, :q ] => 65.5625, [ :q, :r ] => 0.0,
                [ :r, :p ] => 0.0, [ :r, :q ] => 0.0, [ :r, :r ] => 23.040000000000003 } }
    }
  end

  def shared_estimate
    shared_credit.estimate(shared_rows, arms: arms, parts: parts)
  end

  def json(value)
    JSON.parse(JSON.generate(value))
  end

  describe "pins" do
    it "SL15 pins estimate on eight rows of three arms and three parts whose credit is shared between arms" do
      pinned = {
        means: {
          a: [ 1.103448275862069, 0.96551724137931039, 0.44827586206896552 ],
          b: [ 0.54545454545454541, 0.80681818181818177, 0.0 ],
          c: [ 1.28099173553719, 0.7024793388429752, 0.80991735537190079 ]
        },
        covariance: [
          [ 0.099094984878711573, -0.096940254297796777, 0.02704441374785976, -0.0037588074016568235,
            0.00010502550092864648, 0.0, 0.074228375497570429, -0.077889522532987476, 0.021575428603125404 ],
          [ -0.096940254297796777, 0.39078103328097319, -0.011146913320165536, -0.0062278279498039525,
            0.14020259480547556, 0.0, -0.093797807388743729, 0.19720328323999492, -0.063183211415566903 ],
          [ 0.02704441374785976, -0.011146913320165536, 0.064709782957551532, -0.016288165407179568,
            0.00045511050402413476, 0.0, 0.048450911676776919, -0.070596465661193972, 0.021266813724363544 ],
          [ -0.0037588074016568235, -0.0062278279498039525, -0.016288165407179568, 0.054267553445802884,
            0.023591978433850146, 0.0, -0.013134602759938833, 0.025433078510985509, -0.037858560896294288 ],
          [ 0.00010502550092864648, 0.14020259480547556, 0.00045511050402413476, 0.023591978433850146,
            0.10755224119126085, 0.0, -0.01270727835507781, 0.069067001503194073, -0.03662686114110663 ],
          [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
          [ 0.074228375497570429, -0.093797807388743729, 0.048450911676776919, -0.013134602759938833,
            -0.01270727835507781, 0.0, 0.12680883513289098, -0.089568483985508388, 0.0026506949343516983 ],
          [ -0.077889522532987476, 0.19720328323999492, -0.070596465661193972, 0.025433078510985509,
            0.069067001503194073, 0.0, -0.089568483985508388, 0.1836854149280617, -0.10463107427772027 ],
          [ 0.021575428603125404, -0.063183211415566903, 0.021266813724363544, -0.037858560896294288,
            -0.03662686114110663, 0.0, 0.0026506949343516983, -0.10463107427772027, 0.17180253893935937 ]
        ],
        weight_sums: { a: 7.25, b: 14.666666666666668, c: 12.1 },
        effective: { a: 3.7377777777777776, b: 4.6040428061831156, c: 3.1960270683256931 }
      }

      aggregate_failures { expect_pins(shared_estimate, pinned, "estimate") }
    end

    it "SL15 pins linear and contrast on those rows at the fixed weights p 1.0, q 0.35, r 2.5" do
      weights = { p: 1.0, q: 0.35, r: 2.5 }
      pinned = {
        means: { a: 2.5620689655172413, b: 0.82784090909090902, c: 3.5516528925619837 },
        covariance: [
          [ 0.59925859736087916, -0.029049162222020225, 0.22922143125672609 ],
          [ -0.029049162222020225, 0.08395708789542744, -0.12691477076043386 ],
          [ 0.22922143125672609, -0.12691477076043386, 0.99052732272846677 ]
        ]
      }
      contrasts = {
        %i[a b] => { mean: 1.7342280564263324, variance: 0.74131400970034711 },
        %i[a c] => { mean: -0.98958392704474207, variance: 1.1313430575758938 },
        %i[b c] => { mean: -2.7238119834710743, variance: 1.328313952144762 }
      }

      aggregate_failures do
        expect_pins(shared_credit.linear(shared_estimate, weights: weights), pinned, "linear")
        contrasts.each do |(first, second), values|
          expect_pins(shared_credit.contrast(shared_estimate, first, second, weights: weights), values,
                      "contrast #{first} - #{second}")
        end
      end
    end

    it "SL15 pins from_sums on an ordinary test's per-arm sums" do
      pinned = {
        means: {
          a: [ 0.83870967741935487, 0.32258064516129031, 0.87096774193548387 ],
          b: [ 0.58394160583941612, 1.3503649635036497, 0.70072992700729941 ]
        },
        covariance: [
          [ 0.3060027871591442, 0.14325608188660571, -0.12376545850067296, 0.0, 0.0, 0.0 ],
          [ 0.14325608188660571, 0.076013431205137733, -0.084784211728807457, 0.0, 0.0, 0.0 ],
          [ -0.12376545850067296, -0.084784211728807457, 0.13058717668574943, 0.0, 0.0, 0.0 ],
          [ 0.0, 0.0, 0.0, 0.088984934714182284, 0.14199800933565715, -0.15903922386442468 ],
          [ 0.0, 0.0, 0.0, 0.14199800933565715, 0.24747913039538405, -0.31644336317918081 ],
          [ 0.0, 0.0, 0.0, -0.15903922386442468, -0.31644336317918081, 0.47221241794426844 ]
        ],
        weight_sums: { a: 10.333333333333334, b: 6.8499999999999996 },
        effective: { a: 2.662049861495845, b: 2.331842464902472 }
      }

      aggregate_failures do
        expect_pins(shared_credit.from_sums(single_sums, arms: %i[a b], parts: parts), pinned, "from_sums")
      end
    end

    it "SL15 pins estimate on that ordinary test's rows, and from_sums on their sums agrees within 1e-12" do
      pinned = {
        means: {
          a: [ 0.83870967741935487, 0.32258064516129031, 0.87096774193548387 ],
          b: [ 0.58394160583941601, 1.3503649635036497, 0.7007299270072993 ]
        },
        covariance: [
          [ 0.3060027871591442, 0.14325608188660571, -0.12376545850067297, 0.0, 0.0, 0.0 ],
          [ 0.14325608188660571, 0.076013431205137733, -0.084784211728807471, 0.0, 0.0, 0.0 ],
          [ -0.12376545850067297, -0.084784211728807471, 0.13058717668574943, 0.0, 0.0, 0.0 ],
          [ 0.0, 0.0, 0.0, 0.088984934714182298, 0.14199800933565718, -0.15903922386442462 ],
          [ 0.0, 0.0, 0.0, 0.14199800933565718, 0.24747913039538408, -0.31644336317918076 ],
          [ 0.0, 0.0, 0.0, -0.15903922386442462, -0.31644336317918076, 0.47221241794426833 ]
        ],
        weight_sums: { a: 10.333333333333334, b: 6.8499999999999996 },
        effective: { a: 2.662049861495845, b: 2.3318424649024725 }
      }
      from_rows = shared_credit.estimate(single_rows, arms: %i[a b], parts: parts)
      from_sums = shared_credit.from_sums(single_sums, arms: %i[a b], parts: parts)

      aggregate_failures do
        expect_pins(from_rows, pinned, "estimate")
        expect_pins(from_sums, from_rows, "from_sums against estimate")
      end
    end

    it "SL15 pins the accumulator's JSON-safe to_h on two rows, and from_h reads it back unchanged" do
      accumulator = shared_credit.accumulator(arms: arms, parts: parts)
      shared_rows.first(2).each { |row| accumulator.add(row) }
      # Row 1 credits a and b with 2.5 on values (1, 0, 0); row 2 credits a with 1.25 on (0, 2, 1). Per arm: w the
      # credit summed, x the credit times each value. Per pair of arms credited together (the first in the arms'
      # order): w the product of their credits, x that product times each value, xx it times each pair of values (the
      # first part in the parts' order). Every number is exact in binary.
      pinned = {
        "format" => 1,
        "arms" => %w[a b c],
        "parts" => %w[p q r],
        "sums" => {
          "a" => { "w" => 3.75, "x" => { "p" => 2.5, "q" => 2.5, "r" => 1.25 } },
          "b" => { "w" => 2.5, "x" => { "p" => 2.5 } }
        },
        "pairs" => {
          "a" => {
            "a" => { "w" => 7.8125, "x" => { "p" => 6.25, "q" => 3.125, "r" => 1.5625 },
                     "xx" => { "p" => { "p" => 6.25 }, "q" => { "q" => 6.25, "r" => 3.125 }, "r" => { "r" => 1.5625 } } },
            "b" => { "w" => 6.25, "x" => { "p" => 6.25 }, "xx" => { "p" => { "p" => 6.25 } } }
          },
          "b" => {
            "b" => { "w" => 6.25, "x" => { "p" => 6.25 }, "xx" => { "p" => { "p" => 6.25 } } }
          }
        }
      }
      hash = accumulator.to_h

      aggregate_failures do
        expect_pins(hash, pinned, "to_h")
        expect(json(hash)).to eq(hash)
        expect(shared_credit::Accumulator.from_h(json(hash)).to_h).to eq(hash)
        expect(shared_credit::Accumulator.from_h(json(hash)).estimate).to eq(accumulator.estimate)
      end
    end
  end

  describe "readings" do
    it "SL5 answers the arms as given, Strings too, and reads a row's keys and credit by Symbol or String" do
      by_strings = shared_rows.map { |row| { "credit" => row[:credit].transform_keys(&:to_s), "values" => row[:values] } }
      estimate = shared_credit.estimate(by_strings, arms: %w[a b c], parts: %w[p q r])

      expect(estimate.keys).to eq(%i[means covariance weight_sums effective])
      expect(estimate[:means].keys).to eq(%w[a b c])
      expect(estimate[:weight_sums].keys).to eq(%w[a b c])
      expect(estimate[:effective].keys).to eq(%w[a b c])
      expect(estimate[:covariance]).to eq(shared_estimate[:covariance])
      expect(estimate[:means].values).to eq(shared_estimate[:means].values)
      expect(shared_credit.estimate(shared_rows, arms: %i[c a b], parts: parts)[:means].keys).to eq(%i[c a b])
    end

    it "SL5 computes one covariance triangle and mirrors it: every matrix it answers is exactly symmetric" do
      estimate = shared_estimate
      linear = shared_credit.linear(estimate, weights: { p: 1.0, q: 0.35, r: 2.5 })
      from_sums = shared_credit.from_sums(single_sums, arms: %i[a b], parts: parts)

      [ estimate[:covariance], linear[:covariance], from_sums[:covariance] ].each do |matrix|
        matrix.each_index do |i|
          matrix.each_index { |j| expect(matrix[i][j]).to eql(matrix[j][i]), "entry (#{i}, #{j})" }
        end
      end
    end

    it "SL5 gives exactly 0.0 between two arms no row credits together and for a part no credited row holds" do
      rows = [
        { credit: { a: 2.0, b: 0.0 }, values: [ 1.0, 0.0, 0.0 ] },
        { credit: { c: 3.0 }, values: [ 0.0, 0.0, 2.0 ] },
        { credit: { a: 1.25 }, values: [ 3.0, 0.0, 1.0 ] }
      ]
      estimate = shared_credit.estimate(rows, arms: arms, parts: parts)
      covariance = estimate[:covariance]

      expect(estimate[:means][:a][1]).to eql(0.0)
      expect(estimate[:means][:b]).to be_nil
      (0...3).each do |i|
        (3...9).each do |j|
          expect(covariance[i][j]).to eql(0.0), "entry (#{i}, #{j})"
          expect(covariance[j][i]).to eql(0.0), "entry (#{j}, #{i})"
        end
        expect(covariance[1][i]).to eql(0.0)
      end
      expect(covariance[6..8].map { |row| row[6..8] }).to eq([ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ])
    end

    it "SL5 does not depend on the order of the arms within one row's credit" do
      reversed = shared_rows.map { |row| { credit: row[:credit].to_a.reverse.to_h, values: row[:values] } }

      expect(shared_credit.estimate(reversed, arms: arms, parts: parts)).to eq(shared_estimate)
    end

    it "SL5 keeps sparse sums: to_h holds no zero credit, no zero value and no pair of arms no row credits together" do
      accumulator = shared_credit.accumulator(arms: arms, parts: parts)
      accumulator.add({ credit: { a: 2.0, b: 0.0 }, values: [ 1.0, 0.0, 0.0 ] })
      accumulator.add({ credit: { c: 3.0 }, values: [ 0.0, 0.0, 2.0 ] })
      accumulator.add({ credit: {}, values: [ 5.0, 5.0, 5.0 ] })

      expect_pins(accumulator.to_h, {
        "format" => 1, "arms" => %w[a b c], "parts" => %w[p q r],
        "sums" => { "a" => { "w" => 2.0, "x" => { "p" => 2.0 } }, "c" => { "w" => 3.0, "x" => { "r" => 6.0 } } },
        "pairs" => {
          "a" => { "a" => { "w" => 4.0, "x" => { "p" => 4.0 }, "xx" => { "p" => { "p" => 4.0 } } } },
          "c" => { "c" => { "w" => 9.0, "x" => { "r" => 18.0 }, "xx" => { "r" => { "r" => 36.0 } } } }
        }
      })
    end

    it "SL5 names the arms and parts of from_h by Symbols, and merges them with an accumulator that names them by Strings" do
      by_strings = shared_credit.accumulator(arms: %w[a b c], parts: %w[p q r])
      shared_rows.first(4).each { |row| by_strings.add(row) }
      rest = shared_credit.accumulator(arms: %w[a b c], parts: %w[p q r])
      shared_rows.drop(4).each { |row| rest.add(row) }
      read = shared_credit::Accumulator.from_h(json(rest.to_h))
      merged = by_strings.merge(read)

      expect(read.arms).to eq(%i[a b c])
      expect(read.parts).to eq(%i[p q r])
      expect(read.estimate[:means].keys).to eq(%i[a b c])
      expect(merged.arms).to eq(%w[a b c])
      expect(merged.estimate[:means].keys).to eq(%w[a b c])
      expect_pins(merged.estimate[:covariance], shared_estimate[:covariance], "merged")
    end

    it "SL5 merge answers a new accumulator and changes neither; add answers the accumulator itself" do
      first = shared_credit.accumulator(arms: arms, parts: parts)
      expect(first.add(shared_rows[0])).to equal(first)
      second = shared_credit.accumulator(arms: arms, parts: parts)
      shared_rows.drop(1).each { |row| second.add(row) }
      first_before = first.to_h
      second_before = second.to_h

      merged = first.merge(second)
      merged.add(shared_rows[2])

      expect(merged).not_to equal(first)
      expect(merged).not_to equal(second)
      expect(first.to_h).to eq(first_before)
      expect(second.to_h).to eq(second_before)
    end

    it "SL5 refuses a row before it changes the accumulator" do
      accumulator = shared_credit.accumulator(arms: arms, parts: parts)
      accumulator.add(shared_rows[0])
      before = accumulator.to_h

      expect_refused("a bad credit after a good one") do
        accumulator.add({ credit: { a: 1.0, b: Float::NAN }, values: [ 1.0, 1.0, 1.0 ] })
      end
      expect_refused("a bad last value") { accumulator.add({ credit: { a: 1.0 }, values: [ 1.0, 1.0, Float::INFINITY ] }) }
      expect(accumulator.to_h).to eq(before)
    end

    it "SL5 refuses a malformed row" do
      good = { credit: { a: 1.0 }, values: [ 1.0, 0.0, 2.0 ] }
      expect(shared_credit.estimate([ good ], arms: arms, parts: parts)[:means][:a]).to eq([ 1.0, 0.0, 2.0 ])
      {
        "a row that is no Hash" => [ [ { a: 1.0 }, [ 1.0, 0.0, 2.0 ] ] ],
        "a row without values" => [ { credit: { a: 1.0 } } ],
        "a row without credit" => [ { values: [ 1.0, 0.0, 2.0 ] } ],
        "a row with a key of its own" => [ good.merge(weight: 1.0) ],
        "a row naming credit twice" => [ good.merge("credit" => { a: 1.0 }) ],
        "a credit that is no Hash" => [ good.merge(credit: [ :a, 1.0 ]) ],
        "a credit to no arm of the call" => [ good.merge(credit: { z: 1.0 }) ],
        "a credit to one arm twice" => [ good.merge(credit: { :a => 1.0, "a" => 1.0 }) ],
        "a credit below 0" => [ good.merge(credit: { a: -0.5 }) ],
        "values that are no Array" => [ good.merge(values: { p: 1.0 }) ],
        "values one short" => [ good.merge(values: [ 1.0, 0.0 ]) ],
        "values one too many" => [ good.merge(values: [ 1.0, 0.0, 2.0, 3.0 ]) ]
      }.each do |label, rows|
        expect_refused(label) { shared_credit.estimate(rows, arms: arms, parts: parts) }
      end
    end

    it "SL5 reads a number as the other modules do (SL18's one check): Integers and Rationals as Floats, the rest refused" do
      as_floats = [ { credit: { a: 2.5, b: 2.0 }, values: [ 1.0, 0.0, 3.0 ] }, { credit: { a: 0.5 }, values: [ 2.0, 1.0, 0.0 ] } ]
      as_given = [ { credit: { a: Rational(5, 2), b: 2 }, values: [ 1, 0, 3 ] }, { credit: { a: Rational(1, 2) }, values: [ 2, 1, 0 ] } ]

      expect(shared_credit.estimate(as_given, arms: arms, parts: parts))
        .to eq(shared_credit.estimate(as_floats, arms: arms, parts: parts))
      [ Float::NAN, Float::INFINITY, -Float::INFINITY, Complex(1, 0), 10**400, Rational(10**400, 3), "1.0", nil ].each do |bad|
        expect_refused("a credit of #{bad.inspect}") do
          shared_credit.estimate([ { credit: { a: bad }, values: [ 1.0, 0.0, 0.0 ] } ], arms: arms, parts: parts)
        end
        expect_refused("a value of #{bad.inspect}") do
          shared_credit.estimate([ { credit: { a: 1.0 }, values: [ 1.0, bad, 0.0 ] } ], arms: arms, parts: parts)
        end
      end
    end

    it "SL5 refuses arms or parts that are not one or more distinct Symbols or Strings" do
      [ :a, [], [ 1 ], %i[a a], [ :a, "a" ], nil ].each do |bad|
        expect_refused("arms #{bad.inspect}") { shared_credit.accumulator(arms: bad, parts: parts) }
        expect_refused("parts #{bad.inspect}") { shared_credit.accumulator(arms: arms, parts: bad) }
        expect_refused("estimate's arms #{bad.inspect}") { shared_credit.estimate([], arms: bad, parts: parts) }
        expect_refused("from_sums' parts #{bad.inspect}") { shared_credit.from_sums({}, arms: arms, parts: bad) }
      end
    end

    it "SL5 refuses rows that are no Enumerable, takes none at all, and reads a lazy Enumerator once" do
      passes = 0
      lazy = Enumerator.new do |yielder|
        passes += 1
        shared_rows.each { |row| yielder << row }
      end.lazy
      empty = shared_credit.estimate([], arms: arms, parts: parts)

      expect(shared_credit.estimate(lazy, arms: arms, parts: parts)).to eq(shared_estimate)
      expect(passes).to eq(1)
      expect(empty[:means]).to eq({ a: nil, b: nil, c: nil })
      expect(empty[:weight_sums]).to eq({ a: 0.0, b: 0.0, c: 0.0 })
      expect(empty[:effective]).to eq({ a: 0.0, b: 0.0, c: 0.0 })
      expect(empty[:covariance]).to eq(Array.new(9) { Array.new(9, 0.0) })
      expect_refused("a number for rows") { shared_credit.estimate(5, arms: arms, parts: parts) }
    end

    it "SL5 refuses a merge with other arms, other parts, another order or another object" do
      accumulator = shared_credit.accumulator(arms: arms, parts: parts)

      expect_refused("other arms") { accumulator.merge(shared_credit.accumulator(arms: %i[a b], parts: parts)) }
      expect_refused("other parts") { accumulator.merge(shared_credit.accumulator(arms: arms, parts: %i[p q s])) }
      expect_refused("another order") { accumulator.merge(shared_credit.accumulator(arms: %i[b a c], parts: parts)) }
      expect_refused("a Hash") { accumulator.merge(accumulator.to_h) }
    end

    it "SL5 refuses a malformed hash in from_h" do
      accumulator = shared_credit.accumulator(arms: arms, parts: parts)
      shared_rows.first(2).each { |row| accumulator.add(row) }
      good = json(accumulator.to_h)
      expect(shared_credit::Accumulator.from_h(good).to_h).to eq(good)

      {
        "no Hash" => good.to_a,
        "a missing section" => good.except("pairs"),
        "a key of its own" => good.merge("rows" => 2),
        "another format" => good.merge("format" => 2),
        "a credit to no arm" => good.merge("sums" => good["sums"].merge("z" => { "w" => 1.0, "x" => {} })),
        "a sum of no part" => good.merge("sums" => good["sums"].merge("b" => { "w" => 2.5, "x" => { "s" => 1.0 } })),
        "a negative credit" => good.merge("sums" => good["sums"].merge("b" => { "w" => -2.5, "x" => { "p" => 2.5 } })),
        "a pair in the wrong order" => good.merge("pairs" => good["pairs"].merge("b" => good["pairs"]["b"].merge(
          "a" => { "w" => 6.25, "x" => { "p" => 6.25 }, "xx" => { "p" => { "p" => 6.25 } } }))),
        "parts in the wrong order" => good.merge("pairs" => good["pairs"].merge("b" => { "b" => {
          "w" => 6.25, "x" => { "p" => 6.25 }, "xx" => { "q" => { "p" => 6.25 } } } })),
        "a credited arm without its own pair" => good.merge("pairs" => good["pairs"].except("b")),
        "a pair of an arm with no credit" => good.merge("pairs" => good["pairs"].merge("c" => { "c" => {
          "w" => 1.0, "x" => {}, "xx" => {} } })),
        "a pair whose weight is 0 but its sums are not" => good.merge("pairs" => good["pairs"].merge("b" => { "b" => {
          "w" => 0.0, "x" => { "p" => 6.25 }, "xx" => {} } })),
        "a sum that is no number" => good.merge("sums" => good["sums"].merge("b" => { "w" => 2.5, "x" => { "p" => "2.5" } }))
      }.each do |label, hash|
        expect_refused(label) { shared_credit::Accumulator.from_h(hash) }
      end
    end

    describe ".from_sums" do
      def triangle(sums, upper:)
        sums.transform_values do |entry|
          entry.merge(c: entry[:c].select { |(first, second), _| upper ? first <= second : first >= second })
        end
      end

      it "SL5 reads c from either order of a pair and refuses two orders that differ" do
        both = shared_credit.from_sums(single_sums, arms: %i[a b], parts: parts)
        upper = shared_credit.from_sums(triangle(single_sums, upper: true), arms: %i[a b], parts: parts)
        lower = shared_credit.from_sums(triangle(single_sums, upper: false), arms: %i[a b], parts: parts)
        unequal = single_sums.merge(a: single_sums[:a].merge(c: single_sums[:a][:c].merge([ :q, :p ] => 22.3)))

        expect(upper).to eq(both)
        expect(lower).to eq(both)
        expect_refused("two orders that differ") { shared_credit.from_sums(unequal, arms: %i[a b], parts: parts) }
      end

      it "SL5 takes an arm left out, or one whose w is 0 with every other sum 0, as an arm with no credit" do
        left_out = shared_credit.from_sums(single_sums, arms: %i[a b c], parts: parts)
        zero = shared_credit.from_sums(single_sums.merge(c: { w: 0.0, w2: 0.0, a: { p: 0.0 } }), arms: %i[a b c],
                                       parts: parts)

        expect(left_out).to eq(zero)
        expect(left_out[:means][:c]).to be_nil
        expect(left_out[:weight_sums][:c]).to eql(0.0)
        expect(left_out[:effective][:c]).to eql(0.0)
        expect(left_out[:covariance].first(6).map { |row| row.first(6) })
          .to eq(shared_credit.from_sums(single_sums, arms: %i[a b], parts: parts)[:covariance])
        expect(left_out[:covariance][6..]).to all(eq(Array.new(9, 0.0)))
      end

      it "SL5 reads the sums' keys by Symbol or String and every pair's names by Symbol or String" do
        by_strings = single_sums.to_h do |arm, entry|
          [ arm.to_s, { "w" => entry[:w], "w2" => entry[:w2], "a" => entry[:a].transform_keys(&:to_s),
                        "b" => entry[:b].transform_keys(&:to_s), "c" => entry[:c].transform_keys { |pair| pair.map(&:to_s) } } ]
        end

        expect(shared_credit.from_sums(by_strings, arms: %i[a b], parts: parts))
          .to eq(shared_credit.from_sums(single_sums, arms: %i[a b], parts: parts))
      end

      it "SL5 refuses malformed sums" do
        a = single_sums[:a]
        {
          "sums that are no Hash" => single_sums.to_a,
          "sums of no arm of the call" => single_sums.merge(z: a),
          "an arm's sums that are no Hash" => single_sums.merge(a: [ 1.0 ]),
          "a key of its own" => single_sums.merge(a: a.merge(visits: 3)),
          "no w" => single_sums.merge(a: a.except(:w)),
          "no w2" => single_sums.merge(a: a.except(:w2)),
          "a negative w" => single_sums.merge(a: a.merge(w: -1.0)),
          "w2 of 0 beside a w above 0" => single_sums.merge(a: a.merge(w2: 0.0)),
          "a w of 0 beside other sums" => single_sums.merge(a: { w: 0.0, w2: 0.0, a: { p: 1.0 } }),
          "no b" => single_sums.merge(a: a.except(:b)),
          "a part left out of a" => single_sums.merge(a: a.merge(a: a[:a].except(:q))),
          "a part of no name in b" => single_sums.merge(a: a.merge(b: a[:b].merge(s: 1.0))),
          "a pair left out of c" => single_sums.merge(a: a.merge(c: a[:c].except([ :p, :q ], [ :q, :p ]))),
          "a pair named as one String" => single_sums.merge(a: a.merge(c: a[:c].merge("p|s" => 1.0))),
          "a pair of one name" => single_sums.merge(a: a.merge(c: a[:c].merge([ :p ] => 1.0))),
          "a pair of a part of no name" => single_sums.merge(a: a.merge(c: a[:c].merge([ :p, :s ] => 1.0))),
          "a sum that is not finite" => single_sums.merge(a: a.merge(b: a[:b].merge(q: Float::NAN)))
        }.each do |label, sums|
          expect_refused(label) { shared_credit.from_sums(sums, arms: %i[a b], parts: parts) }
        end
      end
    end

    describe ".linear and .contrast" do
      it "SL5 reads the weights as an Array in the parts' order, or as a Hash by part name in any order" do
        estimate = shared_estimate
        by_hash = shared_credit.linear(estimate, weights: { p: 1.0, q: 0.35, r: 2.5 })

        expect(shared_credit.linear(estimate, weights: [ 1.0, 0.35, 2.5 ])).to eq(by_hash)
        expect(shared_credit.linear(estimate, weights: { r: 2.5, q: 0.35, p: 1.0 })).to eq(by_hash)
        expect(shared_credit.linear(estimate, weights: { "q" => Rational(7, 20), :r => 2.5, "p" => 1 })).to eq(by_hash)
        expect(shared_credit.contrast(estimate, :a, :c, weights: { r: 2.5, p: 1.0, q: 0.35 }))
          .to eq(shared_credit.contrast(estimate, :a, :c, weights: [ 1.0, 0.35, 2.5 ]))
      end

      it "SL5 never reads a Hash of weights by position: a part left out, a name of no part or a part twice is refused" do
        estimate = shared_estimate

        [
          [ 1.0, 0.35 ], [ 1.0, Float::NAN, 2.5 ], 1.0,
          { p: 1.0, q: 0.35 },
          { p: 1.0, q: 0.35, r: 2.5, s: 1.0 },
          { "x" => 1, "y" => 0.35, "z" => 2.5 },
          { :p => 1.0, "p" => 1.0, :q => 0.35, :r => 2.5 },
          { p: 1.0, q: 0.35, r: Float::NAN },
          { p: 1.0, q: 0.35, r: 10**400 }
        ].each do |bad|
          expect_refused("weights #{bad.inspect}") { shared_credit.linear(estimate, weights: bad) }
          expect_refused("contrast weights #{bad.inspect}") { shared_credit.contrast(estimate, :a, :b, weights: bad) }
        end
      end

      it "SL5 reads a Hash of weights by name on from_sums, on an accumulator's estimate and on parts named by Strings" do
        from_sums = shared_credit.from_sums(single_sums, arms: %i[a b], parts: parts)
        accumulator = shared_credit.accumulator(arms: arms, parts: parts)
        shared_rows.each { |row| accumulator.add(row) }
        read = shared_credit::Accumulator.from_h(json(accumulator.to_h))
        by_strings = shared_credit.estimate(shared_rows, arms: arms, parts: %w[p q r])
        weights = { r: 2.5, q: 0.35, p: 1.0 }

        [ from_sums, read.estimate, by_strings ].each do |estimate|
          expect(shared_credit.linear(estimate, weights: weights)).to eq(shared_credit.linear(estimate, weights: [ 1.0, 0.35, 2.5 ]))
        end
      end

      it "SL5 answers an estimate that is a Hash of its four keys and names its parts" do
        estimate = shared_estimate

        expect(estimate).to be_a(Hash)
        expect(estimate.keys).to eq(%i[means covariance weight_sums effective])
        expect(estimate.parts).to eq(parts)
        expect(shared_credit.estimate(shared_rows, arms: arms, parts: %w[p q r]).parts).to eq(%w[p q r])
        expect(shared_credit.from_sums(single_sums, arms: %i[a b], parts: parts).parts).to eq(parts)
        expect(estimate.to_h).to eq(estimate)
      end

      it "SL5 refuses a Hash of weights for an estimate that names no parts, or whose parts are not its covariance's" do
        estimate = shared_estimate
        hash = { p: 1.0, q: 0.35, r: 2.5 }

        expect_refused("a plain Hash") { shared_credit.linear(estimate.to_h, weights: hash) }
        expect_refused("one read back from JSON") { shared_credit.linear(json(estimate), weights: hash) }
        expect_refused("contrast on one read back from JSON") { shared_credit.contrast(json(estimate), :a, :b, weights: hash) }
        expect_refused("other parts beside the covariance") do
          shared_credit.linear(estimate.merge(covariance: estimate[:covariance].first(6).map { |row| row.first(6) },
                                              means: estimate[:means].transform_values { |row| row.first(2) }),
                               weights: hash)
        end
      end

      it "SL5 answers nil for an arm with no credit, and 0 for an arm against itself" do
        estimate = shared_credit.estimate(shared_rows, arms: %i[a b c d], parts: parts)
        linear = shared_credit.linear(estimate, weights: [ 1.0, 0.35, 2.5 ])

        expect(linear[:means][:d]).to be_nil
        expect(linear[:covariance][3]).to eq([ 0.0, 0.0, 0.0, 0.0 ])
        expect(linear[:covariance].map(&:last)).to eq([ 0.0, 0.0, 0.0, 0.0 ])
        expect(shared_credit.contrast(estimate, :a, :d, weights: [ 1.0, 0.35, 2.5 ])).to eq({ mean: nil, variance: nil })
        expect(shared_credit.contrast(estimate, :d, :d, weights: [ 1.0, 0.35, 2.5 ])).to eq({ mean: nil, variance: nil })
        expect(shared_credit.contrast(estimate, :b, :b, weights: [ 1.0, 0.35, 2.5 ])).to eq({ mean: 0.0, variance: 0.0 })
      end

      it "SL5 answers 0.0 for a variance that rounding takes below zero" do
        # 1 - 1 - 1 + 0.9999999999999999 is -1.1e-16, the rounding of a matrix that is positive semidefinite by hand.
        one_arm = { means: { a: [ 1.0, 1.0 ] }, covariance: [ [ 1.0, -1.0 ], [ -1.0, 0.9999999999999999 ] ] }
        two_arms = { means: { a: [ 1.0 ], b: [ 1.0 ] }, covariance: [ [ 1.0, 1.0 ], [ 1.0, 0.9999999999999999 ] ] }

        expect(shared_credit.linear(one_arm, weights: [ 1.0, 1.0 ])).to eq({ means: { a: 2.0 }, covariance: [ [ 0.0 ] ] })
        expect(shared_credit.contrast(two_arms, :a, :b, weights: [ 1.0 ])).to eq({ mean: 0.0, variance: 0.0 })
      end

      it "SL5 reads an estimate that went through JSON, and arms by Symbol or String" do
        estimate = shared_estimate
        parsed = json(estimate)
        weights = [ 1.0, 0.35, 2.5 ]
        linear = shared_credit.linear(estimate, weights: weights)

        expect(shared_credit.linear(parsed, weights: weights))
          .to eq({ means: linear[:means].transform_keys(&:to_s), covariance: linear[:covariance] })
        expect(shared_credit.contrast(parsed, :a, "b", weights: weights))
          .to eq(shared_credit.contrast(estimate, "a", :b, weights: weights))
      end

      it "SL5 refuses an arm the estimate does not name and an estimate of the wrong shape" do
        estimate = shared_estimate
        weights = [ 1.0, 0.35, 2.5 ]

        expect_refused("an arm of no name") { shared_credit.contrast(estimate, :a, :z, weights: weights) }
        {
          "no Hash" => estimate.to_a,
          "no means" => estimate.except(:means),
          "no covariance" => estimate.except(:covariance),
          "no arm" => estimate.merge(means: {}),
          "a covariance one row short" => estimate.merge(covariance: estimate[:covariance].first(8)),
          "a covariance row one short" => estimate.merge(covariance: estimate[:covariance].map { |row| row.first(8) }),
          "means one part short" => estimate.merge(means: estimate[:means].merge(a: [ 1.0, 0.5 ])),
          "an entry that is no number" => estimate.merge(covariance: [ [ nil ] * 9 ] * 9)
        }.each do |label, bad|
          expect_refused(label) { shared_credit.linear(bad, weights: weights) }
        end
      end
    end

    # An entry of the covariance is four terms that cancel: where the exact sandwich is 0 the sums leave rounding of
    # either sign, which answers exactly 0.0, and LinearValue takes the matrix. By hand: an arm whose credited rows all
    # hold one value has r_ap = that value, so every (x_ip - r_ap) is 0.
    describe "entries that cancel" do
      def expect_zeros(matrix, label)
        matrix.each_with_index do |row, i|
          row.each_with_index { |value, j| expect(value).to eql(0.0), "#{label}: entry (#{i}, #{j}) is #{value}" }
        end
      end

      # A summary at fixed weight 1 per part, which is refused if Cholesky does not take the covariance.
      def summary_of(estimate, chosen_arms, chosen_parts)
        SkiftetStatistical::LinearValue.summary(
          arms: chosen_arms, parts: chosen_parts, means: estimate[:means], covariance: estimate[:covariance],
          parameters: { one: { fixed: 1.0 } }, weights: ->(drawn) { chosen_parts.to_h { |part| [ part, drawn.fetch(:one) ] } },
          draws: 20, rng: Random.new(3)
        )
      end

      def folded(rows, chosen_arms, chosen_parts)
        rows.each_with_object(shared_credit.accumulator(arms: chosen_arms, parts: chosen_parts)) { |row, sum| sum.add(row) }
      end

      it "SL5 gives exactly 0.0 for one row per arm, and LinearValue takes it" do
        rows = [ { credit: { a: 1.0 / 0.37 }, values: [ 3.0 ] }, { credit: { b: 1.0 / 0.63 }, values: [ 1.0 ] } ]
        estimate = shared_credit.estimate(rows, arms: %i[a b], parts: %i[p])
        summary = summary_of(estimate, %i[a b], %i[p])

        expect_zeros(estimate[:covariance], "one row per arm")
        expect(summary[:a][:p_best]).to eq(1.0)
        expect(summary[:b][:p_best]).to eq(0.0)
        expect(summary[:a][:expected_loss]).to eq(0.0)
        expect(summary[:a][:lower]).to eq(summary[:a][:upper])
      end

      it "SL5 gives exactly 0.0 for one row credited to both arms, and for rows of one value" do
        both = shared_credit.estimate([ { credit: { a: 1.0 / 0.37, b: 1.0 / 0.41 }, values: [ 3.0 ] } ], arms: %i[a b], parts: %i[p])
        three = shared_credit.estimate([ 0.2, 0.3, 0.45 ].map { |share| { credit: { a: 1.0 / share }, values: [ 3.0 ] } },
                                       arms: %i[a], parts: %i[p])
        weight = 1.0 / 0.37
        sums = shared_credit.from_sums({ a: { w: weight, w2: weight * weight, a: { p: 3.0 * weight }, b: { p: 3.0 * weight * weight },
                                              c: { %i[p p] => 9.0 * weight * weight } } }, arms: %i[a], parts: %i[p])

        expect_zeros(both[:covariance], "credit to both")
        expect_zeros(three[:covariance], "three rows")
        expect_zeros(sums[:covariance], "from_sums")
        expect(summary_of(both, %i[a b], %i[p]).keys).to eq(%i[a b])
        expect(summary_of(three, %i[a], %i[p]).keys).to eq(%i[a])
        expect(summary_of(sums, %i[a], %i[p]).keys).to eq(%i[a])
      end

      it "SL5 gives exactly 0.0 on three thousand rows at one share and one value, whose rounding grows with every row" do
        rows = Array.new(3_000) { { credit: { a: 1.0 / 0.37 }, values: [ 0.7 ] } }
        estimate = shared_credit.estimate(rows, arms: %i[a], parts: %i[p])

        expect_zeros(estimate[:covariance], "one share")
        expect(summary_of(estimate, %i[a], %i[p]).keys).to eq(%i[a])
      end

      it "SL5 clears the row of a part with no variance, its other parts' covariance kept to the last bit" do
        # Part p holds 3.0 on every row, so r_ap = r_bp = 3 and rows and columns (a, p) and (b, p) are 0 by hand. Part q
        # varies and is the only part of `without`, whose sums for q are the same sums.
        q_values = [ 1.0, 0.0, 2.5, 0.4 ]
        shares = [ [ 0.37, 0.41 ], [ 0.63, 0.59 ], [ 0.37, 0.59 ], [ 0.63, 0.41 ] ]
        credit = shares.map { |first, second| { a: 1.0 / first, b: 1.0 / second } }
        with = shared_credit.estimate(credit.zip(q_values).map { |row, value| { credit: row, values: [ 3.0, value ] } },
                                      arms: %i[a b], parts: %i[p q])
        without = shared_credit.estimate(credit.zip(q_values).map { |row, value| { credit: row, values: [ value ] } },
                                         arms: %i[a b], parts: %i[q])
        covariance = with[:covariance]

        [ 0, 2 ].each do |cell|
          expect(covariance[cell]).to all(eql(0.0))
          expect(covariance.map { |row| row[cell] }).to all(eql(0.0))
        end
        expect([ [ covariance[1][1], covariance[1][3] ], [ covariance[3][1], covariance[3][3] ] ]).to eq(without[:covariance])
        expect(covariance[1][1]).to be > 0.0
        expect(summary_of(with, %i[a b], %i[p q]).keys).to eq(%i[a b])
      end

      it "SL5 keeps a real covariance far below its terms" do
        # By hand: W = 4, r = 1000.25, psi = 2 (x - r) / 4 = -0.125 and 0.125, so 2 * 0.125^2 = 0.03125, while each of the
        # entry's four terms before the division by W^2 is about 8e6. Every number is exact in binary.
        rows = [ { credit: { a: 2.0 }, values: [ 1000.0 ] }, { credit: { a: 2.0 }, values: [ 1000.5 ] } ]

        expect_pins(shared_credit.estimate(rows, arms: %i[a], parts: %i[p])[:covariance], [ [ 0.03125 ] ], "the variance")
      end

      it "SL5 gives exactly 0.0 on seeded constant arms, credit shared or not, shares alike or not, folded any way" do
        seeded = Random.new(20_261_030)
        chosen_arms = %i[a b c d e]
        chosen_parts = %i[p q r s]
        constants = [ 3.0, 0.7, 1.3, 7.0, 1000.37, 0.1, 0.0, 1.0, 1e-6, 1e6 ]

        300.times do |trial|
          arm_count = 1 + seeded.rand(chosen_arms.size)
          part_count = 1 + seeded.rand(chosen_parts.size)
          trial_arms = chosen_arms.first(arm_count)
          trial_parts = chosen_parts.first(part_count)
          # Arms that share rows share their values, so each group of arms holds one list of values.
          groups = trial_arms.group_by { seeded.rand(1 + seeded.rand(arm_count)) }.values
          values = groups.map { Array.new(part_count) { seeded.rand(2).zero? ? constants[seeded.rand(constants.size)] : seeded.rand * 10 } }
          alike = seeded.rand(2).zero?
          shares = trial_arms.to_h { |arm| [ arm, 0.05 + (seeded.rand * 0.9) ] }
          credit = ->(arm) { 1.0 / (alike ? shares.fetch(arm) : 0.05 + (seeded.rand * 0.9)) }
          # Every arm is credited once, so every arm has means LinearValue reads.
          rows = groups.each_with_index.flat_map do |group, index|
            group.map { |arm| { credit: { arm => credit.call(arm) }, values: values[index] } }
          end
          rows += Array.new(seeded.rand(200)) do
            index = seeded.rand(groups.size)
            chosen = groups[index].select { seeded.rand(2).zero? }
            { credit: chosen.to_h { |arm| [ arm, credit.call(arm) ] }, values: values[index] }
          end
          whole = folded(rows, trial_arms, trial_parts)
          cut = seeded.rand(rows.size)
          halves = folded(rows.first(cut), trial_arms, trial_parts).merge(folded(rows.drop(cut), trial_arms, trial_parts))
          read = shared_credit::Accumulator.from_h(json(whole.to_h))

          [ whole, halves ].each_with_index do |accumulator, way|
            estimate = accumulator.estimate
            expect_zeros(estimate[:covariance], "trial #{trial}, way #{way}")
            expect(summary_of(estimate, trial_arms, trial_parts).keys).to eq(trial_arms)
          end
          expect(read.estimate[:covariance]).to eq(whole.estimate[:covariance])
        end
      end
    end

    # A covariance entry keeps its rounding unless it is cleared. Estimate#rounding is the largest bound the snap used,
    # in the covariance's units: (8 + n) * Float::EPSILON times the sum of the entry's four terms' sizes, over W_a * W_b.
    describe "Estimate#rounding" do
      def draws(estimate, chosen_arms, chosen_parts, rounding)
        SkiftetStatistical::LinearValue.each_draw(
          arms: chosen_arms, parts: chosen_parts, means: estimate[:means], covariance: estimate[:covariance],
          parameters: { one: { fixed: 1.0 } }, weights: ->(drawn) { chosen_parts.to_h { |part| [ part, drawn.fetch(:one) ] } },
          draws: 5, rng: Random.new(3), rounding: rounding
        ).to_a
      end

      def names(prefix, count)
        (0...count).map { |index| :"#{prefix}#{index}" }
      end

      # Rows of mostly zero values at the module's own bounds: counts, yes-or-no parts and amounts, a row crediting
      # about two in five arms at 1 / p.
      def ordinary_rows(seed, arm_count, part_count, row_count)
        rng = Random.new(seed)
        kinds = Array.new(part_count) { [ %i[count binary amount][rng.rand(3)], [ 0.03, 0.1, 0.3, 0.6 ][rng.rand(4)] ] }
        shares = Array.new(arm_count) { 0.02 + (0.3 * rng.rand) }
        total = shares.sum
        shares = shares.map { |share| share / total }
        chosen = names("a", arm_count)
        Array.new(row_count) do
          drawn = (0...arm_count).select { rng.rand < 0.4 }
          drawn = [ rng.rand(arm_count) ] if drawn.empty?
          credit = drawn.to_h { |arm| [ chosen[arm], 1.0 / [ shares[arm] * 1.5, 1.0 ].min ] }
          values = kinds.map do |kind, rate|
            case kind
            when :count then rng.rand < rate ? (1 + rng.rand(3)).to_f : 0.0
            when :binary then rng.rand < rate ? 1.0 : 0.0
            else rng.rand < rate ? [ 30.0, 50.0, 100.0, 200.0, 500.0 ][rng.rand(5)] : 0.0
            end
          end
          { credit: credit, values: values }
        end
      end

      it "SL5 is the largest bound, worked by hand on two rows" do
        # W = 4, r = 1000.25. The pair's sums: w = 8, x = 8002, xx = 8004001. The terms: 8004001, r * 8002 = 8004000.5
        # twice, r * r * 8 = 8004000.5, so their sizes sum to 32016002.5. n is 2 and W * W is 16.
        rows = [ { credit: { a: 2.0 }, values: [ 1000.0 ] }, { credit: { a: 2.0 }, values: [ 1000.5 ] } ]

        expect_pin(shared_credit.estimate(rows, arms: %i[a], parts: %i[p]).rounding, 10.0 * Float::EPSILON * 32_016_002.5 / 16.0, "rounding")
      end

      it "SL5 is the largest of them, wherever in the matrix it stands" do
        # Arm a as above. Arm b's terms sum to 13 + 3 * 12.5 = 50.5 and its cell comes last in the matrix.
        rows = [ { credit: { a: 2.0 }, values: [ 1000.0 ] }, { credit: { a: 2.0 }, values: [ 1000.5 ] },
                 { credit: { b: 2.0 }, values: [ 1.0 ] }, { credit: { b: 2.0 }, values: [ 1.5 ] } ]

        expect_pin(shared_credit.estimate(rows, arms: %i[a b], parts: %i[p]).rounding, 10.0 * Float::EPSILON * 32_016_002.5 / 16.0, "rounding")
      end

      it "SL5 stands beside the four keys, on every door, and is 0.0 with no rows" do
        estimate = shared_estimate
        largest = shared_credit::Accumulator.from_h(json(shared_rows.each_with_object(shared_credit.accumulator(arms: arms, parts: parts)) { |row, sum| sum.add(row) }.to_h)).estimate

        expect(estimate.keys).to eq(%i[means covariance weight_sums effective])
        expect(estimate.rounding).to be > 0.0
        expect(largest.rounding).to eq(estimate.rounding)
        expect(shared_credit.from_sums(single_sums, arms: %i[a b], parts: parts).rounding).to be > 0.0
        expect(shared_credit.estimate([], arms: arms, parts: parts).rounding).to eql(0.0)
      end

      it "SL5 takes the rows a covariance Cholesky.factor refuses for its rounding, of lower rank at the module's bounds" do
        [ [ 7, 12, 50, 992 ], [ 5, 12, 40, 643 ] ].each do |arm_count, part_count, row_count, seed|
          chosen_arms = names("a", arm_count)
          chosen_parts = names("p", part_count)
          estimate = shared_credit.estimate(ordinary_rows(seed, arm_count, part_count, row_count), arms: chosen_arms, parts: chosen_parts)

          expect { SkiftetStatistical::Cholesky.factor(estimate[:covariance]) }.to raise_error(ArgumentError)
          expect(draws(estimate, chosen_arms, chosen_parts, estimate.rounding).size).to eq(5)
        end
      end

      it "SL5 takes two parts one offset apart, and rows of two very different credits, on the estimate's rounding alone" do
        {
          "one offset apart" => [ { credit: { a: 1.0 }, values: [ 100.00158489319246, 102.00158489319246 ] },
                                  { credit: { a: 1.0 }, values: [ 100.0, 102.0 ] } ],
          "credits 1e5 and 1" => [ { credit: { a: 1e5 }, values: [ 0.3, 0.9 ] }, { credit: { a: 1.0 }, values: [ 0.7, 0.2 ] } ]
        }.each do |label, rows|
          estimate = shared_credit.estimate(rows, arms: %i[a], parts: %i[p q])

          expect { draws(estimate, %i[a], %i[p q], 0.0) }.to raise_error(ArgumentError), label
          expect(draws(estimate, %i[a], %i[p q], estimate.rounding).size).to eq(5), label
        end
      end

      it "SL5 is never refused on seeded rows of a large level and a small spread, however they were folded" do
        # Every entry is a small difference of large terms, so a covariance of lower rank is all rounding below its rank.
        taken_by_root = 0
        240.times do |trial|
          rng = Random.new(trial + 1)
          arm_count, part_count, row_count = [ [ 7, 12, 50 ], [ 5, 12, 40 ], [ 4, 8, 20 ], [ 3, 6, 10 ], [ 2, 2, 2 ], [ 1, 12, 8 ] ][trial % 6]
          chosen_arms = names("a", arm_count)
          chosen_parts = names("p", part_count)
          level = 10.0**(1 + rng.rand(7))
          rows = ordinary_rows(trial + 1, arm_count, part_count, row_count).map do |row|
            { credit: row[:credit], values: row[:values].map { |value| (value * 1e-3) + level } }
          end
          rows += chosen_arms.map { |arm| { credit: { arm => 1.0 + rng.rand }, values: Array.new(part_count) { level + rng.rand(2) } } }
          cut = rng.rand(rows.size + 1)
          first = rows.first(cut).each_with_object(shared_credit.accumulator(arms: chosen_arms, parts: chosen_parts)) { |row, sum| sum.add(row) }
          second = rows.drop(cut).each_with_object(shared_credit.accumulator(arms: chosen_arms, parts: chosen_parts)) { |row, sum| sum.add(row) }
          estimate = [ first, second ].map { |sum| shared_credit::Accumulator.from_h(json(sum.to_h)) }.reduce(:merge).estimate
          taken_by_root += 1 unless begin
            SkiftetStatistical::Cholesky.factor(estimate[:covariance])
          rescue ArgumentError
            false
          end

          expect(draws(estimate, chosen_arms, chosen_parts, estimate.rounding).size).to eq(5), "trial #{trial}"
        end
        expect(taken_by_root).to be > 20
      end

      it "SL5 clears only what the bound cannot tell from rounding, so a real variance under it reads 0" do
        # Level 1, a coefficient of variation of 5e-6, 1e5 rows: the exact variance of the mean is 2.5e-16, the terms sum
        # to about 4e5, and the bound over W * W is about 1e5 * 2.2e-16 * 4e5 / 1e10 = 8.9e-16. The stated limit.
        rng = Random.new(11)
        rows = Array.new(100_000) { { credit: { a: 1.0 }, values: [ 1.0 + (5e-6 * ((2 * rng.rand) - 1) * Math.sqrt(3.0)) ] } }
        estimate = shared_credit.estimate(rows, arms: %i[a], parts: %i[p])

        expect(estimate[:covariance]).to eq([ [ 0.0 ] ])
        expect(estimate.rounding).to be > 2.5e-16
      end
    end

    # After one heavy row every light row rounds at the heavy sum's scale, so the sums' rounding grows with the rows
    # while Kish's size reads a handful. By hand a constant arm's sandwich is 0.
    describe "the rows folded" do
      let(:constant_rows) do
        [ { credit: { a: 100.0 }, values: [ 1.1 ] } ] + Array.new(100) { { credit: { a: 1.0 }, values: [ 1.1 ] } }
      end

      def folded(rows)
        rows.each_with_object(shared_credit.accumulator(arms: %i[a], parts: %i[p])) { |row, sum| sum.add(row) }
      end

      it "SL5 gives exactly 0.0 on a constant arm whose heavy row comes first, at a Kish's size of 4 over 101 rows" do
        estimate = shared_credit.estimate(constant_rows, arms: %i[a], parts: %i[p])

        expect(estimate[:effective][:a]).to be < 4.0
        expect(estimate[:covariance]).to eq([ [ 0.0 ] ])
      end

      it "SL5 keeps the count where the sums do not say it, through a store and through a merge after one" do
        accumulator = folded(constant_rows)
        stored = json(accumulator.to_h)
        read = shared_credit::Accumulator.from_h(stored)
        halves = [ folded(constant_rows.first(40)), folded(constant_rows.drop(40)) ]
          .map { |half| shared_credit::Accumulator.from_h(json(half.to_h)) }.reduce(:merge)

        expect(stored["sums"]["a"]["rows"]).to eql(101.0)
        expect(read.to_h).to eq(accumulator.to_h)
        expect(read.estimate).to eq(accumulator.estimate)
        expect(read.estimate.rounding).to eq(accumulator.estimate.rounding)
        expect(halves.to_h["sums"]["a"]["rows"]).to eql(101.0)
        expect(halves.estimate[:covariance]).to eq([ [ 0.0 ] ])
      end

      it "SL5 keeps no count that the sums say: Kish's size rounded up is the rows" do
        alike = Array.new(3) { { credit: { a: 2.0 }, values: [ 1.0 ] } }
        apart = folded([ { credit: { a: 3.0 }, values: [ 1.0 ] }, { credit: { a: 1.0 }, values: [ 2.0 ] } ])

        expect(folded(alike).to_h["sums"]["a"].keys).to eq(%w[w x])
        expect(apart.to_h["sums"]["a"].keys).to eq(%w[w x])
        expect(shared_credit::Accumulator.from_h(json(apart.to_h)).estimate.rounding).to eq(apart.estimate.rounding)
      end

      it "SL5 counts an arm's own rows, and keeps a count for the arm that needs one" do
        rows = constant_rows.map { |row| { credit: row[:credit].merge(b: 1.0), values: row[:values] } }.first(4) + constant_rows
        sums = shared_credit.accumulator(arms: %i[a b], parts: %i[p])
        rows.each { |row| sums.add(row) }

        expect(sums.to_h["sums"]["a"]["rows"]).to eql(105.0)
        expect(sums.to_h["sums"]["b"].keys).to eq(%w[w x])
      end

      it "SL5 reads a stored count as a number of at least 0, under an arm's sums and nowhere else" do
        accumulator = folded(constant_rows)
        good = json(accumulator.to_h)
        {
          "a count that is no number" => good.merge("sums" => { "a" => good["sums"]["a"].merge("rows" => "101") }),
          "a negative count" => good.merge("sums" => { "a" => good["sums"]["a"].merge("rows" => -1.0) }),
          "a count beside the arms" => good.merge("rows" => 101.0)
        }.each do |label, hash|
          expect_refused(label) { shared_credit::Accumulator.from_h(hash) }
        end
      end

      it "SL5 reads a hash stored without a count as the rows Kish's size says" do
        hash = json(folded(constant_rows).to_h)
        hash["sums"]["a"].delete("rows")

        expect(shared_credit::Accumulator.from_h(hash).to_h["sums"]["a"].keys).to eq(%w[w x])
      end
    end

    # Every result is finite. Where a product leaves a Float's range nothing is answered: above Float::MAX it is an
    # infinity, and below Float::MIN it loses its digits and then reads 0, either of which would come out as a NaN or a
    # 0.0 that looks like an answer.
    describe "a Float's range" do
      let(:range) { /Float's range/ }

      def estimate_of(rows, chosen_parts = %i[p])
        shared_credit.estimate(rows, arms: %i[a], parts: chosen_parts)
      end

      it "SL18 refuses credits so tiny their squares read 0, and values so tiny theirs do" do
        tiny = [ { credit: { a: 1e-162 }, values: [ 1.0 ] }, { credit: { a: 1e-162 }, values: [ 2.0 ] } ]
        fading = [ { credit: { a: 1e-155 }, values: [ 1.0 ] }, { credit: { a: 1e-155 }, values: [ 2.0 ] } ]
        small = [ { credit: { a: 1.0 }, values: [ 1e-170 ] }, { credit: { a: 1.0 }, values: [ 2e-170 ] } ]

        expect { estimate_of(tiny) }.to raise_error(ArgumentError, range)
        expect { estimate_of(fading) }.to raise_error(ArgumentError, range)
        expect { estimate_of(small) }.to raise_error(ArgumentError, range)
      end

      it "SL18 refuses credits and values whose products overflow" do
        [
          [ { credit: { a: 1.0 }, values: [ 1e154 ] }, { credit: { a: 2.0 }, values: [ -1e154 ] }, { credit: { a: 1.0 }, values: [ 5e153 ] } ],
          [ { credit: { a: 1.0 }, values: [ 1e155 ] }, { credit: { a: 2.0 }, values: [ -1e155 ] }, { credit: { a: 1.0 }, values: [ 5e154 ] } ],
          [ { credit: { a: 1e200 }, values: [ 1.0 ] } ],
          [ { credit: { a: 1e300 }, values: [ 1e10 ] } ]
        ].each do |rows|
          expect { estimate_of(rows) }.to raise_error(ArgumentError, range), rows.inspect
        end
      end

      it "SL18 refuses a row or a merge that would take a sum past Float::MAX, leaving every sum as it was" do
        # One row's w * w * x * x is 1e308, in range. A second would make the sum 2e308; the arms' other sums would fit.
        accumulator = shared_credit.accumulator(arms: %i[a b], parts: %i[p])
        accumulator.add({ credit: { a: 1.0, b: 1.0 }, values: [ 1e154 ] })
        before = json(accumulator.to_h)

        expect { accumulator.add({ credit: { a: 1.0, b: 1.0 }, values: [ 1e154 ] }) }.to raise_error(ArgumentError, range)
        expect(accumulator.to_h).to eq(before)
        expect { accumulator.add({ credit: { a: 2.0 }, values: [ 2e154 ] }) }.to raise_error(ArgumentError, range)
        expect(accumulator.to_h).to eq(before)
        expect { accumulator.merge(accumulator).merge(accumulator) }.to raise_error(ArgumentError, range)
        expect(accumulator.to_h).to eq(before)
      end

      it "SL18 answers tiny and huge credits and values that fit, scaled as the numbers are" do
        rows = [ { credit: { a: 2.0 }, values: [ 1.0 ] }, { credit: { a: 1.0 }, values: [ 3.0 ] }, { credit: { a: 1.5 }, values: [ 2.5 ] } ]
        plain = estimate_of(rows)

        [ [ 1e-140, 1.0 ], [ 1e140, 1.0 ], [ 1.0, 1e-140 ], [ 1.0, 1e140 ], [ 1e60, 1e-60 ] ].each do |credit, value|
          scaled = estimate_of(rows.map { |row| { credit: row[:credit].transform_values { |number| number * credit }, values: row[:values].map { |number| number * value } } })

          expect(scaled[:means][:a][0] / value).to be_within(1e-12).of(plain[:means][:a][0]), [ credit, value ].inspect
          expect(scaled[:covariance][0][0] / value / value).to be_within(1e-12).of(plain[:covariance][0][0]), [ credit, value ].inspect
          expect(scaled[:effective][:a]).to be_within(1e-12).of(plain[:effective][:a]), [ credit, value ].inspect
        end
      end

      it "SL18 refuses finite sums whose estimate would not be, from from_sums and from_h alike" do
        none = { w: 1.0, w2: 1.0, a: { p: 0.0 }, b: { p: 0.0 }, c: { %i[p p] => 0.0 } }
        {
          "a mean that overflows" => none.merge(w: 1e-100, w2: 1e-200, a: { p: 1e300 }),
          "a mean that reads 0" => none.merge(w: 1e100, w2: 1e200, a: { p: 1e-300 }),
          "a weight squared that overflows" => none.merge(w: 1e170),
          "a weight squared that reads 0" => none.merge(w: 1e-170),
          "a Kish's size that overflows" => none.merge(w: 1e150, w2: 1e-50),
          "a Kish's size that reads 0" => none.merge(w: 1e-100, w2: 1e150),
          # r = 1e-100 and b = 1e-250: r * b reads 0 while r * r * w2 = 1e-200 does not, and c = 1e-200 stands for it.
          "a term that reads 0" => none.merge(a: { p: 1e-100 }, b: { p: 1e-250 }, c: { %i[p p] => 1e-200 }),
          "a term that overflows" => none.merge(a: { p: 1e200 }, b: { p: 1e200 }),
          "a mean squared that reads 0" => none.merge(a: { p: 1e-170 }, b: { p: 1.0 }, c: { %i[p p] => 1.0 }),
          # Each term fits and their sum does not: without the refusal the bound is an infinity and clears the entry.
          "terms too large to sum" => none.merge(a: { p: 1e154 }, b: { p: 1e154 }, c: { %i[p p] => 1.5e308 }),
          # The bound still fits here, and the variance 1e110 / 1e-200 does not.
          "a variance that overflows" => none.merge(w: 1e-100, w2: 1e100, c: { %i[p p] => 1e110 })
        }.each do |label, sums|
          expect { shared_credit.from_sums({ a: sums }, arms: %i[a], parts: %i[p]) }.to raise_error(ArgumentError, range), label
        end
        hash = shared_credit.accumulator(arms: %i[a], parts: %i[p]).add({ credit: { a: 1.0 }, values: [ 1.0 ] }).to_h
        hash["sums"]["a"]["w"] = 1e170

        expect { shared_credit::Accumulator.from_h(json(hash)).estimate }.to raise_error(ArgumentError, range)
      end

      it "SL18 refuses either cross term of an entry that reads 0, the other fitting" do
        # Entry (p, q) holds r_p * b_q and r_q * b_p. One is 1e-100 * 1e-250, which reads 0, and the other is 1.
        one = { %i[p p] => 1.0, %i[p q] => 1.0, %i[q q] => 1.0 }
        [
          { w: 1.0, w2: 1.0, a: { p: 1e-100, q: 1.0 }, b: { p: 1.0, q: 1e-250 }, c: one },
          { w: 1.0, w2: 1.0, a: { p: 1.0, q: 1e-100 }, b: { p: 1e-250, q: 1.0 }, c: one }
        ].each do |sums|
          expect { shared_credit.from_sums({ a: sums }, arms: %i[a], parts: %i[p q]) }.to raise_error(ArgumentError, range), sums.inspect
        end
      end

      it "SL18 still stores sums whose estimate it refuses: their rows are not read from a size no Float holds" do
        sums = { a: { w: 1e170, w2: 1.0, a: { p: 0.0 }, b: { p: 0.0 }, c: { %i[p p] => 0.0 } } }
        stored = shared_credit::Accumulator.from_sums(sums, arms: %i[a], parts: %i[p]).to_h

        expect(stored["sums"]["a"]["w"]).to eq(1e170)
        expect(json(stored)).to eq(stored)
      end

      it "SL18 refuses a linear value, a variance or a contrast that overflows, each on its own" do
        small = { means: { a: [ 1.0 ], b: [ 1.0 ] }, covariance: [ [ 1e308, 0.0 ], [ 0.0, 1e308 ] ] }
        large = { means: { a: [ 1e308 ], b: [ -1e308 ] }, covariance: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ] ] }

        expect { shared_credit.linear(large, weights: [ 10.0 ]) }.to raise_error(ArgumentError, range)
        expect { shared_credit.linear(small, weights: [ 10.0 ]) }.to raise_error(ArgumentError, range)
        expect { shared_credit.contrast(large, :a, :b, weights: [ 1.0 ]) }.to raise_error(ArgumentError, range)
        expect { shared_credit.contrast(small, :a, :b, weights: [ 1.0 ]) }.to raise_error(ArgumentError, range)
      end

      it "SL18 answers no NaN where the terms of a linear variance cross" do
        crossing = { means: { a: [ 1.0, 1.0 ] }, covariance: [ [ 1e308, -1e308 ], [ -1e308, 1e308 ] ] }

        expect { shared_credit.linear(crossing, weights: [ 1e10, 1e10 ]) }.to raise_error(ArgumentError, range)
      end
    end
  end
end
