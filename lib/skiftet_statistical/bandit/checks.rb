# frozen_string_literal: true

require_relative "../numbers"
require_relative "../inputs"

module SkiftetStatistical
  class Bandit
    # The bounds the bandit's modules share, each written once on a number read first by the library's one number
    # reader (Numbers.read, the input domain), the one reading of a whole number, the one check of an answer, the one
    # lookup in a caller's Hash by a key and the one reading of one by Symbol names. Private to the bandit: Prior,
    # Decay, Tallies, Draw, NormalArgmax, Pass and Posteriors ask these, so a bound reads and words alike wherever it
    # is checked.
    module Checks
      # What `lookup` answers inside `holds?` for a key the Hash does not hold.
      NONE = Object.new.freeze
      private_constant :NONE

      module_function

      # The value as a Float, by Numbers.read, when it is >= 0; else "<label> must be >= 0 (got <number>)".
      def nonnegative(given, label)
        number = Numbers.read(given, label)
        raise ArgumentError, "#{label} must be >= 0 (got #{number})" if number.negative?

        number
      end

      # The value as a Float, by Numbers.read, when it lies in [0, 1] (a share of a rule, a uniform, a mixing share, a
      # mean, a clip bound of a mean); else "<label> must lie in [0, 1], got <number>", the words BetaArgmax's own
      # [0, 1] bound has used since 0.2.0, so the bound reads alike across the library.
      def unit(given, label)
        number = Numbers.read(given, label)
        raise ArgumentError, "#{label} must lie in [0, 1], got #{number}" unless number.between?(0.0, 1.0)

        number
      end

      # The value as a Float, by Numbers.read, when it lies above 0 and below 1 (a credible mass); else "<label> must
      # be a number above 0 and below 1 (got <value>)", a value that is not a number included.
      def open_unit(given, label)
        number = Numbers.read(given, label) { nil }
        return number if number&.positive? && number < 1.0

        raise ArgumentError, "#{label} must be a number above 0 and below 1 (got #{Inputs.shown(given)})"
      end

      # The value as it is given when it is an Integer that the library's one number reader reads (an Integer beyond
      # a Float's range is no number, and one beyond Numbers::LARGEST is refused there, in the input domain's words),
      # at least `minimum` when it is given and below `below` when it is given; else "<label> must be an Integer[ of
      # at least <minimum>][ and below <below>] within a Float's range (got <value>)". A count of draws, a seed, a top,
      # a pinned position.
      def integer(given, label, minimum: nil, below: nil)
        number = Integer === given ? Numbers.read(given, label) { nil } : nil
        return given if number && (nil.equal?(minimum) || given >= minimum) && (nil.equal?(below) || given < below)

        bound = nil.equal?(minimum) ? "" : " of at least #{minimum}"
        bound += "#{nil.equal?(minimum) ? '' : ' and'} below #{below}" unless nil.equal?(below)
        raise ArgumentError, "#{label} must be an Integer#{bound} within a Float's range (got #{Inputs.shown(given)})"
      end

      # The value as a Float, by Numbers.read, when it is above 0; else "<label> must be above 0 (got <number>)".
      def positive(given, label)
        number = Numbers.read(given, label)
        raise ArgumentError, "#{label} must be above 0 (got #{number})" unless number.positive?

        number
      end

      # THE check of an answer, written once, the last guard of every public method of the bandit that answers a
      # number: `answer` (the numbers themselves unless given), as it is, when every number of `numbers` (the answer's
      # numbers, a nil among them a figure the answer leaves out) is finite; else "<inputs> overflow a Float's range
      # (the answer would read <numbers>)", `inputs` naming what was handed in. No sum over fewer than 10**208
      # in-domain inputs and no product the bandit forms of them overflows (Numbers::LARGEST, whose argument names the
      # widest term), so what it refuses is a quotient by a small input that left a Float.
      def finite_answer(numbers, inputs, answer = numbers)
        return answer if numbers.all? { |number| nil.equal?(number) || number.finite? }

        raise ArgumentError, "#{inputs} overflow a Float's range (the answer would read #{Inputs.shown(numbers)})"
      end

      # THE lookup in a caller's Hash by a key of the caller's (an opaque key: a mixture's branch or item) or a Symbol
      # (a pass's leader), written once: the value the Hash holds at `key`, or `missing` (nil) when it holds
      # none, by Hash's own fetch (Inputs.fetched: a subclass's fetch or [] is never called, and a default the Hash
      # holds, a value or a block, never read). A key of the caller's has its hash and eql? called by the lookup, one
      # call into the caller's object, asked by the library's one such call (Inputs.asked), `label` naming the Hash.
      def lookup(hash, key, label, missing = nil)
        Inputs.asked(key, "#{label}'s lookup of a key") { Inputs.fetched(hash, key, missing) }
      end

      # Whether the Hash holds `key` (by `lookup`).
      def holds?(hash, key, label)
        !NONE.equal?(lookup(hash, key, label, NONE))
      end

      # The value the Hash holds at the Symbol key, else at its String, else nil (`named`, for the one key).
      def fetch(hash, key, label = "a Hash")
        named(hash, [ key ], label).fetch(key, nil)
      end

      # THE reading of a caller's Hash by Symbol names, written once (a tally's sums, the given uniforms' parts, an
      # other's group and own, a level's strength): { key => value } for each of `keys` (Symbols) the Hash names, its
      # value at the Symbol key, else at the key's String, read from the Hash's pairs (Inputs.pairs, Hash's own to_a,
      # so a default the Hash holds is never read), a String key by its copy (Inputs.string) and a Symbol key as the
      # Symbol it is; every other key is not read, and no method of a key is called. Every value is then looked up in
      # the Hash this builds, never in the caller's by a key of the library's, so a Hash that compares its keys by
      # identity reads exactly as a plain one. Two String keys of one name (an identity Hash can hold both, as can a
      # key whose own hash differs from its String's) are an ArgumentError, "<label> names <key> twice".
      def named(hash, keys, label)
        symbols = {}
        strings = {}
        Inputs.pairs(hash).each do |name, value|
          if Symbol === name
            symbols[name] = value if keys.include?(name)
          elsif String === name
            copy = Inputs.string(name)
            key = keys.find { |candidate| candidate.name == copy }
            next if nil.equal?(key)
            raise ArgumentError, "#{label} names #{key} twice" if strings.key?(key)

            strings[key] = value
          end
        end
        strings.empty? ? symbols : strings.merge(symbols)
      end
    end
    private_constant :Checks
  end
end
