# frozen_string_literal: true

require "digest"
require_relative "../numbers"
require_relative "../inputs"
require_relative "checks"

module SkiftetStatistical
  class Bandit
    # The randomness helpers, each written once: a text becomes a uniform, a list of parts becomes a uniform or a
    # seed, and a uniform picks a key by weight. Pure: the same text gives the same uniform in every process, on every
    # machine, so a caller can key a draw by a visit, a pass or a test and get it back without storing it.
    #
    #   SkiftetStatistical::Bandit::Draw.uniform("t")                       # => 0.8895498694938413
    #   SkiftetStatistical::Bandit::Draw.uniform_for("t", "explore")        # => 0.856909028643178
    #   SkiftetStatistical::Bandit::Draw.seed_for("t", "explore")           # => 15807181545831752473
    #   SkiftetStatistical::Bandit::Draw.from_weights({ "a" => 1, "b" => 3 }, 0.5)  # => "b"
    #   SkiftetStatistical::Bandit::Draw.weighted({ "a" => 1, "b" => 3 }, 0.5)      # => ["b", 0.75]
    #
    # A uniform is the first 8 bytes of SHA256 of the text, read as one big-endian unsigned Integer, shifted right by
    # 11 and divided by 2**53: a Float in [0, 1) on the 2**53 grid. A seed is those same 8 bytes as the Integer, for
    # Random.new. A list of parts is read as its parts' bytes joined by one NUL byte, nil read as "", a String by its
    # bytes alone and anything else by its to_s (so 7 and "7" are one part), whose String answer is read the same way
    # (Inputs.text: never a String's own methods), a to_s that answers no String being an ArgumentError; a part
    # holding a NUL byte is an ArgumentError, so two different part lists never join to the same text. `uniform` reads
    # its one text the same way (its bytes, nil as ""), with no NUL rule: it has no parts to keep apart. Both readings
    # are the bandit's one reading of a keyed draw's parts (`KeyParts`, below), which the order rule also reads a key
    # by, once, and keeps for a re-pin.
    #
    # The draw by weight takes a Hash of key => weight in input order and a uniform in [0, 1]. A weight that is not a
    # finite number above 0 (0, nil, NaN, an infinity, a negative number, a numeric String) counts 0, and a number
    # beyond the library's input domain (Numbers::LARGEST either side of 0) is an ArgumentError, as every number the
    # library reads is; when every weight counts 0, every key counts 1. The arithmetic is the one 0.1's Softmax policy
    # drew by, step for step: the total is Array#sum of the counted weights (a compensated sum), the target is uniform *
    # total, and the running sum adds the counted weights one by one in input order; the draw answers the first key that
    # counts whose running sum is at or above the target, else (the running sum ending below the target, as a uniform
    # near 1.0 can make it) the last key that counts. A key that counts 0 is never answered, at a uniform of 0.0
    # included. `weighted` answers that key with its share, its counted weight over the counted total, which is the
    # probability that a uniform draw answers it; `from_weights` is `weighted`'s key. Keys are any objects and are
    # answered as given: neither call reads a key, so an app may pass its own records. An empty Hash answers no key
    # (nil, and [nil, nil] from `weighted`). A uniform that is not a number in [0, 1], and counted weights whose sum is
    # not a finite number, are an ArgumentError.
    #
    # A quantile draw reads its uniform on the open grid (`open_uniform`): clamped to [2**-53, 1 - 2**-53], the
    # smallest and the largest uniform above 0 that Random#rand answers, so a quantile that is infinite at 0 or 1 (the
    # Normal's) never reads either end. It is the one such reading, for every quantile draw of the posterior set and
    # for the tail of its Normal interval; a Beta draw reads its uniform as it is (its position is a comparison key,
    # and its value at an end is 0 or 1).
    #
    #   SkiftetStatistical::Bandit::Draw.open_uniform(0.0)  # => 1.1102230246251565e-16 (2**-53)
    module Draw
      # 2**53: the uniform's grid, the most a Float holds without rounding.
      UNIFORM_SCALE = 9_007_199_254_740_992.0
      # The open grid's ends: 2**-53 and 1 - 2**-53.
      OPEN_LOW = 1.0 / UNIFORM_SCALE
      OPEN_HIGH = 1.0 - OPEN_LOW
      # The byte between two parts.
      SEPARATOR = "\x00".b.freeze

      class << self
        # The uniform of `text`, in [0, 1).
        def uniform(text)
          Checks.finite_answer([ (head(KeyParts.bytes(text)) >> 11) / UNIFORM_SCALE ], "the text").first
        end

        # The uniform of the parts joined by NUL, in [0, 1).
        def uniform_for(*parts)
          Checks.finite_answer([ (head(joined(parts)) >> 11) / UNIFORM_SCALE ], "the parts").first
        end

        # The seed of the parts joined by NUL, an Integer in [0, 2**64), for Random.new.
        def seed_for(*parts)
          Checks.finite_answer([ head(joined(parts)) ], "the parts").first
        end

        # The uniform (a number in [0, 1], read by the bandit's one [0, 1] bound, Checks.unit) clamped to the open
        # grid [OPEN_LOW, OPEN_HIGH]: the one reading of a uniform a quantile draw takes.
        def open_uniform(uniform)
          Checks.finite_answer([ Checks.unit(uniform, "uniform").clamp(OPEN_LOW, OPEN_HIGH) ], "the uniform").first
        end

        # The key a uniform draws from `weights`: `weighted`'s key.
        def from_weights(weights, uniform)
          weighted(weights, uniform).first
        end

        # [key, share]: the key a uniform draws from `weights`, and its counted weight over the counted total.
        def weighted(weights, uniform)
          unless Hash === weights
            raise ArgumentError, "weights must be a Hash of key => weight (got #{Inputs.class_name(weights)})"
          end

          target = Checks.unit(uniform, "uniform")
          entries = Inputs.pairs(weights)
          keys = entries.map(&:first)
          counts = entries.map { |_key, weight| counted(weight) }
          total = counts.sum
          raise ArgumentError, "the weights must sum to a finite number (got #{total})" unless total.finite?

          if total.zero?
            counts = Array.new(counts.size, 1.0)
            total = counts.sum
          end
          return [ nil, nil ] if total.zero?

          target *= total
          running = 0.0
          last = nil
          counts.each_with_index do |weight, index|
            next unless weight.positive?

            running += weight
            last = index
            break if running >= target
          end
          share = counts[last] / total
          Checks.finite_answer([ share ], "the weights and the uniform", [ keys[last], share ])
        end

        private

        # The first 8 bytes of SHA256(text) as one big-endian unsigned Integer.
        def head(text)
          Digest::SHA256.digest(text).unpack1("Q>")
        end

        # The parts as the one reading of a keyed draw's parts reads them (`KeyParts.read`), joined by NUL.
        def joined(parts)
          KeyParts.read(*parts).join(SEPARATOR)
        end

        # A weight as the draw counts it: a number (the library's one number reader, Numbers.read, which refuses one
        # beyond the input domain) above 0, else 0.0, what is not a number at all included.
        def counted(weight)
          value = Numbers.read(weight, "a weight") { 0.0 }
          value.positive? ? value : 0.0
        end
      end
    end

    # THE reading of a keyed draw's parts, written once: Draw's `uniform` reads its text here, `uniform_for` and
    # `seed_for` their parts, and the order rule's key source a key's parts, once, keeping what it read for a re-pin
    # (Order#repin), so a re-pin reads the key as `rank` read it and never asks the caller's parts again. What it
    # answers is library data, frozen binary Strings, which Draw reads again as their own bytes. Private to the bandit.
    module KeyParts
      module_function

      # The parts, in order, each as a keyed draw reads it: its bytes (`bytes`), refused when they hold a NUL byte
      # (Draw::SEPARATOR, the byte between two parts), each part read once and refused in its turn; a frozen Array of
      # frozen binary Strings. Taken as a splat, so a caller's Array is read by its elements alone, none of its own
      # methods called. Draw.uniform_for(*read(*parts), *more) is Draw.uniform_for(*parts, *more), bit for bit: a part
      # read here reads again as the same bytes.
      def read(*parts)
        parts.map do |part|
          text = bytes(part)
          if text.include?(Draw::SEPARATOR)
            raise ArgumentError, "a part must not hold a NUL byte (got the bytes #{text.inspect})"
          end

          text
        end.freeze
      end

      # A part's bytes, a frozen binary String: "" for nil, else the bytes of the library's one reading of a value as
      # text (Inputs.text: a String read by its bytes and encoding alone, by Inputs.string; anything else by its to_s,
      # asked by the library's one call into a caller's object, so a to_s that raises is an ArgumentError, and its
      # String answer read the same way), which must be a String.
      def bytes(part)
        return "".b.freeze if nil.equal?(part)

        text = Inputs.text(part, "a part")
        return text.b.freeze unless nil.equal?(text)

        raise ArgumentError, "a part must answer to_s with a String (got #{Inputs.shown(part)})"
      end
    end
    private_constant :KeyParts
  end
end
