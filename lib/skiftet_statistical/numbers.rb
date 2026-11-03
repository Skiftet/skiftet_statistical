# frozen_string_literal: true

require_relative "inputs"

module SkiftetStatistical
  # The library's one number reader, written once: every module new in 0.4.0 reads every number its caller hands it
  # here (a parameter, a sum of a tally, a share, a probability, a prior, a strength, an age, a cap, a uniform, a
  # weight), then checks its own bounds on the Float it gets back.
  #
  # The input domain: a number is a real Numeric (an Integer, a Float, a Rational, a BigDecimal) whose to_f answers a
  # finite Float of magnitude at most LARGEST. Refused alike, as an ArgumentError: a NaN, an infinity, a Complex
  # (whatever its imaginary part), nil, a String (a numeric one included: "0.5" and "0x14" are not numbers here), any
  # other object (one that answers no method at all, a BasicObject, included), a Numeric that answers no to_f or whose
  # to_f answers anything but a finite Float or raises, and a number beyond LARGEST either side of 0.
  #
  #   SkiftetStatistical::Numbers.read(3, "a count")             # => 3.0
  #   SkiftetStatistical::Numbers.read("0.25", "a share")        # ArgumentError: a share must be a finite number, ...
  #   SkiftetStatistical::Numbers.read(1e200, "a share")         # ArgumentError: a share must lie within the input ...
  #   SkiftetStatistical::Numbers.read(nil, "a weight") { 0.0 }  # => 0.0, the block's answer for what is not a number
  #   SkiftetStatistical::Numbers.pair([ 2, 3 ], "a prior", %w[alpha beta])  # => [2.0, 3.0]
  module Numbers
    # The largest magnitude the library reads: 10**100. Every sum, product and square the new code forms from its inputs
    # then stays inside a Float (whose largest is about 1.8 * 10**308), because no product it forms multiplies more than
    # two numbers of the domain's size: every other factor beside them is a share, a weight or a probability in [0, 1]
    # (a Decay weight, a uniform, a mixing share) or a standard Normal score (at most about 8.3). The products, by the
    # factors of the domain's size in each:
    # - two: a share times a probability (Posteriors.mixture), the weighted outcomes times the views
    #   (WeightedPosterior.beta) and the weighted views squared (its Kish size), each at most 10**200;
    # - one: a standard deviation times a score, a weight times a sum, a mean held at a weight, at most 10**101;
    # - none: the moments of amounts, the amounts rule's mean and gap and every shrunk mean (Bandit's Tallies.moments,
    #   Tallies.amounts, Tallies.shrunk), whose products (a strength times a mean, the views times a sum, a sum
    #   squared, the squares times the views) are formed exactly, as Integers and powers of 2 (Tallies.exact), and
    #   rounded once (Tallies.rounded); and the amounts rule's standard deviation (Tallies.spread), whose three terms
    #   multiply up to four such numbers (k, n and the difference of two means twice) and divide by k + n up to three
    #   times, formed by Math.frexp's mantissas and powers of 2 apart. So no step of either leaves a Float either way,
    #   and only an answer rounds into a Float's range.
    # A sum over n such terms is at most n * 10**200, which leaves a Float only past n = 10**108 terms, no list a
    # computer holds (a sum of n inputs, past 10**208). A quotient by a small input (a sum over views of 10**-300, say)
    # can still leave a Float, but only to an infinity: a clamp or a minimum then reads it as it reads the exact
    # quotient (which is larger than any bound it is clamped to), or it reaches the answer, where the bandit's one check
    # of an answer refuses it. So no sum or product of inputs overflows on its way to an answer, and no answer reads a
    # wrong finite number because one did.
    LARGEST = 1e100

    class << self
      # The value as a Float in [-LARGEST, LARGEST]. A value that is not a number by the rule above is an ArgumentError,
      # "<label> must be a finite number, got <the value inspected>" (Inputs.shown: a value that answers no inspect is
      # named by its class), unless a block is given, whose answer (given the value) is answered instead: a reader
      # with a rule of its own for what is not a number (a weight that counts 0, a bound with its own words). A number
      # beyond LARGEST is always an ArgumentError, "<label> must lie within the input domain, at most <LARGEST> either
      # side of 0 (got <the value inspected>)"; no block waives the domain.
      def read(value, label)
        number = float(value)
        if nil.equal?(number)
          return yield(value) if block_given?

          raise ArgumentError, "#{label} must be a finite number, got #{Inputs.shown(value)}"
        end
        return number if number.abs <= LARGEST

        raise ArgumentError, "#{label} must lie within the input domain, at most #{LARGEST} either side of 0 (got " \
                             "#{Inputs.shown(value)})"
      end

      # THE reading of a pair [first, second], written once for every reader of one (a Beta pair, a Normal pair, a
      # prior, a clip, a member): an Array of exactly two values (read by Array's own to_a, Inputs.elements, never a
      # subclass's methods), else "<label> must be a pair [<names>] (got
      # <value>)"; each value read by `read` as "<label>'s <name>", then, when a block is given, handed to it with
      # that label and its index (0 or 1), and the block's answer kept: the reader's own bound. Answers two Floats.
      def pair(given, label, names)
        elements = Array === given ? Inputs.elements(given) : nil
        unless elements&.size == 2
          raise ArgumentError, "#{label} must be a pair [#{names.join(', ')}] (got #{Inputs.shown(given)})"
        end

        elements.each_with_index.map do |value, index|
          name = "#{label}'s #{names[index]}"
          number = read(value, name)
          block_given? ? yield(number, name, index) : number
        end
      end

      private

      # The value as a finite Float when it is a real Numeric whose to_f answers one, whatever its magnitude; nil for
      # anything else. Its respond_to?, real?, finite? and to_f are calls into the caller's object, asked by the
      # library's one such call (Inputs.asked), told to answer nil when one raises: no number.
      def float(value)
        return nil unless Numeric === value

        number = Inputs.asked(value, "a number's real?, finite? and to_f", otherwise: nil) do
          value.to_f if Inputs.responds?(value, :to_f) && value.real? && value.finite?
        end
        Float === number && number.finite? ? number : nil
      end
    end
  end
end
