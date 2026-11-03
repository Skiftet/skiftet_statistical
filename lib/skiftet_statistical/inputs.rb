# frozen_string_literal: true

module SkiftetStatistical
  # The library's one reading of what a caller hands in besides a number (SkiftetStatistical::Numbers reads those):
  # a random number generator and the values it answers, and a name, each written once, each refusing with an
  # ArgumentError, the one call into a caller's object (`asked`), the one reading of a caller's String (`string`) and
  # the one way a value and a class are shown in a refusal (`shown`, `class_name`). Private to the library: the code
  # new in 0.4.0 asks these; every 0.3.0 module keeps its own readings, as 0.3.0 wrote them.
  #
  # The caller-object contract. Every method the new code calls on a caller's object goes through `asked`: a key
  # part's to_s, a name's hash and eql? (a lookup by it), a list's each, a number's real?, finite? and to_f
  # (Numbers.read), an rng's rand, a status's to_s, an Amount's numbers, Kernel#respond_to? (which asks
  # respond_to_missing?), and, to word a refusal, a value's inspect and an error's message. A StandardError such a
  # call raises, or a NotImplementedError (the error an unfinished method raises), is an ArgumentError naming the
  # input, as an answer of the wrong kind is (or, for the readings that word a refusal, the answer they fall back on).
  # The core values the new code reads from a caller are read by core methods bound to them, never by their own: a
  # String by `string` (String's own b and encoding), a number by Numbers.read, a class's name by Module's own name
  # (`class_name`), a Hash or an Array by Hash's and Array's own methods (`pairs`, `elements`, `fetched`,
  # `with_values`, `merged`), so a subclass's methods, a singleton method or an overridden == are never called, nor a
  # default the Hash holds read. Whatever a caller's object answers is then judged by these one readers. The
  # callables a caller hands in to do its own work (a pass's block, a hold's satisfied_for?) are its own code: what
  # they raise reaches it as raised.
  #
  # An rng is any object that answers rand (asked by Kernel#respond_to?, so an object that answers it only through
  # method_missing, or a BasicObject, is not one); its rand must answer a Float in [0, 1) and its rand(n) an Integer in
  # [0, n). A name is any object that answers hash, eql? and inspect, its hash an Integer, so a Hash can look it up.
  # Each is checked where it is read, before any draw; an answer of rand, which cannot be checked before it is drawn,
  # at the draw that read it (`checked` wraps an rng so that a reader written for a plain rng, such as 0.3.0's
  # Sampler, has each answer checked as it draws it).
  module Inputs
    # Kernel's own methods, bound to the value, so that an object that answers neither (a BasicObject) is read too.
    RESPONDS = Kernel.instance_method(:respond_to?)
    CLASS_OF = Kernel.instance_method(:class)
    # Module's own name, bound to a class, so that a class whose name or to_s is its own is named by Ruby's record.
    MODULE_NAME = Module.instance_method(:name)
    # String's own b and encoding, bound to a caller's String: the one reading of it (`string`).
    STRING_BYTES = String.instance_method(:b)
    STRING_ENCODING = String.instance_method(:encoding)
    # Hash's and Array's own readers, bound to a caller's Hash or Array, so that a subclass's overrides are never
    # called.
    HASH_PAIRS = Hash.instance_method(:to_a)
    HASH_FETCH = Hash.instance_method(:fetch)
    HASH_VALUES = Hash.instance_method(:transform_values)
    HASH_MERGE = Hash.instance_method(:merge)
    ARRAY_ELEMENTS = Array.instance_method(:to_a)
    # What `asked` is told by default to do when the call raises: refuse.
    REFUSE = Object.new.freeze
    private_constant :RESPONDS, :CLASS_OF, :MODULE_NAME, :STRING_BYTES, :STRING_ENCODING, :HASH_PAIRS, :HASH_FETCH,
                     :HASH_VALUES, :HASH_MERGE, :ARRAY_ELEMENTS, :REFUSE

    module_function

    # THE call into a caller's object, written once: the block's answer, the block calling a method of `value` (`label`
    # names the call: "a part's to_s", "may_lead: a name's lookup"). A StandardError or a NotImplementedError the call
    # raises is an ArgumentError, "<label> raised <its class's name>: <its message> (got <value>)", that error its
    # cause; or, when `otherwise` is given, `otherwise` is answered instead (the readings that word a refusal, and
    # Numbers.read, which reads such a call as no number).
    def asked(value, label, otherwise: REFUSE)
      yield
    rescue StandardError, NotImplementedError => e
      return otherwise unless REFUSE.equal?(otherwise)

      raise ArgumentError, "#{label} raised #{class_name(e)}#{described(e)} (got #{shown(value)})"
    end

    # ": <the error's message>", or "" when its message (asked by `asked`) answers no String or raises; the message read
    # by `string` and worded by `words`.
    def described(error)
      text = string(asked(error, "an error's message", otherwise: nil) { error.message })
      nil.equal?(text) ? "" : ": #{words(text)}"
    end

    # The value as a refusal shows it, the one reading that never refuses, since it words the refusals: a String by
    # its copy's inspect (`string`, so none of its own methods is called); any other value by its inspect (asked by
    # `asked`), read by `string`; for a value that answers none (a BasicObject), whose inspect raises or answers no
    # String, "an object of <its class's name>" (`class_name`). Worded by `words`.
    def shown(value)
      copy = string(value)
      return words(copy.inspect) unless nil.equal?(copy)

      text = string(asked(value, "an inspect", otherwise: nil) { value.inspect })
      nil.equal?(text) ? "an object of #{class_name(value)}" : words(text)
    end

    # The name of the value's class, as a refusal words it: read by Kernel#class and Module#name, both bound (never
    # the class's own name or to_s, so building a refusal never calls a caller's method), "an anonymous class" for a
    # class without one; worded by `words`.
    def class_name(value)
      name = MODULE_NAME.bind_call(CLASS_OF.bind_call(value))
      nil.equal?(name) ? "an anonymous class" : words(string(name))
    end

    # THE reading of a caller's String, written once (a key part or its to_s answer, an item name, a status or its
    # to_s answer, a name in a list, and the words of a refusal): a plain, frozen String of the same bytes in the same encoding,
    # copied by String's own b and encoding bound to it, so a subclass's methods, a singleton method (a frozen String
    # keeps those) or an overridden == are never called; nil for a value that is not a String. What the new code reads
    # of the String (==, include?, inspect, its encoding, its bytes) it then reads of the copy, whose methods are
    # String's own.
    def string(value)
      return nil unless String === value

      STRING_BYTES.bind_call(value).force_encoding(STRING_ENCODING.bind_call(value)).freeze
    end

    # THE reading of a value as text, written once (a key part, a status): a String by `string`, so none of its own
    # methods is called, its to_s included; any other value by its to_s (Kernel#respond_to? and the call asked by
    # `asked`, `label` naming the value, so a to_s that raises is an ArgumentError), its answer read by `string`; nil
    # when the value answers no to_s or its to_s answers no String.
    def text(value, label)
      copy = string(value)
      return copy unless nil.equal?(copy)
      return nil unless responds?(value, :to_s, label)

      string(asked(value, "#{label}'s to_s") { value.to_s })
    end

    # A copy (`string`) as a refusal sets it among its words: as it is when its bytes are ASCII in an ASCII-compatible
    # encoding, or valid in the encoding String#inspect answers in (the default internal encoding, else the default
    # external, US-ASCII when that is not ASCII-compatible); else its inspect, its characters escaped in that encoding.
    # So the words of one refusal never meet in incompatible encodings, and building it never raises.
    def words(text)
      return text if text.ascii_only?

      encoding = Encoding.default_internal || Encoding.default_external
      encoding = Encoding::US_ASCII unless encoding.ascii_compatible?
      text.encoding == encoding && text.valid_encoding? ? text : text.inspect
    end

    # A caller's Hash's [key, value] pairs, in its order, by Hash's own to_a: a new Array.
    def pairs(hash)
      HASH_PAIRS.bind_call(hash)
    end

    # A caller's Hash's value at `key`, or `missing` when it holds none, by Hash's own fetch (a default the Hash holds
    # is never read). Looking a caller's key up calls its hash and eql?: the caller asks this through `asked`.
    def fetched(hash, key, missing)
      HASH_FETCH.bind_call(hash, key, missing)
    end

    # A caller's Hash with each value replaced by the block's answer for it, by Hash's own transform_values: a new
    # Hash of its keys as given, never hashed again.
    def with_values(hash, &)
      HASH_VALUES.bind_call(hash, &)
    end

    # A caller's Hash merged with `other`, by Hash's own merge.
    def merged(hash, other)
      HASH_MERGE.bind_call(hash, other)
    end

    # A caller's Array's elements, by Array's own to_a: the Array itself, or for a subclass's, a new plain Array.
    def elements(array)
      ARRAY_ELEMENTS.bind_call(array)
    end

    # Whether the value answers the method, by Kernel#respond_to? (a BasicObject answers nothing), asked by `asked`
    # (respond_to? asks the value's respond_to_missing?), `label` naming the value.
    def responds?(value, method, label = "a value")
      asked(value, "#{label}'s respond_to?(:#{method})") { RESPONDS.bind_call(value, method) }
    end

    # The rng, when it answers rand; else an ArgumentError naming `label`.
    def rng(value, label = "rng")
      return value if responds?(value, :rand, label)

      raise ArgumentError, "#{label} must answer rand (got #{shown(value)})"
    end

    # The rng's next rand (asked by `asked`), when it is a Float in [0, 1); else an ArgumentError.
    def uniform(rng)
      value = asked(rng, "rng.rand") { rng.rand }
      return value if Float === value && value >= 0.0 && value < 1.0

      raise ArgumentError, "rng.rand must answer a Float in [0, 1) (got #{shown(value)})"
    end

    # The rng's next rand(size) (asked by `asked`), when it is an Integer in [0, size); else an ArgumentError.
    def index(rng, size)
      value = asked(rng, "rng.rand(#{size})") { rng.rand(size) }
      return value if Integer === value && value >= 0 && value < size

      raise ArgumentError, "rng.rand(#{size}) must answer an Integer in [0, #{size}) (got #{shown(value)})"
    end

    # The rng (by `rng`), wrapped so that each of its answers is checked when it is drawn: rand by `uniform`, rand(n)
    # by `index`. The wrapper answers the rng's own stream, one rand of the rng a rand of its own; an rng already
    # wrapped is answered as it is.
    def checked(value, label = "rng")
      Checked === value ? value : Checked.new(rng(value, label))
    end

    # Whether the value can name something a Hash looks up: it answers hash, eql? and inspect, and its hash answers
    # an Integer (a Hash raises TypeError on any other hash, and calls eql? on a name whose hash matches a key's). Its
    # hash is asked by `asked`, `label` naming where the name was given: a hash that raises is an ArgumentError.
    def name?(value, label = "a name")
      responds?(value, :hash, label) && responds?(value, :eql?, label) && responds?(value, :inspect, label) &&
        Integer === asked(value, "#{label}: a name's hash") { value.hash }
    end

    # An rng whose answers are checked as they are drawn (`checked`).
    class Checked
      def initialize(rng)
        @rng = rng
      end

      # The rng's rand, checked by `uniform`; with a size, its rand(size), checked by `index`.
      def rand(size = nil)
        nil.equal?(size) ? Inputs.uniform(@rng) : Inputs.index(@rng, size)
      end
    end
    private_constant :Checked
  end
  private_constant :Inputs
end
