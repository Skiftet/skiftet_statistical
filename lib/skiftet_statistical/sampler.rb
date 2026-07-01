# frozen_string_literal: true

module SkiftetStatistical
  # Random sampling used by the stochastic policies (Thompson Sampling, Softmax,
  # Epsilon-Greedy). An injectable RNG (a `Random`) makes every policy fully
  # deterministic under test — pass `rng: Random.new(seed)`.
  class Sampler
    attr_reader :rng

    def initialize(rng = Random.new)
      @rng = rng
    end

    # Standard normal deviate via Box–Muller.
    def gaussian
      u1 = rand_open
      u2 = @rng.rand
      Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
    end

    # Gamma(shape, scale = 1) via Marsaglia–Tsang. Shapes < 1 are handled by the
    # standard boosting identity: Gamma(k) = Gamma(k + 1) * U**(1/k).
    def gamma(shape)
      raise ArgumentError, "shape must be > 0" unless shape.positive?

      return gamma(shape + 1.0) * (rand_open**(1.0 / shape)) if shape < 1.0

      d = shape - (1.0 / 3.0)
      c = 1.0 / Math.sqrt(9.0 * d)
      loop do
        x = gaussian
        v = (1.0 + (c * x))**3
        next if v <= 0.0

        u = @rng.rand
        return d * v if u < 1.0 - (0.0331 * (x**4))
        return d * v if Math.log(u) < (0.5 * x * x) + (d * (1.0 - v + Math.log(v)))
      end
    end

    # Beta(alpha, beta) drawn as G1 / (G1 + G2) with Gi ~ Gamma(., 1).
    def beta(alpha, beta)
      g1 = gamma(alpha)
      g2 = gamma(beta)
      total = g1 + g2
      total.zero? ? 0.5 : g1 / total
    end

    private

    # Uniform on (0, 1] — keeps log(u) finite in Box–Muller / boosting.
    def rand_open
      u = @rng.rand
      u.zero? ? Float::MIN : u
    end
  end
end
