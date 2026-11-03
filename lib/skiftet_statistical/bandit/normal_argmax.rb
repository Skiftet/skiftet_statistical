# frozen_string_literal: true

require_relative "../numbers"
require_relative "../inputs"
require_relative "../gauss_legendre"
require_relative "../significance"
require_relative "../beta_argmax"
require_relative "checks"

module SkiftetStatistical
  class Bandit
    # THE map between two Normals' standardised variables, written once (NormalArgmax's own integrals and panel breaks,
    # and the order odds' points and distribution functions, probabilities.rb): the point at the standard deviate z of
    # a Normal (mean, sd), in the standard units of another (onto_mean, onto_sd), ((mean - onto_mean) + sd * z) /
    # onto_sd, the means' difference formed first, so two near means far from 0 lose no digits to their size, and a
    # finite numerator over a standard deviation reads a Float, or an infinity where the quotient leaves a Float's
    # range, never a NaN. Private to the bandit.
    module Standardised
      module_function

      def carried(mean, deviation, z, onto_mean, onto_deviation)
        ((mean - onto_mean) + (deviation * z)) / onto_deviation
      end
    end
    private_constant :Standardised

    # Which of several independent Normal variables is the largest, by BetaArgmax's rule: the probability that each
    # member is the largest, and the expected value of the largest.
    #
    #   P(member i is the largest) = integral of phi(z) * product (j != i) of Phi((mu_i + sigma_i z - mu_j) / sigma_j)
    #   E[the largest]             = U - integral from L to U of the product of every F_j(x) dx
    #
    # (phi and Phi the standard Normal's density and distribution function, F_j member j's distribution function).
    #
    # Each member's probability is its own integral, in its own standardised variable z = (x - mu_i) / sigma_i, so its
    # density is sampled on its own scale however narrow, wide or far from the others it is. The integral runs over z
    # from the standard quantile at BetaArgmax::TAIL to the one at 1 - TAIL, composite Gauss-Legendre at
    # BetaArgmax::NODES nodes a panel, the panels broken at the standard quantiles at BetaArgmax::BREAKS and, inside
    # that range, at the same quantiles of every narrower member, carried into z. On each panel every factor is then
    # smooth: a narrower member's distribution function lies between two of its own quantiles, and a member at least as
    # wide moves by at most the panel's width in its own standard units. Another member's argument is formed as
    # ((mu_i - mu_j) + sigma_i z) / sigma_j, a finite numerator over a standard deviation (the bandit's one map between
    # two Normals' standardised variables, Standardised.carried), so it reads 0 or 1 where the quotient leaves a
    # Float, never a NaN.
    #
    # The error bound. A member's integral leaves out at most TAIL at each end of its own range, and a factor beyond its
    # member's extreme quantiles moves by at most TAIL, which a rule of positive weights reads to within 2 TAIL of the
    # panels' mass; the rest is the Gauss-Legendre error on smooth panels, below SMOOTH_ERROR (1e-12). So each member's
    # integral lies within 2 k TAIL + 1e-12 of its exact probability (k the number of members), and their sum within k
    # times that of 1: a sum further from 1 is refused, as beyond what the rule can integrate. The answers are the
    # integrals over their sum, each within 2 k (k + 1) TAIL + 2e-12 of the exact probability (at most 8.4e-8 for 20
    # members, 1.2e-9 for two), then mixed with epsilon / k, as BetaArgmax's are. A standard deviation below Float::MIN,
    # where a Float holds fewer digits and sigma_i z rounds coarsely, is beyond what the rule can integrate, for both
    # answers alike.
    #
    # E[the largest] reads no density: [L, U] is the members' joint range, from the lowest of their quantiles at TAIL to
    # the highest at 1 - TAIL, the panels broken at every member's quantiles at BREAKS, on x less the largest mean (so a
    # mean near 10**6 loses no digits to its own size). Its integrand is a product of distribution functions, at most 1,
    # and a narrow member's step sits on its own quantile points however close they round, so no member's mass goes
    # unsampled. Leaving out what lies beyond the joint range costs at most 1.4e-11 standard deviations a member at each
    # end (below L the product is at most the lowest member's F, and above U one minus it is at most the sum of the
    # members' upper tails); the rest is a rounding of the members' means relative to the largest.
    #
    #   SkiftetStatistical::Bandit::NormalArgmax.probabilities([ [ 0.0, 1.0 ], [ 0.0, 1.0 ] ])     # => [0.5, 0.5]
    #   SkiftetStatistical::Bandit::NormalArgmax.expected_maximum([ [ 0.0, 1.0 ], [ 0.0, 1.0 ] ])  # => 1 / sqrt(pi)
    #
    # Members are given as [[mean, standard_deviation], ...], 1 to BetaArgmax::MAX_MEMBERS of them, each pair read by
    # the library's one reading of a pair (Numbers.pair, its numbers in the input domain, Numbers::LARGEST), the
    # standard deviation above 0. Every answer is in the members' order and passes the bandit's one check of an answer
    # (Checks.finite_answer).
    module NormalArgmax
      ROOT_TWO_PI = Math.sqrt(2.0 * Math::PI)
      # The names of a member's two numbers.
      PAIR = [ "mean", "standard deviation" ].freeze
      # The standard Normal's quantiles at BetaArgmax::BREAKS: a member's own panel breaks in its own variable z, the
      # first and the last its range; on the joint range of E[the largest], every member's breaks.
      SCORES = BetaArgmax::BREAKS.map { |level| Significance.normal_quantile(level) }.freeze
      # What the Gauss-Legendre rule may add to a member's integral on smooth panels, beyond the 2 k TAIL above.
      SMOOTH_ERROR = 1e-12
      private_constant :PAIR, :SCORES, :SMOOTH_ERROR

      class << self
        # P(member i is the largest), mixed with epsilon / k, for every member in order; [1.0] for one member.
        # epsilon lies in [0, 1], read by the bandit's one [0, 1] bound (Checks.unit). The answers lie in [0, 1] and sum
        # to 1.
        def probabilities(params, epsilon: 0.0)
          members = members(params)
          epsilon = Checks.unit(epsilon, "epsilon")
          return [ 1.0 ] if members.size == 1

          integrals, total = integrable(members, params)
          k = members.size
          Checks.finite_answer(integrals.map { |integral| (epsilon / k) + ((1.0 - epsilon) * integral / total) },
                               "the members' parameters")
        end

        # E[the largest member]; one member's mean exactly. Members `probabilities` cannot integrate are refused alike
        # (the one guard, `integrable`), and an answer beyond a Float's range by the bandit's one check of an answer
        # (Checks.finite_answer).
        def expected_maximum(params)
          members = members(params)
          return Checks.finite_answer([ members.first.first ], "the members' parameters").first if members.size == 1

          integrable(members, params)
          origin = members.map(&:first).max
          normals = members.map { |mean, deviation| [ mean - origin, deviation ] }
          points = joint_points(normals)
          integral = 0.0
          panels(points) do |x, weight|
            product = 1.0
            normals.each { |mean, deviation| product *= distribution((x - mean) / deviation) }
            integral += weight * product
          end
          Checks.finite_answer([ origin + (points.last - integral) ], "the members' parameters").first
        end

        private

        # THE guard of both answers, written once: [each member's own integral, their total] when every standard
        # deviation is at least Float::MIN and the integrals sum to 1 within k (2 k TAIL + SMOOTH_ERROR); else an
        # ArgumentError.
        def integrable(members, params)
          if members.all? { |_mean, deviation| deviation >= Float::MIN }
            integrals = members.each_index.map { |member| own_integral(members, member) }
            total = integrals.sum
            k = members.size
            return [ integrals, total ] if (total - 1.0).abs <= k * ((2 * k * BetaArgmax::TAIL) + SMOOTH_ERROR)
          end

          raise ArgumentError, "the members' parameters are beyond what the rule can integrate, got #{Inputs.shown(params)}"
        end

        # The member's own integral of phi(z) * the product of the others' Phi(((mu_i - mu_j) + sigma_i z) / sigma_j)
        # (z carried into each other's standard units by Standardised.carried) over its own range in z; at a node where
        # the product reaches 0, the rest of it is not read.
        def own_integral(members, member)
          mean, deviation = members[member]
          others = members.each_with_index.filter_map { |pair, other| pair unless other == member }
          sum = 0.0
          panels(own_points(members, member)) do |z, weight|
            product = Math.exp(-0.5 * z * z) / ROOT_TWO_PI
            others.each do |other_mean, other_deviation|
              product *= distribution(Standardised.carried(mean, deviation, z, other_mean, other_deviation))
              break if product.zero?
            end
            sum += weight * product
          end
          sum
        end

        # The member's own breaks in z (SCORES), and every narrower member's quantiles at BREAKS carried into z,
        # ((mu_j - mu_i) + sigma_j * score) / sigma_i (Standardised.carried), where they lie inside its range; sorted,
        # each once.
        def own_points(members, member)
          mean, deviation = members[member]
          low = SCORES.first
          high = SCORES.last
          points = SCORES.dup
          members.each_with_index do |(other_mean, other_deviation), other|
            next if other == member || other_deviation >= deviation

            SCORES.each do |score|
              z = Standardised.carried(other_mean, other_deviation, score, mean, deviation)
              points << z if z > low && z < high
            end
          end
          points.sort.uniq
        end

        # Every member's quantiles at BetaArgmax::BREAKS, sorted, each once: the first is L and the last U.
        def joint_points(normals)
          normals.flat_map { |mean, deviation| SCORES.map { |score| mean + (deviation * score) } }.sort.uniq
        end

        # Yields each node of the composite rule and its weight, the panel's half width folded in.
        def panels(points)
          rule = GaussLegendre.nodes(BetaArgmax::NODES)
          points.each_cons(2) do |left, right|
            half_width = (right - left) / 2.0
            middle = (left + right) / 2.0
            rule.each { |node, weight| yield middle + (half_width * node), weight * half_width }
          end
        end

        def distribution(score)
          Significance.normal_cdf(score)
        end

        # The members, read by Array's own to_a (Inputs.elements) and each pair by Numbers.pair.
        def members(params)
          given = Array === params ? Inputs.elements(params) : []
          if given.empty?
            raise ArgumentError, "params must be a non-empty Array of [mean, standard_deviation] pairs, got " \
                                 "#{Inputs.shown(params)}"
          end
          if given.size > BetaArgmax::MAX_MEMBERS
            raise ArgumentError, "params may hold at most #{BetaArgmax::MAX_MEMBERS} members, got #{given.size}"
          end

          given.each_with_index.map do |pair, i|
            Numbers.pair(pair, "member #{i}", PAIR) do |number, name, index|
              index == 1 ? Checks.positive(number, name) : number
            end
          end
        end
      end
    end
  end
end
