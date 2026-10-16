# frozen_string_literal: true

module SkiftetStatistical
  class Bandit
    # One arm's Beta posterior as the allocator judges it, from a tally of exposures and conversions. Pure: no
    # randomness, no state.
    #
    #   count = SkiftetStatistical::Bandit::WeightedPosterior.count(exposures: 400, conversions: 40)
    #   SkiftetStatistical::Bandit::WeightedPosterior.beta(count) # => [41.0, 361.0]
    #
    # A tally MAY carry `weighted_exposures` and `weighted_conversions`, the sums of 1 / the arm's share at each
    # draw (inverse-propensity counts), and `weighted_exposures_squared`, the sum of their squares. The weighted
    # conversions are then rescaled to an exposure count n before the posterior is built, so an arm drawn mostly
    # under a small share holds no more evidence than it has. n is the Kish effective sample size
    # (sum w)^2 / sum w^2 when the squared sum is present, never more than the raw count; else the raw exposure
    # count. Without the weighted sums the raw counts are judged.
    module WeightedPosterior
      module_function

      # The numbers of a tally (Symbol or String keys; a nil tally is nobody seen).
      def count(tally)
        tally ||= {}
        weighted_exposures = fetch_key(tally, :weighted_exposures)&.to_f
        weighted_conversions = fetch_key(tally, :weighted_conversions)&.to_f
        weighted_squared = fetch_key(tally, :weighted_exposures_squared)&.to_f
        weighted = !weighted_exposures.nil? && !weighted_conversions.nil? && weighted_exposures.positive?

        {
          exposures: Integer(fetch_key(tally, :exposures) || 0),
          conversions: Integer(fetch_key(tally, :conversions) || 0),
          weighted_exposures: weighted ? weighted_exposures : nil,
          weighted_conversions: weighted ? weighted_conversions : nil,
          weighted_exposures_squared: weighted && weighted_squared&.positive? ? weighted_squared : nil
        }
      end

      # [alpha, beta] = [prior_alpha + c, prior_beta + n - c], c clamped to [0, n] (n from #effective_exposures,
      # c the weighted conversions rescaled to n when the count is weighted).
      def beta(count, prior_alpha: 1.0, prior_beta: 1.0)
        exposures = effective_exposures(count)
        conversions =
          if count[:weighted_exposures]
            (count[:weighted_conversions] * exposures / count[:weighted_exposures]).clamp(0.0, exposures.to_f)
          else
            count[:conversions].to_f.clamp(0.0, exposures.to_f)
          end

        [ prior_alpha + conversions, prior_beta + exposures - conversions ]
      end

      # The exposure count the posterior is built on: the Kish n when the squared sum is present (never above the
      # raw count), else the raw count.
      def effective_exposures(count)
        return count[:exposures] unless count[:weighted_exposures_squared]

        kish = (count[:weighted_exposures]**2) / count[:weighted_exposures_squared]
        [ kish, count[:exposures].to_f ].min
      end

      def fetch_key(hash, key)
        hash.key?(key) ? hash[key] : hash[key.to_s]
      end
    end
  end
end
