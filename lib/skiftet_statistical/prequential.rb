# frozen_string_literal: true

require_relative "significance"

module SkiftetStatistical
  # The prequential log score (Dawid 1984): how many bits an outcome cost a predictor that gave it probability p,
  # -log2 p, scored on outcomes it had not seen. Several depths of one predictor, each deeper than the one before, are
  # compared on the same events, and the gain of a depth is the bits per event it saves over the depth before it.
  #
  #   rows = [ { cluster: 1, log_probs: { shallow: Math.log(0.25), deep: Math.log(0.5) } },
  #            { cluster: 1, log_probs: { shallow: Math.log(0.5),  deep: Math.log(0.5) } },
  #            { cluster: 2, log_probs: { shallow: Math.log(0.5),  deep: Math.log(0.25) } },
  #            { cluster: 3, log_probs: { shallow: Math.log(0.125), deep: Math.log(0.5) } } ]
  #   SkiftetStatistical::Prequential.score(rows, depth: :deep)  # => 1.25
  #   SkiftetStatistical::Prequential.gains(rows, depths: %i[shallow deep])
  #   # => { deep: { bits: 0.5, se: 0.6495, events: 4, clusters: 3 } }
  #
  # Events of one cluster move together, so the standard error comes from the spread between clusters and not between
  # events (Liang and Zeger 1986). With s_g and n_g a cluster's total and count, and G clusters,
  #
  #   mean = sum s_g / sum n_g
  #   se   = sqrt(G / (G - 1) * sum (s_g - mean * n_g)^2) / sum n_g
  #
  # the residuals taken about the pooled mean, so a cluster weighs by its events. `clustered_mean` answers that mean with
  # its normal interval for a caller that stores per-cluster totals and no events; `gains` adds its rows up to the same
  # totals and goes through the same arithmetic, so the two cannot differ. Under 2 clusters there is no spread to read:
  # the mean stands, its standard error and bounds are nil. With no event at all the mean is nil too.
  #
  # A log probability is a number at most 0. A probability of 0 has no finite log: such an outcome costs a predictor
  # infinitely many bits, so the caller smooths it first and -Infinity is refused like any other non-finite number. A
  # value above 0 is refused too (it is no log of a probability, and it is what a flipped sign looks like). Depths are
  # named by Symbols or Strings, each once, and answered as given; a row's fields and depths are read by either spelling
  # and each must be there once, a depth nobody asked for is not read. A cluster is any Hash key but nil. Every
  # number is a finite real number, finite as a Float too (a NaN, an infinity, a Complex, an Integer beyond a Float's
  # range and a non-number are refused alike), a count is above 0, and a total a Float cannot hold is refused rather
  # than answered as NaN or infinity. Every refusal is an ArgumentError.
  module Prequential
    LOG2 = Math.log(2.0)

    class << self
      # The mean of -log2 p at one depth over `rows`, any Enumerable of `{ log_probs: { depth => Float } }`, read once;
      # nil with no rows.
      def score(rows, depth:)
        names = read_names([ depth ])
        # The rows are read once into their log probabilities and counted by that list's size, never by a counter
        # inside sum's block: Ruby 3.4.12's Array#sum may call its block more than once for an element.
        logs = read_rows(rows).map { |row| read_log_probs(row, names).first }.to_a
        return nil if logs.empty?

        finite((0.0 - (logs.sum(0.0) / logs.size)) / LOG2)
      end

      # Per depth after the first, its gain over the depth before it in the order given: `{ depth => { bits:, se:,
      # events:, clusters: } }`, bits the mean gain per event and se its cluster-robust standard error. `rows` is any
      # Enumerable of `{ cluster:, log_probs: { depth => Float } }`, read once.
      def gains(rows, depths:)
        names = read_names(depths)
        after = names.drop(1)
        counts = Hash.new(0)
        totals = after.map { Hash.new(0.0) }

        read_rows(rows).each do |row|
          logs = read_log_probs(row, names)
          cluster = read_cluster(row)
          counts[cluster] += 1
          totals.each_with_index { |total, i| total[cluster] += (logs[i + 1] - logs[i]) / LOG2 }
        end

        after.zip(totals).to_h do |depth, total|
          pooled = moments(total, counts)
          [ depth, { bits: pooled[:mean], se: pooled[:se], events: pooled[:n], clusters: pooled[:clusters] } ]
        end
      end

      # The mean of per-cluster totals `sums` over per-cluster `counts`, on one key set, with its cluster-robust normal
      # interval at `level`: `{ mean:, lower:, upper:, n:, clusters: }`, n the summed counts as given.
      def clustered_mean(sums:, counts:, level: 0.95)
        level = read_level(level)
        pooled = moments(*read_clusters(sums, counts))
        lower, upper = bounds(pooled[:mean], pooled[:se], level)
        { mean: pooled[:mean], lower: lower, upper: upper, n: pooled[:n], clusters: pooled[:clusters] }
      end

      private

      # The one arithmetic: `totals` and `counts` hold the same keys, the totals Floats. A cluster's total and count
      # meet by key, never by position.
      def moments(totals, counts)
        size = totals.size
        n = counts.values.sum
        return { mean: nil, se: nil, n: n, clusters: size } if size.zero?

        events = number(n, "the summed counts")
        mean = finite(totals.values.sum / events)
        { mean: mean, se: standard_error(totals, counts, mean, events), n: n, clusters: size }
      end

      def standard_error(totals, counts, mean, events)
        size = totals.size
        return nil if size < 2

        squares = totals.sum do |cluster, total|
          residual = total - (mean * counts.fetch(cluster).to_f)
          residual * residual
        end
        finite(Math.sqrt(size.to_f / (size - 1) * squares) / events)
      end

      # The lower tail's quantile, so that a level just under 1 still reads a finite z: its upper tail rounds to 1.
      def bounds(mean, se, level)
        return [ nil, nil ] if se.nil?

        half_width = finite(-Significance.normal_quantile((1.0 - level) / 2.0) * se)
        [ finite(mean - half_width), finite(mean + half_width) ]
      end

      def read_level(value)
        level = number(value, "level")
        raise ArgumentError, "level must be strictly between 0 and 1, got #{value.inspect}" unless level > 0.0 && level < 1.0

        level
      end

      # [totals as Floats, counts as given], checked.
      def read_clusters(sums, counts)
        raise ArgumentError, "sums must be a Hash of cluster => total, got #{sums.class}" unless sums.is_a?(Hash)
        raise ArgumentError, "counts must be a Hash of cluster => count, got #{counts.class}" unless counts.is_a?(Hash)
        unless sums.size == counts.size && sums.each_key.all? { |cluster| counts.key?(cluster) }
          raise ArgumentError, "sums and counts must name the same clusters"
        end

        sums.each_key { |cluster| cluster_key(cluster) }
        counts.each do |cluster, count|
          label = "the count of cluster #{cluster.inspect}"
          raise ArgumentError, "#{label} must be above 0, got #{count.inspect}" unless number(count, label).positive?
        end
        [ sums.to_h { |cluster, total| [ cluster, number(total, "the total of cluster #{cluster.inspect}") ] }, counts ]
      end

      def read_rows(rows)
        return rows if rows.is_a?(Enumerable)

        raise ArgumentError, "rows must be an Enumerable of rows, got #{rows.class}"
      end

      # The names as given, checked: an Array of one or more Symbols or Strings, each named once.
      def read_names(list)
        raise ArgumentError, "depths must be an Array of names, got #{list.class}" unless list.is_a?(Array)
        raise ArgumentError, "depths must name at least one" if list.empty?

        list.each do |name|
          raise ArgumentError, "a depth must be a Symbol or a String, got #{name.inspect}" unless name.is_a?(Symbol) || name.is_a?(String)
        end
        raise ArgumentError, "depths must each be named once, got #{list.inspect}" unless list.map(&:to_s).uniq.size == list.size

        list
      end

      # One rule for both doors: a cluster is any Hash key but nil, which reads as a key nobody filled in.
      def cluster_key(cluster)
        raise ArgumentError, "a cluster must not be nil" if cluster.nil?

        cluster
      end

      def read_cluster(row)
        cluster_key(entry(row, :cluster, "a row"))
      end

      # The row's log probabilities at `names`, as Floats, in that order.
      def read_log_probs(row, names)
        logs = entry(row, :log_probs, "a row")
        names.map do |name|
          log_prob = number(entry(logs, name, "a row's log_probs"), "the log probability at #{name.inspect}")
          raise ArgumentError, "the log probability at #{name.inspect} must be at most 0, got #{log_prob}" if log_prob.positive?

          log_prob
        end
      end

      # The one entry of `hash` that `name` spells, by Symbol or String.
      def entry(hash, name, label)
        raise ArgumentError, "#{label} must be a Hash, got #{hash.class}" unless hash.is_a?(Hash)

        found = [ name.to_sym, name.to_s ].select { |key| hash.key?(key) }
        raise ArgumentError, "#{label} must hold #{name.inspect} once" unless found.size == 1

        hash[found.first]
      end

      # A finite real number, finite as a Float too (an Integer or Rational beyond a Float's range is not), as a Float.
      def number(value, label)
        number = value.to_f if value.is_a?(Numeric) && value.real? && value.finite?
        return number if number&.finite?

        raise ArgumentError, "#{label} must be a finite number, got #{value.inspect}"
      end

      def finite(value)
        return value if value.finite?

        raise ArgumentError, "the totals are beyond a Float's range"
      end
    end
  end
end
