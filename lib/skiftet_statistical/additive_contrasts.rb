# frozen_string_literal: true

module SkiftetStatistical
  # Additive normal contrasts with crossed groupings (Hong, Kveton, Zaheer and Ghavamzadeh 2022): what differs between
  # cells is pooled on the arms' contrasts against a reference arm, never on the arms' own levels. Per arm a,
  #
  #   delta(a | cell) = delta_root(a) + u_tree(a) + sum over groupings g of u_g(a)
  #
  # The tree is nested: a node's value is its parent's plus a deviation N(0, tau^2 of its depth), "tree:1" the first
  # level under the root, whose value has a flat prior. Every other grouping is crossed with it: each key's deviation is
  # N(0, tau^2_g). A cell's contrast is read as its value plus normal noise of the sampling variance given. No term joins
  # two arms, so each arm is fitted alone.
  #
  #   SkiftetStatistical::AdditiveContrasts.fit(
  #     cells: [ { id: 1, contrasts: { a: [ 0.0, 1.0 ] }, tree: %i[root x], groups: {} },
  #              { id: 2, contrasts: { a: [ 1.0, 1.0 ] }, tree: %i[root y], groups: {} },
  #              { id: 3, contrasts: { a: [ 5.0, 1.0 ] }, tree: %i[root z], groups: {} } ],
  #     arms: %i[a])
  #   # => { variances: { "tree:1" => { a: 6.0 } },
  #   #      effects: { "tree:1" => { x: { a: [ -1.714, 2.571 ] }, y: { a: [ -0.857, 2.571 ] }, ... } },
  #   #      cells: { 1 => { a: [ 0.286, 0.905 ] }, 2 => { a: [ 1.143, 0.905 ] }, 3 => { a: [ 4.571, 0.905 ] } } }
  #
  # A tree of one level with no other grouping is plain normal shrinkage of its nodes toward the root, as above: this is
  # the library's one home of normal shrinkage.
  #
  # VARIANCES, by moments on the contrasts, per depth, grouping and arm: tau^2 = max(0, weighted spread of the group
  # estimates - their mean sampling variance). A group's estimate is the precision-weighted mean of what it holds, P its
  # precision, and with K groups around each weighted mean, the spread and the mean sampling variance are
  #
  #   spread = sum P * (estimate - mean)^2 / scale      mean sampling variance = sum (K - 1) / scale
  #   scale  = sum (sum P - sum P^2 / sum P)
  #
  # which is unbiased whatever the precisions are, and with equal ones is the sample variance less the plain mean. A
  # tree level's groups are its nodes, spread about their own parent's mean, and a node's sampling variance carries the
  # depths under it, so the tree is read from its deepest level up. Fewer than two groups show no spread: tau^2 is 0.
  # `variances:` holds what it names fixed (an earlier answer's `variances`, read by Symbol or String): each depth,
  # grouping and arm it names keeps that value, takes no moment step and stays in the answer, used here or not, and
  # whatever it does not name is fitted.
  #
  # POSTERIORS are normal-normal. The tree is solved exactly for given variances, by one pass up and one down: a node's
  # prior is its parent's posterior with the node's own contrasts taken back out, its variance widened by tau^2, so the
  # parent's uncertainty is carried down and nothing is counted twice. A key of a crossed grouping starts from
  # N(0, tau^2_g). The terms are backfitted: each crossed grouping in turn, then the tree, each on the contrasts less the
  # others' current means, `sweeps` times, the first sweep reading the tree as the plain weighted mean. The crossed
  # groupings go first because a key holds many cells while a leaf may hold one, and a leaf fitted first takes its key's
  # signal for its own. Variances move with the sweeps. Backfitting keeps no joint posterior, so a cell's variance is
  # the sum of its terms' variances, each given the others' means.
  #
  # The answer's `effects` are the deviations, per key: a node's is its value less its parent's, with no entry for the
  # root; `cells` are the whole delta(a | cell) with its variance.
  #
  # A cell leaves out an arm it has no contrast for (a nil counts as left out, and `fit` reads no arm outside `arms:`).
  # The cell adds nothing for that arm, and still gets its answer from its node and keys, which is then the prior they
  # give it. An arm no cell has a contrast for has no answer and raises ArgumentError, so it stays out of `arms:`.
  # A sampling variance is above 0: exact knowledge has no normal likelihood, so a caller hands a contrast with a nil
  # mean or a variance of 0.0 as nil.
  #
  # `.baseline` is the other half: each cell's level from all its visits, once `fit` has the effects. An arm left out of
  # `arms:` keeps its visits there, named in `effects:` at [0.0, 0.0] in every cell.
  #
  # POOLING PER (ARM, PART) IS WHAT REACHES A DECISION: theta(a | cell) = mu(cell) + delta(a | cell) for each part p.
  # `fit`'s `arms:` name the pairs, each by `pair(arm, part)`, one `.baseline` runs per part, its rows naming the same
  # pairs and the reference arm by any name no cell holds, and `.pooled_estimate` puts the two back together as a
  # SharedCredit estimate of one cell, which LinearValue and the allocator read as they read the cell's own:
  #
  #   means[a][p]         = mu_p + delta_ap                    (the reference arm's delta is 0)
  #   sd_pooled(a, p)^2   = Var(mu_p) + Var(delta_ap)          (the reference arm's is Var(mu_p) alone)
  #   cov[(a, p), (b, q)] = rho_own[(a, p), (b, q)] * sd_pooled(a, p) * sd_pooled(b, q)
  #
  # rho_own is the correlation `own` holds, so what shared credit cancels between two arms stays cancelled; an own
  # variance of 0 has correlation 0 off the diagonal and sd_pooled^2 on it, and so does an arm no row credits, which
  # keeps its nil means and its zero rows and columns. `own`'s `weight_sums` and `effective` stand as they are. The
  # answer is D * R * D, with R the correlation, so it is positive semidefinite whenever `own` is. It answers #parts,
  # and #rounding: `own`'s bound on an entry carried by the largest factor an sd moved, twice, as an entry moves by two,
  # plus the rounding of moving an entry.
  #
  # The reference arm is read as `.baseline` reads it: the one arm with credit no cell of `effects` names, under any of
  # its pairs. Any other arm with credit and no effect for the cell raises, and so do two arms no cell names.
  #
  # Arms are named by Symbols or Strings, each once, and the answers name them as given; every name inside a cell, a row
  # or `variances:` is read by Symbol or String. Every number is a finite real number, finite as a Float too (a NaN, an
  # infinity, a Complex, an Integer beyond a Float's range and a non-number are refused alike); every refusal is an
  # ArgumentError. There is no random input.
  module AdditiveContrasts
    # A depth's name in the answers is this and the depth, the first level under the root being 1.
    TREE = "tree:"
    # What a grouping may not be named: it would share a tree level's line in `variances`.
    TREE_LEVEL = /\Atree:\d+\z/
    # What joins a pair's arm and part in its name.
    PAIR_JOIN = "|"
    private_constant :TREE, :TREE_LEVEL, :PAIR_JOIN

    class << self
      # cells: [{ id:, contrasts: { arm => [estimate, variance] }, tree: [root, ..., leaf], groups: { name => key } }]
      # -> { variances: { "tree:<depth>" | name => { arm => Float } },
      #      effects: { "tree:<depth>" | name => { key => { arm => [mean, variance] } } },
      #      cells: { id => { arm => [mean, variance] } } }
      def fit(cells:, arms:, sweeps: 5, variances: nil)
        names = arm_names(arms)
        rounds = sweep_count(sweeps)
        design = Design.new(cells)
        held = held_variances(variances, design, names)
        fits = names.to_h { |arm| [ arm, arm_fit(design, arm, held).run(rounds) ] }

        { variances: variance_table(design, fits, held), effects: effect_table(design, fits),
          cells: cell_table(design, fits) }
      end

      # rows: [{ cell:, arm:, covariate:, weight:, value:, weight_squared: nil, value_squared: nil, cross: nil }], the
      # sums over the visits of one (cell, arm, covariate) with weights w > 0 and values x: weight = sum w, value =
      # sum w * x, weight_squared = sum w^2, value_squared = sum w^2 * x^2, cross = sum w^2 * x. With all three squared
      # sums nil the row is one visit; some but not all of them nil raises, and so does a weight not above 0.
      # effects: `fit`'s `:cells`; only the means are read, and a cell of `effects` with no row is left out.
      #
      # A row's effect e is its cell's, 0 for the reference arm: the one arm of the rows no cell of `effects` names.
      # Another arm without an effect for its cell raises, and so do two arms no cell names. Then, pooled over every row
      # of every cell, with S' = S - e * W: the overall mean m = sum S' / sum W, a covariate value k's mean m_k = (sum
      # over k's rows of S') / (sum over k's rows of W), and each row moves by its value's gap, o = m - m_k - e. A
      # cell's mean is its visits' weighted mean of x + o, and its variance the Hajek sandwich over those moved values
      # with the effects and gaps held fixed:
      #
      #   mean = sum (S + o * W) / W_c      variance = sum (B + 2 * d * C + d^2 * Q) / W_c^2, d = o - mean
      #
      # -> { cell => [mean, variance] }
      def baseline(rows:, effects:)
        moved_rows(sum_rows(rows), effects).group_by { |row, _offset| row.cell }.transform_values do |moved|
          weight = moved.sum(0.0) { |row, _offset| row.weight }
          mean = moved.sum(0.0) { |row, offset| row.value + (offset * row.weight) } / weight
          spread = moved.sum(0.0) { |row, offset| row.spread(offset - mean) }
          [ mean, [ spread / (weight * weight), 0.0 ].max ]
        end
      end

      # The name of (arm, part) in `arms:`, in `fit`'s answers and in a `.baseline`'s rows. A name holding the join is
      # refused: "a|b" with "c" and "a" with "b|c" would read alike.
      def pair(arm, part)
        [ arm, part ].each do |name|
          next if Read.name?(name) && !name.to_s.include?(PAIR_JOIN)

          raise ArgumentError, "a pair's names must be Symbols or Strings without #{PAIR_JOIN}, got #{name.inspect}"
        end

        "#{arm}#{PAIR_JOIN}#{part}"
      end

      # own: `cell`'s SharedCredit estimate, as `estimate` or `from_sums` answers it: one read back from JSON names no
      # parts and is refused. effects: `fit`'s `:cells` over the pairs. baselines: { part => `.baseline`'s answer over
      # that part }. Only `cell`'s baselines are read, and none when no arm has credit.
      # -> a SharedCredit::Estimate of own's arms, parts and order, with #parts and #rounding
      def pooled_estimate(own:, effects:, baselines:, cell:)
        unless own.is_a?(SharedCredit::Estimate)
          raise ArgumentError, "own must be a SharedCredit estimate, which names its parts, got #{own.class}"
        end

        parts = own.parts
        arms = own[:means].keys
        credited = arms.reject { |arm| own[:means][arm].nil? }
        reference = reference_arms(credited, effects) { |arm| parts.map { |part| pair(arm, part) } }.first
        levels = credited.empty? ? [] : parts.map { |part| level_of(baselines, part, cell) }
        terms = credited.to_h do |arm|
          [ arm, parts.each_with_index.map do |part, at|
            sum_of(levels[at], arm == reference ? [ 0.0, 0.0 ] : pooled_effect(effects, cell, pair(arm, part)))
          end ]
        end
        variances = arms.flat_map { |arm| terms.key?(arm) ? terms[arm].map(&:last) : Array.new(parts.size, 0.0) }
        moved = Rescale.new(own_covariance(own, variances.size), variances)

        SharedCredit::Estimate.new(parts, {
          means: arms.to_h { |arm| [ arm, terms[arm]&.map(&:first) ] },
          covariance: moved.covariance,
          weight_sums: own[:weight_sums].dup,
          effective: own[:effective].dup
        }, moved.rounding(own.rounding))
      end

      private

      def arm_names(arms)
        raise ArgumentError, "arms must be an Array, got #{arms.class}" unless arms.is_a?(Array)
        raise ArgumentError, "arms must name at least one" if arms.empty?

        arms.each do |arm|
          raise ArgumentError, "arms must be Symbols or Strings, got #{arm.inspect}" unless Read.name?(arm)
        end
        raise ArgumentError, "arms must each be named once" unless arms.map(&:to_s).uniq.size == arms.size

        arms
      end

      def sweep_count(sweeps)
        return sweeps if sweeps.is_a?(Integer) && sweeps >= 1

        raise ArgumentError, "sweeps must be an Integer >= 1, got #{sweeps.inspect}"
      end

      # What `variances:` names, as { name => { arm => Float } }, each name and arm spelled as this fit spells it.
      def held_variances(variances, design, arms)
        return {} if variances.nil?

        names = spelling(design.variance_names)
        spelled = spelling(arms)
        named_once(variances, "variances").each_with_object({}) do |(name, by_arm), held|
          held[names.fetch(name.to_s, name)] = named_once(by_arm, "the variances of #{name}").to_h do |arm, value|
            [ spelled.fetch(arm.to_s, arm), Read.variance(value, "the variance of #{name}, #{arm}") ]
          end
        end
      end

      def spelling(names)
        names.to_h { |name| [ name.to_s, name ] }
      end

      def named_once(hash, label)
        raise ArgumentError, "#{label} must be a Hash, got #{hash.inspect}" unless hash.is_a?(Hash)

        hash.each_key do |name|
          raise ArgumentError, "#{label} must be keyed by Symbols or Strings, got #{name.inspect}" unless Read.name?(name)
        end
        raise ArgumentError, "#{label} names one name twice" unless hash.keys.map(&:to_s).uniq.size == hash.size

        hash
      end

      def arm_fit(design, arm, held)
        estimates = Array.new(design.size, 0.0)
        precisions = Array.new(design.size, 0.0)
        design.cells.each_with_index do |cell, index|
          contrast = Read.lookup(cell[:contrasts], arm)
          estimates[index], precisions[index] = contrast_of(contrast, cell, arm) unless contrast.nil?
        end
        own = held.each_with_object({}) { |(name, by_arm), out| out[name] = by_arm[arm] if by_arm.key?(arm) }

        ArmFit.new(design, estimates, precisions, own, arm)
      end

      # A contrast as [estimate, precision].
      def contrast_of(contrast, cell, arm)
        label = "the contrast of cell #{cell[:id].inspect}, arm #{arm}"
        unless contrast.is_a?(Array) && contrast.size == 2
          raise ArgumentError, "#{label} must be [estimate, variance], got #{contrast.inspect}"
        end

        estimate = Read.number(contrast[0], "#{label}: its estimate")
        variance = Read.number(contrast[1], "#{label}: its variance")
        precision = 1.0 / variance if variance.positive?
        return [ estimate, precision ] if precision&.finite?

        raise ArgumentError, "#{label}: its variance must be above 0 with a finite inverse, got #{contrast[1].inspect}"
      end

      def variance_table(design, fits, held)
        table = design.variance_names.to_h { |name| [ name, fits.transform_values { |fit| fit.tau_squared(name) } ] }
        held.each { |name, by_arm| table[name] = by_arm.merge(table.fetch(name, {})) }
        table
      end

      def effect_table(design, fits)
        table = {}
        (1..design.depth).each do |depth|
          table[Design.level_name(depth)] = design.levels[depth].to_h do |node|
            [ design.keys[node], fits.transform_values { |fit| fit.node_effect(node) } ]
          end
        end
        design.groupings.each_with_index do |grouping, index|
          table[grouping.name] = grouping.keys.each_with_index.to_h do |key, at|
            [ key, fits.transform_values { |fit| fit.key_effect(index, at) } ]
          end
        end
        table
      end

      def cell_table(design, fits)
        design.ids.each_with_index.to_h { |id, at| [ id, fits.transform_values { |fit| fit.cell(at) } ] }
      end

      def sum_rows(rows)
        raise ArgumentError, "rows must be an Enumerable of rows, got #{rows.class}" unless rows.is_a?(Enumerable)

        rows.map { |row| SumRow.new(row) }
      end

      # [[row, offset]] for every row, in order.
      def moved_rows(rows, effects)
        reader = effect_reader(rows, effects)
        effected = rows.map { |row| [ row, reader.call(row) ] }
        overall = adjusted_mean(effected)
        means = effected.group_by { |row, _effect| row.covariate }.transform_values { |rows_of| adjusted_mean(rows_of) }

        effected.map { |row, effect| [ row, overall - means.fetch(row.covariate) - effect ] }
      end

      # sum S' / sum W over [[row, effect]].
      def adjusted_mean(effected)
        adjusted = effected.sum(0.0) { |row, effect| row.value - (effect * row.weight) }
        adjusted / effected.sum(0.0) { |row, _effect| row.weight }
      end

      # row -> e(c, a), the reference arm's being 0.0. Every arm of the rows is read here, once.
      def effect_reader(rows, effects)
        unnamed = reference_arms(rows.map { |row| row.arm.to_s }.uniq, effects) { |arm| [ arm ] }

        ->(row) { unnamed.include?(row.arm.to_s) ? 0.0 : effect_of(row, effects) }
      end

      def effect_of(row, effects)
        pair = effect_pair(effects, row.cell, row.arm)

        Read.number(pair.first, "the effect of cell #{row.cell.inspect}, arm #{row.arm}")
      end

      # Those of `arms` that no cell of `effects` names, under any of the names the block gives an arm: the reference arm.
      # Two are refused.
      def reference_arms(arms, effects)
        raise ArgumentError, "effects must be a Hash, got #{effects.class}" unless effects.is_a?(Hash)

        named = {}
        effects.each do |cell, by_arm|
          unless by_arm.is_a?(Hash)
            raise ArgumentError, "the effects of cell #{cell.inspect} must be a Hash, got #{by_arm.inspect}"
          end

          by_arm.each_key { |arm| named[arm.to_s] = true }
        end
        unnamed = arms.reject { |arm| yield(arm).any? { |name| named.key?(name) } }
        return unnamed unless unnamed.size > 1

        raise ArgumentError, "more than one arm is named by no cell: #{unnamed.join(', ')}"
      end

      def effect_pair(effects, cell, arm)
        by_arm = effects[cell]
        pair = Read.lookup(by_arm, arm) unless by_arm.nil?
        return pair if pair.is_a?(Array)

        raise ArgumentError, "cell #{cell.inspect} has no effect for arm #{arm}, got #{pair.inspect}"
      end

      def pooled_effect(effects, cell, name)
        normal(effect_pair(effects, cell, name), "the effect of cell #{cell.inspect}, #{name}")
      end

      # [mu_p, Var(mu_p)] of a cell.
      def level_of(baselines, part, cell)
        raise ArgumentError, "baselines must be a Hash, got #{baselines.class}" unless baselines.is_a?(Hash)

        by_cell = Read.lookup(baselines, part)
        raise ArgumentError, "part #{part} has no baseline, got #{by_cell.inspect}" unless by_cell.is_a?(Hash)

        normal(by_cell[cell], "the baseline of part #{part}, cell #{cell.inspect}")
      end

      # A [mean, variance] as two Floats.
      def normal(pair, label)
        unless pair.is_a?(Array) && pair.size == 2
          raise ArgumentError, "#{label} must be [mean, variance], got #{pair.inspect}"
        end

        [ Read.number(pair[0], "#{label}: its mean"), Read.variance(pair[1], "#{label}: its variance") ]
      end

      # mu + delta, whose variances add: no term joins them.
      def sum_of(level, effect)
        [ Read.number(level[0] + effect[0], "a pooled mean"), Read.number(level[1] + effect[1], "a pooled variance") ]
      end

      # own's covariance, checked `size` rows of `size` Floats.
      def own_covariance(own, size)
        rows = own[:covariance]
        unless rows.is_a?(Array) && rows.size == size && rows.all? { |row| row.is_a?(Array) && row.size == size }
          raise ArgumentError, "own's covariance must be #{size} rows of #{size}"
        end

        rows.map { |row| row.map { |value| Read.number(value, "an entry of own's covariance") } }
      end
    end

    # The checks every part of this module reads.
    module Read
      module_function

      def name?(value)
        value.is_a?(Symbol) || value.is_a?(String)
      end

      # A finite real number, finite as a Float too (an Integer or Rational beyond a Float's range is not), as a Float:
      # the library's one number check.
      def number(value, label)
        number = value.to_f if value.is_a?(Numeric) && value.real? && value.finite?
        return number if number&.finite?

        raise ArgumentError, "#{label} must be a finite number, got #{value.inspect}"
      end

      def variance(value, label)
        variance = number(value, label)
        raise ArgumentError, "#{label} must be at least 0, got #{value.inspect}" if variance.negative?

        variance
      end

      # hash[name], by Symbol or String; nil when neither is there.
      def lookup(hash, name)
        return hash[name] if hash.key?(name)

        hash[name.is_a?(Symbol) ? name.to_s : name.to_sym]
      end
    end
    private_constant :Read

    # A covariance moved to new variances, each entry keeping its correlation.
    class Rescale
      # The rounding of one entry moved, relative to it: the units of Float::EPSILON its quotient, its products and the
      # square roots under them can add.
      ENTRY_ROUNDING = 8 * Float::EPSILON

      def initialize(own, variances)
        @own = own
        @variances = variances
        @from = own.each_index.map { |index| own[index][index].positive? ? Math.sqrt(own[index][index]) : 0.0 }
        @to = variances.map { |variance| Math.sqrt(variance) }
      end

      # One triangle, mirrored, so it is exactly symmetric, the diagonal the variances themselves.
      def covariance
        size = @variances.size
        answer = Array.new(size) { Array.new(size, 0.0) }
        size.times do |row|
          answer[row][row] = @variances[row]
          ((row + 1)...size).each do |column|
            value = Read.number(correlation(row, column) * @to[row] * @to[column], "a pooled covariance")
            answer[row][column] = value
            answer[column][row] = value
          end
        end
        answer
      end

      # own's bound on an entry, in the new units. The moved matrix is S * own * S, S the sds' ratios, so entry (i, j)'s
      # error is its own times S_i * S_j, which the largest ratio twice bounds. A row whose own variance is 0 holds
      # nothing to carry.
      def rounding(own_rounding)
        stretch = @from.zip(@to).map { |from, to| from.positive? ? to / from : 0.0 }.max || 0.0
        carried = own_rounding * stretch * stretch

        Read.number(carried + (ENTRY_ROUNDING * (@variances.max || 0.0)), "the pooled rounding")
      end

      private

      # Rounding can take a correlation past 1, and it takes nothing the bound does not hold: the exact one is within 1.
      def correlation(row, column)
        scale = @from[row] * @from[column]

        scale.positive? ? (@own[row][column] / scale).clamp(-1.0, 1.0) : 0.0
      end
    end
    private_constant :Rescale

    # One row of `baseline`, read and checked: its numbers Floats, its three squared sums filled in for one visit.
    class SumRow
      REQUIRED = %i[cell arm covariate weight value].freeze
      SQUARED = %i[weight_squared value_squared cross].freeze

      attr_reader :cell, :arm, :covariate, :weight, :value

      def initialize(row)
        check_fields(row)
        @cell = row[:cell]
        @arm = row[:arm]
        @covariate = row[:covariate]
        @weight = Read.number(row[:weight], "a row's weight")
        raise ArgumentError, "a row's weight must be above 0, got #{row[:weight].inspect}" unless @weight.positive?

        @value = Read.number(row[:value], "a row's value")
        read_squared(row)
      end

      # sum of w^2 * (x + shift)^2 over the row's visits.
      def spread(shift)
        @value_squared + (2.0 * shift * @cross) + (shift * shift * @weight_squared)
      end

      private

      def check_fields(row)
        raise ArgumentError, "a row must be a Hash, got #{row.inspect}" unless row.is_a?(Hash)

        missing = REQUIRED.reject { |field| row.key?(field) }
        raise ArgumentError, "a row has no #{missing.join(', ')}: #{row.inspect}" unless missing.empty?
        raise ArgumentError, "a row names no cell: #{row.inspect}" if row[:cell].nil?
        return if Read.name?(row[:arm])

        raise ArgumentError, "a row's arm must be a Symbol or String, got #{row[:arm].inspect}"
      end

      def read_squared(row)
        given = SQUARED.map { |field| row[field] }
        if given.all?(&:nil?)
          @weight_squared = @weight * @weight
          @value_squared = @value * @value
          @cross = @weight * @value
        elsif given.any?(&:nil?)
          raise ArgumentError, "a row gives all of #{SQUARED.join(', ')} or none: #{row.inspect}"
        else
          @weight_squared, @value_squared, @cross = SQUARED.map { |field| Read.number(row[field], "a row's #{field}") }
        end
      end
    end
    private_constant :SumRow

    # A crossed grouping: its name, its keys, and each cell's key as an index into them.
    Grouping = Struct.new(:name, :keys, :of_cell)
    private_constant :Grouping

    # The shape every arm shares, read from the cells: the tree as arrays of node indexes (the root is node 0) and the
    # crossed groupings, in the order of their names. Nothing here reads a contrast.
    class Design
      attr_reader :cells, :ids, :parents, :keys, :levels, :sibling_sets, :node_of, :groupings

      def self.level_name(depth)
        "#{TREE}#{depth}"
      end

      def initialize(cells)
        raise ArgumentError, "cells must be an Enumerable of cells, got #{cells.class}" unless cells.is_a?(Enumerable)

        @cells = cells.to_a
        @cells.each { |cell| check_cell(cell) }
        @ids = @cells.map { |cell| cell[:id] }
        raise ArgumentError, "cells must each have their own id" unless @ids.uniq.size == @ids.size

        read_tree
        read_groupings
      end

      def size
        @cells.size
      end

      # The deepest level under the root.
      def depth
        [ @levels.size - 1, 0 ].max
      end

      def variance_names
        (1..depth).map { |level| self.class.level_name(level) } + @groupings.map(&:name)
      end

      private

      def check_cell(cell)
        raise ArgumentError, "a cell must be a Hash, got #{cell.inspect}" unless cell.is_a?(Hash)
        raise ArgumentError, "a cell has no id: #{cell.inspect}" if cell[:id].nil?

        label = "cell #{cell[:id].inspect}"
        unless cell[:contrasts].is_a?(Hash)
          raise ArgumentError, "the contrasts of #{label} must be a Hash, got #{cell[:contrasts].inspect}"
        end
        path = cell[:tree]
        unless path.is_a?(Array) && !path.empty? && path.none?(&:nil?)
          raise ArgumentError, "the tree of #{label} must be a path [root, ..., leaf], got #{path.inspect}"
        end
        return if cell[:groups].is_a?(Hash) && cell[:groups].all? { |name, key| Read.name?(name) && !key.nil? }

        raise ArgumentError, "the groups of #{label} must be { name => key }, got #{cell[:groups].inspect}"
      end

      # A node is known by its depth and key, so a key sits under one parent only: the answers name it by that key.
      def read_tree
        @parents = []
        @keys = []
        @levels = []
        @sibling_sets = []
        known = []
        @node_of = @cells.map do |cell|
          cell[:tree].each_with_index.inject(nil) { |parent, (key, depth)| node(known, parent, key, depth, cell) }
        end
      end

      def node(known, parent, key, depth, cell)
        level = (known[depth] ||= {})
        found = level[key]
        if depth.zero? && found.nil? && !level.empty?
          raise ArgumentError, "cells must share one root, got #{key.inspect} and #{@keys[0].inspect}"
        end
        return add_node(level, parent, key, depth) if found.nil?
        return found if @parents[found] == parent

        raise ArgumentError, "node #{key.inspect} of cell #{cell[:id].inspect} sits under two parents"
      end

      def add_node(level, parent, key, depth)
        index = @parents.size
        @parents << parent
        @keys << key
        level[key] = index
        (@levels[depth] ||= []) << index
        ((@sibling_sets[depth] ||= {})[parent] ||= []) << index
        index
      end

      def read_groupings
        names = @cells.empty? ? [] : @cells.first[:groups].keys
        labels = names.map(&:to_s).sort
        names.each do |name|
          raise ArgumentError, "a grouping may not be named #{name}: that is a tree level" if TREE_LEVEL.match?(name.to_s)
        end
        @cells.each do |cell|
          next if cell[:groups].keys.map(&:to_s).sort == labels

          raise ArgumentError, "cell #{cell[:id].inspect} must name just the groupings #{labels.join(', ')}"
        end
        raise ArgumentError, "a cell names one grouping twice" unless labels.uniq.size == labels.size

        @groupings = names.sort_by(&:to_s).map { |name| grouping(name) }
      end

      def grouping(name)
        keys = {}
        of_cell = @cells.map { |cell| keys[Read.lookup(cell[:groups], name)] ||= keys.size }

        Grouping.new(name, keys.keys, of_cell)
      end
    end
    private_constant :Design

    # One arm's fit. Its arrays run by cell, by node or by key; a cell with no contrast holds precision 0.
    class ArmFit
      def initialize(design, estimates, precisions, held, arm)
        @design = design
        @estimates = estimates
        @precisions = precisions
        @held = held
        @arm = arm
        @taus = {}
        @tree_effects = Array.new(design.parents.size)
        @key_means = design.groupings.map { |grouping| Array.new(grouping.keys.size, 0.0) }
        @key_variances = design.groupings.map { |grouping| Array.new(grouping.keys.size, 0.0) }
        @tree_means = Array.new(design.size, start) unless design.cells.empty?
      end

      def run(sweeps)
        (@design.groupings.empty? ? 1 : sweeps).times do
          @design.groupings.each_index { |index| fit_grouping(index) }
          fit_tree
        end
        return self if [ @node_means, @key_means, @taus.values ].flatten.all?(&:finite?)

        raise ArgumentError, "the contrasts of arm #{@arm} overflow a Float"
      end

      def tau_squared(name)
        @taus.fetch(name)
      end

      def node_effect(node)
        @tree_effects[node]
      end

      def key_effect(grouping, key)
        [ @key_means[grouping][key], @key_variances[grouping][key] ]
      end

      # A cell's whole delta: its node's value and its keys' deviations.
      def cell(index)
        node = @design.node_of[index]
        mean = @node_means[node]
        variance = @node_variances[node]
        @design.groupings.each_with_index do |grouping, at|
          key = grouping.of_cell[index]
          mean += @key_means[at][key]
          variance += @key_variances[at][key]
        end
        [ mean, variance ]
      end

      private

      # The first sweep's tree: the weighted mean of every contrast.
      def start
        total = @precisions.sum(0.0)
        raise ArgumentError, "no cell has a contrast for arm #{@arm}" unless total.positive?

        weighted = 0.0
        @precisions.each_with_index { |precision, cell| weighted += precision * @estimates[cell] }
        weighted / total
      end

      # Each key's posterior given everything else: prior N(0, tau^2), data its cells' contrasts less the other terms.
      def fit_grouping(index)
        grouping = @design.groupings[index]
        precision = Array.new(grouping.keys.size, 0.0)
        sum = Array.new(grouping.keys.size, 0.0)
        grouping.of_cell.each_with_index do |key, cell|
          weight = @precisions[cell]
          precision[key] += weight
          sum[key] += weight * (@estimates[cell] - @tree_means[cell] - other_keys(cell, index))
        end

        tau_squared = current(grouping.name) { moment([ precision.zip(sum) ]) }
        precision.each_index do |key|
          scale = 1.0 + (tau_squared * precision[key])
          @key_means[index][key] = tau_squared * sum[key] / scale
          @key_variances[index][key] = tau_squared / scale
        end
      end

      # The crossed groupings' current sum for a cell, without grouping `skip`.
      def other_keys(cell, skip = nil)
        total = 0.0
        @design.groupings.each_with_index do |grouping, at|
          total += @key_means[at][grouping.of_cell[cell]] unless at == skip
        end
        total
      end

      # Normal messages on a tree, in natural parameters (a precision, and that precision times the mean), so a node
      # with no contrast (0, 0) and a flat root need no case of their own. Up: what each subtree says about its node.
      # Down: a node's prior, its parent's whole posterior with the node's own message taken back out.
      def fit_tree
        up_precision, up_sum = own_terms
        sent_precision = Array.new(up_precision.size, 0.0)
        sent_sum = Array.new(up_precision.size, 0.0)

        @design.depth.downto(1) do |depth|
          tau_squared = level_tau(depth, up_precision, up_sum)
          @design.levels[depth].each do |node|
            scale = 1.0 + (tau_squared * up_precision[node])
            sent_precision[node] = up_precision[node] / scale
            sent_sum[node] = up_sum[node] / scale
            parent = @design.parents[node]
            up_precision[parent] += sent_precision[node]
            up_sum[parent] += sent_sum[node]
          end
        end

        spread_down(up_precision, up_sum, sent_precision, sent_sum)
        @tree_means = @design.node_of.map { |node| @node_means[node] }
      end

      # What each node's own cells say, less the crossed groupings.
      def own_terms
        precision = Array.new(@design.parents.size, 0.0)
        sum = Array.new(@design.parents.size, 0.0)
        @design.node_of.each_with_index do |node, cell|
          weight = @precisions[cell]
          precision[node] += weight
          sum[node] += weight * (@estimates[cell] - other_keys(cell))
        end
        [ precision, sum ]
      end

      def level_tau(depth, precision, sum)
        current(Design.level_name(depth)) do
          moment(@design.sibling_sets[depth].values.map { |nodes| nodes.map { |node| [ precision[node], sum[node] ] } })
        end
      end

      def spread_down(up_precision, up_sum, sent_precision, sent_sum)
        total_precision = up_precision.dup
        total_sum = up_sum.dup

        (1..@design.depth).each do |depth|
          tau_squared = @taus.fetch(Design.level_name(depth))
          @design.levels[depth].each do |node|
            parent = @design.parents[node]
            rest_precision = total_precision[parent] - sent_precision[node]
            rest_sum = total_sum[parent] - sent_sum[node]
            scale = 1.0 + (tau_squared * rest_precision)
            total_precision[node] = up_precision[node] + (rest_precision / scale)
            total_sum[node] = up_sum[node] + (rest_sum / scale)
            @tree_effects[node] = deviation(tau_squared, up_precision[node], up_sum[node], rest_precision, rest_sum)
          end
        end

        @node_means = total_sum.each_with_index.map { |sum, node| sum / total_precision[node] }
        @node_variances = total_precision.map { |precision| 1.0 / precision }
      end

      # A node's deviation u ~ N(0, tau^2) from its parent, given the two independent normals that bracket it: its
      # subtree's (precision a) and the rest of the tree's (precision b). With T = 1 / a + 1 / b + tau^2,
      # mean = tau^2 * (subtree mean - rest mean) / T and variance = tau^2 * (1 / a + 1 / b) / T, multiplied through by
      # a * b, so a flat side (precision 0) needs no case of its own: it leaves the prior.
      def deviation(tau_squared, a, a_sum, b, b_sum)
        total = a + b + (tau_squared * a * b)
        [ tau_squared * ((a_sum * b) - (b_sum * a)) / total, tau_squared * (a + b) / total ]
      end

      # The variance `variances:` holds, or the block's moment: from here on the current one.
      def current(name)
        @taus[name] = @held.fetch(name) { yield }
      end

      # tau^2 by moments over sets of groups, each set around its own weighted mean, a group being [precision,
      # precision * estimate]. A group with no contrast is left out.
      def moment(sets)
        spread = 0.0
        degrees = 0
        scale = 0.0
        sets.each do |set|
          groups = set.select { |precision, _sum| precision.positive? }
          next if groups.size < 2

          total = groups.sum(0.0) { |precision, _sum| precision }
          mean = groups.sum(0.0) { |_precision, sum| sum } / total
          spread += groups.sum(0.0) { |precision, sum| precision * (((sum / precision) - mean)**2) }
          degrees += groups.size - 1
          scale += total - (groups.sum(0.0) { |precision, _sum| precision * precision } / total)
        end
        scale.positive? ? [ (spread - degrees) / scale, 0.0 ].max : 0.0
      end
    end
    private_constant :ArmFit
  end
end
