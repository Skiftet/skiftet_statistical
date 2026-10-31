# frozen_string_literal: true

module SkiftetStatistical
  # The Cholesky factor of a covariance matrix: the lower-triangular L with L * L^T = matrix, for drawing correlated
  # normals as means + L * n from independent standard normals n. Pure Ruby, no matrix library.
  #
  #   SkiftetStatistical::Cholesky.factor([ [ 4.0, 2.0 ], [ 2.0, 5.0 ] ])
  #   # => [[2.0, 0.0], [1.0, 2.0]]
  #
  # Rows in, rows out: n rows of n numbers in, n rows of n Floats out, 0.0 above the diagonal; the input is never
  # changed. The matrix must be square, of finite real numbers (finite as Floats too) and symmetric: entries (i, j) and
  # (j, i) may differ by at most 1e-12 times the larger of their sizes and sqrt(|a_ii| * |a_jj|), the scale of a
  # covariance's entry, so the rounding of a computed zero passes and a real asymmetry does not. The lower triangle is
  # read.
  #
  # Column by column, the pivot of column j is a_jj - sum over k < j of L_jk^2, and the residual of a row i > j under
  # it is r_ij = a_ij - sum over k < j of L_ik * L_jk:
  # - a pivot below -1e-9 times the trace raises ArgumentError: the matrix is not positive semidefinite. A trace not
  #   above 0 gives no rounding at all, so a pivot of exactly 0 passes and any pivot below 0 raises;
  # - a pivot below 1e-12 times the largest diagonal entry, or not above 0, is set to 0 with its whole column, so a
  #   part with no variance (or none left once the others are known) is drawn at its mean and still takes its normal.
  #   Every residual under it must be within what that pivot holds plus rounding: |r_ij| at most
  #   sqrt(pivot * q_ij) + 1e-9 times the trace, where q_ij = a_ii - sum over k < j of L_ik^2 is the pivot row i has
  #   at column j (a pivot or a q_ij not above 0 counts as 0). The square root is the bound every positive
  #   semidefinite matrix keeps; the rounding is the one a pivot is given. A larger residual raises ArgumentError: the
  #   matrix is not positive semidefinite (a part with no variance cannot covary), and setting the column to 0 would
  #   drop the residual from L * L^T;
  # - otherwise L_jj = sqrt(pivot) and L_ij = r_ij / L_jj below it.
  #
  # A matrix of lower rank is the one `factor` can refuse although it is positive semidefinite: in the given order a
  # pivot of what is left may be rounding, kept because it is above 1e-12 of the largest diagonal entry, and dividing by
  # it carries that rounding into every later pivot. `root` is the same arithmetic taken largest pivot first, so nothing
  # is divided by rounding while a larger pivot is left:
  #
  #   SkiftetStatistical::Cholesky.root([ [ 1.0, 2.0 ], [ 2.0, 4.0 ] ])
  #   # => [[1.0, 0.0], [2.0, 0.0]]
  #
  # It answers a square root S, n rows of n Floats with S * S^T = matrix within the tolerance below, each row the row of
  # the same index given. S is NOT triangular in the given order: column k holds the k-th pivot taken, and the columns
  # after the last pivot kept are 0.0. The matrix is read as `factor` reads it. `rounding`, a finite number of at least
  # 0, is how far the caller says each entry given may be off. The tolerance of residual (i, j) is the larger of 1e-9
  # times the trace (a trace not above 0 gives none) and rounding * reach_i * reach_j. A residual row is the row given
  # less multiples of the pivots' rows given, and reach_i is 1 plus the sizes of those multiples, so that product is
  # what `rounding` on every entry can move the residual by; before any pivot it is `rounding` itself. A pivot kept is
  # above its own tolerance, so a large multiple raises the tolerance less than it lowers a pivot it did not earn: an
  # indefinite matrix cannot pass on its own multiples. At each step the largest residual pivot left is taken:
  # - a residual pivot below minus its tolerance raises ArgumentError: the matrix is not positive semidefinite;
  # - one within its tolerance reads as zero and ends its column, which stays 0.0. Every residual beside it must be
  #   within what that pivot holds plus their tolerance, or it raises ArgumentError as `factor` does: the column would
  #   drop the residual from S * S^T;
  # - otherwise S takes the pivot's column and the residuals give up its share.
  # An accepted matrix is rebuilt within about twice the tolerance, the largest residual dropped.
  module Cholesky
    # A pivot below this times the largest diagonal entry is no variance: set to 0 with its column.
    ZERO_PIVOT = 1e-12
    # A pivot below minus this times the trace is no rounding: the matrix is not positive semidefinite. A residual
    # under a pivot set to 0 is given the same rounding beyond what that pivot holds. A trace not above 0 gives none.
    NEGATIVE_PIVOT = 1e-9
    # How far apart entries (i, j) and (j, i) may be, relative to their scale.
    SYMMETRY = 1e-12

    class << self
      def factor(rows)
        matrix = read(rows)
        size = matrix.size
        diagonal = Array.new(size) { |i| matrix[i][i] }
        zero_below = ZERO_PIVOT * (diagonal.max || 0.0)
        rounding = NEGATIVE_PIVOT * [ diagonal.sum(0.0), 0.0 ].max
        lower = Array.new(size) { Array.new(size, 0.0) }

        size.times do |j|
          pivot = matrix[j][j] - (0...j).sum(0.0) { |k| lower[j][k] * lower[j][k] }
          if pivot < -rounding
            raise ArgumentError, "the matrix is not positive semidefinite (pivot #{pivot} in column #{j})"
          end

          if pivot < zero_below || pivot <= 0.0
            check_held(matrix, lower, j, pivot, rounding)
            next
          end

          root = Math.sqrt(pivot)
          lower[j][j] = root
          ((j + 1)...size).each do |i|
            lower[i][j] = (matrix[i][j] - (0...j).sum(0.0) { |k| lower[i][k] * lower[j][k] }) / root
          end
        end

        lower
      end

      # The square root S of the matrix, largest pivot first (see the module's notes).
      def root(rows, rounding: 0.0)
        margin = read_rounding(rounding)
        matrix = read(rows)
        floor = [ (0...matrix.size).sum(0.0) { |i| NEGATIVE_PIVOT * matrix[i][i] }, 0.0 ].max
        Elimination.new(matrix, floor, margin).square_root
      end

      private

      def read_rounding(value)
        number = value.to_f if value.is_a?(Numeric) && value.real? && value.finite?
        return number if number&.finite? && !number.negative?

        raise ArgumentError, "rounding must be a finite number of at least 0, got #{value.inspect}"
      end

      # Column j's pivot is set to 0 with its column: every residual under it must be within what the pivot holds,
      # sqrt(pivot * q_ij), plus rounding, or it would be dropped from L * L^T.
      def check_held(matrix, lower, j, pivot, rounding)
        held = pivot.positive? ? pivot : 0.0
        ((j + 1)...matrix.size).each do |i|
          residual = matrix[i][j] - (0...j).sum(0.0) { |k| lower[i][k] * lower[j][k] }
          own = matrix[i][i] - (0...j).sum(0.0) { |k| lower[i][k] * lower[i][k] }
          next if residual.abs <= Math.sqrt(held * (own.positive? ? own : 0.0)) + rounding

          raise ArgumentError, "the matrix is not positive semidefinite (residual #{residual} in row #{i} under " \
                               "the zero pivot of column #{j})"
        end
      end

      # The rows as Floats, checked square, finite and symmetric.
      def read(rows)
        raise ArgumentError, "the matrix must be an Array of rows, got #{rows.class}" unless rows.is_a?(Array)

        size = rows.size
        matrix = rows.each_with_index.map do |row, i|
          unless row.is_a?(Array) && row.size == size
            raise ArgumentError, "the matrix must be square: row #{i} is not #{size} entries"
          end

          row.each_with_index.map { |value, j| entry(value, i, j) }
        end
        check_symmetric(matrix)
        matrix
      end

      # A finite real number, finite as a Float too (an Integer or Rational beyond a Float's range is not).
      def entry(value, i, j)
        number = value.to_f if value.is_a?(Numeric) && value.real? && value.finite?
        return number if number&.finite?

        raise ArgumentError, "entry (#{i}, #{j}) must be a finite number, got #{value.inspect}"
      end

      def check_symmetric(matrix)
        matrix.each_index do |i|
          (0...i).each do |j|
            upper = matrix[j][i]
            low = matrix[i][j]
            scale = [ upper.abs, low.abs, Math.sqrt(matrix[i][i].abs * matrix[j][j].abs) ].max
            next if (upper - low).abs <= SYMMETRY * scale

            raise ArgumentError, "the matrix is not symmetric: entries (#{i}, #{j}) and (#{j}, #{i}) differ"
          end
        end
      end
    end

    # One call of `root`: the residuals left, the column of S each pivot gave, and how far `rounding` reaches each row.
    class Elimination
      def initialize(matrix, floor, margin)
        @size = matrix.size
        @floor = floor
        @margin = margin
        @residual = Array.new(@size) { |i| Array.new(@size) { |j| i < j ? matrix[j][i] : matrix[i][j] } }
        @square = Array.new(@size) { Array.new(@size, 0.0) }
        @multiples = Array.new(@size) { Array.new(@size, 0.0) }
        @reach = Array.new(@size, 1.0)
        @left = (0...@size).to_a
      end

      def square_root
        @size.times do |column|
          row = largest_pivot
          @left.delete(row)
          if @residual[row][row] > tolerance(row, row)
            take_pivot(row, column)
          else
            check_dropped(row)
          end
        end

        @square
      end

      private

      def tolerance(first, second)
        [ @floor, @margin * @reach[first] * @reach[second] ].max
      end

      # The row of the largest residual pivot left, each read first: elimination only lowers a pivot, so one beyond its
      # tolerance (or an infinity or a NaN, which an overflow of an indefinite matrix leaves) never comes back.
      def largest_pivot
        largest = nil
        @left.each do |i|
          pivot = @residual[i][i]
          unless pivot.finite? && pivot >= -tolerance(i, i)
            raise ArgumentError, "the matrix is not positive semidefinite (pivot #{pivot} in column #{i})"
          end

          largest = i if largest.nil? || pivot > @residual[largest][largest]
        end
        largest
      end

      # Column `column` of S from row `row`'s pivot, and what is left of the rows still to come.
      def take_pivot(row, column)
        top = Math.sqrt(@residual[row][row])
        @square[row][column] = top
        @left.each { |i| @square[i][column] = @residual[i][row] / top }
        @left.each do |i|
          share = @square[i][column]
          @left.each { |j| @residual[i][j] -= share * @square[j][column] }
          carry(i, row, column, share / top) unless @margin.zero?
        end
      end

      # Row i gives up `multiple` of the pivot's row, itself the row given less the multiples it gave up.
      def carry(row, pivot, column, multiple)
        multiples = @multiples[row]
        given = @multiples[pivot]
        column.times { |earlier| multiples[earlier] -= multiple * given[earlier] }
        multiples[column] = multiple
        @reach[row] = 1.0 + multiples.sum(0.0, &:abs)
      end

      # Row `row`'s pivot reads as zero: what stands beside it must be within what it holds, or S * S^T would drop it.
      def check_dropped(row)
        held = [ @residual[row][row], 0.0 ].max
        @left.each do |i|
          beside = @residual[i][row]
          next if beside.abs <= Math.sqrt(held * [ @residual[i][i], 0.0 ].max) + tolerance(i, row)

          raise ArgumentError, "the matrix is not positive semidefinite (residual #{beside} in row #{i} under " \
                               "the zero pivot of column #{row})"
        end
      end
    end
    private_constant :Elimination
  end
end
