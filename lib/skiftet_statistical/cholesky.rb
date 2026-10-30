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
  # - a pivot below -1e-9 times the trace raises ArgumentError: the matrix is not positive semidefinite;
  # - a pivot below 1e-12 times the largest diagonal entry, or not above 0, is set to 0 with its whole column, so a
  #   part with no variance (or none left once the others are known) is drawn at its mean and still takes its normal.
  #   Every residual under it must be within what that pivot holds plus rounding: |r_ij| at most
  #   sqrt(pivot * q_ij) + 1e-9 times the trace, where q_ij = a_ii - sum over k < j of L_ik^2 is the pivot row i has
  #   at column j (a pivot or a q_ij not above 0 counts as 0). The square root is the bound every positive
  #   semidefinite matrix keeps; the rounding is the one a pivot is given. A larger residual raises ArgumentError: the
  #   matrix is not positive semidefinite (a part with no variance cannot covary), and setting the column to 0 would
  #   drop the residual from L * L^T;
  # - otherwise L_jj = sqrt(pivot) and L_ij = r_ij / L_jj below it.
  module Cholesky
    # A pivot below this times the largest diagonal entry is no variance: set to 0 with its column.
    ZERO_PIVOT = 1e-12
    # A pivot below minus this times the trace is no rounding: the matrix is not positive semidefinite. A residual
    # under a pivot set to 0 is given the same rounding beyond what that pivot holds.
    NEGATIVE_PIVOT = 1e-9
    # How far apart entries (i, j) and (j, i) may be, relative to their scale.
    SYMMETRY = 1e-12

    class << self
      def factor(rows)
        matrix = read(rows)
        size = matrix.size
        diagonal = Array.new(size) { |i| matrix[i][i] }
        zero_below = ZERO_PIVOT * (diagonal.max || 0.0)
        rounding = NEGATIVE_PIVOT * diagonal.sum(0.0)
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

      private

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
  end
end
