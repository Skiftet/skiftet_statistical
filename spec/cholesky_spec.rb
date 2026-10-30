# frozen_string_literal: true

# SL4: the Cholesky factor the linear-value draws use (once per call). Every expected factor below is worked out by
# hand; its entries are dyadic fractions or a named square root, so the comparisons are exact.
#
# The rule: lower-triangular L with L * L^T = matrix, its rows answered as rows (n rows of n entries, 0.0 above the
# diagonal). The pivot of column j is matrix[j][j] - sum over k < j of L[j][k]^2. A pivot below 1e-12 times the largest
# diagonal entry is set to 0 together with its column (so a part with no variance is drawn at its mean); a pivot below
# -1e-9 times the trace raises ArgumentError (the matrix is not positive semidefinite).
RSpec.describe "SkiftetStatistical::Cholesky" do
  def factor(rows)
    SkiftetStatistical::Cholesky.factor(rows)
  end

  def product(lower)
    size = lower.size
    Array.new(size) { |i| Array.new(size) { |j| (0...size).sum { |k| lower[i][k] * lower[j][k] } } }
  end

  it "SL4 factors a positive definite matrix into the rows of its lower-triangular L" do
    # By hand, L = [[2, 0, 0, 0], [1, 2, 0, 0], [0.5, 0, 1, 0], [0, 0.5, 0.5, 1]] and the matrix is L * L^T:
    #   row 0: 4, 2, 1, 0
    #   row 1: 2, 1 + 4 = 5, 0.5, 1
    #   row 2: 1, 0.5, 0.25 + 1 = 1.25, 0.5
    #   row 3: 0, 1, 0.5, 0.25 + 0.25 + 1 = 1.5
    # Column by column: L00 = sqrt 4 = 2; L10 = 2 / 2 = 1, L11 = sqrt(5 - 1) = 2; L20 = 1 / 2, L21 = (0.5 - 0.5) / 2 = 0,
    # L22 = sqrt(1.25 - 0.25) = 1; L30 = 0, L31 = 1 / 2, L32 = (0.5 - 0) / 1, L33 = sqrt(1.5 - 0.25 - 0.25) = 1.
    matrix = [
      [ 4.0, 2.0, 1.0, 0.0 ], [ 2.0, 5.0, 0.5, 1.0 ], [ 1.0, 0.5, 1.25, 0.5 ], [ 0.0, 1.0, 0.5, 1.5 ]
    ]
    lower = factor(matrix)

    expect(lower).to eq([
      [ 2.0, 0.0, 0.0, 0.0 ], [ 1.0, 2.0, 0.0, 0.0 ], [ 0.5, 0.0, 1.0, 0.0 ], [ 0.0, 0.5, 0.5, 1.0 ]
    ])
    expect(product(lower)).to eq(matrix)
  end

  it "SL4 factors a one-by-one matrix into its square root" do
    expect(factor([ [ 9.0 ] ])).to eq([ [ 3.0 ] ])
  end

  it "SL4 sets a zero pivot to 0 with its whole column, and factors the rest" do
    # The middle part has no variance: L00 = 2; L10 = 0 / 2 = 0 and the pivot 0 - 0 = 0 is below 1e-12 * 4, so L11 = 0
    # and L21 = 0; L20 = 2 / 2 = 1, L22 = sqrt(2 - 1 - 0) = 1.
    lower = factor([ [ 4.0, 0.0, 2.0 ], [ 0.0, 0.0, 0.0 ], [ 2.0, 0.0, 2.0 ] ])

    expect(lower).to eq([ [ 2.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 1.0, 0.0, 1.0 ] ])
  end

  it "SL4 factors a rank-one matrix with its second pivot set to 0" do
    # L00 = 1, L10 = 1 / 1 = 1, pivot 1 - 1 = 0 < 1e-12: L11 = 0. L * L^T = [[1, 1], [1, 1]] again.
    lower = factor([ [ 1.0, 1.0 ], [ 1.0, 1.0 ] ])

    expect(lower).to eq([ [ 1.0, 0.0 ], [ 1.0, 0.0 ] ])
    expect(product(lower)).to eq([ [ 1.0, 1.0 ], [ 1.0, 1.0 ] ])
  end

  it "SL4 sets a pivot below 1e-12 times the largest diagonal entry to 0, and keeps one at or above it" do
    # Largest diagonal entry 1: 1e-13 is below 1e-12 and goes to 0; 1e-11 is not, and L11 = sqrt(1e-11).
    expect(factor([ [ 1.0, 0.0 ], [ 0.0, 1e-13 ] ])).to eq([ [ 1.0, 0.0 ], [ 0.0, 0.0 ] ])
    expect(factor([ [ 1.0, 0.0 ], [ 0.0, 1e-11 ] ])).to eq([ [ 1.0, 0.0 ], [ 0.0, Math.sqrt(1e-11) ] ])
  end

  it "SL4 sets a pivot that rounding took just below 0 to 0 rather than raising" do
    # The pivot (1 - 1e-14) - 1 is about -1e-14: below 1e-12 * 1, so set to 0, and above -1e-9 * trace (about -2e-9).
    expect(factor([ [ 1.0, 1.0 ], [ 1.0, 1.0 - 1e-14 ] ])).to eq([ [ 1.0, 0.0 ], [ 1.0, 0.0 ] ])
  end

  it "SL4 sets every pivot of an all-zero matrix to 0" do
    # Every diagonal entry is 0, so the threshold is 0 itself; a part with no variance is still drawn at its mean, so
    # the factor is all zeros (never NaN from 0 / 0).
    expect(factor([ [ 0.0, 0.0 ], [ 0.0, 0.0 ] ])).to eq([ [ 0.0, 0.0 ], [ 0.0, 0.0 ] ])
  end

  it "SL4 factors a matrix whose first part has no variance" do
    # Pivot 0 < 1e-12 * 4: L00 = 0 and L10 = 0; then L11 = sqrt(4 - 0) = 2.
    expect(factor([ [ 0.0, 0.0 ], [ 0.0, 4.0 ] ])).to eq([ [ 0.0, 0.0 ], [ 0.0, 2.0 ] ])
  end

  it "SL4 raises ArgumentError on a matrix that is not positive semidefinite" do
    # [[1, 2], [2, 1]]: pivot 1 - 2^2 = -3, below -1e-9 * 2. [[1, 0], [0, -1]]: pivot -1, below -1e-9 * 0.
    expect { factor([ [ 1.0, 2.0 ], [ 2.0, 1.0 ] ]) }.to raise_error(ArgumentError)
    expect { factor([ [ 1.0, 0.0 ], [ 0.0, -1.0 ] ]) }.to raise_error(ArgumentError)
  end

  it "SL4 leaves its input unchanged" do
    matrix = [ [ 4.0, 2.0 ], [ 2.0, 5.0 ] ].map(&:freeze).freeze

    expect(factor(matrix)).to eq([ [ 2.0, 0.0 ], [ 1.0, 2.0 ] ])
    expect(matrix).to eq([ [ 4.0, 2.0 ], [ 2.0, 5.0 ] ])
  end

  it "SL4 needs no matrix library" do
    factor([ [ 4.0, 2.0 ], [ 2.0, 5.0 ] ])

    expect(defined?(::Matrix)).to be_nil
  end
end
