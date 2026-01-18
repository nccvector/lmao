const std = @import("std");
const simd = @import("simd");
const builtin = @import("builtin");

pub inline fn debugPrintVector(
    comptime T: type,
    comptime R: usize,
    comptime C: usize,
    v: @Vector(R * C, T),
) void {
    const a: [R * C]T = v; // safe: vector -> array copy

    for (0..R) |r| {
        std.debug.print("[", .{});
        for (0..C) |c| {
            const x = a[r * C + c];

            // Pick a decent default format for floats vs others
            if (@typeInfo(T) == .float) {
                std.debug.print("{d:.6}", .{x});
            } else {
                std.debug.print("{any}", .{x});
            }

            if (c + 1 != C) std.debug.print(", ", .{});
        }
        std.debug.print("]\n", .{});
    }
}

pub inline fn index2d(comptime R: usize, comptime C: usize, i: usize) struct { row: usize, col: usize } {
    std.debug.assert(i < R * C); // Removed in ReleaseFast
    return .{
        .row = i / C,
        .col = i % C,
    };
}

pub inline fn index1d(comptime R: usize, comptime C: usize, row: usize, col: usize) usize {
    std.debug.assert(row < R); // Removed in ReleaseFast
    std.debug.assert(col < C); // Removed in ReleaseFast
    return row * C + col;
}

pub inline fn vectorFromArray(comptime T: type, comptime N: usize, ptr: *const [N]T) @Vector(N, T) {
    // Copy load
    const tmp: [N]T = ptr.*;
    return tmp;
}

pub inline fn arrayFromVector(comptime T: type, comptime N: usize, v: @Vector(N, T)) [N]T {
    return v;
}

// Helper for fused multiply-add that works with both floats and integers
pub inline fn mulAdd(comptime T: type, comptime N: usize, a: @Vector(N, T), b: @Vector(N, T), c: @Vector(N, T)) @Vector(N, T) {
    if (comptime @typeInfo(T) == .float) {
        return @mulAdd(@Vector(N, T), a, b, c);
    } else {
        return a * b + c;
    }
}

pub inline fn dotRCK(comptime T: type, comptime R: usize, comptime C: usize, comptime K: usize, a: @Vector(R * C, T), b: @Vector(C * K, T)) @Vector(R * K, T) {
    var out: @Vector(R * K, T) = @splat(0);
    inline for (0..R) |row| {
        var acc: @Vector(K, T) = @splat(0);
        inline for (0..C) |col| {
            const av: @Vector(K, T) = @splat(a[row * C + col]);

            comptime var idx: [K]i32 = undefined;
            inline for (0..K) |i| idx[i] = @intCast(col * K + i);
            const bv: @Vector(K, T) = @shuffle(T, b, undefined, @as(@Vector(K, i32), idx)); // get row
            acc = mulAdd(T, K, av, bv, acc);
        }
        inline for (0..K) |i| {
            out[row * K + i] = acc[i];
        }
    }

    return out;
}

pub inline fn identity(comptime T: type, comptime N: usize) @Vector(N * N, T) {
    var I: @Vector(N * N, T) = @splat(0);
    for (0..N) |i| I[i * N + i] = 1;
    return I;
}

// TODO: profile and optimize this...(try if shuffle is faster)
pub inline fn transposeRCCR(comptime T: type, comptime R: usize, comptime C: usize, v: @Vector(R * C, T)) @Vector(C * R, T) {
    var out: @Vector(C * R, T) = undefined;
    for (0..R) |r| {
        for (0..C) |c| {
            const src = r * C + c;
            const dst = c * R + r;
            out[dst] = v[src];
        }
    }
    return out;
}

pub inline fn splitRows(comptime T: type, comptime R: usize, comptime C: usize, v: @Vector(R * C, T)) [R]@Vector(C, T) {
    @setRuntimeSafety(false);
    const arr_ptr: *const [R * C]T = @ptrCast(&v);
    var rows: [R]@Vector(C, T) = undefined;
    inline for (0..R) |r| {
        const row_ptr: *const [C]T = arr_ptr.*[r * C ..][0..C];
        rows[r] = @as(*const @Vector(C, T), @ptrCast(@alignCast(row_ptr))).*;
    }
    return rows;
}

pub inline fn joinRows(comptime T: type, comptime R: usize, comptime C: usize, rows: [R]@Vector(C, T)) @Vector(R * C, T) {
    var out_arr: [R * C]T = undefined; // view vector storage as array

    // Write rows one by one
    for (0..R) |r| {
        const row_arr: [C]T = @bitCast(rows[r]);
        std.mem.copyForwards(T, out_arr[r * C .. r * C + C], row_arr[0..]);
    }

    return @bitCast(out_arr);
}

pub inline fn horzcatInto(
    comptime T: type,
    comptime R1: usize,
    comptime C1: usize,
    comptime R2: usize,
    comptime C2: usize,
    out: *@Vector(R1 * (C1 + C2), T),
    v1: *const @Vector(R1 * C1, T),
    v2: *const @Vector(R2 * C2, T),
) void {
    comptime {
        if (R1 != R2) @compileError("v1 and v2 must have same number of rows");
    }

    @setRuntimeSafety(false);

    const OutLen = R1 * (C1 + C2);

    // View vectors as flat arrays (no copy, just pointer reinterpretation).
    const out_a: *[OutLen]T = @ptrCast(out);
    const a1: *const [R1 * C1]T = @ptrCast(v1);
    const a2: *const [R2 * C2]T = @ptrCast(v2);

    const out_cols = C1 + C2;

    inline for (0..R1) |r| {
        const o = r * out_cols;
        const ii1 = r * C1;
        const ii2 = r * C2;

        // Copy row r from v1 into out
        @memcpy(out_a[o .. o + C1], a1[ii1 .. ii1 + C1]);

        // Copy row r from v2 into out (right side)
        @memcpy(out_a[o + C1 .. o + C1 + C2], a2[ii2 .. ii2 + C2]);
    }
}

pub inline fn horzcat(
    comptime T: type,
    comptime R1: usize,
    comptime C1: usize,
    comptime R2: usize,
    comptime C2: usize,
    v1: @Vector(R1 * C1, T),
    v2: @Vector(R2 * C2, T),
) @Vector(R1 * (C1 + C2), T) {
    var out: @Vector(R1 * (C1 + C2), T) = undefined;
    horzcatInto(T, R1, C1, R2, C2, &out, &v1, &v2);
    return out;
}

pub inline fn rowEchelonForm(comptime T: type, comptime R: usize, comptime C: usize, aug_eqs: *[R]@Vector(C, T)) void {
    comptime {
        if (@typeInfo(T) != .float) @compileError("function requires floating point type T");
        if (R == 0) @compileError("R must be non-zero");
        if (C == 0) @compileError("C must be non-zero");
    }

    const D = @min(R, C); // num diagonals

    // Loop over diagonals
    for (0..D) |c| {
        var pivot: T = aug_eqs[c][c]; // Start with current diagonal as pivot
        var pivot_row: usize = c;
        // Find best pivot candidate in column values below diagonal
        for (c..D) |r| {
            if (r <= c) continue; // skip current row
            if (@abs(aug_eqs[r][c]) > @abs(pivot)) {
                pivot = aug_eqs[r][c];
                pivot_row = r;
            }
        }

        // Check feasibility (non zero pivot check)
        if (@abs(pivot) < 1e-6) {
            @trap();
        }

        // Swap selected pivot row with current
        if (pivot != aug_eqs[c][c]) {
            const tmp = aug_eqs[c];
            aug_eqs[c] = aug_eqs[pivot_row];
            aug_eqs[pivot_row] = tmp;
        }

        // sm[c] is now the current active row that contains current pivot

        // Divide current row by current pivot (to make the pivot element 1)
        aug_eqs[c] /= @splat(pivot);

        // Zero all values below current pivot
        for (c..D) |r| {
            if (r <= c) continue; // skip current row
            const bp: @Vector(C, T) = @splat(-aug_eqs[r][c]);
            aug_eqs[r] = @mulAdd(@Vector(C, T), bp, aug_eqs[c], aug_eqs[r]);
        }
    }
}

/// Row echelon form for determinant calculation.
/// Unlike rowEchelonForm, this does NOT normalize rows (no division by pivot)
/// and returns the number of row swaps (for sign) or null if singular.
pub inline fn rowEchelonFormForDet(comptime T: type, comptime N: usize, rows: *[N]@Vector(N, T)) ?usize {
    comptime {
        if (@typeInfo(T) != .float) @compileError("function requires floating point type T");
        if (N == 0) @compileError("N must be non-zero");
    }

    const eps = std.math.floatEps(T);
    var swap_count: usize = 0;

    for (0..N) |col| {
        // Find pivot (largest absolute value in column)
        var pivot: T = rows[col][col];
        var pivot_row: usize = col;
        for (col + 1..N) |r| {
            if (@abs(rows[r][col]) > @abs(pivot)) {
                pivot = rows[r][col];
                pivot_row = r;
            }
        }

        // Singular if pivot is ~0
        if (@abs(pivot) < eps) return null;

        // Swap rows if needed
        if (pivot_row != col) {
            const tmp = rows[col];
            rows[col] = rows[pivot_row];
            rows[pivot_row] = tmp;
            swap_count += 1;
        }

        // Eliminate below (without normalizing the pivot row)
        for (col + 1..N) |r| {
            const factor: @Vector(N, T) = @splat(-rows[r][col] / rows[col][col]);
            rows[r] = @mulAdd(@Vector(N, T), factor, rows[col], rows[r]);
        }
    }

    return swap_count;
}

pub inline fn reducedRowEchelonForm(comptime T: type, comptime R: usize, comptime C: usize, aug_eqs: *[R]@Vector(C, T)) void {
    comptime {
        if (@typeInfo(T) != .float) @compileError("function requires floating point type T");
        if (R == 0) @compileError("R must be non-zero");
        if (C == 0) @compileError("C must be non-zero");
    }

    // Compute row echelon form (forward pass)
    rowEchelonForm(T, R, C, aug_eqs);

    const D = @min(R, C); // num diagonals

    // Compute reduced row echelon form (backwards pass)
    for (0..D) |c| {
        const r = D - c - 1; // Reverse loop (bottom-up, currently chosen pivot row)
        if (r <= 0) break; // There is nothing above the first row

        // Forward loop on rows (top to bottom until currently chosen pivot)
        for (0..r) |row| {
            // Element of interest
            const eoi: @Vector(C, T) = @splat(-aug_eqs[row][r]);
            // Eliminate element of interest from current row using the pivots row
            aug_eqs[row] = @mulAdd(@Vector(C, T), eoi, aug_eqs[r], aug_eqs[row]);
        }
    }
}

/// QR decomposition via Householder reflections.
/// Decomposes input matrix A (R×C) into Q (R×R orthogonal) and R_out (R×C upper triangular)
/// such that A = Q * R_out.
/// Requires floating point type T.
pub inline fn qrHouseholder(
    comptime T: type,
    comptime R: usize,
    comptime C: usize,
    A: @Vector(R * C, T),
    Q: *@Vector(R * R, T),
    R_out: *@Vector(R * C, T),
) void {
    comptime {
        if (@typeInfo(T) != .float) @compileError("qrHouseholder requires floating point type T");
        if (R == 0) @compileError("R must be non-zero");
        if (C == 0) @compileError("C must be non-zero");
    }

    const eps = std.math.floatEps(T);
    const K = @min(R, C); // Number of Householder steps

    // Initialize R_out = A
    R_out.* = A;

    // Initialize Q = I
    Q.* = identity(T, R);

    // Convert to arrays for easier indexing
    var r_arr: *[R * C]T = @ptrCast(R_out);
    var q_arr: *[R * R]T = @ptrCast(Q);

    // Householder vector buffer (only indices k..R-1 are used)
    var v: [R]T = undefined;

    // For each column k
    for (0..K) |k| {
        // Step 3: Form the Householder vector from column k, rows k..R-1

        // Compute norm of x = R[k:R-1, k]
        var norm_x_sq: T = 0;
        for (k..R) |i| {
            const val = r_arr[i * C + k];
            norm_x_sq += val * val;
        }
        const norm_x = @sqrt(norm_x_sq);

        // Skip if norm is too small (nothing to eliminate)
        if (norm_x < eps) continue;

        // x0 = R[k, k]
        const x0 = r_arr[k * C + k];

        // alpha = -sign(x0) * norm_x
        const sign_x0: T = if (x0 < 0) -1 else 1;
        const alpha = -sign_x0 * norm_x;

        // Build v: v[i] = x[i] for i in k..R-1, then v[k] -= alpha
        for (k..R) |i| {
            v[i] = r_arr[i * C + k];
        }
        v[k] -= alpha;

        // Compute v^T v
        var vtv: T = 0;
        for (k..R) |i| {
            vtv += v[i] * v[i];
        }

        // Skip if v^T v is too small
        if (vtv < eps) continue;

        // beta = 2 / (v^T v)
        const beta = 2 / vtv;

        // Step 4: Apply reflector to R_out (left-multiply): R = H * R
        // For columns j in k..C-1
        for (k..C) |j| {
            // dot = v^T * R[k:R-1, j]
            var dot: T = 0;
            for (k..R) |i| {
                dot += v[i] * r_arr[i * C + j];
            }
            const tau = beta * dot;
            // R[k:R-1, j] -= tau * v
            for (k..R) |i| {
                r_arr[i * C + j] -= tau * v[i];
            }
        }

        // Step 5: Accumulate Q (right-multiply): Q = Q * H
        // For each row i in 0..R-1
        for (0..R) |i| {
            // dot = Q[i, k:R-1] * v[k:R-1]
            var dot: T = 0;
            for (k..R) |j| {
                dot += q_arr[i * R + j] * v[j];
            }
            const tau = beta * dot;
            // Q[i, k:R-1] -= tau * v
            for (k..R) |j| {
                q_arr[i * R + j] -= tau * v[j];
            }
        }

        // Step 6: Clean up numerical artifacts - force subdiagonal to zero in column k
        for (k + 1..R) |i| {
            r_arr[i * C + k] = 0;
        }
    }
}

/// Compact QR decomposition via Householder reflections (LAPACK-style).
/// Overwrites A in-place:
///   - Upper triangle contains R
///   - Below diagonal contains Householder vectors (v[k]=1 implicit, v[i>k] stored)
/// Returns tau array with one scalar per reflector.
/// This avoids forming Q explicitly - use applyQTranspose to apply Q^T to vectors.
pub inline fn qrHouseholderCompact(
    comptime T: type,
    comptime R: usize,
    comptime C: usize,
    A: *[R]@Vector(C, T),
    tau: *[R]T,
) void {
    comptime {
        if (@typeInfo(T) != .float) @compileError("qrHouseholderCompact requires floating point type T");
        if (R == 0) @compileError("R must be non-zero");
        if (C == 0) @compileError("C must be non-zero");
    }

    const eps = std.math.floatEps(T);
    const NumReflectors = @min(R, C);

    // Initialize tau to zero
    tau.* = [_]T{0} ** R;

    // For each column k
    for (0..NumReflectors) |k| {
        // Compute norm of x = A[k:R-1, k]
        var norm_x_sq: T = 0;
        for (k..R) |i| {
            const val = A[i][k];
            norm_x_sq = @mulAdd(T, val, val, norm_x_sq);
        }
        const norm_x = @sqrt(norm_x_sq);

        // Skip if norm is too small (nothing to eliminate)
        if (norm_x < eps) continue;

        // x0 = A[k, k]
        const x0 = A[k][k];

        // alpha = -sign(x0) * norm_x (new diagonal value for R)
        const sign_x0: T = if (x0 < 0) -1 else 1;
        const alpha = -sign_x0 * norm_x;

        // Householder vector: v = x - alpha*e1, where x is column k below row k
        // v[k] = x0 - alpha, v[i] = x[i] = A[i,k] for i > k
        const v_k = x0 - alpha;

        // Compute v^T v using FMA
        var vtv: T = v_k * v_k;
        for (k + 1..R) |i| {
            const vi = A[i][k];
            vtv = @mulAdd(T, vi, vi, vtv);
        }

        // Skip if v^T v is too small
        if (vtv < eps) continue;

        // beta = 2 / (v^T v)
        const beta = 2 / vtv;

        // Store scaled tau: with v' = v/v_k (so v'[k]=1), tau' = beta * v_k^2
        tau[k] = beta * v_k * v_k;

        // Apply reflector to A[k:, k+1:] (columns k+1..C-1 only)
        // H = I - beta * v * v^T

        // Only process if there are trailing columns
        if (k + 1 < C) {
            // Phase 1: Compute w = beta * (v^T * A[k:, k+1:]) using vectorized ops
            // w is a row vector for trailing columns

            // Start with contribution from row k (where v[k] = v_k)
            const v_k_splat: @Vector(C, T) = @splat(v_k);
            var w: @Vector(C, T) = v_k_splat * A[k];

            // Contribution from rows k+1..R-1 (where v[i] = A[i,k])
            for (k + 1..R) |i| {
                const vi_splat: @Vector(C, T) = @splat(A[i][k]);
                w = @mulAdd(@Vector(C, T), vi_splat, A[i], w);
            }

            // Scale by beta (vectorized)
            const beta_splat: @Vector(C, T) = @splat(beta);
            w *= beta_splat;

            // Zero out w[0..k] - we only want to update columns k+1..C-1
            // Columns 0..k contain R values or will be overwritten
            for (0..k + 1) |j| {
                w[j] = 0;
            }

            // Phase 2: Update A[k:, k+1:] -= v * w (vectorized)
            // Update row k: A[k] -= v_k * w
            A[k] -= v_k_splat * w;

            // Precompute 1/v_k for scaling reflectors (single division)
            const inv_v_k = 1 / v_k;

            // Update rows k+1..R-1, also store scaled reflector
            for (k + 1..R) |i| {
                const vi = A[i][k];
                const vi_splat: @Vector(C, T) = @splat(vi);
                // A[i] -= vi * w (vectorized with FMA: A[i] = -vi * w + A[i])
                A[i] = @mulAdd(@Vector(C, T), -vi_splat, w, A[i]);
                // Store scaled reflector (v[i] / v_k)
                A[i][k] = vi * inv_v_k;
            }
        } else {
            // No trailing columns, just store scaled reflector
            const inv_v_k = 1 / v_k;
            for (k + 1..R) |i| {
                A[i][k] *= inv_v_k;
            }
        }

        // Store alpha (R diagonal) in A[k,k]
        A[k][k] = alpha;
    }
}

/// Apply Q^T to a set of column vectors B (in-place) using stored reflectors.
/// A contains R in upper triangle, reflectors below diagonal (from qrHouseholderCompact).
/// tau contains the reflector scalars.
/// B is R x K matrix of K column vectors to transform.
/// Convention: v[k] = 1 (implicit), v[i>k] = A[i,k]
pub inline fn applyQTranspose(
    comptime T: type,
    comptime R: usize,
    comptime C: usize,
    comptime K: usize,
    A: *const [R]@Vector(C, T),
    tau: *const [R]T,
    B: *[R]@Vector(K, T),
) void {
    comptime {
        if (@typeInfo(T) != .float) @compileError("applyQTranspose requires floating point type T");
    }

    const NumReflectors = @min(R, C);

    // Apply reflectors in forward order (k = 0, 1, ..., NumReflectors-1)
    // Each reflector: B = H_k * B = (I - tau * v * v^T) * B = B - tau * v * (v^T * B)
    // With v[k] = 1 implicit, v[i>k] = A[i,k]
    for (0..NumReflectors) |k| {
        const beta = tau[k];
        if (beta == 0) continue;

        // Phase 1: Compute w = v^T * B (w is 1 x K row vector)
        // Start with contribution from row k (where v[k] = 1 implicit)
        var w: @Vector(K, T) = B[k];

        // Contribution from rows k+1..R-1 (where v[i] = A[i,k])
        for (k + 1..R) |i| {
            const vi: @Vector(K, T) = @splat(A[i][k]);
            w = @mulAdd(@Vector(K, T), vi, B[i], w);
        }

        // Scale by beta
        w *= @as(@Vector(K, T), @splat(beta));

        // Phase 2: B[i] -= v[i] * w (using FMA: B[i] = -v[i] * w + B[i])
        B[k] -= w; // v[k] = 1
        for (k + 1..R) |i| {
            const neg_vi: @Vector(K, T) = @splat(-A[i][k]);
            B[i] = @mulAdd(@Vector(K, T), neg_vi, w, B[i]);
        }
    }
}

/// Upper triangular back-substitution: solve R * x = b
/// R is upper triangular (R x C matrix, only upper R x R part used for square solve)
/// b is R x K matrix of K right-hand sides
/// Solution x overwrites b in-place.
pub inline fn backSubstitute(
    comptime T: type,
    comptime R: usize,
    comptime C: usize,
    comptime K: usize,
    A: *const [R]@Vector(C, T),
    b: *[R]@Vector(K, T),
) void {
    comptime {
        if (@typeInfo(T) != .float) @compileError("backSubstitute requires floating point type T");
        if (R > C) @compileError("backSubstitute requires R <= C for upper triangular solve");
    }

    // Solve from bottom to top
    var i: usize = R;
    while (i > 0) {
        i -= 1;
        // x[i] = (b[i] - sum(R[i,j] * x[j] for j > i)) / R[i,i]
        var sum: @Vector(K, T) = @splat(0);
        for (i + 1..R) |j| {
            sum = @mulAdd(@Vector(K, T), @as(@Vector(K, T), @splat(A[i][j])), b[j], sum);
        }
        b[i] = (b[i] - sum) / @as(@Vector(K, T), @splat(A[i][i]));
    }
}

/// Solve A * x = b using QR decomposition.
/// A is R x C matrix (R <= C for overdetermined/square systems).
/// b is R x K matrix of K right-hand sides.
/// Returns x (C x K) by computing: x = R^-1 * Q^T * b
/// Note: For tall matrices (R > C), this solves the least squares problem.
pub inline fn qrSolve(
    comptime T: type,
    comptime R: usize,
    comptime C: usize,
    comptime K: usize,
    A: [R]@Vector(C, T),
    b: [R]@Vector(K, T),
) [R]@Vector(K, T) {
    comptime {
        if (@typeInfo(T) != .float) @compileError("qrSolve requires floating point type T");
    }

    // Copy A for factorization
    var A_copy = A;
    var tau: [R]T = undefined;

    // Factor: A_copy now contains R (upper) and reflectors (lower)
    qrHouseholderCompact(T, R, C, &A_copy, &tau);

    // Copy b for transformation
    var x = b;

    // Apply Q^T to b: x = Q^T * b
    applyQTranspose(T, R, C, K, &A_copy, &tau, &x);

    // Back-substitute: solve R * result = x
    backSubstitute(T, R, C, K, &A_copy, &x);

    return x;
}

pub fn Matrix(comptime T: type, comptime R: usize, comptime C: usize) type {
    const enable_ct_log = false; // flip when debugging

    comptime {
        const info = @typeInfo(T);
        if (info != .float and info != .int) @compileError("Matrix type must be a numeric type (float or int).");
        if (R == 0) @compileError("Matrix row count must be non-zero.");
        if (C == 0) @compileError("Matrix column count must be non-zero.");
    }

    return struct {
        data: @Vector(R * C, T),

        pub const Self = @This();
        pub const rows = R;
        pub const cols = C;
        pub const ScalarType = T;

        comptime {
            if (enable_ct_log) {
                @compileLog(
                    "Matrix " ++ @typeName(T) ++
                        " (" ++ std.fmt.comptimePrint("{}x{}", .{ R, C }) ++ ")" ++
                        " size=" ++ std.fmt.comptimePrint("{}", .{@sizeOf(Self)}) ++
                        " align=" ++ std.fmt.comptimePrint("{}", .{@alignOf(Self)}),
                );
            }
        }

        pub fn fromArray(a: *const [R * C]T) Self {
            return .{ .data = vectorFromArray(T, R * C, a) };
        }

        pub fn toArray(self: Self) [R * C]T {
            return arrayFromVector(T, R * C, self.data);
        }

        pub fn add(self: Self, other: Self) Self {
            return .{ .data = self.data + other.data };
        }

        pub fn sub(self: Self, other: Self) Self {
            return .{ .data = self.data - other.data };
        }

        pub fn multiply(self: Self, other: Self) Self {
            return .{ .data = self.data * other.data };
        }

        pub fn transpose(self: Self) Matrix(T, C, R) {
            return .{ .data = transposeRCCR(T, R, C, self.data) };
        }

        pub fn dotSIMD(self: Self, other: anytype) Matrix(T, R, @TypeOf(other).cols) {
            comptime {
                const OT = @TypeOf(other);
                // Enforce that other has the expected static shape
                if (!@hasDecl(OT, "rows") or !@hasDecl(OT, "cols"))
                    @compileError("dot: other must be a MatrixX(...) type");
                if (OT.rows != C)
                    @compileError("dot: dimension mismatch: self.cols must equal other.rows");
                if (OT.ScalarType != T)
                    @compileError("dot: scalar type mismatch");
            }

            const K: usize = @TypeOf(other).cols;

            return Matrix(T, R, K).fromArray(&dotRCK(T, R, C, K, self.data, other.data));
        }

        pub fn dot(self: Self, other: anytype) Matrix(T, R, @TypeOf(other).cols) {
            comptime {
                const OT = @TypeOf(other);
                // Enforce that other has the expected static shape
                if (!@hasDecl(OT, "rows") or !@hasDecl(OT, "cols"))
                    @compileError("dot: other must be a MatrixX(...) type");
                if (OT.rows != C)
                    @compileError("dot: dimension mismatch: self.cols must equal other.rows");
                if (OT.ScalarType != T)
                    @compileError("dot: scalar type mismatch");
            }

            const K: usize = @TypeOf(other).cols;
            var out: Matrix(T, R, K) = undefined;

            for (0..R) |r| {
                for (0..K) |k| {
                    var acc: T = 0;
                    for (0..C) |c| {
                        acc += self.data[r * C + c] * other.data[c * K + k];
                    }
                    out.data[r * K + k] = acc;
                }
            }
            return out;
        }

        fn isVec() bool {
            return C == 1 and R >= 2 and R <= 4;
        }

        /// Swizzle using comptime string: v.sw("yzx") or v.sw("wxyz")
        pub fn sw(self: Self, comptime pattern: []const u8) Matrix(T, pattern.len, 1) {
            comptime {
                if (!isVec()) @compileError("sw: only valid for vectors MatrixX(T, R, 1) where R=2..4");
            }

            var out: Matrix(T, pattern.len, 1) = undefined;

            inline for (pattern, 0..) |ch, i| {
                const idx: usize = switch (ch) {
                    'x' => 0,
                    'y' => 1,
                    'z' => 2,
                    'w' => 3,
                    else => @compileError("sw: pattern may only contain 'x','y','z','w'"),
                };
                comptime {
                    if (idx >= R)
                        @compileError("sw: component out of range for this vector size");
                }
                out.data[i] = self.data[idx];
            }
            return out;
        }

        // Optional single-component helpers (method style)
        pub inline fn x(self: Self) T {
            comptime if (!(isVec() and R >= 1)) @compileError("x only for vec");
            return self.data[0];
        }
        pub inline fn y(self: Self) T {
            comptime if (!(isVec() and R >= 2)) @compileError("y only for vec2..4");
            return self.data[1];
        }
        pub inline fn z(self: Self) T {
            comptime if (!(isVec() and R >= 3)) @compileError("z only for vec3..4");
            return self.data[2];
        }
        pub inline fn w(self: Self) T {
            comptime if (!(isVec() and R == 4)) @compileError("w only for vec4");
            return self.data[3];
        }

        // ========================================================================
        // Linear Algebra Methods
        // ========================================================================

        /// Creates an identity matrix. Only available for square matrices.
        /// Returns a Matrix with 1s on the diagonal and 0s elsewhere.
        pub fn identity() Self {
            comptime {
                if (R != C) @compileError("identity() is only available for square matrices (R == C)");
            }
            return .{ .data = lmao_identity(T, R) };
        }

        /// Computes the row echelon form of the matrix using Gaussian elimination.
        /// Returns a new matrix in row echelon form with leading 1s.
        /// Requires floating point type.
        pub fn rowEchelonForm(self: Self) Self {
            comptime {
                if (@typeInfo(T) != .float) @compileError("rowEchelonForm requires floating point type");
            }
            var row_data = splitRows(T, R, C, self.data);
            lmao_rowEchelonForm(T, R, C, &row_data);
            return .{ .data = joinRows(T, R, C, row_data) };
        }

        /// Computes the reduced row echelon form of the matrix using Gauss-Jordan elimination.
        /// Returns a new matrix in reduced row echelon form.
        /// Requires floating point type.
        pub fn reducedRowEchelonForm(self: Self) Self {
            comptime {
                if (@typeInfo(T) != .float) @compileError("reducedRowEchelonForm requires floating point type");
            }
            var row_data = splitRows(T, R, C, self.data);
            lmao_reducedRowEchelonForm(T, R, C, &row_data);
            return .{ .data = joinRows(T, R, C, row_data) };
        }

        /// Result type for QR decomposition
        pub const QRResult = struct {
            Q: Matrix(T, R, R),
            R_mat: Self,
        };

        /// Computes QR decomposition using Householder reflections.
        /// Returns Q (RxR orthogonal matrix) and R (RxC upper triangular matrix)
        /// such that A = Q * R.
        /// Requires floating point type.
        pub fn qrDecompose(self: Self) QRResult {
            comptime {
                if (@typeInfo(T) != .float) @compileError("qrDecompose requires floating point type");
            }
            var Q: @Vector(R * R, T) = undefined;
            var R_out: @Vector(R * C, T) = undefined;
            qrHouseholder(T, R, C, self.data, &Q, &R_out);
            return .{
                .Q = .{ .data = Q },
                .R_mat = .{ .data = R_out },
            };
        }

        /// Solves the linear system Ax = b using QR decomposition.
        /// self is the matrix A, b is a column vector (Rx1 matrix).
        /// Returns the solution vector x.
        /// Requires floating point type and R == C (square matrix).
        pub fn solve(self: Self, b: Matrix(T, R, 1)) Matrix(T, R, 1) {
            comptime {
                if (@typeInfo(T) != .float) @compileError("solve requires floating point type");
                if (R > C) @compileError("solve requires R <= C (square or wide matrix)");
            }
            const A_rows = splitRows(T, R, C, self.data);
            const b_rows = splitRows(T, R, 1, b.data);
            const x_rows = qrSolve(T, R, C, 1, A_rows, b_rows);
            return .{ .data = joinRows(T, R, 1, x_rows) };
        }

        /// Solves the linear system AX = B using QR decomposition for multiple right-hand sides.
        /// self is the matrix A (RxC), B is a matrix of K column vectors (RxK matrix).
        /// Returns the solution matrix X (RxK).
        /// Requires floating point type and R <= C.
        pub fn solveMulti(self: Self, comptime K: usize, B: Matrix(T, R, K)) Matrix(T, R, K) {
            comptime {
                if (@typeInfo(T) != .float) @compileError("solveMulti requires floating point type");
                if (R > C) @compileError("solveMulti requires R <= C (square or wide matrix)");
            }
            const A_rows = splitRows(T, R, C, self.data);
            const B_rows = splitRows(T, R, K, B.data);
            const X_rows = qrSolve(T, R, C, K, A_rows, B_rows);
            return .{ .data = joinRows(T, R, K, X_rows) };
        }

        /// Horizontally concatenates two matrices with the same number of rows.
        /// Returns a new matrix [self | other] with dimensions R x (C + other.cols).
        pub fn horzcat(self: Self, other: anytype) Matrix(T, R, C + @TypeOf(other).cols) {
            comptime {
                const OT = @TypeOf(other);
                if (!@hasDecl(OT, "rows") or !@hasDecl(OT, "cols"))
                    @compileError("horzcat: other must be a Matrix type");
                if (OT.rows != R)
                    @compileError("horzcat: matrices must have the same number of rows");
                if (OT.ScalarType != T)
                    @compileError("horzcat: scalar type mismatch");
            }
            const OtherC = @TypeOf(other).cols;
            return .{ .data = lmao_horzcat(T, R, C, R, OtherC, self.data, other.data) };
        }

        // ========================================================================
        // Vector Operations (for column vectors, C == 1)
        // ========================================================================

        /// Computes the dot product of two vectors.
        /// Only available for column vectors (C == 1).
        pub fn dotProduct(self: Self, other: Self) T {
            comptime {
                if (C != 1) @compileError("dotProduct is only available for column vectors (C == 1)");
            }
            var result: T = 0;
            inline for (0..R) |i| {
                result += self.data[i] * other.data[i];
            }
            return result;
        }

        /// Computes the squared length (magnitude squared) of a vector.
        /// Only available for column vectors (C == 1).
        pub fn lengthSquared(self: Self) T {
            comptime {
                if (C != 1) @compileError("lengthSquared is only available for column vectors (C == 1)");
            }
            return self.dotProduct(self);
        }

        /// Computes the length (magnitude) of a vector.
        /// Only available for column vectors (C == 1) with floating point type.
        pub fn length(self: Self) T {
            comptime {
                if (C != 1) @compileError("length is only available for column vectors (C == 1)");
                if (@typeInfo(T) != .float) @compileError("length requires floating point type");
            }
            return @sqrt(self.lengthSquared());
        }

        /// Returns a normalized (unit length) version of the vector.
        /// Only available for column vectors (C == 1) with floating point type.
        pub fn normalized(self: Self) Self {
            comptime {
                if (C != 1) @compileError("normalized is only available for column vectors (C == 1)");
                if (@typeInfo(T) != .float) @compileError("normalized requires floating point type");
            }
            const len = self.length();
            if (len < std.math.floatEps(T)) {
                return self; // Return original vector if too small to normalize
            }
            return .{ .data = self.data / @as(@Vector(R, T), @splat(len)) };
        }

        /// Computes the cross product of two 3D vectors.
        /// Only available for Vec3 (R == 3, C == 1).
        pub fn cross(self: Self, other: Self) Self {
            comptime {
                if (R != 3 or C != 1) @compileError("cross product is only available for 3D vectors (R == 3, C == 1)");
            }
            return .{
                .data = .{
                    self.data[1] * other.data[2] - self.data[2] * other.data[1],
                    self.data[2] * other.data[0] - self.data[0] * other.data[2],
                    self.data[0] * other.data[1] - self.data[1] * other.data[0],
                },
            };
        }

        /// Negates the vector/matrix (returns -self).
        pub fn negate(self: Self) Self {
            return .{ .data = -self.data };
        }

        /// Scalar multiplication: returns self * scalar.
        pub fn scale(self: Self, scalar: T) Self {
            return .{ .data = self.data * @as(@Vector(R * C, T), @splat(scalar)) };
        }

        /// Scalar division: returns self / scalar.
        pub fn divScalar(self: Self, scalar: T) Self {
            return .{ .data = self.data / @as(@Vector(R * C, T), @splat(scalar)) };
        }

        /// Element-wise minimum.
        pub fn min(self: Self, other: Self) Self {
            return .{ .data = @min(self.data, other.data) };
        }

        /// Element-wise maximum.
        pub fn max(self: Self, other: Self) Self {
            return .{ .data = @max(self.data, other.data) };
        }

        /// Computes the distance between two points (vectors).
        /// Only available for column vectors (C == 1) with floating point type.
        pub fn distance(self: Self, other: Self) T {
            comptime {
                if (C != 1) @compileError("distance is only available for column vectors (C == 1)");
                if (@typeInfo(T) != .float) @compileError("distance requires floating point type");
            }
            return self.sub(other).length();
        }

        // ========================================================================
        // Matrix Inverse (for square matrices)
        // ========================================================================

        /// Computes the determinant of a square matrix.
        /// Only available for square matrices (R == C).
        /// Uses optimized formulas for 2x2 and 3x3, LU decomposition for larger.
        pub fn determinant(self: Self) T {
            comptime {
                if (R != C) @compileError("determinant is only available for square matrices (R == C)");
                if (@typeInfo(T) != .float) @compileError("determinant requires floating point type");
            }

            if (R == 2) {
                // 2x2: ad - bc
                return self.data[0] * self.data[3] - self.data[1] * self.data[2];
            } else if (R == 3) {
                // 3x3: Sarrus' rule / cofactor expansion
                const a = self.data[0];
                const b = self.data[1];
                const c = self.data[2];
                const d = self.data[3];
                const e = self.data[4];
                const f = self.data[5];
                const g = self.data[6];
                const h = self.data[7];
                const i = self.data[8];
                return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
            } else if (R == 4) {
                // 4x4: Cofactor expansion along first row
                const m = self.data;
                const s0 = m[0] * m[5] - m[4] * m[1];
                const s1 = m[0] * m[6] - m[4] * m[2];
                const s2 = m[0] * m[7] - m[4] * m[3];
                const s3 = m[1] * m[6] - m[5] * m[2];
                const s4 = m[1] * m[7] - m[5] * m[3];
                const s5 = m[2] * m[7] - m[6] * m[3];

                const c5 = m[10] * m[15] - m[14] * m[11];
                const c4 = m[9] * m[15] - m[13] * m[11];
                const c3 = m[9] * m[14] - m[13] * m[10];
                const c2 = m[8] * m[15] - m[12] * m[11];
                const c1 = m[8] * m[14] - m[12] * m[10];
                const c0 = m[8] * m[13] - m[12] * m[9];

                return s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
            } else {
                // General case: Use row echelon form
                // Determinant = (-1)^swaps * product of diagonal elements
                var row_data = splitRows(T, R, C, self.data);

                const swap_count = rowEchelonFormForDet(T, R, &row_data) orelse return 0;

                // Product of diagonal elements
                var det: T = 1;
                for (0..R) |i| {
                    det *= row_data[i][i];
                }

                // Apply sign from row swaps
                if (swap_count % 2 == 1) det = -det;

                return det;
            }
        }

        /// Computes the inverse of a square matrix.
        /// Only available for square matrices (R == C) with floating point type.
        /// Returns null if the matrix is singular (non-invertible).
        pub fn inverse(self: Self) ?Self {
            comptime {
                if (R != C) @compileError("inverse is only available for square matrices (R == C)");
                if (@typeInfo(T) != .float) @compileError("inverse requires floating point type");
            }

            const eps = std.math.floatEps(T);

            if (R == 2) {
                // 2x2 direct formula
                const det = self.determinant();
                if (@abs(det) < eps) return null;
                const inv_det = 1 / det;
                return .{
                    .data = .{
                        self.data[3] * inv_det,
                        -self.data[1] * inv_det,
                        -self.data[2] * inv_det,
                        self.data[0] * inv_det,
                    },
                };
            } else if (R == 3) {
                // 3x3 using cofactors
                const m = self.data;
                const a = m[0];
                const b = m[1];
                const c = m[2];
                const d = m[3];
                const e = m[4];
                const f = m[5];
                const g = m[6];
                const h = m[7];
                const i = m[8];

                const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
                if (@abs(det) < eps) return null;
                const inv_det = 1 / det;

                return .{
                    .data = .{
                        (e * i - f * h) * inv_det,
                        (c * h - b * i) * inv_det,
                        (b * f - c * e) * inv_det,
                        (f * g - d * i) * inv_det,
                        (a * i - c * g) * inv_det,
                        (c * d - a * f) * inv_det,
                        (d * h - e * g) * inv_det,
                        (b * g - a * h) * inv_det,
                        (a * e - b * d) * inv_det,
                    },
                };
            } else if (R == 4) {
                // 4x4 optimized inverse using cofactors
                const m = self.data;

                const s0 = m[0] * m[5] - m[4] * m[1];
                const s1 = m[0] * m[6] - m[4] * m[2];
                const s2 = m[0] * m[7] - m[4] * m[3];
                const s3 = m[1] * m[6] - m[5] * m[2];
                const s4 = m[1] * m[7] - m[5] * m[3];
                const s5 = m[2] * m[7] - m[6] * m[3];

                const c5 = m[10] * m[15] - m[14] * m[11];
                const c4 = m[9] * m[15] - m[13] * m[11];
                const c3 = m[9] * m[14] - m[13] * m[10];
                const c2 = m[8] * m[15] - m[12] * m[11];
                const c1 = m[8] * m[14] - m[12] * m[10];
                const c0 = m[8] * m[13] - m[12] * m[9];

                const det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
                if (@abs(det) < eps) return null;
                const inv_det = 1 / det;

                return .{
                    .data = .{
                        (m[5] * c5 - m[6] * c4 + m[7] * c3) * inv_det,
                        (-m[1] * c5 + m[2] * c4 - m[3] * c3) * inv_det,
                        (m[13] * s5 - m[14] * s4 + m[15] * s3) * inv_det,
                        (-m[9] * s5 + m[10] * s4 - m[11] * s3) * inv_det,
                        (-m[4] * c5 + m[6] * c2 - m[7] * c1) * inv_det,
                        (m[0] * c5 - m[2] * c2 + m[3] * c1) * inv_det,
                        (-m[12] * s5 + m[14] * s2 - m[15] * s1) * inv_det,
                        (m[8] * s5 - m[10] * s2 + m[11] * s1) * inv_det,
                        (m[4] * c4 - m[5] * c2 + m[7] * c0) * inv_det,
                        (-m[0] * c4 + m[1] * c2 - m[3] * c0) * inv_det,
                        (m[12] * s4 - m[13] * s2 + m[15] * s0) * inv_det,
                        (-m[8] * s4 + m[9] * s2 - m[11] * s0) * inv_det,
                        (-m[4] * c3 + m[5] * c1 - m[6] * c0) * inv_det,
                        (m[0] * c3 - m[1] * c1 + m[2] * c0) * inv_det,
                        (-m[12] * s3 + m[13] * s1 - m[14] * s0) * inv_det,
                        (m[8] * s3 - m[9] * s1 + m[10] * s0) * inv_det,
                    },
                };
            } else {
                // General case: Use [A | I] -> RREF -> [I | A^-1]
                // Augment matrix with identity: [A | I]
                const augmented = self.horzcat(Self.identity());
                var row_data = splitRows(T, R, R * 2, augmented.data);

                // Apply reduced row echelon form
                lmao_reducedRowEchelonForm(T, R, R * 2, &row_data);

                // Check if left side is identity (matrix was invertible)
                // If any diagonal element is not ~1, matrix is singular
                for (0..R) |d| {
                    if (@abs(row_data[d][d] - 1) > eps) return null;
                }

                // Extract right half (the inverse)
                var result: Self = undefined;
                for (0..R) |r| {
                    for (0..R) |c| {
                        result.data[r * R + c] = row_data[r][R + c];
                    }
                }
                return result;
            }
        }

        // ========================================================================
        // Transformation Matrix Builders (for 4x4 matrices)
        // ========================================================================

        /// Creates a translation matrix.
        /// Only available for 4x4 matrices.
        pub fn translation(tx: T, ty: T, tz: T) Self {
            comptime {
                if (R != 4 or C != 4) @compileError("translation is only available for 4x4 matrices");
            }
            return .{
                .data = .{
                    1, 0, 0, tx,
                    0, 1, 0, ty,
                    0, 0, 1, tz,
                    0, 0, 0, 1,
                },
            };
        }

        /// Creates a uniform scaling matrix.
        /// Only available for 4x4 matrices.
        pub fn scaling(sx: T, sy: T, sz: T) Self {
            comptime {
                if (R != 4 or C != 4) @compileError("scaling is only available for 4x4 matrices");
            }
            return .{
                .data = .{
                    sx, 0,  0,  0,
                    0,  sy, 0,  0,
                    0,  0,  sz, 0,
                    0,  0,  0,  1,
                },
            };
        }

        /// Creates a rotation matrix around the X axis.
        /// Angle is in radians. Only available for 4x4 matrices.
        pub fn rotationX(angle: T) Self {
            comptime {
                if (R != 4 or C != 4) @compileError("rotationX is only available for 4x4 matrices");
                if (@typeInfo(T) != .float) @compileError("rotationX requires floating point type");
            }
            const c = @cos(angle);
            const s = @sin(angle);
            return .{
                .data = .{
                    1, 0, 0,  0,
                    0, c, -s, 0,
                    0, s, c,  0,
                    0, 0, 0,  1,
                },
            };
        }

        /// Creates a rotation matrix around the Y axis.
        /// Angle is in radians. Only available for 4x4 matrices.
        pub fn rotationY(angle: T) Self {
            comptime {
                if (R != 4 or C != 4) @compileError("rotationY is only available for 4x4 matrices");
                if (@typeInfo(T) != .float) @compileError("rotationY requires floating point type");
            }
            const c = @cos(angle);
            const s = @sin(angle);
            return .{
                .data = .{
                    c,  0, s, 0,
                    0,  1, 0, 0,
                    -s, 0, c, 0,
                    0,  0, 0, 1,
                },
            };
        }

        /// Creates a rotation matrix around the Z axis.
        /// Angle is in radians. Only available for 4x4 matrices.
        pub fn rotationZ(angle: T) Self {
            comptime {
                if (R != 4 or C != 4) @compileError("rotationZ is only available for 4x4 matrices");
                if (@typeInfo(T) != .float) @compileError("rotationZ requires floating point type");
            }
            const c = @cos(angle);
            const s = @sin(angle);
            return .{
                .data = .{
                    c, -s, 0, 0,
                    s, c,  0, 0,
                    0, 0,  1, 0,
                    0, 0,  0, 1,
                },
            };
        }

        /// Creates a look-at view matrix (camera looking from eye to target).
        /// Only available for 4x4 matrices.
        pub fn lookAt(eye: Matrix(T, 3, 1), target: Matrix(T, 3, 1), world_up: Matrix(T, 3, 1)) Self {
            comptime {
                if (R != 4 or C != 4) @compileError("lookAt is only available for 4x4 matrices");
                if (@typeInfo(T) != .float) @compileError("lookAt requires floating point type");
            }
            const forward = target.sub(eye).normalized();
            const right = forward.cross(world_up).normalized();
            const up = right.cross(forward);

            return .{
                .data = .{
                    right.data[0],    right.data[1],    right.data[2],    -right.dotProduct(eye),
                    up.data[0],       up.data[1],       up.data[2],       -up.dotProduct(eye),
                    -forward.data[0], -forward.data[1], -forward.data[2], forward.dotProduct(eye),
                    0,                0,                0,                1,
                },
            };
        }

        /// Transforms a 3D point (w=1) by this 4x4 matrix.
        /// Returns the transformed point (perspective division applied if w != 1).
        /// Only available for 4x4 matrices.
        pub fn transformPoint(self: Self, point: Matrix(T, 3, 1)) Matrix(T, 3, 1) {
            comptime {
                if (R != 4 or C != 4) @compileError("transformPoint is only available for 4x4 matrices");
            }
            const px = point.data[0];
            const py = point.data[1];
            const pz = point.data[2];
            const m = self.data;

            const w_comp = m[12] * px + m[13] * py + m[14] * pz + m[15];
            const inv_w = if (@abs(w_comp) > std.math.floatEps(T)) 1 / w_comp else 1;

            return .{
                .data = .{
                    (m[0] * px + m[1] * py + m[2] * pz + m[3]) * inv_w,
                    (m[4] * px + m[5] * py + m[6] * pz + m[7]) * inv_w,
                    (m[8] * px + m[9] * py + m[10] * pz + m[11]) * inv_w,
                },
            };
        }

        /// Transforms a 3D direction (w=0) by this 4x4 matrix.
        /// Returns the transformed direction (ignores translation).
        /// Only available for 4x4 matrices.
        pub fn transformDirection(self: Self, dir: Matrix(T, 3, 1)) Matrix(T, 3, 1) {
            comptime {
                if (R != 4 or C != 4) @compileError("transformDirection is only available for 4x4 matrices");
            }
            const dx = dir.data[0];
            const dy = dir.data[1];
            const dz = dir.data[2];
            const m = self.data;

            return .{
                .data = .{
                    m[0] * dx + m[1] * dy + m[2] * dz,
                    m[4] * dx + m[5] * dy + m[6] * dz,
                    m[8] * dx + m[9] * dy + m[10] * dz,
                },
            };
        }
    };
}

// Internal aliases to avoid name collision with Matrix methods
const lmao_rowEchelonForm = rowEchelonForm;
const lmao_reducedRowEchelonForm = reducedRowEchelonForm;
const lmao_horzcat = horzcat;
const lmao_identity = identity;

pub const Vec2u = Matrix(u32, 2, 1);
pub const Vec3u = Matrix(u32, 3, 1);
pub const Vec4u = Matrix(u32, 4, 1);

pub const Vec2i = Matrix(i32, 2, 1);
pub const Vec3i = Matrix(i32, 3, 1);
pub const Vec4i = Matrix(i32, 4, 1);

pub const Vec2f = Matrix(f32, 2, 1);
pub const Vec3f = Matrix(f32, 3, 1);
pub const Vec4f = Matrix(f32, 4, 1);

pub const Vec2d = Matrix(f64, 2, 1);
pub const Vec3d = Matrix(f64, 3, 1);
pub const Vec4d = Matrix(f64, 4, 1);

pub const Mat2f = Matrix(f32, 2, 2);
pub const Mat3f = Matrix(f32, 3, 3);
pub const Mat4f = Matrix(f32, 4, 4);

pub const Mat2d = Matrix(f64, 2, 2);
pub const Mat3d = Matrix(f64, 3, 3);
pub const Mat4d = Matrix(f64, 4, 4);
