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

// // IMPORTANT: Pass matrix transpose (transposed RxC matrix is CxR with C rows of R elements each)
// pub fn dotRC1(comptime T: type, comptime R: usize, comptime C: usize, a: @Vector(R * C, T), b: @Vector(C, T)) @Vector(R, T) {
//     var out: @Vector(R, T) = @splat(0);
//     for (0..C) |i| {
//         // Stride: i * C → i * R (each row of transposed CxR matrix has R elements)
//         const av: @Vector(R, T) = loadVN(T, R, a[0..].ptr + i * R);
//         const vsp: @Vector(R, T) = @splat(b[i]);
//         out = mulAdd(T, R, av, vsp, out);
//     }
//     return out;
// }

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
        if (R > C) @compileError(std.fmt.comptimePrint("Number of rows must be less than or equal to number of columns (got R={}, C={})", .{ R, C }));
    }

    // Loop over diagonals
    for (0..R) |c| {
        var pivot: T = aug_eqs[c][c]; // Start with current diagonal as pivot
        var pivot_row: usize = c;
        // Find best pivot candidate in column values below diagonal
        for (c..R) |r| {
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
        for (c..R) |r| {
            if (r <= c) continue; // skip current row
            const bp: @Vector(C, T) = @splat(-aug_eqs[r][c]);
            aug_eqs[r] = @mulAdd(@Vector(C, T), bp, aug_eqs[c], aug_eqs[r]);
        }
    }
}

pub inline fn reducedRowEchelonForm(comptime T: type, comptime R: usize, comptime C: usize, aug_eqs: *[R]@Vector(C, T)) void {
    comptime {
        if (R > C) @compileError(std.fmt.comptimePrint("Number of rows must be less than or equal to number of columns (got R={}, C={})", .{ R, C }));
    }

    // Compute row echelon form (forward pass)
    rowEchelonForm(T, R, C, aug_eqs);

    // Compute reduced row echelon form (backwards pass)
    for (0..R) |c| {
        const r = R - c - 1; // Reverse loop (bottom-up, currently chosen pivot row)
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

pub fn MatrixX(comptime T: type, comptime R: usize, comptime C: usize) type {
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

        pub fn transpose(self: Self) MatrixX(T, C, R) {
            return .{ .data = transposeRCCR(T, R, C, self.data) };
        }

        pub fn dotSIMD(self: Self, other: anytype) MatrixX(T, R, @TypeOf(other).cols) {
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

            return MatrixX(T, R, K).fromArray(&dotRCK(T, R, C, K, self.data, other.data));

            // FUTURE WORK: Handle K == 1 case transpose using simd.interleave or shuffle instructions...
            // if (comptime K == 1) { // <--------- Benchmark both with and without this opt
            //     const a1: [R * C]T = @bitCast(self.transpose().data);
            //     const a2: [C * K]T = @bitCast(other.data);
            //     return MatrixX(T, R, 1).fromArray(&dotRC1(T, R, C, &a1, &a2));
            // } else {
            //     const a1: [R * C]T = @bitCast(self.data);
            //     const a2: [C * K]T = @bitCast(other.data);
            //     return MatrixX(T, R, K).fromArray(&dotRCK(T, R, C, K, &a1, &a2));
            // }
        }

        pub fn dot(self: Self, other: anytype) MatrixX(T, R, @TypeOf(other).cols) {
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
            var out: MatrixX(T, R, K) = undefined;

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
        pub fn sw(self: Self, comptime pattern: []const u8) MatrixX(T, pattern.len, 1) {
            comptime {
                if (!isVec()) @compileError("sw: only valid for vectors MatrixX(T, R, 1) where R=2..4");
            }

            var out: MatrixX(T, pattern.len, 1) = undefined;

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
    };
}

pub const Vec2f = MatrixX(f32, 2, 1);
pub const Vec3f = MatrixX(f32, 3, 1);
pub const Vec4f = MatrixX(f32, 4, 1);

pub const Mat2f = MatrixX(f32, 2, 2);
pub const Mat3f = MatrixX(f32, 3, 3);
pub const Mat4f = MatrixX(f32, 4, 4);

// fn row_echelon_form(mat: *[18]f32) void {
//     const v: @Vector(18, f32) = vectorFromArray(f32, 18, mat);
//     var rows = splitRows(f32, 3, 6, v);
//     rowEchelonForm(f32, 3, 6, &rows) catch {
//         return;
//     };
//     const o: @Vector(18, f32) = joinRows(f32, 3, 6, rows);
//     mat.* = arrayFromVector(f32, 18, o);
// }
//
// export fn reduced_row_echelon_form(mat: *[18]f32) void {
//     const v: @Vector(18, f32) = vectorFromArray(f32, 18, mat);
//     var rows = splitRows(f32, 3, 6, v);
//     reducedRowEchelonForm(f32, 3, 6, &rows) catch {
//         return;
//     };
//     const o: @Vector(18, f32) = joinRows(f32, 3, 6, rows);
//     mat.* = arrayFromVector(f32, 18, o);
// }

fn mat2_dot_vec2(mat: *const [4]f32, vec: *const [2]f32, out: *[2]f32) void {
    out.* = Mat2f.fromArray(mat).dot(Vec2f.fromArray(vec)).toArray();
}

fn mat2_dot_vec2_simd(mat: *const [4]f32, vec: *const [2]f32, out: *[2]f32) void {
    out.* = Mat2f.fromArray(mat).dotSIMD(Vec2f.fromArray(vec)).toArray();
}

fn mat3_dot_vec3(mat: *const [9]f32, vec: *const [3]f32, out: *[3]f32) void {
    out.* = Mat3f.fromArray(mat).dot(Vec3f.fromArray(vec)).toArray();
}

fn mat3_dot_vec3_simd(mat: *const [9]f32, vec: *const [3]f32, out: *[3]f32) void {
    out.* = Mat3f.fromArray(mat).dotSIMD(Vec3f.fromArray(vec)).toArray();
}

fn mat4_dot_vec4(mat: *const [16]f32, vec: *const [4]f32, out: *[4]f32) void {
    out.* = Mat4f.fromArray(mat).dot(Vec4f.fromArray(vec)).toArray();
}

fn mat4_dot_vec4_simd(mat: *const [16]f32, vec: *const [4]f32, out: *[4]f32) void {
    out.* = Mat4f.fromArray(mat).dotSIMD(Vec4f.fromArray(vec)).toArray();
}

fn mat2_dot_mat2(matA: *const [4]f32, matB: *const [4]f32, out: *[4]f32) void {
    out.* = Mat2f.fromArray(matA).dot(Mat2f.fromArray(matB)).toArray();
}

fn mat2_dot_mat2_simd(matA: *const [4]f32, matB: *const [4]f32, out: *[4]f32) void {
    out.* = Mat2f.fromArray(matA).dotSIMD(Mat2f.fromArray(matB)).toArray();
}

fn mat3_dot_mat3(matA: *const [9]f32, matB: *const [9]f32, out: *[9]f32) void {
    out.* = Mat3f.fromArray(matA).dot(Mat3f.fromArray(matB)).toArray();
}

fn mat3_dot_mat3_simd(matA: *const [9]f32, matB: *const [9]f32, out: *[9]f32) void {
    out.* = Mat3f.fromArray(matA).dotSIMD(Mat3f.fromArray(matB)).toArray();
}

fn mat4_dot_mat4(matA: *const [16]f32, matB: *const [16]f32, out: *[16]f32) void {
    out.* = Mat4f.fromArray(matA).dot(Mat4f.fromArray(matB)).toArray();
}

fn mat4_dot_mat4_simd(matA: *const [16]f32, matB: *const [16]f32, out: *[16]f32) void {
    out.* = Mat4f.fromArray(matA).dotSIMD(Mat4f.fromArray(matB)).toArray();
}

fn mat8_dot_vec8(mat: *const [64]f32, vec: *const [8]f32, out: *[8]f32) void {
    out.* = MatrixX(f32, 8, 8).fromArray(mat).dot(MatrixX(f32, 8, 1).fromArray(vec)).toArray();
}

fn mat8_dot_vec8_simd(mat: *const [64]f32, vec: *const [8]f32, out: *[8]f32) void {
    out.* = MatrixX(f32, 8, 8).fromArray(mat).dotSIMD(MatrixX(f32, 8, 1).fromArray(vec)).toArray();
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
