const simd = @import("simd");

fn loadVN(comptime T: type, comptime N: usize, ptr: [*]const T) @Vector(N, T) {
    var tmp: [N]f32 = undefined;
    inline for (0..N) |i| tmp[i] = ptr[i];
    return @bitCast(tmp);
}

fn storeVN(comptime T: type, comptime N: usize, ptr: [*]T, v: @Vector(N, T)) void {
    const tmp: [N]T = @bitCast(v);
    inline for (0..N) |i| ptr[i] = tmp[i];
}

// IMPORTANT: Pass matrix transpose
pub fn dotRC1(comptime T: type, comptime R: usize, comptime C: usize, a: *const [R * C]T, b: *const [C]T) [C]T {
    var out: @Vector(C, f32) = @splat(0);
    for (0..C) |i| {
        const av: @Vector(C, T) = loadVN(T, C, a[0..].ptr + i * C);
        const vsp: @Vector(C, T) = @splat(b[i]);
        out += av * vsp;
    }
    return out;
}

pub fn dotRCK(comptime T: type, comptime R: usize, comptime C: usize, comptime K: usize, a: *const [R * C]T, b: *const [C * K]T) [R * K]T {
    var out: [R * K]T = undefined;
    for (0..R) |row| {
        var acc: @Vector(K, T) = @splat(0.0);
        for (0..C) |col| {
            const av: @Vector(K, T) = @splat(a[row * C + col]);
            const bv: @Vector(K, T) = loadVN(T, K, b[0..].ptr + col * K);
            acc = @mulAdd(@Vector(K, T), av, bv, acc);
        }
        storeVN(T, K, out[0..].ptr + row * K, acc);
    }

    return out;
}

pub fn MatrixX(comptime T: type, comptime R: usize, comptime C: usize) type {
    comptime {
        if (@typeInfo(T) != .float) @compileError("Matrix type must be a float type (f16/f32/f64/f128).");
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
            return .{ .data = a.* };
        }

        pub fn toArray(self: Self) [R * C]T {
            return self.data;
        }

        pub fn add(self: Self, other: Self) Self {
            var out: Self = undefined;
            for (0..(R * C)) |i| {
                out.data[i] = self.data[i] + other.data[i];
            }
            return out;
        }

        pub fn sub(self: Self, other: Self) Self {
            var out: Self = undefined;
            for (0..(R * C)) |i| {
                out.data[i] = self.data[i] - other.data[i];
            }
            return out;
        }

        pub fn multiply(self: Self, other: Self) Self {
            var out: Self = undefined;
            for (0..(R * C)) |i| {
                out.data[i] = self.data[i] * other.data[i];
            }
            return out;
        }

        pub fn transpose(self: Self) Self {
            var out: Self = undefined;
            for (0..R) |r| {
                for (0..C) |c| {
                    const src = r * C + c;
                    const dst = c * R + r;
                    out.data[dst] = self.data[src];
                }
            }
            return out;
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

            if (comptime K == 1) { // <--------- Benchmark both with and without this opt
                const a1: [R * C]T = @bitCast(self.transpose().data);
                const a2: [C * K]T = @bitCast(other.data);
                return MatrixX(T, C, 1).fromArray(&dotRC1(T, R, C, &a1, &a2));
            } else {
                const a1: [R * C]T = @bitCast(self.data);
                const a2: [C * K]T = @bitCast(other.data);
                return MatrixX(T, R, K).fromArray(&dotRCK(T, R, C, K, &a1, &a2));
            }
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

fn mat2_dot_vec2(mat: *const [4]f32, vec: *const [2]f32, out: *[2]f32) void {
    out.* = Mat2f.fromArray(mat).dot(Vec2f.fromArray(vec)).toArray();
}

fn mat3_dot_vec3(mat: *const [9]f32, vec: *const [3]f32, out: *[3]f32) void {
    out.* = Mat3f.fromArray(mat).dot(Vec3f.fromArray(vec)).toArray();
}

export fn mat4_dot_vec4(mat: *const [16]f32, vec: *const [4]f32, out: *[4]f32) void {
    out.* = Mat4f.fromArray(mat).dot(Vec4f.fromArray(vec)).toArray();
}

export fn mat4_dot_vec4_simd(mat: *const [16]f32, vec: *const [4]f32, out: *[4]f32) void {
    out.* = Mat4f.fromArray(mat).dotSIMD(Vec4f.fromArray(vec)).toArray();
}

fn mat2_dot_mat2(matA: *const [4]f32, matB: *const [4]f32, out: *[4]f32) void {
    out.* = Mat2f.fromArray(matA).dot(Mat2f.fromArray(matB)).toArray();
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
