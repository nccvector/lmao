pub fn MatrixX(comptime T: type, comptime R: usize, comptime C: usize) type {
    comptime {
        if (@typeInfo(T) != .float) @compileError("Matrix type must be a float type (f16/f32/f64/f128).");
        if (R == 0) @compileError("Matrix row count must be non-zero.");
        if (C == 0) @compileError("Matrix column count must be non-zero.");
    }

    return struct {
        data: [R * C]T,

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

fn mat4_dot_vec4(mat: *const [16]f32, vec: *const [4]f32, out: *[4]f32) void {
    out.* = Mat4f.fromArray(mat).dot(Vec4f.fromArray(vec)).toArray();
}

fn mat2_dot_mat2(matA: *const [4]f32, matB: *const [4]f32, out: *[4]f32) void {
    out.* = Mat2f.fromArray(matA).dot(Mat2f.fromArray(matB)).toArray();
}

fn mat3_dot_mat3(matA: *const [9]f32, matB: *const [9]f32, out: *[9]f32) void {
    out.* = Mat3f.fromArray(matA).dot(Mat3f.fromArray(matB)).toArray();
}

fn mat4_dot_mat4(matA: *const [16]f32, matB: *const [16]f32, out: *[16]f32) void {
    out.* = Mat4f.fromArray(matA).dot(Mat4f.fromArray(matB)).toArray();
}
