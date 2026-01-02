const std = @import("std");
const lmao = @import("lmao");

const Vec2f = lmao.Vec2f;
const Vec3f = lmao.Vec3f;
const Vec4f = lmao.Vec4f;
const Mat2f = lmao.Mat2f;
const Mat3f = lmao.Mat3f;
const Mat4f = lmao.Mat4f;

pub fn main() !void {
    std.debug.print("All your {s} are belong to us.\n", .{"codebase"});
}

test "rowEchelonForm" {
    const m: @Vector(18, f32) = .{
        1, 2, 3, 1, 0, 0,
        2, 5, 2, 0, 1, 0,
        3, 2, 9, 0, 0, 1,
    };
    std.debug.print("\n Input:\n", .{});
    lmao.debugPrintVector(f32, 3, 6, m);
    var m_rows = lmao.splitRows(f32, 3, 6, m);
    lmao.rowEchelonForm(f32, 3, 6, &m_rows);
    std.debug.print("\n Output:\n", .{});
    lmao.debugPrintVector(f32, 3, 6, lmao.joinRows(f32, 3, 6, m_rows));
}

test "reducedRowEchelonForm" {
    const m: @Vector(18, f32) = .{
        1, 2, 3, 1, 0, 0,
        2, 5, 2, 0, 1, 0,
        3, 2, 9, 0, 0, 1,
    };
    std.debug.print("\n Input:\n", .{});
    lmao.debugPrintVector(f32, 3, 6, m);
    var m_rows = lmao.splitRows(f32, 3, 6, m);
    lmao.reducedRowEchelonForm(f32, 3, 6, &m_rows);
    std.debug.print("\n Output:\n", .{});
    lmao.debugPrintVector(f32, 3, 6, lmao.joinRows(f32, 3, 6, m_rows));
}

// ============================================================================
// Vec2f Tests
// ============================================================================

test "Vec2f fromArray and toArray" {
    const v = Vec2f.fromArray(&.{ 1.0, 2.0 });
    const arr = v.toArray();
    try std.testing.expectEqual(@as(f32, 1.0), arr[0]);
    try std.testing.expectEqual(@as(f32, 2.0), arr[1]);
}

test "Vec2f add" {
    const a = Vec2f.fromArray(&.{ 1.0, 2.0 });
    const b = Vec2f.fromArray(&.{ 3.0, 4.0 });
    const c = a.add(b);
    try std.testing.expectEqual(@as(f32, 4.0), c.x());
    try std.testing.expectEqual(@as(f32, 6.0), c.y());
}

test "Vec2f sub" {
    const a = Vec2f.fromArray(&.{ 5.0, 7.0 });
    const b = Vec2f.fromArray(&.{ 2.0, 3.0 });
    const c = a.sub(b);
    try std.testing.expectEqual(@as(f32, 3.0), c.x());
    try std.testing.expectEqual(@as(f32, 4.0), c.y());
}

test "Vec2f multiply" {
    const a = Vec2f.fromArray(&.{ 2.0, 3.0 });
    const b = Vec2f.fromArray(&.{ 4.0, 5.0 });
    const c = a.multiply(b);
    try std.testing.expectEqual(@as(f32, 8.0), c.x());
    try std.testing.expectEqual(@as(f32, 15.0), c.y());
}

test "Vec2f swizzle" {
    const v = Vec2f.fromArray(&.{ 1.0, 2.0 });
    const yx = v.sw("yx");
    try std.testing.expectEqual(@as(f32, 2.0), yx.x());
    try std.testing.expectEqual(@as(f32, 1.0), yx.y());
}

// ============================================================================
// Vec3f Tests
// ============================================================================

test "Vec3f fromArray and toArray" {
    const v = Vec3f.fromArray(&.{ 1.0, 2.0, 3.0 });
    const arr = v.toArray();
    try std.testing.expectEqual(@as(f32, 1.0), arr[0]);
    try std.testing.expectEqual(@as(f32, 2.0), arr[1]);
    try std.testing.expectEqual(@as(f32, 3.0), arr[2]);
}

test "Vec3f add" {
    const a = Vec3f.fromArray(&.{ 1.0, 2.0, 3.0 });
    const b = Vec3f.fromArray(&.{ 4.0, 5.0, 6.0 });
    const c = a.add(b);
    try std.testing.expectEqual(@as(f32, 5.0), c.x());
    try std.testing.expectEqual(@as(f32, 7.0), c.y());
    try std.testing.expectEqual(@as(f32, 9.0), c.z());
}

test "Vec3f sub" {
    const a = Vec3f.fromArray(&.{ 10.0, 20.0, 30.0 });
    const b = Vec3f.fromArray(&.{ 1.0, 2.0, 3.0 });
    const c = a.sub(b);
    try std.testing.expectEqual(@as(f32, 9.0), c.x());
    try std.testing.expectEqual(@as(f32, 18.0), c.y());
    try std.testing.expectEqual(@as(f32, 27.0), c.z());
}

test "Vec3f multiply" {
    const a = Vec3f.fromArray(&.{ 2.0, 3.0, 4.0 });
    const b = Vec3f.fromArray(&.{ 5.0, 6.0, 7.0 });
    const c = a.multiply(b);
    try std.testing.expectEqual(@as(f32, 10.0), c.x());
    try std.testing.expectEqual(@as(f32, 18.0), c.y());
    try std.testing.expectEqual(@as(f32, 28.0), c.z());
}

test "Vec3f swizzle" {
    const v = Vec3f.fromArray(&.{ 1.0, 2.0, 3.0 });
    const zyx = v.sw("zyx");
    try std.testing.expectEqual(@as(f32, 3.0), zyx.x());
    try std.testing.expectEqual(@as(f32, 2.0), zyx.y());
    try std.testing.expectEqual(@as(f32, 1.0), zyx.z());
}

// ============================================================================
// Vec4f Tests
// ============================================================================

test "Vec4f fromArray and toArray" {
    const v = Vec4f.fromArray(&.{ 1.0, 2.0, 3.0, 4.0 });
    const arr = v.toArray();
    try std.testing.expectEqual(@as(f32, 1.0), arr[0]);
    try std.testing.expectEqual(@as(f32, 2.0), arr[1]);
    try std.testing.expectEqual(@as(f32, 3.0), arr[2]);
    try std.testing.expectEqual(@as(f32, 4.0), arr[3]);
}

test "Vec4f add" {
    const a = Vec4f.fromArray(&.{ 1.0, 2.0, 3.0, 4.0 });
    const b = Vec4f.fromArray(&.{ 5.0, 6.0, 7.0, 8.0 });
    const c = a.add(b);
    try std.testing.expectEqual(@as(f32, 6.0), c.x());
    try std.testing.expectEqual(@as(f32, 8.0), c.y());
    try std.testing.expectEqual(@as(f32, 10.0), c.z());
    try std.testing.expectEqual(@as(f32, 12.0), c.w());
}

test "Vec4f sub" {
    const a = Vec4f.fromArray(&.{ 10.0, 20.0, 30.0, 40.0 });
    const b = Vec4f.fromArray(&.{ 1.0, 2.0, 3.0, 4.0 });
    const c = a.sub(b);
    try std.testing.expectEqual(@as(f32, 9.0), c.x());
    try std.testing.expectEqual(@as(f32, 18.0), c.y());
    try std.testing.expectEqual(@as(f32, 27.0), c.z());
    try std.testing.expectEqual(@as(f32, 36.0), c.w());
}

test "Vec4f multiply" {
    const a = Vec4f.fromArray(&.{ 2.0, 3.0, 4.0, 5.0 });
    const b = Vec4f.fromArray(&.{ 6.0, 7.0, 8.0, 9.0 });
    const c = a.multiply(b);
    try std.testing.expectEqual(@as(f32, 12.0), c.x());
    try std.testing.expectEqual(@as(f32, 21.0), c.y());
    try std.testing.expectEqual(@as(f32, 32.0), c.z());
    try std.testing.expectEqual(@as(f32, 45.0), c.w());
}

test "Vec4f swizzle" {
    const v = Vec4f.fromArray(&.{ 1.0, 2.0, 3.0, 4.0 });
    const wzyx = v.sw("wzyx");
    try std.testing.expectEqual(@as(f32, 4.0), wzyx.x());
    try std.testing.expectEqual(@as(f32, 3.0), wzyx.y());
    try std.testing.expectEqual(@as(f32, 2.0), wzyx.z());
    try std.testing.expectEqual(@as(f32, 1.0), wzyx.w());
}

test "Vec4f swizzle to smaller vector" {
    const v = Vec4f.fromArray(&.{ 1.0, 2.0, 3.0, 4.0 });
    const xy = v.sw("xy");
    try std.testing.expectEqual(@as(f32, 1.0), xy.x());
    try std.testing.expectEqual(@as(f32, 2.0), xy.y());
}

// ============================================================================
// Mat2f Tests
// ============================================================================

test "Mat2f fromArray and toArray" {
    const m = Mat2f.fromArray(&.{ 1.0, 2.0, 3.0, 4.0 });
    const arr = m.toArray();
    try std.testing.expectEqual(@as(f32, 1.0), arr[0]);
    try std.testing.expectEqual(@as(f32, 2.0), arr[1]);
    try std.testing.expectEqual(@as(f32, 3.0), arr[2]);
    try std.testing.expectEqual(@as(f32, 4.0), arr[3]);
}

test "Mat2f add" {
    const a = Mat2f.fromArray(&.{ 1.0, 2.0, 3.0, 4.0 });
    const b = Mat2f.fromArray(&.{ 5.0, 6.0, 7.0, 8.0 });
    const c = a.add(b);
    const arr = c.toArray();
    try std.testing.expectEqual(@as(f32, 6.0), arr[0]);
    try std.testing.expectEqual(@as(f32, 8.0), arr[1]);
    try std.testing.expectEqual(@as(f32, 10.0), arr[2]);
    try std.testing.expectEqual(@as(f32, 12.0), arr[3]);
}

test "Mat2f transpose" {
    // Row-major: [1, 2]
    //            [3, 4]
    const m = Mat2f.fromArray(&.{ 1.0, 2.0, 3.0, 4.0 });
    const t = m.transpose();
    const arr = t.toArray();
    // Transposed: [1, 3]
    //             [2, 4]
    try std.testing.expectEqual(@as(f32, 1.0), arr[0]);
    try std.testing.expectEqual(@as(f32, 3.0), arr[1]);
    try std.testing.expectEqual(@as(f32, 2.0), arr[2]);
    try std.testing.expectEqual(@as(f32, 4.0), arr[3]);
}

// ============================================================================
// Mat3f Tests
// ============================================================================

test "Mat3f fromArray and toArray" {
    const m = Mat3f.fromArray(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    const arr = m.toArray();
    for (0..9) |i| {
        try std.testing.expectEqual(@as(f32, @floatFromInt(i + 1)), arr[i]);
    }
}

test "Mat3f add" {
    const a = Mat3f.fromArray(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    const b = Mat3f.fromArray(&.{ 9, 8, 7, 6, 5, 4, 3, 2, 1 });
    const c = a.add(b);
    const arr = c.toArray();
    for (arr) |val| {
        try std.testing.expectEqual(@as(f32, 10.0), val);
    }
}

test "Mat3f transpose" {
    // [1, 2, 3]
    // [4, 5, 6]
    // [7, 8, 9]
    const m = Mat3f.fromArray(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    const t = m.transpose();
    const arr = t.toArray();
    // [1, 4, 7]
    // [2, 5, 8]
    // [3, 6, 9]
    const expected = [_]f32{ 1, 4, 7, 2, 5, 8, 3, 6, 9 };
    for (0..9) |i| {
        try std.testing.expectEqual(expected[i], arr[i]);
    }
}

// ============================================================================
// Mat4f Tests
// ============================================================================

test "Mat4f fromArray and toArray" {
    var input: [16]f32 = undefined;
    for (0..16) |i| {
        input[i] = @floatFromInt(i + 1);
    }
    const m = Mat4f.fromArray(&input);
    const arr = m.toArray();
    for (0..16) |i| {
        try std.testing.expectEqual(@as(f32, @floatFromInt(i + 1)), arr[i]);
    }
}

test "Mat4f add" {
    var a_arr: [16]f32 = undefined;
    var b_arr: [16]f32 = undefined;
    for (0..16) |i| {
        a_arr[i] = @floatFromInt(i);
        b_arr[i] = @floatFromInt(16 - i);
    }
    const a = Mat4f.fromArray(&a_arr);
    const b = Mat4f.fromArray(&b_arr);
    const c = a.add(b);
    const arr = c.toArray();
    for (arr) |val| {
        try std.testing.expectEqual(@as(f32, 16.0), val);
    }
}

test "Mat4f identity transpose" {
    // Identity matrix should equal its transpose
    const identity = Mat4f.fromArray(&.{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    });
    const t = identity.transpose();
    const arr = t.toArray();
    const expected = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    for (0..16) |i| {
        try std.testing.expectEqual(expected[i], arr[i]);
    }
}

// ============================================================================
// Dot Product Tests
// ============================================================================

test "Mat2f dot Vec2f" {
    // [1, 2] * [5]   = [1*5 + 2*6]   = [17]
    // [3, 4]   [6]     [3*5 + 4*6]     [39]
    const m = Mat2f.fromArray(&.{ 1, 2, 3, 4 });
    const v = Vec2f.fromArray(&.{ 5, 6 });
    const result = m.dot(v);
    try std.testing.expectEqual(@as(f32, 17.0), result.x());
    try std.testing.expectEqual(@as(f32, 39.0), result.y());
}

test "Mat3f dot Vec3f" {
    // [1, 0, 0]   [2]   [2]
    // [0, 1, 0] * [3] = [3]
    // [0, 0, 1]   [4]   [4]
    const identity = Mat3f.fromArray(&.{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    });
    const v = Vec3f.fromArray(&.{ 2, 3, 4 });
    const result = identity.dot(v);
    try std.testing.expectEqual(@as(f32, 2.0), result.x());
    try std.testing.expectEqual(@as(f32, 3.0), result.y());
    try std.testing.expectEqual(@as(f32, 4.0), result.z());
}

test "Mat4f dot Vec4f" {
    // Scale matrix (2x)
    const scale = Mat4f.fromArray(&.{
        2, 0, 0, 0,
        0, 2, 0, 0,
        0, 0, 2, 0,
        0, 0, 0, 1,
    });
    const v = Vec4f.fromArray(&.{ 1, 2, 3, 1 });
    const result = scale.dot(v);
    try std.testing.expectEqual(@as(f32, 2.0), result.x());
    try std.testing.expectEqual(@as(f32, 4.0), result.y());
    try std.testing.expectEqual(@as(f32, 6.0), result.z());
    try std.testing.expectEqual(@as(f32, 1.0), result.w());
}

test "Mat2f dot Mat2f" {
    // [1, 2] * [5, 6]   = [1*5+2*7, 1*6+2*8]   = [19, 22]
    // [3, 4]   [7, 8]     [3*5+4*7, 3*6+4*8]     [43, 50]
    const a = Mat2f.fromArray(&.{ 1, 2, 3, 4 });
    const b = Mat2f.fromArray(&.{ 5, 6, 7, 8 });
    const result = a.dot(b);
    const arr = result.toArray();
    try std.testing.expectEqual(@as(f32, 19.0), arr[0]);
    try std.testing.expectEqual(@as(f32, 22.0), arr[1]);
    try std.testing.expectEqual(@as(f32, 43.0), arr[2]);
    try std.testing.expectEqual(@as(f32, 50.0), arr[3]);
}

test "Mat3f dot Mat3f identity" {
    const identity = Mat3f.fromArray(&.{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    });
    const m = Mat3f.fromArray(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    const result = identity.dot(m);
    const arr = result.toArray();
    for (0..9) |i| {
        try std.testing.expectEqual(@as(f32, @floatFromInt(i + 1)), arr[i]);
    }
}

test "Mat4f dot Mat4f identity" {
    const identity = Mat4f.fromArray(&.{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    });
    var m_arr: [16]f32 = undefined;
    for (0..16) |i| {
        m_arr[i] = @floatFromInt(i + 1);
    }
    const m = Mat4f.fromArray(&m_arr);
    const result = identity.dot(m);
    const arr = result.toArray();
    for (0..16) |i| {
        try std.testing.expectEqual(@as(f32, @floatFromInt(i + 1)), arr[i]);
    }
}

// ============================================================================
// Comprehensive Dot Product Tests (with random values from NumPy)
// ============================================================================

const epsilon: f32 = 1e-4;

fn expectApproxEqual(expected: f32, actual: f32) !void {
    const diff = @abs(expected - actual);
    if (diff > epsilon) {
        std.debug.print("Expected: {d}, Actual: {d}, Diff: {d}\n", .{ expected, actual, diff });
        return error.TestExpectedApproxEqual;
    }
}

fn expectArrayApproxEqual(comptime N: usize, expected: [N]f32, actual: [N]f32) !void {
    for (0..N) |i| {
        try expectApproxEqual(expected[i], actual[i]);
    }
}

// ----------------------------------------------------------------------------
// Square Matrix-Vector Products (dot and dotSIMD)
// ----------------------------------------------------------------------------

test "Mat2f dot Vec2f - random values" {
    const m = Mat2f.fromArray(&.{ -2.509198, 9.014286, 4.639879, 1.973170 });
    const v = Vec2f.fromArray(&.{ -6.879627, -6.880110 });
    const expected = [_]f32{ -44.756935, -45.496262 };

    const result = m.dot(v);
    try expectArrayApproxEqual(2, expected, result.toArray());
}

test "Mat2f dotSIMD Vec2f - random values" {
    const m = Mat2f.fromArray(&.{ -2.509198, 9.014286, 4.639879, 1.973170 });
    const v = Vec2f.fromArray(&.{ -6.879627, -6.880110 });
    const expected = [_]f32{ -44.756935, -45.496262 };

    const result = m.dotSIMD(v);
    try expectArrayApproxEqual(2, expected, result.toArray());
}

test "Mat3f dot Vec3f - random values" {
    const m = Mat3f.fromArray(&.{ -8.838327, 7.323523, 2.022300, 4.161451, -9.588310, 9.398197, 6.648853, -5.753218, -6.363501 });
    const v = Vec3f.fromArray(&.{ -6.331910, -3.915155, 0.495129 });
    const expected = [_]f32{ 28.292059, 15.843105, -22.725945 };

    const result = m.dot(v);
    try expectArrayApproxEqual(3, expected, result.toArray());
}

test "Mat3f dotSIMD Vec3f - random values" {
    const m = Mat3f.fromArray(&.{ -8.838327, 7.323523, 2.022300, 4.161451, -9.588310, 9.398197, 6.648853, -5.753218, -6.363501 });
    const v = Vec3f.fromArray(&.{ -6.331910, -3.915155, 0.495129 });
    const expected = [_]f32{ 28.292059, 15.843105, -22.725945 };

    const result = m.dotSIMD(v);
    try expectArrayApproxEqual(3, expected, result.toArray());
}

test "Mat4f dot Vec4f - random values" {
    const m = Mat4f.fromArray(&.{ -1.361100, -4.175417, 2.237058, -7.210123, -4.157107, -2.672763, -0.878600, 5.703519, -6.006525, 0.284689, 1.848291, -9.070992, 2.150897, -6.589518, -8.698968, 8.977711 });
    const v = Vec4f.fromArray(&.{ 9.312640, 6.167947, -3.907725, -8.046557 });
    const expected = [_]f32{ 10.845680, -97.659470, 11.586983, -58.859749 };

    const result = m.dot(v);
    try expectArrayApproxEqual(4, expected, result.toArray());
}

test "Mat4f dotSIMD Vec4f - random values" {
    const m = Mat4f.fromArray(&.{ -1.361100, -4.175417, 2.237058, -7.210123, -4.157107, -2.672763, -0.878600, 5.703519, -6.006525, 0.284689, 1.848291, -9.070992, 2.150897, -6.589518, -8.698968, 8.977711 });
    const v = Vec4f.fromArray(&.{ 9.312640, 6.167947, -3.907725, -8.046557 });
    const expected = [_]f32{ 10.845680, -97.659470, 11.586983, -58.859749 };

    const result = m.dotSIMD(v);
    try expectArrayApproxEqual(4, expected, result.toArray());
}

// ----------------------------------------------------------------------------
// Rectangular Matrix-Vector Products (dot and dotSIMD)
// ----------------------------------------------------------------------------

const Mat2x3f = lmao.MatrixX(f32, 2, 3);
const Mat2x4f = lmao.MatrixX(f32, 2, 4);
const Mat3x2f = lmao.MatrixX(f32, 3, 2);
const Mat3x4f = lmao.MatrixX(f32, 3, 4);
const Mat4x2f = lmao.MatrixX(f32, 4, 2);
const Mat4x3f = lmao.MatrixX(f32, 4, 3);

test "Mat2x3 dot Vec3 -> Vec2" {
    const m = Mat2x3f.fromArray(&.{ 3.684660, -1.196950, -7.559235, -0.096462, -9.312229, 8.186408 });
    const v = Vec3f.fromArray(&.{ -4.824400, 3.250446, -3.765779 });
    const expected = [_]f32{ 6.799507, -60.631721 };

    const result = m.dot(v);
    try expectArrayApproxEqual(2, expected, result.toArray());
}

test "Mat2x3 dotSIMD Vec3 -> Vec2" {
    const m = Mat2x3f.fromArray(&.{ 3.684660, -1.196950, -7.559235, -0.096462, -9.312229, 8.186408 });
    const v = Vec3f.fromArray(&.{ -4.824400, 3.250446, -3.765779 });
    const expected = [_]f32{ 6.799507, -60.631721 };

    const result = m.dotSIMD(v);
    try expectArrayApproxEqual(2, expected, result.toArray());
}

test "Mat2x4 dot Vec4 -> Vec2" {
    const m = Mat2x4f.fromArray(&.{ 0.401360, 0.934206, -6.302911, 9.391692, 5.502656, 8.789979, 7.896547, 1.958000 });
    const v = Vec4f.fromArray(&.{ 8.437485, -8.230150, -6.080343, -9.095454 });
    const expected = [_]f32{ -51.400032, -91.736877 };

    const result = m.dot(v);
    try expectArrayApproxEqual(2, expected, result.toArray());
}

test "Mat2x4 dotSIMD Vec4 -> Vec2" {
    const m = Mat2x4f.fromArray(&.{ 0.401360, 0.934206, -6.302911, 9.391692, 5.502656, 8.789979, 7.896547, 1.958000 });
    const v = Vec4f.fromArray(&.{ 8.437485, -8.230150, -6.080343, -9.095454 });
    const expected = [_]f32{ -51.400032, -91.736877 };

    const result = m.dotSIMD(v);
    try expectArrayApproxEqual(2, expected, result.toArray());
}

test "Mat3x2 dot Vec2 -> Vec3" {
    const m = Mat3x2f.fromArray(&.{ -3.493393, -2.226454, -4.573020, 6.574750, -2.864933, -4.381310 });
    const v = Vec2f.fromArray(&.{ 0.853922, -7.181516 });
    const expected = [_]f32{ 13.006231, -51.121670, 29.018019 };

    const result = m.dot(v);
    try expectArrayApproxEqual(3, expected, result.toArray());
}

test "Mat3x2 dotSIMD Vec2 -> Vec3" {
    const m = Mat3x2f.fromArray(&.{ -3.493393, -2.226454, -4.573020, 6.574750, -2.864933, -4.381310 });
    const v = Vec2f.fromArray(&.{ 0.853922, -7.181516 });
    const expected = [_]f32{ 13.006231, -51.121670, 29.018019 };

    const result = m.dotSIMD(v);
    try expectArrayApproxEqual(3, expected, result.toArray());
}

test "Mat3x4 dot Vec4 -> Vec3" {
    const m = Mat3x4f.fromArray(&.{ 6.043940, -8.508987, 9.737739, 5.444895, -6.025686, -9.889558, 6.309228, 4.137147, 4.580143, 5.425407, -8.519107, -2.830685 });
    const v = Vec4f.fromArray(&.{ -7.682619, 7.262069, 2.465963, -3.382040 });
    const expected = [_]f32{ -102.628082, -23.959274, -7.222130 };

    const result = m.dot(v);
    try expectArrayApproxEqual(3, expected, result.toArray());
}

test "Mat3x4 dotSIMD Vec4 -> Vec3" {
    const m = Mat3x4f.fromArray(&.{ 6.043940, -8.508987, 9.737739, 5.444895, -6.025686, -9.889558, 6.309228, 4.137147, 4.580143, 5.425407, -8.519107, -2.830685 });
    const v = Vec4f.fromArray(&.{ -7.682619, 7.262069, 2.465963, -3.382040 });
    const expected = [_]f32{ -102.628082, -23.959274, -7.222130 };

    const result = m.dotSIMD(v);
    try expectArrayApproxEqual(3, expected, result.toArray());
}

test "Mat4x2 dot Vec2 -> Vec4" {
    const m = Mat4x2f.fromArray(&.{ -8.728833, -3.780354, -3.496334, 4.592124, 2.751149, 7.744255, -0.555701, -7.608115 });
    const v = Vec2f.fromArray(&.{ 4.264896, 5.215701 });
    const expected = [_]f32{ -56.944759, 9.039644, 52.125084, -42.051666 };

    const result = m.dot(v);
    try expectArrayApproxEqual(4, expected, result.toArray());
}

test "Mat4x2 dotSIMD Vec2 -> Vec4" {
    const m = Mat4x2f.fromArray(&.{ -8.728833, -3.780354, -3.496334, 4.592124, 2.751149, 7.744255, -0.555701, -7.608115 });
    const v = Vec2f.fromArray(&.{ 4.264896, 5.215701 });
    const expected = [_]f32{ -56.944759, 9.039644, 52.125084, -42.051666 };

    const result = m.dotSIMD(v);
    try expectArrayApproxEqual(4, expected, result.toArray());
}

test "Mat4x3 dot Vec3 -> Vec4" {
    const m = Mat4x3f.fromArray(&.{ 1.225544, 5.419343, -0.124088, 0.454657, -1.449180, -9.491617, -7.842172, -9.371416, 2.728208, -3.712880, 0.171414, 8.151329 });
    const v = Vec3f.fromArray(&.{ -5.014155, -1.792342, 5.111023 });
    const expected = [_]f32{ -16.492599, -48.194168, 70.062584, 59.971355 };

    const result = m.dot(v);
    try expectArrayApproxEqual(4, expected, result.toArray());
}

test "Mat4x3 dotSIMD Vec3 -> Vec4" {
    const m = Mat4x3f.fromArray(&.{ 1.225544, 5.419343, -0.124088, 0.454657, -1.449180, -9.491617, -7.842172, -9.371416, 2.728208, -3.712880, 0.171414, 8.151329 });
    const v = Vec3f.fromArray(&.{ -5.014155, -1.792342, 5.111023 });
    const expected = [_]f32{ -16.492599, -48.194168, 70.062584, 59.971355 };

    const result = m.dotSIMD(v);
    try expectArrayApproxEqual(4, expected, result.toArray());
}

// ----------------------------------------------------------------------------
// Square Matrix-Matrix Products (dot and dotSIMD)
// ----------------------------------------------------------------------------

test "Mat2f dot Mat2f - random values" {
    const a = Mat2f.fromArray(&.{ -5.424037, -8.460402, -4.204971, -6.775574 });
    const b = Mat2f.fromArray(&.{ 8.593953, 6.162407, 2.668075, 7.429212 });
    const expected = [_]f32{ -69.186905, -96.279236, -54.215065, -76.249916 };

    const result = a.dot(b);
    try expectArrayApproxEqual(4, expected, result.toArray());
}

test "Mat2f dotSIMD Mat2f - random values" {
    const a = Mat2f.fromArray(&.{ -5.424037, -8.460402, -4.204971, -6.775574 });
    const b = Mat2f.fromArray(&.{ 8.593953, 6.162407, 2.668075, 7.429212 });
    const expected = [_]f32{ -69.186905, -96.279236, -54.215065, -76.249916 };

    const result = a.dotSIMD(b);
    try expectArrayApproxEqual(4, expected, result.toArray());
}

test "Mat3f dot Mat3f - random values" {
    const a = Mat3f.fromArray(&.{ 6.073442, -6.268599, 7.851180, 0.786845, 6.148803, 7.921826, -3.639930, -7.798962, -5.441297 });
    const b = Mat3f.fromArray(&.{ -1.457844, 6.360295, 7.214612, -9.860957, 0.214946, -1.651780, -5.557844, -7.602693, -3.247697 });
    const expected = [_]f32{ 9.324623, -22.408642, 28.673616, -105.808456, -53.900982, -30.207378, 112.453552, 16.541115, 4.293163 };

    const result = a.dot(b);
    try expectArrayApproxEqual(9, expected, result.toArray());
}

test "Mat3f dotSIMD Mat3f - random values" {
    const a = Mat3f.fromArray(&.{ 6.073442, -6.268599, 7.851180, 0.786845, 6.148803, 7.921826, -3.639930, -7.798962, -5.441297 });
    const b = Mat3f.fromArray(&.{ -1.457844, 6.360295, 7.214612, -9.860957, 0.214946, -1.651780, -5.557844, -7.602693, -3.247697 });
    const expected = [_]f32{ 9.324623, -22.408642, 28.673616, -105.808456, -53.900982, -30.207378, 112.453552, 16.541115, 4.293163 };

    const result = a.dotSIMD(b);
    try expectArrayApproxEqual(9, expected, result.toArray());
}

test "Mat4f dot Mat4f - random values" {
    const a = Mat4f.fromArray(&.{ 8.858194, -3.535941, 0.375812, 4.060379, -2.727408, 9.435641, 9.248946, -4.964354, -0.055030, -3.982434, -4.303190, -9.262261, 2.191287, 0.053580, -8.970425, -4.427071 });
    const b = Mat4f.fromArray(&.{ 8.165318, -5.208762, -7.102103, -0.210945, 9.713009, -5.158895, 3.442711, 5.232392, -5.247249, 4.564327, -2.644337, 2.646116, 2.670594, 0.715494, -8.194204, 6.706050 });
    const expected = [_]f32{ 46.856983, -23.278173, -109.350380, 7.853527, 7.589019, 4.192191, 68.076111, 41.128891, -41.286583, -5.436661, 73.956406, -94.325981, 53.660122, -55.801796, 44.618870, -53.606831 };

    const result = a.dot(b);
    try expectArrayApproxEqual(16, expected, result.toArray());
}

test "Mat4f dotSIMD Mat4f - random values" {
    const a = Mat4f.fromArray(&.{ 8.858194, -3.535941, 0.375812, 4.060379, -2.727408, 9.435641, 9.248946, -4.964354, -0.055030, -3.982434, -4.303190, -9.262261, 2.191287, 0.053580, -8.970425, -4.427071 });
    const b = Mat4f.fromArray(&.{ 8.165318, -5.208762, -7.102103, -0.210945, 9.713009, -5.158895, 3.442711, 5.232392, -5.247249, 4.564327, -2.644337, 2.646116, 2.670594, 0.715494, -8.194204, 6.706050 });
    const expected = [_]f32{ 46.856983, -23.278173, -109.350380, 7.853527, 7.589019, 4.192191, 68.076111, 41.128891, -41.286583, -5.436661, 73.956406, -94.325981, 53.660122, -55.801796, 44.618870, -53.606831 };

    const result = a.dotSIMD(b);
    try expectArrayApproxEqual(16, expected, result.toArray());
}

// ----------------------------------------------------------------------------
// Rectangular Matrix-Matrix Products (dot and dotSIMD)
// ----------------------------------------------------------------------------

test "Mat2x3 dot Mat3x2 -> Mat2x2" {
    const a = Mat2x3f.fromArray(&.{ -3.584399, -6.269630, -9.184497, 1.817859, 3.551287, -9.668243 });
    const b = Mat3x2f.fromArray(&.{ 0.241861, -5.470085, 2.903456, -6.512671, 3.818755, -2.265293 });
    const expected = [_]f32{ -54.143860, 81.244583, -26.169975, -11.170803 };

    const result = a.dot(b);
    try expectArrayApproxEqual(4, expected, result.toArray());
}

test "Mat2x3 dotSIMD Mat3x2 -> Mat2x2" {
    const a = Mat2x3f.fromArray(&.{ -3.584399, -6.269630, -9.184497, 1.817859, 3.551287, -9.668243 });
    const b = Mat3x2f.fromArray(&.{ 0.241861, -5.470085, 2.903456, -6.512671, 3.818755, -2.265293 });
    const expected = [_]f32{ -54.143860, 81.244583, -26.169975, -11.170803 };

    const result = a.dotSIMD(b);
    try expectArrayApproxEqual(4, expected, result.toArray());
}

test "Mat2x4 dot Mat4x3 -> Mat2x3" {
    const a = Mat2x4f.fromArray(&.{ 8.734600, -7.249581, -3.178673, -7.730530, 8.493873, 7.546787, -4.841167, 3.199681 });
    const b = Mat4x3f.fromArray(&.{ 6.344444, 1.104016, 0.593012, -5.162954, -8.137944, 7.944315, 8.008361, 2.662029, -3.219404, -3.015809, 4.519114, 7.942205 });
    const expected = [_]f32{ 90.703270, 25.242966, -103.577255, -33.494263, -50.465569, 105.989212 };

    const result = a.dot(b);
    try expectArrayApproxEqual(6, expected, result.toArray());
}

test "Mat2x4 dotSIMD Mat4x3 -> Mat2x3" {
    const a = Mat2x4f.fromArray(&.{ 8.734600, -7.249581, -3.178673, -7.730530, 8.493873, 7.546787, -4.841167, 3.199681 });
    const b = Mat4x3f.fromArray(&.{ 6.344444, 1.104016, 0.593012, -5.162954, -8.137944, 7.944315, 8.008361, 2.662029, -3.219404, -3.015809, 4.519114, 7.942205 });
    const expected = [_]f32{ 90.703270, 25.242966, -103.577255, -33.494263, -50.465569, 105.989212 };

    const result = a.dotSIMD(b);
    try expectArrayApproxEqual(6, expected, result.toArray());
}

test "Mat3x4 dot Mat4x2 -> Mat3x2" {
    const a = Mat3x4f.fromArray(&.{ 7.741728, 5.597511, 2.840633, -8.317201, -6.767426, 7.971084, 2.128581, -9.816059, -7.970569, 3.270035, -9.898768, -6.783839 });
    const b = Mat4x2f.fromArray(&.{ 0.974676, 3.837904, 3.039225, -5.514614, 4.243585, -5.255018, -3.492006, 4.929828 });
    const expected = [_]f32{ 65.655952, -57.086044, 60.940422, -129.507385, -16.147404, -30.048212 };

    const result = a.dot(b);
    try expectArrayApproxEqual(6, expected, result.toArray());
}

test "Mat3x4 dotSIMD Mat4x2 -> Mat3x2" {
    const a = Mat3x4f.fromArray(&.{ 7.741728, 5.597511, 2.840633, -8.317201, -6.767426, 7.971084, 2.128581, -9.816059, -7.970569, 3.270035, -9.898768, -6.783839 });
    const b = Mat4x2f.fromArray(&.{ 0.974676, 3.837904, 3.039225, -5.514614, 4.243585, -5.255018, -3.492006, 4.929828 });
    const expected = [_]f32{ 65.655952, -57.086044, 60.940422, -129.507385, -16.147404, -30.048212 };

    const result = a.dotSIMD(b);
    try expectArrayApproxEqual(6, expected, result.toArray());
}

test "Mat4x2 dot Mat2x3 -> Mat4x3" {
    const a = Mat4x2f.fromArray(&.{ 2.992658, 6.984468, 3.152258, 1.366172, -8.126505, -2.645684, -4.695952, -5.120207 });
    const b = Mat2x3f.fromArray(&.{ 9.460211, -2.138046, 7.840931, 2.622772, 5.896226, 0.052742 });
    const expected = [_]f32{ 46.629845, 34.783562, 23.833597, 33.404182, 1.315588, 24.788691, -83.817474, 1.775287, -63.858902, -57.853836, -20.149738, -37.090687 };

    const result = a.dot(b);
    try expectArrayApproxEqual(12, expected, result.toArray());
}

test "Mat4x2 dotSIMD Mat2x3 -> Mat4x3" {
    const a = Mat4x2f.fromArray(&.{ 2.992658, 6.984468, 3.152258, 1.366172, -8.126505, -2.645684, -4.695952, -5.120207 });
    const b = Mat2x3f.fromArray(&.{ 9.460211, -2.138046, 7.840931, 2.622772, 5.896226, 0.052742 });
    const expected = [_]f32{ 46.629845, 34.783562, 23.833597, 33.404182, 1.315588, 24.788691, -83.817474, 1.775287, -63.858902, -57.853836, -20.149738, -37.090687 };

    const result = a.dotSIMD(b);
    try expectArrayApproxEqual(12, expected, result.toArray());
}

test "Mat3x2 dot Mat2x4 -> Mat3x4" {
    const a = Mat3x2f.fromArray(&.{ 1.538078, -0.149646, -6.095140, 4.449042, -4.384553, -9.513680 });
    const b = Mat2x4f.fromArray(&.{ 2.909446, -6.457787, 8.809172, 9.078571, 8.297288, -2.596826, -9.690867, 8.566371 });
    const expected = [_]f32{ 3.233297, -9.543972, 14.999392, 12.681624, 19.181503, 27.807726, -96.808220, -17.223021, -91.694366, 53.019878, 53.571537, -121.303192 };

    const result = a.dot(b);
    try expectArrayApproxEqual(12, expected, result.toArray());
}

test "Mat3x2 dotSIMD Mat2x4 -> Mat3x4" {
    const a = Mat3x2f.fromArray(&.{ 1.538078, -0.149646, -6.095140, 4.449042, -4.384553, -9.513680 });
    const b = Mat2x4f.fromArray(&.{ 2.909446, -6.457787, 8.809172, 9.078571, 8.297288, -2.596826, -9.690867, 8.566371 });
    const expected = [_]f32{ 3.233297, -9.543972, 14.999392, 12.681624, 19.181503, 27.807726, -96.808220, -17.223021, -91.694366, 53.019878, 53.571537, -121.303192 };

    const result = a.dotSIMD(b);
    try expectArrayApproxEqual(12, expected, result.toArray());
}

test "Mat4x3 dot Mat3x4 -> Mat4x4" {
    const a = Mat4x3f.fromArray(&.{ -1.436317, 9.333097, 9.272400, 7.060189, -4.111022, -2.298045, 7.022733, -3.661560, -6.610145, 1.136025, 8.723096, 3.920596 });
    const b = Mat3x4f.fromArray(&.{ 1.401223, -8.056470, 2.300144, 9.801077, -7.198320, 0.366593, 7.547462, 4.815372, 3.940315, 4.049682, -2.810177, -4.128163 });
    const expected = [_]f32{ -32.659039, 52.543362, 41.080364, -7.413100, 30.430334, -67.693626, -8.330412, 58.888062, 10.151446, -84.689720, 7.093498, 78.486328, -45.751427, 9.922639, 57.432686, 36.954365 };

    const result = a.dot(b);
    try expectArrayApproxEqual(16, expected, result.toArray());
}

test "Mat4x3 dotSIMD Mat3x4 -> Mat4x4" {
    const a = Mat4x3f.fromArray(&.{ -1.436317, 9.333097, 9.272400, 7.060189, -4.111022, -2.298045, 7.022733, -3.661560, -6.610145, 1.136025, 8.723096, 3.920596 });
    const b = Mat3x4f.fromArray(&.{ 1.401223, -8.056470, 2.300144, 9.801077, -7.198320, 0.366593, 7.547462, 4.815372, 3.940315, 4.049682, -2.810177, -4.128163 });
    const expected = [_]f32{ -32.659039, 52.543362, 41.080364, -7.413100, 30.430334, -67.693626, -8.330412, 58.888062, 10.151446, -84.689720, 7.093498, 78.486328, -45.751427, 9.922639, 57.432686, 36.954365 };

    const result = a.dotSIMD(b);
    try expectArrayApproxEqual(16, expected, result.toArray());
}

// ============================================================================
// Row Echelon Form Tests (with ground truth from numpy)
// ============================================================================

// Helper to check row echelon form properties
fn checkRowEchelonForm(comptime R: usize, comptime C: usize, rows: [R]@Vector(C, f32)) !void {
    const tol: f32 = 1e-4;
    var prev_pivot_col: ?usize = null;

    for (0..R) |r| {
        // Find pivot column (first non-zero entry)
        var pivot_col: ?usize = null;
        for (0..C) |c| {
            const arr: [C]f32 = rows[r];
            if (@abs(arr[c]) > tol) {
                pivot_col = c;
                break;
            }
        }

        if (pivot_col) |pc| {
            // Pivot should be 1
            const arr: [C]f32 = rows[r];
            try expectApproxEqual(1.0, arr[pc]);

            // Pivot should be to the right of the previous pivot
            if (prev_pivot_col) |ppc| {
                if (pc <= ppc) {
                    std.debug.print("Pivot at row {} col {} should be > previous pivot col {}\n", .{ r, pc, ppc });
                    return error.InvalidRowEchelonForm;
                }
            }

            // All entries below this pivot should be 0
            for (r + 1..R) |below_r| {
                const below_arr: [C]f32 = rows[below_r];
                if (@abs(below_arr[pc]) > tol) {
                    std.debug.print("Entry at ({}, {}) should be 0, got {}\n", .{ below_r, pc, below_arr[pc] });
                    return error.InvalidRowEchelonForm;
                }
            }

            prev_pivot_col = pc;
        }
    }
}

// Helper to check reduced row echelon form properties
fn checkReducedRowEchelonForm(comptime R: usize, comptime C: usize, rows: [R]@Vector(C, f32)) !void {
    const tol: f32 = 1e-4;

    // First check it's a valid REF
    try checkRowEchelonForm(R, C, rows);

    // Additionally check that all entries above pivots are 0
    for (0..R) |r| {
        // Find pivot column
        var pivot_col: ?usize = null;
        for (0..C) |c| {
            const arr: [C]f32 = rows[r];
            if (@abs(arr[c]) > tol) {
                pivot_col = c;
                break;
            }
        }

        if (pivot_col) |pc| {
            // All entries above this pivot should be 0
            for (0..r) |above_r| {
                const above_arr: [C]f32 = rows[above_r];
                if (@abs(above_arr[pc]) > tol) {
                    std.debug.print("RREF: Entry at ({}, {}) should be 0, got {}\n", .{ above_r, pc, above_arr[pc] });
                    return error.InvalidReducedRowEchelonForm;
                }
            }
        }
    }
}

test "rowEchelonForm - 3x3 matrix" {
    const input: @Vector(9, f32) = .{ 2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0 };
    const expected: @Vector(9, f32) = .{ 1.0, 0.333333, -0.666667, 0.0, 1.0, 0.4, 0.0, 0.0, 1.0 };

    var rows = lmao.splitRows(f32, 3, 3, input);
    lmao.rowEchelonForm(f32, 3, 3, &rows);
    const result = lmao.joinRows(f32, 3, 3, rows);

    // Check result matches expected
    const result_arr: [9]f32 = result;
    const expected_arr: [9]f32 = expected;
    try expectArrayApproxEqual(9, expected_arr, result_arr);

    // Verify REF properties
    try checkRowEchelonForm(3, 3, rows);
}

test "rowEchelonForm - 3x4 augmented matrix" {
    const input: @Vector(12, f32) = .{ 1.0, 2.0, -1.0, 8.0, 2.0, 3.0, 1.0, 11.0, 3.0, 2.0, 2.0, 13.0 };
    const expected: @Vector(12, f32) = .{ 1.0, 0.666667, 0.666667, 4.333333, 0.0, 1.0, -0.2, 1.4, 0.0, 0.0, 1.0, -1.285714 };

    var rows = lmao.splitRows(f32, 3, 4, input);
    lmao.rowEchelonForm(f32, 3, 4, &rows);
    const result = lmao.joinRows(f32, 3, 4, rows);

    const result_arr: [12]f32 = result;
    const expected_arr: [12]f32 = expected;
    try expectArrayApproxEqual(12, expected_arr, result_arr);
    try checkRowEchelonForm(3, 4, rows);
}

test "rowEchelonForm - 4x4 matrix" {
    const input: @Vector(16, f32) = .{ 4.0, -2.0, 1.0, 3.0, 2.0, 1.0, -3.0, 2.0, 1.0, 3.0, 2.0, -1.0, 3.0, 2.0, 1.0, 4.0 };
    const expected: @Vector(16, f32) = .{ 1.0, -0.5, 0.25, 0.75, 0.0, 1.0, 0.5, -0.5, 0.0, 0.0, 1.0, -0.333333, 0.0, 0.0, 0.0, 1.0 };

    var rows = lmao.splitRows(f32, 4, 4, input);
    lmao.rowEchelonForm(f32, 4, 4, &rows);
    const result = lmao.joinRows(f32, 4, 4, rows);

    const result_arr: [16]f32 = result;
    const expected_arr: [16]f32 = expected;
    try expectArrayApproxEqual(16, expected_arr, result_arr);
    try checkRowEchelonForm(4, 4, rows);
}

test "rowEchelonForm - 2x3 wide matrix" {
    const input: @Vector(6, f32) = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const expected: @Vector(6, f32) = .{ 1.0, 1.25, 1.5, 0.0, 1.0, 2.0 };

    var rows = lmao.splitRows(f32, 2, 3, input);
    lmao.rowEchelonForm(f32, 2, 3, &rows);
    const result = lmao.joinRows(f32, 2, 3, rows);

    const result_arr: [6]f32 = result;
    const expected_arr: [6]f32 = expected;
    try expectArrayApproxEqual(6, expected_arr, result_arr);
    try checkRowEchelonForm(2, 3, rows);
}

test "reducedRowEchelonForm - 3x3 matrix (yields identity)" {
    const input: @Vector(9, f32) = .{ 2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0 };
    const expected: @Vector(9, f32) = .{ 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };

    var rows = lmao.splitRows(f32, 3, 3, input);
    lmao.reducedRowEchelonForm(f32, 3, 3, &rows);
    const result = lmao.joinRows(f32, 3, 3, rows);

    const result_arr: [9]f32 = result;
    const expected_arr: [9]f32 = expected;
    try expectArrayApproxEqual(9, expected_arr, result_arr);
    try checkReducedRowEchelonForm(3, 3, rows);
}

test "reducedRowEchelonForm - 3x4 augmented matrix" {
    const input: @Vector(12, f32) = .{ 1.0, 2.0, -1.0, 8.0, 2.0, 3.0, 1.0, 11.0, 3.0, 2.0, 2.0, 13.0 };
    const expected: @Vector(12, f32) = .{ 1.0, 0.0, 0.0, 4.428571, 0.0, 1.0, 0.0, 1.142857, 0.0, 0.0, 1.0, -1.285714 };

    var rows = lmao.splitRows(f32, 3, 4, input);
    lmao.reducedRowEchelonForm(f32, 3, 4, &rows);
    const result = lmao.joinRows(f32, 3, 4, rows);

    const result_arr: [12]f32 = result;
    const expected_arr: [12]f32 = expected;
    try expectArrayApproxEqual(12, expected_arr, result_arr);
    try checkReducedRowEchelonForm(3, 4, rows);
}

test "reducedRowEchelonForm - 3x6 augmented [A|I] for inverse" {
    const input: @Vector(18, f32) = .{ 1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 1.0, 0.0, 3.0, 2.0, 9.0, 0.0, 0.0, 1.0 };
    const expected: @Vector(18, f32) = .{ 1.0, 0.0, 0.0, -2.5625, 0.75, 0.6875, 0.0, 1.0, 0.0, 0.75, 0.0, -0.25, 0.0, 0.0, 1.0, 0.6875, -0.25, -0.0625 };

    var rows = lmao.splitRows(f32, 3, 6, input);
    lmao.reducedRowEchelonForm(f32, 3, 6, &rows);
    const result = lmao.joinRows(f32, 3, 6, rows);

    const result_arr: [18]f32 = result;
    const expected_arr: [18]f32 = expected;
    try expectArrayApproxEqual(18, expected_arr, result_arr);
    try checkReducedRowEchelonForm(3, 6, rows);
}

test "reducedRowEchelonForm - 4x4 matrix (yields identity)" {
    const input: @Vector(16, f32) = .{ 4.0, -2.0, 1.0, 3.0, 2.0, 1.0, -3.0, 2.0, 1.0, 3.0, 2.0, -1.0, 3.0, 2.0, 1.0, 4.0 };
    const expected: @Vector(16, f32) = .{ 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0 };

    var rows = lmao.splitRows(f32, 4, 4, input);
    lmao.reducedRowEchelonForm(f32, 4, 4, &rows);
    const result = lmao.joinRows(f32, 4, 4, rows);

    const result_arr: [16]f32 = result;
    const expected_arr: [16]f32 = expected;
    try expectArrayApproxEqual(16, expected_arr, result_arr);
    try checkReducedRowEchelonForm(4, 4, rows);
}

test "reducedRowEchelonForm - 2x3 wide matrix" {
    const input: @Vector(6, f32) = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const expected: @Vector(6, f32) = .{ 1.0, 0.0, -1.0, 0.0, 1.0, 2.0 };

    var rows = lmao.splitRows(f32, 2, 3, input);
    lmao.reducedRowEchelonForm(f32, 2, 3, &rows);
    const result = lmao.joinRows(f32, 2, 3, rows);

    const result_arr: [6]f32 = result;
    const expected_arr: [6]f32 = expected;
    try expectArrayApproxEqual(6, expected_arr, result_arr);
    try checkReducedRowEchelonForm(2, 3, rows);
}

// ============================================================================
// QR Decomposition Tests (Householder)
// ============================================================================

// Helper to check if Q is orthogonal (Q^T * Q ≈ I)
fn checkOrthogonal(comptime N: usize, Q: @Vector(N * N, f32)) !void {
    const tol: f32 = 1e-4;

    // Compute Q^T * Q
    const Q_arr: [N * N]f32 = Q;
    var result: [N * N]f32 = undefined;

    for (0..N) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..N) |k| {
                // Q^T[i,k] * Q[k,j] = Q[k,i] * Q[k,j]
                sum += Q_arr[k * N + i] * Q_arr[k * N + j];
            }
            result[i * N + j] = sum;
        }
    }

    // Check result ≈ I
    for (0..N) |i| {
        for (0..N) |j| {
            const expected: f32 = if (i == j) 1.0 else 0.0;
            const actual = result[i * N + j];
            if (@abs(actual - expected) > tol) {
                std.debug.print("Q^T*Q[{},{}] = {}, expected {}\n", .{ i, j, actual, expected });
                return error.NotOrthogonal;
            }
        }
    }
}

// Helper to check if R is upper triangular
fn checkUpperTriangular(comptime R: usize, comptime C: usize, mat: @Vector(R * C, f32)) !void {
    const tol: f32 = 1e-4;
    const arr: [R * C]f32 = mat;

    for (0..R) |i| {
        for (0..C) |j| {
            if (i > j) {
                const val = arr[i * C + j];
                if (@abs(val) > tol) {
                    std.debug.print("R[{},{}] = {} should be 0 (below diagonal)\n", .{ i, j, val });
                    return error.NotUpperTriangular;
                }
            }
        }
    }
}

// Helper to check Q * R ≈ A
fn checkQRReconstruction(comptime R: usize, comptime C: usize, A: @Vector(R * C, f32), Q: @Vector(R * R, f32), R_mat: @Vector(R * C, f32)) !void {
    const tol: f32 = 1e-3;

    const A_arr: [R * C]f32 = A;
    const Q_arr: [R * R]f32 = Q;
    const R_arr: [R * C]f32 = R_mat;

    // Compute Q * R
    var result: [R * C]f32 = undefined;
    for (0..R) |i| {
        for (0..C) |j| {
            var sum: f32 = 0;
            for (0..R) |k| {
                sum += Q_arr[i * R + k] * R_arr[k * C + j];
            }
            result[i * C + j] = sum;
        }
    }

    // Check result ≈ A
    for (0..R * C) |idx| {
        if (@abs(result[idx] - A_arr[idx]) > tol) {
            const row = idx / C;
            const col = idx % C;
            std.debug.print("(Q*R)[{},{}] = {}, A[{},{}] = {}\n", .{ row, col, result[idx], row, col, A_arr[idx] });
            return error.ReconstructionFailed;
        }
    }
}

test "qrHouseholder - 2x2 matrix" {
    const A: @Vector(4, f32) = .{ 1.0, 2.0, 3.0, 4.0 };
    var Q: @Vector(4, f32) = undefined;
    var R_mat: @Vector(4, f32) = undefined;

    lmao.qrHouseholder(f32, 2, 2, A, &Q, &R_mat);

    try checkOrthogonal(2, Q);
    try checkUpperTriangular(2, 2, R_mat);
    try checkQRReconstruction(2, 2, A, Q, R_mat);
}

test "qrHouseholder - 3x3 classic example" {
    // Classic QR example from numerical analysis
    const A: @Vector(9, f32) = .{ 12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0 };
    var Q: @Vector(9, f32) = undefined;
    var R_mat: @Vector(9, f32) = undefined;

    lmao.qrHouseholder(f32, 3, 3, A, &Q, &R_mat);

    try checkOrthogonal(3, Q);
    try checkUpperTriangular(3, 3, R_mat);
    try checkQRReconstruction(3, 3, A, Q, R_mat);
}

test "qrHouseholder - 3x3 with negatives" {
    const A: @Vector(9, f32) = .{ 2.5, -1.3, 0.7, -0.4, 3.2, -1.8, 1.1, -0.9, 2.4 };
    var Q: @Vector(9, f32) = undefined;
    var R_mat: @Vector(9, f32) = undefined;

    lmao.qrHouseholder(f32, 3, 3, A, &Q, &R_mat);

    try checkOrthogonal(3, Q);
    try checkUpperTriangular(3, 3, R_mat);
    try checkQRReconstruction(3, 3, A, Q, R_mat);
}

test "qrHouseholder - 4x4 matrix" {
    const A: @Vector(16, f32) = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 };
    var Q: @Vector(16, f32) = undefined;
    var R_mat: @Vector(16, f32) = undefined;

    lmao.qrHouseholder(f32, 4, 4, A, &Q, &R_mat);

    try checkOrthogonal(4, Q);
    try checkUpperTriangular(4, 4, R_mat);
    try checkQRReconstruction(4, 4, A, Q, R_mat);
}

test "qrHouseholder - identity matrix" {
    const A: @Vector(9, f32) = .{ 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
    var Q: @Vector(9, f32) = undefined;
    var R_mat: @Vector(9, f32) = undefined;

    lmao.qrHouseholder(f32, 3, 3, A, &Q, &R_mat);

    try checkOrthogonal(3, Q);
    try checkUpperTriangular(3, 3, R_mat);
    try checkQRReconstruction(3, 3, A, Q, R_mat);

    // For identity: Q should be ±I, R should be ±I
    const Q_arr: [9]f32 = Q;
    const R_arr: [9]f32 = R_mat;
    for (0..3) |i| {
        for (0..3) |j| {
            const expected: f32 = if (i == j) 1.0 else 0.0;
            try expectApproxEqual(expected, @abs(Q_arr[i * 3 + j]));
            try expectApproxEqual(expected, @abs(R_arr[i * 3 + j]));
        }
    }
}

test "qrHouseholder - diagonal matrix" {
    const A: @Vector(9, f32) = .{ 3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 5.0 };
    var Q: @Vector(9, f32) = undefined;
    var R_mat: @Vector(9, f32) = undefined;

    lmao.qrHouseholder(f32, 3, 3, A, &Q, &R_mat);

    try checkOrthogonal(3, Q);
    try checkUpperTriangular(3, 3, R_mat);
    try checkQRReconstruction(3, 3, A, Q, R_mat);
}

test "qrHouseholder - 4x3 tall matrix" {
    const A: @Vector(12, f32) = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    var Q: @Vector(16, f32) = undefined;
    var R_mat: @Vector(12, f32) = undefined;

    lmao.qrHouseholder(f32, 4, 3, A, &Q, &R_mat);

    try checkOrthogonal(4, Q);
    try checkUpperTriangular(4, 3, R_mat);
    try checkQRReconstruction(4, 3, A, Q, R_mat);
}

test "qrHouseholder - 3x4 wide matrix" {
    const A: @Vector(12, f32) = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    var Q: @Vector(9, f32) = undefined;
    var R_mat: @Vector(12, f32) = undefined;

    lmao.qrHouseholder(f32, 3, 4, A, &Q, &R_mat);

    try checkOrthogonal(3, Q);
    try checkUpperTriangular(3, 4, R_mat);
    try checkQRReconstruction(3, 4, A, Q, R_mat);
}

test "qrHouseholder - random 5x5 matrix" {
    // Larger test with more varied values
    const A: @Vector(25, f32) = .{
        4.2,  -1.3, 2.7,  0.8,  -3.1,
        -2.4, 5.6,  -0.9, 3.2,  1.7,
        1.1,  -2.8, 4.5,  -1.6, 2.3,
        -0.7, 3.4,  -2.1, 5.8,  -0.4,
        2.9,  -1.5, 0.6,  -2.7, 4.1,
    };
    var Q: @Vector(25, f32) = undefined;
    var R_mat: @Vector(25, f32) = undefined;

    lmao.qrHouseholder(f32, 5, 5, A, &Q, &R_mat);

    try checkOrthogonal(5, Q);
    try checkUpperTriangular(5, 5, R_mat);
    try checkQRReconstruction(5, 5, A, Q, R_mat);
}

test "qrHouseholder - 6x4 tall matrix" {
    const A: @Vector(24, f32) = .{
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0,  10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0,
    };
    var Q: @Vector(36, f32) = undefined;
    var R_mat: @Vector(24, f32) = undefined;

    lmao.qrHouseholder(f32, 6, 4, A, &Q, &R_mat);

    try checkOrthogonal(6, Q);
    try checkUpperTriangular(6, 4, R_mat);
    try checkQRReconstruction(6, 4, A, Q, R_mat);
}

test "qrHouseholder - 2x5 wide matrix" {
    const A: @Vector(10, f32) = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    var Q: @Vector(4, f32) = undefined;
    var R_mat: @Vector(10, f32) = undefined;

    lmao.qrHouseholder(f32, 2, 5, A, &Q, &R_mat);

    try checkOrthogonal(2, Q);
    try checkUpperTriangular(2, 5, R_mat);
    try checkQRReconstruction(2, 5, A, Q, R_mat);
}
