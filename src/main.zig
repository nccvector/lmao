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
