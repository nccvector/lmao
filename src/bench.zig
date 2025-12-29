const std = @import("std");
const lmao = @import("lmao");

const Mat2f = lmao.Mat2f;
const Mat3f = lmao.Mat3f;
const Mat4f = lmao.Mat4f;
const Vec2f = lmao.Vec2f;
const Vec3f = lmao.Vec3f;
const Vec4f = lmao.Vec4f;
const MatrixX = lmao.MatrixX;

const Mat2x3f = MatrixX(f32, 2, 3);
const Mat3x4f = MatrixX(f32, 3, 4);
const Mat4x3f = MatrixX(f32, 4, 3);
const Mat8f = MatrixX(f32, 8, 8);
const Vec8f = MatrixX(f32, 8, 1);

// Use std.mem.doNotOptimizeAway to prevent DCE
const doNotOptimizeAway = std.mem.doNotOptimizeAway;

fn print(comptime fmt: []const u8, args: anytype) void {
    const stdout = std.fs.File.stdout();
    var buf: [4096]u8 = undefined;
    const slice = std.fmt.bufPrint(&buf, fmt, args) catch return;
    _ = stdout.write(slice) catch {};
}

fn formatTimeBuf(ns: f64, buf: []u8) []const u8 {
    if (ns < 1_000) {
        return std.fmt.bufPrint(buf, "{d:.1}ns", .{ns}) catch "?";
    } else if (ns < 1_000_000) {
        return std.fmt.bufPrint(buf, "{d:.2}us", .{ns / 1_000.0}) catch "?";
    } else if (ns < 1_000_000_000) {
        return std.fmt.bufPrint(buf, "{d:.2}ms", .{ns / 1_000_000.0}) catch "?";
    } else {
        return std.fmt.bufPrint(buf, "{d:.2}s", .{ns / 1_000_000_000.0}) catch "?";
    }
}

pub fn main() !void {
    const args = try std.process.argsAlloc(std.heap.page_allocator);
    defer std.process.argsFree(std.heap.page_allocator, args);

    var iterations: usize = 1_000_000;
    for (args) |arg| {
        if (std.mem.startsWith(u8, arg, "-N=")) {
            iterations = std.fmt.parseInt(usize, arg[3..], 10) catch 1_000_000;
        }
    }

    // Runtime RNG - seed from timestamp so compiler can't know values
    const ts: i128 = std.time.nanoTimestamp();
    var prng = std.Random.DefaultPrng.init(@truncate(@as(u128, @bitCast(ts))));
    const rng = prng.random();

    print("\n Benchmark: dot vs dotSIMD ({d} iterations, ReleaseFast)\n", .{iterations});
    print("{s}\n", .{"═" ** 78});
    print(" {s: <40} │ {s: >10} │ {s: >10} │ {s: >10}\n", .{ "Operation", "dot", "dotSIMD", "Speedup" });
    print("{s}\n", .{"─" ** 78});

    // Pre-generate random data to avoid measuring RNG speed
    const alloc = std.heap.page_allocator;

    // Mat4f dot Vec4f
    {
        const mats = try alloc.alloc([16]f32, iterations);
        defer alloc.free(mats);
        const vecs = try alloc.alloc([4]f32, iterations);
        defer alloc.free(vecs);

        for (0..iterations) |i| {
            for (0..16) |j| mats[i][j] = rng.float(f32) * 2.0 - 1.0;
            for (0..4) |j| vecs[i][j] = rng.float(f32) * 2.0 - 1.0;
        }

        var acc1: f32 = 0;
        const start1 = std.time.nanoTimestamp();
        for (0..iterations) |i| {
            const result = Mat4f.fromArray(&mats[i]).dot(Vec4f.fromArray(&vecs[i]));
            acc1 += result.data[0];
        }
        const end1 = std.time.nanoTimestamp();
        doNotOptimizeAway(acc1);
        const dot_ns = @as(f64, @floatFromInt(end1 - start1)) / @as(f64, @floatFromInt(iterations));

        var acc2: f32 = 0;
        const start2 = std.time.nanoTimestamp();
        for (0..iterations) |i| {
            const result = Mat4f.fromArray(&mats[i]).dotSIMD(Vec4f.fromArray(&vecs[i]));
            acc2 += result.data[0];
        }
        const end2 = std.time.nanoTimestamp();
        doNotOptimizeAway(acc2);
        const simd_ns = @as(f64, @floatFromInt(end2 - start2)) / @as(f64, @floatFromInt(iterations));

        printRow("Mat4f dot Vec4f", dot_ns, simd_ns);
    }

    // Mat3f dot Vec3f
    {
        const mats = try alloc.alloc([9]f32, iterations);
        defer alloc.free(mats);
        const vecs = try alloc.alloc([3]f32, iterations);
        defer alloc.free(vecs);

        for (0..iterations) |i| {
            for (0..9) |j| mats[i][j] = rng.float(f32) * 2.0 - 1.0;
            for (0..3) |j| vecs[i][j] = rng.float(f32) * 2.0 - 1.0;
        }

        var acc1: f32 = 0;
        const start1 = std.time.nanoTimestamp();
        for (0..iterations) |i| {
            const result = Mat3f.fromArray(&mats[i]).dot(Vec3f.fromArray(&vecs[i]));
            acc1 += result.data[0];
        }
        const end1 = std.time.nanoTimestamp();
        doNotOptimizeAway(acc1);
        const dot_ns = @as(f64, @floatFromInt(end1 - start1)) / @as(f64, @floatFromInt(iterations));

        var acc2: f32 = 0;
        const start2 = std.time.nanoTimestamp();
        for (0..iterations) |i| {
            const result = Mat3f.fromArray(&mats[i]).dotSIMD(Vec3f.fromArray(&vecs[i]));
            acc2 += result.data[0];
        }
        const end2 = std.time.nanoTimestamp();
        doNotOptimizeAway(acc2);
        const simd_ns = @as(f64, @floatFromInt(end2 - start2)) / @as(f64, @floatFromInt(iterations));

        printRow("Mat3f dot Vec3f", dot_ns, simd_ns);
    }

    // Mat2f dot Vec2f
    {
        const mats = try alloc.alloc([4]f32, iterations);
        defer alloc.free(mats);
        const vecs = try alloc.alloc([2]f32, iterations);
        defer alloc.free(vecs);

        for (0..iterations) |i| {
            for (0..4) |j| mats[i][j] = rng.float(f32) * 2.0 - 1.0;
            for (0..2) |j| vecs[i][j] = rng.float(f32) * 2.0 - 1.0;
        }

        var acc1: f32 = 0;
        const start1 = std.time.nanoTimestamp();
        for (0..iterations) |i| {
            const result = Mat2f.fromArray(&mats[i]).dot(Vec2f.fromArray(&vecs[i]));
            acc1 += result.data[0];
        }
        const end1 = std.time.nanoTimestamp();
        doNotOptimizeAway(acc1);
        const dot_ns = @as(f64, @floatFromInt(end1 - start1)) / @as(f64, @floatFromInt(iterations));

        var acc2: f32 = 0;
        const start2 = std.time.nanoTimestamp();
        for (0..iterations) |i| {
            const result = Mat2f.fromArray(&mats[i]).dotSIMD(Vec2f.fromArray(&vecs[i]));
            acc2 += result.data[0];
        }
        const end2 = std.time.nanoTimestamp();
        doNotOptimizeAway(acc2);
        const simd_ns = @as(f64, @floatFromInt(end2 - start2)) / @as(f64, @floatFromInt(iterations));

        printRow("Mat2f dot Vec2f", dot_ns, simd_ns);
    }

    // Mat4f dot Mat4f
    {
        const matsA = try alloc.alloc([16]f32, iterations);
        defer alloc.free(matsA);
        const matsB = try alloc.alloc([16]f32, iterations);
        defer alloc.free(matsB);

        for (0..iterations) |i| {
            for (0..16) |j| matsA[i][j] = rng.float(f32) * 2.0 - 1.0;
            for (0..16) |j| matsB[i][j] = rng.float(f32) * 2.0 - 1.0;
        }

        var acc1: f32 = 0;
        const start1 = std.time.nanoTimestamp();
        for (0..iterations) |i| {
            const result = Mat4f.fromArray(&matsA[i]).dot(Mat4f.fromArray(&matsB[i]));
            acc1 += result.data[0];
        }
        const end1 = std.time.nanoTimestamp();
        doNotOptimizeAway(acc1);
        const dot_ns = @as(f64, @floatFromInt(end1 - start1)) / @as(f64, @floatFromInt(iterations));

        var acc2: f32 = 0;
        const start2 = std.time.nanoTimestamp();
        for (0..iterations) |i| {
            const result = Mat4f.fromArray(&matsA[i]).dotSIMD(Mat4f.fromArray(&matsB[i]));
            acc2 += result.data[0];
        }
        const end2 = std.time.nanoTimestamp();
        doNotOptimizeAway(acc2);
        const simd_ns = @as(f64, @floatFromInt(end2 - start2)) / @as(f64, @floatFromInt(iterations));

        printRow("Mat4f dot Mat4f", dot_ns, simd_ns);
    }

    // Mat3f dot Mat3f
    {
        const matsA = try alloc.alloc([9]f32, iterations);
        defer alloc.free(matsA);
        const matsB = try alloc.alloc([9]f32, iterations);
        defer alloc.free(matsB);

        for (0..iterations) |i| {
            for (0..9) |j| matsA[i][j] = rng.float(f32) * 2.0 - 1.0;
            for (0..9) |j| matsB[i][j] = rng.float(f32) * 2.0 - 1.0;
        }

        var acc1: f32 = 0;
        const start1 = std.time.nanoTimestamp();
        for (0..iterations) |i| {
            const result = Mat3f.fromArray(&matsA[i]).dot(Mat3f.fromArray(&matsB[i]));
            acc1 += result.data[0];
        }
        const end1 = std.time.nanoTimestamp();
        doNotOptimizeAway(acc1);
        const dot_ns = @as(f64, @floatFromInt(end1 - start1)) / @as(f64, @floatFromInt(iterations));

        var acc2: f32 = 0;
        const start2 = std.time.nanoTimestamp();
        for (0..iterations) |i| {
            const result = Mat3f.fromArray(&matsA[i]).dotSIMD(Mat3f.fromArray(&matsB[i]));
            acc2 += result.data[0];
        }
        const end2 = std.time.nanoTimestamp();
        doNotOptimizeAway(acc2);
        const simd_ns = @as(f64, @floatFromInt(end2 - start2)) / @as(f64, @floatFromInt(iterations));

        printRow("Mat3f dot Mat3f", dot_ns, simd_ns);
    }

    // Mat2f dot Mat2f
    {
        const matsA = try alloc.alloc([4]f32, iterations);
        defer alloc.free(matsA);
        const matsB = try alloc.alloc([4]f32, iterations);
        defer alloc.free(matsB);

        for (0..iterations) |i| {
            for (0..4) |j| matsA[i][j] = rng.float(f32) * 2.0 - 1.0;
            for (0..4) |j| matsB[i][j] = rng.float(f32) * 2.0 - 1.0;
        }

        var acc1: f32 = 0;
        const start1 = std.time.nanoTimestamp();
        for (0..iterations) |i| {
            const result = Mat2f.fromArray(&matsA[i]).dot(Mat2f.fromArray(&matsB[i]));
            acc1 += result.data[0];
        }
        const end1 = std.time.nanoTimestamp();
        doNotOptimizeAway(acc1);
        const dot_ns = @as(f64, @floatFromInt(end1 - start1)) / @as(f64, @floatFromInt(iterations));

        var acc2: f32 = 0;
        const start2 = std.time.nanoTimestamp();
        for (0..iterations) |i| {
            const result = Mat2f.fromArray(&matsA[i]).dotSIMD(Mat2f.fromArray(&matsB[i]));
            acc2 += result.data[0];
        }
        const end2 = std.time.nanoTimestamp();
        doNotOptimizeAway(acc2);
        const simd_ns = @as(f64, @floatFromInt(end2 - start2)) / @as(f64, @floatFromInt(iterations));

        printRow("Mat2f dot Mat2f", dot_ns, simd_ns);
    }

    // Mat4x3 dot Vec3 -> Vec4
    {
        const mats = try alloc.alloc([12]f32, iterations);
        defer alloc.free(mats);
        const vecs = try alloc.alloc([3]f32, iterations);
        defer alloc.free(vecs);

        for (0..iterations) |i| {
            for (0..12) |j| mats[i][j] = rng.float(f32) * 2.0 - 1.0;
            for (0..3) |j| vecs[i][j] = rng.float(f32) * 2.0 - 1.0;
        }

        var acc1: f32 = 0;
        const start1 = std.time.nanoTimestamp();
        for (0..iterations) |i| {
            const result = Mat4x3f.fromArray(&mats[i]).dot(Vec3f.fromArray(&vecs[i]));
            acc1 += result.data[0];
        }
        const end1 = std.time.nanoTimestamp();
        doNotOptimizeAway(acc1);
        const dot_ns = @as(f64, @floatFromInt(end1 - start1)) / @as(f64, @floatFromInt(iterations));

        var acc2: f32 = 0;
        const start2 = std.time.nanoTimestamp();
        for (0..iterations) |i| {
            const result = Mat4x3f.fromArray(&mats[i]).dotSIMD(Vec3f.fromArray(&vecs[i]));
            acc2 += result.data[0];
        }
        const end2 = std.time.nanoTimestamp();
        doNotOptimizeAway(acc2);
        const simd_ns = @as(f64, @floatFromInt(end2 - start2)) / @as(f64, @floatFromInt(iterations));

        printRow("Mat4x3 dot Vec3 -> Vec4", dot_ns, simd_ns);
    }

    // Mat8x8 dot Vec8
    {
        const mats = try alloc.alloc([64]f32, iterations);
        defer alloc.free(mats);
        const vecs = try alloc.alloc([8]f32, iterations);
        defer alloc.free(vecs);

        for (0..iterations) |i| {
            for (0..64) |j| mats[i][j] = rng.float(f32) * 2.0 - 1.0;
            for (0..8) |j| vecs[i][j] = rng.float(f32) * 2.0 - 1.0;
        }

        var acc1: f32 = 0;
        const start1 = std.time.nanoTimestamp();
        for (0..iterations) |i| {
            const result = Mat8f.fromArray(&mats[i]).dot(Vec8f.fromArray(&vecs[i]));
            acc1 += result.data[0];
        }
        const end1 = std.time.nanoTimestamp();
        doNotOptimizeAway(acc1);
        const dot_ns = @as(f64, @floatFromInt(end1 - start1)) / @as(f64, @floatFromInt(iterations));

        var acc2: f32 = 0;
        const start2 = std.time.nanoTimestamp();
        for (0..iterations) |i| {
            const result = Mat8f.fromArray(&mats[i]).dotSIMD(Vec8f.fromArray(&vecs[i]));
            acc2 += result.data[0];
        }
        const end2 = std.time.nanoTimestamp();
        doNotOptimizeAway(acc2);
        const simd_ns = @as(f64, @floatFromInt(end2 - start2)) / @as(f64, @floatFromInt(iterations));

        printRow("Mat8x8 dot Vec8", dot_ns, simd_ns);
    }

    print("{s}\n\n", .{"═" ** 78});
}

fn printRow(name: []const u8, dot_ns: f64, simd_ns: f64) void {
    var dot_buf: [32]u8 = undefined;
    var simd_buf: [32]u8 = undefined;
    var speedup_buf: [20]u8 = undefined;

    const dot_str = formatTimeBuf(dot_ns, &dot_buf);
    const simd_str = formatTimeBuf(simd_ns, &simd_buf);

    const speedup_str = blk: {
        if (simd_ns < 0.001 and dot_ns < 0.001) {
            break :blk "~1.00x";
        } else if (simd_ns < 0.001) {
            break :blk ">>1x";
        } else if (dot_ns < 0.001) {
            break :blk "<<1x";
        } else {
            const speedup = dot_ns / simd_ns;
            if (speedup >= 1.0) {
                break :blk std.fmt.bufPrint(&speedup_buf, "{d:.2}x", .{speedup}) catch "?";
            } else {
                const slowdown = 1.0 / speedup;
                break :blk std.fmt.bufPrint(&speedup_buf, "{d:.2}x slow", .{slowdown}) catch "?";
            }
        }
    };

    print(" {s: <40} │ {s: >10} │ {s: >10} │ {s: >10}\n", .{ name, dot_str, simd_str, speedup_str });
}
