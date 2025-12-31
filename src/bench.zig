const std = @import("std");
const lmao = @import("lmao");

const MatrixX = lmao.MatrixX;

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

/// Generate random values for a given type
fn randomValue(comptime T: type, rng: std.Random) T {
    return switch (@typeInfo(T)) {
        .float => |info| blk: {
            // std.Random.float doesn't support f16, so use f32 and convert
            if (info.bits == 16) {
                break :blk @as(T, @floatCast(rng.float(f32) * 2.0 - 1.0));
            } else {
                break :blk rng.float(T) * 2.0 - 1.0;
            }
        },
        .int => |info| blk: {
            if (info.signedness == .signed) {
                break :blk @as(T, @intCast(rng.intRangeAtMost(i64, std.math.minInt(T), std.math.maxInt(T))));
            } else {
                break :blk rng.int(T);
            }
        },
        else => @compileError("Unsupported type for random generation"),
    };
}

/// Generic benchmark for MatrixX(T, R, C).dot(MatrixX(T, C, K))
fn benchmarkDot(
    comptime T: type,
    comptime R: usize,
    comptime C: usize,
    comptime K: usize,
    comptime name: []const u8,
    iterations: usize,
    rng: std.Random,
    alloc: std.mem.Allocator,
) !void {
    const MatA = MatrixX(T, R, C);
    const MatB = MatrixX(T, C, K);

    const matsA = try alloc.alloc([R * C]T, iterations);
    defer alloc.free(matsA);
    const matsB = try alloc.alloc([C * K]T, iterations);
    defer alloc.free(matsB);

    // Pre-generate random data
    for (0..iterations) |i| {
        for (0..R * C) |j| matsA[i][j] = randomValue(T, rng);
        for (0..C * K) |j| matsB[i][j] = randomValue(T, rng);
    }

    // Benchmark dot
    var acc1: T = 0;
    const start1 = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        const result = MatA.fromArray(&matsA[i]).dot(MatB.fromArray(&matsB[i]));
        acc1 += result.data[0];
    }
    const end1 = std.time.nanoTimestamp();
    doNotOptimizeAway(acc1);
    const dot_ns = @as(f64, @floatFromInt(end1 - start1)) / @as(f64, @floatFromInt(iterations));

    // Benchmark dotSIMD
    var acc2: T = 0;
    const start2 = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        const result = MatA.fromArray(&matsA[i]).dotSIMD(MatB.fromArray(&matsB[i]));
        acc2 += result.data[0];
    }
    const end2 = std.time.nanoTimestamp();
    doNotOptimizeAway(acc2);
    const simd_ns = @as(f64, @floatFromInt(end2 - start2)) / @as(f64, @floatFromInt(iterations));

    printRow(name, dot_ns, simd_ns);
}

/// Run all benchmarks for a given scalar type
fn runBenchmarks(comptime T: type, iterations: usize, rng: std.Random, alloc: std.mem.Allocator) !void {
    const type_name = @typeName(T);

    print("\n Benchmark: dot vs dotSIMD for {s} ({d} iterations, ReleaseFast)\n", .{ type_name, iterations });
    print("{s}\n", .{"═" ** 78});
    print(" {s: <40} │ {s: >10} │ {s: >10} │ {s: >10}\n", .{ "Operation", "dot", "dotSIMD", "Speedup" });
    print("{s}\n", .{"─" ** 78});

    // Matrix-Vector operations
    try benchmarkDot(T, 2, 2, 1, "Mat2 dot Vec2", iterations, rng, alloc);
    try benchmarkDot(T, 3, 3, 1, "Mat3 dot Vec3", iterations, rng, alloc);
    try benchmarkDot(T, 4, 4, 1, "Mat4 dot Vec4", iterations, rng, alloc);

    // Matrix-Matrix operations
    try benchmarkDot(T, 2, 2, 2, "Mat2 dot Mat2", iterations, rng, alloc);
    try benchmarkDot(T, 3, 3, 3, "Mat3 dot Mat3", iterations, rng, alloc);
    try benchmarkDot(T, 4, 4, 4, "Mat4 dot Mat4", iterations, rng, alloc);

    // Non-square operations
    try benchmarkDot(T, 4, 3, 1, "Mat4x3 dot Vec3 -> Vec4", iterations, rng, alloc);
    try benchmarkDot(T, 8, 8, 1, "Mat8x8 dot Vec8", iterations, rng, alloc);

    print("{s}\n\n", .{"═" ** 78});
}

const ScalarType = enum {
    u8,
    u16,
    u32,
    u64,
    i8,
    i16,
    i32,
    i64,
    f16,
    f32,
    f64,

    fn fromString(s: []const u8) ?ScalarType {
        const map = std.StaticStringMap(ScalarType).initComptime(.{
            .{ "u8", .u8 },
            .{ "u16", .u16 },
            .{ "u32", .u32 },
            .{ "u64", .u64 },
            .{ "i8", .i8 },
            .{ "i16", .i16 },
            .{ "i32", .i32 },
            .{ "i64", .i64 },
            .{ "f16", .f16 },
            .{ "f32", .f32 },
            .{ "f64", .f64 },
        });
        return map.get(s);
    }
};

pub fn main() !void {
    const args = try std.process.argsAlloc(std.heap.page_allocator);
    defer std.process.argsFree(std.heap.page_allocator, args);

    var iterations: usize = 1_000_000;
    var scalar_type: ScalarType = .f32;

    for (args) |arg| {
        if (std.mem.startsWith(u8, arg, "-N=")) {
            iterations = std.fmt.parseInt(usize, arg[3..], 10) catch 1_000_000;
        } else if (std.mem.startsWith(u8, arg, "-T=")) {
            if (ScalarType.fromString(arg[3..])) |t| {
                scalar_type = t;
            } else {
                print("Unknown type: {s}. Supported: u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, f64\n", .{arg[3..]});
                return;
            }
        }
    }

    // Runtime RNG - seed from timestamp so compiler can't know values
    const ts: i128 = std.time.nanoTimestamp();
    var prng = std.Random.DefaultPrng.init(@truncate(@as(u128, @bitCast(ts))));
    const rng = prng.random();

    const alloc = std.heap.page_allocator;

    switch (scalar_type) {
        .u8 => try runBenchmarks(u8, iterations, rng, alloc),
        .u16 => try runBenchmarks(u16, iterations, rng, alloc),
        .u32 => try runBenchmarks(u32, iterations, rng, alloc),
        .u64 => try runBenchmarks(u64, iterations, rng, alloc),
        .i8 => try runBenchmarks(i8, iterations, rng, alloc),
        .i16 => try runBenchmarks(i16, iterations, rng, alloc),
        .i32 => try runBenchmarks(i32, iterations, rng, alloc),
        .i64 => try runBenchmarks(i64, iterations, rng, alloc),
        .f16 => try runBenchmarks(f16, iterations, rng, alloc),
        .f32 => try runBenchmarks(f32, iterations, rng, alloc),
        .f64 => try runBenchmarks(f64, iterations, rng, alloc),
    }
}
