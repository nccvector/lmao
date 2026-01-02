const std = @import("std");
const lmao = @import("lmao");
const common = @import("bench_common.zig");

const print = common.print;
const formatTimeBuf = common.formatTimeBuf;
const randomValue = common.randomValue;
const doNotOptimizeAway = common.doNotOptimizeAway;

fn printRow(name: []const u8, qr_ns: f64) void {
    var qr_buf: [32]u8 = undefined;
    const qr_str = formatTimeBuf(qr_ns, &qr_buf);
    print(" {s: <35} │ {s: >12}\n", .{ name, qr_str });
}

/// Generate a random matrix with values in [-1, 1]
fn generateRandomMatrix(
    comptime T: type,
    comptime N: usize,
    rng: std.Random,
    out: *[N]T,
) void {
    for (0..N) |i| {
        out[i] = randomValue(T, rng);
    }
}

/// Benchmark QR decomposition for RxC matrices
fn benchmarkQR(
    comptime T: type,
    comptime R: usize,
    comptime C: usize,
    comptime name: []const u8,
    iterations: usize,
    rng: std.Random,
    alloc: std.mem.Allocator,
) void {
    // Pre-generate random matrices
    const matrices = alloc.alloc([R * C]T, iterations) catch @panic("OOM");
    defer alloc.free(matrices);

    for (0..iterations) |i| {
        generateRandomMatrix(T, R * C, rng, &matrices[i]);
    }

    // Benchmark qrHouseholder
    var acc: T = 0;
    const start = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        const A: @Vector(R * C, T) = matrices[i];
        var Q: @Vector(R * R, T) = undefined;
        var R_mat: @Vector(R * C, T) = undefined;
        lmao.qrHouseholder(T, R, C, A, &Q, &R_mat);
        // Accumulate to prevent DCE
        acc += Q[0] + R_mat[0];
    }
    const end = std.time.nanoTimestamp();
    doNotOptimizeAway(acc);
    const qr_ns = @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(iterations));

    printRow(name, qr_ns);
}

/// Run all benchmarks for a given scalar type
fn runBenchmarks(comptime T: type, iterations: usize, rng: std.Random, alloc: std.mem.Allocator) void {
    const type_name = @typeName(T);

    print("\n Benchmark: QR Decomposition (Householder) for {s} ({d} iterations, ReleaseFast)\n", .{ type_name, iterations });
    print("{s}\n", .{"═" ** 52});
    print(" {s: <35} │ {s: >12}\n", .{ "Matrix Size", "Time/iter" });
    print("{s}\n", .{"─" ** 52});

    // Square matrices
    benchmarkQR(T, 2, 2, "2x2 (square)", iterations, rng, alloc);
    benchmarkQR(T, 3, 3, "3x3 (square)", iterations, rng, alloc);
    benchmarkQR(T, 4, 4, "4x4 (square)", iterations, rng, alloc);
    benchmarkQR(T, 5, 5, "5x5 (square)", iterations, rng, alloc);
    benchmarkQR(T, 8, 8, "8x8 (square)", iterations, rng, alloc);

    print("{s}\n", .{"─" ** 52});

    // Tall matrices (more rows than columns)
    benchmarkQR(T, 4, 2, "4x2 (tall)", iterations, rng, alloc);
    benchmarkQR(T, 4, 3, "4x3 (tall)", iterations, rng, alloc);
    benchmarkQR(T, 6, 4, "6x4 (tall)", iterations, rng, alloc);
    benchmarkQR(T, 8, 4, "8x4 (tall)", iterations, rng, alloc);

    print("{s}\n", .{"─" ** 52});

    // Wide matrices (more columns than rows)
    benchmarkQR(T, 2, 4, "2x4 (wide)", iterations, rng, alloc);
    benchmarkQR(T, 3, 4, "3x4 (wide)", iterations, rng, alloc);
    benchmarkQR(T, 4, 6, "4x6 (wide)", iterations, rng, alloc);
    benchmarkQR(T, 4, 8, "4x8 (wide)", iterations, rng, alloc);

    print("{s}\n\n", .{"═" ** 52});
}

pub fn main() !void {
    const args = try std.process.argsAlloc(std.heap.page_allocator);
    defer std.process.argsFree(std.heap.page_allocator, args);

    const parsed = common.parseArgs(args);
    const iterations = parsed.iterations;
    const scalar_type = parsed.scalar_type;

    // Runtime RNG - seed from timestamp so compiler can't know values
    const ts: i128 = std.time.nanoTimestamp();
    var prng = std.Random.DefaultPrng.init(@truncate(@as(u128, @bitCast(ts))));
    const rng = prng.random();

    const alloc = std.heap.page_allocator;

    switch (scalar_type) {
        .f16 => runBenchmarks(f16, iterations, rng, alloc),
        .f32 => runBenchmarks(f32, iterations, rng, alloc),
        .f64 => runBenchmarks(f64, iterations, rng, alloc),
        else => {
            print("QR decomposition benchmarks only support float types (f16, f32, f64)\n", .{});
            print("Use -T=f32 (default) or -T=f64\n", .{});
        },
    }
}
