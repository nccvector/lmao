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
    print(" {s: <40} │ {s: >12}\n", .{ name, qr_str });
}

fn printRowCompare(name: []const u8, old_ns: f64, new_ns: f64) void {
    var old_buf: [32]u8 = undefined;
    var new_buf: [32]u8 = undefined;
    const old_str = formatTimeBuf(old_ns, &old_buf);
    const new_str = formatTimeBuf(new_ns, &new_buf);
    const speedup = old_ns / new_ns;
    print(" {s: <20} │ {s: >12} │ {s: >12} │ {d:>6.2}x\n", .{ name, old_str, new_str, speedup });
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


/// Benchmark comparison: old qrHouseholder vs new qrHouseholderCompact
fn benchmarkQRCompare(
    comptime T: type,
    comptime R: usize,
    comptime C: usize,
    comptime name: []const u8,
    iterations: usize,
    rng: std.Random,
    alloc: std.mem.Allocator,
) void {
    // Pre-generate random matrices (flat storage for old impl)
    const matrices_flat = alloc.alloc([R * C]T, iterations) catch @panic("OOM");
    defer alloc.free(matrices_flat);

    // Pre-generate random matrices (row storage for new impl)
    const matrices_rows = alloc.alloc([R]@Vector(C, T), iterations) catch @panic("OOM");
    defer alloc.free(matrices_rows);

    for (0..iterations) |i| {
        generateRandomMatrix(T, R * C, rng, &matrices_flat[i]);
        // Convert to row storage outside timing loop
        matrices_rows[i] = lmao.splitRows(T, R, C, matrices_flat[i]);
    }

    // Benchmark OLD qrHouseholder (builds full Q)
    var acc_old: T = 0;
    const start_old = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        const A: @Vector(R * C, T) = matrices_flat[i];
        var Q: @Vector(R * R, T) = undefined;
        var R_mat: @Vector(R * C, T) = undefined;
        lmao.qrHouseholder(T, R, C, A, &Q, &R_mat);
        acc_old += Q[0] + R_mat[0];
    }
    const end_old = std.time.nanoTimestamp();
    doNotOptimizeAway(acc_old);
    const old_ns = @as(f64, @floatFromInt(end_old - start_old)) / @as(f64, @floatFromInt(iterations));

    // Benchmark NEW qrHouseholderCompact (factorization only, no Q)
    var acc_new: T = 0;
    const start_new = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        var A_rows = matrices_rows[i];
        var tau: [R]T = undefined;
        lmao.qrHouseholderCompact(T, R, C, &A_rows, &tau);
        acc_new += A_rows[0][0] + tau[0];
    }
    const end_new = std.time.nanoTimestamp();
    doNotOptimizeAway(acc_new);
    const new_ns = @as(f64, @floatFromInt(end_new - start_new)) / @as(f64, @floatFromInt(iterations));

    printRowCompare(name, old_ns, new_ns);
}

/// Benchmark full solve pipeline: qrSolve
fn benchmarkQRSolve(
    comptime T: type,
    comptime R: usize,
    comptime C: usize,
    comptime name: []const u8,
    iterations: usize,
    rng: std.Random,
    alloc: std.mem.Allocator,
) void {
    // Pre-generate random matrices (row storage) and RHS vectors
    const matrices_rows = alloc.alloc([R]@Vector(C, T), iterations) catch @panic("OOM");
    defer alloc.free(matrices_rows);
    const rhs_vecs = alloc.alloc([R]@Vector(1, T), iterations) catch @panic("OOM");
    defer alloc.free(rhs_vecs);

    for (0..iterations) |i| {
        var flat: [R * C]T = undefined;
        generateRandomMatrix(T, R * C, rng, &flat);
        matrices_rows[i] = lmao.splitRows(T, R, C, flat);

        var rhs_flat: [R]T = undefined;
        generateRandomMatrix(T, R, rng, &rhs_flat);
        for (0..R) |r| {
            rhs_vecs[i][r][0] = rhs_flat[r];
        }
    }

    // Benchmark qrSolve
    var acc: T = 0;
    const start = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        const x = lmao.qrSolve(T, R, C, 1, matrices_rows[i], rhs_vecs[i]);
        acc += x[0][0];
    }
    const end = std.time.nanoTimestamp();
    doNotOptimizeAway(acc);
    const ns = @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(iterations));

    printRow(name, ns);
}

/// Run all benchmarks for a given scalar type
fn runBenchmarks(comptime T: type, iterations: usize, rng: std.Random, alloc: std.mem.Allocator) void {
    const type_name = @typeName(T);

    // Section 1: Factorization comparison (old vs new)
    print("\n Benchmark: QR Factorization - Old (full Q) vs New (compact) for {s}\n", .{type_name});
    print(" ({d} iterations, ReleaseFast)\n", .{iterations});
    print("{s}\n", .{"═" ** 70});
    print(" {s: <20} │ {s: >12} │ {s: >12} │ {s: >8}\n", .{ "Matrix Size", "Old (Q)", "New (no Q)", "Speedup" });
    print("{s}\n", .{"─" ** 70});

    // Square matrices
    benchmarkQRCompare(T, 2, 2, "2x2", iterations, rng, alloc);
    benchmarkQRCompare(T, 3, 3, "3x3", iterations, rng, alloc);
    benchmarkQRCompare(T, 4, 4, "4x4", iterations, rng, alloc);
    benchmarkQRCompare(T, 5, 5, "5x5", iterations, rng, alloc);
    benchmarkQRCompare(T, 8, 8, "8x8", iterations, rng, alloc);

    print("{s}\n", .{"─" ** 70});

    // Tall matrices
    benchmarkQRCompare(T, 4, 2, "4x2 (tall)", iterations, rng, alloc);
    benchmarkQRCompare(T, 6, 4, "6x4 (tall)", iterations, rng, alloc);
    benchmarkQRCompare(T, 8, 4, "8x4 (tall)", iterations, rng, alloc);

    print("{s}\n\n", .{"═" ** 70});

    // Section 2: Full solve pipeline
    print(" Benchmark: QR Solve Pipeline (factor + Q^T*b + back-sub) for {s}\n", .{type_name});
    print("{s}\n", .{"═" ** 57});
    print(" {s: <40} │ {s: >12}\n", .{ "Matrix Size", "Time/iter" });
    print("{s}\n", .{"─" ** 57});

    benchmarkQRSolve(T, 2, 2, "2x2 solve", iterations, rng, alloc);
    benchmarkQRSolve(T, 3, 3, "3x3 solve", iterations, rng, alloc);
    benchmarkQRSolve(T, 4, 4, "4x4 solve", iterations, rng, alloc);
    benchmarkQRSolve(T, 5, 5, "5x5 solve", iterations, rng, alloc);
    benchmarkQRSolve(T, 8, 8, "8x8 solve", iterations, rng, alloc);

    print("{s}\n\n", .{"═" ** 57});
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
