const std = @import("std");
const lmao = @import("lmao");
const common = @import("bench_common.zig");

const print = common.print;
const formatTimeBuf = common.formatTimeBuf;
const randomValue = common.randomValue;
const doNotOptimizeAway = common.doNotOptimizeAway;

fn printRow(name: []const u8, ref_ns: f64, rref_ns: f64) void {
    var ref_buf: [32]u8 = undefined;
    var rref_buf: [32]u8 = undefined;

    const ref_str = formatTimeBuf(ref_ns, &ref_buf);
    const rref_str = formatTimeBuf(rref_ns, &rref_buf);

    print(" {s: <30} │ {s: >12} │ {s: >12}\n", .{ name, ref_str, rref_str });
}

/// Generate a diagonally dominant random matrix (guaranteed invertible).
/// For each row, the diagonal element is set to be larger than the sum
/// of absolute values of all other elements in that row.
fn generateDiagonallyDominantMatrix(
    comptime T: type,
    comptime N: usize,
    rng: std.Random,
    out: *[N * (2 * N)]T,
) void {
    const C = 2 * N; // Augmented matrix width [A | I]

    // Fill with random values first
    for (0..N) |row| {
        var row_sum: T = 0;
        for (0..N) |col| {
            const val = randomValue(T, rng);
            out[row * C + col] = val;
            if (col != row) {
                row_sum += @abs(val);
            }
        }
        // Make diagonal dominant: |a_ii| > sum of |a_ij| for j != i
        const diag_sign: T = if (out[row * C + row] >= 0) 1 else -1;
        out[row * C + row] = diag_sign * (row_sum + 1.0 + @abs(randomValue(T, rng)));

        // Fill identity portion
        for (N..C) |col| {
            out[row * C + col] = if (col - N == row) 1 else 0;
        }
    }
}

/// Benchmark echelon form operations for NxN matrices
fn benchmarkEchelon(
    comptime T: type,
    comptime N: usize,
    comptime name: []const u8,
    iterations: usize,
    rng: std.Random,
    alloc: std.mem.Allocator,
) void {
    const C = 2 * N; // Augmented matrix [A | I]

    // Pre-generate random diagonally dominant matrices
    const matrices = alloc.alloc([N * C]T, iterations) catch @panic("OOM");
    defer alloc.free(matrices);

    for (0..iterations) |i| {
        generateDiagonallyDominantMatrix(T, N, rng, &matrices[i]);
    }

    // Benchmark rowEchelonForm
    var acc1: T = 0;
    const start1 = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        // Copy matrix since the function modifies in place
        const vec: @Vector(N * C, T) = matrices[i];
        var rows = lmao.splitRows(T, N, C, vec);
        lmao.rowEchelonForm(T, N, C, &rows);
        acc1 += rows[0][0];
    }
    const end1 = std.time.nanoTimestamp();
    doNotOptimizeAway(acc1);
    const ref_ns = @as(f64, @floatFromInt(end1 - start1)) / @as(f64, @floatFromInt(iterations));

    // Benchmark reducedRowEchelonForm
    var acc2: T = 0;
    const start2 = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        // Copy matrix since the function modifies in place
        const vec: @Vector(N * C, T) = matrices[i];
        var rows = lmao.splitRows(T, N, C, vec);
        lmao.reducedRowEchelonForm(T, N, C, &rows);
        acc2 += rows[0][0];
    }
    const end2 = std.time.nanoTimestamp();
    doNotOptimizeAway(acc2);
    const rref_ns = @as(f64, @floatFromInt(end2 - start2)) / @as(f64, @floatFromInt(iterations));

    printRow(name, ref_ns, rref_ns);
}

/// Run all benchmarks for a given scalar type
fn runBenchmarks(comptime T: type, iterations: usize, rng: std.Random, alloc: std.mem.Allocator) void {
    const type_name = @typeName(T);

    print("\n Benchmark: Echelon Form for {s} ({d} iterations, ReleaseFast)\n", .{ type_name, iterations });
    print("{s}\n", .{"═" ** 60});
    print(" {s: <30} │ {s: >12} │ {s: >12}\n", .{ "Matrix Size", "REF", "RREF" });
    print("{s}\n", .{"─" ** 60});

    benchmarkEchelon(T, 2, "2x2 -> 2x4 augmented", iterations, rng, alloc);
    benchmarkEchelon(T, 3, "3x3 -> 3x6 augmented", iterations, rng, alloc);
    benchmarkEchelon(T, 4, "4x4 -> 4x8 augmented", iterations, rng, alloc);
    benchmarkEchelon(T, 8, "8x8 -> 8x16 augmented", iterations, rng, alloc);

    print("{s}\n\n", .{"═" ** 60});
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
            print("Echelon form benchmarks only support float types (f16, f32, f64)\n", .{});
            print("Use -T=f32 (default) or -T=f64\n", .{});
        },
    }
}
