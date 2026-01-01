const std = @import("std");

// Use std.mem.doNotOptimizeAway to prevent DCE
pub const doNotOptimizeAway = std.mem.doNotOptimizeAway;

pub fn print(comptime fmt: []const u8, args: anytype) void {
    const stdout = std.fs.File.stdout();
    var buf: [4096]u8 = undefined;
    const slice = std.fmt.bufPrint(&buf, fmt, args) catch return;
    _ = stdout.write(slice) catch {};
}

pub fn formatTimeBuf(ns: f64, buf: []u8) []const u8 {
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

/// Generate random values for a given type
pub fn randomValue(comptime T: type, rng: std.Random) T {
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

pub const ScalarType = enum {
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

    pub fn fromString(s: []const u8) ?ScalarType {
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

/// Initialize RNG from timestamp
pub fn initRng() std.Random {
    const ts: i128 = std.time.nanoTimestamp();
    var prng = std.Random.DefaultPrng.init(@truncate(@as(u128, @bitCast(ts))));
    return prng.random();
}

/// Parse common benchmark arguments (-N=iterations, -T=type)
pub fn parseArgs(args: []const [:0]const u8) struct { iterations: usize, scalar_type: ScalarType } {
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
            }
        }
    }

    return .{ .iterations = iterations, .scalar_type = scalar_type };
}
