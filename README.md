# Lightning Matrix Algebra Operations
aka. lmao

A fast Zig linear algebra library with SIMD-optimized matrix operations using `@mulAdd` for fused multiply-add instructions.

## Benchmarks

Average of 10 runs with 10,000,000 iterations each (ReleaseFast, Apple Silicon M4):

| Operation              |   dot   | dotSIMD | Speedup |
|------------------------|---------|---------|---------|
| Mat4f dot Vec4f        | 1.84 ns | 0.90 ns |  2.04x  |
| Mat3f dot Vec3f        | 1.17 ns | 0.64 ns |  1.83x  |
| Mat2f dot Vec2f        | 0.80 ns | 0.62 ns |  1.29x  |
| Mat4f dot Mat4f        | 3.91 ns | 1.01 ns |  3.87x  |
| Mat3f dot Mat3f        | 1.51 ns | 0.67 ns |  2.25x  |
| Mat2f dot Mat2f        | 0.86 ns | 0.61 ns |  1.41x  |
| Mat4x3 dot Vec3 -> Vec4| 1.47 ns | 0.68 ns |  2.16x  |
| Mat8x8 dot Vec8        |10.93 ns | 5.17 ns |  2.11x  |

Run benchmarks yourself:
```bash
zig build bench -DN=10000000
```

## Usage

```zig
const lmao = @import("lmao");
const Mat4f = lmao.Mat4f;
const Vec4f = lmao.Vec4f;

// Create matrices and vectors
const mat = Mat4f.fromArray(&.{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 });
const vec = Vec4f.fromArray(&.{ 1, 2, 3, 4 });

// Standard dot product
const result = mat.dot(vec);

// SIMD-optimized dot product (up to 4x faster for larger matrices)
const result_fast = mat.dotSIMD(vec);
```
