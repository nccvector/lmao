# Lightning Matrix Algebra Operations
aka. lmao

A fast Zig linear algebra library with SIMD-optimized matrix operations using `@mulAdd` for fused multiply-add instructions.

## Benchmarks

Average of 10 runs with 10,000,000 iterations each (ReleaseFast, Apple Silicon M4):

| Operation              |   dot   | dotSIMD | Speedup |
|------------------------|---------|---------|---------|
| Mat4f dot Vec4f        | 1.50 ns | 0.85 ns |  1.76x  |
| Mat3f dot Vec3f        | 0.96 ns | 0.70 ns |  1.37x  |
| Mat2f dot Vec2f        | 0.66 ns | 0.70 ns |  0.94x  |
| Mat4f dot Mat4f        | 3.20 ns | 1.01 ns |  3.17x  |
| Mat3f dot Mat3f        | 1.32 ns | 0.66 ns |  2.00x  |
| Mat2f dot Mat2f        | 0.80 ns | 0.60 ns |  1.33x  |
| Mat4x3 dot Vec3 -> Vec4| 1.16 ns | 0.65 ns |  1.78x  |
| Mat8x8 dot Vec8        | 9.21 ns | 6.45 ns |  1.43x  |

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

// SIMD-optimized dot product (up to 3x faster for larger matrices)
const result_fast = mat.dotSIMD(vec);
```
