# Lightning Matrix Algebra Operations
aka. lmao

A fast Zig linear algebra library with SIMD-optimized matrix operations using `@mulAdd` for fused multiply-add instructions.

## Features

- **SIMD-optimized** with up to 5x speedups
- **Row echelon form** (REF) and **reduced row echelon form** (RREF)
- **QR decomposition** via Householder reflections
- **Linear system solver** using QR factorization
- **Compile-time** matrix dimensions with zero runtime overhead
- Support for `f16`, `f32`, `f64` and integer types

## Benchmarks

Average of 5 runs with 10,000,000 iterations each (ReleaseFast, Apple Silicon M4):

### Dot Product

| Operation              |   dot   | dotSIMD | Speedup |
|------------------------|---------|---------|---------|
| Mat2f × Vec2f          | 0.60 ns | 0.60 ns |  1.00x  |
| Mat3f × Vec3f          | 0.84 ns | 0.62 ns |  1.35x  |
| Mat4f × Vec4f          | 1.36 ns | 0.82 ns |  1.66x  |
| Mat2f × Mat2f          | 0.70 ns | 0.60 ns |  1.17x  |
| Mat3f × Mat3f          | 1.08 ns | 0.64 ns |  1.69x  |
| Mat4f × Mat4f          | 1.98 ns | 1.00 ns |  1.98x  |
| Mat4x3 × Vec3 → Vec4   | 1.00 ns | 0.70 ns |  1.43x  |
| Mat8x8 × Vec8          | 9.44 ns | 4.94 ns |  1.91x  |

### Echelon Form

| Matrix Size           |    REF    |   RREF    |
|-----------------------|-----------|-----------|
| 2×2 → 2×4 augmented   |   1.42 ns |   1.58 ns |
| 3×3 → 3×6 augmented   |   5.82 ns |   6.96 ns |
| 4×4 → 4×8 augmented   |   8.74 ns |   9.56 ns |
| 8×8 → 8×16 augmented  | 128.58 ns | 173.06 ns |

### QR Decomposition

| Matrix Size | Full Q (explicit) | Compact (no Q) | Speedup |
|-------------|-------------------|----------------|---------|
| 2×2         |          5.04 ns  |        1.70 ns |  2.96x  |
| 3×3         |          6.90 ns  |        4.42 ns |  1.56x  |
| 4×4         |         49.62 ns  |        8.72 ns |  5.70x  |
| 5×5         |         76.42 ns  |       72.12 ns |  1.06x  |
| 8×8         |        225.12 ns  |      136.00 ns |  1.66x  |
| 4×2 (tall)  |         31.32 ns  |        1.98 ns | 15.87x  |
| 6×4 (tall)  |         83.00 ns  |       50.42 ns |  1.65x  |
| 8×4 (tall)  |        132.14 ns  |       58.42 ns |  2.26x  |

### QR Solve (factor + Q^T*b + back-substitution)

| Matrix Size | Time/iter |
|-------------|-----------|
| 2×2         |   7.06 ns |
| 3×3         |  13.60 ns |
| 4×4         |  25.28 ns |
| 5×5         |  77.08 ns |
| 8×8         | 221.26 ns |

Run benchmarks yourself:
```bash
zig build bench -DN=10000000
```

## Matrix API

### Type Aliases

```zig
const lmao = @import("lmao");

// Vectors (column matrices)
const Vec2f = lmao.Vec2f;  // Matrix(f32, 2, 1)
const Vec3f = lmao.Vec3f;  // Matrix(f32, 3, 1)
const Vec4f = lmao.Vec4f;  // Matrix(f32, 4, 1)

// Square matrices
const Mat2f = lmao.Mat2f;  // Matrix(f32, 2, 2)
const Mat3f = lmao.Mat3f;  // Matrix(f32, 3, 3)
const Mat4f = lmao.Mat4f;  // Matrix(f32, 4, 4)

// Also available: Vec2d, Mat4d (f64), Vec3i (i32), Vec4u (u32), etc.
```

### Creating Matrices

```zig
// From array (row-major order)
const mat = Mat3f.fromArray(&.{
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
});

// Identity matrix
const I = Mat3f.identity();

// From vector data
const vec = Vec3f.fromArray(&.{ 1, 2, 3 });
```

### Basic Operations

```zig
const A = Mat3f.fromArray(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
const B = Mat3f.fromArray(&.{ 9, 8, 7, 6, 5, 4, 3, 2, 1 });
const v = Vec3f.fromArray(&.{ 1, 2, 3 });

// Element-wise operations
const sum = A.add(B);
const diff = A.sub(B);
const prod = A.multiply(B);  // element-wise

// Transpose
const At = A.transpose();  // Returns Mat(f32, 3, 3) transposed

// Matrix-vector multiplication
const result = A.dot(v);      // Standard dot product
const fast = A.dotSIMD(v);    // SIMD-optimized (faster for larger matrices)

// Matrix-matrix multiplication
const C = A.dot(B);
const C_fast = A.dotSIMD(B);
```

### Vector Accessors

```zig
const v = Vec4f.fromArray(&.{ 1, 2, 3, 4 });

// Component access
const x = v.x();  // 1
const y = v.y();  // 2
const z = v.z();  // 3 (Vec3f, Vec4f only)
const w = v.w();  // 4 (Vec4f only)

// Swizzle
const v3 = Vec3f.fromArray(&.{ 1, 2, 3 });
const yzx = v3.sw("yzx");  // Vec3f{ 2, 3, 1 }
const xx = v3.sw("xx");    // Vec2f{ 1, 1 }
```

### Row Echelon Form

```zig
const A = Mat3f.fromArray(&.{
    2,  1, -1,
   -3, -1,  2,
   -2,  1,  2,
});

// Row echelon form (Gaussian elimination forward pass)
const ref = A.rowEchelonForm();

// Reduced row echelon form (Gauss-Jordan elimination)
const rref = A.reducedRowEchelonForm();
```

### QR Decomposition

```zig
const A = Mat3f.fromArray(&.{
    12, -51,   4,
     6, 167, -68,
    -4,  24, -41,
});

// QR decomposition: A = Q * R
const qr = A.qrDecompose();

// qr.Q is a 3x3 orthogonal matrix
// qr.R_mat is a 3x3 upper triangular matrix

// Verify: Q * R ≈ A
const reconstructed = qr.Q.dotSIMD(qr.R_mat);
```

### Solving Linear Systems

```zig
// Solve Ax = b using QR decomposition
const A = Mat3f.fromArray(&.{
    4, 2, 1,
    2, 5, 2,
    1, 2, 4,
});
const b = Vec3f.fromArray(&.{ 11, 18, 17 });

const x = A.solve(b);  // x ≈ [1, 2, 3]

// Solve for multiple right-hand sides: AX = B
const B = lmao.Matrix(f32, 3, 2).fromArray(&.{
    11, 14,
    18, 21,
    17, 20,
});
const X = A.solveMulti(2, B);  // Solves both systems at once
```

### Matrix Concatenation

```zig
const A = Mat3f.fromArray(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
const b = Vec3f.fromArray(&.{ 10, 11, 12 });

// Horizontal concatenation: [A | b]
const Ab = A.horzcat(b);  // Returns Matrix(f32, 3, 4)

// Useful for creating augmented matrices
const augmented = A.horzcat(b);  // For solving Ax = b via row reduction
```

### Custom Matrix Dimensions

```zig
// Create any RxC matrix
const Mat4x3 = lmao.Matrix(f32, 4, 3);
const Mat8x8 = lmao.Matrix(f64, 8, 8);

const m = Mat4x3.fromArray(&.{
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    10, 11, 12,
});

// Matrix multiplication respects dimensions
const v = lmao.Matrix(f32, 3, 1).fromArray(&.{ 1, 2, 3 });
const result = m.dotSIMD(v);  // Returns Matrix(f32, 4, 1)
```

### Converting to/from Arrays

```zig
const mat = Mat3f.fromArray(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });

// Get underlying array (row-major)
const arr: [9]f32 = mat.toArray();

// Access raw vector data
const vec: @Vector(9, f32) = mat.data;
```
