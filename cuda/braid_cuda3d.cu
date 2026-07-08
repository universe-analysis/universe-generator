// braid_cuda3d — GPU batch-reject RSA packing engine (3+1).
//
// Three spatial axes (x,y,w), each two-term slope-1; comoving by sin(z);
// collision = per-time-step Chebyshev <= CELL=2/T over a 3x3x3 neighbourhood.
// Flat dense 3D grid per time-step (comoving coords bounded to [-1,1]); fits a
// 24GB 3090 for the low/moderate T we need for the convergence question.
//
// --torus switches to the new-dogma model: the slope-1 budget binds the wiggle
// term alone (|a*b| = 1), the sin1 term is a free comoving offset, and the
// comoving domain is a period-2 torus (wrapped positions, minimum-image
// collision, no wall).
//
// --phase switches to the phase schema: the wiggle term gains a free phase
// f ~ U[0, pi) on even frequencies (odd ones must stay phase-free for the loop
// to close), each component is offset by a*sin(f) so it still starts at zero,
// and the z grid becomes the symmetric T-interior-points-of-(0, pi) indexing.
//
// --terms K generalizes each axis to K sinusoid terms: the sin1 term plus
// K-1 wiggle terms (K=2 is the legacy model, bit-identical RNG stream). Each
// axis draws K-1 unique integer frequencies; the unit slope budget is split
// uniformly at random across the wiggle group (simplex), each amplitude its
// share divided by its frequency, so the group satisfies sum |a_j*b_j| = 1
// exactly. Every term carries its own even-frequency phase and its own
// a*sin(f) offset. Mirrors the 2+1 engine's --terms.
//
// --sparse replaces the dense per-timestep grid (T*gw^3 cells, the T^4 VRAM
// hog) with per-timestep sorted occupied-cell keys + CSR offsets, looked up by
// binary search, and runs the whole prefilter in fp32: device points stored as
// float32 and candidate trajectories evaluated from precomputed sin(b*z)
// tables, no trig in the kernel (GeForce fp64 is 1/64 throughput). VRAM
// becomes O(N*T), so the T ceiling is set by runtime rather than card memory.
// The fp32 prefilter makes only CERTAIN decisions (hit threshold shrunk, wall
// pushed out, each by more than the fp32 error); gray-zone candidates are
// settled by the exact double-precision host recheck, so the admitted packing
// obeys the same collision rule as the dense path.
//
// The main loop is pipelined: two survivor buffers on one CUDA stream, so
// round r+1's kernel overlaps the host-side admission of round r. The device
// grid lags admissions by a round (on top of the 1%-growth rebuild filter);
// points are never removed, so the stale grid is a subset of the current
// packing and the prefilter can only pass extra candidates -- all settled by
// the authoritative host recheck.
//
// Host work is multithreaded (plain std::thread): survivor admission runs a
// parallel collision precheck (serial only for same-chunk mutual collisions,
// preserving RSA admission order exactly), and the sparse rebuild sorts only
// the points admitted since the last rebuild, merging them into a persistent
// per-timestep sorted order with all timesteps in parallel.
//
// Build:  nvcc -O3 -arch=sm_86 -o braid_cuda3d braid_cuda3d.cu
// Run:    ./braid_cuda3d -t 80 --attempts 5e9 --seed 1 --curve out.csv

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e = (x);                                                               \
        if (e) {                                                                           \
            fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

struct Rng {
    uint64_t s[4];
};
__host__ __device__ inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}
__host__ __device__ inline void rng_seed(Rng& r, uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        r.s[i] = z ^ (z >> 31);
    }
}
__host__ __device__ inline uint64_t rng_next(Rng& r) {
    uint64_t result = rotl(r.s[1] * 5, 7) * 9;
    uint64_t t = r.s[1] << 17;
    r.s[2] ^= r.s[0];
    r.s[3] ^= r.s[1];
    r.s[1] ^= r.s[2];
    r.s[0] ^= r.s[3];
    r.s[2] ^= t;
    r.s[3] = rotl(r.s[3], 45);
    return result;
}
__host__ __device__ inline double rng_f64(Rng& r) {
    return (double)(rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}
__host__ __device__ inline uint32_t rng_below(Rng& r, uint32_t n) {
    return (uint32_t)(rng_next(r) % n);
}
__host__ __device__ inline bool rng_flip(Rng& r) {
    return rng_next(r) & 1ULL;
}
__host__ __device__ inline uint32_t igcd(uint32_t a, uint32_t b) {
    while (b) {
        uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

constexpr double kPi = 3.14159265358979323846;

// Chunked parallel-for over [0, n): fn(lo, hi) runs on worker threads over
// disjoint ranges. Plain std::thread (no OpenMP) so the heterogeneous fleet's
// nvcc host-compiler combos all build it. Small n runs inline.
template <typename Fn>
void parallel_for(size_t n, const Fn& fn) {
    const unsigned hw = std::thread::hardware_concurrency();
    const size_t nt = std::min<size_t>(hw ? hw : 4, 16);
    if (n < 2 * nt) {
        fn(0, n);
        return;
    }
    std::vector<std::thread> ts;
    const size_t chunk = (n + nt - 1) / nt;
    for (size_t t = 0; t < nt; t++) {
        const size_t lo = t * chunk;
        const size_t hi = std::min(n, lo + chunk);
        if (lo >= hi)
            break;
        ts.emplace_back([&fn, lo, hi]() { fn(lo, hi); });
    }
    for (auto& th : ts)
        th.join();
}

// Max wiggle terms per axis (--terms counts sin1, so terms <= kMaxWiggle + 1).
constexpr int kMaxWiggle = 9;

struct Path {
    double ax[kMaxWiggle], ay[kMaxWiggle], aw[kMaxWiggle];  // wiggle amplitudes
    double bx[kMaxWiggle], by[kMaxWiggle], bw[kMaxWiggle];  // wiggle frequencies (integer)
    double fx[kMaxWiggle], fy[kMaxWiggle], fw[kMaxWiggle];  // phases (0 unless --phase + even)
    double ax2, ay2, aw2;                                   // sin1 amplitudes
};

// One biased-or-uniform frequency draw in [2, modmax+1] (smart = max of two
// uniforms, favouring finer wiggles).
__host__ __device__ inline uint32_t draw_freq(Rng& r, uint32_t modmax, bool smart) {
    if (!smart)
        return rng_below(r, modmax) + 2;
    uint32_t a1 = rng_below(r, modmax), a2 = rng_below(r, modmax);
    return (a1 > a2 ? a1 : a2) + 2;
}

// nw unique frequencies for one axis. Duplicates re-draw (bounded); if the
// guard blows, the smallest unused frequency fills the slot deterministically.
__host__ __device__ inline void draw_unique_freqs(
    Rng& r, uint32_t modmax, bool smart, int nw, uint32_t* out) {
    for (int j = 0; j < nw; j++) {
        uint32_t b = 0;
        bool dup = true;
        for (int guard = 0; dup && guard < 64; guard++) {
            b = draw_freq(r, modmax, smart);
            dup = false;
            for (int k = 0; k < j; k++)
                if (out[k] == b)
                    dup = true;
        }
        if (dup) {
            for (uint32_t c = 2; c <= modmax + 1; c++) {
                bool used = false;
                for (int k = 0; k < j; k++)
                    if (out[k] == c)
                        used = true;
                if (!used) {
                    b = c;
                    break;
                }
            }
        }
        out[j] = b;
    }
}

// Split 1 into nw parts, uniform on the simplex: nw-1 sorted cuts of [0,1],
// the parts are the gaps between consecutive cuts.
__host__ __device__ inline void split_unit(Rng& r, int nw, double* w) {
    double cuts[kMaxWiggle + 1];
    cuts[0] = 0.0;
    for (int j = 1; j < nw; j++)
        cuts[j] = rng_f64(r);
    cuts[nw] = 1.0;
    for (int j = 2; j < nw; j++) {  // insertion sort of cuts[1..nw-1]
        double v = cuts[j];
        int k = j;
        while (k > 1 && cuts[k - 1] > v) {
            cuts[k] = cuts[k - 1];
            k--;
        }
        cuts[k] = v;
    }
    for (int j = 0; j < nw; j++)
        w[j] = cuts[j + 1] - cuts[j];
}

// Edge-weighted draw for a sin1-component magnitude in (0,1]. theta ~ U(-pi,pi)
// folded through asin gives a density that rises toward 1 (the comoving edge)
// and vanishes at 0 (the centre) -- the opposite of the flat uniform draw. The
// sign is applied later by rng_flip, so only the magnitude is generated here.
// (u = 0 maps to 1, a degenerate wall-hugging path that the edge wall rejects.)
__host__ __device__ inline double angle_magnitude(Rng& r) {
    const double TWO_OVER_PI = 0.63661977236758134308;
    return 1.0 - TWO_OVER_PI * asin(rng_f64(r));
}

// Torus (new-dogma) model helpers. The comoving domain is periodic with period
// 2: a coordinate is wrapped onto the fundamental domain [-1, 1), and
// separations use the minimum image on the circle.
__host__ __device__ inline double torus_wrap(double x) {
    return x - 2.0 * floor((x + 1.0) * 0.5);
}
__host__ __device__ inline double torus_delta(double d) {
    // Inputs are wrapped, so |d| < 2 and one shift suffices.
    if (d > 1.0)
        return d - 2.0;
    if (d < -1.0)
        return d + 2.0;
    return d;
}
__device__ inline float torus_delta_f(float d) {
    if (d > 1.0f)
        return d - 2.0f;
    if (d < -1.0f)
        return d + 2.0f;
    return d;
}
__device__ inline float torus_wrap_f(float x) {
    return x - 2.0f * floorf((x + 1.0f) * 0.5f);
}

// Binary search over a per-timestep span [lo, hi) of the sorted occupied-cell
// key array (sparse grid). Returns the index of `key`, or -1 if the cell holds
// no points.
__device__ inline int find_key(const int* keys, int lo, int hi, int key) {
    const int hi0 = hi;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (keys[mid] < key)
            lo = mid + 1;
        else
            hi = mid;
    }
    return (lo < hi0 && keys[lo] == key) ? lo : -1;
}

__host__ __device__ inline Path propose(
    Rng& r, uint32_t modmax, bool smart, bool angle, bool torus, bool phase, int nw) {
    Path p = {};
    if (nw == 1) {
        // Legacy single-wiggle model: draw order preserved verbatim so the RNG
        // stream (and thus candidate generation) stays bit-identical to the
        // pre---terms engine.
        double xs, ys, ws;
        if (angle) {
            xs = angle_magnitude(r);
            ys = angle_magnitude(r);
            ws = angle_magnitude(r);
        } else {
            xs = rng_f64(r);
            ys = rng_f64(r);
            ws = rng_f64(r);
        }
        uint32_t bx, by, bw;
        if (!smart) {
            bx = rng_below(r, modmax) + 2;
            by = rng_below(r, modmax) + 2;
            bw = rng_below(r, modmax) + 2;
        } else {
            int guard = 0;
            while (true) {
                uint32_t a1 = rng_below(r, modmax), a2 = rng_below(r, modmax);
                bx = (a1 > a2 ? a1 : a2) + 2;
                uint32_t b1 = rng_below(r, modmax), b2 = rng_below(r, modmax);
                by = (b1 > b2 ? b1 : b2) + 2;
                uint32_t c1 = rng_below(r, modmax), c2 = rng_below(r, modmax);
                bw = (c1 > c2 ? c1 : c2) + 2;
                if ((igcd(bx, by) == 1 && igcd(bx, bw) == 1 && igcd(by, bw) == 1) ||
                    guard >= 24)
                    break;
                guard++;
            }
        }
        p.ax2 = xs;
        p.ay2 = ys;
        p.aw2 = ws;
        if (torus) {
            // New-dogma budget: the slope-1 constraint binds the wiggle term
            // alone (|a*b| = 1); the sin1 term is a free comoving offset -- in
            // comoving coordinates it is a CONSTANT, so a uniform draw makes
            // the packing homogeneous on the torus by construction.
            p.ax[0] = 1.0 / bx;
            p.ay[0] = 1.0 / by;
            p.aw[0] = 1.0 / bw;
        } else {
            p.ax[0] = (1.0 - xs) / bx;
            p.ay[0] = (1.0 - ys) / by;
            p.aw[0] = (1.0 - ws) / bw;
        }
        p.bx[0] = bx;
        p.by[0] = by;
        p.bw[0] = bw;
        if (rng_flip(r))
            p.ax[0] = -p.ax[0];
        if (rng_flip(r))
            p.ay[0] = -p.ay[0];
        if (rng_flip(r))
            p.aw[0] = -p.aw[0];
        if (rng_flip(r))
            p.ax2 = -p.ax2;
        if (rng_flip(r))
            p.ay2 = -p.ay2;
        if (rng_flip(r))
            p.aw2 = -p.aw2;
        // Phase update: a free phase on the wiggle term, drawn only for EVEN
        // frequencies -- the loop closes at z=pi only where sin(b*pi + f)
        // equals sin(f), so odd frequencies stay phase-free. The a*sin(f)
        // offset applied at evaluation time re-pins the component to zero at
        // the endpoints. Drawn last so the baseline (phase off) RNG stream is
        // untouched.
        if (phase) {
            if (bx % 2 == 0)
                p.fx[0] = rng_f64(r) * kPi;
            if (by % 2 == 0)
                p.fy[0] = rng_f64(r) * kPi;
            if (bw % 2 == 0)
                p.fw[0] = rng_f64(r) * kPi;
        }
        return p;
    }
    // Generalized multi-term model. Smart keeps the high-frequency bias per
    // draw but drops the cross-axis coprime rule (no single pair to test).
    double xs, ys, ws;
    if (angle) {
        xs = angle_magnitude(r);
        ys = angle_magnitude(r);
        ws = angle_magnitude(r);
    } else {
        xs = rng_f64(r);
        ys = rng_f64(r);
        ws = rng_f64(r);
    }
    p.ax2 = xs;
    p.ay2 = ys;
    p.aw2 = ws;
    uint32_t bxa[kMaxWiggle], bya[kMaxWiggle], bwa[kMaxWiggle];
    draw_unique_freqs(r, modmax, smart, nw, bxa);
    draw_unique_freqs(r, modmax, smart, nw, bya);
    draw_unique_freqs(r, modmax, smart, nw, bwa);
    double wx[kMaxWiggle], wy[kMaxWiggle], ww[kMaxWiggle];
    split_unit(r, nw, wx);
    split_unit(r, nw, wy);
    split_unit(r, nw, ww);
    // Torus: the whole unit budget goes to the wiggle group (sum |a*b| = 1).
    // Hard wall: the group shares the slope budget with sin1, as before.
    double budx = torus ? 1.0 : (1.0 - xs);
    double budy = torus ? 1.0 : (1.0 - ys);
    double budw = torus ? 1.0 : (1.0 - ws);
    for (int j = 0; j < nw; j++) {
        p.ax[j] = budx * wx[j] / bxa[j];
        p.bx[j] = bxa[j];
        p.ay[j] = budy * wy[j] / bya[j];
        p.by[j] = bya[j];
        p.aw[j] = budw * ww[j] / bwa[j];
        p.bw[j] = bwa[j];
    }
    for (int j = 0; j < nw; j++)
        if (rng_flip(r))
            p.ax[j] = -p.ax[j];
    for (int j = 0; j < nw; j++)
        if (rng_flip(r))
            p.ay[j] = -p.ay[j];
    for (int j = 0; j < nw; j++)
        if (rng_flip(r))
            p.aw[j] = -p.aw[j];
    if (rng_flip(r))
        p.ax2 = -p.ax2;
    if (rng_flip(r))
        p.ay2 = -p.ay2;
    if (rng_flip(r))
        p.aw2 = -p.aw2;
    // Per-term phases, same rule as the single-wiggle model: even frequencies
    // only, and each term is re-pinned to zero by its own a*sin(f) offset.
    if (phase) {
        for (int j = 0; j < nw; j++)
            if (bxa[j] % 2 == 0)
                p.fx[j] = rng_f64(r) * kPi;
        for (int j = 0; j < nw; j++)
            if (bya[j] % 2 == 0)
                p.fy[j] = rng_f64(r) * kPi;
        for (int j = 0; j < nw; j++)
            if (bwa[j] % 2 == 0)
                p.fw[j] = rng_f64(r) * kPi;
    }
    return p;
}

// Dense-grid collision test: exact double-precision trajectory + Chebyshev /
// L2 exclusion. This is the reference prefilter -- bit-compatible with the
// historical engine.
__device__ bool collides_dev(const Path& p,
                             int nw,
                             int T,
                             const double* z,
                             const double* sinz,
                             const double* invz,
                             double cell,
                             int gw,
                             int off,
                             long gw2,
                             long gw3,
                             const int* cellStart,
                             const double* ptsX,
                             const double* ptsY,
                             const double* ptsW,
                             const int* order,
                             bool euclid,
                             bool torus) {
    const double edge = 1.0 - 0.5 * cell;  // path radius CELL/2 hits the wall at |X|=1
    // Phase offsets re-pin each component to zero at the endpoints.
    double offx[kMaxWiggle], offy[kMaxWiggle], offw[kMaxWiggle];
    for (int j = 0; j < nw; j++) {
        offx[j] = p.ax[j] * sin(p.fx[j]);
        offy[j] = p.ay[j] * sin(p.fy[j]);
        offw[j] = p.aw[j] * sin(p.fw[j]);
    }
    // Phase-free single-wiggle paths take the verbatim pre-phase expression, so
    // their per-candidate math is bit-identical to the pre-phase engine
    // (folding a zero phase into the expression shifts FP contraction by an
    // ulp). Full-run output is still scheduling-dependent: survivor admission
    // order comes from atomicAdd slots.
    const bool phased = (p.fx[0] != 0.0) || (p.fy[0] != 0.0) || (p.fw[0] != 0.0);

    for (int oi = 0; oi < T; oi++) {
        const int i = order[oi];

        // Comoving position of the candidate path at this timestep (X = x / sin z).
        double X, Y, W;
        if (nw > 1) {
            double xx = p.ax2 * sinz[i], yy = p.ay2 * sinz[i], wwv = p.aw2 * sinz[i];
            for (int j = 0; j < nw; j++) {
                xx += p.ax[j] * sin(p.bx[j] * z[i] + p.fx[j]) - offx[j];
                yy += p.ay[j] * sin(p.by[j] * z[i] + p.fy[j]) - offy[j];
                wwv += p.aw[j] * sin(p.bw[j] * z[i] + p.fw[j]) - offw[j];
            }
            X = xx * invz[i];
            Y = yy * invz[i];
            W = wwv * invz[i];
        } else if (phased) {
            X = (p.ax[0] * sin(p.bx[0] * z[i] + p.fx[0]) + p.ax2 * sinz[i] - offx[0]) * invz[i];
            Y = (p.ay[0] * sin(p.by[0] * z[i] + p.fy[0]) + p.ay2 * sinz[i] - offy[0]) * invz[i];
            W = (p.aw[0] * sin(p.bw[0] * z[i] + p.fw[0]) + p.aw2 * sinz[i] - offw[0]) * invz[i];
        } else {
            X = (p.ax[0] * sin(p.bx[0] * z[i]) + p.ax2 * sinz[i]) * invz[i];
            Y = (p.ay[0] * sin(p.by[0] * z[i]) + p.ay2 * sinz[i]) * invz[i];
            W = (p.aw[0] * sin(p.bw[0] * z[i]) + p.aw2 * sinz[i]) * invz[i];
        }

        int cx, cy, cw;
        if (torus) {
            // Periodic domain: wrap onto [-1, 1); there is no wall.
            X = torus_wrap(X);
            Y = torus_wrap(Y);
            W = torus_wrap(W);
            cx = (int)floor((X + 1.0) / cell);
            cy = (int)floor((Y + 1.0) / cell);
            cw = (int)floor((W + 1.0) / cell);
        } else {
            // (1) Edge collision: the boundary |X| = 1 is a hard wall; a path whose
            //     centre is within radius CELL/2 of it has its body crossing the wall.
            if (fabs(X) > edge || fabs(Y) > edge || fabs(W) > edge)
                return true;
            cx = (int)floor(X / cell);
            cy = (int)floor(Y / cell);
            cw = (int)floor(W / cell);
        }

        // (2) Path-path collision: scan the 3x3x3 neighbourhood of grid cells and
        //     test every accepted point for Chebyshev distance <= CELL. On the
        //     torus the neighbourhood wraps modulo the grid.
        for (int dx = -1; dx <= 1; dx++) {
            int gx = torus ? (cx + dx + gw) % gw : cx + dx + off;
            if (gx < 0 || gx >= gw)
                continue;

            for (int dy = -1; dy <= 1; dy++) {
                int gy = torus ? (cy + dy + gw) % gw : cy + dy + off;
                if (gy < 0 || gy >= gw)
                    continue;

                for (int dz = -1; dz <= 1; dz++) {
                    int gz = torus ? (cw + dz + gw) % gw : cw + dz + off;
                    if (gz < 0 || gz >= gw)
                        continue;

                    // CSR with a sentinel: cell gc holds points
                    // [cellStart[gc], cellStart[gc+1]).
                    const long gc = (long)i * gw3 + (long)gx * gw2 + (long)gy * gw + gz;
                    const long st = cellStart[gc];
                    const int ln = cellStart[gc + 1] - (int)st;

                    for (int k = 0; k < ln; k++) {
                        double dX = X - ptsX[st + k];
                        double dY = Y - ptsY[st + k];
                        double dW = W - ptsW[st + k];
                        if (torus) {
                            dX = torus_delta(dX);
                            dY = torus_delta(dY);
                            dW = torus_delta(dW);
                        }
                        bool hit;
                        if (euclid)
                            hit = dX * dX + dY * dY + dW * dW <= cell * cell;
                        else
                            hit = fabs(dX) <= cell && fabs(dY) <= cell && fabs(dW) <= cell;
                        if (hit)
                            return true;
                    }
                }
            }
        }
    }
    return false;
}

// Sparse-grid collision test, all in fp32 with NO trig in the loop: the
// trajectory comes from precomputed tables (sinbT[(b-2)*T + i] = sin(b*z_i);
// for the phase schema also cosbm1T = cos(b*z_i) - 1, stored pre-subtracted
// in double so the near-endpoint cancellation costs no precision). GeForce
// cards run fp64 at 1/64 throughput, so this is the difference between a
// trig-bound and a memory-bound kernel.
//
// Decisions stay one-sided so the exact double host recheck remains the
// authority: a hit is flagged only when CERTAIN (<= cellTight, the exclusion
// shrunk by more than the fp32 trajectory error), and the wall rejects only
// when CERTAIN (> edgeLoose, the wall pushed out by the same margin).
// Gray-zone candidates survive the prefilter and are settled on the host.
__device__ bool collides_sparse_dev(const Path& p,
                                    int nw,
                                    int T,
                                    const float* sinbT,
                                    const float* cosbm1T,
                                    const float* sinzF,
                                    const float* invzF,
                                    float cellF,
                                    float cellTightF,
                                    float edgeLooseF,
                                    int gw,
                                    int off,
                                    long gw2,
                                    const int* keys,
                                    const int* keyStart,
                                    const int* cellOff,
                                    const float* ptsXf,
                                    const float* ptsYf,
                                    const float* ptsWf,
                                    const int* order,
                                    bool euclid,
                                    bool torus,
                                    bool phase) {
    const float ax2 = (float)p.ax2, ay2 = (float)p.ay2, aw2 = (float)p.aw2;
    // Per-term amplitudes, table offsets, and phase factors (once, not per
    // timestep). With the a*sin(f) offset folded in,
    // sin(b*z + f) - sin(f) = sinb*cos(f) + (cosb - 1)*sin(f).
    float axF[kMaxWiggle], ayF[kMaxWiggle], awF[kMaxWiggle];
    int obx[kMaxWiggle], oby[kMaxWiggle], obw[kMaxWiggle];
    float cfx[kMaxWiggle], sfx[kMaxWiggle];
    float cfy[kMaxWiggle], sfy[kMaxWiggle];
    float cfw[kMaxWiggle], sfw[kMaxWiggle];
    for (int j = 0; j < nw; j++) {
        axF[j] = (float)p.ax[j];
        ayF[j] = (float)p.ay[j];
        awF[j] = (float)p.aw[j];
        obx[j] = ((int)p.bx[j] - 2) * T;
        oby[j] = ((int)p.by[j] - 2) * T;
        obw[j] = ((int)p.bw[j] - 2) * T;
        cfx[j] = cosf((float)p.fx[j]);
        sfx[j] = sinf((float)p.fx[j]);
        cfy[j] = cosf((float)p.fy[j]);
        sfy[j] = sinf((float)p.fy[j]);
        cfw[j] = cosf((float)p.fw[j]);
        sfw[j] = sinf((float)p.fw[j]);
    }

    for (int oi = 0; oi < T; oi++) {
        const int i = order[oi];

        float X, Y, W;
        if (nw == 1) {
            // Single-wiggle fast path: same expressions as the pre---terms
            // engine (no per-term loop overhead in the legacy configuration).
            if (phase) {
                X = (axF[0] * (sinbT[obx[0] + i] * cfx[0] + cosbm1T[obx[0] + i] * sfx[0]) +
                     ax2 * sinzF[i]) *
                    invzF[i];
                Y = (ayF[0] * (sinbT[oby[0] + i] * cfy[0] + cosbm1T[oby[0] + i] * sfy[0]) +
                     ay2 * sinzF[i]) *
                    invzF[i];
                W = (awF[0] * (sinbT[obw[0] + i] * cfw[0] + cosbm1T[obw[0] + i] * sfw[0]) +
                     aw2 * sinzF[i]) *
                    invzF[i];
            } else {
                X = (axF[0] * sinbT[obx[0] + i] + ax2 * sinzF[i]) * invzF[i];
                Y = (ayF[0] * sinbT[oby[0] + i] + ay2 * sinzF[i]) * invzF[i];
                W = (awF[0] * sinbT[obw[0] + i] + aw2 * sinzF[i]) * invzF[i];
            }
        } else {
            float xx = ax2 * sinzF[i], yy = ay2 * sinzF[i], wwv = aw2 * sinzF[i];
            if (phase) {
                for (int j = 0; j < nw; j++) {
                    xx += axF[j] * (sinbT[obx[j] + i] * cfx[j] + cosbm1T[obx[j] + i] * sfx[j]);
                    yy += ayF[j] * (sinbT[oby[j] + i] * cfy[j] + cosbm1T[oby[j] + i] * sfy[j]);
                    wwv += awF[j] * (sinbT[obw[j] + i] * cfw[j] + cosbm1T[obw[j] + i] * sfw[j]);
                }
            } else {
                for (int j = 0; j < nw; j++) {
                    xx += axF[j] * sinbT[obx[j] + i];
                    yy += ayF[j] * sinbT[oby[j] + i];
                    wwv += awF[j] * sinbT[obw[j] + i];
                }
            }
            X = xx * invzF[i];
            Y = yy * invzF[i];
            W = wwv * invzF[i];
        }

        int cx, cy, cw;
        if (torus) {
            X = torus_wrap_f(X);
            Y = torus_wrap_f(Y);
            W = torus_wrap_f(W);
            cx = (int)floorf((X + 1.0f) / cellF);
            cy = (int)floorf((Y + 1.0f) / cellF);
            cw = (int)floorf((W + 1.0f) / cellF);
        } else {
            if (fabsf(X) > edgeLooseF || fabsf(Y) > edgeLooseF || fabsf(W) > edgeLooseF)
                return true;
            cx = (int)floorf(X / cellF);
            cy = (int)floorf(Y / cellF);
            cw = (int)floorf(W / cellF);
        }

        // The fp32 cell index can differ from the exact one only for points a
        // float-error away from a cell boundary; the 3x3x3 scan still covers
        // every point within cellTight of the position, and boundary cases it
        // could miss are gray-zone by construction (host settles them).
        for (int dx = -1; dx <= 1; dx++) {
            int gx = torus ? (cx + dx + gw) % gw : cx + dx + off;
            if (gx < 0 || gx >= gw)
                continue;

            for (int dy = -1; dy <= 1; dy++) {
                int gy = torus ? (cy + dy + gw) % gw : cy + dy + off;
                if (gy < 0 || gy >= gw)
                    continue;

                for (int dz = -1; dz <= 1; dz++) {
                    int gz = torus ? (cw + dz + gw) % gw : cw + dz + off;
                    if (gz < 0 || gz >= gw)
                        continue;

                    const int c = (int)((long)gx * gw2 + (long)gy * gw + gz);
                    const int j = find_key(keys, keyStart[i], keyStart[i + 1], c);
                    if (j < 0)
                        continue;
                    const int st = cellOff[j];
                    const int ln = cellOff[j + 1] - st;

                    for (int k = 0; k < ln; k++) {
                        float dX = X - ptsXf[st + k];
                        float dY = Y - ptsYf[st + k];
                        float dW = W - ptsWf[st + k];
                        if (torus) {
                            dX = torus_delta_f(dX);
                            dY = torus_delta_f(dY);
                            dW = torus_delta_f(dW);
                        }
                        bool hit;
                        if (euclid)
                            hit = dX * dX + dY * dY + dW * dW <= cellTightF * cellTightF;
                        else
                            hit = fabsf(dX) <= cellTightF && fabsf(dY) <= cellTightF &&
                                  fabsf(dW) <= cellTightF;
                        if (hit)
                            return true;
                    }
                }
            }
        }
    }
    return false;
}

__global__ void test_kernel(uint64_t baseSeed,
                            uint64_t round,
                            int batch,
                            uint32_t modmax,
                            bool smart,
                            bool angle,
                            bool euclid,
                            bool torus,
                            bool phase,
                            int nw,
                            int T,
                            const double* z,
                            const double* sinz,
                            const double* invz,
                            double cell,
                            int gw,
                            int off,
                            long gw2,
                            long gw3,
                            const int* cellStart,
                            const double* ptsX,
                            const double* ptsY,
                            const double* ptsW,
                            const int* order,
                            Path* survOut,
                            int* survCount,
                            int survCap) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch)
        return;
    Rng r;
    rng_seed(r, baseSeed ^ (round * 0xD1B54A32D192ED03ULL) ^
                    ((uint64_t)tid * 0x9E3779B97F4A7C15ULL));
    Path p = propose(r, modmax, smart, angle, torus, phase, nw);
    if (!collides_dev(p, nw, T, z, sinz, invz, cell, gw, off, gw2, gw3, cellStart, ptsX, ptsY,
                      ptsW, order, euclid, torus)) {
        int slot = atomicAdd(survCount, 1);
        if (slot < survCap)
            survOut[slot] = p;
    }
}

__global__ void test_kernel_sparse(uint64_t baseSeed,
                                   uint64_t round,
                                   int batch,
                                   uint32_t modmax,
                                   bool smart,
                                   bool angle,
                                   bool euclid,
                                   bool torus,
                                   bool phase,
                                   int nw,
                                   int T,
                                   const float* sinbT,
                                   const float* cosbm1T,
                                   const float* sinzF,
                                   const float* invzF,
                                   float cellF,
                                   float cellTightF,
                                   float edgeLooseF,
                                   int gw,
                                   int off,
                                   long gw2,
                                   const int* keys,
                                   const int* keyStart,
                                   const int* cellOff,
                                   const float* ptsXf,
                                   const float* ptsYf,
                                   const float* ptsWf,
                                   const int* order,
                                   Path* survOut,
                                   int* survCount,
                                   int survCap) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch)
        return;
    Rng r;
    rng_seed(r, baseSeed ^ (round * 0xD1B54A32D192ED03ULL) ^
                    ((uint64_t)tid * 0x9E3779B97F4A7C15ULL));
    Path p = propose(r, modmax, smart, angle, torus, phase, nw);
    if (!collides_sparse_dev(p, nw, T, sinbT, cosbm1T, sinzF, invzF, cellF, cellTightF,
                             edgeLooseF, gw, off, gw2, keys, keyStart, cellOff, ptsXf, ptsYf,
                             ptsWf, order, euclid, torus, phase)) {
        int slot = atomicAdd(survCount, 1);
        if (slot < survCap)
            survOut[slot] = p;
    }
}

int main(int argc, char** argv) {
    int T = 80;
    double budget = 1e8;
    uint64_t seed = 12345;
    bool smart = true;
    const char* curvePath = nullptr;
    int maxfreq = 0;          // 0 = default modulation cap T/2
    double acceptThresh = 0;  // 0 = run to --attempts; >0 = stop when accept-rate < thresh
    bool angle = false;       // edge-weighted sin1 sampling (vs flat uniform)
    bool euclid = false;      // L2-ball exclusion (vs the default Chebyshev cube)
    bool torus = false;       // new-dogma model: |a*b|=1, free sin1 offset, periodic domain
    bool phase = false;       // phase schema: even-frequency phases + symmetric z grid
    bool sparse = false;      // sparse grid (sorted keys + float32 points): VRAM ~ N*T
    int terms = 2;            // total sinusoid terms per axis, incl. sin1 (2 = legacy)
    const char* diagPrefix = nullptr;  // if set, write occupancy + probe diagnostics
    long long probeN = 5000000;        // fresh proposals fired at the final state
    const char* paramPath = nullptr;   // if set, dump accepted worldline parameters
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-t") || !strcmp(argv[i], "--timesteps"))
            T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--attempts"))
            budget = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed"))
            seed = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--uniform"))
            smart = false;
        else if (!strcmp(argv[i], "--smart"))
            smart = true;
        else if (!strcmp(argv[i], "--maxfreq"))
            maxfreq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--curve"))
            curvePath = argv[++i];
        else if (!strcmp(argv[i], "--until-accept-rate"))
            acceptThresh = atof(argv[++i]);
        else if (!strcmp(argv[i], "--angle-sample"))
            angle = true;
        else if (!strcmp(argv[i], "--euclid-collision"))
            euclid = true;
        else if (!strcmp(argv[i], "--torus"))
            torus = true;
        else if (!strcmp(argv[i], "--phase"))
            phase = true;
        else if (!strcmp(argv[i], "--sparse"))
            sparse = true;
        else if (!strcmp(argv[i], "--terms"))
            terms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--diag"))
            diagPrefix = argv[++i];
        else if (!strcmp(argv[i], "--probe-n"))
            probeN = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--dump-params"))
            paramPath = argv[++i];
    }
    // Fixed-convergence stop: hold acceptance rate (not attempts) constant across T.
    std::deque<std::pair<long long, long long>> window;
    long long winAtt = 0, winAdm = 0;
    long long winTarget =
        acceptThresh > 0 ? (long long)std::min(5e9, std::max(2e7, 30.0 / acceptThresh)) : 0;
    const double PI = 3.14159265358979323846;
    // --maxfreq F sets the ACTUAL max frequency to F (freq draw is [2, modmax+1], so
    // modmax=F-1). Default (no flag) keeps the model's T/2+1 cap (modmax=T/2).
    double cell = 2.0 / T;
    // Sparse prefilter margins. The fp32 table trajectory carries a worst-case
    // absolute error of a few 1e-7 (the physical terms scale with sin z, so
    // the 1/sin z endpoint amplification cancels; the phase schema's
    // (cos(bz) - 1)*sin(f) term, stored pre-subtracted, stays at the same
    // scale). 2e-5 is ~40x that bound and still only 0.3% of a cell at T=300.
    // Decisions are one-sided: a hit is flagged only when CERTAIN (within
    // cell - margin), the wall rejects only when CERTAIN (beyond
    // edge + margin); the (cell - margin, cell] gray zone survives the
    // prefilter and is settled by the exact double-precision host recheck, so
    // the admitted packing obeys the same rule as the dense path.
    const double kF32Margin = 2e-5;
    const double cellTight = cell - kF32Margin;
    const double edgeLoose = 1.0 - 0.5 * cell + kF32Margin;
    uint32_t modmax =
        (maxfreq > 2) ? (uint32_t)(maxfreq - 1) : (uint32_t)(T / 2 > 2 ? T / 2 : 2);
    // Wiggle-term count: --terms counts sin1, so nw = terms-1. Capped by the
    // Path struct (kMaxWiggle) and by the pool of unique frequencies (modmax).
    if (terms < 2)
        terms = 2;
    if (terms - 1 > kMaxWiggle) {
        fprintf(stderr, "--terms %d exceeds kMaxWiggle+1 = %d, clamping\n", terms,
                kMaxWiggle + 1);
        terms = kMaxWiggle + 1;
    }
    if ((uint32_t)(terms - 1) > modmax) {
        fprintf(stderr, "--terms %d exceeds unique-frequency pool %u, clamping\n", terms,
                modmax);
        terms = (int)modmax + 1;
    }
    const int nw = terms - 1;
    // Torus grid: exactly T cells of width CELL span [-1, 1), so the modular
    // neighbour scan wraps cleanly at the seam. Hard-wall grid keeps its margin.
    int gw = torus ? T : T + 4, off = torus ? 0 : (T + 4) / 2;
    long gw2 = (long)gw * gw, gw3 = gw2 * gw;
    // z tables + endpoint-first order. The phase schema replaces the hardcoded
    // 0.01 endpoint clamp with a symmetric grid: T interior points of (0, pi),
    // one step in from each end (matches the viewer's z indexing).
    std::vector<double> z(T), sinz(T), invz(T);
    std::vector<int> order(T);
    double z0, step;
    if (phase) {
        step = PI / (T + 1);
        z0 = step;
    } else {
        z0 = 0.01;
        step = (PI - 0.01 - z0) / (T - 1);
    }
    for (int i = 0; i < T; i++) {
        z[i] = z0 + i * step;
        sinz[i] = sin(z[i]);
        invz[i] = 1.0 / sinz[i];
        order[i] = i;
    }
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return std::min(a, T - 1 - a) < std::min(b, T - 1 - b); });

    double *dz, *dsinz, *dinvz;
    int* dorder;
    CK(cudaMalloc(&dz, T * 8));
    CK(cudaMalloc(&dsinz, T * 8));
    CK(cudaMalloc(&dinvz, T * 8));
    CK(cudaMalloc(&dorder, T * 4));
    CK(cudaMemcpy(dz, z.data(), T * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsinz, sinz.data(), T * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dinvz, invz.data(), T * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dorder, order.data(), T * 4, cudaMemcpyHostToDevice));

    // Sparse-mode frequency tables: the kernel evaluates trajectories from
    // sinbT[(b-2)*T + i] = sin(b * z_i) (fp32, computed in double) instead of
    // calling sin() -- modmax x T floats, small enough to live in L2. The
    // phase schema also needs cosbm1T = cos(b * z_i) - 1, pre-subtracted in
    // double so the near-endpoint cancellation costs no precision.
    float *dSinbT = nullptr, *dCosbm1T = nullptr, *dSinzF = nullptr, *dInvzF = nullptr;
    if (sparse) {
        std::vector<float> sinbT((size_t)modmax * T), sinzF(T), invzF(T);
        std::vector<float> cosbm1T(phase ? (size_t)modmax * T : 0);
        for (uint32_t b = 2; b <= modmax + 1; b++)
            for (int i = 0; i < T; i++) {
                sinbT[(size_t)(b - 2) * T + i] = (float)sin((double)b * z[i]);
                if (phase)
                    cosbm1T[(size_t)(b - 2) * T + i] = (float)(cos((double)b * z[i]) - 1.0);
            }
        for (int i = 0; i < T; i++) {
            sinzF[i] = (float)sinz[i];
            invzF[i] = (float)invz[i];
        }
        CK(cudaMalloc(&dSinbT, sinbT.size() * 4));
        CK(cudaMemcpy(dSinbT, sinbT.data(), sinbT.size() * 4, cudaMemcpyHostToDevice));
        if (phase) {
            CK(cudaMalloc(&dCosbm1T, cosbm1T.size() * 4));
            CK(cudaMemcpy(dCosbm1T, cosbm1T.data(), cosbm1T.size() * 4,
                          cudaMemcpyHostToDevice));
        }
        CK(cudaMalloc(&dSinzF, T * 4));
        CK(cudaMalloc(&dInvzF, T * 4));
        CK(cudaMemcpy(dSinzF, sinzF.data(), T * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dInvzF, invzF.data(), T * 4, cudaMemcpyHostToDevice));
    }

    size_t ncell = (size_t)T * gw3;
    if (sparse)
        fprintf(stderr,
                "3+1%s%s (sparse): T=%d  modmax=%u (maxfreq=%u)  terms=%d  grid VRAM ~ N*T\n",
                torus ? " (torus)" : "", phase ? " (phase)" : "", T, modmax, modmax + 1, terms);
    else
        fprintf(stderr,
                "3+1%s%s: T=%d  modmax=%u (maxfreq=%u)  terms=%d  grid cells=%.2e  "
                "(~%.1f GB)\n",
                torus ? " (torus)" : "", phase ? " (phase)" : "", T, modmax, modmax + 1, terms,
                (double)ncell, (ncell + 1) * 4.0 / 1e9);
    // Dense grid: cellStart holds cumulative point offsets (< N*T, well within
    // int32 even at T=300), so a 32-bit grid index halves device memory vs a
    // 64-bit one. One sentinel entry (cellStart[ncell] = npts) makes lengths
    // derivable as cellStart[gc+1] - cellStart[gc], so no separate length array
    // is stored. In sparse mode the dense arrays are never allocated (they are
    // the T^4 memory hog, on host and device alike).
    std::vector<int> cellStart, cellLen;
    int* dCellStart = nullptr;
    if (!sparse) {
        cellStart.resize(ncell + 1);
        cellLen.resize(ncell);  // host-side scratch for the counting pass
        CK(cudaMalloc(&dCellStart, (ncell + 1) * 4));
    }
    // Sparse grid: per timestep, the sorted keys of occupied cells (linear cell
    // index within the timestep; gw^3 fits int32 comfortably to beyond T=1000)
    // plus a CSR offset per key into the point arrays. keyStart[i] delimits
    // timestep i's span of keys; cellOff gets one global sentinel (= npts).
    std::vector<int> keysH, keyStartH(T + 1), cellOffH;
    int *dKeys = nullptr, *dKeyStart = nullptr, *dCellOff = nullptr;
    if (sparse)
        CK(cudaMalloc(&dKeyStart, (T + 1) * 4));

    std::vector<std::vector<double>> px(T), py(T), pw(T);
    // Incremental sparse-rebuild state: per timestep, the point indices sorted
    // by cell key (kept across rebuilds so only fresh points need sorting),
    // plus per-timestep key/offset scratch for the parallel emit.
    std::vector<std::vector<int>> sortedCell(T), sortedIdx(T);
    std::vector<std::vector<int>> tkeys(T), toffs(T);
    std::vector<Path> acceptedPaths;  // populated only when --dump-params is set
    std::vector<std::unordered_map<long, std::vector<std::array<double, 3>>>> hgrid(T);
    auto key3 = [&](int cx, int cy, int cw) -> long {
        return ((long)cx * 100003L + cy) * 100003L + cw;
    };
    double edge = 1.0 - 0.5 * cell;  // path radius CELL/2 hits the hard wall at |X|=1
    // Host hash-grid cell index. Torus: wrapped positions live in [-1,1), so the
    // index is periodic in [0, gw); neighbours wrap modulo gw. Hard wall: raw
    // floor(v/cell) (the hash key handles negatives).
    auto hix = [&](double v) -> int {
        return torus ? (int)floor((v + 1.0) / cell) : (int)floor(v / cell);
    };
    auto hnb = [&](int c, int d) -> int { return torus ? (c + d + gw) % gw : c + d; };
    auto host_collides = [&](const double* X, const double* Y, const double* W) -> bool {
        for (int oi = 0; oi < T; oi++) {
            int i = order[oi];
            if (!torus && (fabs(X[i]) > edge || fabs(Y[i]) > edge || fabs(W[i]) > edge))
                return true;  // edge collision
            int cx = hix(X[i]), cy = hix(Y[i]), cw = hix(W[i]);
            for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++)
                    for (int dz = -1; dz <= 1; dz++) {
                        auto it = hgrid[i].find(key3(hnb(cx, dx), hnb(cy, dy), hnb(cw, dz)));
                        if (it == hgrid[i].end())
                            continue;
                        for (auto& pt : it->second) {
                            double dX = X[i] - pt[0];
                            double dY = Y[i] - pt[1];
                            double dW = W[i] - pt[2];
                            if (torus) {
                                dX = torus_delta(dX);
                                dY = torus_delta(dY);
                                dW = torus_delta(dW);
                            }
                            bool hit;
                            if (euclid)
                                hit = dX * dX + dY * dY + dW * dW <= cell * cell;
                            else
                                hit = fabs(dX) <= cell && fabs(dY) <= cell && fabs(dW) <= cell;
                            if (hit)
                                return true;
                        }
                    }
        }
        return false;
    };
    // Device-grid cell index for the CSR rebuild (must mirror collides_dev).
    auto gix = [&](double v) -> long {
        return torus ? (long)floor((v + 1.0) / cell) : (long)((int)floor(v / cell) + off);
    };
    // Comoving trajectory of a path at every timestep (wrapped on the torus).
    // Same expressions as collides_dev; thread-safe (read-only captures).
    auto eval_path = [&](const Path& p, double* X, double* Y, double* W) {
        double offx[kMaxWiggle], offy[kMaxWiggle], offw[kMaxWiggle];
        for (int j = 0; j < nw; j++) {
            offx[j] = p.ax[j] * sin(p.fx[j]);
            offy[j] = p.ay[j] * sin(p.fy[j]);
            offw[j] = p.aw[j] * sin(p.fw[j]);
        }
        // Same rule as collides_dev: phase-free single-wiggle paths take the
        // verbatim pre-phase expression (per-candidate math identical).
        const bool phased = (p.fx[0] != 0.0) || (p.fy[0] != 0.0) || (p.fw[0] != 0.0);
        for (int i = 0; i < T; i++) {
            if (nw > 1) {
                double xx = p.ax2 * sinz[i], yy = p.ay2 * sinz[i], wwv = p.aw2 * sinz[i];
                for (int j = 0; j < nw; j++) {
                    xx += p.ax[j] * sin(p.bx[j] * z[i] + p.fx[j]) - offx[j];
                    yy += p.ay[j] * sin(p.by[j] * z[i] + p.fy[j]) - offy[j];
                    wwv += p.aw[j] * sin(p.bw[j] * z[i] + p.fw[j]) - offw[j];
                }
                X[i] = xx * invz[i];
                Y[i] = yy * invz[i];
                W[i] = wwv * invz[i];
            } else if (phased) {
                X[i] = (p.ax[0] * sin(p.bx[0] * z[i] + p.fx[0]) + p.ax2 * sinz[i] - offx[0]) *
                       invz[i];
                Y[i] = (p.ay[0] * sin(p.by[0] * z[i] + p.fy[0]) + p.ay2 * sinz[i] - offy[0]) *
                       invz[i];
                W[i] = (p.aw[0] * sin(p.bw[0] * z[i] + p.fw[0]) + p.aw2 * sinz[i] - offw[0]) *
                       invz[i];
            } else {
                X[i] = (p.ax[0] * sin(p.bx[0] * z[i]) + p.ax2 * sinz[i]) * invz[i];
                Y[i] = (p.ay[0] * sin(p.by[0] * z[i]) + p.ay2 * sinz[i]) * invz[i];
                W[i] = (p.aw[0] * sin(p.bw[0] * z[i]) + p.aw2 * sinz[i]) * invz[i];
            }
            if (torus) {
                X[i] = torus_wrap(X[i]);
                Y[i] = torus_wrap(Y[i]);
                W[i] = torus_wrap(W[i]);
            }
        }
    };

    // Pipelined survivor buffers: two sets on one stream, so round r+1's
    // kernel runs while the host admits round r's survivors. The kernel is
    // only a prefilter and points are never removed, so the one-round-stale
    // device grid is a SUBSET of the current packing: staleness can only pass
    // extra candidates, and every survivor is settled by the authoritative
    // host recheck. Each round copies back a fixed pinned prefix of copyCap
    // survivors without waiting for the count; the (rare, early-run) overflow
    // past copyCap is fetched afterwards on a second stream.
    // Path grew to kMaxWiggle arrays (--terms), so the survivor cap is kept at
    // 2^18 (~176 MB per buffer) rather than the historical 2^20: survivor
    // counts are bounded by the batch, which only grows once the acceptance
    // rate is far below cap/batch, so the cap is never reached in practice.
    const int survCap = 1 << 18;
    const int copyCap = 1 << 13;
    cudaStream_t st, stCopy;
    CK(cudaStreamCreate(&st));
    CK(cudaStreamCreate(&stCopy));
    Path* dSurv[2];
    int* dSurvCount[2];
    Path* hSurvPin[2];
    int* hCount[2];
    cudaEvent_t ev[2];
    for (int b = 0; b < 2; b++) {
        CK(cudaMalloc(&dSurv[b], survCap * sizeof(Path)));
        CK(cudaMalloc(&dSurvCount[b], 4));
        CK(cudaMallocHost(&hSurvPin[b], copyCap * sizeof(Path)));
        CK(cudaMallocHost(&hCount[b], 4));
        CK(cudaEventCreate(&ev[b]));
    }
    cudaEvent_t uploadEv;  // last grid upload; wait before rewriting staging
    CK(cudaEventCreate(&uploadEv));
    std::vector<Path> hSurvOver(survCap);  // overflow staging (pageable)
    size_t ptsCap = 1 << 16;
    double *dPtsX = nullptr, *dPtsY = nullptr, *dPtsW = nullptr;
    float *dPtsXf = nullptr, *dPtsYf = nullptr, *dPtsWf = nullptr;
    std::vector<double> ptsXh, ptsYh, ptsWh;
    std::vector<float> ptsXfH, ptsYfH, ptsWfH;
    // Keys and offsets are bounded by the point count, so they share ptsCap.
    auto alloc_pts = [&]() {
        if (sparse) {
            CK(cudaMalloc(&dPtsXf, ptsCap * 4));
            CK(cudaMalloc(&dPtsYf, ptsCap * 4));
            CK(cudaMalloc(&dPtsWf, ptsCap * 4));
            CK(cudaMalloc(&dKeys, (ptsCap + 1) * 4));
            CK(cudaMalloc(&dCellOff, (ptsCap + 1) * 4));
        } else {
            CK(cudaMalloc(&dPtsX, ptsCap * 8));
            CK(cudaMalloc(&dPtsY, ptsCap * 8));
            CK(cudaMalloc(&dPtsW, ptsCap * 8));
        }
    };
    auto free_pts = [&]() {
        if (sparse) {
            CK(cudaFree(dPtsXf));
            CK(cudaFree(dPtsYf));
            CK(cudaFree(dPtsWf));
            CK(cudaFree(dKeys));
            CK(cudaFree(dCellOff));
        } else {
            CK(cudaFree(dPtsX));
            CK(cudaFree(dPtsY));
            CK(cudaFree(dPtsW));
        }
    };
    alloc_pts();

    long long attempts = 0, N = 0, nextMs = 1000;
    long batch = 1 << 16;
    uint64_t round = 0;
    std::vector<std::pair<long long, long long>> curve;
    // Admission chunking (see the admission loop): per-chunk trajectory
    // buffers, precheck flags, and the chunk's admitted survivor slots.
    const int admitChunk = 64;
    std::vector<double> Xc((size_t)admitChunk * T), Yc((size_t)admitChunk * T),
        Wc((size_t)admitChunk * T);
    std::vector<uint8_t> preHit(admitChunk);
    std::vector<int> chunkAdm;
    long long lastN = -1;  // grid rebuilt only when N grew enough (deep-tail speedup)

    // Rebuild + upload the device grid when N grew enough. Uploads are
    // enqueued on the compute stream, so they land between the in-flight
    // kernel and the next one -- the in-flight kernel never sees a partial
    // grid, and the next kernel sees the whole update.
    auto maybe_rebuild = [&]() {
        if (!(lastN < 0 || N - lastN > N / 100 + 1))
            return;
        // The previous upload may still be in flight from these same staging
        // vectors; do not rewrite them under it.
        CK(cudaEventSynchronize(uploadEv));
        {
            size_t npts = (size_t)N * T;
            if (npts > ptsCap) {
                ptsCap = npts * 2;
                // The in-flight kernel reads the old buffers; drain it first.
                CK(cudaStreamSynchronize(st));
                free_pts();
                alloc_pts();
            }
            if (sparse) {
                // Incremental sparse rebuild: only the points admitted since
                // the last rebuild get sorted (a ~1% sliver); they are then
                // linear-merged into the persistent per-timestep sorted order
                // and the staging arrays + CSR are re-emitted. Timesteps are
                // independent, so the whole pass fans out across cores.
                ptsXfH.resize(npts);
                ptsYfH.resize(npts);
                ptsWfH.resize(npts);
                std::vector<size_t> baseOf(T + 1, 0);
                for (int i = 0; i < T; i++)
                    baseOf[i + 1] = baseOf[i] + px[i].size();
                parallel_for((size_t)T, [&](size_t lo, size_t hi) {
                    std::vector<std::pair<int, int>> fresh;  // (cell key, point index)
                    std::vector<int> mcell, midx;
                    for (size_t i = lo; i < hi; i++) {
                        auto& sc = sortedCell[i];
                        auto& si = sortedIdx[i];
                        const size_t n = px[i].size();
                        const size_t oldn = sc.size();
                        fresh.clear();
                        fresh.reserve(n - oldn);
                        for (size_t j = oldn; j < n; j++)
                            fresh.push_back({(int)(gix(px[i][j]) * gw2 + gix(py[i][j]) * gw +
                                                   gix(pw[i][j])),
                                             (int)j});
                        std::sort(fresh.begin(), fresh.end());
                        // Merge the fresh sliver into the persistent order.
                        // Old-first on equal cells keeps the exact order a
                        // full (cell, index) sort would produce.
                        mcell.resize(n);
                        midx.resize(n);
                        size_t a = 0, b = 0;
                        for (size_t k = 0; k < n; k++) {
                            const bool takeOld =
                                a < oldn && (b >= fresh.size() || sc[a] <= fresh[b].first);
                            if (takeOld) {
                                mcell[k] = sc[a];
                                midx[k] = si[a];
                                a++;
                            } else {
                                mcell[k] = fresh[b].first;
                                midx[k] = fresh[b].second;
                                b++;
                            }
                        }
                        sc.swap(mcell);
                        si.swap(midx);
                        // Emit this timestep's staging points + keys/offsets.
                        auto& tk = tkeys[i];
                        auto& to = toffs[i];
                        tk.clear();
                        to.clear();
                        int prev = -1;
                        const size_t base = baseOf[i];
                        for (size_t k = 0; k < n; k++) {
                            if (sc[k] != prev) {
                                tk.push_back(sc[k]);
                                to.push_back((int)(base + k));
                                prev = sc[k];
                            }
                            const int j = si[k];
                            ptsXfH[base + k] = (float)px[i][j];
                            ptsYfH[base + k] = (float)py[i][j];
                            ptsWfH[base + k] = (float)pw[i][j];
                        }
                    }
                });
                keysH.clear();
                cellOffH.clear();
                for (int i = 0; i < T; i++) {
                    keyStartH[i] = (int)keysH.size();
                    keysH.insert(keysH.end(), tkeys[i].begin(), tkeys[i].end());
                    cellOffH.insert(cellOffH.end(), toffs[i].begin(), toffs[i].end());
                }
                keyStartH[T] = (int)keysH.size();
                cellOffH.push_back((int)npts);  // CSR sentinel
                CK(cudaMemcpyAsync(dKeyStart, keyStartH.data(), (T + 1) * 4,
                                   cudaMemcpyHostToDevice, st));
                if (!keysH.empty())
                    CK(cudaMemcpyAsync(dKeys, keysH.data(), keysH.size() * 4,
                                       cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(dCellOff, cellOffH.data(), cellOffH.size() * 4,
                                   cudaMemcpyHostToDevice, st));
                if (npts) {
                    CK(cudaMemcpyAsync(dPtsXf, ptsXfH.data(), npts * 4, cudaMemcpyHostToDevice,
                                       st));
                    CK(cudaMemcpyAsync(dPtsYf, ptsYfH.data(), npts * 4, cudaMemcpyHostToDevice,
                                       st));
                    CK(cudaMemcpyAsync(dPtsWf, ptsWfH.data(), npts * 4, cudaMemcpyHostToDevice,
                                       st));
                }
            } else {
                // Dense rebuild, parallel over timesteps: timestep i owns the
                // cell slice [i*gw3, (i+1)*gw3) and the point list px[i], so
                // the fill/count/prefix/scatter passes are independent per
                // timestep -- this used to be a single-threaded O(T^4) pass
                // that left the GPU idle through a run's whole growth phase.
                // Each slice's prefix starts at that timestep's cumulative
                // point base, so the CSR layout is identical to the old
                // global pass. After the prefix, the cellLen slice doubles as
                // the scatter cursor (no ncell-sized `cur` copy).
                std::vector<size_t> baseOf(T + 1, 0);
                for (int i = 0; i < T; i++)
                    baseOf[i + 1] = baseOf[i] + px[i].size();
                ptsXh.resize(npts);
                ptsYh.resize(npts);
                ptsWh.resize(npts);
                parallel_for((size_t)T, [&](size_t lo, size_t hi) {
                    for (size_t i = lo; i < hi; i++) {
                        int* len = &cellLen[i * gw3];
                        std::fill(len, len + gw3, 0);
                        const size_t n = px[i].size();
                        for (size_t j = 0; j < n; j++) {
                            long c = gix(px[i][j]) * gw2 + gix(py[i][j]) * gw + gix(pw[i][j]);
                            len[c]++;
                        }
                        long acc = (long)baseOf[i];
                        int* start = &cellStart[i * gw3];
                        for (long c = 0; c < gw3; c++) {
                            start[c] = (int)acc;
                            acc += len[c];
                            len[c] = start[c];  // becomes this cell's scatter cursor
                        }
                        for (size_t j = 0; j < n; j++) {
                            long c = gix(px[i][j]) * gw2 + gix(py[i][j]) * gw + gix(pw[i][j]);
                            long s = len[c]++;
                            ptsXh[s] = px[i][j];
                            ptsYh[s] = py[i][j];
                            ptsWh[s] = pw[i][j];
                        }
                    }
                });
                cellStart[ncell] = (int)npts;  // CSR sentinel: one past the last point
                CK(cudaMemcpyAsync(dCellStart, cellStart.data(), (ncell + 1) * 4,
                                   cudaMemcpyHostToDevice, st));
                if (npts) {
                    CK(cudaMemcpyAsync(dPtsX, ptsXh.data(), npts * 8, cudaMemcpyHostToDevice,
                                       st));
                    CK(cudaMemcpyAsync(dPtsY, ptsYh.data(), npts * 8, cudaMemcpyHostToDevice,
                                       st));
                    CK(cudaMemcpyAsync(dPtsW, ptsWh.data(), npts * 8, cudaMemcpyHostToDevice,
                                       st));
                }
            }
            lastN = N;
        }
        CK(cudaEventRecord(uploadEv, st));
    };

    // Enqueue one prefilter round into buffer b: zero the count, launch the
    // kernel, and start the count + fixed-prefix survivor copies back to
    // pinned memory. Everything is on the compute stream; ev[b] fires when
    // the round's results are readable on the host.
    long roundBatch[2] = {0, 0};
    long long attemptsEnqueued = 0;
    auto enqueue_round = [&](int b) {
        CK(cudaMemsetAsync(dSurvCount[b], 0, 4, st));
        int threads = 256, blocks = (int)((batch + threads - 1) / threads);
        if (sparse)
            test_kernel_sparse<<<blocks, threads, 0, st>>>(
                seed, round, (int)batch, modmax, smart, angle, euclid, torus, phase, nw, T,
                dSinbT, dCosbm1T, dSinzF, dInvzF, (float)cell, (float)cellTight,
                (float)edgeLoose, gw, off, gw2, dKeys, dKeyStart, dCellOff, dPtsXf, dPtsYf,
                dPtsWf, dorder, dSurv[b], dSurvCount[b], survCap);
        else
            test_kernel<<<blocks, threads, 0, st>>>(
                seed, round, (int)batch, modmax, smart, angle, euclid, torus, phase, nw, T, dz,
                dsinz, dinvz, cell, gw, off, gw2, gw3, dCellStart, dPtsX, dPtsY, dPtsW, dorder,
                dSurv[b], dSurvCount[b], survCap);
        CK(cudaMemcpyAsync(hCount[b], dSurvCount[b], 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(hSurvPin[b], dSurv[b], copyCap * sizeof(Path),
                           cudaMemcpyDeviceToHost, st));
        CK(cudaEventRecord(ev[b], st));
        roundBatch[b] = batch;
        attemptsEnqueued += batch;
        round++;
    };

    maybe_rebuild();  // initial (empty) grid upload
    bool stopEnqueue = false;
    int cur = 0;
    enqueue_round(cur);
    while (true) {
        // Keep the GPU fed: enqueue the next round (against the device grid,
        // which lags admissions by a round) before draining this one.
        bool enqueuedNext = false;
        if (attemptsEnqueued < (long long)budget && !stopEnqueue) {
            enqueue_round(cur ^ 1);
            enqueuedNext = true;
        }
        // Drain round `cur`: wait for its survivors while its successor runs.
        CK(cudaEventSynchronize(ev[cur]));
        int sc = *hCount[cur];
        int got = sc < survCap ? sc : survCap;
        const Path* surv = hSurvPin[cur];
        if (got > copyCap) {
            // Rare (early rounds): fetch everything past the pinned prefix on
            // the copy stream, so it does not serialize behind the in-flight
            // kernel on the compute stream.
            memcpy(hSurvOver.data(), hSurvPin[cur], copyCap * sizeof(Path));
            CK(cudaMemcpyAsync(hSurvOver.data() + copyCap, dSurv[cur] + copyCap,
                               (size_t)(got - copyCap) * sizeof(Path), cudaMemcpyDeviceToHost,
                               stCopy));
            CK(cudaStreamSynchronize(stCopy));
            surv = hSurvOver.data();
        }
        attempts += roundBatch[cur];

        // Admission runs in chunks with two phases. Phase 1 (parallel): each
        // survivor's trajectory + collision check against the packing as
        // committed so far -- hgrid/px are read-only there, so it fans out
        // across cores. Phase 2 (serial, slot order): the only collisions
        // phase 1 cannot see are against this same chunk's admissions, so
        // each admittee is compared pairwise against the chunk's admitted
        // trajectories. Chunks are small to keep that quadratic tail short.
        // The admitted set is identical to the old fully-serial loop.
        long long admitted = 0;
        for (int s0 = 0; s0 < got; s0 += admitChunk) {
            const int c = std::min(admitChunk, got - s0);
            auto precheck = [&](size_t lo, size_t hi) {
                for (size_t q = lo; q < hi; q++) {
                    double *X = &Xc[q * T], *Y = &Yc[q * T], *W = &Wc[q * T];
                    eval_path(surv[s0 + q], X, Y, W);
                    preHit[q] = host_collides(X, Y, W) ? 1 : 0;
                }
            };
            if (c >= 24)
                parallel_for((size_t)c, precheck);
            else
                precheck(0, (size_t)c);
            chunkAdm.clear();
            for (int q = 0; q < c; q++) {
                if (preHit[q])
                    continue;
                const double *X = &Xc[(size_t)q * T], *Y = &Yc[(size_t)q * T],
                             *W = &Wc[(size_t)q * T];
                bool hit = false;
                for (int a : chunkAdm) {
                    const double *Xa = &Xc[(size_t)a * T], *Ya = &Yc[(size_t)a * T],
                                 *Wa = &Wc[(size_t)a * T];
                    for (int oi = 0; oi < T; oi++) {
                        const int i = order[oi];
                        double dX = X[i] - Xa[i];
                        double dY = Y[i] - Ya[i];
                        double dW = W[i] - Wa[i];
                        if (torus) {
                            dX = torus_delta(dX);
                            dY = torus_delta(dY);
                            dW = torus_delta(dW);
                        }
                        const bool h =
                            euclid ? dX * dX + dY * dY + dW * dW <= cell * cell
                                   : fabs(dX) <= cell && fabs(dY) <= cell && fabs(dW) <= cell;
                        if (h) {
                            hit = true;
                            break;
                        }
                    }
                    if (hit)
                        break;
                }
                if (hit)
                    continue;
                for (int i = 0; i < T; i++) {
                    px[i].push_back(X[i]);
                    py[i].push_back(Y[i]);
                    pw[i].push_back(W[i]);
                    hgrid[i][key3(hix(X[i]), hix(Y[i]), hix(W[i]))].push_back(
                        {X[i], Y[i], W[i]});
                }
                if (paramPath)
                    acceptedPaths.push_back(surv[s0 + q]);
                chunkAdm.push_back(q);
                N++;
                admitted++;
            }
        }
        if (got > 512 && batch > 4096)
            batch /= 2;
        else if (got < 64 && batch < (1 << 26))
            batch *= 2;
        while (attempts >= nextMs) {
            curve.push_back({attempts, N});
            nextMs = (long long)(nextMs * 1.15) + 1;
        }
        if (winTarget) {
            window.push_back({roundBatch[cur], admitted});
            winAtt += roundBatch[cur];
            winAdm += admitted;
            while (window.size() > 1 && winAtt - window.front().first >= winTarget) {
                winAtt -= window.front().first;
                winAdm -= window.front().second;
                window.pop_front();
            }
            if (winAtt >= winTarget && attempts > 1000000 &&
                (double)winAdm / (double)winAtt < acceptThresh)
                stopEnqueue = true;  // drain the in-flight round, then stop
        }
        maybe_rebuild();
        if (!enqueuedNext)
            break;
        cur ^= 1;
    }
    CK(cudaStreamSynchronize(st));
    curve.push_back({attempts, N});
    fprintf(stderr, "done: N=%lld in %lld attempts\n", N, attempts);
    if (curvePath) {
        FILE* f = fopen(curvePath, "w");
        fprintf(f, "attempts,n\n");
        for (auto& c : curve)
            fprintf(f, "%lld,%lld\n", c.first, c.second);
        fclose(f);
    }

    if (paramPath) {
        // Per-worldline parameters of the final packing, so the full phase space
        // can be reconstructed analytically at any z (e.g. the turnaround z=pi/2).
        // Phase columns are always present (zero when --phase is off).
        // --terms 2 keeps the legacy column layout so existing dump readers are
        // untouched; the multi-term layout leads with the sin1 amplitudes and
        // then one ax_j,bx_j,fx_j,ay_j,by_j,fy_j,aw_j,bw_j,fw_j group per
        // wiggle term (the 2+1 layout extended with the w axis).
        FILE* f = fopen(paramPath, "w");
        if (nw == 1) {
            fprintf(f, "ax,ay,aw,bx,by,bw,ax2,ay2,aw2,fx,fy,fw\n");
            for (auto& p : acceptedPaths)
                fprintf(f,
                        "%.10g,%.10g,%.10g,%.0f,%.0f,%.0f,%.10g,%.10g,%.10g,%.10g,%.10g,"
                        "%.10g\n",
                        p.ax[0], p.ay[0], p.aw[0], p.bx[0], p.by[0], p.bw[0], p.ax2, p.ay2,
                        p.aw2, p.fx[0], p.fy[0], p.fw[0]);
        } else {
            fprintf(f, "ax2,ay2,aw2");
            for (int j = 1; j <= nw; j++)
                fprintf(f, ",ax_%d,bx_%d,fx_%d,ay_%d,by_%d,fy_%d,aw_%d,bw_%d,fw_%d", j, j, j, j,
                        j, j, j, j, j);
            fprintf(f, "\n");
            for (auto& p : acceptedPaths) {
                fprintf(f, "%.10g,%.10g,%.10g", p.ax2, p.ay2, p.aw2);
                for (int j = 0; j < nw; j++)
                    fprintf(f, ",%.10g,%.0f,%.10g,%.10g,%.0f,%.10g,%.10g,%.0f,%.10g", p.ax[j],
                            p.bx[j], p.fx[j], p.ay[j], p.by[j], p.fy[j], p.aw[j], p.bw[j],
                            p.fw[j]);
                fprintf(f, "\n");
            }
        }
        fclose(f);
        fprintf(stderr, "dump-params: wrote %zu worldlines to %s\n", acceptedPaths.size(),
                paramPath);
    }

    if (diagPrefix) {
        const int NB = 50;  // histogram bins across the comoving range [-1, 1]
        auto bin_of = [&](double x) -> int {
            int b = (int)floor((x + 1.0) * 0.5 * NB);
            if (b < 0)
                return 0;
            if (b >= NB)
                return NB - 1;
            return b;
        };

        // (A) Occupancy: where the accepted worldlines actually sit in comoving
        //     X, summed over every time-step. A pile-up in the centre with empty
        //     edges is the signature of the centre-biased uniform proposal.
        std::vector<long long> occ(NB, 0);
        for (int i = 0; i < T; i++)
            for (double x : px[i])
                occ[bin_of(x)]++;
        {
            std::string path = std::string(diagPrefix) + "_occ.csv";
            FILE* f = fopen(path.c_str(), "w");
            fprintf(f, "x_center,count\n");
            for (int b = 0; b < NB; b++)
                fprintf(f, "%.4f,%lld\n", -1.0 + (b + 0.5) * 2.0 / NB, occ[b]);
            fclose(f);
        }

        // (B) Probe the final state: fire fresh uniform proposals at the packing
        //     we just built and record, per X-centre bin, how many are proposed
        //     vs still accepted. Lower acceptance in the centre than at the edges
        //     is exactly "wastes attempts on the full centre while the edges are
        //     barely tried."
        std::vector<long long> prop(NB, 0), acc(NB, 0);
        Rng pr;
        rng_seed(pr, seed ^ 0xABCDEF1234567890ULL);
        std::vector<double> X(T), Y(T), W(T);
        for (long long t = 0; t < probeN; t++) {
            Path p = propose(pr, modmax, smart, angle, torus, phase, nw);
            eval_path(p, X.data(), Y.data(), W.data());
            int b = bin_of(p.ax2);  // bin by the X-centre of the proposal
            prop[b]++;
            if (!host_collides(X.data(), Y.data(), W.data()))
                acc[b]++;
        }
        {
            std::string path = std::string(diagPrefix) + "_probe.csv";
            FILE* f = fopen(path.c_str(), "w");
            fprintf(f, "x_center,proposed,accepted\n");
            for (int b = 0; b < NB; b++)
                fprintf(f, "%.4f,%lld,%lld\n", -1.0 + (b + 0.5) * 2.0 / NB, prop[b], acc[b]);
            fclose(f);
        }
        fprintf(stderr, "diag: wrote %s_occ.csv and %s_probe.csv (probeN=%lld)\n", diagPrefix,
                diagPrefix, probeN);
    }
    return 0;
}
