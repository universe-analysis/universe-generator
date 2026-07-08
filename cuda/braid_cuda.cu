// braid_cuda — GPU batch-reject RSA packing engine (2+1).
//
// Mirrors the Rust engine's model and RNG exactly (xoshiro256** + the same
// `propose`), so candidate generation is bit-identical. The collision test for
// a whole batch of candidates runs on the GPU; the host maintains the accepted
// grid and serially admits the (rare) survivors. Batch size adapts: small while
// acceptance is high (early), huge in the deep tail (rejection-dominated).
//
// --phase switches to the phase schema: the wiggle term gains a free phase
// f ~ U[0, pi) on even frequencies (odd ones must stay phase-free for the loop
// to close), each component is offset by a*sin(f) so it still starts at zero,
// and the z grid becomes the symmetric T-interior-points-of-(0, pi) indexing.
//
// --terms K generalizes the model from one wiggle term to K-1 of them per axis
// (K counts the always-present sin1 term). Each wiggle term gets a unique
// integer frequency; the unit slope budget is split uniformly at random across
// the terms (sorted-cut simplex split) and each term's amplitude is its share
// divided by its frequency, so the group satisfies sum |a_j*b_j| = 1 exactly.
// Every term carries its own even-frequency phase and its own a*sin(f) offset.
// K=2 (the default) reproduces the legacy single-wiggle model bit-for-bit.
//
// Output: the same `attempts,n` curve CSV as the Rust engine.
//
// Build:  nvcc -O3 -o braid_cuda braid_cuda.cu
// Run:    ./braid_cuda -t 120 --attempts 2e9 --seed 1 --curve out.csv

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
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

// ---------- RNG (xoshiro256**), identical to the Rust engine ----------
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

// Compile-time cap on wiggle terms per axis (--terms K uses K-1 of them).
// Bounds the fixed-size Path struct that rides through the survivor buffers.
constexpr int kMaxWiggle = 7;

struct Path {
    double ax[kMaxWiggle], ay[kMaxWiggle];  // wiggle amplitudes
    double bx[kMaxWiggle], by[kMaxWiggle];  // wiggle frequencies (integer-valued)
    double fx[kMaxWiggle], fy[kMaxWiggle];  // wiggle phases (0 unless --phase + even freq)
    double ax2, ay2;                        // sin(1*z) amplitudes
};

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

__host__ __device__ inline Path propose(
    Rng& r, uint32_t modmax, bool smart, bool torus, bool phase, int nw) {
    Path p = {};
    if (nw == 1) {
        // Legacy single-wiggle model: draw order preserved verbatim so the RNG
        // stream (and thus candidate generation) stays bit-identical to the
        // pre---terms engine.
        double xs = rng_f64(r), ys = rng_f64(r);
        uint32_t bx, by;
        if (!smart) {
            bx = rng_below(r, modmax) + 2;
            by = rng_below(r, modmax) + 2;
        } else {
            int guard = 0;
            while (true) {
                uint32_t a1 = rng_below(r, modmax), a2 = rng_below(r, modmax);
                bx = (a1 > a2 ? a1 : a2) + 2;
                uint32_t b1 = rng_below(r, modmax), b2 = rng_below(r, modmax);
                by = (b1 > b2 ? b1 : b2) + 2;
                if (igcd(bx, by) == 1 || guard >= 24)
                    break;
                guard++;
            }
        }
        p.ax2 = xs;
        p.ay2 = ys;
        if (torus) {
            // New-dogma budget: the slope-1 constraint binds the wiggle term
            // alone (|a*b| = 1); the sin1 term is a free comoving offset -- in
            // comoving coordinates it is a CONSTANT, so a uniform draw makes
            // the packing homogeneous on the torus by construction.
            p.ax[0] = 1.0 / bx;
            p.ay[0] = 1.0 / by;
        } else {
            p.ax[0] = (1.0 - xs) / bx;
            p.ay[0] = (1.0 - ys) / by;
        }
        p.bx[0] = bx;
        p.by[0] = by;
        if (rng_flip(r))
            p.ax[0] = -p.ax[0];
        if (rng_flip(r))
            p.ay[0] = -p.ay[0];
        if (rng_flip(r))
            p.ax2 = -p.ax2;
        if (rng_flip(r))
            p.ay2 = -p.ay2;
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
        }
        return p;
    }
    // Generalized multi-term model. Smart keeps the high-frequency bias per
    // draw but drops the cross-axis coprime rule (no single pair to test).
    double xs = rng_f64(r), ys = rng_f64(r);
    p.ax2 = xs;
    p.ay2 = ys;
    uint32_t bxa[kMaxWiggle], bya[kMaxWiggle];
    draw_unique_freqs(r, modmax, smart, nw, bxa);
    draw_unique_freqs(r, modmax, smart, nw, bya);
    double wx[kMaxWiggle], wy[kMaxWiggle];
    split_unit(r, nw, wx);
    split_unit(r, nw, wy);
    // Torus: the whole unit budget goes to the wiggle group (sum |a*b| = 1).
    // Hard wall: the group shares the slope budget with sin1, as before.
    double budx = torus ? 1.0 : (1.0 - xs);
    double budy = torus ? 1.0 : (1.0 - ys);
    for (int j = 0; j < nw; j++) {
        p.ax[j] = budx * wx[j] / bxa[j];
        p.bx[j] = bxa[j];
        p.ay[j] = budy * wy[j] / bya[j];
        p.by[j] = bya[j];
    }
    for (int j = 0; j < nw; j++)
        if (rng_flip(r))
            p.ax[j] = -p.ax[j];
    for (int j = 0; j < nw; j++)
        if (rng_flip(r))
            p.ay[j] = -p.ay[j];
    if (rng_flip(r))
        p.ax2 = -p.ax2;
    if (rng_flip(r))
        p.ay2 = -p.ay2;
    // Per-term phases, same rule as the single-wiggle model: even frequencies
    // only, and each term is re-pinned to zero by its own a*sin(f) offset.
    if (phase) {
        for (int j = 0; j < nw; j++)
            if (bxa[j] % 2 == 0)
                p.fx[j] = rng_f64(r) * kPi;
        for (int j = 0; j < nw; j++)
            if (bya[j] % 2 == 0)
                p.fy[j] = rng_f64(r) * kPi;
    }
    return p;
}

// ---------- device collision test against the CSR grid ----------
__device__ bool collides_dev(const Path& p,
                             int nw,
                             int T,
                             const double* z,
                             const double* sinz,
                             const double* invz,
                             double cell,
                             int gw,
                             int off,
                             const int* cellStart,
                             const int* cellLen,
                             const double* ptsX,
                             const double* ptsY,
                             const int* order,
                             bool torus) {
    double edge = 1.0 - 0.5 * cell;  // path radius CELL/2 hits the hard wall at |X|=1
    // Phase offsets re-pin each component to zero at the endpoints.
    double offx[kMaxWiggle], offy[kMaxWiggle];
    for (int j = 0; j < nw; j++) {
        offx[j] = p.ax[j] * sin(p.fx[j]);
        offy[j] = p.ay[j] * sin(p.fy[j]);
    }
    // Phase-free single-wiggle paths take the verbatim pre-phase expression, so
    // their per-candidate math is bit-identical to the pre-phase engine
    // (folding a zero phase into the expression shifts FP contraction by an
    // ulp). Full-run output is still scheduling-dependent: survivor admission
    // order comes from atomicAdd slots.
    const bool phased = (p.fx[0] != 0.0) || (p.fy[0] != 0.0);
    for (int oi = 0; oi < T; oi++) {
        int i = order[oi];
        double X, Y;
        if (nw > 1) {
            double xx = p.ax2 * sinz[i], yy = p.ay2 * sinz[i];
            for (int j = 0; j < nw; j++) {
                xx += p.ax[j] * sin(p.bx[j] * z[i] + p.fx[j]) - offx[j];
                yy += p.ay[j] * sin(p.by[j] * z[i] + p.fy[j]) - offy[j];
            }
            X = xx * invz[i];
            Y = yy * invz[i];
        } else if (phased) {
            X = (p.ax[0] * sin(p.bx[0] * z[i] + p.fx[0]) + p.ax2 * sinz[i] - offx[0]) * invz[i];
            Y = (p.ay[0] * sin(p.by[0] * z[i] + p.fy[0]) + p.ay2 * sinz[i] - offy[0]) * invz[i];
        } else {
            X = (p.ax[0] * sin(p.bx[0] * z[i]) + p.ax2 * sinz[i]) * invz[i];
            Y = (p.ay[0] * sin(p.by[0] * z[i]) + p.ay2 * sinz[i]) * invz[i];
        }
        int cx, cy;
        if (torus) {
            // Periodic domain: wrap onto [-1, 1); there is no wall.
            X = torus_wrap(X);
            Y = torus_wrap(Y);
            cx = (int)floor((X + 1.0) / cell);
            cy = (int)floor((Y + 1.0) / cell);
        } else {
            if (fabs(X) > edge || fabs(Y) > edge)
                return true;  // edge collision
            cx = (int)floor(X / cell);
            cy = (int)floor(Y / cell);
        }
        for (int dx = -1; dx <= 1; dx++) {
            int gx = torus ? (cx + dx + gw) % gw : cx + dx + off;
            if (gx < 0 || gx >= gw)
                continue;
            for (int dy = -1; dy <= 1; dy++) {
                int gy = torus ? (cy + dy + gw) % gw : cy + dy + off;
                if (gy < 0 || gy >= gw)
                    continue;
                long gc = (long)i * gw * gw + (long)gx * gw + gy;
                int st = cellStart[gc], ln = cellLen[gc];
                for (int k = 0; k < ln; k++) {
                    double dX = X - ptsX[st + k];
                    double dY = Y - ptsY[st + k];
                    if (torus) {
                        dX = torus_delta(dX);
                        dY = torus_delta(dY);
                    }
                    if (fabs(dX) <= cell && fabs(dY) <= cell)
                        return true;
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
                            const int* cellStart,
                            const int* cellLen,
                            const double* ptsX,
                            const double* ptsY,
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
    Path p = propose(r, modmax, smart, torus, phase, nw);
    if (!collides_dev(p, nw, T, z, sinz, invz, cell, gw, off, cellStart, cellLen, ptsX, ptsY,
                      order, torus)) {
        int slot = atomicAdd(survCount, 1);
        if (slot < survCap)
            survOut[slot] = p;
    }
}

// ---------- host ----------
int main(int argc, char** argv) {
    int T = 120;
    double budget = 1e8;
    uint64_t seed = 12345;
    bool smart = true;
    const char* curvePath = nullptr;
    int maxfreq = 0;          // 0 = default modulation cap T/2
    double acceptThresh = 0;  // 0 = run to --attempts; >0 = stop when accept-rate < thresh
    bool torus = false;       // new-dogma model: |a*b|=1, free sin1 offset, periodic domain
    bool phase = false;       // phase schema: even-frequency phases + symmetric z grid
    int terms = 2;            // total sinusoid terms per axis, incl. sin1 (2 = legacy)
    const char* paramPath = nullptr;  // if set, dump accepted worldline parameters
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
        else if (!strcmp(argv[i], "--curve"))
            curvePath = argv[++i];
        else if (!strcmp(argv[i], "--maxfreq"))
            maxfreq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--until-accept-rate"))
            acceptThresh = atof(argv[++i]);
        else if (!strcmp(argv[i], "--torus"))
            torus = true;
        else if (!strcmp(argv[i], "--phase"))
            phase = true;
        else if (!strcmp(argv[i], "--terms"))
            terms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump-params"))
            paramPath = argv[++i];
    }
    // Fixed-convergence stop: hold the acceptance rate (not the attempt count)
    // constant across T so the N_sat(T) slope is unbiased. Window sees ~30 accepts.
    std::deque<std::pair<long long, long long>> window;
    long long winAtt = 0, winAdm = 0;
    long long winTarget =
        acceptThresh > 0 ? (long long)std::min(5e9, std::max(2e7, 30.0 / acceptThresh)) : 0;
    const double PI = 3.14159265358979323846;
    // --maxfreq F sets the ACTUAL max frequency to F (freq draw is [2, modmax+1], so
    // modmax=F-1). Default (no flag) keeps the model's T/2+1 cap (modmax=T/2).
    double cell = 2.0 / T;
    uint32_t modmax =
        (maxfreq > 2) ? (uint32_t)(maxfreq - 1) : (uint32_t)(T / 2 > 2 ? T / 2 : 2);
    // Wiggle-term count: --terms counts sin1, so nw = terms-1. Capped by the
    // Path struct (kMaxWiggle) and by the pool of unique frequencies (modmax).
    if (terms < 2)
        terms = 2;
    if (terms - 1 > kMaxWiggle) {
        fprintf(stderr, "--terms %d exceeds compiled cap %d, clamping\n", terms,
                kMaxWiggle + 1);
        terms = kMaxWiggle + 1;
    }
    if ((uint32_t)(terms - 1) > modmax) {
        fprintf(stderr, "--terms %d exceeds unique-frequency pool %u, clamping\n", terms,
                modmax);
        terms = (int)modmax + 1;
    }
    int nw = terms - 1;
    // Torus grid: exactly T cells of width CELL span [-1, 1), so the modular
    // neighbour scan wraps cleanly at the seam. Hard-wall grid keeps its margin.
    int gw = torus ? T : T + 4, off = torus ? 0 : (T + 4) / 2;
    fprintf(stderr, "2+1%s%s: T=%d  modmax=%u (maxfreq=%u)  terms=%d\n",
            torus ? " (torus)" : "", phase ? " (phase)" : "", T, modmax, modmax + 1, terms);
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

    // device tables
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

    size_t ncell = (size_t)T * gw * gw;
    std::vector<int> cellStart(ncell), cellLen(ncell);
    int *dCellStart, *dCellLen;
    CK(cudaMalloc(&dCellStart, ncell * 4));
    CK(cudaMalloc(&dCellLen, ncell * 4));

    // host accepted points per timestep, host grid for admit re-check
    std::vector<std::vector<double>> px(T), py(T);
    std::vector<Path> acceptedPaths;  // populated only when --dump-params is set
    std::vector<std::unordered_map<long, std::vector<std::pair<double, double>>>> hgrid(T);

    double edge = 1.0 - 0.5 * cell;  // path radius CELL/2 hits the hard wall at |X|=1
    // Host hash-grid cell index. Torus: wrapped positions live in [-1,1), so the
    // index is periodic in [0, gw); neighbours wrap modulo gw. Hard wall: raw
    // floor(v/cell) (the hash key handles negatives).
    auto hix = [&](double v) -> int {
        return torus ? (int)floor((v + 1.0) / cell) : (int)floor(v / cell);
    };
    auto hnb = [&](int c, int d) -> int { return torus ? (c + d + gw) % gw : c + d; };
    auto host_collides = [&](const double* X, const double* Y) -> bool {
        for (int oi = 0; oi < T; oi++) {
            int i = order[oi];
            if (!torus && (fabs(X[i]) > edge || fabs(Y[i]) > edge))
                return true;  // edge collision
            int cx = hix(X[i]), cy = hix(Y[i]);
            for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++) {
                    long key = (long)hnb(cx, dx) * 100000L + hnb(cy, dy);
                    auto it = hgrid[i].find(key);
                    if (it == hgrid[i].end())
                        continue;
                    for (auto& pt : it->second) {
                        double dX = X[i] - pt.first;
                        double dY = Y[i] - pt.second;
                        if (torus) {
                            dX = torus_delta(dX);
                            dY = torus_delta(dY);
                        }
                        if (fabs(dX) <= cell && fabs(dY) <= cell)
                            return true;
                    }
                }
        }
        return false;
    };
    // Device-grid cell index for the CSR rebuild (must mirror collides_dev).
    auto gix = [&](double v) -> int {
        return torus ? (int)floor((v + 1.0) / cell) : (int)floor(v / cell) + off;
    };

    // survivor buffers (the multi-term Path is ~350 B, so the cap is kept at a
    // level that still dwarfs the ~256-survivors/round adaptation target)
    int survCap = 1 << 18;
    Path* dSurv;
    int* dSurvCount;
    CK(cudaMalloc(&dSurv, survCap * sizeof(Path)));
    CK(cudaMalloc(&dSurvCount, 4));
    std::vector<Path> hSurv(survCap);

    // device pts buffers (grow as needed)
    size_t ptsCap = 1 << 16;
    double *dPtsX, *dPtsY;
    CK(cudaMalloc(&dPtsX, ptsCap * 8));
    CK(cudaMalloc(&dPtsY, ptsCap * 8));
    std::vector<double> ptsXh, ptsYh;

    long long attempts = 0, N = 0;
    long long nextMs = 1000;
    long batch = 1 << 16;
    uint64_t round = 0;
    std::vector<std::pair<long long, long long>> curve;
    std::vector<double> Xb(T), Yb(T);

    while (attempts < (long long)budget) {
        // --- build CSR from accepted points ---
        size_t npts = (size_t)N * T;
        std::fill(cellLen.begin(), cellLen.end(), 0);
        for (int i = 0; i < T; i++)
            for (size_t j = 0; j < px[i].size(); j++) {
                int cx = gix(px[i][j]), cy = gix(py[i][j]);
                cellLen[(size_t)i * gw * gw + (size_t)cx * gw + cy]++;
            }
        size_t acc = 0;
        for (size_t c = 0; c < ncell; c++) {
            cellStart[c] = acc;
            acc += cellLen[c];
        }
        if (npts > ptsCap) {
            ptsCap = npts * 2;
            CK(cudaFree(dPtsX));
            CK(cudaFree(dPtsY));
            CK(cudaMalloc(&dPtsX, ptsCap * 8));
            CK(cudaMalloc(&dPtsY, ptsCap * 8));
        }
        ptsXh.assign(npts, 0);
        ptsYh.assign(npts, 0);
        std::vector<int> cur(cellStart);
        for (int i = 0; i < T; i++)
            for (size_t j = 0; j < px[i].size(); j++) {
                int cx = gix(px[i][j]), cy = gix(py[i][j]);
                long gc = (long)i * gw * gw + (long)cx * gw + cy;
                int s = cur[gc]++;
                ptsXh[s] = px[i][j];
                ptsYh[s] = py[i][j];
            }
        CK(cudaMemcpy(dCellStart, cellStart.data(), ncell * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dCellLen, cellLen.data(), ncell * 4, cudaMemcpyHostToDevice));
        if (npts) {
            CK(cudaMemcpy(dPtsX, ptsXh.data(), npts * 8, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dPtsY, ptsYh.data(), npts * 8, cudaMemcpyHostToDevice));
        }

        // --- launch batch test ---
        CK(cudaMemset(dSurvCount, 0, 4));
        int threads = 256, blocks = (int)((batch + threads - 1) / threads);
        test_kernel<<<blocks, threads>>>(
            seed, round, (int)batch, modmax, smart, torus, phase, nw, T, dz, dsinz, dinvz, cell,
            gw, off, dCellStart, dCellLen, dPtsX, dPtsY, dorder, dSurv, dSurvCount, survCap);
        CK(cudaDeviceSynchronize());
        int sc;
        CK(cudaMemcpy(&sc, dSurvCount, 4, cudaMemcpyDeviceToHost));
        int got = sc < survCap ? sc : survCap;
        if (got)
            CK(cudaMemcpy(hSurv.data(), dSurv, got * sizeof(Path), cudaMemcpyDeviceToHost));
        attempts += batch;
        round++;

        // --- serial admission with host re-check ---
        long long admitted = 0;
        for (int s = 0; s < got; s++) {
            Path& p = hSurv[s];
            double offx[kMaxWiggle], offy[kMaxWiggle];
            for (int j = 0; j < nw; j++) {
                offx[j] = p.ax[j] * sin(p.fx[j]);
                offy[j] = p.ay[j] * sin(p.fy[j]);
            }
            // Same rule as collides_dev: phase-free single-wiggle paths take
            // the verbatim pre-phase expression (per-candidate math identical
            // to pre-phase).
            const bool phased = (p.fx[0] != 0.0) || (p.fy[0] != 0.0);
            for (int i = 0; i < T; i++) {
                if (nw > 1) {
                    double xx = p.ax2 * sinz[i], yy = p.ay2 * sinz[i];
                    for (int j = 0; j < nw; j++) {
                        xx += p.ax[j] * sin(p.bx[j] * z[i] + p.fx[j]) - offx[j];
                        yy += p.ay[j] * sin(p.by[j] * z[i] + p.fy[j]) - offy[j];
                    }
                    Xb[i] = xx * invz[i];
                    Yb[i] = yy * invz[i];
                } else if (phased) {
                    Xb[i] =
                        (p.ax[0] * sin(p.bx[0] * z[i] + p.fx[0]) + p.ax2 * sinz[i] - offx[0]) *
                        invz[i];
                    Yb[i] =
                        (p.ay[0] * sin(p.by[0] * z[i] + p.fy[0]) + p.ay2 * sinz[i] - offy[0]) *
                        invz[i];
                } else {
                    Xb[i] = (p.ax[0] * sin(p.bx[0] * z[i]) + p.ax2 * sinz[i]) * invz[i];
                    Yb[i] = (p.ay[0] * sin(p.by[0] * z[i]) + p.ay2 * sinz[i]) * invz[i];
                }
                if (torus) {
                    Xb[i] = torus_wrap(Xb[i]);
                    Yb[i] = torus_wrap(Yb[i]);
                }
            }
            if (host_collides(Xb.data(), Yb.data()))
                continue;
            for (int i = 0; i < T; i++) {
                px[i].push_back(Xb[i]);
                py[i].push_back(Yb[i]);
                long key = (long)hix(Xb[i]) * 100000L + hix(Yb[i]);
                hgrid[i][key].push_back({Xb[i], Yb[i]});
            }
            if (paramPath)
                acceptedPaths.push_back(p);
            N++;
            admitted++;
        }
        // --- adapt batch toward ~256 survivors/round ---
        if (got > 512 && batch > 4096)
            batch /= 2;
        else if (got < 64 && batch < (1 << 26))
            batch *= 2;
        // --- milestones ---
        while (attempts >= nextMs) {
            curve.push_back({attempts, N});
            nextMs = (long long)(nextMs * 1.15) + 1;
        }
        // --- fixed-acceptance-rate stop ---
        if (winTarget) {
            window.push_back({batch, admitted});
            winAtt += batch;
            winAdm += admitted;
            while (window.size() > 1 && winAtt - window.front().first >= winTarget) {
                winAtt -= window.front().first;
                winAdm -= window.front().second;
                window.pop_front();
            }
            if (winAtt >= winTarget && attempts > 1000000 &&
                (double)winAdm / (double)winAtt < acceptThresh)
                break;
        }
    }
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
        // Per-worldline parameters of the final 2+1 packing (two spatial axes).
        // Phase columns are always present (zero when --phase is off).
        // --terms 2 keeps the legacy column layout so existing dump readers
        // (analysis, viewer import) are untouched; the multi-term layout leads
        // with the sin1 amplitudes and then one ax_j,bx_j,fx_j,ay_j,by_j,fy_j
        // group per wiggle term.
        FILE* f = fopen(paramPath, "w");
        if (nw == 1) {
            fprintf(f, "ax,ay,bx,by,ax2,ay2,fx,fy\n");
            for (auto& p : acceptedPaths)
                fprintf(f, "%.10g,%.10g,%.0f,%.0f,%.10g,%.10g,%.10g,%.10g\n", p.ax[0], p.ay[0],
                        p.bx[0], p.by[0], p.ax2, p.ay2, p.fx[0], p.fy[0]);
        } else {
            fprintf(f, "ax2,ay2");
            for (int j = 1; j <= nw; j++)
                fprintf(f, ",ax_%d,bx_%d,fx_%d,ay_%d,by_%d,fy_%d", j, j, j, j, j, j);
            fprintf(f, "\n");
            for (auto& p : acceptedPaths) {
                fprintf(f, "%.10g,%.10g", p.ax2, p.ay2);
                for (int j = 0; j < nw; j++)
                    fprintf(f, ",%.10g,%.0f,%.10g,%.10g,%.0f,%.10g", p.ax[j], p.bx[j], p.fx[j],
                            p.ay[j], p.by[j], p.fy[j]);
                fprintf(f, "\n");
            }
        }
        fclose(f);
        fprintf(stderr, "dump-params: wrote %zu worldlines to %s\n", acceptedPaths.size(),
                paramPath);
    }
    return 0;
}
