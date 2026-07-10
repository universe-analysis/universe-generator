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
// --subpaths adds a second packing phase after the unique phase stops
// (attempts budget, --until-accept-rate, or --max-n). Every unique path owns a
// group ID (its admission index). Subpath candidates are generated and
// collision-tested exactly like uniques, but the acceptance rule flips: a
// candidate touching one or more existing paths that ALL share one group ID
// joins that group; touching two groups rejects, and touching nothing rejects
// (so subpaths never add unique counts). Accepted subpaths enter the grid with
// their adopted group ID and can seed further subpaths. Phase 2 stops on
// --sub-attempts (default: the --attempts budget again), --sub-until-accept-rate
// (windowed acceptance decay), or --sub-until-fill-rate (windowed NEW-VOLUME
// decay: newly occupied cells per attempt, where a cell is one (timestep, cx,
// cy) bucket of the collision hash — the "free space filled" measure).
// --sub-terms K lets the sub phase use a different term count than the seeds.
//
// Output: the same `attempts,n` curve CSV as the Rust engine; with --subpaths
// the curve becomes `attempts,n,nsub,filled` (attempts cumulative across both
// phases; filled = occupied cells of T*gw*gw total) and the params dump gains
// a trailing `gid` column.
//
// Build:  nvcc -O3 -o braid_cuda braid_cuda.cu
// Run:    ./braid_cuda -t 120 --attempts 2e9 --seed 1 --curve out.csv

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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
constexpr double kPi = 3.14159265358979323846;

// Compile-time cap on wiggle terms per axis (--terms K uses K-1 of them).
// Bounds the fixed-size Path struct that rides through the survivor buffers.
constexpr int kMaxWiggle = 9;

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

// One uniform frequency draw in [2, modmax+1]. (The old "smart" sampler --
// max-of-two bias + coprime rule -- was removed 2026-07-09: the FREQ campaign
// showed its D-vs-terms trend was a sampler artifact; see lab notes 07-08.)
__host__ __device__ inline uint32_t draw_freq(Rng& r, uint32_t modmax) {
    return rng_below(r, modmax) + 2;
}

// nw unique frequencies for one axis. Duplicates re-draw (bounded); if the
// guard blows, the smallest unused frequency fills the slot deterministically.
__host__ __device__ inline void draw_unique_freqs(Rng& r,
                                                  uint32_t modmax,
                                                  int nw,
                                                  uint32_t* out) {
    for (int j = 0; j < nw; j++) {
        uint32_t b = 0;
        bool dup = true;
        for (int guard = 0; dup && guard < 64; guard++) {
            b = draw_freq(r, modmax);
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
    Rng& r, uint32_t modmax, bool torus, bool phase, int nw) {
    Path p = {};
    if (nw == 1) {
        // Legacy single-wiggle model: draw order preserved verbatim so the RNG
        // stream (and thus candidate generation) stays bit-identical to the
        // pre---terms engine.
        double xs = rng_f64(r), ys = rng_f64(r);
        // Draw order preserved verbatim from the old uniform branch, so the
        // RNG stream (and thus candidate generation) stays bit-identical to
        // pre-removal --uniform runs.
        uint32_t bx = rng_below(r, modmax) + 2;
        uint32_t by = rng_below(r, modmax) + 2;
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
    // Generalized multi-term model.
    double xs = rng_f64(r), ys = rng_f64(r);
    p.ax2 = xs;
    p.ay2 = ys;
    uint32_t bxa[kMaxWiggle], bya[kMaxWiggle];
    draw_unique_freqs(r, modmax, nw, bxa);
    draw_unique_freqs(r, modmax, nw, bya);
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
// Unique mode (sub=false): any contact rejects (early exit), as always.
// Subpath mode (sub=true): scan EVERY contact; reject on a contact with a
// second distinct group ID, or on touching nothing at all — survivors touch
// exactly one group. The host re-check recomputes the adopted group ID
// against the up-to-date grid, so no gid is returned from here.
__device__ bool collides_dev(const Path& p,
                             int nw,
                             bool sub,
                             const int* ptsGid,
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
    int gid = -1;  // sub mode: the single group this candidate may join
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
                    if (fabs(dX) <= cell && fabs(dY) <= cell) {
                        if (!sub)
                            return true;  // unique mode: any contact rejects
                        int g = ptsGid[st + k];
                        if (gid == -1)
                            gid = g;
                        else if (g != gid)
                            return true;  // touches a second group
                    }
                }
            }
        }
    }
    if (sub && gid == -1)
        return true;  // touched nothing: not a subpath
    return false;
}

__global__ void test_kernel(uint64_t baseSeed,
                            uint64_t round,
                            int batch,
                            uint32_t modmax,
                            bool torus,
                            bool phase,
                            int nw,
                            bool sub,
                            const int* ptsGid,
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
    Path p = propose(r, modmax, torus, phase, nw);
    if (!collides_dev(p, nw, sub, ptsGid, T, z, sinz, invz, cell, gw, off, cellStart, cellLen,
                      ptsX, ptsY, order, torus)) {
        int slot = atomicAdd(survCount, 1);
        if (slot < survCap)
            survOut[slot] = p;
    }
}

// One accepted point in the host hash grid (position + owning group).
struct HPt {
    double x, y;
    int gid;
};

// ---------- host ----------
int main(int argc, char** argv) {
    int T = 120;
    double budget = 1e8;
    uint64_t seed = 12345;
    const char* curvePath = nullptr;
    int maxfreq = 0;             // 0 = default modulation cap T/2
    double acceptThresh = 0;     // 0 = run to --attempts; >0 = stop when accept-rate < thresh
    bool torus = false;          // new-dogma model: |a*b|=1, free sin1 offset, periodic domain
    bool phase = false;          // phase schema: even-frequency phases + symmetric z grid
    int terms = 2;               // total sinusoid terms per axis, incl. sin1 (2 = legacy)
    bool subpaths = false;       // phase 2: pack subpaths into the jammed uniques
    double subBudget = 0;        // --sub-attempts (0 = reuse the --attempts budget)
    double subAcceptThresh = 0;  // --sub-until-accept-rate (windowed acceptance decay)
    double subFillThresh = 0;    // --sub-until-fill-rate (windowed new-cells/attempt decay)
    int subTerms = 0;            // --sub-terms (0 = same as --terms)
    long long maxN = 0;          // --max-n: stop the unique phase at this many paths
    const char* paramPath = nullptr;  // if set, dump accepted worldline parameters
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-t") || !strcmp(argv[i], "--timesteps"))
            T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--attempts"))
            budget = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed"))
            seed = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--uniform")) {
            // Uniform is the only frequency sampler; the flag is kept as a
            // no-op so existing command lines (and the orchestrator's argv,
            // which passes it as a guard against stale binaries) still run.
        } else if (!strcmp(argv[i], "--smart")) {
            fprintf(stderr,
                    "error: --smart was removed 2026-07-09 (the smart sampler's "
                    "D-vs-terms trend was an artifact; see lab notes 2026-07-08). "
                    "Frequencies are always drawn uniformly now.\n");
            return 1;
        } else if (!strcmp(argv[i], "--curve"))
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
        else if (!strcmp(argv[i], "--subpaths"))
            subpaths = true;
        else if (!strcmp(argv[i], "--sub-attempts"))
            subBudget = atof(argv[++i]);
        else if (!strcmp(argv[i], "--sub-until-accept-rate"))
            subAcceptThresh = atof(argv[++i]);
        else if (!strcmp(argv[i], "--sub-until-fill-rate"))
            subFillThresh = atof(argv[++i]);
        else if (!strcmp(argv[i], "--sub-terms"))
            subTerms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-n"))
            maxN = atoll(argv[++i]);
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
    // Sub-phase term count (--sub-terms), clamped like --terms.
    int nwSub = nw;
    if (subTerms > 0) {
        if (subTerms < 2)
            subTerms = 2;
        if (subTerms - 1 > kMaxWiggle)
            subTerms = kMaxWiggle + 1;
        if ((uint32_t)(subTerms - 1) > modmax)
            subTerms = (int)modmax + 1;
        nwSub = subTerms - 1;
    }
    // Torus grid: exactly T cells of width CELL span [-1, 1), so the modular
    // neighbour scan wraps cleanly at the seam. Hard-wall grid keeps its margin.
    int gw = torus ? T : T + 4, off = torus ? 0 : (T + 4) / 2;
    fprintf(stderr, "2+1%s%s: T=%d  modmax=%u (maxfreq=%u)  terms=%d\n",
            torus ? " (torus)" : "", phase ? " (phase)" : "", T, modmax, modmax + 1, terms);
    if (subpaths)
        fprintf(stderr, "subpaths: terms=%d  max-n=%lld\n", nwSub + 1, maxN);
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
    std::vector<int> acceptedGids;    // parallel to acceptedPaths (dump gid column)
    std::vector<int> pathGid;         // group ID per accepted path, admission order
    std::vector<std::unordered_map<long, std::vector<HPt>>> hgrid(T);
    long long filledCells = 0;  // occupied (t, cx, cy) buckets of hgrid

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
                        double dX = X[i] - pt.x;
                        double dY = Y[i] - pt.y;
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
    // Subpath re-check: mirrors collides_dev's sub mode against the up-to-date
    // host grid. Returns the adopted group ID, or -1 on any rejection (edge,
    // two-group contact, or no contact at all).
    auto host_sub_gid = [&](const double* X, const double* Y) -> int {
        int gid = -1;
        for (int oi = 0; oi < T; oi++) {
            int i = order[oi];
            if (!torus && (fabs(X[i]) > edge || fabs(Y[i]) > edge))
                return -1;  // edge collision
            int cx = hix(X[i]), cy = hix(Y[i]);
            for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++) {
                    long key = (long)hnb(cx, dx) * 100000L + hnb(cy, dy);
                    auto it = hgrid[i].find(key);
                    if (it == hgrid[i].end())
                        continue;
                    for (auto& pt : it->second) {
                        double dX = X[i] - pt.x;
                        double dY = Y[i] - pt.y;
                        if (torus) {
                            dX = torus_delta(dX);
                            dY = torus_delta(dY);
                        }
                        if (fabs(dX) <= cell && fabs(dY) <= cell) {
                            if (gid == -1)
                                gid = pt.gid;
                            else if (pt.gid != gid)
                                return -1;  // touches a second group
                        }
                    }
                }
        }
        return gid;  // -1 when it touched nothing
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
    int* dPtsGid;
    CK(cudaMalloc(&dPtsX, ptsCap * 8));
    CK(cudaMalloc(&dPtsY, ptsCap * 8));
    CK(cudaMalloc(&dPtsGid, ptsCap * 4));
    std::vector<double> ptsXh, ptsYh;
    std::vector<int> ptsGidH;

    long long attempts = 0, N = 0, Nsub = 0;
    long long nextMs = 1000;
    long batch = 1 << 16;
    uint64_t round = 0;
    std::vector<std::array<long long, 4>> curve;  // attempts, N, Nsub, filledCells
    std::vector<double> Xb(T), Yb(T);

    // Rebuild the device CSR grid (positions + per-point group IDs) from the
    // accepted points. Shared by both packing phases.
    auto uploadGrid = [&]() {
        size_t npts = px[0].size() * (size_t)T;
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
            CK(cudaFree(dPtsGid));
            CK(cudaMalloc(&dPtsX, ptsCap * 8));
            CK(cudaMalloc(&dPtsY, ptsCap * 8));
            CK(cudaMalloc(&dPtsGid, ptsCap * 4));
        }
        ptsXh.assign(npts, 0);
        ptsYh.assign(npts, 0);
        ptsGidH.assign(npts, 0);
        std::vector<int> cur(cellStart);
        for (int i = 0; i < T; i++)
            for (size_t j = 0; j < px[i].size(); j++) {
                int cx = gix(px[i][j]), cy = gix(py[i][j]);
                long gc = (long)i * gw * gw + (long)cx * gw + cy;
                int s = cur[gc]++;
                ptsXh[s] = px[i][j];
                ptsYh[s] = py[i][j];
                ptsGidH[s] = pathGid[j];
            }
        CK(cudaMemcpy(dCellStart, cellStart.data(), ncell * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dCellLen, cellLen.data(), ncell * 4, cudaMemcpyHostToDevice));
        if (npts) {
            CK(cudaMemcpy(dPtsX, ptsXh.data(), npts * 8, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dPtsY, ptsYh.data(), npts * 8, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dPtsGid, ptsGidH.data(), npts * 4, cudaMemcpyHostToDevice));
        }
    };
    // Evaluate a survivor's trajectory into Xb/Yb (same expressions and the
    // same legacy fast paths as collides_dev).
    auto evalPath = [&](const Path& p, int nwc) {
        double offx[kMaxWiggle], offy[kMaxWiggle];
        for (int j = 0; j < nwc; j++) {
            offx[j] = p.ax[j] * sin(p.fx[j]);
            offy[j] = p.ay[j] * sin(p.fy[j]);
        }
        // Same rule as collides_dev: phase-free single-wiggle paths take the
        // verbatim pre-phase expression (per-candidate math identical to
        // pre-phase).
        const bool phased = (p.fx[0] != 0.0) || (p.fy[0] != 0.0);
        for (int i = 0; i < T; i++) {
            if (nwc > 1) {
                double xx = p.ax2 * sinz[i], yy = p.ay2 * sinz[i];
                for (int j = 0; j < nwc; j++) {
                    xx += p.ax[j] * sin(p.bx[j] * z[i] + p.fx[j]) - offx[j];
                    yy += p.ay[j] * sin(p.by[j] * z[i] + p.fy[j]) - offy[j];
                }
                Xb[i] = xx * invz[i];
                Yb[i] = yy * invz[i];
            } else if (phased) {
                Xb[i] = (p.ax[0] * sin(p.bx[0] * z[i] + p.fx[0]) + p.ax2 * sinz[i] - offx[0]) *
                        invz[i];
                Yb[i] = (p.ay[0] * sin(p.by[0] * z[i] + p.fy[0]) + p.ay2 * sinz[i] - offy[0]) *
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
    };
    // Admit Xb/Yb into the host structures under the given group ID; returns
    // how many previously-empty hash cells it occupied (its NEW volume).
    auto insertPath = [&](int gid) -> long long {
        long long newCells = 0;
        for (int i = 0; i < T; i++) {
            px[i].push_back(Xb[i]);
            py[i].push_back(Yb[i]);
            long key = (long)hix(Xb[i]) * 100000L + hix(Yb[i]);
            auto& v = hgrid[i][key];
            if (v.empty())
                newCells++;
            v.push_back({Xb[i], Yb[i], gid});
        }
        pathGid.push_back(gid);
        filledCells += newCells;
        return newCells;
    };

    while (attempts < (long long)budget && !(maxN && N >= maxN)) {
        uploadGrid();

        // --- launch batch test ---
        CK(cudaMemset(dSurvCount, 0, 4));
        int threads = 256, blocks = (int)((batch + threads - 1) / threads);
        test_kernel<<<blocks, threads>>>(seed, round, (int)batch, modmax, torus, phase, nw,
                                         false, dPtsGid, T, dz, dsinz, dinvz, cell, gw, off,
                                         dCellStart, dCellLen, dPtsX, dPtsY, dorder, dSurv,
                                         dSurvCount, survCap);
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
            evalPath(p, nw);
            if (host_collides(Xb.data(), Yb.data()))
                continue;
            insertPath((int)N);  // a unique path's group ID is its admission index
            if (paramPath) {
                acceptedPaths.push_back(p);
                acceptedGids.push_back((int)N);
            }
            N++;
            admitted++;
            if (maxN && N >= maxN)
                break;  // unique-count cap reached mid-round
        }
        // --- adapt batch toward ~256 survivors/round ---
        if (got > 512 && batch > 4096)
            batch /= 2;
        else if (got < 64 && batch < (1 << 26))
            batch *= 2;
        // --- milestones ---
        while (attempts >= nextMs) {
            curve.push_back({attempts, N, Nsub, filledCells});
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
    fprintf(stderr, "done: N=%lld in %lld attempts, filled %lld/%zu cells (%.3f%%)\n", N,
            attempts, filledCells, ncell, 100.0 * (double)filledCells / (double)ncell);

    // ---------- phase 2: subpath packing ----------
    if (subpaths) {
        double subBudgetEff = subBudget > 0 ? subBudget : budget;
        long long attemptsSub = 0;
        batch = 1 << 16;  // fresh adaptation: sub acceptance starts high again
        // Two independent windowed stops, same 30-events heuristic as phase 1:
        // acceptance decay (admitted/attempt) and new-volume decay (newly
        // occupied cells/attempt).
        std::deque<std::pair<long long, long long>> awin, fwin;
        long long aAtt = 0, aAdm = 0, fAtt = 0, fNew = 0;
        long long aTarget =
            subAcceptThresh > 0
                ? (long long)std::min(5e9, std::max(2e7, 30.0 / subAcceptThresh))
                : 0;
        long long fTarget = subFillThresh > 0
                                ? (long long)std::min(5e9, std::max(2e7, 30.0 / subFillThresh))
                                : 0;
        const char* stopReason = "attempt budget";
        while (attemptsSub < (long long)subBudgetEff) {
            uploadGrid();
            CK(cudaMemset(dSurvCount, 0, 4));
            int threads = 256, blocks = (int)((batch + threads - 1) / threads);
            test_kernel<<<blocks, threads>>>(seed, round, (int)batch, modmax, torus, phase,
                                             nwSub, true, dPtsGid, T, dz, dsinz, dinvz, cell,
                                             gw, off, dCellStart, dCellLen, dPtsX, dPtsY,
                                             dorder, dSurv, dSurvCount, survCap);
            CK(cudaDeviceSynchronize());
            int sc;
            CK(cudaMemcpy(&sc, dSurvCount, 4, cudaMemcpyDeviceToHost));
            int got = sc < survCap ? sc : survCap;
            if (got)
                CK(cudaMemcpy(hSurv.data(), dSurv, got * sizeof(Path), cudaMemcpyDeviceToHost));
            attempts += batch;
            attemptsSub += batch;
            round++;
            long long admitted = 0, roundNewCells = 0;
            for (int s = 0; s < got; s++) {
                Path& p = hSurv[s];
                evalPath(p, nwSub);
                // Host re-check is authoritative: it recomputes the adopted
                // group against the grid INCLUDING this round's admissions.
                int gid = host_sub_gid(Xb.data(), Yb.data());
                if (gid < 0)
                    continue;
                roundNewCells += insertPath(gid);
                if (paramPath) {
                    acceptedPaths.push_back(p);
                    acceptedGids.push_back(gid);
                }
                Nsub++;
                admitted++;
            }
            if (got > 512 && batch > 4096)
                batch /= 2;
            else if (got < 64 && batch < (1 << 26))
                batch *= 2;
            while (attempts >= nextMs) {
                curve.push_back({attempts, N, Nsub, filledCells});
                nextMs = (long long)(nextMs * 1.15) + 1;
            }
            if (aTarget) {
                awin.push_back({batch, admitted});
                aAtt += batch;
                aAdm += admitted;
                while (awin.size() > 1 && aAtt - awin.front().first >= aTarget) {
                    aAtt -= awin.front().first;
                    aAdm -= awin.front().second;
                    awin.pop_front();
                }
                if (aAtt >= aTarget && attemptsSub > 1000000 &&
                    (double)aAdm / (double)aAtt < subAcceptThresh) {
                    stopReason = "accept-rate";
                    break;
                }
            }
            if (fTarget) {
                fwin.push_back({batch, roundNewCells});
                fAtt += batch;
                fNew += roundNewCells;
                while (fwin.size() > 1 && fAtt - fwin.front().first >= fTarget) {
                    fAtt -= fwin.front().first;
                    fNew -= fwin.front().second;
                    fwin.pop_front();
                }
                if (fAtt >= fTarget && attemptsSub > 1000000 &&
                    (double)fNew / (double)fAtt < subFillThresh) {
                    stopReason = "fill-rate";
                    break;
                }
            }
        }
        fprintf(stderr,
                "sub done (%s): Nsub=%lld in %lld attempts, filled %lld/%zu cells (%.3f%%)\n",
                stopReason, Nsub, attemptsSub, filledCells, ncell,
                100.0 * (double)filledCells / (double)ncell);
    }

    curve.push_back({attempts, N, Nsub, filledCells});
    if (curvePath) {
        // Legacy 2-column curve unless --subpaths is on (keeps existing
        // braidlab/analysis readers untouched).
        FILE* f = fopen(curvePath, "w");
        if (subpaths) {
            fprintf(f, "attempts,n,nsub,filled\n");
            for (auto& c : curve)
                fprintf(f, "%lld,%lld,%lld,%lld\n", c[0], c[1], c[2], c[3]);
        } else {
            fprintf(f, "attempts,n\n");
            for (auto& c : curve)
                fprintf(f, "%lld,%lld\n", c[0], c[1]);
        }
        fclose(f);
    }

    if (paramPath) {
        // Per-worldline parameters of the final 2+1 packing (two spatial axes).
        // Phase columns are always present (zero when --phase is off).
        // --terms 2 keeps the legacy column layout so existing dump readers
        // (analysis, viewer import) are untouched; the multi-term layout leads
        // with the sin1 amplitudes and then one ax_j,bx_j,fx_j,ay_j,by_j,fy_j
        // group per wiggle term. With --subpaths a trailing gid column records
        // each path's group (rows are in admission order: uniques then subs,
        // with a unique's gid equal to its own row index); paths generated
        // with fewer terms than the widest layout carry zero-amplitude
        // padding terms.
        FILE* f = fopen(paramPath, "w");
        int dumpNw = subpaths ? std::max(nw, nwSub) : nw;
        if (dumpNw == 1) {
            fprintf(f, "ax,ay,bx,by,ax2,ay2,fx,fy%s\n", subpaths ? ",gid" : "");
            for (size_t r = 0; r < acceptedPaths.size(); r++) {
                Path& p = acceptedPaths[r];
                fprintf(f, "%.10g,%.10g,%.0f,%.0f,%.10g,%.10g,%.10g,%.10g", p.ax[0], p.ay[0],
                        p.bx[0], p.by[0], p.ax2, p.ay2, p.fx[0], p.fy[0]);
                if (subpaths)
                    fprintf(f, ",%d", acceptedGids[r]);
                fprintf(f, "\n");
            }
        } else {
            fprintf(f, "ax2,ay2");
            for (int j = 1; j <= dumpNw; j++)
                fprintf(f, ",ax_%d,bx_%d,fx_%d,ay_%d,by_%d,fy_%d", j, j, j, j, j, j);
            fprintf(f, "%s\n", subpaths ? ",gid" : "");
            for (size_t r = 0; r < acceptedPaths.size(); r++) {
                Path& p = acceptedPaths[r];
                fprintf(f, "%.10g,%.10g", p.ax2, p.ay2);
                for (int j = 0; j < dumpNw; j++)
                    fprintf(f, ",%.10g,%.0f,%.10g,%.10g,%.0f,%.10g", p.ax[j], p.bx[j], p.fx[j],
                            p.ay[j], p.by[j], p.fy[j]);
                if (subpaths)
                    fprintf(f, ",%d", acceptedGids[r]);
                fprintf(f, "\n");
            }
        }
        fclose(f);
        fprintf(stderr, "dump-params: wrote %zu worldlines to %s\n", acceptedPaths.size(),
                paramPath);
    }
    return 0;
}
