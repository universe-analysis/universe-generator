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

struct Path {
    double ax, ay, aw, bx, by, bw, ax2, ay2, aw2;
};

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

__host__ __device__ inline Path propose(
    Rng& r, uint32_t modmax, bool smart, bool angle, bool torus) {
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
            if ((igcd(bx, by) == 1 && igcd(bx, bw) == 1 && igcd(by, bw) == 1) || guard >= 24)
                break;
            guard++;
        }
    }
    Path p;
    p.ax2 = xs;
    p.ay2 = ys;
    p.aw2 = ws;
    if (torus) {
        // New-dogma budget: the slope-1 constraint binds the wiggle term alone
        // (|a*b| = 1); the sin1 term is a free comoving offset -- in comoving
        // coordinates it is a CONSTANT, so a uniform draw makes the packing
        // homogeneous on the torus by construction.
        p.ax = 1.0 / bx;
        p.ay = 1.0 / by;
        p.aw = 1.0 / bw;
    } else {
        p.ax = (1.0 - xs) / bx;
        p.ay = (1.0 - ys) / by;
        p.aw = (1.0 - ws) / bw;
    }
    p.bx = bx;
    p.by = by;
    p.bw = bw;
    if (rng_flip(r))
        p.ax = -p.ax;
    if (rng_flip(r))
        p.ay = -p.ay;
    if (rng_flip(r))
        p.aw = -p.aw;
    if (rng_flip(r))
        p.ax2 = -p.ax2;
    if (rng_flip(r))
        p.ay2 = -p.ay2;
    if (rng_flip(r))
        p.aw2 = -p.aw2;
    return p;
}

__device__ bool collides_dev(const Path& p,
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
                             const int* cellLen,
                             const double* ptsX,
                             const double* ptsY,
                             const double* ptsW,
                             const int* order,
                             bool euclid,
                             bool torus) {
    const double edge = 1.0 - 0.5 * cell;  // path radius CELL/2 hits the wall at |X|=1

    for (int oi = 0; oi < T; oi++) {
        const int i = order[oi];

        // Comoving position of the candidate path at this timestep (X = x / sin z).
        double X = (p.ax * sin(p.bx * z[i]) + p.ax2 * sinz[i]) * invz[i];
        double Y = (p.ay * sin(p.by * z[i]) + p.ay2 * sinz[i]) * invz[i];
        double W = (p.aw * sin(p.bw * z[i]) + p.aw2 * sinz[i]) * invz[i];

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

                    const long gc = (long)i * gw3 + (long)gx * gw2 + (long)gy * gw + gz;
                    const long st = cellStart[gc];
                    const int ln = cellLen[gc];

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

__global__ void test_kernel(uint64_t baseSeed,
                            uint64_t round,
                            int batch,
                            uint32_t modmax,
                            bool smart,
                            bool angle,
                            bool euclid,
                            bool torus,
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
                            const int* cellLen,
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
    Path p = propose(r, modmax, smart, angle, torus);
    if (!collides_dev(p, T, z, sinz, invz, cell, gw, off, gw2, gw3, cellStart, cellLen, ptsX,
                      ptsY, ptsW, order, euclid, torus)) {
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
    uint32_t modmax =
        (maxfreq > 2) ? (uint32_t)(maxfreq - 1) : (uint32_t)(T / 2 > 2 ? T / 2 : 2);
    // Torus grid: exactly T cells of width CELL span [-1, 1), so the modular
    // neighbour scan wraps cleanly at the seam. Hard-wall grid keeps its margin.
    int gw = torus ? T : T + 4, off = torus ? 0 : (T + 4) / 2;
    long gw2 = (long)gw * gw, gw3 = gw2 * gw;
    std::vector<double> z(T), sinz(T), invz(T);
    std::vector<int> order(T);
    double z0 = 0.01, z1 = PI - 0.01, step = (z1 - z0) / (T - 1);
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

    size_t ncell = (size_t)T * gw3;
    fprintf(stderr, "3+1%s: T=%d  modmax=%u (maxfreq=%u)  grid cells=%.2e  (~%.1f GB)\n",
            torus ? " (torus)" : "", T, modmax, modmax + 1, (double)ncell, ncell * 8.0 / 1e9);
    // cellStart holds cumulative point offsets (< N*T, well within int32 even at
    // T=300), so a 32-bit grid index halves device memory vs a 64-bit one.
    std::vector<int> cellStart(ncell);
    std::vector<int> cellLen(ncell);
    int* dCellStart;
    int* dCellLen;
    CK(cudaMalloc(&dCellStart, ncell * 4));
    CK(cudaMalloc(&dCellLen, ncell * 4));

    std::vector<std::vector<double>> px(T), py(T), pw(T);
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

    int survCap = 1 << 20;
    Path* dSurv;
    int* dSurvCount;
    CK(cudaMalloc(&dSurv, survCap * sizeof(Path)));
    CK(cudaMalloc(&dSurvCount, 4));
    std::vector<Path> hSurv(survCap);
    size_t ptsCap = 1 << 16;
    double *dPtsX, *dPtsY, *dPtsW;
    CK(cudaMalloc(&dPtsX, ptsCap * 8));
    CK(cudaMalloc(&dPtsY, ptsCap * 8));
    CK(cudaMalloc(&dPtsW, ptsCap * 8));
    std::vector<double> ptsXh, ptsYh, ptsWh;

    long long attempts = 0, N = 0, nextMs = 1000;
    long batch = 1 << 16;
    uint64_t round = 0;
    std::vector<std::pair<long long, long long>> curve;
    std::vector<double> Xb(T), Yb(T), Wb(T);
    long long lastN = -1;  // grid rebuilt only when N grew enough (deep-tail speedup)

    while (attempts < (long long)budget) {
        if (lastN < 0 || N - lastN > N / 100 + 1) {  // rebuild grid only when it changed enough
            size_t npts = (size_t)N * T;
            std::fill(cellLen.begin(), cellLen.end(), 0);
            for (int i = 0; i < T; i++)
                for (size_t j = 0; j < px[i].size(); j++) {
                    long gc = (long)i * gw3 + gix(px[i][j]) * gw2 + gix(py[i][j]) * gw +
                              gix(pw[i][j]);
                    cellLen[gc]++;
                }
            long acc = 0;
            for (size_t c = 0; c < ncell; c++) {
                cellStart[c] = (int)acc;
                acc += cellLen[c];
            }
            if (npts > ptsCap) {
                ptsCap = npts * 2;
                CK(cudaFree(dPtsX));
                CK(cudaFree(dPtsY));
                CK(cudaFree(dPtsW));
                CK(cudaMalloc(&dPtsX, ptsCap * 8));
                CK(cudaMalloc(&dPtsY, ptsCap * 8));
                CK(cudaMalloc(&dPtsW, ptsCap * 8));
            }
            ptsXh.assign(npts, 0);
            ptsYh.assign(npts, 0);
            ptsWh.assign(npts, 0);
            std::vector<int> cur(cellStart);
            for (int i = 0; i < T; i++)
                for (size_t j = 0; j < px[i].size(); j++) {
                    long gc = (long)i * gw3 + gix(px[i][j]) * gw2 + gix(py[i][j]) * gw +
                              gix(pw[i][j]);
                    long s = cur[gc]++;
                    ptsXh[s] = px[i][j];
                    ptsYh[s] = py[i][j];
                    ptsWh[s] = pw[i][j];
                }
            CK(cudaMemcpy(dCellStart, cellStart.data(), ncell * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dCellLen, cellLen.data(), ncell * 4, cudaMemcpyHostToDevice));
            if (npts) {
                CK(cudaMemcpy(dPtsX, ptsXh.data(), npts * 8, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dPtsY, ptsYh.data(), npts * 8, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dPtsW, ptsWh.data(), npts * 8, cudaMemcpyHostToDevice));
            }
            lastN = N;
        }

        CK(cudaMemset(dSurvCount, 0, 4));
        int threads = 256, blocks = (int)((batch + threads - 1) / threads);
        test_kernel<<<blocks, threads>>>(seed, round, (int)batch, modmax, smart, angle, euclid,
                                         torus, T, dz, dsinz, dinvz, cell, gw, off, gw2, gw3,
                                         dCellStart, dCellLen, dPtsX, dPtsY, dPtsW, dorder,
                                         dSurv, dSurvCount, survCap);
        CK(cudaDeviceSynchronize());
        int sc;
        CK(cudaMemcpy(&sc, dSurvCount, 4, cudaMemcpyDeviceToHost));
        int got = sc < survCap ? sc : survCap;
        if (got)
            CK(cudaMemcpy(hSurv.data(), dSurv, got * sizeof(Path), cudaMemcpyDeviceToHost));
        attempts += batch;
        round++;

        long long admitted = 0;
        for (int s = 0; s < got; s++) {
            Path& p = hSurv[s];
            for (int i = 0; i < T; i++) {
                Xb[i] = (p.ax * sin(p.bx * z[i]) + p.ax2 * sinz[i]) * invz[i];
                Yb[i] = (p.ay * sin(p.by * z[i]) + p.ay2 * sinz[i]) * invz[i];
                Wb[i] = (p.aw * sin(p.bw * z[i]) + p.aw2 * sinz[i]) * invz[i];
                if (torus) {
                    Xb[i] = torus_wrap(Xb[i]);
                    Yb[i] = torus_wrap(Yb[i]);
                    Wb[i] = torus_wrap(Wb[i]);
                }
            }
            if (host_collides(Xb.data(), Yb.data(), Wb.data()))
                continue;
            for (int i = 0; i < T; i++) {
                px[i].push_back(Xb[i]);
                py[i].push_back(Yb[i]);
                pw[i].push_back(Wb[i]);
                hgrid[i][key3(hix(Xb[i]), hix(Yb[i]), hix(Wb[i]))].push_back(
                    {Xb[i], Yb[i], Wb[i]});
            }
            if (paramPath)
                acceptedPaths.push_back(p);
            N++;
            admitted++;
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
        // Per-worldline parameters of the final packing, so the full phase space
        // can be reconstructed analytically at any z (e.g. the turnaround z=pi/2).
        FILE* f = fopen(paramPath, "w");
        fprintf(f, "ax,ay,aw,bx,by,bw,ax2,ay2,aw2\n");
        for (auto& p : acceptedPaths)
            fprintf(f, "%.10g,%.10g,%.10g,%.0f,%.0f,%.0f,%.10g,%.10g,%.10g\n", p.ax, p.ay, p.aw,
                    p.bx, p.by, p.bw, p.ax2, p.ay2, p.aw2);
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
            Path p = propose(pr, modmax, smart, angle, torus);
            for (int i = 0; i < T; i++) {
                X[i] = (p.ax * sin(p.bx * z[i]) + p.ax2 * sinz[i]) * invz[i];
                Y[i] = (p.ay * sin(p.by * z[i]) + p.ay2 * sinz[i]) * invz[i];
                W[i] = (p.aw * sin(p.bw * z[i]) + p.aw2 * sinz[i]) * invz[i];
                if (torus) {
                    X[i] = torus_wrap(X[i]);
                    Y[i] = torus_wrap(Y[i]);
                    W[i] = torus_wrap(W[i]);
                }
            }
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
