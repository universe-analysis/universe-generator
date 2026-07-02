// braid_cuda — GPU batch-reject RSA packing engine (2+1).
//
// Mirrors the Rust engine's model and RNG exactly (xoshiro256** + the same
// `propose`), so candidate generation is bit-identical. The collision test for
// a whole batch of candidates runs on the GPU; the host maintains the accepted
// grid and serially admits the (rare) survivors. Batch size adapts: small while
// acceptance is high (early), huge in the deep tail (rejection-dominated).
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

struct Path {
    double ax, ay, bx, by, ax2, ay2;
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

__host__ __device__ inline Path propose(Rng& r, uint32_t modmax, bool smart, bool torus) {
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
    Path p;
    p.ax2 = xs;
    p.ay2 = ys;
    if (torus) {
        // New-dogma budget: the slope-1 constraint binds the wiggle term alone
        // (|a*b| = 1); the sin1 term is a free comoving offset -- in comoving
        // coordinates it is a CONSTANT, so a uniform draw makes the packing
        // homogeneous on the torus by construction.
        p.ax = 1.0 / bx;
        p.ay = 1.0 / by;
    } else {
        p.ax = (1.0 - xs) / bx;
        p.ay = (1.0 - ys) / by;
    }
    p.bx = bx;
    p.by = by;
    if (rng_flip(r))
        p.ax = -p.ax;
    if (rng_flip(r))
        p.ay = -p.ay;
    if (rng_flip(r))
        p.ax2 = -p.ax2;
    if (rng_flip(r))
        p.ay2 = -p.ay2;
    return p;
}

// ---------- device collision test against the CSR grid ----------
__device__ bool collides_dev(const Path& p,
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
    for (int oi = 0; oi < T; oi++) {
        int i = order[oi];
        double X = (p.ax * sin(p.bx * z[i]) + p.ax2 * sinz[i]) * invz[i];
        double Y = (p.ay * sin(p.by * z[i]) + p.ay2 * sinz[i]) * invz[i];
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
    Path p = propose(r, modmax, smart, torus);
    if (!collides_dev(p, T, z, sinz, invz, cell, gw, off, cellStart, cellLen, ptsX, ptsY, order,
                      torus)) {
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
    // Torus grid: exactly T cells of width CELL span [-1, 1), so the modular
    // neighbour scan wraps cleanly at the seam. Hard-wall grid keeps its margin.
    int gw = torus ? T : T + 4, off = torus ? 0 : (T + 4) / 2;
    fprintf(stderr, "2+1%s: T=%d  modmax=%u (maxfreq=%u)\n", torus ? " (torus)" : "", T, modmax,
            modmax + 1);
    // z tables + endpoint-first order
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

    // survivor buffers
    int survCap = 1 << 20;
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
        test_kernel<<<blocks, threads>>>(seed, round, (int)batch, modmax, smart, torus, T, dz,
                                         dsinz, dinvz, cell, gw, off, dCellStart, dCellLen,
                                         dPtsX, dPtsY, dorder, dSurv, dSurvCount, survCap);
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
            for (int i = 0; i < T; i++) {
                Xb[i] = (p.ax * sin(p.bx * z[i]) + p.ax2 * sinz[i]) * invz[i];
                Yb[i] = (p.ay * sin(p.by * z[i]) + p.ay2 * sinz[i]) * invz[i];
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
        FILE* f = fopen(paramPath, "w");
        fprintf(f, "ax,ay,bx,by,ax2,ay2\n");
        for (auto& p : acceptedPaths)
            fprintf(f, "%.10g,%.10g,%.0f,%.0f,%.10g,%.10g\n", p.ax, p.ay, p.bx, p.by, p.ax2,
                    p.ay2);
        fclose(f);
        fprintf(stderr, "dump-params: wrote %zu worldlines to %s\n", acceptedPaths.size(),
                paramPath);
    }
    return 0;
}
