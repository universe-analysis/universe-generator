// braid_knot — lean GPU packer for the knot-capacity (pinned braid) regime.
//
// The general engines (braid_cuda / braid_cuda3d) are built for the
// many-paths regime: spatial hash grids, adaptive batching, survivor-buffer
// pipelines. The knot-capacity experiment (Chris's braid-viewer setup:
// pin the sin1 comoving offsets to zero + full frequency band, then pack
// until jammed) lives in the opposite regime — a dozen strands, astronomical
// attempt counts — where the grid indexes nothing and the batch controller
// manages a flood that never comes. This tool strips both:
//
//   * NO spatial grid: candidates brute-force against the resident strand
//     list (precomputed trajectories) with per-timestep early exit.
//   * Fixed batch (deterministic ramp 4096 -> 2^24), no adaptation.
//   * A candidate IS its counter index: per-candidate RNG is seeded from
//     (seed, global attempt index), so survivors are recorded as bare
//     uint64 indices and the host regenerates their parameters exactly.
//     Admission processes survivors in index order, rechecking each against
//     strands admitted earlier in the same round — so a run is
//     BIT-REPRODUCIBLE per seed, and every admission is logged at its
//     exact attempt index (no round-boundary quantization).
//
// The proposal distribution and the acceptance predicate are identical to
// the general engines' pinned full-band mode (same unique-frequency pool,
// uniform simplex budget split, sign flips, even-frequency phases, same
// per-timestep Chebyshev 2/T exclusion on the wrapped torus, same
// z-grid z_i = (i+1)*pi/(T+1)) — so RSA statistics match; only the search
// machinery differs. A/B validation against the general engines is part of
// the tool's acceptance (see lab notes 2026-08-01).
//
// Build:  nvcc -O3 -o braid_knot braid_knot.cu
// Run:    ./braid_knot -t 32 --dim 2 --pin-sin1 --seed 1 \
//             --attempts 1e11 --curve knot.csv

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e = (x);                                                               \
        if (e) {                                                                           \
            fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

// ---------------------------------------------------------------------------
// RNG: xoshiro256** + splitmix64 seeding, identical to the engines.
// ---------------------------------------------------------------------------
struct Rng {
    uint64_t s[4];
};
__host__ __device__ inline uint64_t rotl64(uint64_t x, int k) {
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
    uint64_t result = rotl64(r.s[1] * 5, 7) * 9;
    uint64_t t = r.s[1] << 17;
    r.s[2] ^= r.s[0];
    r.s[3] ^= r.s[1];
    r.s[1] ^= r.s[2];
    r.s[0] ^= r.s[3];
    r.s[2] ^= t;
    r.s[3] = rotl64(r.s[3], 45);
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
// This tool exists for the full-band regime: size for terms up to 256.
constexpr int kMaxW = 255;
constexpr int kMaxStrands = 256;
constexpr int kSurvCap = 1 << 20;  // uint64 indices only — 8 MB

// ---------------------------------------------------------------------------
// Candidate: per-axis full-band terms. Axis count is runtime (2 or 3).
// ---------------------------------------------------------------------------
struct Cand {
    double a[3][kMaxW];  // amplitudes (signed)
    int b[3][kMaxW];     // integer frequencies
    double f[3][kMaxW];  // phases (0 on odd frequencies)
    double a2[3];        // sin(1*z) comoving offsets
};

// Uniform simplex split of 1 into nw parts (exponential spacings — the
// same uniform-simplex distribution as the engines' full-band path).
__host__ __device__ inline void split_unit(Rng& r, int nw, double* w) {
    double sum = 0.0;
    for (int j = 0; j < nw; j++) {
        w[j] = -log(1.0 - rng_f64(r));
        sum += w[j];
    }
    for (int j = 0; j < nw; j++)
        w[j] /= sum;
}

// Full-pool unique frequencies 2..modmax+1 via Fisher-Yates (the engines'
// full-band fast path); general nw < pool falls back to rejection draws.
__host__ __device__ inline void draw_freqs(Rng& r, uint32_t modmax, int nw, int* out) {
    if ((uint32_t)nw == modmax) {
        for (int j = 0; j < nw; j++)
            out[j] = j + 2;
        for (int j = nw - 1; j > 0; j--) {
            int k = (int)rng_below(r, (uint32_t)(j + 1));
            int tmp = out[j];
            out[j] = out[k];
            out[k] = tmp;
        }
        return;
    }
    for (int j = 0; j < nw; j++) {
        int bnew = 0;
        bool dup = true;
        for (int guard = 0; dup && guard < 64; guard++) {
            bnew = (int)rng_below(r, modmax) + 2;
            dup = false;
            for (int k = 0; k < j; k++)
                if (out[k] == bnew)
                    dup = true;
        }
        out[j] = bnew;
    }
}

__host__ __device__ inline void propose(
    Rng& r, uint32_t modmax, int nw, int dim, int pin, Cand& c) {
    double w[kMaxW];
    for (int ax = 0; ax < dim; ax++) {
        draw_freqs(r, modmax, nw, c.b[ax]);
        split_unit(r, nw, w);
        for (int j = 0; j < nw; j++) {
            double amp = w[j] / (double)c.b[ax][j];
            if (rng_flip(r))
                amp = -amp;
            c.a[ax][j] = amp;
            c.f[ax][j] = (c.b[ax][j] % 2 == 0) ? rng_f64(r) * kPi : 0.0;
        }
        c.a2[ax] = pin ? 0.0 : (rng_f64(r) * 2.0 - 1.0);
    }
}

__host__ __device__ inline double torus_wrap(double x) {
    return x - 2.0 * floor((x + 1.0) * 0.5);
}
__host__ __device__ inline double torus_delta(double d) {
    if (d > 1.0)
        return d - 2.0;
    if (d < -1.0)
        return d + 2.0;
    return d;
}

// ---------------------------------------------------------------------------
// Kernel: one thread = one candidate = one global attempt index.
// Tables: sinbz/cosbz are (maxfreq+1) x T; strands are dim x n x T wrapped
// comoving positions. Reject on the first timestep that collides with any
// strand; record survivors as bare indices.
// ---------------------------------------------------------------------------
__global__ void knot_kernel(uint64_t seed,
                            uint64_t base,
                            uint32_t batch,
                            uint32_t modmax,
                            int nw,
                            int dim,
                            int pin,
                            int T,
                            const double* __restrict__ sinbz,
                            const double* __restrict__ cosbz,
                            const double* __restrict__ sinz,
                            const double* __restrict__ invsinz,
                            const double* __restrict__ strands,
                            int nStrands,
                            double cell,
                            uint64_t* survIdx,
                            int* survCount) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch)
        return;
    const uint64_t idx = base + tid;
    Rng r;
    rng_seed(r, seed ^ (idx * 0x9E3779B97F4A7C15ULL + 0x1234567ULL));
    Cand c;
    propose(r, modmax, nw, dim, pin, c);

    // Per-term trig of the phases, once.
    double cf[3][kMaxW], sf[3][kMaxW];
    for (int ax = 0; ax < dim; ax++)
        for (int j = 0; j < nw; j++) {
            cf[ax][j] = cos(c.f[ax][j]);
            sf[ax][j] = sin(c.f[ax][j]);
        }

    for (int t = 0; t < T; t++) {
        double pos[3];
        for (int ax = 0; ax < dim; ax++) {
            double wig = 0.0;
            for (int j = 0; j < nw; j++) {
                const int b = c.b[ax][j];
                // a * (sin(b z + f) - sin f)
                wig += c.a[ax][j] * (sinbz[(long)b * T + t] * cf[ax][j] +
                                     cosbz[(long)b * T + t] * sf[ax][j] - sf[ax][j]);
            }
            pos[ax] = torus_wrap(wig * invsinz[t] + c.a2[ax] * 1.0);
        }
        for (int s = 0; s < nStrands; s++) {
            bool hit = true;
            for (int ax = 0; ax < dim; ax++) {
                // Strand layout matches the host buffer: [axis][kMaxStrands][T].
                const double d = torus_delta(
                    pos[ax] - strands[((long)ax * kMaxStrands + s) * T + t]);
                if (fabs(d) > cell) {
                    hit = false;
                    break;
                }
            }
            if (hit)
                return;  // collision — rejected
        }
    }
    const int slot = atomicAdd(survCount, 1);
    if (slot < kSurvCap)
        survIdx[slot] = idx;
}

// ---------------------------------------------------------------------------
// Host: exact-index admission loop.
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int T = 32, dim = 2, terms = 0, pin = 0;
    uint64_t seed = 1;
    double budget = 1e10;
    const char* curvePath = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-t"))
            T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dim"))
            dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--terms"))
            terms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pin-sin1"))
            pin = 1;
        else if (!strcmp(argv[i], "--seed"))
            seed = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--attempts"))
            budget = atof(argv[++i]);
        else if (!strcmp(argv[i], "--curve"))
            curvePath = argv[++i];
        else {
            fprintf(stderr, "unknown flag %s\n", argv[i]);
            return 1;
        }
    }
    if (dim != 2 && dim != 3) {
        fprintf(stderr, "error: --dim must be 2 or 3\n");
        return 1;
    }
    if (terms == 0)
        terms = T;  // the tool's reason to exist: full band by default
    const uint32_t modmax = (uint32_t)(T - 1 > 2 ? T - 1 : 2);
    const int nw = terms - 1;
    if (nw > kMaxW || (uint32_t)nw > modmax) {
        fprintf(stderr, "error: terms %d exceeds cap %d or pool %u\n", terms, kMaxW + 1,
                modmax);
        return 1;
    }
    const double cell = 2.0 / T;

    // z grid and trig tables (frequencies 1..maxfreq inclusive).
    const int maxfreq = (int)modmax + 1;
    std::vector<double> z(T), sinzH(T), invsinzH(T);
    for (int i = 0; i < T; i++) {
        z[i] = (i + 1) * kPi / (T + 1);
        sinzH[i] = sin(z[i]);
        invsinzH[i] = 1.0 / sinzH[i];
    }
    std::vector<double> sinbzH((size_t)(maxfreq + 1) * T), cosbzH((size_t)(maxfreq + 1) * T);
    for (int b = 0; b <= maxfreq; b++)
        for (int t = 0; t < T; t++) {
            sinbzH[(size_t)b * T + t] = sin(b * z[t]);
            cosbzH[(size_t)b * T + t] = cos(b * z[t]);
        }

    double *dSinbz, *dCosbz, *dSinz, *dInvsinz, *dStrands;
    uint64_t* dSurv;
    int* dCount;
    CK(cudaMalloc(&dSinbz, sinbzH.size() * 8));
    CK(cudaMalloc(&dCosbz, cosbzH.size() * 8));
    CK(cudaMalloc(&dSinz, T * 8));
    CK(cudaMalloc(&dInvsinz, T * 8));
    CK(cudaMalloc(&dStrands, (size_t)3 * kMaxStrands * T * 8));
    CK(cudaMalloc(&dSurv, (size_t)kSurvCap * 8));
    CK(cudaMalloc(&dCount, 4));
    CK(cudaMemcpy(dSinbz, sinbzH.data(), sinbzH.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dCosbz, cosbzH.data(), cosbzH.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSinz, sinzH.data(), T * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dInvsinz, invsinzH.data(), T * 8, cudaMemcpyHostToDevice));

    // Host-side strand trajectories, axis-major [ax][strand][t] (wrapped).
    std::vector<double> strandsH((size_t)3 * kMaxStrands * T, 0.0);
    int nStrands = 0;

    auto eval_cand = [&](const Cand& c, std::vector<double>& out) {
        out.assign((size_t)dim * T, 0.0);
        for (int ax = 0; ax < dim; ax++)
            for (int t = 0; t < T; t++) {
                double wig = 0.0;
                for (int j = 0; j < nw; j++) {
                    const int b = c.b[ax][j];
                    wig += c.a[ax][j] *
                           (sinbzH[(size_t)b * T + t] * cos(c.f[ax][j]) +
                            cosbzH[(size_t)b * T + t] * sin(c.f[ax][j]) - sin(c.f[ax][j]));
                }
                out[(size_t)ax * T + t] = torus_wrap(wig * invsinzH[t] + c.a2[ax]);
            }
    };
    auto collides_host = [&](const std::vector<double>& pos) {
        for (int t = 0; t < T; t++)
            for (int s = 0; s < nStrands; s++) {
                bool hit = true;
                for (int ax = 0; ax < dim; ax++) {
                    const double d =
                        torus_delta(pos[(size_t)ax * T + t] -
                                    strandsH[((size_t)ax * kMaxStrands + s) * T + t]);
                    if (fabs(d) > cell) {
                        hit = false;
                        break;
                    }
                }
                if (hit)
                    return true;
            }
        return false;
    };

    FILE* curve = curvePath ? fopen(curvePath, "w") : nullptr;
    if (curve)
        fprintf(curve, "attempt,n\n");
    fprintf(stderr,
            "braid_knot: dim=%d T=%d terms=%d pin=%d seed=%llu budget=%.3g "
            "(exact-index admission, deterministic)\n",
            dim, T, terms, pin, (unsigned long long)seed, budget);

    std::vector<uint64_t> hSurv(kSurvCap);
    uint64_t attempts = 0;
    int round = 0;
    uint64_t nextLog = 1;
    while ((double)attempts < budget && nStrands < kMaxStrands) {
        // Deterministic ramp: 4096 -> 2^24 over the first few rounds.
        uint32_t batch = 4096u << (2 * std::min(round, 6));
        if (batch > (1u << 24))
            batch = 1u << 24;
        if ((double)(attempts + batch) > budget)
            batch = (uint32_t)(budget - (double)attempts);
        if (batch == 0)
            break;

        // Upload current strand set (tiny).
        CK(cudaMemcpy(dStrands, strandsH.data(), (size_t)3 * kMaxStrands * T * 8,
                      cudaMemcpyHostToDevice));
        CK(cudaMemset(dCount, 0, 4));
        const int threads = 128;
        knot_kernel<<<(int)((batch + threads - 1) / threads), threads>>>(
            seed, attempts, batch, modmax, nw, dim, pin, T, dSinbz, dCosbz, dSinz, dInvsinz,
            dStrands, nStrands, cell, dSurv, dCount);
        CK(cudaDeviceSynchronize());
        int sc = 0;
        CK(cudaMemcpy(&sc, dCount, 4, cudaMemcpyDeviceToHost));
        if (sc > kSurvCap) {
            fprintf(stderr, "error: survivor cap overflow (%d)\n", sc);
            return 1;
        }
        if (sc) {
            CK(cudaMemcpy(hSurv.data(), dSurv, (size_t)sc * 8, cudaMemcpyDeviceToHost));
            std::sort(hSurv.begin(), hSurv.begin() + sc);
            std::vector<double> pos;
            for (int k = 0; k < sc && nStrands < kMaxStrands; k++) {
                const uint64_t idx = hSurv[k];
                Rng r;
                rng_seed(r, seed ^ (idx * 0x9E3779B97F4A7C15ULL + 0x1234567ULL));
                Cand c;
                propose(r, modmax, nw, dim, pin, c);
                eval_cand(c, pos);
                if (collides_host(pos))
                    continue;  // collided with an earlier same-round admission
                for (int ax = 0; ax < dim; ax++)
                    for (int t = 0; t < T; t++)
                        strandsH[((size_t)ax * kMaxStrands + nStrands) * T + t] =
                            pos[(size_t)ax * T + t];
                nStrands++;
                if (curve)
                    fprintf(curve, "%llu,%d\n", (unsigned long long)idx + 1, nStrands);
                fprintf(stderr, "admit: N=%d at attempt %llu\n", nStrands,
                        (unsigned long long)idx + 1);
            }
        }
        attempts += batch;
        round++;
        if (attempts >= nextLog) {
            nextLog = attempts * 2;
        }
    }
    if (curve)
        fclose(curve);
    fprintf(stderr, "done: N=%d in %llu attempts\n", nStrands, (unsigned long long)attempts);
    return 0;
}
