// frame_bake — GPU past-light-cone frame baker (2+1).
//
// Ports the wrapped viewer's reference-frame construction
// (docs/viewers/twoplusone_2torus_wrapped.html, v37) to CUDA so observer
// frames can be baked from full campaign dumps (10^4-10^5 worldlines with up
// to ~T sinusoid terms each) instead of the viewer's few hundred paths.
// The viewer's math is the spec, copied term for term:
//
//   * A worldline's unwrapped comoving position is
//         X(z) = [a2 sin z + sum_j a_j (sin(b_j z + f_j) - sin f_j)] / sin z.
//   * An observation is (observer, z_obs): freeze the pulse centre at the
//     observer's comoving position, rewind the universe to zeta while a
//     front expands, and image every worldline where it crosses the front.
//     Front modes:
//       speed  : conformal, chi = c * (eta(z_obs) - eta(z_e)), eta = ln tan(z/2)
//       fit    : conformal with c = -1/eta(z_min) (Bang-born pulse reaches
//                the antipode chi = 1 exactly at pi/2)
//       budget : the wiggle-budget reach front (always Chebyshev)
//     The conformal metric is square (Chebyshev) or circle (L2).
//   * Torus ghosts: each image shift (2 nx, 2 ny), |n| <= K, is scanned
//     separately; K defaults to the viewer's auto cap ceil((chiMax + 4)/2).
//   * Crossings are located on the rewind grid (chi-uniform for conformal
//     fronts, z-uniform for budget) and refined by bisection with fresh
//     worldline evaluations; hits are stored FRAME-INDEPENDENTLY as
//     (source path, dx, dy, chi, z_emit) relative to the pulse centre, so
//     the static/moving observer views (aberration + Doppler) are a cheap
//     remap downstream (braidlab/frames.py), exactly as in the viewer.
//   * Cosmological redshift 1+Z = sin(z_obs)/sin(z_e) is derived downstream
//     from z_obs and z_emit; it is not stored per hit.
//
// Output: one binary .frames file (little-endian) —
//   header : magic "BRF1", u32 n_paths, u32 n_observers, u32 n_instants,
//            u32 front (0 speed, 1 fit, 2 budget), u32 cheb, f64 spd_or_c,
//            f64 zmin, i32 wrapK (-1 = auto per instant)
//   observers (n_observers records): u32 type (0 path, 1 point),
//            i32 path_idx (-1 for points), f64 px, f64 py (0 for paths)
//   per instant: f64 z_obs, f64 chiMax, i32 K,
//     per observer: f64 ox, f64 oy, f64 betax, f64 betay (proper peculiar
//            velocity / front speed, uncapped), u64 n_hits,
//            hits: (u32 src, f32 dx, f32 dy, f32 chi, f32 z_emit) * n_hits
//
// Build:  nvcc -O3 -o frame_bake frame_bake.cu
// Run:    ./frame_bake --dump dump.csv --from 0.25 --to 0.75 --frames 60 \
//             --observer-path 0 --observer-point 0.1,0.3 --front fit \
//             --out frames.bin

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                            \
    do {                                                                                 \
        cudaError_t e = (x);                                                             \
        if (e != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, \
                    __LINE__);                                                           \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

// Grid / refinement constants, matching the viewer.
constexpr int kFrameGrid = 420;
constexpr int kFrameBisect = 24;
constexpr int kBudgetGridCap = 4000;
constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// Dump parsing (host). Handles both 2+1 layouts by header name: the legacy
// single-wiggle columns (ax, ay, bx, by, ax2, ay2[, fx, fy]) and the
// multi-term layout (ax2, ay2, then ax_j, bx_j, fx_j, ay_j, by_j, fy_j per
// term). A trailing gid column (subpath dumps) is ignored.
// ---------------------------------------------------------------------------

struct Paths {
    int n = 0;
    int nw = 0;  // wiggle terms per axis
    std::vector<double> a2x, a2y;
    std::vector<double> ax, bx, fx, ay, by, fy;  // (n * nw), term-major per path
};

static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        size_t comma = line.find(',', start);
        if (comma == std::string::npos) {
            out.push_back(line.substr(start));
            break;
        }
        out.push_back(line.substr(start, comma - start));
        start = comma + 1;
    }
    return out;
}

static int col_of(const std::vector<std::string>& header, const char* name) {
    for (size_t i = 0; i < header.size(); i++)
        if (header[i] == name)
            return (int)i;
    return -1;
}

static Paths load_dump(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "error: cannot open dump %s\n", path);
        exit(1);
    }
    char buf[1 << 16];
    if (!fgets(buf, sizeof buf, f)) {
        fprintf(stderr, "error: empty dump %s\n", path);
        exit(1);
    }
    std::string headerLine(buf);
    while (!headerLine.empty() && (headerLine.back() == '\n' || headerLine.back() == '\r'))
        headerLine.pop_back();
    const std::vector<std::string> header = split_csv(headerLine);
    if (col_of(header, "aw") >= 0 || col_of(header, "aw_1") >= 0) {
        fprintf(stderr, "error: %s is a 3+1 dump; frame_bake is 2+1 only\n", path);
        exit(1);
    }

    Paths P;
    const bool multi = col_of(header, "ax_1") >= 0;
    int nw = 1;
    if (multi) {
        while (col_of(header, ("ax_" + std::to_string(nw + 1)).c_str()) >= 0)
            nw++;
    }
    P.nw = nw;
    // Column indices per field. Legacy layout: single term with optional
    // phase columns (pre-phase dumps lack fx/fy -> phases read as 0).
    std::vector<int> cax(nw), cbx(nw), cfx(nw), cay(nw), cby(nw), cfy(nw);
    int ca2x, ca2y;
    if (multi) {
        ca2x = col_of(header, "ax2");
        ca2y = col_of(header, "ay2");
        for (int j = 0; j < nw; j++) {
            const std::string s = "_" + std::to_string(j + 1);
            cax[j] = col_of(header, ("ax" + s).c_str());
            cbx[j] = col_of(header, ("bx" + s).c_str());
            cfx[j] = col_of(header, ("fx" + s).c_str());
            cay[j] = col_of(header, ("ay" + s).c_str());
            cby[j] = col_of(header, ("by" + s).c_str());
            cfy[j] = col_of(header, ("fy" + s).c_str());
        }
    } else {
        ca2x = col_of(header, "ax2");
        ca2y = col_of(header, "ay2");
        cax[0] = col_of(header, "ax");
        cbx[0] = col_of(header, "bx");
        cfx[0] = col_of(header, "fx");
        cay[0] = col_of(header, "ay");
        cby[0] = col_of(header, "by");
        cfy[0] = col_of(header, "fy");
    }
    if (ca2x < 0 || ca2y < 0 || cax[0] < 0) {
        fprintf(stderr, "error: unrecognized dump header in %s\n", path);
        exit(1);
    }

    while (fgets(buf, sizeof buf, f)) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
            line.pop_back();
        if (line.empty())
            continue;
        const std::vector<std::string> cells = split_csv(line);
        auto val = [&](int c) { return c >= 0 ? atof(cells[c].c_str()) : 0.0; };
        P.a2x.push_back(val(ca2x));
        P.a2y.push_back(val(ca2y));
        for (int j = 0; j < nw; j++) {
            P.ax.push_back(val(cax[j]));
            P.bx.push_back(val(cbx[j]));
            P.fx.push_back(val(cfx[j]));
            P.ay.push_back(val(cay[j]));
            P.by.push_back(val(cby[j]));
            P.fy.push_back(val(cfy[j]));
        }
        P.n++;
    }
    fclose(f);
    return P;
}

// ---------------------------------------------------------------------------
// Worldline evaluation (device + host, identical formulas).
// ---------------------------------------------------------------------------

struct DevPaths {
    int n, nw;
    const double *a2x, *a2y, *ax, *bx, *fx, *ay, *by, *fy;
};

__host__ __device__ inline void comov_xy(
    const DevPaths& P, int p, double z, double* x, double* y) {
    const double s = sin(z);
    double xx = P.a2x[p] * s, yy = P.a2y[p] * s;
    const long base = (long)p * P.nw;
    for (int j = 0; j < P.nw; j++) {
        xx += P.ax[base + j] * (sin(P.bx[base + j] * z + P.fx[base + j]) - sin(P.fx[base + j]));
        yy += P.ay[base + j] * (sin(P.by[base + j] * z + P.fy[base + j]) - sin(P.fy[base + j]));
    }
    *x = xx / s;
    *y = yy / s;
}

__host__ __device__ inline void comov_vel_xy(
    const DevPaths& P, int p, double z, double* vx, double* vy) {
    const double s = sin(z), c = cos(z);
    double xx = P.a2x[p] * s, yy = P.a2y[p] * s;
    double dx = P.a2x[p] * c, dy = P.a2y[p] * c;
    const long base = (long)p * P.nw;
    for (int j = 0; j < P.nw; j++) {
        xx += P.ax[base + j] * (sin(P.bx[base + j] * z + P.fx[base + j]) - sin(P.fx[base + j]));
        yy += P.ay[base + j] * (sin(P.by[base + j] * z + P.fy[base + j]) - sin(P.fy[base + j]));
        dx += P.ax[base + j] * P.bx[base + j] * cos(P.bx[base + j] * z + P.fx[base + j]);
        dy += P.ay[base + j] * P.by[base + j] * cos(P.by[base + j] * z + P.fy[base + j]);
    }
    *vx = (dx * s - xx * c) / (s * s);
    *vy = (dy * s - yy * c) / (s * s);
}

// ---------------------------------------------------------------------------
// Fronts. eta(z) = ln tan(z/2). The budget reach is the viewer's closed form:
// max over b in [2, bMax] of the two-time displacement of a unit-budget term
// (even b free-phase: hypot(d alpha, d beta); odd b: |d beta|).
// ---------------------------------------------------------------------------

__host__ __device__ inline double eta_of(double z) {
    return log(tan(z / 2.0));
}

__host__ __device__ inline double reach_of(double z, double zObs, int bMax) {
    const double s = sin(z), sObs = sin(zObs);
    double best = 0.0;
    for (int b = 2; b <= bMax; b++) {
        const double da = (cos(b * z) - 1.0) / (b * s) - (cos(b * zObs) - 1.0) / (b * sObs);
        const double db = sin(b * z) / (b * s) - sin(b * zObs) / (b * sObs);
        const double d = (b % 2 == 0) ? sqrt(da * da + db * db) : fabs(db);
        if (d > best)
            best = d;
    }
    return best;
}

// ---------------------------------------------------------------------------
// Hit search kernel: one thread per (path, image shift). Walks the rewind
// grid for sign changes of dist(pos + shift - none) - chi_front and bisects
// each crossing with fresh worldline evaluations, exactly as the viewer.
// ---------------------------------------------------------------------------

struct Hit {
    uint32_t src;
    float dx, dy, chi, zEmit;
};

struct FrontSpec {
    int budget;  // 1 = budget reach front
    int cheb;    // 1 = Chebyshev metric
    double spd;  // conformal front speed (reference c in budget mode)
    double zObs, zMin, etaObs, chiMax;
    int M;     // rewind grid rows
    int bMax;  // budget reach band
};

__device__ inline double dist_of(double dx, double dy, int cheb) {
    return cheb ? fmax(fabs(dx), fabs(dy)) : sqrt(dx * dx + dy * dy);
}

__device__ inline double z_at_chi(const FrontSpec F, double chi) {
    return 2.0 * atan(exp(F.etaObs - chi / F.spd));
}

__global__ void hits_kernel(DevPaths P,
                            FrontSpec F,
                            double ox,
                            double oy,
                            int K,
                            int obsIdx,
                            const double* zg,
                            const double* chig,
                            const double* px,
                            const double* py,
                            Hit* hits,
                            unsigned long long* hitCount,
                            unsigned long long hitCap) {
    const int side = 2 * K + 1;
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long)P.n * side * side)
        return;
    const int p = (int)(tid / (side * side));
    const int shift = (int)(tid % (side * side));
    const int nx = shift % side - K;
    const int ny = shift / side - K;

    const double bx = 2.0 * nx - ox, by = 2.0 * ny - oy;
    // Skip rows the front cannot have reached for this image (viewer's
    // wrap-cap bound: unwrapped comoving coords stay within +-2).
    const double dmin = fmax(0.0, 2.0 * fmax(fabs((double)nx), fabs((double)ny)) - 4.0);
    if (dmin >= F.chiMax)
        return;
    const int i0 = F.budget ? 1 : max(1, (int)ceil(dmin * F.M / F.chiMax));
    if (i0 > F.M)
        return;

    const long rowBase = (long)p * (F.M + 1);
    double prev =
        dist_of(px[rowBase + i0 - 1] + bx, py[rowBase + i0 - 1] + by, F.cheb) - chig[i0 - 1];
    // The observer sits exactly ON the front at chi = 0: nudge the trivial
    // self-contact inside so it is not recorded as an image. The observer's
    // own row is snapped from an epsilon band, not exact zero: the pulse
    // centre is evaluated on the host and the grid on the device, and a
    // one-ulp libm difference would otherwise decide the branch at random.
    if (p == obsIdx && nx == 0 && ny == 0 && i0 == 1 && fabs(prev) < 1e-9)
        prev = -1e-12;
    else if (prev == 0.0)
        prev = 1e-12;

    for (int i = i0; i <= F.M; i++) {
        const double g = dist_of(px[rowBase + i] + bx, py[rowBase + i] + by, F.cheb) - chig[i];
        if ((prev < 0.0) != (g < 0.0)) {
            const bool insidePrev = prev < 0.0;
            double chi, zEmit, qx, qy;
            if (F.budget) {
                double lo = zg[i - 1], hi = zg[i];
                for (int it = 0; it < kFrameBisect; it++) {
                    const double mid = (lo + hi) / 2.0;
                    comov_xy(P, p, mid, &qx, &qy);
                    const double gm =
                        dist_of(qx + bx, qy + by, F.cheb) - reach_of(mid, F.zObs, F.bMax);
                    if ((gm < 0.0) == insidePrev)
                        lo = mid;
                    else
                        hi = mid;
                }
                zEmit = (lo + hi) / 2.0;
                chi = reach_of(zEmit, F.zObs, F.bMax);
            } else {
                double lo = chig[i - 1], hi = chig[i];
                for (int it = 0; it < kFrameBisect; it++) {
                    const double mid = (lo + hi) / 2.0;
                    comov_xy(P, p, z_at_chi(F, mid), &qx, &qy);
                    const double gm = dist_of(qx + bx, qy + by, F.cheb) - mid;
                    if ((gm < 0.0) == insidePrev)
                        lo = mid;
                    else
                        hi = mid;
                }
                chi = (lo + hi) / 2.0;
                zEmit = z_at_chi(F, chi);
            }
            comov_xy(P, p, zEmit, &qx, &qy);
            const unsigned long long slot = atomicAdd(hitCount, 1ULL);
            if (slot < hitCap) {
                hits[slot].src = (uint32_t)p;
                hits[slot].dx = (float)(qx + bx);
                hits[slot].dy = (float)(qy + by);
                hits[slot].chi = (float)chi;
                hits[slot].zEmit = (float)zEmit;
            }
        }
        prev = g;
    }
}

// Path-grid kernel: comoving positions of every path on the rewind grid,
// filled once per instant and shared by every observer (the viewer's
// fillPathGrid).
__global__ void grid_kernel(DevPaths P, int M, const double* zg, double* px, double* py) {
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long)P.n * (M + 1))
        return;
    const int p = (int)(tid / (M + 1));
    const int i = (int)(tid % (M + 1));
    double x, y;
    comov_xy(P, p, zg[i], &x, &y);
    px[tid] = x;
    py[tid] = y;
}

// ---------------------------------------------------------------------------
// Host driver.
// ---------------------------------------------------------------------------

struct Observer {
    int type;     // 0 = path, 1 = comoving point
    int pathIdx;  // -1 for points
    double px, py;
};

int main(int argc, char** argv) {
    const char* dumpPath = nullptr;
    const char* outPath = "frames.bin";
    double fromPi = 0.25, toPi = 0.75;
    int nFrames = 1;
    double zMinPi = 0.02;
    std::string front = "fit";
    std::string metric = "square";
    double speed = 0.33;
    int wrapK = -1;  // -1 = viewer auto cap
    std::vector<Observer> observers;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dump"))
            dumpPath = argv[++i];
        else if (!strcmp(argv[i], "--out"))
            outPath = argv[++i];
        else if (!strcmp(argv[i], "--from"))
            fromPi = atof(argv[++i]);
        else if (!strcmp(argv[i], "--to"))
            toPi = atof(argv[++i]);
        else if (!strcmp(argv[i], "--frames"))
            nFrames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--zmin"))
            zMinPi = atof(argv[++i]);
        else if (!strcmp(argv[i], "--front"))
            front = argv[++i];
        else if (!strcmp(argv[i], "--metric"))
            metric = argv[++i];
        else if (!strcmp(argv[i], "--speed"))
            speed = atof(argv[++i]);
        else if (!strcmp(argv[i], "--wraps"))
            wrapK = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--observer-path")) {
            Observer o = {0, atoi(argv[++i]), 0.0, 0.0};
            observers.push_back(o);
        } else if (!strcmp(argv[i], "--observer-point")) {
            double x = 0.0, y = 0.0;
            if (sscanf(argv[++i], "%lf,%lf", &x, &y) != 2) {
                fprintf(stderr, "error: --observer-point wants x,y\n");
                return 1;
            }
            Observer o = {1, -1, x, y};
            observers.push_back(o);
        } else {
            fprintf(stderr, "error: unknown flag %s\n", argv[i]);
            return 1;
        }
    }
    if (!dumpPath || observers.empty() || nFrames < 1) {
        fprintf(stderr,
                "usage: frame_bake --dump D.csv --observer-path N|--observer-point "
                "x,y [...] --from 0.25 --to 0.75 --frames 60 [--front "
                "speed|fit|budget] [--metric square|circle] [--speed c] [--zmin "
                "0.02] [--wraps K] --out F.bin\n");
        return 1;
    }
    const int frontCode = front == "speed" ? 0 : front == "fit" ? 1 : 2;
    if (front != "speed" && front != "fit" && front != "budget") {
        fprintf(stderr, "error: --front must be speed|fit|budget\n");
        return 1;
    }

    Paths P = load_dump(dumpPath);
    fprintf(stderr, "loaded %d paths, %d terms/axis from %s\n", P.n, P.nw, dumpPath);
    for (const Observer& o : observers)
        if (o.type == 0 && (o.pathIdx < 0 || o.pathIdx >= P.n)) {
            fprintf(stderr, "error: observer path %d out of range\n", o.pathIdx);
            return 1;
        }

    // Reach band: the largest frequency in the dump (viewer: sampler band
    // extended by any imported term above it — for dumps the terms ARE the
    // band).
    int bMax = 2;
    for (double b : P.bx)
        if ((int)b > bMax)
            bMax = (int)b;
    for (double b : P.by)
        if ((int)b > bMax)
            bMax = (int)b;

    // Device paths.
    auto up = [](const std::vector<double>& v) {
        double* d;
        CK(cudaMalloc(&d, v.size() * sizeof(double)));
        CK(cudaMemcpy(d, v.data(), v.size() * sizeof(double), cudaMemcpyHostToDevice));
        return d;
    };
    DevPaths D = {P.n,      P.nw,     up(P.a2x), up(P.a2y), up(P.ax),
                  up(P.bx), up(P.fx), up(P.ay),  up(P.by),  up(P.fy)};

    FILE* out = fopen(outPath, "wb");
    if (!out) {
        fprintf(stderr, "error: cannot write %s\n", outPath);
        return 1;
    }
    const double zMin = fmax(2e-3, zMinPi * kPi);
    // Header.
    fwrite("BRF1", 1, 4, out);
    const uint32_t nu[3] = {(uint32_t)P.n, (uint32_t)observers.size(), (uint32_t)nFrames};
    fwrite(nu, 4, 3, out);
    const uint32_t fc[2] = {(uint32_t)frontCode,
                            (uint32_t)(frontCode == 2 || metric == "square")};
    fwrite(fc, 4, 2, out);
    fwrite(&speed, 8, 1, out);
    fwrite(&zMin, 8, 1, out);
    const int32_t wk = wrapK;
    fwrite(&wk, 4, 1, out);
    for (const Observer& o : observers) {
        const uint32_t t = (uint32_t)o.type;
        const int32_t pi_ = o.type == 0 ? o.pathIdx : -1;
        fwrite(&t, 4, 1, out);
        fwrite(&pi_, 4, 1, out);
        fwrite(&o.px, 8, 1, out);
        fwrite(&o.py, 8, 1, out);
    }

    const unsigned long long hitCap = 1ULL << 24;  // 16M hits / block (~400 MB)
    Hit* dHits;
    unsigned long long* dCount;
    CK(cudaMalloc(&dHits, hitCap * sizeof(Hit)));
    CK(cudaMalloc(&dCount, 8));
    std::vector<Hit> hHits;

    for (int fi = 0; fi < nFrames; fi++) {
        const double zObs = fmin(
            (fromPi + (toPi - fromPi) * (nFrames == 1 ? 0.0 : (double)fi / (nFrames - 1))) *
                kPi,
            kPi - 1e-3);
        if (zMin >= zObs) {
            fprintf(stderr, "error: rewind floor %.4f >= z_obs %.4f\n", zMin, zObs);
            return 1;
        }
        FrontSpec F = {};
        F.budget = frontCode == 2;
        F.cheb = frontCode == 2 || metric == "square";
        F.zObs = zObs;
        F.zMin = zMin;
        F.etaObs = eta_of(zObs);
        F.spd = frontCode == 0 ? fmax(1e-3, speed) : -1.0 / eta_of(zMin);
        F.bMax = bMax;
        F.M = F.budget ? std::min(kBudgetGridCap, std::max(kFrameGrid, 2 * bMax)) : kFrameGrid;

        // Rewind grid + front curve (host, tiny).
        std::vector<double> zg(F.M + 1), chig(F.M + 1);
        if (F.budget) {
            F.chiMax = 0.0;
            for (int i = 0; i <= F.M; i++) {
                zg[i] = zObs - (zObs - zMin) * i / F.M;
                chig[i] = reach_of(zg[i], zObs, bMax);
                if (chig[i] > F.chiMax)
                    F.chiMax = chig[i];
            }
        } else {
            F.chiMax = F.spd * (F.etaObs - eta_of(zMin));
            for (int i = 0; i <= F.M; i++) {
                chig[i] = F.chiMax * i / F.M;
                zg[i] = 2.0 * atan(exp(F.etaObs - chig[i] / F.spd));
            }
        }
        const int K = wrapK >= 0 ? wrapK : (int)ceil((F.chiMax + 4.0) / 2.0);

        double *dZg, *dChig, *dPx, *dPy;
        CK(cudaMalloc(&dZg, (F.M + 1) * 8));
        CK(cudaMalloc(&dChig, (F.M + 1) * 8));
        CK(cudaMemcpy(dZg, zg.data(), (F.M + 1) * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dChig, chig.data(), (F.M + 1) * 8, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dPx, (long)P.n * (F.M + 1) * 8));
        CK(cudaMalloc(&dPy, (long)P.n * (F.M + 1) * 8));
        {
            const long total = (long)P.n * (F.M + 1);
            const int threads = 256;
            grid_kernel<<<(int)((total + threads - 1) / threads), threads>>>(D, F.M, dZg, dPx,
                                                                             dPy);
            CK(cudaGetLastError());
        }

        fwrite(&zObs, 8, 1, out);
        fwrite(&F.chiMax, 8, 1, out);
        const int32_t k32 = K;
        fwrite(&k32, 4, 1, out);

        for (const Observer& o : observers) {
            double ox, oy, betax = 0.0, betay = 0.0;
            if (o.type == 0) {
                // Host-side evaluation from the host copies of the arrays.
                double vx, vy;
                const DevPaths H = {P.n,         P.nw,        P.a2x.data(), P.a2y.data(),
                                    P.ax.data(), P.bx.data(), P.fx.data(),  P.ay.data(),
                                    P.by.data(), P.fy.data()};
                comov_xy(H, o.pathIdx, zObs, &ox, &oy);
                comov_vel_xy(H, o.pathIdx, zObs, &vx, &vy);
                betax = sin(zObs) * vx / F.spd;
                betay = sin(zObs) * vy / F.spd;
            } else {
                ox = o.px;
                oy = o.py;
            }

            CK(cudaMemset(dCount, 0, 8));
            const int side = 2 * K + 1;
            const long total = (long)P.n * side * side;
            const int threads = 128;
            hits_kernel<<<(int)((total + threads - 1) / threads), threads>>>(
                D, F, ox, oy, K, o.type == 0 ? o.pathIdx : -1, dZg, dChig, dPx, dPy, dHits,
                dCount, hitCap);
            CK(cudaGetLastError());
            unsigned long long n = 0;
            CK(cudaMemcpy(&n, dCount, 8, cudaMemcpyDeviceToHost));
            if (n > hitCap) {
                fprintf(stderr,
                        "error: hit cap overflow (%llu > %llu) at frame %d -- "
                        "lower --wraps or raise the cap\n",
                        n, hitCap, fi);
                return 1;
            }
            hHits.resize(n);
            CK(cudaMemcpy(hHits.data(), dHits, n * sizeof(Hit), cudaMemcpyDeviceToHost));

            fwrite(&ox, 8, 1, out);
            fwrite(&oy, 8, 1, out);
            fwrite(&betax, 8, 1, out);
            fwrite(&betay, 8, 1, out);
            const uint64_t n64 = n;
            fwrite(&n64, 8, 1, out);
            fwrite(hHits.data(), sizeof(Hit), n, out);
        }
        CK(cudaFree(dZg));
        CK(cudaFree(dChig));
        CK(cudaFree(dPx));
        CK(cudaFree(dPy));
        fprintf(stderr, "frame %d/%d: z_obs=%.4f chiMax=%.3f K=%d\n", fi + 1, nFrames, zObs,
                F.chiMax, K);
    }
    fclose(out);
    fprintf(stderr, "done: wrote %s\n", outPath);
    return 0;
}
