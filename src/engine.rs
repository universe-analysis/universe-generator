//! Core packing engine: path model, per-time-step collision grid, and the
//! random-sequential-adsorption (RSA) loop (sequential + batch-parallel).
//!
//! Faithful to `twoplusone_2torus.html`:
//!   path:   x(z) = ax*sin(bx*z) + ax2*sin(z),  y likewise
//!   slope:  |ax*bx| + |ax2| = 1   (the speed limit)
//!   comoving: X = x/sin(z)  (removes the Bang/Crunch funnel)
//!   collide:  per-time-step Chebyshev distance <= CELL = 2/T
//!
//! Optimisations (bit-identical results to a hash-grid + in-order scan):
//!   - flat dense grid per time-step (comoving X,Y are bounded to [-1,1], so a
//!     T-by-T grid indexes directly — no hashing).
//!   - endpoint-first time-step scan order: the ±1 "seam" piles up at the
//!     Bang/Crunch, so checking endpoint time-steps first makes most rejections
//!     early-exit sooner.

// ---- fast RNG (xoshiro256**) ------------------------------------------------

pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn seed(mut seed: u64) -> Rng {
        let mut next = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Rng { s: [next(), next(), next(), next()] }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    #[inline]
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    #[inline]
    pub fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % n as u64) as u32
    }
    #[inline]
    pub fn flip(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

// ---- path model -------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Path {
    pub ax: f64,
    pub ay: f64,
    pub bx: f64,
    pub by: f64,
    pub ax2: f64,
    pub ay2: f64,
}

// ---- engine -----------------------------------------------------------------

pub struct Engine {
    pub t: usize,
    z: Vec<f64>,
    sin_z: Vec<f64>,
    inv_z: Vec<f64>,
    cell: f64,
    mod_max: u32,
    smart: bool,
    // flat dense grid: grids[timestep][cell] -> points; cell = gx*gw + gy
    grids: Vec<Vec<Vec<(f64, f64)>>>,
    gw: i64,
    goff: i64,
    order: Vec<usize>, // time-step scan order (endpoints first)
    pub n: usize,
    pub paths: Vec<Path>,
}

impl Engine {
    pub fn new(t: usize, smart: bool) -> Engine {
        let (z0, z1) = (0.01_f64, std::f64::consts::PI - 0.01);
        let step = (z1 - z0) / (t as f64 - 1.0);
        let z: Vec<f64> = (0..t).map(|i| z0 + i as f64 * step).collect();
        let sin_z: Vec<f64> = z.iter().map(|v| v.sin()).collect();
        let inv_z: Vec<f64> = sin_z.iter().map(|v| 1.0 / v).collect();
        // |X| <= 1, CELL = 2/T  =>  cell index in [-T/2, T/2]; pad for neighbours
        let gw = (t as i64) + 4;
        let goff = gw / 2;
        let mut order: Vec<usize> = (0..t).collect();
        order.sort_by_key(|&i| i.min(t - 1 - i)); // 0, T-1, 1, T-2, ...
        Engine {
            t,
            z,
            sin_z,
            inv_z,
            cell: 2.0 / t as f64,
            mod_max: (t / 2).max(2) as u32,
            smart,
            grids: (0..t).map(|_| vec![Vec::new(); (gw * gw) as usize]).collect(),
            gw,
            goff,
            order,
            n: 0,
            paths: Vec::new(),
        }
    }

    #[inline]
    fn cell_of(&self, x: f64, y: f64) -> (i64, i64) {
        ((x / self.cell).floor() as i64, (y / self.cell).floor() as i64)
    }

    fn draw_freqs(&self, rng: &mut Rng) -> (f64, f64) {
        if !self.smart {
            return ((rng.below(self.mod_max) + 2) as f64, (rng.below(self.mod_max) + 2) as f64);
        }
        let mut guard = 0;
        loop {
            let bx = rng.below(self.mod_max).max(rng.below(self.mod_max)) + 2;
            let by = rng.below(self.mod_max).max(rng.below(self.mod_max)) + 2;
            if gcd(bx, by) == 1 || guard >= 24 {
                return (bx as f64, by as f64);
            }
            guard += 1;
        }
    }

    pub fn propose(&self, rng: &mut Rng) -> Path {
        let xs = rng.f64();
        let ys = rng.f64();
        let (bx, by) = self.draw_freqs(rng);
        let mut ax2 = xs;
        let mut ay2 = ys;
        let mut ax = (1.0 - xs) / bx;
        let mut ay = (1.0 - ys) / by;
        if rng.flip() {
            ax = -ax;
        }
        if rng.flip() {
            ay = -ay;
        }
        if rng.flip() {
            ax2 = -ax2;
        }
        if rng.flip() {
            ay2 = -ay2;
        }
        Path { ax, ay, bx, by, ax2, ay2 }
    }

    pub fn trajectory(&self, p: &Path, xb: &mut [f64], yb: &mut [f64]) {
        for i in 0..self.t {
            let (z, inv, sz) = (self.z[i], self.inv_z[i], self.sin_z[i]);
            xb[i] = (p.ax * (p.bx * z).sin() + p.ax2 * sz) * inv;
            yb[i] = (p.ay * (p.by * z).sin() + p.ay2 * sz) * inv;
        }
    }

    pub fn collides(&self, xb: &[f64], yb: &[f64]) -> bool {
        let cell = self.cell;
        let (gw, goff) = (self.gw, self.goff);
        for &i in &self.order {
            let (xi, yi) = (xb[i], yb[i]);
            let (cx, cy) = self.cell_of(xi, yi);
            let grid = &self.grids[i];
            for dx in -1..=1 {
                let gx = cx + dx + goff;
                if gx < 0 || gx >= gw {
                    continue;
                }
                for dy in -1..=1 {
                    let gy = cy + dy + goff;
                    if gy < 0 || gy >= gw {
                        continue;
                    }
                    let v = &grid[(gx * gw + gy) as usize];
                    for &(px, py) in v {
                        if (xi - px).abs() <= cell && (yi - py).abs() <= cell {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    pub fn insert(&mut self, xb: &[f64], yb: &[f64], p: Path) {
        let (gw, goff) = (self.gw, self.goff);
        for i in 0..self.t {
            let (xi, yi) = (xb[i], yb[i]);
            let (cx, cy) = self.cell_of(xi, yi);
            let idx = ((cx + goff) * gw + (cy + goff)) as usize;
            self.grids[i][idx].push((xi, yi));
        }
        self.paths.push(p);
        self.n += 1;
    }

    /// Single-threaded RSA. `on_ms(attempts, n)` fires at log-spaced milestones.
    pub fn run_sequential(
        &mut self,
        budget: u64,
        target: usize,
        seed: u64,
        mut on_ms: impl FnMut(u64, usize),
    ) {
        let mut rng = Rng::seed(seed);
        let mut xb = vec![0.0; self.t];
        let mut yb = vec![0.0; self.t];
        let mut attempts: u64 = 0;
        let mut next_ms: u64 = 1000;
        while attempts < budget && self.n < target {
            attempts += 1;
            let p = self.propose(&mut rng);
            self.trajectory(&p, &mut xb, &mut yb);
            if !self.collides(&xb, &yb) {
                self.insert(&xb, &yb, p);
            }
            if attempts >= next_ms {
                on_ms(attempts, self.n);
                while attempts >= next_ms {
                    next_ms = (next_ms as f64 * 1.15) as u64 + 1;
                }
            }
        }
        on_ms(attempts, self.n);
    }

    /// Batch-parallel RSA: each round tests `batch` candidates across `threads`
    /// threads (rejections are embarrassingly parallel), then admits survivors
    /// serially with an intra-batch re-check.
    pub fn run_parallel(
        &mut self,
        budget: u64,
        target: usize,
        threads: usize,
        batch: usize,
        base_seed: u64,
        mut on_ms: impl FnMut(u64, usize),
    ) {
        let per = batch.div_ceil(threads);
        let mut attempts: u64 = 0;
        let mut next_ms: u64 = 1000;
        let mut round: u64 = 0;
        while attempts < budget && self.n < target {
            let survivors: Vec<(Path, Vec<f64>, Vec<f64>)> = std::thread::scope(|sc| {
                let eng = &*self;
                let handles: Vec<_> = (0..threads)
                    .map(|ti| {
                        let seed = base_seed
                            ^ (ti as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            ^ round.wrapping_mul(0xD1B5_4A32_D192_ED03);
                        sc.spawn(move || {
                            let mut rng = Rng::seed(seed);
                            let mut xb = vec![0.0; eng.t];
                            let mut yb = vec![0.0; eng.t];
                            let mut out: Vec<(Path, Vec<f64>, Vec<f64>)> = Vec::new();
                            for _ in 0..per {
                                let p = eng.propose(&mut rng);
                                eng.trajectory(&p, &mut xb, &mut yb);
                                if !eng.collides(&xb, &yb) {
                                    out.push((p, xb.clone(), yb.clone()));
                                }
                            }
                            out
                        })
                    })
                    .collect();
                handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
            });
            attempts += (per * threads) as u64;
            round += 1;
            for (p, xb, yb) in survivors {
                if self.n >= target {
                    break;
                }
                if !self.collides(&xb, &yb) {
                    self.insert(&xb, &yb, p);
                }
            }
            if attempts >= next_ms {
                on_ms(attempts, self.n);
                while attempts >= next_ms {
                    next_ms = (next_ms as f64 * 1.15) as u64 + 1;
                }
            }
        }
        on_ms(attempts, self.n);
    }
}
