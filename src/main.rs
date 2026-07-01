//! braid_engine — command-line RSA packing engine for the 2+1 universe model.
//!
//! Packs as many non-intersecting worldlines as possible at a fixed time-step
//! resolution and records the approach-to-jamming growth curve N(attempts).
//!
//! Run `braid_engine --help` for usage.

mod engine;

use engine::Engine;
use std::time::Instant;

fn print_help() {
    eprintln!(
        "braid_engine — RSA packing engine (2+1 braided universe)

USAGE:
    braid_engine [OPTIONS]

OPTIONS:
    -t, --timesteps <N>   time-step resolution T  (sets CELL=2/T, maxfreq=T/2)   [default: 200]
        --attempts  <N>   max candidate proposals (accepts 1e8 / 2.5e9 style)    [default: 1e8]
        --target    <N>   stop early once N paths are packed                     [default: none]
        --threads   <N>   worker threads (1 = sequential)                        [default: 1]
        --batch     <N>   candidates tested per parallel round                   [default: 8192]
        --smart           coprime + high-frequency proposal bias (default on)
        --uniform         plain uniform frequency proposals (natural RSA)
        --seed      <N>   RNG seed                                               [default: 12345]
        --curve     <F>   write the N(attempts) growth curve as CSV (else stdout)
        --out       <F>   write the packed path set as CSV (ax,ay,bx,by,ax2,ay2)
    -h, --help            show this help

EXAMPLES:
    braid_engine -t 200 --attempts 2e9 --threads 16 --curve curve.csv
    braid_engine -t 120 --target 900 --threads 16 --out paths.csv"
    );
}

/// Parse a count that may be written like 1000000, 1e8, 2.5e9.
fn parse_count(s: &str) -> u64 {
    if let Ok(v) = s.parse::<u64>() {
        return v;
    }
    s.parse::<f64>().map(|v| v as u64).unwrap_or_else(|_| {
        eprintln!("bad number: {s}");
        std::process::exit(1);
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut t = 200usize;
    let mut budget = 100_000_000u64;
    let mut target = usize::MAX;
    let mut threads = 1usize;
    let mut batch = 8192usize;
    let mut smart = true;
    let mut seed = 12345u64;
    let mut curve_path: Option<String> = None;
    let mut out_path: Option<String> = None;

    let mut i = 1;
    let need = |i: usize| -> &String {
        args.get(i + 1).unwrap_or_else(|| {
            eprintln!("missing value for {}", args[i]);
            std::process::exit(1);
        })
    };
    while i < args.len() {
        match args[i].as_str() {
            "-t" | "--timesteps" => {
                t = need(i).parse().unwrap();
                i += 2;
            }
            "--attempts" => {
                budget = parse_count(need(i));
                i += 2;
            }
            "--target" => {
                target = need(i).parse().unwrap();
                i += 2;
            }
            "--threads" => {
                threads = need(i).parse().unwrap();
                i += 2;
            }
            "--batch" => {
                batch = need(i).parse().unwrap();
                i += 2;
            }
            "--smart" => {
                smart = true;
                i += 1;
            }
            "--uniform" => {
                smart = false;
                i += 1;
            }
            "--seed" => {
                seed = need(i).parse().unwrap();
                i += 2;
            }
            "--curve" => {
                curve_path = Some(need(i).clone());
                i += 2;
            }
            "--out" => {
                out_path = Some(need(i).clone());
                i += 2;
            }
            "-h" | "--help" => {
                print_help();
                return;
            }
            other => {
                eprintln!("unknown argument: {other}  (try --help)");
                std::process::exit(1);
            }
        }
    }

    if t < 2 {
        eprintln!("--timesteps must be >= 2");
        std::process::exit(1);
    }

    eprintln!(
        "braid_engine: T={t} (CELL={:.5}, maxfreq={}), {}, threads={}, budget={} attempts",
        2.0 / t as f64,
        (t / 2).max(2),
        if smart { "smart" } else { "uniform" },
        threads,
        budget
    );

    let mut eng = Engine::new(t, smart);
    let mut curve: Vec<(u64, usize)> = Vec::new();
    let record = |a: u64, n: usize, curve: &mut Vec<(u64, usize)>| {
        if curve.last().map(|&(la, _)| la) != Some(a) {
            curve.push((a, n));
        }
    };

    let start = Instant::now();
    if threads <= 1 {
        eng.run_sequential(budget, target, seed, |a, n| record(a, n, &mut curve));
    } else {
        eng.run_parallel(budget, target, threads, batch, seed, |a, n| record(a, n, &mut curve));
    }
    let elapsed = start.elapsed().as_secs_f64();

    let (final_att, final_n) = curve.last().copied().unwrap_or((0, eng.n));
    eprintln!(
        "done: N={} in {} attempts, {:.1}s ({:.2}M attempts/s)",
        final_n,
        final_att,
        elapsed,
        final_att as f64 / elapsed / 1e6
    );

    // growth curve CSV
    let mut csv = String::from("attempts,n\n");
    for (a, n) in &curve {
        csv.push_str(&format!("{a},{n}\n"));
    }
    match &curve_path {
        Some(p) => {
            std::fs::write(p, csv).expect("write curve");
            eprintln!("curve -> {p} ({} rows)", curve.len());
        }
        None => print!("{csv}"),
    }

    // packed path set
    if let Some(p) = &out_path {
        let mut s = String::from("ax,ay,bx,by,ax2,ay2\n");
        for pa in &eng.paths {
            s.push_str(&format!(
                "{},{},{},{},{},{}\n",
                pa.ax, pa.ay, pa.bx, pa.by, pa.ax2, pa.ay2
            ));
        }
        std::fs::write(p, s).expect("write paths");
        eprintln!("paths -> {p} ({} rows)", eng.n);
    }
}
