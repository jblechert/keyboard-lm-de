#!/usr/bin/env python3
"""
Benchmarks a GGUF keyboard model on a connected Android device via ADB.

Downloads llama-bench from the latest llama.cpp release, pushes the
binary and model to the device, runs the benchmark, and reports
keyboard-relevant latency (ms per prediction).

Usage:
  .venv_ml/bin/python 17_benchmark_device.py
  .venv_ml/bin/python 17_benchmark_device.py --model data/de_keyboard.gguf
  .venv_ml/bin/python 17_benchmark_device.py --device <serial> --runs 20
  .venv_ml/bin/python 17_benchmark_device.py --pp 64   # longer context
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REMOTE_DIR       = "/data/local/tmp/kblm_bench"
BENCH_BINARY     = "llama-bench"
LLAMA_GITHUB_API = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
CACHE_DIR        = Path(".cache/llama_bench")

# Keyboard-relevant benchmark parameters:
#   pp = prompt tokens processed before first prediction (recent typing context)
#   n  = tokens generated (next-word candidates shown)
#   r  = repetitions for stable average
DEFAULT_PP   = 32
DEFAULT_N    = 5
DEFAULT_RUNS = 10


# ── ADB helpers ──────────────────────────────────────────────────────────────

def adb(*args, device=None):
    cmd = ["adb", *(("-s", device) if device else ()), *args]
    return subprocess.run(cmd, capture_output=True, text=True)


def find_device(serial=None):
    r = adb("devices")
    devices = [
        line.split()[0]
        for line in r.stdout.splitlines()[1:]
        if line.strip() and "device" in line and "offline" not in line
    ]
    if not devices:
        print("Fehler: Kein ADB-Geraet gefunden. USB-Debugging aktiv?", file=sys.stderr)
        sys.exit(1)
    if serial:
        if serial not in devices:
            print(f"Fehler: {serial} nicht gefunden. Verfuegbar: {devices}", file=sys.stderr)
            sys.exit(1)
        return serial
    if len(devices) > 1:
        print(f"Mehrere Geraete: {devices}\n--device angeben.", file=sys.stderr)
        sys.exit(1)
    return devices[0]


def device_abi(device):
    r = adb("shell", "getprop ro.product.cpu.abi", device=device)
    abi = r.stdout.strip()
    if not abi:
        r = adb("shell", "uname -m", device=device)
        abi = "arm64-v8a" if "aarch64" in r.stdout else "x86_64"
    return abi


def device_info(device):
    model   = adb("shell", "getprop ro.product.model",       device=device).stdout.strip()
    android = adb("shell", "getprop ro.build.version.release", device=device).stdout.strip()
    cpus    = adb("shell", "nproc",                           device=device).stdout.strip()
    return model, android, cpus


# ── Model discovery ───────────────────────────────────────────────────────────

def find_gguf(hint=None):
    if hint:
        p = Path(hint)
        if not p.exists():
            print(f"Fehler: Modell nicht gefunden: {hint}", file=sys.stderr)
            sys.exit(1)
        return p
    data = Path("data")
    if not data.exists():
        print("Fehler: data/ nicht gefunden. --model angeben.", file=sys.stderr)
        sys.exit(1)
    ggufs = sorted(data.glob("*.gguf"))
    if not ggufs:
        print("Kein *.gguf in data/ gefunden. Erst 06_convert_to_gguf.py ausfuehren.", file=sys.stderr)
        sys.exit(1)
    # prefer quantized (Q4) models; otherwise largest
    for p in ggufs:
        if "q4" in p.name.lower() or "Q4" in p.name:
            return p
    return max(ggufs, key=lambda p: p.stat().st_size)


# ── llama-bench download ──────────────────────────────────────────────────────

def get_llama_bench(abi, force=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Prefer locally built NEON-only binary (no SVE — matches what Android ships)
    neon_bin = CACHE_DIR / f"llama-bench-{abi}-neon"
    if neon_bin.exists() and not force:
        print(f"  NEON-only binary (lokal gebaut, kein SVE): {neon_bin}")
        return neon_bin
    cached = CACHE_DIR / f"llama-bench-{abi}"
    if cached.exists() and not force:
        print(f"  llama-bench gecacht: {cached}")
        return cached

    print("  Lade llama.cpp Release-Info von GitHub...")
    req = urllib.request.Request(LLAMA_GITHUB_API,
                                 headers={"User-Agent": "kblm-bench/1.0"})
    with urllib.request.urlopen(req) as resp:
        release = json.loads(resp.read())

    tag = release["tag_name"]
    print(f"  Neuestes Release: {tag}")

    # Asset patterns (naming has changed across releases):
    #   old: llama-bNNNN-bin-android-arm64-v8a.zip
    #   new: llama-bNNNN-bin-android-arm64.tar.gz
    short_abi = abi.split("-")[0]   # arm64-v8a -> arm64
    asset = next(
        (a for a in release["assets"]
         if f"android-{abi}" in a["name"] or f"android-{short_abi}" in a["name"]
         and a["name"].endswith((".zip", ".tar.gz"))),
        None
    )
    if not asset:
        names = [a["name"] for a in release["assets"]]
        print(f"  Kein Android Asset in {tag}. Assets: {names}", file=sys.stderr)
        print("  Tipp: llama-bench manuell in .cache/llama_bench/ ablegen.", file=sys.stderr)
        sys.exit(1)

    mb = asset["size"] // 1024 // 1024
    print(f"  Download: {asset['name']} ({mb} MB)...")

    with tempfile.NamedTemporaryFile(suffix=Path(asset["name"]).suffix, delete=False) as tmp:
        urllib.request.urlretrieve(asset["browser_download_url"], tmp.name)
        tmp_path = Path(tmp.name)

    if asset["name"].endswith(".tar.gz"):
        import tarfile
        with tarfile.open(tmp_path) as tf:
            bench_entry = next(
                (m for m in tf.getmembers()
                 if Path(m.name).name in (BENCH_BINARY, f"{BENCH_BINARY}-android")
                 or ("llama-bench" in m.name and not m.isdir())),
                None
            )
            if not bench_entry:
                print(f"  llama-bench nicht im tar. Inhalt: {[m.name for m in tf.getmembers()[:20]]}", file=sys.stderr)
                sys.exit(1)
            f = tf.extractfile(bench_entry)
            data = f.read()
    else:
        with zipfile.ZipFile(tmp_path) as zf:
            bench_entry = next(
                (n for n in zf.namelist()
                 if Path(n).name in (BENCH_BINARY, f"{BENCH_BINARY}-android")
                 or ("llama-bench" in n and not n.endswith("/"))),
                None
            )
            if not bench_entry:
                print(f"  llama-bench nicht im ZIP. Inhalt: {zf.namelist()[:20]}", file=sys.stderr)
                sys.exit(1)
            data = zf.read(bench_entry)

    tmp_path.unlink(missing_ok=True)
    cached.write_bytes(data)
    cached.chmod(0o755)
    print(f"  Gespeichert: {cached} ({len(data)//1024} KB)")
    return cached


# ── Push + run ────────────────────────────────────────────────────────────────

def push_files(device, gguf, bench_bin):
    remote_bin   = f"{REMOTE_DIR}/{BENCH_BINARY}"
    remote_model = f"{REMOTE_DIR}/{gguf.name}"

    adb("shell", f"mkdir -p {REMOTE_DIR}", device=device)

    sz_bin = bench_bin.stat().st_size // 1024
    print(f"  Push binary  ({sz_bin} KB)... ", end="", flush=True)
    r = adb("push", str(bench_bin), remote_bin, device=device)
    if r.returncode != 0:
        print(f"FEHLER\n{r.stderr}", file=sys.stderr); sys.exit(1)
    adb("shell", f"chmod +x {remote_bin}", device=device)
    print("OK")

    sz_model = gguf.stat().st_size // 1024 // 1024
    print(f"  Push model   ({sz_model} MB)... ", end="", flush=True)
    r = adb("push", str(gguf), remote_model, device=device)
    if r.returncode != 0:
        print(f"FEHLER\n{r.stderr}", file=sys.stderr); sys.exit(1)
    print("OK")

    return remote_bin, remote_model


def run_bench(device, remote_bin, remote_model, pp, n, runs):
    cmd = (f"{remote_bin} -m {remote_model} "
           f"-p {pp} -n {n} -r {runs} "
           f"--numa isolate 2>/dev/null || "
           f"{remote_bin} -m {remote_model} -p {pp} -n {n} -r {runs}")
    print(f"\n  Kommando: llama-bench -p {pp} -n {n} -r {runs}")
    print("  Laeuft... (kann 1-2 Min. dauern)\n")
    r = adb("shell", cmd, device=device)
    return r.stdout, r.stderr


# ── Result parsing ────────────────────────────────────────────────────────────

def parse_bench_output(stdout):
    """
    Parse llama-bench markdown table output.
    Handles column order: model | size | params | backend | threads | test | t/s
    Returns list of dicts with keys: test, kind, tokens, t_s
    """
    results = []
    header_cols = None
    for line in stdout.splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]
        # Detect header row
        if "test" in parts and ("t/s" in parts or "t_s" in parts):
            header_cols = parts
            continue
        if header_cols and re.match(r"^[-: ]+$", parts[0] if parts else ""):
            continue  # separator row
        # Find test and t/s columns by header position
        if header_cols:
            try:
                test_idx = next(i for i, h in enumerate(header_cols) if h == "test")
                ts_idx   = next(i for i, h in enumerate(header_cols) if "t/s" in h or "t_s" in h)
                if len(parts) > max(test_idx, ts_idx):
                    test_str = parts[test_idx]
                    ts_str   = parts[ts_idx]
                    m = re.match(r"^(pp|tg)\s*(\d+)$", test_str)
                    ts_m = re.search(r"([\d.]+)", ts_str)
                    if m and ts_m:
                        results.append({
                            "test": test_str,
                            "kind": m.group(1),
                            "tokens": int(m.group(2)),
                            "t_s": float(ts_m.group(1)),
                        })
                    continue
            except StopIteration:
                pass
        # Fallback: scan all columns for pp/tg pattern
        for i, p in enumerate(parts):
            m = re.match(r"^(pp|tg)\s*(\d+)$", p)
            if m:
                ts_str = parts[-1] if parts else ""
                ts_m = re.search(r"([\d.]+)", ts_str)
                if ts_m:
                    results.append({
                        "test": p,
                        "kind": m.group(1),
                        "tokens": int(m.group(2)),
                        "t_s": float(ts_m.group(1)),
                    })
    return results


# ── Report ────────────────────────────────────────────────────────────────────

def report(results, model_name, device_model, android_ver, pp):
    print(f"\n{'='*60}")
    print(f"  Modell:  {model_name}")
    print(f"  Geraet:  {device_model}  (Android {android_ver})")
    print(f"{'='*60}")

    pp_ts = tg_ts = None
    for r in results:
        bar = "#" * int(r["t_s"] / 20)
        print(f"  {r['test']:<8}  {r['t_s']:>8.1f} t/s  {bar}")
        if r["kind"] == "pp":
            pp_ts = r["t_s"]
        else:
            tg_ts = r["t_s"]

    if pp_ts and tg_ts:
        pp_ms  = pp / pp_ts * 1000
        tg_ms  = 1.0 / tg_ts * 1000

        print(f"\n── Keyboard-Latenz ({'─'*38}")
        print(f"  Kalt-Prefill ({pp} Tokens, einmalig): {pp_ms:7.1f} ms")
        print(f"    (nur beim allerersten Aufruf; danach KV-Cache aktiv)")
        print(f"  Laufend: 1 Token generieren:          {tg_ms:7.1f} ms  <- keyboard feel")
        print(f"  Gboard-Ziel:                           <20.0 ms")

        if tg_ms < 5:
            rating = "sehr gut  -- locker Headroom fuer 72M+"
        elif tg_ms < 10:
            rating = "gut       -- Gboard-Level mit KV-Cache"
        elif tg_ms < 20:
            rating = "akzeptabel -- groessere Modelle riskant"
        else:
            rating = "zu langsam -- Modell zu gross fuer dieses Geraet"
        print(f"  Bewertung:                            {rating}")
        print(f"\n  SVE-Headroom: ~20-30% schneller moeglich wenn Kernel SVE aktiviert")
    elif results:
        print("\n  (Konnte Latenz nicht berechnen -- unvollstaendige Ergebnisse)")
    else:
        print("\n  Keine verwertbaren Ergebnisse. Rohe Ausgabe oben.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    "-m", help="GGUF-Datei (default: auto-detect in data/)")
    ap.add_argument("--device",   "-d", help="ADB-Seriennummer")
    ap.add_argument("--pp",       type=int, default=DEFAULT_PP,
                    help=f"Kontext-Laenge in Tokens (default: {DEFAULT_PP})")
    ap.add_argument("--n",        type=int, default=DEFAULT_N,
                    help=f"Generierte Tokens (default: {DEFAULT_N})")
    ap.add_argument("--runs",     type=int, default=DEFAULT_RUNS,
                    help=f"Wiederholungen (default: {DEFAULT_RUNS})")
    ap.add_argument("--redownload", action="store_true",
                    help="llama-bench neu herunterladen (ignoriert Cache)")
    args = ap.parse_args()

    print("── Geraet ──────────────────────────────────────────────")
    device = find_device(args.device)
    abi    = device_abi(device)
    model_name, android_ver, cpus = device_info(device)
    print(f"  {model_name}  Android {android_ver}  ({abi}, {cpus} CPUs)")

    print("\n── Modell ──────────────────────────────────────────────")
    gguf = find_gguf(args.model)
    print(f"  {gguf}  ({gguf.stat().st_size // 1024 // 1024} MB)")

    print("\n── llama-bench ─────────────────────────────────────────")
    bench_bin = get_llama_bench(abi, force=args.redownload)

    print("\n── Push ────────────────────────────────────────────────")
    remote_bin, remote_model = push_files(device, gguf, bench_bin)

    stdout, stderr = run_bench(device, remote_bin, remote_model, args.pp, args.n, args.runs)

    if stdout.strip():
        print(stdout)

    results = parse_bench_output(stdout)
    report(results, gguf.name, model_name, android_ver, args.pp)

    if not results and stderr.strip():
        print("\n[stderr]:", stderr[:1000])


if __name__ == "__main__":
    main()
