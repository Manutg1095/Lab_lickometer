"""
lickometer_analysis.py
----------------------

Script para analizar la microestructura de lamidos a partir de un
archivo CSV con timestamps.  Produce métricas básicas (total de
lamidos, estadísticos de ILIs, ráfagas, eficiencia, etc.), exporta
tablas y genera gráficas de raster, histograma de ILIs y curva
acumulada de lamidos.

El CSV de entrada debe contener al menos una columna llamada ``t`` con
los tiempos de los lamidos.  Estos tiempos pueden estar en
microsegundos, milisegundos o segundos; el script detecta
automáticamente la unidad y convierte a segundos.  También se puede
especificar ``--ttl`` si el archivo contiene columnas ``t`` y
``value`` de una señal digital (por ejemplo, flancos de una entrada TTL);
en ese caso se detectan los flancos ascendentes como lamidos.

Uso de ejemplo:

    python lickometer_analysis.py --ts licks.csv --out analisis

Opciones principales:

    --ts FILE       Archivo CSV con columna t (o t,value con --ttl)
    --out DIR       Carpeta de salida para los resultados
    --ttl           Interpretar CSV como señal digital (t,value)
    --ili-min ms    ILI mínimo (ms) para análisis (descarta artefactos)
    --ili-max ms    ILI máximo (ms) para análisis (descarta pausas muy largas)
    --burst-th ms   Umbral para detectar ráfagas (ILIs < burst_th)
    --min-burst-licks N  Mínimo de lamidos por ráfaga
    --bin-ms ms     Bin para histograma de ILIs (ms)

El script guarda las métricas en ``session_metrics.csv`` y los datos
intermedios en archivos adicionales.  También crea las figuras
``raster.png``, ``ili_hist.png`` y ``cumulative.png`` dentro de la
carpeta de salida.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _detect_units(ts: np.ndarray) -> str:
    """Intenta deducir la unidad de los timestamps basándose en el valor máximo."""
    m = np.nanmax(ts) if ts.size else 0.0
    if m > 1e6:
        return "us"
    if m > 1e3:
        return "ms"
    return "s"


def _to_seconds(ts: np.ndarray) -> tuple[np.ndarray, str]:
    """Convierte a segundos según la unidad detectada y devuelve el array y la unidad."""
    units = _detect_units(ts)
    if units == "us":
        return ts / 1e6, units
    if units == "ms":
        return ts / 1e3, units
    return ts, units


def load_timestamps(path: str, ttl: bool = False) -> tuple[np.ndarray, str]:
    """Carga un CSV con timestamps.  Si ``ttl`` es True, busca flancos ascendentes."""
    df = pd.read_csv(path)
    if ttl:
        if not {"t", "value"}.issubset(df.columns):
            raise ValueError("Modo --ttl requiere columnas: t,value")
        t = df["t"].to_numpy()
        v = df["value"].to_numpy().astype(int)
        rising = np.where((v[1:] == 1) & (v[:-1] == 0))[0] + 1
        ts = t[rising]
    else:
        if "t" not in df.columns:
            raise ValueError("El CSV debe tener una columna llamada 't'")
        ts = df["t"].to_numpy()
    ts = ts[~np.isnan(ts)]
    ts = np.sort(ts)
    ts_sec, units = _to_seconds(ts)
    if ts_sec.size == 0:
        raise ValueError("No hay timestamps válidos tras limpieza.")
    ts_sec = np.unique(ts_sec)  # eliminar duplicados exactos en segundos
    return ts_sec, units


def detect_bursts(
    ts: np.ndarray, ili_ms: np.ndarray, burst_th: float = 300.0, min_licks_burst: int = 3
) -> list[dict]:
    """Detecta ráfagas (runs de ILIs < burst_th) y devuelve una lista de diccionarios."""
    if ili_ms.size == 0:
        return []
    in_burst = ili_ms < burst_th
    bursts: list[dict] = []
    run_start: int | None = None
    for i, ok in enumerate(in_burst):
        if ok and run_start is None:
            run_start = i
        if (not ok or i == len(in_burst) - 1) and run_start is not None:
            run_end = i if not ok else i
            n_ilis = (run_end - run_start + 1)
            size = n_ilis + 1
            if size >= min_licks_burst:
                lick_start_idx = run_start
                lick_end_idx = run_end + 1
                start_s = ts[lick_start_idx]
                end_s = ts[lick_end_idx]
                bursts.append({
                    "start_s": float(start_s),
                    "end_s": float(end_s),
                    "size": int(size),
                    "duration_s": float(max(0.0, end_s - start_s)),
                    "start_idx": int(lick_start_idx),
                    "end_idx": int(lick_end_idx),
                })
            run_start = None
    return bursts


def analyze_session(
    ts: np.ndarray,
    ili_min: float = 60.0,
    ili_max: float = 1000.0,
    burst_th: float = 300.0,
    min_licks_burst: int = 3,
    bin_ms: float = 5.0,
) -> dict:
    """Analiza la sesión: calcula ILIs, ráfagas y métricas, y prepara datos para graficar."""
    if ts.size < 2:
        raise ValueError("Muy pocos eventos para análisis (<2 lamidos).")
    ilis_ms_all = np.diff(ts) * 1000.0
    valid_mask = (ilis_ms_all >= ili_min) & (ilis_ms_all <= ili_max)
    ilis_ms = ilis_ms_all[valid_mask]
    # Estadísticas de ILIs
    ili_mean = float(np.mean(ilis_ms)) if ilis_ms.size else np.nan
    ili_sd = float(np.std(ilis_ms, ddof=1)) if ilis_ms.size > 1 else np.nan
    ili_median = float(np.median(ilis_ms)) if ilis_ms.size else np.nan
    # Porcentajes por rangos
    p60_180 = float(100.0 * np.mean((ilis_ms >= 60) & (ilis_ms <= 180))) if ilis_ms.size else np.nan
    p180_300 = float(100.0 * np.mean((ilis_ms > 180) & (ilis_ms <= 300))) if ilis_ms.size else np.nan
    pgt300 = float(100.0 * np.mean(ilis_ms > 300)) if ilis_ms.size else np.nan
    # Ráfagas
    bursts = detect_bursts(ts, ilis_ms_all, burst_th=burst_th, min_licks_burst=min_licks_burst)
    burst_sizes = np.array([b["size"] for b in bursts], float) if bursts else np.array([], float)
    burst_count = int(len(bursts))
    burst_size_mean = float(np.mean(burst_sizes)) if burst_sizes.size else 0.0
    # Lick efficiency: ILIs 60–180 ms dentro de ráfagas (aprox. ILIs < 300 ms)
    in_burst_ili = ilis_ms < 300
    primary = (ilis_ms >= 60) & (ilis_ms <= 180)
    lick_eff = float(np.sum(primary & in_burst_ili) / max(1, np.sum(in_burst_ili))) if ilis_ms.size else np.nan
    # Duración y licks/min
    duration_s = float(ts[-1] - ts[0]) if ts[-1] > ts[0] else float(ts[-1])
    edges_min = np.arange(0, math.ceil(duration_s) + 1, 60.0) if duration_s > 0 else np.array([0, 60])
    counts_min, _ = np.histogram(ts - ts[0], bins=edges_min)
    licks_per_min = pd.DataFrame({"minute_end_s": edges_min[1:], "licks": counts_min})
    cum = np.cumsum(counts_min)
    if cum.size and cum[-1] > 0:
        eighty = 0.8 * cum[-1]
        idx80 = int(np.searchsorted(cum, eighty))
        t80 = float(edges_min[min(idx80, len(edges_min) - 1)])
    else:
        t80 = np.nan
    # Pausa más larga (ILI máximo sin filtrar)
    longest_pause_ms = float(np.max(ilis_ms_all)) if ilis_ms_all.size else np.nan
    # Histograma de ILIs (para 60–500 ms)
    bins = np.arange(60, 500 + bin_ms, bin_ms)
    hist_mask = (ilis_ms >= 60) & (ilis_ms <= 500)
    hist_vals, hist_edges = np.histogram(ilis_ms[hist_mask], bins=bins, density=True)
    hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
    metrics = {
        "total_licks": int(ts.size),
        "session_duration_s": duration_s,
        "ili_mean_ms": ili_mean,
        "ili_median_ms": ili_median,
        "ili_sd_ms": ili_sd,
        "pct_ili_60_180": p60_180,
        "pct_ili_180_300": p180_300,
        "pct_ili_gt_300": pgt300,
        "burst_count": burst_count,
        "burst_size_mean": burst_size_mean,
        "lick_efficiency": lick_eff,
        "intake80_time_s": t80,
        "longest_pause_ms": longest_pause_ms,
    }
    results = {
        "metrics": metrics,
        "ilis_ms_all": ilis_ms_all,
        "ilis_ms": ilis_ms,
        "hist_centers_ms": hist_centers,
        "hist_density": hist_vals,
        "ts_s": ts,
        "bursts": bursts,
        "licks_per_min_df": licks_per_min,
    }
    return results


def plot_all(res: dict, outdir: str | Path, title: str = "Session") -> None:
    """Genera las figuras de raster, histograma de ILIs y curva acumulada."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ts = res["ts_s"]
    t0 = ts[0]
    rel = ts - t0
    # Raster
    plt.figure()
    plt.vlines(rel, 0, 1, linewidth=1)
    for b in res["bursts"]:
        plt.axvspan(b["start_s"] - t0, b["end_s"] - t0, alpha=0.18)
    plt.xlabel("Tiempo (s)")
    plt.yticks([])
    plt.title(f"Raster {title}")
    plt.tight_layout()
    plt.savefig(out / "raster.png", dpi=220)
    plt.close()
    # Histograma ILIs
    plt.figure()
    plt.plot(res["hist_centers_ms"], res["hist_density"])
    plt.xlabel("ILI (ms)")
    plt.ylabel("Densidad")
    plt.title("Histograma ILI (60–500 ms)")
    plt.tight_layout()
    plt.savefig(out / "ili_hist.png", dpi=220)
    plt.close()
    # Licks/min y acumulado
    lpm = res["licks_per_min_df"]
    if not lpm.empty:
        cum = np.cumsum(lpm["licks"].to_numpy())
        plt.figure()
        plt.step(lpm["minute_end_s"], lpm["licks"], where="mid", label="Licks/min")
        ax2 = plt.twinx()
        if cum.size:
            ax2.step(lpm["minute_end_s"], 100 * cum / max(1, cum[-1]), where="mid", label="Acum (%)")
        plt.xlabel("Tiempo (s, fin de minuto)")
        plt.title("Licks/min y acumulado")
        plt.tight_layout()
        plt.savefig(out / "cumulative.png", dpi=220)
        plt.close()


def save_results(res: dict, outdir: str | Path, params: dict) -> None:
    """Guarda métricas, ILIs, histograma, ráfagas y licks/min como CSV, y parámetros en JSON."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    # Métricas
    pd.DataFrame([res["metrics"]]).to_csv(out / "session_metrics.csv", index=False)
    # ILIs
    pd.DataFrame({"ili_ms_all": res["ilis_ms_all"]}).to_csv(out / "ilis_all.csv", index=False)
    pd.DataFrame({"ili_ms": res["ilis_ms"]}).to_csv(out / "ilis.csv", index=False)
    # Histograma
    pd.DataFrame({"ili_center_ms": res["hist_centers_ms"], "density": res["hist_density"]}).to_csv(out / "ili_hist.csv", index=False)
    # Ráfagas
    pd.DataFrame(res["bursts"]).to_csv(out / "bursts.csv", index=False)
    # Licks/min
    res["licks_per_min_df"].to_csv(out / "licks_per_min.csv", index=False)
    # Parámetros
    with open(out / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Análisis de microestructura de lamidos")
    parser.add_argument("--ts", required=True, help="CSV con columna t (o t,value si --ttl)")
    parser.add_argument("--out", required=True, help="Carpeta de salida")
    parser.add_argument("--ttl", action="store_true", help="Interpretar CSV como señal TTL (flanco ascendente)")
    parser.add_argument("--ili-min", type=float, default=60.0, help="ILI mínimo (ms)")
    parser.add_argument("--ili-max", type=float, default=1000.0, help="ILI máximo (ms)")
    parser.add_argument("--burst-th", type=float, default=300.0, help="Umbral de ráfaga (ms)")
    parser.add_argument("--min-burst-licks", type=int, default=3, help="Mínimo de lamidos por ráfaga")
    parser.add_argument("--bin-ms", type=float, default=5.0, help="Tamaño de bin para histograma (ms)")
    args = parser.parse_args()
    try:
        ts, units = load_timestamps(args.ts, ttl=args.ttl)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar '{args.ts}': {e}", file=sys.stderr)
        sys.exit(1)
    params = {
        "source_file": args.ts,
        "ttl_mode": bool(args.ttl),
        "input_units_detected": units,
        "ili_min_ms": float(args.ili_min),
        "ili_max_ms": float(args.ili_max),
        "burst_th_ms": float(args.burst_th),
        "min_licks_burst": int(args.min_burst_licks),
        "bin_ms": float(args.bin_ms),
    }
    res = analyze_session(
        ts,
        ili_min=args.ili_min,
        ili_max=args.ili_max,
        burst_th=args.burst_th,
        min_licks_burst=args.min_burst_licks,
        bin_ms=args.bin_ms,
    )
    save_results(res, args.out, params)
    plot_all(res, args.out, title=Path(args.ts).stem)
    m = res["metrics"]
    print(f"[OK] Resultados en: {Path(args.out).resolve()}")
    print(
        f"Licks={m['total_licks']} | Duración={m['session_duration_s']:.1f}s | "
        f"ILIμ={m['ili_mean_ms']:.1f}±{(m['ili_sd_ms'] or 0):.1f} ms | "
        f"Ráfagas={m['burst_count']} (μ tamaño {m['burst_size_mean']:.2f}) | "
        f"Eff={m['lick_efficiency']:.2f} | t80={m['intake80_time_s']:.1f}s | "
        f"PausaMax={m['longest_pause_ms']:.1f} ms"
    )


if __name__ == "__main__":
    main()