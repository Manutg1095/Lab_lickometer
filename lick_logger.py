"""
lick_logger.py
----------------

Este script actúa como registrador (logger) para un licómetro de contacto o
de proximidad conectado por puerto serie.  Escucha eventos de lamido
(ya sea recibiendo una línea por lick o una señal digital 0/1) y
genera un archivo ``licks.csv`` con una columna ``t`` que contiene los
tiempos de cada lamido en segundos desde el inicio de la sesión.

El programa soporta dos modos de operación:

``line``
    Se asume que el dispositivo envía una línea de texto (cualquier
    contenido) cada vez que ocurre un lamido.  Cada línea se registra
    como un evento y se aplica un breve período refractario (debounce)
    para evitar contar duplicados.

``ttl``
    Se asume que el dispositivo envía una secuencia de ceros y unos
    (o cualquier representación convertible a 0/1) y el lamido se
    detecta como un flanco ascendente (paso de 0 a 1).  También se
    aplica un período refractario.

El logger escribe dos archivos en la carpeta de salida:

* ``licks.csv``: columna ``t`` con los tiempos de los lamidos (segundos
  desde el inicio).  Este archivo es apto para el analizador
  ``lickometer_analysis.py``.
* ``raw_log.csv``: incluye un timestamp humano, el timestamp de alta
  resolución (performance counter) y el dato recibido por el puerto
  serie (línea o valor).  Sirve para auditoría y diagnóstico.

Controles durante la ejecución (solo Windows):
    - Barra espaciadora: pausa/reanuda la adquisición.
    - Enter (retorno de carro): detiene la sesión.

Uso de ejemplo:

    python lick_logger.py --port COM3 --baud 115200 --mode line --out S1 --minutes 10

Opciones principales:

    --port          Puerto serie (p. ej. COM3, /dev/ttyUSB0)
    --baud          Velocidad en baudios (ej. 115200)
    --mode          'line' o 'ttl'
    --out           Carpeta de salida (se creará si no existe)
    --minutes       Duración de la grabación en minutos (0 = ilimitado)
    --debounce-ms   Tiempo refractario en milisegundos para ignorar eventos muy cercanos
    --flush-n       Número de licks acumulados antes de volcar a disco

Este script está pensado para Windows; el uso de msvcrt permite
manejar las teclas sin bloquear.  En otras plataformas se pueden
adaptar los controles de pausa/fin.
"""

import argparse
import csv
import sys
import time
import threading
import msvcrt
from pathlib import Path

import serial


_paused = False
_stop = False


def _kbhit_loop():
    """Escucha teclado: espacio para pausar/reanudar, Enter para detener."""
    global _paused, _stop
    while not _stop:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b' ':
                _paused = not _paused
                print("\n[Pausado]" if _paused else "\n[Reanudado]")
            elif key in (b'\r', b'\n'):
                _stop = True
                print("\n[Finalizando por usuario]")
        time.sleep(0.03)


def _now_s() -> float:
    """Devuelve el tiempo actual de alta resolución en segundos (perf_counter)."""
    return time.perf_counter()


def _human_time() -> str:
    """Devuelve una cadena con timestamp humano (fecha y hora con milisegundos)."""
    t = time.localtime()
    ms = int((time.time() * 1000) % 1000)
    return time.strftime("%Y-%m-%d %H:%M:%S", t) + f".{ms:03d}"


def _open_serial(port: str, baud: int, timeout: float = 0.01) -> serial.Serial:
    """Intenta abrir el puerto serie especificado y devuelve el objeto Serial."""
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
        print(f"✅ Conectado a {port} @ {baud} baudios")
        return ser
    except Exception as e:
        print(f"❌ No se pudo abrir {port}: {e}")
        sys.exit(1)


def run_line_mode(
    ser: serial.Serial,
    out_dir: Path,
    duration_s: float,
    flush_n: int,
    debounce_ms: float,
) -> None:
    """Modo line: asume que el dispositivo envía una línea por lamido."""
    out_dir.mkdir(parents=True, exist_ok=True)
    licks_csv = out_dir / "licks.csv"
    raw_csv = out_dir / "raw_log.csv"
    start = _now_s()
    last_evt = -1e9
    deb_s = debounce_ms / 1000.0
    # Preparar archivos
    with open(licks_csv, "w", newline="") as f_licks, open(raw_csv, "w", newline="") as f_raw:
        wl = csv.writer(f_licks); wl.writerow(["t"])
        wr = csv.writer(f_raw); wr.writerow(["human_ts", "perf_s", "line"])
        print("▶ Grabando (modo line). Espacio=pause, Enter=stop.")
        buffer: list[list[str]] = []
        cnt = 0
        while not _stop:
            if duration_s and (_now_s() - start) >= duration_s:
                print("⏱️ Tiempo de grabación completado.")
                break
            if _paused:
                time.sleep(0.05)
                continue
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw:
                time.sleep(0.001)
                continue
            ts = _now_s()
            wr.writerow([_human_time(), f"{ts:.6f}", raw])
            # Cada línea se interpreta como un evento
            if (ts - last_evt) >= deb_s:
                t_rel = ts - start
                buffer.append([f"{t_rel:.6f}"])
                last_evt = ts
                cnt += 1
                if cnt % 50 == 0:
                    print(f"Licks: {cnt}   t={t_rel:.2f}s")
            # Vaciar buffer periódicamente
            if len(buffer) >= flush_n:
                wl.writerows(buffer)
                f_licks.flush()
                buffer.clear()
        # Vaciar cualquier evento pendiente
        if buffer:
            wl.writerows(buffer)
            f_licks.flush()
    print(f"✅ Guardado: {licks_csv}  (total licks ≈ {cnt})")
    print(f"🗒️ Log crudo: {raw_csv}")


def run_ttl_mode(
    ser: serial.Serial,
    out_dir: Path,
    duration_s: float,
    flush_n: int,
    debounce_ms: float,
) -> None:
    """Modo ttl: el dispositivo envía 0/1 continuamente; se detectan flancos ascendentes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    licks_csv = out_dir / "licks.csv"
    raw_csv = out_dir / "raw_log.csv"
    start = _now_s()
    last_evt = -1e9
    last_val = 0
    deb_s = debounce_ms / 1000.0
    with open(licks_csv, "w", newline="") as f_licks, open(raw_csv, "w", newline="") as f_raw:
        wl = csv.writer(f_licks); wl.writerow(["t"])
        wr = csv.writer(f_raw); wr.writerow(["human_ts", "perf_s", "value"])
        print("▶ Grabando (modo ttl). Espacio=pause, Enter=stop.")
        buffer: list[list[str]] = []
        cnt = 0
        while not _stop:
            if duration_s and (_now_s() - start) >= duration_s:
                print("⏱️ Tiempo de grabación completado.")
                break
            if _paused:
                time.sleep(0.05)
                continue
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw:
                time.sleep(0.001)
                continue
            ts = _now_s()
            # Convertir raw a un valor 0/1
            try:
                val = int(raw)
                val = 1 if val != 0 else 0
            except Exception:
                digits = "".join(ch for ch in raw if ch in "-0123456789")
                val = int(digits) if digits not in ("", "-") else 0
                val = 1 if val != 0 else 0
            wr.writerow([_human_time(), f"{ts:.6f}", val])
            # Detectar flanco ascendente
            if last_val == 0 and val == 1 and (ts - last_evt) >= deb_s:
                t_rel = ts - start
                buffer.append([f"{t_rel:.6f}"])
                last_evt = ts
                cnt += 1
                if cnt % 50 == 0:
                    print(f"Licks: {cnt}   t={t_rel:.2f}s")
            last_val = val
            if len(buffer) >= flush_n:
                wl.writerows(buffer)
                f_licks.flush()
                buffer.clear()
        if buffer:
            wl.writerows(buffer)
            f_licks.flush()
    print(f"✅ Guardado: {licks_csv}  (total licks ≈ {cnt})")
    print(f"🗒️ Log crudo: {raw_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Logger para licómetro (genera licks.csv con columna t)")
    parser.add_argument("--port", default="COM3", help="Puerto serie (ej. COM3, /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Velocidad en baudios")
    parser.add_argument("--mode", choices=["line", "ttl"], default="line", help="line: una línea por lick; ttl: 0/1 detecta flancos")
    parser.add_argument("--out", default="grabacion", help="Carpeta de salida")
    parser.add_argument("--minutes", type=float, default=10.0, help="Duración de la grabación (minutos), 0 = sin límite")
    parser.add_argument("--debounce-ms", type=float, default=5.0, help="Refractario SW (ms)")
    parser.add_argument("--flush-n", type=int, default=100, help="Volcar a disco cada N licks")
    args = parser.parse_args()

    out_dir = Path(args.out)
    duration_s = 0.0 if args.minutes <= 0 else args.minutes * 60.0
    ser = _open_serial(args.port, args.baud)
    # Hilo para controles de teclado
    t = threading.Thread(target=_kbhit_loop, daemon=True)
    t.start()
    try:
        if args.mode == "line":
            run_line_mode(ser, out_dir, duration_s, args.flush_n, args.debounce_ms)
        else:
            run_ttl_mode(ser, out_dir, duration_s, args.flush_n, args.debounce_ms)
    except KeyboardInterrupt:
        print("\n[CTRL+C] Deteniendo…")
    finally:
        global _stop
        _stop = True
        ser.close()
        print("🔌 Puerto serie cerrado.")


if __name__ == "__main__":
    main()