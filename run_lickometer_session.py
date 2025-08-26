"""
run_lickometer_session.py
-------------------------

Este script encapsula el flujo completo de un experimento con el
licómetro: registra los eventos de lamido durante un intervalo fijo
(por defecto 10 minutos) y, una vez finalizado el registro, ejecuta
el análisis de microestructura utilizando el módulo lickometer_analysis.

Requiere que los scripts `lick_logger.py` y `lickometer_analysis.py` se
encuentren en el mismo directorio.  El logger genera un archivo
``licks.csv`` en la carpeta de salida indicada y el analizador consume
ese archivo para generar las métricas, tablas y gráficas.

Uso desde la línea de comandos:

    python run_lickometer_session.py

Durante la ejecución, el usuario será preguntado por:

* El nombre de la carpeta donde se guardarán los datos y los
  resultados (por ejemplo ``S1_SCOP``).  Si ya existe, se añadirá
  un número al final para evitar sobrescribir.
* El puerto serie (p. ej., ``COM3``).  Se asume que el microcontrolador
  emite una línea por lamido (modo ``line``); si en lugar de ello
  emite una señal digital 0/1, se puede editar la variable `logger_mode`.
* Opcionalmente, la duración de la sesión en minutos (por defecto
  10 minutos).

Al finalizar la sesión, se invocará automáticamente el analizador y
se notificará al usuario la ubicación de los resultados.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def ask_nonempty(prompt: str, default: str | None = None) -> str:
    """Solicita al usuario un texto no vacío.  Si el usuario solo
    pulsa enter y se ha definido `default`, se devolverá ese valor.
    """
    while True:
        answer = input(prompt).strip()
        if answer:
            return answer
        if default is not None:
            return default
        print("Por favor, ingresa un valor válido.")


def unique_directory(base: Path) -> Path:
    """Devuelve una ruta que no exista, añadiendo sufijos en caso de
    conflicto.  Por ejemplo, si `base` existe, intenta con
    `base_1`, `base_2`, etc.
    """
    if not base.exists():
        return base
    i = 1
    while True:
        candidate = base.parent / f"{base.name}_{i}"
        if not candidate.exists():
            return candidate
        i += 1


def run_command(cmd: list[str]) -> int:
    """Ejecuta un comando y devuelve el código de salida.  Imprime
    los argumentos para que el usuario sepa qué se está ejecutando.
    """
    print("\nEjecutando:", " ".join(cmd))
    try:
        completed = subprocess.run(cmd, check=False)
        return completed.returncode
    except FileNotFoundError as e:
        print(f"No se encontró el ejecutable: {e}")
        return 1
    except Exception as e:
        print(f"Error al ejecutar el comando: {e}")
        return 1


def main() -> None:
    print("=== Sesión de licómetro (logger + análisis) ===")

    # Carpeta de salida
    name = ask_nonempty(
        "Nombre de la carpeta donde guardar los datos y resultados: "
    )
    base_dir = Path(name)
    out_dir = unique_directory(base_dir)

    # Puerto serie
    port = ask_nonempty("Puerto serie (ej. COM3): ", default="COM3")

    # Duración
    dur_str = input("Duración de la sesión (minutos, por defecto 10): ").strip()
    if dur_str:
        try:
            duration_min = float(dur_str)
            if duration_min <= 0:
                raise ValueError
        except ValueError:
            print("Duración no válida; se usará el valor por defecto de 10 minutos.")
            duration_min = 10.0
    else:
        duration_min = 10.0

    logger_mode = "line"  # Cambiar a "ttl" si se usa señal digital 0/1
    baudrate = 115200
    debounce_ms = 5.0
    flush_n = 100

    # Construye el comando para el logger
    logger_script = Path(__file__).with_name("lick_logger.py")
    if not logger_script.exists():
        print(f"No se encontró {logger_script}; asegúrate de que lick_logger.py esté en el mismo directorio.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd_logger = [
        sys.executable,
        str(logger_script),
        "--port", port,
        "--baud", str(baudrate),
        "--mode", logger_mode,
        "--out", str(out_dir),
        "--minutes", f"{duration_min}",
        "--debounce-ms", f"{debounce_ms}",
        "--flush-n", str(flush_n),
    ]

    print(f"\nSe grabarán los lamidos durante {duration_min:.1f} minutos.")
    print(f"Los datos se guardarán en: {out_dir}")
    print("Puedes pausar/reanudar con la barra espaciadora y finalizar con Enter.")

    rc = run_command(cmd_logger)
    if rc != 0:
        print("El logger terminó con errores; no se realizará el análisis.")
        sys.exit(rc)

    # Lanza el análisis
    licks_csv = out_dir / "licks.csv"
    if not licks_csv.exists():
        print(f"No se encontró {licks_csv}; no se puede realizar el análisis.")
        sys.exit(1)

    analysis_out = out_dir / "analisis"
    analysis_script = Path(__file__).with_name("lickometer_analysis.py")
    if not analysis_script.exists():
        print(f"No se encontró {analysis_script}; asegúrate de que lickometer_analysis.py esté en el mismo directorio.")
        sys.exit(1)

    cmd_analysis = [
        sys.executable,
        str(analysis_script),
        "--ts", str(licks_csv),
        "--out", str(analysis_out),
    ]

    print("\nEjecutando análisis...")
    rc2 = run_command(cmd_analysis)
    if rc2 == 0:
        print("\nAnálisis completado. Revisa la carpeta de salida para los resultados.")
        print(f"Métricas y figuras en: {analysis_out.resolve()}")
    else:
        print("El análisis terminó con errores.")


if __name__ == "__main__":
    main()