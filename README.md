# Lick Microstructure Pipeline GUI

Aplicación de escritorio en Python para analizar microestructura de licks desde libros Excel en formato nuevo.

## Ejecución

```bash
pip install -r requirements.txt
python main.py
```

En Mac también puedes usar `run_mac.command`. En Windows, `run_windows.bat`.

## Formato requerido
Cada hoja debe contener columnas `indice_lick`, `t_rel_ms` y `delta_ms`. El programa busca automáticamente la fila de encabezado dentro de las primeras 80 filas.

## Convención de nombres

```text
PREFIJO_##_R##_S#_GRUPO
```

Ejemplos:

```text
OSIP_02_R118_S1_OV
OXO_01_R119_S1_O812
SCL_01_R60_S2_LiCl
HB_02_R6_S2_H03
NB_01_R50_S1_N10
BSC_01_R10_S1_SCOP
```

## Prefijos y grupos

BSC: `Veh`, `SCOP`.

OXO: `Veh`, `O812`, `O227`, `O60`.

OSIP: `VV`, `OV`, `VS`, `OS`.

HB: `Veh`, `H03`, `H10`, `H30`.

NB: `Veh`, `N03`, `N10`, `N30`.

SCL: `Veh`, `NB`, `HB`, `LiCl`.

## Parámetros default

- Duración máxima de sesión: 600 s
- ILI mínimo válido: 60 ms
- Umbral de ráfaga: 330 ms
- Mínimo de licks por ráfaga: 3
- Bins histograma: 5 ms
- Histograma principal: 60–500 ms
- Rango intra-burst bajo: 60–220 ms
- Rango intra-burst alto: 220–330 ms

## Salidas

- Excel completo con hojas de metadata, grupos, por animal, resumen, histograma, licks por minuto y GraphPad.
- Figuras PNG.
- Log de validación.

## Definiciones

Ráfaga: secuencia de al menos 3 licks consecutivos con ILI ≤330 ms, editable desde el panel.

IBI corregido: inicio de la siguiente ráfaga válida menos fin de la ráfaga previa.

Histograma: frecuencia relativa por animal y después media grupal ± SEM.
