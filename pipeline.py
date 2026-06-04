from pathlib import Path
from metadata import preview_workbook_groups, summarize_detected_groups
from io_excel import load_workbook_long
from analysis import analyze_per_animal, summarize_group_session, make_histograms, licks_per_minute
from exporter import export_excel
from plotting import export_figures

def run_pipeline(xlsx_path, output_dir, params, log_fn=print):
    xlsx_path=Path(xlsx_path); output_dir=Path(output_dir); excel_dir=output_dir/'01_excel'; fig_dir=output_dir/'02_figuras'; log_dir=output_dir/'03_logs'
    excel_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    log_fn('Leyendo y validando hojas...'); preview_df=preview_workbook_groups(str(xlsx_path)); group_summary=summarize_detected_groups(preview_df)
    if preview_df.empty: raise ValueError('No se detectaron hojas analizables.')
    if len(preview_df[preview_df['status']=='ERROR']): raise ValueError('Hay errores de detección de hojas. Revisa la tabla de validación antes de ejecutar.')
    log_fn('Cargando datos en formato nuevo...'); long_df=load_workbook_long(str(xlsx_path), preview_df, params)
    log_fn('Calculando métricas por animal...'); per_animal=analyze_per_animal(long_df, params)
    log_fn('Resumiendo por grupo y sesión...'); summary=summarize_group_session(per_animal)
    log_fn('Calculando histogramas S2...'); hist_animal,hist_summary=make_histograms(long_df, params)
    log_fn('Calculando licks por minuto...'); licks_minute,licks_minute_summary=licks_per_minute(long_df, params)
    out_excel=excel_dir/f'{xlsx_path.stem}_microestructura_completo.xlsx'; log_fn(f'Exportando Excel: {out_excel}')
    export_excel(out_excel, preview_df, group_summary, long_df, per_animal, summary, hist_animal, hist_summary, licks_minute, licks_minute_summary, params)
    log_fn('Exportando figuras...'); export_figures(per_animal, hist_summary, fig_dir)
    validation_path=log_dir/'reporte_validacion.txt'
    with open(validation_path,'w',encoding='utf-8') as f:
        f.write('REPORTE DE VALIDACIÓN\n\nParámetros:\n'); [f.write(f'- {k}: {v}\n') for k,v in params.items()]; f.write('\nGrupos encontrados:\n'); f.write(group_summary.to_string(index=False)); f.write('\n\nHojas detectadas:\n'); f.write(preview_df.to_string(index=False))
    log_fn('Análisis terminado.'); return {'excel':str(out_excel),'fig_dir':str(fig_dir),'log':str(validation_path)}
