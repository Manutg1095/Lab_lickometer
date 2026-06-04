import pandas as pd

def normalize_column(c): return str(c).strip().lower().replace(' ', '_')

def find_header_row_new_format(xlsx_path, sheet_name, max_rows=80):
    preview=pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, nrows=max_rows)
    for i in range(len(preview)):
        toks={normalize_column(x) for x in preview.iloc[i].dropna().tolist()}
        if {'indice_lick','delta_ms'}.issubset(toks): return i
    raise ValueError(f"No encontré encabezado de formato nuevo en hoja: {sheet_name}")

def read_new_sheet(xlsx_path, sheet_name):
    header=find_header_row_new_format(xlsx_path, sheet_name)
    df=pd.read_excel(xlsx_path, sheet_name=sheet_name, header=header)
    df.columns=[normalize_column(c) for c in df.columns]
    required={'indice_lick','t_rel_ms','delta_ms'}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"Faltan columnas {missing} en hoja {sheet_name}")
    for c in ['indice_lick','t_rel_ms','delta_ms']:
        df[c]=pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=['t_rel_ms']).sort_values('t_rel_ms').copy()

def load_workbook_long(xlsx_path, preview_df, params):
    rows=[]
    for _, meta in preview_df.iterrows():
        if meta.get('status')!='OK': continue
        sh=meta['sheet']; df=read_new_sheet(xlsx_path, sh)
        max_ms=float(params['session_max_s'])*1000.0
        df=df[(df['t_rel_ms']>=0)&(df['t_rel_ms']<=max_ms)].copy()
        for col in ['sheet','prefix','experiment','rat_id','session','group_code','group','group_order','drug_bla','drug_ip','dose','dose_mgkg']:
            df[col]=meta.get(col) if col in meta.index else None
        rows.append(df)
    if not rows: raise ValueError('No se encontraron hojas válidas para analizar.')
    return pd.concat(rows, ignore_index=True)
