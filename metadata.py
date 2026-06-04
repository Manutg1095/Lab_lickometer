import re
import pandas as pd
from config import GROUP_MAPS, SKIP_SHEETS

def normalize_token(x: str) -> str:
    return str(x).strip().upper().replace("-", "_").replace(" ", "_")

def parse_common_sheet_name(sheet_name: str):
    pat = r"^(?P<prefix>[A-Za-z]+)_(?P<exp>\d+)_R(?P<rat>\d+)_(?P<session>S[12])_(?P<group>.+)$"
    m = re.match(pat, sheet_name, flags=re.IGNORECASE)
    if not m: return None
    d = m.groupdict(); prefix = normalize_token(d['prefix']); group_code = normalize_token(d['group'])
    return {"prefix": prefix, "experiment": f"{prefix}_{d['exp']}", "rat_id": f"R{d['rat']}", "session": d['session'].upper(), "group_code": group_code}

def classify_sheet(sheet_name: str) -> dict:
    parsed = parse_common_sheet_name(sheet_name)
    if parsed is None:
        return {"sheet": sheet_name, "status": "ERROR", "error": "Nombre de hoja no coincide con PREFIJO_##_R##_S#_GRUPO."}
    prefix, group_code = parsed['prefix'], parsed['group_code']
    if prefix not in GROUP_MAPS:
        return {**parsed, "sheet": sheet_name, "status": "ERROR", "error": f"Prefijo experimental no reconocido: {prefix}"}
    if group_code not in GROUP_MAPS[prefix]:
        return {**parsed, "sheet": sheet_name, "status": "ERROR", "error": f"Grupo no reconocido para {prefix}: {group_code}"}
    info = GROUP_MAPS[prefix][group_code]
    return {**parsed, "sheet": sheet_name, "group": info['group'], "group_order": info['order'], "drug_bla": info.get('drug_bla'), "drug_ip": info.get('drug_ip'), "dose": info.get('dose'), "dose_mgkg": info.get('dose_mgkg'), "status": "OK", "error": ""}

def preview_workbook_groups(xlsx_path: str) -> pd.DataFrame:
    xl = pd.ExcelFile(xlsx_path); rows=[]
    for sh in xl.sheet_names:
        if sh.lower() in SKIP_SHEETS: continue
        rows.append(classify_sheet(sh))
    df = pd.DataFrame(rows)
    if not df.empty:
        cols=[c for c in ['experiment','session','group_order','rat_id'] if c in df.columns]
        if cols: df=df.sort_values(cols, na_position='last')
    return df.reset_index(drop=True)

def summarize_detected_groups(preview_df: pd.DataFrame) -> pd.DataFrame:
    if preview_df.empty or 'status' not in preview_df: return pd.DataFrame()
    ok=preview_df[preview_df['status']=='OK'].copy()
    if ok.empty: return pd.DataFrame()
    return (ok.groupby(['prefix','experiment','session','group','group_order'], dropna=False)
              .agg(n_hojas=('sheet','count'), n_ratas=('rat_id','nunique'))
              .reset_index().sort_values(['experiment','session','group_order']).reset_index(drop=True))
