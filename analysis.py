import math
import numpy as np
import pandas as pd

def sem(x):
    x=pd.Series(x).dropna()
    return 0.0 if len(x)<2 else float(x.std(ddof=1)/math.sqrt(len(x)))

def detect_bursts(times_ms, params):
    if len(times_ms)<int(params['min_licks_per_burst']): return []
    min_ili=float(params['min_ili_ms']); thr=float(params['burst_threshold_ms']); min_licks=int(params['min_licks_per_burst'])
    bursts=[]; cur=[float(times_ms[0])]
    for i in range(1,len(times_ms)):
        ili=float(times_ms[i]-times_ms[i-1])
        if min_ili<=ili<=thr: cur.append(float(times_ms[i]))
        else:
            if len(cur)>=min_licks: bursts.append({'start_ms':cur[0],'end_ms':cur[-1],'n_licks':len(cur),'duration_ms':cur[-1]-cur[0]})
            cur=[float(times_ms[i])]
    if len(cur)>=min_licks: bursts.append({'start_ms':cur[0],'end_ms':cur[-1],'n_licks':len(cur),'duration_ms':cur[-1]-cur[0]})
    return bursts

def analyze_per_animal(long_df, params):
    rows=[]
    group_cols=['prefix','experiment','rat_id','session','group','group_code','group_order','drug_bla','drug_ip','dose','dose_mgkg','sheet']
    for keys, sub in long_df.groupby(group_cols, dropna=False):
        meta=dict(zip(group_cols, keys)); sub=sub.sort_values('t_rel_ms')
        times=sub['t_rel_ms'].dropna().to_numpy(float); ili=sub['delta_ms'].dropna().to_numpy(float)
        min_ili=float(params['min_ili_ms']); low_min=float(params['low_range_min_ms']); low_max=float(params['low_range_max_ms']); high_min=float(params['high_range_min_ms']); high_max=float(params['high_range_max_ms'])
        valid=ili[ili>=min_ili]; low=ili[(ili>=low_min)&(ili<low_max)]; high=ili[(ili>=high_min)&(ili<=high_max)]; allin=ili[(ili>=low_min)&(ili<=high_max)]; gt=ili[(ili>high_max)&(ili<=1000)]
        bursts=detect_bursts(times, params)
        sizes=np.array([b['n_licks'] for b in bursts], float); durs=np.array([b['duration_ms'] for b in bursts], float)
        ibis=np.array([bursts[i+1]['start_ms']-bursts[i]['end_ms'] for i in range(len(bursts)-1)], float)
        denom_60_330=len(allin); denom_60_1000=len(ili[(ili>=low_min)&(ili<=1000)])
        rows.append({**meta,
            'n_licks':len(times), 'latencia_primer_lick_ms':float(times[0]) if len(times) else np.nan, 'latencia_primer_burst_ms':bursts[0]['start_ms'] if bursts else np.nan,
            'n_ILI_validos':len(valid),
            'n_ILI_60_220':len(low), 'media_ILI_60_220_ms':np.mean(low) if len(low) else np.nan, 'sd_ILI_60_220_ms':np.std(low,ddof=1) if len(low)>1 else np.nan,
            'n_ILI_220_330':len(high), 'media_ILI_220_330_ms':np.mean(high) if len(high) else np.nan, 'sd_ILI_220_330_ms':np.std(high,ddof=1) if len(high)>1 else np.nan,
            'n_ILI_60_330':len(allin), 'media_ILI_60_330_ms':np.mean(allin) if len(allin) else np.nan, 'sd_ILI_60_330_ms':np.std(allin,ddof=1) if len(allin)>1 else np.nan,
            'pct_ILI_60_220_de_60_330':100*len(low)/denom_60_330 if denom_60_330 else np.nan, 'pct_ILI_220_330_de_60_330':100*len(high)/denom_60_330 if denom_60_330 else np.nan, 'pct_ILI_gt330_de_60_1000':100*len(gt)/denom_60_1000 if denom_60_1000 else np.nan,
            'n_bursts':len(bursts), 'tamano_rafaga_media_licks':np.mean(sizes) if len(sizes) else np.nan, 'duracion_rafaga_media_ms':np.mean(durs) if len(durs) else np.nan,
            'n_IBI_corregido':len(ibis), 'IBI_corregido_media_ms':np.mean(ibis) if len(ibis) else np.nan})
    return pd.DataFrame(rows)

def summarize_group_session(per_animal):
    metrics=['n_licks','latencia_primer_lick_ms','latencia_primer_burst_ms','media_ILI_60_220_ms','sd_ILI_60_220_ms','media_ILI_220_330_ms','sd_ILI_220_330_ms','media_ILI_60_330_ms','sd_ILI_60_330_ms','pct_ILI_60_220_de_60_330','pct_ILI_220_330_de_60_330','pct_ILI_gt330_de_60_1000','n_bursts','tamano_rafaga_media_licks','duracion_rafaga_media_ms','IBI_corregido_media_ms']
    rows=[]; keys=['prefix','experiment','session','group','group_order','drug_bla','drug_ip','dose','dose_mgkg']
    for key_vals, sub in per_animal.groupby(keys, dropna=False):
        row=dict(zip(keys,key_vals)); row['n_animales']=sub['rat_id'].nunique()
        for m in metrics:
            vals=pd.to_numeric(sub[m], errors='coerce').dropna(); row[f'{m}_media']=vals.mean() if len(vals) else np.nan; row[f'{m}_SEM']=sem(vals); row[f'{m}_SD']=vals.std(ddof=1) if len(vals)>1 else np.nan
        rows.append(row)
    out=pd.DataFrame(rows)
    return out.sort_values(['experiment','session','group_order','group']) if not out.empty else out

def make_histograms(long_df, params):
    min_ms=float(params['hist_range_min_ms']); max_ms=float(params['hist_range_max_ms']); bin_ms=float(params['hist_bin_ms'])
    bins=np.arange(min_ms, max_ms+bin_ms, bin_ms); rows=[]
    cols=['prefix','experiment','rat_id','session','group','group_code','group_order','drug_bla','drug_ip','dose','dose_mgkg','sheet']
    s2=long_df[long_df['session']=='S2'].copy()
    for keys, sub in s2.groupby(cols, dropna=False):
        meta=dict(zip(cols,keys)); ili=pd.to_numeric(sub['delta_ms'],errors='coerce').dropna().to_numpy(float); vals=ili[(ili>=min_ms)&(ili<=max_ms)]
        if len(vals)==0: continue
        counts, edges=np.histogram(vals,bins=bins); total=counts.sum()
        for c,b0,b1 in zip(counts,edges[:-1],edges[1:]): rows.append({**meta,'bin_inicio_ms':b0,'bin_fin_ms':b1,'bin_centro_ms':(b0+b1)/2,'conteo_bin':int(c),'n_ILI_rango_hist':int(total),'freq_rel':float(c/total) if total else np.nan})
    ha=pd.DataFrame(rows)
    if ha.empty: return ha, pd.DataFrame()
    srows=[]; keys=['prefix','experiment','group','group_order','drug_bla','drug_ip','dose','dose_mgkg','bin_inicio_ms','bin_fin_ms','bin_centro_ms']
    for key_vals, sub in ha.groupby(keys, dropna=False):
        row=dict(zip(keys,key_vals)); vals=pd.to_numeric(sub['freq_rel'],errors='coerce').dropna(); row['n_animales']=sub['rat_id'].nunique(); row['freq_rel_media']=vals.mean(); row['freq_rel_SD']=vals.std(ddof=1) if len(vals)>1 else np.nan; row['freq_rel_SEM']=sem(vals); srows.append(row)
    hs=pd.DataFrame(srows).sort_values(['experiment','group_order','bin_inicio_ms'])
    return ha, hs

def licks_per_minute(long_df, params):
    n_minutes=int(math.ceil(float(params['session_max_s'])/60)); rows=[]
    cols=['prefix','experiment','rat_id','session','group','group_code','group_order','drug_bla','drug_ip','dose','dose_mgkg','sheet']
    for keys, sub in long_df.groupby(cols, dropna=False):
        meta=dict(zip(cols,keys)); times=pd.to_numeric(sub['t_rel_ms'],errors='coerce').dropna().to_numpy(float)/1000
        for minute in range(1,n_minutes+1): rows.append({**meta,'minuto':minute,'licks':int(((times>=(minute-1)*60)&(times<minute*60)).sum())})
    pm=pd.DataFrame(rows)
    if pm.empty: return pm, pd.DataFrame()
    srows=[]
    for key_vals, sub in pm.groupby(['prefix','experiment','session','group','group_order','minuto'], dropna=False):
        row=dict(zip(['prefix','experiment','session','group','group_order','minuto'],key_vals)); vals=sub['licks']; row['n_animales']=sub['rat_id'].nunique(); row['licks_media']=vals.mean(); row['licks_SEM']=sem(vals); srows.append(row)
    return pm, pd.DataFrame(srows).sort_values(['experiment','session','group_order','minuto'])
