from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_bar_metric(per_animal, metric, ylabel, title, output_path, session_filter=None):
    df=per_animal.copy()
    if session_filter: df=df[df['session']==session_filter]
    if df.empty or metric not in df: return
    rows=[]
    for (session,group,order), sub in df.groupby(['session','group','group_order'], dropna=False):
        vals=pd.to_numeric(sub[metric], errors='coerce').dropna()
        if len(vals): rows.append({'session':session,'group':group,'group_order':order,'mean':vals.mean(),'sem':vals.std(ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else 0})
    s=pd.DataFrame(rows)
    if s.empty: return
    s=s.sort_values(['session','group_order']); x=np.arange(len(s)); labels=[f"{r.session}\n{r.group}" for _,r in s.iterrows()]
    plt.figure(figsize=(max(7,len(s)*0.8),5)); plt.bar(x,s['mean'],yerr=s['sem'],capsize=4); plt.xticks(x,labels,rotation=45,ha='right'); plt.ylabel(ylabel); plt.title(title); plt.tight_layout(); Path(output_path).parent.mkdir(parents=True, exist_ok=True); plt.savefig(output_path,dpi=300); plt.close()

def plot_histogram(hist_summary, output_path):
    if hist_summary.empty: return
    plt.figure(figsize=(8,5))
    for (_,group,_), sub in hist_summary.groupby(['experiment','group','group_order'], dropna=False):
        sub=sub.sort_values('bin_centro_ms'); x=sub['bin_centro_ms']; y=sub['freq_rel_media']; e=sub['freq_rel_SEM']; plt.plot(x,y,label=group); plt.fill_between(x,y-e,y+e,alpha=.2)
    plt.xlabel('ILI (ms)'); plt.ylabel('Frecuencia relativa'); plt.title('Histograma ILI S2'); plt.legend(); plt.tight_layout(); Path(output_path).parent.mkdir(parents=True, exist_ok=True); plt.savefig(output_path,dpi=300); plt.close()

def export_figures(per_animal, hist_summary, fig_dir):
    fig_dir=Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    plot_histogram(hist_summary, fig_dir/'histograma_ILI_S2_60_500ms.png')
    plot_bar_metric(per_animal,'latencia_primer_lick_ms','Latencia primer lick (ms)','Latencia al primer lick S2',fig_dir/'latencia_primer_lick_S2.png','S2')
    plot_bar_metric(per_animal,'latencia_primer_burst_ms','Latencia primer burst (ms)','Latencia al primer burst S2',fig_dir/'latencia_primer_burst_S2.png','S2')
    plot_bar_metric(per_animal,'media_ILI_60_220_ms','Media ILI 60–220 ms','Media ILI 60–220 ms',fig_dir/'media_ILI_60_220_S1_S2.png')
    plot_bar_metric(per_animal,'media_ILI_220_330_ms','Media ILI 220–330 ms','Media ILI 220–330 ms',fig_dir/'media_ILI_220_330_S1_S2.png')
    plot_bar_metric(per_animal,'tamano_rafaga_media_licks','Tamaño de ráfaga (licks)','Tamaño promedio de ráfaga',fig_dir/'tamano_rafaga_S1_S2.png')
    plot_bar_metric(per_animal,'IBI_corregido_media_ms','IBI corregido (ms)','Intervalo interráfagas corregido',fig_dir/'IBI_corregido_S1_S2.png')
