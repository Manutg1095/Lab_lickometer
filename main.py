import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import pandas as pd
from config import DEFAULT_PARAMS
from metadata import preview_workbook_groups, summarize_detected_groups
from pipeline import run_pipeline

class LickPipelineGUI(tk.Tk):
    def __init__(self):
        super().__init__(); self.title('Lick Microstructure Pipeline'); self.geometry('1250x760')
        self.xlsx_path=None; self.output_dir=None; self.preview_df=pd.DataFrame(); self.summary_df=pd.DataFrame(); self.param_vars={}; self._build_ui()
    def _build_ui(self):
        top=ttk.Frame(self,padding=10); top.pack(fill='x')
        ttk.Button(top,text='Seleccionar Excel',command=self.on_load_excel).pack(side='left',padx=5); self.file_label=ttk.Label(top,text='Archivo: no seleccionado'); self.file_label.pack(side='left',padx=10)
        ttk.Button(top,text='Seleccionar carpeta de salida',command=self.on_select_output).pack(side='left',padx=5); self.output_label=ttk.Label(top,text='Salida: no seleccionada'); self.output_label.pack(side='left',padx=10)
        main=ttk.PanedWindow(self,orient='horizontal'); main.pack(fill='both',expand=True,padx=10,pady=10); left=ttk.Frame(main); right=ttk.Frame(main); main.add(left,weight=3); main.add(right,weight=1)
        ttk.Label(left,text='Hojas detectadas').pack(anchor='w'); self.preview_tree=ttk.Treeview(left,show='headings',height=16); self.preview_tree.pack(fill='both',expand=True,pady=(0,10))
        ttk.Label(left,text='Resumen de grupos encontrados').pack(anchor='w'); self.summary_tree=ttk.Treeview(left,show='headings',height=8); self.summary_tree.pack(fill='both',expand=True)
        params_frame=ttk.LabelFrame(right,text='Parámetros de análisis',padding=10); params_frame.pack(fill='x',pady=5)
        labels=[('session_max_s','Duración máxima de sesión (s)'),('min_ili_ms','ILI mínimo válido (ms)'),('burst_threshold_ms','Umbral de ráfaga (ms)'),('min_licks_per_burst','Mínimo licks/ráfaga'),('hist_bin_ms','Bins histograma (ms)'),('hist_range_min_ms','Histograma mínimo (ms)'),('hist_range_max_ms','Histograma máximo (ms)'),('low_range_min_ms','Rango bajo mínimo (ms)'),('low_range_max_ms','Rango bajo máximo (ms)'),('high_range_min_ms','Rango alto mínimo (ms)'),('high_range_max_ms','Rango alto máximo (ms)')]
        for i,(key,label) in enumerate(labels):
            ttk.Label(params_frame,text=label).grid(row=i,column=0,sticky='w',pady=2); var=tk.StringVar(value=str(DEFAULT_PARAMS[key])); self.param_vars[key]=var; ttk.Entry(params_frame,textvariable=var,width=12).grid(row=i,column=1,sticky='e',pady=2)
        self.run_button=ttk.Button(right,text='Ejecutar análisis',command=self.on_run,state='disabled'); self.run_button.pack(fill='x',pady=5)
        ttk.Button(right,text='Restaurar defaults',command=self.reset_defaults).pack(fill='x',pady=5)
        log_frame=ttk.LabelFrame(right,text='Progreso',padding=10); log_frame.pack(fill='both',expand=True,pady=5); self.log_text=tk.Text(log_frame,height=15,wrap='word'); self.log_text.pack(fill='both',expand=True)
    def reset_defaults(self):
        for k,v in DEFAULT_PARAMS.items(): self.param_vars[k].set(str(v))
    def log(self,text): self.log_text.insert('end',str(text)+'\n'); self.log_text.see('end'); self.update_idletasks()
    def get_params(self):
        p={}
        for k,var in self.param_vars.items(): p[k]=int(float(var.get())) if k=='min_licks_per_burst' else float(var.get())
        self.validate_params(p); return p
    def validate_params(self,p):
        errors=[]
        if p['session_max_s']<=0: errors.append('La duración máxima debe ser mayor que 0.')
        if p['min_ili_ms']<0: errors.append('El ILI mínimo no puede ser negativo.')
        if p['burst_threshold_ms']<=p['min_ili_ms']: errors.append('El umbral de ráfaga debe ser mayor que el ILI mínimo.')
        if p['min_licks_per_burst']<2: errors.append('El mínimo de licks por ráfaga debe ser al menos 2.')
        if p['hist_bin_ms']<=0: errors.append('El tamaño de bin debe ser mayor que 0.')
        if p['hist_range_max_ms']<=p['hist_range_min_ms']: errors.append('El rango de histograma es inválido.')
        if p['low_range_max_ms']<=p['low_range_min_ms']: errors.append('El rango intra-burst bajo es inválido.')
        if p['high_range_max_ms']<=p['high_range_min_ms']: errors.append('El rango intra-burst alto es inválido.')
        if p['low_range_max_ms']!=p['high_range_min_ms']: errors.append('El límite superior del rango bajo debe coincidir con el límite inferior del rango alto.')
        if p['high_range_max_ms']!=p['burst_threshold_ms']: errors.append('El límite superior del rango alto debería coincidir con el umbral de ráfaga.')
        if errors: raise ValueError('\n'.join(errors))
    def on_load_excel(self):
        path=filedialog.askopenfilename(title='Selecciona el libro de Excel', filetypes=[('Excel files','*.xlsx')])
        if not path: return
        self.xlsx_path=path; self.file_label.config(text=f'Archivo: {Path(path).name}'); self.log('Leyendo libro...')
        try:
            self.preview_df=preview_workbook_groups(path); self.summary_df=summarize_detected_groups(self.preview_df); self.show_dataframe(self.preview_tree,self.preview_df); self.show_dataframe(self.summary_tree,self.summary_df)
            if self.preview_df.empty: self.log('No se detectaron hojas analizables.'); self.run_button.config(state='disabled'); return
            if (self.preview_df['status']=='ERROR').any(): self.log('Hay errores de detección. Corrige nombres de hojas antes de analizar.'); self.run_button.config(state='disabled')
            else:
                self.log('Libro cargado correctamente. Grupos detectados sin errores.')
                if self.output_dir: self.run_button.config(state='normal')
        except Exception as e: messagebox.showerror('Error al cargar Excel',str(e)); self.run_button.config(state='disabled')
    def show_dataframe(self,tree,df):
        tree.delete(*tree.get_children()); tree['columns']=list(df.columns)
        for col in df.columns: tree.heading(col,text=col); tree.column(col,width=max(90,min(180,len(col)*12)),anchor='w')
        for _,row in df.iterrows(): tree.insert('', 'end', values=['' if pd.isna(row[c]) else row[c] for c in df.columns])
    def on_select_output(self):
        path=filedialog.askdirectory(title='Selecciona carpeta de salida')
        if not path: return
        self.output_dir=path; self.output_label.config(text=f'Salida: {path}')
        if self.xlsx_path and not self.preview_df.empty and not (self.preview_df['status']=='ERROR').any(): self.run_button.config(state='normal')
    def on_run(self):
        if not self.xlsx_path: messagebox.showwarning('Falta Excel','Selecciona un archivo Excel.'); return
        if not self.output_dir: messagebox.showwarning('Falta salida','Selecciona carpeta de salida.'); return
        try:
            params=self.get_params(); self.run_button.config(state='disabled'); self.log('Iniciando análisis...'); result=run_pipeline(self.xlsx_path,self.output_dir,params,log_fn=self.log); self.log(f"Excel generado: {result['excel']}"); self.log(f"Figuras: {result['fig_dir']}"); self.log(f"Log: {result['log']}"); messagebox.showinfo('Terminado','Análisis terminado correctamente.')
        except Exception as e: messagebox.showerror('Error en análisis',str(e)); self.log(f'ERROR: {e}')
        finally:
            if self.xlsx_path and self.output_dir and not self.preview_df.empty: self.run_button.config(state='normal')
if __name__=='__main__': LickPipelineGUI().mainloop()
