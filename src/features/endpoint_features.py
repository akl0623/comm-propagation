#!/usr/bin/env python3
"""Task C Feature Builder v1 — Phase 03. Incremental fold-by-fold processing."""
import os, sys, json, time, hashlib, ast as _ast_module, random as _py_random
import numpy as np, pandas as pd
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout,'reconfigure') else None

PIPELINE_ROOT = os.environ.get('PIPELINE_ROOT', '')
if not PIPELINE_ROOT:
    raise EnvironmentError(
        "PIPELINE_ROOT environment variable must be set to the data workspace root. "
        "See README.md (Data Availability section) for data download instructions."
    )

SC_P = os.path.join(PIPELINE_ROOT, "taskC0R2C_symmetric_SC_and_P1_freeze", "averageConnectivity_tractStrength_0.25density_symmetric_C0R2C.csv")
LAT_P = os.path.join(PIPELINE_ROOT, "raw_inputs", "Reordered_matrix_onset_delay__median.csv")
ROI_P = os.path.join(PIPELINE_ROOT, "taskC0R2C_symmetric_SC_and_P1_freeze", "roi_order_C0R2C.txt")
UNI_P = os.path.join(PIPELINE_ROOT, "taskC0R2D_universe_freeze", "taskC_primary_2636_universe_C0R2D.csv")
FOLD_P = os.path.join(PIPELINE_ROOT, "taskC0R2D_universe_freeze", "taskC_primary_fold_manifest_C0R2D_v2.csv")
K_P = os.path.join(PIPELINE_ROOT, "taskC0R2D_universe_freeze", "C1A_fold_K_selection.csv")
NAV_P = os.path.join(PIPELINE_ROOT, "raw_inputs", "navigation_efficiency_matrix_0.25density.csv")
ROUT_P = os.path.join(PIPELINE_ROOT, "raw_inputs", "rout_efficiency_matrix_0.25density.csv")
SRCH_P = os.path.join(PIPELINE_ROOT, "raw_inputs", "search_information_matrix_0.25density.csv")
COMM_P = os.path.join(PIPELINE_ROOT, "raw_inputs", "communicability_matrix_0.25density.csv")

METRICS=["nav","rout","search","comm"]; SUMMARIES=["mean","sd","min","max","first","last","slope"]
SEEDS=[20260727,20260728,20260729,20260730,20260731]; N_FEAT_56=56

_orig_open=open; _OPENED=[]
def _trace_open(f,*a,**kw):
    p=f if isinstance(f,str) else getattr(f,'name',str(f)); _OPENED.append((p,kw.get('mode','r')))
    return _orig_open(f,*a,**kw)

def sha256_f(p):
    h=hashlib.sha256()
    with _orig_open(p,'rb') as f:
        for c in iter(lambda:f.read(65536),b''):h.update(c)
    return h.hexdigest()

def cell_seed(gs,sid): return int.from_bytes(hashlib.sha256(f"{gs}|{sid}".encode()).digest()[:8],'big')

class P1Engine:
    def __init__(self):
        self._mask=None; self._lat=None; self._mats={}; self._dags={}; self._ctcache={}
        self._roi_order=[]; self._r2i={}; self._sc_tract=None
    def load(self):
        sc=pd.read_csv(SC_P,header=None).values.astype(np.float64); self._sc_tract=sc.copy()
        self._mask=(sc>0).astype(np.int8)
        self._lat=pd.read_csv(LAT_P,index_col=0).values.astype(np.float64)
        for m,p in [('nav',NAV_P),('rout',ROUT_P),('search',SRCH_P),('comm',COMM_P)]:
            self._mats[m]=pd.read_csv(p,header=None).values.astype(np.float64)
        with _orig_open(ROI_P) as f: self._roi_order=[l.strip() for l in f if l.strip()]
        self._r2i={r:i for i,r in enumerate(self._roi_order)}
    def build_dag(self,si):
        hemi=list(range(0,180)) if si<180 else list(range(180,360))
        nodes=[(si,0.0)]
        for ri in hemi:
            if ri==si: continue
            tv=self._lat[si,ri]
            if np.isnan(tv) or tv<=0: continue
            nodes.append((ri,float(tv)))
        if len(nodes)<=1: return None
        nodes.sort(key=lambda x:x[1])
        no=[n[0] for n in nodes]; nl={n[0]:n[1] for n in nodes}
        oe=defaultdict(list)
        for ui in no:
            for vi in no:
                if nl[vi]<=nl[ui]: continue
                if self._mask[ui,vi]: oe[ui].append(vi)
            oe[ui]=sorted(set(oe[ui]))
        cts={}
        for ep in no:
            if ep==si: continue
            ntl={n:{} for n in no}; ntl[ep][0]=1
            eidx=no.index(ep)
            for pos in range(eidx,-1,-1):
                u=no[pos]
                for v in oe.get(u,[]):
                    for Lm,cnt in ntl[v].items():
                        L=Lm+1; ntl[u][L]=ntl[u].get(L,0)+cnt
            for L,cnt in ntl[si].items():
                if L>=1 and cnt>0: cts[(si,ep,L)]=int(cnt)
        return {'node_order':no,'out_edges':dict(oe),'counts':cts}
    def init_dags(self):
        for si in range(360):
            d=self.build_dag(si)
            if d is not None: self._dags[si]=d
    def get_ct(self,si,ei,L):
        k=(si,ei,L)
        if k in self._ctcache: return self._ctcache[k]
        d=self._dags[si]; no=d['node_order']; oe=d['out_edges']
        ct={u:{} for u in no}; ct[ei][0]=1
        for rem in range(1,L+1):
            for u in no:
                tc=sum(ct[v].get(rem-1,0) for v in oe.get(u,[]))
                if tc>0: ct[u][rem]=tc
        self._ctcache[k]=ct; return ct
    def unrank(self,rk,si,ei,L):
        d=self._dags[si]; ct=self.get_ct(si,ei,L); oe=d['out_edges']
        path=[si]; cur=si; rem=L; rr=rk
        while rem>0:
            cands=oe.get(cur,[])
            cvs=[ct[v].get(rem-1,0) for v in cands]
            cum=0; ch=None
            for v,c in zip(cands,cvs):
                if rr<cum+c: ch=v; break
                cum+=c
            if ch is None: return None
            path.append(ch); cur=ch; rem-=1; rr-=cum
        return path
    def sample(self,si,ei,L,n,cs):
        d=self._dags.get(si)
        if d is None: return []
        T=int(d['counts'].get((si,ei,L),0))
        if T==0: return []
        n=min(n,T); rng=_py_random.Random(cs)
        if T<=2**62: rks=sorted(rng.sample(range(T),n))
        else: rks=sorted(self._reservoir(rng,T,n))
        return [p for p in (self.unrank(int(r),si,ei,L) for r in rks) if p is not None]
    def _reservoir(self,rng,T,n):
        s=set()
        for i in range(T-n,T):
            t=rng.randint(0,i+1); s.add(t if t not in s else i)
        return sorted(s)
    @property
    def mats(self): return self._mats
    @property
    def roi_order(self): return self._roi_order
    @property
    def r2i(self): return self._r2i
    @property
    def dags(self): return self._dags
    @property
    def mask(self): return self._mask
    @property
    def lat(self): return self._lat
    @property
    def sc_tract(self): return self._sc_tract

def pp_feats(path,eng):
    if len(path)<2: return np.full(28,np.nan)
    pf=[]
    for m in METRICS:
        ev=np.array([float(eng.mats[m][path[s-1],path[s]]) for s in range(1,len(path))])
        slope=np.polyfit(np.linspace(0,1,len(ev)),ev,1)[0] if len(ev)>=2 else 0.0
        pf.extend([np.mean(ev),np.std(ev,ddof=0),np.min(ev),np.max(ev),ev[0],ev[-1],slope])
    return np.array(pf,dtype=np.float64)

def cell_56(ppa):
    np_=ppa.shape[0]
    if np_==0: return np.full(N_FEAT_56,np.nan)
    cf=np.zeros(N_FEAT_56)
    for j in range(28):
        col=ppa[:,j]; fin=np.isfinite(col)
        if fin.sum()>0:
            cf[2*j]=np.median(col[fin])
            cf[2*j+1]=np.subtract(*np.percentile(col[fin],[75,25])) if fin.sum()>1 else 0.0
        else: cf[2*j]=np.nan; cf[2*j+1]=np.nan
    return cf

def build_features_from_paths(paths,eng,L_max):
    n=len(paths)
    if n==0: return (np.full(N_FEAT_56,np.nan), np.full(L_max*8,np.nan), np.zeros(L_max*4,dtype=np.int8))
    ppa=np.array([pp_feats(p,eng) for p in paths])
    blind56=cell_56(ppa)
    L=len(paths[0])-1
    edge_vals={m:{s:[] for s in range(1,L_max+1)} for m in METRICS}
    for p in paths:
        for s in range(1,len(p)):
            if s>L_max: break
            for m in METRICS: edge_vals[m][s].append(float(eng.mats[m][p[s-1],p[s]]))
    stage_feats=[]; stage_mask=[]
    for s in range(1,L_max+1):
        for m in METRICS:
            vals=np.array(edge_vals[m][s]); fin=vals[np.isfinite(vals)]
            if len(fin)>0:
                stage_feats.extend([np.median(fin), np.subtract(*np.percentile(fin,[75,25])) if len(fin)>1 else 0.0])
                stage_mask.append(1)
            else: stage_feats.extend([np.nan,np.nan]); stage_mask.append(0)
    return blind56, np.array(stage_feats), np.array(stage_mask,dtype=np.int8)

def process_fold(eng, uni, fold_samples, rep_idx, rep_seed, fold, K, L_max):
    """Process one fold and return rows."""
    sample_info={r['sample_id']:(r['seed_roi'],r['endpoint_roi'],int(r['path_length_edges'])) for _,r in uni.iterrows()}
    rows=[]
    for sid in fold_samples:
        seed_roi,ep_roi,L=sample_info.get(sid,(None,None,None))
        if seed_roi is None: continue
        si=eng.r2i.get(seed_roi); ei=eng.r2i.get(ep_roi)
        if si is None or ei is None: continue
        cs=cell_seed(str(rep_seed),sid)
        paths=eng.sample(si,ei,L,K,cs)
        n_p=len(paths); n_ok=0; n_val_fail=0
        for p in paths:
            ok=True
            if p[0]!=si or p[-1]!=ei or len(p)-1!=L: ok=False
            elif len(set(p))!=len(p): ok=False
            else:
                for s in range(1,len(p)):
                    if eng.mask[p[s-1],p[s]]!=1: ok=False; break
            if ok: n_ok+=1
            else: n_val_fail+=1
        blind56,stage_feats,stage_mask=build_features_from_paths(paths,eng,L_max)
        ep=[float(eng.mats[m][si,ei]) for m in METRICS]
        sc_val=float(eng.sc_tract[si,ei])
        row={'sample_id':sid,'seed_roi':seed_roi,'endpoint_roi':ep_roi,
             'repeat':rep_idx,'repeat_seed':rep_seed,'test_fold':fold,
             'L':L,'K':K,'n_paths':n_p,'n_validated':n_ok,'n_val_fail':n_val_fail,
             'sc_strength':sc_val,'graph_distance':L}
        for j,m in enumerate(METRICS): row[f'ep_{m}']=ep[j]
        for j in range(N_FEAT_56): row[f'blind_{j:02d}']=blind56[j]
        for j in range(len(stage_feats)): row[f'stage_{j:03d}']=stage_feats[j]
        for j in range(len(stage_mask)): row[f'stagemask_{j:03d}']=int(stage_mask[j])
        rows.append(row)
    return rows

def main():
    global _OPENED; _OPENED=[]
    T0=time.time()
    print("="*60)
    print("Task C Feature Builder v1 — Phase 03 (Incremental)")
    print("="*60)
    print("[1] Loading P1 engine..."); t0=time.time()
    eng=P1Engine(); eng.load(); eng.init_dags()
    print(f"  {len(eng.dags)} DAGs ({time.time()-t0:.1f}s)")
    print("[2] Loading data...")
    uni=pd.read_csv(UNI_P); folds=pd.read_csv(FOLD_P); kt=pd.read_csv(K_P)
    k_map={(int(r['repeat']),int(r['fold'])):int(r['selected_K']) for _,r in kt.iterrows()}
    L_max=int(uni['path_length_edges'].max())
    print(f"  Universe: {len(uni)}, L_max={L_max}")

    all_rows=[]; processed=0
    for rep_idx in range(5):
        rep_seed=SEEDS[rep_idx]; rep_folds=folds[folds['repeat']==rep_idx]
        for fold in sorted(rep_folds['test_fold'].unique()):
            K=k_map.get((rep_idx,fold))
            if K is None: continue
            fold_sids=rep_folds[rep_folds['test_fold']==fold]['sample_id'].tolist()
            print(f"  [rep={rep_idx} fold={fold}] K={K} samples={len(fold_sids)}...", end="", flush=True)
            t1=time.time()
            rows=process_fold(eng,uni,fold_sids,rep_idx,rep_seed,fold,K,L_max)
            all_rows.extend(rows); processed+=len(fold_sids)
            t2=time.time()
            n_paths=sum(r['n_paths'] for r in rows)
            print(f" {len(rows)} rows, {n_paths} paths ({t2-t1:.1f}s) [{processed}/{len(uni)*5}]", flush=True)

    print(f"\n[3] Assembling DataFrame ({len(all_rows)} rows)...", flush=True)
    feat_df=pd.DataFrame(all_rows)
    print(f"  Shape: {feat_df.shape}", flush=True)

    print("[4] Triple audit...", flush=True)
    prob_opens=[f for f,_ in _OPENED if 'probability' in f.lower()]
    print(f"  Probability files opened: {len(prob_opens)} {'✅' if not prob_opens else '❌'}", flush=True)
    tcols=[c for c in feat_df.columns if 'y_endpoint' in c.lower() or 'probability' in c.lower()]
    print(f"  Target cols: {len(tcols)} {'✅' if not tcols else '❌'}", flush=True)

    print("[5] Writing...", flush=True)
    fp='features/TASK_C_FEATURES_V1.parquet'
    feat_df.to_parquet(fp,index=False)
    fh=sha256_f(fp); fs=os.path.getsize(fp)
    schema={'columns':list(feat_df.columns),'n_rows':len(feat_df),'n_cols':len(feat_df.columns)}
    with _orig_open(fp+'.schema.json','w') as f: json.dump(schema,f,indent=2)
    with _orig_open(fp+'.sha256','w') as f: f.write(fh+'\n')
    print(f"  {fp}: {fs}B sha256={fh[:16]}...", flush=True)
    print(f"\nDONE in {time.time()-T0:.0f}s", flush=True)
    return feat_df, eng, L_max

def ast_self_audit():
    with _orig_open(__file__) as f: tree=_ast_module.parse(f.read())
    for node in _ast_module.walk(tree):
        if isinstance(node,_ast_module.Import):
            for a in node.names:
                if 'taskC_target' in a.name: return False,[f"import {a.name}"]
        if isinstance(node,_ast_module.ImportFrom):
            if node.module and 'taskC_target' in node.module: return False,[f"from {node.module}"]
    return True,[]

if __name__=='__main__':
    clean,v=ast_self_audit()
    if not clean: print(f"❌ AST: {v}"); sys.exit(1)
    print("✅ AST: clean", flush=True)
    feat_df,eng,L_max=main()
