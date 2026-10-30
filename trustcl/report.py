"""Build a reproducible TRUST 2027 results report from already completed runs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def build(root='results/final/benchmarks', out='results/cpu_20261030'):
    root=Path(root)
    if not root.exists():
        root=Path('results/reproduction/benchmarks')
    if not root.exists():
        raise FileNotFoundError('No raw run records. Run: python -m trustcl --mode all --out results/reproduction')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[];decisions=[]
    for path in sorted(root.glob('*_s*_o*_*.json')):
        r=json.loads(path.read_text())
        rows.append({k:v for k,v in r.items() if k not in ('stage_accuracy','task_logs','task_order')})
        for x in r['task_logs']:
            c=x['certificate'] or {}
            decisions.append(dict(dataset=r['dataset'],seed=r['seed'],order=r['order'],method=r['method'],task=x['task'],
               decision=x['decision'],selected=x['selection_accuracy'],teacher_acc=x['teacher_accuracy'],
               harm_upper=c.get('harm_upper'),observed_accept_harm=c.get('observed_harm_rate'),
               response_rmse=c.get('response_rmse'),observer_test_harm=x.get('observer_test_harm'),
               accepted=bool(x['decision']!='fallback'),test_candidate_acc=x.get('observer_test_candidate_accuracy')))
    raw=pd.DataFrame(rows);dec=pd.DataFrame(decisions)
    assert raw.groupby(['dataset','method']).size().min()==6,'Not all six paired benchmark runs completed'
    raw.to_csv(out/'benchmark_raw.csv',index=False);dec.to_csv(out/'proposal_audit.csv',index=False)
    summary=raw.groupby(['dataset','method'],as_index=False).agg(
        n=('accuracy','size'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),
        forgetting_mean=('forgetting','mean'),acquisition_mean=('acquisition','mean'),
        admitted_mean=('admitted','mean'),shared_units_mean=('shared_units','mean'),
        route_drift_total=('route_prediction_mismatches','sum'),
        memory_kib_mean=('persistent_bytes',lambda x:x.mean()/1024),
        train_seconds_mean=('training_seconds','mean'))
    summary.to_csv(out/'table_benchmarks.csv',index=False)
    # Paired seed/order bootstrap CI (repeat pairs; correlation within seeds means descriptive only).
    rng=np.random.default_rng(9921);paired=[]
    for ds in sorted(raw.dataset.unique()):
        sub=raw[raw.dataset==ds]
        for competitor in ('finetune','replay','independent','ungated','famp_naive','no_response'):
            f=sub[sub.method=='famp'].sort_values(['seed','order']).accuracy.values
            b=sub[sub.method==competitor].sort_values(['seed','order']).accuracy.values
            d=f-b;n=len(d)
            boot=np.mean(d[rng.integers(n,size=(5000,n))],axis=1)
            paired.append(dict(dataset=ds,versus=competitor,mean_accuracy_difference_pp=100*d.mean(),
                bootstrap_lo_pp=100*np.quantile(boot,.025),bootstrap_hi_pp=100*np.quantile(boot,.975),pairs=n))
    paired=pd.DataFrame(paired);paired.to_csv(out/'paired_effects.csv',index=False)
    # Uncertainty in small test set: descriptive observed task-wise harm (not ground-truth risk).
    audit=dec[dec.harm_upper.notna()].groupby(['dataset','method'],as_index=False).agg(
        selected_proposals=('harm_upper','size'),acceptance_rate=('accepted','mean'),
        mean_harm_upper=('harm_upper','mean'),mean_test_harm=('observer_test_harm','mean'),
        observed_test_exceedances=('observer_test_harm',lambda x:int((x>.15).sum())))
    audit.to_csv(out/'table_admissions.csv',index=False)
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,4.5))
        for meth in ['finetune','replay','independent','famp_naive','famp','ungated']:
            z=summary[summary.method==meth]
            ax.plot(z.dataset,z.accuracy_mean*100,'o-',label=meth)
        ax.set(ylabel='Final task-incremental accuracy (%)',xlabel='Benchmark',ylim=(60,103))
        ax.legend(ncol=3,fontsize=8);fig.tight_layout();fig.savefig(out/'figure_accuracy.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4))
        for meth in ['independent','famp_naive','famp','no_response','ungated']:
            z=summary[summary.method==meth]
            ax.plot(z.dataset,z.admitted_mean,'o-',label=meth)
        ax.set(ylabel='Published routes / 5 tasks',ylim=(-.1,5.5))
        ax.legend(ncol=3,fontsize=8);fig.tight_layout();fig.savefig(out/'figure_admissions.png',dpi=180);plt.close(fig)
    except ImportError:pass
    md=['# Executed CPU benchmark results','','All values below are automatically derived from completed `*.json` task streams. Each dataset/method cell has 3 seeds × 2 orders (n=6). No GPU experiments are claimed.','','## Final accuracy, forgetting, published units and resource costs','',summary.round(4).to_markdown(index=False),'','## Paired FAMP differences (percentage points)','',paired.round(3).to_markdown(index=False),'','## Admission inspection','',audit.round(4).to_markdown(index=False),'','**Interpretation:** In this small CPU protocol, strict FAMP is not demonstrably more accurate or more memory efficient than independent task units. Reuse may be absent when the response gate is stringent. Apparent test harm uses small held-out sets and is not the population truth. Bootstrap intervals are descriptive, with just three independent seeds, and should not be treated as formal hypothesis tests.']
    (out/'results.md').write_text('\n'.join(md))
    return summary

if __name__=='__main__':
    print(build().to_string(index=False))
