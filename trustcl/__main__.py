"""Run reproducible experiments and create camera-ready data tables with no external downloads."""
import argparse, json, hashlib, platform, sys
from pathlib import Path
import pandas as pd
import numpy as np
from .continual import Config, run
from .calibration import simulate, report

METHODS=('finetune','replay','derpp','ewc','independent','ungated','famp','no_response','famp_naive')

def all_runs(out, datasets=('split_digits','rotated_digits'),seeds=(11,23,37),orders=(0,1),epochs=30):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    items=[]
    for ds in datasets:
      for seed in seeds:
       for order in orders:
        for method in METHODS:
         cfg=Config(dataset=ds,seed=seed,order=order,epochs=epochs)
         path=out/f'{ds}_s{seed}_o{order}_{method}.json'
         if path.exists():
          obj=json.loads(path.read_text())
         else:
          obj=run(cfg,method)
          path.write_text(json.dumps(obj,indent=2,allow_nan=False))
         items.append({k:v for k,v in obj.items() if k not in ('stage_accuracy','task_logs','task_order')})
         print(f'{ds} s{seed} o{order} {method}: accuracy={obj["accuracy"]:.3f} admissions={obj["admitted"]} bytes={obj["persistent_bytes"]} runtime={obj["training_seconds"]:.1f}s',flush=True)
    raw=pd.DataFrame(items);raw.to_csv(out/'benchmark_raw.csv',index=False)
    grouped=raw.groupby(['dataset','method'],as_index=False).agg(
      n=('accuracy','count'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),
      forgetting_mean=('forgetting','mean'),acquisition_mean=('acquisition','mean'),
      admission_mean=('admitted','mean'),sharing_mean=('shared_units','mean'),
      persistent_kib_mean=('persistent_bytes',lambda x:float(x.mean()/1024)),
      train_seconds_mean=('training_seconds','mean'))
    grouped.to_csv(out/'benchmark_summary.csv',index=False)
    print('\n',grouped.round(4).to_string(index=False),flush=True)
    return grouped

def run_calibration(out,reps=120):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    df=simulate(reps=reps,n_per_class=100,tasks=5,epsilon=.25)
    summary=report(df,out)
    print('Calibration:',summary.round(3).to_string(index=False));return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',default='results');p.add_argument('--seeds',default='11,23,37');p.add_argument('--orders',default='0,1');p.add_argument('--epochs',type=int,default=30);p.add_argument('--datasets',default='split_digits,rotated_digits,corrupted_digits');p.add_argument('--calibration-reps',type=int,default=120);p.add_argument('--mode',choices=['all','benchmarks','calibration'],default='all')
 a=p.parse_args();dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
 if a.mode in ('all','calibration'):run_calibration(dest/'calibration',a.calibration_reps)
 if a.mode in ('all','benchmarks'):all_runs(dest/'benchmarks',datasets=tuple(a.datasets.split(',')),seeds=tuple(map(int,a.seeds.split(','))),orders=tuple(map(int,a.orders.split(','))),epochs=a.epochs)
 manifest={'python':sys.version.split()[0],'platform':platform.platform(),'source_protocol':'trustcl-cpu-1.0','mode':a.mode,'seeds':a.seeds,'datasets':a.datasets,'orders':a.orders,'epochs':a.epochs,'calibration_reps':a.calibration_reps}
 (dest/'execution_manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
