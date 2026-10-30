"""CPU-oriented task-incremental neural continual learning with independently audited publication.

Published routes are frozen task-specific networks. Reuse attempts train new heads on old,
immutable feature encoders. Acceptance set is never used for fitting or selection.
The byte ledger charges fallback, every published unit and head, replay, and manifest.
This is an experimental mechanism, not a guarantee of low absolute error.
"""
from __future__ import annotations
import hashlib, json, math, time
from functools import lru_cache
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from scipy.ndimage import rotate
import torch
from torch import nn
from torch.nn import functional as F
from .assurance import Policy, certificate

torch.set_num_threads(2)

@dataclass
class Config:
    dataset: str = 'split_digits'
    seed: int = 11
    order: int = 0
    epochs: int = 30
    batch_size: int = 48
    replay_per_task: int = 24
    lr: float = .003
    risk_tolerance: float = .15
    min_teacher_acc: float = .70
    n_tasks: int = 5
    unit_width: int = 32
    feature_dim: int = 16

class Encoder(nn.Module):
    def __init__(self, width=32, feat=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, width), nn.ReLU(),nn.Linear(width,feat),nn.ReLU())
    def forward(self,x): return self.net(x)

class TaskUnit(nn.Module):
    def __init__(self, width=32, feat=16):
        super().__init__();self.encoder=Encoder(width,feat);self.head=nn.Linear(feat,2)
    def forward(self,x):return self.head(self.encoder(x))

class Fallback(nn.Module):
    def __init__(self,tasks,width=48):
        super().__init__();self.tasks=tasks
        self.net=nn.Sequential(nn.Linear(64+tasks,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,2))
    def forward(self,x,t):
        if isinstance(t,int): t=torch.full((len(x),),t,dtype=torch.long)
        return self.net(torch.cat([x,F.one_hot(t,self.tasks).float()],dim=1))

def _tensor(x):return torch.from_numpy(np.asarray(x,dtype=np.float32))

def _predict(model,x,task=None):
    model.eval();
    with torch.no_grad():
        p=model(_tensor(x),task) if isinstance(model,Fallback) else model(_tensor(x))
        return p.argmax(1).numpy()

def _train(net,x,y, cfg, task=None, buffer=None, seed=0, ewc=None):
    """No use of selection, acceptance, or test samples."""
    rng=np.random.default_rng(seed);opt=torch.optim.Adam(net.parameters(),lr=cfg.lr)
    x=_tensor(x);y=torch.tensor(y,dtype=torch.long)
    if buffer is not None and len(buffer[1]):
        bx=_tensor(buffer[0]);by=torch.tensor(buffer[1],dtype=torch.long);bt=torch.tensor(buffer[2],dtype=torch.long)
    else: bx=by=bt=None
    bl=torch.tensor(buffer[3],dtype=torch.float32) if buffer is not None and len(buffer)>3 else None
    net.train()
    for _ in range(cfg.epochs):
        perm=rng.permutation(len(y))
        for start in range(0,len(y),cfg.batch_size):
            idx=perm[start:start+cfg.batch_size]
            if isinstance(net,Fallback):
                logits=net(x[idx],task)
                loss=F.cross_entropy(logits,y[idx])
                if bx is not None:
                    old=torch.from_numpy(rng.integers(len(by),size=min(len(idx),cfg.batch_size)))
                    oldpred=net(bx[old],bt[old])
                    loss=loss+F.cross_entropy(oldpred,by[old])
                    if bl is not None:loss=loss+.3*F.mse_loss(oldpred,bl[old])
            else:loss=F.cross_entropy(net(x[idx]),y[idx])
            if ewc is not None:
                fi,anchor,coef=ewc
                loss=loss+coef*sum((fi[k]*(p-anchor[k]).square()).sum() for k,p in net.named_parameters())
            opt.zero_grad();loss.backward();opt.step()
    net.eval()

def _fisher(model,x,y,task,max_samples=40):
    model.eval();ids=np.linspace(0,len(y)-1,min(max_samples,len(y)),dtype=int)
    fi={k:torch.zeros_like(p) for k,p in model.named_parameters()}
    for i in ids:
        model.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(_tensor(x[i:i+1]),task),torch.tensor(y[i:i+1],dtype=torch.long))
        loss.backward()
        for k,p in model.named_parameters():
            if p.grad is not None:fi[k]+=p.grad.detach().square()/len(ids)
    model.zero_grad(set_to_none=True)
    return fi

def _fit_head(encoder,x,y,dim=16):
    encoder.eval()
    with torch.no_grad(): z=encoder(_tensor(x)).numpy()
    # Linear least-squares head, fitted on TRAIN only; no cross-task gradient into encoder.
    z=np.column_stack([z,np.ones(len(z))]);onehot=np.eye(2)[np.asarray(y)]
    reg=.025*np.eye(z.shape[1]);reg[-1,-1]=0
    w=np.linalg.solve(z.T@z+reg,z.T@onehot)
    head=nn.Linear(dim,2)
    with torch.no_grad():head.weight.copy_(torch.tensor(w[:-1].T,dtype=torch.float32));head.bias.copy_(torch.tensor(w[-1],dtype=torch.float32))
    return head

class Route(nn.Module):
    def __init__(self,encoder,head):
        super().__init__();self.encoder=encoder;self.head=head
    def forward(self,x):return self.head(self.encoder(x))

def _image_splits(seed):
    """Global index split prevents train/accept leakage across rotated domains."""
    raw=load_digits();x=(raw.data/16).astype(np.float32);labels=raw.target
    rng=np.random.default_rng(seed)
    roles={k:[] for k in ['train','select','accept','test']}
    for d in range(10):
        idx=rng.permutation(np.flatnonzero(labels==d))
        assert len(idx)>=174
        for name,sl in [('train',slice(0,55)),('select',slice(55,75)),('accept',slice(75,150)),('test',slice(150,None))]:roles[name].extend(idx[sl].tolist())
    return x,labels,{k:np.array(v,dtype=np.int64) for k,v in roles.items()}

def _dataset(cfg):
    if cfg.dataset not in ('split_digits','rotated_digits','corrupted_digits'):
        raise ValueError('only reproducible bundled sklearn Digits benchmarks are supported')
    x,y,roles=_image_splits(cfg.seed)
    base_tasks=[]
    for task in range(5):
        splits={}
        for role,indices in roles.items():
            if cfg.dataset=='split_digits':
                take=indices[np.isin(y[indices],[2*task,2*task+1])]
                xx=x[take];yy=(y[take]%2).astype(np.int64)
            else:
                # parity classification across domain rotations; 0,20,40,60,80 deg
                xx=x[indices]; yy=(y[indices]%2).astype(np.int64)
                angle=task*20
                if cfg.dataset=='rotated_digits' and angle:
                    xx=np.stack([rotate(v.reshape(8,8),angle,reshape=False,order=1,mode='constant',cval=0).ravel() for v in xx])
                    xx=np.clip(xx,0,1).astype(np.float32)
                elif cfg.dataset=='corrupted_digits' and task:
                    role_id={'train':11,'select':23,'accept':37,'test':49}[role]
                    crng=np.random.default_rng(cfg.seed*1709+task*101+role_id)
                    xx=np.clip(xx+crng.normal(0,task*.025,size=xx.shape),0,1).astype(np.float32)
            splits[role]=(xx.astype(np.float32),yy)
        base_tasks.append(splits)
    perm=np.random.default_rng(cfg.seed+1000+cfg.order*17).permutation(5) if cfg.order else np.arange(5)
    return [base_tasks[i] for i in perm],perm.tolist()

@lru_cache(maxsize=64)
def _dataset_cached(ds, seed, order):
    return _dataset(Config(dataset=ds, seed=seed, order=order))

def dataset(cfg):
    # All returned data are read-only in training; caching is a compute optimization.
    return _dataset_cached(cfg.dataset, cfg.seed, cfg.order)

def _counts(net):return sum(p.numel()*p.element_size() for p in net.parameters())

def _hash_model(m):
    h=hashlib.sha256()
    for key,v in sorted(m.state_dict().items()):h.update(key.encode());h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()

def run(cfg,method='famp'):
    """Run one complete five-task sequence. Methods: finetune, replay, independent, ungated, famp, no_response.

    FAMP uses valid independent acceptance data once per task, for one val-selected proposal.
    Independent publishes only new units (never reuses), subject to same acceptance rule.
    Ungated ignores the certificate (negative control). No_response omits only response check.
    """
    assert method in ('finetune','replay','derpp','ewc','independent','ungated','famp','no_response','famp_naive')
    torch.manual_seed(cfg.seed);np.random.seed(cfg.seed)
    tasks,perm=dataset(cfg);base=Fallback(cfg.n_tasks)
    memx=[];memy=[];memt=[];memlog=[];units=[];routes={};logs=[];stage_acc=[];route_predictions={};fisher=None;anchor=None
    start=time.perf_counter()
    for t,splits in enumerate(tasks):
        trainx,trainy=splits['train'];selectx,selecty=splits['select'];acceptx,accepty=splits['accept']
        oldbuf=None
        if method not in ('finetune','ewc') and memy:
            oldbuf=(np.concatenate(memx),np.concatenate(memy),np.concatenate(memt))
            if method=='derpp':oldbuf=oldbuf+(np.concatenate(memlog),)
        _train(base,trainx,trainy,cfg,task=t,buffer=oldbuf,seed=cfg.seed+101*t,
               ewc=(fisher,anchor,10.) if method=='ewc' and fisher is not None else None)
        if method=='ewc':
            newfi=_fisher(base,trainx,trainy,t)
            fisher=newfi if fisher is None else {k:.9*fisher[k]+newfi[k] for k in newfi}
            anchor={k:p.detach().clone() for k,p in base.named_parameters()}
        teacher_a=_predict(base,acceptx,t)
        options=[]
        if method not in ('finetune','replay','derpp','ewc'):
            if method in ('famp','ungated','no_response','famp_naive'):
                for uid,u in enumerate(units):
                    head=_fit_head(u,trainx,trainy,cfg.feature_dim)
                    route=Route(u,head)
                    val=float(np.mean(_predict(route,selectx)==selecty))
                    options.append(('reuse',uid,head,route,val))
            new=TaskUnit(cfg.unit_width,cfg.feature_dim)
            _train(new,trainx,trainy,cfg,seed=cfg.seed+700+t)
            val=float(np.mean(_predict(new,selectx)==selecty))
            options.append(('new',None,new.head,new,val))
        chosen=None
        if options:
            if method == 'famp_naive' or method == 'independent':
                chosen=max(options,key=lambda z:(z[4],z[0]=='reuse'))
            else:
                # Risk-aware selection uses the separate selection split only.
                # It DOES NOT inspect acceptance data and does not retry rejected proposals.
                teacher_s=_predict(base,selectx,t)
                candidates=[]
                for z in options:
                    pred_s=_predict(z[3],selectx)
                    harm_s=float(np.mean((teacher_s==selecty)&(pred_s!=selecty)))
                    with torch.no_grad():
                        ps=F.softmax(base(_tensor(selectx),t),dim=1)
                        qs=F.softmax(z[3](_tensor(selectx)),dim=1)
                        rmse_s=float(torch.sqrt(torch.mean((ps-qs)**2)).item())
                    # This is a heuristic selection screen, not a certificate.
                    if harm_s <= cfg.risk_tolerance + .03 and (method=='no_response' or rmse_s <= .20):
                        candidates.append((z,harm_s,rmse_s))
                if candidates:
                    # Select the lowest incremental parameter cost among accurate, plausible proposals.
                    viable=[p for p in candidates if p[0][4]>=max(q[0][4] for q in candidates)-.04]
                    chosen=min(viable,key=lambda p:(0 if p[0][0]=='reuse' else 1,-p[0][4]))[0]
                else:
                    chosen=max(options,key=lambda z:(z[4],z[0]=='reuse'))
        decision='fallback';cert=None;chosen_val=None;observer_test_harm=None;observer_test_candidate_acc=None
        if chosen is not None:
            kind,uid,head,route,chosen_val=chosen
            student_a=_predict(route,acceptx)
            # Test observer ONLY: evaluated after candidate and policy were selected;
            # these fields never influence selection, acceptance or training.
            testx,testy=splits['test']
            teacher_test=_predict(base,testx,t)
            candidate_test=_predict(route,testx)
            observer_test_harm=float(np.mean((teacher_test==testy)&(candidate_test!=testy)))
            observer_test_candidate_acc=float(np.mean(candidate_test==testy))
            pol=Policy(epsilon=cfg.risk_tolerance, family_alpha=.05,tasks=cfg.n_tasks,min_teacher_acc=cfg.min_teacher_acc,
                       per_class=True,max_response_rmse=.20)
            # Response metric: calibrated class-probability discrepancy, not perturbation robustness.
            with torch.no_grad():
                lp1=F.softmax(base(_tensor(acceptx),t),dim=1)
                lp2=F.softmax(route(_tensor(acceptx)),dim=1)
                response_rmse=float(torch.sqrt(torch.mean((lp1-lp2)**2)).item())
            cert=certificate(accepty,teacher_a,student_a,task=t,policy=pol,response_rmse=response_rmse,
                             classes=[0,1],provenance=f'{cfg.dataset}/seed={cfg.seed}/order={cfg.order}/task={t}')
            passed=cert['checks']['competent'] and cert['checks']['bounded_harm']
            if method!='no_response':passed=passed and cert['checks']['response_consistent']
            if method=='ungated':passed=True
            if passed:
                if kind=='new':
                    # The candidate encoder itself is published and frozen.
                    encoder=route.encoder
                    for p in encoder.parameters():p.requires_grad_(False)
                    units.append(encoder)
                    uid=len(units)-1
                routes[t]=Route(units[uid],head)
                route_predictions[t]=_predict(routes[t],splits['test'][0]).copy()
                decision='reuse' if kind=='reuse' else 'new'
        # Keep only unresolved tasks in replay for gated methods; normal replay always stores.
        if method in ('finetune','ewc'):pass
        elif t not in routes:
            rng=np.random.default_rng(cfg.seed+100+t)
            idx=rng.permutation(len(trainy))[:cfg.replay_per_task]
            memx.append(trainx[idx]);memy.append(trainy[idx]);memt.append(np.full(len(idx),t,dtype=np.int64))
            if method=='derpp':
                with torch.no_grad():memlog.append(base(_tensor(trainx[idx]),t).numpy().astype(np.float32))
        # Evaluate task-specific route if published; fallback otherwise.
        scores=[]
        for j in range(t+1):
            xx,yy=tasks[j]['test']
            pred=_predict(routes[j],xx) if j in routes else _predict(base,xx,j)
            scores.append(float(np.mean(pred==yy)))
        stage_acc.append(scores)
        bmem=(sum(a.nbytes for a in memx)+sum(a.nbytes for a in memy)+sum(a.nbytes for a in memt)+sum(a.nbytes for a in memlog))
        if method=='ewc' and fisher is not None:bmem+=sum(v.numel()*v.element_size() for v in fisher.values())+sum(v.numel()*v.element_size() for v in anchor.values())
        bbase=_counts(base);bunits=sum(_counts(u) for u in units)
        bheads=sum(_counts(r.head) for r in routes.values())
        record=dict(task=t,decision=decision,teacher_accuracy=float(np.mean(teacher_a==accepty)),
            selection_accuracy=chosen_val,certificate=cert,accuracy_so_far=float(np.mean(scores)),
            bytes_fallback=bbase,bytes_units=bunits,bytes_heads=bheads,bytes_replay=bmem,
            observer_test_harm=observer_test_harm,observer_test_candidate_accuracy=observer_test_candidate_acc,
            units=len(units),shared_units=sum(sum(r.encoder is u for r in routes.values())>1 for u in units))
        logs.append(record)
    duration=time.perf_counter()-start
    route_prediction_mismatches=0
    for task, earlier in route_predictions.items():
        now=_predict(routes[task],tasks[task]['test'][0])
        route_prediction_mismatches+=int(np.sum(earlier!=now))
    final=float(np.mean(stage_acc[-1]));acq=float(np.mean([stage_acc[i][i] for i in range(5)]))
    forgetting=float(np.mean([max(stage_acc[j][i] for j in range(i,5))-stage_acc[-1][i] for i in range(5)]))
    # JSON metadata estimates included in published ledger to avoid counting only tensors.
    metadata=json.dumps({'method':method,'config':asdict(cfg),'routes':sorted(routes),'unit_hashes':[_hash_model(u) for u in units]},sort_keys=True)
    bytes_total=logs[-1]['bytes_fallback']+logs[-1]['bytes_units']+logs[-1]['bytes_heads']+logs[-1]['bytes_replay']+len(metadata.encode())
    return dict(dataset=cfg.dataset,seed=cfg.seed,order=cfg.order,method=method,accuracy=final,acquisition=acq,forgetting=forgetting,
                training_seconds=duration,persistent_bytes=bytes_total,admitted=sum(x['decision']!='fallback' for x in logs),
                shared_units=logs[-1]['shared_units'],units=len(units),route_prediction_mismatches=route_prediction_mismatches,stage_accuracy=stage_acc,task_logs=logs,
                source_protocol='trustcl-cpu-1.0',task_order=perm)
