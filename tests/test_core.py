import json
import numpy as np
import pytest
from trustcl.assurance import upper_cp,Policy,certificate,verify_certificate
from trustcl.continual import Config,run,dataset
from trustcl.audit import make_manifest,verify_manifest

def test_cp_exact_zero():
    a=.005;n=80
    assert np.isclose(upper_cp(0,n,a),1-a**(1/n))
    assert upper_cp(n,n,a)==1
    assert upper_cp(1,n,a)>upper_cp(0,n,a)

def test_two_class_stratification_and_replay():
    y=np.array([0]*80+[1]*80)
    t=y.copy();c=y.copy()
    policy=Policy(epsilon=.15,tasks=5,per_class=True)
    z=certificate(y,t,c,task=0,policy=policy,classes=[0,1])
    assert z['accepted'] and verify_certificate(z,y,t,c)
    mutated=c.copy();mutated[0]=1
    assert not verify_certificate(z,y,t,mutated)

def test_no_acceptance_data_leakage():
    c=Config(epochs=2)
    split,_=dataset(c)
    for task in split:
        groups=[set(map(lambda a:a.tobytes(),task[k][0])) for k in ('train','select','accept','test')]
        for i in range(4):
            for j in range(i+1,4):assert not (groups[i]&groups[j])

def test_small_run_and_ledger():
    c=Config(dataset='split_digits',seed=99,epochs=3)
    result=run(c,'famp')
    assert len(result['task_logs'])==5
    assert result['persistent_bytes']>0
    assert 0<=result['accuracy']<=1
    for event in result['task_logs']:
        if event['decision']!='fallback':assert event['certificate']['accepted']

def test_tamper_detection(tmp_path):
    (tmp_path/'a.json').write_text('{"foo":1}')
    manifest=make_manifest(tmp_path,tmp_path/'manifest.json')
    assert verify_manifest(tmp_path,manifest)['valid']
    (tmp_path/'a.json').write_text('{"foo":2}')
    assert not verify_manifest(tmp_path,manifest)['valid']
