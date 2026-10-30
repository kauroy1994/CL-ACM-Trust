"""Offline independent audit of saved neural-task publication evidence.

Checks what is verifiable from a manifest and (optionally) held-out labels/predictions.
Digest matching is NOT authenticity without a trusted reference digest.
"""
import hashlib,json
from pathlib import Path
from .assurance import verify_certificate

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def make_manifest(root,out):
    root=Path(root);out=Path(out);records=[]
    for f in sorted(root.rglob('*')):
        if f.is_file() and f.resolve()!=out.resolve():records.append({'path':str(f.relative_to(root)),'sha256':sha256(f),'bytes':f.stat().st_size})
    manifest={'type':'trustcl-integrity-v1','files':records,'limitations':'Integrity only; trust in the manifest itself or a trusted hash root is required.'}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(manifest,indent=2))
    return manifest

def verify_manifest(root,manifest):
    root=Path(root).resolve();manifest=manifest if isinstance(manifest,dict) else json.loads(Path(manifest).read_text())
    errors=[]
    for row in manifest['files']:
        p=(root/row['path']).resolve()
        if not p.is_relative_to(root):errors.append('unsafe-path:'+row['path']);continue
        if not p.is_file():errors.append('missing:'+row['path']);continue
        if p.stat().st_size!=row['bytes'] or sha256(p)!=row['sha256']:
            errors.append('modified:'+row['path'])
    return {'valid':not errors,'failures':errors,'tested_files':len(manifest['files'])}

def audit_prediction_record(record,labels,teacher,candidate):
    return {'certificate_reproduced':verify_certificate(record,labels,teacher,candidate),
            'note':'This reconstructs evidence under the declared policy. It cannot prove held-out independence.'}
