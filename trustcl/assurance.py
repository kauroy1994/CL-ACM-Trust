"""Independent, finite-sample assurance for task-specific functional-memory publication.

The certificate controls teacher-relative H = 1[f(X)=Y, g(X)!=Y], *not*
absolute risk, future distribution shift, or correctness of an uncertain teacher.
Acceptance data must be unseen during training, candidate selection, and prior
adaptive decisions. Exact binomial Clopper--Pearson bounds require independent
Bernoulli draws; stratification uses per-class CP plus a union bound.
"""
from dataclasses import dataclass, asdict
from typing import Optional
import hashlib
import json
import numpy as np
from scipy.stats import beta


def upper_cp(k: int, n: int, alpha: float) -> float:
    """One-sided exact Clopper--Pearson upper confidence bound."""
    if not (isinstance(k, (int, np.integer)) and isinstance(n, (int, np.integer))):
        raise TypeError('k and n must be integers')
    if n < 1 or k < 0 or k > n or not (0 < alpha < 1):
        raise ValueError('invalid binomial count or alpha')
    return 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))


def harm_vector(y, teacher, candidate):
    y, teacher, candidate = [np.asarray(a).reshape(-1) for a in (y, teacher, candidate)]
    if not (len(y) == len(teacher) == len(candidate) and len(y) > 0):
        raise ValueError('nonempty, equal-length vectors required')
    return (teacher == y) & (candidate != y)


@dataclass(frozen=True)
class Policy:
    epsilon: float = .25
    family_alpha: float = .05
    tasks: int = 5
    min_teacher_acc: float = .70
    max_response_rmse: float = .20
    per_class: bool = True

    def __post_init__(self):
        if not (0 < self.epsilon < 1 and 0 < self.family_alpha < 1):
            raise ValueError('epsilon and alpha must lie in (0,1)')
        if self.tasks < 1 or not (0 <= self.min_teacher_acc <= 1):
            raise ValueError('invalid tasks or teacher floor')


def certificate(y, teacher, candidate, *, task: int, policy: Policy,
                response_rmse: float = 0., classes=None, provenance: str = '') -> dict:
    """Produce deterministic certificate from held-out predictions and a frozen policy.

    Per-class upper bounds use Bonferroni alpha/(T*C) and known class weights.
    The result is valid for independent class-stratified observations with fixed
    class weights. A missing class is assigned upper bound 1.
    """
    y, teacher, candidate = [np.asarray(a).reshape(-1) for a in (y, teacher, candidate)]
    h = harm_vector(y, teacher, candidate)
    if task < 0 or task >= policy.tasks:
        raise ValueError('task index outside predeclared family')
    if classes is None:
        classes = np.unique(y).tolist()
    else:
        classes = list(classes)
    if not classes or len(set(classes)) != len(classes):
        raise ValueError('classes must be nonempty and distinct')
    if any(int(v) not in set(classes) for v in np.unique(y)):
        raise ValueError('unexpected label')
    if policy.per_class:
        a = policy.family_alpha / (policy.tasks * len(classes))
        bounds = {}
        for c in classes:
            mask = y == c
            n = int(mask.sum())
            k = int(h[mask].sum())
            bounds[str(c)] = dict(n=n, harm=k, upper=upper_cp(k, n, a) if n else 1.)
        upper = float(np.mean([z['upper'] for z in bounds.values()]))
    else:
        a = policy.family_alpha / policy.tasks
        bounds = {'pooled': {'n': len(y), 'harm': int(h.sum()),
                              'upper': upper_cp(int(h.sum()), len(y), a)}}
        upper = bounds['pooled']['upper']
    teacher_acc = float(np.mean(teacher == y))
    candidate_acc = float(np.mean(candidate == y))
    checks = dict(competent=teacher_acc >= policy.min_teacher_acc,
                  bounded_harm=upper <= policy.epsilon,
                  response_consistent=np.isfinite(response_rmse) and response_rmse <= policy.max_response_rmse)
    evidence = dict(task=int(task), n=int(len(y)), teacher_accuracy=teacher_acc,
                    candidate_accuracy=candidate_acc, observed_harm=int(h.sum()),
                    observed_harm_rate=float(h.mean()), harm_upper=upper,
                    response_rmse=float(response_rmse), class_bounds=bounds,
                    policy=asdict(policy), checks=checks, accepted=bool(all(checks.values())),
                    provenance=provenance)
    evidence['certificate_sha256'] = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    ).hexdigest()
    return evidence


def verify_certificate(record: dict, y, teacher, candidate, *, response_rmse=None) -> bool:
    p = Policy(**record['policy'])
    regenerated = certificate(y, teacher, candidate, task=int(record['task']), policy=p,
                              response_rmse=record['response_rmse'] if response_rmse is None else response_rmse,
                              classes=[int(k) for k in record['class_bounds']] if p.per_class else None,
                              provenance=record.get('provenance', ''))
    return regenerated == record


def decide_ungated(y, teacher, candidate, **kwargs):
    return {'accepted': True, 'observed_harm_rate': float(harm_vector(y, teacher, candidate).mean()),
            'note': 'NEGATIVE CONTROL: no statistical assurance applies'}
