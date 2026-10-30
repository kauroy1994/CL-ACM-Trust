"""Analytical-risk calibration of admission policies, all CPU and offline."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .assurance import Policy, certificate


def simulate(reps=400, n_per_class=35, tasks=5, seed=123, epsilon=.25):
    """Known population model: P(H=1 | class) is set before observations.

    y and predictions are synthesized so the *true* weighted harm is exact.
    The teacher is correct with probability .9 and otherwise wrong; candidates
    flip teacher-correct examples with probability harm/.9. Independent samples
    are drawn for every simulation, task, and class. These are certification
    simulations, NOT neural-network benchmark outcomes.
    """
    rng = np.random.default_rng(seed)
    rows = []
    # Levels intentionally straddle epsilon; no post-hoc selection.
    levels = [.0, .025, .05, .10, .15, .20, .24, .26, .30, .40]
    for harm in levels:
        for rep in range(reps):
            for t in range(tasks):
                ys = np.repeat([0, 1], n_per_class)
                tc = rng.random(len(ys)) < .9
                yp_t = np.where(tc, ys, 1 - ys)
                # Distinct class-conditional harm risks, with exact mean `harm`.
                rates = [max(0., harm - .02), min(.9, harm + .02)] if .02 <= harm <= .88 else [harm, harm]
                flips = np.r_[rng.random(n_per_class) < rates[0] / .9,
                              rng.random(n_per_class) < rates[1] / .9]
                yp_c = np.where(tc & flips, 1 - ys, yp_t)
                pol = Policy(epsilon=epsilon, tasks=tasks, per_class=True)
                cert = certificate(ys, yp_t, yp_c, task=t, policy=pol, classes=[0, 1])
                rows.append({'true_harm': float(np.mean(rates)), 'rep': rep, 'task': t,
                             'upper': cert['harm_upper'], 'accept': int(cert['accepted']),
                             'covered': int(cert['harm_upper'] >= np.mean(rates)),
                             'safe': int(np.mean(rates) <= epsilon),
                             'naive_accept': int(cert['observed_harm_rate'] <= epsilon),
                             'teacher_accuracy': cert['teacher_accuracy']})
    return pd.DataFrame(rows)


def report(df, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out/'calibration_trials.csv', index=False)
    summary = df.groupby('true_harm', as_index=False).agg(
        trials=('covered', 'size'), bound_coverage=('covered', 'mean'),
        certified_accept_rate=('accept', 'mean'), naive_accept_rate=('naive_accept', 'mean'),
        teacher_accuracy=('teacher_accuracy','mean'))
    summary['unsafe_certified_accept_rate'] = summary['certified_accept_rate'].where(summary.true_harm > .25, 0.)
    summary.to_csv(out/'calibration_summary.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    ax.plot(summary.true_harm, summary.certified_accept_rate, 'o-', label='Certified')
    ax.plot(summary.true_harm, summary.naive_accept_rate, 's--', label='Naive plug-in (uncertified)')
    ax.axvline(.25, linestyle=':', color='gray', label='Target harm = 0.25')
    ax.set(xlabel='True harmful-disagreement probability', ylabel='Admission probability', ylim=(-.03,1.03))
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'risk_coverage.png', dpi=180); plt.close(fig)
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reps',type=int,default=400)
    ap.add_argument('--n-per-class',type=int,default=35);ap.add_argument('--seed',type=int,default=123)
    ap.add_argument('--out',default='results/calibration');a=ap.parse_args()
    x=simulate(a.reps,a.n_per_class,seed=a.seed)
    s=report(x,a.out);print(s.round(3).to_string(index=False));print('Saved',a.out)

if __name__=='__main__':main()
