"""Counterfactual diagnostic experiments for independent K and L budgets.

The same acoustic observation, correct label, contextual prior, and original
parameters are used across every candidate K (pre-feedback iterations) and L
(post-feedback iterations). This is an intervention on the computational
procedure, not a fitted human data result.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..schemas import Config, initial_state
from ..models.coupled_pc import pre_feedback_state, learn_from_feedback
from ..models.feedback_learning import readout_probability


def counterfactual_grid(cfg: Config, *, x=0.9, sd=0.9,
                        pre_budgets=(0,1,2,4,8,16,32),
                        post_budgets=(0,1,2,4,8,16), labels=(0,1)):
    """Return one diagnostic result per (label,K,L) comparison.

    Each trial starts from identical state; `learn_from_feedback` creates a
    new state and does not mutate the original. Thus no K condition can
    influence the starting parameters in another condition.
    """
    state=initial_state(cfg)
    rows=[]
    for label in labels:
        for k in pre_budgets:
            z=pre_feedback_state(state,x,sd,k,cfg)
            probability=float(readout_probability(state.w,state.b,z))
            for l in post_budgets:
                new,diag=learn_from_feedback(state,x,sd,label,z,cfg,steps=l)
                rows.append({
                    'category_feedback':label,'pre_steps_K':k,'post_steps_L':l,
                    'observation_x':x,'sensory_sd':sd,
                    'p_before_feedback':probability,
                    'z_pre':float(diag.z_pre),'z_post':float(diag.z_post),
                    'delta_w':float(new.w-state.w),
                    'delta_b':float(new.b-state.b),
                    'joint_energy_initial':float(diag.energy_before),
                    'joint_energy_final':float(diag.energy_after),
                    'activity_gradient_abs_final':float(diag.residual_after),
                })
    return rows


def plot_update_grid(rows, path):
    """Plot parameter updates for each K and L, never human measurements."""
    fig, axes=plt.subplots(1,2,figsize=(12,4.4),sharey=False)
    for axis,label in zip(axes,(0,1)):
        group=[r for r in rows if r['category_feedback']==label]
        for k in sorted({r['pre_steps_K'] for r in group}):
            points=sorted((r for r in group if r['pre_steps_K']==k),
                          key=lambda r:r['post_steps_L'])
            axis.plot([r['post_steps_L'] for r in points],
                      [r['delta_w'] for r in points],marker='o',label=f'K={k}')
        axis.set_title(f'Correct category label = {label}')
        axis.set_xlabel('Post-feedback settling steps (L)')
        axis.set_ylabel('Feedback-dependent change in w')
        axis.grid(alpha=.2)
    axes[1].legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8)
    fig.suptitle('Synthetic counterfactual parameter updates (not human data)')
    fig.tight_layout()
    fig.savefig(path,dpi=150,bbox_inches='tight')
    plt.close(fig)
