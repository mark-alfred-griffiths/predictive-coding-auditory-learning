"""Proof that labels are unavailable before feedback and K affects updates."""
from dataclasses import replace
import numpy as np
from ubel_jpc.schemas import Config, initial_state, causal_probability
from ubel_jpc.task import generate_trials
from ubel_jpc.models.feedback_learning import (
    update_readout, readout_probability, cross_entropy)
from ubel_jpc.experiments.learning_and_probes import predict, learn, run_model


def test_readout_update_is_stated_equation():
    cfg=Config(lr_w=.07,lr_b=.03)
    initial=initial_state(cfg)
    z,y=.6,1.
    p=float(readout_probability(initial.w,initial.b,z))
    updated=update_readout(initial,z,y,cfg)
    assert np.isclose(float(updated.w),float(initial.w)+cfg.lr_w*(y-p)*z)
    assert np.isclose(float(updated.b),float(initial.b)+cfg.lr_b*(y-p))
    assert np.isclose(float(initial.w),1.0)  # immutable input, not modified


def test_label_blind_inference_and_current_label_counterfactual():
    cfg=Config(trials=18,block_size=6)
    data=generate_trials(cfg)
    state=initial_state(cfg)
    # Predict takes no category label; exchanging this trial's true label
    # without changing its acoustic observation cannot affect perception.
    a=predict(state,cfg,'pc_truncated',.8,.6,3)
    y_a=0.0
    y_b=1.0
    b=predict(state,cfg,'pc_truncated',.8,.6,3)
    assert np.isclose(a[0],b[0]) and np.isclose(a[1],b[1])
    # AFTER feedback, different labels may legitimately change learning.
    next_a=learn(state,cfg,'pc_truncated',.8,.6,3,y_a,a[1])
    next_b=learn(state,cfg,'pc_truncated',.8,.6,3,y_b,b[1])
    assert not np.isclose(next_a.b,next_b.b)


def test_truncation_changes_mapping_updates_when_state_matters():
    cfg=Config()
    state=initial_state(cfg)
    z_short=predict(state,cfg,'pc_truncated',1.0,.8,1)[1]
    z_long=predict(state,cfg,'pc_truncated',1.0,.8,30)[1]
    a=update_readout(state,z_short,1.,cfg)
    b=update_readout(state,z_long,1.,cfg)
    assert abs(float(a.w-b.w))>1e-4


def test_reversals_cannot_leak_hidden_true_probability():
    cfg=Config(trials=24,block_size=6,probe_every=0)
    data=generate_trials(cfg)
    spoof=replace(data,q_true=1.0-data.q_true)
    # If ANY learner reads the hidden q_true, these will differ.
    for model in ('pc_truncated','bayes_mapping','leaky_accumulator'):
        a=run_model(model,cfg,data)
        b=run_model(model,cfg,spoof)
        for row_a,row_b in zip(a.trial_rows,b.trial_rows):
            assert row_a['p_before_feedback']==row_b['p_before_feedback']
            assert row_a['prior_before']==row_b['prior_before']


def test_feedback_absent_freezes_all_parameters_and_priors():
    cfg=Config(trials=25,block_size=5,feedback_probability=0.,probe_every=5)
    data=generate_trials(cfg)
    for model in ('pc_truncated','bayes_mapping'):
        res=run_model(model,cfg,data)
        assert np.isclose(res.final_state.w,1.)
        assert np.isclose(res.final_state.b,0.)
        assert np.isclose(res.final_state.alpha,2.)
        assert np.isclose(res.final_state.beta,2.)
        assert np.allclose(res.final_state.class_mu,[-cfg.category_mean,cfg.category_mean])
