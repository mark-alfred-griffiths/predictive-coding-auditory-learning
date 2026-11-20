"""Scientific correctness checks for the feedback-aware inference extension."""
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp

from ubel_jpc.schemas import Config, initial_state
from ubel_jpc.models.coupled_pc import (
    joint_energy, joint_gradient_analytic, pre_feedback_state,
    feedback_settle, learn_from_feedback)
from ubel_jpc.models.feedback_learning import update_readout
from ubel_jpc.task import generate_trials
from ubel_jpc.experiments.learning_and_probes import run_model


def test_joint_activity_gradient_matches_jax_autodiff_and_finite_difference():
    cfg=Config(feedback_weight=1.7)
    state=initial_state(cfg)
    z,x,m,sd,y=.23,1.2,-.15,.8,1.0
    analytic=joint_gradient_analytic(z,x,m,sd,state.w,state.b,y,cfg)
    grad=jax.grad(lambda zz: joint_energy(zz,x,m,sd,state.w,state.b,y,cfg))(z)
    eps=1e-5
    finite=(joint_energy(z+eps,x,m,sd,state.w,state.b,y,cfg)-
            joint_energy(z-eps,x,m,sd,state.w,state.b,y,cfg))/(2*eps)
    assert np.isclose(float(analytic),float(grad),rtol=1e-9)
    assert np.isclose(float(analytic),float(finite),rtol=1e-6)


def test_feedback_lowers_joint_energy_and_residual():
    cfg=Config(post_steps=25,feedback_weight=2.0)
    state=initial_state(cfg)
    zpre=pre_feedback_state(state,.1,.8,1,cfg)
    settled=feedback_settle(state,.1,.8,1,zpre,cfg)
    assert float(settled.energy_after) <= float(settled.energy_before)+1e-10
    first=feedback_settle(state,.1,.8,1,zpre,cfg,steps=0)
    assert float(settled.residual_after) < float(first.residual_after)
    assert not np.isclose(float(settled.z_post),float(settled.z_pre))


def test_zero_post_steps_matches_earlier_fixed_state_gradient_update():
    cfg=Config(post_steps=0)
    state=initial_state(cfg)
    zpre=pre_feedback_state(state,1.2,.8,4,cfg)
    new,settling=learn_from_feedback(state,1.2,.8,1,zpre,cfg)
    old=update_readout(state,zpre,1,cfg)
    assert np.isclose(float(new.w),float(old.w),atol=1e-12)
    assert np.isclose(float(new.b),float(old.b),atol=1e-12)
    assert float(settling.z_pre)==float(settling.z_post)


def test_pre_feedback_never_uses_current_correct_label():
    cfg=Config(trials=24,probe_every=0)
    data=generate_trials(cfg)
    flipped=replace(data,y=1-data.y)
    original=run_model('pc_coupled',cfg,data)
    alternate=run_model('pc_coupled',cfg,flipped)
    assert original.trial_rows[0]['p_before_feedback']==alternate.trial_rows[0]['p_before_feedback']
    # Only the feedback-dependent parameter update and subsequent predictions
    # are allowed to differ (assuming y is revealed on this trial).
    assert not np.isclose(original.trial_rows[0]['w_after'],alternate.trial_rows[0]['w_after'])


def test_feedback_weight_zero_preserves_pre_state_if_first_fully_converged():
    cfg=Config(feedback_weight=0.0,post_steps=16)
    state=initial_state(cfg)
    x,sd=.8,.7
    z_pre=pre_feedback_state(state,x,sd,80,cfg)
    post=feedback_settle(state,x,sd,1,z_pre,cfg)
    assert np.isclose(float(post.z_pre),float(post.z_post),atol=1e-10)


def test_counterfactual_labels_affect_post_state_but_not_pre_state():
    cfg=Config(feedback_weight=2.0,post_steps=8)
    state=initial_state(cfg)
    pre=pre_feedback_state(state,.2,.8,2,cfg)
    yes=feedback_settle(state,.2,.8,1,pre,cfg)
    no=feedback_settle(state,.2,.8,0,pre,cfg)
    assert yes.z_pre==no.z_pre
    assert not np.isclose(float(yes.z_post),float(no.z_post))


def test_no_feedback_and_probes_never_update_coupled_model():
    cfg=Config(trials=25,probe_every=5,feedback_probability=0.0)
    data=generate_trials(cfg)
    with_probes=run_model('pc_coupled',cfg,data,probe_every=1)
    without=run_model('pc_coupled',cfg,data,probe_every=0)
    for item in (with_probes,without):
        assert np.isclose(float(item.final_state.w),1.0)
        assert np.isclose(float(item.final_state.b),0.0)
        assert np.isclose(float(item.final_state.alpha),2.0)
    assert [r['p_before_feedback'] for r in with_probes.trial_rows] == [
        r['p_before_feedback'] for r in without.trial_rows]


def test_joint_energy_requires_valid_configuration():
    import pytest
    for kwargs in ({'post_steps':-1},{'feedback_weight':-1},{'post_rate':1.5}):
        with pytest.raises(ValueError):
            Config(**kwargs).validate()


def test_full_post_feedback_settling_erases_pre_k_difference_in_scalar_model():
    """Crucial negative prediction: the feedback energy has a unique mode.

    For any fixed parameters and fully converged post-feedback inference,
    its final activity depends on x and y, not on the initial pre-state.
    Therefore K can cease to influence the subsequent readout update. This
    is a *model-derived* limit on the proposed learning mechanism.
    """
    state=initial_state(Config())
    cfg=Config(post_steps=64,post_rate=.5)
    outputs=[]
    for k in (0,16):
        pre=pre_feedback_state(state,1.2,.9,k,cfg)
        updated,_=learn_from_feedback(state,1.2,.9,1,pre,cfg)
        outputs.append(float(updated.w))
    assert abs(outputs[0]-outputs[1]) < 1e-8
    # With no extra settling, the prior inference budget does influence the
    # learning update in this controlled example.
    no_post=replace(cfg,post_steps=0)
    outputs=[]
    for k in (0,16):
        pre=pre_feedback_state(state,1.2,.9,k,no_post)
        updated,_=learn_from_feedback(state,1.2,.9,1,pre,no_post)
        outputs.append(float(updated.w))
    assert abs(outputs[0]-outputs[1]) > 1e-3
