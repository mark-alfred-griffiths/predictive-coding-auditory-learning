"""Read-only probing, reproducibility and strict causal update checks."""
from dataclasses import replace
import numpy as np
from ubel_jpc.schemas import Config, initial_state, causal_probability
from ubel_jpc.task import generate_trials
from ubel_jpc.experiments.learning_and_probes import (
    run_model, probe, predict, MODEL_NAMES)


def test_inserting_probes_cannot_change_training():
    cfg=Config(trials=30,block_size=10,probe_every=5)
    data=generate_trials(cfg)
    for model in MODEL_NAMES:
        with_probes=run_model(model,cfg,data,probe_every=1)
        no_probes=run_model(model,cfg,data,probe_every=0)
        assert np.isclose(with_probes.final_state.w,no_probes.final_state.w)
        assert np.isclose(with_probes.final_state.b,no_probes.final_state.b)
        assert np.isclose(with_probes.final_state.alpha,no_probes.final_state.alpha)
        assert np.allclose(with_probes.final_state.class_mu,no_probes.final_state.class_mu)
        assert [r['p_before_feedback'] for r in with_probes.trial_rows] == [
            r['p_before_feedback'] for r in no_probes.trial_rows]


def test_probe_function_does_not_mutate_state():
    cfg=Config()
    state=initial_state(cfg)
    before=(float(state.w),float(state.alpha),np.asarray(state.class_mu).copy())
    for m in MODEL_NAMES:
        p=probe(state,cfg,m,[-2,-1,0,1,2])
        assert len(p)==5 and np.isfinite(p).all()
        assert ((p>0)&(p<1)).all()
    assert float(state.w)==before[0] and float(state.alpha)==before[1]
    assert np.array_equal(np.asarray(state.class_mu),before[2])


def test_data_reproducibility_and_seed_independence():
    cfg=Config(trials=50)
    a=generate_trials(cfg,seed=123)
    b=generate_trials(cfg,seed=123)
    c=generate_trials(cfg,seed=124)
    for name in ('x','y','sd','k','q_true','feedback'):
        assert np.array_equal(np.asarray(getattr(a,name)),np.asarray(getattr(b,name)))
    assert not np.array_equal(np.asarray(a.x),np.asarray(c.x))


def test_causal_priors_identical_until_first_differing_feedback():
    cfg=Config(trials=25,block_size=5,probe_every=0)
    a=generate_trials(cfg)
    alternate=replace(a,y=a.y.at[4].set(1.-a.y[4]))
    for model in ('pc_truncated','bayes_mapping'):
        p=run_model(model,cfg,a)
        q=run_model(model,cfg,alternate)
        for t in range(5):
            # Even at t=4, current feedback must NOT affect current choice.
            assert p.trial_rows[t]['p_before_feedback']==q.trial_rows[t]['p_before_feedback']
            assert p.trial_rows[t]['prior_before']==q.trial_rows[t]['prior_before']
        assert p.trial_rows[5]['prior_before']!=q.trial_rows[5]['prior_before']


def test_no_nans_and_probability_ranges():
    cfg=Config(trials=24,block_size=6,probe_every=8,sd_ambiguous=3.5)
    data=generate_trials(cfg)
    for model in MODEL_NAMES:
        res=run_model(model,cfg,data)
        assert all(np.isfinite(r['p_before_feedback']) and 0<r['p_before_feedback']<1
                   for r in res.trial_rows)
        assert all(np.isfinite(r['p']) and 0<r['p']<1 for r in res.probe_rows)
