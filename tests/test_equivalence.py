"""The structural-identifiability negative control is non-negotiable."""
import numpy as np
from ubel_jpc.models.analytic_gaussian import infer, accumulator, accumulator_equivalent


def test_pc_equals_explicit_equivalent_leaky_accumulator():
    for x in (-2.,-.6,0.,.8,1.7):
        for m in (-.8,0.,1.):
            for sd in (.2,.6,1.5):
                for k in (0,1,3,12,32):
                    pc=infer(x,m,sd,1.1,k,.35)
                    acc=accumulator_equivalent(x,m,sd,1.1,k,.35)
                    assert np.isclose(pc,acc,rtol=1e-12,atol=1e-12)


def test_simple_accumulator_equals_pc_with_zero_contextual_prior():
    for k in (0,1,3,12,32):
        assert np.isclose(infer(.8,0.,.8,1.,k,.35),
                          accumulator(.8,0.,.8,1.,k,.35),atol=1e-12)
