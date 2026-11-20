"""Golden-reference mathematical tests for Gaussian prediction-error inference."""
import numpy as np
import jax
import jax.numpy as jnp
from ubel_jpc.models.analytic_gaussian import (
    infer, equilibrium, energy, gradient, trajectory)


def test_reference_is_stationary_and_energy_minimum():
    for x, m, sd, p in ((1.2,-0.2,.8,1.0),(-2.,.5,1.2,.6),(.1,.1,.5,3.0)):
        z = equilibrium(x,m,sd,p)
        assert abs(float(gradient(z,x,m,sd,p))) < 1e-12
        assert float(energy(z,x,m,sd,p)) <= float(energy(m,x,m,sd,p)) + 1e-12


def test_convergence_matches_closed_form_across_noise_conditions():
    for x in (-2., -.3, 0., .8, 2.):
        for sd in (.2,.6,1.5):
            for k in (0,1,3,10,40,100):
                m=.3
                z=infer(x,m,sd,1.2,k,rate=.35)
                expected=equilibrium(x,m,sd,1.2) + (1-.35)**k*(m-equilibrium(x,m,sd,1.2))
                assert np.isclose(z,expected,rtol=1e-11,atol=1e-11)


def test_energy_decreases_and_residual_reduces():
    z, e, residual = trajectory(1.2,-.3,.7,1.,max_k=50)
    assert np.all(np.diff(np.asarray(e))<=1e-12)
    assert np.all(np.diff(np.asarray(residual))<=1e-12)
    assert float(residual[-1])<1e-7


def test_autodiff_matches_analytic_and_finite_difference():
    x,m,sd,prior_sd=.8,-.3,.7,1.1
    z=.4
    autodiff=jax.grad(lambda zz:energy(zz,x,m,sd,prior_sd))(jnp.asarray(z))
    h=1e-5
    numeric=(energy(z+h,x,m,sd,prior_sd)-energy(z-h,x,m,sd,prior_sd))/(2*h)
    assert np.isclose(autodiff,gradient(z,x,m,sd,prior_sd),atol=1e-12)
    assert np.isclose(autodiff,numeric,atol=1e-8)
