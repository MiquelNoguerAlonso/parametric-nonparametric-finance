"""Invariant resample-move correction for the frozen Wishart--CIR model.

The proposal is auxiliary ancestor selection followed by the recorded defensive
CIR path proposal. Its unnormalized target/proposal ratio is exp(logweight).
Independent Metropolis refreshes jointly update ancestor and the entire daily
path. No jitter, moment matching, likelihood tempering, or model change is used.
"""
import numpy as np
from scipy.special import logsumexp
from wishart_cir import constants, guided_integrals, conditional
from wishart_model import normal_logpdf, systematic


def update_move(y, v, p, x, rng, steps=16, moves=3):
    c, _, g = constants(p)
    n = len(v)
    w = y.transpose(0, 2, 1) @ y
    fw = -np.expm1(-p['kappa_w']) / p['kappa_w']
    fv = -np.expm1(-p['kappa_v']) / p['kappa_v']
    sig = p['eta'] * (c + fw * (w-c)) + (1-p['eta']) * (1+fv*(v-1))[:, None, None] * c
    mu = np.asarray(p['mu']) - .5*p.get('ito_scale', 0)*np.diagonal(sig, axis1=1, axis2=2)
    aux = normal_logpdf(x, mu, sig)
    la = logsumexp(aux)
    prob = np.exp(aux-la)
    xi = np.sqrt(2*p['kappa_v']/p['a'])
    displacement = xi*np.sqrt(1-p['eta'])*np.einsum('i,ni->n', g, np.linalg.solve(sig, (x-mu)[..., None])[..., 0])
    delta = np.clip(np.log1p(np.maximum(displacement, -.8)), -1., 1.5)

    def propose():
        # Independent draws are required for the independence-MH proposal.
        a = rng.choice(n, size=n, p=prob)
        yn, vn, integ, correction = guided_integrals(y[a], v[a], p, rng, steps, delta[a])
        m, cv = conditional(p, integ)
        lw = normal_logpdf(x, m, cv) - aux[a] + correction
        return a, yn, vn, lw

    a, yn, vn, lw = propose()
    norm = logsumexp(lw)
    weights = np.exp(lw-norm)
    pick = systematic(weights, n, rng)
    a, yn, vn, lw = a[pick], yn[pick], vn[pick], lw[pick]
    accepted = 0
    for _ in range(moves):
        ap, yp, vp, lp = propose()
        accept = np.log(rng.random(n)) < lp-lw
        accepted += int(accept.sum())
        a[accept], yn[accept], vn[accept], lw[accept] = ap[accept], yp[accept], vp[accept], lp[accept]
    # Random order avoids retaining artificial ancestor groups across days.
    order = rng.permutation(n)
    return yn[order], vn[order], dict(ess=float(1/np.sum(weights**2)),
        parent_ess=float(1/np.sum(prob**2)), nll=float(-la+np.log(n)-norm+np.log(n)),
        move_acceptance=accepted/(n*moves) if moves else 0.,
        unique_parents=int(len(np.unique(a))),
        variance_mean=float(vn.mean()), variance_second=float(np.mean(vn**2)))
