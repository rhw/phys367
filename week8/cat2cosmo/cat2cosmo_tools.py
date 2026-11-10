"""Helper code for cat2cosmo_3x2pt.ipynb.

The physics (the CCL model, which data and priors to use) lives in the notebook. This file holds the
numerical plumbing: plot style, DALI tensors, evaluating a DALI posterior on a grid, and contour helpers.

DALI in one line: expand the model to second order, d(theta0 + x) ~ d0 + J x + 1/2 H x x, and keep the
exact chi^2 of that expanded model for data d0 + dd:

    chi2(x) = |dd - J x - 1/2 H x x|^2_C
            = F[x,x] + D1[x,x,x] + 1/4 D2[x,x,x,x] - 2 b1[x] - b2[x,x] + dd C^-1 dd

    F = J C^-1 J,  D1 = H C^-1 J,  D2 = H C^-1 H        (the DALI "doublet"; F is the Fisher matrix)
    b1 = J C^-1 dd,  b2 = H C^-1 dd                       (only present when the data are shifted)
"""
from itertools import permutations

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.integrate import trapezoid
from scipy.optimize import minimize

CORAL, BLUE, MAGENTA, GOLD, GREEN, GRAY = '#f07167', '#219ebc', '#c767ac', '#E0A100', '#59a270', '#A2A1A5'


def set_style():
    plt.rcParams.update({
        'figure.dpi': 110, 'font.size': 11, 'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'axes.linewidth': 1.0, 'legend.frameon': False,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
        'axes.prop_cycle': cycler(color=[CORAL, BLUE, MAGENTA, GOLD, GREEN, GRAY]),
    })


def sub_bins(theta_edges, nsub=6):
    """Points and area weights inside each angular bin, for bin-averaged model predictions."""
    t, w = [], []
    for lo, hi in zip(theta_edges[:-1], theta_edges[1:]):
        e = np.geomspace(lo, hi, nsub + 1)
        t.append(np.sqrt(e[1:] * e[:-1]))
        w.append((e[1:] ** 2 - e[:-1] ** 2) / (hi ** 2 - lo ** 2))
    return np.array(t), np.array(w)


# ---------------------------------------------------------------------------------------------- DALI
def _symmetrise(T):
    return sum(np.transpose(T, p) for p in permutations(range(T.ndim))) / len(list(permutations(range(T.ndim))))


def dali_tensors(J, H, cov, mask, dd=None, area=1.0):
    """DALI tensors for the masked data vector. dd = d_obs - d_fid (None: data at the fiducial).
    `area` rescales the survey (C -> C / area)."""
    Ci = area * np.linalg.inv(cov[np.ix_(mask, mask)])
    Jm, Hm = J[mask], H[mask]
    T = {'F': Jm.T @ Ci @ Jm,
         'D1': _symmetrise(np.einsum('iab,ij,jc->abc', Hm, Ci, Jm)),
         'D2': _symmetrise(np.einsum('iab,ij,jcd->abcd', Hm, Ci, Hm)),
         'b1': np.zeros(J.shape[1]), 'b2': np.zeros((J.shape[1],) * 2), 'A0': 0.0}
    if dd is not None:
        r = Ci @ dd[mask]
        T.update(b1=Jm.T @ r, b2=np.einsum('iab,i->ab', Hm, r), A0=dd[mask] @ r)
    return T


def dali_chi2(x, T, prior=None):
    """DALI chi^2 at displacement(s) x = theta - theta0 (shape (..., p)); `prior` is a Gaussian prior Fisher
    matrix centred on theta0."""
    x = np.asarray(x, dtype=float)
    xx = np.einsum('...a,...b->...ab', x, x)
    chi2 = (T['A0'] - 2 * x @ T['b1'] - np.einsum('...ab,ab->...', xx, T['b2'])
            + np.einsum('...ab,ab->...', xx, T['F'])
            + np.einsum('...ab,abc,...c->...', xx, T['D1'], x)
            + 0.25 * np.einsum('...ab,abcd,...cd->...', xx, T['D2'], xx))
    if prior is not None:
        chi2 = chi2 + np.einsum('...ab,ab->...', xx, prior)
    return chi2


def _poly(c, y):
    return c[..., 0] + y * (c[..., 1] + y * (c[..., 2] + y * (c[..., 3] + y * c[..., 4])))


def grid_posterior(T, theta0, prior, bounds, n=(96, 96), nb=(64, 40)):
    """DALI posterior of three parameters inside flat prior `bounds` (3 x 2), plus a Gaussian `prior` Fisher.

    Parameters 0 and 1 go on an n[0] x n[1] grid. At each grid point chi^2 is exactly a quartic in parameter 2,
    which is integrated on nb[1] points around its minimum (so narrow ridges are resolved).
    Returns the 2D density P2 on `axes`, the mean and sd of all three parameters, the best fit (MAP), the
    linear Fisher-bias prediction `lin`, the chi^2 of the data at the best fit, and the Fisher errors.
    """
    theta0, bounds = np.asarray(theta0), np.asarray(bounds)
    Fp = T['F'] + prior
    lin = np.linalg.solve(Fp, T['b1'])
    fisher_sd = np.sqrt(np.diag(np.linalg.inv(Fp)))
    half = 6 * fisher_sd[:2]
    ylo, yhi = bounds[2] - theta0[2]
    nodes = np.arange(-2.0, 3.0)
    to_coef = np.linalg.inv(np.vander(nodes, 5, increasing=True)).T
    for _ in range(4):                                         # widen the box if the posterior touches its edge
        lo = np.maximum(bounds[:2, 0], theta0[:2] + lin[:2] - half)
        hi = np.minimum(bounds[:2, 1], theta0[:2] + lin[:2] + half)
        axes = [np.linspace(lo[i], hi[i], n[i]) for i in range(2)]
        X0, X1 = np.meshgrid(axes[0] - theta0[0], axes[1] - theta0[1], indexing='ij')
        pts = np.stack(np.broadcast_arrays(X0[..., None], X1[..., None], nodes), axis=-1)
        c = dali_chi2(pts, T, prior) @ to_coef                 # quartic coefficients in parameter 2
        yc = np.linspace(ylo, yhi, nb[0])
        ys = yc[np.argmin(_poly(c[..., None, :], yc), axis=-1)]
        for _ in range(5):                                      # Newton steps towards the conditional minimum
            d1 = c[..., 1] + ys * (2 * c[..., 2] + ys * (3 * c[..., 3] + 4 * c[..., 4] * ys))
            d2 = 2 * c[..., 2] + ys * (6 * c[..., 3] + 12 * c[..., 4] * ys)
            ys = np.clip(ys - np.where(d2 > 0, d1 / np.where(d2 > 0, d2, 1), 0), ylo, yhi)
        d2 = 2 * c[..., 2] + ys * (6 * c[..., 3] + 12 * c[..., 4] * ys)
        sig = np.where(d2 > 0, np.sqrt(2 / np.where(d2 > 0, d2, 1)), (yhi - ylo) / 10)
        a, b = np.maximum(ylo, ys - 7 * sig), np.minimum(yhi, ys + 7 * sig)
        Y = a[..., None] + (b - a)[..., None] * np.linspace(0, 1, nb[1])
        chiY = _poly(c[..., None, :], Y)
        w = np.exp(-0.5 * (chiY - chiY.min()))
        P2, Pb1, Pb2 = (trapezoid(w * Y ** k, Y, axis=-1) for k in range(3))
        Pn = P2 / P2.max()
        edge = max(Pn[0].max() if lo[0] > bounds[0, 0] else 0, Pn[-1].max() if hi[0] < bounds[0, 1] else 0,
                   Pn[:, 0].max() if lo[1] > bounds[1, 0] else 0, Pn[:, -1].max() if hi[1] < bounds[1, 1] else 0)
        if edge < 1e-3:
            break
        half = 1.6 * half
    tot = P2.sum()
    m0, m1 = P2.sum(axis=1) / tot, P2.sum(axis=0) / tot
    mean = np.array([m0 @ axes[0], m1 @ axes[1], theta0[2] + Pb1.sum() / tot])
    sd = np.sqrt([m0 @ (axes[0] - mean[0]) ** 2, m1 @ (axes[1] - mean[1]) ** 2, Pb2.sum() / tot - (Pb1.sum() / tot) ** 2])
    i, j = np.unravel_index(np.argmax(P2), P2.shape)
    box = list(zip(bounds[:, 0] - theta0, bounds[:, 1] - theta0))
    xm = minimize(lambda x: dali_chi2(x, T, prior), [X0[i, j], X1[i, j], ys[i, j]], method='L-BFGS-B', bounds=box).x
    return dict(axes=axes, P2=P2 / tot, mean=mean, sd=sd, map=theta0 + xm, lin=theta0 + lin,
                chi2_data=float(dali_chi2(xm, T)), fisher_sd=fisher_sd)


# ---------------------------------------------------------------------------------------------- plotting
def hpd_levels(P, fracs=(0.95, 0.68)):
    """Density thresholds that enclose the given probability fractions."""
    p = np.sort(P.ravel())[::-1]
    return [p[np.searchsorted(np.cumsum(p) / p.sum(), f)] for f in fracs]


def plot_dali(ax, R, color, label=None, filled=True, ls='-'):
    """68% / 95% contours of a grid posterior from grid_posterior()."""
    x, y, P = R['axes'][0], R['axes'][1], R['P2'].T
    levels = hpd_levels(R['P2'])
    if filled:
        for lv, alpha in zip(levels, (0.15, 0.32)):
            ax.contourf(x, y, P, levels=[lv, P.max() * 1.001], colors=[color], alpha=alpha)
    ax.contour(x, y, P, levels=sorted(levels), colors=[color], linestyles=ls, linewidths=1.4)
    if label:
        ax.plot([], [], color=color, ls=ls, lw=2, label=label)


def plot_posteriors(ax, entries, truth, reference=None, title=None, legend='inside', pad=0.15):
    """Omega_m-S8 contours for entries = [(posterior, color, label), ...], with the truth marked.
    `reference` (e.g. the posterior from unbiased data) is drawn as a dashed black outline.
    `legend` is 'inside', 'right' (outside the axes) or 'below'."""
    shown = []
    if reference is not None:
        plot_dali(ax, reference, 'k', 'data at the truth', filled=False, ls='--')
        shown.append(reference)
    for R, color, label in entries:
        plot_dali(ax, R, color, label)
        shown.append(R)
    zoom(ax, shown, pad)
    ax.plot(truth[0], truth[1], 'k+', ms=10)
    ax.set(xlabel=r'$\Omega_m$', ylabel=r'$S_8$', title=title)
    if legend == 'right':
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    elif legend == 'below':
        ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2)
    else:
        ax.legend()


def zoom(ax, posteriors, pad=0.15):
    """Set axis limits around the 95% regions of several grid posteriors."""
    boxes = []
    for R in posteriors:
        i, j = np.nonzero(R['P2'] >= hpd_levels(R['P2'])[0])
        boxes.append([R['axes'][0][i].min(), R['axes'][0][i].max(), R['axes'][1][j].min(), R['axes'][1][j].max()])
    x0, x1, y0, y1 = np.min(boxes, 0)[0], np.max(boxes, 0)[1], np.min(boxes, 0)[2], np.max(boxes, 0)[3]
    ax.set_xlim(x0 - pad * (x1 - x0), x1 + pad * (x1 - x0))
    ax.set_ylim(y0 - pad * (y1 - y0), y1 + pad * (y1 - y0))
