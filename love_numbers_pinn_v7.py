"""
LOVE NUMBERS via PINN  --  v7
=============================================================================
v7 changes (on top of v6):
  [V7-1] PROGRESS_EVERY: train() prints the running loss every N epochs by
         default, so a long run is visibly alive.  Set PROGRESS_EVERY = 0 to
         silence it.  The held-out loss is still evaluated every
         HOLDOUT_EVERY epochs, which is what model selection uses.
  [V7-2] table_sobolev checkpoints every run to JSON and skips runs already
         present, so a Colab disconnect costs one run rather than the table.
         It also prints each row as soon as its seeds finish, and the
         equal-wall-clock block can be turned off.
  [V7-3] table_ampmap reports A_B alongside A_A, so the claim that only
         A_dec is amplified is demonstrated across the map rather than
         asserted at one configuration.
=============================================================================
Changes from v5.1, in order of importance.  Each is marked [V6-n] in the code.

[V6-1] EXACT EXPONENT.  v5.1 used the integer N_int = 2*ell_int+1 both as the
       shift in the tilded parameters (a-N, b-N) and as the power in the basis
       f1 = (1-z)^N.  The exact exponent is N_ex = a+b-c = 2*ell+1 = N_int+2eps.
       The two differ by (1-z)^(-2eps) = 1 + 9.2 eps at z=0.99, i.e. at the same
       order as the quantity being measured.  With the exact exponent the tilded
       parameters are exactly (c-b, c-a, c) and the decomposition
       ytilde = A f1 + B f2 is an identity.
       Set Config(exact_exponent=False) to reproduce the v5.1 behaviour.

[V6-2] FLOAT64.  v5.1 ran the network in float32 against a 50-digit basis.

[V6-3] NO INTERPOLATION.  v5.1 built the Frobenius reference on an 800-point
       grid and then np.interp'd it, injecting ~1e-6 errors -- ten orders of
       magnitude above the series itself.  Methods 3 and 6b were fitting the
       piecewise-linear interpolant.  The series is now evaluated directly at
       every requested point (Horner, mpmath, cached).

[V6-4] FROBENIUS ANALOGUES FOR 6a AND 7.  v5.1 had no noise-free input for
       these, yet claimed the whole local family failed on exact input.

[V6-5] HELD-OUT MODEL SELECTION.  v5.1 kept the snapshot with the lowest
       *sampled* batch loss over thousands of draws, which selects favourable
       batches rather than good parameters and biases the reported floor down.

[V6-6] MULTI-SEED.  Median and range over seeds.

[V6-7] AMPLIFICATION FACTORS.  Each linear estimator is exposed as an explicit
       operator L with (A,B) = L y, so A_A = ||L[0,:]|| ||y|| / |A| can be
       computed with no training at all.

[V6-8] TRUNCATION.  3000 terms for values, 12000 for the N-th derivative.

Global knobs: EPOCHS, LAM_H1, LAM_H2 (see [V6-11] below).

Drivers at the bottom; set TASKS to choose.  Cheap tasks (no training):
    baseline, convention, amplification, ampmap, contamination
Expensive tasks (training):
    sobolev, scan, lms, convergence
=============================================================================
"""

import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mpmath as mp
import matplotlib
HEADLESS = os.environ.get("LOVE_HEADLESS", "1") == "1"
if HEADLESS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P

# [V6-2] ------------------------------------------------------------------
torch.set_default_dtype(torch.float64)
DTYPE = torch.float64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mp.mp.dps = 50
OUTDIR = os.getcwd()

N_TERMS_VAL = 3000     # [V6-8]
N_TERMS_DER = 12000    # [V6-8]

# =============================================================================
# [V6-11] GLOBAL KNOBS
# Training budget and Sobolev coefficients in one place.  Every driver
# defaults to these; pass n_epochs= / lam_h1= / lam_h2= to override per call.
#   LAM_H1 = LAM_H2 = 0  reduces the loss to the plain weighted L2 residual.
# =============================================================================
EPOCHS = 20000         # default training budget
LAM_H1 = 0.1           # weight of the (R')^2 term
LAM_H2 = 0.01          # weight of the (R'')^2 term

# [V7-1] progress reporting during training
PROGRESS_EVERY = 250   # print the batch loss every this many epochs (0 = off)
HOLDOUT_EVERY  = 100   # evaluate the held-out loss every this many epochs


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    """All parameters for one (s, ell, m, a_kerr, eps) point."""

    def __init__(self, s, ell_int, m, eps, a_kerr=0.2, exact_exponent=True):
        self.s, self.ell_int, self.m = s, ell_int, m
        self.a_kerr, self.eps = a_kerr, eps
        self.exact_exponent = exact_exponent

        rt = math.sqrt(1.0 - 4.0 * a_kerr ** 2)
        rp, rm = (1 + rt) / 2, (1 - rt) / 2
        self.gamma = a_kerr / (rp - rm)

        ell = ell_int + eps
        self.ell = ell
        self.a = complex(1 + ell - s, 0.0)
        self.b = complex(1 + ell, 2 * m * self.gamma)
        self.c = complex(1 - s, 2 * m * self.gamma)

        # [V6-9] EXTENDED-PRECISION PARAMETERS.
        # c-b = -(l+s) - eps is a near-integer.  Built in float64 it carries an
        # absolute error ~1e-16, so its eps-part is only good to a relative
        # 1e-16/eps.  Everything downstream that divides by that eps-part --
        # the Frobenius recursion at n = l+s, the coefficients c_k of f2 for
        # k >= N, the Gamma ratios -- inherits a 1e-16/eps error.  That is the
        # 4.4e-11 floor seen at eps=1e-5, and it is NOT series truncation: it
        # is dps-independent and grows exactly as 1/eps.  All mpmath-side
        # quantities are therefore built from mp.mpf(str(...)).
        E = mp.mpf(str(eps))
        aK = mp.mpf(str(a_kerr))
        rt_m = mp.sqrt(1 - 4 * aK ** 2)
        g_m = aK / rt_m                        # = a/(r+ - r-)
        self.gamma_mp = g_m
        ell_m = mp.mpf(ell_int) + E
        self.a_mp = mp.mpc(1 + ell_m - s, 0)
        self.b_mp = mp.mpc(1 + ell_m, 2 * m * g_m)
        self.c_mp = mp.mpc(1 - s, 2 * m * g_m)
        self.Nex_mp = 2 * ell_m + 1

        self.N_int = int(2 * ell_int + 1)          # integer, used for d^N/dz^N
        self.N_ex = 2.0 * ell + 1.0                # == a + b - c, exact

        # [V6-1] the one line that matters
        self.N_pow = self.N_ex if exact_exponent else float(self.N_int)
        self.Npow_mp = self.Nex_mp if exact_exponent else mp.mpf(self.N_int)

        if exact_exponent:
            self.at = self.c - self.b              # == a - N_ex, exactly
            self.bt = self.c - self.a              # == b - N_ex, exactly
        else:
            self.at = self.a - self.N_int
            self.bt = self.b - self.N_int
        self.ct = self.c

        # The Frobenius reference always uses the EXACT tilded parameters,
        # since only for those is the tilded equation hypergeometric.  In
        # legacy mode the network therefore solves a displaced ODE -- that
        # displacement is precisely what Table "convention" measures.
        self.sat, self.sbt, self.sct = self.c - self.b, self.c - self.a, self.c
        self.sat_mp = self.c_mp - self.b_mp
        self.sbt_mp = self.c_mp - self.a_mp
        self.sct_mp = self.c_mp

    # real/imaginary decomposition used by the torch residuals
    def real_params(self):
        a_r, a_i = self.at.real, self.at.imag
        b_r, b_i = self.bt.real, self.bt.imag
        c_r, c_i = self.ct.real, self.ct.imag
        alpha = a_r * b_r - a_i * b_i
        beta = a_r * b_i + a_i * b_r
        return dict(a_r=a_r, a_i=a_i, b_r=b_r, b_i=b_i,
                    c_r=c_r, c_i=c_i, alpha=alpha, beta=beta)

    def yp0(self):
        z = (self.at * self.bt) / self.ct
        return float(z.real), float(z.imag)

    def label(self):
        return (f"s={self.s} l={self.ell_int} m={self.m} "
                f"eps={self.eps:.0e} N={self.N_int} "
                f"{'exact' if self.exact_exponent else 'INTEGER'}")


# =============================================================================
# mpmath helpers
# =============================================================================
def _mpc(z):
    """Accepts float, complex, mp.mpf or mp.mpc without losing precision."""
    return mp.mpmathify(z)


def hyp2f1_exact(a, b, c, z, deriv_order=0):
    Ca, Cb, Cc = _mpc(a), _mpc(b), _mpc(c)
    prefac = mp.mpc(1, 0)
    for k in range(deriv_order):
        prefac *= (Ca + k) * (Cb + k) / (Cc + k)
    Ca, Cb, Cc = Ca + deriv_order, Cb + deriv_order, Cc + deriv_order
    zr = np.atleast_1d(np.asarray(z, dtype=float))
    out = np.empty(zr.shape, dtype=complex)
    for i, zi in enumerate(zr):
        out[i] = complex(prefac * mp.hyp2f1(Ca, Cb, Cc, mp.mpf(float(zi))))
    return out


def _pow_mp(om, p_mp):
    """(1-z)^p with p carried in mpmath, so the eps-part of the exponent is
    not truncated to float64 before exponentiation."""
    om = np.atleast_1d(np.asarray(om, dtype=float))
    out = np.empty(om.shape, dtype=complex)
    for i, o in enumerate(om):
        out[i] = complex(mp.power(mp.mpf(float(o)), p_mp))
    return out


def gamma_ratio_AB(cfg):
    """Analytical connection coefficients.  Convention-independent: these
    involve only (a,b,c), never the tilde redefinition, so they are the same
    reference for the exact and the legacy runs."""
    Ca, Cb, Cc = cfg.a_mp, cfg.b_mp, cfg.c_mp
    G = mp.gamma
    A = (G(Cc) * G(Cc - Ca - Cb)) / (G(Cc - Ca) * G(Cc - Cb))
    B = (G(Cc) * G(Ca + Cb - Cc)) / (G(Ca) * G(Cb))
    return complex(A), complex(B)


def c_coefficient(cfg, k):
    """k-th Taylor coefficient of f2 in powers of (1-z):
           c_k = (c-a)_k (c-b)_k / [ (1-N_ex)_k k! ]
    NOTE: c_N is FINITE, not O(1/eps).  The factor (N - N_ex) = -2eps in the
    denominator is cancelled by (c-b+ell+s) = -eps in the numerator, which is
    present whenever N = 2l+1 >= l+s+1, i.e. whenever l >= s.  In the limit
    c_N -> -A/B, so A_eff = A + B c_N is the (identically vanishing) N-th
    Taylor coefficient of a degree-(l+s) polynomial."""
    Ca, Cb = cfg.c_mp - cfg.a_mp, cfg.c_mp - cfg.b_mp
    den = 1 - cfg.Nex_mp
    return complex(mp.rf(Ca, k) * mp.rf(Cb, k) / (mp.rf(den, k) * mp.factorial(k)))


# =============================================================================
# Basis adapted to z = 1
# =============================================================================
def _basis_2f1(cfg, z):
    """f1 = (1-z)^N_pow  2F1(a, b; a+b-c+1; 1-z)
       f2 =              2F1(c-a, c-b; c-a-b+1; 1-z)      [full 2F1]
    Only the power N_pow depends on the exponent convention."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    om = 1.0 - z
    F = hyp2f1_exact(cfg.a_mp, cfg.b_mp, cfg.a_mp + cfg.b_mp - cfg.c_mp + 1, om)
    f1 = _pow_mp(om, cfg.Npow_mp) * F
    f2 = hyp2f1_exact(cfg.c_mp - cfg.a_mp, cfg.c_mp - cfg.b_mp,
                      cfg.c_mp - cfg.a_mp - cfg.b_mp + 1, om)
    return f1, f2


def _basis_2f1_derivs(cfg, z):
    z = np.atleast_1d(np.asarray(z, dtype=float))
    om = 1.0 - z
    p = cfg.N_pow
    pm = cfg.Npow_mp
    abc1 = cfg.a_mp + cfg.b_mp - cfg.c_mp + 1
    F = hyp2f1_exact(cfg.a_mp, cfg.b_mp, abc1, om, 0)
    Fw = hyp2f1_exact(cfg.a_mp, cfg.b_mp, abc1, om, 1)
    Fww = hyp2f1_exact(cfg.a_mp, cfg.b_mp, abc1, om, 2)
    op0, op1, op2 = _pow_mp(om, pm), _pow_mp(om, pm - 1), _pow_mp(om, pm - 2)
    f1 = op0 * F
    df1 = -(p * op1 * F + op0 * Fw)
    d2f1 = (p * (p - 1) * op2 * F + 2 * p * op1 * Fw + op0 * Fww)

    a2 = cfg.c_mp - cfg.a_mp
    b2 = cfg.c_mp - cfg.b_mp
    c2 = cfg.c_mp - cfg.a_mp - cfg.b_mp + 1
    G = hyp2f1_exact(a2, b2, c2, om, 0)
    Gw = hyp2f1_exact(a2, b2, c2, om, 1)
    Gww = hyp2f1_exact(a2, b2, c2, om, 2)
    f2, df2, d2f2 = G, -Gw, Gww
    return f1, f2, df1, df2, d2f1, d2f2


# =============================================================================
# [V6-10] FROBENIUS BASIS AT z = 1, BUILT FROM THE ODE COEFFICIENTS ALONE
# =============================================================================
# Substituting w = 1 - z into the tilded equation
#     z(1-z) y'' + [ct - (at+bt+1) z] y' - at bt y = 0
# gives
#     w(1-w) y_ww + [C2 - (at+bt+1) w] y_w - at bt y = 0,
#     C2 = at + bt + 1 - ct = 1 - Nex,      Nex = ct - at - bt.
# So the two exponents at z=1 are 0 and Nex, and the two Frobenius branches are
#
#     f2(z) = sum_k g_k w^k ,
#             g_0 = 1,  g_{k+1}/g_k = (k+at)(k+bt) / [(k+1)(k+C2)]
#
#     f1(z) = w^p sum_k h_k w^k ,
#             h_0 = 1,  h_{k+1}/h_k = (k+A1)(k+B1) / [(k+1)(k+C1)]
#             A1 = at + Nex,  B1 = bt + Nex,  C1 = 1 + Nex
#
# Every ingredient is read off the ODE.  Nothing here is the connection
# formula, and nothing requires knowing the global solution -- which is the
# point: this construction survives the confluent limit, where the z=1
# expansion becomes asymptotic but the recursion still runs, and where no
# closed-form 2F1 exists.
#
# Set USE_FROBENIUS_BASIS = False to fall back on mpmath's 2F1.  For the
# hypergeometric case the two are provably identical; verify_basis() checks it.
# =============================================================================
USE_FROBENIUS_BASIS = True
BASIS_TOL = mp.mpf('1e-45')
BASIS_MAXTERMS = 40000


class BasisAtOne:
    """Frobenius pair at z=1 generated from the ODE coefficients."""

    def __init__(self, cfg):
        self.cfg = cfg
        at, bt, ct = cfg.sat_mp, cfg.sbt_mp, cfg.sct_mp
        self.at, self.bt, self.ct = at, bt, ct
        self.Nex = ct - at - bt                 # == a + b - c, from the ODE
        self.C2 = at + bt + 1 - ct              # == 1 - Nex
        self.A1 = at + self.Nex                 # == a
        self.B1 = bt + self.Nex                 # == b
        self.C1 = 1 + self.Nex
        self.p = cfg.Npow_mp                    # exponent convention lives here
        self._g = None
        self._h = None
        self._cache = {}

    def _coeffs(self, A, B, C, n):
        out = [mp.mpc(1, 0)]
        for k in range(n):
            out.append(out[-1] * (k + A) * (k + B) / ((k + 1) * (k + C)))
        return out

    def _n_needed(self, wmax):
        if wmax <= 0:
            return 4
        lw = -mp.log(mp.mpf(float(wmax)))
        if lw <= 0:
            return BASIS_MAXTERMS
        return int(min(BASIS_MAXTERMS, max(50, float(-mp.log(BASIS_TOL) / lw) * 1.3)))

    def _sum3(self, coeffs, w):
        """Forward summation of S = sum c_k w^k together with S' and S''
        (derivatives w.r.t. w), with early termination."""
        W = mp.mpf(float(w))
        S = mp.mpc(0); S1 = mp.mpc(0); S2 = mp.mpc(0)
        pw = mp.mpf(1)                     # w^k
        pw1 = mp.mpf(0)                    # w^(k-1)
        pw2 = mp.mpf(0)                    # w^(k-2)
        for k, c in enumerate(coeffs):
            t = c * pw
            S += t
            if k >= 1:
                S1 += c * k * pw1
            if k >= 2:
                S2 += c * k * (k - 1) * pw2
            if k >= 8 and abs(t) < BASIS_TOL * (abs(S) + 1):
                break
            pw2 = pw1 if k >= 1 else mp.mpf(1)
            pw1 = pw
            pw = pw * W
        return S, S1, S2

    def eval(self, z, derivs=False):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        key = (derivs, z.tobytes())
        if key in self._cache:
            return self._cache[key]
        om = 1.0 - z
        n = self._n_needed(om.max())
        if self._g is None or len(self._g) < n + 1:
            self._g = self._coeffs(self.at, self.bt, self.C2, n)
            self._h = self._coeffs(self.A1, self.B1, self.C1, n)
        f1 = np.empty(z.shape, dtype=complex); f2 = np.empty(z.shape, dtype=complex)
        df1 = np.empty(z.shape, dtype=complex); df2 = np.empty(z.shape, dtype=complex)
        d2f1 = np.empty(z.shape, dtype=complex); d2f2 = np.empty(z.shape, dtype=complex)
        p = self.p
        for i, w in enumerate(om):
            G, Gw, Gww = self._sum3(self._g, w)
            H, Hw, Hww = self._sum3(self._h, w)
            W = mp.mpf(float(w))
            wp = mp.power(W, p)
            f2[i] = complex(G)
            f1[i] = complex(wp * H)
            if derivs:
                wp1 = mp.power(W, p - 1)
                wp2 = mp.power(W, p - 2)
                df2[i] = complex(-Gw)                       # dw/dz = -1
                d2f2[i] = complex(Gww)
                df1[i] = complex(-(p * wp1 * H + wp * Hw))
                d2f1[i] = complex(p * (p - 1) * wp2 * H
                                  + 2 * p * wp1 * Hw + wp * Hww)
        res = (f1, f2) if not derivs else (f1, f2, df1, df2, d2f1, d2f2)
        self._cache[key] = res
        return res


def make_basis(cfg, z):
    if USE_FROBENIUS_BASIS:
        return _basis_obj(cfg).eval(z, derivs=False)
    return _basis_2f1(cfg, z)


def make_basis_with_derivs(cfg, z):
    if USE_FROBENIUS_BASIS:
        return _basis_obj(cfg).eval(z, derivs=True)
    return _basis_2f1_derivs(cfg, z)


_BASIS_CACHE = {}


def _basis_obj(cfg):
    key = (cfg.s, cfg.ell_int, cfg.m, cfg.eps, cfg.a_kerr, cfg.exact_exponent)
    if key not in _BASIS_CACHE:
        _BASIS_CACHE[key] = BasisAtOne(cfg)
    return _BASIS_CACHE[key]


def verify_basis(cfg, z=None):
    """Frobenius-generated basis vs the mpmath 2F1 closed form."""
    if z is None:
        z = np.array([0.02, 0.2, 0.5, 0.8, 0.95, 0.99])
    f1a, f2a = _basis_2f1(cfg, z)
    f1b, f2b = _basis_obj(cfg).eval(z)
    e1 = np.abs(f1a - f1b) / np.abs(f1a)
    e2 = np.abs(f2a - f2b) / np.abs(f2a)
    print(f"  {cfg.label()}")
    print(f"  {'z':<8}{'|df1|/f1':<14}{'|df2|/f2':<14}")
    for i, zz in enumerate(z):
        print(f"  {zz:<8}{e1[i]:<14.3e}{e2[i]:<14.3e}")
    return e1.max(), e2.max()



# =============================================================================
# [V6-3] Frobenius series, evaluated DIRECTLY at each point (no interpolation)
# =============================================================================
class Frobenius:
    """ytilde = sum alpha_n z^n with
           alpha_{n+1}/alpha_n = (n+at)(n+bt)/[(n+1)(n+ct)],  alpha_0 = 1,
       (at,bt,ct) = (c-b, c-a, c).  Closed form: 2F1(c-a,c-b;c;z).
       Horner evaluation in mpmath, results cached per z-array."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._alphas = {}
        self._cache = {}

    def alphas(self, n_terms):
        if n_terms in self._alphas:
            return self._alphas[n_terms]
        Ca, Cb, Cc = self.cfg.sat_mp, self.cfg.sbt_mp, self.cfg.sct_mp
        al = [mp.mpc(1, 0)]
        for n in range(n_terms):
            den = (n + 1) * (n + Cc)
            al.append(al[-1] * (n + Ca) * (n + Cb) / den)
        self._alphas[n_terms] = al
        return al

    def series(self, z, order=0, n_terms=None):
        """order-th z-derivative of the series."""
        if n_terms is None:
            n_terms = N_TERMS_VAL if order == 0 else N_TERMS_DER
        z = np.atleast_1d(np.asarray(z, dtype=float))
        key = (order, n_terms, z.tobytes())
        if key in self._cache:
            return self._cache[key]
        al = self.alphas(n_terms)
        # coefficients of the order-th derivative: b_{n-order} = a_n n!/(n-order)!
        if order == 0:
            coef = al
        else:
            coef = []
            for n in range(order, len(al)):
                fac = mp.mpf(1)
                for j in range(order):
                    fac *= (n - j)
                coef.append(al[n] * fac)
        out = np.empty(z.shape, dtype=complex)
        for i, zi in enumerate(z):
            Z = mp.mpf(float(zi))
            acc = mp.mpc(0)
            for cc in reversed(coef):          # Horner
                acc = acc * Z + cc
            out[i] = complex(acc)
        self._cache[key] = out
        return out

    def y(self, z):
        return self.series(z, 0)

    def dy(self, z):
        return self.series(z, 1, n_terms=N_TERMS_VAL)

    def yN(self, z):
        return self.series(z, self.cfg.N_int, n_terms=N_TERMS_DER)

    def certify(self, z=0.999, n_list=(500, 1000, 3000, 12000), order=0):
        """[V6-12] Compare the truncated series against the closed form,
        ENTIRELY IN mpmath.  The previous version cast both sides to complex
        before comparing, so it saturated at 1e-16 and could not distinguish
        500 terms from 12000 -- it certified double precision, not the series.

        order=0 checks the value; order=N checks the N-th derivative against
            d^N/dz^N 2F1(A,B;C;z) = (A)_N (B)_N / (C)_N * 2F1(A+N,B+N;C+N;z),
        which is the claim the paper actually relies on (the derivative is the
        demanding case, not the value)."""
        A, B, C = self.cfg.sat_mp, self.cfg.sbt_mp, self.cfg.sct_mp
        Z = mp.mpf(str(z))
        pref = mp.mpc(1, 0)
        for k in range(order):
            pref *= (A + k) * (B + k) / (C + k)
        ref = pref * mp.hyp2f1(A + order, B + order, C + order, Z)
        rows = []
        for n in n_list:
            al = self.alphas(n)
            acc = mp.mpc(0)
            pw = mp.mpf(1)
            for j in range(order, len(al)):
                fac = mp.mpf(1)
                for i in range(order):
                    fac *= (j - i)
                acc += al[j] * fac * pw
                pw = pw * Z
            rows.append((n, float(abs(acc - ref) / abs(ref))))
        return complex(ref), rows


# =============================================================================
# NETWORK  (hard-enforced Frobenius BC at z=0)
# =============================================================================
class PINN(nn.Module):
    def __init__(self, hidden=128, depth=4, yp0_re=0.0, yp0_im=0.0):
        super().__init__()
        self.register_buffer('yp0_re', torch.tensor(float(yp0_re)))
        self.register_buffer('yp0_im', torch.tensor(float(yp0_im)))
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        w = self.net(z)
        z2 = z * z
        u = 1.0 + self.yp0_re * z + z2 * w[:, 0:1]
        v = 0.0 + self.yp0_im * z + z2 * w[:, 1:2]
        return u, v


def derivatives(model, z, order=4):
    """[V6-11] `order` caps how many autograd passes are taken.  With
    lam_h2 = 0 the fourth derivative is never used, and with lam_h1 = 0 the
    third is not either; computing them anyway would make an H2-vs-L2
    wall-clock comparison meaningless, since the cost is dominated by the
    autograd passes rather than by the algebra."""
    z = z.requires_grad_(True)
    u, v = model(z)
    g = lambda f: torch.autograd.grad(f.sum(), z, create_graph=True)[0]
    du, dv = g(u), g(v)
    d2u, d2v = g(du), g(dv)
    zero = torch.zeros_like(u)
    if order <= 2:
        return u, v, du, dv, d2u, d2v, zero, zero, zero, zero
    d3u, d3v = g(d2u), g(d2v)
    if order <= 3:
        return u, v, du, dv, d2u, d2v, d3u, d3v, zero, zero
    d4u, d4v = g(d3u), g(d3v)
    return u, v, du, dv, d2u, d2v, d3u, d3v, d4u, d4v


def residuals(pr, z, u, v, du, dv, d2u, d2v):
    P_ = pr['c_r'] - (pr['a_r'] + pr['b_r'] + 1.0) * z
    Q_ = pr['c_i'] - (pr['a_i'] + pr['b_i']) * z
    R_u = z * (1 - z) * d2u + P_ * du - Q_ * dv - pr['alpha'] * u + pr['beta'] * v
    R_v = z * (1 - z) * d2v + P_ * dv + Q_ * du - pr['alpha'] * v - pr['beta'] * u
    return R_u, R_v


def residual_derivatives(pr, z, du, dv, d2u, d2v, d3u, d3v):
    P_ = pr['c_r'] - (pr['a_r'] + pr['b_r'] + 1.0) * z
    Q_ = pr['c_i'] - (pr['a_i'] + pr['b_i']) * z
    Pp = -(pr['a_r'] + pr['b_r'] + 1.0)
    Qp = -(pr['a_i'] + pr['b_i'])
    dR_u = ((1 - 2 * z) * d2u + z * (1 - z) * d3u
            + Pp * du + P_ * d2u - Qp * dv - Q_ * d2v
            - pr['alpha'] * du + pr['beta'] * dv)
    dR_v = ((1 - 2 * z) * d2v + z * (1 - z) * d3v
            + Pp * dv + P_ * d2v + Qp * du + Q_ * d2u
            - pr['alpha'] * dv - pr['beta'] * du)
    return dR_u, dR_v


def residual_second_derivatives(pr, z, d2u, d2v, d3u, d3v, d4u, d4v):
    P_ = pr['c_r'] - (pr['a_r'] + pr['b_r'] + 1.0) * z
    Q_ = pr['c_i'] - (pr['a_i'] + pr['b_i']) * z
    Pp = -(pr['a_r'] + pr['b_r'] + 1.0)
    Qp = -(pr['a_i'] + pr['b_i'])
    d2R_u = (-2 * d2u + 2 * (1 - 2 * z) * d3u + z * (1 - z) * d4u
             + 2 * Pp * d2u + P_ * d3u - 2 * Qp * d2v - Q_ * d3v
             - pr['alpha'] * d2u + pr['beta'] * d2v)
    d2R_v = (-2 * d2v + 2 * (1 - 2 * z) * d3v + z * (1 - z) * d4v
             + 2 * Pp * d2v + P_ * d3v + 2 * Qp * d2u + Q_ * d3u
             - pr['alpha'] * d2v - pr['beta'] * d2u)
    return d2R_u, d2R_v


def sobolev_loss(model, pr, z, lam_h1=None, lam_h2=None, imp_w=None):
    """Weighted H^2-Sobolev norm of the residual, estimated by Monte Carlo.
    The batch mean estimates L/(z_max - z_min); the constant does not move
    the minimiser.  lam_h1 = lam_h2 = 0 gives the plain weighted L2 loss."""
    if lam_h1 is None:
        lam_h1 = LAM_H1
    if lam_h2 is None:
        lam_h2 = LAM_H2
    order = 4 if lam_h2 != 0 else (3 if lam_h1 != 0 else 2)
    u, v, du, dv, d2u, d2v, d3u, d3v, d4u, d4v = derivatives(model, z, order)
    R_u, R_v = residuals(pr, z, u, v, du, dv, d2u, d2v)
    w = 1.0 / (z * (1.0 - z))
    integrand = w * (R_u ** 2 + R_v ** 2)
    if lam_h1 != 0:
        dR_u, dR_v = residual_derivatives(pr, z, du, dv, d2u, d2v, d3u, d3v)
        integrand = integrand + lam_h1 * w * (dR_u ** 2 + dR_v ** 2)
    if lam_h2 != 0:
        d2R_u, d2R_v = residual_second_derivatives(pr, z, d2u, d2v,
                                                   d3u, d3v, d4u, d4v)
        integrand = integrand + lam_h2 * w * (d2R_u ** 2 + d2R_v ** 2)
    if imp_w is not None:
        integrand = integrand * imp_w
    return integrand.mean()


def train(cfg, n_epochs=None, n_pts=400, hidden=128, depth=4,
          z_min=0.01, z_max=0.99, lr=1e-3, seed=0, verbose=True,
          holdout_pts=4000, checkpoints=None, importance=False,
          lam_h1=None, lam_h2=None):
    """[V6-5] model selection on a FIXED held-out grid, not the training batch.
       checkpoints: optional sorted list of epoch numbers at which to stash a
       copy of the state dict (used by the convergence study).
       [V6-11] n_epochs defaults to EPOCHS, lam_h1/lam_h2 to LAM_H1/LAM_H2.
       Model selection uses the SAME lambdas as training, so the held-out
       loss is comparable across epochs but NOT across different lambdas."""
    if n_epochs is None:
        n_epochs = EPOCHS
    if lam_h1 is None:
        lam_h1 = LAM_H1
    if lam_h2 is None:
        lam_h2 = LAM_H2
    torch.manual_seed(seed)
    np.random.seed(seed)
    pr = cfg.real_params()
    yr, yi = cfg.yp0()
    model = PINN(hidden, depth, yr, yi).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-6)

    z_hold_np = np.linspace(z_min, z_max, holdout_pts)
    best, best_state = float('inf'), None
    last_hl = float('nan')
    stash = {}
    ck = set(checkpoints or [])
    t_start = time.time()

    for ep in range(n_epochs):
        if importance:
            # draw from p(z) ~ 1/[z(1-z)] on [z_min,z_max] by inverse CDF of
            # the logit; reweight by 1/p to keep the estimator unbiased
            lo, hi = math.log(z_min / (1 - z_min)), math.log(z_max / (1 - z_max))
            t = torch.rand(n_pts, 1, device=device) * (hi - lo) + lo
            z = torch.sigmoid(t)
            imp_w = z * (1.0 - z) * (hi - lo) / (z_max - z_min)
        else:
            z = torch.rand(n_pts, 1, device=device) * (z_max - z_min) + z_min
            imp_w = None
        loss = sobolev_loss(model, pr, z, lam_h1=lam_h1,
                            lam_h2=lam_h2, imp_w=imp_w)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if (ep + 1) % HOLDOUT_EVERY == 0 or ep == 0 or (ep + 1) in ck:
            zh = torch.tensor(z_hold_np.reshape(-1, 1), device=device)
            hl = sobolev_loss(model, pr, zh,
                              lam_h1=lam_h1, lam_h2=lam_h2).item()
            last_hl = hl
            if hl < best:
                best = hl
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            if (ep + 1) in ck:
                stash[ep + 1] = ({k: t.clone() for k, t in model.state_dict().items()}, hl)

        # [V7-1] heartbeat.  Independent of `verbose`, which only controls the
        # longer per-1000-epoch line, so a caller that passes verbose=False
        # still sees that the run is alive.
        if PROGRESS_EVERY and ((ep + 1) % PROGRESS_EVERY == 0 or ep == 0):
            el = time.time() - t_start
            eta = el / (ep + 1) * (n_epochs - ep - 1)
            print(f"        ep {ep+1:6d}/{n_epochs}  batch={loss.item():.3e}  "
                  f"holdout={last_hl:.3e}  best={best:.3e}  "
                  f"[{el:.0f}s elapsed, ETA {eta/60:.1f} min]", flush=True)
        elif verbose and ((ep + 1) % 1000 == 0 or ep == 0):
            print(f"    epoch {ep+1:6d}  batch={loss.item():.4e}  "
                  f"holdout={last_hl:.4e}  best={best:.4e}", flush=True)

    model.load_state_dict(best_state)
    return model, best, stash


# =============================================================================
# PINN evaluators
# =============================================================================
def eval_pinn(model, z):
    zt = torch.tensor(np.asarray(z, dtype=float).reshape(-1, 1), device=device)
    with torch.no_grad():
        u, v = model(zt)
    return u.cpu().numpy().ravel() + 1j * v.cpu().numpy().ravel()


def eval_pinn_d1(model, z):
    zt = torch.tensor(np.asarray(z, dtype=float).reshape(-1, 1),
                      device=device, requires_grad=True)
    u, v = model(zt)
    du = torch.autograd.grad(u.sum(), zt, retain_graph=True)[0]
    dv = torch.autograd.grad(v.sum(), zt)[0]
    y = u.detach().cpu().numpy().ravel() + 1j * v.detach().cpu().numpy().ravel()
    dy = du.detach().cpu().numpy().ravel() + 1j * dv.detach().cpu().numpy().ravel()
    return y, dy


def eval_pinn_dN(model, z, N):
    z = np.atleast_1d(np.asarray(z, dtype=float))
    yv = np.empty(len(z), dtype=complex)
    yN = np.empty(len(z), dtype=complex)
    for i, zk in enumerate(z):
        zt = torch.tensor([[float(zk)]], device=device, requires_grad=True)
        u, v = model(zt)
        yv[i] = complex(u.item(), v.item())
        du, dv = u, v
        for _ in range(N):
            du = torch.autograd.grad(du.sum(), zt, create_graph=True)[0]
            dv = torch.autograd.grad(dv.sum(), zt, create_graph=True)[0]
        yN[i] = complex(du.item(), dv.item())
    return yv, yN


# =============================================================================
# [V6-7] ESTIMATORS AS EXPLICIT LINEAR OPERATORS
#   Each returns (z_points, L, needs_deriv) with (A,B)^T = L @ input_vector.
#   input_vector is y(z_points), or [y; y'] when needs_deriv.
# =============================================================================
Z_DENSE = np.linspace(0.02, 0.99, 400)
Z_DERIV = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
Z_WLS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999])
Z_SING = np.array([0.99, 0.999])
Z_M7 = np.array([0.92, 0.94, 0.96, 0.97, 0.98, 0.985, 0.99])


def op_integral(cfg):
    z = Z_DENSE
    f1, f2 = make_basis(cfg, z)
    w = (z ** -0.3) * ((1 - z) ** -0.4)
    w = w / w.sum() * len(w)
    sw = np.sqrt(w).astype(complex)
    M = np.column_stack([f1 * sw, f2 * sw])
    return z, np.linalg.pinv(M) @ np.diag(sw), False


def op_moments(cfg, n_moments=4):
    z = Z_DENSE
    f1, f2 = make_basis(cfg, z)
    # trapezoid weights (built by hand: np.trapezoid needs numpy>=2)
    tw = np.zeros_like(z)
    tw[1:-1] = (z[2:] - z[:-2]) / 2
    tw[0] = (z[1] - z[0]) / 2
    tw[-1] = (z[-1] - z[-2]) / 2
    W = np.array([tw * z ** n for n in range(n_moments)]).astype(complex)
    M = np.array([[np.sum(tw * z ** n * f1), np.sum(tw * z ** n * f2)]
                  for n in range(n_moments)])
    return z, np.linalg.pinv(M) @ W, False


def op_singular(cfg):
    z = Z_SING
    f1, f2 = make_basis(cfg, z)
    M = np.column_stack([f1, f2])
    return z, np.linalg.pinv(M), False


def op_derivative(cfg):
    z = Z_DERIV
    f1, f2, df1, df2, _, _ = make_basis_with_derivs(cfg, z)
    M = np.vstack([np.column_stack([f1, f2]), np.column_stack([df1, df2])])
    return z, np.linalg.pinv(M), True


def op_weighted(cfg):
    z = Z_WLS
    f1, f2 = make_basis(cfg, z)
    w = 1.0 / np.sqrt(1.0 - z + 1e-6)
    w = w / w.sum() * len(w)
    sw = np.sqrt(w).astype(complex)
    M = np.column_stack([f1 * sw, f2 * sw])
    return z, np.linalg.pinv(M) @ np.diag(sw), False


LINEAR_OPS = {
    "1. Integral residual": op_integral,
    "2. Moment matching": op_moments,
    "3. Singular asymptotic": op_singular,
    "4. Derivative matching": op_derivative,
    "5. Weighted LS": op_weighted,
}


def apply_op(op, y_of, dy_of):
    z, L, needs_d = op
    vec = y_of(z) if not needs_d else np.concatenate([y_of(z), dy_of(z)])
    r = L @ vec
    return complex(r[0]), complex(r[1])


def amplification(cfg, name):
    """A_A = ||L[0,:]|| * ||y|| / |A|  --  no training required."""
    z, L, needs_d = LINEAR_OPS[name](cfg)
    fr = Frobenius(cfg)
    y = fr.y(z)
    vec = y if not needs_d else np.concatenate([y, fr.dy(z)])
    A_t, B_t = gamma_ratio_AB(cfg)
    nv = np.linalg.norm(vec)
    M_cond = np.linalg.cond(np.column_stack(make_basis(cfg, z)))
    return (np.linalg.norm(L[0, :]) * nv / abs(A_t),
            np.linalg.norm(L[1, :]) * nv / abs(B_t),
            M_cond)


# =============================================================================
# LOCAL ESTIMATORS (6a, 6b, 7) -- for PINN and, [V6-4], for the series
# =============================================================================
def local_pointwise(cfg, y_of, yN_of, z_eval=1.0 - 1e-4):
    z = np.array([z_eval])
    B = y_of(z)[0]
    A = ((-1) ** cfg.N_int) / math.factorial(cfg.N_int) * yN_of(z)[0]
    return complex(A), complex(B)


def local_taylor(cfg, y_of, n_pts=15, half_width=0.02, poly_deg=None):
    N = cfg.N_int
    if poly_deg is None:
        poly_deg = N + 3
    zp = np.linspace(1.0 - half_width, 1.0 - 1e-6, n_pts)
    y = y_of(zp)
    s = zp - 1.0
    cre = P.polyfit(s, y.real, poly_deg)
    cim = P.polyfit(s, y.imag, poly_deg)
    B = complex(cre[0], cim[0])
    A = ((-1) ** N) * complex(cre[N], cim[N])
    return A, B


def local_extrapolate(cfg, y_of, yN_of, z_pts=None, poly_deg=2):
    N = cfg.N_int
    if z_pts is None:
        z_pts = Z_M7
    y = y_of(z_pts)
    yN = yN_of(z_pts)
    s = 1.0 - z_pts
    B = complex(P.polyfit(s, y.real, poly_deg)[0], P.polyfit(s, y.imag, poly_deg)[0])
    yN1 = complex(P.polyfit(s, yN.real, poly_deg)[0],
                  P.polyfit(s, yN.imag, poly_deg)[0])
    A = ((-1) ** N) / math.factorial(N) * yN1
    return A, B


def all_estimators(cfg, y_of, dy_of, yN_of):
    """Returns {name: (A, B)} for all eight estimators, given callables."""
    out = {}
    for name, mk in LINEAR_OPS.items():
        out[name] = apply_op(mk(cfg), y_of, dy_of)
    out["6a. Local pointwise"] = local_pointwise(cfg, y_of, yN_of)
    out["6b. Local Taylor"] = local_taylor(cfg, y_of)
    out["7. Local extrapolated"] = local_extrapolate(cfg, y_of, yN_of)
    return out


def frob_callables(cfg):
    fr = Frobenius(cfg)
    return fr.y, fr.dy, fr.yN, fr


def pinn_callables(model, cfg):
    y_of = lambda z: eval_pinn(model, z)
    dy_of = lambda z: eval_pinn_d1(model, z)[1]
    yN_of = lambda z: eval_pinn_dN(model, z, cfg.N_int)[1]
    return y_of, dy_of, yN_of


def errors(cfg, AB):
    A_t, B_t = gamma_ratio_AB(cfg)
    k_t = A_t / B_t
    A, B = AB
    eA = abs(A - A_t) / abs(A_t)
    eB = abs(B - B_t) / abs(B_t)
    k = A / B if abs(B) > 1e-300 else complex('nan')
    ek = abs(k - k_t) / abs(k_t)
    return eA, eB, ek, k


# =============================================================================
# TABLE DRIVERS
# =============================================================================
def table_baseline(configs=None, eps_list=(1e-2, 1e-5)):
    """Table: all eight estimators on the Frobenius series (no network).
    Any error here is a property of the estimator."""
    if configs is None:
        configs = [(2, 2, 2), (1, 1, 1)]
    print("\n" + "=" * 104)
    print("TABLE: exact-input baseline  (Frobenius series, no network)")
    print("  [V6-13] The last column divides the observed error by the estimator's")
    print("  amplification factor.  The Frobenius input carries eps_y ~ 1e-16 (the")
    print("  float64 cast), so Eq.(amp) predicts |dA|/A_A ~ 1e-16 for EVERY linear")
    print("  estimator, however large A_A is.  This tests the amplification formula")
    print("  at the smallest available eps_y, with no training.")
    print("=" * 104)
    for (s, l, m) in configs:
        for eps in eps_list:
            cfg = Config(s, l, m, eps)
            y_of, dy_of, yN_of, fr = frob_callables(cfg)
            print(f"\n  {cfg.label()}")
            print(f"  {'Method':<26}{'|dA|':<14}{'|dB|':<14}{'|dk|':<14}"
                  f"{'|dA|/A_A':<14}")
            print("  " + "-" * 80)
            implied = []
            for name, AB in all_estimators(cfg, y_of, dy_of, yN_of).items():
                eA, eB, ek, _ = errors(cfg, AB)
                if name in LINEAR_OPS:
                    aA, _, _ = amplification(cfg, name)
                    r = eA / aA
                    implied.append(r)
                    col = f"{r:<14.2e}"
                else:
                    col = f"{'--':<14}"
                print(f"  {name:<26}{eA:<14.3e}{eB:<14.3e}{ek:<14.3e}{col}")
            if implied:
                print(f"  {'':<26}{'':<14}{'':<14}{'':<14}"
                      f"[{min(implied):.1e}, {max(implied):.1e}]")


def table_truncation(cfg=None, n_list=(500, 1000, 3000, 8000, 12000)):
    """[V6-12] Certify the series truncation against the closed form, for the
    VALUE and for the N-th derivative.  The derivative is the demanding case:
    the coefficients decay as n^-(1+Nex) but term-by-term differentiation
    boosts them by ~n^N, leaving ~n^-1 near z=1."""
    cfg = cfg or Config(2, 2, 2, 1e-5)
    fr = Frobenius(cfg)
    N = cfg.N_int
    print("\n" + "=" * 74)
    print(f"TRUNCATION CHECK  {cfg.label()}   (compared in mpmath, not float64)")
    print("=" * 74)
    for order, tag in ((0, "value  y(z)"), (N, f"derivative  y^({N})(z)")):
        print(f"\n  --- {tag} ---")
        for z in (0.99, 0.999):
            ref, rows = fr.certify(z, n_list=n_list, order=order)
            print(f"  z = {z}   closed form = {ref:.10g}")
            for n, e in rows:
                flag = "  <-- adequate" if e < 1e-30 else ""
                print(f"    {n:6d} terms : rel err {e:.3e}{flag}")


def table_convention(cfg_seed=(2, 2, 2), eps_list=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5)):
    """[V6-1] integer vs exact exponent, integral-residual estimator on the
    Frobenius series.  No network.  The left block reproduces the spurious
    O(eps) 'conditioning floor' of v5.1."""
    s, l, m = cfg_seed
    print("\n" + "=" * 88)
    print("TABLE: exponent convention  (integral residual on Frobenius series)")
    print(f"  s={s} l={l} m={m}  N={2*l+1}")
    print("=" * 88)
    print(f"  {'eps':<10}{'|dA| int':<14}{'|dB| int':<14}{'|dA| exact':<14}{'|dB| exact':<14}")
    print("  " + "-" * 66)
    for eps in eps_list:
        row = []
        for exact in (False, True):
            cfg = Config(s, l, m, eps, exact_exponent=exact)
            fr = Frobenius(cfg)
            AB = apply_op(op_integral(cfg), fr.y, fr.dy)
            eA, eB, _, _ = errors(cfg, AB)
            row += [eA, eB]
        print(f"  {eps:<10.0e}{row[0]:<14.3e}{row[1]:<14.3e}{row[2]:<14.3e}{row[3]:<14.3e}")


def table_amplification(configs=((2, 2, 2), (1, 1, 1)), eps=1e-3):
    """[V6-7] amplification factors -- no training."""
    print("\n" + "=" * 88)
    print("TABLE: amplification factors  A_A = ||L[0,:]|| ||y|| / |A|")
    print("=" * 88)
    for (s, l, m) in configs:
        cfg = Config(s, l, m, eps)
        print(f"\n  {cfg.label()}")
        print(f"  {'Estimator':<26}{'A_A':<14}{'A_B':<14}{'cond M':<14}")
        print("  " + "-" * 66)
        for name in LINEAR_OPS:
            aA, aB, cM = amplification(cfg, name)
            print(f"  {name:<26}{aA:<14.3e}{aB:<14.3e}{cM:<14.3e}")


def table_contamination(cfg_seed=(2, 2, 2), eps_list=(1e-2, 1e-4, 1e-5, 1e-8)):
    """What the local estimators actually return: A_eff = A + B c_N.
    At eps=0 this is the N-th Taylor coefficient of a degree-(l+s) polynomial,
    hence identically zero.  c_N is FINITE and tends to -k_ell."""
    s, l, m = cfg_seed
    N = 2 * l + 1
    print("\n" + "=" * 96)
    print("TABLE: local-estimator contamination")
    print(f"  s={s} l={l} m={m}  N={N}   ytilde is a degree-{l+s} polynomial at eps=0,")
    print(f"  and {l+s} < {N}, so the (1-z)^N Taylor coefficient vanishes identically.")
    print("=" * 96)
    print(f"  {'eps':<10}{'|A|':<14}{'|B c_N|':<14}{'|c_N|':<14}"
          f"{'|A_eff|/|A|':<14}{'|c_N + k|':<14}")
    print("  " + "-" * 80)
    for eps in eps_list:
        cfg = Config(s, l, m, eps)
        A, B = gamma_ratio_AB(cfg)
        cN = c_coefficient(cfg, N)
        Aeff = A + B * cN
        k = A / B
        print(f"  {eps:<10.0e}{abs(A):<14.6f}{abs(B*cN):<14.6f}{abs(cN):<14.6f}"
              f"{abs(Aeff)/abs(A):<14.3e}{abs(cN + k):<14.3e}")
    # the (1-z)^(N_ex-N) factor that explains the 6a excess at large eps
    print("\n  6a evaluates at z = 1-1e-4, where f1 carries (1-z)^(N_ex-N) = (1e-4)^(2eps):")
    for eps in eps_list:
        print(f"    eps={eps:.0e}:  (1e-4)^(2eps) = {(1e-4)**(2*eps):.4f}")


# =============================================================================
# [V6-15] AMPLIFICATION MAP OVER (s, ell, m)   --  NO TRAINING
# =============================================================================
def parameters_supported(cfg):
    """False when c = 1-s+2i m gamma is a non-positive integer.

    That happens exactly when m*gamma = 0 and s >= 1.  The regulator does NOT
    lift it, because ell does not appear in c.  In that case:
      * Gamma(c) is on a pole, so A_dec and B_grow do not exist separately
        (their ratio k_ell does, since Gamma(c) cancels);
      * the horizon exponents 0 and 1-c = s differ by an integer, so z=0 is
        degenerate too;
      * the Frobenius recursion divides by (n + ct) = 0 at n = s-1.
    Schwarzschild and axisymmetric Kerr perturbations with s >= 1 therefore
    need a regulator acting on s or on c, not on ell.  s = 0, m = 0 is fine
    (c = 1)."""
    cr = complex(cfg.c_mp)
    if abs(cr.imag) > 1e-30:
        return True
    return not (cr.real <= 0 and abs(cr.real - round(cr.real)) < 1e-30)


COND_LIMIT = 1.0 / np.finfo(float).eps      # ~4.5e15


def table_ampmap(s_list=(2,), ell_list=(2, 3, 4, 5), m_list=None, eps=1e-3,
                 estimators=("1. Integral residual", "2. Moment matching",
                             "5. Weighted LS"),
                 eps_y=1e-5, include_two_point=False):
    """[V6-15] Amplification factor across angular numbers.  Requires no
    training, so the usable range of (s, ell, m) can be mapped in advance.

    Reports |A_dec|, k_ell, cond M and A_A for each estimator, and the
    predicted observable error eps_A = A_A * eps_y for a solver of the given
    quality.  Rows with c a non-positive integer are skipped with a note.

    The two-point estimator (Method 3) is off by default: its design matrix
    has cond M > 1e16 for N > 7, so in double precision the pseudo-inverse
    discards the singular direction carrying A_dec and the computed A_A is an
    artifact.  Set include_two_point=True to see it, with a warning flag."""
    ests = list(estimators) + (["3. Singular asymptotic"]
                               if include_two_point else [])
    print("\n" + "=" * (52 + 14 * len(ests)))
    print(f"TABLE: amplification map over (s, ell, m)   eps={eps:.0e}")
    print(f"  predicted eps_A = A_A * eps_y with eps_y = {eps_y:.0e}")
    print("=" * (52 + 14 * len(ests)))
    hdr = (f"  {'s':<3}{'l':<3}{'m':<3}{'N':<4}{'|A_dec|':<11}"
           f"{'Im k':<12}{'cond M':<11}")
    for e in ests:
        hdr += f"{'A_A ' + e.split('.')[0]:<13}"
    hdr += f"{'A_B max':<10}{'eps_A (best)':<14}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    out = []
    for s in s_list:
        for l in ell_list:
            if l < abs(s):
                continue
            ms = range(0, l + 1) if m_list is None else m_list
            for m in ms:
                if abs(m) > l:
                    continue
                cfg = Config(s, l, m, eps)
                if not parameters_supported(cfg):
                    print(f"  {s:<3}{l:<3}{m:<3}{cfg.N_int:<4}"
                          f"  -- skipped: c = {complex(cfg.c_mp).real:.0f} is a "
                          f"non-positive integer (m*gamma = 0, s >= 1)")
                    continue
                A, B = gamma_ratio_AB(cfg)
                k = A / B
                aa, bb, cm = [], [], None
                for e in ests:
                    a, b, c_ = amplification(cfg, e)
                    aa.append(a); bb.append(b)
                    if cm is None:
                        cm = c_
                best = min(aa)
                flag = "  (*)" if cm > COND_LIMIT else ""
                line = (f"  {s:<3}{l:<3}{m:<3}{cfg.N_int:<4}{abs(A):<11.5f}"
                        f"{k.imag:<12.3e}{cm:<11.3e}")
                for a in aa:
                    line += f"{a:<13.3e}"
                line += f"{max(bb):<10.2f}{best * eps_y:<14.2e}{flag}"
                print(line)
                out.append(dict(s=s, l=l, m=m, N=cfg.N_int, absA=abs(A),
                                k=k, condM=cm,
                                A_A={e: a for e, a in zip(ests, aa)},
                                A_B={e: b for e, b in zip(ests, bb)},
                                eps_A_best=best * eps_y))
    if any(r['condM'] > COND_LIMIT for r in out):
        print("  (*) cond M exceeds double precision; A_A unreliable on that row.")
    print("  Rows with eps_A > 1 are not usable at the stated solver accuracy.")
    allB = [b for r in out for b in r['A_B'].values()]
    if allB:
        print(f"  A_B over the whole map: {min(allB):.2f} to {max(allB):.2f} "
              f"-- only A_dec is amplified.")

    # growth summaries
    by_m = {}
    for r in out:
        by_m.setdefault((r['s'], r['m']), []).append(r)
    print("\n  growth of A_A with ell at fixed (s,m), first estimator:")
    e0 = ests[0]
    for key, rs in sorted(by_m.items()):
        rs = sorted(rs, key=lambda r: r['l'])
        if len(rs) < 2:
            continue
        fac = [rs[i]['A_A'][e0] / rs[i - 1]['A_A'][e0] for i in range(1, len(rs))]
        print(f"    s={key[0]} m={key[1]}: factors per unit ell = "
              + ", ".join(f"{f:.1f}" for f in fac))
    by_l = {}
    for r in out:
        by_l.setdefault((r['s'], r['l']), []).append(r)
    print("  dependence on |m| at fixed (s,ell), first estimator:")
    for key, rs in sorted(by_l.items()):
        rs = sorted(rs, key=lambda r: r['m'])
        if len(rs) < 2:
            continue
        print(f"    s={key[0]} l={key[1]}: "
              + ", ".join(f"m={r['m']}:{r['A_A'][e0]:.2e}" for r in rs))
    return out


def table_scan(cfg_seed=(2, 2, 2), eps_list=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
               seeds=(0, 1, 2), n_epochs=None, report="1. Integral residual",
               checkpoint="lovescan.json", resume=True):
    """[V6-6/V7-2] PINN scan over eps -> the Love-number table.

    Reports ONE fixed estimator rather than the per-row best, which would be a
    selection effect.  Every run is checkpointed to JSON and skipped on
    restart, so a disconnect costs one run.

    The Re k_ell column should be read as a bound, not a measurement: at small
    eps the analytical value falls below A_A * eps_y, so the network cannot
    resolve it.  Im k_ell is far above the noise and is a measurement."""
    import json
    if n_epochs is None:
        n_epochs = EPOCHS
    s_, l_, m_ = cfg_seed
    cache = []
    if checkpoint and resume and os.path.exists(checkpoint):
        cache = json.load(open(checkpoint))
        print(f"    resuming from {checkpoint}: {len(cache)} run(s) on disk",
              flush=True)

    def lookup(e, sd):
        for r in cache:
            if r["eps"] == e and r["seed"] == sd and r["epochs"] == n_epochs:
                return r
        return None

    print("\n" + "=" * 104)
    print(f"TABLE: PINN scan over eps   s={s_} l={l_} m={m_}   "
          f"estimator = {report}")
    print(f"  {len(seeds)} seeds, {n_epochs} epochs"
          + (f", checkpoint {checkpoint}" if checkpoint else ""))
    print("=" * 104)
    print(f"  {'eps':<9}{'eps_y':<12}{'eps_A':<12}{'Re k PINN':<14}"
          f"{'Im k PINN':<14}{'Re k an':<14}{'Im k an':<14}")
    print("  " + "-" * 92, flush=True)

    out = {}
    for eps in eps_list:
        cfg = Config(s_, l_, m_, eps)
        op = LINEAR_OPS[report](cfg)
        fr = Frobenius(cfg)
        z_ref = np.linspace(0.02, 0.99, 400)
        y_ref = fr.y(z_ref)
        nrm = np.linalg.norm(y_ref)
        A_t, B_t = gamma_ratio_AB(cfg)
        kt = A_t / B_t
        runs = []
        for sd in seeds:
            hit = lookup(eps, sd)
            if hit is not None:
                runs.append(hit)
                print(f"      eps={eps:.0e} seed {sd}: cached", flush=True)
                continue
            print(f"      eps={eps:.0e} seed {sd}: training "
                  f"{n_epochs} epochs ...", flush=True)
            t0 = time.time()
            model, hl, _ = train(cfg, n_epochs=n_epochs, seed=sd, verbose=False)
            eA, eB, ek, k = errors(cfg, apply_op(op, y_of_pinn(model),
                                                 dy_of_pinn(model)))
            ey = np.linalg.norm(eval_pinn(model, z_ref) - y_ref) / nrm
            rec = dict(eps=eps, seed=sd, epochs=n_epochs, t=time.time() - t0,
                       loss=hl, eps_y=ey, eps_A=eA, eps_B=eB, eps_k=ek,
                       re_k=k.real, im_k=k.imag,
                       s=s_, l=l_, m=m_, estimator=report)
            if checkpoint:
                save_rows(rec, checkpoint)
                cache.append(rec)
            runs.append(rec)
            print(f"      eps={eps:.0e} seed {sd}: {rec['t']:.0f}s  "
                  f"eps_y={ey:.3e}  eps_A={eA:.3e}  "
                  f"k={k.real:+.3e}{k.imag:+.3e}j", flush=True)
        med = lambda key: float(np.median([r[key] for r in runs]))
        print(f"  {eps:<9.0e}{med('eps_y'):<12.3e}{med('eps_A'):<12.3e}"
              f"{med('re_k'):<14.3e}{med('im_k'):<14.4e}"
              f"{kt.real:<14.4e}{kt.imag:<14.5e}", flush=True)
        out[eps] = runs
    return out


def y_of_pinn(model):
    return lambda z: eval_pinn(model, z)


def dy_of_pinn(model):
    return lambda z: eval_pinn_d1(model, z)[1]


def save_rows(rows, path):
    """Append records to a json file so a grid survives a disconnect."""
    import json
    old = []
    if os.path.exists(path):
        old = json.load(open(path))
    old.extend(rows if isinstance(rows, list) else [rows])
    json.dump(old, open(path, "w"), indent=1)
    return path


def lms_row(s, l, m, eps=1e-3, seeds=(0, 1, 2), n_epochs=None,
            report="1. Integral residual", verbose=True,
            checkpoint="lms.json", resume=True, cache=None):
    """[V7-5] One row of the (s,l,m) table, checkpointed per run.

    Alongside the measured errors it records the PREDICTION
        eps_A(pred) = A_A * eps_y
    with A_A from Eq.(amp) and eps_y measured on the same run.  The ratio of
    measured to predicted is the quantity of interest: if it stays of order
    one across the grid, the training-free map of Section on the amplification
    fixes the usable range of (s,l,m) without further training."""
    import json
    if n_epochs is None:
        n_epochs = EPOCHS
    if cache is None:
        cache = []
        if checkpoint and resume and os.path.exists(checkpoint):
            cache = json.load(open(checkpoint))
    cfg = Config(s, l, m, eps)
    if not parameters_supported(cfg):
        if verbose:
            print(f"  {s:<3}{l:<3}{m:<3}{cfg.N_int:<4}  -- skipped: "
                  f"c = {complex(cfg.c_mp).real:.0f} is a non-positive integer")
        return None
    A_t, B_t = gamma_ratio_AB(cfg)
    kt = A_t / B_t
    aA, aB, _ = amplification(cfg, report)
    op = LINEAR_OPS[report](cfg)
    fr = Frobenius(cfg)
    z_ref = np.linspace(0.02, 0.99, 400)
    y_ref = fr.y(z_ref)
    nrm = np.linalg.norm(y_ref)

    runs = []
    for sd in seeds:
        hit = None
        for r in cache:
            if (r["s"], r["l"], r["m"], r["eps"], r["seed"],
                    r["epochs"]) == (s, l, m, eps, sd, n_epochs):
                hit = r
                break
        if hit is not None:
            runs.append(hit)
            if verbose:
                print(f"      (s,l,m)=({s},{l},{m}) seed {sd}: cached",
                      flush=True)
            continue
        if verbose:
            print(f"      (s,l,m)=({s},{l},{m}) seed {sd}: training "
                  f"{n_epochs} epochs ...", flush=True)
        t0 = time.time()
        model, hl, _ = train(cfg, n_epochs=n_epochs, seed=sd, verbose=False)
        eA, eB, ek, k = errors(cfg, apply_op(op, y_of_pinn(model),
                                             dy_of_pinn(model)))
        ey = np.linalg.norm(eval_pinn(model, z_ref) - y_ref) / nrm
        rec = dict(s=s, l=l, m=m, N=cfg.N_int, eps=eps, seed=sd,
                   epochs=n_epochs, t=time.time() - t0, loss=hl,
                   eps_y=ey, eps_A=eA, eps_B=eB, eps_k=ek,
                   A_A=aA, A_B=aB, eps_A_pred=aA * ey,
                   re_k=k.real, im_k=k.imag,
                   re_k_an=kt.real, im_k_an=kt.imag, estimator=report)
        if checkpoint:
            save_rows(rec, checkpoint)
            cache.append(rec)
        runs.append(rec)
        if verbose:
            print(f"      (s,l,m)=({s},{l},{m}) seed {sd}: {rec['t']:.0f}s  "
                  f"eps_y={ey:.3e}  eps_A={eA:.3e}  "
                  f"pred={aA*ey:.3e}", flush=True)

    med = lambda key: float(np.median([r[key] for r in runs]))
    out = dict(s=s, l=l, m=m, N=cfg.N_int, eps=eps, A_A=aA,
               eps_y=med('eps_y'), eps_A=med('eps_A'),
               eps_A_pred=med('eps_A_pred'),
               ratio=med('eps_A') / max(med('eps_A_pred'), 1e-300),
               re_k=med('re_k'), im_k=med('im_k'),
               re_k_an=kt.real, im_k_an=kt.imag,
               dA_min=float(min(r['eps_A'] for r in runs)),
               dA_max=float(max(r['eps_A'] for r in runs)))
    if verbose:
        print(f"  {s:<3}{l:<3}{m:<3}{out['N']:<4}{aA:<11.3e}"
              f"{out['eps_y']:<11.3e}{out['eps_A_pred']:<12.3e}"
              f"{out['eps_A']:<12.3e}{out['ratio']:<9.2f}"
              f"{out['im_k']:<13.4e}{kt.imag:<13.4e}", flush=True)
    return out


def table_lms(rows=None, eps=1e-3, seeds=(0, 1, 2), n_epochs=None,
              report="1. Integral residual", checkpoint="lms.json"):
    """[V7-5] Network results across angular numbers, against the prediction.

    Default rows span the predicted usable range rather than covering a grid:
    an easy corner, a middle one and a hard one.  Three rows chosen to test
    the prediction are worth more than fourteen chosen for coverage, because
    the prediction is what the paper claims."""
    import json
    if rows is None:
        rows = [(2, 2, 2), (2, 3, 1), (2, 4, 1)]
    if n_epochs is None:
        n_epochs = EPOCHS
    cache = []
    if checkpoint and os.path.exists(checkpoint):
        cache = json.load(open(checkpoint))
        print(f"    resuming from {checkpoint}: {len(cache)} run(s)",
              flush=True)
    print("\n" + "=" * 104)
    print(f"TABLE: network results across (s,l,m) at eps={eps:.0e}, "
          f"{len(seeds)} seeds, {n_epochs} epochs, estimator = {report}")
    print("=" * 104)
    print(f"  {'s':<3}{'l':<3}{'m':<3}{'N':<4}{'A_A':<11}{'eps_y':<11}"
          f"{'eps_A pred':<12}{'eps_A meas':<12}{'ratio':<9}"
          f"{'Im k PINN':<13}{'Im k an':<13}")
    print("  " + "-" * 100, flush=True)
    recs = []
    for (s_, l_, m_) in rows:
        r = lms_row(s_, l_, m_, eps=eps, seeds=seeds, n_epochs=n_epochs,
                    report=report, checkpoint=checkpoint, cache=cache)
        if r:
            recs.append(r)
    if recs:
        rr = [r['ratio'] for r in recs]
        print("  " + "-" * 100)
        print(f"  measured / predicted eps_A: {min(rr):.2f} to {max(rr):.2f}")
        print("  A ratio of order one across the grid means the training-free")
        print("  amplification map fixes the usable range without training.")
    return recs


def table_sobolev(cfg_seed=(2, 2, 2), eps=1e-3, seeds=(0, 1, 2),
                  n_epochs=None, report="1. Integral residual",
                  variants=None, equal_walltime=True,
                  checkpoint="sobolev.json", resume=True):
    """[V7-2] Does the Sobolev loss help?

    Trains the same architecture at the same configuration with different
    (lam_h1, lam_h2) and compares BOTH links of the chain: the solution error
    eps_y and the coefficient error eps_A.  The two need not improve together,
    which is the point of the paper.

    Every completed run is appended to `checkpoint` (JSON) immediately, and on
    restart any run already present is skipped.  A disconnect therefore costs
    one run, not the table.  Pass checkpoint=None to disable, resume=False to
    recompute.

    Two caveats handled here:
      * the held-out loss is NOT comparable across variants, since each is the
        value of a different objective.  Compare eps_y and eps_A.
      * the H2 loss needs four autograd passes per step against two for the
        plain L2, so equal epochs is not equal cost.  With
        equal_walltime=True the cheaper variants are re-run with the epoch
        count scaled by the measured time ratio.
    """
    import json
    if n_epochs is None:
        n_epochs = EPOCHS
    if variants is None:
        variants = [("H2  (%.3g, %.3g)" % (LAM_H1, LAM_H2), LAM_H1, LAM_H2),
                    ("H1  (%.3g, 0)" % LAM_H1,              LAM_H1, 0.0),
                    ("L2  (0, 0)",                          0.0,    0.0)]
    s_, l_, m_ = cfg_seed
    cfg = Config(s_, l_, m_, eps)
    fr = Frobenius(cfg)
    z_ref = np.linspace(0.02, 0.99, 400)
    y_ref = fr.y(z_ref)
    nrm = np.linalg.norm(y_ref)
    op = LINEAR_OPS[report](cfg)

    cache = []
    if checkpoint and resume and os.path.exists(checkpoint):
        cache = json.load(open(checkpoint))
        print(f"    resuming from {checkpoint}: {len(cache)} run(s) on disk",
              flush=True)

    def lookup(tag, nep, sd):
        for r in cache:
            if r["tag"] == tag and r["epochs"] == nep and r["seed"] == sd:
                return r
        return None

    n_runs_total = len(seeds) * (len(variants) +
                                 (len(variants) - 1 if equal_walltime else 0))
    state = dict(done=0, t0=time.time())

    def one(tag, lam1, lam2, nep, sd):
        hit = lookup(tag, nep, sd)
        if hit is not None:
            state['done'] += 1
            print(f"      {tag} seed {sd}: cached "
                  f"[{state['done']}/{n_runs_total}]", flush=True)
            return hit
        print(f"      {tag} seed {sd}: training {nep} epochs ...", flush=True)
        t0 = time.time()
        model, hl, _ = train(cfg, n_epochs=nep, seed=sd, verbose=False,
                             lam_h1=lam1, lam_h2=lam2)
        dt = time.time() - t0
        y_of, dy_of, _ = pinn_callables(model, cfg)
        eA, eB, ek, k = errors(cfg, apply_op(op, y_of, dy_of))
        ey = np.linalg.norm(eval_pinn(model, z_ref) - y_ref) / nrm
        rec = dict(tag=tag, epochs=nep, seed=sd, t=dt, loss=hl,
                   ey=ey, eA=eA, eB=eB, ek=ek,
                   re_k=k.real, im_k=k.imag,
                   s=s_, l=l_, m=m_, eps=eps, estimator=report)
        if checkpoint:
            save_rows(rec, checkpoint)
            cache.append(rec)
        state['done'] += 1
        elapsed = time.time() - state['t0']
        eta = elapsed / max(state['done'], 1) * (n_runs_total - state['done'])
        print(f"      {tag} seed {sd}: {dt:6.0f}s  loss={hl:.3e}  "
              f"eps_y={ey:.3e}  eps_A={eA:.3e}   "
              f"[{state['done']}/{n_runs_total}, ETA {eta/60:.0f} min]",
              flush=True)
        return rec

    print("\n" + "=" * 96)
    print(f"TABLE: does the Sobolev loss help?   {cfg.label()}")
    print(f"  estimator = {report},  {len(seeds)} seeds,  {n_epochs} epochs")
    if checkpoint:
        print(f"  checkpointing every run to {checkpoint}")
    print("=" * 96)
    hdr = (f"  {'variant':<18}{'epochs':<9}{'s/run':<9}{'loss*':<12}"
           f"{'eps_y':<12}{'eps_A':<12}{'eps_k':<12}")
    print(hdr)
    print("  " + "-" * 82, flush=True)

    def row(name, nep, runs, note=""):
        med = lambda key: float(np.median([r[key] for r in runs]))
        print(f"  {name:<18}{nep:<9}{med('t'):<9.0f}{med('loss'):<12.3e}"
              f"{med('ey'):<12.3e}{med('eA'):<12.3e}{med('ek'):<12.3e}{note}",
              flush=True)

    out = {}
    for name, lam1, lam2 in variants:
        runs = [one(name, lam1, lam2, n_epochs, sd) for sd in seeds]
        out[name] = dict(equal_epochs=runs)
        row(name, n_epochs, runs)

    if equal_walltime and len(variants) > 1:
        t_ref = float(np.median([r['t'] for r in
                                 out[variants[0][0]]['equal_epochs']]))
        print("  " + "-" * 82)
        print(f"  equal wall-clock (budget matched to '{variants[0][0]}'):",
              flush=True)
        for name, lam1, lam2 in variants[1:]:
            t_here = float(np.median([r['t'] for r in
                                      out[name]['equal_epochs']]))
            nep = max(1, int(round(n_epochs * t_ref / max(t_here, 1e-9))))
            runs = [one(name, lam1, lam2, nep, sd) for sd in seeds]
            out[name]['equal_time'] = runs
            row(name, nep, runs)

    print("  " + "-" * 82)
    print("  * loss values are NOT comparable across variants: each is the")
    print("    value of a different objective.  Compare eps_y and eps_A.")
    aA, aB, _ = amplification(cfg, report)
    print(f"  predicted amplification for this estimator: A_A = {aA:.3e}")
    print(f"  so eps_A ~ {aA:.3g} * eps_y is expected in every row.")
    return out


def study_convergence(cfg_seed=(2, 2, 2), eps=1e-3, seeds=(0, 1, 2),
                      n_epochs=None, report=("1. Integral residual",
                                              "5. Weighted LS",
                                              "3. Singular asymptotic")):
    """[V6-7] test eps_A ~ A_A * eps_y.  Cheap trick: checkpoint ONE long run
    at logarithmically spaced epochs instead of retraining a grid."""
    s, l, m = cfg_seed
    cfg = Config(s, l, m, eps)
    fr = Frobenius(cfg)
    z_ref = np.linspace(0.02, 0.99, 400)
    y_ref = fr.y(z_ref)
    if n_epochs is None:
        n_epochs = 2 * EPOCHS
    cks = sorted(set(int(x) for x in np.logspace(np.log10(500), np.log10(n_epochs), 12)))
    print("\n" + "=" * 96)
    print(f"CONVERGENCE STUDY  {cfg.label()}  checkpoints={cks}")
    for nm in report:
        aA, aB, _ = amplification(cfg, nm)
        print(f"  predicted A_A[{nm}] = {aA:.3e}")
    print("=" * 96)
    print(f"  {'seed':<6}{'epoch':<9}{'loss':<12}{'eps_y':<12}" +
          "".join(f"{nm.split('.')[0]:>14}" for nm in report))
    pts = []
    for sd in seeds:
        _, _, stash = train(cfg, n_epochs=n_epochs, seed=sd, verbose=False,
                            checkpoints=cks)
        for ep in cks:
            if ep not in stash:
                continue
            state, hl = stash[ep]
            mdl = PINN(128, 4, *cfg.yp0()).to(device)
            mdl.load_state_dict(state)
            y_of, dy_of, yN_of = pinn_callables(mdl, cfg)
            ey = np.linalg.norm(eval_pinn(mdl, z_ref) - y_ref) / np.linalg.norm(y_ref)
            eAs = []
            for nm in report:
                eA, _, _, _ = errors(cfg, apply_op(LINEAR_OPS[nm](cfg), y_of, dy_of))
                eAs.append(eA)
            pts.append((sd, ep, hl, ey, eAs))
            print(f"  {sd:<6}{ep:<9}{hl:<12.3e}{ey:<12.3e}" +
                  "".join(f"{e:>14.3e}" for e in eAs))
    # slope-1 check
    ey_all = np.array([p[3] for p in pts])
    fig, ax = plt.subplots(figsize=(7, 5))
    for j, nm in enumerate(report):
        eA_all = np.array([p[4][j] for p in pts])
        ax.loglog(ey_all, eA_all, 'o', label=nm)
        sl, ic = np.polyfit(np.log10(ey_all), np.log10(eA_all), 1)
        print(f"  fit {nm}: slope={sl:.3f}  intercept(A_A)={10**ic:.3e}")
    ax.set_xlabel(r'$\epsilon_y$'); ax.set_ylabel(r'$\epsilon_A$')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
    fig.savefig(os.path.join(OUTDIR, 'convergence.png'), dpi=130, bbox_inches='tight')
    if not HEADLESS:
        plt.show()
    return pts


def study_zmax(cfg_seed=(2, 2, 2), eps=1e-3, zmaxes=(0.95, 0.99, 0.995),
               seeds=(0, 1), n_epochs=None):
    s, l, m = cfg_seed
    cfg = Config(s, l, m, eps)
    print("\n" + "=" * 70)
    print(f"z_max SENSITIVITY  {cfg.label()}")
    print("=" * 70)
    for zm in zmaxes:
        eAs, hls = [], []
        for sd in seeds:
            model, hl, _ = train(cfg, n_epochs=n_epochs, seed=sd,
                                 z_max=zm, verbose=False)
            y_of, dy_of, _ = pinn_callables(model, cfg)
            eA, _, _, _ = errors(cfg, apply_op(op_integral(cfg), y_of, dy_of))
            eAs.append(eA); hls.append(hl)
        print(f"  z_max={zm:<8}loss={np.median(hls):.3e}  |dA|={np.median(eAs):.3e}")


def study_sampling(cfg_seed=(2, 2, 2), eps=1e-3, seeds=(0, 1), n_epochs=None):
    s, l, m = cfg_seed
    cfg = Config(s, l, m, eps)
    print("\n" + "=" * 70)
    print(f"UNIFORM vs IMPORTANCE SAMPLING  {cfg.label()}")
    print("=" * 70)
    for imp in (False, True):
        eAs, hls = [], []
        for sd in seeds:
            model, hl, _ = train(cfg, n_epochs=n_epochs, seed=sd,
                                 importance=imp, verbose=False)
            y_of, dy_of, _ = pinn_callables(model, cfg)
            eA, _, _, _ = errors(cfg, apply_op(op_integral(cfg), y_of, dy_of))
            eAs.append(eA); hls.append(hl)
        tag = "importance" if imp else "uniform"
        print(f"  {tag:<12}loss={np.median(hls):.3e}  |dA|={np.median(eAs):.3e}")


def plot_profiles(cfg_seed=(2, 2, 2), eps=1e-5, seed=0, n_epochs=None):
    s, l, m = cfg_seed
    cfg = Config(s, l, m, eps)
    model, hl, _ = train(cfg, n_epochs=n_epochs, seed=seed)
    fr = Frobenius(cfg)
    z = np.linspace(0.01, 0.99, 300)
    y_fr = fr.series(z, 0)
    dy_fr = fr.series(z, 1, n_terms=N_TERMS_VAL)
    d2y_fr = fr.series(z, 2, n_terms=N_TERMS_VAL)
    zt = torch.tensor(z.reshape(-1, 1), device=device, requires_grad=True)
    u, v = model(zt)
    g = lambda f: torch.autograd.grad(f.sum(), zt, create_graph=True)[0]
    du, dv = g(u), g(v)
    d2u, d2v = g(du), g(dv)
    nz = lambda t: t.detach().cpu().numpy().ravel()
    fig, ax = plt.subplots(3, 2, figsize=(13, 11))
    data = [(nz(u), y_fr.real, 'Re y'), (nz(v), y_fr.imag, 'Im y'),
            (nz(du), dy_fr.real, "Re y'"), (nz(dv), dy_fr.imag, "Im y'"),
            (nz(d2u), d2y_fr.real, "Re y''"), (nz(d2v), d2y_fr.imag, "Im y''")]
    for k, (pn, ex, ttl) in enumerate(data):
        a = ax[k // 2, k % 2]
        a.plot(z, ex, 'k-', lw=2, label='Frobenius')
        a.plot(z, pn, 'r--', lw=1.4, label='PINN')
        a.set_title(ttl); a.set_xlabel('z'); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle(cfg.label(), fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f'profiles_eps{eps:.0e}.png'),
                dpi=130, bbox_inches='tight')
    if not HEADLESS:
        plt.show()


# =============================================================================
# [V7-6] SIDE BY SIDE: all eight estimators, NETWORK vs FROBENIUS input
# =============================================================================
def table_sidebyside(cfg_seed=(2, 2, 2), eps=1e-3, seeds=(0, 1, 2),
                     n_epochs=None, checkpoint="sidebyside.json",
                     resume=True):
    """[V7-6] The main diagnostic of the paper.

    The SAME eight estimators are driven by two inputs: the trained network
    and the exact Frobenius series.  The Frobenius column is a property of the
    estimator alone and no solver can improve it; the difference between the
    columns is the solver's contribution.  For the five linear estimators the
    prediction A_A * eps_y is shown alongside, so the table tests
    Eq.(amp) estimator by estimator at the network's own eps_y rather than at
    machine precision.

    Cost is one training run per seed: the eight estimators are then evaluated
    on the same model, which is cheap.  Runs are checkpointed."""
    import json
    if n_epochs is None:
        n_epochs = EPOCHS
    s_, l_, m_ = cfg_seed
    cfg = Config(s_, l_, m_, eps)
    fr = Frobenius(cfg)
    z_ref = np.linspace(0.02, 0.99, 400)
    y_ref = fr.y(z_ref)
    nrm = np.linalg.norm(y_ref)

    cache = []
    if checkpoint and resume and os.path.exists(checkpoint):
        cache = json.load(open(checkpoint))
        print(f"    resuming from {checkpoint}: {len(cache)} run(s)",
              flush=True)

    def lookup(sd):
        for r in cache:
            if (r["s"], r["l"], r["m"], r["eps"], r["seed"],
                    r["epochs"]) == (s_, l_, m_, eps, sd, n_epochs):
                return r
        return None

    # --- Frobenius column: no training, one evaluation -----------------------
    fy, fdy, fyN, _ = frob_callables(cfg)
    frob = {}
    for name, AB in all_estimators(cfg, fy, fdy, fyN).items():
        eA, eB, ek, _ = errors(cfg, AB)
        frob[name] = dict(eA=eA, eB=eB, ek=ek)

    # --- network column: one model per seed, all eight estimators on it ------
    runs = []
    for sd in seeds:
        hit = lookup(sd)
        if hit is not None:
            runs.append(hit)
            print(f"      seed {sd}: cached", flush=True)
            continue
        print(f"      seed {sd}: training {n_epochs} epochs ...", flush=True)
        t0 = time.time()
        model, hl, _ = train(cfg, n_epochs=n_epochs, seed=sd, verbose=False)
        y_of, dy_of, yN_of = pinn_callables(model, cfg)
        ey = np.linalg.norm(eval_pinn(model, z_ref) - y_ref) / nrm
        per = {}
        for name, AB in all_estimators(cfg, y_of, dy_of, yN_of).items():
            eA, eB, ek, k = errors(cfg, AB)
            per[name] = dict(eA=eA, eB=eB, ek=ek, re_k=k.real, im_k=k.imag)
        rec = dict(s=s_, l=l_, m=m_, eps=eps, seed=sd, epochs=n_epochs,
                   t=time.time() - t0, loss=hl, eps_y=ey, per=per)
        if checkpoint:
            save_rows(rec, checkpoint)
            cache.append(rec)
        runs.append(rec)
        print(f"      seed {sd}: {rec['t']:.0f}s  loss={hl:.3e}  "
              f"eps_y={ey:.3e}", flush=True)

    ey_med = float(np.median([r["eps_y"] for r in runs]))

    print("\n" + "=" * 108)
    print(f"TABLE: same estimators, NETWORK vs FROBENIUS input   {cfg.label()}")
    print(f"  {len(seeds)} seeds, {n_epochs} epochs, median eps_y = {ey_med:.3e}")
    print("=" * 108)
    print(f"  {'Method':<24}{'PINN |dA|':<13}{'Frob |dA|':<13}"
          f"{'A_A':<12}{'pred A_A*eps_y':<16}{'meas/pred':<11}"
          f"{'PINN |dB|':<13}")
    print("  " + "-" * 104, flush=True)

    out = {}
    for name in list(LINEAR_OPS) + ["6a. Local pointwise", "6b. Local Taylor",
                                    "7. Local extrapolated"]:
        pA = float(np.median([r["per"][name]["eA"] for r in runs]))
        pB = float(np.median([r["per"][name]["eB"] for r in runs]))
        fA = frob[name]["eA"]
        if name in LINEAR_OPS:
            aA, _, _ = amplification(cfg, name)
            pred = aA * ey_med
            print(f"  {name:<24}{pA:<13.3e}{fA:<13.3e}{aA:<12.3e}"
                  f"{pred:<16.3e}{pA/max(pred,1e-300):<11.2f}{pB:<13.3e}")
            out[name] = dict(pinn_eA=pA, frob_eA=fA, A_A=aA, pred=pred,
                             ratio=pA / max(pred, 1e-300), pinn_eB=pB)
        else:
            print(f"  {name:<24}{pA:<13.3e}{fA:<13.3e}{'--':<12}"
                  f"{'--':<16}{'--':<11}{pB:<13.3e}")
            out[name] = dict(pinn_eA=pA, frob_eA=fA, pinn_eB=pB)
    print("  " + "-" * 104)
    print("  Frobenius column: property of the estimator; no solver improves it.")
    print("  PINN minus Frobenius: the solver's contribution.")
    print("  meas/pred near 1 confirms Eq.(amp) at the network's own eps_y.")
    return out, frob, runs


# =============================================================================
# [V7-4] ONE-CALL RUNNER
# =============================================================================
def run_all(n_epochs=5000, seeds=(0, 1, 2), cfg_seed=(2, 2, 2),
            cheap=True, sobolev=True, sidebyside=True, scan=True, lms=True,
            convergence=False, equal_walltime=False,
            lms_rows=((2, 2, 2), (2, 3, 1), (2, 4, 1))):
    """Everything, in the order that maximises information per hour.

    Phase 0  cheap    minutes, no training, cannot be invalidated later.
    Phase 1  sobolev  the only untested claim in the method section.
    Phase 2  side     all eight estimators, network vs Frobenius input.
                      The paper's main diagnostic.
    Phase 3  scan     the Love-number table, in eps at one (s,l,m).
    Phase 4  lms      the (s,l,m) dependence of the NETWORK results, against
                      the training-free prediction of phase 0.
    Phase 5  converge optional; the amplification relation swept in eps_y.

    Phases 1-3 checkpoint every run to JSON, so a disconnect resumes where it
    stopped: just call run_all() again with the same arguments.  Training
    prints a heartbeat every PROGRESS_EVERY epochs.

    Run counts at the defaults: phase 1 is 9 runs, phase 2 is 3, phase 3 is
    15, phase 4 is 9 -- 36 training runs.  Time one run first and multiply.
    """
    t00 = time.time()

    def banner(n, name, note=""):
        print("\n" + "#" * 78)
        print(f"#  PHASE {n}: {name}   [{(time.time()-t00)/60:.0f} min elapsed]")
        if note:
            print(f"#  {note}")
        print("#" * 78, flush=True)

    if cheap:
        banner(0, "no-training tables", "minutes; these are final once run")
        table_truncation()
        table_contamination()
        table_convention()
        table_amplification()
        table_ampmap(s_list=(2,), ell_list=(2, 3, 4, 5, 6), eps=1e-3)
        table_baseline()

    if sobolev:
        banner(1, "does the Sobolev loss help?",
               f"{len(seeds)*3} runs at {n_epochs} epochs"
               + (" + equal-wall-clock block" if equal_walltime else ""))
        table_sobolev(cfg_seed=cfg_seed, eps=1e-3, seeds=seeds,
                      n_epochs=n_epochs, equal_walltime=equal_walltime,
                      checkpoint="sobolev.json")

    if sidebyside:
        banner(2, "all eight estimators: network vs Frobenius",
               f"{len(seeds)} runs at {n_epochs} epochs; the eight estimators "
               f"are then free")
        table_sidebyside(cfg_seed=cfg_seed, eps=1e-3, seeds=seeds,
                         n_epochs=n_epochs, checkpoint="sidebyside.json")

    if scan:
        banner(3, "Love number versus the regulator",
               f"{len(seeds)*5} runs at {n_epochs} epochs")
        table_scan(cfg_seed=cfg_seed, seeds=seeds, n_epochs=n_epochs,
                   checkpoint="lovescan.json")

    if lms:
        banner(4, "network results across (s,l,m)",
               f"{len(seeds)*len(lms_rows)} runs at {n_epochs} epochs; "
               f"tests the training-free map")
        table_lms(rows=lms_rows, eps=1e-3, seeds=seeds, n_epochs=n_epochs,
                  checkpoint="lms.json")

    if convergence:
        banner(5, "convergence: does the observable follow the residual?",
               f"{len(seeds)} long runs")
        study_convergence(cfg_seed=cfg_seed, eps=1e-3, seeds=seeds,
                          n_epochs=4 * n_epochs)

    print("\n" + "#" * 78)
    print(f"#  DONE in {(time.time()-t00)/60:.0f} min")
    print("#" * 78, flush=True)


# =============================================================================
# MAIN
# =============================================================================
# Cheap tasks run in a few minutes on CPU and need no GPU.
# "sobolev" TRAINS: 3 variants x SOBOLEV_SEEDS seeds at SOBOLEV_EPOCHS, plus
# 2 more variants for the equal-wall-clock block.  Smoke-test it first with
#     table_sobolev(seeds=(0,), n_epochs=400)
# before running the full thing.
# =============================================================================
# Run everything with one call.  Re-running resumes from the JSON checkpoints.
# =============================================================================
if __name__ == "__main__":
    print("=" * 72)
    print(f"device={device}  dtype={torch.get_default_dtype()}  mp.dps={mp.mp.dps}")
    print(f"EPOCHS default={EPOCHS}  LAM_H1={LAM_H1}  LAM_H2={LAM_H2}  "
          f"PROGRESS_EVERY={PROGRESS_EVERY}")
    print("=" * 72)
    run_all(n_epochs=5000, seeds=(0, 1, 2), cfg_seed=(2, 2, 2),
            cheap=True, sobolev=True, sidebyside=True, scan=True, lms=True,
            convergence=False, equal_walltime=False)
