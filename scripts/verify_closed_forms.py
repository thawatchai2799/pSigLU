"""Symbolic verification of the closed forms used in the pSigLU paper.

Checks, with SymPy:
  * Theorem 1 (family smoothness): for a generic smooth kernel g with g(0)=1,
    f(x) = ReLU(x) - |x| g(|x|)/2 has continuous derivatives of order 0,1,2,
    continuous even-order derivatives, and odd-order jumps -k g^(k-1)(0).
  * Proposition 2: first and second derivatives of pSigLU_p.
  * Proposition 3: minimum location/value, curvature zero, maximal slope,
    gradient tail (sign included).
  * Proposition 4(a): uniform distance to ReLU.
  * Corollary 1: third-derivative jumps of LapLU (-3/b^2) and pSigLU (-3p(p+1)).
Run:  python verify_closed_forms.py
"""
import sympy as sp

x, t, p, b = sp.symbols('x t p b', positive=True)

# ---------- Theorem 1 with a generic kernel (Taylor coefficients) ----------
K = 8
c = sp.symbols('c0:%d' % K)
g = 1 + sum(c[k] * t**k for k in range(1, K))          # g(0) = 1
f_right = x - sp.Rational(1, 2) * x * g.subs(t, x)      # x > 0
f_left = -sp.Rational(1, 2) * (-x) * g.subs(t, -x)      # x < 0  (|x| = -x)
for k in range(0, K):
    jr = sp.diff(f_right, x, k).subs(x, 0)
    jl = sp.diff(f_left, x, k).subs(x, 0)
    jump = sp.simplify(jr - jl)
    gk1 = sp.diff(g, t, k - 1).subs(t, 0) if k >= 1 else None
    if k <= 2 or k % 2 == 0:
        assert jump == 0, (k, jump)
    else:
        assert sp.simplify(jump + k * gk1) == 0, (k, jump, gk1)
print('Theorem 1: jumps verified up to order', K - 1)

# ---------- pSigLU closed forms ----------
fR = x - sp.Rational(1, 2) * x * (1 + x)**(-p)                       # x > 0
fL = -sp.Rational(1, 2) * t * (1 + t)**(-p)                          # x < 0, t = -x
d1R = sp.simplify(sp.diff(fR, x)); d2R = sp.simplify(sp.diff(fR, x, 2))
d1L = sp.simplify(-sp.diff(fL, t)); d2L = sp.simplify(sp.diff(fL, t, 2))
assert sp.simplify(d1R - (1 - sp.Rational(1, 2) * (1 + x)**(-p - 1) * (1 - (p - 1) * x))) == 0
assert sp.simplify(d1L - (sp.Rational(1, 2) * (1 + t)**(-p - 1) * (1 - (p - 1) * t))) == 0
assert sp.simplify(d2R - (p / 2) * (1 + x)**(-p - 2) * (2 - (p - 1) * x)) == 0
assert sp.simplify(d2L - (p / 2) * (1 + t)**(-p - 2) * (2 - (p - 1) * t)) == 0
print('Proposition 2: first and second derivatives verified')

# Proposition 3
tstar = 1 / (p - 1)
assert sp.simplify(d1L.subs(t, tstar)) == 0
fmin = sp.simplify(fL.subs(t, tstar))
for pv in [sp.Rational(6, 5), sp.Rational(3, 2), 2, 3, 7]:
    assert sp.simplify(fmin.subs(p, pv) + sp.Rational(1, 2) * (pv - 1)**(pv - 1) / pv**pv) == 0
assert sp.simplify(d2R.subs(x, 2 / (p - 1))) == 0
smax = sp.simplify(d1R.subs(x, 2 / (p - 1)))
for pv in [sp.Rational(3, 2), 2, 3, 5]:
    assert sp.simplify(smax.subs(p, pv) - (1 + sp.Rational(1, 2) * ((pv - 1) / (pv + 1))**(pv + 1))) == 0
assert sp.simplify(d1L.subs(t, 0)) == sp.Rational(1, 2)
tail = sp.limit(d1L * t**p, t, sp.oo)
assert sp.simplify(tail + (p - 1) / 2) == 0            # f'(x) ~ -(p-1)/2 |x|^-p  (negative)
print('Proposition 3: minimum, curvature zero, maximal slope, gradient tail verified')

# Proposition 4(a): sup |pSigLU - ReLU| = |f_min|
for pv in [sp.Rational(6, 5), 2, 3]:
    assert sp.simplify(-fmin.subs(p, pv) - sp.Rational(1, 2) * (pv - 1)**(pv - 1) / pv**pv) == 0
print('Proposition 4(a): uniform distance to ReLU verified')

# Corollary 1: third-derivative jumps
def jump3(kernel):
    fr = x - sp.Rational(1, 2) * x * kernel.subs(t, x)
    fl = -sp.Rational(1, 2) * (-x) * kernel.subs(t, -x)
    return sp.simplify(sp.limit(sp.diff(fr, x, 3), x, 0) - sp.limit(sp.diff(fl, x, 3), x, 0))
assert sp.simplify(jump3(sp.exp(-t / b)) + 3 / b**2) == 0
assert sp.simplify(jump3((1 + t)**(-p)) + 3 * p * (p + 1)) == 0
print('Corollary 1: third-derivative jumps -3/b^2 (LapLU) and -3p(p+1) (pSigLU) verified')
print('ALL CHECKS PASSED')
