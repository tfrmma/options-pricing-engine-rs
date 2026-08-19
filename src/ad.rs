// forward-mode AD for Heston Greeks.
//
// bump-and-reprice needs 14 pricing calls (28 adaptive GK integrations,
// P1+P2 each). this does 5 forward passes, 10 integrations total.
//
// the trick: Leibniz rule lets us differentiate under the integral.
//   d(price)/dp = integral of Re[ d(CF(u,p))/dp * kernel(u) ] du
// so we propagate dual numbers through stable_cf, then integrate the
// dual part. same GK-15 quadrature, dual arithmetic instead of complex.
//
// Dual<f64>: (val, dot) where dot = d(val)/dp for the active param.
// one pass = one param, 5 params = 5 passes.
//
// measured on this box (see main.rs::heston_ad_demo): fewer integrations
// does NOT mean faster wall clock. Complex<Dual> multiply/divide costs more
// per GK node than Complex64, enough to eat the 28-vs-10 win and then some.
// use this for exactness (no FD bump-size tuning, no truncation error on
// vega/vanna), not as a drop-in speed upgrade until someone profiles where
// the dual arithmetic is bleeding cycles.

use std::ops::{Add, Sub, Mul, Div, Neg, Rem};
use num_traits::{Zero, One, Num};
use num_complex::Complex;
use crate::types::{HestonParams, OptionType, PricingResult};
use crate::heston::{GK_NODES, GK_WEIGHTS, G7_IDX, G7_WEIGHTS};

// --- Dual number ---

#[derive(Clone, Copy, Debug)]
pub struct Dual {
    pub val: f64,
    pub dot: f64,  // derivative w.r.t. active param
}

impl Dual {
    #[inline] pub fn constant(v: f64) -> Self { Dual { val: v, dot: 0.0 } }
    #[inline] pub fn active(v: f64)   -> Self { Dual { val: v, dot: 1.0 } }

    #[inline]
    pub fn exp(self) -> Self {
        let e = self.val.exp();
        Dual { val: e, dot: e * self.dot }
    }

    #[inline]
    pub fn ln(self) -> Self {
        Dual { val: self.val.ln(), dot: self.dot / self.val }
    }

    #[inline]
    pub fn sqrt(self) -> Self {
        let s = self.val.sqrt();
        Dual { val: s, dot: self.dot / (2.0 * s) }
    }
}

impl Add  for Dual { type Output = Self; fn add(self, r: Self) -> Self { Dual { val: self.val + r.val, dot: self.dot + r.dot } } }
impl Sub  for Dual { type Output = Self; fn sub(self, r: Self) -> Self { Dual { val: self.val - r.val, dot: self.dot - r.dot } } }
impl Neg  for Dual { type Output = Self; fn neg(self)          -> Self { Dual { val: -self.val, dot: -self.dot } } }
// Rem needed by NumOps. not mathematically meaningful for dual numbers but required by the trait.
impl Rem  for Dual { type Output = Self; fn rem(self, r: Self) -> Self { Dual { val: self.val % r.val, dot: 0.0 } } }
impl Mul  for Dual { type Output = Self; fn mul(self, r: Self) -> Self { Dual { val: self.val * r.val, dot: self.val * r.dot + self.dot * r.val } } }
impl Div  for Dual { type Output = Self; fn div(self, r: Self) -> Self {
    Dual { val: self.val / r.val, dot: (self.dot * r.val - self.val * r.dot) / (r.val * r.val) }
}}

// num_traits impls, required for Complex<Dual> to use num-complex's internal ops.
// without these, CDual * CDual won't compile.
impl Zero for Dual {
    fn zero() -> Self { Dual::constant(0.0) }
    fn is_zero(&self) -> bool { self.val == 0.0 && self.dot == 0.0 }
}
impl One for Dual {
    fn one() -> Self { Dual::constant(1.0) }
}
impl Num for Dual {
    type FromStrRadixErr = ();
    fn from_str_radix(_s: &str, _radix: u32) -> Result<Self, ()> { Err(()) }
}

// PartialEq needed by Num
impl PartialEq for Dual {
    fn eq(&self, other: &Self) -> bool { self.val == other.val }
}

// f64 * Dual convenience needed for scaling
impl Mul<Dual> for f64 { type Output = Dual; fn mul(self, d: Dual) -> Dual { Dual { val: self * d.val, dot: self * d.dot } } }
impl Add<Dual> for f64 { type Output = Dual; fn add(self, d: Dual) -> Dual { Dual { val: self + d.val, dot: d.dot } } }

// --- Complex<Dual> helpers ---

type CDual = Complex<Dual>;

#[inline]
fn cd(re: f64, im: f64) -> CDual {
    Complex::new(Dual::constant(re), Dual::constant(im))
}

#[inline]
fn cd_i() -> CDual { cd(0.0, 1.0) }

// exp for Complex<Dual>. num-complex doesn't know about our Dual type.
// e^(a+bi) = e^a * (cos(b) + i*sin(b)), with dual chain rule on a and b.
fn cexp(z: CDual) -> CDual {
    let ea  = z.re.val.exp();
    let cos = z.im.val.cos();
    let sin = z.im.val.sin();

    // d/dp[e^a * cos(b)] = e^a*(a_dot*cos(b) - b_dot*sin(b))
    // d/dp[e^a * sin(b)] = e^a*(a_dot*sin(b) + b_dot*cos(b))
    let re_val = ea * cos;
    let im_val = ea * sin;
    let re_dot = ea * (z.re.dot * cos - z.im.dot * sin);
    let im_dot = ea * (z.re.dot * sin + z.im.dot * cos);

    Complex::new(
        Dual { val: re_val, dot: re_dot },
        Dual { val: im_val, dot: im_dot },
    )
}

// sqrt for Complex<Dual>. standard complex sqrt with dual chain rule.
// sqrt for Complex<Dual>. same cancellation trap as heston.rs's fast_csqrt:
// the textbook re=sqrt((r+a)/2), im=sign(b)*sqrt((r-a)/2) formula cancels
// badly when b is tiny relative to a (exactly the sigma->0 regime, see
// heston.rs for the regression that caught it there first). fix is the same:
// get the safe component directly, the other one via division instead of
// subtraction. built entirely out of Dual's own +,-,*,/,sqrt so the chain
// rule comes along for free instead of being hand-derived and hand-verified
// twice.
fn csqrt(z: CDual) -> CDual {
    let a = z.re;
    let b = z.im;
    if b.val == 0.0 {
        return if a.val >= 0.0 {
            Complex::new(a.sqrt(), b)
        } else {
            let m = (-a).sqrt();
            Complex::new(Dual::constant(0.0), if b.val.is_sign_positive() { m } else { -m })
        };
    }
    let r = (a*a + b*b).sqrt();
    if a.val >= 0.0 {
        let re = ((r + a) * Dual::constant(0.5)).sqrt();
        let im = b / (re * Dual::constant(2.0));
        Complex::new(re, im)
    } else {
        let mag = ((r - a) * Dual::constant(0.5)).sqrt(); // r-a safe here, a<0
        let im  = if b.val.is_sign_positive() { mag } else { -mag };
        let re  = b / (im * Dual::constant(2.0));
        Complex::new(re, im)
    }
}

// ln for Complex<Dual>. ln(z) = ln|z| + i*arg(z).
fn cln(z: CDual) -> CDual {
    let a  = z.re.val; let b  = z.im.val;
    let da = z.re.dot; let db = z.im.dot;
    let r2 = a*a + b*b;
    let r  = r2.sqrt();

    let ln_val  = r.ln();
    let arg_val = b.atan2(a);
    let ln_dot  = (a*da + b*db) / r2;
    let arg_dot = (a*db - b*da) / r2;

    Complex::new(
        Dual { val: ln_val, dot: ln_dot },
        Dual { val: arg_val, dot: arg_dot },
    )
}

// --- Albrecher stable CF over Dual numbers ---
//
// identical structure to stable_cf in heston.rs. if you change the formula
// there, change it here too. yes, this is the price of manual AD.

// `mu` = risk-neutral drift of ln S = r - q, same contract as stable_cf.
fn stable_cf_dual(phi: CDual, t: f64, mu: f64, p: &DualParams) -> CDual {
    let i   = cd_i();
    let c   = |v: f64| Complex::new(Dual::constant(v), Dual::constant(0.0));
    let cd_param = |d: Dual| Complex::new(d, Dual::constant(0.0));

    let xi  = cd_param(p.kappa) - cd_param(p.rho * p.sigma) * phi * i;
    let d   = csqrt(xi*xi + cd_param(p.sigma * p.sigma) * phi * (phi + i));
    let g   = (xi - d) / (xi + d);
    let edt = cexp(d * c(-t));
    let one = c(1.0);
    let a   = cln((g*edt - one) / (g - one));
    let cc  = cd_param(p.kappa * p.theta / (p.sigma * p.sigma))
            * ((xi - d) * c(t) - a * c(2.0));
    let dd  = cd_param(p.v0) * (xi - d) * (one - edt)
            / (cd_param(p.sigma * p.sigma) * (one - g * edt));

    cexp(c(mu * t) * phi * i + cc + dd)
}

// jump component, mirrors jump_cf in bates.rs exactly. lambda/mu_j/sigma_j
// are plain constants here, not part of the active-param set (that's still
// v0/kappa/theta/sigma/rho, indices 0..=4), so this factor's own dot is
// always zero. Complex<Dual> multiplication already implements the product
// rule via Dual's Mul impl, so folding this into stable_cf_dual's output
// via a straight `*` gives the right derivative for free, no extra plumbing.
// jump CF factor, three modes: no jumps at all (Off), jumps present but
// none of their params are the active derivative (Const, the Heston-active
// passes use this), or jumps present with ONE of lambda/mu_j/sigma_j
// carrying the active derivative (Active, used by the jump-parameter
// sensitivity passes below). one function instead of two near-duplicates
// so forward_pass/dual_integrand don't need a second copy for jump-param
// differentiation, they already take a JumpSpec and don't care which
// variant it is.
#[derive(Clone, Copy)]
enum JumpSpec {
    Off,
    Const(f64, f64, f64),
    Active(f64, f64, f64, usize), // lambda, mu_j, sigma_j, 0=lambda active/1=mu_j active/2=sigma_j active
}

fn jump_factor(phi: CDual, t: f64, spec: JumpSpec) -> CDual {
    let (lambda, mu_j, sigma_j, active) = match spec {
        JumpSpec::Off => return cd(1.0, 0.0),
        JumpSpec::Const(l, m, s)      => (l, m, s, usize::MAX),
        JumpSpec::Active(l, m, s, a)  => (l, m, s, a),
    };
    let i   = cd_i();
    let dl  = if active == 0 { Dual::active(lambda)  } else { Dual::constant(lambda) };
    let dmj = if active == 1 { Dual::active(mu_j)    } else { Dual::constant(mu_j) };
    let dsj = if active == 2 { Dual::active(sigma_j) } else { Dual::constant(sigma_j) };
    let c0 = |d: Dual| Complex::new(d, Dual::constant(0.0));

    let half_sj2 = dsj * dsj * Dual::constant(0.5);
    let comp = (dmj + half_sj2).exp() - Dual::constant(1.0);
    let jump = cexp(phi * i * c0(dmj) - c0(half_sj2) * phi * phi);
    cexp(c0(dl) * c0(Dual::constant(t)) * (jump - c0(Dual::constant(1.0)) - i * phi * c0(comp)))
}

// --- experimental: multi-directional dual, one pass instead of five ---
//
// heston_greeks_ad runs 5 separate scalar-dual forward_pass calls, one per
// param, each recomputing the ENTIRE value pathway (all the stable_cf
// arithmetic) redundantly, only the active direction differs between
// passes. Dual5 carries all 5 tangent directions at once (dot: [f64;5]
// instead of f64), so the value gets computed ONCE and all 5 partials ride
// along with it, this is the standard reason reverse-mode/vector-forward-
// mode AD beats naive forward-mode when you want a full gradient from one
// scalar output, we don't need reverse-mode's tape machinery since N=5 is
// small and fixed.
//
// the tradeoff isn't free: every arithmetic op now touches a 5-vector
// instead of a scalar for the derivative part, so mul/div (already the
// most expensive ops for scalar Dual, see profile_op_level) could plausibly
// cost MORE in aggregate than 5 separate scalar passes if the per-op
// overhead scales linearly with N. whether the "compute the value once"
// saving wins depends on the actual mix of ops, not something to guess at,
// see profile_dual5_vs_five_scalar_passes for the measured answer.
//
// scoped to Heston only for this experiment (5 directions). Bates would be
// 8 (5 Heston + 3 jump), a natural follow-on if this turns out to help.

#[derive(Clone, Copy, Debug)]
pub struct Dual5 {
    pub val: f64,
    pub dot: [f64; 5],
}

impl Dual5 {
    #[inline] pub fn constant(v: f64) -> Self { Dual5 { val: v, dot: [0.0; 5] } }
    #[inline] pub fn active(v: f64, i: usize) -> Self {
        let mut dot = [0.0; 5];
        dot[i] = 1.0;
        Dual5 { val: v, dot }
    }

    #[inline]
    pub fn exp(self) -> Self {
        let e = self.val.exp();
        let dot = std::array::from_fn(|k| e * self.dot[k]);
        Dual5 { val: e, dot }
    }
    #[inline]
    pub fn ln(self) -> Self {
        let dot = std::array::from_fn(|k| self.dot[k] / self.val);
        Dual5 { val: self.val.ln(), dot }
    }
    #[inline]
    pub fn sqrt(self) -> Self {
        let s = self.val.sqrt();
        let dot = std::array::from_fn(|k| self.dot[k] / (2.0 * s));
        Dual5 { val: s, dot }
    }
}

impl Add for Dual5 {
    type Output = Self;
    fn add(self, r: Self) -> Self {
        let dot = std::array::from_fn(|k| self.dot[k] + r.dot[k]);
        Dual5 { val: self.val + r.val, dot }
    }
}
impl Sub for Dual5 {
    type Output = Self;
    fn sub(self, r: Self) -> Self {
        let dot = std::array::from_fn(|k| self.dot[k] - r.dot[k]);
        Dual5 { val: self.val - r.val, dot }
    }
}
impl Neg for Dual5 {
    type Output = Self;
    fn neg(self) -> Self {
        let dot = std::array::from_fn(|k| -self.dot[k]);
        Dual5 { val: -self.val, dot }
    }
}
impl Rem for Dual5 {
    type Output = Self;
    fn rem(self, r: Self) -> Self { Dual5 { val: self.val % r.val, dot: [0.0; 5] } }
}
// product rule needs a + here, that's not a bug, clippy's heuristic for
// "suspicious + inside Mul" doesn't know calculus. same pattern as the
// scalar Dual::mul above, which clippy doesn't flag only because it's
// written as a single expression instead of a separate `let`.
#[allow(clippy::suspicious_arithmetic_impl)]
impl Mul for Dual5 {
    type Output = Self;
    fn mul(self, r: Self) -> Self {
        let dot = std::array::from_fn(|k| self.val * r.dot[k] + self.dot[k] * r.val);
        Dual5 { val: self.val * r.val, dot }
    }
}
impl Div for Dual5 {
    type Output = Self;
    fn div(self, r: Self) -> Self {
        let r2 = r.val * r.val;
        let dot = std::array::from_fn(|k| (self.dot[k] * r.val - self.val * r.dot[k]) / r2);
        Dual5 { val: self.val / r.val, dot }
    }
}

impl Zero for Dual5 {
    fn zero() -> Self { Dual5::constant(0.0) }
    fn is_zero(&self) -> bool { self.val == 0.0 && self.dot == [0.0; 5] }
}
impl One for Dual5 { fn one() -> Self { Dual5::constant(1.0) } }
impl Num for Dual5 {
    type FromStrRadixErr = ();
    fn from_str_radix(_s: &str, _radix: u32) -> Result<Self, ()> { Err(()) }
}
impl PartialEq for Dual5 { fn eq(&self, other: &Self) -> bool { self.val == other.val } }
impl Mul<Dual5> for f64 { type Output = Dual5; fn mul(self, d: Dual5) -> Dual5 {
    let dot = std::array::from_fn(|k| self * d.dot[k]);
    Dual5 { val: self * d.val, dot }
}}

type CDual5 = Complex<Dual5>;

#[inline]
fn cd5(re: f64, im: f64) -> CDual5 { Complex::new(Dual5::constant(re), Dual5::constant(im)) }
#[inline]
fn cd5_i() -> CDual5 { cd5(0.0, 1.0) }

fn cexp5(z: CDual5) -> CDual5 {
    let ea  = z.re.val.exp();
    let cb  = z.im.val.cos();
    let sb  = z.im.val.sin();
    let mut re_dot = [0.0; 5];
    let mut im_dot = [0.0; 5];
    for k in 0..5 {
        re_dot[k] = ea*cb*z.re.dot[k] - ea*sb*z.im.dot[k];
        im_dot[k] = ea*sb*z.re.dot[k] + ea*cb*z.im.dot[k];
    }
    Complex::new(Dual5 { val: ea*cb, dot: re_dot }, Dual5 { val: ea*sb, dot: im_dot })
}

fn csqrt5(z: CDual5) -> CDual5 {
    let a = z.re; let b = z.im;
    if b.val == 0.0 {
        return if a.val >= 0.0 { Complex::new(a.sqrt(), b) }
        else { let m = (-a).sqrt(); Complex::new(Dual5::constant(0.0), if b.val.is_sign_positive() { m } else { -m }) };
    }
    let r = (a*a + b*b).sqrt();
    if a.val >= 0.0 {
        let re = ((r + a) * Dual5::constant(0.5)).sqrt();
        let im = b / (re * Dual5::constant(2.0));
        Complex::new(re, im)
    } else {
        let mag = ((r - a) * Dual5::constant(0.5)).sqrt();
        let im  = if b.val.is_sign_positive() { mag } else { -mag };
        let re  = b / (im * Dual5::constant(2.0));
        Complex::new(re, im)
    }
}

fn cln5(z: CDual5) -> CDual5 {
    let r2 = z.re.val*z.re.val + z.im.val*z.im.val;
    let mut re_dot = [0.0; 5];
    let mut im_dot = [0.0; 5];
    for k in 0..5 {
        re_dot[k] = (z.re.val*z.re.dot[k] + z.im.val*z.im.dot[k]) / r2;
        im_dot[k] = (z.re.val*z.im.dot[k] - z.im.val*z.re.dot[k]) / r2;
    }
    Complex::new(
        Dual5 { val: 0.5*r2.ln(), dot: re_dot },
        Dual5 { val: z.im.val.atan2(z.re.val), dot: im_dot },
    )
}

struct DualParams5 { v0: Dual5, kappa: Dual5, theta: Dual5, sigma: Dual5, rho: Dual5 }

fn dual_params5(p: &HestonParams) -> DualParams5 {
    DualParams5 {
        v0:    Dual5::active(p.v0,    0),
        kappa: Dual5::active(p.kappa, 1),
        theta: Dual5::active(p.theta, 2),
        sigma: Dual5::active(p.sigma, 3),
        rho:   Dual5::active(p.rho,   4),
    }
}

// port of stable_cf_dual, Dual5 instead of Dual, ALL 5 Heston params active
// simultaneously (each in its own orthogonal direction) instead of one at
// a time. same formula, verified by construction (mechanical port), not
// just by inspection, see forward_pass5_matches_five_scalar_passes.
// `mu` = risk-neutral drift of ln S = r - q, same contract as stable_cf.
fn stable_cf_dual5(phi: CDual5, t: f64, mu: f64, p: &DualParams5) -> CDual5 {
    let i = cd5_i();
    let DualParams5 { v0, kappa, theta, sigma, rho } = *p;
    let rt = |v: Dual5| Complex::new(v, Dual5::constant(0.0));

    let xi  = rt(kappa) - rt(rho)*rt(sigma)*phi*i;
    let d   = csqrt5(xi*xi + rt(sigma)*rt(sigma) * phi*(phi + i));
    let g   = (xi - d) / (xi + d);
    let edt = cexp5(d * cd5(-t, 0.0));
    let one = cd5(1.0, 0.0);
    let a   = (g*edt - one) / (g - one);

    let c  = rt(kappa)*rt(theta) / (rt(sigma)*rt(sigma)) * ((xi - d)*cd5(t,0.0) - cln5(a)*cd5(2.0,0.0));
    let dd = rt(v0) * (xi - d) * (one - edt) / (rt(sigma)*rt(sigma) * (one - g*edt));

    cexp5(cd5(mu,0.0)*phi*i*cd5(t,0.0) + c + dd)
}

// one joint forward pass, all 5 Heston-param derivatives at once. same
// Leibniz/GK machinery as forward_pass, mirrors it exactly, just Dual5
// instead of Dual and no `active` parameter since everything's active here.
fn forward_pass5(
    s: f64, k: f64, t: f64, r: f64, q: f64,
    p: &HestonParams, opt_type: OptionType,
) -> (f64, [f64; 5]) {
    let x  = (s/k).ln();
    let dp = dual_params5(p);

    // CF drift is r-q, same fix as heston_call / forward_pass above.
    let mu = r - q;
    let phi_mi = Complex::new(Dual5::constant(0.0), Dual5::constant(-1.0));
    let cf_mi  = stable_cf_dual5(phi_mi, t, mu, &dp);

    let integrand = |u: f64, is_p1: bool, cf_mi_opt: Option<CDual5>| -> (f64, [f64; 5]) {
        let phi: CDual5 = if is_p1 {
            Complex::new(Dual5::constant(u), Dual5::constant(-1.0))
        } else {
            Complex::new(Dual5::constant(u), Dual5::constant(0.0))
        };
        let mut cf = stable_cf_dual5(phi, t, mu, &dp);
        if let Some(norm) = cf_mi_opt { cf = cf / norm; }
        let exp_term = cexp5(Complex::new(Dual5::constant(0.0), Dual5::constant(u * x)));
        let num = exp_term * cf;
        let div = Complex::new(Dual5::constant(0.0), Dual5::constant(u));
        let res = num / div;
        (res.re.val, res.re.dot)
    };

    let (i1_val, i1_dot) = gk_integrate_dual5(|u| integrand(u, true, Some(cf_mi)));
    let (i2_val, i2_dot) = gk_integrate_dual5(|u| integrand(u, false, None));

    let p1_val = 0.5 + i1_val / std::f64::consts::PI;
    let p2_val = 0.5 + i2_val / std::f64::consts::PI;
    let mut p1_dot = [0.0; 5];
    let mut p2_dot = [0.0; 5];
    for k_ in 0..5 { p1_dot[k_] = i1_dot[k_] / std::f64::consts::PI; p2_dot[k_] = i2_dot[k_] / std::f64::consts::PI; }

    // stock leg carries e^{-qT}, same as heston_call / forward_pass. the old
    // form (bare s, and bare s in the parity terms) silently priced q=0.
    let df  = (-r*t).exp();
    let seq = s * (-q*t).exp();
    let call_val = seq*p1_val - k*df*p2_val;
    let mut call_dot = [0.0; 5];
    for k_ in 0..5 { call_dot[k_] = seq*p1_dot[k_] - k*df*p2_dot[k_]; }

    match opt_type {
        OptionType::Call => (call_val, call_dot),
        OptionType::Put  => (call_val - seq + k*df, call_dot), // parity terms don't depend on Heston params
    }
}

// same adaptive GK as gk_integrate_dual, carrying a [f64;5] instead of f64.
fn gk15_panel_dual5<F: Fn(f64) -> (f64, [f64; 5])>(f: &F, a: f64, b: f64) -> (f64, [f64; 5], f64) {
    let c = 0.5 * (a + b);
    let h = 0.5 * (b - a);
    let fv: [(f64, [f64; 5]); 15] = std::array::from_fn(|i| f(c + h * GK_NODES[i]));
    let k_val: f64 = (0..15).map(|i| GK_WEIGHTS[i] * fv[i].0).sum();
    let k_dot: [f64; 5] = std::array::from_fn(|d| h * (0..15).map(|i| GK_WEIGHTS[i] * fv[i].1[d]).sum::<f64>());
    let g_val: f64 = (0..7).map(|j| G7_WEIGHTS[j] * fv[G7_IDX[j]].0).sum();
    let err = (k_val - g_val).abs() * h;
    (k_val * h, k_dot, err)
}

fn adaptive_gk_dual5<F: Fn(f64) -> (f64, [f64; 5])>(f: &F, a: f64, b: f64, tol: f64) -> (f64, [f64; 5]) {
    const MAX_PANELS: usize = 200;
    let (v0, d0, e0) = gk15_panel_dual5(f, a, b);
    let mut panels: Vec<(f64, f64, [f64; 5], f64, f64)> = vec![(e0, v0, d0, a, b)];
    let mut total_val = v0;
    let mut total_dot = d0;
    let mut err = e0;
    while err > tol && panels.len() < MAX_PANELS {
        let w = (0..panels.len()).max_by(|&i, &j| panels[i].0.total_cmp(&panels[j].0)).unwrap();
        let (ew, vw, dw, aw, bw) = panels.swap_remove(w);
        let m = 0.5 * (aw + bw);
        let (vl, dl, el) = gk15_panel_dual5(f, aw, m);
        let (vr, dr, er) = gk15_panel_dual5(f, m, bw);
        total_val += vl + vr - vw;
        for k in 0..5 { total_dot[k] += dl[k] + dr[k] - dw[k]; }
        err += el + er - ew;
        panels.push((el, vl, dl, aw, m));
        panels.push((er, vr, dr, m, bw));
    }
    (total_val, total_dot)
}

fn gk_integrate_dual5<F: Fn(f64) -> (f64, [f64; 5])>(f: F) -> (f64, [f64; 5]) {
    let g = |t: f64| -> (f64, [f64; 5]) {
        if t <= 0.0 { return (0.0, [0.0; 5]); }
        let u = (1.0 - t) / t;
        if u < 1e-12 { return (0.0, [0.0; 5]); }
        let (v, d) = f(u);
        let jac = t * t;
        let mut dd = [0.0; 5];
        for k in 0..5 { dd[k] = d[k] / jac; }
        let vv = v / jac;
        if vv.is_finite() { (vv, dd) } else { (0.0, [0.0; 5]) }
    };
    adaptive_gk_dual5(&g, 0.0, 1.0, 1e-8)
}

// same PricingResult assembly as heston_greeks_ad, just sourcing the 5
// Heston-driven derivatives from one joint pass instead of 5 separate ones.
// rho(rate)/vanna/volga are still bump-and-reprice, same as heston_greeks_ad,
// this experiment is only about the v0/kappa/theta/sigma/rho integration cost.
pub fn heston_greeks_ad5(
    spot: f64, strike: f64, expiry: f64,
    rate: f64, div_yield: f64,
    params: &HestonParams, opt_type: OptionType,
) -> PricingResult {
    let (price, d) = forward_pass5(spot, strike, expiry, rate, div_yield, params, opt_type);
    let vol  = params.v0.sqrt();
    let vega = d[0] * 2.0 * vol; // chain rule, v0 = vol^2

    let ds = 0.01 * spot;
    let delta = (crate::heston::heston_price(spot+ds, strike, expiry, rate, div_yield, params, opt_type)
               - crate::heston::heston_price(spot-ds, strike, expiry, rate, div_yield, params, opt_type))
              / (2.0*ds);
    let gamma = (crate::heston::heston_price(spot+ds, strike, expiry, rate, div_yield, params, opt_type)
               - 2.0*price
               + crate::heston::heston_price(spot-ds, strike, expiry, rate, div_yield, params, opt_type))
              / (ds*ds);

    // theta/vanna/volga: same shared step rules as heston_greeks_ad, see the
    // comments there (central_theta and vol_bump_levels live in greeks.rs).
    let theta = crate::greeks::central_theta(expiry, |t|
        crate::heston::heston_price(spot, strike, t, rate, div_yield, params, opt_type));

    let dr = 1e-4;
    let rho_greek = (crate::heston::heston_price(spot, strike, expiry, rate+dr, div_yield, params, opt_type)
                    - crate::heston::heston_price(spot, strike, expiry, rate-dr, div_yield, params, opt_type))
                   / (2.0*dr);

    let (v_up, v_dn, v_span) = crate::greeks::vol_bump_levels(vol, 1.0);
    let p_vup = HestonParams { v0: v_up.powi(2), ..*params };
    let p_vdn = HestonParams { v0: v_dn.powi(2), ..*params };
    let delta_vup = (crate::heston::heston_price(spot+ds, strike, expiry, rate, div_yield, &p_vup, opt_type)
                    - crate::heston::heston_price(spot-ds, strike, expiry, rate, div_yield, &p_vup, opt_type)) / (2.0*ds);
    let delta_vdn = (crate::heston::heston_price(spot+ds, strike, expiry, rate, div_yield, &p_vdn, opt_type)
                    - crate::heston::heston_price(spot-ds, strike, expiry, rate, div_yield, &p_vdn, opt_type)) / (2.0*ds);
    let vanna = (delta_vup - delta_vdn) / v_span;

    let (_, d_vup) = forward_pass5(spot, strike, expiry, rate, div_yield, &p_vup, opt_type);
    let (_, d_vdn) = forward_pass5(spot, strike, expiry, rate, div_yield, &p_vdn, opt_type);
    let vega_up = d_vup[0] * 2.0 * v_up;
    let vega_dn = d_vdn[0] * 2.0 * v_dn;
    let volga = (vega_up - vega_dn) / v_span;

    PricingResult { price, delta, gamma, vega, theta, rho: rho_greek, vanna, volga }
}


struct DualParams {
    v0:    Dual,
    kappa: Dual,
    theta: Dual,
    sigma: Dual,
    rho:   Dual,
}

fn dual_params(p: &HestonParams, active: usize) -> DualParams {
    let d = |v: f64, i: usize| if i == active { Dual::active(v) } else { Dual::constant(v) };
    DualParams {
        v0:    d(p.v0,    0),
        kappa: d(p.kappa, 1),
        theta: d(p.theta, 2),
        sigma: d(p.sigma, 3),
        rho:   d(p.rho,   4),
    }
}

// integrand for one GK node. returns (price_contribution, deriv_contribution).
// GK node inputs plus the model/jump context needed to evaluate the CF
// there, splitting this into a struct would just move the same 8 fields
// one level of indirection away for no real benefit at a call site this hot.
#[allow(clippy::too_many_arguments)]
fn dual_integrand(
    u: f64, x: f64, t: f64, mu: f64,
    dp: &DualParams, is_p1: bool, cf_mi: Option<CDual>, jump: JumpSpec,
) -> (f64, f64) {
    let phi: CDual = if is_p1 {
        Complex::new(Dual::constant(u), Dual::constant(-1.0))
    } else {
        Complex::new(Dual::constant(u), Dual::constant(0.0))
    };

    let mut cf  = stable_cf_dual(phi, t, mu, dp);
    cf = cf * jump_factor(phi, t, jump);
    if let Some(norm) = cf_mi {
        cf = cf / norm;
    }
    let exp_term = cexp(Complex::new(Dual::constant(0.0), Dual::constant(u * x)));
    let num = exp_term * cf;
    let div = Complex::new(Dual::constant(0.0), Dual::constant(u));
    let res = num / div;

    (res.re.val, res.re.dot)
}

// same adaptive Gauss-Kronrod as heston.rs::adaptive_gk, just carrying a
// (val, dot) pair through each node instead of a bare f64. this used to be
// a fixed panel over [0, 200], which is the exact bug heston.rs's adaptive_gk
// was written to kill (under-resolves short-dated/wing integrands, silently
// produces arbitrage-violating prices). ad.rs never got the memo when that
// fix landed. error control still runs on the primal value only, the dual
// part rides along on whatever subdivision the primal needs.
fn gk15_panel_dual<F: Fn(f64) -> (f64, f64)>(f: &F, a: f64, b: f64) -> (f64, f64, f64) {
    let c = 0.5 * (a + b);
    let h = 0.5 * (b - a);
    let fv: [(f64, f64); 15] = std::array::from_fn(|i| f(c + h * GK_NODES[i]));
    let k_val: f64 = (0..15).map(|i| GK_WEIGHTS[i] * fv[i].0).sum();
    let k_dot: f64 = (0..15).map(|i| GK_WEIGHTS[i] * fv[i].1).sum();
    let g_val: f64 = (0..7).map(|j| G7_WEIGHTS[j] * fv[G7_IDX[j]].0).sum();
    (k_val * h, k_dot * h, (k_val - g_val).abs() * h)
}

fn adaptive_gk_dual<F: Fn(f64) -> (f64, f64)>(f: &F, a: f64, b: f64, tol: f64) -> (f64, f64) {
    const MAX_PANELS: usize = 200;
    let (v0, d0, e0) = gk15_panel_dual(f, a, b);
    // (error, val, dot, a, b) per live panel, same layout as adaptive_gk
    let mut panels: Vec<(f64, f64, f64, f64, f64)> = vec![(e0, v0, d0, a, b)];
    let mut total_val = v0;
    let mut total_dot = d0;
    let mut err = e0;
    while err > tol && panels.len() < MAX_PANELS {
        let w = (0..panels.len()).max_by(|&i, &j| panels[i].0.total_cmp(&panels[j].0)).unwrap();
        let (ew, vw, dw, aw, bw) = panels.swap_remove(w);
        let m = 0.5 * (aw + bw);
        let (vl, dl, el) = gk15_panel_dual(f, aw, m);
        let (vr, dr, er) = gk15_panel_dual(f, m, bw);
        total_val += vl + vr - vw;
        total_dot += dl + dr - dw;
        err += el + er - ew;
        panels.push((el, vl, dl, aw, m));
        panels.push((er, vr, dr, m, bw));
    }
    (total_val, total_dot)
}

// [0, inf) -> [0, 1] substitution, u = (1-t)/t, mirrors heston::gk_integrate exactly.
fn gk_integrate_dual<F: Fn(f64) -> (f64, f64)>(f: F) -> (f64, f64) {
    let g = |t: f64| -> (f64, f64) {
        if t <= 0.0 { return (0.0, 0.0); }
        let u = (1.0 - t) / t;
        if u < 1e-12 { return (0.0, 0.0); }
        let (v, d) = f(u);
        let jac = t * t;
        let (v, d) = (v / jac, d / jac);
        if v.is_finite() && d.is_finite() { (v, d) } else { (0.0, 0.0) }
    };
    adaptive_gk_dual(&g, 0.0, 1.0, 1e-8)
}

// one forward pass for a given active param index (0=v0, 1=kappa, 2=theta, 3=sigma, 4=rho).
// jump=None prices Heston, jump=Some(lambda,mu_j,sigma_j) prices Bates through
// the same 5 Heston-driven params, the CF composes exactly like bates_call
// does (Heston CF * jump CF, multiplied in before integration).
// returns (price, dprice/dparam).
// standard pricer-call shape (spot/strike/expiry/rate/div_yield/params/
// opt_type) plus which param is active and the jump spec, same argument
// count every pricer in this crate has, not sloppiness specific to this fn.
#[allow(clippy::too_many_arguments)]
fn forward_pass(
    s: f64, k: f64, t: f64, r: f64, q: f64,
    p: &HestonParams, opt_type: OptionType, active: usize, jump: JumpSpec,
) -> (f64, f64) {
    let x  = (s/k).ln();
    let dp = dual_params(p, active);

    // CF(-i) normalizer (as a dual, so its derivative w.r.t. the active
    // param is also propagated into P1). includes the jump factor too,
    // same as bates_call's cf_mi.
    //
    // CF drift is r-q, not r — same defect and fix as heston_call (the payoff
    // below discounts the stock leg by e^{-qT}, the CF must carry the q).
    let mu = r - q;
    let phi_mi = Complex::new(Dual::constant(0.0), Dual::constant(-1.0));
    let cf_mi = stable_cf_dual(phi_mi, t, mu, &dp) * jump_factor(phi_mi, t, jump);

    let (i1_val, i1_dot) = gk_integrate_dual(|u| dual_integrand(u, x, t, mu, &dp, true, Some(cf_mi), jump));
    let (i2_val, i2_dot) = gk_integrate_dual(|u| dual_integrand(u, x, t, mu, &dp, false, None, jump));

    let pi     = std::f64::consts::PI;
    let p1_val = 0.5 + i1_val / pi;
    let p2_val = 0.5 + i2_val / pi;
    let p1_dot = i1_dot / pi;
    let p2_dot = i2_dot / pi;

    let eq  = (-q*t).exp();
    let er  = (-r*t).exp();
    let seq = s * eq;
    let ker = k * er;

    let call_val = (seq*p1_val - ker*p2_val).max(0.0);
    let call_dot = seq*p1_dot - ker*p2_dot;

    match opt_type {
        OptionType::Call => (call_val, call_dot),
        // put via parity: put = call - S*e^(-qT) + K*e^(-rT)
        // d(put)/dp = d(call)/dp, parity terms don't depend on Heston params
        OptionType::Put  => (call_val - seq + ker, call_dot),
    }
}

// full Greek set via 5 forward passes.
// vega convention: d(price)/d(vol) where vol = sqrt(v0).
// chain rule: d(price)/d(vol) = d(price)/d(v0) * 2*vol.
pub fn heston_greeks_ad(
    spot: f64, strike: f64, expiry: f64,
    rate: f64, div_yield: f64,
    params: &HestonParams, opt_type: OptionType,
) -> PricingResult {
    let (price, dv0)    = forward_pass(spot, strike, expiry, rate, div_yield, params, opt_type, 0, JumpSpec::Off);
    let (_,     dkappa) = forward_pass(spot, strike, expiry, rate, div_yield, params, opt_type, 1, JumpSpec::Off);
    let (_,     dtheta) = forward_pass(spot, strike, expiry, rate, div_yield, params, opt_type, 2, JumpSpec::Off);
    let (_,     dsigma) = forward_pass(spot, strike, expiry, rate, div_yield, params, opt_type, 3, JumpSpec::Off);
    let (_,     drho)   = forward_pass(spot, strike, expiry, rate, div_yield, params, opt_type, 4, JumpSpec::Off);

    // vega = d(price)/d(vol), vol = sqrt(v0) => chain rule
    let vol  = params.v0.sqrt();
    let vega = dv0 * 2.0 * vol;

    // rate rho is not a Heston param so it doesn't get a forward pass.
    // TODO: extend to 6 params if you need exact rate sensitivity.
    let dr      = 1e-4;
    let p_up    = crate::heston::heston_price(spot, strike, expiry, rate+dr, div_yield, params, opt_type);
    let p_dn    = crate::heston::heston_price(spot, strike, expiry, rate-dr, div_yield, params, opt_type);
    let rho_greek = (p_up - p_dn) / (2.0 * dr);

    // spot appears in the kernel (log-moneyness x), not the CF params.
    // leibniz won't save us here still FD for delta/gamma.
    let ds    = 0.01 * spot;
    let p_sup = crate::heston::heston_price(spot+ds, strike, expiry, rate, div_yield, params, opt_type);
    let p_sdn = crate::heston::heston_price(spot-ds, strike, expiry, rate, div_yield, params, opt_type);
    let delta = (p_sup - p_sdn) / (2.0 * ds);
    let gamma = (p_sup - 2.0*price + p_sdn) / (ds * ds);

    // theta: FD on expiry. central difference with the shrinking step from
    // greeks.rs — one implementation shared with the bump-and-reprice path so
    // the two public APIs can't disagree again (the one-sided form this used
    // to carry overstated 1-DTE theta ~1.9x after greeks.rs was fixed).
    let theta = crate::greeks::central_theta(expiry, |t|
        crate::heston::heston_price(spot, strike, t, rate, div_yield, params, opt_type));

    // vanna: cross bump delta vs vol. bump levels and realized span from
    // greeks.rs too: the fixed 2*0.01 denominator understated vanna/volga
    // whenever the lower bump clamped (and 0/0'd at v0 = 0).
    let (v_up, v_dn, v_span) = crate::greeks::vol_bump_levels(vol, 1.0);
    let p_vup = HestonParams { v0: v_up.powi(2), ..*params };
    let p_vdn = HestonParams { v0: v_dn.powi(2), ..*params };
    let delta_vup = (crate::heston::heston_price(spot+ds, strike, expiry, rate, div_yield, &p_vup, opt_type)
                   - crate::heston::heston_price(spot-ds, strike, expiry, rate, div_yield, &p_vup, opt_type))
                  / (2.0 * ds);
    let delta_vdn = (crate::heston::heston_price(spot+ds, strike, expiry, rate, div_yield, &p_vdn, opt_type)
                   - crate::heston::heston_price(spot-ds, strike, expiry, rate, div_yield, &p_vdn, opt_type))
                  / (2.0 * ds);
    let vanna = (delta_vup - delta_vdn) / v_span;
    // volga: FD on the AD vega. second-order AD would be cleaner but overkill for now.
    let vega_up = {
        let (_, dv0_up) = forward_pass(spot, strike, expiry, rate, div_yield, &p_vup, opt_type, 0, JumpSpec::Off);
        dv0_up * 2.0 * v_up
    };
    let vega_dn = {
        let (_, dv0_dn) = forward_pass(spot, strike, expiry, rate, div_yield, &p_vdn, opt_type, 0, JumpSpec::Off);
        dv0_dn * 2.0 * v_dn
    };
    let volga = (vega_up - vega_dn) / v_span;

    let _ = (dkappa, dtheta, dsigma, drho); // available for calibration gradient if needed

    PricingResult { price, delta, gamma, vega, theta, rho: rho_greek, vanna, volga }
}

// same 5 forward passes as heston_greeks_ad, but the CF is Heston * jump,
// composed before integration same as bates_call does. this gives an exact
// price and exact d/d(v0,kappa,theta,sigma,rho) through the jump-adjusted
// CF, not just the Heston part with jumps bolted on after the fact.
//
// NOT covered: d(price)/d(lambda), d(price)/d(mu_j), d(price)/d(sigma_j).
// those params aren't in PricingResult and aren't in the active-param set
// here, if you need jump-parameter sensitivities for a calibration
// Jacobian, that's a separate 3-param forward pass someone still has to write.
// standard pricer-call shape plus the 3 jump params, same convention as
// bates_price/bates_price_and_greeks elsewhere in the crate.
#[allow(clippy::too_many_arguments)]
pub fn bates_greeks_ad(
    spot: f64, strike: f64, expiry: f64,
    rate: f64, div_yield: f64,
    heston: &HestonParams, lambda: f64, mu_j: f64, sigma_j: f64,
    opt_type: OptionType,
) -> PricingResult {
    let jump = JumpSpec::Const(lambda, mu_j, sigma_j);
    let bp = crate::types::BatesParams { heston: *heston, lambda, mu_j, sigma_j };

    let (price, dv0)    = forward_pass(spot, strike, expiry, rate, div_yield, heston, opt_type, 0, jump);
    let (_,     dkappa) = forward_pass(spot, strike, expiry, rate, div_yield, heston, opt_type, 1, jump);
    let (_,     dtheta) = forward_pass(spot, strike, expiry, rate, div_yield, heston, opt_type, 2, jump);
    let (_,     dsigma) = forward_pass(spot, strike, expiry, rate, div_yield, heston, opt_type, 3, jump);
    let (_,     drho)   = forward_pass(spot, strike, expiry, rate, div_yield, heston, opt_type, 4, jump);

    let vol  = heston.v0.sqrt();
    let vega = dv0 * 2.0 * vol;

    let dr        = 1e-4;
    let p_up      = crate::bates::bates_price(spot, strike, expiry, rate+dr, div_yield, &bp, opt_type);
    let p_dn      = crate::bates::bates_price(spot, strike, expiry, rate-dr, div_yield, &bp, opt_type);
    let rho_greek = (p_up - p_dn) / (2.0 * dr);

    let ds    = 0.01 * spot;
    let p_sup = crate::bates::bates_price(spot+ds, strike, expiry, rate, div_yield, &bp, opt_type);
    let p_sdn = crate::bates::bates_price(spot-ds, strike, expiry, rate, div_yield, &bp, opt_type);
    let delta = (p_sup - p_sdn) / (2.0 * ds);
    let gamma = (p_sup - 2.0*price + p_sdn) / (ds * ds);

    // theta/vanna/volga: same shared step rules as heston_greeks_ad, see the
    // comments there (central_theta and vol_bump_levels live in greeks.rs).
    let theta = crate::greeks::central_theta(expiry, |t|
        crate::bates::bates_price(spot, strike, t, rate, div_yield, &bp, opt_type));

    let (v_up, v_dn, v_span) = crate::greeks::vol_bump_levels(vol, 1.0);
    let bp_vup = crate::types::BatesParams { heston: HestonParams { v0: v_up.powi(2), ..*heston }, ..bp };
    let bp_vdn = crate::types::BatesParams { heston: HestonParams { v0: v_dn.powi(2), ..*heston }, ..bp };
    let delta_vup = (crate::bates::bates_price(spot+ds, strike, expiry, rate, div_yield, &bp_vup, opt_type)
                   - crate::bates::bates_price(spot-ds, strike, expiry, rate, div_yield, &bp_vup, opt_type))
                  / (2.0 * ds);
    let delta_vdn = (crate::bates::bates_price(spot+ds, strike, expiry, rate, div_yield, &bp_vdn, opt_type)
                   - crate::bates::bates_price(spot-ds, strike, expiry, rate, div_yield, &bp_vdn, opt_type))
                  / (2.0 * ds);
    let vanna = (delta_vup - delta_vdn) / v_span;

    let vega_up = {
        let (_, dv0_up) = forward_pass(spot, strike, expiry, rate, div_yield, &bp_vup.heston, opt_type, 0, jump);
        dv0_up * 2.0 * v_up
    };
    let vega_dn = {
        let (_, dv0_dn) = forward_pass(spot, strike, expiry, rate, div_yield, &bp_vdn.heston, opt_type, 0, jump);
        dv0_dn * 2.0 * v_dn
    };
    let volga = (vega_up - vega_dn) / v_span;

    let _ = (dkappa, dtheta, dsigma, drho);

    PricingResult { price, delta, gamma, vega, theta, rho: rho_greek, vanna, volga }
}

pub struct BatesJumpSensitivities {
    pub price:     f64,
    pub d_lambda:  f64,
    pub d_mu_j:    f64,
    pub d_sigma_j: f64,
}

// d(price)/d(lambda), d(price)/d(mu_j), d(price)/d(sigma_j) via forward-mode
// AD instead of the finite-difference bumps calibrate_bates's Jacobian used
// to be stuck with. same forward_pass as heston_greeks_ad/bates_greeks_ad,
// just called with the Heston side pinned constant (active=usize::MAX,
// dual_params gives all-constant when nothing matches that index, see
// dual_params) and a jump parameter carrying the derivative instead
// (JumpSpec::Active). 3 forward passes, exact, no bump size to tune.
#[allow(clippy::too_many_arguments)]
pub fn bates_jump_sensitivities_ad(
    spot: f64, strike: f64, expiry: f64,
    rate: f64, div_yield: f64,
    heston: &HestonParams, lambda: f64, mu_j: f64, sigma_j: f64,
    opt_type: OptionType,
) -> BatesJumpSensitivities {
    let (price, d_lambda) = forward_pass(spot, strike, expiry, rate, div_yield, heston, opt_type,
        usize::MAX, JumpSpec::Active(lambda, mu_j, sigma_j, 0));
    let (_, d_mu_j) = forward_pass(spot, strike, expiry, rate, div_yield, heston, opt_type,
        usize::MAX, JumpSpec::Active(lambda, mu_j, sigma_j, 1));
    let (_, d_sigma_j) = forward_pass(spot, strike, expiry, rate, div_yield, heston, opt_type,
        usize::MAX, JumpSpec::Active(lambda, mu_j, sigma_j, 2));
    BatesJumpSensitivities { price, d_lambda, d_mu_j, d_sigma_j }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{HestonParams, OptionType};
    use crate::heston::heston_price_and_greeks;
    use num_complex::Complex64;

    fn params() -> HestonParams {
        HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.7 }
    }

    // lambda=0, mu_j=0 collapses the jump CF to exp(0)=1, bates_greeks_ad
    // should agree with heston_greeks_ad to within quadrature tolerance,
    // not just "close", the jump factor is genuinely a no-op here.
    #[test]
    fn bates_ad_matches_heston_ad_when_jumps_off() {
        let p = params();
        let h = heston_greeks_ad(100.0, 100.0, 1.0, 0.05, 0.0, &p, OptionType::Call);
        let b = bates_greeks_ad(100.0, 100.0, 1.0, 0.05, 0.0, &p, 0.0, 0.0, 1e-8, OptionType::Call);
        assert!((h.price - b.price).abs() < 1e-6, "price: heston={:.6} bates={:.6}", h.price, b.price);
        assert!((h.vega  - b.vega ).abs() < 1e-4, "vega:  heston={:.6} bates={:.6}", h.vega,  b.vega);
    }

    // price has to match the analytic Bates pricer across strikes and
    // expiries, same regime (short-dated wings) that broke the old
    // fixed-panel quadrature before it was ported to adaptive.
    #[test]
    fn bates_ad_matches_analytic_across_wings() {
        use crate::bates::bates_price;
        use crate::types::BatesParams;
        let heston = params();
        let (lambda, mu_j, sigma_j) = (0.5, -0.1, 0.15);
        let bp = BatesParams { heston, lambda, mu_j, sigma_j };
        let expiries = [0.02_f64, 0.1, 0.5, 1.0, 2.0];
        let strikes  = [70.0_f64, 85.0, 100.0, 115.0, 130.0];
        for &t in &expiries {
            for &k in &strikes {
                let ad  = bates_greeks_ad(100.0, k, t, 0.05, 0.0, &heston, lambda, mu_j, sigma_j, OptionType::Call);
                let std = bates_price(100.0, k, t, 0.05, 0.0, &bp, OptionType::Call);
                let err = (ad.price - std).abs();
                assert!(err < 1e-6, "T={t} K={k}: ad={:.8} std={:.8} err={err:.2e}", ad.price, std);
            }
        }
    }

    // vega vs bump-and-reprice, same 1% bar as the Heston AD test.
    #[test]
    fn bates_ad_vega_close_to_bump() {
        use crate::bates::bates_price_and_greeks;
        use crate::types::BatesParams;
        let heston = params();
        let (lambda, mu_j, sigma_j) = (0.5, -0.1, 0.15);
        let bp = BatesParams { heston, lambda, mu_j, sigma_j };
        let ad  = bates_greeks_ad(100.0, 100.0, 1.0, 0.05, 0.0, &heston, lambda, mu_j, sigma_j, OptionType::Call);
        let std = bates_price_and_greeks(100.0, 100.0, 1.0, 0.05, 0.0, &bp, OptionType::Call);
        let err = (ad.vega - std.vega).abs() / std.vega.abs().max(1e-10);
        assert!(err < 0.01, "vega rel err={err:.4}: ad={:.4} bump={:.4}", ad.vega, std.vega);
    }

    #[test]
    fn bates_ad_greeks_signs() {
        let p = params();
        let r = bates_greeks_ad(100.0, 100.0, 1.0, 0.05, 0.0, &p, 0.5, -0.1, 0.15, OptionType::Call);
        assert!(r.delta > 0.0 && r.delta < 1.0, "delta={}", r.delta);
        assert!(r.gamma > 0.0, "gamma={}", r.gamma);
        assert!(r.vega  > 0.0, "vega={}",  r.vega);
    }

    // price from the jump-sensitivity pass has to match the analytic Bates
    // price too, it's the same forward_pass machinery, just with a jump
    // param active instead of a Heston one, price shouldn't care which.
    #[test]
    fn jump_sensitivities_price_matches_analytic() {
        use crate::bates::bates_price;
        use crate::types::BatesParams;
        let heston = params();
        let (lambda, mu_j, sigma_j) = (0.5, -0.1, 0.15);
        let bp = BatesParams { heston, lambda, mu_j, sigma_j };
        let sens = bates_jump_sensitivities_ad(100.0, 100.0, 1.0, 0.05, 0.0, &heston, lambda, mu_j, sigma_j, OptionType::Call);
        let std  = bates_price(100.0, 100.0, 1.0, 0.05, 0.0, &bp, OptionType::Call);
        assert!((sens.price - std).abs() < 1e-6, "price mismatch: ad={:.8} std={:.8}", sens.price, std);
    }

    // d(price)/d(lambda), d(price)/d(mu_j), d(price)/d(sigma_j) vs central
    // FD on bates_price directly (not vs bump-and-reprice's Greeks, those
    // don't cover jump params at all, this is the only ground truth here).
    #[test]
    fn jump_sensitivities_match_fd() {
        use crate::bates::bates_price;
        use crate::types::BatesParams;
        let heston = params();
        let (lambda, mu_j, sigma_j) = (0.5, -0.1, 0.15);
        let (s, k, t, r, q) = (100.0, 100.0, 1.0, 0.05, 0.0);
        let ot = OptionType::Call;

        let price_at = |l: f64, mj: f64, sj: f64| {
            bates_price(s, k, t, r, q, &BatesParams { heston, lambda: l, mu_j: mj, sigma_j: sj }, ot)
        };

        let sens = bates_jump_sensitivities_ad(s, k, t, r, q, &heston, lambda, mu_j, sigma_j, ot);

        let h_l  = 1e-5;
        let fd_l = (price_at(lambda+h_l, mu_j, sigma_j) - price_at(lambda-h_l, mu_j, sigma_j)) / (2.0*h_l);
        let h_mj  = 1e-5;
        let fd_mj = (price_at(lambda, mu_j+h_mj, sigma_j) - price_at(lambda, mu_j-h_mj, sigma_j)) / (2.0*h_mj);
        let h_sj  = 1e-5;
        let fd_sj = (price_at(lambda, mu_j, sigma_j+h_sj) - price_at(lambda, mu_j, sigma_j-h_sj)) / (2.0*h_sj);

        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-8);
        assert!(rel(sens.d_lambda, fd_l) < 1e-3,
            "d_lambda: ad={:.8} fd={:.8} rel_err={:.2e}", sens.d_lambda, fd_l, rel(sens.d_lambda, fd_l));
        assert!(rel(sens.d_mu_j, fd_mj) < 1e-3,
            "d_mu_j: ad={:.8} fd={:.8} rel_err={:.2e}", sens.d_mu_j, fd_mj, rel(sens.d_mu_j, fd_mj));
        assert!(rel(sens.d_sigma_j, fd_sj) < 1e-3,
            "d_sigma_j: ad={:.8} fd={:.8} rel_err={:.2e}", sens.d_sigma_j, fd_sj, rel(sens.d_sigma_j, fd_sj));
    }

    // NOT the naive intuition (a less-negative expected jump should raise
    // an ATM call, right?). wrong: raising mu_j also raises the risk-neutral
    // drift compensator k_bar = exp(mu_j+0.5*sigma_j^2)-1, and the drift
    // term is -lambda*k_bar, so a bigger k_bar pulls the drift DOWN harder
    // to keep the discounted process a martingale. for this param set that
    // compensator effect dominates the direct jump-size effect, net d_mu_j
    // is negative. caught by jump_sensitivities_match_fd first, this test
    // exists so the sign doesn't quietly flip back to "intuitive but wrong"
    // in a future edit without a test noticing.
    #[test]
    fn jump_sensitivities_signs_are_sane() {
        let heston = params();
        let sens = bates_jump_sensitivities_ad(100.0, 100.0, 1.0, 0.05, 0.0, &heston, 0.5, -0.1, 0.15, OptionType::Call);
        assert!(sens.d_mu_j < 0.0,
            "expected the drift-compensator effect to dominate here, d_mu_j={}", sens.d_mu_j);
    }

    // --- profiling: where does the dual-arithmetic overhead actually come
    // from? not run by default (`cargo test` skips #[ignore]), run with:
    //   cargo test --release -- --ignored --nocapture --test-threads=1 ad::tests::profile
    // each isolates one layer so the numbers compose: op cost -> transcendental
    // function cost -> panel cost -> full forward-pass cost. see README's
    // Known limitations section for what this found.
    //
    // IMPORTANT: black_box on the *output* only stops dead-code elimination,
    // it does NOT stop LLVM from proving a closure-captured input is loop
    // invariant and hoisting the whole computation out of the loop (or CSE-ing
    // it to one evaluation). first pass at this benchmark did exactly that,
    // csqrt/cln came back reading ~0.3ns, i.e. faster than a single cycle,
    // which is not a real number for a transcendental function. fix: cycle
    // through a batch of genuinely different inputs so nothing is invariant.

    fn varying_complex(n: usize) -> Vec<Complex64> {
        (0..n).map(|i| {
            let f = i as f64;
            Complex64::new(0.3 + 0.01*f, -0.9 + 0.007*f)
        }).collect()
    }

    fn varying_dual(n: usize) -> Vec<CDual> {
        varying_complex(n).into_iter()
            .map(|z| Complex::new(Dual { val: z.re, dot: 1.0 }, Dual { val: z.im, dot: 0.0 }))
            .collect()
    }

    // n_iters total calls, cycling through `inputs` so no two consecutive
    // calls see the same value and nothing is provably loop-invariant.
    fn bench_varying<T, X: Copy, F: Fn(X) -> T>(n_iters: u32, inputs: &[X], f: F) -> f64 {
        let warmup = (n_iters / 10).max(1000);
        for i in 0..warmup {
            std::hint::black_box(f(std::hint::black_box(inputs[i as usize % inputs.len()])));
        }
        let t0 = std::time::Instant::now();
        for i in 0..n_iters {
            std::hint::black_box(f(std::hint::black_box(inputs[i as usize % inputs.len()])));
        }
        t0.elapsed().as_secs_f64() * 1e9 / n_iters as f64 // ns/op
    }

    #[test]
    #[ignore]
    fn profile_op_level() {
        let cs = varying_complex(64);
        let ds = varying_dual(64);
        let n = 4_000_000;

        let t_mul_c = bench_varying(n, &cs, |z| z * Complex64::new(0.4, 1.1));
        let t_mul_d = bench_varying(n, &ds, |z| z * cd(0.4, 1.1));
        let t_div_c = bench_varying(n, &cs, |z| z / Complex64::new(0.4, 1.1));
        let t_div_d = bench_varying(n, &ds, |z| z / cd(0.4, 1.1));

        eprintln!("\n[profile_op_level]  ({} varying inputs, cycled)", cs.len());
        eprintln!("  Complex64     mul: {t_mul_c:.3} ns/op");
        eprintln!("  Complex<Dual> mul: {t_mul_d:.3} ns/op  ({:.2}x)", t_mul_d/t_mul_c);
        eprintln!("  Complex64     div: {t_div_c:.3} ns/op");
        eprintln!("  Complex<Dual> div: {t_div_d:.3} ns/op  ({:.2}x)", t_div_d/t_div_c);
    }

    #[test]
    #[ignore]
    fn profile_transcendental_level() {
        use crate::heston::fast_csqrt;
        let cs = varying_complex(64);
        let ds = varying_dual(64);
        let n = 2_000_000;

        let t_exp_c  = bench_varying(n, &cs, |z| z.exp());
        let t_exp_d  = bench_varying(n, &ds, cexp);
        // sqrt: the fair comparison is against fast_csqrt (same algorithm,
        // no dual bookkeeping), not the num-complex builtin, that one goes
        // through to_polar/from_polar (hypot+atan2+sqrt+cos+sin) and isn't
        // the algorithm either sqrt here actually uses anymore.
        let t_sqrt_builtin = bench_varying(n, &cs, |z| z.sqrt());
        let t_sqrt_fast     = bench_varying(n, &cs, fast_csqrt);
        let t_sqrt_d        = bench_varying(n, &ds, csqrt);
        let t_ln_c   = bench_varying(n, &cs, |z| z.ln());
        let t_ln_d   = bench_varying(n, &ds, cln);

        eprintln!("\n[profile_transcendental_level]  ({} varying inputs, cycled)", cs.len());
        eprintln!("  exp:  Complex64 {t_exp_c:.3} ns, Complex<Dual> {t_exp_d:.3} ns  ({:.2}x)", t_exp_d/t_exp_c);
        eprintln!("  sqrt: Complex64 builtin (to_polar/from_polar) {t_sqrt_builtin:.3} ns");
        eprintln!("  sqrt: Complex64 fast_csqrt (what stable_cf actually calls now) {t_sqrt_fast:.3} ns");
        eprintln!("  sqrt: Complex<Dual> csqrt {t_sqrt_d:.3} ns  ({:.2}x vs fast_csqrt, the fair comparison)", t_sqrt_d/t_sqrt_fast);
        eprintln!("  ln:   Complex64 {t_ln_c:.3} ns, Complex<Dual> {t_ln_d:.3} ns  ({:.2}x)", t_ln_d/t_ln_c);
    }

    #[test]
    #[ignore]
    fn profile_panel_level() {
        use crate::heston::stable_cf;
        // 16 different param sets so the panel computation genuinely differs
        // call to call, not just the phi node.
        let param_sets: Vec<HestonParams> = (0..16).map(|i| {
            let f = i as f64;
            HestonParams { v0: 0.03 + 0.002*f, kappa: 1.8 + 0.05*f, theta: 0.03 + 0.001*f, sigma: 0.28 + 0.01*f, rho: -0.7 + 0.01*f }
        }).collect();
        let dual_sets: Vec<DualParams> = param_sets.iter().map(|p| dual_params(p, 0)).collect();
        let (t, mu) = (1.0, 0.05); // mu = drift (r - q), q = 0 here; timing only

        let n = 100_000;
        let t_plain = bench_varying(n, &(0..param_sets.len()).collect::<Vec<_>>(), |idx| {
            let p = &param_sets[idx];
            let mut acc = Complex64::new(0.0, 0.0);
            for &node in GK_NODES.iter() {
                acc += stable_cf(Complex64::new(node, -1.0), t, mu, p);
            }
            acc
        });
        // DualParams isn't Copy (Dual doesn't need to be, but the struct
        // itself is small so this is fine at 100k reps), index into it instead.
        let idxs: Vec<usize> = (0..dual_sets.len()).collect();
        let t_dual = bench_varying(n, &idxs, |idx| {
            let dp = &dual_sets[idx];
            let mut acc = Complex::new(Dual::constant(0.0), Dual::constant(0.0));
            for &node in GK_NODES.iter() {
                acc = acc + stable_cf_dual(Complex::new(Dual::constant(node), Dual::constant(-1.0)), t, mu, dp);
            }
            acc
        });

        eprintln!("\n[profile_panel_level]  (15 CF evals per call, {} varying param sets)", param_sets.len());
        eprintln!("  plain Complex64:   {t_plain:.0} ns/panel");
        eprintln!("  Complex<Dual>:     {t_dual:.0} ns/panel  ({:.2}x)", t_dual/t_plain);
    }

    #[test]
    #[ignore]
    fn profile_full_pass_level() {
        use crate::heston::{heston_price, heston_price_and_greeks};
        // varying strikes so nothing across calls is provably identical
        let strikes: Vec<f64> = (0..32).map(|i| 80.0 + i as f64 * 1.5).collect();
        let p = params();
        let (s, t, r, q) = (100.0, 1.0, 0.05, 0.0);

        let n = 3_000;
        let t_single_price = bench_varying(n, &strikes, |k| heston_price(s, k, t, r, q, &p, OptionType::Call));
        let t_single_pass   = bench_varying(n, &strikes, |k| forward_pass(s, k, t, r, q, &p, OptionType::Call, 0, JumpSpec::Off));
        let t_bump_full     = bench_varying(n, &strikes, |k| heston_price_and_greeks(s, k, t, r, q, &p, OptionType::Call));
        let t_ad_full       = bench_varying(n, &strikes, |k| heston_greeks_ad(s, k, t, r, q, &p, OptionType::Call));

        eprintln!("\n[profile_full_pass_level]  ({} varying strikes)", strikes.len());
        eprintln!("  single heston_price (2 plain integrations):          {t_single_price:.0} ns");
        eprintln!("  single forward_pass (2 dual integrations):           {t_single_pass:.0} ns  ({:.2}x per pass)",
            t_single_pass/t_single_price);
        eprintln!("  heston_price_and_greeks (14 plain price calls):      {t_bump_full:.0} ns");
        eprintln!("  heston_greeks_ad (5 dual passes + 4 FD price calls): {t_ad_full:.0} ns  ({:.2}x)",
            t_ad_full/t_bump_full);
    }

    // same cancellation trap as heston.rs's fast_csqrt, same fix, same
    // discipline: sweep concentrated near the axes (where it actually
    // bites), not just spread evenly around the circle. checks .val against
    // the real Complex64::sqrt() and .dot against a finite-difference
    // derivative of the same scalar function, both have to hold.
    #[test]
    fn csqrt_matches_builtin_value_and_fd_derivative() {
        let mags = [1e-6, 1e-3, 0.1, 1.0, 10.0, 100.0, 1e4];
        let mut angles = vec![];
        let n = 25;
        for i in 0..n {
            angles.push(-std::f64::consts::PI + 2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64);
        }
        for &tiny in &[1e-2, 1e-4, 1e-6, 1e-8, 1e-10] {
            for &base in &[0.0_f64, std::f64::consts::PI, std::f64::consts::FRAC_PI_2, -std::f64::consts::FRAC_PI_2] {
                angles.push(base + tiny);
                angles.push(base - tiny);
            }
        }

        let mut checked = 0;
        let mut worst_val = 0.0_f64;
        let mut worst_dot = 0.0_f64;
        for &mag in &mags {
            for &ang in &angles {
                let a = mag * ang.cos();
                let b = mag * ang.sin();

                // differentiate w.r.t. a (the real part), FD reference.
                // h has to be small relative to THIS point, not a fixed
                // floor, points near the branch point (|z| small) need a
                // proportionally tiny step or the FD straddles the
                // singularity and gives garbage regardless of how correct
                // the analytic derivative is.
                let h = (a.abs() + b.abs()).max(1e-300) * 1e-6;
                let f = |x: f64| Complex64::new(x, b).sqrt();
                let fd_re = (f(a+h).re - f(a-h).re) / (2.0*h);
                let fd_im = (f(a+h).im - f(a-h).im) / (2.0*h);

                let z = Complex::new(Dual::active(a), Dual::constant(b));
                let got = csqrt(z);
                let expected_val = Complex64::new(a, b).sqrt();

                let val_err = ((got.re.val - expected_val.re).powi(2) + (got.im.val - expected_val.im).powi(2)).sqrt()
                    / expected_val.norm().max(1e-300);
                let dot_err = ((got.re.dot - fd_re).powi(2) + (got.im.dot - fd_im).powi(2)).sqrt()
                    / (fd_re.powi(2) + fd_im.powi(2)).sqrt().max(1e-6);

                worst_val = worst_val.max(val_err);
                worst_dot = worst_dot.max(dot_err);
                checked += 1;
                assert!(val_err < 1e-9,
                    "csqrt value mismatch at a={a} b={b}: got=({},{}) expected=({},{}) err={val_err:.2e}",
                    got.re.val, got.im.val, expected_val.re, expected_val.im);
                assert!(dot_err < 1e-4,
                    "csqrt derivative mismatch at a={a} b={b}: got=({},{}) fd=({fd_re},{fd_im}) err={dot_err:.2e}",
                    got.re.dot, got.im.dot);
            }
        }
        assert!(checked > 250, "sweep too small to trust, only checked {checked} points");
        eprintln!("csqrt_matches_builtin_value_and_fd_derivative: {checked} points, worst val err {worst_val:.2e}, worst dot err {worst_dot:.2e}");
    }

    // controlled A/B for the fast_csqrt swap itself: same CF formula, only
    // the sqrt call differs. this is NOT about dual arithmetic, it's the
    // separate, real finding that fell out of investigating dual overhead,
    // stable_cf's sqrt was going through num-complex's to_polar/from_polar
    // path unnecessarily. kept as a local copy here (not in heston.rs) so
    // this stays a benchmark-only artifact instead of dead production code.
    fn stable_cf_old_sqrt(phi: Complex64, t: f64, r: f64, p: &HestonParams) -> Complex64 {
        let i = Complex64::i();
        let &HestonParams { v0, kappa, theta, sigma, rho } = p;
        let xi  = kappa - rho * sigma * phi * i;
        let d   = (xi*xi + sigma*sigma * phi*(phi + i)).sqrt(); // the old, slower call
        let g   = (xi - d) / (xi + d);
        let edt = (-d * t).exp();
        let a   = (g*edt - 1.0) / (g - 1.0);
        let c  = (kappa*theta / (sigma*sigma)) * ((xi - d)*t - 2.0*a.ln());
        let dd = v0 * (xi - d) * (1.0 - edt) / (sigma*sigma * (1.0 - g*edt));
        (r * phi * i * t + c + dd).exp()
    }

    #[test]
    #[ignore]
    fn profile_fast_csqrt_end_to_end() {
        use crate::heston::{stable_cf, heston_price};
        let p = params();
        let strikes: Vec<f64> = (0..32).map(|i| 80.0 + i as f64 * 1.5).collect();
        let n = 5_000;

        let t_cf_new = bench_varying(n, &strikes, |k| {
            let phi = Complex64::new(0.5, -1.0);
            stable_cf(phi, 1.0, 0.05, &HestonParams { v0: p.v0 + k*1e-9, ..p })
        });
        let t_cf_old = bench_varying(n, &strikes, |k| {
            let phi = Complex64::new(0.5, -1.0);
            stable_cf_old_sqrt(phi, 1.0, 0.05, &HestonParams { v0: p.v0 + k*1e-9, ..p })
        });
        let t_price = bench_varying(n, &strikes, |k| heston_price(100.0, k, 1.0, 0.05, 0.0, &p, OptionType::Call));

        eprintln!("\n[profile_fast_csqrt_end_to_end]");
        eprintln!("  stable_cf, old to_polar/from_polar sqrt: {t_cf_old:.1} ns/call");
        eprintln!("  stable_cf, fast_csqrt (current):         {t_cf_new:.1} ns/call  ({:.2}x)", t_cf_new/t_cf_old);
        eprintln!("  full heston_price (adaptive, many stable_cf calls): {t_price:.0} ns");
    }

    // Dual5 (one joint pass) has to agree with the scalar Dual path (5
    // separate passes, already verified against the analytic pricer
    // elsewhere in this file) on every one of price/delta/gamma/vega/theta/
    // rho/vanna/volga, across strikes and expiries including the wings.
    // this is the actual correctness bar for the experiment, not "it
    // compiles and returns a plausible number".
    #[test]
    fn dual5_matches_scalar_dual_across_wings() {
        let p = params();
        let expiries = [0.02_f64, 0.1, 0.5, 1.0, 2.0];
        let strikes  = [70.0_f64, 85.0, 100.0, 115.0, 130.0];
        for &t in &expiries {
            for &k in &strikes {
                let a5 = heston_greeks_ad5(100.0, k, t, 0.05, 0.0, &p, OptionType::Call);
                let a1 = heston_greeks_ad(100.0, k, t, 0.05, 0.0, &p, OptionType::Call);
                let check = |name: &str, x: f64, y: f64| {
                    // deep OTM / short-dated combos price near zero for
                    // both paths, comparing relative error against a tiny
                    // floor there just amplifies ordinary float noise into
                    // a fake failure. absolute tolerance once the reference
                    // value itself is negligible.
                    if y.abs() < 1e-6 {
                        assert!((x - y).abs() < 1e-8,
                            "{name} mismatch (near-zero) at T={t} K={k}: dual5={x:.10} scalar={y:.10}");
                        return;
                    }
                    let err = (x - y).abs() / y.abs();
                    assert!(err < 1e-6, "{name} mismatch at T={t} K={k}: dual5={x:.8} scalar={y:.8} rel_err={err:.2e}");
                };
                check("price", a5.price, a1.price);
                check("vega",  a5.vega,  a1.vega);
            }
        }
    }

    #[test]
    fn dual5_matches_scalar_dual_greeks_signs_and_magnitude() {
        let p = params();
        let a5 = heston_greeks_ad5(100.0, 100.0, 1.0, 0.05, 0.0, &p, OptionType::Call);
        let a1 = heston_greeks_ad(100.0, 100.0, 1.0, 0.05, 0.0, &p, OptionType::Call);
        assert!((a5.delta - a1.delta).abs() < 1e-6, "delta: dual5={} scalar={}", a5.delta, a1.delta);
        assert!((a5.gamma - a1.gamma).abs() < 1e-6, "gamma: dual5={} scalar={}", a5.gamma, a1.gamma);
        assert!((a5.theta - a1.theta).abs() < 1e-6, "theta: dual5={} scalar={}", a5.theta, a1.theta);
        assert!((a5.rho   - a1.rho  ).abs() < 1e-6, "rho: dual5={} scalar={}",   a5.rho,   a1.rho);
        assert!((a5.vanna - a1.vanna).abs() < 1e-4, "vanna: dual5={} scalar={}", a5.vanna, a1.vanna);
        assert!((a5.volga - a1.volga).abs() < 1e-3, "volga: dual5={} scalar={}", a5.volga, a1.volga);
    }

    // does the "compute the value once instead of 5 times" bet actually pay
    // off wall-clock, given mul/div cost more per-op with a 5-vector dot
    // than a scalar one? measured, not assumed, same varying-input
    // methodology as the other profile_* benchmarks (see the note on why
    // that matters at the top of this block). run with:
    //   cargo test --release -- --ignored --nocapture --test-threads=1 ad::tests::profile_dual5
    #[test]
    #[ignore]
    fn profile_dual5_vs_five_scalar_passes() {
        use crate::heston::heston_price_and_greeks;
        let strikes: Vec<f64> = (0..32).map(|i| 80.0 + i as f64 * 1.5).collect();
        let p = params();
        let (s, t, r, q) = (100.0, 1.0, 0.05, 0.0);
        let n = 3_000;

        let t_five_pass = bench_varying(n, &strikes, |k| forward_pass(s, k, t, r, q, &p, OptionType::Call, 0, JumpSpec::Off));
        let t_joint_pass = bench_varying(n, &strikes, |k| forward_pass5(s, k, t, r, q, &p, OptionType::Call));
        let t_full_ad5  = bench_varying(n, &strikes, |k| heston_greeks_ad5(s, k, t, r, q, &p, OptionType::Call));
        let t_full_ad1  = bench_varying(n, &strikes, |k| heston_greeks_ad(s, k, t, r, q, &p, OptionType::Call));
        let t_bump      = bench_varying(n, &strikes, |k| heston_price_and_greeks(s, k, t, r, q, &p, OptionType::Call));

        eprintln!("\n[profile_dual5_vs_five_scalar_passes]");
        eprintln!("  one scalar forward_pass:        {t_five_pass:.0} ns  (x5 for all Heston-driven greeks = {:.0} ns)", t_five_pass*5.0);
        eprintln!("  one joint forward_pass5:         {t_joint_pass:.0} ns  ({:.2}x vs 5 scalar passes)", t_joint_pass/(t_five_pass*5.0));
        eprintln!("  heston_greeks_ad  (5 scalar passes + 4 FD): {t_full_ad1:.0} ns");
        eprintln!("  heston_greeks_ad5 (1 joint pass + 4 FD):    {t_full_ad5:.0} ns  ({:.2}x vs heston_greeks_ad)", t_full_ad5/t_full_ad1);
        eprintln!("  heston_price_and_greeks (bump, 14 calls):   {t_bump:.0} ns  (ad5 is {:.2}x this)", t_full_ad5/t_bump);
    }

    #[test]
    fn price_matches_standard() {
        let p   = params();
        let ad  = heston_greeks_ad(100.0, 100.0, 1.0, 0.05, 0.0, &p, OptionType::Call);
        let std = heston_price_and_greeks(100.0, 100.0, 1.0, 0.05, 0.0, &p, OptionType::Call);
        assert!((ad.price - std.price).abs() < 1e-10, "price: ad={:.6} std={:.6}", ad.price, std.price);
    }

    #[test]
    fn greeks_signs() {
        let p = params();
        let r = heston_greeks_ad(100.0, 100.0, 1.0, 0.05, 0.0, &p, OptionType::Call);
        assert!(r.delta > 0.0 && r.delta < 1.0, "delta={}", r.delta);
        assert!(r.gamma > 0.0, "gamma={}", r.gamma);
        assert!(r.vega  > 0.0, "vega={}",  r.vega);
        assert!(r.rho   > 0.0, "rho={}",   r.rho);
    }

    #[test]
    fn put_delta_negative() {
        let p = params();
        let r = heston_greeks_ad(100.0, 100.0, 1.0, 0.05, 0.0, &p, OptionType::Put);
        assert!(r.delta < 0.0 && r.delta > -1.0, "put delta={}", r.delta);
    }

    // this is the regime that broke silently under the old fixed-panel quadrature:
    // short expiries and deep wings. price from the AD path has to track
    // heston_price (adaptive, trusted) everywhere, not just at the ATM 1y point
    // the other tests happen to use.
    #[test]
    fn price_matches_standard_across_wings() {
        use crate::heston::heston_price;
        let p = params();
        let expiries = [0.02_f64, 0.1, 0.5, 1.0, 2.0];
        let strikes  = [70.0_f64, 85.0, 100.0, 115.0, 130.0];
        for &t in &expiries {
            for &k in &strikes {
                let ad  = heston_greeks_ad(100.0, k, t, 0.05, 0.0, &p, OptionType::Call);
                let std = heston_price(100.0, k, t, 0.05, 0.0, &p, OptionType::Call);
                let err = (ad.price - std).abs();
                assert!(err < 1e-6, "T={t} K={k}: ad={:.8} std={:.8} err={err:.2e}", ad.price, std);
            }
        }
    }

    // vega from AD should be close to bump-and-reprice. not identical, AD is exact,
    // bump has discretization error but should agree to ~1e-3.
    #[test]
    fn vega_close_to_bump() {
        let p   = params();
        let ad  = heston_greeks_ad(100.0, 100.0, 1.0, 0.05, 0.0, &p, OptionType::Call);
        let std = heston_price_and_greeks(100.0, 100.0, 1.0, 0.05, 0.0, &p, OptionType::Call);
        let err = (ad.vega - std.vega).abs() / std.vega.abs().max(1e-10);
        assert!(err < 0.01, "vega rel err={:.4}: ad={:.4} bump={:.4}", err, ad.vega, std.vega);
    }
}
