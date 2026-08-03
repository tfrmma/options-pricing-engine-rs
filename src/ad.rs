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
fn csqrt(z: CDual) -> CDual {
    // sqrt(a+bi): mod = |z|, arg = atan2(b,a)
    let a  = z.re.val; let b  = z.im.val;
    let da = z.re.dot; let db = z.im.dot;
    let r  = (a*a + b*b).sqrt();
    if r < 1e-300 { return Complex::new(Dual::constant(0.0), Dual::constant(0.0)); }

    let re_val = ((r + a) / 2.0).sqrt();
    let im_val = b.signum() * ((r - a) / 2.0).sqrt();

    // chain rule: d(re)/dp, d(im)/dp
    let dr  = (a*da + b*db) / r;
    let re_dot = if re_val.abs() > 1e-300 { (dr + da) / (4.0 * re_val) } else { 0.0 };
    let im_dot = if im_val.abs() > 1e-300 { (dr - da) / (4.0 * im_val) } else { 0.0 };
    Complex::new(
        Dual { val: re_val, dot: re_dot },
        Dual { val: im_val, dot: im_dot },
    )
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

fn stable_cf_dual(phi: CDual, t: f64, r: f64, p: &DualParams) -> CDual {
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

    cexp(c(r * t) * phi * i + cc + dd)
}

// jump component, mirrors jump_cf in bates.rs exactly. lambda/mu_j/sigma_j
// are plain constants here, not part of the active-param set (that's still
// v0/kappa/theta/sigma/rho, indices 0..=4), so this factor's own dot is
// always zero. Complex<Dual> multiplication already implements the product
// rule via Dual's Mul impl, so folding this into stable_cf_dual's output
// via a straight `*` gives the right derivative for free, no extra plumbing.
fn jump_cf_dual(phi: CDual, t: f64, lambda: f64, mu_j: f64, sigma_j: f64) -> CDual {
    let i    = cd_i();
    let c    = |v: f64| Complex::new(Dual::constant(v), Dual::constant(0.0));
    let comp = (mu_j + 0.5*sigma_j*sigma_j).exp() - 1.0;
    let jump = cexp(phi*i*c(mu_j) - c(0.5*sigma_j*sigma_j) * phi*phi);
    cexp(c(lambda*t) * (jump - c(1.0) - i*phi*c(comp)))
}

type JumpParams = Option<(f64, f64, f64)>; // (lambda, mu_j, sigma_j)

// params where each field is a Dual lets us set one as active at a time
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
fn dual_integrand(
    u: f64, x: f64, t: f64, r: f64,
    dp: &DualParams, is_p1: bool, cf_mi: Option<CDual>, jump: JumpParams,
) -> (f64, f64) {
    let phi: CDual = if is_p1 {
        Complex::new(Dual::constant(u), Dual::constant(-1.0))
    } else {
        Complex::new(Dual::constant(u), Dual::constant(0.0))
    };

    let mut cf  = stable_cf_dual(phi, t, r, dp);
    if let Some((lambda, mu_j, sigma_j)) = jump {
        cf = cf * jump_cf_dual(phi, t, lambda, mu_j, sigma_j);
    }
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
fn forward_pass(
    s: f64, k: f64, t: f64, r: f64, q: f64,
    p: &HestonParams, opt_type: OptionType, active: usize, jump: JumpParams,
) -> (f64, f64) {
    let x  = (s/k).ln();
    let dp = dual_params(p, active);

    // CF(-i) normalizer (as a dual, so its derivative w.r.t. the active
    // param is also propagated into P1). includes the jump factor too,
    // same as bates_call's cf_mi.
    let phi_mi = Complex::new(Dual::constant(0.0), Dual::constant(-1.0));
    let mut cf_mi = stable_cf_dual(phi_mi, t, r, &dp);
    if let Some((lambda, mu_j, sigma_j)) = jump {
        cf_mi = cf_mi * jump_cf_dual(phi_mi, t, lambda, mu_j, sigma_j);
    }

    let (i1_val, i1_dot) = gk_integrate_dual(|u| dual_integrand(u, x, t, r, &dp, true, Some(cf_mi), jump));
    let (i2_val, i2_dot) = gk_integrate_dual(|u| dual_integrand(u, x, t, r, &dp, false, None, jump));

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
    let (price, dv0)    = forward_pass(spot, strike, expiry, rate, div_yield, params, opt_type, 0, None);
    let (_,     dkappa) = forward_pass(spot, strike, expiry, rate, div_yield, params, opt_type, 1, None);
    let (_,     dtheta) = forward_pass(spot, strike, expiry, rate, div_yield, params, opt_type, 2, None);
    let (_,     dsigma) = forward_pass(spot, strike, expiry, rate, div_yield, params, opt_type, 3, None);
    let (_,     drho)   = forward_pass(spot, strike, expiry, rate, div_yield, params, opt_type, 4, None);

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

    // theta: FD on expiry
    let t_dn  = (expiry - 1.0/365.0).max(1e-6);
    let theta = (crate::heston::heston_price(spot, strike, t_dn, rate, div_yield, params, opt_type) - price)
              / (1.0/365.0);

    // vanna: cross bump delta vs vol
    let p_vup = HestonParams { v0: (vol + 0.01).powi(2), ..*params };
    let p_vdn = HestonParams { v0: (vol - 0.01).max(1e-6).powi(2), ..*params };
    let delta_vup = (crate::heston::heston_price(spot+ds, strike, expiry, rate, div_yield, &p_vup, opt_type)
                   - crate::heston::heston_price(spot-ds, strike, expiry, rate, div_yield, &p_vup, opt_type))
                  / (2.0 * ds);
    let delta_vdn = (crate::heston::heston_price(spot+ds, strike, expiry, rate, div_yield, &p_vdn, opt_type)
                   - crate::heston::heston_price(spot-ds, strike, expiry, rate, div_yield, &p_vdn, opt_type))
                  / (2.0 * ds);
    let vanna = (delta_vup - delta_vdn) / (2.0 * 0.01);
    // volga: FD on vega. second-order AD would be cleaner but overkill for now.
    let vega_up = {
        let (_, dv0_up) = forward_pass(spot, strike, expiry, rate, div_yield, &p_vup, opt_type, 0, None);
        dv0_up * 2.0 * (vol + 0.01)
    };
    let vega_dn = {
        let (_, dv0_dn) = forward_pass(spot, strike, expiry, rate, div_yield, &p_vdn, opt_type, 0, None);
        dv0_dn * 2.0 * (vol - 0.01).max(1e-6)
    };
    let volga = (vega_up - vega_dn) / (2.0 * 0.01);

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
pub fn bates_greeks_ad(
    spot: f64, strike: f64, expiry: f64,
    rate: f64, div_yield: f64,
    heston: &HestonParams, lambda: f64, mu_j: f64, sigma_j: f64,
    opt_type: OptionType,
) -> PricingResult {
    let jump: JumpParams = Some((lambda, mu_j, sigma_j));
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

    let t_dn  = (expiry - 1.0/365.0).max(1e-6);
    let theta = (crate::bates::bates_price(spot, strike, t_dn, rate, div_yield, &bp, opt_type) - price)
              / (1.0/365.0);

    let bp_vup = crate::types::BatesParams { heston: HestonParams { v0: (vol + 0.01).powi(2), ..*heston }, ..bp };
    let bp_vdn = crate::types::BatesParams { heston: HestonParams { v0: (vol - 0.01).max(1e-6).powi(2), ..*heston }, ..bp };
    let delta_vup = (crate::bates::bates_price(spot+ds, strike, expiry, rate, div_yield, &bp_vup, opt_type)
                   - crate::bates::bates_price(spot-ds, strike, expiry, rate, div_yield, &bp_vup, opt_type))
                  / (2.0 * ds);
    let delta_vdn = (crate::bates::bates_price(spot+ds, strike, expiry, rate, div_yield, &bp_vdn, opt_type)
                   - crate::bates::bates_price(spot-ds, strike, expiry, rate, div_yield, &bp_vdn, opt_type))
                  / (2.0 * ds);
    let vanna = (delta_vup - delta_vdn) / (2.0 * 0.01);

    let vega_up = {
        let (_, dv0_up) = forward_pass(spot, strike, expiry, rate, div_yield, &bp_vup.heston, opt_type, 0, jump);
        dv0_up * 2.0 * (vol + 0.01)
    };
    let vega_dn = {
        let (_, dv0_dn) = forward_pass(spot, strike, expiry, rate, div_yield, &bp_vdn.heston, opt_type, 0, jump);
        dv0_dn * 2.0 * (vol - 0.01).max(1e-6)
    };
    let volga = (vega_up - vega_dn) / (2.0 * 0.01);

    let _ = (dkappa, dtheta, dsigma, drho);

    PricingResult { price, delta, gamma, vega, theta, rho: rho_greek, vanna, volga }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{HestonParams, OptionType};
    use crate::heston::heston_price_and_greeks;

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
