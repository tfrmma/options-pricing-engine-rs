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
        let t_exp_d  = bench_varying(n, &ds, |z| cexp(z));
        // sqrt: the fair comparison is against fast_csqrt (same algorithm,
        // no dual bookkeeping), not the num-complex builtin, that one goes
        // through to_polar/from_polar (hypot+atan2+sqrt+cos+sin) and isn't
        // the algorithm either sqrt here actually uses anymore.
        let t_sqrt_builtin = bench_varying(n, &cs, |z| z.sqrt());
        let t_sqrt_fast     = bench_varying(n, &cs, |z| fast_csqrt(z));
        let t_sqrt_d        = bench_varying(n, &ds, |z| csqrt(z));
        let t_ln_c   = bench_varying(n, &cs, |z| z.ln());
        let t_ln_d   = bench_varying(n, &ds, |z| cln(z));

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
        let (t, r) = (1.0, 0.05);

        let n = 100_000;
        let t_plain = bench_varying(n, &(0..param_sets.len()).collect::<Vec<_>>(), |idx| {
            let p = &param_sets[idx];
            let mut acc = Complex64::new(0.0, 0.0);
            for &node in GK_NODES.iter() {
                acc += stable_cf(Complex64::new(node, -1.0), t, r, p);
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
                acc = acc + stable_cf_dual(Complex::new(Dual::constant(node), Dual::constant(-1.0)), t, r, dp);
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
        let t_single_pass   = bench_varying(n, &strikes, |k| forward_pass(s, k, t, r, q, &p, OptionType::Call, 0, None));
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
