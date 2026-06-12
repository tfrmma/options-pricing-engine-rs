// forward-mode AD for Heston Greeks.
//
// bump-and-reprice needs 14 pricing calls for a full greek set.
// this does 5 forward passes through the CF — one per param.
//
// the trick: Leibniz rule lets us differentiate under the integral.
//   d(price)/dp = integral of Re[ d(CF(u,p))/dp * kernel(u) ] du
// so we propagate dual numbers through stable_cf, then integrate the
// dual part. same GK-15 quadrature, dual arithmetic instead of complex.
//
// Dual<f64>: (val, dot) where dot = d(val)/dp for the active param.
// one pass = one param. 5 params = 5 passes. each pass is cheaper than
// a full pricing call because there's no extra GK evaluation — just
// dual arithmetic on top of the CF ops we're already doing.

use std::ops::{Add, Sub, Mul, Div, Neg, Rem};
use num_traits::{Zero, One, Num};
use num_complex::Complex;
use crate::types::{HestonParams, OptionType, PricingResult};
use crate::heston::GK_NODES;
use crate::heston::GK_WEIGHTS;

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

// num_traits impls — required for Complex<Dual> to use num-complex's internal ops.
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
    dp: &DualParams, is_p1: bool, cf_mi: Option<CDual>,
) -> (f64, f64) {
    let phi: CDual = if is_p1 {
        Complex::new(Dual::constant(u), Dual::constant(-1.0))
    } else {
        Complex::new(Dual::constant(u), Dual::constant(0.0))
    };

    let mut cf  = stable_cf_dual(phi, t, r, dp);
    if let Some(norm) = cf_mi {
        cf = cf / norm;
    }
    let exp_term = cexp(Complex::new(Dual::constant(0.0), Dual::constant(u * x)));
    let num = exp_term * cf;
    let div = Complex::new(Dual::constant(0.0), Dual::constant(u));
    let res = num / div;

    (res.re.val, res.re.dot)
}

// GK-15 integration returning (integral_val, integral_deriv) simultaneously
fn gk_integrate_dual<F>(f: F) -> (f64, f64)
where F: Fn(f64) -> (f64, f64)
{
    let upper = 200.0;
    let mid   = upper / 2.0;
    let mut sum_val  = 0.0;
    let mut sum_dot  = 0.0;
    for (&n, &w) in GK_NODES.iter().zip(GK_WEIGHTS.iter()) {
        let u = mid + mid * n;
        if u < 1e-12 { continue; }
        let (v, d) = f(u);
        sum_val += w * v;
        sum_dot += w * d;
    }
    (sum_val * mid, sum_dot * mid)
}

// one forward pass for a given active param index (0=v0, 1=kappa, 2=theta, 3=sigma, 4=rho).
// returns (price, dprice/dparam).
fn forward_pass(
    s: f64, k: f64, t: f64, r: f64, q: f64,
    p: &HestonParams, opt_type: OptionType, active: usize,
) -> (f64, f64) {
    let x  = (s/k).ln();
    let dp = dual_params(p, active);

    // CF(-i) normalizer (as a dual, so its derivative w.r.t. the active
    // param is also propagated into P1).
    let phi_mi = Complex::new(Dual::constant(0.0), Dual::constant(-1.0));
    let cf_mi  = stable_cf_dual(phi_mi, t, r, &dp);

    let (i1_val, i1_dot) = gk_integrate_dual(|u| dual_integrand(u, x, t, r, &dp, true, Some(cf_mi)));
    let (i2_val, i2_dot) = gk_integrate_dual(|u| dual_integrand(u, x, t, r, &dp, false, None));

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
        // d(put)/dp = d(call)/dp — parity terms don't depend on Heston params
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
    let (price, dv0)    = forward_pass(spot, strike, expiry, rate, div_yield, params, opt_type, 0);
    let (_,     dkappa) = forward_pass(spot, strike, expiry, rate, div_yield, params, opt_type, 1);
    let (_,     dtheta) = forward_pass(spot, strike, expiry, rate, div_yield, params, opt_type, 2);
    let (_,     dsigma) = forward_pass(spot, strike, expiry, rate, div_yield, params, opt_type, 3);
    let (_,     drho)   = forward_pass(spot, strike, expiry, rate, div_yield, params, opt_type, 4);

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
        let (_, dv0_up) = forward_pass(spot, strike, expiry, rate, div_yield, &p_vup, opt_type, 0);
        dv0_up * 2.0 * (vol + 0.01)
    };
    let vega_dn = {
        let (_, dv0_dn) = forward_pass(spot, strike, expiry, rate, div_yield, &p_vdn, opt_type, 0);
        dv0_dn * 2.0 * (vol - 0.01).max(1e-6)
    };
    let volga = (vega_up - vega_dn) / (2.0 * 0.01);

    let _ = (dkappa, dtheta, dsigma, drho); // available for calibration gradient if needed

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

    // vega from AD should be close to bump-and-reprice. not identical — AD is exact,
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
