// rational approx for Phi and Phi^-1.
// faster than libm erf in tight loops, ~2x on my box.
// ncdf: delegates to libm::erfc full double precision in the tails.
// replaces A&S 26.2.17 which had ~1.5e-7 tail error. matters for deep OTM IV solving.

const SQRT_2PI_INV: f64 = 0.3989422804014327;
const SQRT_2_INV:   f64 = std::f64::consts::FRAC_1_SQRT_2;

#[inline(always)]
pub fn npdf(x: f64) -> f64 {
    SQRT_2PI_INV * (-0.5 * x * x).exp()
}

// erfc-based. error ~1e-15 uniform, no branch-cut issues in the tails.
// phi(x) = erfc(-x / sqrt(2)) / 2
#[inline]
pub fn ncdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x * SQRT_2_INV)
}

// Acklam (2002). used for IV initial guess.
// split into 3 regions: central, lower tail, upper tail.
// don't call with p=0 or p=1, you'll get garbage.
pub fn ncdf_inv(p: f64) -> f64 {
    debug_assert!(p > 0.0 && p < 1.0);

    const A1: f64 = -3.969683028665376e+01; const A2: f64 =  2.209460984245205e+02;
    const A3: f64 = -2.759285104469687e+02; const A4: f64 =  1.383577518672690e+02;
    const A5: f64 = -3.066479806614716e+01; const A6: f64 =  2.506628277459239e+00;
    const B1: f64 = -5.447609879822406e+01; const B2: f64 =  1.615858368580409e+02;
    const B3: f64 = -1.556989798598866e+02; const B4: f64 =  6.680131188771972e+01;
    const B5: f64 = -1.328068155288572e+01;
    const C1: f64 = -7.784894002430293e-03; const C2: f64 = -3.223964580411365e-01;
    const C3: f64 = -2.400758277161838e+00; const C4: f64 = -2.549732539343734e+00;
    const C5: f64 =  4.374664141464968e+00; const C6: f64 =  2.938163982698783e+00;
    const D1: f64 =  7.784695709041462e-03; const D2: f64 =  3.224671290700398e-01;
    const D3: f64 =  2.445134137142996e+00; const D4: f64 =  3.754408661907416e+00;

    const P_LO: f64 = 0.02425;

    if p >= P_LO && p <= 1.0 - P_LO {
        let q = p - 0.5;
        let r = q * q;
        return q * (((((A1*r+A2)*r+A3)*r+A4)*r+A5)*r+A6)
                 / (1.0 + ((((B1*r+B2)*r+B3)*r+B4)*r+B5)*r);
    }

    let r = if p < P_LO { (-2.0*p.ln()).sqrt() } else { (-2.0*(1.0-p).ln()).sqrt() };
    let x = (((((C1*r+C2)*r+C3)*r+C4)*r+C5)*r+C6) / ((((D1*r+D2)*r+D3)*r+D4)*r+1.0);
    if p < P_LO { x } else { -x }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ncdf_sanity() {
        assert!((ncdf(0.0) - 0.5).abs() < 1e-15);
        assert!((ncdf(1.64485363) - 0.95).abs() < 1e-6);
        assert!(ncdf(-10.0) < 1e-23);
    }

    #[test]
    fn ncdf_tail_precision() {
        let cases = [
            (-4.0_f64, 3.167124183311998e-5),
            (-5.0_f64, 2.866515718791939e-7),
            (-6.0_f64, 9.865876449133282e-10),
        ];
        for (x, expected) in cases {
            let got     = ncdf(x);
            let rel_err = (got - expected).abs() / expected;
            assert!(rel_err < 5e-10, "ncdf({x}) = {got:.6e}, expected {expected:.6e}, rel_err={rel_err:.2e}");
        }
    }

    #[test]
    fn ncdf_inv_roundtrip() {
        for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
            let err = (ncdf(ncdf_inv(p)) - p).abs();
            assert!(err < 1e-6, "p={p} err={err}");
        }
    }
}
