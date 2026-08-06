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

    const A1: f64 = -3.969_683_028_665_376e1; const A2: f64 =  2.209_460_984_245_205e2;
    const A3: f64 = -2.759_285_104_469_687e2; const A4: f64 =  1.383_577_518_672_69e2;
    const A5: f64 = -3.066_479_806_614_716e1; const A6: f64 =  2.506_628_277_459_239;
    const B1: f64 = -5.447_609_879_822_406e1; const B2: f64 =  1.615_858_368_580_409e2;
    const B3: f64 = -1.556_989_798_598_866e2; const B4: f64 =  6.680_131_188_771_972e1;
    const B5: f64 = -1.328_068_155_288_572e1;
    const C1: f64 = -7.784_894_002_430_293e-3; const C2: f64 = -3.223_964_580_411_365e-1;
    const C3: f64 = -2.400_758_277_161_838; const C4: f64 = -2.549_732_539_343_734;
    const C5: f64 =  4.374_664_141_464_968; const C6: f64 =  2.938_163_982_698_783;
    const D1: f64 =  7.784_695_709_041_462e-3; const D2: f64 =  3.224_671_290_700_398e-1;
    const D3: f64 =  2.445_134_137_142_996; const D4: f64 =  3.754_408_661_907_416;

    const P_LO: f64 = 0.02425;

    if (P_LO..=1.0 - P_LO).contains(&p) {
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
