// flat structs, no bullshit abstractions.
// Box<dyn Model> in a pricing loop is how you explain to your PM
// why the vol surface takes 40ms to update.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionType { Call, Put }

impl OptionType {
    #[inline]
    pub fn sign(&self) -> f64 {
        match self { OptionType::Call => 1.0, OptionType::Put => -1.0 }
    }
}

// q = continuous div yield. just set it to 0 if there's no div.
#[derive(Debug, Clone, Copy)]
pub struct OptionContract {
    pub spot:      f64,
    pub strike:    f64,
    pub expiry:    f64,  // years
    pub rate:      f64,  // continuously compounded
    pub div_yield: f64,  // q
    pub vol:       f64,
    pub opt_type:  OptionType,
}

// greeks default to 0 — some models compute them, some don't.
// don't assume they're populated unless you asked for them.
#[derive(Debug, Clone, Copy, Default)]
pub struct PricingResult {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega:  f64,
    pub theta: f64,
    pub rho:   f64,
    pub vanna: f64,
    pub volga: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct IvProblem {
    pub contract:     OptionContract,
    pub market_price: f64,
}

// NOTE: v0 is variance, not vol. vol = sqrt(v0). burned by this once.
#[derive(Debug, Clone, Copy)]
pub struct HestonParams {
    pub v0:    f64,  // initial variance
    pub kappa: f64,  // mean reversion
    pub theta: f64,  // long-run variance
    pub sigma: f64,  // vol of vol
    pub rho:   f64,  // spot-vol corr
}

impl HestonParams {
    // 2*kappa*theta > sigma^2. if this fails, variance hits zero.
    // check before running a calibration or you'll waste 10 minutes.
    pub fn feller_ok(&self) -> bool {
        2.0 * self.kappa * self.theta > self.sigma * self.sigma
    }
}

// Bates = Heston + Merton jump diffusion. clean composition.
#[derive(Debug, Clone, Copy)]
pub struct BatesParams {
    pub heston:  HestonParams,
    pub lambda:  f64,  // avg jumps/year
    pub mu_j:    f64,  // mean log-jump
    pub sigma_j: f64,  // jump vol
}

// flat layout: local_vols[i_k * n_expiry + j_t]
// do NOT use Vec<Vec<f64>> here. cache locality matters.
pub struct LocalVolSurface {
    pub strikes:    Vec<f64>,
    pub expiries:   Vec<f64>,
    pub local_vols: Vec<f64>,
}

impl LocalVolSurface {
    pub fn new(strikes: Vec<f64>, expiries: Vec<f64>, local_vols: Vec<f64>) -> Self {
        assert_eq!(local_vols.len(), strikes.len() * expiries.len());
        Self { strikes, expiries, local_vols }
    }

    #[inline]
    pub fn get(&self, i_k: usize, j_t: usize) -> f64 {
        self.local_vols[i_k * self.expiries.len() + j_t]
    }
}
