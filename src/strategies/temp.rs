use anyhow::Result;
use polars::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::strategies::strategy_base::{SellReason, TradingStrategy};
use super::helpers::{
    can_buy_with_constraints,
    get_block_id,
    get_sol_in_curve,
    get_token_amount,
    get_transaction_type,
    get_price_column_value,
    get_token_price,
};

/// Order Flow Imbalance strategy using Cumulative Volume Delta (CVD)
/// - Computes signed volume per row using `Token Amount` and `Transaction Type`
/// - Maintains rolling CVD and detects short-term bursts (imbalance) within a lookback window
/// - Buys on strong buy-side burst; sells on strong sell-side burst or stop conditions
pub struct OrderFlowImbalanceStrategy {
    // Parameters
    lookback_window: usize,           // Number of recent trades to accumulate for CVD burst check
    burst_threshold_tokens: f64,      // Absolute CVD change in window to trigger signal (tokens)
    min_sol_in_curve: f64,
    price_column: String,
    min_blocks_between_buys: i64,
    #[allow(dead_code)]
    min_hold_blocks: i64, // reserved for future extension
    min_blocks_between_sell_buy: i64,
    max_buys: usize,

    // State
    tokens_to_skip: HashSet<String>,
    signed_volume_history: HashMap<String, VecDeque<f64>>, // per-token signed token amounts
    cumulative_volume_delta: HashMap<String, f64>,          // running CVD (tokens)
    price_history: HashMap<String, VecDeque<f64>>,          // for SMA confirmation
    
    // Slope-based CVD state
    cvd_history: HashMap<String, VecDeque<f64>>,            // per-token CVD history for slope calculation
    slope_history: HashMap<String, VecDeque<f64>>,         // per-token slope history for z-score calculation


    // Intensity slope confirmation state
    intensity_history: HashMap<String, VecDeque<f64>>,     // per-token intensity history for slope calculation
    intensity_slope_history: HashMap<String, VecDeque<f64>>, // per-token intensity slope history for z-score calculation

    // Position tracking
    current_positions: HashMap<String, Vec<f64>>,
    buy_block_ids: HashMap<String, Vec<i64>>,
    buy_transaction_indices: HashMap<String, Vec<usize>>,
    sell_block_ids: HashMap<String, i64>,
    last_buy_z_score: Option<f64>,
    trade_z_scores: HashMap<String, f64>,

    // R:R risk management
    stop_loss_pct: f64,  // e.g., 0.05 = 5%
    take_profit_pct: f64, // set to 2x stop_loss_pct to enforce 1:2

    // Signal options
    use_sol_cvd: bool,             // if true, accumulate SOL-value (price*amount) instead of tokens
    min_trade_tokens: f64,         // ignore trades below this token amount
    use_zscore_cvd: bool,          // if true, use traditional z-score CVD detection
    use_zscore_gate: bool,         // if true, require window delta >= k * std
    zscore_k: f64,                 // multiplier k
    zscore_window: usize,          // history window to compute std from signed volumes
    require_price_above_sma: bool, // require price > SMA for buy confirmation
    price_sma_window: usize,       // SMA window (trades)
    
    // Slope-based CVD options
    use_slope_cvd: bool,           // if true, use slope-based CVD detection
    slope_window: usize,           // window for slope calculation
    slope_threshold: f64,         // minimum slope threshold for signal (as % of avg CVD)
    slope_zscore_gate: bool,      // if true, require slope >= k * std of historical slopes
    slope_zscore_k: f64,          // multiplier for slope z-score
    slope_zscore_window: usize,   // history window for slope z-score calculation
    slope_normalize_window: usize, // window for CVD normalization (recent average)


    // Intensity slope confirmation options
    use_intensity_slope_confirmation: bool,  // if true, use intensity slope as additional confirmation
    intensity_slope_window: usize,          // window for intensity slope calculation
    intensity_slope_threshold: f64,         // minimum intensity slope threshold for confirmation
    intensity_slope_zscore_gate: bool,      // if true, require intensity slope >= k * std
    intensity_slope_zscore_k: f64,          // multiplier for intensity slope z-score
    intensity_slope_zscore_window: usize,   // history window for intensity slope z-score calculation

    // Debug counters
    total_buy_signals: i64,
    total_sell_signals: i64,
    insufficient_data_rejections: i64, // reserved for future extension
    tokens_reached_threshold: HashSet<String>,

    // Output control
    #[allow(dead_code)]
    verbose: bool, // reserved for future extension
}

impl OrderFlowImbalanceStrategy {
    pub fn new(
        lookback_window: usize,
        burst_threshold_tokens: f64,
        min_sol_in_curve: f64,
        price_column: &str,
        min_blocks_between_buys: i64,
        min_hold_blocks: i64,
        min_blocks_between_sell_buy: i64,
        max_buys: usize,
        stop_loss_pct: f64,
        use_sol_cvd: bool,
        min_trade_tokens: f64,
        use_zscore_cvd: bool,
        use_zscore_gate: bool,
        zscore_k: f64,
        zscore_window: usize,
        require_price_above_sma: bool,
        price_sma_window: usize,
        use_slope_cvd: bool,
        slope_window: usize,
        slope_threshold: f64,
        slope_zscore_gate: bool,
        slope_zscore_k: f64,
        slope_zscore_window: usize,
        slope_normalize_window: usize,
        use_intensity_slope_confirmation: bool,
        intensity_slope_window: usize,
        intensity_slope_threshold: f64,
        intensity_slope_zscore_gate: bool,
        intensity_slope_zscore_k: f64,
        intensity_slope_zscore_window: usize,
        verbose: bool,
    ) -> Self {
        Self {
            lookback_window,
            burst_threshold_tokens,
            min_sol_in_curve,
            price_column: price_column.to_string(),
            min_blocks_between_buys,
            min_hold_blocks,
            min_blocks_between_sell_buy,
            max_buys,

            tokens_to_skip: HashSet::new(),
            signed_volume_history: HashMap::new(),
            cumulative_volume_delta: HashMap::new(),
            price_history: HashMap::new(),
            
            // Slope-based CVD state
            cvd_history: HashMap::new(),
            slope_history: HashMap::new(),


            // Intensity slope confirmation state
            intensity_history: HashMap::new(),
            intensity_slope_history: HashMap::new(),

            current_positions: HashMap::new(),
            buy_block_ids: HashMap::new(),
            buy_transaction_indices: HashMap::new(),
            sell_block_ids: HashMap::new(),
            last_buy_z_score: None,
            trade_z_scores: HashMap::new(),

            stop_loss_pct,
            take_profit_pct: stop_loss_pct * 2.0,

            use_sol_cvd,
            min_trade_tokens,
            use_zscore_cvd,
            use_zscore_gate,
            zscore_k,
            zscore_window,
            require_price_above_sma,
            price_sma_window,
            
            // Slope-based CVD parameters
            use_slope_cvd,
            slope_window,
            slope_threshold,
            slope_zscore_gate,
            slope_zscore_k,
            slope_zscore_window,
            slope_normalize_window,


            // Intensity slope confirmation parameters
            use_intensity_slope_confirmation,
            intensity_slope_window,
            intensity_slope_threshold,
            intensity_slope_zscore_gate,
            intensity_slope_zscore_k,
            intensity_slope_zscore_window,

            total_buy_signals: 0,
            total_sell_signals: 0,
            insufficient_data_rejections: 0,
            tokens_reached_threshold: HashSet::new(),

            verbose,
        }
    }

    fn push_signed_volume(&mut self, token_address: &str, signed_value: f64) {
        // Update signed volume history for traditional CVD burst detection
        let history = self
            .signed_volume_history
            .entry(token_address.to_string())
            .or_insert_with(|| VecDeque::with_capacity(self.lookback_window + 1));
        history.push_back(signed_value);
        while history.len() > self.lookback_window {
            history.pop_front();
        }
        
        // Update cumulative volume delta
        let cvd = self
            .cumulative_volume_delta
            .entry(token_address.to_string())
            .or_insert(0.0);
        *cvd += signed_value;
        
        // Store CVD history for slope calculation (Option 1: Store CVD)
        if self.use_slope_cvd {
            let cvd_history = self
                .cvd_history
                .entry(token_address.to_string())
                .or_insert_with(|| VecDeque::with_capacity(self.slope_window + 100));
            
            // Store the current cumulative CVD value
            cvd_history.push_back(*cvd);
            
            // Maintain history size
            while cvd_history.len() > self.slope_window + 100 {
                cvd_history.pop_front();
            }
        }
    }

    fn window_delta(&self, token_address: &str) -> f64 {
        match self.signed_volume_history.get(token_address) {
            Some(hist) => hist.iter().copied().sum::<f64>(),
            None => 0.0,
        }
    }

    /// Calculate the slope of CVD over the specified window using linear regression
    /// 
    /// Handles edge cases:
    /// - Insufficient data points
    /// - All identical values (denominator = 0)
    /// - Single data point
    fn calculate_cvd_slope(&self, token_address: &str) -> Option<f64> {
        let cvd_hist = self.cvd_history.get(token_address)?;
        
        if cvd_hist.len() < self.slope_window {
            return None;
        }
        
        // Take the last slope_window points
        let start_idx = cvd_hist.len().saturating_sub(self.slope_window);
        let points: Vec<f64> = cvd_hist.iter().skip(start_idx).copied().collect();
        
        if points.len() < 2 {
            return None;
        }
        
        // Check if all values are identical (would cause division by zero)
        let first_value = points[0];
        if points.iter().all(|&x| (x - first_value).abs() < 1e-10) {
            return Some(0.0); // Flat line = zero slope
        }
        
        // Simple linear regression to calculate slope
        let n = points.len() as f64;
        let x_mean = (n - 1.0) / 2.0; // x values are 0, 1, 2, ..., n-1
        let y_mean = points.iter().sum::<f64>() / n;
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for (i, &y) in points.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }
        
        // Additional safety check for very small denominators
        if denominator <= 1e-10 {
            return None;
        }
        
        Some(numerator / denominator)
    }

    /// Calculate the normalized slope of CVD (slope as % of recent average CVD)
    /// 
    /// Normalization Options:
    /// 1. Current: (raw_slope / avg_abs_cvd) * 100.0 = "% change per trade"
    /// 2. Alternative: raw_slope = absolute slope value
    /// 3. Alternative: (raw_slope / std_cvd) = z-score normalization
    fn calculate_normalized_cvd_slope(&self, token_address: &str) -> Option<f64> {
        let cvd_hist = self.cvd_history.get(token_address)?;
        
        if cvd_hist.len() < self.slope_window {
            return None;
        }
        // Calculate raw slope
        let raw_slope = self.calculate_cvd_slope(token_address)?;

        // Calculate recent average CVD for normalization
        let normalize_start = cvd_hist.len().saturating_sub(self.slope_normalize_window);
        let recent_cvd_values: Vec<f64> = cvd_hist.iter().skip(normalize_start).copied().collect();
        
        if recent_cvd_values.is_empty() {
            return None;
        }
        // Calculate average absolute CVD for normalization
        let avg_abs_cvd = recent_cvd_values.iter().map(|x| x.abs()).sum::<f64>() / recent_cvd_values.len() as f64;
        // Avoid division by zero and very small values that could cause extreme normalization
        if avg_abs_cvd <= 1e-10 {
            return None;
        }
        
        // OPTION 1: Current approach - "% change per trade" (scale-invariant)
        let normalized_slope = (raw_slope / avg_abs_cvd) * 100.0;
        // OPTION 2: Uncomment for absolute slope values (not scale-invariant)
        // let normalized_slope = raw_slope;
        
        // OPTION 3: Uncomment for z-score normalization (statistical approach)
        // let cvd_std = self.calculate_cvd_std(&recent_cvd_values)?;
        // if cvd_std <= 1e-10 { return None; }
        // let normalized_slope = raw_slope / cvd_std;
        
        Some(normalized_slope)
    }

    /// Helper function to calculate standard deviation of CVD values (for z-score normalization)
    #[allow(dead_code)] // Used in commented-out normalization option
    fn calculate_cvd_std(&self, values: &[f64]) -> Option<f64> {
        if values.len() < 2 {
            return None;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        if variance <= 0.0 {
            return None;
        }
        
        Some(variance.sqrt())
    }

    /// Check if the current CVD slope indicates a buy burst (positive slope above threshold)
    fn has_slope_buy_burst(&self, token_address: &str) -> bool {
        if !self.use_slope_cvd {
            return false;
        }
        // Use normalized slope for scale-invariant detection
        let slope = match self.calculate_normalized_cvd_slope(token_address) {
            Some(slope) => slope,
            None => return false,
        };
        
        if !self.slope_zscore_gate {
            return slope >= self.slope_threshold;
        }
        
        // Z-score gate for normalized slope
        let slope_hist = match self.slope_history.get(token_address) {
            Some(hist) => hist,
            None => return false,
        };
        
        if slope_hist.len() < self.slope_zscore_window {
            return false;
        }
        
        // Calculate mean and std of historical normalized slopes
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        
        for slope_val in slope_hist.iter().rev().take(self.slope_zscore_window) {
            sum += *slope_val;
            sum_sq += slope_val * slope_val;
        }
        
        let mean = sum / self.slope_zscore_window as f64;
        let var = (sum_sq / self.slope_zscore_window as f64) - (mean * mean);
        
        if var <= 0.0 {
            return false;
        }
        
        let std = var.sqrt();
        slope >= self.slope_zscore_k * std
    }

    /// Check if the current CVD slope indicates a sell burst (negative slope below threshold)
    fn has_slope_sell_burst(&self, token_address: &str) -> bool {
        if !self.use_slope_cvd {
            return false;
        }
        
        // Use normalized slope for scale-invariant detection
        let slope = match self.calculate_normalized_cvd_slope(token_address) {
            Some(slope) => slope,
            None => return false,
        };
        
        if !self.slope_zscore_gate {
            return slope <= -self.slope_threshold;
        }
        
        // Z-score gate for normalized slope
        let slope_hist = match self.slope_history.get(token_address) {
            Some(hist) => hist,
            None => return false,
        };
        
        if slope_hist.len() < self.slope_zscore_window {
            return false;
        }
        
        // Calculate mean and std of historical normalized slopes
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        
        for slope_val in slope_hist.iter().rev().take(self.slope_zscore_window) {
            sum += *slope_val;
            sum_sq += slope_val * slope_val;
        }
        
        let mean = sum / self.slope_zscore_window as f64;
        let var = (sum_sq / self.slope_zscore_window as f64) - (mean * mean);
        
        if var <= 0.0 {
            return false;
        }
        
        let std = var.sqrt();
        slope <= -self.slope_zscore_k * std
    }

    /// Calculate trade intensity (trades per window) for the given token
    /// Returns the number of trades in the recent window
    fn calculate_trade_intensity(&self, token_address: &str) -> f64 {
        match self.signed_volume_history.get(token_address) {
            Some(hist) => {
                if hist.len() < 2 {
                    return 0.0; // Need at least 2 trades to calculate intensity
                }
                
                // Use the signed volume history length as a proxy for trade count
                // This gives us the number of trades in the lookback window
                hist.len() as f64
            }
            None => 0.0,
        }
    }

    /// Calculate the slope of trade intensity over the specified window using linear regression
    /// 
    /// Handles edge cases:
    /// - Insufficient data points
    /// - All identical values (denominator = 0)
    /// - Single data point
    fn calculate_intensity_slope(&self, token_address: &str) -> Option<f64> {
        let intensity_hist = self.intensity_history.get(token_address)?;
        
        if intensity_hist.len() < self.intensity_slope_window {
            return None;
        }
        println!("intensity_hist.len(): {}", intensity_hist.len());
        // Take the last intensity_slope_window points
        let start_idx = intensity_hist.len().saturating_sub(self.intensity_slope_window);
        let points: Vec<f64> = intensity_hist.iter().skip(start_idx).copied().collect();
        
        if points.len() < 2 {
            return None;
        }
        
        // Check if all values are identical (would cause division by zero)
        let first_value = points[0];
        if points.iter().all(|&x| (x - first_value).abs() < 1e-10) {
            return Some(0.0); // Flat line = zero slope
        }
        
        // Simple linear regression to calculate slope
        let n = points.len() as f64;
        let x_mean = (n - 1.0) / 2.0; // x values are 0, 1, 2, ..., n-1
        let y_mean = points.iter().sum::<f64>() / n;
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for (i, &y) in points.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }
        println!("x_mean: {}", x_mean);
        // Additional safety check for very small denominators
        if denominator <= 1e-10 {
            return None;
        }
        println!("Intensity slope: {}", numerator / denominator);
        Some(numerator / denominator)
    }

    /// Check if the current intensity slope indicates a buy confirmation (positive slope above threshold)
    fn has_intensity_slope_buy_confirmation(&self, token_address: &str) -> bool {
        if !self.use_intensity_slope_confirmation {
            return false;
        }
        println!("intensity_slope_confirmation: {}", self.use_intensity_slope_confirmation);
        let slope = match self.calculate_intensity_slope(token_address) {
            Some(slope) => slope,
            None => return false,
        };
        
        if !self.intensity_slope_zscore_gate {
            return slope >= self.intensity_slope_threshold;
        }
        
        // Z-score gate for intensity slope
        let slope_hist = match self.intensity_slope_history.get(token_address) {
            Some(hist) => hist,
            None => return false,
        };
        
        if slope_hist.len() < self.intensity_slope_zscore_window {
            return false;
        }
        
        // Calculate mean and std of historical intensity slopes
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        
        for slope_val in slope_hist.iter().rev().take(self.intensity_slope_zscore_window) {
            sum += *slope_val;
            sum_sq += slope_val * slope_val;
        }
        
        let mean = sum / self.intensity_slope_zscore_window as f64;
        let var = (sum_sq / self.intensity_slope_zscore_window as f64) - (mean * mean);
        
        if var <= 0.0 {
            return false;
        }
        
        let std = var.sqrt();
        slope >= self.intensity_slope_zscore_k * std
    }

    fn has_buy_burst(&self, token_address: &str) -> bool {
        if !self.use_zscore_cvd {
            return false;
        }
        
        let delta = self.window_delta(token_address);
        if !self.use_zscore_gate {
            return delta >= self.burst_threshold_tokens;
        }
        // z-score gate using history std of signed values over last zscore_window
        if let Some(hist) = self.signed_volume_history.get(token_address) {
            if hist.len() < self.zscore_window { return false; }
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            // iterate last zscore_window elements
            for v in hist.iter().rev().take(self.zscore_window) {
                sum += *v;
                sum_sq += v * v;
            }
            let mean = sum / self.zscore_window as f64;
            let var = (sum_sq / self.zscore_window as f64) - (mean * mean);
            if var <= 0.0 { return false; }
            let std = var.max(0.0).sqrt();
            return delta >= self.zscore_k * std;
        }
        false
    }

    fn has_sell_burst(&self, token_address: &str) -> bool {
        if !self.use_zscore_cvd {
            return false;
        }
        
        let delta = -self.window_delta(token_address);
        if !self.use_zscore_gate {
            return delta >= self.burst_threshold_tokens;
        }
        if let Some(hist) = self.signed_volume_history.get(token_address) {
            if hist.len() < self.zscore_window { return false; }
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for v in hist.iter().rev().take(self.zscore_window) {
                sum += *v;
                sum_sq += v * v;
            }
            let mean = sum / self.zscore_window as f64;
            let var = (sum_sq / self.zscore_window as f64) - (mean * mean);
            if var <= 0.0 { return false; }
            let std = var.max(0.0).sqrt();
            return delta >= self.zscore_k * std;
        }
        false
    }
}

impl TradingStrategy for OrderFlowImbalanceStrategy {
    fn should_buy(
        &mut self,
        token_address: &str,
        current_price: f64,
        _current_index: usize,
        _row: &AnyValue,
        row_data: &HashMap<String, AnyValue>,
    ) -> Result<bool> {
        let current_sol_curve = get_sol_in_curve(row_data);
        let current_block_id = get_block_id(row_data);

        if self.tokens_to_skip.contains(token_address) {
            return Ok(false);
        }


        // Optional price SMA confirmation
        if self.require_price_above_sma {
            let price = get_token_price(row_data);
            let ph = self
                .price_history
                .entry(token_address.to_string())
                .or_insert_with(|| VecDeque::with_capacity(self.price_sma_window + 1));
            ph.push_back(price);
            while ph.len() > self.price_sma_window { ph.pop_front(); }
            if ph.len() == self.price_sma_window {
                let sma = ph.iter().copied().sum::<f64>() / (ph.len() as f64);
                if price <= sma { return Ok(false); }
            } else {
                return Ok(false);
            }
        }
     

		// Check if we can buy more positions (including SOL threshold check)
		if !can_buy_with_constraints(
			token_address,
			current_block_id,
			current_sol_curve,
			current_price,
			&self.sell_block_ids,
			&self.buy_block_ids,
			self.min_blocks_between_sell_buy,
			self.max_buys,
			self.min_blocks_between_buys,
			self.min_sol_in_curve,
			&self.tokens_reached_threshold,
		) {
			return Ok(false);
		}



        // CVD burst condition (traditional z score based)
        let traditional_buy_signal = self.has_buy_burst(token_address);
        
        // Slope-based CVD buy signal
        let slope_buy_signal = self.has_slope_buy_burst(token_address);
        println!("slope_buy_signal: {}", slope_buy_signal);
        // Intensity slope confirmation signal
        let intensity_slope_confirmation = self.has_intensity_slope_buy_confirmation(token_address);
        println!("intensity_slope_confirmation: {}", intensity_slope_confirmation);
        // Buy if either signal is active (or both), with optional intensity slope confirmation
        if traditional_buy_signal || slope_buy_signal {
            // If intensity slope confirmation is enabled, require it as additional confirmation
            if self.use_intensity_slope_confirmation && !intensity_slope_confirmation {
                return Ok(false);
            }
            
            self.total_buy_signals += 1;
            return Ok(true);
        }

        Ok(false)
    }

    fn should_sell(
        &mut self,
        token_address: &str,
        current_price: f64,
        _current_index: usize,
        _row: &AnyValue,
        _row_data: &HashMap<String, AnyValue>,
    ) -> Result<(bool, SellReason)> {
        // Risk:Reward first - use average entry price from current positions
        if let Some(positions) = self.current_positions.get(token_address) {
            if !positions.is_empty() {
                let avg_entry_price = positions.iter().sum::<f64>() / positions.len() as f64;
                if avg_entry_price > 0.0 {
                    let stop_price = avg_entry_price * (1.0 - self.stop_loss_pct);
                    if current_price <= stop_price {
                        self.total_sell_signals += 1;
                        return Ok((true, SellReason::StopLoss(self.stop_loss_pct)));
                    }

                    let target_price = avg_entry_price * (1.0 + self.take_profit_pct);
                    if current_price >= target_price {
                        self.total_sell_signals += 1;
                        return Ok((true, SellReason::TakeProfit(self.take_profit_pct)));
                    }
                }
            }
        }

        // Only check CVD sell signals if we have an open position
        if let Some(positions) = self.current_positions.get(token_address) {
            if !positions.is_empty() {
                // Traditional sell burst condition
                let traditional_sell_signal = self.has_sell_burst(token_address);
                
                // Slope-based CVD sell signal
                let slope_sell_signal = self.has_slope_sell_burst(token_address);
                
                // Sell if either signal is active (or both)
                if traditional_sell_signal || slope_sell_signal {
                    self.total_sell_signals += 1;
                    let reason = if traditional_sell_signal && slope_sell_signal {
                        "OFI_TRADITIONAL_AND_SLOPE_SELL_BURST"
                    } else if traditional_sell_signal {
                        "OFI_TRADITIONAL_SELL_BURST"
                    } else {
                        "OFI_SLOPE_SELL_BURST"
                    };
                    return Ok((true, SellReason::Strategy(reason.to_string())));
                }
            }
        }
        
        Ok((false, SellReason::Strategy("HOLD".to_string())))
    }

    fn update_data(
        &mut self,
        token_address: &str,
        price: f64,
        _tx_index: usize,
        _date: &str,
        row_data: &HashMap<String, AnyValue>,
    ) -> Result<()> {
        // Maintain signed token amount stream based on transaction type
        let token_amount = get_token_amount(row_data);
        let tx_type = get_transaction_type(row_data);
        if token_amount >= self.min_trade_tokens {
            let signed_tokens = match tx_type.to_lowercase().as_str() {
                "sell" => -token_amount,
                _ => token_amount,
            };
            let signed_value = if self.use_sol_cvd {
                let price: f64 = get_token_price(row_data);
                signed_tokens * price
            } else {
                signed_tokens
            };
            self.push_signed_volume(token_address, signed_value);


            // Track intensity history for slope calculation
            if self.use_intensity_slope_confirmation {
                let current_intensity = self.calculate_trade_intensity(token_address);
                
                let intensity_hist = self
                    .intensity_history
                    .entry(token_address.to_string())
                    .or_insert_with(|| VecDeque::with_capacity(self.intensity_slope_window + 100));
                
                intensity_hist.push_back(current_intensity);
                
                // Maintain history size
                while intensity_hist.len() > self.intensity_slope_window + 100 {
                    intensity_hist.pop_front();
                }
            }
        }

        // Track thresholds for SOL in curve gating
        let current_sol_curve = get_sol_in_curve(row_data);
        if current_sol_curve >= self.min_sol_in_curve {
            self.tokens_reached_threshold.insert(token_address.to_string());
        }

        // Update slope history for slope-based CVD if enabled
        if self.use_slope_cvd {
            // Store normalized slope for scale-invariant z-score calculation
            if let Some(normalized_slope) = self.calculate_normalized_cvd_slope(token_address) {
                let slope_hist = self
                    .slope_history
                    .entry(token_address.to_string())
                    .or_insert_with(|| VecDeque::with_capacity(self.slope_zscore_window + 100));
                slope_hist.push_back(normalized_slope);
                while slope_hist.len() > self.slope_zscore_window + 100 {
                    slope_hist.pop_front();
                }
            }
        }

        // Update intensity slope history for intensity slope confirmation if enabled
        if self.use_intensity_slope_confirmation {
            // Store intensity slope for z-score calculation
            if let Some(intensity_slope) = self.calculate_intensity_slope(token_address) {
                let intensity_slope_hist = self
                    .intensity_slope_history
                    .entry(token_address.to_string())
                    .or_insert_with(|| VecDeque::with_capacity(self.intensity_slope_zscore_window + 100));
                intensity_slope_hist.push_back(intensity_slope);
                while intensity_slope_hist.len() > self.intensity_slope_zscore_window + 100 {
                    intensity_slope_hist.pop_front();
                }
            }
        }

        // Optionally use price for additional filters later
        let _current_price = if self.price_column.is_empty() {
            price
        } else {
            get_price_column_value(row_data, &self.price_column)
        };

        Ok(())
    }

    fn get_strategy_name(&self) -> String {
        "OrderFlowImbalance".to_string()
    }

    fn get_system_name(&self) -> String {
        "OrderFlowImbalanceStrategy".to_string()
    }

    fn get_export_parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("lookback_window".to_string(), self.lookback_window.to_string());
        params.insert(
            "burst_threshold_tokens".to_string(),
            format!("{}", self.burst_threshold_tokens),
        );
        params.insert("min_sol_in_curve".to_string(), format!("{}", self.min_sol_in_curve));
        params.insert("use_zscore_cvd".to_string(), self.use_zscore_cvd.to_string());
        params.insert("use_slope_cvd".to_string(), self.use_slope_cvd.to_string());
        params.insert("slope_window".to_string(), self.slope_window.to_string());
        params.insert("slope_threshold".to_string(), format!("{}", self.slope_threshold));
        params.insert("slope_zscore_gate".to_string(), self.slope_zscore_gate.to_string());
        params.insert("slope_zscore_k".to_string(), format!("{}", self.slope_zscore_k));
        params.insert("slope_zscore_window".to_string(), self.slope_zscore_window.to_string());
        params.insert("slope_normalize_window".to_string(), self.slope_normalize_window.to_string());
        params.insert("use_intensity_slope_confirmation".to_string(), self.use_intensity_slope_confirmation.to_string());
        params.insert("intensity_slope_window".to_string(), self.intensity_slope_window.to_string());
        params.insert("intensity_slope_threshold".to_string(), format!("{}", self.intensity_slope_threshold));
        params.insert("intensity_slope_zscore_gate".to_string(), self.intensity_slope_zscore_gate.to_string());
        params.insert("intensity_slope_zscore_k".to_string(), format!("{}", self.intensity_slope_zscore_k));
        params.insert("intensity_slope_zscore_window".to_string(), self.intensity_slope_zscore_window.to_string());
        params
    }

    fn get_slippage_tolerance(&self) -> f64 {
        0.0
    }

	fn on_buy_executed(
		&mut self,
		token_address: &str,
		current_price: f64,
		_sol_invested: f64,
		_tokens_bought: f64,
		current_index: Option<usize>,
		row_data: Option<&HashMap<String, AnyValue>>,
	) -> Result<()> {
		// Initialize lists if this is the first buy for this token
		if !self.current_positions.contains_key(token_address) {
			self.current_positions
				.insert(token_address.to_string(), Vec::new());
			self.buy_block_ids
				.insert(token_address.to_string(), Vec::new());
			self.buy_transaction_indices
				.insert(token_address.to_string(), Vec::new());
		}

		// Add the new buy to the lists
		self.current_positions
			.get_mut(token_address)
			.unwrap()
			.push(current_price);

		if let Some(index) = current_index {
			self.buy_transaction_indices
				.get_mut(token_address)
				.unwrap()
				.push(index);
		}

		// Store buy block ID
		let current_block_id = if let Some(row_data) = row_data {
			get_block_id(row_data)
		} else {
			0 // Default to 0 if no data available
		};
		self.buy_block_ids
			.get_mut(token_address)
			.unwrap()
			.push(current_block_id);

		// Store the Z-score that triggered this buy
		if let Some(z_score) = self.last_buy_z_score.take() {
			self.trade_z_scores
				.insert(token_address.to_string(), z_score);
		}

		// Verbose logging enabled for comparison
		if self.verbose {
			println!(
				"  📊 Now holding {} positions for {}",
				self.current_positions[token_address].len(),
				&token_address[token_address.len() - 8..]
			);
		}

		Ok(())
	}

	fn on_sell_executed(
		&mut self,
		token_address: &str,
		_current_index: Option<usize>,
		row_data: Option<&HashMap<String, AnyValue>>,
	) -> Result<()> {
		// Clear all positions for this token (sell all at once)
		let positions_count = self
			.current_positions
			.get(token_address)
			.map(|p| p.len())
			.unwrap_or(0);

		if let Some(positions) = self.current_positions.get_mut(token_address) {
			positions.clear();
		}

		// Clear buy block ID tracking
		if let Some(buy_block_ids) = self.buy_block_ids.get_mut(token_address) {
			buy_block_ids.clear();
		}

		// Clear transaction index tracking
		if let Some(indices) =
			self.buy_transaction_indices.get_mut(token_address)
		{
			indices.clear();
		}

		// Track sell block ID for minimum blocks between sell and buy constraint
		let current_block_id = if let Some(row_data) = row_data {
			get_block_id(row_data)
		} else {
			0 // Default to 0 if no data available
		};
		self.sell_block_ids
			.insert(token_address.to_string(), current_block_id);

		// Clean up Z-score tracking for closed position
		self.trade_z_scores.remove(token_address);

		// Verbose logging enabled for comparison
		if self.verbose {
			println!(
				"  📤 Sold all {} positions for {}",
				positions_count,
				&token_address[token_address.len() - 8..]
			);
		}

		Ok(())
	}

    fn get_debug_stats(&self) -> HashMap<String, i64> {
        let mut stats = HashMap::new();
        stats.insert("total_buy_signals".to_string(), self.total_buy_signals);
        stats.insert("total_sell_signals".to_string(), self.total_sell_signals);
        stats.insert(
            "insufficient_data_rejections".to_string(),
            self.insufficient_data_rejections,
        );
        stats
    }
}

































use anyhow::Result;
use rust_backtester::{
    BacktestFramework, TradingStrategy,
};
use rust_backtester::strategies::OrderFlowImbalanceStrategy;
use rust_backtester::strategies::common_config::*;

// CSV file path for testing
const CSV_FILE_PATH: &str =
    "/home/daksh/Downloads/pumpfun_09-08-2025_to_09-11-2025.csv/pumpfun_09-08-2025_to_09-11-2025.csv";

// Order Flow Imbalance Strategy Configuration
const LOOKBACK_WINDOW: usize = 50;           // trades in rolling window
const BURST_THRESHOLD_TOKENS: f64 = 200000000.0; // token units; tune per dataset
const PRICE_COLUMN: &str = "Token Price";
const STOP_LOSS_PCT: f64 = 0.20; // 5% stop, 10% target (1:2)
const MAX_BUYS: usize = 1; // Maximum number of buy positions per token

// CVD Configuration
const USE_SOL_CVD: bool = true;         // accumulate SOL-value instead of tokens
const MIN_TRADE_TOKENS: f64 = 1e5;      // ignore micro trades (example threshold)

// Traditional Z-Score CVD Detection
const USE_ZSCORE_CVD: bool = false;      // enable traditional z-score CVD detection
const USE_ZSCORE_GATE: bool = false;     // require z-score burst
const ZSCORE_K: f64 = 2.0;              // k-sigma threshold
const ZSCORE_WINDOW: usize = 50;        // window for std calc

// Price Confirmation
const REQUIRE_PRICE_ABOVE_SMA: bool = false; // price confirmation
const PRICE_SMA_WINDOW: usize = 30;     // SMA window

// Slope-based CVD Detection
const USE_SLOPE_CVD: bool = true;       // enable slope-based CVD detection
const SLOPE_WINDOW: usize = 200;         // window for slope calculation
const SLOPE_THRESHOLD: f64 = 0.5;        // minimum slope threshold (% of avg CVD)
const SLOPE_ZSCORE_GATE: bool = false;  // require slope z-score
const SLOPE_ZSCORE_K: f64 = 2.0;       // slope z-score multiplier
const SLOPE_ZSCORE_WINDOW: usize = 30;  // window for slope z-score calculation
const SLOPE_NORMALIZE_WINDOW: usize = 200; // window for CVD normalization


// Intensity Slope Confirmation
const USE_INTENSITY_SLOPE_CONFIRMATION: bool = true;  // enable intensity slope confirmation
const INTENSITY_SLOPE_WINDOW: usize = 50;             // window for intensity slope calculation
const INTENSITY_SLOPE_THRESHOLD: f64 = 0.5;           // minimum intensity slope threshold
const INTENSITY_SLOPE_ZSCORE_GATE: bool = false;      // require intensity slope z-score
const INTENSITY_SLOPE_ZSCORE_K: f64 = 2.0;           // intensity slope z-score multiplier
const INTENSITY_SLOPE_ZSCORE_WINDOW: usize = 50;      // window for intensity slope z-score calculation

// Alternative configurations for testing (uncomment to use):
// const USE_SLOPE_CVD: bool = false;      // disable slope-based detection
// const USE_ZSCORE_CVD: bool = false;     // disable traditional detection
// const SLOPE_THRESHOLD: f64 = 10.0;      // higher threshold for fewer signals
// const SLOPE_THRESHOLD: f64 = 2.0;       // lower threshold for more signals

fn test_ofi_strategy() -> Result<()> {
    println!("{}", "=".repeat(80));
    println!("📊 Testing Order Flow Imbalance (CVD) Strategy");
    println!("{}", "=".repeat(80));
    println!("This strategy combines traditional CVD burst detection with slope-based CVD analysis.");
    println!("Both detection methods can be enabled/disabled independently for testing.");
    println!("{}", "=".repeat(80));

    let mut strategy = OrderFlowImbalanceStrategy::new(
        LOOKBACK_WINDOW,
        BURST_THRESHOLD_TOKENS,
        MIN_SOL_IN_CURVE,
        PRICE_COLUMN,
        MIN_BLOCKS_BETWEEN_BUYS,
        MIN_HOLD_BLOCKS,
        MIN_BLOCKS_BETWEEN_SELL_BUY,
        MAX_BUYS,
        STOP_LOSS_PCT,
        USE_SOL_CVD,
        MIN_TRADE_TOKENS,
        USE_ZSCORE_CVD,
        USE_ZSCORE_GATE,
        ZSCORE_K,
        ZSCORE_WINDOW,
        REQUIRE_PRICE_ABOVE_SMA,
        PRICE_SMA_WINDOW,
        USE_SLOPE_CVD,
        SLOPE_WINDOW,
        SLOPE_THRESHOLD,
        SLOPE_ZSCORE_GATE,
        SLOPE_ZSCORE_K,
        SLOPE_ZSCORE_WINDOW,
        SLOPE_NORMALIZE_WINDOW,
        USE_INTENSITY_SLOPE_CONFIRMATION,
        INTENSITY_SLOPE_WINDOW,
        INTENSITY_SLOPE_THRESHOLD,
        INTENSITY_SLOPE_ZSCORE_GATE,
        INTENSITY_SLOPE_ZSCORE_K,
        INTENSITY_SLOPE_ZSCORE_WINDOW,
        VERBOSE,
    );

    let mut framework = BacktestFramework::new(
        POSITION_SIZE_SOL,
        BUY_FEE_PERCENT,
        SELL_FEE_PERCENT,
        FLAT_FEE_SOL_BUY,
        FLAT_FEE_SOL_SELL,
        BLOCK_LATENCY_MAX,
        true,  // export_trades
        VERBOSE,
        USE_DYNAMIC_FEES,
    );

    framework.preprocess_data(
        CSV_FILE_PATH,
        VALIDATE_DATA,
        TOLERANCE_PERCENT,
        DETECT_MANIPULATION,
        MANIPULATION_THRESHOLD,
    )?;

    println!("Running Order Flow Imbalance Strategy with:");
    println!("  📊 CSV file: {}", CSV_FILE_PATH);
    println!("  🔎 Lookback window (trades): {}", LOOKBACK_WINDOW);
    println!("  💥 Burst threshold: {} ({} mode)", BURST_THRESHOLD_TOKENS, if USE_SOL_CVD { "SOL" } else { "token" });
    println!("  🪙 Min trade tokens: {}", MIN_TRADE_TOKENS);
    println!();
    println!("  📈 Traditional CVD Detection:");
    println!("    • Enabled: {} (z-score: {}, k={}, window={})", USE_ZSCORE_CVD, USE_ZSCORE_GATE, ZSCORE_K, ZSCORE_WINDOW);
    println!("  📊 Slope-based CVD Detection:");
    println!("    • Enabled: {} (window={}, threshold={}%)", USE_SLOPE_CVD, SLOPE_WINDOW, SLOPE_THRESHOLD);
    println!("    • Z-score gate: {} (k={}, window={})", SLOPE_ZSCORE_GATE, SLOPE_ZSCORE_K, SLOPE_ZSCORE_WINDOW);
    println!("    • Normalize window: {} trades", SLOPE_NORMALIZE_WINDOW);
    println!();
    println!("  📈 Intensity Slope Confirmation:");
    println!("    • Enabled: {} (window={}, threshold={}, z-score: {}, k={}, z-window={})", USE_INTENSITY_SLOPE_CONFIRMATION, INTENSITY_SLOPE_WINDOW, INTENSITY_SLOPE_THRESHOLD, INTENSITY_SLOPE_ZSCORE_GATE, INTENSITY_SLOPE_ZSCORE_K, INTENSITY_SLOPE_ZSCORE_WINDOW);
    println!();
    println!("  📉 Price Confirmation:");
    println!("    • SMA filter: {} (window={})", REQUIRE_PRICE_ABOVE_SMA, PRICE_SMA_WINDOW);
    println!();
    println!("  💰 Risk Management:");
    println!("    • Position size: {} SOL", POSITION_SIZE_SOL);
    println!("    • Min SOL in curve: {}", MIN_SOL_IN_CURVE);
    println!("    • Stop loss: {:.1}% | Target: {:.1}%", STOP_LOSS_PCT * 100.0, STOP_LOSS_PCT * 200.0);

    let results = framework.run_strategy_on_preprocessed_data(&mut strategy)?;

    if results.completed_trades_df.height() > 0 {
        framework.print_sol_in_curve_analysis(25.0);
        framework.print_results_summary(&results, &strategy.get_strategy_name());
        println!("\n📁 Exporting results to CSV files...");
        let _ = framework.export_equity_curve(&results, &strategy);
        let _ = framework.export_detailed_trades(&results, &strategy);
    } else {
        println!("❌ Backtest failed - no results returned");
    }

    Ok(())
}

fn main() -> Result<()> {
    println!("🚀 Starting OFI Strategy Testing Suite");
    test_ofi_strategy()?;
    println!("\n✅ OFI Strategy Test Complete!");
    Ok(())
}


