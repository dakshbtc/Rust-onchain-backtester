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
    use_zscore_gate: bool,         // if true, require window delta >= k * std (should always be true for z-score CVD)
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
        verbose: bool,
    ) -> Self {
        Self {
            lookback_window,
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
        let normalize_start: usize = cvd_hist.len().saturating_sub(self.slope_normalize_window);
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

    fn has_buy_burst(&self, token_address: &str) -> bool {
        if !self.use_zscore_cvd {
            return false;
        }
        
        let delta = self.window_delta(token_address);
        // Z-score gate should always be true for z-score CVD - no raw delta thresholding
        if !self.use_zscore_gate {
            return false; // Disable raw thresholding - require statistical validation
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
        // Z-score gate should always be true for z-score CVD - no raw delta thresholding
        if !self.use_zscore_gate {
            return false; // Disable raw thresholding - require statistical validation
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

        if self.tokens_to_skip.contains(token_address) {
            return Ok(false);
        }


        // CVD burst condition (traditional z score based)
        let traditional_buy_signal = self.has_sell_burst(token_address);
        
        // Buy when has slope sell burst
        let slope_buy_signal = self.has_slope_sell_burst(token_address);
        
        // Buy if either signal is active (or both)
        if traditional_buy_signal || slope_buy_signal {
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
                let traditional_sell_signal = self.has_buy_burst(token_address);
                
                // Close position when has slope buy burst
                let slope_sell_signal = self.has_slope_buy_burst(token_address);
                
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
        params.insert("min_sol_in_curve".to_string(), format!("{}", self.min_sol_in_curve));
        params.insert("use_zscore_cvd".to_string(), self.use_zscore_cvd.to_string());
        params.insert("use_slope_cvd".to_string(), self.use_slope_cvd.to_string());
        params.insert("slope_window".to_string(), self.slope_window.to_string());
        params.insert("slope_threshold".to_string(), format!("{}", self.slope_threshold));
        params.insert("slope_zscore_gate".to_string(), self.slope_zscore_gate.to_string());
        params.insert("slope_zscore_k".to_string(), format!("{}", self.slope_zscore_k));
        params.insert("slope_zscore_window".to_string(), self.slope_zscore_window.to_string());
        params.insert("slope_normalize_window".to_string(), self.slope_normalize_window.to_string());
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





