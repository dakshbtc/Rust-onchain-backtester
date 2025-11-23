use anyhow::Result;
use chrono::{DateTime, Utc};
use polars::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::strategies::strategy_base::{SellReason, TradingStrategy};
use super::helpers::{
    can_buy_with_constraints,
    can_sell_with_hold_blocks,
    get_block_id,
    get_timestamp_ms,
    get_column_value,
    calculate_sol_value,
    get_sol_in_curve,
    update_threshold_tracking,
    get_price_column_value,
    get_volume_column_value,
};

/// DCA (Dollar Cost Averaging) trading strategy that:
/// 1. ENTRY: Initial buy when price falls below VWAP by buy_threshold percentage
/// 2. DCA: Additional buys every 5% drawdown from average cost (up to max_buys)
/// 3. EXIT: Sells when price rises above VWAP by sell_threshold percentage
/// 4. RISK MGMT: Profit target & stop loss tighten with more DCA entries
///
/// Uses VWAP (Volume Weighted Average Price) for entry/exit signals
/// OPTIMIZED VERSION: Uses VecDeques and caching for high-volume tokens
pub struct VwapStrategy {
	// Strategy parameters
	window_size: usize,
	buy_threshold: f64,  // Percentage below VWAP to trigger buy (e.g., -0.05 for 5% below)
	sell_threshold: f64, // Percentage above VWAP to trigger sell (e.g., 0.03 for 3% above)
	stop_loss_threshold: f64, // Stop loss threshold (e.g., 0.50 for 50% loss)
	min_sol_in_curve: f64,
	price_column: String,   // Column name for price data
	volume_column: String,  // Column name for volume data
	max_buys: usize,
	min_blocks_between_buys: i64,
	min_hold_blocks: i64,
	min_blocks_between_sell_buy: i64,
	max_hold_time_minutes: i64, // Maximum time to hold position in minutes
	use_dynamic_thresholds: bool, // Whether to use dynamic thresholds based on SOL in curve
	slippage_tolerance_percent: f64, // Slippage tolerance for buys (e.g., 0.50 = 50%)

	// Trade velocity parameters
	use_trade_velocity: bool, // Whether to use trade velocity for threshold adjustment
	velocity_window_seconds: f64, // Time window for velocity calculation (e.g., 30.0 = 30 seconds)
	high_velocity_threshold: f64, // Trades per second threshold for "high velocity" (e.g., 2.0)
	low_velocity_threshold: f64, // Trades per second threshold for "low velocity" (e.g., 0.2)
	high_velocity_multiplier: f64, // Threshold multiplier for high velocity (e.g., 0.8 = tighter thresholds)
	low_velocity_multiplier: f64, // Threshold multiplier for low velocity (e.g., 1.2 = wider thresholds)

	// Token tracking - VWAP requires price and volume history
	tokens_to_skip: HashSet<String>,
	price_history: HashMap<String, VecDeque<f64>>, // Price history for VWAP calculation
	volume_history: HashMap<String, VecDeque<f64>>, // Volume history for VWAP calculation
	timestamp_history: HashMap<String, VecDeque<i64>>, // Timestamp history for velocity calculation (milliseconds)
	tokens_ready: HashSet<String>,
	tokens_reached_threshold: HashSet<String>,

	// Position tracking - supports multiple buys per token
	current_positions: HashMap<String, Vec<f64>>, // {token_address: [list of buy_prices]}
	buy_block_ids: HashMap<String, Vec<i64>>, // {token_address: [list of buy_block_ids]}
	buy_transaction_indices: HashMap<String, Vec<usize>>, // {token_address: [list of transaction_indices]}
	buy_timestamps: HashMap<String, Vec<i64>>, // {token_address: [list of buy_timestamps_ms]} - for time-based exit
	sell_block_ids: HashMap<String, i64>, // {token_address: sell_block_id}

	// Stop loss mechanism
	stop_loss_blacklist: HashSet<String>,

	// VWAP tracking for each trade
	trade_vwap_deviations: HashMap<String, f64>,
	last_buy_vwap_deviation: Option<f64>,

	// Debug counters
	total_buy_signals: i64,
	total_sell_signals: i64,
	insufficient_data_rejections: i64,
	vwap_calculations: i64,
	sol_curve_rejections: i64,
	skipped_tokens_count: i64,
	stop_loss_triggers: i64,

	// Output control
	verbose: bool, // Controls debug output verbosity
}

impl VwapStrategy {
	/// Create a new VWAP strategy
	pub fn new(
		window_size: usize,
		buy_threshold: f64,
		sell_threshold: f64,
		stop_loss_threshold: f64,
		min_sol_in_curve: f64,
		price_column: &str,
		volume_column: &str,
		max_buys: usize,
		min_blocks_between_buys: i64,
		min_hold_blocks: i64,
		min_blocks_between_sell_buy: i64,
		max_hold_time_minutes: i64,
		use_dynamic_thresholds: bool,
		slippage_tolerance_percent: f64,
		use_trade_velocity: bool,
		velocity_window_seconds: f64,
		high_velocity_threshold: f64,
		low_velocity_threshold: f64,
		high_velocity_multiplier: f64,
		low_velocity_multiplier: f64,
		verbose: bool,
	) -> Self {
		let tokens_to_skip: HashSet<String> = HashSet::new();

		Self {
			window_size,
			buy_threshold,
			sell_threshold,
			stop_loss_threshold,
			min_sol_in_curve,
			price_column: price_column.to_string(),
			volume_column: volume_column.to_string(),
			max_buys,
			min_blocks_between_buys,
			min_hold_blocks,
			min_blocks_between_sell_buy,
			max_hold_time_minutes,
			use_dynamic_thresholds,
			slippage_tolerance_percent,

			use_trade_velocity,
			velocity_window_seconds,
			high_velocity_threshold,
			low_velocity_threshold,
			high_velocity_multiplier,
			low_velocity_multiplier,

			tokens_to_skip,
			price_history: HashMap::new(),
			volume_history: HashMap::new(),
			timestamp_history: HashMap::new(),
			tokens_ready: HashSet::new(),
			tokens_reached_threshold: HashSet::new(),

			current_positions: HashMap::new(),
			buy_block_ids: HashMap::new(),
			buy_transaction_indices: HashMap::new(),
			buy_timestamps: HashMap::new(),
			sell_block_ids: HashMap::new(),

			stop_loss_blacklist: HashSet::new(),

			trade_vwap_deviations: HashMap::new(),
			last_buy_vwap_deviation: None,

			total_buy_signals: 0,
			total_sell_signals: 0,
			insufficient_data_rejections: 0,
			vwap_calculations: 0,
			sol_curve_rejections: 0,
			skipped_tokens_count: 0,
			stop_loss_triggers: 0,

			verbose,
		}
	}

	/// Calculate VWAP (Volume Weighted Average Price) for a token
	fn calculate_vwap(&mut self, token_address: &str, window_size: usize) -> Option<f64> {
		let price_hist = match self.price_history.get(token_address) {
			Some(hist) => hist,
			None => return None,
		};

		let volume_hist = match self.volume_history.get(token_address) {
			Some(hist) => hist,
			None => return None,
		};

		// Need at least window_size data points
		if price_hist.len() < window_size || volume_hist.len() < window_size {
			self.insufficient_data_rejections += 1;
			return None;
		}

		// Ensure both histories have the same length
		if price_hist.len() != volume_hist.len() {
			return None;
		}

		let start_idx = price_hist.len().saturating_sub(window_size);
		let end_idx = price_hist.len();

		if end_idx <= start_idx {
			return None;
		}

		// Calculate VWAP: Sum(Price * Volume) / Sum(Volume)
		let mut price_volume_sum = 0.0;
		let mut volume_sum = 0.0;

		for i in start_idx..end_idx {
			let price = price_hist[i];
			let volume = volume_hist[i];
			
			price_volume_sum += price * volume;
			volume_sum += volume;
		}

		// Avoid division by zero
		if volume_sum == 0.0 {
			return None;
		}

		let vwap = price_volume_sum / volume_sum;
		self.vwap_calculations += 1;

		Some(vwap)
	}

	/// Calculate price deviation from VWAP as a percentage
	fn calculate_price_deviation_from_vwap(
		&mut self,
		token_address: &str,
		current_price: f64,
		window_size: usize,
	) -> Option<f64> {
		let vwap = match self.calculate_vwap(token_address, window_size) {
			Some(vwap) => vwap,
			None => return None,
		};

		// Avoid division by zero
		if vwap == 0.0 {
			return None;
		}

		// Calculate percentage deviation: (current_price - vwap) / vwap
		let deviation = (current_price - vwap) / vwap;
		
		Some(deviation)
	}




	/// Get dynamic parameters based on current SOL in curve
	fn get_dynamic_thresholds(&self, current_sol_in_curve: f64) -> (f64, f64, usize) {
		// Returns (buy_threshold, sell_threshold, window_size)
		// Negative buy_threshold means price must be below VWAP
		// Positive sell_threshold means price must be above VWAP
		
		if current_sol_in_curve < 100.0 {
			(-0.22, 0.10, 300)   // More aggressive: 8% below VWAP to buy, 5% above to sell, 50-period window
		} else if current_sol_in_curve < 125.0 {
			(-0.20, 0.09, 300)   // 6% below VWAP to buy, 4% above to sell, 75-period window
		} else if current_sol_in_curve < 150.0 {
			(-0.18, 0.09, 300)   // 6% below VWAP to buy, 4% above to sell, 75-period window
		} else if current_sol_in_curve < 250.0 {
			(-0.12, 0.08, 300)  // 5% below VWAP to buy, 3% above to sell, 100-period window
		} else if current_sol_in_curve < 500.0 {
			(-0.10, 0.08, 300) // 4% below VWAP to buy, 2.5% above to sell, 125-period window
		} else {
			(-0.04, 0.025, 125) // 4% below VWAP to buy, 2.5% above to sell, 125-period window
		} 
	}

	/// Calculate trade velocity (trades per second) for a token within the velocity window
	fn calculate_trade_velocity(&self, token_address: &str, current_timestamp_ms: i64) -> f64 {
		if !self.use_trade_velocity {
			return 0.0; // Return 0 if velocity is disabled
		}

		let timestamp_hist = match self.timestamp_history.get(token_address) {
			Some(hist) => hist,
			None => return 0.0,
		};

		if timestamp_hist.is_empty() {
			return 0.0;
		}

		// Convert velocity window from seconds to milliseconds
		let velocity_window_ms = (self.velocity_window_seconds * 1000.0) as i64;
		let window_start_ms = current_timestamp_ms - velocity_window_ms;

		// Count trades within the velocity window
		let mut trades_in_window = 0;
		for &timestamp in timestamp_hist.iter().rev() {
			if timestamp >= window_start_ms {
				trades_in_window += 1;
			} else {
				break; // Timestamps are in chronological order, so we can break
			}
		}

		// Calculate trades per second
		if trades_in_window == 0 {
			0.0
		} else {
			trades_in_window as f64 / self.velocity_window_seconds
		}
	}

	/// Get velocity-adjusted thresholds based on current trade velocity
	fn get_velocity_adjusted_thresholds(&self, token_address: &str, current_timestamp_ms: i64, base_buy_threshold: f64, base_sell_threshold: f64) -> (f64, f64) {
		if !self.use_trade_velocity {
			return (base_buy_threshold, base_sell_threshold);
		}

		let velocity = self.calculate_trade_velocity(token_address, current_timestamp_ms);
		
		let multiplier = if velocity >= self.high_velocity_threshold {
			// High velocity = more activity = tighter thresholds (more conservative)
			self.high_velocity_multiplier
		} else if velocity <= self.low_velocity_threshold {
			// Low velocity = less activity = wider thresholds (more aggressive)
			self.low_velocity_multiplier
		} else {
			// Medium velocity = no adjustment
			1.0
		};

		// Apply multiplier to both thresholds
		// For buy threshold (negative): multiplier < 1.0 makes it less negative (tighter)
		// For sell threshold (positive): multiplier < 1.0 makes it smaller (tighter)
		let adjusted_buy_threshold = base_buy_threshold * multiplier;
		let adjusted_sell_threshold = base_sell_threshold * multiplier;

		(adjusted_buy_threshold, adjusted_sell_threshold)
	}
}

impl TradingStrategy for VwapStrategy {
	fn should_buy(
		&mut self,
		token_address: &str,
		current_price: f64,
		current_index: usize,
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

		// Check if token is in the skip list
		if self.tokens_to_skip.contains(token_address) {
			self.skipped_tokens_count += 1;
			return Ok(false);
		}

		// Check if token triggered stop loss before - never re-enter
		if self.stop_loss_blacklist.contains(token_address) {
			return Ok(false);
		}

		// Get thresholds based on configuration
		let (base_buy_threshold, _, window_size) = if self.use_dynamic_thresholds {
			self.get_dynamic_thresholds(current_sol_curve)
		} else {
			(self.buy_threshold, self.sell_threshold, self.window_size)
		};

		// Apply velocity adjustments if enabled
		let current_timestamp_ms = get_timestamp_ms(row_data);
		let (buy_threshold, _) = self.get_velocity_adjusted_thresholds(
			token_address, 
			current_timestamp_ms, 
			base_buy_threshold, 
			0.0 // We don't need sell threshold here
		);

		// Calculate price deviation from VWAP
		let price_deviation = match self
			.calculate_price_deviation_from_vwap(token_address, current_price, window_size)
		{
			Some(deviation) => deviation,
			None => return Ok(false),
		};

		// Buy signal: Price is below VWAP by buy threshold percentage
		if price_deviation < buy_threshold {
			self.total_buy_signals += 1;
			// Verbose logging enabled for comparison
			if self.verbose {
				let vwap = self.calculate_vwap(token_address, window_size).unwrap_or(0.0);
				let velocity = self.calculate_trade_velocity(token_address, current_timestamp_ms);
				println!(
					"\n[Index {}] 📈 VWAP BUY for {}",
					current_index,
					&token_address[token_address.len() - 8..]
				);
				println!(
					"  Price deviation: {:.3}% (threshold: {:.3}%)",
					price_deviation * 100.0, buy_threshold * 100.0
				);
				if self.use_trade_velocity && base_buy_threshold != buy_threshold {
					println!(
						"  Base threshold: {:.3}%, adjusted for velocity: {:.2} trades/sec",
						base_buy_threshold * 100.0, velocity
					);
				}
				println!(
					"  Current price: {:.10e}, VWAP: {:.10e}",
					current_price, vwap
				);
				println!("  SOL in curve: {:.2}", current_sol_curve);
			}
			// Store price deviation for tracking
			self.last_buy_vwap_deviation = Some(price_deviation);
			
			return Ok(true);
		}

		Ok(false)
	}

	fn should_sell(
		&mut self,
		token_address: &str,
		current_price: f64,
		current_index: usize,
		_row: &AnyValue,
		row_data: &HashMap<String, AnyValue>,
	) -> Result<(bool, SellReason)> {
		let current_sol_curve = get_sol_in_curve(row_data);
		let current_block_id = get_block_id(row_data);

		// Check if we have any positions
		if !self.current_positions.contains_key(token_address)
			|| self.current_positions[token_address].is_empty()
		{
			return Ok((
				false,
				SellReason::Strategy("NO_POSITION".to_string()),
			));
		}

		// Check if token is in the skip list
		if self.tokens_to_skip.contains(token_address) {
			self.skipped_tokens_count += 1;
			return Ok((
				false,
				SellReason::Strategy("SKIPPED_TOKEN".to_string()),
			));
		}

		// Check minimum hold blocks constraint
		if !can_sell_with_hold_blocks(
			token_address,
			current_block_id,
			&self.buy_block_ids,
			self.min_hold_blocks,
		) {
			return Ok((
				false,
				SellReason::Strategy("MIN_HOLD_BLOCKS_NOT_MET".to_string()),
			));
		}

		// Check maximum hold time constraint (time-based exit)
		let current_timestamp_ms = get_timestamp_ms(row_data);
		if let Some(buy_timestamps) = self.buy_timestamps.get(token_address) {
			if !buy_timestamps.is_empty() {
				let first_buy_timestamp_ms = buy_timestamps[0];
				let seconds_held = (current_timestamp_ms - first_buy_timestamp_ms) / 1000;
				
				if seconds_held >= self.max_hold_time_minutes * 60 {
					let minutes_held = seconds_held / 60;
					return Ok((
						true, 
						SellReason::Strategy(format!("MAX_HOLD_TIME_{}min", minutes_held))
					));
				}
			}
		}

		// STOP LOSS: Check for 60% loss first (takes priority over VWAP signal)
		let buy_prices = &self.current_positions[token_address];

		for (i, buy_price) in buy_prices.iter().enumerate() {
			let loss_percentage = (buy_price - current_price) / buy_price;

			if loss_percentage >= self.stop_loss_threshold {
				self.stop_loss_triggers += 1;
				// Add to blacklist - never re-enter this token
				self.stop_loss_blacklist.insert(token_address.to_string());

				let _buy_block_id = if let Some(buy_block_ids) =
					self.buy_block_ids.get(token_address)
				{
					buy_block_ids.get(i).unwrap_or(&0).clone()
				} else {
					0
				};

				return Ok((true, SellReason::StopLoss(loss_percentage)));
			}
		}

		// Get thresholds based on configuration
		let (_, base_sell_threshold, window_size) = if self.use_dynamic_thresholds {
			self.get_dynamic_thresholds(current_sol_curve)
		} else {
			(self.buy_threshold, self.sell_threshold, self.window_size)
		};

		// Apply velocity adjustments if enabled
		let current_timestamp_ms = get_timestamp_ms(row_data);
		let (_, sell_threshold) = self.get_velocity_adjusted_thresholds(
			token_address, 
			current_timestamp_ms, 
			0.0, // We don't need buy threshold here
			base_sell_threshold
		);

		// Calculate price deviation from VWAP
		let price_deviation = match self
			.calculate_price_deviation_from_vwap(token_address, current_price, window_size)
		{
			Some(deviation) => deviation,
			None => {
				return Ok((
					false,
					SellReason::Strategy("NO_VWAP".to_string()),
				));
			}
		};

		// Sell signal: Price is above VWAP by sell threshold percentage
		if price_deviation > sell_threshold {
			self.total_sell_signals += 1;
			let reason = format!("VWAP_DEVIATION_{:.2}%", price_deviation * 100.0);

			// Verbose logging enabled for comparison
			if self.verbose {
				let vwap = self.calculate_vwap(token_address, window_size).unwrap_or(0.0);
				let velocity = self.calculate_trade_velocity(token_address, current_timestamp_ms);
				println!(
					"\n[Index {}] 🚨 VWAP SELL for {}",
					current_index,
					&token_address[token_address.len() - 8..]
				);
				println!(
					"  Price deviation: {:.3}% (threshold: {:.3}%)",
					price_deviation * 100.0, sell_threshold * 100.0
				);
				if self.use_trade_velocity && base_sell_threshold != sell_threshold {
					println!(
						"  Base threshold: {:.3}%, adjusted for velocity: {:.2} trades/sec",
						base_sell_threshold * 100.0, velocity
					);
				}
				println!(
					"  Current price: {:.10e}, VWAP: {:.10e}",
					current_price, vwap
				);
				println!("  SOL in curve: {:.2}", current_sol_curve);
			}
			return Ok((true, SellReason::Strategy(reason)));
		}

		Ok((false, SellReason::Strategy("NO_SELL_SIGNAL".to_string())))
	}

	fn update_data(
		&mut self,
		token_address: &str,
		_price: f64,
		_tx_index: usize,
		_date: &str,
		row_data: &HashMap<String, AnyValue>,
	) -> Result<()> {
		// Skip data tracking for tokens in the skip list
		if self.tokens_to_skip.contains(token_address) {
			return Ok(());
		}

		let current_sol_curve = get_sol_in_curve(row_data);
		let current_price = get_price_column_value(row_data, &self.price_column);
		let current_volume = get_volume_column_value(row_data, &self.volume_column);
		let current_timestamp_ms = get_timestamp_ms(row_data);

		// Initialize VecDeques for new tokens (for VWAP calculation and velocity tracking)
		if !self.price_history.contains_key(token_address) {
			let max_history = self.window_size + 300; // Keep some extra for robustness
			self.price_history.insert(
				token_address.to_string(),
				VecDeque::with_capacity(max_history),
			);
			self.volume_history.insert(
				token_address.to_string(),
				VecDeque::with_capacity(max_history),
			);
			// Initialize timestamp history for velocity calculation
			self.timestamp_history.insert(
				token_address.to_string(),
				VecDeque::with_capacity(max_history),
			);
		}

		// O(1) append operation with automatic eviction of old data
		let price_hist = self.price_history.get_mut(token_address).unwrap();
		let volume_hist = self.volume_history.get_mut(token_address).unwrap();
		let timestamp_hist = self.timestamp_history.get_mut(token_address).unwrap();
		
		if price_hist.len() >= self.window_size + 300 {
			price_hist.pop_front();
		}
		price_hist.push_back(current_price);

		if volume_hist.len() >= self.window_size + 300 {
			volume_hist.pop_front();
		}
		volume_hist.push_back(current_volume);

		// Track timestamps for velocity calculation
		// Keep more timestamp history than VWAP window since velocity uses time-based windows
		let velocity_window_ms = (self.velocity_window_seconds * 1000.0) as i64;
		let max_timestamp_entries = ((velocity_window_ms / 1000) as usize).max(self.window_size + 50);
		
		if timestamp_hist.len() >= max_timestamp_entries {
			timestamp_hist.pop_front();
		}
		timestamp_hist.push_back(current_timestamp_ms);

		// Track when tokens first reach the SOL threshold
		update_threshold_tracking(token_address, current_sol_curve, self.min_sol_in_curve, &mut self.tokens_reached_threshold);

		// Mark token as ready when we have enough data
		if price_hist.len() >= self.window_size && volume_hist.len() >= self.window_size
			&& !self.tokens_ready.contains(token_address)
		{
			self.tokens_ready.insert(token_address.to_string());
		}

		Ok(())
	}

	fn get_strategy_name(&self) -> String {
		let velocity_info = if self.use_trade_velocity {
			format!(" + Velocity({:.1}s window, {:.2}-{:.1} trades/sec, {:.2}x-{:.2}x multipliers)", 
				self.velocity_window_seconds,
				self.low_velocity_threshold,
				self.high_velocity_threshold,
				self.high_velocity_multiplier,
				self.low_velocity_multiplier
			)
		} else {
			"".to_string()
		};

		format!(
			"VWAP w/ {} SOL Threshold Gate [Price: {}, Volume: {}]{} (max_buys: {}, hold_blocks: {}, sell_to_buy_blocks: {}, buy_to_buy_blocks: {}, max_hold_time: {}min)",
			self.min_sol_in_curve,
			self.price_column,
			self.volume_column,
			velocity_info,
			self.max_buys,
			self.min_hold_blocks,
			self.min_blocks_between_sell_buy,
			self.min_blocks_between_buys,
			self.max_hold_time_minutes
		)
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
			self.buy_timestamps
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

		// Store buy block ID and timestamp
		let (current_block_id, current_timestamp_ms) = if let Some(row_data) = row_data {
			(get_block_id(row_data), get_timestamp_ms(row_data))
		} else {
			(0, 0) // Default to 0 if no data available
		};
		
		self.buy_block_ids
			.get_mut(token_address)
			.unwrap()
			.push(current_block_id);
			
		self.buy_timestamps
			.get_mut(token_address)
			.unwrap()
			.push(current_timestamp_ms);

		// Store the VWAP deviation that triggered this buy
		if let Some(deviation) = self.last_buy_vwap_deviation.take() {
			self.trade_vwap_deviations
				.insert(token_address.to_string(), deviation);
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

		// Clear buy timestamp tracking
		if let Some(timestamps) = self.buy_timestamps.get_mut(token_address) {
			timestamps.clear();
		}

		// Track sell block ID for minimum blocks between sell and buy constraint
		let current_block_id = if let Some(row_data) = row_data {
			get_block_id(row_data)
		} else {
			0 // Default to 0 if no data available
		};
		self.sell_block_ids
			.insert(token_address.to_string(), current_block_id);

		// Clean up VWAP tracking for closed position
		self.trade_vwap_deviations.remove(token_address);

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
		stats.insert(
			"sol_curve_rejections".to_string(),
			self.sol_curve_rejections,
		);
		stats.insert(
			"vwap_calculations".to_string(),
			self.vwap_calculations,
		);
		stats.insert(
			"tokens_ready_count".to_string(),
			self.tokens_ready.len() as i64,
		);
		stats.insert(
			"total_tokens_tracked".to_string(),
			self.price_history.len() as i64,
		);
		stats.insert(
			"tokens_reached_threshold".to_string(),
			self.tokens_reached_threshold.len() as i64,
		);
		stats.insert(
			"skipped_tokens_count".to_string(),
			self.skipped_tokens_count,
		);
		stats.insert("stop_loss_triggers".to_string(), self.stop_loss_triggers);
		stats.insert(
			"blacklisted_tokens_count".to_string(),
			self.stop_loss_blacklist.len() as i64,
		);

		// Calculate open positions
		let open_positions: usize = self
			.current_positions
			.values()
			.map(|positions| positions.len())
			.sum();
		let tokens_with_positions = self
			.current_positions
			.values()
			.filter(|positions| !positions.is_empty())
			.count();

		stats.insert("open_positions".to_string(), open_positions as i64);
		stats.insert(
			"tokens_with_positions".to_string(),
			tokens_with_positions as i64,
		);

		stats
	}

	fn get_export_parameters(&self) -> HashMap<String, String> {
		let mut params = HashMap::new();
		// Format all values as proper JSON values (numbers as numbers, strings as quoted strings)
		params.insert("window_size".to_string(), self.window_size.to_string());
		params.insert(
			"buy_threshold".to_string(),
			self.buy_threshold.to_string(),
		);
		params.insert(
			"sell_threshold".to_string(),
			self.sell_threshold.to_string(),
		);
		params.insert("max_buys".to_string(), self.max_buys.to_string());
		params.insert(
			"price_column".to_string(),
			format!("\"{}\"", self.price_column),
		);
		params.insert(
			"volume_column".to_string(),
			format!("\"{}\"", self.volume_column),
		);
		params.insert(
			"min_sol_in_curve".to_string(),
			self.min_sol_in_curve.to_string(),
		);
		params.insert(
			"slippage_tolerance_percent".to_string(),
			self.slippage_tolerance_percent.to_string(),
		);
		// Add velocity parameters
		params.insert("use_trade_velocity".to_string(), self.use_trade_velocity.to_string());
		params.insert("velocity_window_seconds".to_string(), self.velocity_window_seconds.to_string());
		params.insert("high_velocity_threshold".to_string(), self.high_velocity_threshold.to_string());
		params.insert("low_velocity_threshold".to_string(), self.low_velocity_threshold.to_string());
		params.insert("high_velocity_multiplier".to_string(), self.high_velocity_multiplier.to_string());
		params.insert("low_velocity_multiplier".to_string(), self.low_velocity_multiplier.to_string());
		params
	}

	fn get_system_name(&self) -> String {
		"VwapStrategy".to_string()
	}

	fn get_slippage_tolerance(&self) -> f64 {
		self.slippage_tolerance_percent
	}

}
