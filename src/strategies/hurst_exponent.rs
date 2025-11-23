use anyhow::Result;
use polars::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::strategies::strategy_base::{SellReason, TradingStrategy};
use super::helpers::{can_buy_with_constraints, can_sell_with_hold_blocks, get_block_id, get_column_value, calculate_sol_value, get_sol_in_curve, update_threshold_tracking};

/// DCA (Dollar Cost Averaging) trading strategy that:
/// 1. ENTRY: Initial buy when Z-score falls below buy_threshold (oversold)
/// 2. DCA: Additional buys every 5% drawdown from average cost (up to max_buys)
/// 3. EXIT: Sells on trailing profit target OR tightening stop loss
/// 4. RISK MGMT: Profit target & stop loss tighten with more DCA entries
///
/// Evolved from gemini-mean-reversion.py with advanced DCA risk management
/// OPTIMIZED VERSION: Uses VecDeques and caching for high-volume tokens
pub struct InterTradeStrategy {
	// Strategy parameters
	window_size: usize,
	buy_threshold: f64,
	sell_threshold: f64,
	min_sol_in_curve: f64,
	z_score_column: String, // Configurable column for z-score calculation
	max_buys: usize,
	min_hold_blocks: i64,
	min_blocks_between_buys: i64,
	min_blocks_between_sell_buy: i64,

	// Token tracking
	tokens_to_skip: HashSet<String>,
	z_score_data_history: HashMap<String, VecDeque<f64>>, // Renamed to be column-agnostic
	tokens_ready: HashSet<String>,
	tokens_reached_threshold: HashSet<String>,

	// Position tracking - supports multiple buys per token
	current_positions: HashMap<String, Vec<f64>>, // {token_address: [list of buy_prices]}
	buy_block_ids: HashMap<String, Vec<i64>>, // {token_address: [list of buy_block_ids]}
	buy_transaction_indices: HashMap<String, Vec<usize>>, // {token_address: [list of transaction_indices]}
	sell_block_ids: HashMap<String, i64>, // {token_address: sell_block_id}

	// Stop loss mechanism
	stop_loss_blacklist: HashSet<String>,

	// Z-score tracking for each trade
	trade_z_scores: HashMap<String, f64>,
	last_buy_z_score: Option<f64>,

	// Debug counters
	total_buy_signals: i64,
	total_sell_signals: i64,
	insufficient_data_rejections: i64,
	z_score_calculations: i64,
	sol_curve_rejections: i64,
	skipped_tokens_count: i64,
	stop_loss_triggers: i64,

	// Output control
	verbose: bool, // Controls debug output verbosity
}

impl HurstExponentStrategy {
	/// Create a new mean reversion strategy
	pub fn new(
		window_size: usize,
		buy_threshold: f64,
		sell_threshold: f64,
		min_sol_in_curve: f64,
		z_score_column: &str,
		max_buys: usize,
		min_blocks_between_buys: i64,
		min_hold_blocks: i64,
		min_blocks_between_sell_buy: i64,
		verbose: bool,
	) -> Self {
		let tokens_to_skip: HashSet<String> = HashSet::new();

		Self {
			window_size,
			buy_threshold,
			sell_threshold,
			min_sol_in_curve,
			z_score_column: z_score_column.to_string(),
			max_buys,
			min_hold_blocks,
			min_blocks_between_buys,
			min_blocks_between_sell_buy,

			tokens_to_skip,
			z_score_data_history: HashMap::new(),
			tokens_ready: HashSet::new(),
			tokens_reached_threshold: HashSet::new(),

			current_positions: HashMap::new(),
			buy_block_ids: HashMap::new(),
			buy_transaction_indices: HashMap::new(),
			sell_block_ids: HashMap::new(),

			stop_loss_blacklist: HashSet::new(),

			trade_z_scores: HashMap::new(),
			last_buy_z_score: None,

			total_buy_signals: 0,
			total_sell_signals: 0,
			insufficient_data_rejections: 0,
			z_score_calculations: 0,
			sol_curve_rejections: 0,
			skipped_tokens_count: 0,
			stop_loss_triggers: 0,

			verbose,
		}
	}

	/// Calculate the current Z-score for a token's specified column value (optimized)
	fn calculate_z_score(
		&mut self,
		token_address: &str,
		current_value: f64,
	) -> Option<f64> {
		let history = match self.z_score_data_history.get(token_address) {
			Some(history) => history,
			None => return None,
		};

		// Need at least window_size data points
		if history.len() < self.window_size {
			self.insufficient_data_rejections += 1;
			return None;
		}

		// Calculate mean and std directly from VecDeque without allocation
		let start_idx = history.len().saturating_sub(self.window_size);
		let end_idx = history.len(); // Use all available historical data

		if end_idx <= start_idx {
			return None;
		}

		// Calculate rolling mean directly from deque
		let mut sum = 0.0;
		let mut count = 0;
		for i in start_idx..end_idx {
			sum += history[i];
			count += 1;
		}

		if count != self.window_size {
			return None;
		}

		let rolling_mean = sum / count as f64;

		// Calculate variance directly without collecting
		let mut variance_sum = 0.0;
		for i in start_idx..end_idx {
			let diff = history[i] - rolling_mean;
			variance_sum += diff * diff;
		}

		let variance = variance_sum / count as f64;
		let rolling_std = variance.sqrt();

		// Avoid division by zero
		if rolling_std == 0.0 {
			return None;
		}

		// Calculate Z-score
		let z_score = (current_value - rolling_mean) / rolling_std;
		self.z_score_calculations += 1;

		Some(z_score)
	}

	/// Get the configurable z-score column value from row data
	fn get_z_score_column_value(
		&self,
		row_data: &HashMap<String, AnyValue>,
	) -> f64 {
		get_column_value(row_data, &self.z_score_column)
	}
}

impl TradingStrategy for HurstExponentStrategy {
	fn should_buy(
		&mut self,
		token_address: &str,
		current_price: f64,
		current_index: usize,
		_row: &AnyValue,
		row_data: &HashMap<String, AnyValue>,
	) -> Result<bool> {
		let current_sol_curve = get_sol_in_curve(row_data);
		let current_z_score_value = self.get_z_score_column_value(row_data);
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

		// Calculate Z-score using the configurable column
		let z_score = match self
			.calculate_z_score(token_address, current_z_score_value)
		{
			Some(z_score) => z_score,
			None => return Ok(false),
		};

		// Buy signal: Z-score below threshold
		if z_score < self.buy_threshold {
			self.total_buy_signals += 1;
			// Verbose logging enabled for comparison
			if self.verbose {
				println!(
					"\n[Index {}] 📈 MEAN REVERSION BUY for {}",
					current_index,
					&token_address[token_address.len() - 8..]
				);
				println!(
					"  Z-score: {:.3} (threshold: {}) [{}]",
					z_score, self.buy_threshold, self.z_score_column
				);
				println!(
					"  {} value: {:.6}",
					self.z_score_column, current_z_score_value
				);
				println!("  SOL in curve: {:.2}", current_sol_curve);
				println!("  Current price: {:.10e}", current_price);
			}
			// Store Z-score for tracking
			self.last_buy_z_score = Some(z_score);
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
		let current_z_score_value = self.get_z_score_column_value(row_data);
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

		// Check minimum hold blocks constraint (using universal constraints)
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

		// STOP LOSS: Check for 65% loss first (takes priority over mean reversion)
		let buy_prices = &self.current_positions[token_address];

		for (i, buy_price) in buy_prices.iter().enumerate() {
			let loss_percentage = (buy_price - current_price) / buy_price;

			if loss_percentage >= 0.65 {
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

				// println!("\n[Index {}] 🛑 STOP LOSS TRIGGERED for {}", current_index, token_address);
				// println!("  Position {}/{} triggered stop loss", i + 1, buy_prices.len());
				// println!("  Buy block ID: {}", buy_block_id);
				// println!("  Loss: {:.1}% (buy: {:.10e}, current: {:.10e})", loss_percentage * 100.0, buy_price, current_price);
				// println!("  Token blacklisted - will never re-enter");

				return Ok((true, SellReason::StopLoss(loss_percentage)));
			}
		}

		// Calculate Z-score using the configurable column
		let z_score = match self
			.calculate_z_score(token_address, current_z_score_value)
		{
			Some(z_score) => z_score,
			None => {
				return Ok((
					false,
					SellReason::Strategy("NO_Z_SCORE".to_string()),
				));
			}
		};

		// Sell signal: Z-score above threshold
		if z_score > self.sell_threshold {
			self.total_sell_signals += 1;
			let reason = format!("MEAN_REVERSION_Z_{:.2}", z_score);

			// Verbose logging enabled for comparison
			if self.verbose {
				println!(
					"\n[Index {}] 🚨 MEAN REVERSION SELL for {}",
					current_index,
					&token_address[token_address.len() - 8..]
				);
				println!(
					"  Z-score: {:.3} (threshold: {}) [{}]",
					z_score, self.sell_threshold, self.z_score_column
				);
				println!(
					"  {} value: {:.6}",
					self.z_score_column, current_z_score_value
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
		let current_z_score_value = self.get_z_score_column_value(row_data);

		// Initialize VecDeque for new tokens (for z-score calculation)
		if !self.z_score_data_history.contains_key(token_address) {
			let max_history = self.window_size + 500; // Keep some extra for robustness
			self.z_score_data_history.insert(
				token_address.to_string(),
				VecDeque::with_capacity(max_history),
			);
		}

		// O(1) append operation with automatic eviction of old data
		let history = self.z_score_data_history.get_mut(token_address).unwrap();
		if history.len() >= self.window_size + 500 {
			history.pop_front();
		}
		history.push_back(current_z_score_value);

		// Track when tokens first reach the SOL threshold
		update_threshold_tracking(token_address, current_sol_curve, self.min_sol_in_curve, &mut self.tokens_reached_threshold);

		// Mark token as ready when we have enough data
		if history.len() >= self.window_size
			&& !self.tokens_ready.contains(token_address)
		{
			self.tokens_ready.insert(token_address.to_string());
		}

		Ok(())
	}

	fn get_strategy_name(&self) -> String {
		format!(
			"Mean Reversion w/ {} SOL Threshold Gate [Z-score on: {}] (max_buys: {}, hold_blocks: {}, sell_to_buy_blocks: {}, buy_to_buy_blocks: {})",
			self.min_sol_in_curve,
			self.z_score_column,
			self.max_buys,
			self.min_hold_blocks,
			self.min_blocks_between_sell_buy,
			self.min_blocks_between_buys
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
		stats.insert(
			"sol_curve_rejections".to_string(),
			self.sol_curve_rejections,
		);
		stats.insert(
			"z_score_calculations".to_string(),
			self.z_score_calculations,
		);
		stats.insert(
			"tokens_ready_count".to_string(),
			self.tokens_ready.len() as i64,
		);
		stats.insert(
			"total_tokens_tracked".to_string(),
			self.z_score_data_history.len() as i64,
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

	fn get_buy_z_score(&self) -> Option<f64> {
		self.last_buy_z_score
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
			"z_score_column".to_string(),
			format!("\"{}\"", self.z_score_column),
		);
		params.insert(
			"min_sol_in_curve".to_string(),
			self.min_sol_in_curve.to_string(),
		);
		params
	}

	fn get_system_name(&self) -> String {
		"MeanReversionStrategy".to_string()
	}

	fn get_slippage_tolerance(&self) -> f64 {
		crate::strategies::common_config::SLIPPAGE_TOLERANCE_PERCENT
	}

}
