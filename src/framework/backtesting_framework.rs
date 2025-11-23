use anyhow::Result;
use chrono::{DateTime, Utc};
use polars::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::framework::fees::get_fee_percent;
use crate::framework::manipulation_detector::ManipulationDetector;
use crate::strategies::strategy_base::{SellReason, TradingStrategy};

/// Portfolio entry for a token position
#[derive(Debug, Clone)]
pub struct PortfolioEntry {
	pub tokens_bought: f64,
	pub sol_invested: f64,
	pub buy_tx_index: usize,
	pub buy_execution_index: usize,
	pub buy_latency_seconds: f64, // Actually represents blocks away from trigger (kept for compatibility)
	pub buy_execution_logic: String,
	pub buy_execution_time: String, // Actual execution timestamp from LatencyData
	pub buy_price_ref: f64,
	pub our_effective_buy_price: f64,
	pub price_after_our_buy: f64,
	pub sol_in_curve_at_our_buy: f64,
	pub sol_in_curve_before_our_buy: f64,
}

/// Trade summary for completed trades
#[derive(Debug, Clone)]
pub struct TradeSummary {
	pub token_address: String,
	pub buy_tx_index: usize,
	pub buy_execution_index: usize,
	pub buy_latency_seconds: f64, // Actually represents blocks away from trigger (kept for compatibility)
	pub buy_execution_time: String, // Actual buy execution timestamp
	pub sell_tx_index: String,    // Can be "LIQUIDATION"
	pub sell_execution_index: Option<usize>,
	pub sell_latency_seconds: Option<f64>, // Actually represents blocks away from trigger (kept for compatibility)
	pub sell_execution_logic: Option<String>,
	pub sell_execution_time: Option<String>, // Actual sell execution timestamp
	pub buy_price_ref: f64,
	pub our_sol_invested: f64,
	pub our_tokens_bought: f64,
	pub our_effective_buy_price: f64,
	pub price_after_our_buy: f64,
	pub sol_in_curve_at_our_buy: f64,
	pub sol_in_curve_before_our_buy: f64,
	pub current_sol_in_curve: f64,
	pub sell_price_ref: f64,
	pub our_sol_received: f64,
	pub our_effective_sell_price: f64,
	pub price_after_our_sell: f64,
	pub pnl_sol: f64,
	pub exit_reason: String,
}

/// Equity curve entry
#[derive(Debug, Clone)]
pub struct EquityEntry {
	pub trade_number: usize,
	pub token_address: String,
	pub trade_pnl: f64,
	pub cumulative_pnl: f64,
	pub peak_balance: f64,
	pub current_drawdown: f64,
	pub max_drawdown: f64,
	pub tx_index: String,
	pub execution_index: usize,
	pub exit_date: String,
	pub exit_index: usize,
	pub trade_type: String,
	pub execution_latency_seconds: f64, // Actually represents blocks away from trigger (kept for compatibility)
}

/// Block ID-based latency-adjusted curve data
#[derive(Debug, Clone)]
pub struct LatencyData {
	pub sol_in_curve: f64,
	pub outstanding_shares: f64,
	pub token_price: f64,
	pub execution_index: usize,
	pub execution_time: DateTime<Utc>,
	pub latency_actual: f64, // Number of blocks away from trigger block
	pub execution_logic: String,
}

/// Individual transaction record for CSV export
#[derive(Debug, Clone)]
pub struct IndividualTransaction {
	pub token_address: String,
	pub timestamp: String,
	pub transaction_type: String, // "BUY" or "SELL"
	pub price: f64,
	pub sol_amount: f64,
}

/// Debug trade information for detailed comparison
#[derive(Debug, Clone)]
pub struct DebugTrade {
	pub trade_number: usize,
	pub action: String, // "BUY" or "SELL"
	pub token_address: String,
	pub tx_index: usize,
	pub timestamp: String,
	pub sol_in_curve_before: f64,
	pub sol_in_curve_after: f64,
	pub token_price_reference: f64,
	pub tokens_transacted: f64,
	pub sol_gross: f64,
	pub sol_fees: f64,
	pub sol_net: f64,
	pub effective_price_per_token: f64,
	pub z_score: Option<f64>,
	pub pnl: Option<f64>, // Only for sells
	pub reason: String,   // Buy reason or sell reason
}

/// Last trade data for each token (for liquidation)
#[derive(Debug, Clone)]
pub struct LastTradeData {
	pub timestamp: String,
	pub sol_in_curve: f64,
	pub outstanding_shares: f64,
	pub token_price: f64,
}

/// Large win details for reporting
#[derive(Debug, Clone)]
pub struct LargeWin {
	pub token_address: String,
	pub pnl: f64,
	pub exit_date: String,
	pub reason: String,
	pub trigger_tx_index: usize, // Original buy transaction index (trigger)
	pub exit_tx_index: usize, // Original sell transaction index (latency adjusted trade)
	pub trigger_execution_index: usize, // Latency adjusted buy execution index
	pub exit_execution_index: usize, // Latency adjusted sell execution index
}

/// Detailed token trade tracking
#[derive(Debug, Clone)]
pub struct TokenTradeDetail {
	pub action: String, // "BUY" or "SELL"
	pub index: usize,
	pub date: String,
	pub sol_amount: f64,
	pub tokens: f64,
	pub price: f64,
	pub sol_in_curve: f64,
	pub pnl: Option<f64>,       // Only for sells
	pub reason: Option<String>, // Only for sells
}

/// SOL level analysis data for parameter optimization
#[derive(Debug, Clone)]
pub struct SolLevelAnalysis {
	pub sol_level: i32,
	pub trade_count: usize,
	pub total_pnl: f64,
	pub win_percentage: f64,
	pub avg_win: f64,
	pub avg_loss: f64,
	pub profit_factor: f64,
	pub min_sol_range: i32,
	pub max_sol_range: i32,
}

/// Trade statistics summary for analysis and export
#[derive(Debug, Clone)]
pub struct TradeStatistics {
	pub total_trades: usize,
	pub cumulative_pnl: f64,
	pub peak_balance: f64,
	pub max_drawdown: f64,
	pub win_count: usize,
	pub loss_count: usize,
	pub win_rate: f64, // Percentage
	pub avg_win: f64,
	pub avg_loss: f64,
	pub profit_factor: f64,
	pub total_wins: f64,
	pub total_losses: f64,
	pub large_wins_count: usize,
	pub large_wins_total: f64,
}

/// Backtest results
#[derive(Debug)]
pub struct BacktestResults {
	pub completed_trades_df: DataFrame,
	pub equity_curve: Vec<EquityEntry>,
	pub cumulative_pnl: f64,
	pub peak_balance: f64,
	pub max_drawdown: f64,
	pub portfolio: HashMap<String, Vec<PortfolioEntry>>, // Track individual DCA buys
	pub large_wins: Vec<LargeWin>,
}

/// Core backtesting framework for token trading strategies
pub struct BacktestFramework {
	// Framework Constants
	virtual_sol_base: f64,
	k_constant: f64,
	initial_tokens: f64,

	// Trading Fees
	buy_fee_percent: f64,
	sell_fee_percent: f64,
	flat_fee_sol_buy: f64,
	flat_fee_sol_sell: f64,
	use_dynamic_fees: bool,

	// Position sizing
	position_size_sol: f64,

	// Latency simulation
	block_latency_max: i64,

	// Portfolio tracking
	portfolio: HashMap<String, Vec<PortfolioEntry>>, // Track individual DCA buys
	completed_trades: Vec<TradeSummary>,
	individual_transactions: Vec<IndividualTransaction>, // Actual executed trades for CSV export

	// Equity curve tracking
	equity_curve: Vec<EquityEntry>,
	cumulative_pnl: f64,
	peak_balance: f64,
	max_drawdown: f64,

	// Data storage
	df: Option<Arc<DataFrame>>,

	// Debug tracking for first 100 trades
	#[allow(dead_code)]
	debug_trades: Vec<DebugTrade>,
	#[allow(dead_code)]
	debug_trade_count: usize,

	// Large wins tracking
	large_wins: Vec<LargeWin>,

	// Export control
	export_trades: bool,

	// Dataset tracking
	dataset_path: Option<Arc<String>>,

	// Precomputed token last trade data for efficient liquidation
	token_last_trade_data: Arc<HashMap<String, LastTradeData>>,

	// Debug output control
	verbose: bool,

	// Slippage tracking
	slippage_failures: i64,
}

impl BacktestFramework {
	/// Create a new backtest framework
	pub fn new(
		position_size_sol: f64,
		buy_fee_percent: f64,
		sell_fee_percent: f64,
		flat_fee_sol_buy: f64,
		flat_fee_sol_sell: f64,
		block_latency_max: i64,
		export_trades: bool,
		verbose: bool,
		use_dynamic_fees: bool,
	) -> Self {
		Self {
			// Framework Constants (from Python)
			virtual_sol_base: 30.0,
			k_constant: 32190005730.0,
			initial_tokens: 1073000191.0,

			// Trading Fees
			buy_fee_percent,
			sell_fee_percent,
			flat_fee_sol_buy,
			flat_fee_sol_sell,
			use_dynamic_fees,

			// Position sizing
			position_size_sol,

			// Latency simulation
			block_latency_max,

			// Portfolio tracking
			portfolio: HashMap::new(), // Now HashMap<String, Vec<PortfolioEntry>>
			completed_trades: Vec::new(),
			individual_transactions: Vec::new(),

			// Equity curve tracking
			equity_curve: Vec::new(),
			cumulative_pnl: 0.0,
			peak_balance: 0.0,
			max_drawdown: 0.0,

			// Data storage
			df: None,

			// Debug tracking
			debug_trades: Vec::new(),
			debug_trade_count: 0,

			// Large wins tracking
			large_wins: Vec::new(),

			// Export control
			export_trades,

			// Dataset tracking
			dataset_path: None,

			// Precomputed token last trade data
			token_last_trade_data: Arc::new(HashMap::new()),

			// Debug output control
			verbose,

			// Reset slippage tracking
			slippage_failures: 0,
		}
	}

	/// Load transaction data from CSV file using Polars
	pub fn load_data<P: AsRef<Path>>(
		&mut self,
		csv_file_path: P,
	) -> Result<bool> {
		match std::fs::File::open(&csv_file_path) {
			Ok(file) => {
				match CsvReader::new(file).finish() {
					Ok(mut df) => {
						// Parse Date column to datetime and add original index
						println!(
							"🔍 Parsing Date column and adding original index..."
						);
						df = df
							.lazy()
							.with_row_index("original_index", None)
							.with_columns([col("Date")
								.str()
								.strptime(
									DataType::Datetime(
										TimeUnit::Milliseconds,
										None,
									),
									StrptimeOptions::default(),
									lit("raise"),
								)
								.alias("Date_parsed")])
							.collect()?;

						self.df = Some(Arc::new(df));
						Ok(true)
					}
					Err(e) => {
						println!("Error reading CSV: {}", e);
						Ok(false)
					}
				}
			}
			Err(e) => {
				println!("Error opening file: {}", e);
				Ok(false)
			}
		}
	}

	/// Validate and clean data with tolerance checking
	pub fn validate_and_clean_data(
		&mut self,
		tolerance_percent: f64,
	) -> Result<HashMap<String, i64>> {
		let df = match &self.df {
			Some(df) => df,
			None => return Ok(HashMap::new()),
		};

		if df.height() == 0 {
			return Ok(HashMap::new());
		}

		println!(
			"🔍 Validating transaction data with {:.1}% tolerance...",
			tolerance_percent * 100.0
		);

		// First apply dust filter
		let original_count = df.height();
		let filtered_df = (**df)
			.clone()
			.lazy()
			.filter(col("Token Amount").gt_eq(1))
			.collect()?;

		let dust_removed = original_count - filtered_df.height();
		println!("  Dust trades removed (< 1 tokens): {}", dust_removed);

		// Apply price manipulation filter (remove trades with 5x+ price spikes) - FAST vectorized approach
		// IMPORTANT: Only compare prices within the same token
		let after_dust_count = filtered_df.height();

		let manipulation_filtered_df = filtered_df
			.clone()
			.lazy()
			.with_columns([
				// Shift prices WITHIN each token group
				col("Token Price")
					.shift(lit(1))
					.over([col("Token Address")])
					.alias("prev_price"),
				col("Token Price")
					.shift(lit(-1))
					.over([col("Token Address")])
					.alias("next_price"),
			])
			.filter(
				// Keep rows where price is NOT 5x higher than both prev and next WITHIN same token
				(col("Token Price").lt(col("prev_price") * lit(5.0)))
                .or(col("Token Price").lt(col("next_price") * lit(5.0)))
                .or(col("prev_price").is_null())  // Keep first trade of each token
                .or(col("next_price").is_null()), // Keep last trade of each token
			)
			.drop(["prev_price", "next_price"])
			.collect()?;

		let manipulation_removed =
			after_dust_count - manipulation_filtered_df.height();
		println!(
			"  Manipulation trades removed (5x+ price spikes): {}",
			manipulation_removed
		);

		// Apply MEV sandwich detection and removal - FAST vectorized approach
		let after_manipulation_count = manipulation_filtered_df.height();

		let mev_filtered_df =
			self.detect_and_remove_mev_sandwiches(manipulation_filtered_df)?;

		let mev_removed = after_manipulation_count - mev_filtered_df.height();
		println!(
			"  MEV sandwich trades removed (same-block 50%+ spike->revert pattern): {}",
			mev_removed
		);

		// Group by token for validation
		let validation_input_count = mev_filtered_df.height();
		println!(
			"  📊 Grouping {} transactions by token...",
			validation_input_count
		);

		// For now, we'll implement a simplified validation
		// In a full implementation, you'd want to replicate the exact Python logic
		self.df = Some(Arc::new(mev_filtered_df));

		let mut stats = HashMap::new();
		stats.insert(
			"removed_trades".to_string(),
			(dust_removed + manipulation_removed + mev_removed) as i64,
		);
		stats.insert("dust_removed".to_string(), dust_removed as i64);
		stats.insert(
			"manipulation_removed".to_string(),
			manipulation_removed as i64,
		);
		stats.insert("mev_removed".to_string(), mev_removed as i64);
		stats.insert("total_trades".to_string(), original_count as i64);
		stats.insert(
			"valid_trades".to_string(),
			self.df.as_ref().unwrap().as_ref().height() as i64,
		);

		println!("📊 Data validation complete:");
		println!("  Original trades: {}", original_count);
		println!("  Dust trades removed: {}", dust_removed);
		println!("  Manipulation trades removed: {}", manipulation_removed);
		println!("  MEV sandwich trades removed: {}", mev_removed);
		println!(
			"  Final valid trades: {}",
			self.df.as_ref().unwrap().height()
		);

		Ok(stats)
	}

	/// Detect and remove MEV sandwich attacks using vectorized operations
	///
	/// MEV sandwich pattern (all trades must be in SAME BLOCK):
	/// 1. Price spikes 50%+ from previous trade (within same token & block)
	/// 2. Within next 1-3 trades (same block), price reverts back to ~original level (±10%)
	/// 3. All trades in the same-block sandwich pattern are removed
	fn detect_and_remove_mev_sandwiches(
		&self,
		df: DataFrame,
	) -> Result<DataFrame> {
		if df.height() == 0 {
			return Ok(df);
		}

		// Add price shift columns within each token group for MEV detection
		// Note: Data is already sorted by Token Address and Date
		let mev_analysis_df = df
			.lazy()
			.with_columns([
				// Previous price (1 trade back) within same token
				col("Token Price")
					.shift(lit(1))
					.over([col("Token Address")])
					.alias("price_1_back"),
				// Previous block ID to ensure same block constraint
				col("Block ID")
					.shift(lit(1))
					.over([col("Token Address")])
					.alias("block_1_back"),
				// Next 1-3 prices and blocks within same token
				col("Token Price")
					.shift(lit(-1))
					.over([col("Token Address")])
					.alias("price_1_ahead"),
				col("Block ID")
					.shift(lit(-1))
					.over([col("Token Address")])
					.alias("block_1_ahead"),
				col("Token Price")
					.shift(lit(-2))
					.over([col("Token Address")])
					.alias("price_2_ahead"),
				col("Block ID")
					.shift(lit(-2))
					.over([col("Token Address")])
					.alias("block_2_ahead"),
				col("Token Price")
					.shift(lit(-3))
					.over([col("Token Address")])
					.alias("price_3_ahead"),
				col("Block ID")
					.shift(lit(-3))
					.over([col("Token Address")])
					.alias("block_3_ahead"),
			])
			.collect()?;

		// Create MEV sandwich detection logic
		let filtered_df = mev_analysis_df
			.lazy()
			.with_columns([
				// Detect if current trade is start of MEV sandwich:
				// 1. Current price is 50%+ higher than previous price
				// 2. Previous trade is in the SAME BLOCK (critical for MEV)
				(col("Token Price").gt(col("price_1_back") * lit(1.5)))
					.and(col("price_1_back").is_not_null())
					.and(col("Block ID").eq(col("block_1_back"))) // Same block constraint
					.alias("is_mev_spike"),
				// Detect if price reverts within next 1-3 trades back to ±10% of original
				// AND all reverting trades are in the SAME BLOCK
				(
					// Check if any of the next 1-3 prices revert AND are in same block
					(col("price_1_ahead")
						.lt(col("price_1_back") * lit(1.1))
						.and(
							col("price_1_ahead")
								.gt(col("price_1_back") * lit(0.9)),
						)
						.and(col("Block ID").eq(col("block_1_ahead"))))
					.or(col("price_2_ahead")
						.lt(col("price_1_back") * lit(1.1))
						.and(
							col("price_2_ahead")
								.gt(col("price_1_back") * lit(0.9)),
						)
						.and(col("Block ID").eq(col("block_2_ahead"))))
					.or(col("price_3_ahead")
						.lt(col("price_1_back") * lit(1.1))
						.and(
							col("price_3_ahead")
								.gt(col("price_1_back") * lit(0.9)),
						)
						.and(col("Block ID").eq(col("block_3_ahead"))))
				)
				.and(col("price_1_back").is_not_null())
				.alias("price_reverts_soon_same_block"),
			])
			.with_columns([
				// Mark trades that are part of MEV sandwich pattern
				col("is_mev_spike")
					.and(col("price_reverts_soon_same_block"))
					.alias("is_mev_sandwich_start"),
			])
			.collect()?;

		// Now we need to mark all trades in the sandwich for removal
		// Only remove trades that are part of the same-block MEV sandwich pattern
		let final_df = filtered_df
			.lazy()
			.with_columns([
				// Mark trades for removal if they are part of same-block MEV sandwich:
				// 1. The MEV front-run trade itself (the spike)
				// 2. The victim trade(s) (sandwiched)
				// 3. The MEV back-run trade (revert)
				col("is_mev_sandwich_start")
					.or(
						// Remove victim trade: next trade if previous was MEV start AND same block
						col("is_mev_sandwich_start")
							.shift(lit(1))
							.over([col("Token Address")])
							.fill_null(false)
							.and(col("Block ID").eq(col("block_1_back"))),
					)
					.or(
						// Remove back-run: trade 2 positions after MEV start AND same block as MEV start
						col("is_mev_sandwich_start")
							.shift(lit(2))
							.over([col("Token Address")])
							.fill_null(false)
							.and(
								col("Block ID").eq(col("block_1_back")
									.shift(lit(-1))
									.over([col("Token Address")])),
							),
					)
					.or(
						// Remove any additional trades in sandwich: 3 positions after MEV start AND same block
						col("is_mev_sandwich_start")
							.shift(lit(3))
							.over([col("Token Address")])
							.fill_null(false)
							.and(
								col("Block ID").eq(col("block_1_back")
									.shift(lit(-2))
									.over([col("Token Address")])),
							),
					)
					.alias("remove_mev_trade"),
			])
			.filter(
				col("remove_mev_trade")
					.eq(lit(false))
					.or(col("remove_mev_trade").is_null()),
			)
			.drop([
				"price_1_back",
				"block_1_back",
				"price_1_ahead",
				"block_1_ahead",
				"price_2_ahead",
				"block_2_ahead",
				"price_3_ahead",
				"block_3_ahead",
				"is_mev_spike",
				"price_reverts_soon_same_block",
				"is_mev_sandwich_start",
				"remove_mev_trade",
			])
			.collect()?;

		Ok(final_df)
	}

	/// Calculate x from price using bonding curve formula
	pub fn calculate_x_from_price(&self, price: f64) -> Option<f64> {
		if price <= 0.0 {
			None
		} else {
			Some((price * self.k_constant).sqrt())
		}
	}

	/// Simulate buying tokens using bonding curve
	pub fn simulate_buy(
		&self,
		sol_in_curve_before: f64,
		outstanding_shares_before: f64,
		sol_to_invest: f64,
	) -> (f64, f64) {
		let tokens_remaining_before =
			self.initial_tokens - outstanding_shares_before;
		let x_after =
			sol_in_curve_before + sol_to_invest + self.virtual_sol_base;
		let tokens_remaining_after = self.k_constant / x_after;

		let tokens_received = tokens_remaining_before - tokens_remaining_after;
		let sol_in_curve_after = sol_in_curve_before + sol_to_invest;

		(tokens_received, sol_in_curve_after)
	}

	/// Simulate selling tokens using bonding curve
	pub fn simulate_sell(
		&self,
		sol_in_curve_before: f64,
		outstanding_shares_before: f64,
		tokens_to_sell: f64,
	) -> (f64, f64) {
		let x_before = sol_in_curve_before + self.virtual_sol_base;
		let sol_in_curve_after_with_virtual = self.k_constant
			/ (self.initial_tokens - outstanding_shares_before
				+ tokens_to_sell);
		let sol_received = x_before - sol_in_curve_after_with_virtual;
		let sol_in_curve_after =
			sol_in_curve_after_with_virtual - self.virtual_sol_base;

		(sol_received, sol_in_curve_after)
	}

	/// Get latency-adjusted curve data using Block ID-based latency simulation
	/// Finds the last transaction for the same token within configurable additional blocks
	pub fn get_latency_adjusted_curve_data(
		&self,
		trigger_index: usize,
		token_address: &str,
	) -> Result<LatencyData> {
		let df = self.df.as_ref().unwrap();

		if trigger_index >= df.height() {
			return Err(anyhow::anyhow!("Trigger index out of bounds"));
		}

		// Get trigger row data
		let trigger_row = df.slice(trigger_index as i64, 1);

		// Extract trigger values - fail if critical data is missing
		let trigger_sol_in_curve = trigger_row
			.column("SOL in Curve")?
			.f64()?
			.get(0)
			.ok_or_else(|| anyhow::anyhow!("Missing SOL in Curve data at trigger index {}", trigger_index))?;
		let trigger_outstanding_shares = trigger_row
			.column("Outstanding Shares")?
			.f64()?
			.get(0)
			.ok_or_else(|| anyhow::anyhow!("Missing Outstanding Shares data at trigger index {}", trigger_index))?;
		let trigger_token_price = trigger_row
			.column("Token Price")?
			.f64()?
			.get(0)
			.ok_or_else(|| anyhow::anyhow!("Missing Token Price data at trigger index {}", trigger_index))?;
		let trigger_block_id =
			trigger_row.column("Block ID")?.i64()?.get(0)
				.ok_or_else(|| anyhow::anyhow!("Missing Block ID data at trigger index {}", trigger_index))?;
		let trigger_time_ms = trigger_row
			.column("Date_parsed")?
			.datetime()?
			.get(0)
			.ok_or_else(|| anyhow::anyhow!("Missing Date_parsed data at trigger index {}", trigger_index))?;

		// Calculate target block range: trigger_block to trigger_block + block_latency_max
		let max_block_id = trigger_block_id + self.block_latency_max;

		// Get column data for efficient iteration
		let token_addresses = df.column("Token Address")?.str()?;
		let block_ids = df.column("Block ID")?.i64()?;
		let sol_curves = df.column("SOL in Curve")?.f64()?;
		let outstanding_shares_col = df.column("Outstanding Shares")?.f64()?;
		let token_prices = df.column("Token Price")?.f64()?;
		let original_indices = df.column("original_index")?.u32()?;
		let date_parsed = df.column("Date_parsed")?.datetime()?;

		// Search for the last transaction of the same token within the block range (trigger + configurable additional)
		let mut best_candidate: Option<usize> = None;

		// Search from trigger_index onwards (include trigger block)
		for i in trigger_index..df.height() {
			let row_token = token_addresses.get(i)
				.ok_or_else(|| anyhow::anyhow!("Missing token address at index {}", i))?;
			let row_block_id = block_ids.get(i)
				.ok_or_else(|| anyhow::anyhow!("Missing block ID at index {}", i))?;

			// Stop searching if we've passed the max block range
			if row_block_id > max_block_id {
				break;
			}

			// Skip if not the same token
			if row_token != token_address {
				continue;
			}

			// Check if this transaction is within our target range (trigger_block to trigger_block + 2)
			if row_block_id >= trigger_block_id && row_block_id <= max_block_id
			{
				// Keep track of the last (most recent) transaction in the range
				best_candidate = Some(i);
			}
		}

		// If we found a transaction within the 3-block range, use it
		if let Some(idx) = best_candidate {
			let execution_block_id = block_ids.get(idx)
				.ok_or_else(|| anyhow::anyhow!("Missing block ID at execution index {}", idx))?;
			let block_latency = execution_block_id - trigger_block_id; // Blocks away from trigger
			let original_execution_index =
				original_indices.get(idx)
					.ok_or_else(|| anyhow::anyhow!("Missing original index at execution index {}", idx))? as usize;
			let execution_time_ms =
				date_parsed.get(idx).unwrap_or(trigger_time_ms);

			return Ok(LatencyData {
				sol_in_curve: sol_curves.get(idx)
					.ok_or_else(|| anyhow::anyhow!("Missing SOL in curve at execution index {}", idx))?,
				outstanding_shares: outstanding_shares_col
					.get(idx)
					.ok_or_else(|| anyhow::anyhow!("Missing outstanding shares at execution index {}", idx))?,
				token_price: token_prices.get(idx)
					.ok_or_else(|| anyhow::anyhow!("Missing token price at execution index {}", idx))?,
				execution_index: original_execution_index,
				execution_time: chrono::DateTime::from_timestamp_millis(
					execution_time_ms,
				)
				.unwrap_or_default()
				.with_timezone(&Utc),
				latency_actual: block_latency as f64, // Number of blocks away
				execution_logic: format!(
					"block_latency_{}_blocks",
					block_latency
				),
			});
		}

		// Fallback: use trigger trade data with 0 block latency
		let original_trigger_index =
			original_indices.get(trigger_index)
				.ok_or_else(|| anyhow::anyhow!("Missing original index at trigger index {}", trigger_index))? as usize;

		Ok(LatencyData {
			sol_in_curve: trigger_sol_in_curve,
			outstanding_shares: trigger_outstanding_shares,
			token_price: trigger_token_price,
			execution_index: original_trigger_index,
			execution_time: chrono::DateTime::from_timestamp_millis(
				trigger_time_ms,
			)
			.unwrap_or_default()
			.with_timezone(&Utc),
			latency_actual: 0.0,
			execution_logic: "trigger_same_block".to_string(),
		})
	}

	/// Execute a buy order
	pub fn execute_buy(
		&mut self,
		token_address: &str,
		original_tx_index: usize,
		filtered_tx_index: usize,
		strategy: &mut dyn TradingStrategy,
	) -> Result<bool> {
		// DEBUG: Removed - confirmed strategy execution is correct
		// Get latency-adjusted curve data (use filtered index for data access)
		let latency_data = self.get_latency_adjusted_curve_data(
			filtered_tx_index,
			token_address,
		)?;

		let sol_to_invest_gross = self.position_size_sol;

		let buy_fee_percent = if self.use_dynamic_fees {
			get_fee_percent(latency_data.token_price)
		} else {
			self.buy_fee_percent
		};

		let sol_to_invest_net = sol_to_invest_gross
			* (1.0 + buy_fee_percent)
			+ self.flat_fee_sol_buy;

		// Check slippage tolerance for buy orders
		let slippage_tolerance = strategy.get_slippage_tolerance();
		if slippage_tolerance > 0.0 {
			// Get trigger price from the filtered transaction row (same as current trigger)
			let df = self.df.as_ref().unwrap();
			let trigger_row = df.slice(filtered_tx_index as i64, 1);
			let trigger_price = trigger_row
				.column("Token Price")?
				.f64()?
				.get(0)
				.ok_or_else(|| anyhow::anyhow!(
					"Missing trigger price for slippage check at index {}", 
					filtered_tx_index
				))?;

			// Only check slippage if we have valid price data
			let price_change = (latency_data.token_price - trigger_price) / trigger_price;
			if price_change > slippage_tolerance {
				self.slippage_failures += 1;
				return Ok(false);
			}
		}

		// DEBUG: Print latency-adjusted curve data for buy
		if self.verbose {
			println!(
				"  🔧 BLOCK LATENCY BUY DEBUG: {} (blocks latency: {:.0})",
				latency_data.execution_logic, latency_data.latency_actual
			);
		}

		let (tokens_we_bought, sol_in_curve_after) = self.simulate_buy(
			latency_data.sol_in_curve,
			latency_data.outstanding_shares,
			sol_to_invest_gross,
		);

		if tokens_we_bought > 0.0 {
			let our_buy_price_effective = if tokens_we_bought > 0.0 {
				sol_to_invest_net / tokens_we_bought
			} else {
				0.0
			};

			let price_after_our_buy = if self.k_constant > 0.0 {
				(sol_in_curve_after + self.virtual_sol_base).powi(2)
					/ self.k_constant
			} else {
				0.0
			};

			let portfolio_entry = PortfolioEntry {
				tokens_bought: tokens_we_bought,
				sol_invested: sol_to_invest_net,
				buy_tx_index: original_tx_index, // Use original index for tracking
				buy_execution_index: latency_data.execution_index,
				buy_latency_seconds: latency_data.latency_actual,
				buy_execution_logic: latency_data.execution_logic,
				buy_execution_time: latency_data.execution_time.to_rfc3339(),
				buy_price_ref: latency_data.token_price,
				our_effective_buy_price: our_buy_price_effective,
				price_after_our_buy,
				sol_in_curve_at_our_buy: sol_in_curve_after,
				sol_in_curve_before_our_buy: latency_data.sol_in_curve,
			};

			// Handle DCA or new position - always track individual buys
			if let Some(existing_buys) = self.portfolio.get_mut(token_address) {
				// DCA addition - add as separate buy transaction
				existing_buys.push(portfolio_entry);
			} else {
				// First buy for this token
				self.portfolio
					.insert(token_address.to_string(), vec![portfolio_entry]);
			}

			// Store individual buy transaction for CSV export
			self.individual_transactions.push(IndividualTransaction {
				token_address: token_address.to_string(),
				timestamp: latency_data.execution_time.to_rfc3339(),
				transaction_type: "BUY".to_string(),
				price: our_buy_price_effective,
				sol_amount: sol_to_invest_net,
			});

			// Notify strategy with proper row data including Block ID
			let df = self.df.as_ref().unwrap();
			let mut buy_row_data = HashMap::new();

			// Get the actual row data for the latency-adjusted execution
			if let Ok(block_ids) = df.column("Block ID") {
				if let Ok(block_id_series) = block_ids.i64() {
					if let Some(block_id) =
						block_id_series.get(filtered_tx_index)
					{
						buy_row_data.insert(
							"Block ID".to_string(),
							AnyValue::Int64(block_id),
						);
					}
				}
			}
			if let Ok(dates) = df.column("Date") {
				if let Ok(date_series) = dates.str() {
					if let Some(date) = date_series.get(filtered_tx_index) {
						buy_row_data
							.insert("Date".to_string(), AnyValue::String(date));
					}
				}
			}

			strategy.on_buy_executed(
				token_address,
				latency_data.token_price,
				sol_to_invest_net,
				tokens_we_bought,
				Some(original_tx_index), // Use original index for strategy notification
				Some(&buy_row_data),
			)?;

			Ok(true)
		} else {
			if self.verbose {
				println!(
					"  BUY SKIPPED for {}: Simulation resulted in 0 tokens",
					token_address
				);
			}
			Ok(false)
		}
	}

	/// Execute a sell order
	pub fn execute_sell(
		&mut self,
		token_address: &str,
		original_tx_index: usize,
		filtered_tx_index: usize,
		date: &str,
		reason: &SellReason,
		strategy: &mut dyn TradingStrategy,
	) -> Result<bool> {
		// DEBUG: Removed - orphaned sells confirmed NOT happening during execution
		let buy_transactions = match self.portfolio.remove(token_address) {
			Some(buys) => buys,
			None => return Ok(false),
		};

		if buy_transactions.is_empty() {
			return Ok(false);
		}

		// Get latency-adjusted curve data (use filtered index for data access)
		let latency_data = self.get_latency_adjusted_curve_data(
			filtered_tx_index,
			token_address,
		)?;

		// Calculate total position size from all individual buys
		let tokens_to_sell: f64 =
			buy_transactions.iter().map(|buy| buy.tokens_bought).sum();

		// DEBUG: Print latency-adjusted curve data for sell
		if self.verbose {
			println!(
				"  🔧 BLOCK LATENCY SELL DEBUG: {} (blocks latency: {:.0})",
				latency_data.execution_logic, latency_data.latency_actual
			);
		}

		let sell_fee_percent = if self.use_dynamic_fees {
			get_fee_percent(latency_data.token_price)
		} else {
			self.sell_fee_percent
		};

		let (sol_we_received_gross, sol_in_curve_after) = self.simulate_sell(
			latency_data.sol_in_curve,
			latency_data.outstanding_shares,
			tokens_to_sell,
		);

		let sol_we_received_net = sol_we_received_gross * (1.0 - sell_fee_percent)
			- self.flat_fee_sol_sell;

		if sol_we_received_net > 1e-9 || tokens_to_sell == 0.0 {
			let our_sell_price_effective = if tokens_to_sell > 1e-9 {
				sol_we_received_net / tokens_to_sell
			} else {
				0.0
			};

			let price_after_our_sell = if self.k_constant > 1e-9 {
				(sol_in_curve_after + self.virtual_sol_base).powi(2)
					/ self.k_constant
			} else {
				0.0
			};

			// Calculate total PnL for position tracking
			let total_sol_invested: f64 =
				buy_transactions.iter().map(|buy| buy.sol_invested).sum();
			let total_pnl = sol_we_received_net - total_sol_invested;

			// Update equity curve tracking, excluding large wins from cumulative P&L
			let pnl_for_results = if total_pnl > 0.3 { 0.0 } else { total_pnl }; // Exclude large wins from final results
			self.cumulative_pnl += pnl_for_results;

			// Store individual sell transaction for CSV export (ONE real sell)
			self.individual_transactions.push(IndividualTransaction {
				token_address: token_address.to_string(),
				timestamp: latency_data.execution_time.to_rfc3339(),
				transaction_type: "SELL".to_string(),
				price: our_sell_price_effective,
				sol_amount: sol_we_received_net,
			});

			// Create ONE trade summary for the real accumulated sell transaction
			// Use weighted averages for buy-side data since we're selling entire accumulated position
			let total_sol_invested: f64 =
				buy_transactions.iter().map(|buy| buy.sol_invested).sum();
			let total_tokens_bought: f64 =
				buy_transactions.iter().map(|buy| buy.tokens_bought).sum();

			// Use first buy transaction for reference data (FIFO approach)
			let first_buy = &buy_transactions[0];

			// Calculate weighted average buy price
			let weighted_avg_buy_price = if total_tokens_bought > 1e-9 {
				total_sol_invested / total_tokens_bought
			} else {
				first_buy.our_effective_buy_price
			};

			// Add to equity curve (only for non-large-win trades)
			if total_pnl <= 1.0 {
				self.equity_curve.push(EquityEntry {
					trade_number: self.completed_trades.len() + 1,
					token_address: token_address.to_string(),
					trade_pnl: pnl_for_results, // Use same filtered value as framework cumulative
					cumulative_pnl: self.cumulative_pnl,
					peak_balance: 0.0, // Will be calculated from equity curve data
					current_drawdown: 0.0, // Will be calculated from equity curve data
					max_drawdown: 0.0, // Will be calculated from equity curve data
					tx_index: original_tx_index.to_string(), // Use original index for tracking
					execution_index: latency_data.execution_index,
					exit_date: date.to_string(),
					exit_index: latency_data.execution_index,
					trade_type: reason.as_string(),
					execution_latency_seconds: latency_data.latency_actual,
				});
			}

			// Track large wins (over 1 SOL)
			if total_pnl > 1.0 {
				self.large_wins.push(LargeWin {
					token_address: token_address.to_string(),
					pnl: total_pnl,
					exit_date: date.to_string(),
					reason: reason.as_string(),
					trigger_tx_index: first_buy.buy_tx_index, // Use first buy as reference
					exit_tx_index: original_tx_index, // Original sell transaction index (latency adjusted trade)
					trigger_execution_index: first_buy.buy_execution_index, // First buy execution index
					exit_execution_index: latency_data.execution_index, // Latency adjusted sell execution index
				});
			}

			// Create ONE trade summary for the real accumulated position (only for non-large-win trades)
			if total_pnl <= 1.0 {
				let trade_summary = TradeSummary {
					token_address: token_address.to_string(),
					buy_tx_index: first_buy.buy_tx_index + 1, // Use first buy as reference
					buy_execution_index: first_buy.buy_execution_index + 1, // Use first buy as reference
					buy_latency_seconds: first_buy.buy_latency_seconds,
					buy_execution_time: first_buy.buy_execution_time.clone(),
					sell_tx_index: (original_tx_index + 1).to_string(), // Convert to CSV line number (1-based)
					sell_execution_index: Some(
						latency_data.execution_index + 1,
					), // Convert to CSV line number (1-based)
					sell_latency_seconds: Some(latency_data.latency_actual),
					sell_execution_logic: Some(
						latency_data.execution_logic.clone(),
					),
					sell_execution_time: Some(
						latency_data.execution_time.to_rfc3339(),
					),
					buy_price_ref: first_buy.buy_price_ref,
					our_sol_invested: total_sol_invested, // REAL total investment
					our_tokens_bought: total_tokens_bought, // REAL total tokens
					our_effective_buy_price: weighted_avg_buy_price, // Weighted average buy price
					price_after_our_buy: first_buy.price_after_our_buy, // Use first buy as reference
					sol_in_curve_at_our_buy: first_buy.sol_in_curve_at_our_buy, // Use first buy as reference
					sol_in_curve_before_our_buy: first_buy
						.sol_in_curve_before_our_buy, // Use first buy as reference
					current_sol_in_curve: latency_data.sol_in_curve,
					sell_price_ref: latency_data.token_price,
					our_sol_received: sol_we_received_net, // REAL total proceeds
					our_effective_sell_price: our_sell_price_effective, // REAL sell price
					price_after_our_sell,
					pnl_sol: total_pnl, // REAL total PnL
					exit_reason: reason.as_string(),
				};

				self.completed_trades.push(trade_summary);
			}

			// Notify strategy with proper row data including Block ID
			let df = self.df.as_ref().unwrap();
			let mut sell_row_data = HashMap::new();

			// Get the actual row data for the latency-adjusted execution
			if let Ok(block_ids) = df.column("Block ID") {
				if let Ok(block_id_series) = block_ids.i64() {
					if let Some(block_id) =
						block_id_series.get(filtered_tx_index)
					{
						sell_row_data.insert(
							"Block ID".to_string(),
							AnyValue::Int64(block_id),
						);
					}
				}
			}
			if let Ok(dates) = df.column("Date") {
				if let Ok(date_series) = dates.str() {
					if let Some(date) = date_series.get(filtered_tx_index) {
						sell_row_data
							.insert("Date".to_string(), AnyValue::String(date));
					}
				}
			}

			strategy.on_sell_executed(
				token_address,
				Some(original_tx_index),
				Some(&sell_row_data),
			)?; // Use original index

			Ok(true)
		} else {
			if self.verbose {
				println!(
					"  {} SKIPPED for {}: Simulation resulted in 0 SOL received.",
					reason.as_string(),
					token_address
				);
			}
			Ok(false)
		}
	}

	/// Preprocess data once (load, validate, detect manipulation) for efficient parameter testing
	pub fn preprocess_data<P: AsRef<Path>>(
		&mut self,
		csv_file_path: P,
		validate_data: bool,
		tolerance_percent: f64,
		detect_manipulation: bool,
		manipulation_threshold: f64,
	) -> Result<()> {
		// Store the dataset path for export metadata
		self.dataset_path = Some(Arc::new(
			csv_file_path.as_ref().to_string_lossy().to_string(),
		));

		if !self.load_data(csv_file_path)? {
			return Err(anyhow::anyhow!("Failed to load data"));
		}

		// Detect and filter manipulation BEFORE validation
		if detect_manipulation {
			println!("🚨 Running manipulation detection...");
			let mut detector =
				ManipulationDetector::new(manipulation_threshold, 85.0, 3);
			let blacklisted_tokens =
				detector.detect_manipulation(self.df.as_ref().unwrap())?;

			if !blacklisted_tokens.is_empty() {
				// Actually filter out the blacklisted tokens
				let original_count = self.df.as_ref().unwrap().height();

				// Create a filter using lazy evaluation
				let mut lazy_df = (**self.df.as_ref().unwrap()).clone().lazy();

				// Filter out each blacklisted token
				for blacklisted_token in &blacklisted_tokens {
					lazy_df = lazy_df.filter(
						col("Token Address")
							.neq(lit(blacklisted_token.clone())),
					);
				}

				let filtered_df = lazy_df.collect()?;
				let filtered_count = filtered_df.height();
				let removed_count = original_count - filtered_count;

				self.df = Some(Arc::new(filtered_df));

				println!(
					"🚫 Filtered {} transactions from {} manipulated tokens",
					removed_count,
					detector.get_blacklisted_tokens().len()
				);
				println!(
					"   Remaining transactions: {} (was {})",
					filtered_count, original_count
				);
			} else {
				println!("✅ No manipulation detected");
			}
		}

		// Validate and clean data if requested
		if validate_data {
			self.validate_and_clean_data(tolerance_percent)?;
		}

		// Precompute token last trade data for efficient liquidation
		println!("🕐 Computing token last trade data...");
		if let Some(ref df) = self.df {
			// Sort by Date and get the last row for each token with all fields
			let last_trades_df = (**df)
				.clone()
				.lazy()
				.sort(vec!["Date"], SortMultipleOptions::default())
				.group_by([col("Token Address")])
				.agg([
					col("Date").last().alias("last_timestamp"),
					col("SOL in Curve").last().alias("last_sol_in_curve"),
					col("Outstanding Shares").last().alias("last_outstanding_shares"),
					col("Token Price").last().alias("last_token_price"),
				])
				.collect()?;

			let token_addresses = last_trades_df.column("Token Address")?.str()?;
			let last_timestamps = last_trades_df.column("last_timestamp")?.str()?;
			let last_sol_in_curves = last_trades_df.column("last_sol_in_curve")?.f64()?;
			let last_outstanding_shares = last_trades_df.column("last_outstanding_shares")?.f64()?;
			let last_token_prices = last_trades_df.column("last_token_price")?.f64()?;

			let mut last_trade_map = HashMap::new();
			for i in 0..token_addresses.len() {
				if let (Some(token), Some(timestamp), Some(sol_in_curve), Some(outstanding_shares), Some(token_price)) = (
					token_addresses.get(i),
					last_timestamps.get(i),
					last_sol_in_curves.get(i),
					last_outstanding_shares.get(i),
					last_token_prices.get(i),
				) {
					last_trade_map.insert(
						token.to_string(),
						LastTradeData {
							timestamp: timestamp.to_string(),
							sol_in_curve,
							outstanding_shares,
							token_price,
						},
					);
				}
			}

			println!(
				"   Computed last trade data for {} tokens",
				last_trade_map.len()
			);

			// Wrap in Arc after building
			self.token_last_trade_data = Arc::new(last_trade_map);
		}

		println!(
			"✅ Data preprocessing complete - {} valid transactions ready for strategy testing",
			self.df.as_ref().unwrap().height()
		);

		Ok(())
	}

	/// Clone framework for new parameter test (preserving preprocessed data, resetting trade tracking)
	pub fn clone_for_new_test(&self) -> Self {
		Self {
			// Copy framework constants and fees
			virtual_sol_base: self.virtual_sol_base,
			k_constant: self.k_constant,
			initial_tokens: self.initial_tokens,
			buy_fee_percent: self.buy_fee_percent,
			sell_fee_percent: self.sell_fee_percent,
			flat_fee_sol_buy: self.flat_fee_sol_buy,
			flat_fee_sol_sell: self.flat_fee_sol_sell,
			position_size_sol: self.position_size_sol,
			block_latency_max: self.block_latency_max,
			use_dynamic_fees: self.use_dynamic_fees,

			// Preserve preprocessed data (Arc prevents copying)
			df: self.df.as_ref().map(Arc::clone),

			// Reset all trading/tracking state
			portfolio: HashMap::new(), // Now HashMap<String, Vec<PortfolioEntry>>
			completed_trades: Vec::new(),
			individual_transactions: Vec::new(),
			equity_curve: Vec::new(),
			cumulative_pnl: 0.0,
			peak_balance: 0.0,
			max_drawdown: 0.0,
			debug_trades: Vec::new(),
			debug_trade_count: 0,
			large_wins: Vec::new(),

			// Preserve export setting
			export_trades: self.export_trades,

			// Preserve dataset path (Arc clone is cheap)
			dataset_path: self.dataset_path.clone(),

			// Preserve precomputed last trade data (Arc clone is cheap)
			token_last_trade_data: Arc::clone(&self.token_last_trade_data),

			// Preserve verbose setting
			verbose: self.verbose,

			// Reset slippage tracking
			slippage_failures: 0,
		}
	}

	/// Check if framework is in verbose mode
	pub fn is_verbose(&self) -> bool {
		self.verbose
	}

	/// Get the loaded dataframe
	pub fn get_dataframe(&self) -> Option<&DataFrame> {
		self.df.as_ref().map(|arc| arc.as_ref())
	}

	/// Run strategy on already preprocessed data (for parameter optimization)
	pub fn run_strategy_on_preprocessed_data(
		&mut self,
		strategy: &mut dyn TradingStrategy,
	) -> Result<BacktestResults> {
		if self.df.is_none() {
			return Err(anyhow::anyhow!(
				"No preprocessed data available. Call preprocess_data first."
			));
		}

		if self.verbose {
			println!(
				"Running strategy {} on preprocessed data",
				strategy.get_strategy_name()
			);
		}

		// Clear any stale data from previous runs
		self.portfolio.clear();
		self.completed_trades.clear();
		self.individual_transactions.clear();
		self.equity_curve.clear();
		self.cumulative_pnl = 0.0;
		self.peak_balance = 0.0;
		self.max_drawdown = 0.0;
		self.slippage_failures = 0;

		let df_height = self.df.as_ref().unwrap().height();

		// Extract all data at once for vectorized processing (avoiding borrow conflicts)
		let (
			token_address_vec,
			price_vec,
			date_vec,
			sol_curve_vec,
			platform_vec,
			original_index_vec,
		) = {
			let df = self.df.as_ref().unwrap();
			let token_addresses = df.column("Token Address")?.str()?;
			let token_prices = df.column("Token Price")?.f64()?;
			let dates = df.column("Date")?.str()?;
			let sol_curves = df.column("SOL in Curve")?.f64()?;
			let platforms = df.column("Platform")?.str()?;
			let original_indices = df.column("original_index")?.u32()?;

			// Convert to owned vectors to avoid borrowing issues
			// If data is missing at valid indices, it indicates corrupted dataframe
			let mut token_addr_vec = Vec::with_capacity(df_height);
			let mut price_vec = Vec::with_capacity(df_height);
			let mut date_vec = Vec::with_capacity(df_height);
			let mut sol_curve_vec = Vec::with_capacity(df_height);
			let mut platform_vec = Vec::with_capacity(df_height);
			let mut original_index_vec = Vec::with_capacity(df_height);
			
			for i in 0..df_height {
				token_addr_vec.push(
					token_addresses.get(i)
						.ok_or_else(|| anyhow::anyhow!("Missing token address at row {}", i))?
						.to_string()
				);
				price_vec.push(
					token_prices.get(i)
						.ok_or_else(|| anyhow::anyhow!("Missing token price at row {}", i))?
				);
				date_vec.push(
					dates.get(i)
						.ok_or_else(|| anyhow::anyhow!("Missing date at row {}", i))?
						.to_string()
				);
				sol_curve_vec.push(
					sol_curves.get(i)
						.ok_or_else(|| anyhow::anyhow!("Missing SOL in curve at row {}", i))?
				);
				platform_vec.push(
					platforms.get(i)
						.ok_or_else(|| anyhow::anyhow!("Missing platform at row {}", i))?
						.to_string()
				);
				original_index_vec.push(
					original_indices.get(i)
						.ok_or_else(|| anyhow::anyhow!("Missing original index at row {}", i))?
				);
			}

			(
				token_addr_vec,
				price_vec,
				date_vec,
				sol_curve_vec,
				platform_vec,
				original_index_vec,
			)
		};

		// Process each transaction
		for i in 0..df_height {
			let token_address = &token_address_vec[i];
			let price_after_tx = price_vec[i];
			let date = &date_vec[i];
			let sol_in_curve = sol_curve_vec[i];
			let platform = &platform_vec[i];
			let original_index = original_index_vec[i] as usize;

			// Create row data for strategy - extract all needed columns at once to avoid borrow issues
			let mut row_data = HashMap::new();
			row_data.insert("Date".to_string(), AnyValue::String(date));
			row_data.insert(
				"SOL in Curve".to_string(),
				AnyValue::Float64(sol_in_curve),
			);
			row_data.insert("Platform".to_string(), AnyValue::String(platform));

			// Extract all DataFrame columns in one scope to release borrows before mutable operations
			{
				let df = self.df.as_ref().unwrap();

				// Add Block ID
				if let Ok(block_ids) = df.column("Block ID") {
					if let Ok(block_id_series) = block_ids.i64() {
						if let Some(block_id) = block_id_series.get(i) {
							row_data.insert(
								"Block ID".to_string(),
								AnyValue::Int64(block_id),
							);
						}
					}
				}

				// Add Token Price
				if let Ok(token_prices) = df.column("Token Price") {
					if let Ok(price_series) = token_prices.f64() {
						if let Some(token_price) = price_series.get(i) {
							row_data.insert(
								"Token Price".to_string(),
								AnyValue::Float64(token_price),
							);
						}
					}
				}

				// Add Token Amount
				if let Ok(token_amounts) = df.column("Token Amount") {
					if let Ok(amount_series) = token_amounts.f64() {
						if let Some(token_amount) = amount_series.get(i) {
							row_data.insert(
								"Token Amount".to_string(),
								AnyValue::Float64(token_amount),
							);
						}
					}
				}

				// Add Transaction Type
				if let Ok(transaction_types) = df.column("Transaction Type") {
					if let Ok(type_series) = transaction_types.str() {
						if let Some(transaction_type) = type_series.get(i) {
							row_data.insert(
								"Transaction Type".to_string(),
								AnyValue::StringOwned(
									transaction_type.to_string().into(),
								),
							);
						}
					}
				}

				// Add Wallet Address (for copy trading strategies)
				if let Ok(wallet_addresses) = df.column("Wallet Address") {
					if let Ok(address_series) = wallet_addresses.str() {
						if let Some(wallet_address) = address_series.get(i) {
							row_data.insert(
								"Wallet Address".to_string(),
								AnyValue::StringOwned(
									wallet_address.to_string().into(),
								),
							);
						}
					}
				}
			} // DataFrame borrows are released here

			// Check sell signals for existing positions
			let positions_to_check: Vec<String> = self
				.portfolio
				.keys()
				.filter(|addr| **addr == *token_address)
				.cloned()
				.collect();

			for held_token_address in positions_to_check {
				let (should_sell, sell_reason) = strategy.should_sell(
					&held_token_address,
					price_after_tx,
					original_index,
					&AnyValue::Null,
					&row_data,
				)?;

				if should_sell {
					self.execute_sell(
						&held_token_address,
						original_index,
						i,
						date,
						&sell_reason,
						strategy,
					)?;
				}
			}

			// Check buy signals
			if strategy.should_buy(
				token_address,
				price_after_tx,
				original_index,
				&AnyValue::Null,
				&row_data,
			)? {
				self.execute_buy(token_address, original_index, i, strategy)?;
			}

			// Update strategy with latest data after buy/sell decisions
			strategy.update_data(
				token_address,
				price_after_tx,
				i,
				date,
				&row_data,
			)?;
		}

		// Liquidate remaining positions
		self.liquidate_remaining_positions()?;

		Ok(self.compile_results()?)
	}

	/// Liquidate all remaining positions at end of simulation
	fn liquidate_remaining_positions(&mut self) -> Result<()> {
		if self.portfolio.is_empty() {
			return Ok(());
		}

		if self.verbose {
			println!(
				"\nSimulating end-of-period liquidation for remaining tokens..."
			);
		}
		let tokens_to_liquidate: Vec<String> =
			self.portfolio.keys().cloned().collect();

		for token_address in tokens_to_liquidate {
			if let Some(buy_transactions) =
				self.portfolio.remove(&token_address)
			{
				// Get last trade data for this token
				let last_trade_data = match self.token_last_trade_data.get(&token_address) {
					Some(data) => data,
					None => {
						// If no last trade data, skip this token (shouldn't happen with valid data)
						if self.verbose {
							println!("  WARNING: No last trade data for token {}, skipping liquidation", token_address);
						}
						continue;
					}
				};

				// Aggregate all buy positions for this token
				let total_tokens: f64 = buy_transactions.iter()
					.map(|buy| buy.tokens_bought)
					.sum();
				let total_sol_invested: f64 = buy_transactions.iter()
					.map(|buy| buy.sol_invested)
					.sum();

				// Find the earliest buy transaction for reporting
				let earliest_buy = buy_transactions.iter()
					.min_by_key(|buy| buy.buy_tx_index)
					.unwrap();

				// Simulate selling total position against last known curve state
				let (sol_received_gross, sol_in_curve_after) = self.simulate_sell(
					last_trade_data.sol_in_curve,
					last_trade_data.outstanding_shares,
					total_tokens,
				);

				// Apply fees to get net SOL received
				let sell_fee_percent = if self.use_dynamic_fees {
					get_fee_percent(last_trade_data.token_price)
				} else {
					self.sell_fee_percent
				};
				let sol_received_net = sol_received_gross * (1.0 - sell_fee_percent) 
					- self.flat_fee_sol_sell;

				// Calculate aggregate PnL
				let pnl = sol_received_net - total_sol_invested;

				// Update equity tracking
				self.cumulative_pnl += pnl;

				// Calculate effective prices
				let effective_sell_price = if total_tokens > 1e-9 {
					sol_received_net / total_tokens
				} else {
					0.0
				};

				let price_after_our_sell = if self.k_constant > 0.0 {
					(sol_in_curve_after + self.virtual_sol_base).powi(2) / self.k_constant
				} else {
					0.0
				};

				// Store individual liquidation transaction for CSV export
				self.individual_transactions.push(IndividualTransaction {
					token_address: token_address.to_string(),
					timestamp: last_trade_data.timestamp.clone(),
					transaction_type: "SELL".to_string(),
					price: effective_sell_price,
					sol_amount: sol_received_net,
				});

				// Create aggregated trade summary for all positions
				let trade_summary = TradeSummary {
					token_address: token_address.clone(),
					buy_tx_index: earliest_buy.buy_tx_index + 1, // Convert to CSV line number (1-based)
					buy_execution_index: earliest_buy.buy_execution_index + 1, // Convert to CSV line number (1-based)
					buy_latency_seconds: earliest_buy.buy_latency_seconds,
					buy_execution_time: earliest_buy.buy_execution_time.clone(),
					sell_tx_index: "LIQUIDATION".to_string(),
					sell_execution_index: None,
					sell_latency_seconds: None,
					sell_execution_logic: None,
					sell_execution_time: Some(last_trade_data.timestamp.clone()),
					buy_price_ref: earliest_buy.buy_price_ref,
					our_sol_invested: total_sol_invested,
					our_tokens_bought: total_tokens,
					our_effective_buy_price: if total_tokens > 1e-9 {
						total_sol_invested / total_tokens
					} else {
						0.0
					},
					price_after_our_buy: earliest_buy.price_after_our_buy,
					sol_in_curve_at_our_buy: earliest_buy.sol_in_curve_at_our_buy,
					sol_in_curve_before_our_buy: earliest_buy.sol_in_curve_before_our_buy,
					current_sol_in_curve: last_trade_data.sol_in_curve,
					sell_price_ref: last_trade_data.token_price,
					our_sol_received: sol_received_net,
					our_effective_sell_price: effective_sell_price,
					price_after_our_sell,
					pnl_sol: pnl,
					exit_reason: SellReason::Liquidation.as_string(),
				};

				self.completed_trades.push(trade_summary);

				// Add liquidation to equity curve
				self.equity_curve.push(EquityEntry {
					trade_number: self.completed_trades.len(),
					token_address: token_address.to_string(),
					trade_pnl: pnl,
					cumulative_pnl: self.cumulative_pnl,
					peak_balance: 0.0, // Will be calculated from equity curve data
					current_drawdown: 0.0, // Will be calculated from equity curve data
					max_drawdown: 0.0, // Will be calculated from equity curve data
					tx_index: "LIQUIDATION".to_string(),
					execution_index: 0,
					exit_date: last_trade_data.timestamp.clone(),
					exit_index: 0,
					trade_type: "LIQUIDATION".to_string(),
					execution_latency_seconds: 0.0,
				});

				if self.verbose {
					println!(
						"  LIQUIDATED: {} - Sold {:.2} tokens for {:.4} SOL (PnL: {:.4} SOL)",
						&token_address[token_address.len().saturating_sub(8)..],
						total_tokens,
						sol_received_net,
						pnl
					);
				}
			}
		}

		Ok(())
	}

	/// Compile and return backtest results (using equity curve as single source of truth)
	fn compile_results(&self) -> Result<BacktestResults> {
		// Convert completed trades to DataFrame
		let token_addresses: Vec<String> = self
			.completed_trades
			.iter()
			.map(|t| t.token_address.clone())
			.collect();
		let buy_indices: Vec<i64> = self
			.completed_trades
			.iter()
			.map(|t| t.buy_tx_index as i64)
			.collect();
		let sell_indices: Vec<String> = self
			.completed_trades
			.iter()
			.map(|t| t.sell_tx_index.clone())
			.collect();
		let pnl_values: Vec<f64> =
			self.completed_trades.iter().map(|t| t.pnl_sol).collect();
		let exit_reasons: Vec<String> = self
			.completed_trades
			.iter()
			.map(|t| t.exit_reason.clone())
			.collect();

		let completed_trades_df = DataFrame::new(vec![
			Series::new("token_address".into(), token_addresses).into(),
			Series::new("buy_tx_index".into(), buy_indices).into(),
			Series::new("sell_tx_index".into(), sell_indices).into(),
			Series::new("pnl_sol".into(), pnl_values).into(),
			Series::new("exit_reason".into(), exit_reasons).into(),
		])?;

		// Sort equity curve chronologically after all trades and liquidations are complete
		let mut sorted_equity_curve = self.equity_curve.clone();
		sorted_equity_curve.sort_by(|a, b| a.exit_date.cmp(&b.exit_date));

		// Calculate all summary statistics from sorted equity curve (single source of truth)
		let (cumulative_pnl, peak_balance, max_drawdown) =
			if sorted_equity_curve.is_empty() {
				(0.0, 0.0, 0.0)
			} else {
				// Recalculate cumulative P&L in chronological order
				let mut chronological_cumulative_pnl = 0.0;
				let mut chronological_peak = 0.0;
				let mut max_drawdown = 0.0;

				for entry in &sorted_equity_curve {
					chronological_cumulative_pnl += entry.trade_pnl;

					// Update peak balance
					if chronological_cumulative_pnl > chronological_peak {
						chronological_peak = chronological_cumulative_pnl;
					}

					// Calculate current drawdown and update max
					let current_drawdown =
						chronological_peak - chronological_cumulative_pnl;
					if current_drawdown > max_drawdown {
						max_drawdown = current_drawdown;
					}
				}

				(
					chronological_cumulative_pnl,
					chronological_peak,
					max_drawdown,
				)
			};

		Ok(BacktestResults {
			completed_trades_df,
			equity_curve: sorted_equity_curve, // Use the single sorted equity curve
			cumulative_pnl,
			peak_balance,
			max_drawdown,
			portfolio: self.portfolio.clone(),
			large_wins: self.large_wins.clone(),
		})
	}

	/// Print debug information for first 100 trades
	#[allow(dead_code)]
	fn print_debug_trades(&self) {
		if self.debug_trades.is_empty() {
			println!("\n=== DEBUG: No trades to show ===");
			return;
		}

		println!(
			"\n=== DEBUG: First {} Trades Details ===",
			self.debug_trades.len()
		);
		println!(
			"{:<4} {:<6} {:<10} {:<8} {:<24} {:<10} {:<10} {:<15} {:<15} {:<10} {:<10} {:<15} {:<8} {:<10} {:<12}",
			"T#",
			"Action",
			"Token",
			"TxIdx",
			"Timestamp",
			"SOL_Before",
			"SOL_After",
			"Tokens",
			"SOL_Gross",
			"SOL_Fees",
			"SOL_Net",
			"Price/Token",
			"Z-Score",
			"P&L",
			"Reason"
		);
		println!("{}", "=".repeat(200));

		for trade in &self.debug_trades {
			let short_token = if trade.token_address.len() >= 8 {
				&trade.token_address[trade.token_address.len() - 8..]
			} else {
				&trade.token_address
			};

			let z_score_str = match trade.z_score {
				Some(z) => format!("{:.3}", z),
				None => "N/A".to_string(),
			};

			let pnl_str = match trade.pnl {
				Some(p) => format!("{:.6}", p),
				None => "N/A".to_string(),
			};

			println!(
				"{:<4} {:<6} {:<10} {:<8} {:<24} {:<10.2} {:<10.2} {:<15.6} {:<15.6} {:<10.6} {:<10.6} {:<15.10} {:<8} {:<10} {:<12}",
				trade.trade_number,
				trade.action,
				short_token,
				trade.tx_index,
				trade.timestamp,
				trade.sol_in_curve_before,
				trade.sol_in_curve_after,
				trade.tokens_transacted,
				trade.sol_gross,
				trade.sol_fees,
				trade.sol_net,
				trade.effective_price_per_token,
				z_score_str,
				pnl_str,
				trade.reason.split(' ').next().unwrap_or(&trade.reason)
			);
		}
		println!("{}", "=".repeat(200));
		println!();
	}

	/// Print results summary
	/// Get completed trades for per-segment analysis
	pub fn get_completed_trades(&self) -> &Vec<TradeSummary> {
		&self.completed_trades
	}

	/// Calculate comprehensive trade statistics from equity curve (single source of truth)
	pub fn calculate_trade_statistics(
		results: &BacktestResults,
	) -> TradeStatistics {
		// Equity curve is already chronologically sorted
		let total_trades = results.equity_curve.len();

		// Handle empty results case
		if total_trades == 0 {
			return TradeStatistics {
				total_trades: 0,
				cumulative_pnl: 0.0,
				peak_balance: 0.0,
				max_drawdown: 0.0,
				win_count: 0,
				loss_count: 0,
				win_rate: 0.0,
				avg_win: 0.0,
				avg_loss: 0.0,
				profit_factor: 0.0,
				total_wins: 0.0,
				total_losses: 0.0,
				large_wins_count: results.large_wins.len(),
				large_wins_total: results
					.large_wins
					.iter()
					.map(|w| w.pnl)
					.sum(),
			};
		}

		// Recalculate all statistics chronologically from equity curve
		let mut chronological_cumulative_pnl = 0.0;
		let mut chronological_peak = 0.0;
		let mut max_drawdown = 0.0;

		for entry in &results.equity_curve {
			chronological_cumulative_pnl += entry.trade_pnl;

			// Update peak balance
			if chronological_cumulative_pnl > chronological_peak {
				chronological_peak = chronological_cumulative_pnl;
			}

			// Calculate current drawdown and update max
			let current_drawdown =
				chronological_peak - chronological_cumulative_pnl;
			if current_drawdown > max_drawdown {
				max_drawdown = current_drawdown;
			}
		}

		// Calculate win/loss statistics from equity curve
		let winning_trades: Vec<&EquityEntry> = results
			.equity_curve
			.iter()
			.filter(|e| e.trade_pnl > 0.0)
			.collect();
		let losing_trades: Vec<&EquityEntry> = results
			.equity_curve
			.iter()
			.filter(|e| e.trade_pnl < 0.0)
			.collect();

		let win_count = winning_trades.len();
		let loss_count = losing_trades.len();
		let win_rate = if total_trades > 0 {
			win_count as f64 / total_trades as f64 * 100.0
		} else {
			0.0
		};

		// Calculate average win/loss
		let avg_win = if !winning_trades.is_empty() {
			winning_trades.iter().map(|e| e.trade_pnl).sum::<f64>()
				/ winning_trades.len() as f64
		} else {
			0.0
		};

		let avg_loss = if !losing_trades.is_empty() {
			losing_trades.iter().map(|e| e.trade_pnl).sum::<f64>()
				/ losing_trades.len() as f64
		} else {
			0.0
		};

		// Calculate profit factor
		let total_wins = if !winning_trades.is_empty() {
			winning_trades.iter().map(|e| e.trade_pnl).sum::<f64>()
		} else {
			0.0
		};

		let total_losses = if !losing_trades.is_empty() {
			losing_trades.iter().map(|e| e.trade_pnl.abs()).sum::<f64>()
		} else {
			0.0
		};

		let profit_factor = if total_losses > 0.0 {
			total_wins / total_losses
		} else if total_wins > 0.0 {
			f64::INFINITY
		} else {
			0.0
		};

		// Calculate large wins statistics
		let large_wins_count = results.large_wins.len();
		let large_wins_total = results.large_wins.iter().map(|w| w.pnl).sum();

		TradeStatistics {
			total_trades,
			cumulative_pnl: chronological_cumulative_pnl, // Use chronologically calculated value
			peak_balance: chronological_peak, // Use chronologically calculated value
			max_drawdown,                     // Use chronologically calculated value
			win_count,
			loss_count,
			win_rate,
			avg_win,
			avg_loss,
			profit_factor,
			total_wins,
			total_losses,
			large_wins_count,
			large_wins_total,
		}
	}

	/// Get SOL in curve analysis data for parameter optimization
	pub fn get_sol_in_curve_analysis_data(
		&self,
		sol_group_level: f64,
	) -> std::collections::HashMap<i32, SolLevelAnalysis> {
		let mut sol_level_data = std::collections::HashMap::new();

		if self.completed_trades.is_empty() {
			return sol_level_data;
		}

		// Group trades by SOL level (floor-based for sequential ranges) - same logic as print function
		let mut sol_level_groups: std::collections::HashMap<
			i32,
			Vec<&TradeSummary>,
		> = std::collections::HashMap::new();

		for trade in &self.completed_trades {
			let sol_level =
				((trade.sol_in_curve_before_our_buy / sol_group_level).floor()
					as i32) * (sol_group_level as i32);
			sol_level_groups
				.entry(sol_level)
				.or_insert_with(Vec::new)
				.push(trade);
		}

		// Calculate metrics for each SOL level
		for (sol_level, trades) in sol_level_groups {
			let trade_count = trades.len();
			let total_pnl: f64 = trades.iter().map(|t| t.pnl_sol).sum();

			// Win/loss statistics
			let winning_trades: Vec<&&TradeSummary> =
				trades.iter().filter(|t| t.pnl_sol > 0.0).collect();
			let losing_trades: Vec<&&TradeSummary> =
				trades.iter().filter(|t| t.pnl_sol <= 0.0).collect();

			let win_count = winning_trades.len();
			let win_percentage = if trade_count > 0 {
				(win_count as f64 / trade_count as f64) * 100.0
			} else {
				0.0
			};

			let avg_win = if !winning_trades.is_empty() {
				winning_trades.iter().map(|t| t.pnl_sol).sum::<f64>()
					/ winning_trades.len() as f64
			} else {
				0.0
			};

			let avg_loss = if !losing_trades.is_empty() {
				losing_trades.iter().map(|t| t.pnl_sol).sum::<f64>()
					/ losing_trades.len() as f64
			} else {
				0.0
			};

			// Profit Factor
			let total_wins: f64 =
				winning_trades.iter().map(|t| t.pnl_sol).sum();
			let total_losses: f64 =
				losing_trades.iter().map(|t| t.pnl_sol.abs()).sum();
			let profit_factor = if total_losses > 0.0 {
				total_wins / total_losses
			} else {
				0.0
			};

			// Calculate range (min-max SOL levels in this group)
			let min_sol = trades
				.iter()
				.map(|t| t.sol_in_curve_before_our_buy as i32)
				.min()
				.unwrap_or(sol_level);
			let max_sol = trades
				.iter()
				.map(|t| t.sol_in_curve_before_our_buy as i32)
				.max()
				.unwrap_or(sol_level);

			let analysis = SolLevelAnalysis {
				sol_level,
				trade_count,
				total_pnl,
				win_percentage,
				avg_win,
				avg_loss,
				profit_factor,
				min_sol_range: min_sol,
				max_sol_range: max_sol,
			};

			sol_level_data.insert(sol_level, analysis);
		}

		sol_level_data
	}

	/// Print SOL in curve trigger analysis
	pub fn print_sol_in_curve_analysis(&self, sol_group_level: f64) {
		if self.completed_trades.is_empty() {
			println!(
				"No completed trades available for SOL in curve analysis."
			);
			return;
		}

		println!("\n{}", "=".repeat(100));
		println!("📊 SOL IN CURVE TRIGGER ANALYSIS");
		println!("{}", "=".repeat(100));
		println!(
			"Using 'sol_in_curve_before_our_buy' column for SOL in curve analysis"
		);
		println!("SOL grouping level: {} SOL increments", sol_group_level);

		// Group trades by SOL level (floor-based for sequential ranges)
		let mut sol_level_groups: std::collections::HashMap<
			i32,
			Vec<&TradeSummary>,
		> = std::collections::HashMap::new();

		for trade in &self.completed_trades {
			let sol_level =
				((trade.sol_in_curve_before_our_buy / sol_group_level).floor()
					as i32) * (sol_group_level as i32);
			sol_level_groups
				.entry(sol_level)
				.or_insert_with(Vec::new)
				.push(trade);
		}

		let total_trades = self.completed_trades.len();
		let total_sol_levels = sol_level_groups.len();

		println!(
			"Analyzing {} trades across {} different SOL in curve levels:",
			total_trades, total_sol_levels
		);
		println!();

		// Collect and sort SOL levels
		let mut sol_levels: Vec<i32> =
			sol_level_groups.keys().cloned().collect();
		sol_levels.sort();

		// Print header
		println!(
			"{:<8} {:<7} {:<10} {:<6} {:<8} {:<9} {:<5} {:<12} {:<8} {:<8}",
			"SOL Lvl",
			"Trades",
			"Total PnL",
			"Win%",
			"Avg Win",
			"Avg Loss",
			"PF",
			"Range",
			"Best",
			"Worst"
		);
		println!("{}", "-".repeat(90));

		let mut grand_total_pnl = 0.0;

		for sol_level in sol_levels {
			let trades = sol_level_groups.get(&sol_level).unwrap();
			let trade_count = trades.len();
			let total_pnl: f64 = trades.iter().map(|t| t.pnl_sol).sum();

			// Win/loss statistics
			let winning_trades: Vec<&&TradeSummary> =
				trades.iter().filter(|t| t.pnl_sol > 0.0).collect();
			let losing_trades: Vec<&&TradeSummary> =
				trades.iter().filter(|t| t.pnl_sol <= 0.0).collect();

			let win_count = winning_trades.len();
			let win_percentage = if trade_count > 0 {
				(win_count as f64 / trade_count as f64) * 100.0
			} else {
				0.0
			};

			let avg_win = if !winning_trades.is_empty() {
				winning_trades.iter().map(|t| t.pnl_sol).sum::<f64>()
					/ winning_trades.len() as f64
			} else {
				0.0
			};

			let avg_loss = if !losing_trades.is_empty() {
				losing_trades.iter().map(|t| t.pnl_sol).sum::<f64>()
					/ losing_trades.len() as f64
			} else {
				0.0
			};

			// Profit Factor
			let total_wins: f64 =
				winning_trades.iter().map(|t| t.pnl_sol).sum();
			let total_losses: f64 =
				losing_trades.iter().map(|t| t.pnl_sol.abs()).sum();
			let profit_factor = if total_losses > 0.0 {
				total_wins / total_losses
			} else {
				0.0
			};

			// Best and worst trades for this level
			let best_trade = trades
				.iter()
				.max_by(|a, b| a.pnl_sol.partial_cmp(&b.pnl_sol).unwrap());
			let worst_trade = trades
				.iter()
				.min_by(|a, b| a.pnl_sol.partial_cmp(&b.pnl_sol).unwrap());

			// Calculate range (min-max SOL levels in this group)
			let min_sol = trades
				.iter()
				.map(|t| t.sol_in_curve_before_our_buy as i32)
				.min()
				.unwrap_or(sol_level);
			let max_sol = trades
				.iter()
				.map(|t| t.sol_in_curve_before_our_buy as i32)
				.max()
				.unwrap_or(sol_level);
			let range_str = format!("{}-{}", min_sol, max_sol);

			grand_total_pnl += total_pnl;

			println!(
				"{:<8} {:<7} {:<+10.4} {:<6.1} {:<+8.4} {:<+9.4} {:<5.2} {:<12} {:<+8.4} {:<+8.4}",
				sol_level,
				trade_count,
				total_pnl,
				win_percentage,
				avg_win,
				avg_loss,
				profit_factor,
				range_str,
				best_trade.map(|t| t.pnl_sol).unwrap_or(0.0),
				worst_trade.map(|t| t.pnl_sol).unwrap_or(0.0)
			);
		}

		println!("{}", "-".repeat(90));
		println!("TOTAL    {}    {:+.4}", total_trades, grand_total_pnl);
		println!();
	}

	/// Print SOL in curve trigger analysis for a given set of trades
	pub fn print_sol_in_curve_analysis_for_trades(
		&self,
		trades: &[TradeSummary],
		sol_group_level: f64,
		title: &str,
	) {
		if trades.is_empty() {
			println!(
				"No completed trades available for SOL in curve analysis for {}.",
				title
			);
			return;
		}

		println!("\n{}", "=".repeat(100));
		println!("📊 {}", title.to_uppercase());
		println!("{}", "=".repeat(100));
		println!(
			"Using 'sol_in_curve_before_our_buy' column for SOL in curve analysis"
		);
		println!("SOL grouping level: {} SOL increments", sol_group_level);

		// Group trades by SOL level (floor-based for sequential ranges)
		let mut sol_level_groups: std::collections::HashMap<
			i32,
			Vec<&TradeSummary>,
		> = std::collections::HashMap::new();

		for trade in trades {
			let sol_level =
				((trade.sol_in_curve_before_our_buy / sol_group_level).floor()
					as i32) * (sol_group_level as i32);
			sol_level_groups
				.entry(sol_level)
				.or_insert_with(Vec::new)
				.push(trade);
		}

		let total_trades = trades.len();
		let total_sol_levels = sol_level_groups.len();

		println!(
			"Analyzing {} trades across {} different SOL in curve levels:",
			total_trades, total_sol_levels
		);
		println!();

		// Collect and sort SOL levels
		let mut sol_levels: Vec<i32> =
			sol_level_groups.keys().cloned().collect();
		sol_levels.sort();

		// Print header
		println!(
			"{:<8} {:<7} {:<10} {:<6} {:<8} {:<9} {:<5} {:<12} {:<8} {:<8}",
			"SOL Lvl",
			"Trades",
			"Total PnL",
			"Win%",
			"Avg Win",
			"Avg Loss",
			"PF",
			"Range",
			"Best",
			"Worst"
		);
		println!("{}", "-".repeat(90));

		let mut grand_total_pnl = 0.0;

		for sol_level in sol_levels {
			let trades = sol_level_groups.get(&sol_level).unwrap();
			let trade_count = trades.len();
			let total_pnl: f64 = trades.iter().map(|t| t.pnl_sol).sum();

			// Win/loss statistics
			let winning_trades: Vec<&&TradeSummary> =
				trades.iter().filter(|t| t.pnl_sol > 0.0).collect();
			let losing_trades: Vec<&&TradeSummary> =
				trades.iter().filter(|t| t.pnl_sol <= 0.0).collect();

			let win_count = winning_trades.len();
			let win_percentage = if trade_count > 0 {
				(win_count as f64 / trade_count as f64) * 100.0
			} else {
				0.0
			};

			let avg_win = if !winning_trades.is_empty() {
				winning_trades.iter().map(|t| t.pnl_sol).sum::<f64>()
					/ winning_trades.len() as f64
			} else {
				0.0
			};

			let avg_loss = if !losing_trades.is_empty() {
				losing_trades.iter().map(|t| t.pnl_sol).sum::<f64>()
					/ losing_trades.len() as f64
			} else {
				0.0
			};

			// Profit Factor
			let total_wins: f64 =
				winning_trades.iter().map(|t| t.pnl_sol).sum();
			let total_losses: f64 =
				losing_trades.iter().map(|t| t.pnl_sol.abs()).sum();
			let profit_factor = if total_losses > 0.0 {
				total_wins / total_losses
			} else {
				0.0
			};

			// Best and worst trades for this level
			let best_trade = trades
				.iter()
				.max_by(|a, b| a.pnl_sol.partial_cmp(&b.pnl_sol).unwrap());
			let worst_trade = trades
				.iter()
				.min_by(|a, b| a.pnl_sol.partial_cmp(&b.pnl_sol).unwrap());

			// Calculate range (min-max SOL levels in this group)
			let min_sol = trades
				.iter()
				.map(|t| t.sol_in_curve_before_our_buy as i32)
				.min()
				.unwrap_or(sol_level);
			let max_sol = trades
				.iter()
				.map(|t| t.sol_in_curve_before_our_buy as i32)
				.max()
				.unwrap_or(sol_level);
			let range_str = format!("{}-{}", min_sol, max_sol);

			grand_total_pnl += total_pnl;

			println!(
				"{:<8} {:<7} {:<+10.4} {:<6.1} {:<+8.4} {:<+9.4} {:<5.2} {:<12} {:<+8.4} {:<+8.4}",
				sol_level,
				trade_count,
				total_pnl,
				win_percentage,
				avg_win,
				avg_loss,
				profit_factor,
				range_str,
				best_trade.map(|t| t.pnl_sol).unwrap_or(0.0),
				worst_trade.map(|t| t.pnl_sol).unwrap_or(0.0)
			);
		}

		println!("{}", "-".repeat(90));
		println!("TOTAL    {}    {:+.4}", total_trades, grand_total_pnl);
		println!();
	}

	pub fn print_results_summary(
		&self,
		results: &BacktestResults,
		strategy_name: &str,
	) {
		if results.completed_trades_df.height() == 0 {
			println!(
				"No {} trades were completed during the simulation.",
				strategy_name
			);
			return;
		}

		println!("\n--- {} Results Summary ---", strategy_name);

		// Use centralized calculation method
		let stats = Self::calculate_trade_statistics(results);

		println!("Total Trades: {}", stats.total_trades);
		println!("Total PnL: {:.4} SOL", stats.cumulative_pnl);

		if !results.equity_curve.is_empty() {
			println!("Final Balance: {:+.4} SOL", stats.cumulative_pnl);
			println!("Peak Balance: {:+.4} SOL", stats.peak_balance);
			println!("Maximum Drawdown: {:+.4} SOL", stats.max_drawdown);
			println!(
				"Win Rate: {:.1}% ({}/{})",
				stats.win_rate, stats.win_count, stats.total_trades
			);
			println!("Average Win: {:+.4} SOL", stats.avg_win);
			println!("Average Loss: {:+.4} SOL", stats.avg_loss);
			println!("Profit Factor: {:.2}", stats.profit_factor);
		}

		// Print slippage failure stats
		if self.slippage_failures > 0 {
			println!("Slippage Failures: {} buy orders rejected", self.slippage_failures);
		}

		// Print large wins (over 1 SOL) - EXCLUDED FROM FINAL RESULTS
		if !results.large_wins.is_empty() {
			println!("\n🚀 Large Wins (>1 SOL) - EXCLUDED FROM FINAL RESULTS:");
			for (i, win) in results.large_wins.iter().enumerate() {
				println!(
					"   {}. {:.4} SOL - {} (Trigger TX: {}, Latency Adjusted TX: {}) [{}]",
					i + 1,
					win.pnl,
					win.token_address,
					win.trigger_tx_index + 1, // Convert to CSV line number (1-based)
					win.exit_tx_index + 1,    // Convert to CSV line number (1-based)
					win.reason
				);
			}
			println!(
				"   Total large wins: {} (removed from dataset and final results)",
				results.large_wins.len()
			);
		}
	}

	/// Export equity curve to CSV with identical format to Python version
	pub fn export_equity_curve(
		&self,
		results: &BacktestResults,
		strategy: &dyn TradingStrategy,
	) -> Result<()> {
		if !self.export_trades {
			return Ok(());
		}

		if results.equity_curve.is_empty() {
			println!("No equity curve data to export.");
			return Ok(());
		}

		// Create equity curves directory
		let home_dir =
			std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
		let equity_curves_dir = format!(
			"{}/Downloads/rust-backtester-main/equity_curves",
			home_dir
		);
		std::fs::create_dir_all(&equity_curves_dir)?;

		// Generate timestamp for filename (matching Python format)
		let now = chrono::Utc::now();
		let timestamp = now.format("%Y%m%d_%H%M%S").to_string();

	// Use clean system name for filename (no sanitization needed)
	let system_name = strategy.get_system_name();

	let filename = format!(
		"{}/{}_{}.csv",
		equity_curves_dir, system_name, timestamp
	);

		// Sort equity curve chronologically by exit_date (matching Python behavior)
		let mut sorted_equity = results.equity_curve.clone();
		sorted_equity.sort_by(|a, b| a.exit_date.cmp(&b.exit_date));

		// Create CSV content with exact same columns as Python version
		let mut csv_content = String::new();
		csv_content.push_str("trade_number,token_address,exit_date,cumulative_pnl,trade_pnl,trade_type,execution_latency_seconds\n");

		// Recalculate trade numbers and cumulative P&L in chronological order
		let mut chronological_cumulative_pnl = 0.0;
		for (index, entry) in sorted_equity.iter().enumerate() {
			chronological_cumulative_pnl += entry.trade_pnl;

			csv_content.push_str(&format!(
				"{},{},{},{:.4},{:.4},{},{:.1}\n",
				index + 1, // Sequential trade number starting from 1
				entry.token_address,
				entry.exit_date,
				chronological_cumulative_pnl, // Recalculated cumulative P&L
				entry.trade_pnl,
				entry.trade_type,
				entry.execution_latency_seconds
			));
		}

		// Write to file
		std::fs::write(&filename, csv_content)?;

		println!("📊 Equity curve exported to: {}", filename);
		println!(
			"   - Contains {} trades with columns: trade_number, token_address, exit_date, cumulative_pnl, trade_pnl, trade_type, execution_latency_seconds",
			results.equity_curve.len()
		);

		Ok(())
	}

	/// Export detailed trades to CSV (matching Python detailed trades export)
	pub fn export_detailed_trades(
		&self,
		results: &BacktestResults,
		strategy: &dyn TradingStrategy,
	) -> Result<()> {
		if !self.export_trades {
			return Ok(());
		}

		if results.completed_trades_df.height() == 0 {
			println!("No detailed trades to export.");
			return Ok(());
		}

		// Create merged_trades directory
		let home_dir =
			std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
		let merged_trades_dir = format!(
			"{}/Downloads/rust-backtester-main/merged_trades",
			home_dir
		);
		std::fs::create_dir_all(&merged_trades_dir)?;

		// Generate timestamp for filename (matching Python format)
		let now = chrono::Utc::now();
		let timestamp = now.format("%Y%m%d_%H%M%S").to_string();
		let iso_timestamp = now.to_rfc3339();

		// Get strategy name and parameters
		let strategy_name = strategy.get_strategy_name();
		let strategy_params = strategy.get_export_parameters();

		// Use clean system name for filename (no sanitization needed)
		let system_name = strategy.get_system_name();

		let filename =
			format!("{}/{}_{}.csv", merged_trades_dir, system_name, timestamp);

		// Create CSV content with header metadata and trade data
		let mut csv_content = String::new();

		// Add header metadata with clean section format
		csv_content.push_str(
			"# Header contains dataset reference and backtest results\n",
		);

		// Dataset section
		csv_content.push_str("#DATASET\n");
		let dataset_path = self
			.dataset_path
			.as_ref()
			.map(|arc| arc.as_str())
			.unwrap_or("/path/to/original_trades.csv");
		csv_content.push_str(&format!("path,{}\n", dataset_path));
		csv_content.push_str(&format!("generated,{}\n", iso_timestamp));
		csv_content.push_str("\n");

		// Strategy section - use system name from strategy
		csv_content.push_str("#STRATEGY\n");
		let system_name = strategy.get_system_name();
		csv_content.push_str(&format!("name,{}\n", system_name));
		csv_content.push_str("\n");

		// Strategy parameters section - all the detailed parameters
		csv_content.push_str("#PARAMETERS\n");
		if strategy_params.is_empty() {
			csv_content.push_str("# No strategy parameters\n");
		} else {
			for (key, value) in strategy_params.iter() {
				csv_content.push_str(&format!("{},{}\n", key, value));
			}
		}
		csv_content.push_str("\n");

		// Framework settings section
		csv_content.push_str("#SETTINGS\n");
		csv_content.push_str(&format!(
			"position_size_sol,{}\n",
			self.position_size_sol
		));
		csv_content
			.push_str(&format!("buy_fee_percent,{}\n", self.buy_fee_percent));
		csv_content
			.push_str(&format!("sell_fee_percent,{}\n", self.sell_fee_percent));
		csv_content
			.push_str(&format!("flat_fee_sol_buy,{}\n", self.flat_fee_sol_buy));
		csv_content.push_str(&format!(
			"flat_fee_sol_sell,{}\n",
			self.flat_fee_sol_sell
		));
		csv_content.push_str(&format!(
			"block_latency_max,{}\n",
			self.block_latency_max
		));
		csv_content.push_str("\n");

		// Trade statistics section - calculate and add for frontend consumption
		let stats = Self::calculate_trade_statistics(results);
		csv_content.push_str("#STATS\n");
		csv_content.push_str(&format!("total_trades,{}\n", stats.total_trades));
		csv_content
			.push_str(&format!("cumulative_pnl,{:.4}\n", stats.cumulative_pnl));
		csv_content
			.push_str(&format!("peak_balance,{:.4}\n", stats.peak_balance));
		csv_content
			.push_str(&format!("max_drawdown,{:.4}\n", stats.max_drawdown));
		csv_content.push_str(&format!("win_count,{}\n", stats.win_count));
		csv_content.push_str(&format!("loss_count,{}\n", stats.loss_count));
		csv_content.push_str(&format!("win_rate,{:.2}\n", stats.win_rate));
		csv_content.push_str(&format!("avg_win,{:.4}\n", stats.avg_win));
		csv_content.push_str(&format!("avg_loss,{:.4}\n", stats.avg_loss));
		csv_content
			.push_str(&format!("profit_factor,{:.4}\n", stats.profit_factor));
		csv_content.push_str(&format!("total_wins,{:.4}\n", stats.total_wins));
		csv_content
			.push_str(&format!("total_losses,{:.4}\n", stats.total_losses));
		csv_content.push_str(&format!(
			"large_wins_count,{}\n",
			stats.large_wins_count
		));
		csv_content.push_str(&format!(
			"large_wins_total,{:.4}\n",
			stats.large_wins_total
		));
		csv_content.push_str("\n");

		// Trade data section
		csv_content.push_str("#TRADES\n");
		csv_content
			.push_str("token_address,timestamp,trade_type,price,sol_amount\n");

		// Add trade data - use actual individual transactions
		for transaction in &self.individual_transactions {
			csv_content.push_str(&format!(
				"{},{},{},{:.8},{:.6}\n",
				transaction.token_address,
				transaction.timestamp,
				transaction.transaction_type,
				transaction.price,
				transaction.sol_amount
			));
		}

		// Add equity curve section (reusing logic from export_equity_curve)
		if !results.equity_curve.is_empty() {
			csv_content.push_str("\n");
			csv_content.push_str("#EQUITY\n");
			csv_content.push_str("trade_number,token_address,exit_date,cumulative_pnl,trade_pnl,trade_type,execution_latency_seconds\n");

			// Sort equity curve chronologically by exit_date (same as standalone export)
			let mut sorted_equity = results.equity_curve.clone();
			sorted_equity.sort_by(|a, b| a.exit_date.cmp(&b.exit_date));

			// Recalculate trade numbers and cumulative P&L in chronological order (same logic as standalone)
			let mut chronological_cumulative_pnl = 0.0;
			for (index, entry) in sorted_equity.iter().enumerate() {
				chronological_cumulative_pnl += entry.trade_pnl;

				csv_content.push_str(&format!(
					"{},{},{},{:.4},{:.4},{},{:.1}\n",
					index + 1, // Sequential trade number starting from 1
					entry.token_address,
					entry.exit_date,
					chronological_cumulative_pnl, // Recalculated cumulative P&L
					entry.trade_pnl,
					entry.trade_type,
					entry.execution_latency_seconds
				));
			}
		}

		// Write to file
		std::fs::write(&filename, csv_content)?;

		println!("📋 Detailed trades exported to: {}", filename);
		println!(
			"   - Contains {} individual transactions",
			self.individual_transactions.len()
		);
		if !results.equity_curve.is_empty() {
			println!(
				"   - Contains {} equity curve entries",
				results.equity_curve.len()
			);
		}
		println!("   - Strategy: {}", strategy_name);
		let sections = if results.equity_curve.is_empty() {
			"DATASET, STRATEGY, PARAMETERS, SETTINGS, STATS, TRADES"
		} else {
			"DATASET, STRATEGY, PARAMETERS, SETTINGS, STATS, TRADES, EQUITY"
		};
		println!(
			"   - Includes clean CSV headers with sections: {}",
			sections
		);

		Ok(())
	}

	pub fn print_market_cap_analysis(&self, sol_group_level: f64) {
		if self.completed_trades.is_empty() {
			println!("No completed trades available for Market Cap analysis.");
			return;
		}

		println!("\n{}", "=".repeat(100));
		println!("💹 MARKET CAP ANALYSIS BY SOL IN CURVE");
		println!("{}", "=".repeat(100));
		println!("(Only includes trades within 1% of each SOL level)");

		// Group trades by the nearest SOL level, if they are within 1% tolerance
		let mut sol_level_groups: std::collections::HashMap<i32, Vec<&TradeSummary>> =
			std::collections::HashMap::new();

		for trade in &self.completed_trades {
			let sol_in_curve = trade.sol_in_curve_before_our_buy;
			if sol_in_curve <= 0.0 { continue; }

			let nearest_level = (sol_in_curve / sol_group_level).round() as i32 * sol_group_level as i32;
			if nearest_level == 0 { continue; }

			let tolerance = nearest_level as f64 * 0.01;
			if (sol_in_curve - nearest_level as f64).abs() <= tolerance {
				sol_level_groups
					.entry(nearest_level)
					.or_insert_with(Vec::new)
					.push(trade);
			}
		}

		let mut sol_levels: Vec<i32> = sol_level_groups.keys().cloned().collect();
		sol_levels.sort();

		println!(
			"{:<12} {:<15} {:<20}",
			"SOL Level", "Trade Count", "Avg Market Cap ($)"
		);
		println!("{}", "-".repeat(50));

		for sol_level in sol_levels {
			let trades = sol_level_groups.get(&sol_level).unwrap();
			let trade_count = trades.len();

			let mut total_market_cap = 0.0;
			for trade in trades.iter() {
				// Market cap is token price * 1 billion
				let market_cap = trade.buy_price_ref * 1_000_000_000.0;
				total_market_cap += market_cap;
			}

			let avg_market_cap = if trade_count > 0 {
				total_market_cap / trade_count as f64
			} else {
				0.0
			};

			println!(
				"{:<12} {:<15} ${:<20.0}",
				sol_level,
				trade_count,
				avg_market_cap.round()
			);
		}
		println!("{}", "-".repeat(50));
		println!();
	}

}
