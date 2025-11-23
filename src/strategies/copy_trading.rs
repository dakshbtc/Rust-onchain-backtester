use anyhow::Result;
use polars::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::{framework::backtesting_framework::TradeSummary, strategies::strategy_base::SellReason};
use crate::strategies::strategy_base::TradingStrategy;
use super::helpers::{can_buy_with_constraints, can_sell_with_hold_blocks, get_block_id, get_wallet_address, get_transaction_type, get_sol_in_curve, get_token_amount, get_token_price, update_threshold_tracking};

/// Simple copy trading strategy that mirrors a specific wallet's trades
/// - When target wallet buys a token, we buy the same token
/// - When target wallet sells a token, we sell the same token  
/// - Uses fixed position sizing (0.1 SOL per trade)
pub struct CopyTradingStrategy {
	// Target wallet to copy
	target_wallet: String,
	
	// Position tracking - supports multiple buys per token like other strategies
	current_positions: HashMap<String, Vec<f64>>, // {token_address: [list of buy_prices]}
	buy_block_ids: HashMap<String, Vec<i64>>, // {token_address: [list of buy_block_ids]}
	sell_block_ids: HashMap<String, i64>, // {token_address: sell_block_id}
	
	// Target wallet tracking
	target_wallet_positions: HashMap<String, Vec<(f64, f64, f64, f64)>>, // {token_address: [(token_amount, cost_in_sol, sol_in_curve, cost_in_sol_no_fees)]}
	target_wallet_pnl: f64,
	target_wallet_pnl_no_fees: f64,
	target_wallet_completed_trades: Vec<TradeSummary>,
	
	// Threshold tracking
	tokens_reached_threshold: HashSet<String>,
	
	// Fee structure (same as ours for comparison)
	buy_fee_percent: f64,
	sell_fee_percent: f64,
	flat_fee_sol_buy: f64,
	flat_fee_sol_sell: f64,
	
	// Constraints
	min_sol_in_curve: f64,
	min_blocks_between_buys: i64,
	min_hold_blocks: i64,
	min_blocks_between_sell_buy: i64,
	max_buys: usize,
	
	// Debug counters
	total_buy_signals: i64,
	total_sell_signals: i64,
	target_wallet_buys: i64,
	target_wallet_sells: i64,
	sol_curve_rejections: i64,
	
	// Output control
	verbose: bool,
}

impl CopyTradingStrategy {
	/// Create a new copy trading strategy
	pub fn new(
		target_wallet: &str,
		min_sol_in_curve: f64,
		buy_fee_percent: f64,
		sell_fee_percent: f64,
		flat_fee_sol_buy: f64,
		flat_fee_sol_sell: f64,
		max_buys: usize,
		min_blocks_between_buys: i64,
		min_hold_blocks: i64,
		min_blocks_between_sell_buy: i64,
		verbose: bool,
	) -> Self {
		Self {
			target_wallet: target_wallet.to_string(),
			current_positions: HashMap::new(),
			buy_block_ids: HashMap::new(),
			sell_block_ids: HashMap::new(),
			target_wallet_positions: HashMap::new(),
			target_wallet_pnl: 0.0,
			target_wallet_pnl_no_fees: 0.0,
			target_wallet_completed_trades: Vec::new(),
			tokens_reached_threshold: HashSet::new(),
			
			buy_fee_percent,
			sell_fee_percent,
			flat_fee_sol_buy,
			flat_fee_sol_sell,
			
			min_sol_in_curve,
			min_blocks_between_buys,
			min_hold_blocks,
			min_blocks_between_sell_buy,
			max_buys,
			
			total_buy_signals: 0,
			total_sell_signals: 0,
			target_wallet_buys: 0,
			target_wallet_sells: 0,
			sol_curve_rejections: 0,
			
			verbose,
		}
	}








	/// Track target wallet's buy
	fn track_target_wallet_buy(&mut self, token_address: &str, token_amount: f64, token_price: f64, sol_in_curve: f64) {
		// Calculate cost with fees: token_amount * token_price * (1 + fee) + flat_fee
		let cost_in_sol = token_amount * token_price * (1.0 + self.buy_fee_percent) + self.flat_fee_sol_buy;
		let cost_in_sol_no_fees = token_amount * token_price;

		// Add to positions
		if !self.target_wallet_positions.contains_key(token_address) {
			self.target_wallet_positions.insert(token_address.to_string(), Vec::new());
		}
		self.target_wallet_positions.get_mut(token_address).unwrap().push((token_amount, cost_in_sol, sol_in_curve, cost_in_sol_no_fees));
	}

	/// Track target wallet's sell
	fn track_target_wallet_sell(&mut self, token_address: &str, token_amount: f64, token_price: f64, row_data: &HashMap<String, AnyValue>) -> (f64, f64) {
		let mut pnl = 0.0;
		let mut pnl_no_fees = 0.0;

		if let Some(positions) = self.target_wallet_positions.get_mut(token_address) {
			let proceeds_in_sol = token_amount * token_price * (1.0 - self.sell_fee_percent) - self.flat_fee_sol_sell;
			let proceeds_in_sol_no_fees = token_amount * token_price;

			let mut remaining_to_sell = token_amount;
			let mut total_cost_of_sold_positions = 0.0;
			let mut total_cost_of_sold_positions_no_fees = 0.0;
			let mut total_tokens_from_sold_positions = 0.0;
			let mut sol_in_curve_at_first_buy = 0.0;
			let mut buys_consumed_count = 0;

			let original_positions_count = positions.len();

			for i in 0..original_positions_count {
				if remaining_to_sell <= 0.0 {
					break;
				}

				let (pos_amount, pos_cost, pos_sol_in_curve, pos_cost_no_fees) = positions[i];

				if i == 0 {
					sol_in_curve_at_first_buy = pos_sol_in_curve;
				}

				if pos_amount <= remaining_to_sell {
					// Consume whole position
					total_cost_of_sold_positions += pos_cost;
					total_cost_of_sold_positions_no_fees += pos_cost_no_fees;
					total_tokens_from_sold_positions += pos_amount;
					remaining_to_sell -= pos_amount;
					buys_consumed_count += 1;
				} else {
					// Consume partial position
					let portion_to_sell = remaining_to_sell / pos_amount;
					total_cost_of_sold_positions += pos_cost * portion_to_sell;
					total_cost_of_sold_positions_no_fees += pos_cost_no_fees * portion_to_sell;
					total_tokens_from_sold_positions += remaining_to_sell;

					// Update position
					positions[i] = (pos_amount - remaining_to_sell, pos_cost * (1.0 - portion_to_sell), pos_sol_in_curve, pos_cost_no_fees * (1.0 - portion_to_sell));
					remaining_to_sell = 0.0;
				}
			}

			if buys_consumed_count > 0 {
				positions.drain(0..buys_consumed_count);
			}

			if total_tokens_from_sold_positions > 0.0 {
				// The proceeds are for `token_amount`, but we might sell less if we don't have enough.
				// The logic should be based on `total_tokens_from_sold_positions`.
				let proceeds_per_token = proceeds_in_sol / token_amount;
				let actual_proceeds = proceeds_per_token * total_tokens_from_sold_positions;
				pnl = actual_proceeds - total_cost_of_sold_positions;

				let proceeds_per_token_no_fees = proceeds_in_sol_no_fees / token_amount;
				let actual_proceeds_no_fees = proceeds_per_token_no_fees * total_tokens_from_sold_positions;
				pnl_no_fees = actual_proceeds_no_fees - total_cost_of_sold_positions_no_fees;

				let trade_summary = TradeSummary {
					token_address: token_address.to_string(),
					pnl_sol: pnl,
					sol_in_curve_before_our_buy: sol_in_curve_at_first_buy,
					our_sol_invested: total_cost_of_sold_positions,
					our_tokens_bought: total_tokens_from_sold_positions,
					our_sol_received: actual_proceeds,
					exit_reason: "COPY_TARGET_WALLET".to_string(),
					// Fill other fields with default/dummy values
					buy_tx_index: 0,
					buy_execution_index: 0,
					buy_latency_seconds: 0.0,
					buy_execution_time: "".to_string(),
					sell_tx_index: "".to_string(),
					sell_execution_index: None,
					sell_latency_seconds: None,
					sell_execution_logic: None,
					sell_execution_time: None,
					buy_price_ref: 0.0,
					our_effective_buy_price: if total_tokens_from_sold_positions > 0.0 {
						total_cost_of_sold_positions / total_tokens_from_sold_positions
					} else {
						0.0
					},
					price_after_our_buy: 0.0,
					sol_in_curve_at_our_buy: 0.0,
					current_sol_in_curve: get_sol_in_curve(row_data),
					sell_price_ref: token_price,
					our_effective_sell_price: if total_tokens_from_sold_positions > 0.0 {
						actual_proceeds / total_tokens_from_sold_positions
					} else {
						0.0
					},
					price_after_our_sell: 0.0,
				};
				self.target_wallet_completed_trades.push(trade_summary);
			}
		}

		(pnl, pnl_no_fees)
	}

	/// Get target wallet's current P/L
	pub fn get_target_wallet_pnl(&self) -> f64 {
		self.target_wallet_pnl
	}

	/// Get target wallet's current P/L without fees
	pub fn get_target_wallet_pnl_no_fees(&self) -> f64 {
		self.target_wallet_pnl_no_fees
	}

	/// Get target wallet's open positions count
	pub fn get_target_wallet_open_positions(&self) -> usize {
		self.target_wallet_positions.values().map(|v| v.len()).sum()
	}

	pub fn get_target_wallet_completed_trades(&self) -> &Vec<TradeSummary> {
		&self.target_wallet_completed_trades
	}
}

impl TradingStrategy for CopyTradingStrategy {
	fn should_buy(
		&mut self,
		token_address: &str,
		current_price: f64,
		current_index: usize,
		_row: &AnyValue,
		row_data: &HashMap<String, AnyValue>,
	) -> Result<bool> {
		let wallet_address = get_wallet_address(row_data);
		let transaction_type = get_transaction_type(row_data);
		let current_sol_curve = get_sol_in_curve(row_data);
		let current_block_id = get_block_id(row_data);


		// Only consider transactions from our target wallet
		if wallet_address != self.target_wallet {
			return Ok(false);
		}

		if self.verbose {
			println!("🔍 Target wallet transaction detected: {} {}", transaction_type, &token_address[token_address.len().saturating_sub(8)..]);
		}

		// Only buy if the target wallet is buying
		if transaction_type.to_lowercase() != "buy" {
			if self.verbose {
				println!("  ❌ Not a buy transaction: {}", transaction_type);
			}
			return Ok(false);
		}

		self.target_wallet_buys += 1;

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
			if self.verbose {
				println!("  ❌ Constraint violation for {}", &token_address[token_address.len().saturating_sub(8)..]);
			}
			return Ok(false);
		}

		self.total_buy_signals += 1;

		if self.verbose {
			println!(
				"\n[Index {}] 🎯 COPY BUY for {}",
				current_index,
				&token_address[token_address.len().saturating_sub(8)..]
			);
			println!("  Target wallet: {}", &wallet_address[..8]);
			println!("  Current price: {:.10e}", current_price);
			println!("  SOL in curve: {:.2}", current_sol_curve);
		}

		Ok(true)
	}

	fn should_sell(
		&mut self,
		token_address: &str,
		current_price: f64,
		current_index: usize,
		_row: &AnyValue,
		row_data: &HashMap<String, AnyValue>,
	) -> Result<(bool, SellReason)> {
		let wallet_address = get_wallet_address(row_data);
		let transaction_type = get_transaction_type(row_data);
		let current_block_id = get_block_id(row_data);

		// Check if we have a position to sell
		if !self.current_positions.contains_key(token_address)
			|| self.current_positions[token_address].is_empty()
		{
			return Ok((false, SellReason::Strategy("NO_POSITION".to_string())));
		}

		// Only consider transactions from our target wallet
		if wallet_address != self.target_wallet {
			return Ok((false, SellReason::Strategy("NOT_TARGET_WALLET".to_string())));
		}


		// Only sell if the target wallet is selling
		if transaction_type.to_lowercase() != "sell" {
			return Ok((false, SellReason::Strategy("TARGET_WALLET_NOT_SELLING".to_string())));
		}

		self.target_wallet_sells += 1;

		// Check minimum hold blocks constraint (using universal constraints)
		if !can_sell_with_hold_blocks(
			token_address,
			current_block_id,
			&self.buy_block_ids,
			self.min_hold_blocks,
		) {
			return Ok((false, SellReason::Strategy("MIN_HOLD_BLOCKS_NOT_MET".to_string())));
		}

		self.total_sell_signals += 1;

		if self.verbose {
			println!(
				"\n[Index {}] 🎯 COPY SELL for {}",
				current_index,
				&token_address[token_address.len().saturating_sub(8)..]
			);
			println!("  Target wallet: {}", &wallet_address[..8]);
			println!("  Current price: {:.10e}", current_price);
		}

		Ok((true, SellReason::Strategy("COPY_TARGET_WALLET".to_string())))
	}

	fn update_data(
		&mut self,
		token_address: &str,
		_price: f64,
		_tx_index: usize,
		_date: &str,
		row_data: &HashMap<String, AnyValue>,
	) -> Result<()> {
		let wallet_address = get_wallet_address(row_data);
		let transaction_type = get_transaction_type(row_data);
		
		// Track target wallet's trades for P/L calculation
		if wallet_address == self.target_wallet {
			let token_amount = get_token_amount(row_data);
			let token_price = get_token_price(row_data);
			let sol_in_curve = get_sol_in_curve(row_data);
			
			match transaction_type.to_lowercase().as_str() {
				"buy" => {
					self.track_target_wallet_buy(token_address, token_amount, token_price, sol_in_curve);
				}
				"sell" => {
					let (trade_pnl, trade_pnl_no_fees) = self.track_target_wallet_sell(token_address, token_amount, token_price, row_data);
					self.target_wallet_pnl += trade_pnl;
					self.target_wallet_pnl_no_fees += trade_pnl_no_fees;
				}
				_ => {}
			}
		}
		
		// Track when tokens first reach the SOL threshold (for all transactions, not just target wallet)
		let current_sol_curve = get_sol_in_curve(row_data);
		update_threshold_tracking(token_address, current_sol_curve, self.min_sol_in_curve, &mut self.tokens_reached_threshold);
		
		Ok(())
	}

	fn get_strategy_name(&self) -> String {
		format!(
			"Copy Trading {} (min_sol: {}, max_buys: {}, hold_blocks: {}, sell_to_buy_blocks: {}, buy_to_buy_blocks: {})",
			&self.target_wallet[..8],
			self.min_sol_in_curve,
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
		_current_index: Option<usize>,
		row_data: Option<&HashMap<String, AnyValue>>,
	) -> Result<()> {
		// Initialize lists if this is the first buy for this token
		if !self.current_positions.contains_key(token_address) {
			self.current_positions
				.insert(token_address.to_string(), Vec::new());
			self.buy_block_ids
				.insert(token_address.to_string(), Vec::new());
		}

		// Add the new buy to the lists
		self.current_positions
			.get_mut(token_address)
			.unwrap()
			.push(current_price);

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

		if self.verbose {
			println!(
				"  📊 Now holding {} positions for {}",
				self.current_positions[token_address].len(),
				&token_address[token_address.len().saturating_sub(8)..]
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

		// Track sell block ID for minimum blocks between sell and buy constraint
		let current_block_id = if let Some(row_data) = row_data {
			get_block_id(row_data)
		} else {
			0 // Default to 0 if no data available
		};
		self.sell_block_ids
			.insert(token_address.to_string(), current_block_id);

		if self.verbose {
			println!(
				"  📤 Sold all {} positions for {}",
				positions_count,
				&token_address[token_address.len().saturating_sub(8)..]
			);
		}

		Ok(())
	}

	fn get_debug_stats(&self) -> HashMap<String, i64> {
		let mut stats = HashMap::new();

		stats.insert("total_buy_signals".to_string(), self.total_buy_signals);
		stats.insert("total_sell_signals".to_string(), self.total_sell_signals);
		stats.insert("target_wallet_buys".to_string(), self.target_wallet_buys);
		stats.insert("target_wallet_sells".to_string(), self.target_wallet_sells);
		stats.insert("sol_curve_rejections".to_string(), self.sol_curve_rejections);
		// Calculate open positions
		let open_positions: usize = self
			.current_positions
			.values()
			.map(|positions| positions.len())
			.sum();
		stats.insert("open_positions".to_string(), open_positions as i64);
		stats.insert("target_wallet_open_positions".to_string(), self.get_target_wallet_open_positions() as i64);

		stats
	}

	fn get_export_parameters(&self) -> HashMap<String, String> {
		let mut params = HashMap::new();
		params.insert("target_wallet".to_string(), format!("\"{}\"", self.target_wallet));
		params.insert("min_sol_in_curve".to_string(), self.min_sol_in_curve.to_string());
		params.insert("max_buys".to_string(), self.max_buys.to_string());
		params.insert("min_blocks_between_buys".to_string(), self.min_blocks_between_buys.to_string());
		params.insert("min_hold_blocks".to_string(), self.min_hold_blocks.to_string());
		params.insert("min_blocks_between_sell_buy".to_string(), self.min_blocks_between_sell_buy.to_string());
		params
	}

	fn get_system_name(&self) -> String {
		"CopyTradingStrategy".to_string()
	}

	fn get_slippage_tolerance(&self) -> f64 {
		crate::strategies::common_config::SLIPPAGE_TOLERANCE_PERCENT
	}

}
