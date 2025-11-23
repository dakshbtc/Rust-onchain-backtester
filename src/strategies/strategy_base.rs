use anyhow::Result;
use polars::prelude::*;
use std::collections::HashMap;

/// Sell reason returned by trading strategies
#[derive(Debug, Clone)]
pub enum SellReason {
	Strategy(String),
	StopLoss(f64), // Loss percentage
	TakeProfit(f64), // Profit percentage
	Liquidation,
}

impl SellReason {
	pub fn as_string(&self) -> String {
		match self {
			SellReason::Strategy(reason) => reason.clone(),
			SellReason::StopLoss(loss_pct) => {
				format!("STOP_LOSS_{:.0}%", loss_pct * 100.0)
			}
			SellReason::TakeProfit(profit_pct) => {
				format!("TAKE_PROFIT_{:.0}%", profit_pct * 100.0)
			}
			SellReason::Liquidation => "LIQUIDATION".to_string(),
		}
	}
}

/// Strategy data that can be passed between buy/sell operations
pub type StrategyData = HashMap<String, String>;

/// Abstract trait for trading strategies
pub trait TradingStrategy {
	/// Determine if we should buy based on strategy logic
	fn should_buy(
		&mut self,
		token_address: &str,
		current_price: f64,
		current_index: usize,
		row: &AnyValue,
		row_data: &HashMap<String, AnyValue>,
	) -> Result<bool>;

	/// Determine if we should sell based on strategy logic
	/// Returns (should_sell, sell_reason)
	fn should_sell(
		&mut self,
		token_address: &str,
		current_price: f64,
		current_index: usize,
		row: &AnyValue,
		row_data: &HashMap<String, AnyValue>,
	) -> Result<(bool, SellReason)>;

	/// Update any strategy-specific data structures
	fn update_data(
		&mut self,
		token_address: &str,
		price: f64,
		tx_index: usize,
		date: &str,
		row_data: &HashMap<String, AnyValue>,
	) -> Result<()>;

	/// Return the name of the strategy
	fn get_strategy_name(&self) -> String;

	/// Called when a buy is executed (optional override)
	fn on_buy_executed(
		&mut self,
		_token_address: &str,
		_current_price: f64,
		_sol_invested: f64,
		_tokens_bought: f64,
		_current_index: Option<usize>,
		_row_data: Option<&HashMap<String, AnyValue>>,
	) -> Result<()> {
		Ok(())
	}

	/// Called when a sell is executed (optional override)
	fn on_sell_executed(
		&mut self,
		_token_address: &str,
		_current_index: Option<usize>,
		_row_data: Option<&HashMap<String, AnyValue>>,
	) -> Result<()> {
		Ok(())
	}

	/// Get entry size for this token (optional override)
	fn get_entry_size(
		&self,
		_token_address: &str,
		_price: f64,
		_tx_index: usize,
		_row_data: &HashMap<String, AnyValue>,
	) -> f64 {
		0.1 // Default position size
	}

	/// Get debug statistics (optional override)
	fn get_debug_stats(&self) -> HashMap<String, i64> {
		HashMap::new()
	}

	/// Get the Z-score that triggered the last buy signal (optional override)
	fn get_buy_z_score(&self) -> Option<f64> {
		None
	}

	/// Get strategy parameters for export (simple key-value pairs)
	fn get_export_parameters(&self) -> HashMap<String, String> {
		HashMap::new()
	}

	/// Get the system name of the strategy (e.g., "MeanReversionStrategy")
	fn get_system_name(&self) -> String {
		// Default fallback - strategies should override this
		"UnknownStrategy".to_string()
	}

	/// Get slippage tolerance for buy orders as percentage (e.g., 0.50 = 50%)
	/// Returns 0.0 to disable slippage checking (default behavior)
	fn get_slippage_tolerance(&self) -> f64 {
		0.0 // Default: no slippage checking
	}
}


