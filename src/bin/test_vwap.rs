/*!
VWAP Strategy Test Suite
========================

This test file contains all configurable parameters for testing the VWAP Strategy.
All trading fees, position sizes, strategy parameters, and validation settings are configured here.

To modify the strategy behavior, adjust the parameters in the TRADING CONFIGURATION section below.
*/

use anyhow::Result;
use rust_backtester::{
	BacktestFramework, VwapStrategy, TradingStrategy,
};
use rust_backtester::strategies::common_config::*;

// CSV file path for testing
const CSV_FILE_PATH: &str =
	"/home/daksh/Downloads/pumpfun_09-08-2025_to_09-11-2025.csv/pumpfun_09-08-2025_to_09-11-2025.csv";
    

// TRADING CONFIGURATION PARAMETERS
// =================================

// Strategy-Specific Configuration (universal parameters imported from common_config)
// All common parameters now imported from common_config

// VWAP Strategy Configuration
const DEFAULT_WINDOW_SIZE: usize = 120; // Default rolling window for VWAP calculation
const DEFAULT_BUY_THRESHOLD: f64 = -0.20; // Buy when price is 20% below VWAP
const DEFAULT_SELL_THRESHOLD: f64 = 0.11; // Sell when price is 11% above VWAP
const DEFAULT_STOP_LOSS_THRESHOLD: f64 = 0.60; // Stop loss at 60% loss
const DEFAULT_MAX_HOLD_TIME_MINUTES: i64 = 130; // Maximum hold time in minutes
const PRICE_COLUMN: &str = "Token Price"; // Column to use for price in VWAP calculation
const VOLUME_COLUMN: &str = "Token Amount"; // Column to use for volume in VWAP calculation
const MAX_BUYS: usize = 1; // Maximum number of buy positions per token


// Dynamic Threshold Configuration
const USE_DYNAMIC_THRESHOLDS: bool = false; // Use dynamic thresholds based on SOL in curve (true) or static thresholds from above (false)

// Trade Velocity Configuration
const USE_TRADE_VELOCITY: bool = true; // Use trade velocity for threshold adjustment
const VELOCITY_WINDOW_SECONDS: f64 = 30.0; // Time window for velocity calculation (30 seconds)
const HIGH_VELOCITY_THRESHOLD: f64 = 2.0; // Trades per second threshold for "high velocity"
const LOW_VELOCITY_THRESHOLD: f64 = 0.2; // Trades per second threshold for "low velocity" 
const HIGH_VELOCITY_MULTIPLIER: f64 = 0.8; // Threshold multiplier for high velocity (tighter thresholds)
const LOW_VELOCITY_MULTIPLIER: f64 = 1.2; // Threshold multiplier for low velocity (wider thresholds)



fn test_vwap_strategy() -> Result<()> {
	println!("{}", "=".repeat(80));
	println!("📊 Testing VWAP Strategy");
	println!("{}", "=".repeat(80));

	// Create strategy instance with configured parameters (verbose=true for single test)
	let mut strategy = VwapStrategy::new(
		DEFAULT_WINDOW_SIZE,
		DEFAULT_BUY_THRESHOLD,
		DEFAULT_SELL_THRESHOLD,
		DEFAULT_STOP_LOSS_THRESHOLD,
		MIN_SOL_IN_CURVE,
		PRICE_COLUMN,
		VOLUME_COLUMN,
		MAX_BUYS,
		MIN_BLOCKS_BETWEEN_BUYS, // minimum blocks between consecutive buys
		MIN_HOLD_BLOCKS, // minimum blocks to hold before selling
		MIN_BLOCKS_BETWEEN_SELL_BUY, // minimum blocks between sell and next buy
		DEFAULT_MAX_HOLD_TIME_MINUTES,
		USE_DYNAMIC_THRESHOLDS, // use dynamic thresholds based on SOL in curve or static thresholds
		SLIPPAGE_TOLERANCE_PERCENT, // slippage tolerance for buy orders
		USE_TRADE_VELOCITY, // use trade velocity for threshold adjustment
		VELOCITY_WINDOW_SECONDS, // velocity window in seconds
		HIGH_VELOCITY_THRESHOLD, // high velocity threshold (trades/sec)
		LOW_VELOCITY_THRESHOLD, // low velocity threshold (trades/sec)  
		HIGH_VELOCITY_MULTIPLIER, // high velocity multiplier
		LOW_VELOCITY_MULTIPLIER, // low velocity multiplier
		VERBOSE, // verbose=true to show detailed debug output during testing
	);

	// Create framework instance with configured fees and position size
	let mut framework = BacktestFramework::new(
		POSITION_SIZE_SOL,
		BUY_FEE_PERCENT,
		SELL_FEE_PERCENT,
		FLAT_FEE_SOL_BUY,
		FLAT_FEE_SOL_SELL,
		BLOCK_LATENCY_MAX,
		true, // export_trades = true for single strategy testing
		VERBOSE, // verbose = true for detailed debug output during testing
		USE_DYNAMIC_FEES,
	);

	framework.preprocess_data(
		CSV_FILE_PATH,
		VALIDATE_DATA,
		TOLERANCE_PERCENT,
		DETECT_MANIPULATION,
		MANIPULATION_THRESHOLD,
	)?;

	// Run strategy on preprocessed data
	println!("Running strategy with:");
	println!("  📊 CSV file: {}", CSV_FILE_PATH);
	println!("  📈 Rolling window: {} periods", DEFAULT_WINDOW_SIZE);
	println!(
		"  🔻 Buy threshold: {:.1}% below VWAP (Price: {}, Volume: {})",
		DEFAULT_BUY_THRESHOLD * 100.0, PRICE_COLUMN, VOLUME_COLUMN
	);
	println!(
		"  🔺 Sell threshold: {:.1}% above VWAP (Price: {}, Volume: {})",
		DEFAULT_SELL_THRESHOLD * 100.0, PRICE_COLUMN, VOLUME_COLUMN
	);
	println!("  💰 Position size: {} SOL", POSITION_SIZE_SOL);
	println!("  🚫 Min SOL in curve: {}", MIN_SOL_IN_CURVE);
	println!("  🔢 Max buys per token: {}", MAX_BUYS);
	println!("  🔄 Min blocks between buys: {}", MIN_BLOCKS_BETWEEN_BUYS);
	println!("  ⏳ Min hold blocks: {}", MIN_HOLD_BLOCKS);
	println!("  🔀 Min blocks sell → buy: {}", MIN_BLOCKS_BETWEEN_SELL_BUY);
	if USE_DYNAMIC_FEES {
		println!("  💸 Fees: Dynamic based on market cap");
	} else {
		println!(
			"  💸 Buy fee: {:.2}% + {:.6} SOL",
			BUY_FEE_PERCENT * 100.0,
			FLAT_FEE_SOL_BUY
		);
		println!(
			"  💸 Sell fee: {:.2}% + {:.6} SOL",
			SELL_FEE_PERCENT * 100.0,
			FLAT_FEE_SOL_SELL
		);
	}
	println!("  ⏰ Block latency max: {} blocks", BLOCK_LATENCY_MAX);
	println!(
		"  🎯 Strategy: VWAP-based entry and exit with {:.0}% stop loss",
		DEFAULT_STOP_LOSS_THRESHOLD * 100.0
	);
    println!("  ⏰ Max hold time: {} minutes", DEFAULT_MAX_HOLD_TIME_MINUTES);
	println!("  📊 Buy slippage tolerance: {:.0}%", SLIPPAGE_TOLERANCE_PERCENT * 100.0);
	if USE_TRADE_VELOCITY {
		println!(
			"  🚀 Trade Velocity: {:.1}s window, {:.2}-{:.1} trades/sec thresholds, {:.2}x-{:.2}x multipliers",
			VELOCITY_WINDOW_SECONDS,
			LOW_VELOCITY_THRESHOLD,
			HIGH_VELOCITY_THRESHOLD,
			HIGH_VELOCITY_MULTIPLIER,
			LOW_VELOCITY_MULTIPLIER
		);
	} else {
		println!("  🚀 Trade Velocity: Disabled");
	}
	println!();

	let results = framework.run_strategy_on_preprocessed_data(&mut strategy)?;

	// Print results
	if results.completed_trades_df.height() > 0 {
		// Print SOL in curve trigger analysis first (using default 25.0 SOL grouping)
		framework.print_sol_in_curve_analysis(25.0);

		// Then print main results summary right below it
		framework
			.print_results_summary(&results, &strategy.get_strategy_name());
	} else {
		println!("❌ Backtest failed - no results returned");
		return Ok(());
	}

	// Print strategy-specific debug stats
	let debug_stats = strategy.get_debug_stats();
	println!("\n--- VWAP Strategy Debug Statistics ---");
	println!(
		"Total buy signals: {}",
		debug_stats.get("total_buy_signals").unwrap_or(&0)
	);
	println!(
		"Total sell signals: {}",
		debug_stats.get("total_sell_signals").unwrap_or(&0)
	);
	println!(
		"Insufficient data rejections: {}",
		debug_stats
			.get("insufficient_data_rejections")
			.unwrap_or(&0)
	);
	println!(
		"SOL curve rejections (< {}): {}",
		MIN_SOL_IN_CURVE,
		debug_stats.get("sol_curve_rejections").unwrap_or(&0)
	);
	println!(
		"VWAP calculations performed: {}",
		debug_stats.get("vwap_calculations").unwrap_or(&0)
	);
	println!(
		"Tokens ready for trading: {}",
		debug_stats.get("tokens_ready_count").unwrap_or(&0)
	);
	println!(
		"Total tokens tracked: {}",
		debug_stats.get("total_tokens_tracked").unwrap_or(&0)
	);

	// Export results to CSV files (matching Python export format)
	if results.completed_trades_df.height() > 0 {
		println!("\n📁 Exporting results to CSV files...");

		// Export equity curve
		if let Err(e) = framework
			.export_equity_curve(&results, &strategy)
		{
			println!("⚠️  Failed to export equity curve: {}", e);
		}

		// Export detailed trades
		if let Err(e) = framework.export_detailed_trades(&results, &strategy) {
			println!("⚠️  Failed to export detailed trades: {}", e);
		}

	} else {
		println!("\n⚠️  No trades completed - nothing to export");
	}

	Ok(())
}

fn main() -> Result<()> {
	println!("🚀 Starting VWAP Strategy Testing Suite");
	println!("VWAP-based entry and exit signals with volume weighting");
	println!();

	// Test default parameters
	test_vwap_strategy()?;

	println!("\n✅ VWAP Strategy Test Complete!");

	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_strategy_creation() {
		let strategy = VwapStrategy::new(
			DEFAULT_WINDOW_SIZE,
			DEFAULT_BUY_THRESHOLD,
			DEFAULT_SELL_THRESHOLD,
			DEFAULT_STOP_LOSS_THRESHOLD,
			MIN_SOL_IN_CURVE,
			PRICE_COLUMN,
			VOLUME_COLUMN,
			MAX_BUYS,
			MIN_BLOCKS_BETWEEN_BUYS, // minimum blocks between consecutive buys
			MIN_HOLD_BLOCKS, // minimum blocks to hold before selling  
			MIN_BLOCKS_BETWEEN_SELL_BUY, // minimum blocks between sell and next buy
			DEFAULT_MAX_HOLD_TIME_MINUTES,
			false, // use_dynamic_thresholds=false for tests
			SLIPPAGE_TOLERANCE_PERCENT, // slippage_tolerance_percent
			false, // use_trade_velocity=false for tests
			30.0,  // velocity_window_seconds 
			2.0,   // high_velocity_threshold
			0.2,   // low_velocity_threshold
			0.8,   // high_velocity_multiplier
			1.2,   // low_velocity_multiplier
			false, // verbose=false for tests
		);

		assert_eq!(
			strategy.get_strategy_name().contains("VWAP"),
			true
		);
	}

	#[test]
	fn test_framework_creation() {
		let framework = BacktestFramework::new(
			POSITION_SIZE_SOL,
			BUY_FEE_PERCENT,
			SELL_FEE_PERCENT,
			FLAT_FEE_SOL_BUY,
			FLAT_FEE_SOL_SELL,
			BLOCK_LATENCY_MAX,
			false, // export_trades = false for tests
			false, // verbose = false for tests
			false, // use_dynamic_fees
		);

		// Basic smoke test - framework should be created without panicking
		assert!(true);
	}
}