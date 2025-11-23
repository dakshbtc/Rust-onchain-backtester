/*! 
Inter-Trade Strategy Test Suite
==============================

This test file contains all configurable parameters for testing the Inter-Trade (enhanced mean reversion) Strategy.
All trading fees, position sizes, strategy parameters, and validation settings are configured here.

To modify the strategy behavior, adjust the parameters in the TRADING CONFIGURATION section below.
*/

use anyhow::Result;
use rust_backtester::{
	BacktestFramework, TradingStrategy,
};
use rust_backtester::strategies::InterTradeStrategy;
use rust_backtester::strategies::common_config::*;

// CSV file path for testing
const CSV_FILE_PATH: &str =
	"/home/daksh/Downloads/pumpfun_09-08-2025_to_09-11-2025.csv/pumpfun_09-08-2025_to_09-11-2025.csv";

// TRADING CONFIGURATION PARAMETERS
// =================================

// Inter-Trade Strategy Configuration (universal parameters imported from common_config)
const DEFAULT_WINDOW_SIZE: usize = 400; // Default rolling window for z-score
const DEFAULT_BUY_THRESHOLD: f64 = -3.5; // Buy when Z-score < -3.5 (oversold)
const DEFAULT_SELL_THRESHOLD: f64 = 0.8; // Sell when Z-score > 0.8 (mean reversion)
const Z_SCORE_COLUMN: &str = "Token Price"; // Column to use for Z-score calculation ("SOL in Curve", "Token Price", or "SOL")
const MAX_BUYS: usize = 1; // Maximum number of buy positions per token
const SELL_VELOCITY_BUY_TRIGGER_THRESHOLD: f64 = 5.0;
const SELL_BUY_RATIO_THRESHOLD: f64 = 5.0;

fn test_inter_trade_strategy() -> Result<()> {
	println!("{}", "=".repeat(80));
	println!("📊 Testing Inter-Trade Strategy");
	println!("{}", "=".repeat(80));

	// Create strategy instance with configured parameters (verbose=true for single test)
	let mut strategy = InterTradeStrategy::new(
		DEFAULT_WINDOW_SIZE,
		DEFAULT_BUY_THRESHOLD,
		DEFAULT_SELL_THRESHOLD,
		MIN_SOL_IN_CURVE,
		Z_SCORE_COLUMN,
		MAX_BUYS,
		MIN_BLOCKS_BETWEEN_BUYS,
		MIN_HOLD_BLOCKS,
		MIN_BLOCKS_BETWEEN_SELL_BUY,
		SELL_VELOCITY_BUY_TRIGGER_THRESHOLD,
		SELL_BUY_RATIO_THRESHOLD,
		VERBOSE, // verbose from common config
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
		VERBOSE, // verbose from common config
		USE_DYNAMIC_FEES, // use dynamic fees per common config
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
		"  🔻 Buy threshold: {} (Z-score on {})",
		DEFAULT_BUY_THRESHOLD, Z_SCORE_COLUMN
	);
	println!(
		"  🔺 Sell threshold: {} (Z-score on {})",
		DEFAULT_SELL_THRESHOLD, Z_SCORE_COLUMN
	);
	println!("  💰 Position size: {} SOL", POSITION_SIZE_SOL);
	println!("  🚫 Min SOL in curve: {}", MIN_SOL_IN_CURVE);
	println!("  🔢 Max buys per token: {}", MAX_BUYS);
	println!("  🔄 Min blocks between buys: {}", MIN_BLOCKS_BETWEEN_BUYS);
	println!("  ⏳ Min hold blocks: {}", MIN_HOLD_BLOCKS);
	println!("  🔀 Min blocks sell → buy: {}", MIN_BLOCKS_BETWEEN_SELL_BUY);
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
	println!("\n--- Inter-Trade Strategy Debug Statistics ---");
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
		"Z-score calculations performed: {}",
		debug_stats.get("z_score_calculations").unwrap_or(&0)
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
	println!("🚀 Starting Inter-Trade Strategy Testing Suite");
	println!("Enhanced mean reversion with trade velocity context");
	println!();

	// Test default parameters
	test_inter_trade_strategy()?;

	println!("\n✅ Inter-Trade Strategy Test Complete!");

	Ok(())
}

// #[cfg(test)]
// mod tests {
// 	use super::*;

// 	#[test]
// 	fn test_strategy_creation() {
// 		let strategy = InterTradeStrategy::new(
// 			DEFAULT_WINDOW_SIZE,
// 			DEFAULT_BUY_THRESHOLD,
// 			DEFAULT_SELL_THRESHOLD,
// 			MIN_SOL_IN_CURVE,
// 			Z_SCORE_COLUMN,
// 			MAX_BUYS,
// 			MIN_BLOCKS_BETWEEN_BUYS,
// 			MIN_HOLD_BLOCKS,
// 			MIN_BLOCKS_BETWEEN_SELL_BUY,
// 			false, // verbose=false for tests
// 		);

// 		assert_eq!(
// 			strategy.get_strategy_name().contains("Mean Reversion"),
// 			true
// 		);
// 	}

// 	#[test]
// 	fn test_framework_creation() {
// 		let framework = BacktestFramework::new(
// 			POSITION_SIZE_SOL,
// 			BUY_FEE_PERCENT,
// 			SELL_FEE_PERCENT,
// 			FLAT_FEE_SOL_BUY,
// 			FLAT_FEE_SOL_SELL,
// 			BLOCK_LATENCY_MAX,
// 			false, // export_trades = false for tests
// 			false, // verbose = false for tests
// 			USE_DYNAMIC_FEES, // use_dynamic_fees
// 		);

// 		// Basic smoke test - framework should be created without panicking
// 		assert!(true);
// 	}
// }
