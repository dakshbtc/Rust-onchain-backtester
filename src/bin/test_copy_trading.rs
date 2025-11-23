/*!
Copy Trading Strategy Test Suite
================================

This test file contains all configurable parameters for testing the Copy Trading Strategy.
All trading fees, position sizes, strategy parameters, and validation settings are configured here.

To modify the strategy behavior, adjust the parameters in the TRADING CONFIGURATION section below.
*/

use anyhow::Result;
use rust_backtester::{
	BacktestFramework, CopyTradingStrategy, TradingStrategy,
};
use rust_backtester::strategies::common_config::*;

// CSV file path for testing
const CSV_FILE_PATH: &str =
	"../solana-wallet-analytics/trades/STorre_20250916_160302.csv";

// TRADING CONFIGURATION PARAMETERS
// =================================

// Copy Trading Strategy Configuration (universal parameters imported from common_config)
const TARGET_WALLET: &str = "STorreSu8X6yPLmiEScNHSGDinqV7j84hWjmvH9SPwk"; // Wallet to copy

// Copy Trading Strategy Overrides (different from common config defaults)
const MIN_SOL_IN_CURVE_OVERRIDE: f64 = 0.1; // Lower threshold for copy trading
const VERBOSE_OVERRIDE: bool = false; // Less verbose output for copy trading
const DETECT_MANIPULATION_OVERRIDE: bool = false; // No manipulation detection for copy trading
const USE_DYNAMIC_FEES_OVERRIDE: bool = false; // Fixed fees for copy trading

fn test_copy_trading_strategy() -> Result<()> {
	println!("{}", "=".repeat(80));
	println!("🎯 Testing Copy Trading Strategy");
	println!("{}", "=".repeat(80));

	// Create strategy instance with configured parameters (verbose=true for single test)
	let mut strategy = CopyTradingStrategy::new(
		TARGET_WALLET,
		MIN_SOL_IN_CURVE_OVERRIDE,
		BUY_FEE_PERCENT,
		SELL_FEE_PERCENT,
		FLAT_FEE_SOL_BUY,
		FLAT_FEE_SOL_SELL,
		1, // max_buys - copy trading typically does 1:1 copying
		MIN_BLOCKS_BETWEEN_BUYS,
		MIN_HOLD_BLOCKS,
		MIN_BLOCKS_BETWEEN_SELL_BUY,
		VERBOSE_OVERRIDE, // verbose=false for copy trading
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
		VERBOSE_OVERRIDE, // verbose = false for copy trading
		USE_DYNAMIC_FEES_OVERRIDE,
	);

	// Preprocess data once (same approach as optimizer)
	println!(
		"🔄 Preprocessing data (load CSV, detect manipulation, validate)..."
	);
	framework.preprocess_data(
		CSV_FILE_PATH,
		VALIDATE_DATA,
		TOLERANCE_PERCENT,
		DETECT_MANIPULATION_OVERRIDE,
		MANIPULATION_THRESHOLD,
	)?;

	// Run strategy on preprocessed data
	println!("Running strategy with:");
	println!("  📊 CSV file: {}", CSV_FILE_PATH);
	println!("  🎯 Target wallet: {}...{}", &TARGET_WALLET[..8], &TARGET_WALLET[TARGET_WALLET.len()-8..]);
	println!("  💰 Position size: {} SOL", POSITION_SIZE_SOL);
	println!("  🚫 Min SOL in curve: {}", MIN_SOL_IN_CURVE_OVERRIDE);
		if USE_DYNAMIC_FEES_OVERRIDE {
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
		"  🎯 Strategy: Copy trades from target wallet"
	);
	println!();

	let results = framework.run_strategy_on_preprocessed_data(&mut strategy)?;

	// Print results
	if results.completed_trades_df.height() > 0 {
		// Print SOL in curve trigger analysis first (using default 25.0 SOL grouping)
		framework.print_sol_in_curve_analysis(25.0);
		framework.print_market_cap_analysis(10.0);

		// Then print main results summary right below it
		framework
			.print_results_summary(&results, &strategy.get_strategy_name());
	} else {
		println!("❌ Backtest failed - no results returned");
		return Ok(());
	}

	// Print strategy-specific debug stats
	let debug_stats = strategy.get_debug_stats();
	println!("\n--- Copy Trading Strategy Debug Statistics ---");
	println!(
		"Target wallet buy signals detected: {}",
		debug_stats.get("target_wallet_buys").unwrap_or(&0)
	);
	println!(
		"Target wallet sell signals detected: {}",
		debug_stats.get("target_wallet_sells").unwrap_or(&0)
	);
	println!(
		"Total buy signals executed: {}",
		debug_stats.get("total_buy_signals").unwrap_or(&0)
	);
	println!(
		"Total sell signals executed: {}",
		debug_stats.get("total_sell_signals").unwrap_or(&0)
	);
	println!(
		"SOL curve rejections (< {}): {}",
		MIN_SOL_IN_CURVE_OVERRIDE,
		debug_stats.get("sol_curve_rejections").unwrap_or(&0)
	);
	println!(
		"Open positions: {}",
		debug_stats.get("open_positions").unwrap_or(&0)
	);
	println!(
		"Target wallet open positions: {}",
		debug_stats.get("target_wallet_open_positions").unwrap_or(&0)
	);

	// Display target wallet P/L
	let target_wallet_trades = strategy.get_target_wallet_completed_trades();
	framework.print_sol_in_curve_analysis_for_trades(
		target_wallet_trades,
		25.0, // Same sol group level as main analysis
		"Target Wallet P/L Analysis",
	);
	println!(
		"Target wallet total P/L: {:.4} SOL (using same fee structure)",
		strategy.get_target_wallet_pnl()
	);
	println!(
		"Target wallet total P/L (no fees): {:.4} SOL",
		strategy.get_target_wallet_pnl_no_fees()
	);
	println!(
		"Target wallet remaining open positions: {}",
		strategy.get_target_wallet_open_positions()
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
	println!("🚀 Starting Copy Trading Strategy Testing Suite");
	println!("Mirror trades from a specific wallet address");
	println!();

	// Test default parameters
	test_copy_trading_strategy()?;

	println!("\n✅ Copy Trading Strategy Test Complete!");

	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_strategy_creation() {
		let strategy = CopyTradingStrategy::new(
			TARGET_WALLET,
			MIN_SOL_IN_CURVE_OVERRIDE,
			BUY_FEE_PERCENT,
			SELL_FEE_PERCENT,
			FLAT_FEE_SOL_BUY,
			FLAT_FEE_SOL_SELL,
			1, // max_buys - copy trading typically does 1:1 copying
			MIN_BLOCKS_BETWEEN_BUYS,
			MIN_HOLD_BLOCKS,
			MIN_BLOCKS_BETWEEN_SELL_BUY,
			false, // verbose=false for tests
		);

		assert_eq!(
			strategy.get_strategy_name().contains("Copy Trading"),
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
