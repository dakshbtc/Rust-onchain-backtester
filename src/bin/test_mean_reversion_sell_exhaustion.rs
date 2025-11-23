use anyhow::Result;
use rust_backtester::{BacktestFramework, TradingStrategy};
use rust_backtester::strategies::MeanReversionSellExhaustionStrategy;
use rust_backtester::strategies::mean_reversion_sell_exhaustion::ExhaustionMethod;
use rust_backtester::strategies::common_config::*;

// CSV file path for testing
const CSV_FILE_PATH: &str =
	"/home/daksh/Desktop/pumpfun_10-03-2025_to_10-07-2025.csv/pumpfun_10-03-2025_to_10-07-2025.csv";

// Conservative defaults (higher quality signals)
const SELL_RATE_WINDOW: usize = 20; // trades
const SURGE_ZSCORE_THRESHOLD: f64 = 2.5; // standard deviations
const SURGE_MIN_DURATION_TRADES: usize = 10; // trades
const EXHAUSTION_METHOD: ExhaustionMethod = ExhaustionMethod::Combined;
const EXHAUSTION_PCT_THRESHOLD: f64 = -0.35; // -35%
const EXHAUSTION_CUSUM_THRESHOLD: f64 = 4.0;
const EXHAUSTION_CUSUM_DRIFT: f64 = 0.5;
const MIN_SELL_TRADES_IN_SURGE: usize = 15;
const COOLDOWN_AFTER_ENTRY: usize = 100; // trades
const USE_VOLUME_WEIGHTING: bool = true;

fn test_mean_reversion_sell_exhaustion() -> Result<()> {
	println!("{}", "=".repeat(80));
	println!("📊 Testing Mean Reversion Sell Exhaustion Strategy");
	println!("{}", "=".repeat(80));
	println!("Detects sell surge, peak, and exhaustion to enter long with cooldown.");
	println!("{}", "=".repeat(80));

	let mut strategy = MeanReversionSellExhaustionStrategy::new(
		SELL_RATE_WINDOW,
		SURGE_ZSCORE_THRESHOLD,
		SURGE_MIN_DURATION_TRADES,
		EXHAUSTION_METHOD,
		EXHAUSTION_PCT_THRESHOLD,
		EXHAUSTION_CUSUM_THRESHOLD,
		EXHAUSTION_CUSUM_DRIFT,
		MIN_SELL_TRADES_IN_SURGE,
		COOLDOWN_AFTER_ENTRY,
		USE_VOLUME_WEIGHTING,
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

	println!("Running Mean Reversion Sell Exhaustion Strategy with:");
	println!("  📊 CSV file: {}", CSV_FILE_PATH);
	println!("  🧮 Sell rate window (trades): {}", SELL_RATE_WINDOW);
	println!("  ⚠️ Surge z-score threshold: {:.2}", SURGE_ZSCORE_THRESHOLD);
	println!("  ⏱️ Surge min duration (trades): {}", SURGE_MIN_DURATION_TRADES);
	println!("  ✅ Exhaustion method: {:?}", EXHAUSTION_METHOD);
	println!("  📉 Pct decline threshold: {:.0}%", EXHAUSTION_PCT_THRESHOLD * 100.0);
	println!("  📈 CUSUM threshold/drift: {:.2} / {:.2}", EXHAUSTION_CUSUM_THRESHOLD, EXHAUSTION_CUSUM_DRIFT);
	println!("  🧪 Min sells in surge: {}", MIN_SELL_TRADES_IN_SURGE);
	println!("  🔄 Cooldown (trades): {}", COOLDOWN_AFTER_ENTRY);
	println!("  🧪 Volume weighting: {}", USE_VOLUME_WEIGHTING);

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
	println!("🚀 Starting Mean Reversion Sell Exhaustion Strategy Test");
	test_mean_reversion_sell_exhaustion()?;
	println!("\n✅ Mean Reversion Sell Exhaustion Test Complete!");
	Ok(())
}


