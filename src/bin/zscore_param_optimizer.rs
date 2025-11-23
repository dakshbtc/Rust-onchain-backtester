/*!
Mean Reversion Strategy Parameter Optimizer
==========================================

This tool efficiently tests combinations of Mean Reversion strategy parameters:
- Window Size: Rolling window for Z-score calculation
- Buy Threshold: Z-score threshold for buy signals
- Sell Threshold: Z-score threshold for sell signals

Data preprocessing (token manipulation detection, date parsing) runs only once.
Results are grouped by SOL in curve levels, with top 3 performers saved for each group.
*/

use anyhow::Result;
use rayon::prelude::*;
use rust_backtester::{BacktestFramework, MeanReversionStrategy};
use rust_backtester::strategies::common_config::*;
use std::collections::HashMap;
use std::sync::{
	Arc,
	atomic::{AtomicUsize, Ordering},
};

// CSV file path for testing
const CSV_FILE_PATH: &str =
	"../solana-wallet-analytics/trades/pumpfun_08-25-2025_to_08-28-2025.csv";

// MEAN REVERSION OPTIMIZER CONFIGURATION (universal parameters imported from common_config)
// Fee overrides for this optimizer (using different fee structure)
const BUY_FEE_PERCENT_OVERRIDE: f64 = 0.003;
const SELL_FEE_PERCENT_OVERRIDE: f64 = 0.035;
const FLAT_FEE_SOL_BUY_OVERRIDE: f64 = 0.00004;
const FLAT_FEE_SOL_SELL_OVERRIDE: f64 = 0.00004;
const USE_DYNAMIC_FEES_OVERRIDE: bool = false; // Override common config

// Z-SCORE COLUMN CONFIGURATION
const Z_SCORE_COLUMN: &str = "Token Price"; // Column to use for Z-score calculation ("SOL in Curve", "Token Price", or "SOL")
const MAX_BUYS_RANGE: &[usize] = &[1]; // Maximum number of buy positions per token (range to test)

// DCA STRATEGY PARAMETER RANGES TO TEST
const DCA_DRAWDOWN_INTERVALS: &[f64] = &[0.03]; // Drawdown intervals for DCA entries
const BASE_PROFIT_TARGETS: &[f64] = &[0.05, 0.06]; // Base profit targets above average cost
const BASE_STOP_LOSSES: &[f64] = &[0.30, 0.6]; // Base stop losses below average cost
const STOP_LOSS_TIGHTENING_PER_DCAS: &[f64] = &[0.03, 0.1]; // Stop loss tightening per DCA entry
const PROFIT_TARGET_TIGHTENING_PER_DCAS: &[f64] = &[0.02, 0.05]; // Profit target tightening per DCA entry

// SOL LEVEL GROUPING CONFIGURATION
const SOL_GROUP_LEVEL: f64 = 5000.0; // SOL level increment for grouping (e.g., 25.0 groups by 25 SOL increments)

// PARAMETER RANGES TO TEST
// ========================

// Window sizes to test (rolling window for mean reversion)
const WINDOW_SIZES: &[usize] = &[80];

// Buy thresholds to test (Z-score values - more negative = more oversold)
const BUY_THRESHOLDS: &[f64] = &[-2.0];

// Sell thresholds to test (Z-score values - more positive = more overbought)
const SELL_THRESHOLDS: &[f64] = &[0.8];


#[derive(Debug, Clone)]
struct ParameterSet {
	window_size: usize,
	buy_threshold: f64,
	sell_threshold: f64,
	max_buys: usize,
	dca_drawdown_interval: f64,
	base_profit_target: f64,
	base_stop_loss: f64,
	stop_loss_tightening_per_dca: f64,
	profit_target_tightening_per_dca: f64,
}

#[derive(Debug, Clone)]
struct TestResult {
	params: ParameterSet,
	total_pnl: f64,
	total_trades: usize,
	win_rate: f64,
	avg_win: f64,
	avg_loss: f64,
	profit_factor: f64,
	max_drawdown: f64,
	mar_ratio: f64, // Managed Account Ratio (Total Return / Max Drawdown)
	sol_group: String,
	trades_in_group: usize,
}

impl ParameterSet {
	fn new(
		window_size: usize,
		buy_threshold: f64,
		sell_threshold: f64,
		max_buys: usize,
		dca_drawdown_interval: f64,
		base_profit_target: f64,
		base_stop_loss: f64,
		stop_loss_tightening_per_dca: f64,
		profit_target_tightening_per_dca: f64,
	) -> Self {
		Self {
			window_size,
			buy_threshold,
			sell_threshold,
			max_buys,
			dca_drawdown_interval,
			base_profit_target,
			base_stop_loss,
			stop_loss_tightening_per_dca,
			profit_target_tightening_per_dca,
		}
	}
}

fn generate_parameter_combinations() -> Vec<ParameterSet> {
	let mut combinations = Vec::new();

	for &window_size in WINDOW_SIZES {
		for &buy_threshold in BUY_THRESHOLDS {
			for &sell_threshold in SELL_THRESHOLDS {
				for &max_buys in MAX_BUYS_RANGE {
					for &dca_drawdown_interval in DCA_DRAWDOWN_INTERVALS {
						for &base_profit_target in BASE_PROFIT_TARGETS {
							for &base_stop_loss in BASE_STOP_LOSSES {
								for &stop_loss_tightening_per_dca in
									STOP_LOSS_TIGHTENING_PER_DCAS
								{
									for &profit_target_tightening_per_dca in
										PROFIT_TARGET_TIGHTENING_PER_DCAS
									{
										combinations.push(ParameterSet::new(
											window_size,
											buy_threshold,
											sell_threshold,
											max_buys,
											dca_drawdown_interval,
											base_profit_target,
											base_stop_loss,
											stop_loss_tightening_per_dca,
											profit_target_tightening_per_dca,
										));
									}
								}
							}
						}
					}
				}
			}
		}
	}

	println!(
		"📊 Generated {} parameter combinations to test",
		combinations.len()
	);
	println!("   Window sizes: {:?}", WINDOW_SIZES);
	println!(
		"   Buy thresholds: {:?} (Z-score on {})",
		BUY_THRESHOLDS, Z_SCORE_COLUMN
	);
	println!(
		"   Sell thresholds: {:?} (Z-score on {}) - LEGACY",
		SELL_THRESHOLDS, Z_SCORE_COLUMN
	);
	println!("   Max buys per token: {:?}", MAX_BUYS_RANGE);
	println!("   DCA drawdown intervals: {:?}", DCA_DRAWDOWN_INTERVALS);
	println!("   Base profit targets: {:?}", BASE_PROFIT_TARGETS);
	println!("   Base stop losses: {:?}", BASE_STOP_LOSSES);
	println!(
		"   Stop loss tightening per DCA: {:?}",
		STOP_LOSS_TIGHTENING_PER_DCAS
	);
	println!(
		"   Profit target tightening per DCA: {:?}",
		PROFIT_TARGET_TIGHTENING_PER_DCAS
	);

	combinations
}

fn test_parameter_set_with_preprocessed_framework(
	params: &ParameterSet,
	min_sol_in_curve: f64,
	preprocessed_framework: &BacktestFramework,
) -> Result<
	Option<(
		rust_backtester::framework::backtesting_framework::BacktestResults,
		BacktestFramework,
	)>,
> {
	// Create strategy with test parameters (verbose=false for parameter optimization)
	let mut strategy = MeanReversionStrategy::new(
		params.window_size,
		params.buy_threshold,
		params.sell_threshold,
		min_sol_in_curve,
		Z_SCORE_COLUMN,
		params.max_buys,
		MIN_BLOCKS_BETWEEN_BUYS,
		MIN_HOLD_BLOCKS,
		MIN_BLOCKS_BETWEEN_SELL_BUY,
		false, // verbose=false to suppress debug output during optimization
	);

	// Clone the preprocessed framework to avoid data reprocessing
	let mut framework = preprocessed_framework.clone_for_new_test();

	// Run backtest on already preprocessed data
	let results = framework.run_strategy_on_preprocessed_data(&mut strategy)?;

	if results.completed_trades_df.height() > 0 {
		Ok(Some((results, framework)))
	} else {
		Ok(None)
	}
}

/// Calculate per-segment equity curves and maximum drawdown for each SOL level
fn calculate_segment_drawdowns(
	framework: &BacktestFramework,
) -> std::collections::HashMap<String, f64> {
	use std::collections::HashMap;

	// Get completed trades from framework
	let completed_trades = framework.get_completed_trades();

	if completed_trades.is_empty() {
		return HashMap::new();
	}

	// Group trades by SOL level (same SOL_GROUP_LEVEL SOL grouping logic as analysis)
	let mut sol_level_groups: HashMap<
		String,
		Vec<&rust_backtester::framework::backtesting_framework::TradeSummary>,
	> = HashMap::new();

	for trade in completed_trades {
		// Use same SOL level grouping logic: round to nearest SOL_GROUP_LEVEL
		let sol_level = ((trade.sol_in_curve_before_our_buy / SOL_GROUP_LEVEL)
			.round() as i32)
			* (SOL_GROUP_LEVEL as i32);
		let sol_group_name = format!("SOL_{:04}", sol_level);

		sol_level_groups
			.entry(sol_group_name)
			.or_insert_with(Vec::new)
			.push(trade);
	}

	// Calculate drawdown for each SOL level group
	let mut segment_drawdowns: HashMap<String, f64> = HashMap::new();

	for (sol_group_name, mut trades) in sol_level_groups {
		// Sort trades chronologically by buy_tx_index (should already be in order, but ensure)
		trades.sort_by_key(|trade| trade.buy_tx_index);

		// Calculate equity curve and maximum drawdown for this segment
		let mut running_balance = 0.0;
		let mut peak_balance = 0.0;
		let mut max_drawdown = 0.0;

		for trade in trades {
			running_balance += trade.pnl_sol;

			// Track new peak
			if running_balance > peak_balance {
				peak_balance = running_balance;
			}

			// Calculate current drawdown
			let current_drawdown = peak_balance - running_balance;

			// Track maximum drawdown
			if current_drawdown > max_drawdown {
				max_drawdown = current_drawdown;
			}
		}

		segment_drawdowns.insert(sol_group_name, max_drawdown);
	}

	segment_drawdowns
}

fn analyze_results_by_sol_groups(
	params: &ParameterSet,
	_results: &rust_backtester::framework::backtesting_framework::BacktestResults,
	framework: &BacktestFramework,
) -> Result<Vec<TestResult>> {
	let mut group_results = Vec::new();

	// Calculate per-segment drawdowns first
	let segment_drawdowns = calculate_segment_drawdowns(framework);

	// Use the existing SOL level analysis from the framework
	let sol_level_data =
		framework.get_sol_in_curve_analysis_data(SOL_GROUP_LEVEL);

	if sol_level_data.is_empty() {
		println!("    ⚠️  No SOL level data available");
		return Ok(group_results);
	}

	// Convert each SOL level analysis to a TestResult for parameter optimization
	for (sol_level, analysis) in sol_level_data {
		let sol_group_name = format!("SOL_{:04}", sol_level);

		// Get actual max drawdown for this SOL level from our calculation
		let group_max_drawdown = segment_drawdowns
			.get(&sol_group_name)
			.cloned()
			.unwrap_or(0.0);

		// Calculate MAR ratio (Managed Account Ratio): Total Return / Max Drawdown
		let mar_ratio = if group_max_drawdown > 1e-9 {
			analysis.total_pnl / group_max_drawdown
		} else if analysis.total_pnl > 0.0 {
			// If no drawdown but positive returns, use a very high MAR ratio
			f64::INFINITY
		} else if analysis.total_pnl == 0.0 {
			// If no returns and no drawdown, MAR is 0
			0.0
		} else {
			// Negative returns with no drawdown (shouldn't happen, but handle gracefully)
			f64::NEG_INFINITY
		};

		let test_result = TestResult {
			params: params.clone(),
			total_pnl: analysis.total_pnl,
			total_trades: analysis.trade_count,
			win_rate: analysis.win_percentage,
			avg_win: analysis.avg_win,
			avg_loss: analysis.avg_loss,
			profit_factor: analysis.profit_factor,
			max_drawdown: group_max_drawdown,
			mar_ratio,
			sol_group: sol_group_name,
			trades_in_group: analysis.trade_count,
		};

		group_results.push(test_result);
	}

	// Only print debug messages if framework is in verbose mode (not during parameter optimization)
	if framework.is_verbose() {
		println!(
			"    ✅ Created {} SOL level groups for analysis",
			group_results.len()
		);
	}

	Ok(group_results)
}

fn find_top_performers_by_group(
	all_results: Vec<TestResult>,
) -> (HashMap<String, Vec<TestResult>>, HashMap<String, Vec<TestResult>>) {
	let mut grouped_results: HashMap<String, Vec<TestResult>> = HashMap::new();

	// Group results by SOL curve group
	for result in all_results {
		grouped_results
			.entry(result.sol_group.clone())
			.or_insert_with(Vec::new)
			.push(result);
	}

	let mut mar_sorted_groups: HashMap<String, Vec<TestResult>> = HashMap::new();
	let mut pnl_sorted_groups: HashMap<String, Vec<TestResult>> = HashMap::new();

	// Sort each group by both MAR ratio and P/L (descending) - prioritizing risk-adjusted returns
	for (group_name, results) in grouped_results.iter_mut() {
		// Clone results for P/L sorting (we need both sortings)
		let mut pnl_sorted_results = results.clone();

		// Sort original by MAR ratio
		results.sort_by(|a, b| {
			// Handle infinite MAR ratios (perfect strategies with no drawdown)
			match (a.mar_ratio.is_finite(), b.mar_ratio.is_finite()) {
				(true, true) => b
					.mar_ratio
					.partial_cmp(&a.mar_ratio)
					.unwrap_or(std::cmp::Ordering::Equal),
				(false, true) => {
					if a.mar_ratio.is_sign_positive() {
						std::cmp::Ordering::Less
					} else {
						std::cmp::Ordering::Greater
					}
				}
				(true, false) => {
					if b.mar_ratio.is_sign_positive() {
						std::cmp::Ordering::Greater
					} else {
						std::cmp::Ordering::Less
					}
				}
				(false, false) => {
					// Both infinite - compare by sign, then by total PnL as tiebreaker
					match (
						a.mar_ratio.is_sign_positive(),
						b.mar_ratio.is_sign_positive(),
					) {
						(true, true) => b
							.total_pnl
							.partial_cmp(&a.total_pnl)
							.unwrap_or(std::cmp::Ordering::Equal), // Both +inf
						(false, false) => a
							.total_pnl
							.partial_cmp(&b.total_pnl)
							.unwrap_or(std::cmp::Ordering::Equal), // Both -inf
						(true, false) => std::cmp::Ordering::Less, // +inf beats -inf
						(false, true) => std::cmp::Ordering::Greater, // -inf loses to +inf
					}
				}
			}
		});

		// Sort clone by total P/L (descending)
		pnl_sorted_results.sort_by(|a, b| {
			b.total_pnl
				.partial_cmp(&a.total_pnl)
				.unwrap_or(std::cmp::Ordering::Equal)
		});

		// Print TOP 3 by MAR RATIO
		println!("\n🏆 TOP 3 PERFORMERS (by MAR Ratio) for {}:", group_name);
		for (i, result) in results.iter().take(3).enumerate() {
			let mar_display = if result.mar_ratio.is_finite() {
				format!("{:.2}", result.mar_ratio)
			} else if result.mar_ratio.is_sign_positive() {
				"∞".to_string()
			} else {
				"-∞".to_string()
			};

			println!(
				"  {}. Window: {}, Buy: {:.1}, Sell: {:.1}, MaxBuys: {}, DCA: {:.1}%, PT: {:.1}%, SL: {:.1}%, SLT: {:.1}%, PTT: {:.1}% → PnL: {:+.4} SOL, MAR: {} ({} trades, {:.1}% win rate)",
				i + 1,
				result.params.window_size,
				result.params.buy_threshold,
				result.params.sell_threshold,
				result.params.max_buys,
				result.params.dca_drawdown_interval * 100.0,
				result.params.base_profit_target * 100.0,
				result.params.base_stop_loss * 100.0,
				result.params.stop_loss_tightening_per_dca * 100.0,
				result.params.profit_target_tightening_per_dca * 100.0,
				result.total_pnl,
				mar_display,
				result.total_trades,
				result.win_rate
			);
		}

		// Print TOP 3 by P/L
		println!("\n💰 TOP 3 PERFORMERS (by Total P/L) for {}:", group_name);
		for (i, result) in pnl_sorted_results.iter().take(3).enumerate() {
			let mar_display = if result.mar_ratio.is_finite() {
				format!("{:.2}", result.mar_ratio)
			} else if result.mar_ratio.is_sign_positive() {
				"∞".to_string()
			} else {
				"-∞".to_string()
			};

			println!(
				"  {}. Window: {}, Buy: {:.1}, Sell: {:.1}, MaxBuys: {}, DCA: {:.1}%, PT: {:.1}%, SL: {:.1}%, SLT: {:.1}%, PTT: {:.1}% → PnL: {:+.4} SOL, MAR: {} ({} trades, {:.1}% win rate)",
				i + 1,
				result.params.window_size,
				result.params.buy_threshold,
				result.params.sell_threshold,
				result.params.max_buys,
				result.params.dca_drawdown_interval * 100.0,
				result.params.base_profit_target * 100.0,
				result.params.base_stop_loss * 100.0,
				result.params.stop_loss_tightening_per_dca * 100.0,
				result.params.profit_target_tightening_per_dca * 100.0,
				result.total_pnl,
				mar_display,
				result.total_trades,
				result.win_rate
			);
		}

		// Keep top 3 for both MAR ratio and P/L for CSV export
		let mar_top_3 = results.iter().take(3).cloned().collect();
		let pnl_top_3 = pnl_sorted_results.iter().take(3).cloned().collect();
		
		mar_sorted_groups.insert(group_name.clone(), mar_top_3);
		pnl_sorted_groups.insert(group_name.clone(), pnl_top_3);
	}

	(mar_sorted_groups, pnl_sorted_groups)
}

fn export_optimization_results(
	mar_top_performers: &HashMap<String, Vec<TestResult>>,
	pnl_top_performers: &HashMap<String, Vec<TestResult>>,
) -> Result<()> {
	use std::fs;
	use std::io::Write;

	// Create results directory
	let home_dir = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
	let results_dir = format!(
		"{}/Documents/github/rust-backtester/optimization_results",
		home_dir
	);
	fs::create_dir_all(&results_dir)?;

	// Helper function to export results to CSV
	let export_to_csv = |top_performers: &HashMap<String, Vec<TestResult>>, file_suffix: &str, description: &str| -> Result<String> {
		// Sort SOL groups by SOL level (ascending)
		let mut sorted_groups: Vec<(i32, String, &Vec<TestResult>)> =
			top_performers
				.iter()
				.map(|(group_name, results)| {
					// Extract SOL level from group name (e.g., "SOL_1450" -> 1450)
					let sol_level = group_name
						.strip_prefix("SOL_")
						.unwrap_or("0")
						.parse::<i32>()
						.unwrap_or(0);
					(sol_level, group_name.clone(), results)
				})
				.collect();

		// Sort by SOL level ascending (lowest first)
		sorted_groups.sort_by(|a, b| a.0.cmp(&b.0));

		// Export detailed results to CSV
		let mut csv_content = String::new();
		csv_content.push_str("SOL_Level,Rank,Window_Size,Buy_Threshold,Sell_Threshold,Max_Buys,DCA_Drawdown_Interval,Base_Profit_Target,Base_Stop_Loss,Stop_Loss_Tightening_Per_DCA,Profit_Target_Tightening_Per_DCA,Total_PnL,Total_Trades,Win_Rate,Avg_Win,Avg_Loss,Profit_Factor,Max_Drawdown,MAR_Ratio,Trades_In_Group\n");

		for (i, (sol_level, _group_name, results)) in
			sorted_groups.iter().enumerate()
		{
			// Add blank row between groups (except for first group)
			if i > 0 {
				csv_content.push_str("\n");
			}

			for (rank, result) in results.iter().enumerate() {
				// Handle infinite MAR ratios for CSV export
				let mar_csv = if result.mar_ratio.is_finite() {
					format!("{:.4}", result.mar_ratio)
				} else if result.mar_ratio.is_sign_positive() {
					"INF".to_string()
				} else {
					"-INF".to_string()
				};

				csv_content.push_str(&format!(
					"{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.6},{},{:.2},{:.6},{:.6},{:.2},{:.6},{},{}\n",
					sol_level,
					rank + 1,
					result.params.window_size,
					result.params.buy_threshold,
					result.params.sell_threshold,
					result.params.max_buys,
					result.params.dca_drawdown_interval,
					result.params.base_profit_target,
					result.params.base_stop_loss,
					result.params.stop_loss_tightening_per_dca,
					result.params.profit_target_tightening_per_dca,
					result.total_pnl,
					result.total_trades,
					result.win_rate,
					result.avg_win,
					result.avg_loss,
					result.profit_factor,
					result.max_drawdown,
					mar_csv,
					result.trades_in_group
				));
			}
		}

		let filename = format!("{}/mean_reversion_top_performers_{}.csv", results_dir, file_suffix);
		let mut file = std::fs::File::create(&filename)?;
		file.write_all(csv_content.as_bytes())?;

		println!("\n📁 Exported {} results to: {}", description, filename);
		println!(
			"   📊 {} SOL levels with top 3 performers per group",
			sorted_groups.len()
		);

		Ok(filename)
	};

	// Export MAR ratio top performers
	export_to_csv(mar_top_performers, "mar_ratio", "MAR Ratio top performers")?;
	
	// Export P/L top performers
	export_to_csv(pnl_top_performers, "pnl", "P/L top performers")?;

	Ok(())
}

fn run_parameter_optimization() -> Result<()> {
	println!("{}", "=".repeat(100));
	println!("🧪 MEAN REVERSION PARAMETER OPTIMIZATION");
	println!("{}", "=".repeat(100));
	println!(
		"📊 Testing {} parameter combinations",
		WINDOW_SIZES.len()
			* BUY_THRESHOLDS.len()
			* SELL_THRESHOLDS.len()
			* MAX_BUYS_RANGE.len()
			* DCA_DRAWDOWN_INTERVALS.len()
			* BASE_PROFIT_TARGETS.len()
			* BASE_STOP_LOSSES.len()
			* STOP_LOSS_TIGHTENING_PER_DCAS.len()
			* PROFIT_TARGET_TIGHTENING_PER_DCAS.len()
	);
	println!("🎯 Data preprocessing will run ONLY ONCE at the beginning");
	println!(
		"📈 Results grouped by actual SOL in curve levels ({} SOL increments)",
		SOL_GROUP_LEVEL
	);
	println!(
		"📊 OPTIMIZATION CRITERION: MAR Ratio (Managed Account Ratio = Total Return / Max Drawdown)"
	);
	println!("🔢 Z-score calculation using: {}", Z_SCORE_COLUMN);

	println!("\n📊 SOL Level Analysis:");
	println!(
		"  Uses same {}-SOL grouping as main analysis (0, {}, {}, {}, {}, {}, ...)",
		SOL_GROUP_LEVEL as i32,
		SOL_GROUP_LEVEL as i32,
		(SOL_GROUP_LEVEL * 2.0) as i32,
		(SOL_GROUP_LEVEL * 3.0) as i32,
		(SOL_GROUP_LEVEL * 4.0) as i32,
		(SOL_GROUP_LEVEL * 5.0) as i32
	);
	println!();

	// STEP 1: Preprocess data once (this is the expensive part)
	println!(
		"🔄 STEP 1: Preprocessing data (load CSV, detect manipulation, validate)..."
	);
	let mut master_framework = BacktestFramework::new(
		POSITION_SIZE_SOL,
		BUY_FEE_PERCENT_OVERRIDE,
		SELL_FEE_PERCENT_OVERRIDE,
		FLAT_FEE_SOL_BUY_OVERRIDE,
		FLAT_FEE_SOL_SELL_OVERRIDE,
		BLOCK_LATENCY_MAX,
		false, // export_trades = false for parameter optimization (prevents file explosion)
		false, // verbose = false for parameter optimization (prevents debug spam)
		USE_DYNAMIC_FEES_OVERRIDE,
	);

	master_framework.preprocess_data(
		CSV_FILE_PATH,
		VALIDATE_DATA,
		TOLERANCE_PERCENT,
		DETECT_MANIPULATION,
		MANIPULATION_THRESHOLD,
	)?;

	println!(
		"✅ Data preprocessing complete! Now testing {} parameter combinations efficiently...\n",
		WINDOW_SIZES.len()
			* BUY_THRESHOLDS.len()
			* SELL_THRESHOLDS.len()
			* MAX_BUYS_RANGE.len()
			* DCA_DRAWDOWN_INTERVALS.len()
			* BASE_PROFIT_TARGETS.len()
			* BASE_STOP_LOSSES.len()
			* STOP_LOSS_TIGHTENING_PER_DCAS.len()
			* PROFIT_TARGET_TIGHTENING_PER_DCAS.len()
	);

	// Wrap in Arc for safe sharing across threads
	let master_framework = Arc::new(master_framework);

	// STEP 2: Test each parameter combination on preprocessed data (IN PARALLEL)
	let parameter_combinations = generate_parameter_combinations();
	println!(
		"🚀 Running {} tests in parallel across all CPU cores...",
		parameter_combinations.len()
	);

	// Use a reasonable default for min_sol_in_curve during testing
	let min_sol_in_curve = 85.0;

	// Create atomic counter for progress tracking (no synchronization overhead)
	let completed_counter = Arc::new(AtomicUsize::new(0));
	let total_combinations = parameter_combinations.len();

	println!(
		"🚀 Starting {} parameter combinations...",
		total_combinations
	);

	// Run parameter tests in parallel and collect successful results
	let all_results: Vec<TestResult> = parameter_combinations
		.par_iter()
		.filter_map(|params| {
			let result = match test_parameter_set_with_preprocessed_framework(
				params,
				min_sol_in_curve,
				&master_framework,
			) {
				Ok(Some((results, framework))) => {
					// Analyze results by SOL groups
					match analyze_results_by_sol_groups(
						params, &results, &framework,
					) {
						Ok(group_results) => Some(group_results),
						Err(_) => None,
					}
				}
				Ok(None) => None,
				Err(_) => None,
			};

			// Only increment counter AFTER the actual work is completed
			let completed =
				completed_counter.fetch_add(1, Ordering::Relaxed) + 1;
			if completed % 5 == 0 || completed == total_combinations {
				println!(
					"📊 Completed {}/{} combinations ({:.1}%) - Result: {}",
					completed,
					total_combinations,
					(completed as f64 / total_combinations as f64) * 100.0,
					if result.is_some() {
						"SUCCESS"
					} else {
						"NO TRADES"
					}
				);
			}

			result
		})
		.flatten()
		.collect();

	println!(
		"✅ All {} parameter combinations completed!",
		total_combinations
	);

	let successful_tests = all_results.len()
		/ if all_results.is_empty() {
			1
		} else {
			all_results
				.iter()
				.map(|r| &r.sol_group)
				.collect::<std::collections::HashSet<_>>()
				.len()
		};

	println!("\n{}", "=".repeat(100));
	println!("📊 OPTIMIZATION RESULTS SUMMARY");
	println!("{}", "=".repeat(100));

	if all_results.is_empty() {
		println!("❌ No successful test results to analyze");
		return Ok(());
	}

	println!(
		"🎯 Found {} total group results from {} successful parameter tests",
		all_results.len(),
		successful_tests
	);

	// Print distribution of results by SOL group
	let mut group_counts: HashMap<String, usize> = HashMap::new();
	for result in &all_results {
		*group_counts.entry(result.sol_group.clone()).or_insert(0) += 1;
	}

	println!("\n📊 Results distribution by SOL group:");
	for (group_name, count) in &group_counts {
		println!("  {}: {} parameter combinations tested", group_name, count);
	}

	// Find top performers by group
	let (mar_top_performers, pnl_top_performers) = find_top_performers_by_group(all_results);

	// Print summary of top performers (using MAR ratio sorted results)
	println!("\n🏆 SUMMARY OF TOP PERFORMERS:");
	for (group_name, performers) in &mar_top_performers {
		if !performers.is_empty() {
			let best = &performers[0];
			let mar_display = if best.mar_ratio.is_finite() {
				format!("{:.2}", best.mar_ratio)
			} else if best.mar_ratio.is_sign_positive() {
				"∞".to_string()
			} else {
				"-∞".to_string()
			};

			println!(
				"  {} Best: PnL {:+.4} SOL, MAR {}, Win Rate {:.1}% ({} trades)",
				group_name,
				best.total_pnl,
				mar_display,
				best.win_rate,
				best.total_trades
			);
			println!(
				"    Params: Window: {}, Buy: {:.1}, Sell: {:.1}, MaxBuys: {}, DCA: {:.1}%, PT: {:.1}%, SL: {:.1}%, SLT: {:.1}%, PTT: {:.1}%",
				best.params.window_size,
				best.params.buy_threshold,
				best.params.sell_threshold,
				best.params.max_buys,
				best.params.dca_drawdown_interval * 100.0,
				best.params.base_profit_target * 100.0,
				best.params.base_stop_loss * 100.0,
				best.params.stop_loss_tightening_per_dca * 100.0,
				best.params.profit_target_tightening_per_dca * 100.0
			);
		}
	}

	// Export results - both MAR ratio and P/L top performers
	export_optimization_results(&mar_top_performers, &pnl_top_performers)?;

	println!("\n✅ Parameter optimization complete!");
	println!(
		"🦀 Best parameter combinations identified for each SOL group (by both MAR Ratio and P/L)"
	);

	Ok(())
}

fn main() -> Result<()> {
	println!("🚀 Starting Mean Reversion Parameter Optimization");
	println!("Based on efficient testing of parameter combinations");
	println!("🔢 Z-score calculation column: {}", Z_SCORE_COLUMN);
	println!();

	run_parameter_optimization()?;

	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_parameter_generation() {
		let combinations = generate_parameter_combinations();
		let expected_count =
			WINDOW_SIZES.len() * BUY_THRESHOLDS.len() * SELL_THRESHOLDS.len();
		assert_eq!(combinations.len(), expected_count);
	}

	#[test]
	fn test_parameter_set_creation() {
		let params =
			ParameterSet::new(200, -2.5, 0.5, 1, 0.03, 0.05, 0.30, 0.03, 0.02);
		assert_eq!(params.window_size, 200);
		assert_eq!(params.buy_threshold, -2.5);
		assert_eq!(params.sell_threshold, 0.5);
	}
}
