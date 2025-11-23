/*!
VWAP Strategy Parameter Optimizer
=================================

This tool efficiently tests combinations of VWAP strategy parameters:
- Window Size: Rolling window for VWAP calculation
- Buy Threshold: Percentage below VWAP threshold for buy signals
- Sell Threshold: Percentage above VWAP threshold for sell signals

Data preprocessing (token manipulation detection, date parsing) runs only once.
Results are grouped by SOL in curve levels, with top 20 performers saved for each group.
*/

use anyhow::Result;
use rayon::prelude::*;
use rust_backtester::{BacktestFramework, VwapStrategy};
use rust_backtester::strategies::common_config::*;
use std::collections::HashMap;
use std::sync::{
	Arc,
	atomic::{AtomicUsize, Ordering},
};

// CSV file path for testing
const CSV_FILE_PATH: &str =
	"../solana-wallet-analytics/trades/pumpfun_09-08-2025_to_09-11-2025.csv";

// VWAP OPTIMIZER CONFIGURATION (universal parameters imported from common_config)
const PRICE_COLUMN: &str = "Token Price"; // Column to use for price in VWAP calculation
const VOLUME_COLUMN: &str = "Token Amount"; // Column to use for volume in VWAP calculation
const MAX_BUYS_RANGE: &[usize] = &[1]; // Maximum number of buy positions per token (range to test)

// SOL LEVEL GROUPING CONFIGURATION
const SOL_GROUP_LEVEL: f64 = 10000.0; // SOL level increment for grouping (e.g., 25.0 groups by 25 SOL increments)

// DYNAMIC THRESHOLD CONFIGURATION
const USE_DYNAMIC_THRESHOLDS: bool = false; // Use dynamic thresholds based on SOL in curve (true) or static parameter ranges (false)

// Window sizes to test (rolling window for VWAP calculation)
const WINDOW_SIZES: &[usize] = &[300];
const BUY_THRESHOLDS: &[f64] = &[-0.20];
const SELL_THRESHOLDS: &[f64] = &[0.10];
const STOP_LOSS_THRESHOLDS: &[f64] = &[1111.0];
const MAX_HOLD_TIME_MINUTES: &[i64] = &[130]; // Various time limits

// TRADE VELOCITY CONFIGURATION
const USE_TRADE_VELOCITY: bool = true; // Use trade velocity for threshold adjustment
// Trade velocity parameters to test
const VELOCITY_WINDOW_SECONDS: &[f64] = &[15.0, 30.0, 60.0]; // Time windows for velocity calculation (seconds)
const HIGH_VELOCITY_THRESHOLDS: &[f64] = &[1.0, 2.0, 3.0]; // High velocity thresholds (trades/sec)
const LOW_VELOCITY_THRESHOLDS: &[f64] = &[0.1, 0.2, 0.5]; // Low velocity thresholds (trades/sec)
const HIGH_VELOCITY_MULTIPLIERS: &[f64] = &[0.6, 0.8, 0.9]; // High velocity multipliers (tighter thresholds)
const LOW_VELOCITY_MULTIPLIERS: &[f64] = &[1.1, 1.2, 1.4]; // Low velocity multipliers (wider thresholds)

// Note: SOL level grouping now uses the same SOL_GROUP_LEVEL-SOL level grouping as the main analysis
// This provides detailed breakdowns like: SOL_0000, SOL_0025, SOL_0050, SOL_0075, etc. (when SOL_GROUP_LEVEL=25)

#[derive(Debug, Clone)]
struct ParameterSet {
	window_size: usize,
	buy_threshold: f64,
	sell_threshold: f64,
	stop_loss_threshold: f64,
	max_hold_time_minutes: i64,
	max_buys: usize,
	min_blocks_between_buys: i64,
	min_hold_blocks: i64,
	min_blocks_between_sell_buy: i64,
	price_column: String,
	volume_column: String,
	slippage_tolerance_percent: f64,
	use_trade_velocity: bool,
	velocity_window_seconds: f64,
	high_velocity_threshold: f64,
	low_velocity_threshold: f64,
	high_velocity_multiplier: f64,
	low_velocity_multiplier: f64,
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
		stop_loss_threshold: f64,
		max_hold_time_minutes: i64,
		max_buys: usize,
		min_blocks_between_buys: i64,
		min_hold_blocks: i64,
		min_blocks_between_sell_buy: i64,
		price_column: String,
		volume_column: String,
		slippage_tolerance_percent: f64,
		use_trade_velocity: bool,
		velocity_window_seconds: f64,
		high_velocity_threshold: f64,
		low_velocity_threshold: f64,
		high_velocity_multiplier: f64,
		low_velocity_multiplier: f64,
	) -> Self {
		Self {
			window_size,
			buy_threshold,
			sell_threshold,
			stop_loss_threshold,
			max_hold_time_minutes,
			max_buys,
			min_blocks_between_buys,
			min_hold_blocks,
			min_blocks_between_sell_buy,
			price_column,
			volume_column,
			slippage_tolerance_percent,
			use_trade_velocity,
			velocity_window_seconds,
			high_velocity_threshold,
			low_velocity_threshold,
			high_velocity_multiplier,
			low_velocity_multiplier,
		}
	}
}

fn generate_parameter_combinations() -> Vec<ParameterSet> {
	let mut combinations = Vec::new();

	for &window_size in WINDOW_SIZES {
		for &buy_threshold in BUY_THRESHOLDS {
			for &sell_threshold in SELL_THRESHOLDS {
				for &stop_loss_threshold in STOP_LOSS_THRESHOLDS {
					for &max_hold_time_minutes in MAX_HOLD_TIME_MINUTES {
						for &max_buys in MAX_BUYS_RANGE {
							if USE_TRADE_VELOCITY {
								// Generate combinations with velocity parameters
								for &velocity_window in VELOCITY_WINDOW_SECONDS {
									for &high_velocity_threshold in HIGH_VELOCITY_THRESHOLDS {
										for &low_velocity_threshold in LOW_VELOCITY_THRESHOLDS {
											for &high_velocity_multiplier in HIGH_VELOCITY_MULTIPLIERS {
												for &low_velocity_multiplier in LOW_VELOCITY_MULTIPLIERS {
													combinations.push(ParameterSet::new(
														window_size,
														buy_threshold,
														sell_threshold,
														stop_loss_threshold,
														max_hold_time_minutes,
														max_buys,
														MIN_BLOCKS_BETWEEN_BUYS,
														MIN_HOLD_BLOCKS,
														MIN_BLOCKS_BETWEEN_SELL_BUY,
														PRICE_COLUMN.to_string(),
														VOLUME_COLUMN.to_string(),
														SLIPPAGE_TOLERANCE_PERCENT,
														USE_TRADE_VELOCITY,
														velocity_window,
														high_velocity_threshold,
														low_velocity_threshold,
														high_velocity_multiplier,
														low_velocity_multiplier,
													));
												}
											}
										}
									}
								}
							} else {
								// Generate combinations without velocity (default parameters)
								combinations.push(ParameterSet::new(
									window_size,
									buy_threshold,
									sell_threshold,
									stop_loss_threshold,
									max_hold_time_minutes,
									max_buys,
									MIN_BLOCKS_BETWEEN_BUYS,
									MIN_HOLD_BLOCKS,
									MIN_BLOCKS_BETWEEN_SELL_BUY,
									PRICE_COLUMN.to_string(),
									VOLUME_COLUMN.to_string(),
									SLIPPAGE_TOLERANCE_PERCENT,
									false, // use_trade_velocity
									30.0,  // velocity_window_seconds (default)
									2.0,   // high_velocity_threshold (default)
									0.2,   // low_velocity_threshold (default)
									0.8,   // high_velocity_multiplier (default)
									1.2,   // low_velocity_multiplier (default)
								));
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
		"   Buy thresholds: {:?} (% below VWAP, Price: {}, Volume: {})",
		BUY_THRESHOLDS, PRICE_COLUMN, VOLUME_COLUMN
	);
	println!(
		"   Sell thresholds: {:?} (% above VWAP, Price: {}, Volume: {})",
		SELL_THRESHOLDS, PRICE_COLUMN, VOLUME_COLUMN
	);
	println!("   Stop loss thresholds: {:?} (% loss)", STOP_LOSS_THRESHOLDS);
	println!("   Max hold time: {:?} (minutes)", MAX_HOLD_TIME_MINUTES);
	println!("   Max buys per token: {:?}", MAX_BUYS_RANGE);
	println!("   Min blocks between buys: {}", MIN_BLOCKS_BETWEEN_BUYS);
	println!("   Min hold blocks: {}", MIN_HOLD_BLOCKS);
	println!("   Min blocks sell → buy: {}", MIN_BLOCKS_BETWEEN_SELL_BUY);
	println!("   Slippage tolerance: {:.1}%", SLIPPAGE_TOLERANCE_PERCENT * 100.0);
	
	if USE_TRADE_VELOCITY {
		println!("   🚀 Trade Velocity ENABLED:");
		println!("     Velocity windows: {:?} (seconds)", VELOCITY_WINDOW_SECONDS);
		println!("     High velocity thresholds: {:?} (trades/sec)", HIGH_VELOCITY_THRESHOLDS);
		println!("     Low velocity thresholds: {:?} (trades/sec)", LOW_VELOCITY_THRESHOLDS);
		println!("     High velocity multipliers: {:?} (tighter)", HIGH_VELOCITY_MULTIPLIERS);
		println!("     Low velocity multipliers: {:?} (wider)", LOW_VELOCITY_MULTIPLIERS);
	} else {
		println!("   🚀 Trade Velocity: DISABLED");
	}

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
	let mut strategy = VwapStrategy::new(
		params.window_size,
		params.buy_threshold,
		params.sell_threshold,
		params.stop_loss_threshold,
		min_sol_in_curve,
		&params.price_column,
		&params.volume_column,
		params.max_buys,
		params.min_blocks_between_buys, // minimum blocks between consecutive buys
		params.min_hold_blocks, // minimum blocks to hold before selling
		params.min_blocks_between_sell_buy, // minimum blocks between sell and next buy
		params.max_hold_time_minutes,
		USE_DYNAMIC_THRESHOLDS, // use dynamic thresholds based on SOL in curve or static parameter ranges
		params.slippage_tolerance_percent, // slippage tolerance for buy orders
		params.use_trade_velocity, // use trade velocity for threshold adjustment
		params.velocity_window_seconds, // velocity window in seconds
		params.high_velocity_threshold, // high velocity threshold (trades/sec)
		params.low_velocity_threshold, // low velocity threshold (trades/sec)
		params.high_velocity_multiplier, // high velocity multiplier
		params.low_velocity_multiplier, // low velocity multiplier
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
) -> (HashMap<String, Vec<TestResult>>, HashMap<String, Vec<TestResult>>, HashMap<String, Vec<TestResult>>) {
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
	let mut drawdown_sorted_groups: HashMap<String, Vec<TestResult>> = HashMap::new();

	// Sort each group by MAR ratio, P/L, and drawdown
	for (group_name, results) in grouped_results.iter_mut() {
		// Clone results for P/L and drawdown sorting (we need all three sortings)
		let mut pnl_sorted_results = results.clone();
		let mut drawdown_sorted_results = results.clone();

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

		// Sort another clone by drawdown (ascending - lowest first)
		drawdown_sorted_results.sort_by(|a, b| {
			a.max_drawdown
				.partial_cmp(&b.max_drawdown)
				.unwrap_or(std::cmp::Ordering::Equal)
		});

		// Print TOP 20 by MAR RATIO
		println!("\n🏆 TOP 20 PERFORMERS (by MAR Ratio) for {}:", group_name);
		for (i, result) in results.iter().take(20).enumerate() {
			let mar_display = if result.mar_ratio.is_finite() {
				format!("{:.2}", result.mar_ratio)
			} else if result.mar_ratio.is_sign_positive() {
				"∞".to_string()
			} else {
				"-∞".to_string()
			};

			let velocity_info = if result.params.use_trade_velocity {
				format!(", Velocity: {:.1}s/{:.2}-{:.1}tps/{:.2}x-{:.2}x", 
					result.params.velocity_window_seconds,
					result.params.low_velocity_threshold,
					result.params.high_velocity_threshold,
					result.params.high_velocity_multiplier,
					result.params.low_velocity_multiplier
				)
			} else {
				"".to_string()
			};

			println!(
				"  {}. Window: {}, Buy: {:.1}%, Sell: {:.1}%, StopLoss: {:.0}%, MaxHoldTime: {}MIN, MaxBuys: {}, Slippage: {:.1}%{} → PnL: {:+.4} SOL, MAR: {} ({} trades, {:.1}% win rate)",
				i + 1,
				result.params.window_size,
				result.params.buy_threshold * 100.0,
				result.params.sell_threshold * 100.0,
				result.params.stop_loss_threshold * 100.0,
				result.params.max_hold_time_minutes,
				result.params.max_buys,
				result.params.slippage_tolerance_percent * 100.0,
				velocity_info,
				result.total_pnl,
				mar_display,
				result.total_trades,
				result.win_rate
			);
		}

		// Print TOP 20 by P/L
		println!("\n💰 TOP 20 PERFORMERS (by Total P/L) for {}:", group_name);
		for (i, result) in pnl_sorted_results.iter().take(20).enumerate() {
			let mar_display = if result.mar_ratio.is_finite() {
				format!("{:.2}", result.mar_ratio)
			} else if result.mar_ratio.is_sign_positive() {
				"∞".to_string()
			} else {
				"-∞".to_string()
			};

			let velocity_info = if result.params.use_trade_velocity {
				format!(", Velocity: {:.1}s/{:.2}-{:.1}tps/{:.2}x-{:.2}x", 
					result.params.velocity_window_seconds,
					result.params.low_velocity_threshold,
					result.params.high_velocity_threshold,
					result.params.high_velocity_multiplier,
					result.params.low_velocity_multiplier
				)
			} else {
				"".to_string()
			};

			println!(
				"  {}. Window: {}, Buy: {:.1}%, Sell: {:.1}%, StopLoss: {:.0}%, MaxHoldTime: {}MIN, MaxBuys: {}, Slippage: {:.1}%{} → PnL: {:+.4} SOL, MAR: {} ({} trades, {:.1}% win rate)",
				i + 1,
				result.params.window_size,
				result.params.buy_threshold * 100.0,
				result.params.sell_threshold * 100.0,
				result.params.stop_loss_threshold * 100.0,
				result.params.max_hold_time_minutes,
				result.params.max_buys,
				result.params.slippage_tolerance_percent * 100.0,
				velocity_info,
				result.total_pnl,
				mar_display,
				result.total_trades,
				result.win_rate
			);
		}

		// Print TOP 20 by LOWEST DRAWDOWN
		println!("\n📉 TOP 20 PERFORMERS (by Lowest Drawdown) for {}:", group_name);
		for (i, result) in drawdown_sorted_results.iter().take(20).enumerate() {
			let mar_display = if result.mar_ratio.is_finite() {
				format!("{:.2}", result.mar_ratio)
			} else if result.mar_ratio.is_sign_positive() {
				"∞".to_string()
			} else {
				"-∞".to_string()
			};

			let velocity_info = if result.params.use_trade_velocity {
				format!(", Velocity: {:.1}s/{:.2}-{:.1}tps/{:.2}x-{:.2}x", 
					result.params.velocity_window_seconds,
					result.params.low_velocity_threshold,
					result.params.high_velocity_threshold,
					result.params.high_velocity_multiplier,
					result.params.low_velocity_multiplier
				)
			} else {
				"".to_string()
			};

			println!(
				"  {}. Window: {}, Buy: {:.1}%, Sell: {:.1}%, StopLoss: {:.0}%, MaxHoldTime: {}MIN, MaxBuys: {}, Slippage: {:.1}%{} → Drawdown: {:.4} SOL, PnL: {:+.4} SOL, MAR: {} ({} trades, {:.1}% win rate)",
				i + 1,
				result.params.window_size,
				result.params.buy_threshold * 100.0,
				result.params.sell_threshold * 100.0,
				result.params.stop_loss_threshold * 100.0,
				result.params.max_hold_time_minutes,
				result.params.max_buys,
				result.params.slippage_tolerance_percent * 100.0,
				velocity_info,
				result.max_drawdown,
				result.total_pnl,
				mar_display,
				result.total_trades,
				result.win_rate
			);
		}

		// Keep top 20 for MAR ratio, P/L, and drawdown for CSV export
		let mar_top_20 = results.iter().take(20).cloned().collect();
		let pnl_top_20 = pnl_sorted_results.iter().take(20).cloned().collect();
		let drawdown_top_20 = drawdown_sorted_results.iter().take(20).cloned().collect();
		
		mar_sorted_groups.insert(group_name.clone(), mar_top_20);
		pnl_sorted_groups.insert(group_name.clone(), pnl_top_20);
		drawdown_sorted_groups.insert(group_name.clone(), drawdown_top_20);
	}

	(mar_sorted_groups, pnl_sorted_groups, drawdown_sorted_groups)
}

fn export_optimization_results(
	mar_top_performers: &HashMap<String, Vec<TestResult>>,
	pnl_top_performers: &HashMap<String, Vec<TestResult>>,
	drawdown_top_performers: &HashMap<String, Vec<TestResult>>,
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
		csv_content.push_str("SOL_Level,Rank,Window_Size,Buy_Threshold,Sell_Threshold,Stop_Loss_Threshold,Max_Hold_Time_Minutes,Max_Buys,Min_Blocks_Between_Buys,Min_Hold_Blocks,Min_Blocks_Between_Sell_Buy,Price_Column,Volume_Column,Slippage_Tolerance_Percent,Use_Trade_Velocity,Velocity_Window_Seconds,High_Velocity_Threshold,Low_Velocity_Threshold,High_Velocity_Multiplier,Low_Velocity_Multiplier,Total_PnL,Total_Trades,Win_Rate,Avg_Win,Avg_Loss,Profit_Factor,Max_Drawdown,MAR_Ratio,Trades_In_Group\n");

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
					"{},{},{},{:.4},{:.4},{:.4},{},{},{},{},{},{},{},{:.4},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.6},{},{:.2},{:.6},{:.6},{:.2},{:.6},{},{}\n",
					sol_level,
					rank + 1,
					result.params.window_size,
					result.params.buy_threshold,
					result.params.sell_threshold,
					result.params.stop_loss_threshold,
					result.params.max_hold_time_minutes,
					result.params.max_buys,
					result.params.min_blocks_between_buys,
					result.params.min_hold_blocks,
					result.params.min_blocks_between_sell_buy,
					result.params.price_column,
					result.params.volume_column,
					result.params.slippage_tolerance_percent,
					result.params.use_trade_velocity,
					result.params.velocity_window_seconds,
					result.params.high_velocity_threshold,
					result.params.low_velocity_threshold,
					result.params.high_velocity_multiplier,
					result.params.low_velocity_multiplier,
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

		let filename = format!("{}/vwap_top_performers_{}_{}.csv", results_dir, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(), file_suffix);
		let mut file = std::fs::File::create(&filename)?;
		file.write_all(csv_content.as_bytes())?;

		println!("\n📁 Exported {} results to: {}", description, filename);
			println!(
				"   📊 {} SOL levels with top 20 performers per group",
				sorted_groups.len()
			);

		Ok(filename)
	};

	// Export MAR ratio top performers
	export_to_csv(mar_top_performers, "mar_ratio", "MAR Ratio top performers")?;
	
	// Export P/L top performers
	export_to_csv(pnl_top_performers, "pnl", "P/L top performers")?;
	
	// Export drawdown top performers
	export_to_csv(drawdown_top_performers, "drawdown", "Lowest Drawdown top performers")?;

	Ok(())
}

fn run_parameter_optimization() -> Result<()> {
	println!("{}", "=".repeat(100));
	println!("🧪 VWAP PARAMETER OPTIMIZATION");
	println!("{}", "=".repeat(100));
	println!(
		"📊 Testing {} parameter combinations",
		WINDOW_SIZES.len()
			* BUY_THRESHOLDS.len()
			* SELL_THRESHOLDS.len()
			* STOP_LOSS_THRESHOLDS.len()
			* MAX_HOLD_TIME_MINUTES.len()
			* MAX_BUYS_RANGE.len()
	);
	println!("🎯 Data preprocessing will run ONLY ONCE at the beginning");
	println!(
		"📈 Results grouped by actual SOL in curve levels ({} SOL increments)",
		SOL_GROUP_LEVEL
	);
	println!(
		"📊 OPTIMIZATION CRITERION: MAR Ratio (Managed Account Ratio = Total Return / Max Drawdown)"
	);
	println!("🔢 VWAP calculation using: Price: {}, Volume: {}", PRICE_COLUMN, VOLUME_COLUMN);
	println!("📊 Slippage tolerance: {:.1}% for all parameter combinations", SLIPPAGE_TOLERANCE_PERCENT * 100.0);

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
		BUY_FEE_PERCENT,
		SELL_FEE_PERCENT,
		FLAT_FEE_SOL_BUY,
		FLAT_FEE_SOL_SELL,
		BLOCK_LATENCY_MAX,
		false, // export_trades = false for parameter optimization (prevents file explosion)
		false, // verbose = false for parameter optimization (prevents debug spam)
		USE_DYNAMIC_FEES,
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
			* STOP_LOSS_THRESHOLDS.len()
			* MAX_HOLD_TIME_MINUTES.len()
			* MAX_BUYS_RANGE.len()
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
			let (result, temp_framework, temp_results) = match test_parameter_set_with_preprocessed_framework(
				params,
				min_sol_in_curve,
				&master_framework,
			) {
				Ok(Some((results, framework))) => {
					// Analyze results by SOL groups
					match analyze_results_by_sol_groups(
						params, &results, &framework,
					) {
						Ok(group_results) => (Some(group_results), Some(framework), Some(results)),
						Err(_) => (None, Some(framework), Some(results)),
					}
				}
				Ok(None) => (None, None, None),
				Err(_) => (None, None, None),
			};

			// Explicitly drop heavy objects to clear memory between parameter tests
			drop(temp_framework);

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
	let (mar_top_performers, pnl_top_performers, drawdown_top_performers) = find_top_performers_by_group(all_results);

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
			let velocity_summary = if best.params.use_trade_velocity {
				format!(", Velocity: {:.1}s/{:.2}-{:.1}tps/{:.2}x-{:.2}x", 
					best.params.velocity_window_seconds,
					best.params.low_velocity_threshold,
					best.params.high_velocity_threshold,
					best.params.high_velocity_multiplier,
					best.params.low_velocity_multiplier
				)
			} else {
				"".to_string()
			};

			println!(
				"    Params: Window: {}, Buy: {:.1}%, Sell: {:.1}%, StopLoss: {:.0}%, MaxHoldTime: {}min, MaxBuys: {}, Slippage: {:.1}%{}",
				best.params.window_size,
				best.params.buy_threshold * 100.0,
				best.params.sell_threshold * 100.0,
				best.params.stop_loss_threshold * 100.0,
				best.params.max_hold_time_minutes,
				best.params.max_buys,
				best.params.slippage_tolerance_percent * 100.0,
				velocity_summary
			);
		}
	}

	// Export results - MAR ratio, P/L, and drawdown top performers
	export_optimization_results(&mar_top_performers, &pnl_top_performers, &drawdown_top_performers)?;

	println!("\n✅ Parameter optimization complete!");
	println!(
		"🦀 Best parameter combinations identified for each SOL group (by MAR Ratio, P/L, and Lowest Drawdown)"
	);

	Ok(())
}

fn main() -> Result<()> {
	println!("🚀 Starting VWAP Parameter Optimization");
	println!("Based on efficient testing of parameter combinations");
	println!("🔢 VWAP calculation columns: Price: {}, Volume: {}", PRICE_COLUMN, VOLUME_COLUMN);
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
			WINDOW_SIZES.len() * BUY_THRESHOLDS.len() * SELL_THRESHOLDS.len() * STOP_LOSS_THRESHOLDS.len() * MAX_HOLD_TIME_MINUTES.len() * MAX_BUYS_RANGE.len();
		assert_eq!(combinations.len(), expected_count);
	}

	#[test]
	fn test_parameter_set_creation() {
		let params = ParameterSet::new(
			200, 
			-0.05, 
			0.03, 
			0.50,
			360,
			1,
			5,     // min_blocks_between_buys
			5,     // min_hold_blocks  
			10,    // min_blocks_between_sell_buy
			"Token Price".to_string(), 
			"Token Amount".to_string(),
			0.05,
			true,  // use_trade_velocity
			30.0,  // velocity_window_seconds
			2.0,   // high_velocity_threshold
			0.2,   // low_velocity_threshold
			0.8,   // high_velocity_multiplier
			1.2    // low_velocity_multiplier
		);
		assert_eq!(params.window_size, 200);
		assert_eq!(params.buy_threshold, -0.05);
		assert_eq!(params.sell_threshold, 0.03);
		assert_eq!(params.stop_loss_threshold, 0.50);
		assert_eq!(params.max_hold_time_minutes, 360);
		assert_eq!(params.min_blocks_between_buys, 5);
		assert_eq!(params.min_hold_blocks, 5);
		assert_eq!(params.min_blocks_between_sell_buy, 10);
		assert_eq!(params.slippage_tolerance_percent, 0.05);
		assert_eq!(params.use_trade_velocity, true);
		assert_eq!(params.velocity_window_seconds, 30.0);
	}
}
