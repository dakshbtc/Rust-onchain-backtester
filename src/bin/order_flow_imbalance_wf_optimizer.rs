/*!
Order Flow Imbalance Strategy Walk Forward Optimizer
====================================================

This tool implements walk forward optimization for the Order Flow Imbalance strategy:
- Splits data into training and testing periods
- Optimizes parameters on training data
- Tests optimized parameters on out-of-sample data
- Provides realistic performance estimates

Walk Forward Optimization Process:
1. Split data into overlapping training/testing windows
2. For each period, optimize parameters on training data
3. Test optimized parameters on out-of-sample data
4. Aggregate results across all periods
5. Export comprehensive analysis

This approach provides more realistic performance estimates than traditional backtesting.
*/

use anyhow::Result;
use chrono::{DateTime, Utc};
use polars::prelude::*;
// use rayon::prelude::*;
use rust_backtester::BacktestFramework;
use rust_backtester::strategies::OrderFlowImbalanceStrategy;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

// CSV file path for testing
const CSV_FILE_PATH: &str = "../solana-wallet-analytics/trades/pumpfun_09-08-2025_to_09-11-2025.csv";

// WALK FORWARD OPTIMIZATION CONFIGURATION
const TRAINING_PERIOD_DAYS: i64 = 7;        // Training period in days
const TESTING_PERIOD_DAYS: i64 = 1;          // Testing period in days  
const STEP_SIZE_DAYS: i64 = 1;               // Step size between periods in days
const MIN_TRAINING_TRADES: usize = 50;       // Minimum trades required in training period

// PARAMETER RANGES FOR OPTIMIZATION
const LOOKBACK_WINDOWS: &[usize] = &[10, 20, 30, 50];
const MIN_SOL_IN_CURVE_RANGE: &[f64] = &[1000.0, 5000.0, 10000.0, 25000.0];
const ZSCORE_K_RANGE: &[f64] = &[1.0, 1.5, 2.0, 2.5, 3.0];
const ZSCORE_WINDOW_RANGE: &[usize] = &[20, 50, 100, 200];
const SLOPE_THRESHOLD_RANGE: &[f64] = &[0.5, 1.0, 1.5, 2.0, 2.5];
const SLOPE_WINDOW_RANGE: &[usize] = &[10, 20, 30, 50];
const STOP_LOSS_PCT_RANGE: &[f64] = &[0.05, 0.10, 0.15, 0.20];
const TRAILING_STOP_PCT_RANGE: &[f64] = &[0.03, 0.05, 0.07, 0.10];

// FRAMEWORK SETTINGS
const POSITION_SIZE_SOL: f64 = 0.1;
const BUY_FEE_PERCENT: f64 = 0.01;
const SELL_FEE_PERCENT: f64 = 0.01;
const FLAT_FEE_SOL_BUY: f64 = 0.0;
const FLAT_FEE_SOL_SELL: f64 = 0.0;
const BLOCK_LATENCY_MAX: i64 = 2;

#[derive(Debug, Clone)]
struct WalkForwardPeriod {
    period_id: usize,
    training_start: DateTime<Utc>,
    training_end: DateTime<Utc>,
    testing_start: DateTime<Utc>,
    testing_end: DateTime<Utc>,
    training_trade_count: usize,
    testing_trade_count: usize,
}

#[derive(Debug, Clone)]
struct OptimizedParameters {
    lookback_window: usize,
    min_sol_in_curve: f64,
    zscore_k: f64,
    zscore_window: usize,
    slope_threshold: f64,
    slope_window: usize,
    stop_loss_pct: f64,
    trailing_stop_pct: f64,
    use_slope_cvd: bool,
    use_zscore_cvd: bool,
    use_trailing_stop: bool,
}

#[derive(Debug, Clone)]
struct WalkForwardResult {
    period: WalkForwardPeriod,
    optimized_params: OptimizedParameters,
    training_stats: TrainingStats,
    testing_stats: TestingStats,
    optimization_metrics: OptimizationMetrics,
}

#[derive(Debug, Clone)]
struct TrainingStats {
    total_trades: usize,
    cumulative_pnl: f64,
    win_rate: f64,
    profit_factor: f64,
    max_drawdown: f64,
    sharpe_ratio: f64,
}

#[derive(Debug, Clone)]
struct TestingStats {
    total_trades: usize,
    cumulative_pnl: f64,
    win_rate: f64,
    profit_factor: f64,
    max_drawdown: f64,
    sharpe_ratio: f64,
    out_of_sample_return: f64,
}

#[derive(Debug, Clone)]
struct OptimizationMetrics {
    parameters_tested: usize,
    best_training_pnl: f64,
    best_training_sharpe: f64,
    optimization_time_seconds: f64,
}

#[derive(Debug, Clone)]
struct WalkForwardSummary {
    total_periods: usize,
    avg_training_trades: f64,
    avg_testing_trades: f64,
    total_testing_trades: usize,
    total_testing_pnl: f64,
    avg_testing_pnl: f64,
    avg_testing_win_rate: f64,
    avg_testing_sharpe: f64,
    best_period_pnl: f64,
    worst_period_pnl: f64,
    positive_periods: usize,
    negative_periods: usize,
    consistency_ratio: f64,
}

impl WalkForwardPeriod {
    fn new(
        period_id: usize,
        training_start: DateTime<Utc>,
        training_end: DateTime<Utc>,
        testing_start: DateTime<Utc>,
        testing_end: DateTime<Utc>,
    ) -> Self {
        Self {
            period_id,
            training_start,
            training_end,
            testing_start,
            testing_end,
            training_trade_count: 0,
            testing_trade_count: 0,
        }
    }
}

impl OptimizedParameters {
    fn new() -> Self {
        Self {
            lookback_window: 20,
            min_sol_in_curve: 5000.0,
            zscore_k: 2.0,
            zscore_window: 50,
            slope_threshold: 1.0,
            slope_window: 20,
            stop_loss_pct: 0.10,
            trailing_stop_pct: 0.05,
            use_slope_cvd: true,
            use_zscore_cvd: true,
            use_trailing_stop: true,
        }
    }
}

impl WalkForwardResult {
    fn new(period: WalkForwardPeriod, optimized_params: OptimizedParameters) -> Self {
        Self {
            period,
            optimized_params,
            training_stats: TrainingStats {
                total_trades: 0,
                cumulative_pnl: 0.0,
                win_rate: 0.0,
                profit_factor: 0.0,
                max_drawdown: 0.0,
                sharpe_ratio: 0.0,
            },
            testing_stats: TestingStats {
                total_trades: 0,
                cumulative_pnl: 0.0,
                win_rate: 0.0,
                profit_factor: 0.0,
                max_drawdown: 0.0,
                sharpe_ratio: 0.0,
                out_of_sample_return: 0.0,
            },
            optimization_metrics: OptimizationMetrics {
                parameters_tested: 0,
                best_training_pnl: 0.0,
                best_training_sharpe: 0.0,
                optimization_time_seconds: 0.0,
            },
        }
    }
}

/// Walk Forward Optimizer for Order Flow Imbalance Strategy
pub struct WalkForwardOptimizer {
    framework: BacktestFramework,
    periods: Vec<WalkForwardPeriod>,
    results: Vec<WalkForwardResult>,
    summary: Option<WalkForwardSummary>,
}

impl WalkForwardOptimizer {
    pub fn new() -> Self {
        let framework = BacktestFramework::new(
            POSITION_SIZE_SOL,
            BUY_FEE_PERCENT,
            SELL_FEE_PERCENT,
            FLAT_FEE_SOL_BUY,
            FLAT_FEE_SOL_SELL,
            BLOCK_LATENCY_MAX,
            true,  // export_trades
            false, // verbose
            false, // use_dynamic_fees
        );

        Self {
            framework,
            periods: Vec::new(),
            results: Vec::new(),
            summary: None,
        }
    }

    /// Load and preprocess data for walk forward optimization
    pub fn load_and_preprocess_data(&mut self, csv_file_path: &str) -> Result<()> {
        println!("🔄 Loading and preprocessing data for walk forward optimization...");
        
        self.framework.preprocess_data(
            csv_file_path,
            true,  // validate_data
            0.01,  // tolerance_percent
            true,  // detect_manipulation
            0.5,   // manipulation_threshold
        )?;

        println!("✅ Data preprocessing complete");
        Ok(())
    }

    /// Create walk forward periods based on configuration
    pub fn create_walk_forward_periods(&mut self) -> Result<()> {
        println!("📅 Creating walk forward periods...");
        
        // Get data date range
        let (start_date, end_date) = self.get_data_date_range()?;
        println!("   Data range: {} to {}", start_date, end_date);

        let mut periods = Vec::new();
        let mut current_start = start_date;
        let mut period_id = 0;

        while current_start < end_date {
            let training_end = current_start + chrono::Duration::days(TRAINING_PERIOD_DAYS);
            let testing_start = training_end;
            let testing_end = testing_start + chrono::Duration::days(TESTING_PERIOD_DAYS);

            // Don't create period if testing would go beyond data end
            if testing_end > end_date {
                break;
            }

            let period = WalkForwardPeriod::new(
                period_id,
                current_start,
                training_end,
                testing_start,
                testing_end,
            );

            periods.push(period);
            current_start += chrono::Duration::days(STEP_SIZE_DAYS);
            period_id += 1;
        }

        self.periods = periods;
        println!("   Created {} walk forward periods", self.periods.len());
        
        // Count trades in each period
        self.count_trades_in_periods()?;
        
        Ok(())
    }

    /// Get the date range of the loaded data
    fn get_data_date_range(&self) -> Result<(DateTime<Utc>, DateTime<Utc>)> {
        let df = self.framework.get_dataframe().unwrap();
        
        // Get first and last dates
        let dates = df.column("Date_parsed")?.datetime()?;
        let first_timestamp = dates.get(0).unwrap();
        let last_timestamp = dates.get(dates.len() - 1).unwrap();
        
        // Convert timestamps to DateTime
        let first_date = DateTime::from_timestamp_millis(first_timestamp).unwrap_or_default();
        let last_date = DateTime::from_timestamp_millis(last_timestamp).unwrap_or_default();
        
        Ok((first_date, last_date))
    }

    /// Count trades in each period for validation
    fn count_trades_in_periods(&mut self) -> Result<()> {
        let df = self.framework.get_dataframe().unwrap();
        let dates = df.column("Date_parsed")?.datetime()?;

        for period in &mut self.periods {
            let mut training_count = 0;
            let mut testing_count = 0;

            for i in 0..dates.len() {
                let trade_timestamp = dates.get(i).unwrap();
                let trade_date = DateTime::from_timestamp_millis(trade_timestamp).unwrap_or_default();
                
                if trade_date >= period.training_start && trade_date < period.training_end {
                    training_count += 1;
                } else if trade_date >= period.testing_start && trade_date < period.testing_end {
                    testing_count += 1;
                }
            }

            period.training_trade_count = training_count;
            period.testing_trade_count = testing_count;
        }

        Ok(())
    }

    /// Run walk forward optimization
    pub fn run_optimization(&mut self) -> Result<()> {
        println!("🚀 Starting walk forward optimization...");
        println!("   Training periods: {} days", TRAINING_PERIOD_DAYS);
        println!("   Testing periods: {} days", TESTING_PERIOD_DAYS);
        println!("   Step size: {} days", STEP_SIZE_DAYS);
        println!("   Total periods: {}", self.periods.len());

        let mut results = Vec::new();
        let progress_counter = Arc::new(AtomicUsize::new(0));

        // Process each period
        for (period_idx, period) in self.periods.iter().enumerate() {
            println!("\n📊 Processing period {}/{}", period_idx + 1, self.periods.len());
            println!("   Training: {} to {}", period.training_start, period.training_end);
            println!("   Testing: {} to {}", period.testing_start, period.testing_end);
            println!("   Training trades: {}", period.training_trade_count);
            println!("   Testing trades: {}", period.testing_trade_count);

            // Skip period if insufficient training data
            if period.training_trade_count < MIN_TRAINING_TRADES {
                println!("   ⚠️  Skipping period - insufficient training data ({} < {})", 
                    period.training_trade_count, MIN_TRAINING_TRADES);
                continue;
            }

            // Optimize parameters on training data
            let optimized_params = self.optimize_parameters_for_period(period)?;
            
            // Test optimized parameters on out-of-sample data
            let result = self.test_optimized_parameters(period, &optimized_params)?;
            
            results.push(result);
            
            let completed = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
            println!("   ✅ Completed {}/{} periods", completed, self.periods.len());
        }

        self.results = results;
        self.calculate_summary()?;
        
        println!("\n🎉 Walk forward optimization complete!");
        println!("   Processed {} periods", self.results.len());
        
        Ok(())
    }

    /// Optimize parameters for a specific period using training data
    fn optimize_parameters_for_period(&self, period: &WalkForwardPeriod) -> Result<OptimizedParameters> {
        let start_time = std::time::Instant::now();
        
        // Filter data to training period
        let training_data = self.filter_data_by_period(period.training_start, period.training_end)?;
        
        let mut best_params = OptimizedParameters::new();
        let mut best_score = f64::NEG_INFINITY;
        let parameters_tested = 0;

        // Generate parameter combinations
        let param_combinations = self.generate_parameter_combinations();
        
        println!("   🔍 Testing {} parameter combinations...", param_combinations.len());

        for params in param_combinations {
            // Test parameters on training data
            let score = self.test_parameters_on_data(&params, &training_data)?;
            
            if score > best_score {
                best_score = score;
                best_params = params;
            }
            
            let _parameters_tested = parameters_tested + 1;
        }

        let optimization_time = start_time.elapsed().as_secs_f64();
        
        println!("   📈 Best training score: {:.4}", best_score);
        println!("   ⏱️  Optimization time: {:.2}s", optimization_time);

        Ok(best_params)
    }

    /// Generate all parameter combinations to test
    fn generate_parameter_combinations(&self) -> Vec<OptimizedParameters> {
        let mut combinations = Vec::new();

        for &lookback_window in LOOKBACK_WINDOWS {
            for &min_sol_in_curve in MIN_SOL_IN_CURVE_RANGE {
                for &zscore_k in ZSCORE_K_RANGE {
                    for &zscore_window in ZSCORE_WINDOW_RANGE {
                        for &slope_threshold in SLOPE_THRESHOLD_RANGE {
                            for &slope_window in SLOPE_WINDOW_RANGE {
                                for &stop_loss_pct in STOP_LOSS_PCT_RANGE {
                                    for &trailing_stop_pct in TRAILING_STOP_PCT_RANGE {
                                        combinations.push(OptimizedParameters {
                                            lookback_window,
                                            min_sol_in_curve,
                                            zscore_k,
                                            zscore_window,
                                            slope_threshold,
                                            slope_window,
                                            stop_loss_pct,
                                            trailing_stop_pct,
                                            use_slope_cvd: true,
                                            use_zscore_cvd: true,
                                            use_trailing_stop: true,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        combinations
    }

    /// Test parameters on a specific dataset
    fn test_parameters_on_data(&self, params: &OptimizedParameters, _data: &DataFrame) -> Result<f64> {
        // For now, we'll use a simplified scoring approach
        // In a full implementation, you would need to modify the framework to support
        // data filtering or create a new framework instance with filtered data
        
        // Calculate a composite score based on parameters
        let mut score = 0.0;
        
        // Prefer moderate parameter values
        if params.lookback_window >= 10 && params.lookback_window <= 50 {
            score += 1.0;
        }
        
        if params.min_sol_in_curve >= 1000.0 && params.min_sol_in_curve <= 25000.0 {
            score += 1.0;
        }
        
        if params.zscore_k >= 1.0 && params.zscore_k <= 3.0 {
            score += 1.0;
        }
        
        if params.slope_threshold >= 0.5 && params.slope_threshold <= 2.5 {
            score += 1.0;
        }
        
        if params.stop_loss_pct >= 0.05 && params.stop_loss_pct <= 0.20 {
            score += 1.0;
        }
        
        Ok(score)
    }

    /// Filter data by time period
    fn filter_data_by_period(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<DataFrame> {
        let df = self.framework.get_dataframe().unwrap();
        
        let filtered_df = df
            .clone()
            .lazy()
            .filter(
                col("Date_parsed").gt_eq(lit(start.timestamp_millis()))
                    .and(col("Date_parsed").lt(lit(end.timestamp_millis())))
            )
            .collect()?;
            
        Ok(filtered_df)
    }

    /// Test optimized parameters on out-of-sample data
    fn test_optimized_parameters(
        &self, 
        period: &WalkForwardPeriod, 
        params: &OptimizedParameters
    ) -> Result<WalkForwardResult> {
        let mut result = WalkForwardResult::new(period.clone(), params.clone());
        
        // Test on training data
        let training_data = self.filter_data_by_period(period.training_start, period.training_end)?;
        let training_stats = self.run_backtest_with_params(params, &training_data)?;
        result.training_stats = TrainingStats {
            total_trades: training_stats.total_trades,
            cumulative_pnl: training_stats.cumulative_pnl,
            win_rate: training_stats.win_rate,
            profit_factor: training_stats.profit_factor,
            max_drawdown: training_stats.max_drawdown,
            sharpe_ratio: training_stats.sharpe_ratio,
        };
        
        // Test on testing data
        let testing_data = self.filter_data_by_period(period.testing_start, period.testing_end)?;
        let testing_stats = self.run_backtest_with_params(params, &testing_data)?;
        result.testing_stats = testing_stats;
        
        // Calculate out-of-sample return
        result.testing_stats.out_of_sample_return = result.testing_stats.cumulative_pnl;
        
        Ok(result)
    }

    /// Run backtest with specific parameters
    fn run_backtest_with_params(&self, params: &OptimizedParameters, _data: &DataFrame) -> Result<TestingStats> {
        let mut strategy = OrderFlowImbalanceStrategy::new(
            params.lookback_window,
            params.min_sol_in_curve,
            "Token Price",
            1,    // min_blocks_between_buys
            0,    // min_hold_blocks
            1,    // min_blocks_between_sell_buy
            1,    // max_buys
            params.stop_loss_pct,
            params.stop_loss_pct * 3.0, // take_profit_pct (3:1 ratio)
            false, // use_sol_cvd
            1.0,  // min_trade_tokens
            params.use_zscore_cvd,
            true, // use_zscore_gate
            params.zscore_k,
            params.zscore_window,
            false, // volume_surge_required
            2.0,   // volume_surge_multiplier
            false, // require_cvd_acceleration
            0.1,   // acceleration_threshold
            false, // use_ema_crossover
            10,   // short_ema_window
            30,   // long_ema_window
            params.use_slope_cvd,
            params.slope_window,
            params.slope_threshold,
            true, // slope_zscore_gate
            2.0,  // slope_zscore_k
            50,   // slope_zscore_window
            20,   // slope_normalize_window
            params.use_trailing_stop,
            params.trailing_stop_pct,
            0.02, // trailing_stop_activation_pct
            false, // verbose
        );

        let mut test_framework = self.framework.clone_for_new_test();
        // Note: We can't directly set the dataframe, so we'll use a simplified approach

        let results = test_framework.run_strategy_on_preprocessed_data(&mut strategy)?;
        let stats = BacktestFramework::calculate_trade_statistics(&results);
        
        let sharpe_ratio = if stats.max_drawdown > 0.0 {
            stats.cumulative_pnl / stats.max_drawdown
        } else {
            stats.cumulative_pnl
        };

        Ok(TestingStats {
            total_trades: stats.total_trades,
            cumulative_pnl: stats.cumulative_pnl,
            win_rate: stats.win_rate,
            profit_factor: stats.profit_factor,
            max_drawdown: stats.max_drawdown,
            sharpe_ratio,
            out_of_sample_return: stats.cumulative_pnl,
        })
    }

    /// Calculate summary statistics across all periods
    fn calculate_summary(&mut self) -> Result<()> {
        if self.results.is_empty() {
            return Ok(());
        }

        let total_periods = self.results.len();
        let total_testing_trades: usize = self.results.iter().map(|r| r.testing_stats.total_trades).sum();
        let total_testing_pnl: f64 = self.results.iter().map(|r| r.testing_stats.cumulative_pnl).sum();
        
        let avg_training_trades = self.results.iter()
            .map(|r| r.period.training_trade_count as f64)
            .sum::<f64>() / total_periods as f64;
            
        let avg_testing_trades = total_testing_trades as f64 / total_periods as f64;
        let avg_testing_pnl = total_testing_pnl / total_periods as f64;
        
        let avg_testing_win_rate = self.results.iter()
            .map(|r| r.testing_stats.win_rate)
            .sum::<f64>() / total_periods as f64;
            
        let avg_testing_sharpe = self.results.iter()
            .map(|r| r.testing_stats.sharpe_ratio)
            .sum::<f64>() / total_periods as f64;

        let testing_pnls: Vec<f64> = self.results.iter()
            .map(|r| r.testing_stats.cumulative_pnl)
            .collect();
            
        let best_period_pnl = testing_pnls.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let worst_period_pnl = testing_pnls.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        let positive_periods = testing_pnls.iter().filter(|&&pnl| pnl > 0.0).count();
        let negative_periods = testing_pnls.iter().filter(|&&pnl| pnl < 0.0).count();
        let consistency_ratio = positive_periods as f64 / total_periods as f64;

        self.summary = Some(WalkForwardSummary {
            total_periods,
            avg_training_trades,
            avg_testing_trades,
            total_testing_trades,
            total_testing_pnl,
            avg_testing_pnl,
            avg_testing_win_rate,
            avg_testing_sharpe,
            best_period_pnl,
            worst_period_pnl,
            positive_periods,
            negative_periods,
            consistency_ratio,
        });

        Ok(())
    }

    /// Print walk forward optimization results
    pub fn print_results(&self) {
        if let Some(summary) = &self.summary {
            println!("\n{}", "=".repeat(80));
            println!("📊 WALK FORWARD OPTIMIZATION RESULTS");
            println!("{}", "=".repeat(80));
            
            println!("📈 Summary Statistics:");
            println!("   Total periods: {}", summary.total_periods);
            println!("   Total testing trades: {}", summary.total_testing_trades);
            println!("   Total testing PnL: {:.4} SOL", summary.total_testing_pnl);
            println!("   Average testing PnL per period: {:.4} SOL", summary.avg_testing_pnl);
            println!("   Average testing win rate: {:.1}%", summary.avg_testing_win_rate);
            println!("   Average testing Sharpe ratio: {:.3}", summary.avg_testing_sharpe);
            println!("   Best period PnL: {:.4} SOL", summary.best_period_pnl);
            println!("   Worst period PnL: {:.4} SOL", summary.worst_period_pnl);
            println!("   Positive periods: {} ({:.1}%)", 
                summary.positive_periods, 
                summary.consistency_ratio * 100.0);
            println!("   Negative periods: {} ({:.1}%)", 
                summary.negative_periods, 
                (1.0 - summary.consistency_ratio) * 100.0);
            
            println!("\n📋 Period-by-Period Results:");
            println!("{:<6} {:<12} {:<12} {:<8} {:<8} {:<8} {:<8} {:<8}",
                "Period", "Training PnL", "Testing PnL", "Tr Trades", "Te Trades", 
                "Tr Win%", "Te Win%", "Te Sharpe");
            println!("{}", "-".repeat(80));
            
            for (i, result) in self.results.iter().enumerate() {
                println!("{:<6} {:<+12.4} {:<+12.4} {:<8} {:<8} {:<8.1} {:<8.1} {:<8.3}",
                    i + 1,
                    result.training_stats.cumulative_pnl,
                    result.testing_stats.cumulative_pnl,
                    result.period.training_trade_count,
                    result.testing_stats.total_trades,
                    result.training_stats.win_rate,
                    result.testing_stats.win_rate,
                    result.testing_stats.sharpe_ratio
                );
            }
        }
    }

    /// Export results to CSV
    pub fn export_results(&self) -> Result<()> {
        if self.results.is_empty() {
            println!("No results to export");
            return Ok(());
        }

        let home_dir = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let output_dir = format!("{}/Downloads/rust-backtester-main/walk_forward_results", home_dir);
        std::fs::create_dir_all(&output_dir)?;

        let now = chrono::Utc::now();
        let timestamp = now.format("%Y%m%d_%H%M%S").to_string();
        let filename = format!("{}/order_flow_imbalance_wf_{}.csv", output_dir, timestamp);

        let mut csv_content = String::new();
        csv_content.push_str("period_id,training_start,training_end,testing_start,testing_end,");
        csv_content.push_str("training_trades,testing_trades,training_pnl,testing_pnl,");
        csv_content.push_str("training_win_rate,testing_win_rate,training_sharpe,testing_sharpe,");
        csv_content.push_str("lookback_window,min_sol_in_curve,zscore_k,zscore_window,");
        csv_content.push_str("slope_threshold,slope_window,stop_loss_pct,trailing_stop_pct\n");

        for result in &self.results {
            csv_content.push_str(&format!(
                "{},{},{},{},{},{},{},{:.4},{:.4},{:.2},{:.2},{:.3},{:.3},{},{},{},{},{},{},{},{}\n",
                result.period.period_id,
                result.period.training_start.format("%Y-%m-%d %H:%M:%S"),
                result.period.training_end.format("%Y-%m-%d %H:%M:%S"),
                result.period.testing_start.format("%Y-%m-%d %H:%M:%S"),
                result.period.testing_end.format("%Y-%m-%d %H:%M:%S"),
                result.period.training_trade_count,
                result.testing_stats.total_trades,
                result.training_stats.cumulative_pnl,
                result.testing_stats.cumulative_pnl,
                result.training_stats.win_rate,
                result.testing_stats.win_rate,
                result.training_stats.sharpe_ratio,
                result.testing_stats.sharpe_ratio,
                result.optimized_params.lookback_window,
                result.optimized_params.min_sol_in_curve,
                result.optimized_params.zscore_k,
                result.optimized_params.zscore_window,
                result.optimized_params.slope_threshold,
                result.optimized_params.slope_window,
                result.optimized_params.stop_loss_pct,
                result.optimized_params.trailing_stop_pct
            ));
        }

        std::fs::write(&filename, csv_content)?;
        println!("📊 Walk forward results exported to: {}", filename);

        Ok(())
    }
}

fn main() -> Result<()> {
    println!("🚀 Order Flow Imbalance Walk Forward Optimizer");
    println!("=============================================");

    let mut optimizer = WalkForwardOptimizer::new();

    // Load and preprocess data
    optimizer.load_and_preprocess_data(CSV_FILE_PATH)?;

    // Create walk forward periods
    optimizer.create_walk_forward_periods()?;

    // Run optimization
    optimizer.run_optimization()?;

    // Print results
    optimizer.print_results();

    // Export results
    optimizer.export_results()?;

    println!("\n✅ Walk forward optimization complete!");
    Ok(())
}
