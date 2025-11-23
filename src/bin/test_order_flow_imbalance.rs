use anyhow::Result;
use rust_backtester::{
    BacktestFramework, TradingStrategy,
};
use rust_backtester::strategies::OrderFlowImbalanceStrategy;
use rust_backtester::strategies::common_config::*;

// CSV file path for testing
const CSV_FILE_PATH: &str =
    "/home/daksh/Desktop/pumpfun_08-20-2025_to_09-03-2025.csv/pumpfun_first_10000000.csv";
    // /home/daksh/Desktop/pumpfun_10-03-2025_to_10-07-2025.csv/pumpfun_10-03-2025_to_10-07-2025.csv
    // /home/daksh/Downloads/pumpfun_09-08-2025_to_09-11-2025.csv/pumpfun_09-08-2025_to_09-11-2025.csv
    // /home/daksh/Desktop/pumpfun_08-20-2025_to_09-03-2025.csv/pumpfun_first_10000000.csv
// Order Flow Imbalance Strategy Configuration
const LOOKBACK_WINDOW: usize = 100;           // trades in rolling window
const PRICE_COLUMN: &str = "Token Price";
const STOP_LOSS_PCT: f64 = 0.20; // 10% stop loss
const TAKE_PROFIT_PCT: f64 = 0.40; // 30% take profit (configurable)
const MAX_BUYS: usize = 1; // Maximum number of buy positions per token

// Trailing Stop Loss Configuration
const USE_TRAILING_STOP: bool = true;       // enable trailing stop loss
const TRAILING_STOP_PCT: f64 = 0.05;        // 3% trailing stop
const TRAILING_STOP_ACTIVATION_PCT: f64 = 0.30; // 2% profit to activate trailing stop

// CVD Configuration
const USE_SOL_CVD: bool = true;         // accumulate SOL-value instead of tokens
const MIN_TRADE_TOKENS: f64 = 1e5;      // ignore micro trades (example threshold)

// Traditional Z-Score CVD Detection
const USE_ZSCORE_CVD: bool = false;      // enable traditional z-score CVD detection
const USE_ZSCORE_GATE: bool = false;      // require z-score burst (should always be true for z-score CVD)
const ZSCORE_K: f64 = 2.0;              // k-sigma threshold
const ZSCORE_WINDOW: usize = 200;        // window for std calc

// Volume Surge Detection
const VOLUME_SURGE_REQUIRED: bool = false;      // Enable volume surge filter
const VOLUME_SURGE_MULTIPLIER: f64 = 1.3;      // Require 2x average volume

// CVD Acceleration Filter
const REQUIRE_CVD_ACCELERATION: bool = false;   // Enable CVD acceleration filter
const ACCELERATION_THRESHOLD: f64 = 0.05;       // Minimum acceleration threshold

// EMA Crossover Configuration
const USE_EMA_CROSSOVER: bool = true;  // enable EMA crossover for buy signals
const SHORT_EMA_WINDOW: usize = 50;    // short EMA window (trades)
const LONG_EMA_WINDOW: usize = 200;     // long EMA window (trades)

// Slope-based CVD Detection
const USE_SLOPE_CVD: bool = true;       // enable slope-based CVD detection
const SLOPE_WINDOW: usize = 200;         // window for slope calculation
const SLOPE_THRESHOLD: f64 = 0.7;        // minimum slope threshold (% of avg CVD)
const SLOPE_ZSCORE_GATE: bool = false;  // require slope z-score
const SLOPE_ZSCORE_K: f64 = 2.0;       // slope z-score multiplier
const SLOPE_ZSCORE_WINDOW: usize = 200;  // window for slope z-score calculation
const SLOPE_NORMALIZE_WINDOW: usize = 200; // window for CVD normalization

// Alternative configurations for testing (uncomment to use):
// const USE_SLOPE_CVD: bool = false;      // disable slope-based detection
// const USE_ZSCORE_CVD: bool = false;     // disable traditional detection
// const SLOPE_THRESHOLD: f64 = 10.0;      // higher threshold for fewer signals
// const SLOPE_THRESHOLD: f64 = 2.0;       // lower threshold for more signals

fn test_ofi_strategy() -> Result<()> {
    println!("{}", "=".repeat(80));
    println!("📊 Testing Order Flow Imbalance (CVD) Strategy");
    println!("{}", "=".repeat(80));
    println!("This strategy combines traditional CVD burst detection with slope-based CVD analysis.");
    println!("Both detection methods can be enabled/disabled independently for testing.");
    println!("Trailing stop loss functionality is now available for enhanced risk management.");
    println!("{}", "=".repeat(80));

    let mut strategy = OrderFlowImbalanceStrategy::new(
        LOOKBACK_WINDOW,
        MIN_SOL_IN_CURVE,
        PRICE_COLUMN,
        MIN_BLOCKS_BETWEEN_BUYS,
        MIN_HOLD_BLOCKS,
        MIN_BLOCKS_BETWEEN_SELL_BUY,
        MAX_BUYS,
        STOP_LOSS_PCT,
        TAKE_PROFIT_PCT,
        USE_SOL_CVD,
        MIN_TRADE_TOKENS,
        USE_ZSCORE_CVD,
        USE_ZSCORE_GATE,
        ZSCORE_K,
        ZSCORE_WINDOW,
        VOLUME_SURGE_REQUIRED,
        VOLUME_SURGE_MULTIPLIER,
        REQUIRE_CVD_ACCELERATION,
        ACCELERATION_THRESHOLD,
        USE_EMA_CROSSOVER,
        SHORT_EMA_WINDOW,
        LONG_EMA_WINDOW,
        USE_SLOPE_CVD,
        SLOPE_WINDOW,
        SLOPE_THRESHOLD,
        SLOPE_ZSCORE_GATE,
        SLOPE_ZSCORE_K,
        SLOPE_ZSCORE_WINDOW,
        SLOPE_NORMALIZE_WINDOW,
        USE_TRAILING_STOP,
        TRAILING_STOP_PCT,
        TRAILING_STOP_ACTIVATION_PCT,
        VERBOSE,
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

    println!("Running Order Flow Imbalance Strategy with:");
    println!("  📊 CSV file: {}", CSV_FILE_PATH);
    println!("  🔎 Lookback window (trades): {}", LOOKBACK_WINDOW);
    println!("  🪙 Min trade tokens: {}", MIN_TRADE_TOKENS);
    println!();
    println!("  📈 Traditional CVD Detection:");
    println!("    • Enabled: {} (z-score: {}, k={}, window={})", USE_ZSCORE_CVD, USE_ZSCORE_GATE, ZSCORE_K, ZSCORE_WINDOW);
    println!("  📊 Slope-based CVD Detection:");
    println!("    • Enabled: {} (window={}, threshold={}%)", USE_SLOPE_CVD, SLOPE_WINDOW, SLOPE_THRESHOLD);
    println!("    • Z-score gate: {} (k={}, window={})", SLOPE_ZSCORE_GATE, SLOPE_ZSCORE_K, SLOPE_ZSCORE_WINDOW);
    println!("    • Normalize window: {} trades", SLOPE_NORMALIZE_WINDOW);
    println!();
    println!("  📈 Volume Surge Detection:");
    println!("    • Enabled: {} (multiplier: {}x)", VOLUME_SURGE_REQUIRED, VOLUME_SURGE_MULTIPLIER);
    println!();
    println!("  🚀 CVD Acceleration Filter:");
    println!("    • Enabled: {} (threshold: {})", REQUIRE_CVD_ACCELERATION, ACCELERATION_THRESHOLD);
    println!();
    println!("  📉 EMA Crossover:");
    println!("    • Enabled: {} (short={}, long={})", USE_EMA_CROSSOVER, SHORT_EMA_WINDOW, LONG_EMA_WINDOW);
    println!("    • Buy signal: Short EMA crosses above Long EMA + Slope confirmation");
    println!();
    println!("  💰 Risk Management:");
    println!("    • Position size: {} SOL", POSITION_SIZE_SOL);
    println!("    • Min SOL in curve: {}", MIN_SOL_IN_CURVE);
    println!("    • Stop loss: {:.1}% | Take profit: {:.1}%", STOP_LOSS_PCT * 100.0, TAKE_PROFIT_PCT * 100.0);
    println!("  🎯 Trailing Stop Loss:");
    println!("    • Enabled: {}", USE_TRAILING_STOP);
    if USE_TRAILING_STOP {
        println!("    • Trailing stop: {:.1}%", TRAILING_STOP_PCT * 100.0);
        println!("    • Activation threshold: {:.1}% profit", TRAILING_STOP_ACTIVATION_PCT * 100.0);
    }

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
    println!("🚀 Starting OFI Strategy Testing Suite");
    test_ofi_strategy()?;
    println!("\n✅ OFI Strategy Test Complete!");
    Ok(())
}

