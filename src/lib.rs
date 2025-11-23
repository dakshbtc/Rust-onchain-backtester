//! Rust Backtesting Framework
//!
//! A high-performance backtesting framework built with Polars that replicates
//! the exact functionality of the Python backtesting system.

pub mod framework;
pub mod strategies;
// moved: trading_strategy is now strategies::strategy_base

pub use framework::backtesting_framework::BacktestFramework;
pub use framework::manipulation_detector::ManipulationDetector;
pub use strategies::{CopyTradingStrategy, MeanReversionStrategy, VwapStrategy};
pub use strategies::strategy_base::TradingStrategy;
