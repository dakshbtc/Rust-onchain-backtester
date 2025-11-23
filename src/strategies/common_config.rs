/*!
Universal Trading Configuration
==============================

This file contains common configuration parameters that are shared across all trading strategies.
These parameters include position sizing, fees, validation settings, and other universal constants.

Strategy-specific parameters should remain in their respective files.
*/

// =================================
// POSITION AND FEE CONFIGURATION
// =================================

pub const VERBOSE: bool = true;
pub const POSITION_SIZE_SOL: f64 = 0.1;
pub const USE_DYNAMIC_FEES: bool = true;
pub const BUY_FEE_PERCENT: f64 = 0.0125;
pub const SELL_FEE_PERCENT: f64 = 0.0125;
pub const FLAT_FEE_SOL_BUY: f64 = 0.0000667;
pub const FLAT_FEE_SOL_SELL: f64 = 0.0000667;
pub const SLIPPAGE_TOLERANCE_PERCENT: f64 = 0.05;
pub const BLOCK_LATENCY_MAX: i64 = 2;
pub const MIN_SOL_IN_CURVE: f64 = 150.0;

// =================================
// BLOCK CONSTRAINT CONFIGURATION
// =================================
pub const MIN_BLOCKS_BETWEEN_BUYS: i64 = 5;
pub const MIN_HOLD_BLOCKS: i64 = 5;
pub const MIN_BLOCKS_BETWEEN_SELL_BUY: i64 = 10;

// =================================
// DATA VALIDATION CONFIGURATION
// =================================
pub const VALIDATE_DATA: bool = true;
pub const TOLERANCE_PERCENT: f64 = 0.01;
pub const DETECT_MANIPULATION: bool = true;
pub const MANIPULATION_THRESHOLD: f64 = 60.0;

