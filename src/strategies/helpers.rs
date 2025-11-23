use std::collections::{HashMap, HashSet};
use polars::prelude::*;
use chrono::{DateTime, Utc};

/// Get Block ID from row data
pub fn get_block_id(row_data: &HashMap<String, AnyValue>) -> i64 {
    match row_data.get("Block ID") {
        Some(AnyValue::Int64(block_id)) => *block_id,
        Some(AnyValue::Int32(block_id)) => *block_id as i64,
        Some(AnyValue::UInt64(block_id)) => *block_id as i64,
        Some(AnyValue::UInt32(block_id)) => *block_id as i64,
        _ => 0,
    }
}

/// Get timestamp in milliseconds from row data
pub fn get_timestamp_ms(row_data: &HashMap<String, AnyValue>) -> i64 {
    // Try Date_parsed first (preprocessed datetime), then fallback to Date
    match row_data.get("Date_parsed") {
        Some(AnyValue::Datetime(timestamp_ms, _, _)) => *timestamp_ms,
        _ => {
            // Fallback: try to parse original Date string if Date_parsed not available
            match row_data.get("Date") {
                Some(AnyValue::String(date_str)) => {
                    // Parse the ISO 8601 date string (e.g., "2025-08-31T20:08:15Z")
                    match date_str.parse::<DateTime<Utc>>() {
                        Ok(datetime) => datetime.timestamp_millis(),
                        Err(_) => {
                            // If parsing fails, return 0 (should rarely happen with valid data)
                            0
                        }
                    }
                }
                _ => 0,
            }
        }
    }
}

/// Get column value from row data (generic method for any numeric column)
pub fn get_column_value(
    row_data: &HashMap<String, AnyValue>,
    column_name: &str,
) -> f64 {
    // Special handling for computed "SOL" column
    if column_name == "SOL" {
        return calculate_sol_value(row_data);
    }

    // Standard column lookup for existing CSV columns
    match row_data.get(column_name) {
        Some(AnyValue::Float64(val)) => *val,
        Some(AnyValue::Float32(val)) => *val as f64,
        Some(AnyValue::Int64(val)) => *val as f64,
        Some(AnyValue::Int32(val)) => *val as f64,
        Some(AnyValue::UInt64(val)) => *val as f64,
        Some(AnyValue::UInt32(val)) => *val as f64,
        _ => 0.0,
    }
}

/// Get boolean column value from row data
pub fn get_boolean_column_value(
    row_data: &HashMap<String, AnyValue>,
    column_name: &str,
) -> bool {
    match row_data.get(column_name) {
        Some(AnyValue::Boolean(val)) => *val,
        Some(AnyValue::Int64(val)) => *val != 0,
        Some(AnyValue::Int32(val)) => *val != 0,
        Some(AnyValue::UInt64(val)) => *val != 0,
        Some(AnyValue::UInt32(val)) => *val != 0,
        Some(AnyValue::Float64(val)) => *val != 0.0,
        Some(AnyValue::Float32(val)) => *val != 0.0,
        Some(AnyValue::String(val)) => val.to_lowercase() == "true" || *val == "1",
        Some(AnyValue::StringOwned(val)) => val.to_lowercase() == "true" || val.as_str() == "1",
        _ => false,
    }
}

/// Get the price column value from row data using provided column name
pub fn get_price_column_value(
    row_data: &HashMap<String, AnyValue>,
    price_column: &str,
) -> f64 {
    get_column_value(row_data, price_column)
}

/// Get the volume column value from row data using provided column name
pub fn get_volume_column_value(
    row_data: &HashMap<String, AnyValue>,
    volume_column: &str,
) -> f64 {
    get_column_value(row_data, volume_column)
}

/// Calculate SOL value: Token Price * Token Amount, with sign based on Transaction Type
pub fn calculate_sol_value(row_data: &HashMap<String, AnyValue>) -> f64 {
    // Get Token Price
    let token_price = get_column_value(row_data, "Token Price");

    // Get Token Amount
    let token_amount = get_column_value(row_data, "Token Amount");

    // Calculate base SOL value
    let sol_value = token_price * token_amount;

    // Get Transaction Type to determine sign
    let transaction_type = match row_data.get("Transaction Type") {
        Some(AnyValue::String(tx_type)) => tx_type,
        Some(AnyValue::StringOwned(tx_type)) => tx_type.as_str(),
        _ => "Buy", // Default to Buy if not found
    };

    // Apply sign: Buy = positive, Sell = negative
    match transaction_type.to_lowercase().as_str() {
        "sell" => -sol_value,
        _ => sol_value, // Buy or any other type stays positive
    }
}

/// Get SOL in curve from row data (convenience method)
pub fn get_sol_in_curve(row_data: &HashMap<String, AnyValue>) -> f64 {
    get_column_value(row_data, "SOL in Curve")
}

/// Get wallet address from row data
pub fn get_wallet_address(row_data: &HashMap<String, AnyValue>) -> String {
    match row_data.get("Wallet Address") {
        Some(AnyValue::String(addr)) => addr.to_string(),
        Some(AnyValue::StringOwned(addr)) => addr.to_string(),
        _ => String::new(),
    }
}

/// Get transaction type from row data
pub fn get_transaction_type(row_data: &HashMap<String, AnyValue>) -> String {
    match row_data.get("Transaction Type") {
        Some(AnyValue::String(tx_type)) => tx_type.to_string(),
        Some(AnyValue::StringOwned(tx_type)) => tx_type.to_string(),
        _ => "Unknown".to_string(),
    }
}

/// Get token amount from row data
pub fn get_token_amount(row_data: &HashMap<String, AnyValue>) -> f64 {
    get_column_value(row_data, "Token Amount")
}

/// Get token price from row data
pub fn get_token_price(row_data: &HashMap<String, AnyValue>) -> f64 {
    get_column_value(row_data, "Token Price")
}

/// Update threshold tracking when a token reaches the SOL threshold
pub fn update_threshold_tracking(token_address: &str, current_sol_curve: f64, min_sol_in_curve: f64, tokens_reached_threshold: &mut HashSet<String>) {
    if current_sol_curve >= min_sol_in_curve {
        tokens_reached_threshold.insert(token_address.to_string());
    }
}

/// Check if we can buy based on block and transaction constraints
pub fn can_buy_with_constraints(
    token_address: &str,
    current_block_id: i64,
    current_sol_curve: f64,
    current_price: f64,
    sell_block_ids: &HashMap<String, i64>,
    buy_block_ids: &HashMap<String, Vec<i64>>,
    min_blocks_between_sell_buy: i64,
    max_buys: usize,
    min_blocks_between_buys: i64,
    min_sol_in_curve: f64,
    tokens_reached_threshold: &HashSet<String>,
) -> bool {
    // MIN SOL IN CURVE CHECK - minimum price constraint
    if current_price < 0.000000094 {
        return false;
    }

    // Check minimum SOL in curve threshold (only for tokens that haven't reached threshold)
    if !tokens_reached_threshold.contains(token_address) {
        if current_sol_curve < min_sol_in_curve {
            return false;
        }
    }

    // Check block constraint after last sell
    if let Some(last_sell_block_id) = sell_block_ids.get(token_address) {
        let blocks_since_sell = current_block_id - last_sell_block_id;
        if blocks_since_sell < min_blocks_between_sell_buy {
            return false;
        }
    }

    // Check if this is the first buy - apply SOL curve constraint
    if !buy_block_ids.contains_key(token_address) 
        || buy_block_ids[token_address].is_empty() 
    {
        // For first buy, block if SOL curve is outside 25.0-350.0 range
        if current_sol_curve < 25.0 || current_sol_curve > 350.0 {
            return false;  // Block first buy if outside range
        }
        // If SOL curve is good, continue with other checks below
    }
    
    // Check max buys constraint
    if buy_block_ids.get(token_address).map_or(0, |v| v.len()) >= max_buys {
        return false;
    }

    // Check block constraint between buys (only if there are previous buys)
    if let Some(buy_block_ids) = buy_block_ids.get(token_address) {
        if let Some(last_buy_block_id) = buy_block_ids.last() {
            let blocks_since_last_buy = current_block_id - last_buy_block_id;
            if blocks_since_last_buy < min_blocks_between_buys {
                return false;
            }
        }
    }

    true
}

/// Check if we can sell based on minimum hold blocks constraint
pub fn can_sell_with_hold_blocks(
    token_address: &str,
    current_block_id: i64,
    buy_block_ids: &HashMap<String, Vec<i64>>,
    min_hold_blocks: i64,
) -> bool {
    if !buy_block_ids.contains_key(token_address)
        || buy_block_ids[token_address].is_empty()
    {
        return true; // No buy blocks tracked, can sell
    }

    // Check if enough blocks have passed since the FIRST buy (most restrictive)
    let first_buy_block_id = buy_block_ids[token_address][0];
    let blocks_held = current_block_id - first_buy_block_id;

    if blocks_held < min_hold_blocks {
        return false;
    }

    true
}
