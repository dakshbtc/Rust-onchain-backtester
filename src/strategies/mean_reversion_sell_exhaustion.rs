use anyhow::Result;
use polars::prelude::*;
use std::collections::{HashMap, VecDeque};

use crate::strategies::strategy_base::{SellReason, TradingStrategy};
use super::helpers::{
	get_timestamp_ms,
	get_transaction_type,
	get_token_amount,
	get_token_price,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurgeState {
	Normal,
	SellSurgeDetected,
	PeakReached,
	Exhausted,
	Cooldown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExhaustionMethod {
	PercentageFromPeak,
	CUSUM,
	DualWindow,
	Combined,
}

pub struct MeanReversionSellExhaustionStrategy {
	// Configuration
	sell_rate_window: usize,
	surge_zscore_threshold: f64,
	surge_min_duration_trades: usize,
	exhaustion_method: ExhaustionMethod,
	exhaustion_pct_threshold: f64, // negative percentage e.g., -0.30 means -30%
	exhaustion_cusum_threshold: f64,
	exhaustion_cusum_drift: f64,
	min_sell_trades_in_surge: usize,
	cooldown_after_entry: usize,
	use_volume_weighting: bool,

	// Trade tracking
	sell_timestamps: HashMap<String, VecDeque<i64>>, // ms
	buy_timestamps: HashMap<String, VecDeque<i64>>,  // optional context
	sell_volumes: HashMap<String, VecDeque<f64>>,    // if volume weighted
	trade_index: usize,

	// Sell rate tracking
	sell_rate_history: HashMap<String, VecDeque<f64>>,
	current_sell_rate: HashMap<String, f64>,

	// State management
	token_state: HashMap<String, SurgeState>,
	peak_sell_rate: HashMap<String, f64>,
	surge_start_index: HashMap<String, usize>,
	cusum_positive: HashMap<String, f64>,

	// Position management
	last_entry_index: HashMap<String, usize>,
}

impl MeanReversionSellExhaustionStrategy {
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		sell_rate_window: usize,
		surge_zscore_threshold: f64,
		surge_min_duration_trades: usize,
		exhaustion_method: ExhaustionMethod,
		exhaustion_pct_threshold: f64,
		exhaustion_cusum_threshold: f64,
		exhaustion_cusum_drift: f64,
		min_sell_trades_in_surge: usize,
		cooldown_after_entry: usize,
		use_volume_weighting: bool,
	) -> Self {
		Self {
			sell_rate_window,
			surge_zscore_threshold,
			surge_min_duration_trades,
			exhaustion_method,
			exhaustion_pct_threshold,
			exhaustion_cusum_threshold,
			exhaustion_cusum_drift,
			min_sell_trades_in_surge,
			cooldown_after_entry,
			use_volume_weighting,
			sell_timestamps: HashMap::new(),
			buy_timestamps: HashMap::new(),
			sell_volumes: HashMap::new(),
			trade_index: 0,
			sell_rate_history: HashMap::new(),
			current_sell_rate: HashMap::new(),
			token_state: HashMap::new(),
			peak_sell_rate: HashMap::new(),
			surge_start_index: HashMap::new(),
			cusum_positive: HashMap::new(),
			last_entry_index: HashMap::new(),
		}
	}

	fn calculate_sell_rate(&self, token: &str) -> Option<f64> {
		let ts = self.sell_timestamps.get(token)?;
		if ts.len() < 2 { return None; }
		let start = ts.front().copied()?;
		let end = ts.back().copied()?;
		let span_ms = (end - start).max(1);
		let base_rate = (ts.len() as f64) / (span_ms as f64 / 1000.0);
		if !self.use_volume_weighting { return Some(base_rate); }
		let vols = match self.sell_volumes.get(token) { Some(v) => v, None => return Some(base_rate) };
		if vols.is_empty() { return Some(base_rate); }
		let total_volume: f64 = vols.iter().copied().sum::<f64>().max(0.0);
		let trade_count = ts.len() as f64;
		Some(base_rate * (if trade_count > 0.0 { total_volume / trade_count } else { 1.0 }))
	}

	fn mean_std(values: &[f64]) -> Option<(f64, f64)> {
		if values.len() < 3 { return None; }
		let mean = values.iter().copied().sum::<f64>() / (values.len() as f64);
		let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() as f64);
		Some((mean, var.sqrt()))
	}

	fn z_score(&self, token: &str, current: f64) -> Option<f64> {
		let hist = self.sell_rate_history.get(token)?;
		let vals: Vec<f64> = hist.iter().copied().collect();
		let (mean, std) = Self::mean_std(&vals)?;
		if std <= 1e-9 { return None; }
		Some((current - mean) / std)
	}

	fn update_state_machine(&mut self, token: &str, current_rate: f64) {
		let state = *self.token_state.get(token).unwrap_or(&SurgeState::Normal);
		match state {
			SurgeState::Normal => {
				if let Some(z) = self.z_score(token, current_rate) {
					if z > self.surge_zscore_threshold {
						self.token_state.insert(token.to_string(), SurgeState::SellSurgeDetected);
						self.surge_start_index.insert(token.to_string(), self.trade_index);
						self.peak_sell_rate.insert(token.to_string(), current_rate);
					}
				}
			}
			SurgeState::SellSurgeDetected => {
				let peak = self.peak_sell_rate.entry(token.to_string()).or_insert(0.0);
				if current_rate > *peak { *peak = current_rate; }
				let start = *self.surge_start_index.get(token).unwrap_or(&self.trade_index);
				let duration = self.trade_index.saturating_sub(start);
				if duration >= self.surge_min_duration_trades {
					if current_rate < *peak * 0.95 {
						self.token_state.insert(token.to_string(), SurgeState::PeakReached);
					}
				}
			}
			SurgeState::PeakReached => {
				if self.detect_exhaustion(token, current_rate) {
					let start = *self.surge_start_index.get(token).unwrap_or(&self.trade_index);
					let sell_count = self.trade_index.saturating_sub(start);
					if sell_count >= self.min_sell_trades_in_surge {
						self.token_state.insert(token.to_string(), SurgeState::Exhausted);
					} else {
						self.token_state.insert(token.to_string(), SurgeState::Normal);
					}
				}
			}
			SurgeState::Exhausted => {
				// Transition to cooldown handled when buy is executed
			}
			SurgeState::Cooldown => {
				// Cooldown expiration checked in should_buy
			}
		}
	}

	fn detect_exhaustion(&mut self, token: &str, current_rate: f64) -> bool {
		match self.exhaustion_method {
			ExhaustionMethod::PercentageFromPeak => self.detect_percentage_exhaustion(token, current_rate),
			ExhaustionMethod::CUSUM => self.detect_cusum_exhaustion(token, current_rate),
			ExhaustionMethod::DualWindow => self.detect_dual_window_exhaustion(token),
			ExhaustionMethod::Combined => {
				let a = self.detect_percentage_exhaustion(token, current_rate);
				let b = self.detect_cusum_exhaustion(token, current_rate);
				let c = self.detect_dual_window_exhaustion(token);
				[a, b, c].into_iter().filter(|x| *x).count() >= 2
			}
		}
	}

	fn detect_percentage_exhaustion(&self, token: &str, current_rate: f64) -> bool {
		let peak = *self.peak_sell_rate.get(token).unwrap_or(&0.0);
		if peak <= 0.0 { return false; }
		let decline_pct = (current_rate - peak) / peak; // e.g., -0.30 for -30%
		decline_pct <= self.exhaustion_pct_threshold
	}

	fn detect_cusum_exhaustion(&mut self, token: &str, current_rate: f64) -> bool {
		let hist = match self.sell_rate_history.get(token) { Some(h) => h, None => return false };
		if hist.is_empty() { return false; }
		let mean = hist.iter().copied().sum::<f64>() / (hist.len() as f64);
		let deviation = mean - current_rate; // decline when positive
		let entry = self.cusum_positive.entry(token.to_string()).or_insert(0.0);
		*entry = (0.0f64).max(*entry + deviation - self.exhaustion_cusum_drift);
		*entry > self.exhaustion_cusum_threshold
	}

	fn detect_dual_window_exhaustion(&self, token: &str) -> bool {
		let hist = match self.sell_rate_history.get(token) { Some(h) => h, None => return false };
		if hist.len() < 8 { return false; }
		let recent_window = 5.min(hist.len());
		let recent_avg = hist.iter().rev().take(recent_window).copied().sum::<f64>() / (recent_window as f64);
		let historical_avg = hist.iter().copied().sum::<f64>() / (hist.len() as f64);
		if historical_avg <= 1e-12 { return false; }
		let ratio = recent_avg / historical_avg;
		ratio < 0.7
	}
}

impl TradingStrategy for MeanReversionSellExhaustionStrategy {
	fn should_buy(
		&mut self,
		token_address: &str,
		_current_price: f64,
		current_index: usize,
		_row: &AnyValue,
		_row_data: &HashMap<String, AnyValue>,
	) -> Result<bool> {
		// Cooldown handling
		if let Some(state) = self.token_state.get(token_address) {
			if *state == SurgeState::Cooldown {
				let last = *self.last_entry_index.get(token_address).unwrap_or(&0);
				if current_index.saturating_sub(last) >= self.cooldown_after_entry {
					self.token_state.insert(token_address.to_string(), SurgeState::Normal);
				} else {
					return Ok(false);
				}
			}
		}

		let current_rate = match self.current_sell_rate.get(token_address).copied() { Some(r) => r, None => return Ok(false) };
		self.update_state_machine(token_address, current_rate);

		if let Some(SurgeState::Exhausted) = self.token_state.get(token_address).copied() {
			self.last_entry_index.insert(token_address.to_string(), current_index);
			self.token_state.insert(token_address.to_string(), SurgeState::Cooldown);
			return Ok(true);
		}

		Ok(false)
	}

	fn should_sell(
		&mut self,
		_token_address: &str,
		_current_price: f64,
		_current_index: usize,
		_row: &AnyValue,
		_row_data: &HashMap<String, AnyValue>,
	) -> Result<(bool, SellReason)> {
		Ok((false, SellReason::Strategy("Hold".to_string())))
	}

	fn update_data(
		&mut self,
		token_address: &str,
		_price: f64,
		tx_index: usize,
		_date: &str,
		row_data: &HashMap<String, AnyValue>,
	) -> Result<()> {
		self.trade_index = tx_index;
		let timestamp_ms = get_timestamp_ms(row_data);
		let tx_type = get_transaction_type(row_data).to_lowercase();
		if tx_type == "sell" {
			let tsq = self
				.sell_timestamps
				.entry(token_address.to_string())
				.or_insert_with(|| VecDeque::with_capacity(self.sell_rate_window + 16));
			tsq.push_back(timestamp_ms);
			while tsq.len() > self.sell_rate_window { tsq.pop_front(); }

			if self.use_volume_weighting {
				let vq = self
					.sell_volumes
					.entry(token_address.to_string())
					.or_insert_with(|| VecDeque::with_capacity(self.sell_rate_window + 16));
				vq.push_back(get_token_amount(row_data).abs());
				while vq.len() > self.sell_rate_window { vq.pop_front(); }
			}
		} else if tx_type == "buy" {
			let bq = self
				.buy_timestamps
				.entry(token_address.to_string())
				.or_insert_with(|| VecDeque::with_capacity(self.sell_rate_window + 16));
			bq.push_back(timestamp_ms);
			while bq.len() > self.sell_rate_window { bq.pop_front(); }
		}

		// Recalculate current sell rate if we have enough sells
		if let Some(rate) = self.calculate_sell_rate(token_address) {
			self.current_sell_rate.insert(token_address.to_string(), rate);
			let hist = self
				.sell_rate_history
				.entry(token_address.to_string())
				.or_insert_with(|| VecDeque::with_capacity(self.sell_rate_window * 4 + 64));
			hist.push_back(rate);
			while hist.len() > self.sell_rate_window * 4 + 64 { hist.pop_front(); }
		}

		// Optionally update peak price confirmation (not required for core logic)
		let _price = get_token_price(row_data);

		Ok(())
	}

	fn get_strategy_name(&self) -> String {
		"MeanReversionSellExhaustion".to_string()
	}

	fn get_system_name(&self) -> String {
		"MeanReversionSellExhaustionStrategy".to_string()
	}

	fn get_export_parameters(&self) -> HashMap<String, String> {
		let mut m = HashMap::new();
		m.insert("sell_rate_window".to_string(), self.sell_rate_window.to_string());
		m.insert("surge_zscore_threshold".to_string(), format!("{:.3}", self.surge_zscore_threshold));
		m.insert("surge_min_duration_trades".to_string(), self.surge_min_duration_trades.to_string());
		m.insert("exhaustion_method".to_string(), match self.exhaustion_method {
			ExhaustionMethod::PercentageFromPeak => "PercentageFromPeak",
			ExhaustionMethod::CUSUM => "CUSUM",
			ExhaustionMethod::DualWindow => "DualWindow",
			ExhaustionMethod::Combined => "Combined",
		}.to_string());
		m.insert("exhaustion_pct_threshold".to_string(), format!("{:.3}", self.exhaustion_pct_threshold));
		m.insert("exhaustion_cusum_threshold".to_string(), format!("{:.3}", self.exhaustion_cusum_threshold));
		m.insert("exhaustion_cusum_drift".to_string(), format!("{:.3}", self.exhaustion_cusum_drift));
		m.insert("min_sell_trades_in_surge".to_string(), self.min_sell_trades_in_surge.to_string());
		m.insert("cooldown_after_entry".to_string(), self.cooldown_after_entry.to_string());
		m.insert("use_volume_weighting".to_string(), self.use_volume_weighting.to_string());
		m
	}
}


