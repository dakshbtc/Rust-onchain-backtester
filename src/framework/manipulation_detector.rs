use anyhow::Result;
use chrono::{DateTime, Utc};
use polars::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
// tqdm imported elsewhere

/// Manipulation details for a token
#[derive(Debug, Clone)]
pub struct ManipulationDetails {
	pub sol_amount: f64,
	pub timestamp: String,
	pub curve_sol: f64,
}

/// Detects token manipulation patterns in trading data
///
/// Identifies tokens where single actors purchase >70 SOL in same second
/// before reaching 85 SOL minimum threshold.
pub struct ManipulationDetector {
	manipulation_threshold_sol: f64,
	min_curve_threshold: f64,
	time_window_seconds: i64,

	// Results
	blacklisted_tokens: HashSet<String>,
	manipulation_details: HashMap<String, ManipulationDetails>,
}

impl ManipulationDetector {
	/// Create a new manipulation detector
	pub fn new(
		manipulation_threshold_sol: f64,
		min_curve_threshold: f64,
		time_window_seconds: i64,
	) -> Self {
		Self {
			manipulation_threshold_sol,
			min_curve_threshold,
			time_window_seconds,
			blacklisted_tokens: HashSet::new(),
			manipulation_details: HashMap::new(),
		}
	}

	/// Analyze DataFrame to detect manipulation patterns using optimized vectorized approach
	pub fn detect_manipulation(
		&mut self,
		df: &DataFrame,
	) -> Result<HashSet<String>> {
		println!(
			"🚨 Analyzing {} transactions for manipulation patterns...",
			df.height()
		);
		println!(
			"   Logic: SOL spike of {}+ to reach {}+ in {} seconds = manipulation",
			self.manipulation_threshold_sol,
			self.min_curve_threshold,
			self.time_window_seconds
		);

		// Reset state
		self.blacklisted_tokens.clear();
		self.manipulation_details.clear();

		// Parse dates and sort data once for entire dataset
		println!("   📅 Converting dates and pre-processing data...");
		let df_processed = df
			.clone()
			.lazy()
			.with_columns([col("Date")
				.str()
				.strptime(
					DataType::Datetime(TimeUnit::Milliseconds, None),
					StrptimeOptions::default(),
					lit("raise"),
				)
				.alias("Date_parsed")])
			.sort(
				["Token Address", "Date_parsed"],
				SortMultipleOptions::default(),
			)
			.collect()?;

		// Process tokens in single pass (data is sorted by token)
		println!("   🔍 Processing tokens efficiently...");
		let token_col = df_processed.column("Token Address")?.str()?;

		let mut current_token = String::new();
		let mut token_rows = Vec::new();
		let mut token_count = 0;

		// Progress bar setup
		let total_rows = df_processed.height();
		let pb = tqdm::tqdm(0..total_rows);

		for row_idx in pb {
			if let Some(token_address) = token_col.get(row_idx) {
				if token_address != current_token {
					// Process previous token if we have data
					if !token_rows.is_empty()
						&& !self.blacklisted_tokens.contains(&current_token)
					{
						let _ = self.process_token_fast(
							&df_processed,
							&current_token,
							&token_rows,
						);
					}

					// Start new token
					current_token = token_address.to_string();
					token_rows.clear();
					token_count += 1;
				}

				token_rows.push(row_idx);
			}
		}

		// Process final token
		if !token_rows.is_empty()
			&& !self.blacklisted_tokens.contains(&current_token)
		{
			let _ = self.process_token_fast(
				&df_processed,
				&current_token,
				&token_rows,
			);
		}

		println!("   📊 Processed {} unique tokens", token_count);

		println!(
			"🚫 Found {} manipulated tokens",
			self.blacklisted_tokens.len()
		);
		Ok(self.blacklisted_tokens.clone())
	}

	/// Fast token processing using row indices on pre-sorted data
	fn process_token_fast(
		&mut self,
		df: &DataFrame,
		token_address: &str,
		token_rows: &[usize],
	) -> Result<()> {
		if token_rows.is_empty() {
			return Ok(());
		}

		let sol_col = df.column("SOL in Curve")?.f64()?;
		let date_col = df.column("Date_parsed")?.datetime()?;

		// Find first transaction where SOL >= min_curve_threshold
		let mut hit_threshold_idx = None;
		for &row_idx in token_rows {
			if let Some(sol) = sol_col.get(row_idx) {
				if sol >= self.min_curve_threshold {
					hit_threshold_idx = Some(row_idx);
					break;
				}
			}
		}

		let hit_threshold_idx = match hit_threshold_idx {
			Some(idx) => idx,
			None => return Ok(()), // Token never reached threshold
		};

		// Get the time and SOL when it hit threshold
		let hit_time = match date_col.get(hit_threshold_idx) {
			Some(time) => time,
			None => return Ok(()),
		};

		let hit_sol = sol_col.get(hit_threshold_idx).unwrap_or(0.0);

		// Calculate lookback time
		let check_time_millis = hit_time - (self.time_window_seconds * 1000);

		// Find best match <= check_time_millis (data is sorted by time)
		let mut earlier_sol = 0.0; // Default baseline

		// Since data is sorted, search backwards from threshold hit for efficiency
		let hit_pos = token_rows
			.iter()
			.position(|&idx| idx == hit_threshold_idx)
			.unwrap_or(0);

		for &row_idx in token_rows[..hit_pos].iter().rev() {
			if let (Some(date_millis), Some(sol)) =
				(date_col.get(row_idx), sol_col.get(row_idx))
			{
				if date_millis <= check_time_millis {
					// Found the best match - data is sorted so this is optimal
					earlier_sol = sol;
					break;
				}
			}
		}

		// Check for manipulation: if SOL increased by more than threshold
		let sol_change = hit_sol - earlier_sol;
		if sol_change >= self.manipulation_threshold_sol {
			// MANIPULATION DETECTED
			self.blacklisted_tokens.insert(token_address.to_string());

			let hit_time_str = DateTime::<Utc>::from_timestamp_millis(hit_time)
				.unwrap_or_default()
				.format("%Y-%m-%d %H:%M:%S")
				.to_string();

			self.manipulation_details.insert(
				token_address.to_string(),
				ManipulationDetails {
					sol_amount: sol_change,
					timestamp: hit_time_str.clone(),
					curve_sol: hit_sol,
				},
			);
		}

		Ok(())
	}

	/// Get blacklisted tokens
	pub fn get_blacklisted_tokens(&self) -> &HashSet<String> {
		&self.blacklisted_tokens
	}

	/// Save blacklist to file
	pub fn save_blacklist(&self, filename: &str) -> Result<()> {
		let mut file = File::create(filename)?;

		writeln!(file, "# Blacklisted tokens due to manipulation detection")?;
		writeln!(
			file,
			"# Logic: SOL spike of {}+ to reach {}+ in {} seconds = manipulation",
			self.manipulation_threshold_sol,
			self.min_curve_threshold,
			self.time_window_seconds
		)?;
		writeln!(
			file,
			"# Note: Tokens without {}+ seconds of history assume baseline of 0",
			self.time_window_seconds
		)?;
		writeln!(
			file,
			"# Generated: {}",
			chrono::Utc::now().format("%Y-%m-%d %H:%M:%S")
		)?;
		writeln!(file)?;

		let mut sorted_tokens: Vec<_> =
			self.blacklisted_tokens.iter().collect();
		sorted_tokens.sort();

		for token in sorted_tokens {
			if let Some(details) = self.manipulation_details.get(token) {
				writeln!(
					file,
					"{},{:.2},{},{:.2}",
					token,
					details.sol_amount,
					details.timestamp,
					details.curve_sol
				)?;
			} else {
				writeln!(file, "{},0.00,unknown,0.00", token)?;
			}
		}

		println!("💾 Blacklist saved to {}", filename);
		Ok(())
	}

	/// Load blacklist from file
	pub fn load_blacklist(&mut self, filename: &str) -> Result<bool> {
		match std::fs::read_to_string(filename) {
			Ok(content) => {
				for line in content.lines() {
					let line = line.trim();
					if line.is_empty() || line.starts_with('#') {
						continue;
					}

					let parts: Vec<&str> = line.split(',').collect();
					if !parts.is_empty() {
						let token = parts[0].to_string();
						self.blacklisted_tokens.insert(token.clone());

						if parts.len() >= 4 {
							if let (Ok(sol_amount), Ok(curve_sol)) = (
								parts[1].parse::<f64>(),
								parts[3].parse::<f64>(),
							) {
								self.manipulation_details.insert(
									token,
									ManipulationDetails {
										sol_amount,
										timestamp: parts[2].to_string(),
										curve_sol,
									},
								);
							}
						}
					}
				}

				println!(
					"📂 Loaded {} blacklisted tokens from {}",
					self.blacklisted_tokens.len(),
					filename
				);
				Ok(true)
			}
			Err(_) => {
				println!("⚠️  Blacklist file {} not found", filename);
				Ok(false)
			}
		}
	}
}
