/*!
Example: Configurable Take Profit Parameter
==========================================

This example demonstrates how to use the new configurable take profit parameter
in the Order Flow Imbalance strategy instead of relying on the stop_loss_pct parameter.
*/

use anyhow::Result;

fn main() -> Result<()> {
    println!("🎯 Configurable Take Profit Parameter Example");
    println!("=============================================");
    
    println!("\n📊 Before (Hardcoded):");
    println!("  • Stop Loss: 10%");
    println!("  • Take Profit: 40% (hardcoded as stop_loss_pct * 4.0)");
    println!("  • Risk:Reward Ratio: 1:4");
    
    println!("\n✅ After (Configurable):");
    println!("  • Stop Loss: 10%");
    println!("  • Take Profit: 30% (configurable TAKE_PROFIT_PCT)");
    println!("  • Risk:Reward Ratio: 1:3");
    
    println!("\n🔧 Configuration in test_order_flow_imbalance.rs:");
    println!("  const STOP_LOSS_PCT: f64 = 0.10;     // 10% stop loss");
    println!("  const TAKE_PROFIT_PCT: f64 = 0.30;  // 30% take profit");
    
    println!("\n📈 Strategy Constructor Changes:");
    println!("  OrderFlowImbalanceStrategy::new(");
    println!("      // ... other parameters ...");
    println!("      STOP_LOSS_PCT,     // stop_loss_pct");
    println!("      TAKE_PROFIT_PCT,   // take_profit_pct (NEW!)");
    println!("      // ... other parameters ...");
    println!("  );");
    
    println!("\n💡 Benefits:");
    println!("  • Independent control over stop loss and take profit");
    println!("  • Flexible risk:reward ratios");
    println!("  • Easy parameter optimization");
    println!("  • Clear separation of risk management parameters");
    
    println!("\n🎛️ Example Configurations:");
    println!("  Conservative: Stop Loss 5%, Take Profit 10% (1:2 ratio)");
    println!("  Moderate:    Stop Loss 10%, Take Profit 30% (1:3 ratio)");
    println!("  Aggressive:  Stop Loss 15%, Take Profit 60% (1:4 ratio)");
    
    println!("\n✅ To test with your own parameters:");
    println!("  1. Edit TAKE_PROFIT_PCT in test_order_flow_imbalance.rs");
    println!("  2. Run: cargo run --bin test_order_flow_imbalance");
    println!("  3. Compare results with different take profit levels");
    
    Ok(())
}
