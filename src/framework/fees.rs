pub fn get_fee_percent(token_price: f64) -> f64 {
    // Calculate marketcap: token_price * 1 billion, rounded to 2 decimals
    let market_cap = (token_price * 1_000_000_000.0 * 100.0).round() / 100.0;
    
    if market_cap <= 420.0 {
        0.01250 // 1.250%
    } else if market_cap <= 1470.0 {
        0.01200 // 1.200%
    } else if market_cap <= 2460.0 {
        0.01150 // 1.150%
    } else if market_cap <= 3440.0 {
        0.01100 // 1.100%
    } else if market_cap <= 4420.0 {
        0.01050 // 1.050%
    } else if market_cap <= 9820.0 {
        0.01000 // 1.000%
    } else if market_cap <= 14740.0 {
        0.00950 // 0.950%
    } else if market_cap <= 19650.0 {
        0.00900 // 0.900%
    } else if market_cap <= 24560.0 {
        0.00850 // 0.850%
    } else if market_cap <= 29470.0 {
        0.00800 // 0.800%
    } else if market_cap <= 34380.0 {
        0.00750 // 0.750%
    } else if market_cap <= 39300.0 {
        0.00700 // 0.700%
    } else if market_cap <= 44210.0 {
        0.00650 // 0.650%
    } else if market_cap <= 49120.0 {
        0.00600 // 0.600%
    } else if market_cap <= 54030.0 {
        0.00550 // 0.550%
    } else if market_cap <= 58940.0 {
        0.00525 // 0.525%
    } else if market_cap <= 63860.0 {
        0.00500 // 0.500%
    } else if market_cap <= 68770.0 {
        0.00475 // 0.475%
    } else if market_cap <= 73681.0 {
        0.00450 // 0.450%
    } else if market_cap <= 78590.0 {
        0.00425 // 0.425%
    } else if market_cap <= 83500.0 {
        0.00400 // 0.400%
    } else if market_cap <= 88400.0 {
        0.00375 // 0.375%
    } else if market_cap <= 93330.0 {
        0.00350 // 0.350%
    } else if market_cap <= 98240.0 {
        0.00325 // 0.325%
    } else {
        0.00300 // 0.300%
    }
}
