To run:

Bins are located in the /bin directory. these test bins run control our strategy parameters, run the actual strategies and display strategy performace. Make sure to run on --release for speed. you'll need to update the location of the data file in each bin, relative to where you have it on your computer.

cargo run --bin (bin name) --release

example: cargo run --bin test_vwap --release

We cannot change our common parameters, as these are fixed for backtesting purposes with fee structure, latency, etc. They are located in the strategies/common_config.rs file.

When you create a new strategy, please use the same structure as the existing ones (vwap.rs or zscore.rs). We always need to use the buy and sell contraints, which you'll need to include the should_buy and should_sell functions in the actual strategies strategies.

For parameter optimiztion, look at the vwap_param_optimizer.rs or zscore_param_optimizer.rs as examples. just copy the files, swap out what you need to to run your strategy and it's parameter ranges for testing.

Please do not edit or change the backtesting framework. If you think there is an error or bug in the system, please let me know and we'll look into it. The only thing you might want to change is where csv files get exported to in the backtesting framework. 

Two files are created when you run a strategy, an equity curve csv and a merged trades csv. You can 

You have full creative freedom! Our goal is have a reasonably steady equity curve, maximize return with minimizing drawdown. A strategy should return thousands of trades. Any strategy with fewer than 500 trades should be reworked. Too few trades will just result in overfitting. For instance, we usually trade about 5k-20k trades on this data set when backtesting.

FYI - most of the bots we see that are successful employ a combination of mean reversion and momentum in combination. I don't want to steer you in any direction, just let you know.

Please let me know when you have any questions. 

