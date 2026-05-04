# Optimal Battery Dispatch via Stochastic Energy Market Modelling

This project develops a stochastic optimization framework for optimal battery charging and discharging under uncertain electricity prices, renewable generation, and load demand. The study uses hourly data from the German DE/LU electricity market and focuses on minimizing total electricity purchase cost while respecting practical battery constraints.

## Project Overview

Electricity prices, wind generation, solar output, and load demand fluctuate significantly over time. These uncertainties make battery scheduling a complex decision problem. This project models these uncertainties statistically and then solves a battery dispatch optimization problem to determine when the battery should charge, discharge, or remain idle.

The main objective is to reduce electricity cost through energy arbitrage: charging the battery when electricity prices are low and discharging when prices are high.

## Data

The analysis uses hourly electricity market data from the German DE/LU market, including:

- Day-ahead electricity price
- Load demand
- Wind generation
- Solar generation

The dataset was cleaned, scaled, and converted into a consistent hourly time-series format suitable for stochastic modelling and optimization. :contentReference[oaicite:0]{index=0}

## Stochastic Models Used

The project models uncertainty using the following distributions and processes:

- **Electricity Price:** Ornstein-Uhlenbeck mean-reverting process
- **Wind Generation:** OU dynamics with Weibull marginal distribution
- **Load Demand:** Hourly Normal distribution
- **Solar Generation:** Hourly Beta distribution

These models are used to estimate expected price, load, wind, and solar values for the battery dispatch optimization.

## Battery Dispatch Optimization

The optimization model determines the hourly battery schedule subject to:

- Battery energy capacity limits
- Charging and discharging power limits
- Charging and discharging efficiency losses
- State-of-charge balance
- Initial and terminal battery state constraints
- No simultaneous charging and discharging

The model was first formulated as a Linear/Mixed-Integer Linear Programming problem for a one-day horizon. For the one-month horizon, a finite-horizon dynamic programming approach was used to improve computational efficiency.

## Key Results

The one-month comparison shows that battery dispatch reduces electricity cost across all tested scenarios. Although percentage savings are small because they are measured against very large monthly electricity costs, the absolute savings are economically significant.

The reported savings range from approximately:

- **€1.25 million to €1.98 million**
- **0.11% to 0.21% of total monthly cost**

The highest savings occur under the real historical realised path because actual market prices contain stronger volatility and larger price spreads, creating better opportunities for buy-low/sell-high battery arbitrage.

## Conclusion

This project demonstrates that a stochastic modelling framework can support effective battery scheduling under uncertain energy market conditions. Even with simplified assumptions, the battery dispatch model produces meaningful cost reductions while respecting realistic operational constraints.

The results also show that expected-value optimization gives stable but conservative performance, while realised historical price paths can generate higher arbitrage profits due to stronger price fluctuations.

## Limitations

The model is intentionally simplified. It does not include battery degradation cost, transaction fees, reserve-market revenue, network constraints, price jumps, regime changes, or full scenario-based stochastic optimization. The analysis is also limited to a one-month evaluation horizon.

## Future Work

Possible extensions include:

- Scenario-based stochastic optimization
- Battery degradation cost modelling
- Jump-diffusion or regime-switching price models
- Multi-market participation
- Weather-driven renewable forecasting
- Full-year backtesting
- Reinforcement learning for adaptive battery scheduling
