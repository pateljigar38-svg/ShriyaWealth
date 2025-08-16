You are an expert mutual fund analyst with 35+ years of experience and a proven record of consistently generating 20%+ XIRR while ensuring no fund underperforms its category return.

Build a mutual fund selection and backtesting model with the following rules:

🔹 Backtesting & Accuracy

Use historical AMFI NAV data for backtesting.

Any recommendation must achieve ≥95% accuracy when validated against actual returns over subsequent periods.

Example: If JM Flexicap was suggested in 2022 for 20%+ XIRR, validation in 2025 should show ≥95% alignment.


🔹 Return Guardrails

In bear markets, minimum XIRR ≥ 13%.

In bull markets, minimum XIRR ≥ 18%.

At no point should a recommended fund underperform its category average.


🔹 Ranking & Selection

Recommend only funds in the Top 5 performers across 5Y, 3Y, 1Y, 6M, and 3M returns.

Categories covered: Large Cap, Mid Cap, Small Cap, Flexicap, Thematic, Hybrid, International.

Allocation Rules:

Avoid over-diversification (max 7 funds per portfolio).

Avoid over-concentration (no single fund >30% allocation).

Conservative clients → max 3 funds.

Aggressive clients → up to 7 funds with balanced exposure.



🔹 Outputs Required

1. Recommended funds with rank, returns, and comparison vs. category.


2. Historical backtest validation with accuracy %.


3. SIP outcome simulator (₹1,000–₹1,00,000, adjustable tenor).


4. Risk-adjusted metrics (Sharpe ratio, volatility, max drawdown).


5. Projection tables for SIP & Lump Sum across 1Y, 3Y, 5Y, 10Y.


6. Export options: Google Sheet + one-click PDF report.



🔹 Default Settings

Default plan: Regular – Growth Option.

Allow toggle to Dividend Option if required by client.


provide input python file and output file with all above logic
