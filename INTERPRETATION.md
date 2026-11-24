# Interpretation of v1

OLS uses Sale_Price in dollars as the outcome. Accident is 1 for yes, 0 for no.
The fixed split is 168 training sales and 72 test sales (random_state=42).

The five-feature training model has adjusted R² 0.869054. Num_Owners has
p=0.865346. Removing it gives adjusted R² 0.869835, so the four-feature model
is selected as v1. This selection uses training evidence only.

Holding the other included characteristics fixed:

| Feature | Estimated price change | Training p-value |
| --- | ---: | ---: |
| Mileage_K | -$78.15 per additional 1,000 miles | 4.32e-8 |
| Age_Years | -$1,291.63 per additional year | 1.16e-11 |
| Horsepower | +$75.02 per horsepower | 7.79e-35 |
| Accident | -$2,176.64 for an accident history | 3.25e-5 |

These are conditional associations in this teaching dataset, not causal effects.
The intercept ($21,762.40) is the algebraic prediction at zero mileage, age, and
horsepower without an accident; zero horsepower is outside ordinary business use.

On held-out cars, baseline R²=-0.013678 and RMSE=$6,494.15; v1 R²=0.822679
and RMSE=$2,716.14. A negative baseline R² is possible because its training-set
mean differs from the test-set mean. RMSE is the square root of mean squared
prediction error, in dollars; it is not mean absolute error or a guaranteed range.

On the 40 independent auction lots, baseline buys 14 cars, of which 10 lose money,
for -$6,000 total profit. v1 buys 17 cars, of which 6 lose money, for $15,800 total
profit. Incremental profit is $15,800 - (-$6,000) = $21,800 for this auction.
The model buys more cars and ties up more capital; there is no budget constraint
in this simulation. No annual extrapolation is justified by this single auction.

The rule uses a strict comparison: buy if Auction_Price + 1500 < predicted price;
pass at equality. Lot 1 is a six-year-old, 71,000-mile, 120-HP car with no reported
accident and three owners. Its $21,500 auction price implies $23,000 all-in cost.
v1 predicts $17,466.17 and passes. The baseline predicts $25,851.19 and buys.
The model's break-even maximum bid is $15,966.17; it must bid below that to have
positive predicted profit under the exact assignment rule.

Deployment recommendation: use v1 as decision support and monitor realized error.
It still buys six money-losing cars. The data do not include model, make, local
demand or condition beyond accident status. More auction validation would be
needed before relying on the rule in a real dealership.
