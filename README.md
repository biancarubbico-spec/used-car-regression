# Used-car auction regression

Bianca Rubbico — Projects in Programming and AI, individual regression assignment.

Predict retail sale prices from vehicle characteristics, then buy only when predicted
retail proceeds exceed the auction price plus the prompt's $1,500 preparation cost.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analysis.py
```

The input CSVs are the course-provided teaching datasets. There are 240 past sales
(the prompt describes them as the last year's sales) and 40 separate auction lots.
Exact calendar dates and whether the teaching data were synthesized are not specified.

The script holds out 72 sales using random seed 42 and fits only the other 168.
It compares an OLS model with all five features against one without Num_Owners,
using training p-values and adjusted R-squared, never test performance, to select v1.
Accident is encoded yes=1 and no=0. The baseline predicts the training-set mean.
The auction lots are never used to fit or select the model. Later_Sale_Price is used
only after decisions to calculate realized profit.

`results/report.json` contains all reported statistics; `results/ols_summary.txt`
contains coefficient estimates, p-values, and full regression diagnostics.
`results/test_predictions.csv` and `results/auction_decisions.csv` allow row-level checks.
The chart is generated directly by the analysis script.

Profit is retail sale price minus auction price minus $1,500, summed only for cars
the rule buys. The comparison assumes all selected cars can be purchased at the
listed winning price, no capital constraint, and no further costs beyond the stated
$1,500. Auction results are a retrospective simulation, not a guaranteed forecast.
