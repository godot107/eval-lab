# Model-quality report: german_credit

1000 rows, 30% positive. Cost of a miss 5, of a false alarm 1, so the decision threshold is 0.167.

## Leaderboard (5-fold CV on the development split; 90% bootstrap intervals)

| Model | log loss | Brier | ROC AUC | PR AUC | calibration error | expected cost |
|---|---|---|---|---|---|---|
| no skill (predict the base rate) | 0.611 (0.587-0.633) | 0.210 (0.198-0.221) | 0.500 (0.500-0.500) | 0.300 (0.271-0.326) | 0.048 (0.025-0.060) | 0.700 (0.674-0.729) |
| logistic | 0.521 (0.485-0.555) | 0.174 (0.161-0.187) | 0.770 (0.740-0.796) | 0.587 (0.528-0.636) | 0.054 (0.043-0.083) | 0.537 (0.481-0.600) |
| random forest | 0.500 (0.477-0.523) | 0.165 (0.156-0.175) | 0.797 (0.769-0.824) | 0.628 (0.578-0.686) | 0.063 (0.054-0.096) | 0.515 (0.476-0.555) |
| gradient boosting | 0.589 (0.529-0.652) | 0.178 (0.163-0.196) | 0.779 (0.747-0.810) | 0.574 (0.520-0.633) | 0.098 (0.079-0.123) | 0.583 (0.510-0.655) |

## Winner on the untouched holdout (200 rows): random forest

| Metric | Value |
|---|---|
| log loss | 0.494 |
| Brier | 0.162 |
| ROC AUC | 0.801 |
| PR AUC | 0.649 |
| calibration error | 0.079 |
| expected cost | 0.550 |
| cost, approve/accept everyone | 1.500 |
| cost, reject/flag everyone | 0.700 |

At the cost threshold: 57 positives caught, 3 missed, 95 false alarms, 45 correct negatives.

## Permutation importance on the holdout (top 10)

- checking_status: +0.0753
- credit_amount: +0.0139
- duration: +0.0120
- credit_history: +0.0117
- purpose: +0.0094
- age: +0.0089
- savings_status: +0.0051
- other_payment_plans: +0.0045
- property_magnitude: +0.0042
- installment_commitment: +0.0039

![report](figures/l8_german_credit_report.png)
