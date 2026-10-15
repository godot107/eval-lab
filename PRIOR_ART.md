# Prior art and references

Found through textbook-kb (Willie's library) on 2026-10-15. Section-level citations.

| Topic | Source | What this lab takes from it |
|---|---|---|
| Wrong vs right way to cross-validate | Hastie, Tibshirani, Friedman, *Elements of Statistical Learning* (2009), 7.10.2 | Lesson 4a reproduces the experiment: 50 samples, 5,000 noise features, 100 selected, 1-NN |
| Bootstrap and the .632 estimator | ESL 7.11 | Lesson 3's bootstrap estimator |
| Conditional vs expected test error | ESL 7.12 | Lesson 3's headline: CV barely tracks the error of the model actually fitted |
| Nested cross-validation | Witten, Frank, Hall, Pal, *Data Mining* (4th ed.), ch. 5; Albon, *ML with Python Cookbook* | Lesson 5 |
| Bootstrap optimism correction; Brier and log scores as proper scoring rules | Harrell, *Regression Modeling Strategies*, ch. 5 | Lessons 1 and 3 |
| Purged and embargoed K-fold | Lopez de Prado, *Advances in Financial Machine Learning* (2018), ch. 7 | Lesson 4c |
| Regression error metrics | Jansen, *ML for Algorithmic Trading* (2nd ed.), ch. 6 | Lesson 2 |
| Stratified and grouped K-fold in scikit-learn | Guido and Muller, *Introduction to ML with Python*, ch. 5; scikit-learn user guide, "Cross-validation" (the link on the Trello card) | Lessons 3 and 4b |

| Conformalized quantile regression | Romano, Patterson, Candes, "Conformalized Quantile Regression", NeurIPS 2019 | Lesson 2c |

| Testing whether one classifier beats another; the resampled t-test's inflated false positives; 5x2cv | Dietterich, "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms", *Neural Computation* 10(7), 1998 | Lesson 6 |
| Corrected resampled t-test | Nadeau and Bengio, "Inference for the Generalization Error", *Machine Learning* 52, 2003 | Lesson 6 |
| Learning and validation curves | Guido and Muller ch. 5; scikit-learn user guide | Lesson 7 |
| AutoML model-quality report | SageMaker Autopilot model quality reports (AWS ML blog, linked on the Trello card) | Lesson 8 |

**Where this lab differs from the usual tutorial:** tutorials show *how* to call `cross_val_score`. Here every
evaluation method is scored against the truth it estimates, so "how wrong can this be?" gets a number.
