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

To add when lessons 5 and 6 are built (verify before citing): Cawley and Talbot (2010) on selection bias in
model evaluation; Nadeau and Bengio (2003) on the corrected resampled t-test; Dietterich (1998) on statistical tests
for comparing classifiers.

**Where this lab differs from the usual tutorial:** tutorials show *how* to call `cross_val_score`. Here every
evaluation method is scored against the truth it estimates, so "how wrong can this be?" gets a number.
