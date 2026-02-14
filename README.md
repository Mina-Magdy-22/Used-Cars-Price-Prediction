# Used Cars Price Prediction

Purpose:
Build a reliable price-prediction baseline for used cars while preventing data leakage and providing a reproducible workflow.

Project summary:
This repository contains a Jupyter notebook that performs data auditing, robust preprocessing, LOOCV-style feature engineering (brand/model encoding), baseline linear modeling, and a high-performing Random Forest. Emphasis is on transparent diagnostics and CV-safe feature engineering.

Key results:
- Linear Regression (baseline): R² ≈ 0.55, MAE ≈ 1,844
- Random Forest (best): R² ≈ 0.875
- Diagnostics indicate heteroscedastic residuals; consider robust losses or target transforms for final models.

Dataset:
Autos.csv — vehicle attributes (brand, model, yearOfRegistration, powerPS, kilometer, fuelType, gearbox, notRepairedDamage, price, etc.)
Target: price

What I did (concise):
1. Data audit: types, missingness, unrealistic values (yearOfRegistration, powerPS).
2. Cleaning: dropped irrelevant columns, winsorized extreme powerPS, removed invalid/zero prices.
3. Imputation: mode for categorical; KNN/SimpleImputer for numeric when appropriate.
4. Feature engineering: age, Power_per_Age, LOOCV Brand_Model_Value (row-excluded).
5. Encoding: one-hot for low-cardinality categories; frequency/LOOCV encodings for grouped features.
6. Modeling: pipeline-based Linear Regression and Random Forest; cross-validation used for validation.
7. Diagnostics: residual plots, Durbin–Watson, Spearman correlations, OLS summary for interpretability.

Data-leakage checks & mitigation:
- Performed checks for columns containing "price" and for near-perfect correlations with the target.
- LOOCV brand/model encoding was computed excluding the current row. Important: such encodings must be computed inside each CV fold (or using CV-safe encoders) during training to avoid leakage.
Recommendations:
- Put imputation, encoding and scaling inside a Pipeline/ColumnTransformer.
- Compute any target-dependent encodings within training folds (use CV-safe encoders).
- If data is temporal, use time-aware splits (train chronologically before test).

Short checklist before finalizing:
- [ ] Ensure LOOCV / target encodings are CV-safe (computed inside folds).
- [ ] Use a held-out test set untouched during feature engineering.
- [ ] Re-run validation with nested CV or a final holdout for robust metrics.
- [ ] Document limitations (heteroscedasticity, geographic/temporal biases).
