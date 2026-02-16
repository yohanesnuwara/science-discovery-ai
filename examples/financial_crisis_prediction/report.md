## Data integration plan

IMPLEMENTATION ROADMAP:
Step 1: ESRB Processing (Solve Challenge #1, #6, #9)
- Parse dates: pd.to_datetime() with errors='coerce'
- Create annual labels: Country by year match
- Filter crisis types: crisis_onset = (year between start and end)
Step 2: JST Processing (Solve Challenge #2, #4, #5, #7)
- Column mapping: Use priority system for feature mappings
- Missing data: 3-stage imputation (forecast, interpolate, backward-fill)
- Sequence building: Precedes window for causality
Step 3: GPR Processing (Solve Challenge #3, #10)
- Monthly to annual: Aggregate per quarter/year
- Critical window detection: Use ESRB crisis start years
- Lag analysis: For year Y, use Y-1, Y-2, Y-3
Step 4: Integration (Solve Challenge #11, #12)
- Align by country-year: Merge all datasets on (country, year)
- Time-based splits: Split years, NOT by observation ratio
- Weighted loss: class_weights = [1, balance_factor]
Step 5: Explainability (Solve Challenge #13, #15)
- Feature importance: Attention weights from TCN
- Country-specific: Per-country prediction metrics
- SHAP-style: Feature contribution analysis
KEY FIXES:
1. Explicit mappings: Don't rely on partial name matching
2. Causal sequences: Always use PREVIOUS years for prediction
3. Validation safety: Exclude crisis years from validation set
4. Imputation strategy: Multiple-stage for historical data
5. Output standardization: Unified (country, year) format
Would you like me to implement this complete solution?