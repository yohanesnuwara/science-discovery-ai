# Form template for research input for the agent

## Objective of research

As a **quantitative researcher** in **statistical learning / time-series modeling**,
I want to **study and forecast the probability and timing of next banking/financial crisis onsets**, incorporating macro-financial cycles and geopolitical risk.

My objectives are: 

- [ ] Build a **crisis-onset label** (annual, country-level) from the **ESRB financial crises database** and align it with predictors.
- [ ] Construct a **country-year panel with multi-year sequences** from the **Jordà–Schularick–Taylor (JST) Macrohistory Database** plus annualized features from the **Geopolitical Risk (GPR) index**.
- [ ] Train and evaluate a **GPU-based Temporal Convolutional Network (TCN)** to estimate **crisis-onset risk** (and optionally “within-h-years” risk), including interpretable diagnostics on key drivers.

## Data input

I have the following data: 

- [ ] **European Financial Crises Data** (filepath `/workspace/science-discovery-ai/.input/esrb.fcdb20220120.en.csv`)
- [ ] **Jordà–Schularick–Taylor (JST) Macrohistory Data** (filepath `/workspace/science-discovery-ai/.input/JSTdatasetR6.csv`)
- [ ] **Caldara & Iacoviello Geopolitical Risk (GPR) dataset** (filepath `/workspace/science-discovery-ai/.input/data_gpr_export.csv`)

The CSV contains column names. Read **NOTES** at the end of the form.

## Planned methodology

This is my preferred methodology

**Primary approach: GPU-based Temporal Convolutional Network (TCN)** for rare-event prediction in a country-year panel.

1) **Label construction (ESRB)**
- Create annual targets:
  - **Onset label**: `y[c,t]=1` if a crisis starts in year `t` for country `c`, else 0.
  - Optionally create **in-crisis state** `s[c,t]` for filtering “at-risk” years.
- Train on **at-risk** samples (exclude years already in crisis when predicting onset).

2) **Feature engineering (JST + GPR)**
- JST: use macro-financial predictors and engineered transforms (YoY growth, acceleration, rolling volatility, slope `ltrate-stir`, etc.).
- GPR: convert monthly series to annual features (mean, max, volatility, tail-count) and **lag** them to avoid leakage.
- Optional: include country-share GPR series (`GPRC_*` or `GPRHC_*`) as country-differentiated geopolitics exposure (lagged).

3) **Sequence building**
- For each country-year (c,t), build an input sequence window of length `L` years:
  - `X[c,t] = [z[c,t-L], …, z[c,t-1]]`
- Predict `y[c,t]` with a causal TCN (no future information).

4) **Training + evaluation**
- Loss: **weighted BCE** or **focal loss** for class imbalance.
- Validation: **time-based splits** (blocked CV by year) to prevent look-ahead bias.
- Metrics: **AUCPR**, Brier score (calibration), Precision@k, and lead-time performance.

You can use these equations if you want 

- **Discrete-time hazard formulation (onset risk)**
  
  \[
  p_{c,t} = \Pr(y_{c,t}=1 \mid \text{at risk}) = \sigma(f_\theta(X_{c,t}))
  \]
  
  where \(f_\theta(\cdot)\) is the TCN mapping a length-\(L\) history into an onset probability.

- **Annualization of monthly GPR**
  
  \[
  \text{GPR\_mean}_t = \frac{1}{12}\sum_{m=1}^{12} \text{GPR}_{t,m},\quad
  \text{GPR\_max}_t = \max_m \text{GPR}_{t,m}
  \]
  
  \[
  \text{GPR\_tailcount}_t = \sum_{m=1}^{12}\mathbb{1}\{\text{GPR}_{t,m}>\text{P90}(\text{GPR})\}
  \]
  
  (P90 computed on training period only; use lagged versions for prediction.)

## Expected outcome

I want to make a forecast of next financial crisis:

- **Crisis risk over time** (predicted probability vs. actual onset) by country and aggregated over regions.
- **Top-risk country-years** (ranked risk) and how this changes with GPR spikes.
- **Driver diagnostics** (feature importance via ablation or permutation; optional saliency over time steps).

And the output data of:

- `predicted_risk_country_year.csv` (country, year, predicted probability, label, split)
- `gpr_annual_features.csv` (year, aggregated GPR features)
- `model_eval_metrics.csv` (AUCPR, Brier, Precision@k by split)

in **CSV** format, plus visualizations saved as **PNG**.

Also create a brief research report in **PDF** and provide the research log / summary in **Markdown**.

---

## Notes

### A) Jordà–Schularick–Taylor (JST) Macrohistory Database — column descriptions (as shared)

- `year`: Year  
- `country`: Country  
- `iso`: ISO 3-letter code  
- `ifs`: IFS 3-number country-code  
- `pop`: Population  
- `rgdpmad`: Real GDP per capita (PPP, 1990 Int$, Maddison)  
- `rgdbarro`: Real GDP per capita (index, 2005=100)  
- `rconsbarro`: Real consumption per capita (index, 2006=100)  
- `gdp`: GDP (nominal, local currency)  
- `iy`: Investment-to-GDP ratio  
- `cpi`: Consumer prices (index, 1990=100)  
- `ca`: Current account (nominal, local currency)  
- `imports`: Imports (nominal, local currency)  
- `exports`: Exports (nominal, local currency)  
- `narrowm`: Narrow money (nominal, local currency)  
- `money`: Broad money (nominal, local currency)  
- `stir`: Short-term interest rate (nominal, percent per year)  
- `ltrate`: Long-term interest rate (nominal, percent per year)  
- `hpnnom`: House prices (nominal index, 1990=100)  
- `unemp`: Unemployment rate (percent)  
- `wage`: Wages (index, 1990=100)  
- `debtgdp`: Public debt-to-GDP ratio  
- `revenue`: Government revenues (nominal, local currency)  
- `expenditure`: Government expenditure (nominal, local currency)  
- `xrusd`: USD exchange rate (local currency/USD)  
- `peg`: Peg dummy  
- `peg_strict`: Strict peg dummy  
- `crisisJST`: Systemic financial crises (0-1 dummy); included since R5  
- `crisisJST_old`: Systemic financial crises (0-1 dummy); as coded in all prior releases (R1–R4)  
- `JSTtrilemmaIV`: JST trilemma instrument (raw base rate changes)  
- `tloans`: Total loans to non-financial private sector (nominal, local currency)  
- `tmort`: Mortgage loans to non-financial private sector (nominal, local currency)  
- `thh`: Total loans to households (nominal, local currency)  
- `tbus`: Total loans to business (nominal, local currency)  
- `bdebt`: Corporate debt (nominal, local currency)  
- `peg_type`: Peg type (BASE, PEG, FLOAT)  
- `peg_base`: Peg base (GBR, USA, DEU, HYBRID, NA)  
- `eq_tr`: Equity total return, nominal. r[t] = [[p[t] + d[t]] / p[t-1]] - 1  
- `housing_tr`: Housing total return, nominal. r[t] = [[p[t] + d[t]] / p[t-1]] - 1  
- `bond_tr`: Government bond total return, nominal. r[t] = [[p[t] + coupon[t]] / p[t-1]] - 1  
- `bill_rate`: Bill rate, nominal. r[t] = coupon[t] / p[t-1]  
- `rent_ipolated`: 1 if housing rental yields interpolated e.g. wartime  
- `housing_capgain_ipolated`: 1 if housing capital gains and total returns interpolated e.g. wartime  
- `housing_capgain`: Housing capital gain, nominal. cg[t] = [p[t] / p[t-1]] - 1  
- `housing_rent_rtn`: Housing rental return. dp_rtn[t] = rent[t] / p[t-1]  
- `housing_rent_yd`: Housing rental yield. dp[t] = rent[t] / p[t]  
- `eq_capgain`: Equity capital gain, nominal. cg[t] = [p[t] / p[t-1]] - 1  
- `eq_dp`: Equity dividend yield. dp[t] = dividend[t] / p[t]  
- `eq_capgain_interp`: 1 if equity cap. gain interpolated to cover exchange closure  
- `eq_tr_interp`: 1 if equity total return interpolated to cover exchange closure  
- `eq_dp_interp`: 1 if equity dividend interpolated or assumed zero to cover exchange closure  
- `bond_rate`: Gov. bond rate, rate[t] = coupon[t] / p[t-1], or yield to maturity at t  
- `eq_div_rtn`: Equity dividend return. dp_rtn[t] = dividend[t] / p[t-1]  
- `capital_tr`: Tot. rtn. on wealth, nominal. Wtd. avg. of housing, equity, bonds and bills  
- `risky_tr`: Tot. rtn. on risky assets, nominal. Wtd. avg. of housing and equity  
- `safe_tr`: Tot. rtn. on safe assets, nominal. Equally wtd. avg. of bonds and bills  
- `lev`: Banks, capital ratio (in %)  
- `ltd`: Banks, loans-to-deposits ratio (in %)  
- `noncore`: Banks, noncore funding ratio (in %)  

### B) Caldara & Iacoviello Geopolitical Risk (GPR) dataset — column descriptions (as shared)

- `month`: Date (year/month)  
- `GPR`: Recent GPR (Index: 1985:2019=100)  
- `GPRT`: Recent GPR Threats (Index: 1985:2019=100)  
- `GPRA`: Recent GPR Acts (Index: 1985:2019=100)  
- `GPRH`: Historical GPR (Index: 1900:2019=100)  
- `GPRHT`: Historical GPR Threats (Index: 1900:2019=100)  
- `GPRHA`: Historical GPR Acts (Index: 1900:2019=100)  
- `SHARE_GPR`: Percent of Recent GPR Articles  
- `N10`: Number of articles (10 recent newspapers, 1985-)  
- `SHARE_GPRH`: Percent of Historical GPR Articles  
- `N3H`: Number of articles (3 historical newspapers, 1900-)  
- `GPRH_NOEW`: Historical GPR Index including Excluded Words (1900-2019=100)  
- `GPR_NOEW`: Recent GPR Index including Excluded Words (1985-2019=100)  
- `GPRH_AND`: Historical GPR Index (broader search criteria) replacing N/2 with AND (1900-2019=100)  
- `GPR_AND`: Recent GPR Index (broader search criteria) replacing N/2 with AND (1985-2019=100)  
- `GPRH_BASIC`: Historical Basic GPR Index (narrow search criteria) (1900-2019=100)  
- `GPR_BASIC`: Recent Basic GPR Index (narrow search criteria) (1985-2019=100)  
- `SHAREH_CAT_1`: Share of articles Cat.1  
- `SHAREH_CAT_2`: Share of articles Cat.2  
- `SHAREH_CAT_3`: Share of articles Cat.3  
- `SHAREH_CAT_4`: Share of articles Cat.4  
- `SHAREH_CAT_5`: Share of articles Cat.5  
- `SHAREH_CAT_6`: Share of articles Cat.6  
- `SHAREH_CAT_7`: Share of articles Cat.7  
- `SHAREH_CAT_8`: Share of articles Cat.8  

**Country GPR (Recent): Percent of articles**
- `GPRC_ARG`: Argentina  
- `GPRC_AUS`: Australia  
- `GPRC_BEL`: Belgium  
- `GPRC_BRA`: Brazil  
- `GPRC_CAN`: Canada  
- `GPRC_CHE`: Switzerland  
- `GPRC_CHL`: Chile  
- `GPRC_CHN`: China  
- `GPRC_COL`: Colombia  
- `GPRC_DEU`: Germany  
- `GPRC_DNK`: Denmark  
- `GPRC_EGY`: Egypt  
- `GPRC_ESP`: Spain  
- `GPRC_FIN`: Finland  
- `GPRC_FRA`: France  
- `GPRC_GBR`: United Kingdom  
- `GPRC_HKG`: Hong Kong  
- `GPRC_HUN`: Hungary  
- `GPRC_IDN`: Indonesia  
- `GPRC_IND`: India  
- `GPRC_ISR`: Israel  
- `GPRC_ITA`: Italy  
- `GPRC_JPN`: Japan  
- `GPRC_KOR`: South Korea  
- `GPRC_MEX`: Mexico  
- `GPRC_MYS`: Malaysia  
- `GPRC_NLD`: Netherlands  
- `GPRC_NOR`: Norway  
- `GPRC_PER`: Peru  
- `GPRC_PHL`: Philippines  
- `GPRC_POL`: Poland  
- `GPRC_PRT`: Portugal  
- `GPRC_RUS`: Russia  
- `GPRC_SAU`: Saudi Arabia  
- `GPRC_SWE`: Sweden  
- `GPRC_THA`: Thailand  
- `GPRC_TUN`: Tunisia  
- `GPRC_TUR`: Turkey  
- `GPRC_TWN`: Taiwan  
- `GPRC_UKR`: Ukraine  
- `GPRC_USA`: United States  
- `GPRC_VEN`: Venezuela  
- `GPRC_VNM`: Vietnam  
- `GPRC_ZAF`: South Africa  

**Country GPR (Historical): Percent of articles**
- `GPRHC_ARG`: Argentina  
- `GPRHC_AUS`: Australia  
- `GPRHC_BEL`: Belgium  
- `GPRHC_BRA`: Brazil  
- `GPRHC_CAN`: Canada  
- `GPRHC_CHE`: Switzerland  
- `GPRHC_CHL`: Chile  
- `GPRHC_CHN`: China  
- `GPRHC_COL`: Colombia  
- `GPRHC_DEU`: Germany  
- `GPRHC_DNK`: Denmark  
- `GPRHC_EGY`: Egypt  
- `GPRHC_ESP`: Spain  
- `GPRHC_FIN`: Finland  
- `GPRHC_FRA`: France  
- `GPRHC_GBR`: United Kingdom  
- `GPRHC_HKG`: Hong Kong  
- `GPRHC_HUN`: Hungary  
- `GPRHC_IDN`: Indonesia  
- `GPRHC_IND`: India  
- `GPRHC_ISR`: Israel  
- `GPRHC_ITA`: Italy  
- `GPRHC_JPN`: Japan  
- `GPRHC_KOR`: South Korea  
- `GPRHC_MEX`: Mexico  
- `GPRHC_MYS`: Malaysia  
- `GPRHC_NLD`: Netherlands  
- `GPRHC_NOR`: Norway  
- `GPRHC_PER`: Peru  
- `GPRHC_PHL`: Philippines  
- `GPRHC_POL`: Poland  
- `GPRHC_PRT`: Portugal  
- `GPRHC_RUS`: Russia  
- `GPRHC_SAU`: Saudi Arabia  
- `GPRHC_SWE`: Sweden  
- `GPRHC_THA`: Thailand  
- `GPRHC_TUN`: Tunisia  
- `GPRHC_TUR`: Turkey  
- `GPRHC_TWN`: Taiwan  
- `GPRHC_UKR`: Ukraine  
- `GPRHC_USA`: United States  
- `GPRHC_VEN`: Venezuela  
- `GPRHC_VNM`: Vietnam  
- `GPRHC_ZAF`: South Africa  

**Important implementation note:** When using ESRB crisis periods, convert episode start/end dates into annual labels and ensure predictors (JST and annualized GPR) are **lagged** to avoid label leakage.