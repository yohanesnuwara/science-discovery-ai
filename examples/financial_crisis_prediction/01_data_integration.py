"""
Financial Crisis Forecasting - COMPREHENSIVE APPROACH
Trying 4 different target strategies:
1. Multi-label crisis type classification
2. Time-to-crisis regression
3. Crisis severity prediction
4. Multi-year ahead prediction
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import precision_recall_curve, auc, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("=" * 70)

# ============================================================
# 1. LOAD AND PROCESS ALL DATA
# ============================================================

print("\n=== LOADING DATA ===")

# Load datasets
jst = pd.read_csv(".input/JSTdatasetR6.csv")
esrb = pd.read_csv(".input/esrb.fcdb20220120.en.csv")
gpr = pd.read_csv(".input/data_gpr_export.csv")

print(f"JST: {jst.shape}")
print(f"ESRB: {esrb.shape}")
print(f"GPR: {gpr.shape}")

# Parse dates
esrb["Start date"] = pd.to_datetime(esrb["Start date"], format="%Y-%m", errors="coerce")
esrb["year"] = esrb["Start date"].dt.year
gpr["date"] = pd.to_datetime(gpr["month"], format="%d.%m.%Y", errors="coerce")
gpr["year"] = gpr["date"].dt.year

# Country code mapping
iso_map = {
    "AT": "AUT",
    "BE": "BEL",
    "BG": "BGR",
    "CY": "CYP",
    "CZ": "CZE",
    "DE": "DEU",
    "DK": "DNK",
    "EE": "EST",
    "EL": "GRC",
    "ES": "ESP",
    "FI": "FIN",
    "FR": "FRA",
    "HR": "HRV",
    "HU": "HUN",
    "IE": "IRL",
    "IT": "ITA",
    "LT": "LTU",
    "LU": "LUX",
    "LV": "LVA",
    "MT": "MLT",
    "NL": "NLD",
    "PL": "POL",
    "PT": "PRT",
    "RO": "ROU",
    "SE": "SWE",
    "SI": "SVN",
    "SK": "SVK",
    "UK": "GBR",
    "US": "USA",
}
esrb["iso"] = esrb["Country"].map(iso_map)

# ============================================================
# 2. CREATE MULTIPLE TARGETS
# ============================================================

print("\n=== CREATING TARGETS ===")

# Sort JST
jst = jst.sort_values(["country", "year"]).reset_index(drop=True)

# Target 1: Multi-label crisis types (from ESRB)
crisis_types = [
    "Banking",
    "Significant asset price correction",
    "Currency / BoP / Capital flow",
    "Sovereign",
]

for ctype in crisis_types:
    jst[f"crisis_{ctype.replace(' / ', '_').replace(' ', '_')}"] = 0

# Map ESRB crisis types to JST
for _, row in esrb.iterrows():
    if pd.notna(row["year"]) and pd.notna(row["iso"]):
        mask = (jst["iso"] == row["iso"]) & (jst["year"] == row["year"])
        for ctype in crisis_types:
            if ctype in esrb.columns and pd.notna(row[ctype]):
                jst.loc[
                    mask, f"crisis_{ctype.replace(' / ', '_').replace(' ', '_')}"
                ] = row[ctype]

# Also use JST crisis as backup
jst["crisis_any"] = jst["crisisJST"]

# Target 2: Time-to-crisis (regression)
print("Creating time-to-crisis target...")
jst["years_to_crisis"] = np.nan
for country in jst["country"].unique():
    mask = jst["country"] == country
    country_df = jst[mask].sort_values("year")
    crisis_years = country_df[country_df["crisisJST"] == 1]["year"].values

    years_to_crisis = []
    for year in country_df["year"]:
        future_crises = crisis_years[crisis_years > year]
        if len(future_crises) > 0:
            years_to_crisis.append(future_crises[0] - year)
        else:
            years_to_crisis.append(10)  # Cap at 10 years
    jst.loc[mask, "years_to_crisis"] = years_to_crisis

# Target 3: Crisis severity (using GDP decline during crisis)
print("Creating crisis severity target...")
jst["crisis_severity"] = 0
for country in jst["country"].unique():
    mask = jst["country"] == country
    country_df = jst[mask].sort_values("year")

    for i, row in country_df.iterrows():
        if row["crisisJST"] == 1:
            # Calculate GDP decline from pre-crisis peak
            if i > 0:
                gdp_change = (
                    jst.loc[i, "rgdpmad"] / jst.loc[i - 1, "rgdpmad"] - 1
                    if jst.loc[i - 1, "rgdpmad"] > 0
                    else 0
                )
                jst.loc[i, "crisis_severity"] = max(0, -gdp_change)  # Positive = worse

# Target 4: Multi-year ahead binary targets
print("Creating multi-year ahead targets...")
for horizon in [2, 3, 5]:
    jst[f"crisis_in_{horizon}yr"] = 0
    for country in jst["country"].unique():
        mask = jst["country"] == country
        data = jst.loc[mask, "crisisJST"].values
        target = np.zeros(len(data))
        for i in range(len(data) - horizon):
            # Crisis within next h years
            if np.any(data[i + 1 : i + horizon + 1] == 1) and data[i] == 0:
                target[i] = 1
        jst.loc[mask, f"crisis_in_{horizon}yr"] = target

# Show target statistics
print("\n=== TARGET STATISTICS ===")
print("\n1. Multi-label Crisis Types:")
for ctype in crisis_types:
    col = f"crisis_{ctype.replace(' / ', '_').replace(' ', '_')}"
    print(f"  {ctype}: {jst[col].sum():.0f} episodes")

print(f"\n2. Time-to-crisis: mean={jst['years_to_crisis'].mean():.1f} years")
print(
    f"   (0-1 years: {(jst['years_to_crisis'] <= 1).sum()} obs, 1-2 years: {((jst['years_to_crisis'] > 1) & (jst['years_to_crisis'] <= 2)).sum()} obs)"
)

print(f"\n3. Crisis severity: {jst['crisis_severity'].sum():.0f} total severity")

print("\n4. Multi-year ahead:")
for horizon in [2, 3, 5]:
    col = f"crisis_in_{horizon}yr"
    print(
        f"   {horizon}-year ahead: {jst[col].sum():.0f} positive / {len(jst)} total ({jst[col].mean() * 100:.1f}%)"
    )

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

print("\n=== FEATURE ENGINEERING ===")

feature_cols = []

# Growth rates
for col in ["rgdpmad", "tloans", "gdp", "hpnom", "cpi", "money", "narrowm"]:
    if col in jst.columns:
        jst[f"{col}_gr"] = jst.groupby("country")[col].pct_change()
        jst[f"{col}_gr2"] = jst.groupby("country")[f"{col}_gr"].diff()
        feature_cols.extend([f"{col}_gr", f"{col}_gr2"])

# Volatility
for col in ["rgdpmad_gr", "tloans_gr"]:
    if col in jst.columns:
        for window in [3, 5]:
            jst[f"{col}_vol{window}"] = (
                jst.groupby("country")[col]
                .rolling(window, min_periods=2)
                .std()
                .reset_index(0, drop=True)
            )
            feature_cols.append(f"{col}_vol{window}")

# Spreads and ratios
if "ltrate" in jst.columns and "stir" in jst.columns:
    jst["yield_spread"] = jst["ltrate"] - jst["stir"]
    jst["yield_spread_chg"] = jst.groupby("country")["yield_spread"].diff()
    feature_cols.extend(["yield_spread", "yield_spread_chg"])

if "tloans" in jst.columns and "gdp" in jst.columns:
    jst["credit_gdp"] = jst["tloans"] / jst["gdp"]
    jst["credit_gdp_gr"] = jst.groupby("country")["credit_gdp"].pct_change()
    feature_cols.extend(["credit_gdp", "credit_gdp_gr"])

# Bank health
for col in ["lev", "ltd", "noncore"]:
    if col in jst.columns:
        jst[f"{col}_chg"] = jst.groupby("country")[col].diff()
        feature_cols.extend([col, f"{col}_chg"])

# Asset returns
for col in ["eq_tr", "housing_tr", "bond_tr"]:
    if col in jst.columns:
        for lag in [1, 2]:
            jst[f"{col}_lag{lag}"] = jst.groupby("country")[col].shift(lag)
            feature_cols.append(f"{col}_lag{lag}")

# Macroeconomic
for col in ["ca", "debtgdp", "unemp"]:
    if col in jst.columns:
        if col == "ca" and "gdp" in jst.columns:
            jst["ca_gdp"] = jst["ca"] / jst["gdp"]
            jst["ca_gdp_chg"] = jst.groupby("country")["ca_gdp"].diff()
            feature_cols.extend(["ca_gdp", "ca_gdp_chg"])
        elif col == "debtgdp":
            jst[f"{col}_chg"] = jst.groupby("country")[col].diff()
            feature_cols.extend([col, f"{col}_chg"])
        elif col == "unemp":
            jst[f"{col}_chg"] = jst.groupby("country")[col].diff()
            feature_cols.extend([col, f"{col}_chg"])

# GPR
gpr_annual = (
    gpr.groupby("year")
    .agg(
        {"GPR": ["mean", "max", "std"], "GPRT": "mean", "GPRA": "mean", "GPRH": "mean"}
    )
    .reset_index()
)
gpr_annual.columns = ["year"] + ["_".join(c) for c in gpr_annual.columns[1:]]

for col in [c for c in gpr_annual.columns if c != "year"]:
    gpr_annual[f"{col}_lag1"] = gpr_annual[col].shift(1)

gpr_features = [c for c in gpr_annual.columns if "lag1" in c]
jst = jst.merge(gpr_annual[["year"] + gpr_features], on="year", how="left")
feature_cols.extend(gpr_features)

print(f"Total features: {len(feature_cols)}")

# ============================================================
# 4. CREATE SEQUENCES
# ============================================================

print("\n=== CREATING SEQUENCES ===")

SEQ_LEN = 5


def create_sequences(df, features, target_col, seq_len=5):
    """Create sequences for a specific target"""
    X, y, meta = [], [], []

    for country in df["country"].unique():
        cdf = df[df["country"] == country].sort_values("year")
        if len(cdf) < seq_len + 1:
            continue

        feat = cdf[features].values
        tgt = cdf[target_col].values
        yrs = cdf["year"].values
        isos = cdf["iso"].values

        for i in range(seq_len, len(feat)):
            if not np.isnan(feat[i - seq_len : i]).any() and not np.isnan(tgt[i]):
                X.append(feat[i - seq_len : i])
                y.append(tgt[i])
                meta.append({"country": country, "iso": isos[i], "year": yrs[i]})

    return np.array(X), np.array(y), meta


# ============================================================
# 5. MODEL DEFINITIONS
# ============================================================


class MultitaskModel(nn.Module):
    """Multi-task model for crisis prediction"""

    def __init__(self, input_size, seq_len, num_crisis_types=4, dropout=0.4):
        super().__init__()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * seq_len, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Task-specific heads
        # 1. Multi-label classification (crisis types)
        self.crisis_type_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_crisis_types),
            nn.Sigmoid(),
        )

        # 2. Time to crisis (regression)
        self.time_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

        # 3. Crisis severity (regression)
        self.severity_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.ReLU(),  # Severity is positive
        )

        # 4. Multi-year ahead (binary classifications)
        self.horizon2_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.horizon3_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.horizon5_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        shared = self.encoder(x)
        return {
            "crisis_types": self.crisis_type_head(shared),
            "time_to_crisis": self.time_head(shared).squeeze(),
            "severity": self.severity_head(shared).squeeze(),
            "horizon2": self.horizon2_head(shared).squeeze(),
            "horizon3": self.horizon3_head(shared).squeeze(),
            "horizon5": self.horizon5_head(shared).squeeze(),
        }


class SingleTaskModel(nn.Module):
    """Single-task model for specific target"""

    def __init__(self, input_size, seq_len, task="binary", dropout=0.4):
        super().__init__()
        self.task = task

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * seq_len, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        if task == "binary":
            self.activation = nn.Sigmoid()
        elif task == "regression":
            self.activation = lambda x: x

    def forward(self, x):
        out = self.net(x)
        if self.task == "binary":
            out = self.activation(out)
        return out.squeeze()


# ============================================================
# 6. TRAINING FUNCTIONS
# ============================================================


def train_multitask(X_train, y_dict_train, X_val, y_dict_val, epochs=100):
    """Train multi-task model"""
    model = MultitaskModel(X_train.shape[2], SEQ_LEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Normalize
    scaler = RobustScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    scaler.fit(X_train_flat)
    X_train_norm = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_norm = scaler.transform(X_val_flat).reshape(X_val.shape)

    X_train_t = torch.FloatTensor(X_train_norm).to(device)
    X_val_t = torch.FloatTensor(X_val_norm).to(device)

    # Convert targets
    y_train_t = {k: torch.FloatTensor(v).to(device) for k, v in y_dict_train.items()}
    y_val_t = {k: torch.FloatTensor(v).to(device) for k, v in y_dict_val.items()}

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(X_train_t)

        # Multi-task loss
        loss = 0
        # Crisis types (multi-label BCE)
        loss += F.binary_cross_entropy(out["crisis_types"], y_train_t["crisis_types"])
        # Time to crisis (MSE)
        loss += F.mse_loss(out["time_to_crisis"], y_train_t["time_to_crisis"])
        # Severity (MSE)
        loss += F.mse_loss(out["severity"], y_train_t["severity"])
        # Horizons (BCE)
        loss += F.binary_cross_entropy(out["horizon2"], y_train_t["horizon2"])
        loss += F.binary_cross_entropy(out["horizon3"], y_train_t["horizon3"])
        loss += F.binary_cross_entropy(out["horizon5"], y_train_t["horizon5"])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(X_val_t)
                val_loss = F.binary_cross_entropy(
                    val_out["horizon2"], y_val_t["horizon2"]
                )
            print(
                f"Epoch {epoch + 1}: Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}"
            )

    return model, scaler


def train_single_task(X_train, y_train, X_val, y_val, task="binary", epochs=100):
    """Train single-task model"""
    model = SingleTaskModel(X_train.shape[2], SEQ_LEN, task=task).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Normalize
    scaler = RobustScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    scaler.fit(X_train_flat)
    X_train_norm = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_norm = scaler.transform(X_val_flat).reshape(X_val.shape)

    X_train_t = torch.FloatTensor(X_train_norm).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val_norm).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    best_metric = 0 if task == "binary" else float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(X_train_t)

        if task == "binary":
            # Weighted BCE for imbalance
            pos_weight = torch.tensor(
                [(y_train == 0).sum() / max((y_train == 1).sum(), 1)]
            ).to(device)
            weights = torch.where(
                y_train_t == 1, pos_weight, torch.ones_like(pos_weight)
            )
            loss = (
                F.binary_cross_entropy(pred, y_train_t, reduction="none") * weights
            ).mean()
        else:
            loss = F.mse_loss(pred, y_train_t)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Eval
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t).cpu().numpy()

            if task == "binary" and y_val.sum() > 0:
                precision, recall, _ = precision_recall_curve(y_val, val_pred)
                metric = auc(recall, precision)
                print(
                    f"Epoch {epoch + 1}: Loss={loss.item():.4f}, Val AUCPR={metric:.4f}"
                )
                if metric > best_metric:
                    best_metric = metric
                    best_state = model.state_dict().copy()
            else:
                metric = mean_squared_error(y_val, val_pred)
                print(
                    f"Epoch {epoch + 1}: Loss={loss.item():.4f}, Val MSE={metric:.4f}"
                )
                if metric < best_metric:
                    best_metric = metric
                    best_state = model.state_dict().copy()

    if best_state:
        model.load_state_dict(best_state)

    return model, scaler, best_metric


# ============================================================
# 7. CREATE TRAIN/VAL SPLIT
# ============================================================

print("\n=== CREATING DATA SPLITS ===")

# Use all features
all_features = [f for f in feature_cols if f in jst.columns]

# Create sequences for each target
print("Creating sequences...")

# 1. Multi-label crisis types
crisis_type_cols = [
    f"crisis_{ctype.replace(' / ', '_').replace(' ', '_')}" for ctype in crisis_types
]

# Create a lookup from (iso, year) to crisis types
jst_lookup = jst.set_index(["iso", "year"])[crisis_type_cols].to_dict("index")

X_types, y_types, meta_types = create_sequences(
    jst, all_features, crisis_type_cols[0], SEQ_LEN
)
# Get all type labels from jst_lookup
y_types_full = []
for m in meta_types:
    key = (m["iso"], m["year"])
    if key in jst_lookup:
        y_types_full.append([jst_lookup[key][col] for col in crisis_type_cols])
    else:
        y_types_full.append([0] * len(crisis_type_cols))
y_types_full = np.array(y_types_full)

# 2. Time to crisis
X_time, y_time, meta_time = create_sequences(
    jst, all_features, "years_to_crisis", SEQ_LEN
)

# 3. Severity
X_sev, y_sev, meta_sev = create_sequences(jst, all_features, "crisis_severity", SEQ_LEN)

# 4. Multi-year ahead
X_h2, y_h2, meta_h2 = create_sequences(jst, all_features, "crisis_in_2yr", SEQ_LEN)
X_h3, y_h3, meta_h3 = create_sequences(jst, all_features, "crisis_in_3yr", SEQ_LEN)
X_h5, y_h5, meta_h5 = create_sequences(jst, all_features, "crisis_in_5yr", SEQ_LEN)

print(f"Multi-label: {X_types.shape}, {y_types_full.sum()} total type labels")
print(f"Time-to-crisis: {X_time.shape}, mean={y_time.mean():.1f} years")
print(f"Severity: {X_sev.shape}, total={y_sev.sum():.1f}")
print(f"2-year: {X_h2.shape}, positives={y_h2.sum()}")
print(f"3-year: {X_h3.shape}, positives={y_h3.sum()}")
print(f"5-year: {X_h5.shape}, positives={y_h5.sum()}")

# Split by time
years_types = np.array([m["year"] for m in meta_types])
train_mask = years_types < 2000
val_mask = (years_types >= 2000) & (years_types < 2008)

print(f"\nTrain: {train_mask.sum()}, Val: {val_mask.sum()}")

# ============================================================
# 8. TRAIN ALL MODELS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING ALL APPROACHES")
print("=" * 70)

results = {}

# Approach 1: Multi-label crisis types
print("\n--- Approach 1: Multi-label Crisis Types ---")
try:
    y_dict_train = {
        "crisis_types": torch.FloatTensor(y_types_full[train_mask]).to(device),
        "time_to_crisis": torch.FloatTensor(y_time[train_mask]).to(device),
        "severity": torch.FloatTensor(y_sev[train_mask]).to(device),
        "horizon2": torch.FloatTensor(y_h2[train_mask]).to(device),
        "horizon3": torch.FloatTensor(y_h3[train_mask]).to(device),
        "horizon5": torch.FloatTensor(y_h5[train_mask]).to(device),
    }
    y_dict_val = {
        "crisis_types": torch.FloatTensor(y_types_full[val_mask]).to(device),
        "time_to_crisis": torch.FloatTensor(y_time[val_mask]).to(device),
        "severity": torch.FloatTensor(y_sev[val_mask]).to(device),
        "horizon2": torch.FloatTensor(y_h2[val_mask]).to(device),
        "horizon3": torch.FloatTensor(y_h3[val_mask]).to(device),
        "horizon5": torch.FloatTensor(y_h5[val_mask]).to(device),
    }

    model_multi, scaler_multi = train_multitask(
        X_types[train_mask], y_dict_train, X_types[val_mask], y_dict_val, epochs=80
    )
    results["multitask"] = "trained"
    print("✓ Multi-task model trained")
except Exception as e:
    print(f"✗ Multi-task failed: {e}")
    results["multitask"] = "failed"

# Approach 2: Individual models for each target
print("\n--- Approach 2-5: Individual Task Models ---")

# 2. Time to crisis
print("\n2. Time-to-Crisis Regression:")
try:
    valid_time = ~np.isnan(y_time)
    model_time, scaler_time, metric_time = train_single_task(
        X_time[train_mask & valid_time[train_mask]],
        y_time[train_mask & valid_time[train_mask]],
        X_time[val_mask & valid_time[val_mask]],
        y_time[val_mask & valid_time[val_mask]],
        task="regression",
        epochs=80,
    )
    print(f"   Best Val MSE: {metric_time:.4f}")
    results["time_to_crisis"] = metric_time
except Exception as e:
    print(f"   ✗ Failed: {e}")
    results["time_to_crisis"] = None

# 3. Crisis severity
print("\n3. Crisis Severity Regression:")
try:
    model_sev, scaler_sev, metric_sev = train_single_task(
        X_sev[train_mask],
        y_sev[train_mask],
        X_sev[val_mask],
        y_sev[val_mask],
        task="regression",
        epochs=80,
    )
    print(f"   Best Val MSE: {metric_sev:.4f}")
    results["severity"] = metric_sev
except Exception as e:
    print(f"   ✗ Failed: {e}")
    results["severity"] = None

# 4. Multi-year ahead
for horizon, X_h, y_h, meta_h in [
    (2, X_h2, y_h2, meta_h2),
    (3, X_h3, y_h3, meta_h3),
    (5, X_h5, y_h5, meta_h5),
]:
    print(f"\n{horizon}-year ahead prediction:")
    try:
        years_h = np.array([m["year"] for m in meta_h])
        train_mask_h = years_h < 2000
        val_mask_h = (years_h >= 2000) & (years_h < 2008)

        model_h, scaler_h, metric_h = train_single_task(
            X_h[train_mask_h],
            y_h[train_mask_h],
            X_h[val_mask_h],
            y_h[val_mask_h],
            task="binary",
            epochs=80,
        )
        print(f"   Best Val AUCPR: {metric_h:.4f}")
        results[f"horizon_{horizon}"] = metric_h
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        results[f"horizon_{horizon}"] = None

# ============================================================
# 9. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("COMPREHENSIVE RESULTS SUMMARY")
print("=" * 70)

print("\n1. Multi-task Learning:")
print(f"   Status: {results.get('multitask', 'N/A')}")

print("\n2. Time-to-Crisis (Regression):")
if results.get("time_to_crisis"):
    print(f"   Val MSE: {results['time_to_crisis']:.4f}")
else:
    print("   Failed")

print("\n3. Crisis Severity (Regression):")
if results.get("severity"):
    print(f"   Val MSE: {results['severity']:.4f}")
else:
    print("   Failed")

print("\n4. Multi-year Ahead Prediction:")
for h in [2, 3, 5]:
    key = f"horizon_{h}"
    if results.get(key):
        print(f"   {h}-year: AUCPR = {results[key]:.4f}")
    else:
        print(f"   {h}-year: Failed")

print("\n5. Comparison to Original (1-year ahead):")
print(f"   Original 1-year AUCPR: 0.269")
for h in [2, 3, 5]:
    key = f"horizon_{h}"
    if results.get(key):
        if results[key] > 0.269:
            print(f"   {h}-year: IMPROVED ✓ ({results[key]:.4f} > 0.269)")
        else:
            print(f"   {h}-year: Not improved ({results[key]:.4f} < 0.269)")

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print("""
- Multi-label approach: Predicts specific crisis types (banking, asset price, etc.)
- Time-to-crisis: Predicts years until next crisis (continuous)
- Severity: Predicts GDP decline magnitude during crisis
- Multi-year: Predicts crisis within 2/3/5 years (more positives = easier)

The multi-year approaches (2,3,5 year) have MORE training examples because:
- 1-year: Crisis must start exactly next year
- 5-year: Crisis can start anytime in next 5 years (5x more positives!)
""")

# Save results
results_df = pd.DataFrame([results])
results_df.to_csv(".output/comprehensive_results.csv", index=False)
print("\nResults saved to: comprehensive_results.csv")
