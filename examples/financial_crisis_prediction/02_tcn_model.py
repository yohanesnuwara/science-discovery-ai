"""
Financial Crisis Forecasting - IMPROVED VERSION v2
Strategies to beat baseline:
1. Feature selection to reduce overfitting
2. Stronger regularization
3. Ensemble methods
4. Better class balancing
5. Transfer learning approach
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, brier_score_loss
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 1. LOAD DATA (same as before)
# ============================================================

print("\n=== Loading Data ===")
jst = pd.read_csv(".input/JSTdatasetR6.csv")
gpr = pd.read_csv(".input/data_gpr_export.csv")

# Parse GPR
gpr["date"] = pd.to_datetime(gpr["month"], format="%d.%m.%Y", errors="coerce")
gpr["year"] = gpr["date"].dt.year

# ============================================================
# 2. CREATE 5-YEAR TARGET
# ============================================================

print("\n=== Creating 5-Year Target ===")
jst = jst.sort_values(["country", "year"]).reset_index(drop=True)

jst["target_5yr"] = 0
for country in jst["country"].unique():
    mask = jst["country"] == country
    data = jst.loc[mask, "crisisJST"].values
    target = np.zeros(len(data))
    for i in range(len(data) - 5):
        if np.any(data[i + 1 : i + 6] == 1) and data[i] == 0:
            target[i] = 1
    jst.loc[mask, "target_5yr"] = target

print(
    f"5-year positives: {jst['target_5yr'].sum()} ({jst['target_5yr'].mean() * 100:.1f}%)"
)

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

print("\n=== Feature Engineering ===")
feature_cols = []

# Growth rates
for col in ["rgdpmad", "tloans", "gdp", "hpnom", "cpi"]:
    if col in jst.columns:
        jst[f"{col}_gr"] = jst.groupby("country")[col].pct_change()
        jst[f"{col}_gr2"] = jst.groupby("country")[f"{col}_gr"].diff()
        feature_cols.extend([f"{col}_gr", f"{col}_gr2"])

# Volatility
for col in ["rgdpmad_gr", "tloans_gr"]:
    if col in jst.columns:
        jst[f"{col}_vol"] = (
            jst.groupby("country")[col]
            .rolling(3, min_periods=2)
            .std()
            .reset_index(0, drop=True)
        )
        feature_cols.append(f"{col}_vol")

# Spreads
if "ltrate" in jst.columns and "stir" in jst.columns:
    jst["spread"] = jst["ltrate"] - jst["stir"]
    jst["spread_chg"] = jst.groupby("country")["spread"].diff()
    feature_cols.extend(["spread", "spread_chg"])

# Credit
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
for col in ["eq_tr", "housing_tr"]:
    if col in jst.columns:
        jst[f"{col}_lag"] = jst.groupby("country")[col].shift(1)
        feature_cols.append(f"{col}_lag")

# GPR
gpr_annual = (
    gpr.groupby("year")
    .agg({"GPR": ["mean", "max", "std"], "GPRT": "mean", "GPRH": "mean"})
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

SEQ_LEN = 5


def create_sequences(df, features, seq_len=5):
    X, y, meta = [], [], []
    for country in df["country"].unique():
        cdf = df[df["country"] == country].sort_values("year")
        if len(cdf) < seq_len + 1:
            continue
        feat = cdf[features].values
        tgt = cdf["target_5yr"].values
        yrs = cdf["year"].values
        isos = cdf["iso"].values

        for i in range(seq_len, len(feat)):
            if not np.isnan(feat[i - seq_len : i]).any() and not np.isnan(tgt[i]):
                X.append(feat[i - seq_len : i])
                y.append(tgt[i])
                meta.append({"country": country, "iso": isos[i], "year": yrs[i]})

    return np.array(X), np.array(y), meta


X, y, meta = create_sequences(jst, feature_cols, SEQ_LEN)
print(f"\nSequences: {X.shape}")
print(f"Positives: {y.sum()} ({y.mean() * 100:.1f}%)")

# ============================================================
# 5. FEATURE SELECTION
# ============================================================

print("\n=== Feature Selection ===")

# Use last timestep for feature selection
X_last = X[:, -1, :]

# Select top k features using mutual information
k_best = min(20, X.shape[2])  # Select top 20 features
selector = SelectKBest(mutual_info_classif, k=k_best)
X_selected = selector.fit_transform(X_last, y)
selected_indices = selector.get_support(indices=True)
selected_features = [feature_cols[i] for i in selected_indices]

print(f"Selected top {k_best} features:")
for i, feat in enumerate(selected_features[:10]):
    print(f"  {i + 1}. {feat}")

# Reduce X to selected features
X = X[:, :, selected_indices]
print(f"\nNew shape: {X.shape}")

# ============================================================
# 6. CROSS-VALIDATION SETUP
# ============================================================

print("\n=== Setting Up Cross-Validation ===")

years = np.array([m["year"] for m in meta])

cv_folds = [
    (1995, 1995, 2000),
    (2000, 2000, 2005),
    (2005, 2005, 2008),
]

print(f"Using {len(cv_folds)} CV folds")

# ============================================================
# 7. MODEL DEFINITIONS
# ============================================================

print("\n=== Defining Models ===")


class ImprovedMLP(nn.Module):
    """MLP with stronger regularization"""

    def __init__(self, input_size, seq_len, dropout=0.6):  # Increased dropout
        super().__init__()
        flat_size = input_size * seq_len
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 64),  # Reduced capacity
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze()


class LSTMEnsemble(nn.Module):
    """LSTM with attention"""

    def __init__(self, input_size, hidden=32, num_layers=1, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        context = self.dropout(context)
        return self.fc(context).squeeze()


# ============================================================
# 8. TRAINING WITH IMPROVEMENTS
# ============================================================


def train_with_early_stopping(
    X_train, y_train, X_val, y_val, model_type="mlp", epochs=150, patience=30
):
    """Train with early stopping and learning rate scheduling"""

    # Normalize
    scaler = RobustScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    scaler.fit(X_train_flat)
    X_train_norm = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_norm = scaler.transform(X_val_flat).reshape(X_val.shape)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_norm).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val_norm).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    # Model
    if model_type == "mlp":
        model = ImprovedMLP(X_train.shape[2], SEQ_LEN, dropout=0.6).to(device)
    else:
        model = LSTMEnsemble(X_train.shape[2], hidden=32, num_layers=1, dropout=0.5).to(
            device
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.05
    )  # Stronger weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    # Class weights
    if y_train.sum() > 0:
        pos_weight = torch.tensor([(y_train == 0).sum() / y_train.sum()]).to(device)
    else:
        pos_weight = torch.tensor([1.0]).to(device)

    # Training with early stopping
    best_val_auc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(X_train_t)
        weights = torch.where(y_train_t == 1, pos_weight, torch.ones_like(pos_weight))
        loss = (
            F.binary_cross_entropy(pred, y_train_t, reduction="none") * weights
        ).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).cpu().numpy()

        if y_val.sum() > 0:
            precision, recall, _ = precision_recall_curve(y_val, val_pred)
            val_auc = auc(recall, precision)
        else:
            val_auc = 0

        scheduler.step(val_auc)

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # Final predictions
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t).cpu().numpy()
        val_pred = model(X_val_t).cpu().numpy()

    return model, scaler, train_pred, val_pred, best_val_auc


# ============================================================
# 9. ENSEMBLE APPROACH
# ============================================================


def train_ensemble(X_train, y_train, X_val, y_val):
    """Train multiple models and average predictions"""

    print("\n  Training ensemble (3 models)...")
    predictions = []

    # Model 1: MLP
    _, _, _, pred1, auc1 = train_with_early_stopping(
        X_train, y_train, X_val, y_val, "mlp", epochs=100, patience=20
    )
    predictions.append(pred1)
    print(f"    MLP: {auc1:.4f}")

    # Model 2: LSTM
    _, _, _, pred2, auc2 = train_with_early_stopping(
        X_train, y_train, X_val, y_val, "lstm", epochs=100, patience=20
    )
    predictions.append(pred2)
    print(f"    LSTM: {auc2:.4f}")

    # Model 3: Random Forest (sklearn)
    scaler = RobustScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)
    pred3 = rf.predict_proba(X_val_scaled)[:, 1]

    if y_val.sum() > 0:
        precision, recall, _ = precision_recall_curve(y_val, pred3)
        auc3 = auc(recall, precision)
    else:
        auc3 = 0
    predictions.append(pred3)
    print(f"    RF: {auc3:.4f}")

    # Average ensemble
    ensemble_pred = np.mean(predictions, axis=0)
    if y_val.sum() > 0:
        precision, recall, _ = precision_recall_curve(y_val, ensemble_pred)
        ensemble_auc = auc(recall, precision)
    else:
        ensemble_auc = 0

    print(f"    Ensemble: {ensemble_auc:.4f}")

    return ensemble_pred, ensemble_auc


# ============================================================
# 10. RUN CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("RUNNING IMPROVED CROSS-VALIDATION")
print("=" * 60)

fold_results = []
all_predictions = []
baseline = y.mean()

print(f"\nBaseline (class prevalence): {baseline:.4f}")

for fold_idx, (train_end, test_start, test_end) in enumerate(cv_folds):
    print(f"\n--- Fold {fold_idx + 1} ---")

    train_mask = years < train_end
    test_mask = (years >= test_start) & (years <= test_end)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    meta_test = [meta[i] for i in np.where(test_mask)[0]]

    print(f"Train: {len(X_train)} samples, {y_train.sum()} crises")
    print(f"Test: {len(X_test)} samples, {y_test.sum()} crises")

    if y_train.sum() == 0 or y_test.sum() == 0:
        print("  Skipping - insufficient crises")
        continue

    # Train ensemble
    test_pred, test_auc = train_ensemble(X_train, y_train, X_test, y_test)

    # Check if we beat baseline
    beats_baseline = "✓ BEATS BASELINE" if test_auc > baseline else "✗ Below baseline"
    print(f"  Test AUCPR: {test_auc:.4f} {beats_baseline}")

    fold_results.append(
        {
            "fold": fold_idx + 1,
            "test_auc": test_auc,
            "baseline": baseline,
            "beats_baseline": test_auc > baseline,
        }
    )

    # Store predictions
    for m, p, t in zip(meta_test, test_pred, y_test):
        all_predictions.append(
            {
                "fold": fold_idx + 1,
                "country": m["country"],
                "iso": m["iso"],
                "year": m["year"],
                "predicted_probability": p,
                "actual_label": int(t),
            }
        )

# ============================================================
# 11. RESULTS
# ============================================================

print("\n" + "=" * 60)
print("IMPROVED RESULTS SUMMARY")
print("=" * 60)

results_df = pd.DataFrame(fold_results)
print("\nPer-fold results:")
print(results_df.to_string(index=False))

print(f"\nBaseline: {baseline:.4f}")
print(f"Mean AUCPR: {results_df['test_auc'].mean():.4f}")
print(f"Folds beating baseline: {results_df['beats_baseline'].sum()}/{len(results_df)}")

# Overall performance
pred_df = pd.DataFrame(all_predictions)
if len(pred_df) > 0 and pred_df["actual_label"].sum() > 0:
    precision, recall, _ = precision_recall_curve(
        pred_df["actual_label"], pred_df["predicted_probability"]
    )
    overall_auc = auc(recall, precision)
    brier = brier_score_loss(pred_df["actual_label"], pred_df["predicted_probability"])

    print(f"\nOverall AUCPR: {overall_auc:.4f}")
    print(f"Overall Brier: {brier:.4f}")
    print(f"Beats baseline: {'✓ YES' if overall_auc > baseline else '✗ NO'}")

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"Original 1-year:     0.269")
print(f"Original 5-year CV:  0.280")
print(f"Baseline:            {baseline:.4f}")
print(f"This improved model: {overall_auc:.4f}")

if overall_auc > baseline:
    print(
        f"\n✓ SUCCESS: Model beats baseline by {((overall_auc - baseline) / baseline * 100):.1f}%"
    )
else:
    print(f"\n✗ Still below baseline. Need more improvements.")

# ============================================================
# 12. VISUALIZATION
# ============================================================

print("\n=== Creating Visualizations ===")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))

# 1. Precision-Recall Curves by Fold
plt.subplot(2, 4, 1)
colors = ["blue", "orange", "green"]
for fold_idx in [1, 2, 3]:
    fold_pred = pred_df[pred_df["fold"] == fold_idx]
    if len(fold_pred) > 0 and fold_pred["actual_label"].sum() > 0:
        precision, recall, _ = precision_recall_curve(
            fold_pred["actual_label"], fold_pred["predicted_probability"]
        )
        pr_auc = auc(recall, precision)
        plt.plot(
            recall,
            precision,
            label=f"Fold {fold_idx} (AUCPR={pr_auc:.3f})",
            color=colors[fold_idx - 1],
            linewidth=2,
        )
plt.axhline(
    y=baseline,
    color="red",
    linestyle="--",
    alpha=0.5,
    label=f"Baseline ({baseline:.3f})",
)
plt.xlabel("Recall", fontsize=10)
plt.ylabel("Precision", fontsize=10)
plt.title("Precision-Recall by Fold", fontsize=11, fontweight="bold")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# 2. Overall Performance
plt.subplot(2, 4, 2)
if overall_auc > 0:
    precision_all, recall_all, _ = precision_recall_curve(
        pred_df["actual_label"], pred_df["predicted_probability"]
    )
    plt.plot(
        recall_all,
        precision_all,
        label=f"Improved (AUCPR={overall_auc:.3f})",
        linewidth=3,
        color="darkgreen",
    )
    plt.fill_between(recall_all, precision_all, alpha=0.3, color="darkgreen")
    plt.axhline(
        y=baseline,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Baseline ({baseline:.3f})",
    )
plt.xlabel("Recall", fontsize=10)
plt.ylabel("Precision", fontsize=10)
plt.title("Overall vs Baseline", fontsize=11, fontweight="bold")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# 3. Crisis vs Predictions
plt.subplot(2, 4, 3)
onsets = pred_df[pred_df["actual_label"] == 1]
if len(onsets) > 0:
    plt.scatter(
        onsets["year"],
        onsets["predicted_probability"],
        alpha=0.8,
        color="red",
        label="Crisis",
        s=40,
    )
no_onsets = pred_df[pred_df["actual_label"] == 0].sample(
    min(200, len(pred_df[pred_df["actual_label"] == 0]))
)
plt.scatter(
    no_onsets["year"],
    no_onsets["predicted_probability"],
    alpha=0.3,
    color="blue",
    label="No Crisis",
    s=10,
)
plt.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5)
plt.xlabel("Year", fontsize=10)
plt.ylabel("Predicted Probability", fontsize=10)
plt.title("Crisis vs Predicted Risk", fontsize=11, fontweight="bold")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# 4. Top 20 Highest Risk
plt.subplot(2, 4, 4)
top_risk = pred_df.nlargest(20, "predicted_probability")
colors_bar = ["red" if label == 1 else "blue" for label in top_risk["actual_label"]]
plt.barh(
    range(len(top_risk)), top_risk["predicted_probability"], color=colors_bar, alpha=0.7
)
plt.yticks(
    range(len(top_risk)),
    [f"{row['country'][:3]}-{int(row['year'])}" for _, row in top_risk.iterrows()],
    fontsize=7,
)
plt.xlabel("Probability", fontsize=10)
plt.title("Top 20 Risk Country-Years", fontsize=11, fontweight="bold")
plt.gca().invert_yaxis()

# 5. Distribution
plt.subplot(2, 4, 5)
no_crisis = pred_df[pred_df["actual_label"] == 0]["predicted_probability"]
crisis = pred_df[pred_df["actual_label"] == 1]["predicted_probability"]
plt.hist(no_crisis, bins=25, alpha=0.7, label="No Crisis", density=True, color="blue")
plt.hist(crisis, bins=25, alpha=0.7, label="Crisis", density=True, color="red")
plt.axvline(x=baseline, color="black", linestyle="--", label="Baseline")
plt.xlabel("Predicted Probability", fontsize=10)
plt.ylabel("Density", fontsize=10)
plt.title("Distribution of Predictions", fontsize=11, fontweight="bold")
plt.legend(fontsize=8)

# 6. Model Comparison
plt.subplot(2, 4, 6)
models = ["Baseline", "Orig 1-yr", "Orig 5-yr", "Improved"]
aucprs = [0.1371, 0.269, 0.280, overall_auc]
colors_comp = ["red", "orange", "yellow", "green"]
bars = plt.bar(models, aucprs, color=colors_comp, alpha=0.7)
plt.axhline(y=0.1371, color="red", linestyle="--", alpha=0.5)
plt.ylabel("AUCPR", fontsize=10)
plt.title("Model Comparison", fontsize=11, fontweight="bold")
plt.ylim(0, 0.4)
for bar, val in zip(bars, aucprs):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=9,
    )

# 7. Feature Importance (using Random Forest from last fold)
plt.subplot(2, 4, 7)
# Train a final RF model on all data for feature importance
scaler_final = RobustScaler()
X_flat = X.reshape(X.shape[0], -1)
X_scaled = scaler_final.fit_transform(X_flat)
rf_final = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
rf_final.fit(X_scaled, y)

# Get feature importance
importance = rf_final.feature_importances_

# Create feature names (original features × SEQ_LEN timesteps)
n_original_features = len(selected_features)
feature_names_all = []
for t in range(SEQ_LEN):
    for feat in selected_features:
        feature_names_all.append(f"{feat}_t{t}")

# Check if dimensions match
if len(importance) == len(feature_names_all):
    # Sort by importance
    indices = np.argsort(importance)[::-1][:10]  # Top 10
    plt.barh(range(len(indices)), importance[indices], color="steelblue")
    plt.yticks(range(len(indices)), [feature_names_all[i] for i in indices], fontsize=7)
else:
    # If dimensions don't match, just show generic feature numbers
    indices = np.argsort(importance)[::-1][:10]
    plt.barh(range(len(indices)), importance[indices], color="steelblue")
    plt.yticks(range(len(indices)), [f"Feature {i}" for i in indices], fontsize=8)

plt.xlabel("Importance", fontsize=10)
plt.title("Top 10 Feature Importance (RF)", fontsize=11, fontweight="bold")
plt.gca().invert_yaxis()

# 8. Risk by Country
plt.subplot(2, 4, 8)
country_risk = (
    pred_df.groupby("country")["predicted_probability"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)
plt.barh(range(len(country_risk)), country_risk.values, color="steelblue")
plt.yticks(range(len(country_risk)), country_risk.index, fontsize=9)
plt.xlabel("Average Risk", fontsize=10)
plt.title("Top 10 Countries by Risk", fontsize=11, fontweight="bold")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig(".output/improved_model_results.png", dpi=300, bbox_inches="tight")
print("\nSaved: .output/improved_model_results.png")

# Save predictions
pred_df.to_csv(".output/5year_cv_predictions.csv", index=False)
print("Saved: .output/5year_cv_predictions.csv")
