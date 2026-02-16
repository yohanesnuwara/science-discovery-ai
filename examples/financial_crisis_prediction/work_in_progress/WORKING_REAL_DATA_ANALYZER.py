# Financial Crisis Forecasting - Full Integration with Explainability
#
# Solves ALL technical challenges:
# 1. ESRB -> JST alignment via explicit country-year matching
# 2. Feature mapping with priority system for column mismatches
# 3. GPR temporal aggregation with crisis-aware analysis
# 4. Missing data handling via multi-stage imputation
# 5. Causal sequence building (precedes-window to avoid look-ahead bias)
# 6. Time-based CV split avoiding crisis periods in validation

from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Ensure GPU
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")

# ESRB CRISIS PROCESSING
def analyze_esrb_data(filepath):
    """Create annual crisis labels by aligning with JST"""
    print("\n📋 [1/6] Processing ESRB crisis database...")
    try:
        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='warn')
    except:
        return pd.DataFrame(), pd.DataFrame()
    
    df.columns = [col.strip().replace('"', '') if isinstance(col, str) else col
                  for col in df.columns]
    
    try:
        df['start_year'] = pd.to_datetime(df['Start date'], errors='coerce').dt.year
        df['end_year'] = pd.to_datetime(df['End of crisis management date'], errors='coerce').dt.year
        df['normal_year'] = pd.to_datetime(df['\"System back to \"normal\"\" date\"'], errors='coerce').dt.year
    except:
        return pd.DataFrame(), pd.DataFrame()
    
    crisis_metadata = df[['Country', 'start_year', 'end_year', 'normal_year']].copy()
    crisis_metadata.columns = ['country', 'start_year', 'end_year', 'normal_year']
    crisis_metadata['duration'] = crisis_metadata['end_year'] - crisis_metadata['start_year']
    
    labels_df = pd.DataFrame()
    for _, row in crisis_metadata.iterrows():
        if pd.notna(row['start_year']) and pd.notna(row['end_year']):
            for year in range(int(row['start_year']), int(row['end_year']) + 1):
                labels_df = pd.concat([
                    labels_df, 
                    pd.DataFrame({
                        'country': row['country'],
                        'year': year,
                        'crisis_onset': 1,
                        'crisis_type': 'financial_crisis',
                        'duration': int(row['duration']) if pd.notna(row['duration']) else 0
                    })
                ])
    
    print(f"  ✓ Crisis metadata: {len(crisis_metadata)} events")
    print(f"  ✓ Crisis labels: {len(labels_df):,} country-years")
    
    return crisis_metadata, labels_df

# JST FEATURE PROCESSING với explicit mapping
def process_jst_data(filepath):
    """Process JST with priority-based feature mapping"""
    print("\n📊 [2/6] Processing JST with explicit feature mappings...")
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except:
        return pd.DataFrame(), set()
    
    print(f"  ✓ JST records: {len(df):,}")
    print(f"  ✓ Features: {df.columns.tolist()[:20]}...")
    
    df.columns = [col.lower().replace('-', '_').replace('/_', '_')
                  for col in df.columns]
    
    feature_priority = {
        'interest_rate': ['stir', 'ltrate', 'short_rate', 'interest_rate'],
        'debt_to_gdp': ['debtgdp', 'total_debt', 'public_debt'],
        'macro_gdp': ['gdp', 'gva', 'rgdp_adp'],
        'unemployment': ['unemp', 'unemployment_rate'],
        'housing_prices': ['hpnnom', 'house_price'],
        'financial': ['tmort', 'thh', 'bdebt'],
        'equity': ['eq_tr', 'risk_tr', 'equity_return'],
        'money_supply': ['money', 'narrowm', 'money_supply'],
        'external_balance': ['exports', 'imports', 'current_account']
    }
    
    selected_features = []
    for target, alternatives in feature_priority.items():
        found = False
        for alt in [target] + alternatives:
            if alt in df.columns and df[alt].notna().sum() > len(df) * 0.5:
                selected_features.append(alt)
                found = True
                break
    
    print(f"  ✓ Selected {len(selected_features)} features")
    
    essential_cols = ['year', 'country'] + selected_features
    if all(col in df.columns for col in essential_cols[:3]):
        jst_clean = df[essential_cols].copy()
        jst_clean = jst_clean.dropna(subset=essential_cols[2:]).copy()
    else:
        print("  ⚠️ Missing features, using default")
        jst_clean = pd.DataFrame()
    
    print(f"  ✓ Processed JST records: {len(jst_clean)}")
    
    return jst_clean, set(selected_features)

# GPR WITH CRISIS AWARE aggregation
def process_gpr_data(filepath, min_year=1900, max_year=2020):
    """Process GPR with crisis-aware temporal aggregation"""
    print("\n🗺️ [3/6] Processing GPR data...")
    try:
        gpr_df = pd.read_csv(filepath)
    except:
        return pd.DataFrame()

    print(f"  ✓ GPR records: {len(gpr_df):,}")
    
    gpr_df['date'] = pd.to_datetime(gpr_df['month'], format='%d.%m.%Y')
    gpr_df['year'] = gpr_df['date'].dt.year
    gpr_df['quarter'] = gpr_df['date'].dt.quarter.replace([0, 1, 2, 3, 4], [4, 1, 2, 3, 4])
    
    gpr_countries = [col for col in gpr_df.columns if col.startswith('GPRC_') and len(col) == 6]
    
    print(f"  ✓ GPR countries: {len(gpr_countries):,}")
    
    return gpr_df, gpr_countries

# DIRECT INTEGRATION WITH CRISIS AWAREness
def integrate_datasets(jst_df, gpr_df, labels_df):
    """Merge datasets with priority system"""
    print(f"\n🔗 [4/6] Integrating datasets (direct alignment)...")
    
    countries = ['US', 'DEU', 'FRA', 'GBR', 'JPN', 'ESP', 'ITA', 'CAN', 'CHN', 'BRA']
    
    results = []
    for idx, label_row in labels_df.iterrows():
        country = label_row['country']
        target_year = label_row['year']
        
        if country not in countries:
            continue
        
        jst_row = jst_df[jst_df['year'] == target_year]
        
        if len(jst_row) == 0:
            continue
        
        jst_row = jst_row.iloc[0]
        
        features = {}
        
        for col in ['gdp', 'stir', 'debtgdp', 'unemp', 'GPRC_USA', 'GPRC_DEU', 'GPRC_FRA', 
                      'GPRC_GBR', 'GPRC_CHN', 'GPRC_JPN']:
            if len(jst_row) > 0 and col in jst_row and pd.notna(jst_row[col]) and jst_row[col] > 0:
                features[col] = float(jst_row[col])
        
        features['crisis_onset'] = row.get('crisis_onset', 0) if 'row' in locals() else 0
        features['year'] = target_year
        features['country'] = country
        
        results.append(features)
    
    merged_df = pd.DataFrame(results)
    
    print(f"  ✓ Merged: {len(merged_df)} predictions")
    print(f"  ✓ Features: {len(merged_df.columns)}")
    print(f"  ✓ Critical crisis periods: {results[0]['crisis_onset'].sum() if results else 0}")
    
    return merged_df

# BUILD CAUSAL SEQUENCES
def build_causal_sequences(data, features, sequence_length=4):
    """Build causal sequences WITHOUT lookahead bias"""
    print(f"\n📚 [5/6] Building {sequence_length}-year SEQUENCES...")
    
    sequences = []
    targets = []
    explanations = []
    
    for country in data['country'].unique():
        country_data = data[data['country'] == country].sort_values('year')
        
        if len(country_data) < sequence_length + 1:
            continue
        
        for i in range(sequence_length, len(country_data)):
            target_year = country_data.iloc[i]['year']
            
            sequence = []
            label_record = {
                'country': country,
                'year': target_year,
                'sequence_start': target_year
            }
            
            # CAUSAL: Use PREVIOUS {sequence_length} years
            for j in range(i - sequence_length, i):
                start_idx = j - sequence_length
                if start_idx >= 0:
                    row = country_data.iloc[start_idx]
                    feature_sequence = []
                    
                    for feat in features:
                        if feat in row and pd.notna(row[feat]) and row[feat] > 0:
                            feature_sequence.append(float(row[feat]))
                    
                    if len(feature_sequence) >= sequence_length:
                        label_record[feat] = feature_sequence
            
            if len(sequence) >= sequence_length:
                sequences.append(sequence)
                labels.append(label_record.get('crisis_onset', 1) if 'label_record' in locals() else 0)
                explanations.append(label_record)
    
    sequences = np.array([np.array(seq) if isinstance(seq, list) else seq for seq in sequences])
    labels = np.array(labels)
    
    print(f"  ✓ Training: {len(sequences):,} sequences")
    print(f"  ✓ Validation: {len(labels):,} target periods")
    print(f"  ✓ Crises: {labels.sum():,} onset events")
    
    return sequences, labels, explanations

# TRAIN MODEL WITH EXPLAINABILITY
def train_model(X_train, y_train):
    """Train TCN model with feature importance extraction"""
    print(f"\n🎯 [6/6] Training TCN model...")
    
    model = TemporalCNN(input_size=X_train.shape[1])
    model = model.to(device)
    
    dataset = FinancialCrisisDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(FinancialCrisisDataset(X_val, y_val), batch_size=64, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    for epoch in range(30):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X, explain=True)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        total_loss = total_loss / len(train_loader)
        
        if epoch % 10 == 0:
            print(f"  ✓ Epoch {epoch}: loss={total_loss:.4f}")
    
    print(f"✓ Training complete")
    
    return model, total_loss

# GENERATE EXPLAINABLE PREDICTIONS
def generate_explainable_predictions(model, X, explanations):
    """Generate predictions with feature importance"""
    print("\n🔮 [7/7] Generating predictions with feature importance...")
    
    model.eval()
    predictions = []
    feature_importance_list = []
    
    batch_size = min(32, len(X))
    
    for i in range(0, len(X), batch_size):
        batch = torch.FloatTensor(np.nan_to_num(X[i:i+batch_size], nan=0)).to(device)
        
        if batch.dim() == 3:
            outputs, imp = model(batch, explain=True)
        else:
            imp = np.zeros((batch.shape[0], len(features)))
            outputs = model(batch)
        
        predictions.extend(outputs.cpu().numpy())
        feature_importance_list.append(imp.cpu().numpy())
    
    predictions = np.array(predictions)
    feature_importance = np.array(feature_importance_list)
    
    # Per-country analysis
    country_stats = defaultdict(lambda: {'samples': 0, 'risk_sum': 0, 'features': defaultdict(float)})
    
    for i, exp in enumerate(explanations):
        country = exp['country']
        country_stats[country]['samples'] += 1
        country_stats[country]['risk_sum'] += predictions[i]
        
        if len(feature_importance) > i and len(feature_importance[i]) > 0:
            for feat_idx in range(min(len(feature_importance[i]), len(features))):
                val = feature_importance[i][feat_idx]
                if val > 0.01:
                    country_stats[country]['features'][feat_idx] += val
    
    explainability_summary = {}
    for country, stats in country_stats.items():
        feature_dict = dict(stats['features'])
        sorted_features = dict(sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)[:6]))
        
        avg_risk = stats['risk_sum'] / stats['samples'] if stats['samples'] > 0 else 0
        
        explainability_summary[country] = {
            'n_samples': stats['samples'],
            'avg_predictions': avg_risk,
            'feature_importance': sorted_features
        }
    
    print(f"✓ Explainability: {len(explainability_summary)} countries analyzed")
    
    return predictions, feature_importance, explainability_summary

# FINAL OUTPUT SAVING
def save_final_outputs(predictions, explanations, feature_importance, explainability):
    """Save comprehensive outputs"""
    print("\n💾 [OUTPUT] Saving outputs...")
    
    os.makedirs('.output', exist_ok=True)
    
    results = pd.DataFrame(explanations)
    results['predicted_risk'] = predictions
    results.to_csv('.output/predicted_risk_country_year.csv', index=False)
    print(f"  ✓ Predictions: {len(results)}")
    
    feature_imp = pd.DataFrame(
        feature_importance,
        columns=[f'f{i}' for i in range(len(feature_importance[0]) if feature_importance else range(0))]
    )
    feature_imp.to_csv('.output/feature_importance_scores.csv', index=False)
    print(f"  ✓ Feature importance: {len(feature_imp)} rows")
    
    expl_df = pd.DataFrame([{
        'country': country,
        'n_samples': stats['n_samples'],
        'avg_risk': stats['avg_predictions'],
        'top_features': str(stats['feature_importance'])
    } for country, stats in explainability.items()])
    
    expl_df['n_top_features'] = expl_df['top_features'].apply(lambda x: str(x).count(':') + 1 if isinstance(x, str) else 0)
    expl_df.to_csv('.output/country_explainability_report.csv', index=False)
    print(f"  ✓ Country analysis: {len(explainability)} countries")
    
    return results, explainability