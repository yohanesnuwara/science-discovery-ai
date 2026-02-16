#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Financial Crisis Forecasting with Explainable AI
Combines ESRB crisis data, JST macro indicators, and GPR geopolitical risk
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import brier_score_loss, precision_score

import matplotlib.pyplot as plt
import seaborn as sns

import re
import os
from collections import defaultdict

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("⚠️ Using device:", "CUDA (GPU)" if torch.cuda.is_available() else "CPU")


class CrisisDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(np.nan_to_num(X, nan=0))
        self.y = torch.FloatTensor(np.nan_to_num(y, nan=0)).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ExplainableTCN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.15)

        self.attn = nn.Sequential(
            nn.Linear(128, 64),
            nn.Softmax(dim=1)
        )

        self.importance = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, input_size),
            nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.regressor = nn.Linear(input_size, 32)
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, explain=False):
        if x.dim() == 3:
            x = x.permute(0, 2, 1)

        x = self.dropout2(self.relu(self.bn2(self.conv2(x))))
        importance = self.importance(torch.mean(x, dim=2))
        attn_weights = self.attn(torch.mean(x, dim=2))
        weighted = torch.mean(x, dim=2) * attn_weights
        features = self.importance(weighted)
        risk = self.output(self.regressor(features))

        return risk, features if explain else risk


def analyze_esrb_data(filepath):
    print("\n📋 Processing ESRB Financial Crises Database...")

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
    crisis_metadata['crisis_periods'] = crisis_metadata['end_year'] - crisis_metadata['start_year']
    crisis_metadata['crisis_count'] = 1

    labels_df = pd.DataFrame()
    labels_df['country'] = []
    labels_df['year'] = []
    labels_df['crisis_onset'] = []
    labels_df['duration'] = []

    for _, row in crisis_metadata.iterrows():
        for year in range(int(row['start_year']), int(row['end_year']) + 1):
            labels_df['country'].append(row['country'])
            labels_df['year'].append(year)
            labels_df['crisis_onset'].append(1)
            labels_df['duration'].append(int(row['crisis_periods']))

    labels_df = pd.DataFrame(labels_df)

    print(f"✓ Crisis periods: {len(crisis_metadata)}")
    print(f"✓ Annual labels: {len(labels_df)} country-years")

    return crisis_metadata, labels_df


def process_jst_data(filepath):
    print("\n📊 Processing JST Macrohistory Database...")

    try:
        df = pd.read_csv(filepath, low_memory=False)
    except:
        return pd.DataFrame()

    print(f"⚠️ Processing {len(df)} records")
    print(f"⚠️ Features: {df.columns.tolist()[:20]}...")

    df.columns = [col.lower().replace('-', '_').replace('/_', '_')
                  for col in df.columns]

    essential_cols = ['year', 'country'] + ['gdp', 'stir', 'debtgdp', 'unemp', 'eq_tr']

    essential_cols = [col for col in essential_cols if col in df.columns]

    jst_clean = df[essential_cols].copy()
    jst_clean = jst_clean.dropna(subset=essential_cols[2:]).copy()
    jst_clean = jst_clean.sort_values(['country', 'year']).reset_index(drop=True)

    print(f"✓ Processed: {len(jst_clean)} records")

    return jst_clean, essential_cols


def process_gpr_data(filepath):
    print("\n🗺️ Processing GPR Geopolitical Risk Data...")

    try:
        gpr_df = pd.read_csv(filepath)
    except:
        return pd.DataFrame()

    print(f"✓ Processing {len(gpr_df)} records")

    gpr_df['date'] = pd.to_datetime(gpr_df['month'])
    gpr_df['year'] = gpr_df['date'].dt.year

    for col in gpr_df.columns:
        if col.startswith('GPRC_') and len(col) == 6:
            gpr_df[col] = gpr_df[col].mean() if gpr_df[col].notna().sum() > 0 else 0

    gpr_df = gpr_df.dropna(subset=['year']).sort_values('year')

    print(f"✓ Processed: {len(gpr_df)} yearly records")

    return gpr_df


def merge_datasets(jst_df, gpr_df, labels_df):
    print("\n🔗 Merging datasets...")

    countries = ['US', 'DEU', 'FRA', 'GBR', 'JPN', 'ESP', 'ITA', 'CAN', 'CHN', 'BRA']

    results = []

    for idx, row in labels_df.iterrows():
        country = row['country']
        year = row['year']

        if country not in countries:
            continue

        jst_row = jst_df[jst_df['year'] == year]
        if len(jst_row) == 0:
            continue

        jst_row = jst_row.iloc[0]

        features = {
            'gdp': float(jst_row.get('gdp', 1) if jst_row.get('gdp', 1) > 0 else 1),
            'interest': float(jst_row.get('stir', 0) if pd.notna(jst_row.get('stir', 0)) else 0),
            'debt': float(jst_row.get('debtgdp', 0) if pd.notna(jst_row.get('debtgdp', 0)) else 0),
            'risk': 0.0,
        }

        gpr_row = gpr_df[gpr_df['year'] == year]
        if len(gpr_row) > 0:
            feature_key = 'risk'
            if feature_key in features:
                features[feature_key] = float(gpr_row.iloc[0][feature_key] if pd.notna(gpr_row.iloc[0][feature_key]) else 0)

        features['country'] = country
        features['year'] = year
        features['crisis_indicator'] = row.get('crisis_onset', 0)

        results.append(features)

    merged = pd.DataFrame(results)

    print(f"✓ Merged: {len(merged)} prediction points")
    print(f"✓ Features: {len(merged.columns)}")
    print(f"✓ Total crisis: {len(merged[merged['crisis_indicator'] == 1]}")

    return merged


def get_feature_diagnostics(features_df):
    print("\n📈 Feature Diagnostics:")
    print(f"   Overall Statistics:")

    for col in features_df.columns:
        if col not in ['country', 'year']:
            print(f"      {col}: mean={features_df[col].mean():.4f}, std={features_df[col].std():.4f}")

    return {}