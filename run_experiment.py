"""
COM748 — Full experiment pipeline (local version).
Reproduces all figures and metric tables from the work doc.
Run: python3 run_experiment.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap
import lime
import lime.lime_tabular

RANDOM_STATE = 42
OUTPUT_DIR   = '/home/khalid/research'

# ── 1. Load data ─────────────────────────────────────────────────────────────
print('Loading data...')
df_raw = pd.read_csv(f'{OUTPUT_DIR}/heart_disease_uci.csv')
print(f'Raw shape: {df_raw.shape}')

# ── 2. Dataset understanding ─────────────────────────────────────────────────
print('\n=== Dataset Distribution ===')
print(df_raw['dataset'].value_counts())

print('\n=== Missing Values ===')
missing    = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(1)
missing_df  = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df  = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
print(missing_df)

# Save missing values bar chart
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(missing_df.index, missing_df['Missing %'], color='steelblue')
ax.set_ylabel('Missing (%)')
ax.set_title('Missing Value Percentage per Feature')
ax.set_ylim(0, 100)
for i, (col, row) in enumerate(missing_df.iterrows()):
    ax.text(i, row['Missing %'] + 1, f"{row['Missing %']}%", ha='center', fontsize=9)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/missing_values.png', dpi=300)
plt.close()
print('Saved: missing_values.png')

# ── 3. Preprocessing ──────────────────────────────────────────────────────────
print('\nPreprocessing...')
df = df_raw.copy()
df.drop(columns=[c for c in ['id', 'dataset', 'index'] if c in df.columns], inplace=True)
df.replace('', np.nan, inplace=True)

# Binarise target: 0 = no disease, 1 = disease present
df['target'] = (df['num'] > 0).astype(int)
df.drop(columns=['num'], inplace=True)

# Clip physically impossible negative oldpeak values before IQR capping
df['oldpeak'] = pd.to_numeric(df['oldpeak'], errors='coerce').clip(lower=0)

# IQR outlier capping — 'ca' excluded (discrete integer count 0–3)
cap_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
before_stats = df[cap_cols].describe()

for col in cap_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR     = Q3 - Q1
    df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

after_stats = df[cap_cols].describe()

# Outlier capping box plots
fig, axes = plt.subplots(2, len(cap_cols), figsize=(18, 8))
fig.suptitle('Outlier Capping — Before vs After (IQR 1.5×)', fontsize=13)
for i, col in enumerate(cap_cols):
    axes[0, i].boxplot(df_raw[col].dropna())
    axes[0, i].set_title(f'{col} (Before)')
    axes[1, i].boxplot(df[col].dropna())
    axes[1, i].set_title(f'{col} (After)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/outlier_capping.png', dpi=300)
plt.close()
print('Saved: outlier_capping.png')

# ── 4. Feature setup and split ───────────────────────────────────────────────
numeric_features     = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

X = df[numeric_features + categorical_features]
y = df['target']

print(f'\nFinal dataset: {X.shape[0]} rows, {X.shape[1]} features')
print(f'Disease-positive rate: {y.mean():.1%}  (n={y.sum()}, no disease n={len(y)-y.sum()})')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f'Train: {X_train.shape[0]} | Test: {X_test.shape[0]}')
print(f'Train disease rate: {y_train.mean():.1%} | Test disease rate: {y_test.mean():.1%}')

# ── 5. Preprocessing pipeline ────────────────────────────────────────────────
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ── 6. Classifiers ───────────────────────────────────────────────────────────
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=RANDOM_STATE),
    'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=None,
                                                   min_samples_split=2, random_state=RANDOM_STATE),
    'XGBoost':             XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                                          eval_metric='logloss', random_state=RANDOM_STATE,
                                          verbosity=0),
    'SVM':                 SVC(C=1.0, kernel='rbf', gamma='scale', probability=True,
                                random_state=RANDOM_STATE),
    'KNN':                 KNeighborsClassifier(n_neighbors=5, metric='minkowski'),
}

# Hyperparameter table
print('\n=== Hyperparameters ===')
hyper_rows = [
    ('Logistic Regression', 'max_iter=1000, solver=lbfgs'),
    ('Random Forest',       'n_estimators=100, max_depth=None, min_samples_split=2'),
    ('XGBoost',             'n_estimators=100, learning_rate=0.1, max_depth=6, eval_metric=logloss'),
    ('SVM',                 'C=1.0, kernel=rbf, gamma=scale, probability=True'),
    ('KNN',                 'n_neighbors=5, metric=minkowski'),
]
for name, params in hyper_rows:
    print(f'  {name}: {params}')

# ── 7. Cross-validation with SMOTE inside folds ──────────────────────────────
print('\nRunning 5-fold cross-validation...')
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

cv_results = {}
for name, clf in classifiers.items():
    pipe = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote',        SMOTE(k_neighbors=5, random_state=RANDOM_STATE)),
        ('classifier',   clf)
    ])
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring,
                             return_train_score=False, n_jobs=-1)
    cv_results[name] = scores
    print(f'  {name} — AUC: {scores["test_roc_auc"].mean():.3f} ± {scores["test_roc_auc"].std():.3f}')

cv_summary = []
for name, scores in cv_results.items():
    cv_summary.append({
        'Model':     name,
        'Accuracy':  f"{scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}",
        'Precision': f"{scores['test_precision'].mean():.3f} ± {scores['test_precision'].std():.3f}",
        'Recall':    f"{scores['test_recall'].mean():.3f} ± {scores['test_recall'].std():.3f}",
        'F1':        f"{scores['test_f1'].mean():.3f} ± {scores['test_f1'].std():.3f}",
        'AUC':       f"{scores['test_roc_auc'].mean():.3f} ± {scores['test_roc_auc'].std():.3f}",
    })
cv_df = pd.DataFrame(cv_summary)

print('\n=== 5-Fold Cross-Validation Results (mean ± std) ===')
print(cv_df.to_string(index=False))

# ── 8. Fit final models on full training set ─────────────────────────────────
print('\nFitting final models on full training set...')
fitted_models = {}
test_results  = []

for name, clf in classifiers.items():
    pipe = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote',        SMOTE(k_neighbors=5, random_state=RANDOM_STATE)),
        ('classifier',   clf)
    ])
    pipe.fit(X_train, y_train)
    fitted_models[name] = pipe

    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    cm      = confusion_matrix(y_test, y_pred)

    test_results.append({
        'Model':     name,
        'Accuracy':  round(accuracy_score(y_test, y_pred), 3),
        'Precision': round(precision_score(y_test, y_pred), 3),
        'Recall':    round(recall_score(y_test, y_pred), 3),
        'F1':        round(f1_score(y_test, y_pred), 3),
        'AUC':       round(roc_auc_score(y_test, y_proba), 3),
        'TP':        int(cm[1, 1]),
        'FP':        int(cm[0, 1]),
        'TN':        int(cm[0, 0]),
        'FN':        int(cm[1, 0]),
    })
    print(f'  {name} — AUC: {test_results[-1]["AUC"]}, Recall: {test_results[-1]["Recall"]}')

test_df = pd.DataFrame(test_results)
print('\n=== Test Set Results — All 5 Models ===')
print(test_df.to_string(index=False))

# ── 9. Confusion matrices ─────────────────────────────────────────────────────
print('\nGenerating confusion matrices...')
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle('Confusion Matrices — Test Set', fontsize=13)

for ax, (name, pipe) in zip(axes, fitted_models.items()):
    y_pred = pipe.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    disp   = ConfusionMatrixDisplay(cm, display_labels=['No Disease', 'Disease'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name, fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrices.png', dpi=300)
plt.close()
print('Saved: confusion_matrices.png')

# ── 10. ROC curves ────────────────────────────────────────────────────────────
print('Generating ROC curves...')
fig, ax = plt.subplots(figsize=(8, 6))
for name, pipe in fitted_models.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
ax.set_title('ROC Curves — All 5 Classifiers')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/roc_curves.png', dpi=300)
plt.close()
print('Saved: roc_curves.png')

# ── 11. SHAP — Global interpretability (Random Forest) ───────────────────────
print('\nGenerating SHAP plots...')
rf_pipe = fitted_models['Random Forest']
prep    = rf_pipe.named_steps['preprocessor']
rf_clf  = rf_pipe.named_steps['classifier']

cat_feature_names = prep.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features).tolist()
all_feature_names = numeric_features + cat_feature_names

X_test_prep = prep.transform(X_test)

explainer  = shap.TreeExplainer(rf_clf)
shap_values = explainer.shap_values(X_test_prep)

# Handle both 2D (binary) and 3D (per-class) SHAP array formats
if isinstance(shap_values, list):
    sv_positive = shap_values[1]
elif shap_values.ndim == 3:
    sv_positive = shap_values[:, :, 1]
else:
    sv_positive = shap_values

# SHAP summary (bee swarm)
plt.figure()
shap.summary_plot(sv_positive, X_test_prep, feature_names=all_feature_names,
                  show=False, plot_type='dot')
plt.title('SHAP Summary Plot — Random Forest (Disease class)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: shap_summary.png')

# SHAP bar plot
plt.figure()
shap.summary_plot(sv_positive, X_test_prep, feature_names=all_feature_names,
                  plot_type='bar', show=False)
plt.title('SHAP Mean Absolute Feature Importance — Random Forest')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/shap_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: shap_bar.png')

# ── 12. SHAP — Waterfall (single true-positive patient) ──────────────────────
y_pred_rf = rf_pipe.predict(X_test)
pos_idx   = np.where((y_pred_rf == 1) & (y_test.values == 1))[0][0]

ev = explainer.expected_value
base_val = float(np.array(ev).flatten()[1]) if isinstance(ev, (list, np.ndarray)) else float(ev)

shap_exp = shap.Explanation(
    values        = sv_positive[pos_idx],
    base_values   = base_val,
    data          = X_test_prep[pos_idx],
    feature_names = all_feature_names,
)

plt.figure()
shap.waterfall_plot(shap_exp, max_display=12, show=False)
plt.title('SHAP Waterfall — True Positive Patient')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/shap_waterfall.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: shap_waterfall.png')

# ── 13. LIME — Local explanation (positive and negative patient) ──────────────
print('\nGenerating LIME explanations...')

def rf_predict_proba(X_array):
    return rf_clf.predict_proba(X_array)

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data  = prep.transform(X_train),
    feature_names  = all_feature_names,
    class_names    = ['No Disease', 'Disease'],
    discretize_continuous = True,
    random_state   = RANDOM_STATE,
)

# True-positive patient
lime_exp_pos = lime_explainer.explain_instance(
    data_row   = X_test_prep[pos_idx],
    predict_fn = rf_predict_proba,
    num_features = 10,
)
fig = lime_exp_pos.as_pyplot_figure()
fig.suptitle('LIME Explanation — True Positive Patient (Disease predicted)', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/lime_positive.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: lime_positive.png')

# True-negative patient
neg_idx = np.where((y_pred_rf == 0) & (y_test.values == 0))[0][0]
lime_exp_neg = lime_explainer.explain_instance(
    data_row   = X_test_prep[neg_idx],
    predict_fn = rf_predict_proba,
    num_features = 10,
)
fig = lime_exp_neg.as_pyplot_figure()
fig.suptitle('LIME Explanation — True Negative Patient (No Disease predicted)', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/lime_negative.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: lime_negative.png')

# ── 14. Final summary ─────────────────────────────────────────────────────────
print('\n' + '='*60)
print('PHASE 1 COMPLETE — All outputs generated')
print('='*60)
print(f'\nDataset: {X.shape[0]} rows, {X.shape[1]} features')
print(f'Disease-positive: {y.sum()} ({y.mean():.1%}) | No disease: {len(y)-y.sum()}')
print(f'Train/Test split: {X_train.shape[0]} / {X_test.shape[0]}')

print('\n=== Cross-Validation (5-fold, mean ± std) ===')
print(cv_df.to_string(index=False))

print('\n=== Test Set Results ===')
print(test_df[['Model','Accuracy','Precision','Recall','F1','AUC','TP','FP','TN','FN']].to_string(index=False))

print('\nFiles saved:')
files_saved = [
    'missing_values.png', 'outlier_capping.png', 'confusion_matrices.png',
    'roc_curves.png', 'shap_summary.png', 'shap_bar.png', 'shap_waterfall.png',
    'lime_positive.png', 'lime_negative.png',
]
for f in files_saved:
    print(f'  {OUTPUT_DIR}/{f}')
