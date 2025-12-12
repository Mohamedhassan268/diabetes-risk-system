import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap

# Set plot style for professional output
plt.style.use('seaborn-v0_8-whitegrid')

# ---------------------------------------------------------
# PHASE 1: ADVANCED PREPROCESSING & ICD-9 MAPPING
# ---------------------------------------------------------

def map_icd9_code(code):
    """
    Maps specific ICD-9 codes to 9 distinct clinical categories.
    This reduces dimensionality from 700+ codes to 9 distinct features.
    """
    if code == '?' or pd.isna(code):
        return 'Missing'
    
    # Handle codes starting with V or E (Supplementary classification)
    if str(code).startswith('V') or str(code).startswith('E'):
        return 'Other'

    try:
        code_float = float(code)
    except ValueError:
        return 'Other'

    if (390 <= code_float <= 459) or code_float == 785:
        return 'Circulatory'
    elif (460 <= code_float <= 519) or code_float == 786:
        return 'Respiratory'
    elif (520 <= code_float <= 579) or code_float == 787:
        return 'Digestive'
    elif (250 <= code_float < 251):
        return 'Diabetes'
    elif (800 <= code_float <= 999):
        return 'Injury'
    elif (710 <= code_float <= 739):
        return 'Musculoskeletal'
    elif (580 <= code_float <= 629) or code_float == 788:
        return 'Genitourinary'
    elif (140 <= code_float <= 239):
        return 'Neoplasms'
    else:
        return 'Other'

def load_and_clean_data(filepath):
    print(">>> Loading Data...")
    # '?' is the placeholder for missing values in this dataset
    df = pd.read_csv(filepath, na_values='?')
    
    print(f"Original Shape: {df.shape}")

    # 1. Handling Missing Data "Intelligently"
    # Drop columns with >40% missing data.
    drop_cols = ['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr']
    df = df.drop(columns=drop_cols)
    
    # Drop rows with missing gender or diagnosis (very few rows, safe to drop)
    df = df.dropna(subset=['gender', 'diag_1', 'diag_2', 'diag_3'])
    
    # Remove 'Unknown/Invalid' gender
    df = df[df['gender'] != 'Unknown/Invalid']

    # 2. Target Engineering (Risk Definition)
    # We want to predict HIGH RISK: Readmission < 30 days.
    df['target'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    df = df.drop(columns=['readmitted'])
    
    print(f"Class Distribution (1 = High Risk): \n{df['target'].value_counts(normalize=True)}")
    
    return df

# ---------------------------------------------------------
# PHASE 2: FEATURE ENGINEERING (The "Imaginative" Part)
# ---------------------------------------------------------

def engineer_features(df):
    print(">>> Engineering Features...")
    
    # 1. Service Utilization (System Efficiency Proxy)
    df['service_utilization'] = (df['number_outpatient'] + 
                                 df['number_emergency'] + 
                                 df['number_inpatient'])
    
    # 2. Complexity Interaction
    df['complexity_score'] = df['num_medications'] * df['num_lab_procedures']
    
    # 3. ICD-9 Grouping (Dimensionality Reduction)
    df['diag_1_group'] = df['diag_1'].apply(map_icd9_code)
    
    # Drop raw diagnosis codes now that we have groups
    df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'])
    
    # 4. Medication Changes
    df['med_change'] = df['change'].apply(lambda x: 1 if x == 'Ch' else 0)
    df = df.drop(columns=['change'])

    # 5. Diabetes Meds
    df['on_diabetes_med'] = df['diabetesMed'].apply(lambda x: 1 if x == 'Yes' else 0)
    df = df.drop(columns=['diabetesMed'])
    
    # 6. One-Hot Encoding for remaining categorical variables
    cat_cols = ['race', 'gender', 'age', 'admission_type_id', 
                'discharge_disposition_id', 'admission_source_id', 'diag_1_group', 
                'max_glu_serum', 'A1Cresult']
    
    # Get dummies, drop_first=True to avoid dummy variable trap (collinearity)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # --- CRITICAL FIX FOR XGBOOST ---
    # XGBoost crashes if column names contain brackets [] or < symbols
    # We replace them with underscores
    print(">>> Sanitizing Column Names for XGBoost...")
    df.columns = df.columns.str.replace(r'[\[\]<]', '_', regex=True)
    
    # Drop remaining medication columns (20+ columns) for this demo
    med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
                'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
                'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
                'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 
                'metformin-pioglitazone']
    df = df.drop(columns=med_cols)
    
    print(f"Final Feature Set Shape: {df.shape}")
    return df

# ---------------------------------------------------------
# PHASE 3: MODELING (Minimizing Risk)
# ---------------------------------------------------------

def run_pipeline():
    # --- Load Data ---
    try:
        df = load_and_clean_data('diabetic_data.csv')
    except FileNotFoundError:
        print("ERROR: 'diabetic_data.csv' not found. Please download from UCI link.")
        return

    # --- Feature Engineering ---
    df = engineer_features(df)

    # --- Split Data ---
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # --- Model 1: Baseline (Logistic Regression) ---
    print("\n>>> Training Baseline Model (Logistic Regression)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    
    y_pred_lr = lr.predict(X_test_scaled)
    print("\n--- Baseline Results ---")
    print(classification_report(y_test, y_pred_lr))
    print(f"ROC-AUC: {roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:,1]):.3f}")

    # --- Model 2: Advanced (XGBoost) ---
    print("\n>>> Training Advanced Model (XGBoost)...")
    # Calculate scale_pos_weight: count(negative) / count(positive)
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio, 
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    print("\n--- XGBoost Results (The 'Systems' Model) ---")
    print(classification_report(y_test, y_pred_xgb))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_xgb):.3f}")

    # ---------------------------------------------------------
    # PHASE 4: EXPLAINABILITY & BUSINESS VALUE
    # ---------------------------------------------------------
    print("\n>>> Generating SHAP Explanations...")
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.title("Key Risk Drivers (Beshara Group Analysis)")
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    print("SHAP Summary Plot saved as 'shap_summary.png'")
    
    print("\n>>> BUSINESS INTERPRETATION (Manager Language) <<<")
    print("1. Look at 'service_utilization' on the plot. High values (red) push risk to the right.")
    print("   -> INSIGHT: Frequent fliers need a dedicated care coordinator.")
    print("2. Look at 'discharge_disposition'.")
    print("   -> INSIGHT: Patients sent home without support (vs SNF) are higher risk.")
    print("3. Recall vs Precision: We accepted lower precision to maximize Recall.")
    print("   -> VALUE: We catch more high-risk patients, preventing costly readmissions.")

if __name__ == "__main__":
    run_pipeline()