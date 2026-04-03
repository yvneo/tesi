import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt

BASE_DIR = "/storage_14tb/workspace_idio/tesi_anna_martini/risultati"
MODELS = ["rf", "knn", "xgboost"]
NUM_PACKETS = ["10", "20"]

def load_data():
    data = []
    for model in MODELS:
        for num in NUM_PACKETS:
            folder = os.path.join(BASE_DIR, model, num)
            if os.path.exists(folder):
                files = [f for f in os.listdir(folder) if f.endswith(".csv") and f.startswith("report_")]
                for file in files:
                    file_path = os.path.join(folder, file)
                    df = pd.read_csv(file_path)
                    scenario = file.replace("report_", "").replace(".csv", "")
                    f1_id_folds = df['f1_ID'].values
                    f1_ood_folds = df['f1_OOD_or_JD'].values
                    data.append({
                        "Model": model,
                        "Num_Packets": num,
                        "Scenario": scenario,
                        "F1_ID_Mean": np.mean(f1_id_folds),
                        "F1_OOD_Mean": np.mean(f1_ood_folds),
                        "F1_ID_Folds": f1_id_folds,
                        "F1_OOD_Folds": f1_ood_folds
                    })
    return pd.DataFrame(data)


def perform_statistical_tests(df):
    scenari = df['Scenario'].unique()

    for sce in scenari: 
        for num in NUM_PACKETS:
            rf_data = df[(df['Model'] == 'rf') & (df['Num_Packets'] == num) & (df['Scenario'] == sce)]
            xgboost_data = df[(df['Model'] == 'xgboost') & (df['Num_Packets'] == num) & (df['Scenario'] == sce)]
            
            if not rf_data.empty and not xgboost_data.empty:
                t_stat, p_value = stats.ttest_rel(rf_data.iloc[0]['F1_ID_Folds'], xgboost_data.iloc[0]['F1_ID_Folds'])

                print(f"Scenario: {sce}, Num_Packets: {num} - p-value: {p_value:.4f} if p-value < 0.05: {'Significant' if p_value < 0.05 else 'Not Significant'}")

           
