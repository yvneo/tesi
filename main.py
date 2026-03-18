from data_loader import load_and_split_data
from feature_engineering import extract_features
from model_utils import train_and_evaluate_model
import numpy as np


DATASET_PATH = "/storage_14tb/workspace_idio/UniVR_Data/dataset_df_exact_no0load_saturated_handshake_tail_no0loadFilter_extended_94bde95_7e0dbc17/"

print("--- FASE 1: Caricamento Dati ---")
home_df, univr_df = load_and_split_data(DATASET_PATH)
print(f"Dati Home caricati: {len(home_df)} | Dati UniVR caricati: {len(univr_df)}")

print("\n--- FASE 2: Estrazione Feature ---")
X_home, y_home = extract_features(home_df)
X_univr, y_univr = extract_features(univr_df)
print(f"Matrice Home. Forma: {X_home.shape}") 
print(f"Matrice UniVR. Forma: {X_univr.shape}") 

print("\n--- FASE 3: Addestramento e Valutazione Modello ---")

common_classes = np.intersect1d(np.unique(y_home), np.unique(y_univr))

filter_home = np.isin(y_home, common_classes)
filter_univr = np.isin(y_univr, common_classes)

X_home_filtered = X_home[filter_home]
y_home_filtered = y_home[filter_home]
X_univr_filtered = X_univr[filter_univr]
y_univr_filtered = y_univr[filter_univr]
print(f"Campioni Home dopo filtro: {len(y_home_filtered)} | Campioni UniVR dopo filtro: {len(y_univr_filtered)}")

#SCENARIO 1: Addestramento su Home, test su UniVR
print("\nSCENARIO 1: Addestramento su Home, test su UniVR")
report_scenario_1 = train_and_evaluate_model(X_home_filtered, y_home_filtered, X_univr_filtered, y_univr_filtered, scenario_name="Home_vs_UniVR")
report_scenario_1.to_csv("risultati/report_scenario_1.csv", index=False)

#SCENARIO 2: Addestramento su UniVR, test su Home
print("\nSCENARIO 2: Addestramento su UniVR, test su Home")
report_scenario_2 = train_and_evaluate_model(X_univr_filtered, y_univr_filtered, X_home_filtered, y_home_filtered, scenario_name="UniVR_vs_Home")
report_scenario_2.to_csv("risultati/report_scenario_2.csv", index=False)

#SCENARIO 3: Addestramento su dati combinati, test su Home
print("\nSCENARIO 3: Addestramento su dati combinati, test su Home")
X_combined = np.vstack((X_home_filtered, X_univr_filtered))
y_combined = np.concatenate((y_home_filtered, y_univr_filtered))
report_scenario_3 = train_and_evaluate_model(X_combined, y_combined, X_home_filtered, y_home_filtered, scenario_name="Combined_vs_Home")
report_scenario_3.to_csv("risultati/report_scenario_3.csv", index=False)
