from data_loader import load_and_split_data
from feature_engineering import extract_features
from model_utils import train_and_evaluate_model
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
NUM_PACKETS = 20 # 10 o 20 
MODEL_TYPE = 'xgboost' # 'rf', 'knn', 'xgboost'
DATASET_PATH = "/storage_14tb/workspace_idio/UniVR_Data/dataset_df_exact_no0load_saturated_handshake_tail_no0loadFilter_extended_94bde95_7e0dbc17/"

print("--- FASE 1: Caricamento Dati ---")

home_df, univr_df = load_and_split_data(DATASET_PATH)
print(f"Dati Home caricati: {len(home_df)} | Dati UniVR caricati: {len(univr_df)}")

print(f"\n--- FASE 2: Estrazione Feature con {NUM_PACKETS} pacchetti ---")

X_home, y_home = extract_features(home_df, num_packets=NUM_PACKETS)
X_univr, y_univr = extract_features(univr_df, num_packets=NUM_PACKETS)
print(f"Matrice Home. Forma: {X_home.shape}") 
print(f"Matrice UniVR. Forma: {X_univr.shape}") 

print("\n--- FASE 3: Addestramento e Valutazione Modello ---")

#garantisco che il modello sia testato e addestrato su stesse applicazioni in entrambe location usando intersezione 
common_classes = np.intersect1d(np.unique(y_home), np.unique(y_univr))  

filter_home = np.isin(y_home, common_classes)
filter_univr = np.isin(y_univr, common_classes)

X_home_filtered = X_home[filter_home]
y_home_filtered = y_home[filter_home]
X_univr_filtered = X_univr[filter_univr]
y_univr_filtered = y_univr[filter_univr]
print(f"Campioni Home dopo filtro: {len(y_home_filtered)} | Campioni UniVR dopo filtro: {len(y_univr_filtered)}")

all_results = [] #per salvare i risultati di tutti gli scenari

def collect_results(df, location_name, label):
    for _, row in df.iterrows():
        all_results.append({
            'Location': location_name,
            'Scenario': label,
            'F1 Score': row['f1_OOD_or_JD'] 
        })

#1. addestramento su Home, test su Home
report_home_to_home = train_and_evaluate_model(X_home_filtered, y_home_filtered, X_home_filtered, y_home_filtered, scenario_name="Home_vs_Home", taxonomy_label="ID", model_type=MODEL_TYPE, seed=SEED, num_packets=NUM_PACKETS)
collect_results(report_home_to_home, location_name="Home", label="ID")
report_home_to_home.to_csv(f"risultati/{MODEL_TYPE}/{NUM_PACKETS}/report_home_to_home.csv", index=False)

#2. addestramento su Home, test su UniVR
report_home_to_univr = train_and_evaluate_model(X_home_filtered, y_home_filtered, X_univr_filtered, y_univr_filtered, scenario_name="Home_vs_UniVR", taxonomy_label="OOD", model_type=MODEL_TYPE, seed=SEED, num_packets=NUM_PACKETS)
collect_results(report_home_to_univr, location_name="UniVR", label="OOD")
report_home_to_univr.to_csv(f"risultati/{MODEL_TYPE}/{NUM_PACKETS}/report_home_to_univr.csv", index=False)

#3. addestramento su UniVR, test su UniVR
report_univr_to_univr = train_and_evaluate_model(X_univr_filtered, y_univr_filtered, X_univr_filtered, y_univr_filtered, scenario_name="UniVR_vs_UniVR", taxonomy_label="ID", model_type=MODEL_TYPE, seed=SEED, num_packets=NUM_PACKETS)
collect_results(report_univr_to_univr, location_name="UniVR", label="ID")
report_univr_to_univr.to_csv(f"risultati/{MODEL_TYPE}/{NUM_PACKETS}/report_univr_to_univr.csv", index=False)

#4. addestramento su UniVR, test su Home
report_univr_to_home = train_and_evaluate_model(X_univr_filtered, y_univr_filtered, X_home_filtered, y_home_filtered, scenario_name="UniVR_vs_Home", taxonomy_label="OOD", model_type=MODEL_TYPE, seed=SEED, num_packets=NUM_PACKETS)
collect_results(report_univr_to_home, location_name="Home", label="OOD")
report_univr_to_home.to_csv(f"risultati/{MODEL_TYPE}/{NUM_PACKETS}/report_univr_to_home.csv", index=False)

#5. addestramento su dati combinati, test su UniVR
X_combined = np.vstack((X_home_filtered, X_univr_filtered)) #unisco dati
y_combined = np.concatenate((y_home_filtered, y_univr_filtered)) #unisco etichette
report_jd_to_univr = train_and_evaluate_model(X_combined, y_combined, X_univr_filtered, y_univr_filtered, scenario_name="Combined_vs_UniVR", taxonomy_label="JD", model_type=MODEL_TYPE, seed=SEED, num_packets=NUM_PACKETS)
collect_results(report_jd_to_univr, location_name="UniVR", label="JD")
report_jd_to_univr.to_csv(f"risultati/{MODEL_TYPE}/{NUM_PACKETS}/report_jd_to_univr.csv", index=False)

#6. addestramento su dati combinati, test su Home
report_jd_to_home = train_and_evaluate_model(X_combined, y_combined, X_home_filtered, y_home_filtered, scenario_name="Combined_vs_Home", taxonomy_label="JD", model_type=MODEL_TYPE, seed=SEED, num_packets=NUM_PACKETS)
collect_results(report_jd_to_home, location_name="Home", label="JD")
report_jd_to_home.to_csv(f"risultati/{MODEL_TYPE}/{NUM_PACKETS}/report_jd_to_home.csv", index=False)    

#generazione barplot finale 
final_plot_df = pd.DataFrame(all_results)
plt.figure(figsize=(10, 6))
sns.barplot(data=final_plot_df, x='Location', y='F1 Score', hue='Scenario', capsize=0.1, errorbar='sd')
plt.title('Valutazione Cross-Domain: ID vs OOD vs JD')
plt.ylabel('F1 Score (%)')
plt.ylim(0, 105)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"grafici/{MODEL_TYPE}/{NUM_PACKETS}/barplot_finale_tassonomia.png", dpi=300, bbox_inches='tight')
plt.show()