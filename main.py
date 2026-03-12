from data_loader import load_and_split_data
from feature_engineering import extract_features
from model_utils import train_and_evaluate_model

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
results_df, report_df = train_and_evaluate_model(X_home, y_home, X_univr, y_univr)

report_df.to_csv("risultati/RFreport.csv", index=False)
print("\nReport salvato come 'report.csv'")

results_df.to_csv("risultati/RFresults.csv", index=False)
print("Risultati salvati come 'results.csv'")

