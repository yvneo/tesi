from data_loader import load_and_split_data
from feature_engineering import extract_features

DATASET_PATH = "/storage_14tb/workspace_idio/UniVR_Data/dataset_df_exact_no0load_saturated_handshake_tail_no0loadFilter_extended_94bde95_7e0dbc17/"

print("--- FASE 1: Caricamento Dati ---")
home_df, univr_df = load_and_split_data(DATASET_PATH)
print(f"Dati Home caricati: {len(home_df)} | Dati UniVR caricati: {len(univr_df)}")

print("\n--- FASE 2: Estrazione Feature ---")
X_home, y_home = extract_features(home_df)
print(f"Matrice Home pronta. Forma: {X_home.shape}") 
