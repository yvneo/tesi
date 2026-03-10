import pandas as pd
import os

DATASET_PATH = "/storage_14tb/workspace_idio/UniVR_Data/dataset_df_exact_no0load_saturated_handshake_tail_no0loadFilter_extended_94bde95_7e0dbc17/"

print("Verifica connessione ai dati...")

if os.path.exists(DATASET_PATH):
    files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.parquet')]
    print(f"Successo! Trovati {len(files)} file .parquet")
    
    if files:
        primo_file = os.path.join(DATASET_PATH, files[0])
        df = pd.read_parquet(primo_file)
        print(f"Dati caricati correttamente. Forma del dataframe: {df.shape}")
else:
    print("Errore: Non riesco a trovare il path del dataset. Controlla se ci sono errori di battitura.")