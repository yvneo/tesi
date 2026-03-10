#CARICAMENTO E DIVISIONE DEI DATI 
#La procedura inizia con il caricamento dei dati in formato parquet tramite la libreria pandas. 
#I singoli file vengono aggregati in un unico DataFrame globale attraverso il metodo concat. 
#Per valutare la robustezza del classificatore in contesti eterogenei, il dataset viene 
#partizionato in due sotto-insiemi distinti in base alla colonna Where: Home1 (indicata come 
#Location A) e UniVR (indicata come Location B).

import pandas as pd
import os

#definzione del path del dataset
#DATASET_PATH = "/storage_14tb/workspace_idio/UniVR_Data/dataset_df_exact_no0load_saturated_handshake_tail_no0loadFilter_extended_94bde95_7e0dbc17/"

def load_and_split_data(DATASET_PATH):
    #creo il persorso completo per ogni file .parquet presente nella cartella
    all_files = [os.path.join(DATASET_PATH, f) for f in os.listdir(DATASET_PATH) if f.endswith('.parquet')]
    
    #leggo tutti i file e li concateno in un unico DataFrame
    df_list = [pd.read_parquet(file) for file in all_files] #leggo ogni file e lo trasformo in tabella 
    df = pd.concat(df_list, ignore_index=True) #concateno tutte le tabelle in un unico DataFrame
    
    #divido il dataset in due sottoinsiemi basati sulla colonna 'Where'
    #copy per evitare avvisi di SettingWithCopyWarning quando si lavora con i sottoinsiemi del DataFrame
    df_home = df[df['Where'] == 'Home 1'].copy()  # Location A
    df_univr = df[df['Where'] == 'UniVR'].copy()  # Location B

    return df_home, df_univr

#if __name__ == "__main__":
    #print("Caricamento e divisione dei dati in corso...")
    #df_home, df_univr = load_and_split_data()
    #print(f"Location A (Home 1) - Numero di campioni: {len(df_home)}")
    #print(f"Location B (UniVR) - Numero di campioni: {len(df_univr)}")