#CARICAMENTO E DIVISIONE DEI DATI 
#La procedura inizia con il caricamento dei dati in formato parquet tramite la libreria pandas. 
#I singoli file vengono aggregati in un unico DataFrame globale attraverso il metodo concat. 
#Per valutare la robustezza del classificatore in contesti eterogenei, il dataset viene 
#partizionato in due sotto-insiemi distinti in base alla colonna Where: Home1 (indicata come 
#Location A) e UniVR (indicata come Location B).

import pandas as pd
import os


def load_and_split_data(DATASET_PATH):
    #genera la lista dei percorsi completi per tutti i file .parquet presenti nella directory specificata da DATASET_PATH
    all_files = [os.path.join(DATASET_PATH, f) for f in os.listdir(DATASET_PATH) if f.endswith('.parquet')]
    
    #converte ogni file .parquet in un DataFrame e li concatena in un unico DataFrame globale
    df_list = [pd.read_parquet(file) for file in all_files] #leggo ogni file e lo trasformo in tabella 
    df = pd.concat(df_list, ignore_index=True) #concateno tutte le tabelle in un unico DataFrame
    
    #partiziona il dataset in due sotto-insiemi distinti in base alla colonna Where: Home1 (indicata come Location A) e UniVR (indicata come Location B)
    df_home = df[df['Where'] == 'Home 1'].copy()  
    df_univr = df[df['Where'] == 'UniVR'].copy()  

    return df_home, df_univr

