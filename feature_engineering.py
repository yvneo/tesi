#INGEGNERIZZAZIONE DELLE FEAUTURE E RAPPRESENTAZIONE DELL'INPUT 
#primi 10 pacchetti di ciascun biflusso 
#per ognni pacchetto, estraggo 4 feauture: L4_payload_lenght, iat_micros, packet_dir e TCP_win_size
#flussi < 10 pacchetti sistemati con zero padding
#risultato array numpy con N righe e 40 colonne 

import numpy as np
def extract_features(df, num_packets=10):
    features_list = [] #contiene righe da 40 numeri
    labels_list = [] #contiene etichette 
    
    #itero su ogni riga (biflusso) 
    for _, row in df.iterrows():
        #estraggo 4 campi richiesti per i primi 10 pacchetti
        pl = row['L4_payload_bytes_dir'][:num_packets]
        iat = row['iat_micros'][:num_packets]
        dire = row['packet_dir'][:num_packets]
        win = row['BF_TCP_win_size_dir'][:num_packets]

        #creo vettore di 10 zeri per il padding
        pl_padded = np.zeros(num_packets)
        iat_padded = np.zeros(num_packets)
        dire_padded = np.full(num_packets, 0.5)
        win_padded = np.zeros(num_packets)

        #riempio i vettori con i valori estratti, lasciando zero per quelli mancanti
        pl_padded[:len(pl)] = pl
        iat_padded[:len(iat)] = iat
        dire_padded[:len(dire)] = dire
        win_padded[:len(win)] = win 

        #concateno le feauture in un unico vettore di 40 elementi
        row_features = np.concatenate([pl_padded, iat_padded, dire_padded, win_padded])
        features_list.append(row_features)
        labels_list.append(row['BF_label']) 
    
    return np.array(features_list), np.array(labels_list)