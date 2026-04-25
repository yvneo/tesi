#INGEGNERIZZAZIONE DELLE FEAUTURE E RAPPRESENTAZIONE DELL'INPUT 
#trasforma i biflussi in formato tabellare NumPy. Per ogni biflusso vengono estratti i primi N pacchetti, e per ciascun pacchetto vengono selezionati 4 campi:
#L4_payload_bytes_dir, iat_micros, packet_dir, BF_TCP_win_size_dir.
#Se un biflusso contiene meno di N pacchetti, i campi mancanti vengono riempiti con zero (o 0.5 per packet_dir).

import numpy as np
def extract_features(df, num_packets):
    features_list = [] #ospita i vettori finali (con 40/80 elementi ciasuno)
    labels_list = [] #ospita le etichette di classificazione 
    
    #iterazione su ogni biflusso presente del DataFrame 
    for _, row in df.iterrows():

        #estrazione delle 4 feature target limitate ai primi N pacchetti
        pl = row['L4_payload_bytes_dir'][:num_packets]
        iat = row['iat_micros'][:num_packets]
        dire = row['packet_dir'][:num_packets]
        win = row['BF_TCP_win_size_dir'][:num_packets]

        #inizializzazione dei vettori di padding con valori neautri (zero o 0.5 per direzione)
        pl_padded = np.zeros(num_packets)
        iat_padded = np.zeros(num_packets)
        dire_padded = np.full(num_packets, 0.5)
        win_padded = np.zeros(num_packets)

        #inserimento dei dati estratti nei vettori preallocati
        pl_padded[:len(pl)] = pl
        iat_padded[:len(iat)] = iat
        dire_padded[:len(dire)] = dire
        win_padded[:len(win)] = win 

        #concatenazione delle feature in un unico vettore riga 
        row_features = np.concatenate([pl_padded, iat_padded, dire_padded, win_padded])
        features_list.append(row_features)
        labels_list.append(row['BF_label']) 
    
    #ritorna i dati in formato NumPy array, pronti per essere utilizzati come input per i modelli di machine learning
    return np.array(features_list), np.array(labels_list)