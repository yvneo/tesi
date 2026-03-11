#PROTOCOLLO DI VALUTAZIONE CROSS-LOCATION
#addestrare il modello su location a e testarlo sia su a che su b 
#divido Home in 5 parti 
#normalizzazione con fit solo su training set 
#label encoding 

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def train_and_evaluate_model(X_home, y_home, X_univr, y_univr):
    #codifica le etichette
    label_encoder = LabelEncoder()
    y_home_encoded = label_encoder.fit_transform(y_home)
    y_univr_encoded = label_encoder.transform(y_univr) 

    #configuro il K-Fold Cross Validation con K=5
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    results_report = []

    for fold, (train_index, test_index) in enumerate(kf.split(X_home)): #in ogni fold prendo 4 parti per il training e 1 parte per il test
        #divido i dati in training e test set
        X_train, X_test = X_home[train_index], X_home[test_index]
        y_train, y_test = y_home_encoded[train_index], y_home_encoded[test_index]

        #eseguo normalizzazione con MinMaxScaler
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train) #solo su training set
        X_test_scaled = scaler.transform(X_test) 
        X_univr_scaled = scaler.transform(X_univr) #applico la stessa trasformazione anche a UniVR

        #addestro il modello 
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        #valuto il modello sia su Home che su UniVR
        y_pred_home = model.predict(X_test_scaled) #modello prova a indovinare le etichette del test set di Home

        for real, pred in zip(y_test, y_pred_home):
            #zip mi mette risposta vera e quella predetta una accanto all'altra
            results.append({'fold': fold+1, 'Domain': 'Home', 'Ground Truth': label_encoder.inverse_transform([real])[0], 'Predicted': label_encoder.inverse_transform([pred])[0]}) 

        y_pred_univr = model.predict(X_univr_scaled)
        for real, pred in zip(y_univr_encoded, y_pred_univr):
            results.append({'fold': fold+1, 'Domain': 'UniVR', 'Ground Truth': label_encoder.inverse_transform([real])[0], 'Predicted': label_encoder.inverse_transform([pred])[0]}) 

        results_report.append({'fold': fold+1, 'acc_home': accuracy_score(y_test, y_pred_home), 'acc_univr': accuracy_score(y_univr_encoded, y_pred_univr)}) #salvo statistiche generali 

        df_results = pd.DataFrame(results)
        df_report = pd.DataFrame(results_report)

        
    return df_results, df_report