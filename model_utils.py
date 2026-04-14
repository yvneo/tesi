#PROTOCOLLO DI VALUTAZIONE CROSS-LOCATION
#addestrare il modello su location a e testarlo sia su a che su b 
#divido Home in 5 parti 
#normalizzazione con fit solo su training set 
#label encoding 

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sage 
import os

def get_model(model_name, seed):
    if model_name == 'rf':
        return RandomForestClassifier(n_estimators=100, 
                                    random_state=seed, 
                                    class_weight='balanced')
    elif model_name == 'knn':
        return KNeighborsClassifier(n_neighbors=20, 
                                    metric = 'manhattan', 
                                    weights='distance')
    elif model_name == 'xgboost':
        return XGBClassifier(
            n_estimators=250,           # Bilancio ideale
            learning_rate=0.08,         # Abbastanza basso per precisione, abbastanza alto per convergere
            tree_method='hist',         # FONDAMENTALE: usa gli istogrammi 
            max_bin=256,                # Velocizza il calcolo riducendo i "cestini" delle feature, standard per hist
            
            max_depth=5,                # Profondità media: cattura i pattern senza "imparare a memoria"
            subsample=0.8,              # Usa l'80% dei dati per albero (riduce rumore e accelera)
            colsample_bytree=0.8,       # Usa l'80% delle feature per albero (fondamentale per i 10 pacchetti)
            
            gamma=1,                    # Impedisce split di rami se il guadagno è minimo (pulisce il rumore)
            random_state=seed,
            eval_metric='mlogloss')
    else:
        raise ValueError(f"Model {model_name} not supported.")

def train_and_evaluate_model(X_train_data, y_train_data, X_test_ext, y_test_ext, scenario_name, taxonomy_label, model_type, seed, num_packets):
    #RECAP: X_train_data, y_train_data sono i dati su cui addestro il modello, X_test_ext, y_test_ext sono i dati su cui testare il modello
    # X -> dati, y -> etichette 

    # 1. LABEL ENCODING
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_data) #imparo le classi dal training set e associo un numero a ciascuna classe
    y_ext_encoded = label_encoder.transform(y_test_ext) #applico la stessa codifica al test set esterno
    class_names = label_encoder.classes_

    # 2. CONFIGURAZIONE DELLA CROSS-VALIDATION
    kf = KFold(n_splits=5, shuffle=True, random_state=seed) #4 parti per train e una per test
    results_report = [] #per salvare i risultati di ogni fold
    cm_internal_list = [] #per test fatti sulla stessa location di train
    cm_external_list = [] #per test fatti su location diversa da quella di train

    #definisco i gruppi di feature per lo scaling: L4_payload_lenght, iat_micros, packet_dir, TCP_win_size
    feature_groups =[range(0, num_packets), range(num_packets, num_packets*2), range(num_packets*2, num_packets*3), range(num_packets*3, num_packets*4)]

    fold_importances = [] #per salvare l'importanza delle feature in ogni fold, solo per rf e xgboost

    for fold, (train_index, test_index) in enumerate(kf.split(X_train_data)):
        
        #divido i dati in train e validazione val per questo fold
        X_train, X_val = X_train_data[train_index], X_train_data[test_index]
        y_train, y_val = y_train_encoded[train_index], y_train_encoded[test_index]

        # 3.SCALING per feature, non per pacchetto 
        #inizializzo vettori vuoti con stessa forma di originali 
        X_train_scaled = X_train.copy().astype(float)
        X_val_scaled = X_val.copy().astype(float)
        X_ext_scaled = X_test_ext.copy().astype(float)

        for group in feature_groups:
            scaler = MinMaxScaler()
            # Trasformo il gruppo in un'unica colonna gigante per calcolare Min/Max globali del gruppo
            train_group = X_train[:, group]
            scaler.fit(train_group.reshape(-1, 1))
            
            # Applico la trasformazione all'intero blocco di colonne 
            X_train_scaled[:, group] = scaler.transform(train_group.reshape(-1, 1)).reshape(train_group.shape)
            X_val_scaled[:, group] = scaler.transform(X_val[:, group].reshape(-1, 1)).reshape(X_val[:, group].shape)
            X_ext_scaled[:, group] = scaler.transform(X_test_ext[:, group].reshape(-1, 1)).reshape(X_test_ext[:, group].shape)

        # 4. ADDDESTRAMENTO MODELLO
        model = get_model(model_type, seed)
        model.fit(X_train_scaled, y_train)

        #CAMPIONAMENTO BILANCIATO PER SAGE 
        df_temp = pd.DataFrame(X_train_scaled)
        df_temp['label'] = y_train
        samples_per_class = 25
        balanced_samples = df_temp.groupby('label').apply(lambda x: x.sample(n = min(len(x), samples_per_class), replace=True, random_state=seed)).reset_index(drop=True)
        X_train_balanced_samples = balanced_samples.drop(columns=['label']).values
        
        #utilizzo sage per calcolare l'importanza delle feature 
        imputer = sage.MarginalImputer(model, X_train_balanced_samples) #utilizzo campioni bilanciati per calcolare importanza delle feature, soprattutto per xgboost che è molto lento con molti campioni
        estimator = sage.PermutationEstimator(imputer)
        n_samples = min(15, len(X_val_scaled)) #limito il numero di campioni per velocizzare il calcolo, soprattutto con xgboost   
        sage_values = estimator(X_val_scaled[:n_samples], y_val[: n_samples], n_permutations=10, thresh = 0.08, detect_convergence = False) #calcolo importanza delle feature con 10 permutazioni, bilanciando accuratezza e tempo di calcolo
        fold_importances.append(sage_values.values) #salvo importanza delle feature per questo fold, solo per rf e xgboost
        
        # 5. PREDIZIONI
        y_pred_int = model.predict(X_val_scaled) #test su stessa location di train
        y_pred_ext = model.predict(X_ext_scaled) #test su location diversa da quella di train

        # 6. CALCOLO METRICHE: bal_acc = media della precisione di ogni singola classe, f1 = media armonica di precisione e recall, macro = media non pesata delle metriche per ogni classe
        f1_int = f1_score(y_val, y_pred_int, average='macro')*100
        f1_ext = f1_score(y_ext_encoded, y_pred_ext, average='macro')*100
        bal_acc_int = balanced_accuracy_score(y_val, y_pred_int)*100
        bal_acc_ext = balanced_accuracy_score(y_ext_encoded, y_pred_ext)*100

        results_report.append({
            'fold': fold+1, 
            'f1_ID': f1_int, 'bal_acc_ID': bal_acc_int,
            'f1_OOD_or_JD': f1_ext, 'bal_acc_OOD_or_JD': bal_acc_ext,
            'Taxonomy': taxonomy_label
        })

        # 7. MATRICI DI CONFUSIONE NORMALIZZATE IN PERCENTUALE
        cm_int = confusion_matrix(y_val, y_pred_int, labels=range(len(class_names)), normalize='true') * 100
        cm_ext = confusion_matrix(y_ext_encoded, y_pred_ext, labels=range(len(class_names)), normalize='true') * 100

        cm_internal_list.append(cm_int)
        cm_external_list.append(cm_ext)
        print("Ho finito il fold ", fold+1, " su ", scenario_name)
        
    # 8. REPoRT E VISUALIZZAZIONE
    df_report = pd.DataFrame(results_report)

    if len(fold_importances) > 0:
        avg_importances = np.mean(fold_importances, axis=0)
        feature_data = []
        for i in range(num_packets): 
            feature_data.append({'Feature': 'L4_payload', 'Packet': i+1, 'Importance': avg_importances[i]})
            feature_data.append({'Feature': 'iat_micros', 'Packet': i+1, 'Importance': avg_importances[i + num_packets]})
            feature_data.append({'Feature': 'packet_dir', 'Packet': i+1, 'Importance': avg_importances[i + num_packets*2]})
            feature_data.append({'Feature': 'TCP_win_size', 'Packet': i+1, 'Importance': avg_importances[i + num_packets*3]})
        df_importance = pd.DataFrame(feature_data)
        df_importance.to_csv(f"risultati/{model_type}/{num_packets}/feature_importance_{scenario_name}.csv", index=False)
    
    #calcolo media e deviazione standard 
    numeric_cols = df_report.select_dtypes(include=[np.number]).drop(columns=['fold'])
    stats = numeric_cols.agg(['mean', 'std']).T
    stats['Type'] = 'Global Metric (%)'

    #media delle 5 matrici dei 5 fold per internal ed external
    cm_internal_avg = np.mean(cm_internal_list, axis=0)
    cm_external_avg = np.mean(cm_external_list, axis=0)

    #creo dataframe per classi 
    class_report = pd.DataFrame({
        'mean': np.diag(cm_internal_avg), #percentuale di successo media quando train e test in stessa locarion
        'std': np.std([np.diag(cm) for cm in cm_internal_list], axis=0), #oscillazione di questa percentuale nei 5 fold, train e test in stessa location
        'mean_ext': np.diag(cm_external_avg), #percentuale di successo media quando train e test in location diversa
        'std_ext': np.std([np.diag(cm) for cm in cm_external_list], axis=0), #oscillazione di questa percentuale nei 5 fold, train e test in location diversa
        'Type': 'Per Class'
    }, index=class_names)
    class_report['Type'] = 'Per Class Accuracy (%)'
    
    final_report = pd.concat([stats, class_report], axis=0)
    final_report = final_report.round(2)
    final_report.to_csv(f"risultati/{model_type}/{num_packets}/final_report_{scenario_name}.csv", index=True)

    # Visualizzazione
    plot_results(df_report, cm_internal_avg, cm_external_avg, class_names, scenario_name, taxonomy_label, model_type, num_packets)

    return df_report


def plot_results(df_report, cm_internal_avg, cm_external_avg, class_names, scenario_name, taxonomy_label, model_type, num_packets):
    #heatmap 
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm_internal_avg, annot=True, fmt=".1f", cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f'Avg Confusion Matrix % - ID ')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    sns.heatmap(cm_external_avg, annot=True, fmt=".1f", cmap='Oranges', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f'Avg Confusion Matrix % - {taxonomy_label} ({scenario_name})')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig(f"grafici/{model_type}/{num_packets}/heatmap_{scenario_name}.png", dpi=300)
    plt.show()

