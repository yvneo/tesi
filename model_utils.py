#PROTOCOLLO DI VALUTAZIONE CROSS-LOCATION
#addestrare il modello su location a e testarlo sia su a che su b 
#divido Home in 5 parti 
#normalizzazione con fit solo su training set 
#label encoding 

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_and_evaluate_model(X_train_data, y_train_data, X_test_ext, y_test_ext, scenario_name):
    #RECAP: X_train_data, y_train_data sono i dati su cui addestro il modello, X_test_ext, y_test_ext sono i dati su cui testare il modello
    # X -> dati, y -> etichette 

    # 1. LABEL ENCODING
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_data) #imparo le classi dal training set e associo un numero a ciascuna classe
    y_ext_encoded = label_encoder.transform(y_test_ext) #applico la stessa codifica al test set esterno
    class_names = label_encoder.classes_

    # 2. CONFIGURAZIONE DELLA CROSS-VALIDATION
    kf = KFold(n_splits=5, shuffle=True, random_state=42) #4 parti per train e una per test
    results_report = [] #per salvare i risultati di ogni fold
    cm_internal_list = [] #per test fatti sulla stessa location di train
    cm_external_list = [] #per test fatti su location diversa da quella di train

    #definisco i gruppi di feature per lo scaling: L4_payload_lenght, iat_micros, packet_dir, TCP_win_size
    feature_groups =[range(0, 10), range(10, 20), range(20, 30), range(30, 40)]


    for fold, (train_index, test_index) in enumerate(kf.split(X_train_data)):
        
        #divido i dati in train e validazione val per questo fold
        X_train, X_val = X_train_data[train_index], X_train_data[test_index]
        y_train, y_val = y_train_encoded[train_index], y_train_encoded[test_index]

        # 3.SCALING per feature, non per pacchetto 
        #inizializzo vettori vuoti con stessa forma di originali 
        X_train_scaled = np.zeros_like(X_train, dtype=float)
        X_val_scaled = np.zeros_like(X_val, dtype=float)
        X_ext_scaled = np.zeros_like(X_test_ext, dtype=float)

        for group in feature_groups:
            scaler = MinMaxScaler()
            #fit scaler solo sui dati di train per questo gruppo di feature
            data_to_fit = X_train[:, group].reshape(-1, 1) #nella matrice di addestramento prendo tutte le righe delle colonne del gruppo di feature, reshape per avere una matrice colonna
            scaler.fit(data_to_fit) #calcolo min e max su questi dati per questo gruppo di feature
            #applico la trasformazione a train, val e test esterno per questo gruppo di feature
            for i in group:
                X_train_scaled[:, i] = scaler.transform(X_train[:, i].reshape(-1, 1)).flatten()
                X_val_scaled[:, i] = scaler.transform(X_val[:, i].reshape(-1, 1)).flatten()
                X_ext_scaled[:, i] = scaler.transform(X_test_ext[:, i].reshape(-1, 1)).flatten()

        # 4. ADDDESTRAMENTO MODELLO
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train_scaled, y_train)

        # 5. PREDIZIONI
        y_pred_int = model.predict(X_val_scaled) #test su stessa location di train
        y_pred_ext = model.predict(X_ext_scaled) #test su location diversa da quella di train

        # 6. CALCOLO METRICHE: bal_acc = media della precisione di ogni singola classe, f1 = media armonica di precisione e recall, macro = media non pesata delle metriche per ogni classe
        f1_int = f1_score(y_val, y_pred_int, average='macro')
        f1_ext = f1_score(y_ext_encoded, y_pred_ext, average='macro')
        bal_acc_int = balanced_accuracy_score(y_val, y_pred_int)
        bal_acc_ext = balanced_accuracy_score(y_ext_encoded, y_pred_ext)

        results_report.append({
            'fold': fold+1, 
            'f1_Internal': f1_int, 'bal_acc_Internal': bal_acc_int,
            'f1_External': f1_ext, 'bal_acc_External': bal_acc_ext 
        })

        # 7. MATRICI DI CONFUSIONE NORMALIZZATE IN PERCENTUALE
        cm_int = confusion_matrix(y_val, y_pred_int, labels=range(len(class_names)), normalize='true') * 100
        cm_ext = confusion_matrix(y_ext_encoded, y_pred_ext, labels=range(len(class_names)), normalize='true') * 100

        cm_internal_list.append(cm_int)
        cm_external_list.append(cm_ext)
        
    # 8. REPoRT E VISUALIZZAZIONE
    df_report = pd.DataFrame(results_report)

    #calcolo media e deviazione standard 
    stats = (df_report.drop(columns='fold').agg(['mean', 'std'])*100).T
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
    final_report = final_report.round(3)
    final_report.to_csv(f"risultati/final_report_{scenario_name}.csv", index=True)

    # Visualizzazione
    plot_results(df_report, cm_internal_avg, cm_external_avg, class_names, scenario_name)

    return df_report


def plot_results(df_report, cm_internal_avg, cm_external_avg, class_names, scenario_name):
    #preparazione dei dati per il barplot
    metrics = ['f1', 'bal_acc']
    domains = {'Internal', 'External'}
    plot_data = []

    for _, row in df_report.iterrows():
        for m in metrics:
            for d in domains:
                plot_data.append({
                    'Metric': m,
                    'Location': d,
                    'Score': row[f"{m}_{d}"]
                })

    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    #asse x metrica, y valore medio
    sns.barplot(data = plot_df, x='Metric', y='Score', hue='Location', capsize=0.1, errorbar='sd')
    plt.title(f'Performance Scenario: {scenario_name}')
    plt.ylim(0, 1.1)
    plt.savefig(f"grafici/barplot_performance_{scenario_name}.png", dpi=300, bbox_inches='tight')
    plt.show()

    #heatmap 
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm_internal_avg, annot=True, fmt=".1f", cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f'Avg Confusion Matrix % - Internal ({scenario_name})')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    sns.heatmap(cm_external_avg, annot=True, fmt=".1f", cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f'Avg Confusion Matrix % - External ({scenario_name})')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig(f"grafici/heatmap_{scenario_name}.png", dpi=300)
    plt.show()

