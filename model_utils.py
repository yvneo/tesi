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

    #trovo le classi comuni tra i due dataset
    #common_classes = np.intersect1d(np.unique(y_train_data), np.unique(y_test_ext))

    #filtro Home
    #mask_home = np.isin(y_train_data, common_classes)
    #X_home = X_train_data[mask_home]
    #y_home = y_train_data[mask_home]

    #filtro UniVR
    #mask_univr = np.isin(y_test_ext, common_classes)
    #X_univr = X_test_ext[mask_univr]
    #y_univr = y_test_ext[mask_univr]

    #print("Numero campioni Home dopo filtro:", len(y_home))
    #print("Numero campioni UniVR dopo filtro:", len(y_univr))
    
    # 1. CODIFICA DELLE ETICHETTE 
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_data) #imparo le classi dal training set
    y_ext_encoded = label_encoder.transform(y_test_ext) #applico la stessa codifica al test set esterno
    class_names = label_encoder.classes_

    # 2. CONFIGURAZIONE DELLA CROSS-VALIDATION
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results_report = []
    cm_internal_list = [] #per test fatti sulla stessa location di train
    cm_external_list = [] #per test fatti su location diversa da quella di train

    #definisco i gruppi di feature per lo scaling: L4_payload_lenght, iat_micros, packet_dir, TCP_win_size
    feature_groups =[range(0, 10), range(10, 20), range(20, 30), range(30, 40)]


    for fold, (train_index, test_index) in enumerate(kf.split(X_train_data)):
        
        X_train, X_val = X_train_data[train_index], X_train_data[test_index]
        y_train, y_val = y_train_encoded[train_index], y_train_encoded[test_index]

        # 3.SCALING 
        X_train_scaled = np.zeros_like(X_train, dtype=float)
        X_val_scaled = np.zeros_like(X_val, dtype=float)
        X_ext_scaled = np.zeros_like(X_test_ext, dtype=float)

        for group in feature_groups:
            scaler = MinMaxScaler()
            data_to_fit = X_train[:, group].reshape(-1, 1)
            scaler.fit(data_to_fit)
            for i in group:
                X_train_scaled[:, i] = scaler.transform(X_train[:, i].reshape(-1, 1)).flatten()
                X_val_scaled[:, i] = scaler.transform(X_val[:, i].reshape(-1, 1)).flatten()
                X_ext_scaled[:, i] = scaler.transform(X_test_ext[:, i].reshape(-1, 1)).flatten()

        # 4. ADDDESTRAMENTO MODELLO
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train_scaled, y_train)

        # 5. PREDIZIONI
        y_pred_int = model.predict(X_val_scaled)
        y_pred_ext = model.predict(X_ext_scaled)

        # 6. CALCOLO METRICHE
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
    cm_internal_avg = np.mean(cm_internal_list, axis=0)
    cm_external_avg = np.mean(cm_external_list, axis=0)

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
    axes[0].set_title('Avg Confusion Matrix % - Internal ({scenario_name})')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    sns.heatmap(cm_external_avg, annot=True, fmt=".1f", cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Avg Confusion Matrix % - External ({scenario_name})')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig(f"grafici/heatmap_{scenario_name}.png", dpi=300)
    plt.show()
