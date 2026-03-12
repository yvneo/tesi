#PROTOCOLLO DI VALUTAZIONE CROSS-LOCATION
#addestrare il modello su location a e testarlo sia su a che su b 
#divido Home in 5 parti 
#normalizzazione con fit solo su training set 
#label encoding 

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_and_evaluate_model(X_home, y_home, X_univr, y_univr):
    #codifica le etichette
    label_encoder = LabelEncoder()
    y_home_encoded = label_encoder.fit_transform(y_home)
    y_univr_encoded = label_encoder.transform(y_univr) 
    class_names = label_encoder.classes_ #nomi delle classi per grafici

    #configuro il K-Fold Cross Validation con K=5
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    results_report = []

    cm_home_list = [] #lista per matrici di confusione dei fold 
    cm_univr_list  = []

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
        y_pred_univr = model.predict(X_univr_scaled)

        #calcolo delle metriche f1 e balanced accuracy
        f1_home = f1_score(y_test, y_pred_home, average='macro')
        f1_univr = f1_score(y_univr_encoded, y_pred_univr, average='macro')
        bal_acc_home = balanced_accuracy_score(y_test, y_pred_home)
        bal_acc_univr = balanced_accuracy_score(y_univr_encoded, y_pred_univr)

        #calcolo delle matrici di confusione 
        cm_home = confusion_matrix(y_test, y_pred_home, labels = range(len(class_names)), normalize='true') 
        cm_univr = confusion_matrix(y_univr_encoded, y_pred_univr, labels = range(len(class_names)), normalize='true')

        cm_home_list.append(cm_home)
        cm_univr_list.append(cm_univr)
        
        for real, pred in zip(y_test, y_pred_home):
            #zip mi mette risposta vera e quella predetta una accanto all'altra
            results.append({'fold': fold+1, 'Domain': 'Home', 'Ground Truth': label_encoder.inverse_transform([real])[0], 'Predicted': label_encoder.inverse_transform([pred])[0]}) 

        for real, pred in zip(y_univr_encoded, y_pred_univr):
            results.append({'fold': fold+1, 'Domain': 'UniVR', 'Ground Truth': label_encoder.inverse_transform([real])[0], 'Predicted': label_encoder.inverse_transform([pred])[0]}) 

        results_report.append({'fold': fold+1, 'f1_home': f1_home, 'bal_acc_home': bal_acc_home, 'f1_univr': f1_univr, 'bal_acc_univr': bal_acc_univr })  #salvo statistiche generali 

        df_results = pd.DataFrame(results)
        df_report = pd.DataFrame(results_report)

        #calvolo la matrice di confusione media
        cm_home_avg = np.mean(cm_home_list, axis=0)
        cm_univr_avg = np.mean(cm_univr_list, axis=0)

        #visualizzazione
        plot_results(df_report, cm_home_avg, cm_univr_avg, class_names)

    return df_results, df_report


def plot_results(df_report, cm_home_avg, cm_univr_avg, class_names):
    #preparazione dei dati per il barplot
    metrics = ['f1', 'bal_acc']
    plot_data = []
    for metric in metrics:
        for domain in ['home', 'univr']:
            col_name = f"{metric}_{domain}"
            mean_value = df_report[col_name].mean()
            std_value = df_report[col_name].std()
            plot_data.append({'Metric': metric, 'Domain': domain, 'Mean': mean_value, 'Std': std_value})

    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    sns.barplot(data = plot_df, x='Metric', y='Mean', hue='Domain', capsize=0.1)
    plt.title('Performance Comparison: Home vs UniVR (Mean ± Std)')
    plt.ylabel('Mean Score')
    plt.ylim(0, 1.1)
    plt.savefig(f"grafici/barplot_performance.png", dpi=300, bbox_inches='tight')
    plt.show()

    #heatmap 
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm_home_avg, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Average Confusion Matrix - Home')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    sns.heatmap(cm_univr_avg, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Average Confusion Matrix - UniVR')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(f"grafici/heatmap_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("Grafici salvati in 'grafici/'")