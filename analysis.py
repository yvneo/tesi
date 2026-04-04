#parte di codice per analizzare i risultati dei modelli e generare grafici comparativi
#generate_heatmap_feature_usage: funzione che genera una heatmap per ogni modello, mostrando l'importanza media totale di ogni gruppo di feature per ogni scenario e finestra temporale. I dati vengono letti dai file CSV generati in precedenza e organizzati in un formato adatto per la visualizzazione. La heatmap viene salvata come immagine PNG e mostrata a schermo.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from scipy import stats

MODELS = ['rf', 'xgboost']
WINDOWS = [10,20]
SCENARIOS = ['Home_vs_Home', 'Home_vs_UniVR', 
    'UniVR_vs_UniVR', 'UniVR_vs_Home', 
    'Combined_vs_UniVR', 'Combined_vs_Home']

SCENARIOS_FOR_TTEST = ['home_to_home', 'home_to_univr', 
    'univr_to_univr', 'univr_to_home', 'jd_to_univr', 'jd_to_home']

def generate_heatmap_feature_usage(model):
    data = []
    for w in WINDOWS:
        for s in SCENARIOS:
            path = f'risultati/{model}/{w}/feature_importance_groups_{s}.csv'
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['Window'] = w
                df['Scenario'] = s
                data.append(df)

    df = pd.concat(data)
    df['Feature_Window'] = df['Feature_Group'] + ' (W=' + df['Window'].astype(str) + ')'
    pivot = df.pivot_table(index='Feature_Window', columns='Scenario', values='Total_Importance_Mean')
    pivot = pivot[[s for s in SCENARIOS if s in pivot.columns]]

    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt = '.2f', linewidths =0.5, cbar_kws={'label': 'Mean Total Importance'})
    plt.title(f'Global Feature Importance Summary - {model.upper()}', fontsize=16, pad=20)
    plt.xlabel('Scenario', fontsize=12)
    plt.ylabel('Feature Group (Window Size)', fontsize=12)
    plt.xticks(rotation=30, ha='right')
    plt.savefig(f'grafici/analisi_feature/global_heatmap_{model}.png', bbox_inches='tight', dpi =300)
    plt.show()

def perform_comparative_analysis_ttest(model_a, model_b, windows, scenarios):
    results = []
    for w in windows:
        for s in scenarios:
            path_a = f'risultati/{model_a}/{w}/report_{s}.csv'
            path_b = f'risultati/{model_b}/{w}/report_{s}.csv'
            if os.path.exists(path_a) and os.path.exists(path_b):
                df_a = pd.read_csv(path_a)
                df_b = pd.read_csv(path_b)
                
                metric_col = 'f1_ID' if 'home_to_home' in s or 'univr_to_univr' in s else 'f1_OOD_or_JD'
                scores_a = df_a[metric_col].values
                scores_b = df_b[metric_col].values

                t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
                results.append({
                    'Window': w,
                    'Scenario': s,
                    'Metric': metric_col,
                    f'Mean_{model_a}': scores_a.mean(),
                    f'Mean_{model_b}': scores_b.mean(),
                    'T-Statistic': t_stat,
                    'P-Value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })
    df_results = pd.DataFrame(results)
    df_results.to_csv(f'risultati/statistiche_ttest/comparative_analysis_{model_a}_vs_{model_b}.csv', index=False)
    return df_results



for model in MODELS:
    generate_heatmap_feature_usage(model)
report = perform_comparative_analysis_ttest('rf', 'xgboost', WINDOWS, SCENARIOS_FOR_TTEST)
print(report)