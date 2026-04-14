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

def plot_combined_importance(model_type, num_packets):
    all_data = []
    for scenario in SCENARIOS:
        path = f'risultati/{model_type}/{num_packets}/feature_importance_{scenario}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Scenario'] = scenario
            all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data)
        g = sns.FacetGrid(combined_df, col="Scenario", col_wrap=3, height=4, hue="Feature", aspect=1.2)
        g.map(sns.lineplot, "Packet_Index", "Importance", marker="o")
        g.add_legend()
        g.set_titles("{col_name}")
        g.set_axis_labels("Packet Index", "Average Importance")
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(f'Feature Importance Across Scenarios: {model_type} - {num_packets} Packets', fontsize=16)
        plt.savefig(f"grafici/{model_type}/{num_packets}/combined_feature_importance.png", dpi=300, bbox_inches='tight')
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