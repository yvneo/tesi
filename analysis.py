import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from scipy import stats


#configurazione dei parametri per l'analisi comparativa e visualizzazione
MODELS = ['rf', 'xgboost']
WINDOWS = [10,20]

SCENARIOS = ['Home_vs_Home', 'Home_vs_UniVR', 
    'UniVR_vs_UniVR', 'UniVR_vs_Home', 
    'Combined_vs_UniVR', 'Combined_vs_Home']

SCENARIOS_FOR_TTEST = ['home_to_home', 'home_to_univr', 
    'univr_to_univr', 'univr_to_home', 'jd_to_univr', 'jd_to_home']

def plot_combined_importance(model_type, num_packets):

    """funzione che crea barplot dell'importanza delle feature: 
    - un subplot per ogni feature
    - importanza media con barre di errore 
    - indice pacchetto sull'asse x
    - differenza tra scenari evidenziata da colori diversi"""

    all_data = []

    #caricamento dei file .csv riguardanti la feature importance per ciascuno scenario, se esistono, e concatenazione in un unico DataFrame
    for scenario in SCENARIOS:
        path = f'risultati/{model_type}/{num_packets}/feature_importance_{scenario}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Scenario'] = scenario
            all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data)

        sns.set_theme(style="whitegrid")

        g = sns.catplot(data = combined_df, 
                        kind = 'bar',
                        x = 'Packet',
                        y = 'Importance',
                        hue = 'Scenario',
                        col = 'Feature',
                        col_wrap = 2,
                        height = 5,
                        aspect = 1.5,
                        errorbar = 'sd',
                        palette = "muted")
        g.set_axis_labels("Packet Index", "Average Importance (Log Scale)")
        g.set_titles("{col_name}")
        g.despine(left=True)

        for ax in g.axes.flat: #utilizzo scala logaritmica per evidenziare meglio le differenze tra feature con importanza molto diversa
            ax.set_yscale('log')

        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(f"Feature Importance for {model_type.upper()} - Window {num_packets} Packets", fontsize=16)

        plt.savefig(f'grafici/{model_type}/{num_packets}/combined_feature_importance.png')

def perform_comparative_analysis_ttest(model_a, model_b, windows, scenarios):

    """funzione per eseguire un'analisi comparativa tra due modelli utilizzando il test t di Student per campioni appaiati, su più scenari e finestre di pacchetti"""

    results = []
    for w in windows:
        for s in scenarios:
            path_a = f'risultati/{model_a}/{w}/report_{s}.csv'
            path_b = f'risultati/{model_b}/{w}/report_{s}.csv'
            if os.path.exists(path_a) and os.path.exists(path_b):
                df_a = pd.read_csv(path_a)
                df_b = pd.read_csv(path_b)
                
                metric_col = 'f1_OOD_or_JD'
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


# esecuzione della pipeline di analisi 
for model in MODELS:
    for num_packets in WINDOWS:
        plot_combined_importance(model, num_packets)
plot_combined_importance('knn', 10)

report = perform_comparative_analysis_ttest('rf', 'xgboost', WINDOWS, SCENARIOS_FOR_TTEST)
