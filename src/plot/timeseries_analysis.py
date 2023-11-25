import matplotlib.pyplot as plt
import numpy as np
from src.utility.constants import sifim_features


def plot_with_thresholds(ax, title, data, data_labels=[], plot_label='', sigma=None, fault_indicator=None, fault_detection_indicator=None, ts_sec=30, threshold_perc=2):

    x_ticks = list(range(0, len(data[0]) + 1, 10))
    ax.set_xticks(x_ticks, [i * ts_sec for i in range(len(x_ticks))])
    # Creazione del grafico
    for i, d in enumerate(data):
        ax.plot(d, label=f'{data_labels[i]}', linestyle='-')

    # Aggiunta di threshold al grafico
    if sigma is not None:
        mean = 0
        m2s = np.ones(len(data[0])) * sigma * (-threshold_perc)
        p2s = np.ones(len(data[0])) * sigma * (+threshold_perc)
        ax.axhline(y=mean, color='g', linestyle='--', label='Fault Threshold')
        ax.fill_between(range(len(data[0])), p2s, m2s, color='g', alpha=0.2)
    # Etichette per upper value e lower value
        ax.text(0, sigma * threshold_perc,
                f'{threshold_perc}\u03C3', color='g', fontsize=10, va='bottom', ha='left')
        ax.text(0, sigma * -threshold_perc,
                f'-{threshold_perc}\u03C3', color='g', fontsize=10, va='top', ha='left')

    if fault_indicator is not None:
        ax.axvline(x=fault_indicator, color='b', linestyle='--', label='Fault Effective')
    # if fd:
    if fault_detection_indicator is not None:
        ax.axvline(x=fault_detection_indicator, color='r',
                   linestyle='--', label='Fault Predicted')

    # Aggiunta di etichette e titolo
    ax.set_xlabel('Time')
    ax.set_ylabel(plot_label)
    ax.set_title(title)

    # Aggiunta di una legenda
    ax.legend(loc='upper left')
    ax.grid(True)


def create_subplots(
    dataset,
    models,
    zoom=50,
    feature=[
        ('energia_apparente_importata_sistema',
         'System imported apparent energy effective', 'kWh'),
        ('energia_attiva_importata_di_sistema',
         'System imported active energy effective', 'kWh'),
        ('potenza_attiva_di_sistema', 'Active system power', 'kWh'),
        ('frequenza', 'Frequency', 'Hz'),
        ('tensione_di_sistema', 'System voltage', 'Volt'),
        ('corrente_di_sistema', 'System current', 'Ampere')
    ],
    n_example=0,
    plotpath=None,
    plotshow=True
):
    map_label = {l: i for i, l in enumerate(sifim_features)}
    middle = dataset.x.shape[1] // 2
    start = - (middle + zoom)
    end = - (middle - zoom)
    x_ts, y_ts = dataset.x[:, :-1], dataset.x[:, 1:]
    for key, label, unit in feature:
        f = map_label[key]

        _, axs = plt.subplots(2, len(models), figsize=(13, 5 * len(models)))

        for j, model in enumerate(models):
            ad_labels, ad_predictions, ad_std = model.predict(x_ts, y_ts)
            ad_labels, ad_predictions, ad_std = ad_labels[:,
                                                          :-1], ad_predictions[:, :-1], ad_std[:, :-1]
            std = ad_std[n_example, start:end, f]
            labels_eff = dataset.y[n_example, start:end, f]
            labels_pred = ad_labels[n_example, start:end, f]
            treshold_eff = np.where(labels_eff == 1)[0][0]
            treshold_pred = None
            if (len(np.where(labels_pred == 1)[0]) > 0):
                treshold_pred = np.where(labels_pred == 1)[0][0]

            model_name = model.__class__.__name__
            plot_with_thresholds(
                axs[0] if len(list(axs.shape)) == 1 else axs[j, 0],
                f'{model_name} Standard deviation (q): {label}',
                [std],
                sigma=model.sigma[f].item(),
                threshold_perc=model.threshold_perc,
                data_labels=['q'],
                plot_label='\u03C3',
                fault_indicator=treshold_eff,
                fault_detection_indicator=treshold_pred,
            )  # standard deviation

            plot_with_thresholds(
                axs[1] if len(list(axs.shape)) == 1 else axs[j, 1],
                f'{model_name} Timeseries prediction: {label}',
                [dataset.x[n_example, start:end, f],
                    ad_predictions[n_example, start:end, f]],
                fault_indicator=treshold_eff,
                fault_detection_indicator=treshold_pred,
                plot_label=unit,
                data_labels=["Timeseries Effective", "Timeseries Prediction"],
            )  # timeseries effective and prediction
            plt.tight_layout()

        if plotpath is not None:
            plt.savefig(f'{plotpath}{label}.png')

        if plotshow:
            plt.show()


if __name__ == '__main__':
    # Esempio di utilizzo
    data1 = np.random.random(100)
    data2 = np.random.random(100)
    thresholds = [0.3, 0.6, 0.8]
    fault_indicator = int(np.random.uniform(50, 100))
    drift_indicator = int(np.random.uniform(50, 100))
    fault_detection_indicator = int(np.random.uniform(50, 100))

    # Chiamata alla funzione
    plot_with_thresholds(data1, data2, thresholds, fault_indicator,
                         drift_indicator, fault_detection_indicator)
