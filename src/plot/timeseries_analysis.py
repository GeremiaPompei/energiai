import matplotlib.pyplot as plt
import numpy as np


def plot_with_thresholds(title, data, thresholds=[], fault_indicator=None, drift_indicator=None,
                         fault_detection_indicator=None):
    plt.figure(figsize=(10, 6))
    # Creazione del grafico
    for i, d in enumerate(data):
        plt.plot(d, label=f'Dati {i}', linestyle='-')

    # Aggiunta di threshold al grafico
    for i, threshold in enumerate(thresholds):
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold {i + 1}')

    if fault_indicator is not None:
        plt.axvline(x=fault_indicator, color='g', linestyle='--', label='Fault')
    # if d:
    if drift_indicator is not None:
        plt.axvline(x=drift_indicator, color='b', linestyle='--', label='Drift')
    # if fd:
    if fault_detection_indicator is not None:
        plt.axvline(x=fault_detection_indicator, color='m', linestyle='--', label='Fault Detection')

    # Aggiunta di etichette e titolo
    plt.xlabel('Tempo')
    plt.ylabel('Valore')
    plt.title(title)

    # Aggiunta di una legenda
    plt.legend(loc='upper left')

    # Visualizzazione del grafico
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
    plot_with_thresholds(data1, data2, thresholds, fault_indicator, drift_indicator, fault_detection_indicator)
