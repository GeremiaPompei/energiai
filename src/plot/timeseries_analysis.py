import matplotlib.pyplot as plt
import numpy as np


def plot_with_thresholds(data1, data2, thresholds, fault_indicator, drift_indicator, fault_detection_indicator):
    plt.figure(figsize=(10, 6))
    # Creazione del grafico
    plt.plot(data1, label='Dati 1', linestyle='-')
    plt.plot(data2, label='Dati 2', linestyle='-')

    # Aggiunta di threshold al grafico
    for i, threshold in enumerate(thresholds):
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold {i + 1}')

    plt.axvline(x=fault_indicator, color='g', linestyle='--', label='Fault')
    # if d:
    plt.axvline(x=drift_indicator, color='b', linestyle='--', label='Drift')
    # if fd:
    plt.axvline(x=fault_detection_indicator, color='m', linestyle='--', label='Fault Detection')

    # Aggiunta di etichette e titolo
    plt.xlabel('Tempo')
    plt.ylabel('Valore')
    plt.title('Grafico con Thresholds, Fault, Drift e Rilevazione di Fault')

    # Aggiunta di una legenda
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

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
