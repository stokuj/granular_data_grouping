import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def kNNAlgorithm(data, k, folder, show_picture, save_picture, name):
    if not os.path.exists(folder+"/kNN"):
        os.makedirs(folder+"/kNN")
        
    # Inicjalizacja modelu k-nn
    knn_model = NearestNeighbors(n_neighbors=k+1)

    # Dopasowanie modelu do danych
    knn_model.fit(data)

    # Znalezienie k najbliższych sąsiadów dla każdego punktu
    distances, indices = knn_model.kneighbors(data)

    # Obliczenie średnich odległości
    mean_distances = np.mean(distances[:, 1:], axis=1)

    # Obliczenie średniej wartości z średnich odległości
    mean_mean_distance = np.mean(mean_distances)


    if save_picture:
        plt.figure(figsize=(8, 6))

        # Wyświetlenie punktów danych
        plt.scatter(data[:, 0], data[:, 1], c='blue', marker='o', label='Punkty danych')

        # Wyświetlenie linii łączących punkty z ich k najbliższymi sąsiadami
        for i, neighbors in enumerate(indices):
            for neighbor_index in neighbors:
                plt.plot([data[i, 0], data[neighbor_index, 0]], [data[i, 1], data[neighbor_index, 1]], 'k--', lw=0.5)

        plt.title(f'K najbliższych sąsiadów dla k={k}')
        plt.xlabel('Współrzędna X')
        plt.ylabel('Współrzędna Y')
        plt.legend()
        plt.grid(True)

        # Zapisanie wykresu do pliku
        plot_file = f"{folder}/kNN/kNN_{k}_plot.png"
        plt.savefig(plot_file)

        # Wyświetlenie wykresu
        if show_picture:
            plt.show()
        plt.close()
    
    return mean_mean_distance
