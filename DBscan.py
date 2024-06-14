import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
from matplotlib.colors import PowerNorm
from math import sqrt
import KNN as KNN
from sklearn.metrics import accuracy_score


def DBscanAlgorithmLoop(min_epsilon, max_epsilon, epsilon_step, min_min_samples, max_min_samples, folder):

    # Wczytanie danych 
    filename = f"{folder}/points.txt"
    data = np.loadtxt(filename)
    
    # Wybranie tylko dwóch pierwszych kolumn (x i y), ignorując kolumnę z etykietami
    dataForDBSCAN = data[:, :2]
    
    if not os.path.exists(f"{folder}/dbScan"):
        os.makedirs(f"{folder}/dbScan")
    
    # Przygotowanie list na wyniki
    results = []
    results_yes = []
    
    # Pętla po różnych wartościach epsilon i min_samples
    epsilon = min_epsilon

    while epsilon <= max_epsilon + epsilon_step:
        for min_samples in range(min_min_samples, max_min_samples + 1):
            db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(dataForDBSCAN)
            labels = db.labels_

            # Obliczenie metryk
            no_clusters = len(np.unique(labels))
            no_noise = np.sum(labels == -1)
            p_cluster = np.sum(labels == 0)
            mean_distance = 0
            std_distance = 0

            # Obliczenie odległości między punktami
            if p_cluster > 0:
                distances = pdist(dataForDBSCAN[labels == 0])  # Oblicz odległości tylko dla punktów w klastrze
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)

            # Zaokrąglenie wartości do 2 miejsc po przecinku
            epsilon_rounded = round(epsilon, 2)

            p1 = p2 = p3 = p4 = p5 = p6 = p7 = p8 = p9 = p10 = p11 = p12 = p13 = p14 = p15 = 0
            if mean_distance != 0 and std_distance != 0:
                method = "dbScan"
                cluster_data =(dataForDBSCAN[labels == 0])
                
                path = folder
                k3 = KNN.kNNAlgorithm(data=cluster_data, k=3, folder=path+"/"+ method +"/", show_picture=False, save_picture =False, name=method)
                k5 = KNN.kNNAlgorithm(data=cluster_data, k=5, folder=path+"/"+ method +"/", show_picture=False, save_picture =False, name=method)
                k7 = KNN.kNNAlgorithm(data=cluster_data, k=7, folder=path+"/"+ method +"/", show_picture=False, save_picture =False, name=method)
                k9 = KNN.kNNAlgorithm(data=cluster_data, k=9, folder=path+"/"+ method +"/", show_picture=False, save_picture =False, name=method)
                
                p1 = p_cluster         * (1 / mean_distance)
                p2 = p_cluster         * (1 / mean_distance**2)
                p3 = sqrt(p_cluster)   * (1 / mean_distance**2)
                
                p4 = p_cluster        * (1 / k3)
                p5 = p_cluster        * (1 / k3**2)
                p6 = sqrt(p_cluster)  * (1 / k3**2)
                
                p7 = p_cluster        * (1 / k5)
                p8 = p_cluster        * (1 / k5**2)
                p9 = sqrt(p_cluster)  * (1 / k5**2)
                
                p10 = p_cluster        * (1 / k7 )
                p11 = p_cluster        * (1 / k7**2)
                p12 = sqrt(p_cluster)  * (1 / k7**2)
                
                p13 = p_cluster        * (1 / k9 )
                p14 = p_cluster        * (1 / k9**2)
                p15 = sqrt(p_cluster)  * (1 / k9**2)

            result = [epsilon_rounded, min_samples, p_cluster, no_noise, mean_distance, std_distance, no_clusters,
                p1,     p2,     p3, 
                p4,     p5,     p6,  
                p7,     p8,     p9,  
                p10,    p11,    p12,
                p13,    p14,    p15
                ]
            if no_clusters == 2 and no_noise != len(dataForDBSCAN) and p_cluster > 100:
                result.insert(4, "yes")
                results_yes.append(result)
            else:
                result.insert(4, "no")
            results.append(result)

        epsilon += epsilon_step
        epsilon = round(epsilon,2)

    # Zapisz wyniki do plików CSV
    columns = ["epsilon", "min_samples", "p_cluster", "no_noise", "condition", "mean_distance", "std_distance", "no_clusters",
                "p1",   "p2",   "p3",
                "p4",   "p5",   "p6",
                "p7",   "p8",   "p9",
                "p10",  "p11",  "p12",
                "p13",  "p14",  "p15"
            ]
    results_df = pd.DataFrame(results, columns=columns)
    results_yes_df = pd.DataFrame(results_yes, columns=columns)

    results_df.to_csv(f"{folder}/dbScan/dbScanResults.csv", index=False)
    results_yes_df.to_csv(f"{folder}/dbScan/dbScanResultsYes.csv", index=False)

def DBscanChart(folder, show_picture, p_id):
    # Wczytanie danych z pliku CSV
    filename = f"{folder}/dbScan/dbScanResults.csv"
    data = pd.read_csv(filename)

    # Filtracja danych, aby uwzględnić tylko te wiersze, gdzie warunek jest spełniony
    data_yes = data[data['condition'] == 'yes']

    # Podział danych na odpowiednie zestawy
    x_yes = data_yes['epsilon'].tolist()
    y_yes = data_yes['min_samples'].tolist()
    test = data_yes[p_id].tolist()

    # Utworzenie wykresu
    plt.clf()
    plt.figure(figsize=(10, 10))
    #test_inv_log = np.exp(test)  # Odwrotność logarytmu
    y_per=np.percentile(y_yes,90)
    sc = plt.scatter(x_yes, y_yes, c=test, cmap='plasma', s=5, alpha=0.6, norm=PowerNorm(gamma=50,vmin=y_per ))
    plt.title(f"dbScan p={p_id}")
    plt.xlabel('Epsilon')
    plt.ylabel('Min Samples')
    plt.colorbar(sc, label='Średnia wartość', format='%.2f')  # Dodajemy formatowanie do kolorów
    plt.grid(True)
    plt.savefig(f"{folder}/dbScan/dbScanGraph_{p_id}.png", dpi=1000) 
    if(show_picture):
        plt.show()

    
    # Znalezienie maksymalnej wartości testu
    max_test_value = max(test)
    max_test_index = test.index(max_test_value)

    # Pobranie odpowiadających epsilon i min_samples dla maksymalnego testu
    epsilon_max_test = x_yes[max_test_index]
    min_samples_max_test = y_yes[max_test_index]
    
     # Zaznaczenie maksymalnego punktu
    plt.scatter(epsilon_max_test, min_samples_max_test, color='red', marker='x', label='Maksymalna wartość testu')
    plt.close()
    plt.clf()
    return epsilon_max_test, min_samples_max_test, max_test_value

def DBscan(file_path, folder, epsilon, samples, p_id, p_val):
    
    # Wczytanie danych
    data = np.loadtxt(file_path)

    # Wybranie tylko dwóch pierwszych kolumn (x i y), ignorując kolumnę z etykietami
    dataForDBSCAN = data[:, :2]

    # Wykonanie klasteryzacji DBSCAN
    dbscan = DBSCAN(eps=epsilon, min_samples=samples)
    dbscan.fit(dataForDBSCAN)

    
    # Obliczenie dokładności
    true_labels = data[:, 2].astype(int)

    # Obliczanie dokładności
    #print(dbscan.labels_)
    modified_labels = np.where(dbscan.labels_ == -1, 2, 1)
    #print(modified_labels)
   # print(true_labels)
    accuracy = accuracy_score(true_labels, modified_labels)
    #print(f"Accuracy of DBSCAN clustering: {accuracy}")

    # Wizualizacja klastrów
    plt.figure()
    plt.scatter(dataForDBSCAN[:, 0], dataForDBSCAN[:, 1], c=dbscan.labels_, cmap='viridis_r', s=10)
    plt.title(f"p={p_id}={p_val} E={epsilon} S:{samples} A:{accuracy}")
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    output_file = f"{folder}/dbScan/dbScan_{p_id}.png"
    plt.savefig(output_file, dpi=1000) 
    plt.close()
