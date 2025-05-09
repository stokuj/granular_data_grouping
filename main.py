import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from math import sqrt
import os
import csv
from sklearn.metrics import accuracy_score
####
#importy własne
import point_generators as pg
import DBscan as db
import KNN as KNN
import time

#################################################################################################################################
### Single/complete linkage
#################################################################################################################################

def LinkageAlgorithmLoop(path,file_path, method, max_d, num_measurements, max_d_range, result_path):

    # Wczytanie danych z pliku
    data = np.loadtxt(file_path)
    
    # Wykonanie algorytmu aglomeracyjnego single linkage
    Z = linkage(data, method=method)
    max_d_step = (max_d_range - max_d) / (num_measurements - 1)

    # Wyświetlenie wyniku
    #print("Wartość kroku:", max_d_step)
    #######################################################################

    with open(result_path, "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Zapisz nagłówki kolumn
        csvwriter.writerow(["max_d", "largest_cluster_points", "remaining_points", "num_clusters", 
                            "mean_cluster_distance", "std_dev_distance",
                            "second_mean_cluster_distance", "second_std_dev_distance", 
                            "p1",   "p2",   "p3",
                            "p4",   "p5",   "p6",
                            "p7",   "p8",   "p9",
                            "p10",  "p11",  "p12",
                            "p13",  "p14",  "p15"
                            ])

        while max_d <= max_d_range: 
            clusters = fcluster(Z, max_d, criterion='distance')

            # Stworzenie słownika do przechowywania liczności klastrów
            clusters_count = {}

            # Obliczenie liczby punktów w każdym klastrze
            for cluster_id in clusters:
                if cluster_id not in clusters_count:
                    clusters_count[cluster_id] = 1
                else:
                    clusters_count[cluster_id] += 1
                    
            # Znalezienie klastra z największą liczbą punktów
            largest_cluster_id = max(clusters_count, key=clusters_count.get)
            largest_cluster_points = clusters_count[largest_cluster_id]
            
            # Początkowa liczba klastrów
            num_clusters = len(set(clusters))
            if num_clusters > 2:
                # Znalezienie identyfikatora klastra z drugą największą liczbą punktów
                second_largest_cluster_id = sorted(clusters_count, key=clusters_count.get)[-2]
            #######################################################################  
            while len(clusters_count) > 2:
                # Znalezienie identyfikatora klastra z najmniejszą liczbą punktów
                smallest_cluster_id = min(clusters_count, key=clusters_count.get)

                # Podmiana identyfikatora klastra najmniejszego na identyfikator klastra pośrodku
                clusters = np.where(clusters == smallest_cluster_id, second_largest_cluster_id, clusters)

                # Zaktualizowanie statystyki klastrów
                clusters_count = {}
                for cluster_id in clusters:
                    if cluster_id not in clusters_count:
                        clusters_count[cluster_id] = 1
                    else:
                        clusters_count[cluster_id] += 1
            #######################################################################
            # Obliczenie średniej i odchylenia standardowego odległości między punktami w największym i drugim co do wielkości klastrze
            if len(clusters_count) >= 2:
                largest_cluster_id = max(clusters_count, key=clusters_count.get)
                largest_cluster_indices = np.where(clusters == largest_cluster_id)[0]
                if len(largest_cluster_indices) >= 2:
                    largest_cluster_data = data[largest_cluster_indices]
                    largest_cluster_distances = pdist(largest_cluster_data)
                    mean_cluster_distance = np.mean(largest_cluster_distances)
                    std_dev_distance = np.std(largest_cluster_distances)
                else:
                    mean_cluster_distance = 0
                    std_dev_distance = 0

                second_largest_cluster_id = sorted(clusters_count, key=clusters_count.get)[-2]
                second_largest_cluster_indices = np.where(clusters == second_largest_cluster_id)[0]
                if len(second_largest_cluster_indices) >= 2:
                    second_largest_cluster_data = data[second_largest_cluster_indices]
                    second_largest_cluster_distances = pdist(second_largest_cluster_data)
                    second_mean_cluster_distance = np.mean(second_largest_cluster_distances)
                    second_std_dev_distance = np.std(second_largest_cluster_distances)
                else:
                    second_mean_cluster_distance = 0
                    second_std_dev_distance = 0
            else:
                mean_cluster_distance = 0
                std_dev_distance = 0
                second_mean_cluster_distance = 0
                second_std_dev_distance = 0

            # Obliczenie liczby pozostałych punktów
            remaining_points = len(clusters) - largest_cluster_points

            # Obliczenie liczby klastrów
            num_clusters = len(set(clusters))
            
            p1, p2, p3, p4 = 0, 0, 0, 0
            p5, p6, p7, p8 = 0, 0, 0, 0
            p9, p10, p11, p12 = 0, 0, 0, 0
            p13, p14, p15 = 0, 0, 0
            
            if mean_cluster_distance!=0 or second_mean_cluster_distance !=0:

                k3 = KNN.kNNAlgorithm(data=largest_cluster_data, k=3, folder=path+"/"+ method +"/", show_picture=False, save_picture =False, name=method + " " + str(max_d))
                k5 = KNN.kNNAlgorithm(data=largest_cluster_data, k=5, folder=path+"/"+ method +"/", show_picture=False, save_picture =False, name=method + " " + str(max_d))
                k7 = KNN.kNNAlgorithm(data=largest_cluster_data, k=7, folder=path+"/"+ method +"/", show_picture=False, save_picture =False, name=method + " " + str(max_d))
                k9 = KNN.kNNAlgorithm(data=largest_cluster_data, k=9, folder=path+"/"+ method +"/", show_picture=False, save_picture =False, name=method + " " + str(max_d))
                
                p1 = largest_cluster_points         * (1 / mean_cluster_distance)
                p2 = largest_cluster_points         * (1 / mean_cluster_distance**2)
                p3 = sqrt(largest_cluster_points)   * (1 / mean_cluster_distance**2)
                
                p4 = largest_cluster_points        * (1 / k3)
                p5 = largest_cluster_points        * (1 / k3**2)
                p6 = sqrt(largest_cluster_points)  * (1 / k3**2)
                
                p7 = largest_cluster_points        * (1 / k5)
                p8 = largest_cluster_points        * (1 / k5**2)
                p9 = sqrt(largest_cluster_points)  * (1 / k5**2)
                
                p10 = largest_cluster_points        * (1 / k7 )
                p11 = largest_cluster_points        * (1 / k7**2)
                p12 = sqrt(largest_cluster_points)  * (1 / k7**2)
                
                p13 = largest_cluster_points        * (1 / k9 )
                p14 = largest_cluster_points        * (1 / k9**2)
                p15 = sqrt(largest_cluster_points)  * (1 / k9**2)
            # Zapisz dane do pliku CSV
            if remaining_points > largest_cluster_points:
                output_data = [max_d, largest_cluster_points, remaining_points, num_clusters, 
                                second_mean_cluster_distance, second_std_dev_distance, 
                                mean_cluster_distance, std_dev_distance,
                                p1,     p2,     p3, 
                                p4,     p5,     p6,  
                                p7,     p8,     p9,  
                                p10,    p11,    p12,
                                p13,    p14,    p15
                                ]
            else:
                output_data = [max_d, largest_cluster_points, remaining_points, num_clusters, 
                                mean_cluster_distance, std_dev_distance, 
                                second_mean_cluster_distance, second_std_dev_distance,
                                p1,     p2,     p3, 
                                p4,     p5,     p6,  
                                p7,     p8,     p9,  
                                p10,    p11,    p12,
                                p13,    p14,    p15
                                ]
                
            csvwriter.writerow(output_data)

            # Zwiększenie wartości d_max o krok
            max_d = max_d + max_d_step

def LinkageAlgorithm(file_path, method, max_d, name, folder, show_picture=False):
    # Wczytanie danych z pliku
    data = np.loadtxt(file_path)
    # Wybranie tylko dwóch pierwszych kolumn (x i y), ignorując kolumnę z etykietami
    data_xy = data[:, :2]
    # Wykonanie algorytmu aglomeracyjnego single linkage
    Z = linkage(data_xy, method=method)

    #######################################################################

    # Wyznaczenie klastrów na podstawie dendrogramu
    clusters = fcluster(Z, max_d, criterion='distance')

    #######################################################################

    # Stworzenie słownika do przechowywania liczności klastrów
    clusters_count = {}

    # Obliczenie liczby punktów w każdym klastrze
    for cluster_id in clusters:
        if cluster_id not in clusters_count:
            clusters_count[cluster_id] = 1
        else:
            clusters_count[cluster_id] += 1

    # Znalezienie klastra z największą liczbą punktów
    largest_cluster_id = max(clusters_count, key=clusters_count.get)
    largest_cluster_count = clusters_count[largest_cluster_id]

    # Znalezienie identyfikatora klastra z drugą największą liczbą punktów
    second_largest_cluster_id = sorted(clusters_count, key=clusters_count.get)[-2]
    
    # Wydrukowanie informacji o klastrze z największą liczbą punktów
    #print(f"Największy klaster: {largest_cluster_id} - {largest_cluster_count} punktów")
    # Początkowa liczba klastrów
    num_clusters = len(set(clusters))
    if num_clusters > 2:
        while len(clusters_count) > 2:

            # Znalezienie identyfikatora klastra z najmniejszą liczbą punktów
            smallest_cluster_id = min(clusters_count, key=clusters_count.get)

            # Podmiana identyfikatora klastra najmniejszego na identyfikator klastra drugiego największego (szum)
            clusters = np.where(clusters == smallest_cluster_id, second_largest_cluster_id, clusters)

            # Zaktualizowanie statystyki klastrów
            clusters_count = {}
            for cluster_id in clusters:
                if cluster_id not in clusters_count:
                    clusters_count[cluster_id] = 1
                else:
                    clusters_count[cluster_id] += 1

    #######################################################################
        # Znalezienie identyfikatora klastra z najmniejszą
        smallest_cluster_id = min(clusters_count, key=clusters_count.get)
        #print(smallest_cluster_id)
        # Znalezienie identyfikatora klastra z największą liczbą punktów
        largest_cluster_id = max(clusters_count, key=clusters_count.get)
        #print(largest_cluster_id)
        

    # Podmiana identyfikatora klastra największego na 1, a drugiego na 2
    if(largest_cluster_count>=500):
        clusters = np.where(clusters == largest_cluster_id, 1, 2)
    else:
        clusters = np.where(clusters == largest_cluster_id, 2, 1)
    
    # Zaktualizowanie statystyki klastrów
    clusters_count = {}
    for cluster_id in clusters:
        if cluster_id not in clusters_count:
            clusters_count[cluster_id] = 1
        else:
            clusters_count[cluster_id] += 1
            
    # Obliczenie dokładności
    true_labels = data[:, 2].astype(int)

    accuracy = accuracy_score(true_labels, clusters)
    # print("Accuracy:", accuracy)
    #######################################################################
    # Wyświetlenie klastrów na płaszczyźnie 2D
    plt.figure()

    # Wyświetlenie punktów danych
    plt.scatter(data[:,0], data[:,1], c=clusters, cmap='viridis', alpha=0.5, s=5)
    
    # Tworzenie tablicy Z dla konturów klastrów
    x = np.linspace(min(data[:,0]), max(data[:,0]), 100)
    y = np.linspace(min(data[:,1]), max(data[:,1]), 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            dists = np.sqrt((data[:,0] - X[j, i])**2 + (data[:,1] - Y[j, i])**2)
            if np.min(dists) < max_d:
                Z[j, i] = clusters[np.argmin(dists)]
            else:
                Z[j, i] = np.nan  # Ustawienie obszarów poza zasięgiem klastrów na NaN

    plt.contourf(X, Y, Z, alpha=0.1, cmap='viridis')

    # Ustawienie zakresu osi x i y
    plt.xlim(min(data[:,0]) - 5, max(data[:,0]) + 5)
    plt.ylim(min(data[:,1]) - 5, max(data[:,1]) + 5)


    plt.title(f"{method} -- {name}: {round(max_d,2)} Accuracy: {accuracy}")
    plt.xlabel('Współrzędna X')
    plt.ylabel('Współrzędna Y')
    plt.colorbar(label='Numer klastra')

    output_file = f"{folder}/{method}/clustering_{name}.png"
    plt.savefig(output_file, dpi=1000) 
    if(show_picture):
        plt.show()
    plt.close()

    #######################################################################
    # Tworzenie listy punktów w formacie "x_punktu y_punktu id_clustra"
    points_with_clusters = []
    for i in range(len(data)):
        points_with_clusters.append((data[i][0], data[i][1], clusters[i]))

    # Zapisywanie punktów do pliku CSV
    output_file = f"{folder}/{method}/LinkageResults_{name}.csv"
    with open(output_file, mode="w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Zapisz nagłówki kolumn
        csvwriter.writerow(["x_punktu", "y_punktu", "id_klastra"])
        for point in points_with_clusters:
            csvwriter.writerow(point)

    #print(f"Punkty zapisano do pliku {output_file}")

def find_max_and_d_max(param, d_max_values):
    param_index = np.argmax(param)
    param_d_max = d_max_values[param_index]
    return param_d_max

def LinkageChart(file_path, draw):
    # Wczytanie danych z pliku CSV
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

    # Usunięcie wierszy z wartościami zerowymi
    data = data[data[:, 5] != 0]

    # Wyodrębnienie danych
    d_max_values = data[:, 0]
    largest_cluster_points = data[:, 1]
    remaining_points = data[:, 2]

    p1 = data[:, 8]
    p2 = data[:, 9]
    p3 = data[:, 10]


    k3_p1 = data[:, 11]
    k3_p2 = data[:, 12]
    k3_p3 = data[:, 13]

    
    k5_p1 = data[:, 14]
    k5_p2 = data[:, 15]
    k5_p3 = data[:, 16]

    
    k7_p1 = data[:, 17]
    k7_p2 = data[:, 18]
    k7_p3 = data[:, 19]

    
    k9_p1 = data[:, 20]
    k9_p2 = data[:, 21]
    k9_p3 = data[:, 22]


    p1_index = np.argmax(p1)
    p1_d_max = d_max_values[p1_index]
    p2_d_max = find_max_and_d_max(p2, d_max_values)
    p3_d_max = find_max_and_d_max(p3, d_max_values)

    k3_p1_d_max = find_max_and_d_max(k3_p1, d_max_values)
    k3_p2_d_max = find_max_and_d_max(k3_p2, d_max_values)
    k3_p3_d_max = find_max_and_d_max(k3_p3, d_max_values)

    k5_p1_d_max = find_max_and_d_max(k5_p1, d_max_values)
    k5_p2_d_max = find_max_and_d_max(k5_p2, d_max_values)
    k5_p3_d_max = find_max_and_d_max(k5_p3, d_max_values)

    k7_p1_d_max = find_max_and_d_max(k7_p1, d_max_values)
    k7_p2_d_max = find_max_and_d_max(k7_p2, d_max_values)
    k7_p3_d_max = find_max_and_d_max(k7_p3, d_max_values)

    k9_p1_d_max = find_max_and_d_max(k9_p1, d_max_values)
    k9_p2_d_max = find_max_and_d_max(k9_p2, d_max_values)
    k9_p3_d_max = find_max_and_d_max(k9_p3, d_max_values)
    
    if (draw):
        # Tworzenie wykresu zależności od d_max
        plt.figure(figsize=(15, 6))

        # Wykres dla liczby punktów w największym klastrze
        plt.plot(d_max_values, largest_cluster_points, label='Liczba punktów w klastrze', color='blue')
        plt.plot(d_max_values, remaining_points, label='Liczba punktów w szumie', color='red')
        plt.xlabel('d_max')
        plt.ylabel('Liczba punktów')
        plt.title('Liczba punktów w klastrze oraz szumie')
        plt.grid(True)
        plt.legend()
        
        # Tworzenie wykresu zależności od d_max
        plt.figure(figsize=(15, 6))

        # Wykres dla liczby punktów w największym klastrze
        plt.plot(d_max_values, p1, label='test', color='green')
        plt.scatter(p1_d_max, p1[p1_index], color='black', label=f"D_max dla max ({p1_d_max})")
        plt.xlabel('d_max')
        plt.ylabel('Liczba punktów')
        plt.title('Wartość test dla parametru d_max')
        plt.grid(True)
        plt.legend()
        plt.show()
    return [p1_d_max, p2_d_max, p3_d_max, k3_p1_d_max, k3_p2_d_max, k3_p3_d_max, k5_p1_d_max, k5_p2_d_max, k5_p3_d_max, k7_p1_d_max, k7_p2_d_max, k7_p3_d_max, k9_p1_d_max, k9_p2_d_max, k9_p3_d_max]

def makeDendrogram(file_path, method, draw):
    # Wczytanie danych z pliku
    data = np.loadtxt(file_path)

    # Wykonanie algorytmu aglomeracyjnego single linkage
    Z = linkage(data, method=method)
    
    min_distance = Z[-1, 2]
    #print("Minimalna odległość w dendrogramie:", min_distance)
    max_distance = Z[0, 2]
    #print("Maksymalna odległość w dendrogramie:", max_distance)
    
    if (draw):
        plt.figure(figsize=(10, 5))
        dendrogram(Z)
        plt.title('Dendrogram - Algorytm aglomeracyjny')
        plt.xlabel('Indeksy danych')
        plt.ylabel('Odległość')
        plt.show()
    return max_distance,min_distance

#################################################################################################################################
#################################################################################################################################

def p_name(param):
    for name, value in globals().items():
        if value is param:
            return name
    return None

maxSize = 100     
#Słownik
generators = {
    "circle_A"  : (pg.generate_points_circle,  None),
    "circle_B"  : (pg.generate_points_mises,   None),
    "circle_C"  : (pg.generate_points_normal,  None),
    "ring"      : (pg.generate_points_ring,    None),
    # "shape_A"   : (pg.generate_points_in_shape, "drawn_figure_A.txt"),
    # "shape_B"   : (pg.generate_points_in_shape, "drawn_figure_B.txt"),
    # "shape_C"   : (pg.generate_points_in_shape, "drawn_figure_C.txt"),
}
# Ścieżka do pliku CSV
csv_file = 'wyniki_czasu_wykonania.csv'

# # Tworzenie nagłówków pliku CSV
# with open(csv_file, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['generator', 'noisePoints', 'circle_radius', 'time_dbscan', 'time_single', 'time_complete'])

# # Pętla główna
for noisePoints in range(300, 701, 100):
    num_points = 1000 - noisePoints
    
    for folder, (generator_function, shape_name) in generators.items():
        for circle_radius in np.arange(30, 31, 20):
            print(circle_radius, noisePoints, generator_function)
            
            path = f"{folder}/{circle_radius}/{noisePoints}" if generator_function != pg.generate_points_in_shape else f"{folder}/{noisePoints}"
            
            for subfolder in [path, path+"/single", path+"/complete"]:
                if not os.path.exists(subfolder):
                    os.makedirs(subfolder)
                
                
            # ### Generowanie punktów    
            #pg.generate_points(generator_function, path, noisePoints, circle_radius, shape_name, num_points, maxSize)
            
            # # Wczytanie danych z pliku
            data = np.loadtxt(f"{path}/points.txt")
            
            # ### KNN
            k_names = [3, 5, 7, 9]
            p_names = [
                'p1', 'p2', 'p3',
                'k3_p1', 'k3_p2', 'k3_p3',
                'k5_p1', 'k5_p2', 'k5_p3',
                'k7_p1', 'k7_p2', 'k7_p3',
                'k9_p1', 'k9_p2', 'k9_p3'
            ]
            
            # ### Single i complete linkage
            # methods = ['single', 'complete']

            # for method in methods:
            #     start_time = time.time()
                
            #     # Tworzenie dendrogramu i uruchamianie pętli algorytmu Linkage
            #     min_distance, max_distance = makeDendrogram(file_path=f"{path}/points.txt", method=method, draw=False)
            #     result_path = f"{path}/{method}/{method}LinkageLoopResults.csv"
            #     LinkageAlgorithmLoop(path=path,file_path=f"{path}/points.txt", method=method, max_d=min_distance, max_d_range=max_distance, num_measurements=200, result_path=result_path)
            #     p_val = LinkageChart(file_path=result_path, draw=False)

            #     # Uruchamianie algorytmu kNN i algorytmu Linkage
            #     for k in k_names:
            #         k_val = KNN.kNNAlgorithm(data, k=k, folder=path, show_picture=False, save_picture=False, name=None)
            #         LinkageAlgorithm(file_path=f"{path}/points.txt", method=method, max_d=k_val, name=p_name(k) + str(k), folder=path, show_picture=False)

            #     # Uruchamianie algorytmu Linkage dla każdego p w p_names
            #     for i, p in enumerate(p_names):
            #         LinkageAlgorithm(file_path=f"{path}/points.txt", method=method, max_d=p_val[i], name=p, folder=path, show_picture=False)
            #     end_time = time.time()
            #     elapsed_time = end_time - start_time

            #     if method == 'single':
            #         time_single = elapsed_time
            #     else:
            #         time_complete = elapsed_time
            
            # ### DBscan
            # start_time = time.time()
            
            # db.DBscanAlgorithmLoop(2, 20.0, 0.2, 20, 150, folder=path)
            
            # for i in range(1, 21):
            #     ep, samp, p_val = db.DBscanChart(folder=path, show_picture=False, p_id=f"p{i}")
            #     db.DBscan(file_path=f"{path}/points.txt", folder=path, epsilon=ep, samples=samp, p_id=f"p{i}", p_val=p_val)
                
            # end_time = time.time()
            # time_dbscan = end_time - start_time
            
            # ### Zapis wyników do pliku CSV
            # with open(csv_file, mode='a', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow([folder, noisePoints, circle_radius, time_dbscan, time_single, time_complete])
            
            # ### Wykresy
            pg.create_combined_plot(path)
            
            ### Bez zbędnego powtarzania pętli na kształów
            if generator_function == pg.generate_points_in_shape:
                break
