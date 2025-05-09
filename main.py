import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
import time
from math import sqrt
import os
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import LogNorm, PowerNorm
import matplotlib.image as mpimg
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import matplotlib.image as mpimg
import csv
import math
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
#################################################################################################################################
### Generator liczb
#################################################################################################################################

def generate_points_circle(noisePoints,     circle_radius,  num_points, maxSize):
    offsetX = np.random.uniform(-(maxSize - circle_radius), maxSize - circle_radius)
    offsetY = np.random.uniform(-(maxSize - circle_radius), maxSize - circle_radius)

    theta = np.random.uniform(0.0, 2.0*np.pi, num_points)
    zero_to_one = np.random.uniform(0.0, 1.0, num_points)

    x_circle = circle_radius * np.sqrt(zero_to_one) * np.cos(theta) + offsetX
    y_circle = circle_radius * np.sqrt(zero_to_one) * np.sin(theta) + offsetY
    
    # Maskowanie punktów znajdujących się poza obszarem
    inside_circle_mask = (x_circle >= -maxSize) & (x_circle <= maxSize) & (y_circle >= -maxSize) & (y_circle <= maxSize)
    x_circle = x_circle[inside_circle_mask]
    y_circle = y_circle[inside_circle_mask]

    x_random = np.random.uniform(-maxSize, maxSize, noisePoints)
    y_random = np.random.uniform(-maxSize, maxSize, noisePoints)
    
    print(f"Liczba wszystkich punktów: {noisePoints + num_points}, Procentowy udział szumu: {(noisePoints / (noisePoints + num_points)) * 100}%, Rozmiar koła: {circle_radius}")
    return x_circle, y_circle, x_random, y_random

def generate_points_mises(noisePoints,      circle_radius,  num_points, maxSize):
    offsetX = np.random.uniform(-(maxSize - circle_radius), maxSize - circle_radius)
    offsetY = np.random.uniform(-(maxSize - circle_radius), maxSize - circle_radius)

    # Generowanie punktów na okręgu z rozkładem Von Misesa
    theta = np.random.uniform(0.0, 2.0*np.pi, num_points)
    r = np.random.vonmises(0, 10, num_points) * circle_radius
    x_circle = r * np.cos(theta) + offsetX
    y_circle = r * np.sin(theta) + offsetY
    
    # Maskowanie punktów znajdujących się poza obszarem
    inside_circle_mask = (x_circle >= -maxSize) & (x_circle <= maxSize) & (y_circle >= -maxSize) & (y_circle <= maxSize)
    x_circle = x_circle[inside_circle_mask]
    y_circle = y_circle[inside_circle_mask]

    x_random = np.random.uniform(-maxSize, maxSize, noisePoints)
    y_random = np.random.uniform(-maxSize, maxSize, noisePoints)
    
    print(f"Liczba wszystkich punktów: {noisePoints + num_points}, Procentowy udział szumu: {(noisePoints / (noisePoints + num_points)) * 100}%, Rozmiar koła: {circle_radius}")
    return x_circle, y_circle, x_random, y_random

def generate_points_normal(noisePoints,     circle_radius,  num_points, maxSize):
    offsetX = np.random.uniform(-(maxSize - circle_radius), maxSize - circle_radius)
    offsetY = np.random.uniform(-(maxSize - circle_radius), maxSize - circle_radius)

    # Generowanie punktów na okręgu z rozkładem normalnym
    theta = np.random.uniform(0.0, 2.0*np.pi, num_points)
    r = np.random.normal(0, circle_radius, num_points)
    x_circle = r * np.cos(theta) + offsetX
    y_circle = r * np.sin(theta) + offsetY
    
    # Maskowanie punktów znajdujących się poza obszarem
    inside_circle_mask = (x_circle >= -maxSize) & (x_circle <= maxSize) & (y_circle >= -maxSize) & (y_circle <= maxSize)
    x_circle = x_circle[inside_circle_mask]
    y_circle = y_circle[inside_circle_mask]

    x_random = np.random.uniform(-maxSize, maxSize, noisePoints)
    y_random = np.random.uniform(-maxSize, maxSize, noisePoints)
    
    print(f"Liczba wszystkich punktów: {noisePoints + num_points}, Procentowy udział szumu: {(noisePoints / (noisePoints + num_points)) * 100}%, Rozmiar koła: {circle_radius}")
    return x_circle, y_circle, x_random, y_random

def generate_points_ring(noisePoints,       circle_radius,  num_points, maxSize):

    offsetX = np.random.uniform(-(maxSize - circle_radius), maxSize - circle_radius)
    offsetY = np.random.uniform(-(maxSize - circle_radius), maxSize - circle_radius)

    theta = np.random.uniform(0, 2.0*np.pi, num_points)
    zero_to_one = np.random.uniform(circle_radius/2, circle_radius, num_points)

    x_circle = 1 * (zero_to_one) * np.cos(theta) + offsetX
    y_circle = 1 * (zero_to_one) * np.sin(theta) + offsetY
    
    # Maskowanie punktów znajdujących się poza obszarem
    inside_circle_mask = (x_circle >= -maxSize) & (x_circle <= maxSize) & (y_circle >= -maxSize) & (y_circle <= maxSize)
    x_circle = x_circle[inside_circle_mask]
    y_circle = y_circle[inside_circle_mask]

    x_random = np.random.uniform(-maxSize, maxSize, noisePoints)
    y_random = np.random.uniform(-maxSize, maxSize, noisePoints)
    
    print(f"Liczba wszystkich punktów: {noisePoints + num_points}, Procentowy udział szumu: {(noisePoints / (noisePoints + num_points)) * 100}%, Rozmiar koła: {circle_radius}")
    return x_circle, y_circle, x_random, y_random

def generate_points_in_shape(noisePoints,   shape_name,     num_points, maxSize):
    x_random = np.random.uniform(-maxSize, maxSize, noisePoints)
    y_random = np.random.uniform(-maxSize, maxSize, noisePoints)
    
    shape = Path(read_shape_from_file(shape_name))
    bounds = [(-maxSize, maxSize), (-maxSize, maxSize)]
    points_inside = []
    while len(points_inside) < num_points:
        x = np.random.uniform(*bounds[0])
        y = np.random.uniform(*bounds[1])
        if shape.contains_point((x, y)):
            points_inside.append((x, y))
            
    print(f"Liczba wszystkich punktów: {noisePoints + num_points}, Procentowy udział szumu: {(noisePoints / (noisePoints + num_points)) * 100}%, Kształt: {shape_name}")
    x_points, y_points = zip(*points_inside)
    return x_points, y_points, x_random, y_random

def plot_points(x_circle, y_circle, x_random, y_random,folder, save_picture, show_picture):
    plt.figure()
    plt.plot(x_circle, y_circle, "ro", ms=2)
    plt.plot(x_random, y_random, "bo", ms=2)
    if(save_picture):
        output_file = f"{folder}/points.png"
        plt.savefig(output_file, dpi=1000) 
    if(show_picture):
        plt.show()
    plt.close()
    
def create_combined_plot(path):
    image_paths_and_titles = [
    (f"{path}/complete/clustering_p1.png", "Complete p1"),
    (f"{path}/complete/clustering_p2.png", "Complete p2"),
    (f"{path}/complete/clustering_p3.png", "Complete p3"),
    (f"{path}/complete/clustering_p4.png", "Complete p4"),
    
    (f"{path}/complete/clustering_k3_p1.png", "Complete k3_p1"),
    (f"{path}/complete/clustering_k3_p2.png", "Complete k3_p2"),
    (f"{path}/complete/clustering_k3_p3.png", "Complete k3_p3"),
    (f"{path}/complete/clustering_k3_p4.png", "Complete k3_p4"),
    
    (f"{path}/complete/clustering_k5_p1.png", "Complete k5_p1"),
    (f"{path}/complete/clustering_k5_p2.png", "Complete k5_p2"),
    (f"{path}/complete/clustering_k5_p3.png", "Complete k5_p3"),
    (f"{path}/complete/clustering_k5_p4.png", "Complete k5_p4"),
    
    (f"{path}/complete/clustering_k7_p1.png", "Complete k7_p1"),
    (f"{path}/complete/clustering_k7_p2.png", "Complete k7_p2"),
    (f"{path}/complete/clustering_k7_p3.png", "Complete k7_p3"),
    (f"{path}/complete/clustering_k7_p4.png", "Complete k7_p4"),
    
    (f"{path}/complete/clustering_k9_p1.png", "Complete k9_p1"),
    (f"{path}/complete/clustering_k9_p2.png", "Complete k9_p2"),
    (f"{path}/complete/clustering_k9_p3.png", "Complete k9_p3"),
    (f"{path}/complete/clustering_k9_p4.png", "Complete k9_p4"),
    
    (f"{path}/single/clustering_p1.png", "Single p1"),
    (f"{path}/single/clustering_p2.png", "Single p2"),
    (f"{path}/single/clustering_p3.png", "Single p3"),
    (f"{path}/single/clustering_p4.png", "Single p4"),

    (f"{path}/single/clustering_k3.png", "Single k3"),
    (f"{path}/single/clustering_k5.png", "Single k5"),
    (f"{path}/single/clustering_k7.png", "Single k7"),
    (f"{path}/single/clustering_k9.png", "Single k9"),
    
    (f"{path}/single/clustering_k3_p1.png", "Single k3_p1"),
    (f"{path}/single/clustering_k3_p2.png", "Single k3_p2"),
    (f"{path}/single/clustering_k3_p3.png", "Single k3_p3"),
    (f"{path}/single/clustering_k3_p4.png", "Single k3_p4"),
    
    (f"{path}/single/clustering_k5_p1.png", "Single k5_p1"),
    (f"{path}/single/clustering_k5_p2.png", "Single k5_p2"),
    (f"{path}/single/clustering_k5_p3.png", "Single k5_p3"),
    (f"{path}/single/clustering_k5_p4.png", "Single k5_p4"),
    
    (f"{path}/single/clustering_k7_p1.png", "Single k7_p1"),
    (f"{path}/single/clustering_k7_p2.png", "Single k7_p2"),
    (f"{path}/single/clustering_k7_p3.png", "Single k7_p3"),
    (f"{path}/single/clustering_k7_p4.png", "Single k7_p4"),
    
    (f"{path}/single/clustering_k9_p1.png", "Single k9_p1"),
    (f"{path}/single/clustering_k9_p2.png", "Single k9_p2"),
    (f"{path}/single/clustering_k9_p3.png", "Single k9_p3"),
    (f"{path}/single/clustering_k9_p4.png", "Single k9_p4"),
    
    (f"{path}/dbScan/dbScan_p1.png", ""),
    (f"{path}/dbScan/dbScan_p2.png", ""),
    (f"{path}/dbScan/dbScan_p3.png", ""),
    (f"{path}/dbScan/dbScan_p4.png", ""),
    
    (f"{path}/dbScan/dbScan_p5.png", ""),
    (f"{path}/dbScan/dbScan_p6.png", ""),
    (f"{path}/dbScan/dbScan_p7.png", ""),
    (f"{path}/dbScan/dbScan_p8.png", ""),
    
    (f"{path}/dbScan/dbScan_p9.png", ""),
    (f"{path}/dbScan/dbScan_p10.png", ""),
    (f"{path}/dbScan/dbScan_p11.png", ""),
    (f"{path}/dbScan/dbScan_p12.png", ""),
    
    (f"{path}/dbScan/dbScan_p13.png", ""),
    (f"{path}/dbScan/dbScan_p14.png", ""),
    (f"{path}/dbScan/dbScan_p15.png", ""),
    (f"{path}/dbScan/dbScan_p16.png", ""),
    
    (f"{path}/dbScan/dbScan_p17.png", ""),
    (f"{path}/dbScan/dbScan_p18.png", ""),
    (f"{path}/dbScan/dbScan_p19.png", ""),
    (f"{path}/dbScan/dbScan_p20.png", ""),
]
    # # Filtracja istniejących plików
    # image_paths_and_titles = [(img_path, title) for img_path, title in image_paths_and_titles if os.path.exists(img_path)]

    # # Ścieżka wyjściowa dla złożonego wykresu
    # output_path = f"{path}/combined_plot.png"

    # # Oblicz liczbę wierszy i kolumn potrzebnych do wyświetlenia wszystkich obrazów
    # num_images = len(image_paths_and_titles)
    # cols = 4
    # rows = math.ceil(num_images / cols)

    # # Utwórz nowy wykres
    # fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))  # Dostosowanie rozmiaru wykresu do liczby wierszy
    
    # # Wczytaj każde zdjęcie i umieść je na odpowiednim miejscu w wykresie
    # for i, (img_path, title) in enumerate(image_paths_and_titles):
    #     row = i // cols  # Oblicz indeks wiersza
    #     col = i % cols   # Oblicz indeks kolumny
    #     img = mpimg.imread(img_path,format="png")
    #     axs[row, col].imshow(img)
    #     axs[row, col].axis('off')       # Wyłącz osie dla czytelności
    #     axs[row, col].set_title(title)  # Dodaj opisowy tytuł dla każdego obrazka
    
    # # Jeśli mamy mniej obrazów niż miejsc na siatce, wyłącz pozostałe osie
    # for j in range(i + 1, rows * cols):
    #     row = j // cols
    #     col = j % cols
    #     axs[row, col].axis('off')

    # # Ustaw przestrzeń między podwykresami
    # plt.subplots_adjust(wspace=0.2, hspace=0.3)
    
    # # Zapisz wykres do pliku
    # plt.savefig(output_path)
    
    # # Zamknij figurę, aby zwolnić pamięć
    # plt.close(fig)
    # print("Done")
    ############################################################################################################################################################
    # Filtracja istniejących plików
    image_paths_and_titles = [(img_path, title) for img_path, title in image_paths_and_titles if os.path.exists(img_path)]

    # Oblicz liczbę wierszy i kolumn potrzebnych do wyświetlenia wszystkich obrazów
    num_images = len(image_paths_and_titles)
    cols = 4
    rows = math.ceil(num_images / cols)

    # Wczytaj wszystkie obrazy
    images = [Image.open(img_path) for img_path, _ in image_paths_and_titles]

    # Rozmiar pojedynczego obrazu (zakładamy, że wszystkie obrazy mają ten sam rozmiar)
    img_width, img_height = images[0].size

    # Utworzenie nowego obrazu o odpowiednich wymiarach
    combined_image = Image.new('RGB', (cols * img_width, rows * img_height))

    # Umieszczenie każdego obrazu w odpowiednim miejscu w złożonym obrazie
    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        combined_image.paste(image, (col * img_width, row * img_height))

    # Ścieżka wyjściowa dla złożonego obrazu
    output_path = f"{path}/combined_plot.png"

    # Zapisz złożony obraz
    combined_image.save(output_path, format='PNG', dpi=(1000, 1000))

    print("Done")
    
    ################################################################################################################################
    
def save_points_to_file(x_circle, y_circle, x_random, y_random, folder):
    data = np.column_stack((np.concatenate([x_circle, x_random]), np.concatenate([y_circle, y_random])))
    file_path = f"{folder}/points.txt"
    np.savetxt(file_path, data, fmt='%.4f')

#################################################################################################################################
### Single/complete linkage
#################################################################################################################################

def LinkageAlgorithmLoop(file_path, method, max_d, num_measurements, max_d_range, result_path):
    # Wczytanie danych z pliku
    data = np.loadtxt(file_path)
    
    # Wykonanie algorytmu aglomeracyjnego single linkage
    Z = linkage(data, method=method)
    max_d_step = (max_d_range - max_d) / (num_measurements - 1)

    # Wyświetlenie wyniku
    print("Wartość kroku:", max_d_step)
    #######################################################################

    with open(result_path, "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Zapisz nagłówki kolumn
        csvwriter.writerow(["max_d", "largest_cluster_points", "remaining_points", "num_clusters", 
                            "mean_cluster_distance", "std_dev_distance",
                            "second_mean_cluster_distance", "second_std_dev_distance", 
                            "p1", "p2", "p3", "p4",
                            "k3_p1", "k3_p2", "k3_p3", "k3_p4",
                            "k5_p1", "k5_p2", "k5_p3", "k5_p4",
                            "k7_p1", "k7_p2", "k7_p3", "k7_p4",
                            "k9_p1", "k9_p2", "k9_p3", "k9_p4",
                            "test"])

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
            if mean_cluster_distance!=0 or second_mean_cluster_distance !=0:
                p1 = largest_cluster_points * (1/mean_cluster_distance)
                p2 = sqrt(largest_cluster_points)  * (1 / mean_cluster_distance)
                p3 = sqrt(largest_cluster_points)  * (1 / mean_cluster_distance**2)
                p4 = sqrt(largest_cluster_points)  * (1 / mean_cluster_distance)**2
                
                k3 = kNNAlgorithm(data=largest_cluster_data, k=3, folder=path+"/"+ method +"/", show_picture=False, save_picture =False, name=method + " " + str(max_d))
                k5 = kNNAlgorithm(data=largest_cluster_data, k=5, folder=path+"/"+ method +"/", show_picture=False, save_picture =False, name=method + " " + str(max_d))
                k7 = kNNAlgorithm(data=largest_cluster_data, k=7, folder=path+"/"+ method +"/", show_picture=False, save_picture =False, name=method + " " + str(max_d))
                k9 = kNNAlgorithm(data=largest_cluster_data, k=9, folder=path+"/"+ method +"/", show_picture=False, save_picture =False, name=method + " " + str(max_d))
                
                k3_p1 = largest_cluster_points * (1/k3)
                k3_p2 = sqrt(largest_cluster_points)  * (1 / k3)
                k3_p3 = sqrt(largest_cluster_points)  * (1 / k3**2)
                k3_p4 = sqrt(largest_cluster_points)  * (1 / k3)**2
                
                k5_p1 = largest_cluster_points * (1/k5)
                k5_p2 = sqrt(largest_cluster_points)  * (1 / k5)
                k5_p3 = sqrt(largest_cluster_points)  * (1 / k5**2)
                k5_p4 = sqrt(largest_cluster_points)  * (1 / k5)**2
                
                k7_p1 = largest_cluster_points  * (1/k7 )
                k7_p2 = sqrt(largest_cluster_points)  * (1 / k7 )
                k7_p3 = sqrt(largest_cluster_points)  * (1 / k7 **2)
                k7_p4 = sqrt(largest_cluster_points)  * (1 / k7 )**2
                
                k9_p1 = largest_cluster_points * (1/k9 )
                k9_p2 = sqrt(largest_cluster_points)  * (1 / k9 )
                k9_p3 = sqrt(largest_cluster_points)  * (1 / k9 **2)
                k9_p4 = sqrt(largest_cluster_points)  * (1 / k9 )**2
            # Zapisz dane do pliku CSV
            if remaining_points > largest_cluster_points:
                output_data = [max_d, largest_cluster_points, remaining_points, num_clusters, 
                               second_mean_cluster_distance, second_std_dev_distance, 
                               mean_cluster_distance, std_dev_distance,
                               p1, p2, p3, p4,
                               k3_p1, k3_p2, k3_p3, k3_p4,
                               k5_p1, k5_p2, k5_p3, k5_p4,
                               k7_p1, k7_p2, k7_p3, k7_p4,
                               k9_p1, k9_p2, k9_p3, k9_p4,
                               0]
            else:
                output_data = [max_d, largest_cluster_points, remaining_points, num_clusters, 
                               mean_cluster_distance, std_dev_distance, 
                               second_mean_cluster_distance, second_std_dev_distance,
                               p1, p2, p3, p4,
                               k3_p1, k3_p2, k3_p3, k3_p4,
                               k5_p1, k5_p2, k5_p3, k5_p4,
                               k7_p1, k7_p2, k7_p3, k7_p4,
                               k9_p1, k9_p2, k9_p3, k9_p4,
                               1]
            csvwriter.writerow(output_data)

            # Zwiększenie wartości d_max o krok
            max_d = max_d + max_d_step

def LinkageAlgorithm(file_path, method, max_d, name, folder, show_picture=False):
    # Wczytanie danych z pliku
    data = np.loadtxt(file_path)

    # Wykonanie algorytmu aglomeracyjnego single linkage
    Z = linkage(data, method=method)

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

    # # Wydrukowanie statystyki klastrów po zakończeniu pętli
    # print("Po aktualizacji:")
    # for cluster_id, count in clusters_count.items():
    #     print(f"Klaster {cluster_id}: {count} punktów")

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


    plt.title(f"{method} -- {name}: {round(max_d,4)}")
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

    print(f"Punkty zapisano do pliku {output_file}")

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
    mean_distances = data[:, 4]
    std_dev_distances = data[:, 5]
    second_mean_distances = data[:, 6]
    second_std_dev_distances = data[:, 7]

    p1 = data[:, 8]
    p2 = data[:, 9]
    p3 = data[:, 10]
    p4 = data[:, 11]

    k3_p1 = data[:, 12]
    k3_p2 = data[:, 13]
    k3_p3 = data[:, 14]
    k3_p4 = data[:, 15]
    
    k5_p1 = data[:, 16]
    k5_p2 = data[:, 17]
    k5_p3 = data[:, 18]
    k5_p4 = data[:, 19]
    
    k7_p1 = data[:, 20]
    k7_p2 = data[:, 21]
    k7_p3 = data[:, 22]
    k7_p4 = data[:, 23]
    
    k9_p1 = data[:, 24]
    k9_p2 = data[:, 25]
    k9_p3 = data[:, 26]
    k9_p4 = data[:, 27]
    
    # Znalezienie indeksu najmniejszego odchylenia standardowego
    min_std_dev_index = np.argmin(std_dev_distances)
    min_std_dev_d_max = d_max_values[min_std_dev_index]
    min_std_dev_value = std_dev_distances[min_std_dev_index]

    # Znalezienie indeksu najmniejszego odchylenia standardowego dla drugiego klastra
    min_second_std_dev_index = np.argmin(second_std_dev_distances)
    min_second_std_dev_d_max = d_max_values[min_second_std_dev_index]
    min_second_std_dev_value = second_std_dev_distances[min_second_std_dev_index]

    p1_index = np.argmax(p1)
    p1_d_max = d_max_values[p1_index]
    
    p2_d_max = find_max_and_d_max(p2, d_max_values)
    p3_d_max = find_max_and_d_max(p3, d_max_values)
    p4_d_max = find_max_and_d_max(p4, d_max_values)

    k3_p1_d_max = find_max_and_d_max(k3_p1, d_max_values)
    k3_p2_d_max = find_max_and_d_max(k3_p2, d_max_values)
    k3_p3_d_max = find_max_and_d_max(k3_p3, d_max_values)
    k3_p4_d_max = find_max_and_d_max(k3_p4, d_max_values)

    k5_p1_d_max = find_max_and_d_max(k5_p1, d_max_values)
    k5_p2_d_max = find_max_and_d_max(k5_p2, d_max_values)
    k5_p3_d_max = find_max_and_d_max(k5_p3, d_max_values)
    k5_p4_d_max = find_max_and_d_max(k5_p4, d_max_values)

    k7_p1_d_max = find_max_and_d_max(k7_p1, d_max_values)
    k7_p2_d_max = find_max_and_d_max(k7_p2, d_max_values)
    k7_p3_d_max = find_max_and_d_max(k7_p3, d_max_values)
    k7_p4_d_max = find_max_and_d_max(k7_p4, d_max_values)

    k9_p1_d_max = find_max_and_d_max(k9_p1, d_max_values)
    k9_p2_d_max = find_max_and_d_max(k9_p2, d_max_values)
    k9_p3_d_max = find_max_and_d_max(k9_p3, d_max_values)
    k9_p4_d_max = find_max_and_d_max(k9_p4, d_max_values)
    
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
    return p1_d_max, p2_d_max, p3_d_max, p4_d_max, k3_p1_d_max, k3_p2_d_max, k3_p3_d_max, k3_p4_d_max,k5_p1_d_max,k5_p2_d_max, k5_p3_d_max, k5_p4_d_max,k7_p1_d_max,k7_p2_d_max, k7_p3_d_max, k7_p4_d_max,k9_p1_d_max, k9_p2_d_max, k9_p3_d_max,k9_p4_d_max

def makeDendrogram(file_path, method, draw):
    # Wczytanie danych z pliku
    data = np.loadtxt(file_path)

    # Wykonanie algorytmu aglomeracyjnego single linkage
    Z = linkage(data, method=method)
    
    min_distance = Z[-1, 2]
    print("Minimalna odległość w dendrogramie:", min_distance)
    max_distance = Z[0, 2]
    print("Maksymalna odległość w dendrogramie:", max_distance)
    
    if (draw):
        plt.figure(figsize=(10, 5))
        dendrogram(Z)
        plt.title('Dendrogram - Algorytm aglomeracyjny')
        plt.xlabel('Indeksy danych')
        plt.ylabel('Odległość')
        plt.show()
    return max_distance,min_distance

#################################################################################################################################
### DBscan 
#################################################################################################################################

def DBscanAlgorithmLoop(min_epsilon, max_epsilon, epsilon_step, min_min_samples, max_min_samples, folder):
    start_time = time.time()  # Start pomiaru czasu

    # Wczytanie danych 
    filename = f"{folder}/points.txt"
    dataForDBSCAN = np.loadtxt(filename)

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

            p1 = p2 = p3 = p4 = p5 = p6 = p7 = p8 = p9 = p10 = p11 = p12 = p13 = p14 = p15 = p16 = p17 = p18 = p19 = p20 = 0
            if mean_distance != 0 and std_distance != 0:
                p1 = p_cluster              * (1 /mean_distance)
                p2 = sqrt(p_cluster)              * ((1 / mean_distance)**2)
                p3 = sqrt(p_cluster)             * (1 / mean_distance)
                p4 = sqrt(p_cluster)             * (1 / mean_distance**2)
                
                method = "dbScan"
                cluster_data =(dataForDBSCAN[labels == 0])
                k3 = kNNAlgorithm(data=cluster_data, k=3, folder=path+"/"+ method +"/", show_picture=False, save_picture =False, name=method)
                k5 = kNNAlgorithm(data=cluster_data, k=5, folder=path+"/"+ method +"/", show_picture=False, save_picture =False, name=method)
                k7 = kNNAlgorithm(data=cluster_data, k=7, folder=path+"/"+ method +"/", show_picture=False, save_picture =False, name=method)
                k9 = kNNAlgorithm(data=cluster_data, k=9, folder=path+"/"+ method +"/", show_picture=False, save_picture =False, name=method)
                
                p5 = p_cluster * (1/k3)
                p6 = sqrt(p_cluster)  * (1 / k3)
                p7 = sqrt(p_cluster)  * (1 / k3**2)
                p8 = sqrt(p_cluster)  * (1 / k3)**2
                
                p9 = p_cluster * (1/k5)
                p10 = sqrt(p_cluster)  * (1 / k5)
                p11 = sqrt(p_cluster)  * (1 / k5**2)
                p12 = sqrt(p_cluster)  * (1 / k5)**2
                
                p13 = p_cluster  * (1/k7 )
                p14 = sqrt(p_cluster)  * (1 / k7 )
                p15 = sqrt(p_cluster)  * (1 / k7 **2)
                p16 = sqrt(p_cluster)  * (1 / k7 )**2
                
                p17 = p_cluster * (1/k9 )
                p18 = sqrt(p_cluster)  * (1 / k9 )
                p19 = sqrt(p_cluster)  * (1 / k9 **2)
                p20 = sqrt(p_cluster)  * (1 / k9 )**2

            result = [epsilon_rounded, min_samples, p_cluster, no_noise, mean_distance, std_distance, no_clusters,
                p1,  p2,  p3, p4,
                p5,  p6,  p7, p8,
                p9,  p10, p11, p12,
                p13, p14, p15, p16,
                p17, p18, p19, p20]
            if no_clusters == 2 and no_noise != len(dataForDBSCAN) and p_cluster > 0:
                result.insert(4, "yes")
                results_yes.append(result)
            else:
                result.insert(4, "no")
            results.append(result)

        epsilon += epsilon_step

    # Zapisz wyniki do plików CSV
    columns =   ["epsilon", "min_samples", "p_cluster", "no_noise", "condition", "mean_distance", "std_distance", "no_clusters",
               "p1",   "p2",  "p3",  "p4",
                "p5",  "p6",  "p7",  "p8",
                "p9",  "p10", "p11", "p12",
                "p13", "p14", "p15", "p16",
                "p17", "p18", "p19", "p20"
               ]
    results_df = pd.DataFrame(results, columns=columns)
    results_yes_df = pd.DataFrame(results_yes, columns=columns)

    results_df.to_csv(f"{folder}/dbScan/dbScanResults.csv", index=False)
    results_yes_df.to_csv(f"{folder}/dbScan/dbScanResultsYes.csv", index=False)

    end_time = time.time()  # Koniec pomiaru czasu
    print(f"Czas działania: {end_time - start_time} sekund")

def DBscanChart(folder, show_picture, p_id):
    # Wczytanie danych z pliku CSV
    filename = f"{folder}/dbScan/dbScanResults.csv"
    data = pd.read_csv(filename)

    # Filtracja danych, aby uwzględnić tylko te wiersze, gdzie warunek jest spełniony
    data_yes = data[data['condition'] == 'yes']

    # Podział danych na odpowiednie zestawy
    x_yes = data_yes['epsilon'].tolist()
    y_yes = data_yes['min_samples'].tolist()
    mean_values = data_yes['mean_distance'].tolist()
    std_values = data_yes['std_distance'].tolist()
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
    dataForDBSCAN = np.loadtxt(file_path)

    # Wykonanie klasteryzacji DBSCAN
    dbscan = DBSCAN(eps=epsilon, min_samples=samples)
    dbscan.fit(dataForDBSCAN)

    # Wizualizacja klastrów
    plt.figure(figsize=(10, 6))
    plt.scatter(dataForDBSCAN[:, 0], dataForDBSCAN[:, 1], c=dbscan.labels_, cmap='viridis_r', s=10)
    plt.title(f"p={p_id}={p_val} E=ps{epsilon} Samples:{samples}")
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    output_file = f"{folder}/dbScan/dbScan_{p_id}.png"
    plt.savefig(output_file, dpi=1000) 
    plt.close()
#################################################################################################################################

def read_shape_from_file(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            # Rozdziel linie na części używając przecinka jako separatora
            x, y = line.strip().split(',')
            # Dodaj punkt do listy, przekonwertujając x i y na liczby zmiennoprzecinkowe
            points.append((float(x), float(y)))
    return points                                

def generate_points(generator_function, folder, noisePoints, circle_radius, shape_name, num_points, maxSize):

    if(shape_name):
        x_circle, y_circle, x_random, y_random = generator_function(noisePoints, shape_name,    num_points, maxSize)
    else:
        x_circle, y_circle, x_random, y_random = generator_function(noisePoints, circle_radius, num_points, maxSize)
        
    plot_points(x_circle, y_circle, x_random, y_random, folder=folder, save_picture=True, show_picture=False)
    save_points_to_file(x_circle, y_circle, x_random, y_random, folder=folder)

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

    # # Zapisywanie wyników do pliku CSV
    # if name:
    #     output_file = f"{folder}/kNN/kNN_{name}_{k}_results.csv"
    # else:
    #     output_file = f"{folder}/kNN/kNN_{k}_results.csv"
    
    # with open(output_file, mode="w", newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     # Zapisz nagłówki kolumn
    #     csvwriter.writerow(["Index", "Neighbors", "Distances", "Mean Distance", "Mean Mean Distance"])
    #     for i, (neighbors, dist, mean_dist) in enumerate(zip(indices[:, 1:], distances[:, 1:], mean_distances)):
    #         neighbors_str = ' '.join(str(index) for index in neighbors)
    #         dist_str = ' '.join(f"{d:.2f}" for d in dist)
    #         csvwriter.writerow([i, neighbors_str, dist_str, f"{mean_dist:.2f}", f"{mean_mean_distance:.2f}"])

    #print(f"Wyniki k-nn zostały zapisane do pliku {output_file}")
    #print(f"Średnia wartość z średnich odległości: {mean_mean_distance:.2f}")

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

maxSize = 100     
#Słownik
generators = {
    # "circle_A"  : (generate_points_circle,  None),
    # "circle_B"  : (generate_points_mises,   None),
    # "circle_C"  : (generate_points_normal,  None),
    # "ring"      : (generate_points_ring,    None),
    # "shape_A"   : (generate_points_in_shape, "drawn_figure_A.txt"),
     "shape_B"   : (generate_points_in_shape, "drawn_figure_B.txt"),
    # "shape_C"   : (generate_points_in_shape, "drawn_figure_C.txt")
}
# # Pętla główna
for noisePoints in range(400, 401, 100):
    num_points = 1000 - noisePoints
    
    for folder, (generator_function, shape_name) in generators.items():
        for circle_radius in np.arange(40, 41, 20):
    
            if(generator_function != generate_points_in_shape):
                path = f"{folder}/{circle_radius}/{noisePoints}"
            else:
                path = f"{folder}/{noisePoints}"
                
            if not os.path.exists(path):
                os.makedirs(path)
                
            if not os.path.exists(path+"/single"):
                os.makedirs(path+"/single")
                
            if not os.path.exists(path+"/complete"):
                os.makedirs(path+"/complete")
                
                
            # # ### Generowanie punktów    
            # # generate_points(generator_function, path, noisePoints, circle_radius, shape_name, num_points, maxSize)
            
            # # Wczytanie danych z pliku
            # data = np.loadtxt(f"{path}/points.txt")
            
            # ### KNN
            # k3 = kNNAlgorithm(data, k=3, folder=path, show_picture=False, save_picture =False, name=None)
            # k5 = kNNAlgorithm(data, k=5, folder=path, show_picture=False, save_picture =False, name=None)
            # k7 = kNNAlgorithm(data, k=7, folder=path, show_picture=False, save_picture =False, name=None)
            # k9 = kNNAlgorithm(data, k=9, folder=path, show_picture=False, save_picture =False, name=None)
            
            # ### Single linkage 
            # min_distance, max_distance = makeDendrogram(file_path = f"{path}/points.txt", method= 'single', draw=False)
            # LinkageAlgorithmLoop(file_path = f"{path}/points.txt", method= 'single', max_d = min_distance, max_d_range = max_distance, num_measurements = 40, result_path=f"{path}/single/singleLinkageLoopResults.csv")
            # p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20 = LinkageChart(file_path = f"{path}/single/singleLinkageLoopResults.csv", draw = False)

            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=p1,   name="p1",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=p2,   name="p2",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=p3,   name="p3",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=p4,   name="p4",  folder=path, show_picture= False)

            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=k3,   name="k3",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=k5,   name="k5",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=k7,   name="k7",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=k9,   name="k9",  folder=path, show_picture= False)
            
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=p5,   name="k3_p1",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=p6,   name="k3_p2",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=p7,   name="k3_p3",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=p8,   name="k3_p4",  folder=path, show_picture= False)
            
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=p9,   name="k5_p1",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=p10,  name="k5_p2",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=p11,  name="k5_p3",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=p12,  name="k5_p4",  folder=path, show_picture= False)
            
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=p13,  name="k7_p1",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=p14,  name="k7_p2",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=p15,  name="k7_p3",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=p16,  name="k7_p4",  folder=path, show_picture= False)
            
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=p17,  name="k9_p1",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=p18,  name="k9_p2",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=p19,  name="k9_p3",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'single', max_d=p20,  name="k9_p4",  folder=path, show_picture= False)


            # # ### Complete linkage
            # min_distance, max_distance = makeDendrogram(file_path = f"{path}/points.txt", method= 'complete', draw=False)
            # LinkageAlgorithmLoop(file_path = f"{path}/points.txt", method= 'complete', max_d = min_distance, max_d_range = max_distance, num_measurements = 40, result_path=f"{path}/complete/completeLinkageLoopResults.csv")
            # p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20 =  LinkageChart(file_path = f"{path}/complete/completeLinkageLoopResults.csv", draw = False)
             
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'complete', max_d=p1,  name="p1",    folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'complete', max_d=p2,  name="p2",    folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'complete', max_d=p3,  name="p3",    folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'complete', max_d=p4,  name="p4",    folder=path, show_picture= False)
            
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'complete', max_d=p5,   name="k3_p1",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'complete', max_d=p6,   name="k3_p2",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'complete', max_d=p7,   name="k3_p3",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'complete', max_d=p8,   name="k3_p4",  folder=path, show_picture= False)
            
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'complete', max_d=p9,   name="k5_p1",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'complete', max_d=p10,  name="k5_p2",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'complete', max_d=p11,  name="k5_p3",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'complete', max_d=p12,  name="k5_p4",  folder=path, show_picture= False)
            
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'complete', max_d=p13,  name="k7_p1",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'complete', max_d=p14,  name="k7_p2",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'complete', max_d=p15,  name="k7_p3",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'complete', max_d=p16,  name="k7_p4",  folder=path, show_picture= False)
            
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'complete', max_d=p17,  name="k9_p1",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'complete', max_d=p18,  name="k9_p2",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'complete', max_d=p19,  name="k9_p3",  folder=path, show_picture= False)
            # LinkageAlgorithm(file_path = f"{path}/points.txt", method= 'complete', max_d=p20,  name="k9_p4",  folder=path, show_picture= False)


            ## DBscan
            DBscanAlgorithmLoop(0.5, 20.0, 0.1, 10, 400, folder=path) #min_epsilon, max_epsilon, epsilon_step, min_min_samples, max_min_samples
            
            # Przygotowanie parametrów p1 do p20
            params = []
            for i in range(1, 21):
                ep, samp, p_val = DBscanChart(folder=path, show_picture=False, p_id=f"p{i}")
                params.append((ep, samp, p_val, f"p{i}"))

            # Wykonanie DBscan dla każdego zestawu parametrów
            for ep, samp, p_val, p_id in params:
                DBscan(file_path=f"{path}/points.txt", folder=path, epsilon=ep, samples=samp, p_id=p_id, p_val=p_val)

            
            ### Połączenie wykresów
            create_combined_plot(path)