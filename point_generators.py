# point_generators.py

import numpy as np
import matplotlib.pyplot as plt
import os
import math
from matplotlib.path import Path
from PIL import Image

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
 
def save_points_to_file(x_circle, y_circle, x_random, y_random, folder):
    
    # Convert tuples to NumPy arrays if not already arrays
    x_circle = np.array(x_circle) if isinstance(x_circle, tuple) else x_circle
    y_circle = np.array(y_circle) if isinstance(y_circle, tuple) else y_circle
    x_random = np.array(x_random) if isinstance(x_random, tuple) else x_random
    y_random = np.array(y_random) if isinstance(y_random, tuple) else y_random
    
    # Tworzenie etykiet dla danych: 1 dla danych z circle, 2 dla danych random
    labels_circle = np.ones(x_circle.shape[0])      # Wszystkie punkty z circle mają etykietę 1
    labels_random = np.full(x_random.shape[0], 2)   # Wszystkie punkty z random mają etykietę 2

    # Łączenie danych x, y oraz etykiet
    data_circle = np.column_stack((x_circle, y_circle, labels_circle))
    data_random = np.column_stack((x_random, y_random, labels_random))

    # Łączenie wszystkich danych w jedną macierz
    data = np.vstack((data_circle, data_random))

    # Określenie ścieżki do pliku
    file_path = f"{folder}/points.txt"

    # Zapisywanie danych do pliku z użyciem formatu z 4 miejscami po przecinku dla wartości x i y
    np.savetxt(file_path, data, fmt='%.4f %.4f %d')
    
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
    
def create_combined_plot(path):
    image_paths_and_titles = [
    (f"{path}/complete/clustering_p1.png", "Complete p1"),
    (f"{path}/complete/clustering_p2.png", "Complete p2"),
    (f"{path}/complete/clustering_p3.png", "Complete p3"),
    
    (f"{path}/complete/clustering_k3_p1.png", "Complete k3_p1"),
    (f"{path}/complete/clustering_k3_p2.png", "Complete k3_p2"),
    (f"{path}/complete/clustering_k3_p3.png", "Complete k3_p3"),
    
    (f"{path}/complete/clustering_k5_p1.png", "Complete k5_p1"),
    (f"{path}/complete/clustering_k5_p2.png", "Complete k5_p2"),
    (f"{path}/complete/clustering_k5_p3.png", "Complete k5_p3"),
    
    (f"{path}/complete/clustering_k7_p1.png", "Complete k7_p1"),
    (f"{path}/complete/clustering_k7_p2.png", "Complete k7_p2"),
    (f"{path}/complete/clustering_k7_p3.png", "Complete k7_p3"),
    
    (f"{path}/complete/clustering_k9_p1.png", "Complete k9_p1"),
    (f"{path}/complete/clustering_k9_p2.png", "Complete k9_p2"),
    (f"{path}/complete/clustering_k9_p3.png", "Complete k9_p3"),
    
    (f"{path}/single/clustering_p1.png", "Single p1"),
    (f"{path}/single/clustering_p2.png", "Single p2"),
    (f"{path}/single/clustering_p3.png", "Single p3"),

    (f"{path}/single/clustering_k3_p1.png", "Single k3_p1"),
    (f"{path}/single/clustering_k3_p2.png", "Single k3_p2"),
    (f"{path}/single/clustering_k3_p3.png", "Single k3_p3"),
    
    (f"{path}/single/clustering_k5_p1.png", "Single k5_p1"),
    (f"{path}/single/clustering_k5_p2.png", "Single k5_p2"),
    (f"{path}/single/clustering_k5_p3.png", "Single k5_p3"),
    
    (f"{path}/single/clustering_k7_p1.png", "Single k7_p1"),
    (f"{path}/single/clustering_k7_p2.png", "Single k7_p2"),
    (f"{path}/single/clustering_k7_p3.png", "Single k7_p3"),
    
    (f"{path}/single/clustering_k9_p1.png", "Single k9_p1"),
    (f"{path}/single/clustering_k9_p2.png", "Single k9_p2"),
    (f"{path}/single/clustering_k9_p3.png", "Single k9_p3"),

    (f"{path}/single/clustering_k3.png", "Single k3"),
    (f"{path}/single/clustering_k5.png", "Single k5"),
    (f"{path}/single/clustering_k7.png", "Single k7"),
    
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
]

    # Filtracja istniejących plików
    image_paths_and_titles = [(img_path, title) for img_path, title in image_paths_and_titles if os.path.exists(img_path)]

    # Oblicz liczbę wierszy i kolumn potrzebnych do wyświetlenia wszystkich obrazów
    num_images = len(image_paths_and_titles)
    cols = 3
    rows = math.ceil(num_images / cols)

    # Wczytaj wszystkie obrazy
    images = [Image.open(img_path) for img_path, _ in image_paths_and_titles]

    # Rozmiar pojedynczego obrazu (zakładamy, że wszystkie obrazy mają ten sam rozmiar)
    img_width, img_height = images[0].size
    print(img_width)
    print(img_height)
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
