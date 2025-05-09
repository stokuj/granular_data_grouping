# Hierarchiczne grupowanie danych w Pythonie

## Wstęp

Grupowanie danych (ang. clustering) jest jedną z fundamentalnych technik eksploracji danych, stosowaną w wielu dziedzinach nauki i przemysłu, takich jak:

- Analiza biochemiczna (grupowanie sekwencji genów)
- Segmentacja klientów w marketingu
- Wykrywanie anomalii w cyberbezpieczeństwie
- Wizualizacja dużych zbiorów danych

W niniejszym projekcie skupiamy się na **granularnym grupowaniu danych w jedną grupę**. Celem jest wyodrębnienie dużego, wewnętrznie spójnego klastra oraz sklasyfikowanie pozostałych punktów jako szum.

## Dziedzina problemu

1. **Analiza skupień (Cluster Analysis)**
   - Techniki nienadzorowanej klasyfikacji danych bez etykiet.
2. **Obliczenia ziarniste (Granular Computing)**
   - Zarządzanie informacją przez tworzenie granulek danych o różnym poziomie szczegółowości.
3. **Detekcja skupień w szumie**
   - Identyfikacja istotnych struktur nawet w obecności dużej liczby punktów zakłócających.

## Cel badania

- Zaprojektowanie i implementacja metod granularnego grupowania, umożliwiających selekcję **jednego dominującego klastra**.
- Opracowanie ilościowych kryteriów oceny granic klastra.
- Porównanie trzech podejść klasteryzacyjnych: DBSCAN, Single Linkage, Complete Linkage.
- Przeprowadzenie eksperymentów na syntetycznych zbiorach o różnych kształtach i poziomach szumu.
- Ocena wydajności obliczeniowej wszystkich algorytmów.

## Dlaczego jedna grupa?

Typowe algorytmy klasteryzacji dzielą dane na wiele klastrów. W naszym podejściu interesuje nas **jedynie** identyfikacja jednego głównego klastra:

- Koncentracja na najistotniejszej strukturze w danych.
- Uniknięcie nadmiernej segmentacji prowadzącej do artefaktów.
- Jednoznaczne kryteria definiujące granice klastra.

## Zasada uzasadnionej granulacji

Granularne grupowanie opiera się na łączeniu dwóch komplementarnych miar:

1. **Wielkość klastra (N)**
2. **Jednorodność wewnętrzna**:
   - Średnia odległość między punktami (d̄)
   - Średnia odległość do k-tego sąsiada (d_k)

Formuły kryteriów:
```math
P1 = N \times \frac{1}{\bar{d}}                \\
P2 = N \times \left(\frac{1}{\bar{d}}\right)^2  \\
P3 = \sqrt{N} \times \left(\frac{1}{\bar{d}}\right)^2
```
oraz warianty:
```math
P1_k = N \times \frac{1}{d_k},
P2_k = N \times \left(\frac{1}{d_k}\right)^2,
P3_k = \sqrt{N} \times \left(\frac{1}{d_k}\right)^2.
```

## Teoretyczne podstawy algorytmów

### DBSCAN
- Złożoność: \(O(n \log n)\) przy zastosowaniu indeksów przestrzennych; w przeciwnym wypadku \(O(n^2)\).
- Dwa parametry kluczowe: promień ε i minimalna liczba punktów MinPts.

### Single Linkage
- Zasada: odległość między klastrami = minimalna odległość między punktami.
- Zaburzenia: może łączyć długie łańcuchy.

### Complete Linkage
- Zasada: odległość między klastrami = maksymalna odległość między punktami.
- Zaleta: tworzy bardziej zwarte klastry; wada: wrażliwy na izolowane punkty.

## Struktura projektu

```text
project_root/
├── point_generators.py      # Generowanie danych
├── DBscan.py                # Implementacja DBSCAN
├── KNN.py                   # Implementacja algorytmu kNN
├── main.py                  # Skrypt eksperymentalny
├── wyniki_czasu_wykonania.csv
├── requirements.txt
└── results/
    ├── circle/...
    ├── ring/...
    └── normal/...
```

## Instalacja

```bash
git clone https://github.com/użytkownik/projekt-klastrowania.git
cd projekt-klastrowania
pip install -r requirements.txt
```

## Użytkowanie przykładowe

### Generowanie danych
```python
import point_generators as pg
# generuj 800 punktów w kole o promieniu 50 + 200 punktów szumu
pg.generate_points_circle(
    output_folder='results/circle',
    noise_points=200,
    circle_radius=50,
    num_points=800,
    max_size=100
)
```

### DBSCAN
```python
from DBscan import DBscanAlgorithmLoop, DBscanChart, DBscan
# iteracja po eps ∈ [2,20], step=0.5, MinPts ∈ [5,50]
DBscanAlgorithmLoop(2,20,0.5,5,50,'results/circle')
# analiza wykresów i wybór optymalnych parametrów
eps, min_pts, p_val = DBscanChart('results/circle', show_picture=True, p_id='p1')
# wykonanie docelowe
DBscan(
    file_path='results/circle/points.txt',
    folder='results/circle',
    epsilon=eps,
    samples=min_pts,
    p_id='p1',
    p_val=p_val
)
```

### Single / Complete Linkage
```python
from main import makeDendrogram, LinkageAlgorithmLoop, LinkageAlgorithm
# oblicz zakres d_min, d_max
d_min, d_max = makeDendrogram('results/circle/points.txt', method='single', draw=True)
# iteracja po 200 krokach
LinkageAlgorithmLoop(
    path='results/circle',
    file_path='results/circle/points.txt',
    method='single',
    max_d=d_min,
    max_d_range=d_max,
    num_measurements=200,
    result_path='results/circle/single/results_loop.csv'
)
# końcowe uruchomienie dla optymalnego d_max
opt_d = 10.5  # wynik z wykresu
LinkageAlgorithm(
    file_path='results/circle/points.txt',
    method='single',
    max_d=opt_d,
    name='opt',
    folder='results/circle',
    show_picture=True
)
```

## Szczegóły implementacji

1. `point_generators.py`:
   - Funkcja `generate_points`: wrapper wybierający odpowiedni generator.
   - Formaty wyjściowe: `points.txt` z trzema kolumnami (x, y, etykieta).

2. `DBscan.py`:
   - `DBscanAlgorithmLoop`: zapisuje wyniki P1–P15 do CSV.
   - `DBscanChart`: rysuje wykresy miar vs eps / MinPts.
   - `DBscan`: generuje ostateczne klastry i zapisuje `DBscanResults.csv`.

3. `KNN.py`:
   - Łatwa integracja z `LinkageAlgorithmLoop` do obliczania metryk P1–P15.

4. `main.py`:
   - Import wszystkich modułów.
   - Konfiguracja parametrów eksperymentu.
   - Pętle po poziomach szumu, promieniach i kształtach.
   - Zapis wyników do `wyniki_czasu_wykonania.csv`.

## Wyniki eksperymentów

| Algorytm        | Dokładność (średnia) | Czas relatywny | Uwagi                             |
|-----------------|----------------------|----------------|-----------------------------------|
| DBSCAN          | 0.89                 | 1.0            | Wrażliwy na ε, MinPts             |
| Single Linkage  | 0.82                 | 0.2            | Dobre dla kształtów łancuchowych  |
| Complete Linkage| 0.75                 | 0.25           | Najgorsze przy dużym szumie       |

## Wnioski

- **Najlepszy** algorytm do granularnego wykrywania jednego klastra: DBSCAN, ale wymaga optymalizacji parametrów.
- **Single Linkage** sprawdza się lepiej przy nietypowych kształtach (von Mises).
- **Complete Linkage** nie polecany w obecności silnego szumu.
- Wzrost udziału szumu powyżej 60% znacząco obniża jakość wszystkich algorytmów.

## Możliwości rozwoju

- Automatyczny dobór parametrów przy użyciu technik optymalizacji globalnej lub uczących się heurystyk.
- Zastosowanie algorytmów hybrydowych (DBSCAN + hierarchiczne).
- Ekstrapolacja na dane wielowymiarowe i dynamiczne strumienie danych.

## Autor i licencja

- **Autor**: Twoje Imię i Nazwisko
- **Licencja**: MIT
