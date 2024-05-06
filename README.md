# SWIZ Projekt Rekonstrukcji 3D

## Opis projektu

Projekt Rekonstrukcji 3D realizujący zagadnienia ze stereowizji i triangulacji budynków.  
Program będzie tworzyć chmurę punktów w przestrzeni 3D reprezentujących szkielet domu na podstawie jego zdjęć z 2 kamer.

## Etapy pracy
- Wykrywanie punktów 2D
- Dopasowanie punktów 2D na 2 obrazach
- Geometria epipolarna
- 3a. Jeśli znane są zarówno wewnętrzne, jak i zewnętrzne parametry kamery, rekonstrukcja za pomocą macierzy projekcji.
  - 3b. Jeśli znane są tylko parametry wewnętrzne, znormalizuj współrzędne i oblicz macierz podstawową.
  - 3c. Jeśli nie są znane ani parametry wewnętrzne, ani zewnętrzne, oblicz macierz podstawową.
- Mając macierz podstawową lub zasadniczą, przyjmij P1 = [I 0] i oblicz parametry kamery 2.
- Wykonaj triangulację, wiedząc, że x1 = P1 * X i x2 = P2 * X.
- Dostosowanie wiązki w celu zminimalizowania błędów odwzorowania i dopracowania współrzędnych 3D.

Przetłumaczono z DeepL.com (wersja darmowa)

## Autorzy
- Jakub Karpiński 
- Krzysztof Miśków 

## Zbiór Danych
Wykorzystamy poniższy zbiór danych zawierający zdjęcia z budynków oraz parametry kamer
https://www.robots.ox.ac.uk/~vgg/data/mview/

## Technologie

- **Python**: Język programowania.
- **OpenCV**: Biblioteka do przetwarzania obrazów.
