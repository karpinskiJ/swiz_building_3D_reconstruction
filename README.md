# Realizacja

## Podstawowa idea
Podstawową ideą w rekonstrukcji obrazów 3D jest to, że mając zestaw obrazów z których każdy jest wykonany z innego punktu widzenia, możemy dokonać rekonstrukcji trójwymiarowej sceny. Znając ruch kamer względem globalnego układu możemy określić ich ruch, czyli macierze projekcji.

## Asocjacja danych
Aby to wykonać, skonstruujemy pipeline składający się z dwóch głównych części: asocjacji danych i struktury z ruchu (SfM).

### Asocjacja danych
Asocjacja danych jest używana do sprawdzenia, czy para obrazów jest podobna do siebie. Dwa obrazy mogą być sprawdzone pod kątem podobieństwa, używając korespondencji obrazów (SIFT) oraz geometrii dwuobrazowej (two-view geometry).

1. **Znajdowanie powiązanych komponentów**:
   - Mając zestaw nieustrukturyzowanych obrazów, najpierw znajdujemy powiązane komponenty w tych obrazach. Pomaga to znaleźć nakładające się widoki w obrazach.
   - Używamy algorytmu SIFT, który pomaga wyodrębnić punkty kluczowe z obrazów.
   - Następnie wykonujemy korespondencje obrazów lub punktów kluczowych, używając algorytmu geometrii dwuobrazowej.
   - Otrzymujemy mapowanie cechy w jednym obrazie na podobną cechę w innym obrazie.

2. **Złożoność przeszukiwania par obrazów**:
   - Jeśli zestaw wejściowych obrazów N jest duży, to przeszukiwanie par obrazów staje się niewykonalne.
   - Złożoność zapytania dla jednego obrazu wynosi 𝑂(𝑁 ∙ 𝐾^2), gdzie K to liczba punktów kluczowych w każdym obrazie.
   - Używana jest efektywna metoda oparta na drzewach do wyszukiwania obrazów, co redukuje złożoność zapytania do 𝑂(𝐾 ∙ 𝑏 ∙ 𝐿), gdzie K to cechy w zapytaniu obrazu, b to gałęzie w drzewie, a l to poziomy w drzewie.

## Structure From Motion (SfM)
SfM jest odpowiedzialna za początkową rekonstrukcję, używając estymacji pozycji i technik triangulacji, a następnie udoskonalenie tego przy użyciu algorytmu dopasowania wiązki. MVS jest następnie stosowany, aby uzyskać gęstą reprezentację 3D.

### Incremental SfM
W naszym zastosowaniu użyliśmy Incremental SfM, która działa w następujący sposób:

1. **Wybór widoków**:
   - Najpierw wybieramy dwa niepanoramiczne widoki z grafu sceny wygenerowanego przez krok asocjacji danych.

2. **Algorytm 8-punktowy**:
   - Algorytm 8-punktowy jest używany do obliczenia macierzy fundamentalnej lub istotnej.
   - Macierz fundamentalna może być również uważana za projekcję kamery, którą można rozłożyć na dwie macierze P i P'. P' reprezentuje wewnętrzną kalibrację kamery.

3. **Liniowa triangulacja**:
   - Następnie stosujemy algorytm liniowej triangulacji, aby obliczyć korespondencje i uzyskać punkty 3D.

4. **Algorytm dopasowania wiązki**:
   - Algorytm dopasowania wiązki jest stosowany do udoskonalenia punktów 3D uzyskanych z poprzedniego kroku.

5. **Korespondencje 2D-3D**:
   - Następnie znajdujemy korespondencje 2D-3D i dodajemy więcej widoków do systemu.
   - Korespondencja 2D-2D jest najpierw ustanawiana między nowo dodanym obrazem a poprzednim obrazem, a następnie ustanawia się korespondencję 2D-3D, aby uzyskać punkty 3D.

6. **Obliczanie pozycji obrazów**:
   - Gdy ustanowimy korespondencje 2D-3D dla wszystkich obrazów, używamy algorytmu Perspective-n-Point (PnP), aby obliczyć pozycję obrazów względem światowych współrzędnych.

7. **Gęsta chmura punktów**:
   - Dodatkowo, więcej korespondencji 2D-2D może być wybranych w sąsiednich obrazach i zastosować liniową triangulację, aby znaleźć ich punkty 3D. To pomaga uzyskać bardziej gęstą chmurę punktów do rekonstrukcji 3D.

8. **Udoskonalenie punktów 3D**:
   - Po uzyskaniu punktów 3D, następnym krokiem jest udoskonalenie tych punktów przy użyciu algorytmu dopasowania wiązki.
   - Algorytm dopasowania wiązki jest stosowany zarówno do punktów 3D, jak i szacunków pozycji kamery uzyskanych z algorytmu PnP.
