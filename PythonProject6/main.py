from itertools import combinations, chain
from collections import defaultdict

def wczytaj_transakcje(plik):
    with open(plik, 'r', encoding='utf-8') as f:
        return [set(linia.strip().split(',')) for linia in f if linia.strip()]

def licz_czestotliwosci(transakcje, kandydaci):
    licznik = defaultdict(int)
    for transakcja in transakcje:
        for kandydat in kandydaci:
            if kandydat.issubset(transakcja):
                licznik[frozenset(kandydat)] += 1
    return licznik

def generuj_kandydatow(poprz_zbiory, rozmiar):
    unikalne = set(chain.from_iterable(poprz_zbiory))
    return [set(x) for x in combinations(unikalne, rozmiar)]

def filtruj_frequent(licznik, prog):
    return {zbior for zbior, liczba in licznik.items() if liczba >= prog}

def generuj_reguly(frequent_sets, transakcje, prog_jakosci):
    D = len(transakcje)
    reguly = []
    wsparcie_map = {}
    for zbior in frequent_sets:
        wsparcie_map[zbior] = sum(1 for t in transakcje if zbior.issubset(t)) / D

    for zbior in frequent_sets:
        if len(zbior) < 2:
            continue
        for i in range(1, len(zbior)):
            for poprzednik in combinations(zbior, i):
                poprzednik = frozenset(poprzednik)
                nastepnik = zbior - poprzednik
                if not nastepnik:
                    continue
                wsp = wsparcie_map[zbior]
                ufn = wsp / wsparcie_map.get(poprzednik, 1e-10)
                jakosc = wsp * ufn
                if jakosc >= prog_jakosci:
                    reguly.append((poprzednik, nastepnik, round(wsp, 3), round(ufn, 3), round(jakosc, 3)))
    return reguly

def apriori(transakcje, prog_czestosci=2, prog_jakosci=1/3):
    czestotliwosci = []
    frequent_sets = []
    rozmiar = 1

    kandydaci = [set([item]) for item in set(chain.from_iterable(transakcje))]
    while True:
        licznik = licz_czestotliwosci(transakcje, kandydaci)
        czeste = filtruj_frequent(licznik, prog_czestosci)
        if not czeste:
            break
        frequent_sets.extend(czeste)
        czestotliwosci.append(licznik)
        rozmiar += 1
        kandydaci = generuj_kandydatow(czeste, rozmiar)

    reguly = generuj_reguly(frequent_sets, transakcje, prog_jakosci)
    return frequent_sets, reguly

if __name__ == '__main__':
    plik = 'paragony.txt'
    transakcje = wczytaj_transakcje(plik)
    czeste_zbiory, reguly = apriori(transakcje, prog_czestosci=2, prog_jakosci=1/3)

    print("\nCZĘSTE ZBIORY (z częstościami):")
    wszystkie_transakcje = len(transakcje)
    for zbior in czeste_zbiory:
        licznik = sum(1 for t in transakcje if zbior.issubset(t))
        print(f"{set(zbior)} → wystąpień: {licznik}, wsparcie: {round(licznik / wszystkie_transakcje, 3)}")

    print("\nREGUŁY ASOCJACYJNE:")
    for poprzednik, nastepnik, wsp, ufn, jakosc in reguly:
        print(f"{set(poprzednik)} => {set(nastepnik)} | wsparcie={wsp}, ufność={ufn}, jakość={jakosc}")
