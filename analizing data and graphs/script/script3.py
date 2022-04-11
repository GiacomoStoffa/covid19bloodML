import statistics

import pandas as pd
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from pylab import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from matplotlib import rcParams, patches, gridspec
import seaborn as sns

dataset = pd.read_excel("selected_dataset_diagnostico_COVID.xls")
data = []
for i in dataset["DATA"]:
    data.append((i - datetime.datetime(2020, 2, 7)).days + 1)
dataset["DATA"] = data

# verifico quali punti non hanno 7 giorni precedenti
for i in range(0, len(dataset["DATA"]) - 1):
    if abs(dataset["DATA"][i] - dataset["DATA"][i + 1]) > 5:
        print(i)
# il primo il secondo e il 1405 potrebbero dare problemi.

# sostituisco i nan con gli 0 (che poi non considero nel grafico)
dataset['PLT1'] = dataset['PLT1'].fillna(0)
dataset["PCR"] = dataset['PCR'].fillna(0)
dataset["FG"] = dataset['FG'].fillna(0)
dataset["LY"] = dataset['LY'].fillna(0)
dataset["WBC"] = dataset['WBC'].fillna(0)
dataset["AST"] = dataset['AST'].fillna(0)
dataset["LDH"] = dataset['LDH'].fillna(0)
dataset["CA"] = dataset['CA'].fillna(0)


# aggiungo gli zeri dove non c'è il giorno
# definisco una funzione di supporto
def contains(lista, valore):
    cnt = 0
    for i in lista:
        if i == valore:
            return cnt
        cnt += 1
    return -1


giorni = []
valori = []
valoriPLT1 = []
valoriPCR = []
valoriFG = []
valoriLY = []
valoriWBC = []
valoriAST = []
valoriLDH = []
valoriCA = []
tamponi = []
sesso = []
età = []

cnt = 1
for i in np.arange(1, 115):
    a = contains(dataset["DATA"], i)
    if a == -1:
        giorni.append(cnt)
        valoriPLT1.append(0)
        valoriPCR.append(0)
        valoriFG.append(0)
        valoriLY.append(0)
        valoriWBC.append(0)
        valoriAST.append(0)
        valoriLDH.append(0)
        valoriCA.append(0)
        # per tamponi e sesso (che hanno valori binari 0/1) uso nan al posto di zero
        tamponi.append(-1)
        sesso.append(-1)
        età.append(0)
    else:
        indice = a
        while (indice < len(dataset["DATA"]) - 7) & (dataset["DATA"][indice] == dataset["DATA"][indice + 1]):
            giorni.append(dataset["DATA"][indice])
            valoriPLT1.append(dataset["PLT1"][indice])
            valoriPCR.append(dataset["PCR"][indice])
            valoriFG.append(dataset["FG"][indice])
            valoriLY.append(dataset["LY"][indice])
            valoriWBC.append(dataset["WBC"][indice])
            valoriAST.append(dataset["AST"][indice])
            valoriLDH.append(dataset["LDH"][indice])
            valoriCA.append(dataset["CA"][indice])
            tamponi.append(dataset["TAMPONI"][indice])
            sesso.append(dataset["Sex"][indice])
            età.append(dataset["Age"][indice])
            indice += 1
        giorni.append(dataset["DATA"][indice])
        valoriPLT1.append(dataset["PLT1"][indice])
        valoriPCR.append(dataset["PCR"][indice])
        valoriFG.append(dataset["FG"][indice])
        valoriLY.append(dataset["LY"][indice])
        valoriWBC.append(dataset["WBC"][indice])
        valoriAST.append(dataset["AST"][indice])
        valoriLDH.append(dataset["LDH"][indice])
        valoriCA.append(dataset["CA"][indice])
        tamponi.append(dataset["TAMPONI"][indice])
        sesso.append(dataset["Sex"][indice])
        età.append(dataset["Age"][indice])
    cnt += 1
# ora gli utlimi mancanti
for indice in range(1635, 1641):
    giorni.append(dataset["DATA"][indice])
    valoriPLT1.append(dataset["PLT1"][indice])
    valoriPCR.append(dataset["PCR"][indice])
    valoriFG.append(dataset["FG"][indice])
    valoriLY.append(dataset["LY"][indice])
    valoriWBC.append(dataset["WBC"][indice])
    valoriAST.append(dataset["AST"][indice])
    valoriLDH.append(dataset["LDH"][indice])
    valoriCA.append(dataset["CA"][indice])
    tamponi.append(dataset["TAMPONI"][indice])
    sesso.append(dataset["Sex"][indice])
    età.append(dataset["Age"][indice])

# ora aggiorno il dataset con gli zeri
dict = {
    "DATA": giorni,
    "PLT1": valoriPLT1,
    "PCR": valoriPCR,
    "FG": valoriFG,
    "LY": valoriLY,
    "WBC": valoriWBC,
    "AST": valoriAST,
    "LDH": valoriLDH,
    "CA": valoriCA,
    "TAMPONI": tamponi,
    "Sex": sesso,
    "Age": età
}
dataset = pd.DataFrame(dict)

# verifico la nuova posizione dei punti che potevano dare problemi
for i in range(0, len(dataset["DATA"]) - 1):
    if (dataset["PCR"][i] == 1.1) & (dataset["FG"][i] == 324) & (dataset["LY"][i] == 33.2):
        print(dataset["DATA"][i])


# ora si trova in posizione 1024


# parto con PLT1

# definisco una funzione di supporto
def aggiornaListaSupporto(counter, lista, c, nomeGrandezza):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(dataset[nomeGrandezza][r])
        r += 1
    return nuovaLista


# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiorno = []
i = 0
while i < len(dataset["DATA"]) - 1:
    if dataset["DATA"][i] != dataset["DATA"][i + 1]:
        indiciCambioGiorno.append(i)
    i += 1

# inserisco le medie dei primi giorni (dall' ottavo giorno al sedicesimo, esclusi il 9 e il 10
# perché non ci sono dati nei 7 giorni precedenti)
medieGiorniPrecedentiPLT1 = [446, 200, 200, 200, 200, 200, 200]  # contiene le medie dei valori dei 7 giorni precedenti
stdevGiorniPrecedentiPLT1 = [0, 0, 0, 0, 0, 0, 0]
# inserisco anche per la lista di supporto i valori dei primi giorni
listaSupporto = [200, 0, 0, 0, 0, 0, 168]

i = 16
cnt = 8  # indica il giorno da cui parte la lista di supporto
nPLT1 = [1, 1, 1, 1, 1, 1, 1]
for i in range(16, 1655):
    if i in indiciCambioGiorno:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(dataset["PLT1"][i])
        listaSupporto = aggiornaListaSupporto(indiciCambioGiorno[cnt], listaSupporto,
                                              indiciCambioGiorno[cnt + 7], "PLT1")
        # elimino gli zeri dalla lista
        listaSupporto = [i for i in listaSupporto if i != 0]
        nPLT1.append(len(listaSupporto))
        medieGiorniPrecedentiPLT1.append(mean(listaSupporto))
        if len(listaSupporto) == 0:
            stdevGiorniPrecedentiPLT1.append(0)
        stdevGiorniPrecedentiPLT1.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(dataset["PLT1"][i])
# aggiungo l'ultimo valore mancante alle deviazioni e alle medie
medieGiorniPrecedentiPLT1.append(233.18)
stdevGiorniPrecedentiPLT1.append(71.8)

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno
for i in range(0, 10):
    medieGiorniPrecedentiPLT1.pop(0)
    stdevGiorniPrecedentiPLT1.pop(0)
print(len(medieGiorniPrecedentiPLT1))
print(len(stdevGiorniPrecedentiPLT1))

dict = {
    "giorno": np.arange(20, 116).tolist(),
    "media": medieGiorniPrecedentiPLT1,
    "dev standard": stdevGiorniPrecedentiPLT1
}

datiPLT1 = pd.DataFrame(dict)

# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
# gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])
# gs.update(wspace=0.03, hspace=0.03)
# ax = plt.subplot(gs[1])
sns.lineplot(x="giorno", y="media", data=datiPLT1, marker=".", markersize=8, color="#1034a6")

# creo la linea verticale tratteggiata
# plt.plot([117] * 430, np.arange(150, 580), linestyle='dashed', color="black", linewidth=0.8)

# dato che non mostra gli intervalli di confidenza li calcolo io
# ho già calcolato n, devo solo aggiungergli alcuni valori mancanti
nPLT1.append(46)
for i in range(0, 10):
    nPLT1.pop(0)

icPLT1 = []
cnt = 0
for i in datiPLT1["dev standard"]:
    icPLT1.append(1.96 * i / sqrt(nPLT1[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinPLT1 = []
cnt = 0
for i in datiPLT1["dev standard"]:
    icMinPLT1.append(datiPLT1["media"][cnt] - icPLT1[cnt])
    cnt += 1
icMaxPLT1 = []
cnt = 0
for i in datiPLT1["dev standard"]:
    icMaxPLT1.append(datiPLT1["media"][cnt] + icPLT1[cnt])
    cnt += 1

# disegno gli intervalli di confidenza sul grafico
plt.margins(0.05)


# voglio colorare con un colore più chiaro la fascia se N è basso e più alto se N è alto

# devo definire una funzione di supporto per fare i rettangoli e non dei poligoni con i lati storti
# questa esatta funzione verrà riusata anche per gli altri dati, quindi non commentarla
def creaRettangolo(giorni, icMin, icMax, n):
    nuoviGiorni = []
    nuoviIcMin = []
    nuoviIcMax = []
    nuoviN = []
    for i in giorni:
        nuoviGiorni.append(i - 0.5)
        nuoviGiorni.append(i + 0.5)
    for i in icMin:
        nuoviIcMin.append(i)
        nuoviIcMin.append(i)
    for i in icMax:
        nuoviIcMax.append(i)
        nuoviIcMax.append(i)
    for i in n:
        nuoviN.append(i)
        nuoviN.append(i)
    return [nuoviGiorni, nuoviIcMin, nuoviIcMax, nuoviN]


datiIC = creaRettangolo(datiPLT1["giorno"], icMinPLT1, icMaxPLT1, nPLT1)
giorni = datiIC[0]
minIC = datiIC[1]
maxIC = datiIC[2]
nPLT1 = datiIC[3]


for b in range(0, len(giorni)):
    i = nPLT1[b]
    if i <= 5:
        alpha = 0.02
    if i >= 30:
        alpha = 0.8
    if (i > 5) & (i < 30):
        alpha = (i - 5) / 25
    ax.fill_between(giorni[b:(b + 2)], minIC[b: (b + 2)], maxIC[b: (b + 2)], edgecolor="face",
                    color="#1b6ca8", alpha=alpha)
    # salto un punto
    b += 1

ax.fill_between(datiPLT1["giorno"], icMinPLT1, icMaxPLT1, alpha=0.9, color="#1b6ca8",
                label="95% confidence interval").set_visible(False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xlabel("Time (days)", fontsize=12)
plt.ylabel("previous 7 days mean PLT1", fontsize=12)

plt.xlim(8, 115.2)
plt.xticks([38, 69, 99], labels=[])
ax.tick_params(which='major', length=6, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([24, 55, 85], minor=True)
ax.tick_params(which='minor', length=10, color='black')
ax.text(8, 152, "F", size=10)
ax.text(23.6, 152, "M", size=10)
ax.text(54.7, 152, "A", size=10)
ax.text(84.6, 152, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda, ma devo fare un bbox per metterla in una posizione comoda

plt.legend(loc="upper right", ncol=1, fancybox=True,
           frameon=False)
plt.show()


# faccio il grafico per PCR

# definisco una funzione di supporto
def aggiornaListaSupporto(counter, lista, c, nomeGrandezza):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(dataset[nomeGrandezza][r])
        r += 1
    return nuovaLista


# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiorno = []
i = 0
while i < len(dataset["DATA"]) - 1:
    if dataset["DATA"][i] != dataset["DATA"][i + 1]:
        indiciCambioGiorno.append(i)
    i += 1


# inserisco le medie dei primi giorni (dall' ottavo giorno al sedicesimo, esclusi il 9 e il 10
# perché non ci sono dati nei 7 giorni precedenti)
medieGiorniPrecedentiPCR = [6.5, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4]  # contiene le medie dei valori dei 7 giorni precedenti
stdevGiorniPrecedentiPCR = [0, 0, 0, 0, 0, 0, 0]
# inserisco anche per la lista di supporto i valori dei primi giorni
listaSupporto = [1.4, 0, 0, 0, 0, 0, 14.3]


i = 16
cnt = 8  # indica il giorno da cui parte la lista di supporto
nPCR = [1, 1, 1, 1, 1, 1, 1]
for i in range(16, 1655):
    if i in indiciCambioGiorno:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(dataset["PCR"][i])
        listaSupporto = aggiornaListaSupporto(indiciCambioGiorno[cnt], listaSupporto,
                                              indiciCambioGiorno[cnt + 7], "PCR")
        # elimino gli zeri dalla lista
        listaSupporto = [i for i in listaSupporto if i != 0]
        nPCR.append(len(listaSupporto))
        medieGiorniPrecedentiPCR.append(mean(listaSupporto))
        if len(listaSupporto) == 0:
            stdevGiorniPrecedentiPCR.append(0)
        stdevGiorniPrecedentiPCR.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(dataset["PCR"][i])
# aggiungo l'ultimo valore mancante alle deviazioni e alle medie
medieGiorniPrecedentiPCR.append(42.53)
stdevGiorniPrecedentiPCR.append(59.12)
for i in range(0, 10):
    medieGiorniPrecedentiPCR.pop(0)
    stdevGiorniPrecedentiPCR.pop(0)

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno

dict = {
    "giorno": np.arange(20, 116),
    "media": medieGiorniPrecedentiPCR,
    "dev standard": stdevGiorniPrecedentiPCR
}

datiPCR = pd.DataFrame(dict)



# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
# gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])
# gs.update(wspace=0.03, hspace=0.03)
# ax = plt.subplot(gs[1])
sns.lineplot(x="giorno", y="media", data=datiPCR, marker=".", markersize=8, color="#1034a6")

# creo la linea verticale tratteggiata
# plt.plot([117] * 430, np.arange(150, 580), linestyle='dashed', color="black", linewidth=0.8)

# dato che non mostra gli intervalli di confidenza li calcolo io
# ho già calcolato n, devo solo aggiungergli alcuni valori mancanti
nPCR.append(43)
for i in range(0, 10):
    nPCR.pop(0)


icPCR = []
cnt = 0
for i in datiPCR["dev standard"]:
    icPCR.append(1.96 * i / sqrt(nPCR[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinPCR = []
cnt = 0
for i in datiPCR["dev standard"]:
    icMinPCR.append(datiPCR["media"][cnt] - icPCR[cnt])
    cnt += 1
icMaxPCR = []
cnt = 0
for i in datiPCR["dev standard"]:
    icMaxPCR.append(datiPCR["media"][cnt] + icPCR[cnt])
    cnt += 1

# disegno gli intervalli di confidenza sul grafico
plt.margins(0.05)

datiIC = creaRettangolo(datiPCR["giorno"], icMinPCR, icMaxPCR, nPCR)
giorni = datiIC[0]
minIC = datiIC[1]
maxIC = datiIC[2]
nPCR = datiIC[3]

for b in range(0, len(nPCR)):
    i = nPCR[b]
    if i <= 5:
        alpha = 0.02
    if i >= 30:
        alpha = 0.8
    if (i > 5) & (i < 30):
        alpha = (i - 5) / 25
    ax.fill_between(giorni[b:(b + 2)], minIC[b: (b + 2)], maxIC[b: (b + 2)], edgecolor="face", color="#1b6ca8", alpha=alpha)
    # salto un punto, altrimenti ripeto la stessa cosa
    b += 1

# ne faccio uno invisibile solo per mettere la label
ax.fill_between(datiPCR["giorno"], icMinPCR, icMaxPCR, alpha=0.9, color="#1b6ca8",  label="95% confidence interval").set_visible(False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xlabel("Time (days)", fontsize=12)
plt.ylabel("previous 7 days mean PCR", fontsize=12)

plt.xlim(7.8, 115.2)
plt.xticks([38, 69, 99], labels=[])
ax.tick_params(which='major', length=6, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([24, 55, 85], minor=True)
ax.tick_params(which='minor', length=4, color='black')
ax.text(8, -25, "F", size=10)
ax.text(23.6, -25, "M", size=10)
ax.text(55.7, -25, "A", size=10)
ax.text(85.6, -25, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda, ma devo fare un bbox per metterla in una posizione comoda

plt.legend(loc="upper right", ncol=1, fancybox=True,
           frameon=False)
plt.show()


# passo al grafico per FG


# definisco una funzione di supporto
def aggiornaListaSupporto(counter, lista, c, nomeGrandezza):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(dataset[nomeGrandezza][r])
        r += 1
    return nuovaLista


# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiorno = []
i = 0
while i < len(dataset["DATA"]) - 1:
    if dataset["DATA"][i] != dataset["DATA"][i + 1]:
        indiciCambioGiorno.append(i)
    i += 1


# inserisco le medie dei primi giorni; parto dal diaciasettesimo giorno
medieGiorniPrecedentiFG = [0]  # contiene le medie dei valori dei 7 giorni precedenti
stdevGiorniPrecedentiFG = [0]
# inserisco anche per la lista di supporto i valori dei primi giorni
listaSupporto = []

print(indiciCambioGiorno)
i = 23
cnt = 0  # indica il giorno da cui parte la lista di supporto
nFG = []
for i in range(17, 1655):
    if i in indiciCambioGiorno:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(dataset["FG"][i])
        listaSupporto = aggiornaListaSupporto(indiciCambioGiorno[cnt], listaSupporto,
                                              indiciCambioGiorno[cnt + 7], "FG")
        # elimino gli zeri dalla lista
        listaSupporto = [i for i in listaSupporto if i != 0]
        nFG.append(len(listaSupporto))
        if len(listaSupporto) == 0:
            medieGiorniPrecedentiFG.append(0)
        else:
            medieGiorniPrecedentiFG.append(mean(listaSupporto))
        if len(listaSupporto) <= 1:
            stdevGiorniPrecedentiFG.append(0)
        else:
            stdevGiorniPrecedentiFG.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(dataset["FG"][i])
# aggiungo l'ultimo valore mancante alle deviazioni e alle medie
lista = [373.58, 397.45, 397.64, 379.84, 362.93, 385.35, 420.19, 435.65]
medieGiorniPrecedentiFG = medieGiorniPrecedentiFG + lista
lista = [147.65, 148.446, 176.02, 154.8, 147.57, 146.884, 155.225]
stdevGiorniPrecedentiFG = stdevGiorniPrecedentiFG + lista
stdevGiorniPrecedentiFG.append(162.64)
for i in range(0, 12):
    medieGiorniPrecedentiFG.pop(0)
    stdevGiorniPrecedentiFG.pop(0)



# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno

dict = {
    "giorno": np.arange(20, 115),
    "media": medieGiorniPrecedentiFG,
    "dev standard": stdevGiorniPrecedentiFG
}

datiFG = pd.DataFrame(dict)

nFG = nFG + [12, 11, 14, 13, 14, 17, 16, 17, 17]
for i in range(0, 12):
    nFG.pop(0)
print(len(nFG))

# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
# gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])
# gs.update(wspace=0.03, hspace=0.03)
# ax = plt.subplot(gs[1])
sns.lineplot(x="giorno", y="media", data=datiFG, marker=".", markersize=8, color="#1034a6")

# creo la linea verticale tratteggiata
# plt.plot([117] * 430, np.arange(150, 580), linestyle='dashed', color="black", linewidth=0.8)


icFG = []
cnt = 0
for i in datiFG["dev standard"]:
    icFG.append(1.96 * (i / sqrt(nFG[cnt])))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinFG = []
cnt = 0
for i in datiFG["dev standard"]:
    icMinFG.append(datiFG["media"][cnt] - icFG[cnt])
    cnt += 1
icMaxFG = []
cnt = 0
for i in datiFG["dev standard"]:
    icMaxFG.append(datiFG["media"][cnt] + icFG[cnt])
    cnt += 1

# disegno gli intervalli di confidenza sul grafico
# ne sto facendo diverse versioni per ogni tipo di dato. Per FG ne lascio commentata qualcuna, per gli altri c'è solo
# l'utlima che sto provando in ogni momento

# versione poligono, punto centrato nella riga, non nella fascia
"""
plt.margins(0.05)
for b in range(0, len(nFG)):
    i = nFG[b]
    if i <= 5:
        alpha = 0.02
    if i >= 30:
        alpha = 0.8
    if (i > 5) & (i < 30):
        alpha = (i - 5) / 25
    ax.fill_between(datiFG["giorno"][b:(b + 2)], icMinFG[b: (b + 2)], icMaxFG[b: (b + 2)], edgecolor="face", color="#1b6ca8", alpha=alpha)
"""
# versione poligono storto, ic centrato nella fascia
"""
plt.margins(0.05)
for b in range(0, len(nFG)):
    i = nFG[b]
    if i <= 5:
        alpha = 0.02
    if i >= 30:
        alpha = 0.8
    if (i > 5) & (i < 30):
        alpha = (i - 5) / 25
    ax.fill_between(datiFG["giorno"][b:(b + 2)] - 0.5, icMinFG[b: (b + 2)], icMaxFG[b: (b + 2)], edgecolor="face", color="#1b6ca8", alpha=alpha)
"""
# versione rettangolo ( poi eventualmente ci sarebbe anche la versione con la sola riga più grande)

plt.margins(0.05)
datiIC = creaRettangolo(datiFG["giorno"], icMinFG, icMaxFG, nFG)
giorni = datiIC[0]
minIC = datiIC[1]
maxIC = datiIC[2]
nFG = datiIC[3]

for b in range(0, len(nFG)):
    i = nFG[b]
    if i <= 5:
        alpha = 0.02
    if i >= 30:
        alpha = 0.8
    if (i > 5) & (i < 30):
        alpha = (i - 5) / 25
    ax.fill_between(giorni[b:(b + 2)], minIC[b: (b + 2)], maxIC[b: (b + 2)], edgecolor="face", color="#1b6ca8", alpha=alpha)
    # salto un punto, altrimenti rifaccio quello già fatto
    b += 1


ax.fill_between(datiFG["giorno"], icMinFG, icMaxFG, alpha=0.9, color="#1b6ca8",  label="95% confidence interval").set_visible(False)


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xlabel("Time (days)", fontsize=12)
plt.ylabel("previous 7 days mean FG", fontsize=12)

plt.xlim(7.8, 115.2)
plt.xticks([38, 69, 99], labels=[])
ax.tick_params(which='major', length=6, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([24, 55, 85], minor=True)
ax.tick_params(which='minor', length=4, color='black')
ax.text(8, 205, "F", size=10)
ax.text(23.6, 205, "M", size=10)
ax.text(54.7, 205, "A", size=10)
ax.text(84.6, 205, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda, ma devo fare un bbox per metterla in una posizione comoda

plt.legend(loc="upper right", ncol=1, fancybox=True,
           frameon=False)
plt.show()



# passo ad LY

# definisco una funzione di supporto
def aggiornaListaSupporto(counter, lista, c, nomeGrandezza):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(dataset[nomeGrandezza][r])
        r += 1
    return nuovaLista


# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiorno = []
i = 0
while i < len(dataset["DATA"]) - 1:
    if dataset["DATA"][i] != dataset["DATA"][i + 1]:
        indiciCambioGiorno.append(i)
    i += 1


# inserisco le medie dei primi giorni (dall' ottavo giorno al sedicesimo, esclusi il 9 e il 10
# perché non ci sono dati nei 7 giorni precedenti)
medieGiorniPrecedentiLY = [0]  # contiene le medie dei valori dei 7 giorni precedenti
stdevGiorniPrecedentiLY = [0]
# inserisco anche per la lista di supporto i valori dei primi giorni
listaSupporto = []


i = 12
cnt = 0  # indica il giorno da cui parte la lista di supporto
nLY = []
for i in range(12, 1656):
    if i in indiciCambioGiorno:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(dataset["LY"][i])
        listaSupporto = aggiornaListaSupporto(indiciCambioGiorno[cnt], listaSupporto,
                                              indiciCambioGiorno[cnt + 7], "LY")
        # elimino gli zeri dalla lista
        listaSupporto = [i for i in listaSupporto if i != 0]
        nLY.append(len(listaSupporto))
        medieGiorniPrecedentiLY.append(mean(listaSupporto))
        if len(listaSupporto) <= 1:
            stdevGiorniPrecedentiLY.append(0)
        else:
            stdevGiorniPrecedentiLY.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(dataset["LY"][i])
# aggiungo l'ultimo valore mancante alle deviazioni e alle medie
medieGiorniPrecedentiLY.remove(0)
medieGiorniPrecedentiLY.pop(0)
medieGiorniPrecedentiLY.pop(0)
lista = [16.61, 16.9, 17.73, 16.14, 16.82, 15.94]
medieGiorniPrecedentiLY = medieGiorniPrecedentiLY + lista
stdevGiorniPrecedentiLY.remove(0)
stdevGiorniPrecedentiLY.remove(0)
stdevGiorniPrecedentiLY.remove(0)
lista = [10.5, 10.99, 11.66, 11.39, 11.3, 9.92]
stdevGiorniPrecedentiLY = stdevGiorniPrecedentiLY + lista

for i in range(0, 9):
    medieGiorniPrecedentiLY.pop(0)
    stdevGiorniPrecedentiLY.pop(0)

print(len(medieGiorniPrecedentiLY))
print(len(stdevGiorniPrecedentiLY))
# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno

dict = {
    "giorno": np.arange(20, 116),
    "media": medieGiorniPrecedentiLY,
    "dev standard": stdevGiorniPrecedentiLY
}

datiLY = pd.DataFrame(dict)


# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
# gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])
# gs.update(wspace=0.03, hspace=0.03)
# ax = plt.subplot(gs[1])
sns.lineplot(x="giorno", y="media", data=datiLY, marker=".", markersize=8, color="#1034a6")

# creo la linea verticale tratteggiata
# plt.plot([117] * 430, np.arange(150, 580), linestyle='dashed', color="black", linewidth=0.8)

# dato che non mostra gli intervalli di confidenza li calcolo io
# ho già calcolato n, devo solo aggiungergli alcuni valori mancanti
# nLY.append(43)
nLY.pop(0)
nLY.pop(0)
nLY = nLY + [28, 28, 31, 32, 29, 30]
for i in range(0, 9):
    nLY.pop(0)

icLY = []
cnt = 0
for i in datiLY["dev standard"]:
    icLY.append(1.96 * i / sqrt(nLY[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinLY = []
cnt = 0
for i in datiLY["dev standard"]:
    icMinLY.append(datiLY["media"][cnt] - icLY[cnt])
    cnt += 1
icMaxLY = []
cnt = 0
for i in datiLY["dev standard"]:
    icMaxLY.append(datiLY["media"][cnt] + icLY[cnt])
    cnt += 1

# disegno gli intervalli di confidenza sul grafico
plt.margins(0.05)

datiIC = creaRettangolo(datiLY["giorno"], icMinLY, icMaxLY, nLY)
giorni = datiIC[0]
minIC = datiIC[1]
maxIC = datiIC[2]
nLY = datiIC[3]

for b in range(0, len(nFG)):
    i = nLY[b]
    if i <= 5:
        alpha = 0.02
    if i >= 30:
        alpha = 0.8
    if (i > 5) & (i < 30):
        alpha = (i - 5) / 25
    ax.fill_between(giorni[b:(b + 2)], minIC[b: (b + 2)], maxIC[b: (b + 2)], edgecolor="face", color="#1b6ca8", alpha=alpha)
    # salto un punto, altrimenti rifarei quello già fatto
    b += 1

ax.fill_between(datiLY["giorno"], icMinLY, icMaxLY, alpha=0.9, color="#1b6ca8",  label="95% confidence interval").set_visible(False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xlabel("Time (days)", fontsize=12)
plt.ylabel("previous 7 days mean LY", fontsize=12)
plt.xlim(7.8, 115.2)
plt.xticks([38, 69, 99], labels=[])
ax.tick_params(which='major', length=6, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([24, 55, 85], minor=True)
ax.tick_params(which='minor', length=10, color='black')
ax.text(8, 9, "F", size=10)
ax.text(23.6, 9, "M", size=10)
ax.text(54.7, 9, "A", size=10)
ax.text(84.6, 9, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda, ma devo fare un bbox per metterla in una posizione comoda

plt.legend(loc="upper right", ncol=1, fancybox=True,
           frameon=False)
plt.show()



# passo a WBC

# definisco una funzione di supporto
def aggiornaListaSupporto(counter, lista, c, nomeGrandezza):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(dataset[nomeGrandezza][r])
        r += 1
    return nuovaLista


# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiorno = []
i = 0
while i < len(dataset["DATA"]) - 1:
    if dataset["DATA"][i] != dataset["DATA"][i + 1]:
        indiciCambioGiorno.append(i)
    i += 1


# inserisco le medie dei primi giorni (dall' ottavo giorno al sedicesimo, esclusi il 9 e il 10
# perché non ci sono dati nei 7 giorni precedenti)
medieGiorniPrecedentiWBC = [9.6, 6.8, 6.8, 6.8, 6.8, 6.8, 6.8]  # contiene le medie dei valori dei 7 giorni precedenti
stdevGiorniPrecedentiWBC = [0, 0, 0, 0, 0, 0, 0]
# inserisco anche per la lista di supporto i valori dei primi giorni
listaSupporto = [6.8, 0, 0, 0, 0, 0, 7.7]


i = 16
cnt = 8  # indica il giorno da cui parte la lista di supporto
nWBC = [1, 1, 1, 1, 1, 1, 1]
for i in range(16, 1655):
    if i in indiciCambioGiorno:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(dataset["WBC"][i])
        listaSupporto = aggiornaListaSupporto(indiciCambioGiorno[cnt], listaSupporto,
                                              indiciCambioGiorno[cnt + 7], "WBC")
        # elimino gli zeri dalla lista
        listaSupporto = [i for i in listaSupporto if i != 0]
        nWBC.append(len(listaSupporto))
        medieGiorniPrecedentiWBC.append(mean(listaSupporto))
        if len(listaSupporto) <= 1:
            stdevGiorniPrecedentiWBC.append(0)
        else:
            stdevGiorniPrecedentiWBC.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(dataset["WBC"][i])
# aggiungo l'ultimo valore mancante alle deviazioni e alle medie
medieGiorniPrecedentiWBC.append(11.22)
stdevGiorniPrecedentiWBC.append(11.2)
for i in range(0, 10):
    medieGiorniPrecedentiWBC.pop(0)
    stdevGiorniPrecedentiWBC.pop(0)
print(len(medieGiorniPrecedentiWBC))
print(len(stdevGiorniPrecedentiWBC))



# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno

dict = {
    "giorno": np.arange(20, 116),
    "media": medieGiorniPrecedentiWBC,
    "dev standard": stdevGiorniPrecedentiWBC
}

datiWBC = pd.DataFrame(dict)



# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
# gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])
# gs.update(wspace=0.03, hspace=0.03)
# ax = plt.subplot(gs[1])
sns.lineplot(x="giorno", y="media", data=datiWBC, marker=".", markersize=8, color="#1034a6")

# creo la linea verticale tratteggiata
# plt.plot([117] * 430, np.arange(150, 580), linestyle='dashed', color="black", linewidth=0.8)

# dato che non mostra gli intervalli di confidenza li calcolo io
# ho già calcolato n, devo solo aggiungergli alcuni valori mancanti
nWBC.append(46)
for i in range(0, 10):
    nWBC.pop(0)

print(len(nWBC))


icWBC = []
cnt = 0
for i in datiWBC["dev standard"]:
    icWBC.append(1.96 * i / sqrt(nWBC[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinWBC = []
cnt = 0
for i in datiWBC["dev standard"]:
    icMinWBC.append(datiWBC["media"][cnt] - icWBC[cnt])
    cnt += 1
icMaxWBC = []
cnt = 0
for i in datiWBC["dev standard"]:
    icMaxWBC.append(datiWBC["media"][cnt] + icWBC[cnt])
    cnt += 1

# disegno gli intervalli di confidenza sul grafico
plt.margins(0.05)

datiIC = creaRettangolo(datiWBC["giorno"], icMinWBC, icMaxWBC, nWBC)
giorni = datiIC[0]
minIC = datiIC[1]
maxIC = datiIC[2]
nWBC = datiIC[3]

for b in range(0, len(nWBC)):
    i = nWBC[b]
    if i <= 5:
        alpha = 0.02
    if i >= 30:
        alpha = 0.8
    if (i > 5) & (i < 30):
        alpha = (i - 5) / 25
    ax.fill_between(giorni[b:(b + 2)], minIC[b: (b + 2)], maxIC[b: (b + 2)], edgecolor="face", color="#1b6ca8", alpha=alpha)
    # salto un punto, altrimenti rifaccio quello già fatto
    b += 1

ax.fill_between(datiWBC["giorno"], icMinWBC, icMaxWBC, alpha=0.9, color="#1b6ca8",  label="95% confidence interval").set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xlabel("Time (days)", fontsize=12)
plt.ylabel("previous 7 days mean WBC", fontsize=12)

plt.xlim(7.8, 115.2)
plt.xticks([38, 69, 99], labels=[])
ax.tick_params(which='major', length=6, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([24, 55, 85], minor=True)
ax.tick_params(which='minor', length=10, color='black')
ax.text(8, 5, "F", size=10)
ax.text(24.6, 5, "M", size=10)
ax.text(54.7, 5, "A", size=10)
ax.text(84.6, 5, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda, ma devo fare un bbox per metterla in una posizione comoda

plt.legend(loc="upper right", ncol=1, fancybox=True,
           frameon=False)
plt.show()



# passo a AST

# definisco una funzione di supporto
def aggiornaListaSupporto(counter, lista, c, nomeGrandezza):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(dataset[nomeGrandezza][r])
        r += 1
    return nuovaLista


# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiorno = []
i = 0
while i < len(dataset["DATA"]) - 1:
    if dataset["DATA"][i] != dataset["DATA"][i + 1]:
        indiciCambioGiorno.append(i)
    i += 1


# inserisco le medie dei primi giorni (dall' ottavo giorno al sedicesimo, esclusi il 9 e il 10
# perché non ci sono dati nei 7 giorni precedenti)
medieGiorniPrecedentiAST = [0]  # contiene le medie dei valori dei 7 giorni precedenti
stdevGiorniPrecedentiAST = [0]
# inserisco anche per la lista di supporto i valori dei primi giorni
listaSupporto = []


i = 12
cnt = 0  # indica il giorno da cui parte la lista di supporto
nAST = []
for i in range(12, 1656):
    if i in indiciCambioGiorno:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(dataset["AST"][i])
        listaSupporto = aggiornaListaSupporto(indiciCambioGiorno[cnt], listaSupporto,
                                              indiciCambioGiorno[cnt + 7], "AST")
        # elimino gli zeri dalla lista
        listaSupporto = [i for i in listaSupporto if i != 0]
        nAST.append(len(listaSupporto))
        medieGiorniPrecedentiAST.append(mean(listaSupporto))
        if len(listaSupporto) <= 1:
            stdevGiorniPrecedentiAST.append(0)
        else:
            stdevGiorniPrecedentiAST.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(dataset["AST"][i])
# aggiungo l'ultimo valore mancante alle deviazioni e alle medie
medieGiorniPrecedentiAST.remove(0)
medieGiorniPrecedentiAST.pop(0)
medieGiorniPrecedentiAST.pop(0)
lista = [30.90, 28.58, 36.0, 39.224, 36.36, 36.22]
medieGiorniPrecedentiAST = medieGiorniPrecedentiAST + lista
stdevGiorniPrecedentiAST.remove(0)
stdevGiorniPrecedentiAST.remove(0)
stdevGiorniPrecedentiAST.remove(0)
lista = [22.42, 16.40, 38.62, 40.45, 40.92, 39.95]
stdevGiorniPrecedentiAST = stdevGiorniPrecedentiAST + lista
for i in range(0, 9):
    medieGiorniPrecedentiAST.pop(0)
    stdevGiorniPrecedentiAST.pop(0)

print(len(medieGiorniPrecedentiAST))
print(len(stdevGiorniPrecedentiAST))


# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno

dict = {
    "giorno": np.arange(20, 116),
    "media": medieGiorniPrecedentiAST,
    "dev standard": stdevGiorniPrecedentiAST
}

datiAST = pd.DataFrame(dict)


# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
# gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])
# gs.update(wspace=0.03, hspace=0.03)
# ax = plt.subplot(gs[1])
sns.lineplot(x="giorno", y="media", data=datiAST, marker=".", markersize=8, color="#1034a6")

# creo la linea verticale tratteggiata
# plt.plot([117] * 430, np.arange(150, 580), linestyle='dashed', color="black", linewidth=0.8)

# dato che non mostra gli intervalli di confidenza li calcolo io
# ho già calcolato n, devo solo aggiungergli alcuni valori mancanti
nAST.pop(0)
nAST.pop(0)
nAST = nAST + [48, 47, 50, 48, 43, 45]
for i in range(0, 9):
    nAST.pop(0)


icAST = []
cnt = 0
for i in datiAST["dev standard"]:
    icAST.append(1.96 * i / sqrt(nAST[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinAST = []
cnt = 0
for i in datiAST["dev standard"]:
    icMinAST.append(datiAST["media"][cnt] - icAST[cnt])
    cnt += 1
icMaxAST = []
cnt = 0
for i in datiAST["dev standard"]:
    icMaxAST.append(datiAST["media"][cnt] + icAST[cnt])
    cnt += 1

# disegno gli intervalli di confidenza sul grafico
plt.margins(0.05)

datiIC = creaRettangolo(datiAST["giorno"], icMinAST, icMaxAST, nAST)
giorni = datiIC[0]
minIC = datiIC[1]
maxIC = datiIC[2]
nAST = datiIC[3]

for b in range(0, len(nAST)):
    i = nAST[b]
    if i <= 5:
        alpha = 0.02
    if i >= 30:
        alpha = 0.8
    if (i > 5) & (i < 30):
        alpha = (i - 5) / 25
    ax.fill_between(giorni[b:(b + 2)], minIC[b: (b + 2)], maxIC[b: (b + 2)], edgecolor="face", color="#1b6ca8", alpha=alpha)
    # salto un punto, altrimenti rifaccio quello già fatto
    b += 1

ax.fill_between(datiAST["giorno"], icMinAST, icMaxAST, alpha=0.9, color="#1b6ca8",  label="95% confidence interval").set_visible(False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xlabel("Time (days)", fontsize=12)
plt.ylabel("previous 7 days mean AST", fontsize=12)

plt.xlim(7.8, 115.2)
plt.xticks([38, 69, 99], labels=[])
ax.tick_params(which='major', length=6, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([24, 55, 85], minor=True)
ax.tick_params(which='minor', length=10, color='black')
ax.text(8, 12, "F", size=10)
ax.text(23.6, 12, "M", size=10)
ax.text(54.7, 12, "A", size=10)
ax.text(84.6, 12, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda, ma devo fare un bbox per metterla in una posizione comoda

plt.legend(loc="upper right", ncol=1, fancybox=True,
           frameon=False)
plt.show()


# passo a LDH
# definisco una funzione di supporto
def aggiornaListaSupporto(counter, lista, c, nomeGrandezza):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(dataset[nomeGrandezza][r])
        r += 1
    return nuovaLista


# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiorno = []
i = 0
while i < len(dataset["DATA"]) - 1:
    if dataset["DATA"][i] != dataset["DATA"][i + 1]:
        indiciCambioGiorno.append(i)
    i += 1


# inserisco le medie dei primi giorni (dall' ottavo giorno al sedicesimo, esclusi il 9 e il 10
# perché non ci sono dati nei 7 giorni precedenti)
medieGiorniPrecedentiLDH = [0]  # contiene le medie dei valori dei 7 giorni precedenti
stdevGiorniPrecedentiLDH = [0]
# inserisco anche per la lista di supporto i valori dei primi giorni
listaSupporto = []


i = 12
cnt = 0  # indica il giorno da cui parte la lista di supporto
nLDH = []
for i in range(12, 1656):
    if i in indiciCambioGiorno:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(dataset["LDH"][i])
        listaSupporto = aggiornaListaSupporto(indiciCambioGiorno[cnt], listaSupporto,
                                              indiciCambioGiorno[cnt + 7], "LDH")
        # elimino gli zeri dalla lista
        listaSupporto = [i for i in listaSupporto if i != 0]
        nLDH.append(len(listaSupporto))
        medieGiorniPrecedentiLDH.append(mean(listaSupporto))
        if len(listaSupporto) <= 1:
            stdevGiorniPrecedentiLDH.append(0)
        else:
            stdevGiorniPrecedentiLDH.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(dataset["LDH"][i])
# aggiungo l'ultimo valore mancante alle deviazioni e alle medie
medieGiorniPrecedentiLDH.remove(0)
medieGiorniPrecedentiLDH.pop(0)
medieGiorniPrecedentiLDH.pop(0)
lista = [246.97, 247.32, 247.17, 258.25, 258.5, 259.7]
medieGiorniPrecedentiLDH = medieGiorniPrecedentiLDH + lista
stdevGiorniPrecedentiLDH.remove(0)
stdevGiorniPrecedentiLDH.remove(0)
stdevGiorniPrecedentiLDH.remove(0)
lista = [71.23, 80.1, 85.22, 91.63, 94.31, 91.65]
stdevGiorniPrecedentiLDH = stdevGiorniPrecedentiLDH + lista

for i in range(0, 9):
    medieGiorniPrecedentiLDH.pop(0)
    stdevGiorniPrecedentiLDH.pop(0)

print(len(medieGiorniPrecedentiLDH))
print(len(stdevGiorniPrecedentiLDH))


# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno

dict = {
    "giorno": np.arange(20, 116),
    "media": medieGiorniPrecedentiLDH,
    "dev standard": stdevGiorniPrecedentiLDH
}

datiLDH = pd.DataFrame(dict)


# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
# gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])
# gs.update(wspace=0.03, hspace=0.03)
# ax = plt.subplot(gs[1])
sns.lineplot(x="giorno", y="media", data=datiLDH, marker=".", markersize=8, color="#1034a6")

# creo la linea verticale tratteggiata
# plt.plot([117] * 430, np.arange(150, 580), linestyle='dashed', color="black", linewidth=0.8)

# dato che non mostra gli intervalli di confidenza li calcolo io
# ho già calcolato n, devo solo aggiungergli alcuni valori mancanti
nLDH.pop(0)
nLDH.pop(0)
nLDH = nLDH + [48, 47, 50, 48, 43, 45]
for i in range(0, 9):
    nLDH.pop(0)
print(len(nLDH))

icLDH = []
cnt = 0
for i in datiLDH["dev standard"]:
    icLDH.append(1.96 * i / sqrt(nLDH[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinLDH = []
cnt = 0
for i in datiLDH["dev standard"]:
    icMinLDH.append(datiLDH["media"][cnt] - icLDH[cnt])
    cnt += 1
icMaxLDH = []
cnt = 0
for i in datiLDH["dev standard"]:
    icMaxLDH.append(datiLDH["media"][cnt] + icLDH[cnt])
    cnt += 1

# disegno gli intervalli di confidenza sul grafico
# disegno gli intervalli di confidenza sul grafico
plt.margins(0.05)
datiIC = creaRettangolo(datiLDH["giorno"], icMinLDH, icMaxLDH, nLDH)
giorni = datiIC[0]
minIC = datiIC[1]
maxIC = datiIC[2]
nLDH = datiIC[3]

for b in range(0, len(nLDH)):
    i = nLDH[b]
    if i <= 5:
        alpha = 0.02
    if i >= 30:
        alpha = 0.8
    if (i > 5) & (i < 30):
        alpha = (i - 5) / 25
    ax.fill_between(giorni[b:(b + 2)], minIC[b: (b + 2)], maxIC[b: (b + 2)], edgecolor="face", color="#1b6ca8", alpha=alpha)
    # salto un punto, altrimenti rifaccio quello già fatto
    b += 1

ax.fill_between(datiLDH["giorno"], icMinLDH, icMaxLDH, alpha=0.9, color="#1b6ca8",  label="95% confidence interval").set_visible(False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xlabel("Time (days)", fontsize=12)
plt.ylabel("previous 7 days mean LDH", fontsize=12)

plt.xlim(7.8, 115.2)
plt.xticks([38, 69, 99], labels=[])
ax.tick_params(which='major', length=6, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([24, 55, 85], minor=True)
ax.tick_params(which='minor', length=10, color='black')
ax.text(8, 110, "F", size=10)
ax.text(23.6, 110, "M", size=10)
ax.text(54.7, 110, "A", size=10)
ax.text(84.6, 110, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda, ma devo fare un bbox per metterla in una posizione comoda

plt.legend(loc="upper right", ncol=1, fancybox=True,
           frameon=False)
plt.show()


# passo al CA
# definisco una funzione di supporto
def aggiornaListaSupporto(counter, lista, c, nomeGrandezza):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(dataset[nomeGrandezza][r])
        r += 1
    return nuovaLista


# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiorno = []
i = 0
while i < len(dataset["DATA"]) - 1:
    if dataset["DATA"][i] != dataset["DATA"][i + 1]:
        indiciCambioGiorno.append(i)
    i += 1


# inserisco le medie dei primi giorni (dall' ottavo giorno al sedicesimo, esclusi il 9 e il 10
# perché non ci sono dati nei 7 giorni precedenti)
medieGiorniPrecedentiCA = [2.52, 2.44, 2.44, 2.44, 2.44, 2.44, 2.44]  # contiene le medie dei valori dei 7 giorni precedenti
stdevGiorniPrecedentiCA = [0, 0, 0, 0, 0, 0, 0]
# inserisco anche per la lista di supporto i valori dei primi giorni
listaSupporto = [2.44, 0, 0, 0, 0, 0, 2.24]


i = 16
cnt = 8  # indica il giorno da cui parte la lista di supporto
nCA = [1, 1, 1, 1, 1, 1, 1]
for i in range(16, 1655):
    if i in indiciCambioGiorno:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(dataset["CA"][i])
        listaSupporto = aggiornaListaSupporto(indiciCambioGiorno[cnt], listaSupporto,
                                              indiciCambioGiorno[cnt + 7], "CA")
        # elimino gli zeri dalla lista
        listaSupporto = [i for i in listaSupporto if i != 0]
        nCA.append(len(listaSupporto))
        medieGiorniPrecedentiCA.append(mean(listaSupporto))
        if len(listaSupporto) <= 1:
            stdevGiorniPrecedentiCA.append(0)
        else:
            stdevGiorniPrecedentiCA.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(dataset["CA"][i])
# aggiungo l'ultimo valore mancante alle deviazioni e alle medie
medieGiorniPrecedentiCA.append(2.24)
stdevGiorniPrecedentiCA.append(0.197)
for i in range(0, 10):
    medieGiorniPrecedentiCA.pop(0)
    stdevGiorniPrecedentiCA.pop(0)

print(len(medieGiorniPrecedentiCA))
print(len(stdevGiorniPrecedentiCA))



# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno

dict = {
    "giorno": np.arange(20, 116),
    "media": medieGiorniPrecedentiCA,
    "dev standard": stdevGiorniPrecedentiCA
}

datiCA = pd.DataFrame(dict)

# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
# gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])
# gs.update(wspace=0.03, hspace=0.03)
# ax = plt.subplot(gs[1])
sns.lineplot(x="giorno", y="media", data=datiCA, marker=".", markersize=8, color="#1034a6")

# creo la linea verticale tratteggiata
# plt.plot([117] * 430, np.arange(150, 580), linestyle='dashed', color="black", linewidth=0.8)

# dato che non mostra gli intervalli di confidenza li calcolo io
# ho già calcolato n, devo solo aggiungergli alcuni valori mancanti
nCA.append(45)
for i in range(0, 10):
    nCA.pop(0)
print(len(nCA))

icCA = []
cnt = 0
for i in datiCA["dev standard"]:
    icCA.append(1.96 * i / sqrt(nCA[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinCA = []
cnt = 0
for i in datiCA["dev standard"]:
    icMinCA.append(datiCA["media"][cnt] - icCA[cnt])
    cnt += 1
icMaxCA = []
cnt = 0
for i in datiCA["dev standard"]:
    icMaxCA.append(datiCA["media"][cnt] + icCA[cnt])
    cnt += 1

plt.margins(0.05)
datiIC = creaRettangolo(datiCA["giorno"], icMinCA, icMaxCA, nCA)
giorni = datiIC[0]
minIC = datiIC[1]
maxIC = datiIC[2]
nCA = datiIC[3]

for b in range(0, len(nCA)):
    i = nCA[b]
    if i <= 5:
        alpha = 0.02
    if i >= 30:
        alpha = 0.8
    if (i > 5) & (i < 30):
        alpha = (i - 5) / 25
    ax.fill_between(giorni[b:(b + 2)], minIC[b: (b + 2)], maxIC[b: (b + 2)], edgecolor="face", color="#1b6ca8", alpha=alpha)
    # salto un punto, altrimenti rifaccio quello già fatto
    b += 1

ax.fill_between(datiCA["giorno"], icMinCA, icMaxCA, alpha=0.9, color="#1b6ca8",  label="95% confidence interval").set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xlabel("Time (days)", fontsize=12)
plt.ylabel("previous 7 days mean CA", fontsize=12)

plt.xlim(7.8, 115.2)
plt.xticks([38, 69, 99], labels=[])
ax.tick_params(which='major', length=6, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([24, 55, 85], minor=True)
ax.tick_params(which='minor', length=10, color='black')
ax.text(8, 2.07, "F", size=10)
ax.text(23.6, 2.07, "M", size=10)
ax.text(54.7, 2.07, "A", size=10)
ax.text(84.6, 2.07, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda, ma devo fare un bbox per metterla in una posizione comoda

plt.legend(loc="upper right", ncol=1, fancybox=True,
           frameon=False)
plt.show()




# faccio il grafico dell'età minima, massima e media
# devo creare tre liste, una con l'età minima, una con quella media e una con quella massima

# definisco una funzione di supporto
def aggiornaListaSupporto(counter, lista, c):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(dataset["Age"][r])
        r += 1
    return nuovaLista


# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiorno = []
i = 0
while i < len(dataset["DATA"]) - 1:
    if dataset["DATA"][i] != dataset["DATA"][i + 1]:
        indiciCambioGiorno.append(i)
    i += 1


# inserisco le medie dei primi giorni (dall' ottavo giorno al sedicesimo, esclusi il 9 e il 10
# perché non ci sono dati nei 7 giorni precedenti)
etàMinime = [1, 49, 49, 49, 49, 49, 49]  # contiene le medie dei valori dei 7 giorni precedenti
etàMassime = [1, 49, 49, 49, 49, 49, 49]
etàMedie = [1, 49, 49, 49, 49, 49, 49]
# inserisco anche per la lista di supporto i valori dei primi giorni
listaSupporto = [49, 0, 0, 0, 0, 0, 35]

i = 16
cnt = 8  # indica il giorno da cui parte la lista di supporto
for i in range(16, 1655):
    if i in indiciCambioGiorno:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(dataset["Age"][i])
        listaSupporto = aggiornaListaSupporto(indiciCambioGiorno[cnt], listaSupporto,
                                              indiciCambioGiorno[cnt + 7])
        # elimino gli zeri dalla lista
        listaSupporto = [i for i in listaSupporto if i != 0]
        etàMinime.append(min(listaSupporto))
        etàMassime.append(max(listaSupporto))
        etàMedie.append(mean(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(dataset["Age"][i])
# aggiungo l'utlimo valore mancante alle tre liste
etàMinime.append(11)
etàMassime.append(92)
etàMedie.append(63.65)
for i in range(0, 10):
    etàMinime.pop(0)
    etàMassime.pop(0)
    etàMedie.pop(0)
print(len(etàMedie))
# creo un dataframe con tutti i dati
dict = {
    "giorno": np.arange(20, 116),
    "min": etàMinime,
    "max": etàMassime,
    "media": etàMedie
}
datiEtà = pd.DataFrame(dict)
# inzio a disegnare il grafico
matplotlib.rcParams['lines.markeredgewidth'] = 0
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x="giorno", y="min", data=datiEtà, marker=".", markersize=14, color="royalblue", label="min age")
sns.lineplot(x="giorno", y="media", data=datiEtà, marker=".", markersize=14, color="#107dac", label="mean age")
sns.lineplot(x="giorno", y="max", data=datiEtà, marker=".", markersize=14, color="#005073", label="max age")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.grid(axis="y", linewidth=1, alpha=0.7)
plt.xlabel("Time (days)", fontsize=12)
plt.ylabel("previous 7 days Age (years)", fontsize=12)

plt.xlim(7.8, 115.2)
plt.ylim(0, 110)
plt.xticks([38, 69, 99], labels=[])
ax.tick_params(which='major', length=6, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([24, 55, 85], minor=True)
ax.tick_params(which='minor', length=10, color='black')
ax.text(8, -8, "F", size=10)
ax.text(23.6, -8, "M", size=10)
ax.text(54.7, -8, "A", size=10)
ax.text(84.6, -8, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda, ma devo fare un bbox per metterla in una posizione comoda

plt.legend(loc="upper left", ncol=1, fancybox=True,
           frameon=False)
plt.show()



# passo al grafico del genere
# definisco una funzione di supporto
def aggiornaListaSupporto(counter, lista, c):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(dataset["Sex"][r])
        r += 1
    return nuovaLista


# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiorno = []
i = 0
while i < len(dataset["DATA"]) - 1:
    if dataset["DATA"][i] != dataset["DATA"][i + 1]:
        indiciCambioGiorno.append(i)
    i += 1

# definisco una funzione per calcolarne la percentuale
def calcolaPercentualiZero(lista):
    cnt = 0
    for i in lista:
        if i == 0:
            cnt += 1
    return (cnt / len(lista)) * 100


percentualiZero = [100, 0, 0, 0, 0, 0, 0]
percentualiUno = [0, 100, 100, 100, 100, 100, 100]
# inserisco anche per la lista di supporto i valori dei primi giorni
listaSupporto = [1, -1, -1, -1, -1, -1, 0]

i = 16
cnt = 8  # indica il giorno da cui parte la lista di supporto
for i in range(16, 1655):
    if i in indiciCambioGiorno:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(dataset["Sex"][i])
        listaSupporto = aggiornaListaSupporto(indiciCambioGiorno[cnt], listaSupporto,
                                              indiciCambioGiorno[cnt + 7])
        # elimino gli zeri dalla lista
        listaSupporto = [i for i in listaSupporto if i != -1]
        print(listaSupporto)
        perZero = calcolaPercentualiZero(listaSupporto)
        percentualiZero.append(perZero)
        percentualiUno.append(abs(100 - perZero))
        cnt += 1
    else:
        listaSupporto.append(dataset["Sex"][i])
percentualiZero.append(38.8)
percentualiUno.append(62.2)
print(percentualiZero)
print(percentualiUno)



# creo un dataframe con tutti i dati
dict = {
    "giorno": np.arange(20, 116),
    "zeri": percentualiZero,
    "uni": percentualiUno,
}
for i in range(0, 10):
    percentualiUno.pop(0)
    percentualiZero.pop(0)

datiGenere = pd.DataFrame(dict)
# inzio a disegnare il grafico
matplotlib.rcParams['lines.markeredgewidth'] = 0
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x="giorno", y="zeri", data=datiGenere, marker=".", markersize=14, color="#005b96", label="male", alpha=1)
sns.lineplot(x="giorno", y="uni", data=datiGenere, marker=".", markersize=14, color="lightslategrey", label="female", alpha=1)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.grid(axis="y", linewidth=1, alpha=0.7)
plt.xlabel("Time (days)", fontsize=12)
plt.ylabel("previous 7 days percentage per sex", fontsize=12)

plt.xlim(7.8, 115.2)
plt.ylim(0, 110)
plt.xticks([38, 69, 99], labels=[])
ax.tick_params(which='major', length=6, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([24, 55, 85], minor=True)
ax.tick_params(which='minor', length=10, color='black')
ax.text(8, -8, "F", size=10)
ax.text(23.6, -8, "M", size=10)
ax.text(54.7, -8, "A", size=10)
ax.text(84.6, -8, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda, ma devo fare un bbox per metterla in una posizione comoda

plt.legend(loc="upper left", ncol=1, fancybox=True,
           frameon=False)
plt.show()



# passo al grafico dei tamponi
# definisco una funzione di supporto
def aggiornaListaSupporto(counter, lista, c, nomeGrandezza):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(dataset[nomeGrandezza][r])
        r += 1
    return nuovaLista


# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiorno = []
i = 0
while i < len(dataset["DATA"]) - 1:
    if dataset["DATA"][i] != dataset["DATA"][i + 1]:
        indiciCambioGiorno.append(i)
    i += 1

# definisco una funzione per calcolare la percentuale di tamponi positivi dei 7 giorni precedenti
def calcolaPercentualiTamponi(lista):
    cnt = 0
    for i in lista:
        if i == 1:
            cnt += 1
    return (cnt / (len(lista))) * 100

# inserisco le medie dei primi giorni (dall' ottavo giorno al sedicesimo, esclusi il 9 e il 10
# perché non ci sono dati nei 7 giorni precedenti)
percentualiTamponiPositivi = [0, 0, 0, 0, 0, 0, 0]  # contiene le medie dei valori dei 7 giorni precedenti


# inserisco anche per la lista di supporto i valori dei primi giorni
listaSupporto = [0, -1, -1, -1, -1, -1, 0]


i = 16
cnt = 8  # indica il giorno da cui parte la lista di supporto
nTamponi = [1, 1, 1, 1, 1, 1, 1]
for i in range(16, 1655):
    if i in indiciCambioGiorno:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(dataset["TAMPONI"][i])
        listaSupporto = aggiornaListaSupporto(indiciCambioGiorno[cnt], listaSupporto,
                                              indiciCambioGiorno[cnt + 7], "TAMPONI")
        # elimino gli zeri dalla lista
        listaSupporto = [i for i in listaSupporto if i != -1]
        nTamponi.append(len(listaSupporto))
        percentualiTamponiPositivi.append(calcolaPercentualiTamponi(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(dataset["TAMPONI"][i])

percentualiTamponiPositivi.append(4.1)



# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno
for i in range(0, 10):
    percentualiTamponiPositivi.pop(0)


dict = {
    "giorno": np.arange(20, 116),
    "percentuali positivi": percentualiTamponiPositivi
}

datiTamponi = pd.DataFrame(dict)


# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))

sns.lineplot(x="giorno", y="percentuali positivi", data=datiTamponi, marker=".", markersize=14, color="#1034a6")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xlabel("Time (days)", fontsize=12)
plt.ylabel("previous 7 days positive percentage ", fontsize=12)

plt.xlim(8, 115.2)
plt.ylim(0, 100)
plt.xticks([38, 69, 99], labels=[])
plt.grid(axis="y", linewidth=0.8, alpha=0.8)
ax.tick_params(which='major', length=6, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([24, 55, 85], minor=True)
ax.tick_params(which='minor', length=10, color='black')
ax.text(8, -8, "F", size=10)
ax.text(23.6, -8, "M", size=10)
ax.text(54.7, -8, "A", size=10)
ax.text(84.6, -8, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)
plt.show()

