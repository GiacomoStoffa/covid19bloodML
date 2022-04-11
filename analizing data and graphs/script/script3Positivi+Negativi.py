import statistics
import pandas as pd
from pylab import *

pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import seaborn as sns
from scipy.stats import iqr

datasetPos = pd.read_excel("selected_dataset_diagnostico_COVID_POSITIVI.xls")
datasetNeg = pd.read_excel("selected_dataset_diagnostico_COVID_NEGATIVI.xls")

data = []
for i in datasetPos["DATA"]:
    data.append((i - datetime.datetime(2020, 2, 7)).days + 1)
datasetPos["DATA"] = data

data = []
for i in datasetNeg["DATA"]:
    data.append((i - datetime.datetime(2020, 2, 7)).days + 1)
datasetNeg["DATA"] = data

# sostituisco i nan con gli 0 (che poi non considero nel grafico)
datasetPos['PLT1'] = datasetPos['PLT1'].fillna(0)
datasetPos["PCR"] = datasetPos['PCR'].fillna(0)
datasetPos["FG"] = datasetPos['FG'].fillna(0)
datasetPos["LY"] = datasetPos['LY'].fillna(0)
datasetPos["WBC"] = datasetPos['WBC'].fillna(0)
datasetPos["AST"] = datasetPos['AST'].fillna(0)
datasetPos["LDH"] = datasetPos['LDH'].fillna(0)
datasetPos["CA"] = datasetPos['CA'].fillna(0)
datasetNeg['PLT1'] = datasetNeg['PLT1'].fillna(0)
datasetNeg["PCR"] = datasetNeg['PCR'].fillna(0)
datasetNeg["FG"] = datasetNeg['FG'].fillna(0)
datasetNeg["LY"] = datasetNeg['LY'].fillna(0)
datasetNeg["WBC"] = datasetNeg['WBC'].fillna(0)
datasetNeg["AST"] = datasetNeg['AST'].fillna(0)
datasetNeg["LDH"] = datasetNeg['LDH'].fillna(0)
datasetNeg["CA"] = datasetNeg['CA'].fillna(0)


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
    a = contains(datasetPos["DATA"], i)
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
        while (indice < (len(datasetPos["DATA"]) - 1)) and (
                datasetPos["DATA"][indice] == datasetPos["DATA"][indice + 1]):
            giorni.append(datasetPos["DATA"][indice])
            valoriPLT1.append(datasetPos["PLT1"][indice])
            valoriPCR.append(datasetPos["PCR"][indice])
            valoriFG.append(datasetPos["FG"][indice])
            valoriLY.append(datasetPos["LY"][indice])
            valoriWBC.append(datasetPos["WBC"][indice])
            valoriAST.append(datasetPos["AST"][indice])
            valoriLDH.append(datasetPos["LDH"][indice])
            valoriCA.append(datasetPos["CA"][indice])
            tamponi.append(datasetPos["TAMPONI"][indice])
            sesso.append(datasetPos["Sex"][indice])
            età.append(datasetPos["Age"][indice])
            indice += 1
        giorni.append(datasetPos["DATA"][indice])
        valoriPLT1.append(datasetPos["PLT1"][indice])
        valoriPCR.append(datasetPos["PCR"][indice])
        valoriFG.append(datasetPos["FG"][indice])
        valoriLY.append(datasetPos["LY"][indice])
        valoriWBC.append(datasetPos["WBC"][indice])
        valoriAST.append(datasetPos["AST"][indice])
        valoriLDH.append(datasetPos["LDH"][indice])
        valoriCA.append(datasetPos["CA"][indice])
        tamponi.append(datasetPos["TAMPONI"][indice])
        sesso.append(datasetPos["Sex"][indice])
        età.append(datasetPos["Age"][indice])
    cnt += 1

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
datasetPos = pd.DataFrame(dict)

# faccio la stessa per i negativi
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
    a = contains(datasetNeg["DATA"], i)
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
        while (indice < (len(datasetNeg["DATA"]) - 1)) and (
                datasetNeg["DATA"][indice] == datasetNeg["DATA"][indice + 1]):
            giorni.append(datasetNeg["DATA"][indice])
            valoriPLT1.append(datasetNeg["PLT1"][indice])
            valoriPCR.append(datasetNeg["PCR"][indice])
            valoriFG.append(datasetNeg["FG"][indice])
            valoriLY.append(datasetNeg["LY"][indice])
            valoriWBC.append(datasetNeg["WBC"][indice])
            valoriAST.append(datasetNeg["AST"][indice])
            valoriLDH.append(datasetNeg["LDH"][indice])
            valoriCA.append(datasetNeg["CA"][indice])
            tamponi.append(datasetNeg["TAMPONI"][indice])
            sesso.append(datasetNeg["Sex"][indice])
            età.append(datasetNeg["Age"][indice])
            indice += 1
        giorni.append(datasetNeg["DATA"][indice])
        valoriPLT1.append(datasetNeg["PLT1"][indice])
        valoriPCR.append(datasetNeg["PCR"][indice])
        valoriFG.append(datasetNeg["FG"][indice])
        valoriLY.append(datasetNeg["LY"][indice])
        valoriWBC.append(datasetNeg["WBC"][indice])
        valoriAST.append(datasetNeg["AST"][indice])
        valoriLDH.append(datasetNeg["LDH"][indice])
        valoriCA.append(datasetNeg["CA"][indice])
        tamponi.append(datasetNeg["TAMPONI"][indice])
        sesso.append(datasetNeg["Sex"][indice])
        età.append(datasetNeg["Age"][indice])
    cnt += 1

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
datasetNeg = pd.DataFrame(dict)


# definisco una funzione di supporto
def aggiornaListaSupportoPos(counter, lista, c, nomeGrandezza):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(datasetPos[nomeGrandezza][r])
        r += 1
    return nuovaLista


# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiornoPos = []
i = 0
while i < len(datasetPos["DATA"]) - 1:
    if datasetPos["DATA"][i] != datasetPos["DATA"][i + 1]:
        indiciCambioGiornoPos.append(i)
    i += 1


def aggiornaListaSupportoNeg(counter, lista, c, nomeGrandezza):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(datasetNeg[nomeGrandezza][r])
        r += 1
    return nuovaLista


# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiornoNeg = []
i = 0
while i < len(datasetNeg["DATA"]) - 1:
    if datasetNeg["DATA"][i] != datasetNeg["DATA"][i + 1]:
        indiciCambioGiornoNeg.append(i)
    i += 1

# inizio con il grafico per PLT1
# calcolo medie e deviazioni standard dei positivi
mediePLT1Pos = []
stdevPLT1Pos = []
listaSupporto = []

cnt = 13  # indica il giorno da cui parte la lista di supporto
nPLT1Pos = []
for i in range(16, 803):
    if i in indiciCambioGiornoPos:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(datasetPos["PLT1"][i])
        listaSupporto = aggiornaListaSupportoPos(indiciCambioGiornoPos[cnt], listaSupporto,
                                                 indiciCambioGiornoPos[cnt + 7], "PLT1")
        if (cnt < 74) or (cnt > 80):
            # elimino gli zeri dalla lista
            listaSupporto = [i for i in listaSupporto if i != 0]
            nPLT1Pos.append(len(listaSupporto))
            mediePLT1Pos.append(mean(listaSupporto))
            if len(listaSupporto) <= 1:
                stdevPLT1Pos.append(0)
            else:
                stdevPLT1Pos.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(datasetPos["PLT1"][i])

# aggiungo l'ultimo valore mancante alle due liste
mediePLT1Pos.append(219.17)
stdevPLT1Pos.append(93.57)
print(mediePLT1Pos)
print(len(mediePLT1Pos))
print(stdevPLT1Pos)
print(len(stdevPLT1Pos))
print(nPLT1Pos)
print(len(nPLT1Pos))

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno
dict = {
    "giorno": np.arange(21, 82).tolist() + np.arange(89, 115).tolist(),
    "media": mediePLT1Pos,
    "dev standard": stdevPLT1Pos
}
datiPLT1Pos = pd.DataFrame(dict)

# calcolo medie e deviazioni standard dei negativi
mediePLT1Neg = []
stdevPLT1Neg = []
listaSupporto = []

cnt = 12  # indica il giorno da cui parte la lista di supporto
nPLT1Neg = []
for i in range(0, 732):
    if i in indiciCambioGiornoNeg:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(datasetNeg["PLT1"][i])
        listaSupporto = aggiornaListaSupportoNeg(indiciCambioGiornoNeg[cnt], listaSupporto,
                                                 indiciCambioGiornoNeg[cnt + 7], "PLT1")
        if (cnt < 74) or (cnt > 80):
            # elimino gli zeri dalla lista
            listaSupporto = [i for i in listaSupporto if i != 0]
            nPLT1Neg.append(len(listaSupporto))
            mediePLT1Neg.append(mean(listaSupporto))
            if len(listaSupporto) <= 1:
                stdevPLT1Neg.append(0)
            else:
               stdevPLT1Neg.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(datasetNeg["PLT1"][i])

# aggiungo gli ultimi due valori mancanti
mediePLT1Neg += [241.73, 233.82]
stdevPLT1Neg += [75.45, 71.98]

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno
dict = {
    "giorno": np.arange(20, 82).tolist() + np.arange(89, 115).tolist(),
    "media": mediePLT1Neg,
    "dev standard": stdevPLT1Neg
}
datiPLT1Neg = pd.DataFrame(dict)

# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x="giorno", y="media", data=datiPLT1Pos, marker=".", markersize=14, markeredgecolor='None',
             markerfacecolor="#b80f0a", color="#b80f0a", label="Positive", zorder=1.0)
sns.lineplot(x="giorno", y="media", data=datiPLT1Neg, marker=".", markersize=14, markeredgecolor="None",
             markerfacecolor="#1034a6", color="#1034a6", label="Negative", zorder=1.0)

# calcolo gli intervalli di confidenza dei positivi e li disegno

# aggiungo gli ultimi sue valori mancanti alla lista degli n già popolata in precedenza
nPLT1Pos.append(2)

icPLT1Pos = []
cnt = 0
for i in datiPLT1Pos["dev standard"]:
    icPLT1Pos.append(1.96 * i / sqrt(nPLT1Pos[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinPLT1Pos = []
cnt = 0
for i in datiPLT1Pos["dev standard"]:
    icMinPLT1Pos.append(datiPLT1Pos["media"][cnt] - icPLT1Pos[cnt])
    cnt += 1
icMaxPLT1Pos = []
cnt = 0
for i in datiPLT1Pos["dev standard"]:
    icMaxPLT1Pos.append(datiPLT1Pos["media"][cnt] + icPLT1Pos[cnt])
    cnt += 1


# faccio una serie di operazioni per mostrare gli ic centrati rispetto al punto con la dimensione e la trasparenza
# corrette
def calcolaOrdinata(P, Q, x):
    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = a * (P[0]) + b * (P[1])
    if b < 0:
        y = (c - (a * x)) / b
    else:
        y = ((a * x) - c) / b
    return y


ordinateMin = []
ordinateMax = []
for i in range(1, len(datiPLT1Pos["giorno"])):
    ordinateMin.append(calcolaOrdinata([datiPLT1Pos["giorno"][i - 1], icMinPLT1Pos[i - 1]],
                                       [datiPLT1Pos["giorno"][i], icMinPLT1Pos[i]], datiPLT1Pos["giorno"][i - 1] + 0.5))
    ordinateMax.append(calcolaOrdinata([datiPLT1Pos["giorno"][i - 1], icMaxPLT1Pos[i - 1]],
                                       [datiPLT1Pos["giorno"][i], icMaxPLT1Pos[i]], datiPLT1Pos["giorno"][i - 1] + 0.5))
# ora inserisco i nuovi valori nei giorni e negli ic

giorni = [21]
for i in datiPLT1Pos["giorno"]:
    if i >= 22:
        giorni.append(i - 0.5)
        giorni.append(i)

icMinPLT1PosNuovi = []
cnt = 0
for i in range(0, len(icMinPLT1Pos) - 1):
    icMinPLT1PosNuovi.append(icMinPLT1Pos[i])
    icMinPLT1PosNuovi.append(ordinateMin[cnt])
    cnt += 1
icMinPLT1PosNuovi.append(icMinPLT1Pos[len(icMinPLT1Pos) - 1])

icMaxPLT1PosNuovi = []
cnt = 0
for i in range(0, len(icMaxPLT1Pos) - 1):
    icMaxPLT1PosNuovi.append(icMaxPLT1Pos[i])
    icMaxPLT1PosNuovi.append(ordinateMax[cnt])
    cnt += 1
icMaxPLT1PosNuovi.append(icMaxPLT1Pos[len(icMaxPLT1Pos) - 1])

# infine, aggiorno gli nPLT1
nPLT1PosNuovi = [nPLT1Pos[0]]
# il primo e l'ultimo non vanno duplicati
for i in range(1, len(nPLT1Pos) - 1):
    nPLT1PosNuovi.append(nPLT1Pos[i])
    nPLT1PosNuovi.append(nPLT1Pos[i])
nPLT1PosNuovi.append(nPLT1Pos[len(nPLT1Pos) - 1])

plt.margins(0.05)
for b in range(0, len(nPLT1PosNuovi)):
    i = nPLT1PosNuovi[b]
    if i <= 5:
        alpha = 0.08
    if i >= 30:
        alpha = 0.45
    if (i > 5) & (i < 30):
        alpha = i / 66.67
    # se non ci sono dati nei 7 giorni precedenti non mostro nulla
    if i == 0:
        alpha = 0
    facecolor = (1, 0, 0, alpha)
    ax.fill_between(giorni[b:(b + 2)], icMinPLT1PosNuovi[b: (b + 2)], icMaxPLT1PosNuovi[b: (b + 2)],
                    facecolor=facecolor, edgecolor=(0, 0.25, 1, 0), linewidths=0)
# ne faccio uno solo per la legenda, ma non lo mostro
ax.fill_between(giorni[1:(1 + 2)], icMinPLT1PosNuovi[1: (1 + 2)], icMaxPLT1PosNuovi[1: (1 + 2)], facecolor=(1, 0, 0, 0.5),
                label="Positive 95% confidence interval").set_visible(False)

# calcolo gli intervalli di confidenza dei negativi e li disegno
# aggiungo gli ultimi sue valori mancanti alla lista degli n già popolata in precedenza
nPLT1Neg += [43, 41]

icPLT1Neg = []
cnt = 0
for i in datiPLT1Neg["dev standard"]:
    icPLT1Neg.append(1.96 * i / sqrt(nPLT1Neg[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinPLT1Neg = []
cnt = 0
for i in datiPLT1Neg["dev standard"]:
    icMinPLT1Neg.append(datiPLT1Neg["media"][cnt] - icPLT1Neg[cnt])
    cnt += 1
icMaxPLT1Neg = []
cnt = 0
for i in datiPLT1Neg["dev standard"]:
    icMaxPLT1Neg.append(datiPLT1Neg["media"][cnt] + icPLT1Neg[cnt])
    cnt += 1

# faccio una serie di operazioni per rendere cnetrare i punti rieptto agli intervalli di confidenza e fare in modo
# che trasparenza e altezza dell'intervallo siano corretti

ordinateMin = []
ordinateMax = []
for i in range(1, len(datiPLT1Neg["giorno"])):
    ordinateMin.append(calcolaOrdinata([datiPLT1Neg["giorno"][i - 1], icMinPLT1Neg[i - 1]],
                                       [datiPLT1Neg["giorno"][i], icMinPLT1Neg[i]], datiPLT1Neg["giorno"][i - 1] + 0.5))
    ordinateMax.append(calcolaOrdinata([datiPLT1Neg["giorno"][i - 1], icMaxPLT1Neg[i - 1]],
                                       [datiPLT1Neg["giorno"][i], icMaxPLT1Neg[i]], datiPLT1Neg["giorno"][i - 1] + 0.5))
giorni = [20]
for i in datiPLT1Neg["giorno"]:
    if i >= 21:
        giorni.append(i - 0.5)
        giorni.append(i)
print(giorni)
icMinPLT1NegNuovi = []
cnt = 0
for i in range(0, len(icMinPLT1Neg) - 1):
    icMinPLT1NegNuovi.append(icMinPLT1Neg[i])
    icMinPLT1NegNuovi.append(ordinateMin[cnt])
    cnt += 1
icMinPLT1NegNuovi.append(icMinPLT1Neg[len(icMinPLT1Neg) - 1])

icMaxPLT1NegNuovi = []
cnt = 0
for i in range(0, len(icMaxPLT1Neg) - 1):
    icMaxPLT1NegNuovi.append(icMaxPLT1Neg[i])
    icMaxPLT1NegNuovi.append(ordinateMax[cnt])
    cnt += 1
icMaxPLT1NegNuovi.append(icMaxPLT1Neg[len(icMaxPLT1Neg) - 1])

# infine, aggiorno gli nPLT1
nPLT1NegNuovi = [nPLT1Neg[0]]
# il primo e l'ultimo valore non vanno duplicati
for i in range(1, len(nPLT1Neg) - 1):
    nPLT1NegNuovi.append(nPLT1Neg[i])
    nPLT1NegNuovi.append(nPLT1Neg[i])
nPLT1NegNuovi.append(nPLT1Neg[len(nPLT1Neg) - 1])

for b in range(0, len(nPLT1NegNuovi)):
    i = nPLT1NegNuovi[b]
    if i <= 5:
        alpha = 0.08
    if i >= 30:
        alpha = 0.45
    if (i > 5) & (i < 30):
        alpha = i / 66.67
    if i == 0:
        alpha = 0
    facecolor = (0, 0.25, 1, alpha)
    ax.fill_between(giorni[b:(b + 2)], icMinPLT1NegNuovi[b: (b + 2)], icMaxPLT1NegNuovi[b: (b + 2)],
                    facecolor=facecolor, linewidths=0.0)

# coloro di bianco la fascia che devo saltare
ax.fill_between([81.2, 88.9], [150, 150], [420, 420], facecolor=(1, 1, 1, 1), linewidths=0.0, zorder=5.0)

# ne faccio uno solo per la legenda, ma non lo mostro
ax.fill_between(giorni[1:(1 + 2)], icMinPLT1PosNuovi[1: (1 + 2)], icMaxPLT1PosNuovi[1: (1 + 2)],
                facecolor=(0, 0.25, 1, 0.5), label="Negative 95% confidence interval").set_visible(False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xlabel("Time (days)", fontsize=12)
plt.ylabel("previous 7 days mean PLT1", fontsize=12)

plt.xlim(7.8, 115.2)
plt.xticks([38, 69, 99], labels=[])
ax.tick_params(which='major', length=6, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([24, 55, 85], minor=True)
ax.tick_params(which='minor', length=10, color='black')
ax.text(8, 45, "F", size=10)
ax.text(23.6, 45, "M", size=10)
ax.text(54.7, 45, "A", size=10)
ax.text(84.6, 45, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda
plt.legend(loc="upper left", ncol=1, fancybox=True, frameon=False)

plt.show()




# passo a pcr
# calcolo medie e deviazioni standard dei positivi
mediePCRPos = []
stdevPCRPos = []
listaSupporto = []

cnt = 13  # indica il giorno da cui parte la lista di supporto
nPCRPos = []
for i in range(16, 803):
    if i in indiciCambioGiornoPos:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(datasetPos["PCR"][i])
        listaSupporto = aggiornaListaSupportoPos(indiciCambioGiornoPos[cnt], listaSupporto,
                                                 indiciCambioGiornoPos[cnt + 7], "PCR")
        if (cnt < 74) or (cnt > 80):
            # elimino gli zeri dalla lista
            listaSupporto = [i for i in listaSupporto if i != 0]
            nPCRPos.append(len(listaSupporto))
            mediePCRPos.append(mean(listaSupporto))
            if len(listaSupporto) <= 1:
                stdevPCRPos.append(0)
            else:
                stdevPCRPos.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(datasetPos["PCR"][i])

# aggiungo l'ultimo valore mancante alle due liste
mediePCRPos.append(61.55)
stdevPCRPos.append(45.61)

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno
dict = {
    "giorno": np.arange(21, 82).tolist() + np.arange(89, 115).tolist(),
    "media": mediePCRPos,
    "dev standard": stdevPCRPos
}
datiPCRPos = pd.DataFrame(dict)

# calcolo medie e deviazioni standard dei negativi
mediePCRNeg = []
stdevPCRNeg = []
listaSupporto = []

cnt = 12  # indica il giorno da cui parte la lista di supporto
nPCRNeg = []
for i in range(0, 732):
    if i in indiciCambioGiornoNeg:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(datasetNeg["PCR"][i])
        listaSupporto = aggiornaListaSupportoNeg(indiciCambioGiornoNeg[cnt], listaSupporto,
                                                 indiciCambioGiornoNeg[cnt + 7], "PCR")
        if (cnt < 74) or (cnt > 80):
            # elimino gli zeri dalla lista
            listaSupporto = [i for i in listaSupporto if i != 0]
            nPCRNeg.append(len(listaSupporto))
            mediePCRNeg.append(mean(listaSupporto))
            if len(listaSupporto) <= 1:
                stdevPCRNeg.append(0)
            else:
                stdevPCRNeg.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(datasetNeg["PCR"][i])

# aggiungo gli ultimi due valori mancanti
mediePCRNeg += [41.42, 41.6]
stdevPCRNeg += [60.47, 59.99]

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno
dict = {
    "giorno": np.arange(20, 82).tolist() + np.arange(89, 115).tolist(),
    "media": mediePCRNeg,
    "dev standard": stdevPCRNeg
}

datiPCRNeg = pd.DataFrame(dict)

# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x="giorno", y="media", data=datiPCRPos, marker=".", markersize=14, markeredgecolor='None',
             markerfacecolor="#b80f0a", color="#b80f0a", label="Positive")
sns.lineplot(x="giorno", y="media", data=datiPCRNeg, marker=".", markersize=14, markeredgecolor="None",
             markerfacecolor="#1034a6", color="#1034a6", label="Negative")

# calcolo gli intervalli di confidenza dei positivi e li disegno
# aggiungo l'ultimo valore mancante alla lista degli n già popolata in precedenza
nPCRPos.append(2)

icPCRPos = []
cnt = 0
for i in datiPCRPos["dev standard"]:
    icPCRPos.append(1.96 * i / sqrt(nPCRPos[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinPCRPos = []
cnt = 0
for i in datiPCRPos["dev standard"]:
    icMinPCRPos.append(datiPCRPos["media"][cnt] - icPCRPos[cnt])
    cnt += 1
icMaxPCRPos = []
cnt = 0
for i in datiPCRPos["dev standard"]:
    icMaxPCRPos.append(datiPCRPos["media"][cnt] + icPCRPos[cnt])
    cnt += 1


# faccio una serie di operazioni per mostrare gli ic centrati rispetto al punto con la dimensione e la trasparenza
# corrette
def calcolaOrdinata(P, Q, x):
    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = a * (P[0]) + b * (P[1])
    if b < 0:
        y = (c - (a * x)) / b
    else:
        y = ((a * x) - c) / b
    return y


ordinateMin = []
ordinateMax = []
for i in range(1, len(datiPCRPos["giorno"])):
    ordinateMin.append(calcolaOrdinata([datiPCRPos["giorno"][i - 1], icMinPCRPos[i - 1]],
                                       [datiPCRPos["giorno"][i], icMinPCRPos[i]], datiPCRPos["giorno"][i - 1] + 0.5))
    ordinateMax.append(calcolaOrdinata([datiPCRPos["giorno"][i - 1], icMaxPCRPos[i - 1]],
                                       [datiPCRPos["giorno"][i], icMaxPCRPos[i]], datiPCRPos["giorno"][i - 1] + 0.5))
# ora inserisco i nuovi valori nei giorni e negli ic

giorni = [21]
for i in datiPCRPos["giorno"]:
    if i >= 22:
        giorni.append(i - 0.5)
        giorni.append(i)

icMinPCRPosNuovi = []
cnt = 0
for i in range(0, len(icMinPCRPos) - 1):
    icMinPCRPosNuovi.append(icMinPCRPos[i])
    icMinPCRPosNuovi.append(ordinateMin[cnt])
    cnt += 1
icMinPCRPosNuovi.append(icMinPCRPos[len(icMinPCRPos) - 1])

icMaxPCRPosNuovi = []
cnt = 0
for i in range(0, len(icMaxPCRPos) - 1):
    icMaxPCRPosNuovi.append(icMaxPCRPos[i])
    icMaxPCRPosNuovi.append(ordinateMax[cnt])
    cnt += 1
icMaxPCRPosNuovi.append(icMaxPCRPos[len(icMaxPCRPos) - 1])

# infine, aggiorno gli nPCR
nPCRPosNuovi = [nPCRPos[0]]
# il primo e l'ultimo non vanno duplicati
for i in range(1, len(nPCRPos) - 1):
    nPCRPosNuovi.append(nPCRPos[i])
    nPCRPosNuovi.append(nPCRPos[i])
nPCRPosNuovi.append(nPCRPos[len(nPCRPos) - 1])

plt.margins(0.05)
for b in range(0, len(nPCRPosNuovi)):
    i = nPCRPosNuovi[b]
    if i <= 5:
        alpha = 0.08
    if i >= 30:
        alpha = 0.45
    if (i > 5) & (i < 30):
        alpha = i / 66.67
    # se non ci sono dati nei 7 giorni precedenti non mostro nulla
    if i == 0:
        alpha = 0
    facecolor = (1, 0, 0, alpha)
    ax.fill_between(giorni[b:(b + 2)], icMinPCRPosNuovi[b: (b + 2)], icMaxPCRPosNuovi[b: (b + 2)],
                    facecolor=facecolor, edgecolor=(0, 0.25, 1, 0), linewidths=0)
# ne faccio uno solo per la legenda, ma non lo mostro
ax.fill_between(giorni[1:(1 + 2)], icMinPCRPosNuovi[1: (1 + 2)], icMaxPCRPosNuovi[1: (1 + 2)], facecolor=(1, 0, 0, 0.5),
                label="Positive 95% confidence interval").set_visible(False)

# calcolo gli intervalli di confidenza dei negativi e li disegno
# aggiungo gli ultimi sue valori mancanti alla lista degli n già popolata in precedenza
nPCRNeg += [43, 41]

icPCRNeg = []
cnt = 0
for i in datiPCRNeg["dev standard"]:
    icPCRNeg.append(1.96 * i / sqrt(nPCRNeg[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinPCRNeg = []
cnt = 0
for i in datiPCRNeg["dev standard"]:
    icMinPCRNeg.append(datiPCRNeg["media"][cnt] - icPCRNeg[cnt])
    cnt += 1
icMaxPCRNeg = []
cnt = 0
for i in datiPCRNeg["dev standard"]:
    icMaxPCRNeg.append(datiPCRNeg["media"][cnt] + icPCRNeg[cnt])
    cnt += 1

# faccio una serie di operazioni per rendere cnetrare i punti rieptto agli intervalli di confidenza e fare in modo
# che trasparenza e altezza dell'intervallo siano corretti

ordinateMin = []
ordinateMax = []
for i in range(1, len(datiPCRNeg["giorno"])):
    ordinateMin.append(calcolaOrdinata([datiPCRNeg["giorno"][i - 1], icMinPCRNeg[i - 1]],
                                       [datiPCRNeg["giorno"][i], icMinPCRNeg[i]], datiPCRNeg["giorno"][i - 1] + 0.5))
    ordinateMax.append(calcolaOrdinata([datiPCRNeg["giorno"][i - 1], icMaxPCRNeg[i - 1]],
                                       [datiPCRNeg["giorno"][i], icMaxPCRNeg[i]], datiPCRNeg["giorno"][i - 1] + 0.5))
giorni = [20]
for i in datiPCRNeg["giorno"]:
    if i >= 21:
        giorni.append(i - 0.5)
        giorni.append(i)
icMinPCRNegNuovi = []
cnt = 0
for i in range(0, len(icMinPCRNeg) - 1):
    icMinPCRNegNuovi.append(icMinPCRNeg[i])
    icMinPCRNegNuovi.append(ordinateMin[cnt])
    cnt += 1
icMinPCRNegNuovi.append(icMinPCRNeg[len(icMinPCRNeg) - 1])

icMaxPCRNegNuovi = []
cnt = 0
for i in range(0, len(icMaxPCRNeg) - 1):
    icMaxPCRNegNuovi.append(icMaxPCRNeg[i])
    icMaxPCRNegNuovi.append(ordinateMax[cnt])
    cnt += 1
icMaxPCRNegNuovi.append(icMaxPCRNeg[len(icMaxPCRNeg) - 1])

# infine, aggiorno gli nPCR
nPCRNegNuovi = [nPCRNeg[0]]
# il primo e l'ultimo valore non vanno duplicati
for i in range(1, len(nPCRNeg) - 1):
    nPCRNegNuovi.append(nPCRNeg[i])
    nPCRNegNuovi.append(nPCRNeg[i])
nPCRNegNuovi.append(nPCRNeg[len(nPCRNeg) - 1])

for b in range(0, len(nPCRNegNuovi)):
    i = nPCRNegNuovi[b]
    if i <= 5:
        alpha = 0.08
    if i >= 30:
        alpha = 0.45
    if (i > 5) & (i < 30):
        alpha = i / 66.67
    if i == 0:
        alpha = 0
    facecolor = (0, 0.25, 1, alpha)
    ax.fill_between(giorni[b:(b + 2)], icMinPCRNegNuovi[b: (b + 2)], icMaxPCRNegNuovi[b: (b + 2)],
                    facecolor=facecolor, linewidths=0.0)
# coloro di bianco la fascia che devo saltare
ax.fill_between([81.2, 89], [0, 0], [200, 200], facecolor=(1, 1, 1, 1), linewidths=0.0, zorder=5.0)

# ne faccio uno solo per la legenda, ma non lo mostro
ax.fill_between(giorni[1:(1 + 2)], icMinPCRPosNuovi[1: (1 + 2)], icMaxPCRPosNuovi[1: (1 + 2)],
                facecolor=(0, 0.25, 1, 0.5), label="Negative 95% confidence interval").set_visible(False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xlabel("Time (days)", fontsize=12)
plt.ylabel("previous 7 days mean PCR", fontsize=12)

plt.xlim(7.8, 115.2)
plt.xticks([38, 69, 99], labels=[])
ax.tick_params(which='major', length=6, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([24, 55, 85], minor=True)
ax.tick_params(which='minor', length=10, color='black')
ax.text(8, -100, "F", size=10)
ax.text(23.6, -100, "M", size=10)
ax.text(54.7, -100, "A", size=10)
ax.text(84.6, -100, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda
plt.legend(loc="upper left", ncol=1, fancybox=True, frameon=False)

plt.show()



# passo a a FG
# calcolo medie e deviazioni standard dei positivi
medieFGPos = []
stdevFGPos = []
listaSupporto = []

cnt = 13  # indica il giorno da cui parte la lista di supporto
nFGPos = []
for i in range(16, 803):
    if i in indiciCambioGiornoPos:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(datasetPos["FG"][i])
        listaSupporto = aggiornaListaSupportoPos(indiciCambioGiornoPos[cnt], listaSupporto,
                                                 indiciCambioGiornoPos[cnt + 7], "FG")
        if (cnt < 74) or (cnt > 80):
            # elimino gli zeri dalla lista
            listaSupporto = [i for i in listaSupporto if i != 0]
            nFGPos.append(len(listaSupporto))
            medieFGPos.append(mean(listaSupporto))
            if len(listaSupporto) <= 1:
                stdevFGPos.append(0)
            else:
                stdevFGPos.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(datasetPos["FG"][i])

# aggiungo l'ultimo valore mancante alle due liste
medieFGPos.append(511.5)
stdevFGPos.append(65.76)

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno
dict = {
    "giorno": np.arange(21, 82).tolist() + np.arange(89, 115).tolist(),
    "media": medieFGPos,
    "dev standard": stdevFGPos
}
datiFGPos = pd.DataFrame(dict)

# calcolo medie e deviazioni standard dei negativi
medieFGNeg = []
stdevFGNeg = []
listaSupporto = []

cnt = 12  # indica il giorno da cui parte la lista di supporto
nFGNeg = []
for i in range(0, 732):
    if i in indiciCambioGiornoNeg:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(datasetNeg["FG"][i])
        listaSupporto = aggiornaListaSupportoNeg(indiciCambioGiornoNeg[cnt], listaSupporto,
                                                 indiciCambioGiornoNeg[cnt + 7], "FG")
        if (cnt < 74) or (cnt > 80):
            # elimino gli zeri dalla lista
            listaSupporto = [i for i in listaSupporto if i != 0]
            nFGNeg.append(len(listaSupporto))
            medieFGNeg.append(mean(listaSupporto))
            if len(listaSupporto) <= 1:
                stdevFGNeg.append(0)
            else:
                stdevFGNeg.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(datasetNeg["FG"][i])

# aggiungo gli ultimi due valori mancanti
medieFGNeg += [428, 426.87]
stdevFGNeg += [164.64, 170.35]

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno
dict = {
    "giorno": np.arange(20, 82).tolist() + np.arange(89, 115).tolist(),
    "media": medieFGNeg,
    "dev standard": stdevFGNeg
}

datiFGNeg = pd.DataFrame(dict)

# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x="giorno", y="media", data=datiFGPos, marker=".", markersize=14, markeredgecolor='None',
             markerfacecolor="#b80f0a", color="#b80f0a", label="Positive")
sns.lineplot(x="giorno", y="media", data=datiFGNeg, marker=".", markersize=14, markeredgecolor="None",
             markerfacecolor="#1034a6", color="#1034a6", label="Negative")

# calcolo gli intervalli di confidenza dei positivi e li disegno
# aggiungo l'ultimo valore mancante alla lista degli n già popolata in precedenza
nFGPos.append(2)

icFGPos = []
cnt = 0
for i in datiFGPos["dev standard"]:
    icFGPos.append(1.96 * i / sqrt(nFGPos[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinFGPos = []
cnt = 0
for i in datiFGPos["dev standard"]:
    icMinFGPos.append(datiFGPos["media"][cnt] - icFGPos[cnt])
    cnt += 1
icMaxFGPos = []
cnt = 0
for i in datiFGPos["dev standard"]:
    icMaxFGPos.append(datiFGPos["media"][cnt] + icFGPos[cnt])
    cnt += 1


# faccio una serie di operazioni per mostrare gli ic centrati rispetto al punto con la dimensione e la trasparenza
# corrette
def calcolaOrdinata(P, Q, x):
    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = a * (P[0]) + b * (P[1])
    if b < 0:
        y = (c - (a * x)) / b
    else:
        y = ((a * x) - c) / b
    return y


ordinateMin = []
ordinateMax = []
for i in range(1, len(datiFGPos["giorno"])):
    ordinateMin.append(calcolaOrdinata([datiFGPos["giorno"][i - 1], icMinFGPos[i - 1]],
                                       [datiFGPos["giorno"][i], icMinFGPos[i]], datiFGPos["giorno"][i - 1] + 0.5))
    ordinateMax.append(calcolaOrdinata([datiFGPos["giorno"][i - 1], icMaxFGPos[i - 1]],
                                       [datiFGPos["giorno"][i], icMaxFGPos[i]], datiFGPos["giorno"][i - 1] + 0.5))
# ora inserisco i nuovi valori nei giorni e negli ic

giorni = [21]
for i in datiFGPos["giorno"]:
    if i >= 22:
        giorni.append(i - 0.5)
        giorni.append(i)

icMinFGPosNuovi = []
cnt = 0
for i in range(0, len(icMinFGPos) - 1):
    icMinFGPosNuovi.append(icMinFGPos[i])
    icMinFGPosNuovi.append(ordinateMin[cnt])
    cnt += 1
icMinFGPosNuovi.append(icMinFGPos[len(icMinFGPos) - 1])

icMaxFGPosNuovi = []
cnt = 0
for i in range(0, len(icMaxFGPos) - 1):
    icMaxFGPosNuovi.append(icMaxFGPos[i])
    icMaxFGPosNuovi.append(ordinateMax[cnt])
    cnt += 1
icMaxFGPosNuovi.append(icMaxFGPos[len(icMaxFGPos) - 1])

# infine, aggiorno gli nFG
nFGPosNuovi = [nFGPos[0]]
# il primo e l'ultimo non vanno duplicati
for i in range(1, len(nFGPos) - 1):
    nFGPosNuovi.append(nFGPos[i])
    nFGPosNuovi.append(nFGPos[i])
nFGPosNuovi.append(nFGPos[len(nFGPos) - 1])

plt.margins(0.05)
for b in range(0, len(nFGPosNuovi)):
    i = nFGPosNuovi[b]
    if i <= 5:
        alpha = 0.08
    if i >= 30:
        alpha = 0.45
    if (i > 5) & (i < 30):
        alpha = i / 66.67
    # se non ci sono dati nei 7 giorni precedenti non mostro nulla
    if i == 0:
        alpha = 0
    facecolor = (1, 0, 0, alpha)
    ax.fill_between(giorni[b:(b + 2)], icMinFGPosNuovi[b: (b + 2)], icMaxFGPosNuovi[b: (b + 2)],
                    facecolor=facecolor, edgecolor=(0, 0.25, 1, 0), linewidths=0)
# ne faccio uno solo per la legenda, ma non lo mostro
ax.fill_between(giorni[1:(1 + 2)], icMinFGPosNuovi[1: (1 + 2)], icMaxFGPosNuovi[1: (1 + 2)], facecolor=(1, 0, 0, 0.5),
                label="Positive 95% confidence interval").set_visible(False)

# calcolo gli intervalli di confidenza dei negativi e li disegno
# aggiungo gli ultimi sue valori mancanti alla lista degli n già popolata in precedenza
nFGNeg += [16, 15]

icFGNeg = []
cnt = 0
for i in datiFGNeg["dev standard"]:
    icFGNeg.append(1.96 * i / sqrt(nFGNeg[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinFGNeg = []
cnt = 0
for i in datiFGNeg["dev standard"]:
    icMinFGNeg.append(datiFGNeg["media"][cnt] - icFGNeg[cnt])
    cnt += 1
icMaxFGNeg = []
cnt = 0
for i in datiFGNeg["dev standard"]:
    icMaxFGNeg.append(datiFGNeg["media"][cnt] + icFGNeg[cnt])
    cnt += 1

# faccio una serie di operazioni per rendere cnetrare i punti rieptto agli intervalli di confidenza e fare in modo
# che trasparenza e altezza dell'intervallo siano corretti

ordinateMin = []
ordinateMax = []
for i in range(1, len(datiFGNeg["giorno"])):
    ordinateMin.append(calcolaOrdinata([datiFGNeg["giorno"][i - 1], icMinFGNeg[i - 1]],
                                       [datiFGNeg["giorno"][i], icMinFGNeg[i]], datiFGNeg["giorno"][i - 1] + 0.5))
    ordinateMax.append(calcolaOrdinata([datiFGNeg["giorno"][i - 1], icMaxFGNeg[i - 1]],
                                       [datiFGNeg["giorno"][i], icMaxFGNeg[i]], datiFGNeg["giorno"][i - 1] + 0.5))
giorni = [20]
for i in datiFGNeg["giorno"]:
    if i >= 21:
        giorni.append(i - 0.5)
        giorni.append(i)
icMinFGNegNuovi = []
cnt = 0
for i in range(0, len(icMinFGNeg) - 1):
    icMinFGNegNuovi.append(icMinFGNeg[i])
    icMinFGNegNuovi.append(ordinateMin[cnt])
    cnt += 1
icMinFGNegNuovi.append(icMinFGNeg[len(icMinFGNeg) - 1])

icMaxFGNegNuovi = []
cnt = 0
for i in range(0, len(icMaxFGNeg) - 1):
    icMaxFGNegNuovi.append(icMaxFGNeg[i])
    icMaxFGNegNuovi.append(ordinateMax[cnt])
    cnt += 1
icMaxFGNegNuovi.append(icMaxFGNeg[len(icMaxFGNeg) - 1])

# infine, aggiorno gli nFG
nFGNegNuovi = [nFGNeg[0]]
# il primo e l'ultimo valore non vanno duplicati
for i in range(1, len(nFGNeg) - 1):
    nFGNegNuovi.append(nFGNeg[i])
    nFGNegNuovi.append(nFGNeg[i])
nFGNegNuovi.append(nFGNeg[len(nFGNeg) - 1])
print(nFGNeg)
print(nFGNegNuovi)

for b in range(0, len(nFGNegNuovi)):
    i = nFGNegNuovi[b]
    if i <= 5:
        alpha = 0.08
    if i >= 30:
        alpha = 0.45
    if (i > 5) & (i < 30):
        alpha = i / 66.67
    if i == 0:
        alpha = 0
    facecolor = (0, 0.25, 1, alpha)
    ax.fill_between(giorni[b:(b + 2)], icMinFGNegNuovi[b: (b + 2)], icMaxFGNegNuovi[b: (b + 2)],
                    facecolor=facecolor, linewidths=0.0)
# coloro di bianco la fascia che devo saltare
ax.fill_between([81.2, 89], [200, 200], [700, 700], facecolor=(1, 1, 1, 1), linewidths=0.0, zorder=5.0)

# ne faccio uno solo per la legenda, ma non lo mostro
ax.fill_between(giorni[1:(1 + 2)], icMinFGPosNuovi[1: (1 + 2)], icMaxFGPosNuovi[1: (1 + 2)],
                facecolor=(0, 0.25, 1, 0.5), label="Negative 95% confidence interval").set_visible(False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xlabel("Time (days)", fontsize=12)
plt.ylabel("previous 7 days mean FG", fontsize=12)

plt.xlim(7.8, 115.2)
plt.xticks([38, 69, 99], labels=[])
ax.tick_params(which='major', length=6, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([24, 55, 85], minor=True)
ax.tick_params(which='minor', length=10, color='black')
ax.text(8, -220, "F", size=10)
ax.text(23.6, -220, "M", size=10)
ax.text(54.7, -220, "A", size=10)
ax.text(84.6, -220, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda
plt.legend(loc="upper left", ncol=1, fancybox=True, frameon=False)
plt.show()



# passo a LY
# calcolo medie e deviazioni standard dei positivi
medieLYPos = []
stdevLYPos = []
listaSupporto = []

cnt = 13  # indica il giorno da cui parte la lista di supporto
nLYPos = []
for i in range(16, 803):
    if i in indiciCambioGiornoPos:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(datasetPos["LY"][i])
        listaSupporto = aggiornaListaSupportoPos(indiciCambioGiornoPos[cnt], listaSupporto,
                                                 indiciCambioGiornoPos[cnt + 7], "LY")
        if (cnt < 74) or (cnt > 80):
            # elimino gli zeri dalla lista
            listaSupporto = [i for i in listaSupporto if i != 0]
            nLYPos.append(len(listaSupporto))
            medieLYPos.append(mean(listaSupporto))
            if len(listaSupporto) <= 1:
                stdevLYPos.append(0)
            else:
                stdevLYPos.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(datasetPos["LY"][i])

# aggiungo l'ultimo valore mancante alle due liste
medieLYPos.append(10.7)
stdevLYPos.append(12.02)

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno
dict = {
    "giorno": np.arange(21, 82).tolist() + np.arange(89, 115).tolist(),
    "media": medieLYPos,
    "dev standard": stdevLYPos
}
datiLYPos = pd.DataFrame(dict)

# calcolo medie e deviazioni standard dei negativi
medieLYNeg = []
stdevLYNeg = []
listaSupporto = []

cnt = 12  # indica il giorno da cui parte la lista di supporto
nLYNeg = []
for i in range(0, 732):
    if i in indiciCambioGiornoNeg:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(datasetNeg["LY"][i])
        listaSupporto = aggiornaListaSupportoNeg(indiciCambioGiornoNeg[cnt], listaSupporto,
                                                 indiciCambioGiornoNeg[cnt + 7], "LY")
        if (cnt < 74) or (cnt > 80):
            # elimino gli zeri dalla lista
            listaSupporto = [i for i in listaSupporto if i != 0]
            nLYNeg.append(len(listaSupporto))
            medieLYNeg.append(mean(listaSupporto))
            if len(listaSupporto) <= 1:
                stdevLYNeg.append(0)
            else:
                stdevLYNeg.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(datasetNeg["LY"][i])

# aggiungo gli ultimi due valori mancanti
medieLYNeg += [17.344, 16.372]
stdevLYNeg += [11.15, 9.90]

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno
dict = {
    "giorno": np.arange(20, 82).tolist() + np.arange(89, 115).tolist(),
    "media": medieLYNeg,
    "dev standard": stdevLYNeg
}

datiLYNeg = pd.DataFrame(dict)

# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x="giorno", y="media", data=datiLYPos, marker=".", markersize=14, markeredgecolor='None',
             markerfacecolor="#b80f0a", color="#b80f0a", label="Positive")
sns.lineplot(x="giorno", y="media", data=datiLYNeg, marker=".", markersize=14, markeredgecolor="None",
             markerfacecolor="#1034a6", color="#1034a6", label="Negative")

# calcolo gli intervalli di confidenza dei positivi e li disegno
# aggiungo l'ultimo valore mancante alla lista degli n già popolata in precedenza
nLYPos.append(2)

icLYPos = []
cnt = 0
for i in datiLYPos["dev standard"]:
    icLYPos.append(1.96 * i / sqrt(nLYPos[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinLYPos = []
cnt = 0
for i in datiLYPos["dev standard"]:
    icMinLYPos.append(datiLYPos["media"][cnt] - icLYPos[cnt])
    cnt += 1
icMaxLYPos = []
cnt = 0
for i in datiLYPos["dev standard"]:
    icMaxLYPos.append(datiLYPos["media"][cnt] + icLYPos[cnt])
    cnt += 1


# faccio una serie di operazioni per mostrare gli ic centrati rispetto al punto con la dimensione e la trasparenza
# corrette
def calcolaOrdinata(P, Q, x):
    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = a * (P[0]) + b * (P[1])
    if b < 0:
        y = (c - (a * x)) / b
    else:
        y = ((a * x) - c) / b
    return y


ordinateMin = []
ordinateMax = []
for i in range(1, len(datiLYPos["giorno"])):
    ordinateMin.append(calcolaOrdinata([datiLYPos["giorno"][i - 1], icMinLYPos[i - 1]],
                                       [datiLYPos["giorno"][i], icMinLYPos[i]], datiLYPos["giorno"][i - 1] + 0.5))
    ordinateMax.append(calcolaOrdinata([datiLYPos["giorno"][i - 1], icMaxLYPos[i - 1]],
                                       [datiLYPos["giorno"][i], icMaxLYPos[i]], datiLYPos["giorno"][i - 1] + 0.5))
# ora inserisco i nuovi valori nei giorni e negli ic

giorni = [21]
for i in datiLYPos["giorno"]:
    if i >= 22:
        giorni.append(i - 0.5)
        giorni.append(i)

icMinLYPosNuovi = []
cnt = 0
for i in range(0, len(icMinLYPos) - 1):
    icMinLYPosNuovi.append(icMinLYPos[i])
    icMinLYPosNuovi.append(ordinateMin[cnt])
    cnt += 1
icMinLYPosNuovi.append(icMinLYPos[len(icMinLYPos) - 1])

icMaxLYPosNuovi = []
cnt = 0
for i in range(0, len(icMaxLYPos) - 1):
    icMaxLYPosNuovi.append(icMaxLYPos[i])
    icMaxLYPosNuovi.append(ordinateMax[cnt])
    cnt += 1
icMaxLYPosNuovi.append(icMaxLYPos[len(icMaxLYPos) - 1])

# infine, aggiorno gli nLY
nLYPosNuovi = [nLYPos[0]]
# il primo e l'ultimo non vanno duplicati
for i in range(1, len(nLYPos) - 1):
    nLYPosNuovi.append(nLYPos[i])
    nLYPosNuovi.append(nLYPos[i])
nLYPosNuovi.append(nLYPos[len(nLYPos) - 1])

plt.margins(0.05)
for b in range(0, len(nLYPosNuovi)):
    i = nLYPosNuovi[b]
    if i <= 5:
        alpha = 0.08
    if i >= 30:
        alpha = 0.45
    if (i > 5) & (i < 30):
        alpha = i / 66.67
    # se non ci sono dati nei 7 giorni precedenti non mostro nulla
    if i == 0:
        alpha = 0
    facecolor = (1, 0, 0, alpha)
    ax.fill_between(giorni[b:(b + 2)], icMinLYPosNuovi[b: (b + 2)], icMaxLYPosNuovi[b: (b + 2)],
                    facecolor=facecolor, edgecolor=(0, 0.25, 1, 0), linewidths=0)
# ne faccio uno solo per la legenda, ma non lo mostro
ax.fill_between(giorni[1:(1 + 2)], icMinLYPosNuovi[1: (1 + 2)], icMaxLYPosNuovi[1: (1 + 2)], facecolor=(1, 0, 0, 0.5),
                label="Positive 95% confidence interval").set_visible(False)

# calcolo gli intervalli di confidenza dei negativi e li disegno
# aggiungo gli ultimi sue valori mancanti alla lista degli n già popolata in precedenza
nLYNeg += [28, 28]

icLYNeg = []
cnt = 0
for i in datiLYNeg["dev standard"]:
    icLYNeg.append(1.96 * i / sqrt(nLYNeg[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinLYNeg = []
cnt = 0
for i in datiLYNeg["dev standard"]:
    icMinLYNeg.append(datiLYNeg["media"][cnt] - icLYNeg[cnt])
    cnt += 1
icMaxLYNeg = []
cnt = 0
for i in datiLYNeg["dev standard"]:
    icMaxLYNeg.append(datiLYNeg["media"][cnt] + icLYNeg[cnt])
    cnt += 1

# faccio una serie di operazioni per rendere cnetrare i punti rieptto agli intervalli di confidenza e fare in modo
# che trasparenza e altezza dell'intervallo siano corretti

ordinateMin = []
ordinateMax = []
for i in range(1, len(datiLYNeg["giorno"])):
    ordinateMin.append(calcolaOrdinata([datiLYNeg["giorno"][i - 1], icMinLYNeg[i - 1]],
                                       [datiLYNeg["giorno"][i], icMinLYNeg[i]], datiLYNeg["giorno"][i - 1] + 0.5))
    ordinateMax.append(calcolaOrdinata([datiLYNeg["giorno"][i - 1], icMaxLYNeg[i - 1]],
                                       [datiLYNeg["giorno"][i], icMaxLYNeg[i]], datiLYNeg["giorno"][i - 1] + 0.5))
giorni = [20]
for i in datiLYNeg["giorno"]:
    if i >= 21:
        giorni.append(i - 0.5)
        giorni.append(i)
icMinLYNegNuovi = []
cnt = 0
for i in range(0, len(icMinLYNeg) - 1):
    icMinLYNegNuovi.append(icMinLYNeg[i])
    icMinLYNegNuovi.append(ordinateMin[cnt])
    cnt += 1
icMinLYNegNuovi.append(icMinLYNeg[len(icMinLYNeg) - 1])

icMaxLYNegNuovi = []
cnt = 0
for i in range(0, len(icMaxLYNeg) - 1):
    icMaxLYNegNuovi.append(icMaxLYNeg[i])
    icMaxLYNegNuovi.append(ordinateMax[cnt])
    cnt += 1
icMaxLYNegNuovi.append(icMaxLYNeg[len(icMaxLYNeg) - 1])

# infine, aggiorno gli nLY
nLYNegNuovi = [nLYNeg[0]]
# il primo e l'ultimo valore non vanno duplicati
for i in range(1, len(nLYNeg) - 1):
    nLYNegNuovi.append(nLYNeg[i])
    nLYNegNuovi.append(nLYNeg[i])
nLYNegNuovi.append(nLYNeg[len(nLYNeg) - 1])
print(nLYNeg)
print(nLYNegNuovi)

for b in range(0, len(nLYNegNuovi)):
    i = nLYNegNuovi[b]
    if i <= 5:
        alpha = 0.08
    if i >= 30:
        alpha = 0.45
    if (i > 5) & (i < 30):
        alpha = i / 66.67
    if i == 0:
        alpha = 0
    facecolor = (0, 0.25, 1, alpha)
    ax.fill_between(giorni[b:(b + 2)], icMinLYNegNuovi[b: (b + 2)], icMaxLYNegNuovi[b: (b + 2)],
                    facecolor=facecolor, linewidths=0.0)
# coloro di bianco la fascia che devo saltare
ax.fill_between([81.2, 89], [10, 10], [30, 30], facecolor=(1, 1, 1, 1), linewidths=0.0, zorder=5.0)

# ne faccio uno solo per la legenda, ma non lo mostro
ax.fill_between(giorni[1:(1 + 2)], icMinLYPosNuovi[1: (1 + 2)], icMaxLYPosNuovi[1: (1 + 2)],
                facecolor=(0, 0.25, 1, 0.5), label="Negative 95% confidence interval").set_visible(False)

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
ax.text(8, -20, "F", size=10)
ax.text(23.6, -20, "M", size=10)
ax.text(54.7, -20, "A", size=10)
ax.text(84.6, -20, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda
plt.legend(loc="upper left", ncol=1, fancybox=True, frameon=False)
plt.show()



# passo a WBC
# calcolo medie e deviazioni standard dei positivi
medieWBCPos = []
stdevWBCPos = []
listaSupporto = []

cnt = 13  # indica il giorno da cui parte la lista di supporto
nWBCPos = []
for i in range(16, 803):
    if i in indiciCambioGiornoPos:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(datasetPos["WBC"][i])
        listaSupporto = aggiornaListaSupportoPos(indiciCambioGiornoPos[cnt], listaSupporto,
                                                 indiciCambioGiornoPos[cnt + 7], "WBC")
        if (cnt < 74) or (cnt > 80):
            # elimino gli zeri dalla lista
            listaSupporto = [i for i in listaSupporto if i != 0]
            nWBCPos.append(len(listaSupporto))
            medieWBCPos.append(mean(listaSupporto))
            if len(listaSupporto) <= 1:
                stdevWBCPos.append(0)
            else:
                stdevWBCPos.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(datasetPos["WBC"][i])

# aggiungo l'ultimo valore mancante alle due liste
medieWBCPos.append(13.75)
stdevWBCPos.append(11.38)

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno
dict = {
    "giorno": np.arange(21, 82).tolist() + np.arange(89, 115).tolist(),
    "media": medieWBCPos,
    "dev standard": stdevWBCPos
}
datiWBCPos = pd.DataFrame(dict)

# calcolo medie e deviazioni standard dei negativi
medieWBCNeg = []
stdevWBCNeg = []
listaSupporto = []

cnt = 12  # indica il giorno da cui parte la lista di supporto
nWBCNeg = []
for i in range(0, 732):
    if i in indiciCambioGiornoNeg:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(datasetNeg["WBC"][i])
        listaSupporto = aggiornaListaSupportoNeg(indiciCambioGiornoNeg[cnt], listaSupporto,
                                                 indiciCambioGiornoNeg[cnt + 7], "WBC")
        if (cnt < 74) or (cnt > 80):
            # elimino gli zeri dalla lista
            listaSupporto = [i for i in listaSupporto if i != 0]
            nWBCNeg.append(len(listaSupporto))
            medieWBCNeg.append(mean(listaSupporto))
            if len(listaSupporto) <= 1:
                stdevWBCNeg.append(0)
            else:
                stdevWBCNeg.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(datasetNeg["WBC"][i])

# aggiungo gli ultimi due valori mancanti
medieWBCNeg += [11.15, 11.11]
stdevWBCNeg += [11.42, 11.31]

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno
dict = {
    "giorno": np.arange(20, 82).tolist() + np.arange(89, 115).tolist(),
    "media": medieWBCNeg,
    "dev standard": stdevWBCNeg
}

datiWBCNeg = pd.DataFrame(dict)

# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x="giorno", y="media", data=datiWBCPos, marker=".", markersize=14, markeredgecolor='None',
             markerfacecolor="#b80f0a", color="#b80f0a", label="Positive")
sns.lineplot(x="giorno", y="media", data=datiWBCNeg, marker=".", markersize=14, markeredgecolor="None",
             markerfacecolor="#1034a6", color="#1034a6", label="Negative")

# calcolo gli intervalli di confidenza dei positivi e li disegno
# aggiungo l'ultimo valore mancante alla lista degli n già popolata in precedenza
nWBCPos.append(2)

icWBCPos = []
cnt = 0
for i in datiWBCPos["dev standard"]:
    icWBCPos.append(1.96 * i / sqrt(nWBCPos[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinWBCPos = []
cnt = 0
for i in datiWBCPos["dev standard"]:
    icMinWBCPos.append(datiWBCPos["media"][cnt] - icWBCPos[cnt])
    cnt += 1
icMaxWBCPos = []
cnt = 0
for i in datiWBCPos["dev standard"]:
    icMaxWBCPos.append(datiWBCPos["media"][cnt] + icWBCPos[cnt])
    cnt += 1


# faccio una serie di operazioni per mostrare gli ic centrati rispetto al punto con la dimensione e la trasparenza
# corrette
def calcolaOrdinata(P, Q, x):
    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = a * (P[0]) + b * (P[1])
    if b < 0:
        y = (c - (a * x)) / b
    else:
        y = ((a * x) - c) / b
    return y


ordinateMin = []
ordinateMax = []
for i in range(1, len(datiWBCPos["giorno"])):
    ordinateMin.append(calcolaOrdinata([datiWBCPos["giorno"][i - 1], icMinWBCPos[i - 1]],
                                       [datiWBCPos["giorno"][i], icMinWBCPos[i]], datiWBCPos["giorno"][i - 1] + 0.5))
    ordinateMax.append(calcolaOrdinata([datiWBCPos["giorno"][i - 1], icMaxWBCPos[i - 1]],
                                       [datiWBCPos["giorno"][i], icMaxWBCPos[i]], datiWBCPos["giorno"][i - 1] + 0.5))
# ora inserisco i nuovi valori nei giorni e negli ic

giorni = [21]
for i in datiWBCPos["giorno"]:
    if i >= 22:
        giorni.append(i - 0.5)
        giorni.append(i)

icMinWBCPosNuovi = []
cnt = 0
for i in range(0, len(icMinWBCPos) - 1):
    icMinWBCPosNuovi.append(icMinWBCPos[i])
    icMinWBCPosNuovi.append(ordinateMin[cnt])
    cnt += 1
icMinWBCPosNuovi.append(icMinWBCPos[len(icMinWBCPos) - 1])

icMaxWBCPosNuovi = []
cnt = 0
for i in range(0, len(icMaxWBCPos) - 1):
    icMaxWBCPosNuovi.append(icMaxWBCPos[i])
    icMaxWBCPosNuovi.append(ordinateMax[cnt])
    cnt += 1
icMaxWBCPosNuovi.append(icMaxWBCPos[len(icMaxWBCPos) - 1])

# infine, aggiorno gli nWBC
nWBCPosNuovi = [nWBCPos[0]]
# il primo e l'ultimo non vanno duplicati
for i in range(1, len(nWBCPos) - 1):
    nWBCPosNuovi.append(nWBCPos[i])
    nWBCPosNuovi.append(nWBCPos[i])
nWBCPosNuovi.append(nWBCPos[len(nWBCPos) - 1])

plt.margins(0.05)
for b in range(0, len(nWBCPosNuovi)):
    i = nWBCPosNuovi[b]
    if i <= 5:
        alpha = 0.08
    if i >= 30:
        alpha = 0.45
    if (i > 5) & (i < 30):
        alpha = i / 66.67
    # se non ci sono dati nei 7 giorni precedenti non mostro nulla
    if i == 0:
        alpha = 0
    facecolor = (1, 0, 0, alpha)
    ax.fill_between(giorni[b:(b + 2)], icMinWBCPosNuovi[b: (b + 2)], icMaxWBCPosNuovi[b: (b + 2)],
                    facecolor=facecolor, edgecolor=(0, 0.25, 1, 0), linewidths=0)
# ne faccio uno solo per la legenda, ma non lo mostro
ax.fill_between(giorni[1:(1 + 2)], icMinWBCPosNuovi[1: (1 + 2)], icMaxWBCPosNuovi[1: (1 + 2)], facecolor=(1, 0, 0, 0.5),
                label="Positive 95% confidence interval").set_visible(False)

# calcolo gli intervalli di confidenza dei negativi e li disegno
# aggiungo gli ultimi sue valori mancanti alla lista degli n già popolata in precedenza
nWBCNeg += [43, 44]

icWBCNeg = []
cnt = 0
for i in datiWBCNeg["dev standard"]:
    icWBCNeg.append(1.96 * i / sqrt(nWBCNeg[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinWBCNeg = []
cnt = 0
for i in datiWBCNeg["dev standard"]:
    icMinWBCNeg.append(datiWBCNeg["media"][cnt] - icWBCNeg[cnt])
    cnt += 1
icMaxWBCNeg = []
cnt = 0
for i in datiWBCNeg["dev standard"]:
    icMaxWBCNeg.append(datiWBCNeg["media"][cnt] + icWBCNeg[cnt])
    cnt += 1

# faccio una serie di operazioni per rendere cnetrare i punti rieptto agli intervalli di confidenza e fare in modo
# che trasparenza e altezza dell'intervallo siano corretti

ordinateMin = []
ordinateMax = []
for i in range(1, len(datiWBCNeg["giorno"])):
    ordinateMin.append(calcolaOrdinata([datiWBCNeg["giorno"][i - 1], icMinWBCNeg[i - 1]],
                                       [datiWBCNeg["giorno"][i], icMinWBCNeg[i]], datiWBCNeg["giorno"][i - 1] + 0.5))
    ordinateMax.append(calcolaOrdinata([datiWBCNeg["giorno"][i - 1], icMaxWBCNeg[i - 1]],
                                       [datiWBCNeg["giorno"][i], icMaxWBCNeg[i]], datiWBCNeg["giorno"][i - 1] + 0.5))
giorni = [20]
for i in datiWBCNeg["giorno"]:
    if i >= 21:
        giorni.append(i - 0.5)
        giorni.append(i)
icMinWBCNegNuovi = []
cnt = 0
for i in range(0, len(icMinWBCNeg) - 1):
    icMinWBCNegNuovi.append(icMinWBCNeg[i])
    icMinWBCNegNuovi.append(ordinateMin[cnt])
    cnt += 1
icMinWBCNegNuovi.append(icMinWBCNeg[len(icMinWBCNeg) - 1])

icMaxWBCNegNuovi = []
cnt = 0
for i in range(0, len(icMaxWBCNeg) - 1):
    icMaxWBCNegNuovi.append(icMaxWBCNeg[i])
    icMaxWBCNegNuovi.append(ordinateMax[cnt])
    cnt += 1
icMaxWBCNegNuovi.append(icMaxWBCNeg[len(icMaxWBCNeg) - 1])

# infine, aggiorno gli nWBC
nWBCNegNuovi = [nWBCNeg[0]]
# il primo e l'ultimo valore non vanno duplicati
for i in range(1, len(nWBCNeg) - 1):
    nWBCNegNuovi.append(nWBCNeg[i])
    nWBCNegNuovi.append(nWBCNeg[i])
nWBCNegNuovi.append(nWBCNeg[len(nWBCNeg) - 1])
print(nWBCNeg)
print(nWBCNegNuovi)

for b in range(0, len(nWBCNegNuovi)):
    i = nWBCNegNuovi[b]
    if i <= 5:
        alpha = 0.08
    if i >= 30:
        alpha = 0.45
    if (i > 5) & (i < 30):
        alpha = i / 66.67
    if i == 0:
        alpha = 0
    facecolor = (0, 0.25, 1, alpha)
    ax.fill_between(giorni[b:(b + 2)], icMinWBCNegNuovi[b: (b + 2)], icMaxWBCNegNuovi[b: (b + 2)],
                    facecolor=facecolor, linewidths=0.0)
# coloro di bianco la fascia che devo saltare
ax.fill_between([81.2, 89], [0, 0], [20, 20], facecolor=(1, 1, 1, 1), linewidths=0.0, zorder=5.0)

# ne faccio uno solo per la legenda, ma non lo mostro
ax.fill_between(giorni[1:(1 + 2)], icMinWBCPosNuovi[1: (1 + 2)], icMaxWBCPosNuovi[1: (1 + 2)],
                facecolor=(0, 0.25, 1, 0.5), label="Negative 95% confidence interval").set_visible(False)

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
ax.text(8, -9, "F", size=10)
ax.text(23.6, -9, "M", size=10)
ax.text(54.7, -9, "A", size=10)
ax.text(84.6, -9, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda
plt.legend(loc="upper left", ncol=1, fancybox=True, frameon=False)
plt.show()



# passo ad AST
# calcolo medie e deviazioni standard dei positivi
medieASTPos = []
stdevASTPos = []
listaSupporto = []

cnt = 13  # indica il giorno da cui parte la lista di supporto
nASTPos = []
for i in range(16, 803):
    if i in indiciCambioGiornoPos:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(datasetPos["AST"][i])
        listaSupporto = aggiornaListaSupportoPos(indiciCambioGiornoPos[cnt], listaSupporto,
                                                 indiciCambioGiornoPos[cnt + 7], "AST")
        if (cnt < 74) or (cnt > 80):
            # elimino gli zeri dalla lista
            listaSupporto = [i for i in listaSupporto if i != 0]
            nASTPos.append(len(listaSupporto))
            medieASTPos.append(mean(listaSupporto))
            if len(listaSupporto) <= 1:
                stdevASTPos.append(0)
            else:
                stdevASTPos.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(datasetPos["AST"][i])

# aggiungo l'ultimo valore mancante alle due liste
medieASTPos.append(40.75)
stdevASTPos.append(4.57)

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno
dict = {
    "giorno": np.arange(21, 82).tolist() + np.arange(89, 115).tolist(),
    "media": medieASTPos,
    "dev standard": stdevASTPos
}
datiASTPos = pd.DataFrame(dict)

# calcolo medie e deviazioni standard dei negativi
medieASTNeg = []
stdevASTNeg = []
listaSupporto = []

cnt = 12  # indica il giorno da cui parte la lista di supporto
nASTNeg = []
for i in range(0, 732):
    if i in indiciCambioGiornoNeg:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(datasetNeg["AST"][i])
        listaSupporto = aggiornaListaSupportoNeg(indiciCambioGiornoNeg[cnt], listaSupporto,
                                                 indiciCambioGiornoNeg[cnt + 7], "AST")
        if (cnt < 74) or (cnt > 80):
            # elimino gli zeri dalla lista
            listaSupporto = [i for i in listaSupporto if i != 0]
            nASTNeg.append(len(listaSupporto))
            medieASTNeg.append(mean(listaSupporto))
            if len(listaSupporto) <= 1:
                stdevASTNeg.append(0)
            else:
                stdevASTNeg.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(datasetNeg["AST"][i])

# aggiungo gli ultimi due valori mancanti
medieASTNeg += [36.18, 36.0]
stdevASTNeg += [41.39, 40.88]

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno
dict = {
    "giorno": np.arange(20, 82).tolist() + np.arange(89, 115).tolist(),
    "media": medieASTNeg,
    "dev standard": stdevASTNeg
}

datiASTNeg = pd.DataFrame(dict)

# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x="giorno", y="media", data=datiASTPos, marker=".", markersize=14, markeredgecolor='None',
             markerfacecolor="#b80f0a", color="#b80f0a", label="Positive")
sns.lineplot(x="giorno", y="media", data=datiASTNeg, marker=".", markersize=14, markeredgecolor="None",
             markerfacecolor="#1034a6", color="#1034a6", label="Negative")

# calcolo gli intervalli di confidenza dei positivi e li disegno
# aggiungo l'ultimo valore mancante alla lista degli n già popolata in precedenza
nASTPos.append(2)

icASTPos = []
cnt = 0
for i in datiASTPos["dev standard"]:
    icASTPos.append(1.96 * i / sqrt(nASTPos[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinASTPos = []
cnt = 0
for i in datiASTPos["dev standard"]:
    icMinASTPos.append(datiASTPos["media"][cnt] - icASTPos[cnt])
    cnt += 1
icMaxASTPos = []
cnt = 0
for i in datiASTPos["dev standard"]:
    icMaxASTPos.append(datiASTPos["media"][cnt] + icASTPos[cnt])
    cnt += 1


# faccio una serie di operazioni per mostrare gli ic centrati rispetto al punto con la dimensione e la trasparenza
# corrette
def calcolaOrdinata(P, Q, x):
    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = a * (P[0]) + b * (P[1])
    if b < 0:
        y = (c - (a * x)) / b
    else:
        y = ((a * x) - c) / b
    return y


ordinateMin = []
ordinateMax = []
for i in range(1, len(datiASTPos["giorno"])):
    ordinateMin.append(calcolaOrdinata([datiASTPos["giorno"][i - 1], icMinASTPos[i - 1]],
                                       [datiASTPos["giorno"][i], icMinASTPos[i]], datiASTPos["giorno"][i - 1] + 0.5))
    ordinateMax.append(calcolaOrdinata([datiASTPos["giorno"][i - 1], icMaxASTPos[i - 1]],
                                       [datiASTPos["giorno"][i], icMaxASTPos[i]], datiASTPos["giorno"][i - 1] + 0.5))
# ora inserisco i nuovi valori nei giorni e negli ic

giorni = [21]
for i in datiASTPos["giorno"]:
    if i >= 22:
        giorni.append(i - 0.5)
        giorni.append(i)

icMinASTPosNuovi = []
cnt = 0
for i in range(0, len(icMinASTPos) - 1):
    icMinASTPosNuovi.append(icMinASTPos[i])
    icMinASTPosNuovi.append(ordinateMin[cnt])
    cnt += 1
icMinASTPosNuovi.append(icMinASTPos[len(icMinASTPos) - 1])

icMaxASTPosNuovi = []
cnt = 0
for i in range(0, len(icMaxASTPos) - 1):
    icMaxASTPosNuovi.append(icMaxASTPos[i])
    icMaxASTPosNuovi.append(ordinateMax[cnt])
    cnt += 1
icMaxASTPosNuovi.append(icMaxASTPos[len(icMaxASTPos) - 1])

# infine, aggiorno gli nAST
nASTPosNuovi = [nASTPos[0]]
# il primo e l'ultimo non vanno duplicati
for i in range(1, len(nASTPos) - 1):
    nASTPosNuovi.append(nASTPos[i])
    nASTPosNuovi.append(nASTPos[i])
nASTPosNuovi.append(nASTPos[len(nASTPos) - 1])

plt.margins(0.05)
for b in range(0, len(nASTPosNuovi)):
    i = nASTPosNuovi[b]
    if i <= 5:
        alpha = 0.08
    if i >= 30:
        alpha = 0.45
    if (i > 5) & (i < 30):
        alpha = i / 66.67
    # se non ci sono dati nei 7 giorni precedenti non mostro nulla
    if i == 0:
        alpha = 0
    facecolor = (1, 0, 0, alpha)
    ax.fill_between(giorni[b:(b + 2)], icMinASTPosNuovi[b: (b + 2)], icMaxASTPosNuovi[b: (b + 2)],
                    facecolor=facecolor, edgecolor=(0, 0.25, 1, 0), linewidths=0)
# ne faccio uno solo per la legenda, ma non lo mostro
ax.fill_between(giorni[1:(1 + 2)], icMinASTPosNuovi[1: (1 + 2)], icMaxASTPosNuovi[1: (1 + 2)], facecolor=(1, 0, 0, 0.5),
                label="Positive 95% confidence interval").set_visible(False)

# calcolo gli intervalli di confidenza dei negativi e li disegno
# aggiungo gli ultimi sue valori mancanti alla lista degli n già popolata in precedenza
nASTNeg += [41, 39]

icASTNeg = []
cnt = 0
for i in datiASTNeg["dev standard"]:
    icASTNeg.append(1.96 * i / sqrt(nASTNeg[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinASTNeg = []
cnt = 0
for i in datiASTNeg["dev standard"]:
    icMinASTNeg.append(datiASTNeg["media"][cnt] - icASTNeg[cnt])
    cnt += 1
icMaxASTNeg = []
cnt = 0
for i in datiASTNeg["dev standard"]:
    icMaxASTNeg.append(datiASTNeg["media"][cnt] + icASTNeg[cnt])
    cnt += 1

# faccio una serie di operazioni per rendere cnetrare i punti rieptto agli intervalli di confidenza e fare in modo
# che trasparenza e altezza dell'intervallo siano corretti

ordinateMin = []
ordinateMax = []
for i in range(1, len(datiASTNeg["giorno"])):
    ordinateMin.append(calcolaOrdinata([datiASTNeg["giorno"][i - 1], icMinASTNeg[i - 1]],
                                       [datiASTNeg["giorno"][i], icMinASTNeg[i]], datiASTNeg["giorno"][i - 1] + 0.5))
    ordinateMax.append(calcolaOrdinata([datiASTNeg["giorno"][i - 1], icMaxASTNeg[i - 1]],
                                       [datiASTNeg["giorno"][i], icMaxASTNeg[i]], datiASTNeg["giorno"][i - 1] + 0.5))
giorni = [20]
for i in datiASTNeg["giorno"]:
    if i >= 21:
        giorni.append(i - 0.5)
        giorni.append(i)
icMinASTNegNuovi = []
cnt = 0
for i in range(0, len(icMinASTNeg) - 1):
    icMinASTNegNuovi.append(icMinASTNeg[i])
    icMinASTNegNuovi.append(ordinateMin[cnt])
    cnt += 1
icMinASTNegNuovi.append(icMinASTNeg[len(icMinASTNeg) - 1])

icMaxASTNegNuovi = []
cnt = 0
for i in range(0, len(icMaxASTNeg) - 1):
    icMaxASTNegNuovi.append(icMaxASTNeg[i])
    icMaxASTNegNuovi.append(ordinateMax[cnt])
    cnt += 1
icMaxASTNegNuovi.append(icMaxASTNeg[len(icMaxASTNeg) - 1])

# infine, aggiorno gli nAST
nASTNegNuovi = [nASTNeg[0]]
# il primo e l'ultimo valore non vanno duplicati
for i in range(1, len(nASTNeg) - 1):
    nASTNegNuovi.append(nASTNeg[i])
    nASTNegNuovi.append(nASTNeg[i])
nASTNegNuovi.append(nASTNeg[len(nASTNeg) - 1])
print(nASTNeg)
print(nASTNegNuovi)

for b in range(0, len(nASTNegNuovi)):
    i = nASTNegNuovi[b]
    if i <= 5:
        alpha = 0.08
    if i >= 30:
        alpha = 0.45
    if (i > 5) & (i < 30):
        alpha = i / 66.67
    if i == 0:
        alpha = 0
    facecolor = (0, 0.25, 1, alpha)
    ax.fill_between(giorni[b:(b + 2)], icMinASTNegNuovi[b: (b + 2)], icMaxASTNegNuovi[b: (b + 2)],
                    facecolor=facecolor, linewidths=0.0)
# coloro di bianco la fascia che devo saltare
ax.fill_between([81.2, 89], [0, 0], [120, 120], facecolor=(1, 1, 1, 1), linewidths=0.0, zorder=5.0)

# ne faccio uno solo per la legenda, ma non lo mostro
ax.fill_between(giorni[1:(1 + 2)], icMinASTPosNuovi[1: (1 + 2)], icMaxASTPosNuovi[1: (1 + 2)],
                facecolor=(0, 0.25, 1, 0.5), label="Negative 95% confidence interval").set_visible(False)

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
ax.text(8, -50, "F", size=10)
ax.text(23.6, -50, "M", size=10)
ax.text(54.7, -50, "A", size=10)
ax.text(84.6, -50, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda
plt.legend(loc="upper right", ncol=1, fancybox=True, frameon=False)
plt.show()


# passo a LDH
# calcolo medie e deviazioni standard dei positivi
medieLDHPos = []
stdevLDHPos = []
listaSupporto = []

cnt = 13  # indica il giorno da cui parte la lista di supporto
nLDHPos = []
for i in range(16, 803):
    if i in indiciCambioGiornoPos:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(datasetPos["LDH"][i])
        listaSupporto = aggiornaListaSupportoPos(indiciCambioGiornoPos[cnt], listaSupporto,
                                                 indiciCambioGiornoPos[cnt + 7], "LDH")
        if (cnt < 74) or (cnt > 80):
            # elimino gli zeri dalla lista
            listaSupporto = [i for i in listaSupporto if i != 0]
            nLDHPos.append(len(listaSupporto))
            medieLDHPos.append(mean(listaSupporto))
            if len(listaSupporto) <= 1:
                stdevLDHPos.append(0)
            else:
                stdevLDHPos.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(datasetPos["LDH"][i])

# aggiungo l'ultimo valore mancante alle due liste
medieLDHPos.append(342.75)
stdevLDHPos.append(148.85)

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno
dict = {
    "giorno": np.arange(21, 82).tolist() + np.arange(89, 115).tolist(),
    "media": medieLDHPos,
    "dev standard": stdevLDHPos
}
datiLDHPos = pd.DataFrame(dict)

# calcolo medie e deviazioni standard dei negativi
medieLDHNeg = []
stdevLDHNeg = []
listaSupporto = []

cnt = 12  # indica il giorno da cui parte la lista di supporto
nLDHNeg = []
for i in range(0, 732):
    if i in indiciCambioGiornoNeg:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(datasetNeg["LDH"][i])
        listaSupporto = aggiornaListaSupportoNeg(indiciCambioGiornoNeg[cnt], listaSupporto,
                                                 indiciCambioGiornoNeg[cnt + 7], "LDH")
        if (cnt < 74) or (cnt > 80):
            # elimino gli zeri dalla lista
            listaSupporto = [i for i in listaSupporto if i != 0]
            nLDHNeg.append(len(listaSupporto))
            medieLDHNeg.append(mean(listaSupporto))
            if len(listaSupporto) <= 1:
                stdevLDHNeg.append(0)
            else:
                stdevLDHNeg.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(datasetNeg["LDH"][i])

# aggiungo gli ultimi due valori mancanti
medieLDHNeg += [253.56, 255.59]
stdevLDHNeg += [90.21, 88.84]

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno
dict = {
    "giorno": np.arange(20, 82).tolist() + np.arange(89, 115).tolist(),
    "media": medieLDHNeg,
    "dev standard": stdevLDHNeg
}

datiLDHNeg = pd.DataFrame(dict)

# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x="giorno", y="media", data=datiLDHPos, marker=".", markersize=14, markeredgecolor='None',
             markerfacecolor="#b80f0a", color="#b80f0a", label="Positive")
sns.lineplot(x="giorno", y="media", data=datiLDHNeg, marker=".", markersize=14, markeredgecolor="None",
             markerfacecolor="#1034a6", color="#1034a6", label="Negative")

# calcolo gli intervalli di confidenza dei positivi e li disegno
# aggiungo l'ultimo valore mancante alla lista degli n già popolata in precedenza
nLDHPos.append(2)

icLDHPos = []
cnt = 0
for i in datiLDHPos["dev standard"]:
    icLDHPos.append(1.96 * i / sqrt(nLDHPos[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinLDHPos = []
cnt = 0
for i in datiLDHPos["dev standard"]:
    icMinLDHPos.append(datiLDHPos["media"][cnt] - icLDHPos[cnt])
    cnt += 1
icMaxLDHPos = []
cnt = 0
for i in datiLDHPos["dev standard"]:
    icMaxLDHPos.append(datiLDHPos["media"][cnt] + icLDHPos[cnt])
    cnt += 1


# faccio una serie di operazioni per mostrare gli ic centrati rispetto al punto con la dimensione e la trasparenza
# corrette
def calcolaOrdinata(P, Q, x):
    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = a * (P[0]) + b * (P[1])
    if b < 0:
        y = (c - (a * x)) / b
    else:
        y = ((a * x) - c) / b
    return y


ordinateMin = []
ordinateMax = []
for i in range(1, len(datiLDHPos["giorno"])):
    ordinateMin.append(calcolaOrdinata([datiLDHPos["giorno"][i - 1], icMinLDHPos[i - 1]],
                                       [datiLDHPos["giorno"][i], icMinLDHPos[i]], datiLDHPos["giorno"][i - 1] + 0.5))
    ordinateMax.append(calcolaOrdinata([datiLDHPos["giorno"][i - 1], icMaxLDHPos[i - 1]],
                                       [datiLDHPos["giorno"][i], icMaxLDHPos[i]], datiLDHPos["giorno"][i - 1] + 0.5))
# ora inserisco i nuovi valori nei giorni e negli ic

giorni = [21]
for i in datiLDHPos["giorno"]:
    if i >= 22:
        giorni.append(i - 0.5)
        giorni.append(i)

icMinLDHPosNuovi = []
cnt = 0
for i in range(0, len(icMinLDHPos) - 1):
    icMinLDHPosNuovi.append(icMinLDHPos[i])
    icMinLDHPosNuovi.append(ordinateMin[cnt])
    cnt += 1
icMinLDHPosNuovi.append(icMinLDHPos[len(icMinLDHPos) - 1])

icMaxLDHPosNuovi = []
cnt = 0
for i in range(0, len(icMaxLDHPos) - 1):
    icMaxLDHPosNuovi.append(icMaxLDHPos[i])
    icMaxLDHPosNuovi.append(ordinateMax[cnt])
    cnt += 1
icMaxLDHPosNuovi.append(icMaxLDHPos[len(icMaxLDHPos) - 1])

# infine, aggiorno gli nLDH
nLDHPosNuovi = [nLDHPos[0]]
# il primo e l'ultimo non vanno duplicati
for i in range(1, len(nLDHPos) - 1):
    nLDHPosNuovi.append(nLDHPos[i])
    nLDHPosNuovi.append(nLDHPos[i])
nLDHPosNuovi.append(nLDHPos[len(nLDHPos) - 1])

plt.margins(0.05)
for b in range(0, len(nLDHPosNuovi)):
    i = nLDHPosNuovi[b]
    if i <= 5:
        alpha = 0.08
    if i >= 30:
        alpha = 0.45
    if (i > 5) & (i < 30):
        alpha = i / 66.67
    # se non ci sono dati nei 7 giorni precedenti non mostro nulla
    if i == 0:
        alpha = 0
    facecolor = (1, 0, 0, alpha)
    ax.fill_between(giorni[b:(b + 2)], icMinLDHPosNuovi[b: (b + 2)], icMaxLDHPosNuovi[b: (b + 2)],
                    facecolor=facecolor, edgecolor=(0, 0.25, 1, 0), linewidths=0)
# ne faccio uno solo per la legenda, ma non lo mostro
ax.fill_between(giorni[1:(1 + 2)], icMinLDHPosNuovi[1: (1 + 2)], icMaxLDHPosNuovi[1: (1 + 2)], facecolor=(1, 0, 0, 0.5),
                label="Positive 95% confidence interval").set_visible(False)

# calcolo gli intervalli di confidenza dei negativi e li disegno
# aggiungo gli ultimi sue valori mancanti alla lista degli n già popolata in precedenza
nLDHNeg += [38, 40]

icLDHNeg = []
cnt = 0
for i in datiLDHNeg["dev standard"]:
    icLDHNeg.append(1.96 * i / sqrt(nLDHNeg[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinLDHNeg = []
cnt = 0
for i in datiLDHNeg["dev standard"]:
    icMinLDHNeg.append(datiLDHNeg["media"][cnt] - icLDHNeg[cnt])
    cnt += 1
icMaxLDHNeg = []
cnt = 0
for i in datiLDHNeg["dev standard"]:
    icMaxLDHNeg.append(datiLDHNeg["media"][cnt] + icLDHNeg[cnt])
    cnt += 1

# faccio una serie di operazioni per rendere cnetrare i punti rieptto agli intervalli di confidenza e fare in modo
# che trasparenza e altezza dell'intervallo siano corretti

ordinateMin = []
ordinateMax = []
for i in range(1, len(datiLDHNeg["giorno"])):
    ordinateMin.append(calcolaOrdinata([datiLDHNeg["giorno"][i - 1], icMinLDHNeg[i - 1]],
                                       [datiLDHNeg["giorno"][i], icMinLDHNeg[i]], datiLDHNeg["giorno"][i - 1] + 0.5))
    ordinateMax.append(calcolaOrdinata([datiLDHNeg["giorno"][i - 1], icMaxLDHNeg[i - 1]],
                                       [datiLDHNeg["giorno"][i], icMaxLDHNeg[i]], datiLDHNeg["giorno"][i - 1] + 0.5))
giorni = [20]
for i in datiLDHNeg["giorno"]:
    if i >= 21:
        giorni.append(i - 0.5)
        giorni.append(i)
icMinLDHNegNuovi = []
cnt = 0
for i in range(0, len(icMinLDHNeg) - 1):
    icMinLDHNegNuovi.append(icMinLDHNeg[i])
    icMinLDHNegNuovi.append(ordinateMin[cnt])
    cnt += 1
icMinLDHNegNuovi.append(icMinLDHNeg[len(icMinLDHNeg) - 1])

icMaxLDHNegNuovi = []
cnt = 0
for i in range(0, len(icMaxLDHNeg) - 1):
    icMaxLDHNegNuovi.append(icMaxLDHNeg[i])
    icMaxLDHNegNuovi.append(ordinateMax[cnt])
    cnt += 1
icMaxLDHNegNuovi.append(icMaxLDHNeg[len(icMaxLDHNeg) - 1])

# infine, aggiorno gli nLDH
nLDHNegNuovi = [nLDHNeg[0]]
# il primo e l'ultimo valore non vanno duplicati
for i in range(1, len(nLDHNeg) - 1):
    nLDHNegNuovi.append(nLDHNeg[i])
    nLDHNegNuovi.append(nLDHNeg[i])
nLDHNegNuovi.append(nLDHNeg[len(nLDHNeg) - 1])
print(nLDHNeg)
print(nLDHNegNuovi)

for b in range(0, len(nLDHNegNuovi)):
    i = nLDHNegNuovi[b]
    if i <= 5:
        alpha = 0.08
    if i >= 30:
        alpha = 0.45
    if (i > 5) & (i < 30):
        alpha = i / 66.67
    if i == 0:
        alpha = 0
    facecolor = (0, 0.25, 1, alpha)
    ax.fill_between(giorni[b:(b + 2)], icMinLDHNegNuovi[b: (b + 2)], icMaxLDHNegNuovi[b: (b + 2)],
                    facecolor=facecolor, linewidths=0.0)
# coloro di bianco la fascia che devo saltare
ax.fill_between([81.2, 89], [100, 100], [500, 500], facecolor=(1, 1, 1, 1), linewidths=0.0, zorder=5.0)

# ne faccio uno solo per la legenda, ma non lo mostro
ax.fill_between(giorni[1:(1 + 2)], icMinLDHPosNuovi[1: (1 + 2)], icMaxLDHPosNuovi[1: (1 + 2)],
                facecolor=(0, 0.25, 1, 0.5), label="Negative 95% confidence interval").set_visible(False)

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
ax.text(8, -10, "F", size=10)
ax.text(23.6, -10, "M", size=10)
ax.text(54.7, -10, "A", size=10)
ax.text(84.6, -10, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda
plt.legend(loc="upper right", ncol=1, fancybox=True, frameon=False)
plt.show()


# passo a CA
# calcolo medie e deviazioni standard dei positivi
medieCAPos = []
stdevCAPos = []
listaSupporto = []

cnt = 13  # indica il giorno da cui parte la lista di supporto
nCAPos = []
for i in range(16, 803):
    if i in indiciCambioGiornoPos:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(datasetPos["CA"][i])
        listaSupporto = aggiornaListaSupportoPos(indiciCambioGiornoPos[cnt], listaSupporto,
                                                 indiciCambioGiornoPos[cnt + 7], "CA")
        if (cnt < 74) or (cnt > 80):
            # elimino gli zeri dalla lista
            listaSupporto = [i for i in listaSupporto if i != 0]
            nCAPos.append(len(listaSupporto))
            medieCAPos.append(mean(listaSupporto))
            if len(listaSupporto) <= 1:
                stdevCAPos.append(0)
            else:
                stdevCAPos.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(datasetPos["CA"][i])

# aggiungo l'ultimo valore mancante alle due liste
medieCAPos.append(2.0017)
stdevCAPos.append(0.083)

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno
dict = {
    "giorno": np.arange(21, 82).tolist() + np.arange(89, 115).tolist(),
    "media": medieCAPos,
    "dev standard": stdevCAPos
}
datiCAPos = pd.DataFrame(dict)

# calcolo medie e deviazioni standard dei negativi
medieCANeg = []
stdevCANeg = []
listaSupporto = []

cnt = 12  # indica il giorno da cui parte la lista di supporto
nCANeg = []
for i in range(0, 732):
    if i in indiciCambioGiornoNeg:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(datasetNeg["CA"][i])
        listaSupporto = aggiornaListaSupportoNeg(indiciCambioGiornoNeg[cnt], listaSupporto,
                                                 indiciCambioGiornoNeg[cnt + 7], "CA")
        if (cnt < 74) or (cnt > 80):
            # elimino gli zeri dalla lista
            listaSupporto = [i for i in listaSupporto if i != 0]
            nCANeg.append(len(listaSupporto))
            medieCANeg.append(mean(listaSupporto))
            if len(listaSupporto) <= 1:
                stdevCANeg.append(0)
            else:
                stdevCANeg.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(datasetNeg["CA"][i])

# aggiungo gli ultimi due valori mancanti
medieCANeg += [2.238, 2.252]
stdevCANeg += [0.19, 0.19]

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno
dict = {
    "giorno": np.arange(20, 82).tolist() + np.arange(89, 115).tolist(),
    "media": medieCANeg,
    "dev standard": stdevCANeg
}

datiCANeg = pd.DataFrame(dict)

# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x="giorno", y="media", data=datiCAPos, marker=".", markersize=14, markeredgecolor='None',
             markerfacecolor="#b80f0a", color="#b80f0a", label="Positive")
sns.lineplot(x="giorno", y="media", data=datiCANeg, marker=".", markersize=14, markeredgecolor="None",
             markerfacecolor="#1034a6", color="#1034a6", label="Negative")

# calcolo gli intervalli di confidenza dei positivi e li disegno
# aggiungo l'ultimo valore mancante alla lista degli n già popolata in precedenza
nCAPos.append(2)

icCAPos = []
cnt = 0
for i in datiCAPos["dev standard"]:
    icCAPos.append(1.96 * i / sqrt(nCAPos[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinCAPos = []
cnt = 0
for i in datiCAPos["dev standard"]:
    icMinCAPos.append(datiCAPos["media"][cnt] - icCAPos[cnt])
    cnt += 1
icMaxCAPos = []
cnt = 0
for i in datiCAPos["dev standard"]:
    icMaxCAPos.append(datiCAPos["media"][cnt] + icCAPos[cnt])
    cnt += 1


# faccio una serie di operazioni per mostrare gli ic centrati rispetto al punto con la dimensione e la trasparenza
# corrette
def calcolaOrdinata(P, Q, x):
    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = a * (P[0]) + b * (P[1])
    if b < 0:
        y = (c - (a * x)) / b
    else:
        y = ((a * x) - c) / b
    return y


ordinateMin = []
ordinateMax = []
for i in range(1, len(datiCAPos["giorno"])):
    ordinateMin.append(calcolaOrdinata([datiCAPos["giorno"][i - 1], icMinCAPos[i - 1]],
                                       [datiCAPos["giorno"][i], icMinCAPos[i]], datiCAPos["giorno"][i - 1] + 0.5))
    ordinateMax.append(calcolaOrdinata([datiCAPos["giorno"][i - 1], icMaxCAPos[i - 1]],
                                       [datiCAPos["giorno"][i], icMaxCAPos[i]], datiCAPos["giorno"][i - 1] + 0.5))
# ora inserisco i nuovi valori nei giorni e negli ic

giorni = [21]
for i in datiCAPos["giorno"]:
    if i >= 22:
        giorni.append(i - 0.5)
        giorni.append(i)

icMinCAPosNuovi = []
cnt = 0
for i in range(0, len(icMinCAPos) - 1):
    icMinCAPosNuovi.append(icMinCAPos[i])
    icMinCAPosNuovi.append(ordinateMin[cnt])
    cnt += 1
icMinCAPosNuovi.append(icMinCAPos[len(icMinCAPos) - 1])

icMaxCAPosNuovi = []
cnt = 0
for i in range(0, len(icMaxCAPos) - 1):
    icMaxCAPosNuovi.append(icMaxCAPos[i])
    icMaxCAPosNuovi.append(ordinateMax[cnt])
    cnt += 1
icMaxCAPosNuovi.append(icMaxCAPos[len(icMaxCAPos) - 1])

# infine, aggiorno gli nCA
nCAPosNuovi = [nCAPos[0]]
# il primo e l'ultimo non vanno duplicati
for i in range(1, len(nCAPos) - 1):
    nCAPosNuovi.append(nCAPos[i])
    nCAPosNuovi.append(nCAPos[i])
nCAPosNuovi.append(nCAPos[len(nCAPos) - 1])

plt.margins(0.05)
for b in range(0, len(nCAPosNuovi)):
    i = nCAPosNuovi[b]
    if i <= 5:
        alpha = 0.08
    if i >= 30:
        alpha = 0.45
    if (i > 5) & (i < 30):
        alpha = i / 66.67
    # se non ci sono dati nei 7 giorni precedenti non mostro nulla
    if i == 0:
        alpha = 0
    facecolor = (1, 0, 0, alpha)
    ax.fill_between(giorni[b:(b + 2)], icMinCAPosNuovi[b: (b + 2)], icMaxCAPosNuovi[b: (b + 2)],
                    facecolor=facecolor, edgecolor=(0, 0.25, 1, 0), linewidths=0)
# ne faccio uno solo per la legenda, ma non lo mostro
ax.fill_between(giorni[1:(1 + 2)], icMinCAPosNuovi[1: (1 + 2)], icMaxCAPosNuovi[1: (1 + 2)], facecolor=(1, 0, 0, 0.5),
                label="Positive 95% confidence interval").set_visible(False)

# calcolo gli intervalli di confidenza dei negativi e li disegno
# aggiungo gli ultimi sue valori mancanti alla lista degli n già popolata in precedenza
nCANeg += [38, 40]

icCANeg = []
cnt = 0
for i in datiCANeg["dev standard"]:
    icCANeg.append(1.96 * i / sqrt(nCANeg[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinCANeg = []
cnt = 0
for i in datiCANeg["dev standard"]:
    icMinCANeg.append(datiCANeg["media"][cnt] - icCANeg[cnt])
    cnt += 1
icMaxCANeg = []
cnt = 0
for i in datiCANeg["dev standard"]:
    icMaxCANeg.append(datiCANeg["media"][cnt] + icCANeg[cnt])
    cnt += 1

# faccio una serie di operazioni per rendere cnetrare i punti rieptto agli intervalli di confidenza e fare in modo
# che trasparenza e altezza dell'intervallo siano corretti

ordinateMin = []
ordinateMax = []
for i in range(1, len(datiCANeg["giorno"])):
    ordinateMin.append(calcolaOrdinata([datiCANeg["giorno"][i - 1], icMinCANeg[i - 1]],
                                       [datiCANeg["giorno"][i], icMinCANeg[i]], datiCANeg["giorno"][i - 1] + 0.5))
    ordinateMax.append(calcolaOrdinata([datiCANeg["giorno"][i - 1], icMaxCANeg[i - 1]],
                                       [datiCANeg["giorno"][i], icMaxCANeg[i]], datiCANeg["giorno"][i - 1] + 0.5))
giorni = [20]
for i in datiCANeg["giorno"]:
    if i >= 21:
        giorni.append(i - 0.5)
        giorni.append(i)
icMinCANegNuovi = []
cnt = 0
for i in range(0, len(icMinCANeg) - 1):
    icMinCANegNuovi.append(icMinCANeg[i])
    icMinCANegNuovi.append(ordinateMin[cnt])
    cnt += 1
icMinCANegNuovi.append(icMinCANeg[len(icMinCANeg) - 1])

icMaxCANegNuovi = []
cnt = 0
for i in range(0, len(icMaxCANeg) - 1):
    icMaxCANegNuovi.append(icMaxCANeg[i])
    icMaxCANegNuovi.append(ordinateMax[cnt])
    cnt += 1
icMaxCANegNuovi.append(icMaxCANeg[len(icMaxCANeg) - 1])

# infine, aggiorno gli nCA
nCANegNuovi = [nCANeg[0]]
# il primo e l'ultimo valore non vanno duplicati
for i in range(1, len(nCANeg) - 1):
    nCANegNuovi.append(nCANeg[i])
    nCANegNuovi.append(nCANeg[i])
nCANegNuovi.append(nCANeg[len(nCANeg) - 1])
print(nCANeg)
print(nCANegNuovi)

for b in range(0, len(nCANegNuovi)):
    i = nCANegNuovi[b]
    if i <= 5:
        alpha = 0.08
    if i >= 30:
        alpha = 0.45
    if (i > 5) & (i < 30):
        alpha = i / 66.67
    if i == 0:
        alpha = 0
    facecolor = (0, 0.25, 1, alpha)
    ax.fill_between(giorni[b:(b + 2)], icMinCANegNuovi[b: (b + 2)], icMaxCANegNuovi[b: (b + 2)],
                    facecolor=facecolor, linewidths=0.0)
# coloro di bianco la fascia che devo saltare
ax.fill_between([81.2, 89], [1.8, 1.8], [2.5, 2.5], facecolor=(1, 1, 1, 1), linewidths=0.0, zorder=5.0)

# ne faccio uno solo per la legenda, ma non lo mostro
ax.fill_between(giorni[1:(1 + 2)], icMinCAPosNuovi[1: (1 + 2)], icMaxCAPosNuovi[1: (1 + 2)],
                facecolor=(0, 0.25, 1, 0.5), label="Negative 95% confidence interval").set_visible(False)

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
ax.text(8, 1.51, "F", size=10)
ax.text(23.6, 1.51, "M", size=10)
ax.text(54.7, 1.51, "A", size=10)
ax.text(84.6, 1.51, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda
plt.legend(loc="upper left", ncol=1, fancybox=True, frameon=False)
plt.show()



# faccio i due grafici delle età (media + sd uno, mediana + iqr l'altro)
medieEtàPos = []
stdevEtàPos = []
medianeEtàPos = []
iqrEtàPos = []
listaSupporto = []

cnt = 13  # indica il giorno da cui parte la lista di supporto
for i in range(16, 803):
    if i in indiciCambioGiornoPos:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(datasetPos["Age"][i])
        listaSupporto = aggiornaListaSupportoPos(indiciCambioGiornoPos[cnt], listaSupporto,
                                              indiciCambioGiornoPos[cnt + 7], "Age")
        if (cnt < 70) or (cnt > 80):
            # elimino gli zeri dalla lista
            listaSupporto = [i for i in listaSupporto if i != 0]
            medieEtàPos.append(mean(listaSupporto))
            medianeEtàPos.append(np.median(listaSupporto))
            iqrEtàPos.append(iqr(listaSupporto))
            if len(listaSupporto) <= 1:
                stdevEtàPos.append(0)
            else:
                stdevEtàPos.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(datasetPos["Age"][i])

# aggiungo l'ultimo dato mancante
medieEtàPos.append(82.5)
stdevEtàPos.append(9.19)
medianeEtàPos.append(82.5)
iqrEtàPos.append(6.5)

medieEtàNeg = []
stdevEtàNeg = []
medianeEtàNeg = []
iqrEtàNeg = []
listaSupporto = []

cnt = 12  # indica il giorno da cui parte la lista di supporto
for i in range(0, 732):
    if i in indiciCambioGiornoNeg:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(datasetNeg["Age"][i])
        listaSupporto = aggiornaListaSupportoNeg(indiciCambioGiornoNeg[cnt], listaSupporto,
                                              indiciCambioGiornoNeg[cnt + 7], "Age")
        if (cnt < 70) or (cnt > 80):
            # elimino gli zeri dalla lista
            listaSupporto = [i for i in listaSupporto if i != 0]
            medieEtàNeg.append(mean(listaSupporto))
            medianeEtàNeg.append(np.median(listaSupporto))
            iqrEtàNeg.append(iqr(listaSupporto))
            if len(listaSupporto) <= 1:
                stdevEtàNeg.append(0)
            else:
                stdevEtàNeg.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(datasetNeg["Age"][i])

# aggiungo gli ultimi due dati mancanti
medieEtàNeg += [63.42, 62.85]
stdevEtàNeg += [22.6, 22.85]
medianeEtàNeg += [69, 67]
iqrEtàNeg += [34, 33.5]

# inzio col disgnare i due lineplot
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x=np.arange(21, 78).tolist() + np.arange(89, 115).tolist(), y=medieEtàPos, marker=".", markersize=14, markeredgecolor='None',
             markerfacecolor="#b80f0a", color="#b80f0a", label="Positive")
sns.lineplot(x=np.arange(20, 78).tolist() + np.arange(89, 115).tolist(), y=medieEtàNeg, marker=".", markersize=14, markeredgecolor='None',
             markerfacecolor="#1034a6", color="#1034a6", label="Negative")

# ora faccio gli intervalli della deviazione standard
stDevMinPos = []
stDevMaxPos = []
cnt = 0
for i in stdevEtàPos:
    stDevMinPos.append(medieEtàPos[cnt] - i)
    stDevMaxPos.append(medieEtàPos[cnt] + i)
    cnt += 1
ax.fill_between(np.arange(21, 78).tolist() + np.arange(89, 115).tolist(), stDevMinPos, stDevMaxPos, facecolor=(1, 0, 0, 0.2))

# creo gli intervalli della deviazione standard
stDevMinNeg = []
stDevMaxNeg = []
cnt = 0
for i in stdevEtàNeg:
    stDevMinNeg.append(medieEtàNeg[cnt] - i)
    stDevMaxNeg.append(medieEtàNeg[cnt] + i)
    cnt += 1
ax.fill_between(np.arange(20, 78).tolist() + np.arange(89, 115).tolist(), stDevMinNeg, stDevMaxNeg, facecolor=(0, 0.25, 1, 0.2))

# coloro di bianco la fascia che devo saltare
ax.fill_between([77.2, 89], [10, 10], [125, 125], facecolor=(1, 1, 1, 1), linewidths=0.0, zorder=5.0)

plt.ylabel("previous 7 days \n mean age (years) \n standard deviation", rotation=0, fontsize=12)
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.08, 0.5)
plt.xlabel("Time (weeks)", fontsize=12)
plt.xticks([38, 69, 99], labels=[])
ax.tick_params(which='major', length=6, color='black')
plt.ylabel("previous 7 days \n mean age (years) \n standard deviation", rotation=0, fontsize=12)
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.08, 0.5)
plt.xlabel("Time (weeks)", fontsize=12)
plt.xticks([38, 69, 99], labels=[])
ax.tick_params(which='major', length=6, color='black')
# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([24, 55, 85], minor=True)
ax.tick_params(which='minor', length=10, color='black')
ax.text(8, -8, "F", size=10)
ax.text(23.6, -8, "M", size=10)
ax.text(54.7, -8, "A", size=10)
ax.text(84.6, -8, "M", size=10)
plt.xlim(7.8, 115.2)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.legend(loc="upper left", ncol=1, fancybox=True, frameon=False)
plt.show()


# ora quello delle mediane
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x=np.arange(21, 78).tolist() + np.arange(89, 115).tolist(), y=medianeEtàPos, marker=".", markersize=14, markeredgecolor='None',
             markerfacecolor="#b80f0a", color="#b80f0a", label="Positive")
sns.lineplot(x=np.arange(20, 78).tolist() + np.arange(89, 115).tolist(), y=medianeEtàNeg, marker=".", markersize=14, markeredgecolor='None',
             markerfacecolor="#1034a6", color="#1034a6", label="Negative")

# creo gli intervalli dello scarto interquartile dei positivi
iqrMinPos = []
iqrMaxPos = []
cnt = 0
for i in iqrEtàPos:
    iqrMinPos.append(medianeEtàPos[cnt] - i)
    iqrMaxPos.append(medianeEtàPos[cnt] + i)
    cnt += 1
ax.fill_between(np.arange(21, 78).tolist() + np.arange(89, 115).tolist(), iqrMinPos, iqrMaxPos, facecolor=(1, 0, 0, 0.2))
# creo gli intervalli dello scarto interquartile dei positivi
iqrMinNeg = []
iqrMaxNeg = []
cnt = 0
for i in iqrEtàNeg:
    iqrMinNeg.append(medianeEtàNeg[cnt] - i)
    iqrMaxNeg.append(medianeEtàNeg[cnt] + i)
    cnt += 1

ax.fill_between(np.arange(20, 78).tolist() + np.arange(89, 115).tolist(), iqrMinNeg, iqrMaxNeg, facecolor=(0, 0.25, 1, 0.2))

# coloro di bianco la fascia che devo saltare
ax.fill_between([77.2, 89], [10, 10], [125, 125], facecolor=(1, 1, 1, 1), linewidths=0.0, zorder=5.0)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.ylabel("previous 7 days \n median age (years) \n IQR", rotation=0, fontsize=12)
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.08, 0.5)
plt.xlabel("Time (weeks)", fontsize=12)
plt.xticks([38, 69, 99], labels=[])
ax.tick_params(which='major', length=6, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([24, 55, 85], minor=True)
ax.tick_params(which='minor', length=10, color='black')
ax.text(8, -18, "F", size=10)
ax.text(23.6, -18, "M", size=10)
ax.text(54.7, -18, "A", size=10)
ax.text(84.6, -18, "M", size=10)
plt.xlim(7.8, 115.2)
plt.show()

