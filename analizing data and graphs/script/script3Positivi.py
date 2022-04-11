import statistics
import pandas as pd
from pylab import *

pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import seaborn as sns
from scipy.stats import iqr

dataset = pd.read_excel("selected_dataset_diagnostico_COVID_POSITIVI.xls")
data = []
for i in dataset["DATA"]:
    data.append((i - datetime.datetime(2020, 2, 7)).days + 1)
dataset["DATA"] = data

# verifico quali punti non hanno 7 giorni precedenti
for i in range(0, len(dataset["DATA"]) - 1):
    if (abs(dataset["DATA"][i] - dataset["DATA"][i + 1])) > 6:
        print(dataset["DATA"][i])

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
        while (indice < (len(dataset["DATA"]) - 1)) and (dataset["DATA"][indice] == dataset["DATA"][indice + 1]):
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


# parto con PLT1

# inserisco le medie dei primi giorni (dal 27 febbraio al 4 marzo)
medieGiorniPrecedentiPLT1 = []
stdevGiorniPrecedentiPLT1 = []
listaSupporto = []

cnt = 13  # indica il giorno da cui parte la lista di supporto
nPLT1 = []
for i in range(16, 803):
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
        if len(listaSupporto) <= 1:
            stdevGiorniPrecedentiPLT1.append(0)
        else:
            stdevGiorniPrecedentiPLT1.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(dataset["PLT1"][i])

# aggiungo l'ultimo valore mancante alle due liste
medieGiorniPrecedentiPLT1.append(219.17)
stdevGiorniPrecedentiPLT1.append(93.57)
print(medieGiorniPrecedentiPLT1)
print(stdevGiorniPrecedentiPLT1)

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno

dict = {
    "giorno": np.arange(21, 115).tolist(),
    "media": medieGiorniPrecedentiPLT1,
    "dev standard": stdevGiorniPrecedentiPLT1
}

datiPLT1 = pd.DataFrame(dict)

# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x="giorno", y="media", data=datiPLT1, marker=".", markersize=8, color="#1034a6")

# dato che non mostra gli intervalli di confidenza li calcolo io
# ho già calcolato n, devo solo aggiungergli l'utlimo valore mancante
nPLT1.append(2)
print(nPLT1)

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


# definisco una funzione che uso per costruire i rettangoli degli intervalli di confidenza
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
        alpha = 0.035
    if i >= 30:
        alpha = 0.9
    if (i > 5) & (i < 30):
        alpha = (i - 5) / 25
    # se non ci sono dati nei 7 giorni precedenti non mostro nulla
    if i == 0:
        alpha = 0
    ax.fill_between(giorni[b:(b + 2)], minIC[b: (b + 2)], maxIC[b: (b + 2)], edgecolor="face",
                    color="#1b6ca8", alpha=alpha)
    # salto un punto, altrimenti disegnerei sopra quello già disegnato
    b += 1

# serve solo per la label nella legenda, non lo mostro
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
ax.text(8, -28, "F", size=10)
ax.text(23.6, -28, "M", size=10)
ax.text(54.7, -28, "A", size=10)
ax.text(84.6, -28, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# inserisco la legenda
plt.legend(loc="upper right", ncol=1, fancybox=True,
           frameon=False)
plt.show()


# passo a PCR

indiciCambioGiorno = []
i = 0
while i < len(dataset["DATA"]) - 1:
    if dataset["DATA"][i] != dataset["DATA"][i + 1]:
        indiciCambioGiorno.append(i)
    i += 1

medieGiorniPrecedentiPCR = []  # contiene le medie dei valori dei 7 giorni precedenti
stdevGiorniPrecedentiPCR = []  # contiene le deviazioni standard dei 7 giorni precedenti
listaSupporto = []

cnt = 13  # indica il giorno da cui parte la lista di supporto
nPCR = []

for i in range(16, 803):
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
        if len(listaSupporto) <= 1:
            stdevGiorniPrecedentiPCR.append(0)
        else:
            stdevGiorniPrecedentiPCR.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(dataset["PCR"][i])

# aggiungo l'utlimo valore mancante
medieGiorniPrecedentiPCR.append(61.55)
stdevGiorniPrecedentiPCR.append(45.61)

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno
dict = {
    "giorno": np.arange(21, 115),
    "media": medieGiorniPrecedentiPCR,
    "dev standard": stdevGiorniPrecedentiPCR
}

datiPCR = pd.DataFrame(dict)

# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x="giorno", y="media", data=datiPCR, marker=".", markersize=8, color="#1034a6")

# aggiungo l'ultimo valore mancante
nPCR.append(2)

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
        alpha = 0.035
    if i >= 30:
        alpha = 0.9
    if (i > 5) & (i < 30):
        alpha = (i - 5) / 25
    ax.fill_between(giorni[b:(b + 2)], minIC[b: (b + 2)], maxIC[b: (b + 2)], edgecolor="face", color="#1b6ca8",
                    alpha=alpha)
    # salto un punto, altrimenti ripeto la stessa cosa
    b += 1

# ne faccio uno invisibile solo per mettere la label
ax.fill_between(datiPCR["giorno"], icMinPCR, icMaxPCR, alpha=0.9, color="#1b6ca8",
                label="95% confidence interval").set_visible(False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xlabel("Time (days)", fontsize=12)
plt.ylabel("previous 7 days mean PCR", fontsize=12)

plt.xlim(8, 115.2)
plt.xticks([38, 69, 99], labels=[])
ax.tick_params(which='major', length=6, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([24, 55, 85], minor=True)
ax.tick_params(which='minor', length=10, color='black')
ax.text(8, -125, "F", size=10)
ax.text(23.6, -125, "M", size=10)
ax.text(54.7, -125, "A", size=10)
ax.text(84.6, -125, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda
plt.legend(loc="upper right", ncol=1, fancybox=True,
           frameon=False)
plt.show()

# passo a FG
medieGiorniPrecedentiFG = []  # contiene le medie dei valori dei 7 giorni precedenti
stdevGiorniPrecedentiFG = []
listaSupporto = []

cnt = 13  # indica il giorno da cui parte la lista di supporto
nFG = []
for i in range(16, 803):
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

# aggiungo l'ultimo valore mancante alle medie e alle deviazioni
medieGiorniPrecedentiFG.append(511.5)
stdevGiorniPrecedentiFG.append(65.76)

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno

dict = {
    "giorno": np.arange(21, 115),
    "media": medieGiorniPrecedentiFG,
    "dev standard": stdevGiorniPrecedentiFG
}

datiFG = pd.DataFrame(dict)

# aggiungo l'ultimo valore mancante agli n
nFG.append(2)

# inizio a disegnare il grafico

# siccome c'è un punto, tale per cui non c'è alcun valore nei sette giorni precedenti, elimino il punto dalla lista
# delle medie
nuovaListaMedie = []
nuovaListaGiorni = []
cnt = 21
for i in medieGiorniPrecedentiFG:
    if i != 0:
        nuovaListaMedie.append(i)
        nuovaListaGiorni.append(cnt)
    cnt += 1
medieGiorniPrecedentiFG = nuovaListaMedie
giorni = nuovaListaGiorni

fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x=giorni, y=medieGiorniPrecedentiFG, marker=".", markersize=8, color="#1034a6")

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

plt.margins(0.05)
datiIC = creaRettangolo(datiFG["giorno"], icMinFG, icMaxFG, nFG)
giorni = datiIC[0]
minIC = datiIC[1]
maxIC = datiIC[2]
nFG = datiIC[3]

for b in range(0, len(nFG)):
    i = nFG[b]
    if i <= 5:
        alpha = 0.035
    if i >= 30:
        alpha = 0.9
    if (i > 5) & (i < 30):
        alpha = (i - 5) / 25
    # se non ci sono valori nei sette giorni precedenti, non mostro nulla
    if i == 0:
        alpha = 0
    ax.fill_between(giorni[b:(b + 2)], minIC[b: (b + 2)], maxIC[b: (b + 2)], edgecolor="face", color="#1b6ca8",
                    alpha=alpha)
    # salto un punto, altrimenti rifaccio quello già fatto
    b += 1

# serve solo per inserire la label nella legenda
ax.fill_between(datiFG["giorno"], icMinFG, icMaxFG, alpha=0.9, color="#1b6ca8",
                label="95% confidence interval").set_visible(False)

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
ax.text(8, -220, "F", size=10)
ax.text(23.6, -220, "M", size=10)
ax.text(54.7, -220, "A", size=10)
ax.text(84.6, -220, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

plt.legend(loc="upper left", ncol=1, fancybox=True,
           frameon=False)
plt.show()


# passo a LY

medieGiorniPrecedentiLY = []  # contiene le medie dei valori dei 7 giorni precedenti
stdevGiorniPrecedentiLY = []
# inserisco anche per la lista di supporto i valori dei primi giorni
listaSupporto = []

cnt = 13  # indica il giorno da cui parte la lista di supporto
nLY = []
for i in range(16, 800):
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

# sostituisco nan con 0 e aggiungo i valori mancanti
medieGiorniPrecedentiLY[len(medieGiorniPrecedentiLY) - 1] = 0
medieGiorniPrecedentiLY += [0, 2.2, 2.2, 10.7]
stdevGiorniPrecedentiLY += [0, 0, 0, 12.02]

print(medieGiorniPrecedentiLY)
print(stdevGiorniPrecedentiLY)

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno

dict = {
    "giorno": np.arange(21, 115),
    "media": medieGiorniPrecedentiLY,
    "dev standard": stdevGiorniPrecedentiLY
}

datiLY = pd.DataFrame(dict)

# inizio a disegnare il grafico
# ci sono degli zeri nelle medie perché per alcuni punti non ci sono dati nei sette giorni precedenti. Elimino gli zeri
# dalla lista delle medie
nuovaListaMedie = []
nuovaListaGiorni = []
cnt = 21
for i in medieGiorniPrecedentiLY:
    if i != 0:
        nuovaListaMedie.append(i)
        nuovaListaGiorni.append(cnt)
    cnt += 1
medieGiorniPrecedentiLY = nuovaListaMedie
giorni = nuovaListaGiorni

fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x=giorni, y=medieGiorniPrecedentiLY, marker=".", markersize=8, color="#1034a6")

nLY += [0, 1, 1, 2]

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
        alpha = 0.035
    if i >= 30:
        alpha = 0.9
    if (i > 5) & (i < 30):
        alpha = (i - 5) / 25
    # se non ci sono valori, non mostro nulla
    if i == 0:
        alpha = 0
    ax.fill_between(giorni[b:(b + 2)], minIC[b: (b + 2)], maxIC[b: (b + 2)], edgecolor="face", color="#1b6ca8",
                    alpha=alpha)
    # salto un punto, altrimenti rifarei quello già fatto
    b += 1

# serve solo per aggiungere la label alla legenda
ax.fill_between(datiLY["giorno"], icMinLY, icMaxLY, alpha=0.9, color="#1b6ca8",
                label="95% confidence interval").set_visible(False)

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

# aggiungo la legenda
plt.legend(loc="upper right", ncol=1, fancybox=True,
           frameon=False)
plt.show()


# passo a WBC

medieGiorniPrecedentiWBC = []
stdevGiorniPrecedentiWBC = []
listaSupporto = []
cnt = 13  # indica il giorno da cui parte la lista di supporto
nWBC = []

for i in range(16, 803):
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

# aggiungo l'ultimo valore mancante
medieGiorniPrecedentiWBC.append(13.75)
stdevGiorniPrecedentiWBC.append(11.38)


# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno

dict = {
    "giorno": np.arange(21, 115),
    "media": medieGiorniPrecedentiWBC,
    "dev standard": stdevGiorniPrecedentiWBC
}

datiWBC = pd.DataFrame(dict)

# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x="giorno", y="media", data=datiWBC, marker=".", markersize=8, color="#1034a6")

# dato che non mostra gli intervalli di confidenza li calcolo io
# ho già calcolato n, devo solo aggiungergli alcuni valori mancanti
nWBC.append(2)

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
        alpha = 0.035
    if i >= 30:
        alpha = 0.9
    if (i > 5) & (i < 30):
        alpha = (i - 5) / 25
    ax.fill_between(giorni[b:(b + 2)], minIC[b: (b + 2)], maxIC[b: (b + 2)], edgecolor="face", color="#1b6ca8",
                    alpha=alpha)
    # salto un punto, altrimenti rifaccio quello già fatto
    b += 1

ax.fill_between(datiWBC["giorno"], icMinWBC, icMaxWBC, alpha=0.9, color="#1b6ca8",
                label="95% confidence interval").set_visible(False)
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
ax.text(8, -7.8, "F", size=10)
ax.text(23.6, -7.8, "M", size=10)
ax.text(54.7, -7.8, "A", size=10)
ax.text(84.6, -7.8, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda
plt.legend(loc="upper right", ncol=1, fancybox=True,
           frameon=False)
plt.show()


# proseguo con AST
medieGiorniPrecedentiAST = []  # contiene le medie dei valori dei 7 giorni precedenti
stdevGiorniPrecedentiAST = []
listaSupporto = []

cnt = 13  # indica il giorno da cui parte la lista di supporto
nAST = []
for i in range(16, 803):
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

print(medieGiorniPrecedentiAST)
print(stdevGiorniPrecedentiAST)

# aggiungo l'ultimo valore mancante
medieGiorniPrecedentiAST.append(40.75)
stdevGiorniPrecedentiAST.append(4.57)

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno

dict = {
    "giorno": np.arange(21, 115),
    "media": medieGiorniPrecedentiAST,
    "dev standard": stdevGiorniPrecedentiAST
}

datiAST = pd.DataFrame(dict)

# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x="giorno", y="media", data=datiAST, marker=".", markersize=8, color="#1034a6")

# aggiungo l'utlimo valore mancante alla lista degli n, già popolata in precedenza
nAST.append(2)

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
        alpha = 0.035
    if i >= 30:
        alpha = 0.9
    if (i > 5) & (i < 30):
        alpha = (i - 5) / 25
    if i == 0:
        alpha = 0
    ax.fill_between(giorni[b:(b + 2)], minIC[b: (b + 2)], maxIC[b: (b + 2)], edgecolor="face", color="#1b6ca8",
                    alpha=alpha)
    # salto un punto, altrimenti rifaccio quello già fatto
    b += 1

ax.fill_between(datiAST["giorno"], icMinAST, icMaxAST, alpha=0.9, color="#1b6ca8",
                label="95% confidence interval").set_visible(False)

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
ax.text(8, -40, "F", size=10)
ax.text(23.6, -40, "M", size=10)
ax.text(54.7, -40, "A", size=10)
ax.text(84.6, -40, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda
plt.legend(loc="upper right", ncol=1, fancybox=True,
           frameon=False)
plt.show()

# passo a LDH
medieGiorniPrecedentiLDH = []
stdevGiorniPrecedentiLDH = []
listaSupporto = []

cnt = 13  # indica il giorno da cui parte la lista di supporto
nLDH = []
for i in range(16, 803):
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
# aggiungo l'ultimo valore mancante
medieGiorniPrecedentiLDH.append(342.75)
stdevGiorniPrecedentiLDH.append(148.85)

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno

dict = {
    "giorno": np.arange(21, 115),
    "media": medieGiorniPrecedentiLDH,
    "dev standard": stdevGiorniPrecedentiLDH
}

datiLDH = pd.DataFrame(dict)

# inizio a disegnare il grafico
# elimino i valori nan dalla lista delle medie e creo una lista coi giorni coerente rispetto alla nuova lista
medieGiorniPrecedentiLDH[len(medieGiorniPrecedentiLDH) - 4] = 0
medieGiorniPrecedentiLDH[len(medieGiorniPrecedentiLDH) - 5] = 0
nuovaListaMedie = []
nuovaListaGiorni = []
cnt = 21
for i in medieGiorniPrecedentiLDH:
    if i != 0:
        nuovaListaMedie.append(i)
        nuovaListaGiorni.append(cnt)
    cnt += 1
medieGiorniPrecedentiLDH = nuovaListaMedie
giorni = nuovaListaGiorni

fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x=giorni, y=medieGiorniPrecedentiLDH, data=datiLDH, marker=".", markersize=8, color="#1034a6")

nLDH.append(2)

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
plt.margins(0.05)
datiIC = creaRettangolo(datiLDH["giorno"], icMinLDH, icMaxLDH, nLDH)
giorni = datiIC[0]
minIC = datiIC[1]
maxIC = datiIC[2]
nLDH = datiIC[3]

for b in range(0, len(nLDH)):
    i = nLDH[b]
    if i <= 5:
        alpha = 0.035
    if i >= 30:
        alpha = 0.9
    if (i > 5) & (i < 30):
        alpha = (i - 5) / 25
    # se non ci sono valori, non mostro alcun intervallo di confidenza
    if i == 0:
        alpha = 0
    ax.fill_between(giorni[b:(b + 2)], minIC[b: (b + 2)], maxIC[b: (b + 2)], edgecolor="face", color="#1b6ca8",
                    alpha=alpha)
    # salto un punto, altrimenti rifaccio quello già fatto
    b += 1

ax.fill_between(datiLDH["giorno"], icMinLDH, icMaxLDH, alpha=0.9, color="#1b6ca8",
                label="95% confidence interval").set_visible(False)

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
ax.text(8, 15, "F", size=10)
ax.text(23.6, 15, "M", size=10)
ax.text(54.7, 15, "A", size=10)
ax.text(84.6, 15, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda, ma devo fare un bbox per metterla in una posizione comoda

plt.legend(loc="upper right", ncol=1, fancybox=True,
           frameon=False)
plt.show()


# passo a CA
medieGiorniPrecedentiCA = []
stdevGiorniPrecedentiCA = []
listaSupporto = []

cnt = 13  # indica il giorno da cui parte la lista di supporto
nCA = []
for i in range(16, 803):
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

# aggiungo gli ultimi l'ultimo dato mancante
medieGiorniPrecedentiCA.append(2.0017)
stdevGiorniPrecedentiCA.append(0.083)

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno

dict = {
    "giorno": np.arange(21, 115),
    "media": medieGiorniPrecedentiCA,
    "dev standard": stdevGiorniPrecedentiCA
}

datiCA = pd.DataFrame(dict)

# inizio a disegnare il grafico
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x="giorno", y="media", data=datiCA, marker=".", markersize=8, color="#1034a6")
nCA.append(2)

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
        alpha = 0.035
    if i >= 30:
        alpha = 0.9
    if (i > 5) & (i < 30):
        alpha = (i - 5) / 25
    if i == 0:
        alpha = 0
    ax.fill_between(giorni[b:(b + 2)], minIC[b: (b + 2)], maxIC[b: (b + 2)], edgecolor="face", color="#1b6ca8",
                    alpha=alpha)
    # salto un punto, altrimenti rifaccio quello già fatto
    b += 1

ax.fill_between(datiCA["giorno"], icMinCA, icMaxCA, alpha=0.9, color="#1b6ca8",
                label="95% confidence interval").set_visible(False)
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

# aggiungo la legenda, ma devo fare un bbox per metterla in una posizione comoda

plt.legend(loc="upper left", ncol=1, fancybox=True,
           frameon=False)
plt.show()



# ora faccio i due grafici dell'età
medieGiorniPrecedentiEtà = []
stdevGiorniPrecedentiEtà = []
medianeGiorniPrecedentiEtà = []
iqrGiorniPrecedentiEtà = []
listaSupporto = []

cnt = 13  # indica il giorno da cui parte la lista di supporto
for i in range(16, 803):
    if i in indiciCambioGiorno:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(dataset["Age"][i])
        listaSupporto = aggiornaListaSupporto(indiciCambioGiorno[cnt], listaSupporto,
                                              indiciCambioGiorno[cnt + 7], "Age")
        # elimino gli zeri dalla lista
        listaSupporto = [i for i in listaSupporto if i != 0]
        medieGiorniPrecedentiEtà.append(mean(listaSupporto))
        medianeGiorniPrecedentiEtà.append(np.median(listaSupporto))
        iqrGiorniPrecedentiEtà.append(iqr(listaSupporto))
        if len(listaSupporto) <= 1:
            stdevGiorniPrecedentiEtà.append(0)
        else:
            stdevGiorniPrecedentiEtà.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(dataset["Age"][i])

# aggiungo l'ultimo dato mancante
medieGiorniPrecedentiEtà.append(82.5)
stdevGiorniPrecedentiEtà.append(9.19)
medianeGiorniPrecedentiEtà.append(82.5)
iqrGiorniPrecedentiEtà.append(6.5)

# disegno il grafico
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x=np.arange(21, 115), y=medieGiorniPrecedentiEtà, marker=".", color="royalblue", markersize=10, markeredgecolor="royalblue")

# creo gli intervalli della deviazione standard
stDevMin = []
stDevMax = []
cnt = 0
for i in stdevGiorniPrecedentiEtà:
    stDevMin.append(medieGiorniPrecedentiEtà[cnt] - i)
    stDevMax.append(medieGiorniPrecedentiEtà[cnt] + i)
    cnt += 1
ax.fill_between(np.arange(21, 115), stDevMin, stDevMax, alpha=0.45, color="#939ec3", label="standard deviation")
plt.ylabel("previous 7 days \n mean age (years) \n standard deviation", rotation=0, fontsize=12)
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.08, 0.5)
plt.xlabel("Time (weeks)", fontsize=12)
plt.xticks([38, 69, 99], labels=[])
ax.tick_params(which='major', length=6, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([24, 55, 85], minor=True)
ax.tick_params(which='minor', length=10, color='black')
ax.text(8, -11, "F", size=10)
ax.text(23.6, -11, "M", size=10)
ax.text(54.7, -11, "A", size=10)
ax.text(84.6, -11, "M", size=10)
plt.xlim(7.8, 115.2)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.show()



# faccio il grafico con la mediana e lo scarto interquartile
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x=np.arange(21, 115), y=medianeGiorniPrecedentiEtà, marker=".", color="royalblue", markersize=10, markeredgecolor="royalblue")

# creo gli intervalli dello scarto interquartile
iqrMin = []
iqrMax = []
cnt = 0
for i in iqrGiorniPrecedentiEtà:
    iqrMin.append(medianeGiorniPrecedentiEtà[cnt] - i)
    iqrMax.append(medianeGiorniPrecedentiEtà[cnt] + i)
    cnt += 1
ax.fill_between(np.arange(21, 115), iqrMin, iqrMax, alpha=0.45, color="#939ec3", label="IQR")
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

