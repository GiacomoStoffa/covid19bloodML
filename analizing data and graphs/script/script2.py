import statistics

import pandas as pd
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from pylab import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from matplotlib import rcParams, patches, gridspec
import seaborn as sns

# leggo i dataset
# AST19 = pd.read_excel("TOTALE_18_19_AST-.xlsx")
# AST20 = pd.read_excel("TOTALE_19_20_AST-.xlsx")
# LDH19 = pd.read_excel("TOTALE_18_19_LDH-.xlsx")
# LDH20 = pd.read_excel("TOTALE_19_20_LDH-.xlsx")
AST19 = pd.read_excel("ASTSintomi19.xlsx")
AST20 = pd.read_excel("ASTSintomi20.xlsx")
LDH19 = pd.read_excel("LDHSintomi19.xlsx")
LDH20 = pd.read_excel("LDHSintomi20.xlsx")


# leggo i dati dei tamponi
numeroPositivi = pd.read_excel("datiPositivi2.xlsx")

# parto con quello per LDH


# per prima cosa tronco i valori superiori a mille
LDH19["valore"][LDH19["valore"] > 1000] = 1000
LDH20["valore"][LDH20["valore"] > 1000] = 1000
AST20["valore"][AST20["valore"] > 1000] = 1000
AST19["valore"][AST19["valore"] > 1000] = 1000



# per prima cosa inserisco zeri dove manca un giorno
# definisco una funzione di supporto
def contains (lista, valore):
    cnt = 0
    for i in lista:
        if i == valore:
            return cnt
        cnt += 1
    return -1


giorni = []
valori = []
dateNascita = []
cnt = 1
for i in np.arange(1, 149):
    a = contains(LDH19["giorno"], i)
    if a == -1:
        giorni.append(cnt)
        valori.append(0)
        dateNascita.append(0)
    else:
        indice = a
        while (indice < len(LDH19["giorno"])) & (LDH19["giorno"][indice] == LDH19["giorno"][indice + 1]):
            giorni.append(LDH19["giorno"][indice])
            valori.append(LDH19["valore"][indice])
            dateNascita.append(LDH19["data di nascita"][indice])
            indice += 1
        giorni.append(LDH19["giorno"][indice])
        valori.append(LDH19["valore"][indice])
        dateNascita.append(LDH19["data di nascita"][indice])
    cnt += 1
# aggiungo gli utlimi due mancanti:
giorni.append(149)
giorni.append(149)
valori.append(201)
valori.append(214)
dateNascita.append("26/09/1950")
dateNascita.append("12/04/2002")
dict = {"giorno": giorni,
        "valore": valori,
        "data di nascita": dateNascita}
LDH19ConZeri = pd.DataFrame(dict)


# faccio la stessa cosa per LDH20
giorni = []
valori = []
dateNascita = []
cnt = 1
for i in np.arange(1, 152):
    a = contains(LDH20["giorno"], i)
    if a == -1:
        giorni.append(cnt)
        valori.append(0)
        dateNascita.append(0)
    else:
        indice = a
        while (indice < len(LDH20["giorno"])) & (LDH20["giorno"][indice] == LDH20["giorno"][indice + 1]):
            giorni.append(LDH20["giorno"][indice])
            valori.append(LDH20["valore"][indice])
            dateNascita.append(LDH20["data di nascita"][indice])
            indice += 1
        giorni.append(LDH20["giorno"][indice])
        valori.append(LDH20["valore"][indice])
        dateNascita.append(LDH20["data di nascita"][indice])
    cnt += 1
# aggiungo gli ultimi mancanti:
for i in np.arange(0, 14):
    giorni.append(152)
valori = valori + [128, 168, 192, 224, 246, 277, 330, 333, 361, 393, 419, 432, 484, 734]
dateNascita = dateNascita + ["17/09/1949", "17/05/1985", "01/10/1993", "06/01/2003", "10/07/1941", "28/04/1938",
                             "30/12/1984", "07/06/1999", "08/12/1963", "10/02/1978", "16/04/1959",
                             "15/05/1941", "23/09/1927", "24/08/1938"]
dict = {"giorno": giorni,
        "valore": valori,
        "data di nascita": dateNascita}
LDH20ConZeri = pd.DataFrame(dict)

# stessa cosa per AST19
giorni = []
valori = []
dateNascita = []
cnt = 1
for i in np.arange(1, 151):
    a = contains(AST19["giorno"], i)
    if a == -1:
        giorni.append(cnt)
        valori.append(0)
        dateNascita.append(0)
    else:
        indice = a
        while (indice < len(AST19["giorno"])) & (AST19["giorno"][indice] == AST19["giorno"][indice + 1]):
            giorni.append(AST19["giorno"][indice])
            valori.append(AST19["valore"][indice])
            dateNascita.append(AST19["data nascita"][indice])
            indice += 1
        giorni.append(AST19["giorno"][indice])
        valori.append(AST19["valore"][indice])
        dateNascita.append(AST19["data nascita"][indice])
    cnt += 1
# aggiungo gli utlimi due mancanti:
giorni.append(151)
giorni.append(151)
valori.append(32)
valori.append(31)
dateNascita.append("15/03/1934")
dateNascita.append("18/03/1926")
dict = {"giorno": giorni,
        "valore": valori,
        "data di nascita": dateNascita}
AST19ConZeri = pd.DataFrame(dict)

# stessa cosa per AST20
giorni = []
valori = []
dateNascita = []
cnt = 1
for i in np.arange(1, 152):
    a = contains(AST20["data"], i)
    if a == -1:
        giorni.append(cnt)
        valori.append(0)
        dateNascita.append(0)
    else:
        indice = a
        while (indice < len(AST20["data"])) & (AST20["data"][indice] == AST20["data"][indice + 1]):
            giorni.append(AST20["data"][indice])
            valori.append(AST20["valore"][indice])
            dateNascita.append(AST20["data nascita"][indice])
            indice += 1
        giorni.append(AST20["data"][indice])
        valori.append(AST20["valore"][indice])
        dateNascita.append(AST20["data nascita"][indice])
    cnt += 1
# aggiungo quelli che mancano
for i in np.arange(0, 14):
    giorni.append(152)
valori = valori + [17, 21, 25, 27, 29, 33, 35, 36, 55, 55, 62, 71, 93, 104]
dateNascita = dateNascita + ["17/05/1985", "01/10/1993", "17/09/1949", "07/06/1999", "30/12/1984", "16/04/1959",
                             "08/12/1963", "28/04/1938", "24/08/1938", "10/02/1978",
                             "06/01/2003", "02/02/1948", "15/05/1941", "23/09/1927"]
dict = {"giorno": giorni,
        "valore": valori,
        "data di nascita": dateNascita}
AST20ConZeri = pd.DataFrame(dict)


# inserisco nei quattro dataframe una colonna che indica la settimana
settimana = []
cnt = 1
i = 0
for i in LDH19ConZeri["giorno"]:
    settimana.append(math.floor((i / 7) - 0.0001) + 1)
    i += 1
LDH19ConZeri["settimana"] = settimana
print(settimana)
# stessa cosa per LDH20
settimana = []
for i in LDH20ConZeri["giorno"]:
    settimana.append(math.floor((i / 7) - 0.0001) + 1)
    i += 1
LDH20ConZeri["settimana"] = settimana
print(settimana)
# stessa cosa per AST19
settimana = []
cnt = 1
i = 0
for i in AST19ConZeri["giorno"]:
    settimana.append(math.floor((i / 7) - 0.0001) + 1)
    i += 1
AST19ConZeri["settimana"] = settimana
print(settimana)
# stessa cosa per AST20
settimana = []
cnt = 1
i = 0
for i in AST20ConZeri["giorno"]:
    settimana.append(math.floor((i / 7) - 0.0001) + 1)
    i += 1

AST20ConZeri["settimana"] = settimana
print(settimana)

# faccio la stessa cosa per quelli senza zeri (mi serve per il grafico delle età)
settimana = []
cnt = 1
i = 0
for i in LDH19["giorno"]:
    settimana.append(math.floor((i / 7) - 0.0001) + 1)
    i += 1
LDH19["settimana"] = settimana

# stessa cosa per LDH20
settimana = []
for i in LDH20["giorno"]:
    settimana.append(math.floor((i / 7) - 0.0001) + 1)
    i += 1
LDH20["settimana"] = settimana

# stessa cosa per AST19
settimana = []
cnt = 1
i = 0
for i in AST19["giorno"]:
    settimana.append(math.floor((i / 7) - 0.0001) + 1)
    i += 1
AST19["settimana"] = settimana

# stessa cosa per AST20
settimana = []
cnt = 1
i = 0
for i in AST20["data"]:
    settimana.append(math.floor((i / 7) - 0.0001) + 1)
    i += 1

AST20["settimana"] = settimana


# devo calcolare le medie e le deviazioni standard di ogni settimana
mediaSettimanaLDH19 = []
stdevSettimanaLDH19 = []
listaSupporto = []

for i in range(0, len(LDH19ConZeri["valore"]) - 1):
    if LDH19ConZeri["settimana"][i] != LDH19ConZeri["settimana"][i + 1]:
        if LDH19ConZeri["valore"][i] != 0:
            listaSupporto.append(LDH19ConZeri["valore"][i])
        mediaSettimanaLDH19.append(mean(listaSupporto))
        stdevSettimanaLDH19.append(statistics.stdev(listaSupporto))
        listaSupporto = []
    else:
        if LDH19ConZeri["valore"][i] != 0:
            listaSupporto.append(LDH19ConZeri["valore"][i])

# ora aggiungo la media degli ultimi giorni rimanenti (che non finiscono una settimana intera)
mediaSettimanaLDH19.append(217.33)

# ora metto nel datframe le medie
medieNuova = []
cnt = 0
for i in range(0, len(LDH19ConZeri["valore"]) - 1):
    if LDH19ConZeri["settimana"][i] != LDH19ConZeri["settimana"][i + 1]:
        medieNuova.append(mediaSettimanaLDH19[cnt])
        cnt += 1
    else:
        medieNuova.append(mediaSettimanaLDH19[cnt])
medieNuova.append(217.33)
LDH19ConZeri["media settimanale"] = medieNuova

# faccio la setssa cosa per LDH20
# devo calcolare le medie e le deviazioni standard di ogni settimana
mediaSettimanaLDH20 = []
stdevSettimanaLDH20 = []
listaSupporto = []

for i in range(0, len(LDH20ConZeri["valore"]) - 1):
    if LDH20ConZeri["settimana"][i] != LDH20ConZeri["settimana"][i + 1]:
        if LDH20ConZeri["valore"][i] != 0:
            listaSupporto.append(LDH20ConZeri["valore"][i])
        mediaSettimanaLDH20.append(mean(listaSupporto))
        if len(listaSupporto) > 1:
            stdevSettimanaLDH20.append(statistics.stdev(listaSupporto))
        else:
            stdevSettimanaLDH20.append(0)
        listaSupporto = []
    else:
        if LDH20ConZeri["valore"][i] != 0:
            listaSupporto.append(LDH20ConZeri["valore"][i])


# ora aggiungo la media degli ultimi giorni rimanenti (che non finiscono una settimana intera)
mediaSettimanaLDH20.append(379.86)
# ora metto nel dataframe le medie
medieNuova = []
cnt = 0
for i in range(0, len(LDH20ConZeri["valore"]) - 1):
    if LDH20ConZeri["settimana"][i] != LDH20ConZeri["settimana"][i + 1]:
        medieNuova.append(mediaSettimanaLDH20[cnt])
        cnt += 1
    else:
        medieNuova.append(mediaSettimanaLDH20[cnt])
medieNuova.append(379.86)
print(medieNuova)
LDH20ConZeri["media settimanale"] = medieNuova


# faccio lo stesso identico grafico, ma inserendo sopra il grafico con le media delle età e la deviazioni standard
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 6])
gs.update(wspace=0.03, hspace=0.03)
ax0 = plt.subplot(gs[2])
sns.lineplot(x="settimana", y="media settimanale", color="blue", data=LDH19ConZeri, estimator=np.mean, marker="o", ci=95,
             err_style="bars", label="2018/19", ax=ax0)
sns.lineplot(x="settimana", y="media settimanale", color="red", data=LDH20ConZeri, estimator=np.mean, marker="o", ci=95,
             err_style="bars", label="2019/20", ax=ax0)

# disegno la fascia colorata che indica la deviazione standard dei dati del 18/19
stdevMinLDH19 = []
stdevMaxLDH19 = []
cnt = 0
for i in range(0, len(mediaSettimanaLDH19) - 1):
    stdevMinLDH19.append(mediaSettimanaLDH19[i] - stdevSettimanaLDH19[cnt])
    stdevMaxLDH19.append(mediaSettimanaLDH19[i] + stdevSettimanaLDH19[cnt])
    cnt += 1
ax0.fill_between(np.arange(1, 22), stdevMinLDH19, stdevMaxLDH19, alpha=0.2, label="18/19 standard deviation")


# devo calcolare e disegnare  gli intervalli di confidenza dato che no li calcola seaborn
nLDH19 = []
cnt = 0
for i in range(0, len(LDH19ConZeri["valore"]) - 1):
    if LDH19ConZeri["settimana"][i] != LDH19ConZeri["settimana"][i + 1]:
        if LDH19ConZeri["valore"][i] != 0:
            cnt += 1
        nLDH19.append(cnt)
        cnt = 0
    else:
        if LDH19ConZeri["valore"][i] != 0:
            cnt += 1
nLDH20 = []
cnt = 0
for i in range(0, len(LDH20ConZeri["valore"]) - 1):
    if LDH20ConZeri["settimana"][i] != LDH20ConZeri["settimana"][i + 1]:
        if LDH20ConZeri["valore"][i] != 0:
            cnt += 1
        nLDH20.append(cnt)
        cnt = 0
    else:
        if LDH20ConZeri["valore"][i] != 0:
            cnt += 1
icLDH19 = []
cnt = 0
for i in stdevSettimanaLDH19:
    if i != 0:
        icLDH19.append(1.96 * (i / sqrt(nLDH19[cnt])))
    else:
        icLDH19.append(0)
    cnt += 1
icLDH20 = []
cnt = 0
for i in stdevSettimanaLDH20:
    if i != 0:
        icLDH20.append(1.96 * (i / sqrt(nLDH20[cnt])))
    else:
        icLDH20.append(0)
    cnt += 1
# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinLDH19 = []
cnt = 0
for i in stdevSettimanaLDH19:
    icMinLDH19.append(mediaSettimanaLDH19[cnt] - icLDH19[cnt])
    cnt += 1
icMaxLDH19 = []
cnt = 0
for i in stdevSettimanaLDH19:
    icMaxLDH19.append(mediaSettimanaLDH19[cnt] + icLDH19[cnt])
    cnt += 1

icMinLDH20 = []
cnt = 0
for i in stdevSettimanaLDH20:
    icMinLDH20.append(mediaSettimanaLDH20[cnt] - icLDH20[cnt])
    cnt += 1
icMaxLDH20 = []
cnt = 0
for i in stdevSettimanaLDH20:
    icMaxLDH20.append(mediaSettimanaLDH20[cnt] + icLDH20[cnt])
    cnt += 1

# disegno gli ic
plt.vlines(np.arange(1, 22), icMinLDH19, icMaxLDH19, colors=["blue"]*21)
plt.vlines(np.arange(1, 22), icMinLDH20, icMaxLDH20, colors=["red"]*21)

# disegno la linea che indica il limite patologico
sns.lineplot(x=np.arange(0, 22), y=[220] * 22, color="black", label="normal/pathological limit", ax=ax0)

# creo la linea verticale tratteggiata
plt.plot([16.7] * 530, np.arange(60, 590), linestyle='dashed', color="black", linewidth=0.8)

plt.xlim(1, 21.1)
plt.ylim(60, 590)
plt.xticks([1, 5, 9, 13, 17, 21])
plt.xlabel("Time (weeks)", fontsize=12)
# scrivo le indicazioni relative al mese. Uso i minor ticks
ax0.set_xticks([5.28, 9.71, 14.14, 18.28], minor=True)
ax0.tick_params(which='minor', length=3, color='black')
ax0.text(1, 10, "N", size=10)
ax0.text(5.28, 10, "D", size=10)
ax0.text(9.71, 10, "J", size=10)
ax0.text(14.14, 10, "F", size=10)
ax0.text(18.28, 10, "M", size=10)
# sposto l'etichetta dell'asse più in basso
ax0.xaxis.set_label_coords(0.5, -0.15)

plt.ylabel("LDH (U/L)", fontsize=12, rotation=0)
ax0.yaxis.set_label_coords(-0.07, 0.5)

# aggiungo la legenda
plt.legend(loc="upper left", frameon=False)

# ora mostro un nuovo grafico sopra con il numero dei positivi
# per prima cosa devo prendere i dati settimanali
numeroPositiviSettimana = [0]
for i in numeroPositivi["riepilogo settimanale"]:
    if not isnan(i):
        numeroPositiviSettimana.append(i)
ax1 = plt.subplot(gs[1])
sns.lineplot(x=[16.7, 17.7, 18.7, 19.7, 20.7], y=numeroPositiviSettimana, ax=ax1)
plt.xlim(1, 21.1)
ax1.set_xticks([])
plt.ylabel("No. of COVID \n positive tests", rotation=0, fontsize=12)
ax1.yaxis.tick_right()  # metto i ticks sulla destra
ax1.yaxis.set_label_coords(-0.07, 0.5)  # posiziono la label dell'asse y
plt.ylim(0, max(numeroPositiviSettimana))


# ora inserisco il grafico con le età e deviazioni standard
# per prima cosa calcolo l'età di ognuno e aggiungo una colonna per raccoglierla
etàLDH19 = []
for i in LDH19ConZeri["data di nascita"]:
    if i != 0:
        etàLDH19.append(2020 - int(i[6: 11]))
etàLDH20 = []
for i in LDH20ConZeri["data di nascita"]:
    if i != 0:
        etàLDH20.append(2020 - int(i[6: 11]))
etàAST19 = []
for i in AST19ConZeri["data di nascita"]:
    if i != 0:
        etàAST19.append(2020 - int(i[6: 11]))
etàAST20 = []
for i in AST20ConZeri["data di nascita"]:
    if i != 0:
        etàAST20.append(2020 - int(i[6: 11]))
etàLDHTOT = etàLDH19 + etàLDH20
etàASTTOT = etàAST19 + etàAST20



# inserisco nel dataframe una colonna con questo dato
LDH19["età"] = etàLDH19
LDH20["età"] = etàLDH20
AST19["età"] = etàAST19
AST20["età"] = etàAST20



# iniz"io a disegnare il grafico
ax2 = plt.subplot(gs[0])
sns.lineplot(x="settimana", y="età", estimator=np.mean, data=LDH19, ci="sd", ax=ax2)
sns.lineplot(x="settimana", y="età", estimator=np.mean, data=LDH20, ci="sd", ax=ax2, color="red")
ax2.yaxis.set_label_coords(-0.06, 0.5)
plt.xlim(1, 21.1)
ax2.yaxis.tick_right()  # metto i ticks sulla destra
plt.ylabel("mean age (yrs) \n standard deviation", rotation=0, fontsize=12)
plt.xlabel("")
plt.xticks([])


plt.show()




# ora faccio lo stesso identico grafico, ma con sopra gli altri due grafici
# devo calcolare le medie e le deviazioni standard di ogni settimana
mediaSettimanaAST19 = []
stdevSettimanaAST19 = []
listaSupporto = []

for i in range(0, len(AST19ConZeri["valore"]) - 1):
    if AST19ConZeri["settimana"][i] != AST19ConZeri["settimana"][i + 1]:
        if AST19ConZeri["valore"][i] != 0:
            listaSupporto.append(AST19ConZeri["valore"][i])
        mediaSettimanaAST19.append(mean(listaSupporto))
        stdevSettimanaAST19.append(statistics.stdev(listaSupporto))
        listaSupporto = []
    else:
        if AST19ConZeri["valore"][i] != 0:
            listaSupporto.append(AST19ConZeri["valore"][i])

# ora aggiungo la media degli ultimi giorni rimanenti (che non finiscono una settimana intera)
mediaSettimanaAST19.append(29.5)

# ora metto nel dataframe le medie
medieNuova = []
cnt = 0
for i in range(0, len(AST19ConZeri["valore"]) - 1):
    if AST19ConZeri["settimana"][i] != AST19ConZeri["settimana"][i + 1]:
        medieNuova.append(mediaSettimanaAST19[cnt])
        cnt += 1
    else:
        medieNuova.append(mediaSettimanaAST19[cnt])
medieNuova.append(29.5)
AST19ConZeri["media settimanale"] = medieNuova
print(medieNuova)

# faccio la stessa cosa per LDH20
# devo calcolare le medie e le deviazioni standard di ogni settimana
mediaSettimanaAST20 = []
stdevSettimanaAST20 = []
listaSupporto = []

for i in range(0, len(AST20ConZeri["valore"]) - 1):
    if AST20ConZeri["settimana"][i] != AST20ConZeri["settimana"][i + 1]:
        if AST20ConZeri["valore"][i] != 0:
            listaSupporto.append(AST20ConZeri["valore"][i])
        mediaSettimanaAST20.append(mean(listaSupporto))
        if len(listaSupporto) > 1:
            stdevSettimanaAST20.append(statistics.stdev(listaSupporto))
        else:
            stdevSettimanaAST20.append(0)
        listaSupporto = []
    else:
        if AST20ConZeri["valore"][i] != 0:
            listaSupporto.append(AST20ConZeri["valore"][i])


# ora aggiungo la media degli ultimi giorni rimanenti (che non finiscono una settimana intera)
mediaSettimanaAST20.append(54.47)

# ora metto nel dataframe le medie
medieNuova = []
cnt = 0
for i in range(0, len(AST20ConZeri["valore"]) - 1):
    if AST20ConZeri["settimana"][i] != AST20ConZeri["settimana"][i + 1]:
        medieNuova.append(mediaSettimanaAST20[cnt])
        cnt += 1
    else:
        medieNuova.append(mediaSettimanaAST20[cnt])
medieNuova.append(54.47)
print(medieNuova)
AST20ConZeri["media settimanale"] = medieNuova


fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 6])
gs.update(wspace=0.03, hspace=0.03)
ax0 = plt.subplot(gs[2])
sns.lineplot(x="settimana", y="media settimanale", color="blue", data=AST19ConZeri, estimator=np.mean, marker="o", ci=95,
             err_style="bars", label="2018/19", ax=ax0)
sns.lineplot(x="settimana", y="media settimanale", color="red", data=AST20ConZeri, estimator=np.mean, marker="o", ci=95,
             err_style="bars", label="2019/20", ax=ax0)
# disegno la fascia colorata che indica la deviazione standard dei dati del 18/19
stdevMinAST19 = []
stdevMaxAST19 = []
cnt = 0
for i in range(0, len(mediaSettimanaLDH19) - 1):
    stdevMinAST19.append(mediaSettimanaAST19[i] - stdevSettimanaAST19[cnt])
    stdevMaxAST19.append(mediaSettimanaAST19[i] + stdevSettimanaAST19[cnt])
    cnt += 1
ax0.fill_between(np.arange(1, 22), stdevMinAST19, stdevMaxAST19, alpha=0.2, label="18/19 standard deviation")

# devo calcolare e disegnare  gli intervalli di confidenza dato che non li calcola seaborn
nAST19 = []
cnt = 0
for i in range(0, len(AST19ConZeri["valore"]) - 1):
    if AST19ConZeri["settimana"][i] != AST19ConZeri["settimana"][i + 1]:
        if AST19ConZeri["valore"][i] != 0:
            cnt += 1
        nAST19.append(cnt)
        cnt = 0
    else:
        if AST19ConZeri["valore"][i] != 0:
            cnt += 1
nAST20 = []
cnt = 0
for i in range(0, len(AST20ConZeri["valore"]) - 1):
    if AST20ConZeri["settimana"][i] != AST20ConZeri["settimana"][i + 1]:
        if AST20ConZeri["valore"][i] != 0:
            cnt += 1
        nAST20.append(cnt)
        cnt = 0
    else:
        if AST20ConZeri["valore"][i] != 0:
            cnt += 1
icAST19 = []
cnt = 0
for i in stdevSettimanaAST19:
    if i != 0:
        icAST19.append(1.96 * (i / sqrt(nAST19[cnt])))
    else:
        icAST19.append(0)
    cnt += 1
icAST20 = []
cnt = 0
for i in stdevSettimanaAST20:
    if i != 0:
        icAST20.append(1.96 * (i / sqrt(nAST20[cnt])))
    else:
        icAST20.append(0)
    cnt += 1
# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinAST19 = []
cnt = 0
for i in stdevSettimanaAST19:
    icMinAST19.append(mediaSettimanaAST19[cnt] - icAST19[cnt])
    cnt += 1
icMaxAST19 = []
cnt = 0
for i in stdevSettimanaAST19:
    icMaxAST19.append(mediaSettimanaAST19[cnt] + icAST19[cnt])
    cnt += 1

icMinAST20 = []
cnt = 0
for i in stdevSettimanaAST20:
    icMinAST20.append(mediaSettimanaAST20[cnt] - icAST20[cnt])
    cnt += 1
icMaxAST20 = []
cnt = 0
for i in stdevSettimanaAST20:
    icMaxAST20.append(mediaSettimanaAST20[cnt] + icAST20[cnt])
    cnt += 1

# disegno gli ic
plt.vlines(np.arange(1, 22), icMinAST19, icMaxAST19, colors=["blue"]*21)
plt.vlines(np.arange(1, 22), icMinAST20, icMaxAST20, colors=["red"]*21)


# disegno la linea che indica il limite patologico
sns.lineplot(x=np.arange(0, 22), y=[35] * 22, color="black", label="normal/pathological limit", ax=ax0)

# creo la linea verticale tratteggiata
plt.plot([16.7] * 120, np.arange(0, 120), linestyle='dashed', color="black", linewidth=0.8)

# scrivo le indicazioni relative al mese. Uso i minor ticks
plt.xticks([1, 5, 9, 13, 17, 21])
plt.xlabel("Time (weeks)", fontsize=12)
# scrivo le indicazioni relative al mese. Uso i minor ticks
ax0.set_xticks([5.28, 9.71, 14.14, 18.28], minor=True)
ax0.tick_params(which='minor', length=3, color='black')
ax0.text(1, -11, "N", size=10)
ax0.text(5.28, -11, "D", size=10)
ax0.text(9.71, -11, "J", size=10)
ax0.text(14.14, -11, "F", size=10)
ax0.text(18.28, -11, "M", size=10)
plt.xlim(1, 21.1)
plt.ylim(0, 120)
# sposto l'etichetta dell'asse più in basso
ax0.xaxis.set_label_coords(0.5, -0.15)

ax0.yaxis.set_label_coords(-0.07, 0.5)

plt.ylabel("AST (U/L)", fontsize=12, rotation=0)

# aggiungo la legenda
plt.legend(loc="upper left", frameon=False)

# ora mostro sullo stesso grafico il numero di positivi
ax1 = plt.subplot(gs[1])
sns.lineplot(x=[16.7, 17.7, 18.7, 19.7, 20.7], y=numeroPositiviSettimana, ax=ax1)
plt.xlim(1, 21.1)
ax1.set_xticks([])
plt.ylabel("No. of COVID \n positive tests", rotation=0, fontsize=12)
ax1.yaxis.tick_right()  # metto i ticks sulla destra
ax1.yaxis.set_label_coords(-0.07, 0.5)  # poisiziono la label dell'asse y
plt.ylim(0, max(numeroPositiviSettimana))

# disegno il grafico sopra
ax2 = plt.subplot(gs[0])
sns.lineplot(x="settimana", y="età", estimator=np.mean, data=AST19, ci="sd", ax=ax2)
sns.lineplot(x="settimana", y="età", estimator=np.mean, data=AST20, ci="sd", ax=ax2, color="red")
ax2.yaxis.set_label_coords(-0.06, 0.5)
plt.xlim(1, 21.1)
ax2.yaxis.tick_right()  # metto i ticks sulla destra
plt.ylabel("mean age (yrs) \n standard deviation", rotation=0, fontsize=12)
plt.xlabel("")
plt.xticks([])
plt.show()



print("------------------")




# passo all'altro tipo di grafico




# devo calcolare per ogni giorno la media dei valori dei 7 giorni precedenti.
# per i primi 7 giorni non ha senso farlo.

# LDH19 = pd.read_excel("LDHSintomi19ConZeri.xlsx")

i = 0
listaSupporto = []
# per prima cosa faccio un ciclo nel quale aggiungo i valori dei primi 7 giorni alla listaSupporto
i = 0
fine = False
while fine != True:
    if LDH19ConZeri["giorno"][i] == 8:
        fine = True
    else:
        listaSupporto.append(LDH19ConZeri["valore"][i])
    i += 1


# definisco una funzione di supporto
def aggiornaListaSupporto(counter, lista, c):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(LDH19ConZeri["valore"][r])
        r += 1
    return nuovaLista



# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiornoLDH19 = []
i = 0
while i < len(LDH19ConZeri["giorno"]) - 1:
    if LDH19ConZeri["giorno"][i] != LDH19ConZeri["giorno"][i + 1]:
        indiciCambioGiornoLDH19.append(i)
    i += 1



i = 11  # 10 perchè ho già messo i valori della prima settimana
cnt = 0  # indica il giorno da cui parte la lista di supporto
listaSupporto = [i for i in listaSupporto if i != 0]
medieGiorniPrecedentiLDH19 = []  # contiene le medie dei valori dei 7 giorni precedenti
stdevGiorniPrecedentiLDH19 = []  # conterrà le deviazioni standard dei valori dei 7 giorniprecedenti
medieGiorniPrecedentiLDH19.append(mean(listaSupporto))  # così aggiungo la media dei primi 7 giorni
stdevGiorniPrecedentiLDH19.append(statistics.stdev(listaSupporto))
nLDH19 = [8]
while i < len(LDH19ConZeri) - 1:
    if i in indiciCambioGiornoLDH19:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(LDH19ConZeri["valore"][i])
        listaSupporto = aggiornaListaSupporto(indiciCambioGiornoLDH19[cnt], listaSupporto,
                                              indiciCambioGiornoLDH19[cnt + 7])
        # elimino gli zeri dalla lista
        listaSupporto = [i for i in listaSupporto if i != 0]
        nLDH19.append(len(listaSupporto))
        medieGiorniPrecedentiLDH19.append(mean(listaSupporto))
        stdevGiorniPrecedentiLDH19.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(LDH19ConZeri["valore"][i])
    i += 1

# aggiungo l'ultimo dato mancante:
medieGiorniPrecedentiLDH19.append(259.57)
stdevGiorniPrecedentiLDH19.append(91.62)


# ora faccio la stessa cosa per LDH20
i = 0
listaSupporto = []
# per prima cosa faccio un ciclo nel quale aggiungo i valori dei primi 7 giorni alla listaSupporto
fine = False
while fine != True:
    if LDH20ConZeri["giorno"][i] == 8:
        fine = True
    else:
        listaSupporto.append(LDH20ConZeri["valore"][i])
    i += 1


# definisco una funzione di supporto
def aggiornaListaSupporto20(counter, lista, c):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(LDH20ConZeri["valore"][r])
        r += 1
    return nuovaLista


# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiornoLDH20 = []
i = 0
while i < len(LDH20ConZeri) - 1:
    if LDH20ConZeri["giorno"][i] != LDH20ConZeri["giorno"][i + 1]:
        indiciCambioGiornoLDH20.append(i)
    i += 1
i = 10  # 10 perchè ho già messo i valori della prima settimana
cnt = 0  # indica il giorno da cui parte la lista di supporto
medieGiorniPrecedentiLDH20 = []  # contiene le medie dei valori dei 7 giorni precedenti
stdevGiorniPrecedentiLDH20 = []  # contiene le dev standard dei valori dei 7 giorni precedenti
medieGiorniPrecedentiLDH20.append(mean(listaSupporto))  # così aggiungo la media dei primi 7 giorni
stdevGiorniPrecedentiLDH20.append(statistics.stdev(listaSupporto))
nLDH20 = [5]  # n è una lista che userò dopo per gli intrevalli di confidenza
while i < len(LDH20ConZeri) - 1:
    if i in indiciCambioGiornoLDH20:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(LDH20ConZeri["valore"][i])
        if cnt < 261:
            listaSupporto = aggiornaListaSupporto20(indiciCambioGiornoLDH20[cnt], listaSupporto,
                                                    indiciCambioGiornoLDH20[cnt + 7])
        else:
            listaSupporto = aggiornaListaSupporto20(indiciCambioGiornoLDH20[cnt], listaSupporto,
                                                    indiciCambioGiornoLDH20[cnt + 6])
        # elimino gli zeri dalla lista
        listaSupporto = [i for i in listaSupporto if i != 0]
        if len(listaSupporto) > 1:
            nLDH20.append(len(listaSupporto))
            medieGiorniPrecedentiLDH20.append(mean(listaSupporto))
            stdevGiorniPrecedentiLDH20.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(LDH20ConZeri["valore"][i])
    i += 1
# sistemo i primi e gli ultimi, che sono sbagliati/mancano


# ci sono dei valori sbagliati, li sistemo
medieGiorniPrecedentiLDH20.remove(237.33333333333334)
medieGiorniPrecedentiLDH20.remove(303.2307692307692)
medieGiorniPrecedentiLDH20[49] = 360
medieGiorniPrecedentiLDH20[50] = 217
medieGiorniPrecedentiLDH20.insert(51, 237.33)
medieGiorniPrecedentiLDH20.insert(52, 303.23)
medieGiorniPrecedentiLDH20.insert(53, 318.21)
medieGiorniPrecedentiLDH20.insert(56, 300.45)
medieGiorniPrecedentiLDH20.append(406.51)
medieGiorniPrecedentiLDH20.append(395.985)

for i in np.arange(0, len(medieGiorniPrecedentiLDH20)):
    print(i + 7)
    print(medieGiorniPrecedentiLDH20[i])
print("-----------------------------------------------")
stdevGiorniPrecedentiLDH20[49] = 0
stdevGiorniPrecedentiLDH20[50] = 0
stdevGiorniPrecedentiLDH20.insert(51, 71.54)
stdevGiorniPrecedentiLDH20.insert(52, 136.41)
stdevGiorniPrecedentiLDH20.append(162.39)
stdevGiorniPrecedentiLDH20.append(168.37)

for i in np.arange(0, len(stdevGiorniPrecedentiLDH20)):
    print(i + 7)
    print(stdevGiorniPrecedentiLDH20[i])
print("-------------------------------------------------")



medieGiorniPrecedentiLDH20[0] = 299.6
medieGiorniPrecedentiLDH20.remove(217)
medieGiorniPrecedentiLDH20.remove(237.33)
stdevGiorniPrecedentiLDH20.remove(0)
stdevGiorniPrecedentiLDH20.remove(0)
a = np.arange(8, 58).tolist()
b = np.arange(60, 154).tolist()
print(len(stdevGiorniPrecedentiLDH20))

# ora creo un dataframe con le medie dei 7 giorni precedenti e il giorno

dict = {
    "giorno": np.arange(8, 151),
    "media": medieGiorniPrecedentiLDH19,
    "dev standard": stdevGiorniPrecedentiLDH19
}
datiLDH19 = pd.DataFrame(dict)
dict2 = {
    "giorno": (a + b),
    "media": medieGiorniPrecedentiLDH20,
    "dev standard": stdevGiorniPrecedentiLDH20
}
datiLDH20 = pd.DataFrame(dict2)

# inizio a disegnare il grafico
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])
gs.update(wspace=0.03, hspace=0.03)
ax = plt.subplot(gs[1])
sns.lineplot(x="giorno", y="media", data=datiLDH19, marker=".", markersize=8, label="2018/19", ax=ax)

# creo la linea verticale tratteggiata
plt.plot([117] * 430, np.arange(150, 580), linestyle='dashed', color="black", linewidth=0.8)

# dato che non mostra gli intervalli di confidenza li calcolo io
# ho già calcolato n, devo solo aggiungergli alcuni valori mancanti
nLDH19.append(15)



icLDH19 = []
cnt = 0
for i in datiLDH19["dev standard"]:
    icLDH19.append(1.96 * i / sqrt(nLDH19[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinLDH19 = []
cnt = 0
for i in datiLDH19["dev standard"]:
    icMinLDH19.append(datiLDH19["media"][cnt] - icLDH19[cnt])
    cnt += 1
icMaxLDH19 = []
cnt = 0
for i in datiLDH19["dev standard"]:
    icMaxLDH19.append(datiLDH19["media"][cnt] + icLDH19[cnt])
    cnt += 1

# disegno gli intervalli di confidenza sul grafico
plt.margins(0.05)
ax.fill_between(np.arange(8, 151), icMinLDH19, icMaxLDH19, alpha=0.2, label="18/19 confidence Interval 95%")


# ora faccio lineplot e fascia per LDH20
sns.lineplot(x="giorno", y="media", color="red", data=datiLDH20, marker=".", markersize=8, label="2019/20", ax=ax)
# disegno la linea che indica il limite patologico/normale
sns.lineplot(x=np.arange(8, 272), y=[220] * 264, color="black", label="normal/pathological limit", ax=ax)

# dato che non mostra gli intervalli di confidenza li calcolo io
# per prima cosa devo calcolare n (di cui poi farò la radice).
#  L'ho già calcolato prima, devo solo aggiungergli gli utlimi due valori, che mancano
nLDH20.insert(49, 1)
nLDH20.insert(50, 1)
nLDH20.append(142)
nLDH20.append(133)
nLDH20.remove(1)
nLDH20.remove(1)
print(len(nLDH20))
print(len(stdevGiorniPrecedentiLDH20))
print(len(medieGiorniPrecedentiLDH20))

cnt = 0
icLDH20 = []
for i in datiLDH20["dev standard"]:
    icLDH20.append(1.96 * i / sqrt(nLDH20[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinLDH20 = []
cnt = 0
for i in datiLDH20["dev standard"]:
    icMinLDH20.append(datiLDH20["media"][cnt] - icLDH20[cnt])
    cnt += 1
icMaxLDH20 = []
cnt = 0
for i in datiLDH20["dev standard"]:
    icMaxLDH20.append(datiLDH20["media"][cnt] + icLDH20[cnt])
    cnt += 1

# disegno gli intervalli di confidenza sul grafico
plt.margins(0.05)
ax.fill_between(a + b, icMinLDH20, icMaxLDH20, alpha=0.2, color="red",
                label="19/20 confidence interval 95%")
plt.xlabel("time (day)", fontsize=12)
plt.ylabel("previous 7 day mean LDH (U/L)", fontsize=12)
plt.xlim(8, 151.2)
plt.ylim(150, 580)
plt.xticks([8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 151])
ax.tick_params(which='major', length=8, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([30, 61, 92, 121], minor=True)
ax.tick_params(which='minor', length=4, color='black')
ax.text(8, 112, "N", size=10)
ax.text(30, 112, "D", size=10)
ax.text(61, 112, "J", size=10)
ax.text(92, 112, "F", size=10)
ax.text(121, 112, "M", size=10)



# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda, ma devo fare un bbox per metterla in una posizione comoda

plt.legend(bbox_to_anchor=(0.1, 0.97), loc="upper center", ncol=1, fancybox=True,
           frameon=False)


# ora creo il grafico da mettere sopra
# per prima cosa devo prendere i dati giornalieri
numeroPositivi = pd.read_excel("DatiPositivi2.xlsx")
numeroPositiviGiorno = []
for i in numeroPositivi["numero positivi"]:
    numeroPositiviGiorno.append(i)
# faccio il grafico per il numero di positivi giornaliero
ax1 = plt.subplot(gs[0])
sns.lineplot(x=np.arange(117, 152), y=numeroPositiviGiorno, ax=ax1)
plt.xlim(8, 151)
plt.ylim(0, max(numeroPositiviGiorno))
plt.xlim(8, 151)
plt.ylabel("No. of COVID \n positive tests", rotation=0, fontsize=12)
ax1.yaxis.set_label_coords(-0.05, 0.45)
ax1.set_xticks([])
ax1.yaxis.tick_right()  # metto i ticks sulla destra

plt.show()


# faccio lo stesso grafico per AST

# devo calcolare per ogni giorno la media dei valori dei 7 giorni precedenti.
# per i primi 7 giorni non ha senso farlo.
i = 0
listaSupporto = []
# per prima cosa faccio un ciclo nel quale aggiungo i valori dei primi 7 giorni alla listaSupporto
fine = False
while fine != True:
    if AST19ConZeri["giorno"][i] == 8:
        fine = True
    else:
        listaSupporto.append(AST19ConZeri["valore"][i])
    i += 1


# definisco una funzione di supporto
def aggiornaListaSupportoAST19(counter, lista, c):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(AST19ConZeri["valore"][r])
        r += 1
    return nuovaLista


# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiornoAST19 = []
i = 0
while i < len(AST19ConZeri) - 1:
    if AST19ConZeri["giorno"][i] != AST19ConZeri["giorno"][i + 1]:
        indiciCambioGiornoAST19.append(i)
    i += 1

i = 19  # 556 perchè ho già messo i valori della prima settimana
cnt = 0  # indica il giorno da cui parte la lista di supporto
medieGiorniPrecedentiAST19 = []  # contiene le medie dei valori dei 7 giorni precedenti
stdevGiorniPrecedentiAST19 = []  # conterrà le deviazioni standard dei valori dei 7 giorni precedenti
listaSupporto = [i for i in listaSupporto if i != 0]
medieGiorniPrecedentiAST19.append(mean(listaSupporto))  # così aggiungo la media dei primi 7 giorni
stdevGiorniPrecedentiAST19.append(statistics.stdev(listaSupporto))
nAST19 = [17]
while i < len(AST19) - 1:
    if i in indiciCambioGiornoAST19:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(AST19ConZeri["valore"][i])
        listaSupporto = aggiornaListaSupportoAST19(indiciCambioGiornoAST19[cnt], listaSupporto,
                                                   indiciCambioGiornoAST19[cnt + 7])
        listaSupporto = [i for i in listaSupporto if i != 0]
        nAST19.append(len(listaSupporto))
        medieGiorniPrecedentiAST19.append(mean(listaSupporto))
        stdevGiorniPrecedentiAST19.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(AST19ConZeri["valore"][i])
    i += 1


medieGiorniPrecedentiAST19[len(medieGiorniPrecedentiAST19) -3] = 36.6
medieGiorniPrecedentiAST19[len(medieGiorniPrecedentiAST19) -2] = 34.2
medieGiorniPrecedentiAST19[len(medieGiorniPrecedentiAST19) -1] = 28.48
medieGiorniPrecedentiAST19.append(28.9)
medieGiorniPrecedentiAST19.append(29.1)


stdevGiorniPrecedentiAST19.append(15.7)
stdevGiorniPrecedentiAST19.append(17.9)
stdevGiorniPrecedentiAST19.append(16.9)
stdevGiorniPrecedentiAST19.remove(33.15116890850155)






# ora faccio la stessa cosa per AST20
i = 0
listaSupporto = []
# per prima cosa faccio un ciclo nel quale aggiungo i valori dei primi 7 giorni alla listaSupporto
fine = False
while fine != True:
    if AST20ConZeri["giorno"][i] == 8:
        fine = True
    else:
        listaSupporto.append(AST20ConZeri["valore"][i])
    i += 1


# definisco una funzione di supporto
def aggiornaListaSupportoAST20(counter, lista, c):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(AST20ConZeri["valore"][r])
        r += 1
    return nuovaLista


# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiornoAST20 = []
i = 0
while i < len(AST20ConZeri) - 1:
    if AST20ConZeri["giorno"][i] != AST20ConZeri["giorno"][i + 1]:
        indiciCambioGiornoAST20.append(i)
    i += 1
i = 10  # 10 perchè ho già messo i valori della prima settimana
cnt = 0  # indica il giorno da cui parte la lista di supporto
medieGiorniPrecedentiAST20 = []  # contiene le medie dei valori dei 7 giorni precedenti
stdevGiorniPrecedentiAST20 = []  # contiene le dev standard dei valori dei 7 giorni precedenti
listaSupporto = [i for i in listaSupporto if i != 0]
medieGiorniPrecedentiAST20.append(mean(listaSupporto))  # così aggiungo la media dei primi 7 giorni
stdevGiorniPrecedentiAST20.append(statistics.stdev(listaSupporto))
nAST20 = [10]
while i < len(AST20ConZeri) - 1:
    if i in indiciCambioGiornoAST20:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(AST20ConZeri["valore"][i])
        if cnt < 144:
            listaSupporto = aggiornaListaSupportoAST20(indiciCambioGiornoAST20[cnt], listaSupporto,
                                                       indiciCambioGiornoAST20[cnt + 7])
        else:
            listaSupporto = aggiornaListaSupportoAST20(indiciCambioGiornoAST20[cnt], listaSupporto,
                                                       indiciCambioGiornoAST20[cnt + 6])
        listaSupporto = [i for i in listaSupporto if i != 0]
        nAST20.append(len(listaSupporto))
        medieGiorniPrecedentiAST20.append(mean(listaSupporto))
        stdevGiorniPrecedentiAST20.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(AST20ConZeri["valore"][i])
    i += 1

medieGiorniPrecedentiAST20[len(medieGiorniPrecedentiAST20) - 1] = 57.13
stdevGiorniPrecedentiAST20[len(stdevGiorniPrecedentiAST20) - 1] = 52.99

print(len(medieGiorniPrecedentiAST19))
print(len(stdevGiorniPrecedentiAST19))


# ora creo un dataframe con le medie dei 7 giorni preceedenti e il giorno

dict = {
    "giorno": np.arange(8, 152),
    "media": medieGiorniPrecedentiAST19,
    "dev standard": stdevGiorniPrecedentiAST19
}
datiAST19 = pd.DataFrame(dict)
dict2 = {
    "giorno": np.arange(8, 154),
    "media": medieGiorniPrecedentiAST20,
    "dev standard": stdevGiorniPrecedentiAST20
}
datiAST20 = pd.DataFrame(dict2)

# inizio a disegnare il grafico
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])
gs.update(wspace=0.03, hspace=0.03)
ax = plt.subplot(gs[1])

sns.lineplot(x="giorno", y="media", data=datiAST19, marker=".", markersize=9, label="2018/19")

# creo la linea verticale tratteggiata
plt.plot([117] * 175, np.arange(0, 175), linestyle='dashed', color="black", linewidth=0.8)

# disegno la linea che indica il limite patologico/normale
sns.lineplot(x=np.arange(8, 272), y=[35] * 264, color="black", label="normal/pathological limit")


# dato che non mostra gli intervalli di confidenza li calcolo io
# per prima cosa devo aggiungere gli utlimi valori a n già calcolato prima
nAST19.append(30)
nAST19.append(22)
nAST19.append(22)


cnt = 0
icAST19 = []
for i in datiAST19["dev standard"]:
    icAST19.append(1.96 * i / sqrt(nAST19[cnt]))
    cnt += 1

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinAST19 = []
cnt = 0
for i in datiAST19["dev standard"]:
    icMinAST19.append(datiAST19["media"][cnt] - icAST19[cnt])
    cnt += 1
icMaxAST19 = []
cnt = 0
for i in datiAST19["dev standard"]:
    icMaxAST19.append(datiAST19["media"][cnt] + icAST19[cnt])
    cnt += 1

# disegno gli intervalli di confidenza sul grafico
plt.margins(0.05)
ax.fill_between(np.arange(8, 152), icMinAST19, icMaxAST19, alpha=0.2, label="18/19 confidence Interval 95%")


# ora faccio lineplot e fascia per AST20
sns.lineplot(x="giorno", y="media", color="red", data=datiAST20, marker=".", markersize=9, label="2019/20")

# dato che non mostra gli intervalli di confidenza li calcolo io
nAST20[len(nAST20) - 1] = 124



cnt = 0
icAST20 = []
for i in datiAST20["dev standard"]:
    icAST20.append(1.96 * i / sqrt(nAST20[cnt]))

# ora creo due liste una con gli estremi superiori e una con gli estremi inferiori
icMinAST20 = []
cnt = 0
for i in datiAST20["dev standard"]:
    icMinAST20.append(datiAST20["media"][cnt] - icAST20[cnt])
    cnt += 1
icMaxAST20 = []
cnt = 0
for i in datiAST20["dev standard"]:
    icMaxAST20.append(datiAST20["media"][cnt] + icAST20[cnt])
    cnt += 1

# disegno gli intervalli di confidenza sul grafico
plt.margins(0.05)
ax.fill_between(np.arange(8, 154), icMinAST20, icMaxAST20, alpha=0.2, color="red",
                label="19/20 confidence interval 95%")
plt.xlabel("time(day)", fontsize=12)
plt.ylabel("previous 7 day mean AST (U/L)", fontsize=12)
plt.xlim(8, 154)
plt.ylim(0, 100)
plt.xticks([8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 152])
ax.tick_params(which='major', length=8, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([30, 61, 92, 121], minor=True)
ax.tick_params(which='minor', length=4, color='black')
ax.text(8, -12, "N", size=10)
ax.text(30, -12, "D", size=10)
ax.text(61, -12, "J", size=10)
ax.text(92, -12, "F", size=10)
ax.text(121, -12, "M", size=10)


# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)

ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda, ma devo fare un bbox per metterla in una posizione comoda

plt.legend(bbox_to_anchor=(0.1, 0.97), loc="upper center", ncol=1, fancybox=True,
           frameon=False)


# ora mostro sullo stesso grafico il numero di positivi
# per prima cosa devo prendere i dati settimanali
numeroPositiviGiorno = []
for i in numeroPositivi["numero positivi"]:
    numeroPositiviGiorno.append(i)

# ora creo il grafico da mettere sopra
# per prima cosa devo prendere i dati settimanali
numeroPositiviGiorno = []
for i in numeroPositivi["numero positivi"]:
    numeroPositiviGiorno.append(i)

# faccio il grafico per il numero di positivi giornaliero
numeroPositivi = pd.read_excel("DatiPositivi2.xlsx")
numeroPositiviGiorno = []
for i in numeroPositivi["numero positivi"]:
    numeroPositiviGiorno.append(i)
# faccio il grafico per il numero di positivi giornaliero
ax1 = plt.subplot(gs[0])
sns.lineplot(x=np.arange(117, 152), y=numeroPositiviGiorno, ax=ax1)
plt.xlim(8, 151)
plt.ylim(0, max(numeroPositiviGiorno))
plt.xlim(8, 151)
plt.ylabel("No. of COVID \n positive tests", rotation=0, fontsize=12)
ax1.yaxis.set_label_coords(-0.05, 0.45)
ax1.set_xticks([])
ax1.yaxis.tick_right()  # metto i ticks sulla destra

plt.show()

print(datiAST20["giorno"])
print(datiAST20["media"])
print(datiAST20["dev standard"])



# ora faccio il grafico età con la deviazione standard
# per prima cosa calcolo l'età di ognuno e aggiungo una colonna per raccoglierla
etàLDH19 = []
for i in LDH19["data di nascita"]:
    etàLDH19.append(2020 - int(i[6: 11]))
etàLDH20 = []
for i in LDH20["data di nascita"]:
    etàLDH20.append(2020 - int(i[6: 11]))
etàAST19 = []
for i in AST19["data nascita"]:
    etàAST19.append(2020 - int(i[6: 11]))
etàAST20 = []
for i in AST20["data nascita"]:
    etàAST20.append(2020 - int(i[6: 11]))
etàLDHTOT = etàLDH19 + etàLDH20
etàASTTOT = etàAST19 + etàAST20
print(etàLDHTOT)

# inserisco nel dataframe una colonna con questo dato
LDH19["età"] = etàLDH19
LDH20["età"] = etàLDH20
AST19["età"] = etàAST19
AST20["età"] = etàAST20

