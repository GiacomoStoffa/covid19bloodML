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
AST19 = pd.read_excel("TOTALE_18_19_AST-.xlsx")
AST20 = pd.read_excel("TOTALE_19_20_AST-.xlsx")
LDH19 = pd.read_excel("TOTALE_18_19_LDH-.xlsx")
LDH20 = pd.read_excel("TOTALE_19_20_LDH-.xlsx")
# AST19 = pd.read_excel("ASTSintomi19.xlsx")
# AST20 = pd.read_excel("ASTSintomi20.xlsx")
# LDH19 = pd.read_excel("LDHSintomi19.xlsx")
# LDH20 = pd.read_excel("LDHSintomi19.xlsx")


# leggo i dati dei tamponi
numeroPositivi = pd.read_excel("datiPositivi.xlsx")


# parto con quello per LDH
# calcolo le mediane

# per prima cosa tronco i valori superiori a mille
LDH19["valore"][LDH19["valore"] > 1000] = 1000
LDH20["valore"][LDH20["valore"] > 1000] = 1000
AST20["valore"][AST20["valore"] > 1000] = 1000
AST19["valore"][AST19["valore"] > 1000] = 1000


# inserisco nei quattro dataframe una colonna che indica la settimana
settimana = []
cnt = 1
i = 0
while i < len(LDH19) - 1:
    if (LDH19["giorno"][i] % 7 == 0) & (LDH19["giorno"][i + 1] % 7 != 0):
        cnt += 1
    settimana.append(cnt)
    i += 1
# aggiungo la settimana per l'utlimo valore che è rimansto fuori
settimana.append(cnt)
LDH19["settimana"] = settimana

# stessa cosa per LDH20
settimana = []
cnt = 1
i = 0
while i < len(LDH20) - 1:
    if (LDH20["giorno"][i] % 7 == 0) & (LDH20["giorno"][i + 1] % 7 != 0):
        cnt += 1
    settimana.append(cnt)
    i += 1
# aggiungo la settimana per l'ultimo valore che è rimansto fuori
settimana.append(cnt)
LDH20["settimana"] = settimana

# stessa cosa per AST19
settimana = []
cnt = 1
i = 0
while i < len(AST19) - 1:
    if (AST19["giorno"][i] % 7 == 0) & (AST19["giorno"][i + 1] % 7 != 0):
        cnt += 1
    settimana.append(cnt)
    i += 1
# aggiungo la settimana per l'utlimo valore che è rimasto fuori
settimana.append(cnt)
AST19["settimana"] = settimana

# stessa cosa per AST20
settimana = []
cnt = 1
i = 0
while i < len(AST20) - 1:
    if (AST20["data"][i] % 7 == 0) & (AST20["data"][i + 1] % 7 != 0):
        cnt += 1
    settimana.append(cnt)
    i += 1
# aggiungo la settimana per l'ultimo valore che è rimasto fuori
settimana.append(cnt)
AST20["settimana"] = settimana


fig, ax = plt.subplots()
sns.lineplot(x="settimana", y="valore", color="blue", data=LDH19, estimator=np.median, marker="o", ci=95,
             err_style="bars", label="2018/19")
sns.lineplot(x="settimana", y="valore", color="red", data=LDH20, estimator=np.median, marker="o", ci=95,
             err_style="bars", label="2019/20")


# disegno la fascia colorata che indica la deviazione standard dei dati del 18/19
sns.lineplot(x="settimana", y="valore", color="blue", data=LDH19, marker="o", ci="sd",
             err_style="band", alpha=0, err_kws={"alpha": 0.2, "label": "18/19 standard deviation"})
# stampo tutte le deviazioni standard di ogni settimana
valoriSettimana = []
i = 0
while i < len(LDH19) - 1:
    if (LDH19["giorno"][i] % 7 == 0) & (LDH19["giorno"][i + 1] % 7 != 0):
        print(int(statistics.stdev(valoriSettimana)))
        valoriSettimana = []
    valoriSettimana.append(LDH19["valore"][i])
    i += 1

# disegno la linea che indica il limite patologico
sns.lineplot(x=np.arange(0, 40), y=[220] * 40, color="black", label="normal/pathological limit")


ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.xlim(1, 39.1)
plt.ylim(60, 590)
plt.xticks([1, 10, 20, 30, 39])
plt.xlabel("Time (weeks)")
# scrivo le indicazioni relative alla stagione
ax.text(1, -0.3, "Summer", size=12, ha='center')
ax.text(4, -0.3, "Fall", size=12, ha='center')
ax.text(16, -0.3, "Winter", size=12, ha='center')
ax.text(28, -0.3, "Springer", size=12, ha='center')
plt.ylabel("LDH (U/L)")

# aggiungo la legenda
plt.legend(loc="upper left", frameon=False)

# ora mostro sullo stesso grafico il numero di positivi
# per prima cosa devo prendere i dati settimanali
numeroPositiviSettimana = []
for i in numeroPositivi["riepilogo settimanale"]:
    if not isnan(i):
        numeroPositiviSettimana.append(i)

x = numeroPositiviSettimana

axins = inset_axes(ax,
                   width="36.8%",
                   height="3%",
                   loc='lower center',
                   bbox_to_anchor=(0.313, 0.8, 1, 1.5),
                   bbox_transform=ax.transAxes,
                   borderpad=0,
                   )

cmap = LinearSegmentedColormap.from_list("", ["blue", "red"], 512)

axins.imshow([x], cmap=cmap, aspect="auto",
             extent=[x[0] - np.diff(x)[0] / 2, x[-1] + np.diff(x)[0] / 2, 0, 1])
axins.set_title("positive RT-PCR test at Emergency room")
axins.tick_params(right=False, labelright=False, labelbottom=False, left=False, labelleft=False)
plt.show()


# stampo tutte le deviazioni standard di ogni settimana
valoriSettimana = []
i = 0
while i < len(LDH19) - 1:
    if (LDH19["giorno"][i] % 7 == 0) & (LDH19["giorno"][i + 1] % 7 != 0):
        print(int(statistics.stdev(valoriSettimana)))
        valoriSettimana = []
    valoriSettimana.append(LDH19["valore"][i])
    i += 1
    
# faccio lo stesso identico grafico, ma inserendo sopra il grafico con le media delle età e la deviazioni standard
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 6])
gs.update(wspace=0.03, hspace=0.03)
ax0 = plt.subplot(gs[2])
sns.lineplot(x="settimana", y="valore", color="blue", data=LDH19, estimator=np.mean, marker="o", ci=95,
             err_style="bars", label="2018/19", ax=ax0)
sns.lineplot(x="settimana", y="valore", color="red", data=LDH20, estimator=np.mean, marker="o", ci=95,
             err_style="bars", label="2019/20", ax=ax0)


# disegno la fascia colorata che indica la deviazione standard dei dati del 18/19
sns.lineplot(x="settimana", y="valore", data=LDH19, marker="o", ci="sd", ax=ax0,
             err_style="band", alpha=0, err_kws={"alpha": 0.2, "label": "18/19 standard deviation"})
# sns.lineplot(x="settimana", y="valore", color="green", data=LDH19, estimator=np.mean, marker="o", ci=95,
#             err_style="bars", label="2019/20", ax=ax0)


# disegno la linea che indica il limite patologico
sns.lineplot(x=np.arange(0, 40), y=[220] * 40, color="black", label="normal/pathological limit", ax=ax0)

# creo la linea verticale tratteggiata
plt.plot([25] * 530, np.arange(60, 590), linestyle='dashed', color="black", linewidth=0.8)

plt.xlim(1, 39.1)
plt.ylim(60, 590)
plt.xticks([1, 10, 20, 30, 39])
plt.xlabel("Time (weeks)", fontsize=12)
# scrivo le indicazioni relative al mese. Uso i minor ticks
ax0.set_xticks([4.86, 9.29, 13.58, 18, 22.43, 26.57, 31, 35.29], minor=True)
ax0.tick_params(which='minor', length=3, color='black')
ax0.text(0.945, 10, "S", size=10)
ax0.text(4.75, 10, "O", size=10)
ax0.text(9.2, 10, "N", size=10)
ax0.text(13.49, 10, "D", size=10)
ax0.text(18, 10, "J", size=10)
ax0.text(22.43, 10, "F", size=10)
ax0.text(26.5, 10, "M", size=10)
ax0.text(30.92, 10, "A", size=10)
ax0.text(35.2, 10, "M", size=10)
# sposto l'etichetta dell'asse più in basso
ax0.xaxis.set_label_coords(0.5, -0.15)

plt.ylabel("LDH (U/L)", fontsize=12, rotation=0)
ax0.yaxis.set_label_coords(-0.07, 0.5)

# aggiungo la legenda
plt.legend(loc="upper left", frameon=False)

# ora mostro sullo stesso grafico il numero di positivi
# per prima cosa devo prendere i dati settimanali
numeroPositiviSettimana = []
for i in numeroPositivi["riepilogo settimanale"]:
    if not isnan(i):
        numeroPositiviSettimana.append(i)
numeroPositiviSettimana.append(1)
ax1 = plt.subplot(gs[1])
sns.lineplot(x=np.arange(25, 39), y=numeroPositiviSettimana, ax=ax1)
plt.xlim(1, 39)
ax1.set_xticks([])
plt.ylabel("No. of COVID \n positive tests", rotation=0, fontsize=12)
ax1.yaxis.tick_right()  # metto i ticks sulla destra
ax1.yaxis.set_label_coords(-0.07, 0.5)  # poisiziono la label dell'asse y
plt.ylim(0, max(numeroPositiviSettimana))


# ora inserisco il grafico con le età e deviazioni standard
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

# inizio a disegnare il grafico
ax2 = plt.subplot(gs[0])
sns.lineplot(x="settimana", y="età", estimator=np.mean, data=LDH19, ci="sd", ax=ax2)
sns.lineplot(x="settimana", y="età", estimator=np.mean, data=LDH20, ci="sd", ax=ax2, color="red")
ax2.yaxis.set_label_coords(-0.06, 0.5)
plt.xlim(1, 39)
ax2.yaxis.tick_right()  # metto i ticks sulla destra
plt.ylabel("mean age (yrs) \n standard deviation", rotation=0, fontsize=12)
plt.xlabel("")
plt.xticks([])
plt.show()




# ora faccio lo stesso grafico per i valori AST
fig, ax = plt.subplots()
sns.lineplot(x="settimana", y="valore", color="blue", data=AST19, estimator=np.median, marker="o", ci=95,
             err_style="bars", label="2018/19")
sns.lineplot(x="settimana", y="valore", color="red", data=AST20, estimator=np.median, marker="o", ci=95,
             err_style="bars", label="2019/20")
# disegno la fascia colorata che indica la deviazione standard dei dati del 18/19
sns.lineplot(x="settimana", y="valore", color="blue", data=AST19, marker="o", ci="sd",
             err_style="band", alpha=0, err_kws={"alpha": 0.2, "label": "18/19 standard deviation"})
# stampo tutte le deviazioni standard di ogni settimana
valoriSettimana = []
i = 0
while i < len(AST19) - 1:
    if (AST19["giorno"][i] % 7 == 0) & (AST19["giorno"][i + 1] % 7 != 0):
        print(int(statistics.stdev(valoriSettimana)))
        valoriSettimana = []
    valoriSettimana.append(AST19["valore"][i])
    i += 1

# disegno la linea che indica il limite patologico
sns.lineplot(x=np.arange(0, 40), y=[35] * 40, color="black", label="normal/pathological limit")

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
# scrivo le indicazioni relative alla stagione
ax.text(1, -8, "Summer", size=12)
ax.text(4, -8, "Fall", size=12)
ax.text(16, -8, "Winter", size=12)
ax.text(28, -8, "Springer", size=12)
plt.xlim(1, 39.1)
plt.ylim(0, 120)
plt.xticks([1, 10, 20, 30, 39])
plt.xlabel("Time (weeks)")
plt.ylabel("AST (U/L)")

# aggiungo la legenda
plt.legend(loc="upper left", frameon=False)

# ora mostro sullo stesso grafico il numero di positivi
# per prima cosa devo prendere i dati settimanali
numeroPositiviSettimana = []
for i in numeroPositivi["riepilogo settimanale"]:
    if not isnan(i):
        numeroPositiviSettimana.append(i)

x = numeroPositiviSettimana
axins = inset_axes(ax0,
                   width="36.8%",
                   height="3%",
                   loc='lower center',
                   bbox_to_anchor=(0.313, 0.8, 1, 1.5),
                   bbox_transform=ax0.transAxes,
                   borderpad=0,
                   )

cmap = LinearSegmentedColormap.from_list("", ["blue", "red"])
axins.imshow([x], cmap=cmap, aspect="auto",
             extent=[x[0] - np.diff(x)[0] / 2, x[-1] + np.diff(x)[0] / 2, 0, 1])
axins.tick_params(right=False, labelright=False, labelbottom=False, left=False, labelleft=False)
axins.set_title("positive RT-PCR test at Emergency room")
plt.show()



# ora faccio lo stesso identico grafico, ma con sopra gli altri due grafici
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 6])
gs.update(wspace=0.03, hspace=0.03)
ax0 = plt.subplot(gs[2])
sns.lineplot(x="settimana", y="valore", color="blue", data=AST19, estimator=np.mean, marker="o", ci=95,
             err_style="bars", label="2018/19", ax=ax0)
sns.lineplot(x="settimana", y="valore", color="red", data=AST20, estimator=np.mean, marker="o", ci=95,
             err_style="bars", label="2019/20", ax=ax0)
# disegno la fascia colorata che indica la deviazione standard dei dati del 18/19
sns.lineplot(x="settimana", y="valore", data=AST19, marker="o", ci="sd", ax=ax0,
             err_style="band", alpha=0, err_kws={"alpha": 0.2, "label": "18/19 standard deviation"})



# stampo tutte le deviazioni standard di ogni settimana
valoriSettimana = []
i = 0
while i < len(AST19) - 1:
    if (AST19["giorno"][i] % 7 == 0) & (AST19["giorno"][i + 1] % 7 != 0):
        print(int(statistics.stdev(valoriSettimana)))
        valoriSettimana = []
    valoriSettimana.append(AST19["valore"][i])
    i += 1

# disegno la linea che indica il limite patologico
sns.lineplot(x=np.arange(0, 40), y=[35] * 40, color="black", label="normal/pathological limit", ax=ax0)

# creo la linea verticale tratteggiata
plt.plot([25] * 120, np.arange(0, 120), linestyle='dashed', color="black", linewidth=0.8)

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax0.set_xticks([4.86, 9.29, 13.58, 18, 22.43, 26.57, 31, 35.29], minor=True)
ax0.tick_params(which='minor', length=3, color='black')
ax0.text(0.945, -11, "S", size=10)
ax0.text(4.75, -11, "O", size=10)
ax0.text(9.2, -11, "N", size=10)
ax0.text(13.49, -11, "D", size=10)
ax0.text(18, -11, "J", size=10)
ax0.text(22.43, -11, "F", size=10)
ax0.text(26.5, -11, "M", size=10)
ax0.text(30.92, -11, "A", size=10)
ax0.text(35.2, -11, "M", size=10)

plt.xlim(1, 39.1)
plt.ylim(0, 120)
plt.xticks([1, 10, 20, 30, 39])
plt.xlabel("Time (weeks)", fontsize=12)
# sposto l'etichetta dell'asse più in basso
ax0.xaxis.set_label_coords(0.5, -0.15)

ax0.yaxis.set_label_coords(-0.07, 0.5)

plt.ylabel("AST (U/L)", fontsize=12, rotation=0)

# aggiungo la legenda
plt.legend(loc="upper left", frameon=False)

# ora mostro sullo stesso grafico il numero di positivi
numeroPositiviSettimana = []
for i in numeroPositivi["riepilogo settimanale"]:
    if not isnan(i):
        numeroPositiviSettimana.append(i)
numeroPositiviSettimana.append(1)
ax1 = plt.subplot(gs[1])
sns.lineplot(x=np.arange(25, 39), y=numeroPositiviSettimana, ax=ax1)
plt.xlim(1, 39)
ax1.set_xticks([])
plt.ylabel("No. of COVID \n positive tests", rotation=0, fontsize=12)
ax1.yaxis.tick_right()  # metto i ticks sulla destra
ax1.yaxis.set_label_coords(-0.07, 0.5)  # poisiziono la label dell'asse y
plt.ylim(0, max(numeroPositiviSettimana))





# disegno il grafico sopra
ax1 = plt.subplot(gs[0])
sns.lineplot(x="settimana", y="età", estimator=np.mean, data=AST19, ci="sd", ax=ax1)
sns.lineplot(x="settimana", y="età", estimator=np.mean, data=AST20, ci="sd", ax=ax1, color="red")
plt.xlim(1, 39)
plt.ylabel("mean age (yrs) \n standard deviation", rotation=0)
ax1.yaxis.set_label_coords(-0.08, 0.3)
plt.title("mean age 2018/19")
plt.xlabel("")
plt.xticks([])
plt.show()






# passo aall'altro tipo di grafico
# devo calcolare per ogni giorno la media dei valori dei 7 giorni precedenti.
# per i primi 7 giorni non ha senso farlo.
i = 0
listaSupporto = []
# per prima cosa faccio un ciclo nel quale aggiungo i valori dei primi 7 giorni alla listaSupporto
fine = False
while fine != True:
    if LDH19["giorno"][i] == 8:
        fine = True
    else:
        listaSupporto.append(LDH19["valore"][i])
    i += 1


# definisco una funzione di supporto
def aggiornaListaSupporto(counter, lista, c):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(LDH19["valore"][r])
        r += 1
    return nuovaLista


# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiornoLDH19 = []
i = 0
while i < len(LDH19) - 1:
    if LDH19["giorno"][i] != LDH19["giorno"][i + 1]:
        indiciCambioGiornoLDH19.append(i)
    i += 1

i = 299  # 298 perchè ho già messo i valori della prima settimana
cnt = 0  # indica il giorno da cui parte la lista di supporto
medieGiorniPrecedentiLDH19 = []  # contiene le medie dei valori dei 7 giorni precedenti
stdevGiorniPrecedentiLDH19 = []  # conterrà le deviazioni standard dei valori dei 7 giorniprecedenti
medieGiorniPrecedentiLDH19.append(mean(listaSupporto))  # così aggiungo la media dei primi 7 giorni
stdevGiorniPrecedentiLDH19.append(statistics.stdev(listaSupporto))
while i < len(LDH19) - 1:
    if i in indiciCambioGiornoLDH19:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        print(listaSupporto)
        listaSupporto.append(LDH19["valore"][i])
        listaSupporto = aggiornaListaSupporto(indiciCambioGiornoLDH19[cnt], listaSupporto, indiciCambioGiornoLDH19[cnt + 7])
        medieGiorniPrecedentiLDH19.append(mean(listaSupporto))
        stdevGiorniPrecedentiLDH19.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(LDH19["valore"][i])
    i += 1
medieGiorniPrecedentiLDH19[len(medieGiorniPrecedentiLDH19) - 1] = 242.29
medieGiorniPrecedentiLDH19.append(241.32)
stdevGiorniPrecedentiLDH19.append(84.18)

# ora faccio la stessa cosa per LDH20
i = 0
listaSupporto = []
# per prima cosa faccio un ciclo nel quale aggiungo i valori dei primi 7 giorni alla listaSupporto
fine = False
while fine != True:
    if LDH20["giorno"][i] == 8:
        fine = True
    else:
        listaSupporto.append(LDH20["valore"][i])
    i += 1


# definisco una funzione di supporto
def aggiornaListaSupporto20(counter, lista, c):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(LDH20["valore"][r])
        r += 1
    return nuovaLista


# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiornoLDH20 = []
i = 0
while i < len(LDH20) - 1:
    if LDH20["giorno"][i] != LDH20["giorno"][i + 1]:
        indiciCambioGiornoLDH20.append(i)
    i += 1
i = 301  # 301 perchè ho già messo i valori della prima settimana
cnt = 0  # indica il giorno da cui parte la lista di supporto
medieGiorniPrecedentiLDH20 = []  # contiene le medie dei valori dei 7 giorni precedenti
stdevGiorniPrecedentiLDH20 = []  # contiene le dev standard dei valori dei 7 giorni precedenti
medieGiorniPrecedentiLDH20.append(mean(listaSupporto))  # così aggiungo la media dei primi 7 giorni
stdevGiorniPrecedentiLDH20.append(statistics.stdev(listaSupporto))
while i < len(LDH20) - 1:
    if i in indiciCambioGiornoLDH20:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(LDH20["valore"][i])
        if cnt < 261:
            listaSupporto = aggiornaListaSupporto20(indiciCambioGiornoLDH20[cnt], listaSupporto, indiciCambioGiornoLDH20[cnt + 7])
        else:
            listaSupporto = aggiornaListaSupporto20(indiciCambioGiornoLDH20[cnt], listaSupporto, indiciCambioGiornoLDH20[cnt + 6])
        medieGiorniPrecedentiLDH20.append(mean(listaSupporto))
        stdevGiorniPrecedentiLDH20.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(LDH20["valore"][i])
    i += 1
medieGiorniPrecedentiLDH20[len(medieGiorniPrecedentiLDH20) - 1] = 255.67
stdevGiorniPrecedentiLDH20[len(stdevGiorniPrecedentiLDH20) - 1] = 102.61

# ora creo un dataframe con le medie dei 7 giorni preceedenti e il giorno

dict = {
    "giorno": np.arange(8, 272),
    "media": medieGiorniPrecedentiLDH19,
    "dev standard": stdevGiorniPrecedentiLDH19
}
datiLDH19 = pd.DataFrame(dict)
dict2 = {
    "giorno": np.arange(8, 271),
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
plt.plot([175] * 240, np.arange(180, 420), linestyle='dashed', color="black", linewidth=0.8)

# dato che non mostra gli intervalli di confidenza li calcolo io
# per prima cosa devo calcolare n (di cui poi farò la radice)
n = [299]  # inserisco il primo valore a mano opi li inserisco con un ciclo
cnt = 0
for i in range(7, len(indiciCambioGiornoLDH19)):
    n.append(indiciCambioGiornoLDH19[i] - indiciCambioGiornoLDH19[cnt])
    cnt += 1
n.append(280)  # aggiungo l'ultimo valore mancante

icLDH19 = []
cnt = 0
for i in datiLDH19["dev standard"]:
    icLDH19.append(1.96 * i / sqrt(n[cnt]))
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
ax.fill_between(np.arange(8, 272), icMinLDH19, icMaxLDH19, alpha=0.2, label="18/19 confidence Interval 95%")

# ora faccio lineplot e fascia per LDH20
sns.lineplot(x="giorno", y="media", color="red", data=datiLDH20, marker=".", markersize=8, label="2019/20", ax=ax)
# disegno la linea che indica il limite patologico/normale
sns.lineplot(x=np.arange(8, 272), y=[220] * 264, color="black", label="normal/pathological limit", ax=ax)

# dato che non mostra gli intervalli di confidenza li calcolo io
# per prima cosa devo calcolare n (di cui poi farò la radice)
n = [321]  # aggiungo manualmente il primo, poi li calcolcolo con un ciclo
cnt = 0
for i in range(7, len(indiciCambioGiornoLDH20)):
    n.append(indiciCambioGiornoLDH20[i] - indiciCambioGiornoLDH20[cnt])
    cnt += 1
n.append(362)  # aggiungo l'ultimo mancante

cnt = 0
icLDH20 = []
for i in datiLDH20["dev standard"]:
    icLDH20.append(1.96 * i / sqrt(n[cnt]))
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
ax.fill_between(np.arange(8, 271), icMinLDH20, icMaxLDH20, alpha=0.2, color="red",
                label="19/20 confidence interval 95%")
plt.xlabel("time (day)", fontsize=12)
plt.ylabel("previous 7 day mean LDH (U/L)", fontsize=12)
plt.xlim(8, 270.2)
plt.ylim(180, 420)
plt.xticks([8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230,
            240, 250, 260, 270])
ax.tick_params(which='major', length=8, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([27, 58, 88, 119, 150, 179, 210, 240], minor=True)
ax.tick_params(which='minor', length=4, color='black')
ax.text(7.4, 162, "S", size=10)
ax.text(26.5, 162, "O", size=10)
ax.text(57.5, 162, "N", size=10)
ax.text(87.5, 162, "D", size=10)
ax.text(118.5, 162, "J", size=10)
ax.text(149.5, 162, "F", size=10)
ax.text(178.5, 162, "M", size=10)
ax.text(209.5, 162, "A", size=10)
ax.text(239.5, 162, "M", size=10)

# sposto l'etichetta dell'asse più in basso
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.05, 0.4)

# aggiungo la legenda, ma devo fare un bbox per metterla in una posizione comoda

plt.legend(bbox_to_anchor=(0.1, 0.97), loc="upper center", ncol=1, fancybox=True,
           frameon=False)

# ora creo il grafico da mettere sopra
# per prima cosa devo prendere i dati settimanali
numeroPositiviGiorno = []
for i in numeroPositivi["numero positivi"]:
    numeroPositiviGiorno.append(i)

# faccio il grafico per il numero di positivi giornaliero
ax1 = plt.subplot(gs[0])
sns.lineplot(x=np.arange(175, 271), y=numeroPositiviGiorno, ax=ax1)
plt.xlim(8, 270)
plt.ylim(0, max(numeroPositiviGiorno))
plt.xlim(8, 270)
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
    if AST19["giorno"][i] == 8:
        fine = True
    else:
        listaSupporto.append(AST19["valore"][i])
    i += 1


# definisco una funzione di supporto
def aggiornaListaSupportoAST19(counter, lista, c):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(AST19["valore"][r])
        r += 1
    return nuovaLista


# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiornoAST19 = []
i = 0
while i < len(AST19) - 1:
    if AST19["giorno"][i] != AST19["giorno"][i + 1]:
        indiciCambioGiornoAST19.append(i)
    i += 1

i = 556  # 556 perchè ho già messo i valori della prima settimana
cnt = 0  # indica il giorno da cui parte la lista di supporto
medieGiorniPrecedentiAST19 = []  # contiene le medie dei valori dei 7 giorni precedenti
stdevGiorniPrecedentiAST19 = []  # conterrà le deviazioni standard dei valori dei 7 giorniprecedenti
medieGiorniPrecedentiAST19.append(mean(listaSupporto))  # così aggiungo la media dei primi 7 giorni
stdevGiorniPrecedentiAST19.append(statistics.stdev(listaSupporto))
while i < len(AST19) - 1:
    if i in indiciCambioGiornoAST19:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(AST19["valore"][i])
        listaSupporto = aggiornaListaSupportoAST19(indiciCambioGiornoAST19[cnt], listaSupporto, indiciCambioGiornoAST19[cnt + 7])
        medieGiorniPrecedentiAST19.append(mean(listaSupporto))
        stdevGiorniPrecedentiAST19.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(AST19["valore"][i])
    i += 1
medieGiorniPrecedentiAST19.append(34.37)
stdevGiorniPrecedentiAST19.append(53.77)

# ora faccio la stessa cosa per AST20
i = 0
listaSupporto = []
# per prima cosa faccio un ciclo nel quale aggiungo i valori dei primi 7 giorni alla listaSupporto
fine = False
while fine != True:
    if AST20["data"][i] == 8:
        fine = True
    else:
        listaSupporto.append(AST20["valore"][i])
    i += 1


# definisco una funzione di supporto
def aggiornaListaSupportoAST20(counter, lista, c):
    # copio i valori dopo l'indice counter nella nuova lista e la restituisco
    nuovaLista = []
    r = counter + 1
    while r < c + 1:
        nuovaLista.append(AST20["valore"][r])
        r += 1
    return nuovaLista


# creo una lista con tutti gli indici dove c'è il cambio del giorno
indiciCambioGiornoAST20 = []
i = 0
while i < len(AST20) - 1:
    if AST20["data"][i] != AST20["data"][i + 1]:
        indiciCambioGiornoAST20.append(i)
    i += 1
i = 474  # 474 perchè ho già messo i valori della prima settimana
cnt = 0  # indica il giorno da cui parte la lista di supporto
medieGiorniPrecedentiAST20 = []  # contiene le medie dei valori dei 7 giorni precedenti
stdevGiorniPrecedentiAST20 = []  # contiene le dev standard dei valori dei 7 giorni precedenti
medieGiorniPrecedentiAST20.append(mean(listaSupporto))  # così aggiungo la media dei primi 7 giorni
stdevGiorniPrecedentiAST20.append(statistics.stdev(listaSupporto))
while i < len(AST20) - 1:
    if i in indiciCambioGiornoAST20:
        # significa che sono alla fine dei dati del giorno che sto considerando
        # devo calcolare la media per i 7 giorni precedenti
        # inoltre, devo eliminare dalla lista di supporto tutti i dati vecchi (ovvero di 8 giorni prima)
        listaSupporto.append(AST20["valore"][i])
        if cnt < 261:
            listaSupporto = aggiornaListaSupportoAST20(indiciCambioGiornoAST20[cnt], listaSupporto,
                                                       indiciCambioGiornoAST20[cnt + 7])
        else:
            listaSupporto = aggiornaListaSupportoAST20(indiciCambioGiornoAST20[cnt], listaSupporto,
                                                       indiciCambioGiornoAST20[cnt + 6])
        medieGiorniPrecedentiAST20.append(mean(listaSupporto))
        stdevGiorniPrecedentiAST20.append(statistics.stdev(listaSupporto))
        cnt += 1
    else:
        listaSupporto.append(AST20["valore"][i])
    i += 1
# medieGiorniPrecedentiAST20[len(medieGiorniPrecedentiAST20) - 1] = 255.67
# stdevGiorniPrecedentiAST20[len(stdevGiorniPrecedentiAST20) - 1] = 102.61

stdevGiorniPrecedentiAST20.append(41.06)
medieGiorniPrecedentiAST20.append(36.28)


# ora creo un dataframe con le medie dei 7 giorni preceedenti e il giorno

dict = {
    "giorno": np.arange(8, 272),
    "media": medieGiorniPrecedentiAST19,
    "dev standard": stdevGiorniPrecedentiAST19
}
datiAST19 = pd.DataFrame(dict)
dict2 = {
    "giorno": np.arange(8, 272),
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
plt.plot([175] * 55, np.arange(20, 75), linestyle='dashed', color="black", linewidth=0.8)

# disegno la linea che indica il limite patologico/normale
sns.lineplot(x=np.arange(8, 272), y=[35] * 264, color="black", label="normal/pathological limit")

# dato che non mostra gli intervalli di confidenza li calcolo io
# per prima cosa devo calcolare n (di cui poi farò la radice)
n = [556]  # aggiungo manualmente il primo, poi li calcolcolo con un ciclo
cnt = 0
for i in range(7, len(indiciCambioGiornoAST19)):
    n.append(indiciCambioGiornoAST19[i] - indiciCambioGiornoAST19[cnt])
    cnt += 1
n.append(484)  # aggiungo l'ultimo mancante

cnt = 0
icAST19 = []
for i in datiAST19["dev standard"]:
    icAST19.append(1.96 * i / sqrt(n[cnt]))
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
ax.fill_between(np.arange(8, 272), icMinAST19, icMaxAST19, alpha=0.2, label="18/19 confidence Interval 95%")

# ora faccio lineplot e fascia per LDH20
sns.lineplot(x="giorno", y="media", color="red", data=datiAST20, marker=".", markersize=9, label="2019/20")

# dato che non mostra gli intervalli di confidenza li calcolo io
# per prima cosa devo calcolare n (di cui poi farò la radice)
n = [474]  # aggiungo manualmente il primo, poi li calcolcolo con un ciclo
cnt = 0
for i in range(7, len(indiciCambioGiornoAST20)):
    n.append(indiciCambioGiornoAST20[i] - indiciCambioGiornoAST20[cnt])
    cnt += 1
n.append(407)  # aggiungo l'ultimo mancante


cnt = 0
icAST20 = []
for i in datiAST20["dev standard"]:
    icAST20.append(1.96 * i / sqrt(n[cnt]))

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
ax.fill_between(np.arange(8, 272), icMinAST20, icMaxAST20, alpha=0.2, color="red",
                label="19/20 confidence interval 95%")
plt.xlabel("time(day)", fontsize=12)
plt.ylabel("previous 7 day mean AST (U/L)", fontsize=12)
plt.xlim(8, 271)
plt.ylim(20, 70)
plt.xticks([8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220,
            230, 240, 250, 260, 270])
ax.tick_params(which='major', length=8, color='black')

# scrivo le indicazioni relative al mese. Uso i minor ticks
ax.set_xticks([27, 58, 88, 119, 150, 179, 210, 240], minor=True)
ax.tick_params(which='minor', length=4, color='black')
ax.text(7.4, 16, "S", size=10)
ax.text(26.5, 16, "O", size=10)
ax.text(57.5, 16, "N", size=10)
ax.text(87.5, 16, "D", size=10)
ax.text(118.5, 16, "J", size=10)
ax.text(149.5, 16, "F", size=10)
ax.text(178.5, 16, "M", size=10)
ax.text(209.5, 16, "A", size=10)
ax.text(239.5, 16, "M", size=10)


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
ax1 = plt.subplot(gs[0])
sns.lineplot(x=np.arange(175, 271), y=numeroPositiviGiorno, ax=ax1)
plt.xlim(8, 270)
plt.ylim(0, max(numeroPositiviGiorno))
plt.xlim(8, 270)
plt.ylabel("No. of COVID \n positive tests", rotation=0, fontsize=12)
ax1.yaxis.set_label_coords(-0.05, 0.45)
ax1.set_xticks([])
ax1.yaxis.tick_right()  # metto i ticks sulla destra
plt.show()

plt.show()


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

# inizio a fare il grafico media dell'età vs settimana con la dev standard (in totale 4 grafici)
fig, ax = plt.subplots(figsize=(15, 4))
sns.lineplot(x="settimana", y="età", estimator=np.mean, data=LDH19, ci="sd")
plt.xlim(1, 39)
plt.ylabel("mean age (yrs)")
plt.xlabel("")
plt.xticks([])
plt.show()

# faccio lo stesso grafico per LDH20
fig, ax = plt.subplots(figsize=(15, 4))
sns.lineplot(x="settimana", y="età", estimator=np.mean, data=LDH20, ci="sd")
plt.xlim(1, 39)
plt.ylabel("mean age (yrs)")
plt.xlabel("")
plt.xticks([])
plt.show()

# lo stesso per AST19
fig, ax = plt.subplots(figsize=(15, 4))
sns.lineplot(x="settimana", y="età", estimator=np.mean, data=AST19, ci="sd")
plt.xlim(1, 39)
plt.ylabel("mean age (yrs)")
plt.xlabel("")
plt.xticks([])
plt.show()

# infine, lo stesso per AST20
fig, ax = plt.subplots(figsize=(15, 4))
sns.lineplot(x="settimana", y="età", estimator=np.mean, data=AST20, ci="sd")
plt.xlim(1, 39)
plt.ylabel("mean age (yrs)")
plt.xlabel("")
plt.xticks([])
plt.show()












