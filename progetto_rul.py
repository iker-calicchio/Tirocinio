import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error



#CARICAMENTO DEL DATASET


dati = pd.read_csv(
    "train_FD001.txt",
    sep=r"\s+",
    header=None
)

print("Dimensione dataset:")
print(dati.shape)



#NOMI DELLE COLONNE


nomi_colonne = [
    "unit",
    "cycle",
    "setting1",
    "setting2",
    "setting3",
    "sensor1",
    "sensor2",
    "sensor3",
    "sensor4",
    "sensor5",
    "sensor6",
    "sensor7",
    "sensor8",
    "sensor9",
    "sensor10",
    "sensor11",
    "sensor12",
    "sensor13",
    "sensor14",
    "sensor15",
    "sensor16",
    "sensor17",
    "sensor18",
    "sensor19",
    "sensor20",
    "sensor21"
]

dati.columns = nomi_colonne



#PRIME RIGHE DEL DATASET


print("\nPrime 5 righe:")
print(dati.head())



#INFORMAZIONI GENERALI


print("\nInformazioni sul dataset:")
print(dati.info())



#NUMERO DI MOTORI


numero_motori = dati["unit"].nunique()

print("\nNumero di motori:")
print(numero_motori)



#ELENCO DEI MOTORI


print("\nMotori presenti:")
print(dati["unit"].unique())



#CICLO MASSIMO DI OGNI MOTORE


vita_motori = dati.groupby("unit")["cycle"].max()

print("\nNumero di cicli di ogni motore:")
print(vita_motori)



#CREAZIONE DELLA RUL


dati["RUL"] = (
    dati.groupby("unit")["cycle"].transform("max")
    - dati["cycle"]
)

print("\nPrime righe con la RUL:")
print(dati[["unit", "cycle", "RUL"]].head(10))



#CONTROLLO DELLA RUL DEL MOTORE 1


motore1 = dati[dati["unit"] == 1]

print("\nUltimi cicli del motore 1:")
print(
    motore1[["unit", "cycle", "RUL"]].tail()
)



#ELENCO DEI SENSORI


sensori = [
    "sensor1",
    "sensor2",
    "sensor3",
    "sensor4",
    "sensor5",
    "sensor6",
    "sensor7",
    "sensor8",
    "sensor9",
    "sensor10",
    "sensor11",
    "sensor12",
    "sensor13",
    "sensor14",
    "sensor15",
    "sensor16",
    "sensor17",
    "sensor18",
    "sensor19",
    "sensor20",
    "sensor21"
]



#STATISTICHE DEI SENSORI


print("\nStatistiche dei sensori:")
print(dati[sensori].describe())



#VARIANZIONE STANDARD DEI SENSORI


varianze = dati[sensori].var()

print("\nVarianza dei sensori:")
print(varianze.sort_values())

#SPLIT DEI MOTORI IN TRAINING E VALIDATION

from sklearn.model_selection import train_test_split

motori = dati["unit"].unique()

motori_train, motori_val = train_test_split(
    motori,
    test_size=0.2,
    random_state=42
)

dati_train = dati[
    dati["unit"].isin(motori_train)
].copy()

dati_val = dati[
    dati["unit"].isin(motori_val)
].copy()

print("\n===== SPLIT TRAINING / VALIDATION =====")

print("Motori training:", dati_train["unit"].nunique())
print("Motori validation:", dati_val["unit"].nunique())



#ANALISI DELLE CORRELAZIONI SOLO SUL TRAINING


correlazioni = (
    dati_train[sensori + ["RUL"]]
    .corr()["RUL"]
)

print("\nCorrelazione dei sensori con la RUL - SOLO TRAINING:")
print(correlazioni.sort_values())



#HEATMAP DELLE CORRELAZIONI DEL TRAINING

plt.figure(figsize=(12, 8))

sns.heatmap(
    dati_train[sensori + ["RUL"]].corr(),
    cmap="coolwarm"
)

plt.title("Matrice di correlazione - Training")

plt.show()


#IDENTIFICAZIONE DEI SENSORI COSTANTI

varianze_train = dati_train[sensori].var()

print("\nVarianza dei sensori nel training:")
print(varianze_train.sort_values())


sensori_utili = [
    sensore
    for sensore in sensori
    if varianze_train[sensore] > 1e-10
]

print("\nSensori utilizzati dal modello:")
print(sensori_utili)

#SPLIT MOTORI 80 PER TRAIN E 20 PER VALIDATION

from sklearn.model_selection import train_test_split

motori = dati["unit"].unique()

motori_train, motori_val = train_test_split(
    motori,
    test_size=0.2,
    random_state=42
)

dati_train = dati[
    dati["unit"].isin(motori_train)
]

dati_val = dati[
    dati["unit"].isin(motori_val)
]

print("\nMotori training:", dati_train["unit"].nunique())
print("Motori validation:", dati_val["unit"].nunique())


#CREAZIONE X E Y


X_train = dati_train[sensori_utili]
y_train = dati_train["RUL"]

X_val = dati_val[sensori_utili]
y_val = dati_val["RUL"]

print("\nDimensione X_train:", X_train.shape)
print("Dimensione y_train:", y_train.shape)

print("Dimensione X_val:", X_val.shape)
print("Dimensione y_val:", y_val.shape)


#BASELINE: PREVISIONE DELLA RUL MEDIA


rul_media = y_train.mean()

y_pred_baseline = np.full(
    len(y_val),
    rul_media
)

mae_baseline = mean_absolute_error(
    y_val,
    y_pred_baseline
)

rmse_baseline = np.sqrt(
    mean_squared_error(
        y_val,
        y_pred_baseline
    )
)

print("\n===== RISULTATI BASELINE =====")

print("RUL media del training:", rul_media)
print("MAE:", mae_baseline)
print("RMSE:", rmse_baseline)


#IMPLEMENTAZIONE REGRESSIONE LINEARE

from sklearn.linear_model import LinearRegression

modello_lineare = LinearRegression()

modello_lineare.fit(
    X_train,
    y_train
)


y_pred = modello_lineare.predict(X_val)



#VALUTAZIONE ERRORE

mae = mean_absolute_error(
    y_val,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_val,
        y_pred
    )
)

print("\n===== RISULTATI LINEAR REGRESSION =====")

print("MAE:", mae)
print("RMSE:", rmse)



#GRAFICO RUL REALE VS RUL PREDDETTA


plt.figure(figsize=(8, 6))

plt.scatter(
    y_val,
    y_pred,
    s=10
)

plt.plot(
    [0, y_val.max()],
    [0, y_val.max()]
)

plt.xlabel("RUL reale")
plt.ylabel("RUL predetta")
plt.title("Linear Regression - RUL reale vs predetta")

plt.grid()

plt.show()

#CONSIDERAZIONI: IN VALIDATION QUESTO MODELLO SBAGLIA MEDIAMENTE DI 30 CICLI, IL VALORE RMSE DI 38 INDICA LA PRESENZA DI ALCUNE PREDICTION CON UN ERRORE PIUTTOSTO ALTO.
#ORA SI PROCEDE CON L'IMPLEMENTAZIONE DEL MODELLO RANDOM FOREST (200 alberi decisionali e rul che sarà un media delle predizioni) IN MODO DA POTER CONFRONTARE I RISULTATI.

from sklearn.ensemble import RandomForestRegressor

modello_rf = RandomForestRegressor(
    n_estimators=200,  #200 alberi e 42 random states sono numeri arbitrari
    random_state=42,
    n_jobs=-1
)

#ADDESTRAMENTO

modello_rf.fit(
    X_train,
    y_train
)

y_pred_rf = modello_rf.predict(X_val)

#STAMPA MAE E RMSE
mae_rf = mean_absolute_error(
    y_val,
    y_pred_rf
)

rmse_rf = np.sqrt(
    mean_squared_error(
        y_val,
        y_pred_rf
    )
)

print("\n===== RISULTATI RANDOM FOREST =====")

print("MAE:", mae_rf)
print("RMSE:", rmse_rf)

#VISUALIZZAZIONE PREDIZIONI RANDOM FOREST

plt.figure(figsize=(8, 6))

plt.scatter(
    y_val,
    y_pred_rf,
    s=10
)

plt.plot(
    [0, y_val.max()],
    [0, y_val.max()]
)

plt.xlabel("RUL reale")
plt.ylabel("RUL predetta")
plt.title("Random Forest - RUL reale vs predetta")

plt.grid()

plt.show()

#POSSIAMO SFRUTTARE UN PROPRIETA' DI RANDOM FOREST PER VISUALIZZARE L'IMPORTANZA DEI SENSORI NELLE DECISIONI DEL MODELLO

importanze = pd.DataFrame({
    "sensore": sensori_utili,
    "importanza": modello_rf.feature_importances_
})

importanze = importanze.sort_values(
    "importanza",
    ascending=False
)

print("\nImportanza dei sensori:")
print(importanze)


#ORA IMPLEMENTO UN DECISION TREE REGRESSOR

from sklearn.tree import DecisionTreeRegressor

modello_tree = DecisionTreeRegressor(
    random_state=42
)

modello_tree.fit(
    X_train,
    y_train
)

#PREDIZIONI


y_pred_tree = modello_tree.predict(
    X_val
)

#VALUTAZIONE

mae_tree = mean_absolute_error(
    y_val,
    y_pred_tree
)

rmse_tree = np.sqrt(
    mean_squared_error(
        y_val,
        y_pred_tree
    )
)

print("\n===== RISULTATI DECISION TREE =====")

print("MAE:", mae_tree)
print("RMSE:", rmse_tree)

#CONFRONTO DEI TRE MODELLI

print("\n===== CONFRONTO MODELLI =====")

print(
    f"Baseline           -> "
    f"MAE: {mae_baseline:.2f} | RMSE: {rmse_baseline:.2f}"
)

print(
    f"Linear Regression -> "
    f"MAE: {mae:.2f} | RMSE: {rmse:.2f}"
)

print(
    f"Decision Tree     -> "
    f"MAE: {mae_tree:.2f} | RMSE: {rmse_tree:.2f}"
)

print(
    f"Random Forest     -> "
    f"MAE: {mae_rf:.2f} | RMSE: {rmse_rf:.2f}"
)



#CARICAMENTO TEST FD001


test = pd.read_csv(
    "test_FD001.txt",
    sep=r"\s+",
    header=None
)

test.columns = nomi_colonne

print("\n===== TEST FD001 =====")

print("Dimensione test:")
print(test.shape)

print("Numero di motori nel test:")
print(test["unit"].nunique())


#CARICAMENTO RUL REALI DEL TEST


rul_reali = pd.read_csv(
    "RUL_FD001.txt",
    sep=r"\s+",
    header=None
)

rul_reali = rul_reali.iloc[:, 0]

print("\nPrime RUL reali:")
print(rul_reali.head())

print("\nNumero di RUL reali:")
print(len(rul_reali))

#ADDESTRAMENTO DEL MODELLO FINALE SU TUTTI I 100 MOTORI

X_full = dati[sensori_utili]
y_full = dati["RUL"]

modello_finale = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

modello_finale.fit(
    X_full,
    y_full
)

print("\n===== MODELLO FINALE =====")

print("Motori utilizzati per l'addestramento:",
      dati["unit"].nunique())

print("Numero di campioni di training:",
      len(X_full))

#ULTIMA OSSERVAZIONE DI OGNI MOTORE DEL TEST

test_ultima_riga = (
    test
    .groupby("unit")
    .tail(1)
    .copy()
)

print("\n===== ULTIMA OSSERVAZIONE DEI MOTORI TEST =====")

print("Numero di motori:",
      test_ultima_riga["unit"].nunique())

print("\nPrime righe:")
print(
    test_ultima_riga[
        ["unit", "cycle"]
    ].head(10)
)

#PREDIZIONE DELLA RUL SUL TEST

test_ultima_riga = (                            #ORDINO PER UNIT
    test
    .groupby("unit")
    .tail(1)
    .sort_values("unit")
    .copy()
)

X_test = test_ultima_riga[sensori_utili]

y_pred_test = modello_finale.predict(
    X_test
)

print("\n===== PRIME PREDIZIONI TEST =====")

for i in range(10):

    print(
        f"Motore {i+1}: "
        f"RUL predetta = {y_pred_test[i]:.2f}"
    )


#VALUTAZIONE SUL TEST FD001

mae_test = mean_absolute_error(
    rul_reali,
    y_pred_test
)

rmse_test = np.sqrt(
    mean_squared_error(
        rul_reali,
        y_pred_test
    )
)

print("\n===== RISULTATI TEST FD001 =====")

print("MAE:", mae_test)
print("RMSE:", rmse_test)


#CONFRONTO RUL REALE E RUL PREDDETTA

risultati_test = pd.DataFrame({
    "Motore": test_ultima_riga["unit"].values,
    "RUL_reale": rul_reali.values,
    "RUL_predetta": y_pred_test
})

risultati_test["Errore"] = (
    risultati_test["RUL_predetta"]
    - risultati_test["RUL_reale"]
)

print("\n===== CONFRONTO PRIME 10 PREDIZIONI =====")

print(
    risultati_test.head(10)
)


#GRAFICO RUL REALE VS RUL PREDETTA  TEST FD001


plt.figure(figsize=(8, 6))

plt.scatter(
    risultati_test["RUL_reale"],
    risultati_test["RUL_predetta"],
    s=30
)

limite = max(
    risultati_test["RUL_reale"].max(),
    risultati_test["RUL_predetta"].max()
)

plt.plot(
    [0, limite],
    [0, limite],
    linestyle="--"
)

plt.xlabel("RUL reale")
plt.ylabel("RUL predetta")
plt.title("Random Forest - Test FD001")

plt.grid()

plt.savefig(
    "results/FD001/rul_reale_vs_predetta.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


#GRAFICO ERRORE PER MOTORE  TEST FD001

plt.figure(figsize=(12, 5))

plt.bar(
    risultati_test["Motore"],
    risultati_test["Errore"]
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Motore")
plt.ylabel("Errore (RUL predetta - RUL reale)")
plt.title("Errore di predizione per motore - Test FD001")

plt.grid(axis="y")

plt.savefig(
    "results/FD001/errore_per_motore.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()