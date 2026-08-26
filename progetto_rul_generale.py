import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
#funzione caricamento dataset

def carica_train(dataset):

    file = f"train_{dataset}.txt"

    dati = pd.read_csv(
        file,
        sep=r"\s+",
        header=None
    )

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

    return dati

#caricamento train

def carica_test(dataset):

    file = f"test_{dataset}.txt"

    dati = pd.read_csv(
        file,
        sep=r"\s+",
        header=None
    )

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

    return dati


#funzione per caricamento rul test

def carica_rul_test(dataset):

    file = f"RUL_{dataset}.txt"

    rul = pd.read_csv(
        file,
        sep=r"\s+",
        header=None
    )

    return rul.iloc[:, 0].values


#funzione calcolo RUL

def calcola_rul(dati):

    dati["RUL"] = (
        dati.groupby("unit")["cycle"].transform("max")
        - dati["cycle"]
    )

    return dati

#funzione split dei motori

def split_motori(dati):

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

    return dati_train, dati_val

#funzione selezione dei sensori

def seleziona_sensori(dati_train):

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

    varianze = dati_train[sensori].var()

    sensori_utili = [
        sensore
        for sensore in sensori
        if varianze[sensore] > 1e-10
    ]

    correlazioni = (
        dati_train[sensori_utili + ["RUL"]]
        .corr()["RUL"]
        .drop("RUL")
    )

    return sensori_utili, correlazioni

#funzione addestramento modelli

def addestra_modelli(
    dati_train,
    dati_val,
    sensori_utili
):

    X_train = dati_train[sensori_utili]
    y_train = dati_train["RUL"]

    X_val = dati_val[sensori_utili]
    y_val = dati_val["RUL"]

    # Linear Regression

    modello_lr = LinearRegression()

    modello_lr.fit(
        X_train,
        y_train
    )

    pred_lr = modello_lr.predict(X_val)

    mae_lr = mean_absolute_error(
        y_val,
        pred_lr
    )

    rmse_lr = np.sqrt(
        mean_squared_error(
            y_val,
            pred_lr
        )
    )

    # Decision Tree

    modello_tree = DecisionTreeRegressor(
        random_state=42
    )

    modello_tree.fit(
        X_train,
        y_train
    )

    pred_tree = modello_tree.predict(X_val)

    mae_tree = mean_absolute_error(
        y_val,
        pred_tree
    )

    rmse_tree = np.sqrt(
        mean_squared_error(
            y_val,
            pred_tree
        )
    )

    # Random Forest

    modello_rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    modello_rf.fit(
        X_train,
        y_train
    )

    pred_rf = modello_rf.predict(X_val)

    mae_rf = mean_absolute_error(
        y_val,
        pred_rf
    )

    rmse_rf = np.sqrt(
        mean_squared_error(
            y_val,
            pred_rf
        )
    )

    risultati = pd.DataFrame({

        "modello": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest"
        ],

        "MAE": [
            mae_lr,
            mae_tree,
            mae_rf
        ],

        "RMSE": [
            rmse_lr,
            rmse_tree,
            rmse_rf
        ]
    })

    modelli = {
        "Linear Regression": modello_lr,
        "Decision Tree": modello_tree,
        "Random Forest": modello_rf
    }

    return risultati, modelli


#funzione valutazione test

def valuta_test(
    dataset,
    modello,
    sensori_utili,
    dati_train
):

    #caricamento test

    dati_test = carica_test(dataset)

    #caricamento RUL reali

    rul_reali = carica_rul_test(dataset)

    #prendiamo l'ultima osservazione di ogni motore

    dati_test_finali = (
        dati_test
        .groupby("unit")
        .tail(1)
        .sort_values("unit")
    )

    #predizione

    predizioni = modello.predict(
        dati_test_finali[sensori_utili]
    )

    #baseline: RUL media del training

    rul_media = dati_train["RUL"].mean()

    predizioni_baseline = np.full(
        len(rul_reali),
        rul_media
    )

    #metriche modello

    mae = mean_absolute_error(
        rul_reali,
        predizioni
    )

    rmse = np.sqrt(
        mean_squared_error(
            rul_reali,
            predizioni
        )
    )

    #metriche baseline

    mae_baseline = mean_absolute_error(
        rul_reali,
        predizioni_baseline
    )

    rmse_baseline = np.sqrt(
        mean_squared_error(
            rul_reali,
            predizioni_baseline
        )
    )

    #tabella risultati

    risultati_test = pd.DataFrame({

        "unit": dati_test_finali["unit"].values,

        "RUL_reale": rul_reali,

        "RUL_predetta": predizioni,

        "errore": predizioni - rul_reali,

        "RUL_baseline": predizioni_baseline

    })

    return (
        risultati_test,
        mae,
        rmse,
        mae_baseline,
        rmse_baseline
    )


#funzione plot grafici real/prediction e errore per motore
def crea_grafici(
    risultati_test,
    dataset
):

    # Cartella risultati

    cartella = os.path.join(
        "results_generale",
        dataset
    )

    os.makedirs(
        cartella,
        exist_ok=True
    )

    # =========================
    # GRAFICO RUL REALE VS PRED.
    # =========================

    plt.figure(figsize=(8, 6))

    plt.scatter(
        risultati_test["RUL_reale"],
        risultati_test["RUL_predetta"]
    )

    minimo = min(
        risultati_test["RUL_reale"].min(),
        risultati_test["RUL_predetta"].min()
    )

    massimo = max(
        risultati_test["RUL_reale"].max(),
        risultati_test["RUL_predetta"].max()
    )

    plt.plot(
        [minimo, massimo],
        [minimo, massimo],
        linestyle="--"
    )

    plt.xlabel("RUL reale")

    plt.ylabel("RUL predetta")

    plt.title(
        f"{dataset} - RUL reale vs predetta"
    )

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            cartella,
            "rul_reale_vs_predetta.png"
        )
    )

    plt.close()

    # =========================
    # GRAFICO ERRORE PER MOTORE
    # =========================

    plt.figure(figsize=(10, 6))

    plt.bar(
        risultati_test["unit"],
        risultati_test["errore"]
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel("Motore")

    plt.ylabel("Errore RUL")

    plt.title(
        f"{dataset} - Errore per motore"
    )

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            cartella,
            "errore_per_motore.png"
        )
    )

    plt.close()


#funzione per eseguire dataset



def esegui_dataset(dataset):

    print("\n================================")
    print(f"DATASET: {dataset}")
    print("================================")

  
    # CARICAMENTO TRAIN
   

    dati = carica_train(dataset)

  
    # CALCOLO RUL
   

    dati = calcola_rul(dati)

   
    # SPLIT 80/20 DEI MOTORI
  

    dati_train, dati_val = split_motori(dati)

    
    # SELEZIONE SENSORI
    # SOLO SUL TRAINING
   

    sensori_utili, correlazioni = seleziona_sensori(
        dati_train
    )

    print("\nSensori utilizzati:")
    print(sensori_utili)

    
    # TRAINING DEI MODELLI
   

    risultati, modelli = addestra_modelli(
        dati_train,
        dati_val,
        sensori_utili
    )

    print("\nRisultati validation:")
    print(risultati)

    
    # SCELTA MODELLO MIGLIORE
   

    nome_modello, modello_validation = (
        scegli_miglior_modello(
            risultati,
            modelli
        )
    )

    print(
        f"\nModello migliore sulla validation: "
        f"{nome_modello}"
    )

   
    # RETRAINING SU TUTTO IL TRAIN
   

    X_train_completo = dati[sensori_utili]

    y_train_completo = dati["RUL"]

    if nome_modello == "Linear Regression":

        modello_finale = LinearRegression()

    elif nome_modello == "Decision Tree":

        modello_finale = DecisionTreeRegressor(
            random_state=42
        )

    elif nome_modello == "Random Forest":

        modello_finale = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

    modello_finale.fit(
        X_train_completo,
        y_train_completo
    )

    print(
        "\nModello riaddestrato su tutti i motori del training."
    )

  
    # TEST
   

    (
        risultati_test,
        mae_test,
        rmse_test,
        mae_baseline,
        rmse_baseline

    ) = valuta_test(
        dataset,
        modello_finale,
        sensori_utili,
        dati
    )

    
    # RISULTATI TEST
  

    print("\nRisultati TEST:")

    print(
        f"{nome_modello}:"
    )

    print(
        f"MAE  = {mae_test:.4f}"
    )

    print(
        f"RMSE = {rmse_test:.4f}"
    )

    print("\nBaseline RUL media:")

    print(
        f"MAE  = {mae_baseline:.4f}"
    )

    print(
        f"RMSE = {rmse_baseline:.4f}"
    )

    
    # CARTELLA RISULTATI
  

    cartella = os.path.join(
        "results_generale",
        dataset
    )

    os.makedirs(
        cartella,
        exist_ok=True
    )

    
    # SALVATAGGIO CSV
    

    risultati_test.to_csv(
        os.path.join(
            cartella,
            "risultati_test.csv"
        ),
        index=False
    )

   
    # CREAZIONE GRAFICI
   

    crea_grafici(
        risultati_test,
        dataset
    )

    return (
        risultati,
        nome_modello,
        sensori_utili,
        risultati_test,
        mae_test,
        rmse_test,
        mae_baseline,
        rmse_baseline
    )

#scelgo il miglior modello basato sul MAE

def scegli_miglior_modello(risultati, modelli):

    indice_migliore = risultati["MAE"].idxmin()

    nome_modello = risultati.loc[
        indice_migliore,
        "modello"
    ]

    modello_migliore = modelli[nome_modello]

    return nome_modello, modello_migliore


# ESECUZIONE DATASET

dataset = [
    "FD001",
    "FD002",
    "FD003",
    "FD004"
]

risultati_finali = {}

for nome_dataset in dataset:

    risultati_finali[nome_dataset] = (
        esegui_dataset(nome_dataset)
    )