import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def carica_train(dataset):
    file = f"train_{dataset}.txt"
    dati = pd.read_csv(file, sep=r"\s+", header=None)
    dati.columns = [
        "unit", "cycle", "setting1", "setting2", "setting3",
        "sensor1", "sensor2", "sensor3", "sensor4", "sensor5",
        "sensor6", "sensor7", "sensor8", "sensor9", "sensor10",
        "sensor11", "sensor12", "sensor13", "sensor14", "sensor15",
        "sensor16", "sensor17", "sensor18", "sensor19", "sensor20",
        "sensor21"
    ]
    return dati


def carica_test(dataset):
    file = f"test_{dataset}.txt"
    dati = pd.read_csv(file, sep=r"\s+", header=None)
    dati.columns = [
        "unit", "cycle", "setting1", "setting2", "setting3",
        "sensor1", "sensor2", "sensor3", "sensor4", "sensor5",
        "sensor6", "sensor7", "sensor8", "sensor9", "sensor10",
        "sensor11", "sensor12", "sensor13", "sensor14", "sensor15",
        "sensor16", "sensor17", "sensor18", "sensor19", "sensor20",
        "sensor21"
    ]
    return dati


def carica_rul_test(dataset):
    file = f"RUL_{dataset}.txt"
    rul = pd.read_csv(file, sep=r"\s+", header=None)
    return rul.iloc[:, 0].values


def calcola_rul(dati):
    dati["RUL"] = (
        dati.groupby("unit")["cycle"].transform("max")
        - dati["cycle"]
    )
    return dati


def split_motori(dati):
    motori = dati["unit"].unique()

    motori_train, motori_val = train_test_split(
        motori,
        test_size=0.2,
        random_state=42
    )

    dati_train = dati[dati["unit"].isin(motori_train)].copy()
    dati_val = dati[dati["unit"].isin(motori_val)].copy()

    return dati_train, dati_val


def seleziona_feature(dati_train, usa_settings=False):
    sensori = [
        "sensor1", "sensor2", "sensor3", "sensor4", "sensor5",
        "sensor6", "sensor7", "sensor8", "sensor9", "sensor10",
        "sensor11", "sensor12", "sensor13", "sensor14", "sensor15",
        "sensor16", "sensor17", "sensor18", "sensor19", "sensor20",
        "sensor21"
    ]

    settings = ["setting1", "setting2", "setting3"]

    varianze = dati_train[sensori].var()

    sensori_utili = [
        sensore for sensore in sensori
        if varianze[sensore] > 1e-10
    ]

    correlazioni = (
        dati_train[sensori_utili + ["RUL"]]
        .corr()["RUL"]
        .drop("RUL")
    )

    if usa_settings:
        features = sensori_utili + settings
    else:
        features = sensori_utili

    return features, sensori_utili, correlazioni


def crea_modello(nome_modello):
    if nome_modello == "Linear Regression":
        return LinearRegression()

    if nome_modello == "Decision Tree":
        return DecisionTreeRegressor(random_state=42)

    if nome_modello == "Random Forest":
        return RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

    raise ValueError(f"Modello non riconosciuto: {nome_modello}")


def addestra_modelli(dati_train, dati_val, features):
    X_train = dati_train[features]
    y_train = dati_train["RUL"]

    X_val = dati_val[features]
    y_val = dati_val["RUL"]

    nomi_modelli = [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ]

    modelli = {}
    risultati = []

    for nome_modello in nomi_modelli:
        modello = crea_modello(nome_modello)

        modello.fit(X_train, y_train)

        predizioni = modello.predict(X_val)

        mae = mean_absolute_error(y_val, predizioni)
        rmse = np.sqrt(mean_squared_error(y_val, predizioni))

        modelli[nome_modello] = modello

        risultati.append({
            "modello": nome_modello,
            "MAE": mae,
            "RMSE": rmse
        })

    return pd.DataFrame(risultati), modelli


def valuta_test(dataset, modello, features, dati_train):
    dati_test = carica_test(dataset)
    rul_reali = carica_rul_test(dataset)

    dati_test_finali = (
        dati_test
        .groupby("unit")
        .tail(1)
        .sort_values("unit")
    )

    predizioni = modello.predict(dati_test_finali[features])

    rul_media = dati_train["RUL"].mean()

    predizioni_baseline = np.full(
        len(rul_reali),
        rul_media
    )

    mae = mean_absolute_error(rul_reali, predizioni)
    rmse = np.sqrt(mean_squared_error(rul_reali, predizioni))

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


def crea_grafici(risultati_test, dataset, nome_modello, cartella):
    nome_file = nome_modello.lower().replace(" ", "_")

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
        f"{dataset} - {nome_modello}\nRUL reale vs predetta"
    )
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            cartella,
            f"{nome_file}_rul_reale_vs_predetta.png"
        )
    )

    plt.close()

    plt.figure(figsize=(10, 6))

    plt.bar(
        risultati_test["unit"],
        risultati_test["errore"]
    )

    plt.axhline(0, linestyle="--")

    plt.xlabel("Motore")
    plt.ylabel("Errore RUL")
    plt.title(
        f"{dataset} - {nome_modello}\nErrore per motore"
    )
    plt.grid(axis="y")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            cartella,
            f"{nome_file}_errore_per_motore.png"
        )
    )

    plt.close()


def esegui_dataset(dataset, usa_settings, risultati_summary):
    if usa_settings:
        tipo_features = "sensors_plus_settings"
        descrizione_features = "Sensori + Settings"
    else:
        tipo_features = "sensors_only"
        descrizione_features = "Solo sensori"

    print("\n====================================================")
    print(f"DATASET: {dataset}")
    print(f"FEATURES: {descrizione_features}")
    print("====================================================")

    dati = carica_train(dataset)
    dati = calcola_rul(dati)

    dati_train, dati_val = split_motori(dati)

    features, sensori_utili, correlazioni = seleziona_feature(
        dati_train,
        usa_settings=usa_settings
    )

    print("\nFeature utilizzate:")
    print(features)

    risultati_validation, _ = addestra_modelli(
        dati_train,
        dati_val,
        features
    )

    print("\nRisultati VALIDATION:")
    print(risultati_validation.to_string(index=False))

    cartella = os.path.join(
        "results_generale",
        dataset,
        tipo_features
    )

    os.makedirs(cartella, exist_ok=True)

    risultati_validation.to_csv(
        os.path.join(
            cartella,
            "risultati_validation.csv"
        ),
        index=False
    )

    correlazioni.to_csv(
        os.path.join(
            cartella,
            "correlazioni_sensori.csv"
        ),
        header=["correlazione"]
    )

    with open(
        os.path.join(cartella, "feature_utilizzate.txt"),
        "w",
        encoding="utf-8"
    ) as file:

        file.write(f"Dataset: {dataset}\n")
        file.write(
            f"Tipo features: {descrizione_features}\n\n"
        )
        file.write("Features utilizzate:\n")

        for feature in features:
            file.write(f"- {feature}\n")

    risultati_test_tutti = []

    print("\n----------------------------------------------------")
    print("TEST UFFICIALE - TUTTI I MODELLI")
    print("----------------------------------------------------")

    for nome_modello in [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ]:

        print(f"\nTest modello: {nome_modello}")

        # Ogni modello viene riaddestrato su TUTTO il training.
        modello_finale = crea_modello(nome_modello)

        modello_finale.fit(
            dati[features],
            dati["RUL"]
        )

        (
            risultati_test,
            mae_test,
            rmse_test,
            mae_baseline,
            rmse_baseline
        ) = valuta_test(
            dataset,
            modello_finale,
            features,
            dati
        )

        print(f"MAE  = {mae_test:.4f}")
        print(f"RMSE = {rmse_test:.4f}")

        nome_file = nome_modello.lower().replace(" ", "_")

        risultati_test.to_csv(
            os.path.join(
                cartella,
                f"risultati_test_{nome_file}.csv"
            ),
            index=False
        )

        crea_grafici(
            risultati_test,
            dataset,
            nome_modello,
            cartella
        )

        riga_validation = risultati_validation[
            risultati_validation["modello"] == nome_modello
        ].iloc[0]

        riga_summary = {
            "dataset": dataset,
            "features": tipo_features,
            "modello": nome_modello,
            "validation_MAE": riga_validation["MAE"],
            "validation_RMSE": riga_validation["RMSE"],
            "test_MAE": mae_test,
            "test_RMSE": rmse_test,
            "baseline_test_MAE": mae_baseline,
            "baseline_test_RMSE": rmse_baseline
        }

        risultati_test_tutti.append(riga_summary)
        risultati_summary.append(riga_summary)

    pd.DataFrame(risultati_test_tutti).to_csv(
        os.path.join(
            cartella,
            "risultati_test_modelli.csv"
        ),
        index=False
    )


dataset = ["FD001", "FD002", "FD003", "FD004"]

risultati_summary = []

for nome_dataset in dataset:
    esegui_dataset(
        nome_dataset,
        usa_settings=False,
        risultati_summary=risultati_summary
    )

    esegui_dataset(
        nome_dataset,
        usa_settings=True,
        risultati_summary=risultati_summary
    )

summary = pd.DataFrame(risultati_summary)

summary.to_csv(
    "results_summary.csv",
    index=False
)

print("\n\n====================================================")
print("SUMMARY FINALE")
print("====================================================")
print(summary.to_string(index=False))

print("\nRisultati salvati in:")
print("- results_summary.csv")
print("- results_generale/")
