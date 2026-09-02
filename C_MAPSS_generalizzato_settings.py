import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# CONFIGURAZIONE

DATASET = ["FD001", "FD002", "FD003", "FD004"]

N_SPLITS = 5

MODELLI = [
    "Linear Regression",
    "Decision Tree",
    "Random Forest"
]

TIPI_FEATURES = {
    "sensors_only": False,
    "sensors_plus_settings": True
}

CARTELLA_RESULTS = "results_generale"

# NOMI COLONNE

COLONNE = [
    "unit", "cycle",
    "setting1", "setting2", "setting3",
    "sensor1", "sensor2", "sensor3", "sensor4", "sensor5",
    "sensor6", "sensor7", "sensor8", "sensor9", "sensor10",
    "sensor11", "sensor12", "sensor13", "sensor14", "sensor15",
    "sensor16", "sensor17", "sensor18", "sensor19", "sensor20",
    "sensor21"
]

SENSORI = [
    "sensor1", "sensor2", "sensor3", "sensor4", "sensor5",
    "sensor6", "sensor7", "sensor8", "sensor9", "sensor10",
    "sensor11", "sensor12", "sensor13", "sensor14", "sensor15",
    "sensor16", "sensor17", "sensor18", "sensor19", "sensor20",
    "sensor21"
]

SETTINGS = [
    "setting1",
    "setting2",
    "setting3"
]

# CARICAMENTO DATASET

def carica_train(dataset):

    file = f"train_{dataset}.txt"

    dati = pd.read_csv(
        file,
        sep=r"\s+",
        header=None
    )

    dati.columns = COLONNE

    return dati


def carica_test(dataset):

    file = f"test_{dataset}.txt"

    dati = pd.read_csv(
        file,
        sep=r"\s+",
        header=None
    )

    dati.columns = COLONNE

    return dati


def carica_rul_test(dataset):

    file = f"RUL_{dataset}.txt"

    rul = pd.read_csv(
        file,
        sep=r"\s+",
        header=None
    )

    return rul.iloc[:, 0].values

# CALCOLO RUL

def calcola_rul(dati):

    dati = dati.copy()

    dati["RUL"] = (
        dati.groupby("unit")["cycle"].transform("max")
        - dati["cycle"]
    )

    return dati

# MODELLO

def crea_modello(nome_modello):

    if nome_modello == "Linear Regression":

        return LinearRegression()

    if nome_modello == "Decision Tree":

        return DecisionTreeRegressor(
            random_state=42
        )

    if nome_modello == "Random Forest":

        return RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

    raise ValueError(
        f"Modello non riconosciuto: {nome_modello}"
    )


# SELEZIONE FEATURE

def seleziona_feature(
    dati_train,
    usa_settings=False
):
    # 1. Eliminiamo sensori con varianza praticamente nulla

    varianze_sensori = dati_train[SENSORI].var()

    sensori_utili = [
        sensore
        for sensore in SENSORI
        if varianze_sensori[sensore] > 1e-10
    ]

    # 2. Correlazione sensori - RUL
    #
    # Questa viene calcolata SOLO sul training del fold.
    # Non viene utilizzata per eliminare arbitrariamente
    # sensori: viene salvata come informazione.

    correlazioni = (
        dati_train[sensori_utili + ["RUL"]]
        .corr()["RUL"]
        .drop("RUL")
        .sort_values()
    )

    # 3. Aggiunta eventuale operating settings

    if usa_settings:

        varianze_settings = dati_train[SETTINGS].var()

        settings_utili = [
            setting
            for setting in SETTINGS
            if varianze_settings[setting] > 1e-10
        ]

        features = sensori_utili + settings_utili

    else:

        settings_utili = []

        features = sensori_utili

    return (
        features,
        sensori_utili,
        settings_utili,
        correlazioni
    )

# BASELINE

def calcola_baseline(y_train, y_val):

    valore_baseline = y_train.mean()

    predizioni = np.full(
        len(y_val),
        valore_baseline
    )

    mae = mean_absolute_error(
        y_val,
        predizioni
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val,
            predizioni
        )
    )

    return mae, rmse, valore_baseline

# CROSS VALIDATION

def cross_validation(
    dati,
    usa_settings,
    dataset
):

    print("\n")
    print("----------------------------------------------------")
    print(f"CROSS-VALIDATION - {dataset}")
    print(
        "FEATURES:",
        "Sensori + Settings"
        if usa_settings
        else "Solo sensori"
    )
    print("----------------------------------------------------")

    groups = dati["unit"]

    gkf = GroupKFold(
        n_splits=N_SPLITS
    )

    risultati_cv = []

    for fold, (indice_train, indice_val) in enumerate(
        gkf.split(
            dati,
            dati["RUL"],
            groups=groups
        ),
        start=1
    ):

        dati_train = dati.iloc[indice_train].copy()
        dati_val = dati.iloc[indice_val].copy()

        print(f"\nFold {fold}")
        print(
            f"Motori training: "
            f"{dati_train['unit'].nunique()}"
        )
        print(
            f"Motori validation: "
            f"{dati_val['unit'].nunique()}"
        )

        # FEATURE SELECTION
        # Fatta esclusivamente sul training del fold

        (
            features,
            sensori_utili,
            settings_utili,
            correlazioni
        ) = seleziona_feature(
            dati_train,
            usa_settings=usa_settings
        )

        X_train = dati_train[features]
        y_train = dati_train["RUL"]

        X_val = dati_val[features]
        y_val = dati_val["RUL"]

        # BASELINE

        (
            mae_baseline,
            rmse_baseline,
            valore_baseline
        ) = calcola_baseline(
            y_train,
            y_val
        )

        risultati_cv.append({
            "dataset": dataset,
            "features": (
                "sensors_plus_settings"
                if usa_settings
                else "sensors_only"
            ),
            "modello": "Baseline",
            "fold": fold,
            "MAE": mae_baseline,
            "RMSE": rmse_baseline,
            "n_features": len(features)
        })

        # MODELLI

        for nome_modello in MODELLI:

            modello = crea_modello(
                nome_modello
            )

            modello.fit(
                X_train,
                y_train
            )

            predizioni = modello.predict(
                X_val
            )

            mae = mean_absolute_error(
                y_val,
                predizioni
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_val,
                    predizioni
                )
            )

            risultati_cv.append({
                "dataset": dataset,
                "features": (
                    "sensors_plus_settings"
                    if usa_settings
                    else "sensors_only"
                ),
                "modello": nome_modello,
                "fold": fold,
                "MAE": mae,
                "RMSE": rmse,
                "n_features": len(features)
            })

    risultati_cv = pd.DataFrame(
        risultati_cv
    )

    # MEDIA E DEVIAZIONE STANDARD

    risultati_cv_summary = (
        risultati_cv
        .groupby(
            [
                "dataset",
                "features",
                "modello"
            ]
        )
        .agg(
            CV_MAE_mean=("MAE", "mean"),
            CV_MAE_std=("MAE", "std"),
            CV_RMSE_mean=("RMSE", "mean"),
            CV_RMSE_std=("RMSE", "std"),
            n_features=("n_features", "mean")
        )
        .reset_index()
    )

    return (
        risultati_cv,
        risultati_cv_summary
    )

# TRAINING FINALE + TEST UFFICIALE

def valuta_test(
    dataset,
    modello,
    features,
    dati_train_completo
):

    dati_test = carica_test(
        dataset
    )

    rul_reali = carica_rul_test(
        dataset
    )

    # Prendiamo l'ultima osservazione disponibile di ogni
    # motore del test set

    dati_test_finali = (
        dati_test
        .groupby("unit")
        .tail(1)
        .sort_values("unit")
        .reset_index(drop=True)
    )

    # Controllo importante
    if len(dati_test_finali) != len(rul_reali):

        raise ValueError(
            f"{dataset}: numero motori test "
            f"({len(dati_test_finali)}) "
            f"diverso dal numero di RUL reali "
            f"({len(rul_reali)})"
        )

    # Predizione

    predizioni = modello.predict(
        dati_test_finali[features]
    )

    # Baseline

    rul_media = dati_train_completo["RUL"].mean()

    predizioni_baseline = np.full(
        len(rul_reali),
        rul_media
    )

    # Metriche modello

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

    # Metriche baseline

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

    # Risultati per singolo motore

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

# GRAFICI

def crea_grafici(
    risultati_test,
    dataset,
    nome_modello,
    cartella
):

    nome_file = (
        nome_modello
        .lower()
        .replace(" ", "_")
    )

    # RUL reale vs predetta

    plt.figure(
        figsize=(8, 6)
    )

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

    plt.xlabel(
        "RUL reale"
    )

    plt.ylabel(
        "RUL predetta"
    )

    plt.title(
        f"{dataset} - {nome_modello}\n"
        "RUL reale vs predetta"
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

    # Errore per motore

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        risultati_test["unit"],
        risultati_test["errore"]
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "Motore"
    )

    plt.ylabel(
        "Errore RUL"
    )

    plt.title(
        f"{dataset} - {nome_modello}\n"
        "Errore per motore"
    )

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            cartella,
            f"{nome_file}_errore_per_motore.png"
        )
    )

    plt.close()

# IMPORTANZA FEATURE RANDOM FOREST

def salva_importanza_feature(
    modello,
    features,
    dataset,
    cartella,
    tipo_features
):

    if not hasattr(
        modello,
        "feature_importances_"
    ):
        return None

    importanza = pd.DataFrame({

        "feature": features,

        "importance": (
            modello.feature_importances_
        )

    })

    importanza = (
        importanza
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importanza.to_csv(
        os.path.join(
            cartella,
            "random_forest_feature_importance.csv"
        ),
        index=False
    )

    # Grafico

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        importanza["feature"],
        importanza["importance"]
    )

    plt.xlabel(
        "Importanza"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"{dataset} - Random Forest\n"
        f"Feature importance - {tipo_features}"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            cartella,
            "random_forest_feature_importance.png"
        )
    )

    plt.close()

    return importanza


# ESECUZIONE DI UN DATASET

def esegui_dataset(
    dataset,
    usa_settings,
    risultati_cv_completi,
    risultati_cv_summary_completi,
    risultati_test_completi
):

    if usa_settings:

        tipo_features = (
            "sensors_plus_settings"
        )

        descrizione_features = (
            "Sensori + Settings"
        )

    else:

        tipo_features = (
            "sensors_only"
        )

        descrizione_features = (
            "Solo sensori"
        )

    print("\n")
    print("====================================================")
    print(f"DATASET: {dataset}")
    print(f"FEATURES: {descrizione_features}")
    print("====================================================")

    # Caricamento

    dati = carica_train(
        dataset
    )

    dati = calcola_rul(
        dati
    )

    # Cross-validation

    (
        risultati_cv,
        risultati_cv_summary
    ) = cross_validation(
        dati,
        usa_settings,
        dataset
    )

    risultati_cv_completi.append(
        risultati_cv
    )

    risultati_cv_summary_completi.append(
        risultati_cv_summary
    )

    # Cartella risultati


    cartella = os.path.join(
        CARTELLA_RESULTS,
        dataset,
        tipo_features
    )

    os.makedirs(
        cartella,
        exist_ok=True
    )

    # Salvataggio CV

    risultati_cv.to_csv(
        os.path.join(
            cartella,
            "risultati_cross_validation.csv"
        ),
        index=False
    )

    risultati_cv_summary.to_csv(
        os.path.join(
            cartella,
            "risultati_cross_validation_summary.csv"
        ),
        index=False
    )

    # Stampa risultati CV

    print("\nRisultati CROSS-VALIDATION:")

    print(
        risultati_cv_summary.to_string(
            index=False
        )
    )

    # FEATURE SELECTION FINALE
    #
    # Ora possiamo usare tutto il training, perché la
    # cross-validation è terminata.

    (
        features,
        sensori_utili,
        settings_utili,
        correlazioni
    ) = seleziona_feature(
        dati,
        usa_settings=usa_settings
    )

    print("\nFeature utilizzate nel training finale:")

    print(features)

    # Salva correlazioni

    correlazioni.to_csv(
        os.path.join(
            cartella,
            "correlazioni_sensori.csv"
        ),
        header=["correlazione"]
    )

    # Salva feature utilizzate

    with open(
        os.path.join(
            cartella,
            "feature_utilizzate.txt"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"Dataset: {dataset}\n"
        )

        file.write(
            f"Tipo features: "
            f"{descrizione_features}\n\n"
        )

        file.write(
            "Sensori utilizzati:\n"
        )

        for sensor in sensori_utili:

            file.write(
                f"- {sensor}\n"
            )

        file.write(
            "\nOperating settings utilizzati:\n"
        )

        if len(settings_utili) == 0:

            file.write(
                "- Nessuno\n"
            )

        else:

            for setting in settings_utili:

                file.write(
                    f"- {setting}\n"
                )

    # TRAINING FINALE + TEST

    risultati_test_dataset = []

    print("\n")
    print("----------------------------------------------------")
    print("TEST UFFICIALE")
    print("----------------------------------------------------")

    for nome_modello in MODELLI:

        print(
            f"\nTest modello: "
            f"{nome_modello}"
        )

        # Riaddestramento su TUTTO il training

        modello_finale = crea_modello(
            nome_modello
        )

        modello_finale.fit(
            dati[features],
            dati["RUL"]
        )

        # Test ufficiale

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

        print(
            f"MAE  = {mae_test:.4f}"
        )

        print(
            f"RMSE = {rmse_test:.4f}"
        )

        # Salva risultati per motore

        nome_file = (
            nome_modello
            .lower()
            .replace(" ", "_")
        )

        risultati_test.to_csv(
            os.path.join(
                cartella,
                f"risultati_test_{nome_file}.csv"
            ),
            index=False
        )

        # Grafici

        crea_grafici(
            risultati_test,
            dataset,
            nome_modello,
            cartella
        )

        # CV summary del modello

        riga_cv = risultati_cv_summary[
            risultati_cv_summary["modello"]
            == nome_modello
        ].iloc[0]

        # Riga finale summary

        riga_summary = {

            "dataset": dataset,

            "features": tipo_features,

            "modello": nome_modello,

            "CV_MAE_mean":
                riga_cv["CV_MAE_mean"],

            "CV_MAE_std":
                riga_cv["CV_MAE_std"],

            "CV_RMSE_mean":
                riga_cv["CV_RMSE_mean"],

            "CV_RMSE_std":
                riga_cv["CV_RMSE_std"],

            "test_MAE":
                mae_test,

            "test_RMSE":
                rmse_test,

            "baseline_test_MAE":
                mae_baseline,

            "baseline_test_RMSE":
                rmse_baseline,

            "n_features":
                len(features)
        }

        risultati_test_dataset.append(
            riga_summary
        )

        risultati_test_completi.append(
            riga_summary
        )

        # Feature importance RF

        if nome_modello == "Random Forest":

            salva_importanza_feature(
                modello_finale,
                features,
                dataset,
                cartella,
                tipo_features
            )

    # Summary del dataset

    pd.DataFrame(
        risultati_test_dataset
    ).to_csv(
        os.path.join(
            cartella,
            "risultati_test_modelli.csv"
        ),
        index=False
    )


# CREAZIONE SUMMARY TESTUALE

def crea_summary_testuale(
    summary,
    percorso
):

    with open(
        percorso,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "====================================================\n"
        )

        file.write(
            "NASA C-MAPSS - RISULTATI COMPLETI\n"
        )

        file.write(
            "====================================================\n\n"
        )

        file.write(
            "Modelli utilizzati:\n"
        )

        file.write(
            "- Linear Regression\n"
        )

        file.write(
            "- Decision Tree\n"
        )

        file.write(
            "- Random Forest\n\n"
        )

        file.write(
            "Cross-validation: 5-fold GroupKFold\n"
        )

        file.write(
            "Grouping: motore (unit)\n"
        )

        file.write(
            "Ogni motore rimane interamente nello stesso fold.\n\n"
        )

        file.write(
            "====================================================\n"
        )

        file.write(
            "RISULTATI PER DATASET\n"
        )

        file.write(
            "====================================================\n\n"
        )

        for dataset in DATASET:

            file.write(
                f"\n******** {dataset} ********\n\n"
            )

            dati_dataset = summary[
                summary["dataset"]
                == dataset
            ]

            for tipo_features in [
                "sensors_only",
                "sensors_plus_settings"
            ]:

                file.write(
                    f"\n--- {tipo_features} ---\n\n"
                )

                dati_features = dati_dataset[
                    dati_dataset["features"]
                    == tipo_features
                ]

                # Baseline

                baseline = dati_features[
                    dati_features["modello"]
                    == "Baseline"
                ]

                if len(baseline) > 0:

                    # Baseline è presente nel CV
                    pass

                # Modelli

                for _, riga in dati_features.iterrows():

                    file.write(
                        f"Modello: "
                        f"{riga['modello']}\n"
                    )

                    file.write(
                        f"  CV MAE   = "
                        f"{riga['CV_MAE_mean']:.4f} "
                        f"+/- "
                        f"{riga['CV_MAE_std']:.4f}\n"
                    )

                    file.write(
                        f"  CV RMSE  = "
                        f"{riga['CV_RMSE_mean']:.4f} "
                        f"+/- "
                        f"{riga['CV_RMSE_std']:.4f}\n"
                    )

                    file.write(
                        f"  Test MAE = "
                        f"{riga['test_MAE']:.4f}\n"
                    )

                    file.write(
                        f"  Test RMSE = "
                        f"{riga['test_RMSE']:.4f}\n"
                    )

                    file.write(
                        f"  Baseline Test MAE = "
                        f"{riga['baseline_test_MAE']:.4f}\n"
                    )

                    file.write(
                        f"  Baseline Test RMSE = "
                        f"{riga['baseline_test_RMSE']:.4f}\n"
                    )

                    file.write(
                        "\n"
                    )

        # CONFRONTO SETTINGS

        file.write(
            "\n====================================================\n"
        )

        file.write(
            "CONFRONTO: SENSORI vs SENSORI + SETTINGS\n"
        )

        file.write(
            "====================================================\n\n"
        )

        for dataset in DATASET:

            file.write(
                f"\n{dataset}\n"
            )

            for modello in MODELLI:

                sensori = summary[
                    (summary["dataset"] == dataset)
                    &
                    (summary["features"] == "sensors_only")
                    &
                    (summary["modello"] == modello)
                ]

                settings = summary[
                    (summary["dataset"] == dataset)
                    &
                    (summary["features"] == "sensors_plus_settings")
                    &
                    (summary["modello"] == modello)
                ]

                if len(sensori) == 0 or len(settings) == 0:
                    continue

                mae_sensori = sensori.iloc[0]["test_MAE"]
                mae_settings = settings.iloc[0]["test_MAE"]

                rmse_sensori = sensori.iloc[0]["test_RMSE"]
                rmse_settings = settings.iloc[0]["test_RMSE"]

                delta_mae = (
                    mae_sensori
                    - mae_settings
                )

                delta_rmse = (
                    rmse_sensori
                    - rmse_settings
                )

                file.write(
                    f"\n{modello}\n"
                )

                file.write(
                    f"  Test MAE - solo sensori: "
                    f"{mae_sensori:.4f}\n"
                )

                file.write(
                    f"  Test MAE - sensori + settings: "
                    f"{mae_settings:.4f}\n"
                )

                file.write(
                    f"  Miglioramento MAE: "
                    f"{delta_mae:.4f}\n"
                )

                file.write(
                    f"  Test RMSE - solo sensori: "
                    f"{rmse_sensori:.4f}\n"
                )

                file.write(
                    f"  Test RMSE - sensori + settings: "
                    f"{rmse_settings:.4f}\n"
                )

                file.write(
                    f"  Miglioramento RMSE: "
                    f"{delta_rmse:.4f}\n"
                )

        # CONFRONTO FAULT MODES

        file.write(
            "\n====================================================\n"
        )

        file.write(
            "CONFRONTO DATASET E MODALITA' DI GUASTO\n"
        )

        file.write(
            "====================================================\n\n"
        )

        file.write(
            "FD001 e FD003: una condizione operativa.\n"
        )

        file.write(
            "FD002 e FD004: sei condizioni operative.\n\n"
        )

        file.write(
            "FD001/FD003 permettono di osservare il cambiamento "
            "associato alle diverse modalita' di degradazione "
            "in presenza di una sola condizione operativa.\n\n"
        )

        file.write(
            "FD002/FD004 permettono di osservare lo stesso "
            "confronto in presenza di condizioni operative multiple.\n\n"
        )

        # Confronto RF feature importance

        file.write(
            "L'analisi delle feature piu' importanti del "
            "Random Forest e' salvata nelle rispettive "
            "cartelle dei dataset.\n"
        )

        file.write(
            "Questa analisi permette di verificare se i "
            "sensori piu' importanti cambiano tra FD001/FD003 "
            "e FD002/FD004.\n"
        )

# MAIN

if __name__ == "__main__":

    risultati_cv_completi = []

    risultati_cv_summary_completi = []

    risultati_test_completi = []

    # Eseguiamo tutti i dataset

    for nome_dataset in DATASET:

        # SOLO SENSORI
     
        esegui_dataset(
            nome_dataset,
            usa_settings=False,
            risultati_cv_completi=risultati_cv_completi,
            risultati_cv_summary_completi=risultati_cv_summary_completi,
            risultati_test_completi=risultati_test_completi
        )

        # SENSORI + OPERATING SETTINGS


        esegui_dataset(
            nome_dataset,
            usa_settings=True,
            risultati_cv_completi=risultati_cv_completi,
            risultati_cv_summary_completi=risultati_cv_summary_completi,
            risultati_test_completi=risultati_test_completi
        )

    # SUMMARY GLOBALI

    summary_test = pd.DataFrame(
        risultati_test_completi
    )

    summary_cv = pd.concat(
        risultati_cv_summary_completi,
        ignore_index=True
    )

    summary_cv_folds = pd.concat(
        risultati_cv_completi,
        ignore_index=True
    )

    # Salvataggio CSV

    summary_test.to_csv(
        "results_summary.csv",
        index=False
    )

    summary_cv.to_csv(
        "results_cross_validation_summary.csv",
        index=False
    )

    summary_cv_folds.to_csv(
        "results_cross_validation_folds.csv",
        index=False
    )

    # Summary testuale
   
    crea_summary_testuale(
        summary_test,
        "results_summary.txt"
    )

    # STAMPA FINALE

    print("\n\n")
    print("====================================================")
    print("SUMMARY FINALE - TEST")
    print("====================================================")

    print(
        summary_test.to_string(
            index=False
        )
    )

    print("\n")
    print("====================================================")
    print("SUMMARY FINALE - CROSS VALIDATION")
    print("====================================================")

    print(
        summary_cv.to_string(
            index=False
        )
    )

    print("\nRisultati salvati in:")

    print("- results_summary.csv")

    print("- results_summary.txt")

    print("- results_cross_validation_summary.csv")

    print("- results_cross_validation_folds.csv")

    print("- results_generale/")
