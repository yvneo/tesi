# Analisi della Robustezza di Modelli di ML per la Network Traffic Classification 

Benvenuti nella repository dedicata al codice e agli esperimenti della mia tesi di laurea in Informatica presso l'**Università degli Studi di Verona**.

##  Progetto
L'obiettivo di questo lavoro è analizzare la **robustezza**, la **spiegabilità** e la **sostenibilità** dei modelli di Machine Learning applicati alla classificazione del traffico di rete, con un focus particolare sugli scenari **Cross-Location** (quando il modello viene addestrato in un luogo e testato in un altro).

###  Focus della Ricerca
* **Problematica:** L'evoluzione del traffico criptato (TLS, QUIC) rende obsoleta la *Deep Packet Inspection* (DPI).
* **Soluzione:** Utilizzo di algoritmi di ML per classificare il traffico basandosi su caratteristiche statistiche dei flussi.
* **Analisi SAGE:** Implementazione del framework *Shapley Additive Global Importance* (SAGE) per interpretare le decisioni dei modelli.

##  Tecnologie e Algoritmi
Il progetto mette a confronto diversi approcci per valutare quale sia il più affidabile in condizioni reali:

* **Modelli basati su alberi:** `Random Forest`, `XGBoost`
* **Modelli basati su istanze:** `K-Nearest Neighbors (k-NN)`
* **Data Collection:** Sistema `PcapMon` per l'acquisizione e caratterizzazione dei dataset.
* **Librerie principali:** `scikit-learn`, `XGBoost`, `pandas`, `numpy`.

