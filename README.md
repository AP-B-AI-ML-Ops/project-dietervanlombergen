# Describe the problem and your application in the README.md of your repository



# Projectdocumentatie: Solar Energy Predictor

---

## 1. Dataset(s)

### Welke data gebruik je?
Voor dit project maken we gebruik van twee specifieke, reële databronnen die lokaal zijn opgeslagen in de map `./data/`:
1. **Weersvoorspelling Data (`sun_combined.csv`):** Dit bestand bevat historische zonnestralingsmetingen ($W/m^2$) die zijn samengevoegd uit openbare weersbronnen zoals Open-Meteo, het KMI en Kaggle.
2. **Netproductie Data (`productie_comnbined.csv`):** Dit bestand bevat de werkelijke, realtime opgewekte zonne-energie in Vlaanderen, geschaald van Kilowattuur (kWh) naar Megawatt (MW).

### Hoe verkrijgen we Training-, Validatie- en Testdata?

Onze data-preprocessing pipeline splitst deze data op chronologische wijze (in plaats van willekeurig) om data-lekkage door de tijd heen te voorkomen:
* Het script laadt de ruwe data, voert een *inner join* uit op de tijdstempels (`tijd`) om de weersomstandigheden perfect uit te lijnen met de werkelijke netstroom, en berekent automatisch cyclische tijdsfuncties (`sin_hour` en `cos_hour`).
* Vervolgens wordt de dataset opgesplitst op basis van een chronologische $80/20$-verhouding. Deze subsets worden opgeslagen als binaire pickle-bestanden (`train.pkl` en `val.pkl`) die rechtstreeks worden ingeladen tijdens de hyperparameter-tuning en modeltraining.

### Hoe verkrijgen we nieuwe data voor de service?

* **Voor de Web Service (Live REST API):** De eindgebruiker of een front-end client stuurt een standaard JSON-payload via een HTTP POST-verzoek met live wp
---

## 2. Projectuitleg

### Wat doet de service precies?
We hebben een machine learning-toepassing op productieniveau gebouwd die de **zonne-energieproductie (in MW)** voor de komende 24 uur voorspelt op basis van binnenkomende weersvoorspellingen.

### Wat voor soort applicatie is het en wat is het doel?
De applicatie functioneert als een schaalbare microservice die is opgedeeld in drie centrale deployment-patronen:
1. **Een Live REST API (Inference):** Een Flask-endpoint in een Docker-container waarmee externe systemen direct realtime voorspellingen kunnen opvragen.
2. **Een Geplande Batch Engine:** Een geautomatiseerde Prefect-pipeline die dagelijks prestatie- en driftmetingen wegschrijft naar Parquet-rapporten.
3. **Een Centraal Tracking Dashboard:** Een experiment-hub waarin modelversies en parameters nauwkeurig worden bijgehouden via MLflow.

Het uiteindelijke doel van dit project is om netbeheerders en energiehandelaren te helpen bij het anticiperen op fluctuaties in hernieuwbare energie. Hierdoor kan het elektriciteitsnet efficiënter in balans worden gehouden en hoeven er minder fossiele back-upcentrales te worden ingezet.

---

## 3. Flows & Actions

Om dit ML-systeem stabiel en hands-free te laten draaien, hebben we de logica opgedeeld in een hiërarchie van Prefect-flows en -taken.

### Kernflows en Acties:

1. **De End-to-End Parent Flow (`main.py`):**
   Deze flow coördineert ons volledige machine learning-systeem door achtereenvolgens vier specifieke subflows aan te roepen:
   * **Fase 1 - Preprocessing Flow:** Schoont de ruwe data op, voegt de tabellen samen, berekent de cyclische tijdskenmerken en exporteert de binaire numpy-matrices.
   * **Fase 2 - Baseline Training Flow:** Traint een standaard XGBoost-regressiemodel en registreert de startstatistieken via MLflow autologging.
   * **Fase 3 - Hyperparameter Optimalisatie (HPO) Flow:** Start een Optuna-studie die verspreid over 15 trials zoekt naar de absolute beste parametercombinaties (zoals `max_depth` en `learning_rate`).
   * **Fase 4 - Registratie Flow:** Retraint het model op de volledige dataset met de optimale parameters uit de HPO-fase en registreert deze als de nieuwe 'Champion'-versie in de MLflow Model Registry.

2. **De Geplande Batch Inference Flow (`batch.py`):**
   Deze flow draait continu op de achtergrond als een Prefect-worker en voert op een vast schema de volgende acties uit om modeldrift te controleren:
   * **Actie 1:** Communiceert met de MLflow-server en downloadt automatisch de nieuwste 'Champion'-modelbestanden.
   * **Actie 2:** Haalt de recentste weersvoorspellingsdata binnen voor de komende dag.
   * **Actie 3:** Genereert zonne-energievoorspellingen via het ingeladen model.
   * **Actie 4:** Vraagt de werkelijk behaalde opbrengst op bij het Elia-netwerk, berekent de foutmarge (delta) en slaat dit op als een gestructureerd `.parquet`-bestand voor kwaliteitsmonitoring.


## 1. De DevContainers Opstarten

We maken gebruik van een voorgeconfigureerde **VS Code DevContainer** zodat alle benodigde dependencies (zoals Python 3.12, Docker-in-Docker en Git-hooks) direct klaarstaan zonder je lokale machine te vervuilen.

1. Open de projectmap in **VS Code**.
2. Als de DevContainer-extensie is geïnstalleerd, verschijnt er rechtsonder een pop-up. Klik op **Reopen in Container**.
   * *Alternatief:* Open het Command Palette (`Ctrl+Shift+P` of `Cmd+Shift+P`) en typ: `Dev Containers: Reopen in Container`.
3. VS Code bouwt nu de containeromgeving op. Dit kan de eerste keer een paar minuten duren. Zodra de terminal onderin start met `vscode ➜ /workspace`, ben je succesvol ingelogd!
4. Doe een install van alle packages ```pip install -r requirements.txt```
5. Voor lokaal development run ook nog ```export prefect config set PREFECT_API_URL=http://orchestration:4200/api```



---

## 2. De Core Infrastructuur Starten (Docker Compose)

Voordat we code kunnen draaien, moeten de centrale MLOps-services (MLflow en Prefect) op de achtergrond worden opgestart.

1. Kopieer het voorbeeld-omgevingsbestand naar een actieve `.env`:
   ```bash
   cp sample.env .env



## Stap 4.2: De Core Infrastructuur Starten (Docker Compose)

Start de achtergrondservices zoals de MLflow-server en de Prefect-orkestrator:

```bash
docker compose up -d
```

### Toegang tot dashboards

- MLflow UI: `http://localhost:5000`
- Prefect UI: `http://localhost:4200`

---

## Stap 4.3: De Main Trainingspipeline Runnen

Voer de volledige end-to-end pipeline uit (data cleaning, optimalisatie via Optuna en modelregistratie):

```bash
python experiment-tracking/main.py
```

### Resultaat

Zodra dit script klaar is, staat je allerbeste model automatisch geregistreerd als de actieve Champion in de MLflow Model Registry.

---

## Stap 4.4: De Geplande Batch Service Activeren

De batch service controleert periodiek nieuwe weersvoorspellingen tegen de reële netproductiedata van Elia.

### Start de Prefect-worker op

```bash
cd deploy-batch
python batch.py
```

###  Handmatig een proefrun start

Open een nieuw browser tabblad en ga naar de UI op localhost:4200. Daar kan je een nieuwe deployment starten.

### (Optioneel) Handmatig een proefrun forceren

Open een nieuwe, aparte terminal en voer uit:

```bash
prefect deployment run 'run-batch/solar-monthly-batch-deployment'
```

### Output

De gegenereerde rapportages en foutmarges verschijnen als gecomprimeerde `.parquet`-bestanden in de map:

```text
./batch-reports/
```

---

## Stap 4.5: De Live Web Service (REST API) Testen

Laadt het Champion-model in en serveert realtime voorspellingen via een Flask-endpoint.

### Start de webserver op

```bash
python deploy-webservice/predict.py
```

### Test de API-endpoint met `curl`

Open een nieuwe terminal en voer uit:

```bash
curl -X POST http://web-api:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"radiation_wm2": 450.5, "sin_hour": 0.5, "cos_hour": -0.866}'
```

### Verwacht JSON-antwoord

```json
{
  "solar_production_mw_predicted": 22.41,
  "model_version": 5
}
```

---

## Stap 4.6: Kwaliteitscontroles (Kwaliteit & Tests)

### Unit- & Integratietests uitvoeren

```bash
pytest tests/ -v
```

### Codekwaliteit controleren via de pre-commit checks

```bash
pre-commit
```
