# guardian-econ-pipeline

## Descriere

Acest proiect construiește un pipeline pentru știri economice, în mai multe etape:

1. colectează articole de afaceri The Guardian pentru perioada 2020–2026;
2. etichetează articolele cu Gemini într-o schemă JSON validată prin Pydantic;
3. creează un set de referință de 250 de articole, care se etichetează manual;
4. compară etichetele LLM cu setul de referință și calculează acordul, Kappa Cohen, macro-F1 și metrice pe clase;
5. folosește un eșantion GDELT pentru partea big-data și compară lunar indicatorii GDELT cu etichetele sentimentale Guardian.

Etichetarea păstrează următoarele câmpuri: `tema`, `sentiment`, `entitati`, `perspectiva_economica` și `incredere`. Setul de referință este folosit pentru evaluarea calității etichetelor generate de LLM.

## Configurare

Necesare:

- Python 3.14;
- un mediu virtual Python;
- cheile API pentru The Guardian și Gemini.

În Windows PowerShell:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Creează un fișier `.env` în rădăcina proiectului:

```dotenv
GUARDIAN_API_KEY=cheia_ta_guardian
GEMINI_API_KEY=cheia_ta_gemini
```

Fișierul `.env` nu trebuie commit-uit și nu trebuie să conțină chei reale în cod sau în documentație.

## Date și rulare

Rulează comenzile din rădăcina proiectului, în această ordine.

### 1. Colectare articole The Guardian

```powershell
python -m src.collect_guardian --per-month 60 --max-articles 7000
```

Colectează incremental articole din secțiunea Business, deduplicate după URL. Scrie `data/articles.parquet`. Dacă fișierul există deja, păstrează datele existente și adaugă lunile nedepășite.

### 2. Colectare eșantion GDELT

```powershell
python -m src.collect_gdelt --step-days 7
```

Colectează arhive GDELT 2.0 din 2020 până la data curentă, la intervale de 7 zile. Scrie:

- `data/gdelt/events/` — evenimente în Parquet, împărțite pe `year` și `month`;
- `data/gdelt/processed.txt` — marcajele fișierelor procesate sau marcate ca invalide.

### 3. Etichetare cu Gemini

```powershell
python -m src.label_llm --model gemini-3.5-flash-lite --daily-limit 400
Se rulează o singură dată pe zi (cota gratuită Gemini este de 500 de apeluri pe zi); la fiecare rulare continuă de unde a rămas, datorită cache-ului.
```

Eșantionează articolele pe luni, cu seed determinist, și validează răspunsurile cu schema Pydantic `ArticleLabel`. Scrie:

- `data/labels.parquet` — etichetele valid generate;
- `data/llm_cache/` — cache-ul pe disc al apelurilor;
- `data/costs.csv` — modelul și tokenii de intrare/ieșire;
- `data/invalid.jsonl` — răspunsuri JSON/validare invalide;
- `data/failed.jsonl` — apeluri sau erori API eșuate.

### 4. Crearea setului de referință

```powershell
python -m src.make_reference_set
```

Selectează determinist 250 de articole dintre cele etichetate și scrie:

- `data/reference_set.xlsx` — registrul de completat manual;
- `data/reference_urls.txt` — URL-urile selectate, câte unul pe linie.

Completează manual coloanele `tema_manual`, `sentiment_manual` și `perspectiva_manual` în registru, apoi salvează copia completă ca `data/reference_set_final.xlsx`. Etapele care urmează folosesc acest fișier final.

### 5. Evaluarea etichetelor LLM

```powershell
python -m src.evaluate_llm_vs_reference
```

Compară setul manual final cu `data/labels.parquet` și scrie:

- `data/llm_vs_reference.xlsx` — rezumat, metrici pe câmp și matricea de confuzie pentru temă;
- `data/llm_errors.xlsx` — articole pentru care tema manuală diferă de tema LLM.

### 6. Compararea GDELT cu Guardian

```powershell
python -m src.gdelt_vs_guardian --min-articles 8
```

Agregă lunar evenimentele GDELT și etichetele Guardian, apoi scrie:

- `data/gdelt_vs_guardian.xlsx` — seria lunară, pragul minim de articole și indicatorii comparației;
- `data/gdelt_vs_guardian.png` — graficul GDELT AvgTone versus ponderea Guardian cu sentiment negativ.

## Teste

```powershell
python -m pytest
```

## Date locale

Directorul `data/` conține fișiere generate și date locale, deci nu este versionat în Git. Nu adăuga `.env`, chei API, Parquet-uri, Excel-uri, cache-uri LLM sau alte rezultate locale în commit-uri.

## Structura proiectului

```text
.
├── data/
│   ├── gdelt/
│   ├── articles.parquet
│   ├── labels.parquet
│   ├── llm_cache/
│   ├── costs.csv
│   └── fișierele Excel/JSON de evaluare
├── docs/
│   └── agent_incidents.md
├── src/
│   ├── collect_guardian.py
│   ├── collect_gdelt.py
│   ├── make_reference_set.py
│   ├── label_llm.py
│   ├── evaluate_llm_vs_reference.py
│   ├── gdelt_vs_guardian.py
│   └── schema.py
├── tests/
├── .env
├── .gitignore
├── AGENTS.md
├── README.md
└── requirements.txt
```

## Rezultate

de completat