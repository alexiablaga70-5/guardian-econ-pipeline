# AGENTS.md

## Scop
Pipeline: colectare The Guardian + GDELT -> etichetare LLM (JSON validat)
-> fine-tuning DistilBERT. Domeniu: știri economice.

## Reguli generale
- Lucrează în pași mici; un pas = o funcție + un test. Nu schimba fișiere
  în afara celor cerute.
- Python 3.14, dependențe în requirements.txt. Nu adăuga pachete fără să întrebi.
  Dacă un pachet nu se instalează, spune-mi, nu căuta soluții ocolitoare.
- Nu citi și nu scrie chei API în cod; folosește variabile de mediu (.env, ignorat în Git).

## Date
- Stochează în Parquet sau DuckDB, în data/. Deduplicare după URL.
- Colectare incrementală: nu re-descărca ce există deja.
- Doar surse gratuite (The Guardian Open Platform, GDELT).

## Etichetare LLM
- Schema JSON validată cu Pydantic; respinge și loghează răspunsurile invalide.
- Cache pe disc pentru FIECARE apel (cheie = hash prompt + model). Rerularea
  nu trebuie să genereze costuri.
- Maximum 5.000 de articole. Loghează tokenii de intrare/ieșire în costs.csv.

## Teste și documentație
- pytest pentru fiecare modul; README cu pași de rulare.
- Orice decizie de design neașteptată se notează în docs/agent_incidents.md.
