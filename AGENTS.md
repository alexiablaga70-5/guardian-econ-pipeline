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

## Reguli învățate din incidente
- Nu afișa niciodată o cheie API sau un URL cu parametri. La HTTPError afișează doar codul de stare și calea.
- Testează pe date reale, pe un eșantion mic, înainte de rularea completă. Un test cu răspuns simulat nu arată că un endpoint sau un nume există.
- Nu presupune nume de secțiuni, câmpuri sau endpoint-uri: verifică în documentația API.
- Colectarea și etichetarea trebuie să acopere toată perioada 2020–2026, nu doar primele luni. Folosește eșantionare pe luni și ordine amestecată cu seed fix.
- Fiecare script afișează progres și un rezumat final (număr de elemente, interval de date, erori).
- Scripturile se rulează ca module: `python -m src.nume_script`. Nu folosi importuri relative care funcționează doar sub pytest.
- Orice scriere în Excel se testează cu un fișier xlsx real; datele cu fus orar se convertesc înainte.
- La erori de API (429, 500, 503) reîncearcă cu așteptare crescătoare; nu opri tot scriptul pentru un singur element.
- Nu modifica fișiere în afara celor cerute în sarcină.
