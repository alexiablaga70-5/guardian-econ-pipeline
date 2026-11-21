# Raport de incidente al agentului

Agent folosit: GitHub Copilot (modul Agent, VS Code), ghidat de `AGENTS.md`.
Proiect: pipeline de date construit de agent (The Guardian + GDELT), domeniu economic.

Fiecare incident are: ce a făcut agentul, cum a fost depistat și cum a fost corectat.

## Tabel rezumat

| # | Etapă | Incident | Depistat prin | Corectat prin |
|---|---|---|---|---|
| 1 | Colectare Guardian | A tratat „economics" ca secțiune (este tag) | Rulare reală: eroare 404 | Endpoint `/search` cu `section=business`, argument `--tag` |
| 2 | Colectare Guardian | A luat articolele de la cele mai vechi la cele mai noi | Primele titluri, toate din 1 ianuarie 2020 | Eșantionare lunară (`--per-month`) |
| 3 | Colectare Guardian | Script fără mesaje de progres și fără rezumat | Rulare tăcută, fără informații | Progres și rezumat final |
| 4 | Colectare Guardian | Cheia API afișată în mesajul de eroare (încalcă `AGENTS.md`) | Traceback din terminal cu URL-ul complet | Erorile afișează doar codul de stare și calea, fără parametri |
| 5 | Colectare Guardian | Colectarea s-a oprit la limită, cu 2025–2026 aproape goale | Numărătoarea pe ani: 324 (2025) și 22 (2026) | Rulare repetată cu `--per-month 60 --max-articles 7000`, total 6.062 |
| 6 | Etichetare LLM | Import relativ: scriptul nu pornea din linia de comandă | Prima rulare directă (`ImportError`) | Pornire ca modul: `python -m src.label_llm` |
| 7 | Etichetare LLM | Articolele procesate în ordine cronologică, nu în eșantion amestecat | Anii articolelor etichetate: doar 2020 | Ordine amestecată cu seed fix, eșantion pe luni |
| 8 | Etichetare LLM | Prompt fără regulă pentru articole neeconomice | Niciun `altele` în primele 20 de etichete | Regulă în prompt, etichete și cache reluate |
| 9 | Etichetare LLM | Mesaj de eroare vag („status=unknown") și oprire la prima eroare | Oprirea scriptului la 373 de articole |Reîncercări cu așteptare crescătoare și log în `data/failed.jsonl` |
| 10 | Set de referință | Scriere Excel cu date cu fus orar | `ValueError` la rulare | Conversie la date fără fus orar, test cu xlsx real |

## Detalii

### 1. „economics" tratat ca secțiune
Agentul a presupus că `business` și `economics` sunt secțiuni la The Guardian. `economics` este un tag din `business`, așa că API-ul a răspuns 404. Testul cu răspuns simulat (mock) a trecut, deci nu putea prinde greșeala: presupunerea era în cod și în test.

### 2. Ordine cronologică în colectare
Scriptul cerea articolele cu `order-by=oldest` și se oprea la limită, deci tot eșantionul ar fi venit din primele zile ale lui 2020. Specificația cerea acoperire 2020–2026, dar agentul n-a tradus-o în cod.

### 3. Lipsa mesajelor de progres
Scriptul rula fără să afișeze nimic, deci nu se putea ști dacă a reușit. Agentul a respectat literal cererea („scrie scriptul") fără să adauge feedback minim.

### 4. Cheia API în mesajul de eroare
`response.raise_for_status()` includea URL-ul complet, cu cheia, în excepție. Regula din `AGENTS.md` interzicea afișarea cheilor. Testele nu verificau mesajele de eroare, deci nu au prins problema.

### 5. Colectare oprită la limită
Cu limita de 4.998 de articole, anii 2025 și 2026 au rămas aproape goi (324 și 22 de articole). Detectat prin numărătoare pe ani, înainte de etichetare.

### 6. Import relativ
`label_llm.py` folosea `from .schema import ...`. Testele rulau sub `pytest` și treceau, dar scriptul pornit direct dădea `ImportError`. Soluție: se pornește mereu ca modul.

### 7. Ordine cronologică în etichetare
Chiar dacă specificația cerea eșantion răspândit pe luni, articolele erau procesate în ordinea lunilor. Cu limita zilnică, primele zile ar fi etichetat doar 2020. Detectat verificând anii articolelor etichetate (toți 2020).

### 8. Prompt fără categoria „altele"
Secțiunea Business conține și articole neeconomice, dar promptul nu avea o regulă pentru ele. După corectare, etichetele și cache-ul au fost șterse și refăcute cu noul prompt.

### 9. Gestionarea erorilor API
Scriptul afișa „Gemini API error: status=unknown", fără tipul erorii, și se oprea. Cota zilnică Google s-a dovedit a fi cauza uneia dintre opriri (eroare 429).

### 10. Excel cu fus orar
`pandas.to_excel` nu acceptă date cu fus orar. Testul nu scria un xlsx real cu date reale.

## Concluzii

- Testele cu date simulate au trecut în aproape toate cazurile, dar rularea pe date reale a găsit problemele. Verificarea pe un eșantion mic, cu date reale, înainte de rularea completă, a fost cel mai util pas.
- Agentul respectă literal ce i se cere și nu adaugă ce nu este specificat (progres, acoperire în timp, gestionarea erorilor, siguranța cheilor).
- Sarcinile mici și bine delimitate (`Do not change anything else`) au limitat daunele: agentul n-a modificat fișiere în afara celor cerute.
- Regulile învățate au fost adăugate în `AGENTS.md`.
