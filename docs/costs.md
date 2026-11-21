# Costurile proiectului

## Etapa de etichetare LLM

| Model | Apeluri | Tokeni de intrare | Tokeni de ieșire |
|---|---:|---:|---:|
| Gemini 3.5 Flash Lite (`gemini-3.5-flash-lite`) | 924 | 769143 | 101272 |

Suma cheltuită efectiv a fost **0.00 USD**, deoarece a fost folosit nivelul gratuit al API-ului Gemini și nu a fost conectat niciun cont de facturare.

**Comparație ipotetică, nu cost real:** aceiași tokeni la tarifele Claude Haiku 5.5 de 0.10 USD per milion de tokeni de intrare și 0.50 USD per milion de tokeni de ieșire ar costa **0.1275503 USD**.

## Costurile celorlalte etape

| Etapă | Cost |
|---|---:|
| The Guardian API (cheie gratuită pentru dezvoltatori) | 0 USD |
| GDELT (gratuit, fără cheie) | 0 USD |
| GitHub Copilot (plan gratuit) | 0 USD |
| Antrenarea modelului în Colab/Kaggle (GPU gratuit) | 0 USD |

Cache-ul LLM din `data/llm_cache` permite reluarea fără apeluri suplimentare, deci rerulările nu generează costuri API noi.

Cele 924 de apeluri includ aproximativ 20 de apeluri făcute cu o versiune anterioară a promptului.

Nu a fost conectat niciun cont de facturare la furnizorul LLM, deci cheltuiala maximă posibilă a fost 0 USD (limita de cheltuieli cerută în enunț este respectată prin construcție). Nu s-a folosit Batch API, pentru că nivelul gratuit Gemini are limită de apeluri pe zi, iar cache-ul pe disc a evitat reapelurile.
