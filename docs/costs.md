# Costurile proiectului

## Etapa de etichetare LLM

| Model | Apeluri | Tokeni de intrare | Tokeni de ieșire |
|---|---:|---:|---:|
| Gemini 3.5 Flash Lite (`gemini-3.5-flash-lite`) | 1372 | 1139848 | 149804 |

Suma cheltuită efectiv a fost **0.00 USD**, deoarece a fost folosit nivelul gratuit al API-ului Gemini și nu a fost conectat niciun cont de facturare.

**Comparație ipotetică, nu cost real:** aceiași tokeni la tarifele Claude Haiku 5.5 de 0.10 USD per milion de tokeni de intrare și 0.50 USD per milion de tokeni de ieșire ar costa **0.1888868 USD**.

## Costurile celorlalte etape

| Etapă | Cost |
|---|---:|
| The Guardian API (cheie gratuită pentru dezvoltatori) | 0 USD |
| GDELT (gratuit, fără cheie) | 0 USD |
| GitHub Copilot (plan gratuit pentru studenți) | 0 USD |
| Antrenarea modelului în Colab/Kaggle (GPU gratuit) | 0 USD |

Cache-ul LLM din `data/llm_cache` permite reluarea fără apeluri suplimentare, deci rerulările nu generează costuri API noi.
