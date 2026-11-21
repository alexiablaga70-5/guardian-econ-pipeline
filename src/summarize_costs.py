import csv
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COSTS_PATH = PROJECT_ROOT / "data" / "costs.csv"
REPORT_PATH = PROJECT_ROOT / "docs" / "costs.md"
LABELING_MODEL = "gemini-3.5-flash-lite"
REQUIRED_COLUMNS = {"timestamp", "model", "input_tokens", "output_tokens"}


@dataclass(frozen=True)
class CostSummary:
    calls: int
    input_tokens: int
    output_tokens: int

    @property
    def hypothetical_claude_cost(self) -> float:
        return (
            self.input_tokens * 0.10 + self.output_tokens * 0.50
        ) / 1_000_000


def read_cost_summary(costs_path: Path) -> CostSummary:
    calls = 0
    input_tokens = 0
    output_tokens = 0

    with costs_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Coloane lipsă în {costs_path}: {missing}")

        for row_number, row in enumerate(reader, start=2):
            if row["model"] != LABELING_MODEL:
                continue
            try:
                row_input_tokens = int(row["input_tokens"])
                row_output_tokens = int(row["output_tokens"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Valori invalide pentru tokeni la rândul {row_number}"
                ) from error
            if row_input_tokens < 0 or row_output_tokens < 0:
                raise ValueError(
                    f"Valori negative pentru tokeni la rândul {row_number}"
                )

            calls += 1
            input_tokens += row_input_tokens
            output_tokens += row_output_tokens

    return CostSummary(calls, input_tokens, output_tokens)


def render_report(summary: CostSummary) -> str:
    hypothetical_cost = f"{summary.hypothetical_claude_cost:.7f}"
    return f"""# Costurile proiectului

## Etapa de etichetare LLM

| Model | Apeluri | Tokeni de intrare | Tokeni de ieșire |
|---|---:|---:|---:|
| Gemini 3.5 Flash Lite (`{LABELING_MODEL}`) | {summary.calls} | {summary.input_tokens} | {summary.output_tokens} |

Suma cheltuită efectiv a fost **0.00 USD**, deoarece a fost folosit nivelul gratuit al API-ului Gemini și nu a fost conectat niciun cont de facturare.

**Comparație ipotetică, nu cost real:** aceiași tokeni la tarifele Claude Haiku 5.5 de 0.10 USD per milion de tokeni de intrare și 0.50 USD per milion de tokeni de ieșire ar costa **{hypothetical_cost} USD**.

## Costurile celorlalte etape

| Etapă | Cost |
|---|---:|
| The Guardian API (cheie gratuită pentru dezvoltatori) | 0 USD |
| GDELT (gratuit, fără cheie) | 0 USD |
| GitHub Copilot (plan gratuit pentru studenți) | 0 USD |
| Antrenarea modelului în Colab/Kaggle (GPU gratuit) | 0 USD |

Cache-ul LLM din `data/llm_cache` permite reluarea fără apeluri suplimentare, deci rerulările nu generează costuri API noi.
"""


def main() -> None:
    summary = read_cost_summary(COSTS_PATH)
    report = render_report(summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Raport scris in: {REPORT_PATH}")
    print("Etichetare LLM (Gemini 3.5 Flash Lite):")
    print(f"  Apeluri: {summary.calls}")
    print(f"  Tokeni de intrare: {summary.input_tokens}")
    print(f"  Tokeni de iesire: {summary.output_tokens}")
    print("  Suma cheltuita efectiv: 0.00 USD")
    print(
        "  Comparatie ipotetica Claude Haiku 5.5: "
        f"{summary.hypothetical_claude_cost:.7f} USD"
    )
    print(
        "Alte etape: Guardian API=0 USD; GDELT=0 USD; "
        "GitHub Copilot=0 USD; antrenare Colab/Kaggle=0 USD"
    )


if __name__ == "__main__":
    main()
