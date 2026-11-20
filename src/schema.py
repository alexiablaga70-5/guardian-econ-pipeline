from typing import Literal

from pydantic import BaseModel, Field


Tema = Literal[
    "inflatie",
    "piata_muncii",
    "comert",
    "energie",
    "locuinte",
    "piete_financiare",
    "politici_publice",
    "corporatii",
    "altele",
]
Sentiment = Literal["negativ", "neutru", "pozitiv"]
PerspectivaEconomica = Literal[
    "in_scadere",
    "stabila",
    "in_crestere",
    "neclar",
]


class ArticleLabel(BaseModel):
    tema: Tema
    sentiment: Sentiment
    entitati: list[str]
    perspectiva_economica: PerspectivaEconomica
    incredere: float = Field(ge=0, le=1)
