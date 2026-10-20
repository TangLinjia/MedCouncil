"""Drug safety checking tool for MedCouncil."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field
import re


RULES_PATH = Path(__file__).resolve().parents[2] / "data" / "drug_safety_rules.json"


class DrugSafetyCheckInput(BaseModel):
    """Input for drug safety checking."""

    proposed_medications: list[str] = Field(
        description="Medication names being considered for treatment. Pass drug names only, without dosage or instructions."
    )

    current_medications: list[str] = Field(
        default_factory=list,
        description="Medication names the patient is currently taking."
    )

    allergies: list[str] = Field(
        default_factory=list,
        description="Known medication allergies."
    )


@lru_cache
def load_rules() -> dict:
    """Load rules and build lookup indexes once."""

    with RULES_PATH.open("r", encoding="utf-8") as file:
        rules = json.load(file)

    aliases = {}

    for canonical_name, alias_list in rules.get("aliases", {}).items():
        canonical_name = canonical_name.lower()
        aliases[canonical_name] = canonical_name

        for alias in alias_list:
            aliases[alias.lower()] = canonical_name

    drug_classes = {
        drug.lower(): {value.lower() for value in classes}
        for drug, classes in rules.get("drug_classes", {}).items()
    }

    for drug in drug_classes:
        aliases.setdefault(
            drug,
            drug,
        )

    interaction_index = {}

    for interaction in rules.get("interactions", []):
        key = frozenset({
            interaction["drug_a"].lower(),
            interaction["drug_b"].lower(),
        })

        interaction_index[key] = interaction

    allergy_index = {
        allergy.lower(): {value.lower() for value in classes}
        for allergy, classes in rules.get("allergy_classes", {}).items()
    }

    drug_pattern = re.compile(
        r"\b(" +
        "|".join(
            re.escape(name)
            for name in sorted(
                aliases,
                key=len,
                reverse=True,
            )
        ) +
        r")\b",
        re.IGNORECASE,
    )

    return {
        "aliases": aliases,
        "drug_classes": drug_classes,
        "interaction_index": interaction_index,
        "allergy_index": allergy_index,
        "drug_pattern": drug_pattern,
    }


def normalize_drug(name: str, rules: dict) -> str:
    """Normalize a medication string to its canonical drug name."""

    name = name.lower().strip()

    match = rules["drug_pattern"].search(
        name
    )

    if match is None:
        return name

    matched_name = match.group(0).lower()

    return rules["aliases"].get(
        matched_name,
        matched_name,
    )


def get_terms(drug: str, rules: dict) -> set[str]:
    """Return the drug name and its drug classes."""

    return {drug, *rules["drug_classes"].get(drug, set())}


def find_interaction(drug_a: str, drug_b: str, rules: dict) -> dict | None:
    """Look up an interaction using indexed drug names and classes."""

    terms_a = get_terms(
        drug_a,
        rules,
    )

    terms_b = get_terms(
        drug_b,
        rules,
    )

    for term_a in terms_a:
        for term_b in terms_b:
            interaction = rules["interaction_index"].get(
                frozenset({
                    term_a,
                    term_b,
                })
            )

            if interaction: return interaction
    return None


@tool("drug_safety_check", args_schema=DrugSafetyCheckInput)
def drug_safety_check(proposed_medications: list[str], current_medications: list[str] | None = None, allergies: list[str] | None = None) -> str:
    """
    Check proposed medications for known drug interactions,
    allergy conflicts, and duplicate therapy.

    This tool uses a limited local rule set. No alert does not
    guarantee that a medication combination is clinically safe.
    """

    current_medications = current_medications or []
    allergies = allergies or []

    rules = load_rules()

    proposed = [
        normalize_drug(
            drug,
            rules,
        )
        for drug in proposed_medications
    ]

    current = [
        normalize_drug(
            drug,
            rules,
        )
        for drug in current_medications
    ]

    allergies = [
        allergy.lower().replace(" allergy", "").strip()
        for allergy in allergies
    ]

    alerts = []

    for proposed_drug in proposed:
        if proposed_drug in current:
            alerts.append(
                {
                    "type": "already_taking",
                    "severity": "info",
                    "drug": proposed_drug,
                }
            )

        proposed_terms = get_terms(
            proposed_drug,
            rules,
        )

        for allergy in allergies:
            allergy_classes = rules["allergy_index"].get(
                allergy,
                set(),
            )

            if proposed_drug == allergy or not proposed_terms.isdisjoint(allergy_classes):
                alerts.append(
                    {
                        "type": "allergy",
                        "severity": "high",
                        "drug": proposed_drug,
                        "allergy": allergy,
                    }
                )

        for current_drug in current:
            interaction = find_interaction(
                proposed_drug,
                current_drug,
                rules,
            )

            if interaction:
                alerts.append(
                    {
                        "type": "drug_interaction",
                        "source": "proposed_current",
                        "drug_a": proposed_drug,
                        "drug_b": current_drug,
                        "severity": interaction["severity"],
                        "description": interaction["description"],
                        "recommendation": interaction["recommendation"],
                    }
                )

    for index, drug_a in enumerate(proposed):
        for drug_b in proposed[index + 1:]:
            interaction = find_interaction(
                drug_a,
                drug_b,
                rules,
            )

            if interaction:
                alerts.append(
                    {
                        "type": "drug_interaction",
                        "source": "proposed_proposed",
                        "drug_a": drug_a,
                        "drug_b": drug_b,
                        "severity": interaction["severity"],
                        "description": interaction["description"],
                        "recommendation": interaction["recommendation"],
                    }
                )

    return json.dumps(
        {
            "status": "alerts_found" if alerts else "no_known_conflicts",
            "alerts": alerts,
            "limitations": "Limited local rule set; absence of an alert does not prove safety.",
        },
        ensure_ascii=False,
        indent=2,
    )
