"""Manual integration tests for Treatment Agent."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from src.agents.treatment import treatment_agent


class TestMedication(BaseModel):
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    started_on: Optional[str] = None


class TestPatient(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    chief_complaint: str
    symptoms: list[str] = Field(default_factory=list)
    medical_history: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    current_medications: list[TestMedication] = Field(default_factory=list)


class TestDiagnosis(BaseModel):
    primary_diagnosis: Optional[str] = None
    confidence: str
    reasoning: str
    differential_diagnoses: list[str] = Field(default_factory=list)
    recommended_tests: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)


def run_test(name: str, patient: TestPatient, diagnosis: TestDiagnosis):
    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    state = {
        "patient": patient,
        "diagnosis_synthesis": diagnosis,
        "errors": [],
    }

    result = treatment_agent(state)

    print("\n[TREATMENT AGENT RESULT]")

    treatment_plan = result.get(
        "treatment_plan"
    )

    if treatment_plan:
        print(
            treatment_plan.model_dump_json(indent=2)
        )
    else:
        print(result)


# ============================================================
# TEST 4
# Drug-drug interaction
# nitroglycerin + sildenafil
# ============================================================

import json

from src.tools.drug_safety_check import drug_safety_check


print("\n")
print("=" * 70)
print("TEST 4 - DRUG-DRUG INTERACTION")
print("=" * 70)

safety_result = drug_safety_check.invoke(
    {
        "proposed_medications": [
            "nitroglycerin"
        ],
        "current_medications": [
            "sildenafil"
        ],
        "allergies": [],
    }
)

print(
    f"[DRUG SAFETY RESULT]\n{safety_result}"
)

result = json.loads(
    safety_result
)

assert result["status"] == "alerts_found"

interaction_alerts = [
    alert
    for alert in result["alerts"]
    if alert["type"] == "drug_interaction"
]

assert len(interaction_alerts) > 0

assert interaction_alerts[0]["drug_a"] == "nitroglycerin"
assert interaction_alerts[0]["drug_b"] == "sildenafil"
assert interaction_alerts[0]["severity"] == "critical"

print(
    "\n[TEST 4 PASSED] "
    "Nitroglycerin and sildenafil interaction was correctly detected."
)


# ============================================================
# TEST 5
# Drug allergy conflict
# amoxicillin + penicillin allergy
# ============================================================

# import json

# from src.tools.drug_safety_check import drug_safety_check


# print("\n")
# print("=" * 70)
# print("TEST 5 - DRUG ALLERGY")
# print("=" * 70)

# safety_result = drug_safety_check.invoke(
#     {
#         "proposed_medications": [
#             "amoxicillin"
#         ],
#         "current_medications": [],
#         "allergies": [
#             "penicillin"
#         ],
#     }
# )

# print(
#     f"[DRUG SAFETY RESULT]\n{safety_result}"
# )

# result = json.loads(
#     safety_result
# )

# assert result["status"] == "alerts_found"

# allergy_alerts = [
#     alert
#     for alert in result["alerts"]
#     if alert["type"] == "allergy"
# ]

# assert len(allergy_alerts) > 0

# assert allergy_alerts[0]["drug"] == "amoxicillin"
# assert allergy_alerts[0]["allergy"] == "penicillin"
# assert allergy_alerts[0]["severity"] == "high"

# print(
#     "\n[TEST 5 PASSED] "
#     "Amoxicillin was correctly flagged for penicillin allergy."
# )


# ============================================================
# TEST 6
# No medication should be required
# ============================================================

# patient_6 = TestPatient(
#     age=28,
#     gender="male",
#     chief_complaint="mild ankle pain after twisting injury",
#     symptoms=[
#         "mild ankle pain",
#     ],
#     medical_history=[],
#     allergies=[],
#     current_medications=[],
# )

# diagnosis_6 = TestDiagnosis(
#     primary_diagnosis=None,
#     confidence="low",
#     reasoning=(
#         "The available information is insufficient to confirm a "
#         "specific diagnosis. Physical examination is required before "
#         "pharmacologic treatment is considered."
#     ),
#     differential_diagnoses=[
#         "minor ankle sprain",
#     ],
#     recommended_tests=[
#         "physical examination",
#     ],
#     unresolved_questions=[
#         "ability to bear weight",
#         "presence of bony tenderness",
#         "degree of swelling",
#     ],
# )

# run_test(
#     "TEST 6 - NO MEDICATION",
#     patient_6,
#     diagnosis_6,
# )
