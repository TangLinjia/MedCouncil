"""Shared workflow state for the MedCouncil graph."""

from __future__ import annotations

import operator

from typing import Annotated, Optional, TypedDict

from .patient import PatientInfo
from .diagnosis import (
    DiagnosticHypothesis,
    DiagnosisSynthesis,
    HypothesisEvaluation,
)
from .treatment import TreatmentPlan
from .review import FinalReview


class MedCouncilState(TypedDict, total=False):
    """Shared state passed between agents in the MedCouncil workflow."""

    raw_input: str
    patient: Optional[PatientInfo] # input.py
    needs_more_information: bool # general_physician.py
    missing_information: list[str] # general_physician.py
    follow_up_questions: list[str] # general_physician.py
    diagnostic_hypotheses: list[DiagnosticHypothesis] # general_physician.py
    gp_reasoning: str # general_physician.py
    hypothesis_evaluations: Annotated[list[HypothesisEvaluation], operator.add] # diagnosis_expert.py
    diagnosis_synthesis: Optional[DiagnosisSynthesis] # diagnosis_synthesis.py
    treatment_plan: Optional[TreatmentPlan] # treatment.py
    final_review: Optional[FinalReview] # final_review.py
    current_step: str # update every time
    errors: Annotated[list[str], operator.add]


class DiagnosisExpertState(TypedDict):
    """Input state for evaluating a single diagnostic hypothesis."""

    patient: PatientInfo
    hypothesis: DiagnosticHypothesis