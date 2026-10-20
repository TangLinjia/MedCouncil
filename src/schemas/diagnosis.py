"""Diagnosis-related data models for MedCouncil."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

PriorityLevel = Literal[
    "high",
    "medium",
    "low",
]

class ConfidenceLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class DiagnosticHypothesis(BaseModel):
    """Initial diagnostic hypothesis proposed by the General Physician."""

    condition: str
    reasoning: str
    priority: PriorityLevel = "medium"

class InitialAssessment(BaseModel):
    """Initial assessment from the General Physician."""

    sufficient_information: bool
    missing_information: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    hypotheses: list[DiagnosticHypothesis] = Field(default_factory=list, max_length=3,)
    overall_reasoning: str

class HypothesisEvaluation(BaseModel):
    """Independent evaluation of one diagnostic hypothesis."""

    condition: str
    confidence: ConfidenceLevel = ConfidenceLevel.MODERATE
    reasoning: str
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    recommended_tests: list[str] = Field(default_factory=list)
    notes: Optional[str] = None

class DiagnosisSynthesis(BaseModel):
    """Synthesized diagnosis based on expert evaluations."""

    primary_diagnosis: Optional[str] = None
    confidence: ConfidenceLevel = ConfidenceLevel.MODERATE
    reasoning: str
    differential_diagnoses: list[str] = Field(default_factory=list)
    recommended_tests: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
