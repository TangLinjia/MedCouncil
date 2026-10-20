"""Patient-related data models used across the MedCouncil workflow."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class SeverityLevel(str, Enum):
    VERY_MILD = "very_mild"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    VERY_SEVERE = "very_severe"


class DurationUnit(str, Enum):
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    YEARS = "years"


class Symptom(BaseModel):
    """A symptom reported by the patient."""

    name: str
    duration_value: Optional[float] = Field(default=None, ge=0)
    duration_unit: Optional[DurationUnit] = None
    severity: Optional[SeverityLevel] = None
    details: Optional[str] = None


class Allergy(BaseModel):
    """Known patient allergy."""

    allergen: str
    reaction: Optional[str] = None
    severity: Optional[SeverityLevel] = None


class Medication(BaseModel):
    """Medication currently used by the patient."""

    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    started_on: Optional[date] = None


class VitalSigns(BaseModel):
    """Basic clinical vital signs."""

    temperature_c: Optional[float] = Field(
        default=None,
        description="Body temperature in Celsius",
    )
    heart_rate: Optional[int] = Field(
        default=None,
        ge=0,
        description="Heart rate in beats per minute",
    )
    systolic_bp: Optional[int] = Field(default=None, ge=0)
    diastolic_bp: Optional[int] = Field(default=None, ge=0)
    respiratory_rate: Optional[int] = Field(default=None, ge=0)
    oxygen_saturation: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Peripheral oxygen saturation percentage",
    )


class LabResult(BaseModel):
    """Single laboratory test result."""

    test_name: str
    value: str
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    abnormal: bool = False


class PatientInfo(BaseModel):
    """Structured patient information shared between medical agents."""

    patient_id: Optional[str] = None

    name: Optional[str] = None
    age: Optional[int] = Field(default=None, ge=0, le=130)
    gender: Gender = Gender.UNKNOWN

    chief_complaint: str

    symptoms: list[Symptom] = Field(default_factory=list)

    medical_history: list[str] = Field(default_factory=list)
    family_history: list[str] = Field(default_factory=list)

    allergies: list[Allergy] = Field(default_factory=list)
    current_medications: list[Medication] = Field(default_factory=list)

    vital_signs: Optional[VitalSigns] = None
    lab_results: list[LabResult] = Field(default_factory=list)

    raw_input: Optional[str] = Field(
        default=None,
        description="Original unstructured patient input",
    )

    created_at: datetime = Field(default_factory=datetime.now)