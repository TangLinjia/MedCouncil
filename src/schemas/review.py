"""Final review data models for MedCouncil."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ReviewIssue(BaseModel):
    """A potential issue found during final review."""

    category: str
    description: str
    recommendation: Optional[str] = None


class FinalReview(BaseModel):
    """Structured output from the final review agent."""

    summary: str

    diagnosis_consistent: bool = True
    treatment_appropriate: bool = True

    issues: list[ReviewIssue] = Field(default_factory=list)

    missing_information: list[str] = Field(default_factory=list)
    safety_concerns: list[str] = Field(default_factory=list)

    follow_up_recommendations: list[str] = Field(default_factory=list)

    final_notes: Optional[str] = None