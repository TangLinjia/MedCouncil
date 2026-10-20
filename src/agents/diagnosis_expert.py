"""Diagnosis expert agent for evaluating diagnostic hypotheses."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ValidationError

from src.schemas.diagnosis import (
    ConfidenceLevel,
    HypothesisEvaluation,
)
from src.schemas.state import DiagnosisExpertState
from src.services.config import get_settings


logger = logging.getLogger(__name__)


class ExpertAssessment(BaseModel):
    """Structured assessment produced by a diagnosis expert."""

    confidence: ConfidenceLevel = ConfidenceLevel.MODERATE
    reasoning: str

    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    recommended_tests: list[str] = Field(default_factory=list)

    notes: str | None = None

DIAGNOSIS_EXPERT_SYSTEM_PROMPT = """
You are an independent Diagnosis Expert in MedCouncil,
a multi-agent medical consultation system.

You will receive:

1. Structured patient information.
2. One diagnostic hypothesis proposed by the General Physician.

Your task is to independently evaluate whether the provided
diagnostic hypothesis is supported by the available patient information.

Do not simply agree with the General Physician.

Return the following assessment:

{
  "confidence": "low|moderate|high|very_high",

  "reasoning": "concise clinical reasoning evaluating the hypothesis",

  "supporting_evidence": [
    "patient information that supports the hypothesis"
  ],

  "contradicting_evidence": [
    "patient information that argues against the hypothesis"
  ],

  "missing_information": [
    "important information that would help evaluate the hypothesis"
  ],

  "recommended_tests": [
    "diagnostic tests that may help confirm or exclude the hypothesis"
  ],

  "notes": "additional observations or null"
}

Rules:

1. Evaluate only the diagnostic hypothesis provided to you.
2. Perform an independent assessment rather than automatically agreeing
   with the General Physician.
3. Base your evaluation only on the provided patient information.
4. Do not invent symptoms, history, examination findings, laboratory
   results, or risk factors.
5. Clearly distinguish supporting evidence from contradicting evidence.
6. Use missing_information when clinically important information is absent.
7. Recommend tests only when they are relevant to evaluating the hypothesis.
8. Do not provide treatment recommendations.
9. Do not make claims of diagnostic certainty when the evidence is incomplete.
10. Keep the reasoning concise and clinically focused.
"""

def diagnosis_expert_agent(state: DiagnosisExpertState) -> dict:
    """
    LangGraph node that independently evaluates one diagnostic hypothesis.

    Reads:
        state["patient"]
        state["hypothesis"]

    Writes:
        state["hypothesis_evaluations"]
        state["errors"]
    """

    patient = state.get("patient")
    hypothesis = state.get("hypothesis")

    if patient is None or hypothesis is None:
        return {"errors": ["Diagnosis Expert received incomplete input"]}

    logger.info(
        "diagnosis_expert_agent.start",
        extra={"condition": hypothesis.condition},
    )

    settings = get_settings()

    llm = ChatOllama(
        model=settings.ollama_model,
        temperature=0.1,
    )

    structured_llm = llm.with_structured_output(
        ExpertAssessment,
        method="json_schema",
    )

    messages = [
        SystemMessage(content=DIAGNOSIS_EXPERT_SYSTEM_PROMPT),
        HumanMessage(
            content=(f"Patient information:\n\n {patient.model_dump_json(indent=2)} \n\n Diagnostic hypothesis to evaluate:\n\n {hypothesis.model_dump_json(indent=2)}"),
        ),
    ]

    try:
        assessment = structured_llm.invoke(messages)

        evaluation = HypothesisEvaluation(
            condition=hypothesis.condition,
            confidence=assessment.confidence,
            reasoning=assessment.reasoning,
            supporting_evidence=assessment.supporting_evidence,
            contradicting_evidence=assessment.contradicting_evidence,
            missing_information=assessment.missing_information,
            recommended_tests=assessment.recommended_tests,
            notes=assessment.notes,
        )

        logger.info(
            "diagnosis_expert_agent.success",
            extra={
                "condition": hypothesis.condition,
                "confidence": evaluation.confidence.value,
            },
        )

        return {
            "hypothesis_evaluations": [
                evaluation
            ]
        }

    except ValidationError as e:
        logger.exception(
            "diagnosis_expert_agent.validation_error"
        )

        return {
            "errors": [f"Diagnosis Expert validation error for {hypothesis.condition}: {e}"]
        }

    except Exception as e:
        logger.exception(
            "diagnosis_expert_agent.error"
        )

        return {
            "errors": [f"Diagnosis Expert error for {hypothesis.condition}: {e}"]
        }
