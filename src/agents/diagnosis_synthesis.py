"""Diagnosis synthesis agent."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from src.schemas.diagnosis import DiagnosisSynthesis
from src.schemas.state import MedCouncilState
from src.services.config import get_settings


logger = logging.getLogger(__name__)


DIAGNOSIS_SYNTHESIS_SYSTEM_PROMPT = """
You are the Diagnosis Synthesis Agent in MedCouncil,
a multi-agent medical consultation system.

You will receive:

1. Structured patient information.
2. Multiple independent evaluations of diagnostic hypotheses.

Your task is to compare the expert evaluations and synthesize them
into a single preliminary diagnostic assessment.

Return the following information:

{
  "primary_diagnosis": "most supported diagnosis or null",

  "confidence": "low|moderate|high|very_high",

  "reasoning": "brief explanation of how the expert evidence was synthesized",

  "differential_diagnoses": [
    "important alternative diagnosis"
  ],

  "recommended_tests": [
    "tests that may help clarify the diagnosis"
  ],

  "unresolved_questions": [
    "important missing clinical information"
  ]
}

Rules:

1. Base the synthesis only on the patient information and expert evaluations.
2. Compare supporting and contradicting evidence across all hypotheses.
3. Do not automatically choose the hypothesis with the highest confidence.
4. Consider the quality and completeness of the evidence.
5. Preserve clinically relevant alternative diagnoses.
6. Merge duplicate or overlapping recommended tests.
7. Highlight important unresolved information.
8. Do not invent new symptoms, laboratory results, history, or findings.
9. Do not provide treatment recommendations.
10. If the evidence is insufficient to identify a primary diagnosis,
    set primary_diagnosis to null.
11. Treat the result as a preliminary diagnostic synthesis,
    not a definitive diagnosis.
"""

def diagnosis_synthesis_agent(state: MedCouncilState) -> dict:
    """
    LangGraph node that synthesizes expert diagnosis evaluations.

    Reads:
        state["patient"]
        state["hypothesis_evaluations"]

    Writes:
        state["diagnosis_synthesis"]
        state["current_step"]
        state["errors"]
    """

    patient = state.get("patient")
    evaluations = state.get(
        "hypothesis_evaluations",
        [],
    )

    logger.info(
        "diagnosis_synthesis_agent.start",
        extra={
            "evaluation_count": len(evaluations),
        },
    )

    if patient is None:
        return {
            "current_step": "diagnosis_synthesis",
            "errors": ["No patient information provided to Diagnosis Synthesis Agent"],
        }

    if not evaluations:
        return {
            "current_step": "diagnosis_synthesis",
            "errors": ["No expert evaluations provided to Diagnosis Synthesis Agent"],
        }

    settings = get_settings()

    llm = ChatOllama(
        model=settings.ollama_model,
        temperature=0.1,
    )

    structured_llm = llm.with_structured_output(
        DiagnosisSynthesis,
        method="json_schema",
    )

    evaluations_json = json.dumps(
        [
            evaluation.model_dump(mode="json")
            for evaluation in evaluations
        ],
        indent=2,
    )

    messages = [
        SystemMessage(
            content=DIAGNOSIS_SYNTHESIS_SYSTEM_PROMPT,
        ),
        HumanMessage(
            content=(
                f"Patient information:\n\n {patient.model_dump_json(indent=2)} \n\n Independent expert evaluations:\n\n {evaluations_json}"
            ),
        ),
    ]

    try:
        synthesis = structured_llm.invoke(messages)

        logger.info(
            "diagnosis_synthesis_agent.success",
            extra={
                "primary_diagnosis":
                    synthesis.primary_diagnosis,
            },
        )

        return {
            "diagnosis_synthesis": synthesis,
            "current_step": "diagnosis_synthesis",
        }

    except ValidationError as e:
        logger.exception(
            "diagnosis_synthesis_agent.validation_error"
        )

        return {
            "current_step": "diagnosis_synthesis",
            "errors": [f"Diagnosis synthesis validation error: {e}"],
        }

    except Exception as e:
        logger.exception(
            "diagnosis_synthesis_agent.error"
        )

        return {
            "current_step": "diagnosis_synthesis",
            "errors": [f"Diagnosis Synthesis Agent error: {e}"],
        }
