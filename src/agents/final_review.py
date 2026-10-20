"""Final review agent for MedCouncil."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from src.schemas.review import FinalReview
from src.schemas.state import MedCouncilState
from src.services.config import get_settings


logger = logging.getLogger(__name__)


FINAL_REVIEW_SYSTEM_PROMPT = """
You are the Final Review Agent in MedCouncil,
a multi-agent medical consultation system.

You will receive:

1. Structured patient information.
2. A synthesized preliminary diagnostic assessment.
3. A proposed treatment and management plan.

Your task is to perform a final consistency and safety review of the
diagnostic and treatment outputs.

Return the following information:

{
  "summary": "brief overall review of the case",

  "diagnosis_consistent": true,

  "treatment_appropriate": true,

  "issues": [
    {
      "category": "diagnosis|treatment|safety|missing_information|other",
      "description": "description of the identified issue",
      "recommendation": "suggested correction or null"
    }
  ],

  "missing_information": [
    "important information still needed"
  ],

  "safety_concerns": [
    "potential clinical safety concern"
  ],

  "follow_up_recommendations": [
    "recommended follow-up action"
  ],

  "final_notes": "additional review comments or null"
}

Rules:

1. Review the diagnosis and treatment together for consistency.
2. Check whether the treatment plan appropriately addresses the
   synthesized diagnosis.
3. Consider the patient's allergies, medications, medical history,
   vital signs, and laboratory results when available.
4. Identify contradictions, omissions, or unsupported assumptions.
5. Identify important safety concerns.
6. Preserve diagnostic uncertainty when the evidence is incomplete.
7. Do not invent new patient information.
8. Do not introduce a completely new diagnosis unless it is necessary
   to identify a clear safety or consistency problem.
9. Do not create an entirely new treatment plan.
10. When an issue is found, clearly explain what should be reconsidered.
11. Keep the review concise and clinically relevant.
"""


def final_review_agent(state: MedCouncilState,) -> dict:
    """
    LangGraph node that performs the final clinical review.

    Reads:
        state["patient"]
        state["diagnosis_synthesis"]
        state["treatment_plan"]

    Writes:
        state["final_review"]
        state["current_step"]
        state["errors"]
    """

    patient = state.get("patient")
    diagnosis = state.get("diagnosis_synthesis")
    treatment_plan = state.get("treatment_plan")

    logger.info("final_review_agent.start")

    if patient is None:
        return {
            "current_step": "final_review",
            "errors": ["No patient information provided to Final Review Agent"],
        }

    if diagnosis is None:
        return {
            "current_step": "final_review",
            "errors": ["No diagnosis synthesis provided to Final Review Agent"],
        }

    if treatment_plan is None:
        return {
            "current_step": "final_review",
            "errors": ["No treatment plan provided to Final Review Agent"],
        }

    settings = get_settings()

    llm = ChatOllama(
        model=settings.ollama_model,
        temperature=0.1,
    )

    structured_llm = llm.with_structured_output(
        FinalReview,
        method="json_schema",
    )

    messages = [
        SystemMessage(
            content=FINAL_REVIEW_SYSTEM_PROMPT,
        ),
        HumanMessage(
            content=(
                f"Patient information:\n\n {patient.model_dump_json(indent=2)} \n\n Diagnostic synthesis:\n\n {diagnosis.model_dump_json(indent=2)} \n\n Treatment plan:\n\n {treatment_plan.model_dump_json(indent=2)}"
            ),
        ),
    ]

    try:
        review = structured_llm.invoke(messages)

        logger.info(
            "final_review_agent.success",
            extra={
                "diagnosis_consistent":
                    review.diagnosis_consistent,
                "treatment_appropriate":
                    review.treatment_appropriate,
            },
        )

        return {
            "final_review": review,
            "current_step": "final_review",
        }

    except ValidationError as e:
        logger.exception(
            "final_review_agent.validation_error"
        )

        return {
            "current_step": "final_review",
            "errors": [f"Final review validation error: {e}"],
        }

    except Exception as e:
        logger.exception(
            "final_review_agent.error"
        )

        return {
            "current_step": "final_review",
            "errors": [f"Final Review Agent error: {e}"],
        }
