"""Patient input agent for extracting structured clinical information."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from src.schemas.patient import PatientInfo
from src.schemas.state import MedCouncilState
from src.services.config import get_settings
import json


INPUT_SYSTEM_PROMPT = """
You are a patient input assistant in a multi-agent medical consultation system.

Your task is to extract structured patient information from the user's
natural-language description.

Use only information explicitly provided by the patient.
Do not infer, assume, or invent missing clinical information.

Extract the following information:

{
  "name": "patient name or null",
  "age": "integer or null",
  "gender": "male|female|other|unknown",
  "chief_complaint": "main reason for seeking medical care",

  "symptoms": [
    {
      "name": "symptom name",
      "duration_value": "integer or null",
      "duration_unit": "minutes|hours|days|weeks|months|years"
      "severity": "very_mild|mild|moderate|severe|very_severe",
      "details": "additional symptom details or null"
    }
  ],

  "medical_history": [
    "previous or existing medical condition"
  ],

  "family_history": [
    "relevant family medical condition"
  ],

  "allergies": [
    {
      "allergen": "allergen name",
      "reaction": "known reaction or null",
      "severity": "very_mild|mild|moderate|severe|very_severe"
    }
  ],

  "current_medications": [
    {
      "name": "medication name",
      "dosage": "dose or null",
      "frequency": "frequency or null",
      "started_on": "YYYY-MM-DD or null"
    }
  ],

  "vital_signs": {
    "temperature_c": "float or null",
    "heart_rate": "integer or null",
    "systolic_bp": "integer or null",
    "diastolic_bp": "integer or null",
    "respiratory_rate": "integer or null",
    "oxygen_saturation": "float or null"
  },

  "lab_results": [
    {
      "test_name": "laboratory test name",
      "value": "test result",
      "unit": "unit or null",
      "reference_range": "reference range or null",
      "abnormal": "true or false"
    }
  ]
}

Rules:

1. Preserve the patient's original clinical meaning.
2. Do not make a diagnosis.
3. Do not recommend treatment.
4. Do not invent missing information.
5. Use null for unavailable optional scalar values.
6. Use empty lists when no information is provided for list fields.
7. Use "unknown" when gender is not provided.
8. Convert explicitly stated durations into days when possible.
9. Do not mark a lab result as abnormal unless the provided information
   or reference range clearly supports it.
10. Keep the chief complaint concise and based on the patient's main concern.
"""


logger = logging.getLogger(__name__)


def input_agent(state: MedCouncilState) -> dict:
    """
    LangGraph node that converts raw patient input
    into structured PatientInfo.

    Reads:
        state["raw_input"]

    Writes:
        state["patient"]
        state["current_step"]
        state["errors"]
    """

    raw = state.get("raw_input")

    logger.info(
        "input_agent.start",
        extra={"raw_input_len": len(raw or "")},
    )
    if not raw:
        return {
            "patient": None,
            "current_step": "input",
            "errors": ["No raw input provided to input Agent"],
        }

    settings = get_settings()
    llm = ChatOllama(
        model=settings.ollama_model,
        temperature=0.1,
    )

    structured_llm = llm.with_structured_output(
        PatientInfo,
        method="json_schema",
    )

    messages = [
        SystemMessage(content=INPUT_SYSTEM_PROMPT),
        HumanMessage(content=f"Patient narrative:\n\n{raw}"),
    ]

    try:
        patient = structured_llm.invoke(messages)

        patient = patient.model_copy(
            update={
                "raw_input": raw,
            }
        )

        logger.info(
            "input_agent.success",
            extra={"patient_name": patient.name},
        )

        return {
            "patient": patient,
            "current_step": "input",
        }

    except json.JSONDecodeError as e:
            logger.exception("input_agent.json_error")
    
            return {
                "patient": None,
                "current_step": "input",
                "errors": [f"Patient json error: {e}"],
            }

    except ValidationError as e:
        logger.exception("input_agent.validation_error")

        return {
            "patient": None,
            "current_step": "input",
            "errors": [f"Patient validation error: {e}"],
        }

    except Exception as e:
        logger.exception("input_agent.error")

        return {
            "patient": None,
            "current_step": "input",
            "errors": [f"input Agent error: {e}"],
        }
