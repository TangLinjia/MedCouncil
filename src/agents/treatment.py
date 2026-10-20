"""Treatment planning agent for MedCouncil."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ValidationError

from src.schemas.state import MedCouncilState
from src.schemas.treatment import TreatmentPlan
from src.services.config import get_settings
from src.tools.guideline_search import guideline_search
from src.tools.drug_safety_check import drug_safety_check


logger = logging.getLogger(__name__)

class TreatmentDraft(BaseModel):
    """Intermediate treatment plan before drug safety review."""

    proposed_medications: list[str] = Field(
        default_factory=list
    )

    treatment_plan: TreatmentPlan


TREATMENT_SYSTEM_PROMPT = """
You are the Treatment Planning Agent in MedCouncil,
a multi-agent medical consultation system.

You will receive:

1. Structured patient information.
2. A synthesized preliminary diagnostic assessment.

Your task is to generate a structured treatment and management plan
based on the available information.

You have access to the guideline_search tool.

Before generating a treatment or management plan, you must use the
guideline_search tool at least once to retrieve relevant clinical
guideline evidence.

Construct the search query using the patient's clinical context and the
diagnostic synthesis.

After reviewing the retrieved evidence, you may call guideline_search
again if additional guideline information is needed.

Do not claim that guideline evidence was retrieved unless you actually
used the guideline_search tool.

Return the following information:

{
  "primary_condition": "condition the treatment plan is addressing",

  "recommendations": [
    {
      "treatment": "treatment or management recommendation",
      "purpose": "clinical purpose or null",
      "instructions": "general instructions or null",
      "precautions": [
        "important precaution"
      ]
    }
  ],

  "lifestyle_advice": [
    "relevant lifestyle or supportive-care recommendation"
  ],

  "follow_up": "recommended follow-up plan or null",

  "warning_signs": [
    "symptoms or changes that warrant urgent medical attention"
  ],

  "additional_tests": [
    "additional diagnostic or monitoring test"
  ],

  "notes": "additional treatment considerations or null"
}

Rules:

1. Base the plan only on the provided patient information and
   synthesized diagnosis.
2. Address the primary diagnosis identified in the diagnostic synthesis.
3. Consider the patient's allergies, current medications, medical history,
   vital signs, and laboratory findings when available.
4. Do not invent patient information.
5. Do not present the preliminary diagnosis as definitively confirmed.
6. Avoid overly specific prescription dosing unless explicitly supported
   by the provided clinical information.
7. Include relevant precautions and contraindication considerations.
8. Include warning signs when appropriate.
9. Include additional tests when further diagnostic clarification or
   monitoring is still needed.
10. If diagnostic uncertainty is substantial, favor further evaluation
    and follow-up over unsupported treatment recommendations.
11. Keep the plan concise and clinically relevant.
12. When guideline evidence is retrieved, use it as supporting evidence
    rather than blindly copying it.
13. You must retrieve clinical guideline evidence before finalizing
    treatment recommendations.
14. Do not recommend specific medications, doses, or management steps
    based only on internal model knowledge when guideline evidence has
    not been retrieved.
15. Do not repeat information already present in recommendations,
    precautions, warning_signs, additional_tests, or follow_up.
16. Do not repeat the same statement in different wording.
"""


def treatment_agent(state: MedCouncilState) -> dict:
    """
    LangGraph node that generates a treatment plan.

    Reads:
        state["patient"]
        state["diagnosis_synthesis"]

    Writes:
        state["treatment_plan"]
        state["current_step"]
        state["errors"]
    """

    patient = state.get("patient")
    diagnosis = state.get("diagnosis_synthesis")

    logger.info("treatment_agent.start")

    if patient is None:
        return {
            "current_step": "treatment",
            "errors": ["No patient information provided to Treatment Agent"],
        }

    if diagnosis is None:
        return {
            "current_step": "treatment",
            "errors": ["No diagnosis synthesis provided to Treatment Agent"],
        }

    settings = get_settings()

    llm = ChatOllama(
        model=settings.ollama_model,
        temperature=0.1,
    )

    tool_llm = llm.bind_tools(
        [guideline_search],
    )

    draft_llm = llm.with_structured_output(
        TreatmentDraft,
        method="json_schema",
    )

    structured_llm = llm.with_structured_output(
        TreatmentPlan,
        method="json_schema",
    )

    messages = [
        SystemMessage(
            content=TREATMENT_SYSTEM_PROMPT,
        ),
        HumanMessage(
            content=(
                f"Patient information:\n\n {patient.model_dump_json(indent=2)} \n\n Diagnostic synthesis:\n\n {diagnosis.model_dump_json(indent=2)}"
            ),
        ),
    ]

    try:
        tool_messages = list(messages)
        guideline_used = False

        for _ in range(3):
            response = tool_llm.invoke(tool_messages)
            tool_messages.append(response)

            print(
                f"[TREATMENT] tool calls: {response.tool_calls}"
            )

            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:
                if tool_call["name"] != "guideline_search":
                    continue

                guideline_used = True

                logger.info(
                    "treatment_agent.guideline_search",
                    extra={
                        "query": tool_call["args"].get("query"),
                    },
                )

                tool_result = guideline_search.invoke(
                    tool_call["args"],
                )

                print(
                    f"[RAG RESULT]\n{tool_result}"
                )

                tool_messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"],
                    )
                )

        tool_messages.append(
            HumanMessage(
                content=(
                    "Using the patient information, diagnostic synthesis, "
                    "and any retrieved guideline evidence above, generate "
                    "a preliminary treatment plan. Also list every medication "
                    "you intend to recommend in proposed_medications. Use generic "
                    "medication names only, without dosage or instructions."
                ),
            ),
        )

        draft = draft_llm.invoke(tool_messages)

        print(
            f"[TREATMENT DRAFT]\n{draft.model_dump_json(indent=2)}"
        )

        current_medications = [
            medication.name
            for medication in patient.current_medications
            if medication.name
        ]

        if draft.proposed_medications:
            safety_result = drug_safety_check.invoke(
                {
                    "proposed_medications": draft.proposed_medications,
                    "current_medications": current_medications,
                    "allergies": patient.allergies,
                }
            )
        else:
            safety_result = (
                '{"status": "not_applicable", '
                '"alerts": [], '
                '"message": "No medications were proposed."}'
            )

        print(
            f"[DRUG SAFETY RESULT]\n{safety_result}"
        )

        final_messages = list(tool_messages)

        final_messages.append(
            HumanMessage(
                content=(
                    f"Preliminary treatment plan:\n\n"
                    f"{draft.treatment_plan.model_dump_json(indent=2)}\n\n"
                    f"Drug safety check:\n\n"
                    f"{safety_result}\n\n"
                    "Generate the final structured treatment plan. "
                    "Revise or remove medication recommendations when the "
                    "drug safety check identifies a relevant conflict. "
                    "Do not claim that a medication is safe merely because "
                    "no conflict was found in the limited local rule set."
                ),
            ),
        )

        treatment_plan = structured_llm.invoke(final_messages)

        print(
            f"[FINAL TREATMENT PLAN]\n{treatment_plan.model_dump_json(indent=2)}"
        )

        logger.info(
            "treatment_agent.success",
            extra={
                "primary_condition":
                    treatment_plan.primary_condition,
                "guideline_used":
                    guideline_used,
            },
        )

        return {
            "treatment_plan": treatment_plan,
            "current_step": "treatment",
        }

    except ValidationError as e:
        logger.exception(
            "treatment_agent.validation_error"
        )

        return {
            "current_step": "treatment",
            "errors": [f"Treatment plan validation error: {e}"],
        }

    except Exception as e:
        logger.exception(
            "treatment_agent.error"
        )

        return {
            "current_step": "treatment",
            "errors": [f"Treatment Agent error: {e}"],
        }
