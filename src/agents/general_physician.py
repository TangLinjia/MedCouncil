"""General Physician agent for specialist routing."""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from src.schemas.state import MedCouncilState
from src.schemas.diagnosis import InitialAssessment
from src.services.config import get_settings


logger = logging.getLogger(__name__)


GENERAL_PHYSICIAN_SYSTEM_PROMPT = """
You are the General Physician coordinator in MedCouncil, a multi-agent medical consultation system.

Your task is to review the structured patient information and decide whether there is enough information to generate 2 to 3 clinically meaningful preliminary diagnostic hypotheses for further evaluation by independent diagnosis expert agents.

You are responsible for broad initial clinical reasoning and differential diagnosis generation.

Return the following information:

{
"sufficient_information": true,

"missing_information": [
"important information that is required before meaningful diagnostic hypotheses can be generated"
],

"follow_up_questions": [
"concise question to ask the patient when essential information is missing"
],

"hypotheses": [
{
"condition": "possible medical condition",
"reasoning": "brief explanation of why this condition should be investigated",
"priority": "high|medium|low"
}
],

"overall_reasoning": "brief summary of the clinical reasoning behind the assessment"
}

Information sufficiency rules:

1. "sufficient_information" means there is enough information to generate 2 to 3 reasonable preliminary diagnostic hypotheses for further expert evaluation.

2. "sufficient_information" does NOT mean there is enough information to make a definitive diagnosis.

3. Do not require a complete medical history, physical examination, vital signs, laboratory results, imaging results, or specialist testing before generating diagnostic hypotheses.

4. Missing tests, vital signs, or additional history may create diagnostic uncertainty, but they should not automatically make the information insufficient.

5. Set "sufficient_information" to false only when the patient's information is too vague or incomplete to generate meaningful diagnostic hypotheses.

Examples of insufficient information include:

* "I feel sick."
* "Something hurts."
* "I don't feel well."
* A symptom is mentioned without enough context to understand its nature, duration, location, severity, or associated features.

Examples of sufficient information include:

* A specific chief complaint with clinically useful characteristics.
* Information about symptom duration, triggers, relieving factors, associated symptoms, or relevant medical history.
* Enough context to identify at least 2 plausible diagnostic possibilities.

If sufficient_information is false:

1. Set "sufficient_information" to false.
2. Leave "hypotheses" empty.
3. Identify only the clinically important missing information that prevents meaningful hypothesis generation.
4. Generate concise follow-up questions for the patient.
5. Ask only questions that would materially help generate diagnostic hypotheses.
6. Do not ask for laboratory tests, imaging, ECG results, or other investigations unless the patient has already indicated that such results exist and clarification is necessary.

If sufficient_information is true:

1. Set "sufficient_information" to true.
2. Generate at least 2 and no more than 3 diagnostic hypotheses.
3. "missing_information" may be empty.
4. "follow_up_questions" should normally be empty.
5. Do not request additional information merely because more clinical data would improve diagnostic certainty.
6. Further missing evidence, diagnostic tests, and unresolved questions should be handled by downstream diagnosis expert agents.

Diagnostic hypothesis rules:

1. Base all hypotheses only on the provided patient information.
2. Consider the chief complaint, symptoms, symptom duration, severity, triggers, relieving factors, associated symptoms, medical history, family history, current medications, allergies, vital signs, and laboratory results when available.
3. Select clinically meaningful and distinct diagnostic possibilities rather than multiple variations of the same diagnosis.
4. Prioritize hypotheses that are both plausible and important to investigate.
5. Include serious conditions when the patient's presentation reasonably supports considering them.
6. Do not introduce serious or emergency diagnoses without supporting clinical features.
7. Do not make a final diagnosis.
8. Do not provide treatment recommendations.
9. Do not invent symptoms, physical examination findings, medical history, risk factors, laboratory values, imaging findings, or other patient information.
10. When information is uncertain, explicitly preserve that uncertainty rather than filling in missing details.
11. Keep each hypothesis reasoning concise and clinically relevant.
12. The hypotheses will be evaluated independently by downstream diagnosis expert agents.

Priority guidance:

* "high": the hypothesis is especially important or strongly supported enough to warrant prioritized expert evaluation.
* "medium": the hypothesis is reasonably plausible and should remain part of the differential.
* "low": the hypothesis is less supported but still clinically relevant enough to investigate.

The goal of this stage is not to establish a definitive diagnosis. The goal is to determine whether the patient information is sufficient for meaningful differential diagnosis generation and, when it is, to produce 2 to 3 useful hypotheses for downstream expert evaluation.
"""


def general_physician_agent(state: MedCouncilState) -> dict:
    """
    LangGraph node that generates initial diagnostic hypotheses.

    Reads:
        state["patient"]

    Writes:
        state["selected_specialists"]
        state["current_step"]
        state["errors"]
    """

    patient = state.get("patient")

    logger.info("general_physician_agent.start")

    if patient is None:
        return {
            "selected_specialists": [],
            "current_step": "general_physician",
            "errors": ["No patient information provided to General Physician Agent"],
        }

    settings = get_settings()

    llm = ChatOllama(
        model=settings.ollama_model,
        temperature=0.1,
    )

    structured_llm = llm.with_structured_output(
        InitialAssessment,
        method="json_schema",
    )

    messages = [
        SystemMessage(content=GENERAL_PHYSICIAN_SYSTEM_PROMPT,),
        HumanMessage(content=(f"Patient information:\n\n {patient.model_dump_json(indent=2, exclude={'created_at', 'raw_input'},)}"),
        ),
    ]

    try:
        print("GP 实际收到的信息:", messages[1].content)
        assessment = structured_llm.invoke(messages)

        logger.info(
            "general_physician_agent.success",
            extra={
                "hypotheses": [
                    hypothesis.condition
                    for hypothesis in assessment.hypotheses
                ]
            },
        )

        return {
            "needs_more_information": not assessment.sufficient_information,
            "missing_information": assessment.missing_information,
            "follow_up_questions": assessment.follow_up_questions,
            "diagnostic_hypotheses": assessment.hypotheses,
            "gp_reasoning": assessment.overall_reasoning,
            "current_step": "general_physician",
        }

    except ValidationError as e:
        logger.exception(
            "general_physician_agent.validation_error"
        )

        return {
            "current_step": "general_physician",
            "errors": [f"General Physician validation error: {e}"],
        }

    except Exception as e:
        logger.exception(
            "general_physician_agent.error"
        )

        return {
            "current_step": "general_physician",
            "errors": [f"General Physician Agent error: {e}"],
        }
