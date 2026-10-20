"""Main LangGraph workflow for MedCouncil."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents.input import input_agent
from src.agents.general_physician import general_physician_agent
from src.agents.diagnosis_expert import diagnosis_expert_agent
from src.agents.diagnosis_synthesis import diagnosis_synthesis_agent
from src.agents.treatment import treatment_agent
from src.agents.final_review import final_review_agent

from src.schemas.state import (
    DiagnosisExpertState,
    MedCouncilState,
)

from src.graph.hitl import request_more_information

def route_after_general_physician(state: MedCouncilState):
    """Route based on whether more patient information is required."""

    if state.get("needs_more_information", False):
        return "request_more_information"

    patient = state.get("patient")
    hypotheses = state.get("diagnostic_hypotheses", [])

    if patient is None:
        return "request_more_information"

    if not hypotheses:
        return "request_more_information"

    return [
            Send("diagnosis_expert", {"patient": patient, "hypothesis": hypothesis} )
            for hypothesis in hypotheses
        ]


def build_medcouncil_graph():
    """Build and compile the MedCouncil multi-agent workflow."""

    builder = StateGraph(MedCouncilState)

    # -------------------------
    # Nodes
    # -------------------------
    builder.add_node("input", input_agent)
    builder.add_node("general_physician", general_physician_agent)
    builder.add_node("request_more_information", request_more_information)
    builder.add_node("diagnosis_expert", diagnosis_expert_agent, input_schema=DiagnosisExpertState)
    builder.add_node("diagnosis_synthesis", diagnosis_synthesis_agent)
    builder.add_node("treatment", treatment_agent)
    builder.add_node("final_review", final_review_agent)

    # -------------------------
    # Edges
    # -------------------------
    builder.add_edge(START, "input")
    builder.add_edge("input", "general_physician")
    # GP decides:
    # 1. insufficient information -> END
    # 2. sufficient information -> dynamic diagnosis experts
    builder.add_conditional_edges("general_physician", route_after_general_physician)
    builder.add_edge("request_more_information", "input")
    builder.add_edge("diagnosis_expert", "diagnosis_synthesis")
    builder.add_edge("diagnosis_synthesis", "treatment")
    builder.add_edge("treatment", "final_review")
    builder.add_edge("final_review", END)

    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=[
            # patient.py
            ("schemas.patient", "Gender"),
            ("schemas.patient", "SeverityLevel"),
            ("schemas.patient", "Symptom"),
            ("schemas.patient", "Allergy"),
            ("schemas.patient", "Medication"),
            ("schemas.patient", "VitalSigns"),
            ("schemas.patient", "LabResult"),
            ("schemas.patient", "PatientInfo"),

            # diagnosis.py
            ("schemas.diagnosis", "ConfidenceLevel"),
            ("schemas.diagnosis", "DiagnosticHypothesis"),
            ("schemas.diagnosis", "InitialAssessment"),
            ("schemas.diagnosis", "HypothesisEvaluation"),
            ("schemas.diagnosis", "DiagnosisSynthesis"),

            # treatment.py
            ("schemas.treatment", "TreatmentRecommendation"),
            ("schemas.treatment", "TreatmentPlan"),

            # review.py
            ("schemas.review", "ReviewIssue"),
            ("schemas.review", "FinalReview"),
        ]
    )

    checkpointer = InMemorySaver(serde=serializer)
    return builder.compile(checkpointer=checkpointer)

medcouncil_graph = build_medcouncil_graph()
