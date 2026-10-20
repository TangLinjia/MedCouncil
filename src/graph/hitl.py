"""Human-in-the-loop logic for MedCouncil."""

from __future__ import annotations

from langgraph.types import interrupt

from src.schemas.state import MedCouncilState


def request_more_information(state: MedCouncilState) -> dict:
    """
    Pause the workflow and request additional information from the user.

    Reads:
        state["raw_input"]
        state["missing_information"]
        state["follow_up_questions"]

    Writes:
        state["raw_input"]
        state["needs_more_information"]
        state["current_step"]

    The graph pauses at interrupt() and resumes when the user
    provides additional information through Command(resume=...).
    """

    questions = state.get("follow_up_questions", [])

    missing_information = state.get(
        "missing_information", [])

    # Pause the graph and expose these questions to the caller.
    additional_input = interrupt(
        {
            "type": "additional_information_required",
            "message": (
                "Additional patient information is required "
                "before diagnostic evaluation can continue."
            ),
            "missing_information": missing_information,
            "questions": questions,
        }
    )

    # When the graph is resumed, interrupt() returns the user's response.
    previous_input = state.get("raw_input", "")

    # Usually the resume value will simply be a string.
    # Supporting a dict also makes this easier to connect to an API later.
    if isinstance(additional_input, dict):
        additional_text = str(
            additional_input.get(
                "response",
                additional_input.get(
                    "additional_information",
                    "",
                ),
            )
        )
    else:
        additional_text = str(additional_input)

    combined_input = (
        f"{previous_input.strip()}\n\n Additional patient information:\n {additional_text.strip()}"
    ).strip()

    return {
        "raw_input": combined_input,
        "needs_more_information": False,
        "current_step": "additional_information_received",
    }
