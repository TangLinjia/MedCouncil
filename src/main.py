"""Local CLI entry point for testing MedCouncil."""

from __future__ import annotations

from uuid import uuid4

from langgraph.types import Command

from graph.workflow import medcouncil_graph


def print_interrupt(interrupt_data) -> None:
    """Display follow-up questions requested by the workflow."""

    payload = interrupt_data.value

    print("\nAdditional information is required.\n")

    questions = payload.get("questions", [])

    for index, question in enumerate(questions, start=1):
        print(f"{index}. {question}")

    missing_information = payload.get(
        "missing_information",
        [],
    )

    if missing_information:
        print("\nMissing information:")
        for item in missing_information:
            print(f"- {item}")


def print_final_result(result: dict) -> None:
    """Display the final MedCouncil result."""

    print("\n" + "=" * 60)
    print("MEDCOUNCIL RESULT")
    print("=" * 60)

    diagnosis = result.get("diagnosis_synthesis")

    if diagnosis is not None:
        print("\nDiagnosis synthesis:")
        print(
            diagnosis.model_dump_json(
                indent=2,
            )
        )

    treatment = result.get("treatment_plan")

    if treatment is not None:
        print("\nTreatment plan:")
        print(
            treatment.model_dump_json(
                indent=2,
            )
        )

    review = result.get("final_review")

    if review is not None:
        print("\nFinal review:")
        print(
            review.model_dump_json(
                indent=2,
            )
        )

    errors = result.get("errors", [])

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")

def read_multiline_input() -> str:
    """Read multiline patient input until an empty line is entered."""

    print(
        "Describe the patient's condition "
        "(press Enter twice to finish):"
    )

    lines = []

    while True:
        line = input("> ")

        if not line.strip():
            break

        lines.append(line)

    return "\n".join(lines).strip()


def main() -> None:
    """Run MedCouncil from the command line."""

    print("MedCouncil")
    print("-" * 40)

    raw_input = read_multiline_input()

    if not raw_input:
        print("No patient information provided.")
        return

    # Each patient consultation gets its own thread.
    thread_id = str(uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    # ---------------------------------------------------------
    # Initial graph execution
    # ---------------------------------------------------------

    result = medcouncil_graph.invoke(
        {
            "raw_input": raw_input,
            "hypothesis_evaluations": [],
            "errors": [],
        },
        config=config,
    )

    # ---------------------------------------------------------
    # HITL loop
    # ---------------------------------------------------------

    while result.get("__interrupt__"):
        interrupts = result["__interrupt__"]

        # Our current workflow only creates one HITL interrupt
        # at a time.
        interrupt_data = interrupts[0]

        print_interrupt(interrupt_data)

        additional_information = read_multiline_input(
            "\nPlease provide the requested information "
            "(press Enter twice to finish):"
        )

        if not additional_information:
            print(
                "\nAdditional information is required "
                "to continue the assessment."
            )
            return

        # Resume the SAME graph thread.
        result = medcouncil_graph.invoke(
            Command(
                resume=additional_information,
            ),
            config=config,
        )

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    print_final_result(result)


if __name__ == "__main__":
    main()
