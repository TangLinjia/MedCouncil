"""MCP server exposing the MedCouncil workflow."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langgraph.types import Command
from mcp.server import MCPServer
from pydantic import BaseModel

from src.graph.workflow import medcouncil_graph as graph


mcp = MCPServer("MedCouncil")


def to_jsonable(value: Any) -> Any:
    """Convert LangGraph/Pydantic values into MCP-safe JSON values."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")

    if isinstance(value, dict):
        return {
            key: to_jsonable(item) # recursion
            for key, item in value.items()
            if key != "__interrupt__"
        }

    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    return value


def format_result(result: dict, thread_id: str) -> dict:
    """Convert a LangGraph result into an MCP response."""

    interrupts = result.get(
        "__interrupt__",
        [],
    )

    if interrupts:
        return {
            "status": "needs_input",
            "thread_id": thread_id,
            "interrupts": [to_jsonable(interrupt.value) for interrupt in interrupts],
        }

    return {
        "status": "completed",
        "thread_id": thread_id,
        "result": to_jsonable(result),
    }


@mcp.tool()
def start_consultation(initial_state: dict[str, Any], thread_id: str | None = None) -> dict:
    """
    Start a new MedCouncil consultation.

    initial_state must use the same state structure currently
    passed to the MedCouncil LangGraph workflow.
    """

    thread_id = (thread_id or str(uuid4()))

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    result = graph.invoke(
        initial_state,
        config=config,
    )

    return format_result(
        result,
        thread_id,
    )


@mcp.tool()
def resume_consultation(thread_id: str, response: str | dict[str, Any]) -> dict:
    """
    Resume a MedCouncil consultation that paused for human input.
    """

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    result = graph.invoke(
        Command(resume=response),
        config=config,
    )

    return format_result(
        result,
        thread_id,
    )


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8001,
    )
