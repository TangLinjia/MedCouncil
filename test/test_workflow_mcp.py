"""Test MedCouncil workflow through MCP."""

from __future__ import annotations
import anyio
from mcp import Client
import json


async def main():
    async with Client(
        "http://127.0.0.1:8001/mcp"
    ) as client:
        tools = await client.list_tools()

        print(
            "[MCP TOOLS]"
        )

        for tool in tools.tools:
            print(
                tool.name
            )

        result = await client.call_tool(
            "start_consultation",
            {
                "initial_state": {
                    "raw_input": "女"
                }
            },
        )

        print(
            "[IS ERROR]",
            result.is_error,
        )

        print(
            "[CONTENT]",
            result.content,
        )

        data = json.loads(
            result.content[0].text
        )

        print(
            "[START CONSULTATION RESULT]"
        )

        print(
            "[DATA]",
            data,
        )

        print(
            "[STATUS]",
            data["status"],
        )

        print(
            "[THREAD ID]",
            data["thread_id"],
        )

        if data["status"] == "needs_input":
            thread_id = data["thread_id"]

            resumed = await client.call_tool(
                "resume_consultation",
                {
                    "thread_id": thread_id,
                    "response": {
                        "thread_id": thread_id,
                        "response": "干咳, 无呼吸困难, 无胸痛, 无慢性病史, 无已知药物过敏"
                    },
                },
            )

            print(
                "[RESUME IS ERROR]",
                resumed.is_error,
            )

            print(
                "[RESUME CONTENT]",
                resumed.content,
            )

            if resumed.content:
                resumed_data = json.loads(
                    resumed.content[0].text
                )

                print(
                    "[RESUME DATA]",
                    resumed_data,
                )

                print(
                    "[RESUME STATUS]",
                    resumed_data["status"],
                )


if __name__ == "__main__":
    anyio.run(main)
