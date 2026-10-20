import asyncio
import json
from pathlib import Path
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from src.graph.hitl import request_more_information

CASES = json.loads(Path('data/cases/missing_information.json').read_text())


class CaseState(TypedDict, total=False):
    raw_input: str
    missing_information: list[str]
    follow_up_questions: list[str]
    needs_more_information: bool
    current_step: str
    rounds: int


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['id'])
def test_repeated_interrupt_preserves_history_and_completed_nodes(case):
    async def run():
        calls = []
        target = len(case['supplements'])

        def prepare(state):
            calls.append('prepare')
            return {'rounds': 0}

        def advance(state):
            return {'rounds': state['rounds'] + 1}

        builder = StateGraph(CaseState)
        builder.add_node('prepare', prepare)
        builder.add_node('ask', request_more_information)
        builder.add_node('advance', advance)
        builder.add_edge(START, 'prepare')
        builder.add_edge('prepare', 'ask')
        builder.add_edge('ask', 'advance')
        builder.add_conditional_edges('advance', lambda state: 'ask' if state['rounds'] < target else END)
        graph = builder.compile(checkpointer=InMemorySaver())
        config = {'configurable': {'thread_id': case['id']}}
        result = await graph.ainvoke({'raw_input': case['initial_input'],
                                     'missing_information': case['expected_follow_up_topics'],
                                     'follow_up_questions': ['请补充相关信息']}, config)
        for index, supplement in enumerate(case['supplements']):
            assert result['__interrupt__'][0].value['missing_information'] == case['expected_follow_up_topics']
            result = await graph.ainvoke(Command(resume=supplement), config)
            assert case['initial_input'] in result['raw_input']
            for previous in case['supplements'][:index + 1]:
                assert previous in result['raw_input']
        assert not result.get('__interrupt__')
        assert result['rounds'] == target
        assert calls == ['prepare']

    asyncio.run(run())
