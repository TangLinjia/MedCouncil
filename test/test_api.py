import json

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

from src.api.app import create_app
from src.graph.hitl import request_more_information
from src.schemas.state import MedCouncilState


@pytest.fixture
def graph():
    builder = StateGraph(MedCouncilState)
    builder.add_node('ask', request_more_information)
    builder.add_edge(START, 'ask')
    builder.add_edge('ask', END)
    return builder.compile(checkpointer=InMemorySaver())


@pytest.fixture
def client(graph):
    with TestClient(create_app(graph)) as client:
        yield client


def test_start_resume_and_unknown_thread(client):
    started = client.post('/consultations', json={'raw_input': '咳嗽'}).json()
    assert started['status'] == 'needs_input'
    resumed = client.post(f"/consultations/{started['thread_id']}/resume", json={'response': '三天'}).json()
    assert resumed['status'] == 'completed'
    assert '咳嗽' in resumed['result']['raw_input']
    assert '三天' in resumed['result']['raw_input']
    assert client.post('/consultations/missing/resume', json={'response': '三天'}).status_code == 404
    assert client.post(f"/consultations/{started['thread_id']}/resume", json={'response': '重复'}).status_code == 409


def test_stream_interrupt_and_resume(client):
    response = client.post('/consultations/stream', json={'raw_input': '头痛'})
    assert response.headers['content-type'].startswith('text/event-stream')
    frames = response.text.strip().split('\n\n')
    payload = json.loads(frames[-1].split('data: ')[1])
    assert 'event: needs_input' in frames[-1]
    response = client.post(f"/consultations/{payload['thread_id']}/resume/stream", json={'response': '两天'})
    assert 'event: node_completed' in response.text
    assert 'event: completed' in response.text
    assert '两天' in response.text


@pytest.mark.parametrize('value', ['', '   '])
def test_blank_input(client, value):
    assert client.post('/consultations', json={'raw_input': value}).status_code == 422
    assert client.post('/consultations/x/resume', json={'response': value}).status_code == 422


def test_sessions_are_isolated(client):
    first = client.post('/consultations', json={'raw_input': '患者甲'}).json()
    second = client.post('/consultations', json={'raw_input': '患者乙'}).json()
    result = client.post(f"/consultations/{first['thread_id']}/resume", json={'response': '甲的补充'}).json()
    assert '患者乙' not in result['result']['raw_input']
    assert first['thread_id'] != second['thread_id']


def test_stream_unknown_thread(client):
    response = client.post('/consultations/missing/resume/stream', json={'response': '补充'})
    assert response.status_code == 404


def test_workflow_failure_is_reported_without_exception_details():
    builder = StateGraph(MedCouncilState)

    def fail(state):
        raise RuntimeError('private internal detail')

    builder.add_node('fail', fail)
    builder.add_edge(START, 'fail')
    builder.add_edge('fail', END)
    graph = builder.compile(checkpointer=InMemorySaver())
    with TestClient(create_app(graph)) as client:
        response = client.post('/consultations', json={'raw_input': '测试'})
        assert response.status_code == 500
        assert 'private internal detail' not in response.text
        response = client.post('/consultations/stream', json={'raw_input': '测试'})
        assert 'event: error' in response.text
        assert 'event: completed' not in response.text
        assert 'private internal detail' not in response.text


def test_async_execution_keeps_other_sessions_responsive_and_rejects_double_resume():
    import asyncio
    import httpx

    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def wait_for_release(state):
            entered.set()
            await release.wait()
            return {'current_step': 'finished'}

        builder = StateGraph(MedCouncilState)
        builder.add_node('ask', request_more_information)
        builder.add_node('wait', wait_for_release)
        builder.add_edge(START, 'ask')
        builder.add_edge('ask', 'wait')
        builder.add_edge('wait', END)
        app = create_app(builder.compile(checkpointer=InMemorySaver()))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            first = (await client.post('/consultations', json={'raw_input': '甲'})).json()
            url = f"/consultations/{first['thread_id']}/resume"
            pending = asyncio.create_task(client.post(url, json={'response': '补充'}))
            try:
                await asyncio.wait_for(entered.wait(), timeout=5)
                duplicate = await client.post(url, json={'response': '重复'})
                assert duplicate.status_code == 409
                other = await asyncio.wait_for(client.post('/consultations', json={'raw_input': '乙'}), timeout=5)
                assert other.json()['status'] == 'needs_input'
                release.set()
                assert (await pending).json()['status'] == 'completed'
            finally:
                release.set()
                await pending

    asyncio.run(run())
