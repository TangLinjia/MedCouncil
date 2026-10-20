"""Asynchronous HTTP and SSE access to the consultation workflow."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


class StartRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    raw_input: str = Field(min_length=1, max_length=50000)

    @field_validator('raw_input')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Patient information must not be blank')
        return value


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    response: str = Field(min_length=1, max_length=50000)

    @field_validator('response')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Additional information must not be blank')
        return value


def format_result(values, thread_id, interrupts=()):
    interrupts = interrupts or values.get('__interrupt__', ())
    if interrupts:
        return {'status': 'needs_input', 'thread_id': thread_id,
                'interrupts': jsonable_encoder([item.value for item in interrupts])}
    result = {key: value for key, value in values.items() if key != '__interrupt__'}
    return {'status': 'completed', 'thread_id': thread_id,
            'result': jsonable_encoder(result)}


def create_app(graph=None):
    app = FastAPI(title='MedCouncil')
    # Same-process state and locks: use one worker with the in-memory checkpointer.
    locks = {}

    def get_graph():
        nonlocal graph
        if graph is None:
            from src.graph.workflow import medcouncil_graph
            graph = medcouncil_graph
        return graph

    @asynccontextmanager
    async def reserve(thread_id, resume):
        lock = locks.setdefault(thread_id, asyncio.Lock())
        if lock.locked():
            raise HTTPException(409, 'Consultation is already running')
        await lock.acquire()
        try:
            workflow = get_graph()
            config = {'configurable': {'thread_id': thread_id}}
            if resume:
                snapshot = await workflow.aget_state(config)
                if not snapshot.values:
                    raise HTTPException(404, 'Unknown consultation')
                if not any(task.interrupts for task in snapshot.tasks):
                    raise HTTPException(409, 'Consultation is not awaiting input')
            yield workflow, config
        finally:
            lock.release()
            if locks.get(thread_id) is lock:
                del locks[thread_id]

    async def execute(value, thread_id, resume=False):
        async with reserve(thread_id, resume) as (workflow, config):
            try:
                result = await workflow.ainvoke(value, config=config)
                return format_result(result, thread_id)
            except Exception:
                logger.exception('Consultation execution failed')
                raise HTTPException(500, 'Consultation execution failed')

    async def stream(value, thread_id, resume=False):
        # Reserve before returning HTTP 200 so missing/busy sessions get HTTP errors.
        reservation = reserve(thread_id, resume)
        workflow, config = await reservation.__aenter__()

        async def events():
            def frame(event, payload):
                data = json.dumps(jsonable_encoder(payload), ensure_ascii=False)
                return f'event: {event}\ndata: {data}\n\n'

            try:
                yield frame('started', {'thread_id': thread_id})
                async for update in workflow.astream(value, config=config, stream_mode='updates'):
                    for node, output in update.items():
                        if node != '__interrupt__':
                            yield frame('node_completed', {'thread_id': thread_id,
                                                          'node': node, 'output': output})
                snapshot = await workflow.aget_state(config)
                interrupts = [item for task in snapshot.tasks for item in task.interrupts]
                result = format_result(snapshot.values, thread_id, interrupts)
                yield frame(result['status'], result)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception('Consultation stream failed')
                yield frame('error', {'thread_id': thread_id, 'message': 'Consultation execution failed'})
            finally:
                await reservation.__aexit__(None, None, None)

        return StreamingResponse(events(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.post('/consultations')
    async def start(body: StartRequest):
        return await execute({'raw_input': body.raw_input, 'errors': [],
                              'hypothesis_evaluations': []}, str(uuid4()))

    @app.post('/consultations/stream')
    async def start_stream(body: StartRequest):
        return await stream({'raw_input': body.raw_input, 'errors': [],
                             'hypothesis_evaluations': []}, str(uuid4()))

    @app.post('/consultations/{thread_id}/resume')
    async def resume(thread_id: str, body: ResumeRequest):
        return await execute(Command(resume=body.response), thread_id, resume=True)

    @app.post('/consultations/{thread_id}/resume/stream')
    async def resume_stream(thread_id: str, body: ResumeRequest):
        return await stream(Command(resume=body.response), thread_id, resume=True)

    return app


app = create_app()
