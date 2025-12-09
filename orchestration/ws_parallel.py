from __future__ import annotations

import asyncio
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)


class AsyncWorkerPool:
    """Simple asyncio-based worker pool: schedule N workers to consume from an async queue

    Usage:
    - create an asyncio.Queue and push messages
    - provide a `handler(msg)` coroutine to process one message
    - instantiate AsyncWorkerPool(queue, handler, num_workers) and call `start()`
    """

    def __init__(self, queue: asyncio.Queue, handler: Callable[[Any], Any], num_workers: int = 4):
        self.queue = queue
        self.handler = handler
        self.num_workers = int(max(1, num_workers))
        self._workers = []
        self._running = False

    async def _worker(self, wid: int):
        logger.info('worker %s started', wid)
        while self._running:
            try:
                msg = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                await self.handler(msg)
            except Exception:
                logger.exception('worker %s failed to handle message', wid)
            finally:
                try:
                    self.queue.task_done()
                except Exception:
                    pass

    async def start(self):
        if self._running:
            return
        self._running = True
        self._workers = [asyncio.create_task(self._worker(i)) for i in range(self.num_workers)]

    async def stop(self):
        self._running = False
        # wait for workers to exit
        for w in self._workers:
            w.cancel()
        await asyncio.sleep(0)
