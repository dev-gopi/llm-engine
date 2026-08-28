import asyncio

from serving.orchestration import ContinuousStreamScheduler, ReloadableBackend, ReplicaPoolBackend, TokenStepScheduler
from serving.rate_limit import SQLiteRateLimiter


class Backend:
    def __init__(self, name):
        self.name = name
        self.ready = True
        self.started = False
        self.stopped = False
        self.release = asyncio.Event()
        self.release.set()

    async def startup(self):
        self.started = True

    async def shutdown(self):
        self.stopped = True

    async def generate(self, request):
        await self.release.wait()
        return self.name, request

    async def stream(self, request):
        for index in range(3):
            await asyncio.sleep(0)
            yield self.name, request, index


def test_continuous_scheduler_multiplexes_token_streams() -> None:
    async def scenario():
        scheduler = ContinuousStreamScheduler(Backend("model"), max_active=2)
        await scheduler.startup()

        async def collect(value):
            return [event async for event in scheduler.stream(value)]

        first, second = await asyncio.gather(collect("a"), collect("b"))
        await scheduler.shutdown()
        return first, second

    first, second = asyncio.run(scenario())
    assert [event[2] for event in first] == [0, 1, 2]
    assert [event[2] for event in second] == [0, 1, 2]


def test_continuous_scheduler_cancels_disconnected_stream() -> None:
    class EndlessBackend(Backend):
        def __init__(self):
            super().__init__("model")
            self.cancelled = asyncio.Event()

        async def stream(self, request):
            try:
                while True:
                    yield request
                    await asyncio.sleep(0)
            finally:
                self.cancelled.set()

    async def scenario():
        backend = EndlessBackend()
        scheduler = ContinuousStreamScheduler(
            backend, max_active=1, event_queue_size=1
        )
        await scheduler.startup()
        stream = scheduler.stream("request")
        assert await anext(stream) == "request"
        await stream.aclose()
        await asyncio.wait_for(backend.cancelled.wait(), timeout=1)
        await scheduler.shutdown()

    asyncio.run(scenario())


def test_token_step_scheduler_batches_active_sequences_and_admits_work():
    class TokenBackend:
        def __init__(self):
            self.batch_sizes = []
            self.released = []

        async def start_stream(self, request):
            return [request, 0]

        async def decode_stream_batch(self, states):
            self.batch_sizes.append(len(states))
            results = []
            for state in states:
                state[1] += 1
                results.append(((state[0], state[1]), state[1] == 3))
            return results

        def release_stream(self, state):
            self.released.append(state[0])

    async def scenario():
        backend = TokenBackend()
        scheduler = TokenStepScheduler(backend, max_active=4)
        await scheduler.startup()
        results = await asyncio.gather(*(
            asyncio.create_task(collect(scheduler.stream(value))) for value in ("a", "b", "c")
        ))
        await scheduler.shutdown()
        return backend, results

    async def collect(stream):
        return [value async for value in stream]

    backend, results = asyncio.run(scenario())
    assert all(len(values) == 3 for values in results)
    assert any(size > 1 for size in backend.batch_sizes)
    assert sorted(backend.released) == ["a", "b", "c"]


def test_replica_pool_routes_to_least_active_replica() -> None:
    async def scenario():
        first, second = Backend("one"), Backend("two")
        first.release.clear()
        pool = ReplicaPoolBackend([first, second])
        blocked = asyncio.create_task(pool.generate("first"))
        while pool.active[0] == 0:
            await asyncio.sleep(0)
        routed = await pool.generate("second")
        first.release.set()
        await blocked
        return routed

    assert asyncio.run(scenario()) == ("two", "second")


def test_reloadable_backend_warms_replacement_and_drains_inflight_request() -> None:
    async def scenario():
        old, new = Backend("old"), Backend("new")
        old.release.clear()
        wrapper = ReloadableBackend(old, version="v1")
        request = asyncio.create_task(wrapper.generate("request"))
        await asyncio.sleep(0)
        reload_task = asyncio.create_task(wrapper.reload(new, version="v2"))
        await asyncio.sleep(0)
        assert new.started and not old.stopped
        assert await wrapper.generate("new-request") == ("new", "new-request")
        old.release.set()
        assert await request == ("old", "request")
        await reload_task
        assert old.stopped and wrapper.version == "v2"

    asyncio.run(scenario())


def test_sqlite_rate_limiter_is_shared_across_instances(tmp_path) -> None:
    async def scenario():
        path = tmp_path / "limits.sqlite"
        first = SQLiteRateLimiter(path, 1)
        second = SQLiteRateLimiter(path, 1)
        return await first.allow("client"), await second.allow("client")

    assert asyncio.run(scenario()) == (True, False)
