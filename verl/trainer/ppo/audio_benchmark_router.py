"""Loopback HTTP routing for audio benchmarks over multiple rollout replicas."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterable
from types import TracebackType

import aiohttp
from aiohttp import web
from multidict import CIMultiDictProxy
from yarl import URL

_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


def _forward_headers(headers: CIMultiDictProxy[str]) -> Iterable[tuple[str, str]]:
    """Keep end-to-end headers, including duplicates such as Set-Cookie."""
    excluded = _HOP_BY_HOP_HEADERS | {"host"}
    for connection in headers.getall("Connection", []):
        excluded = excluded | {name.strip().lower() for name in connection.split(",")}
    return [(name, value) for name, value in headers.items() if name.lower() not in excluded]


class AudioBenchmarkRouter:
    """Expose live rollout replicas through one local OpenAI-compatible URL.

    The HTTP server runs on its own event loop so synchronous benchmark child
    processes can use it while the caller waits in ``subprocess.run``. Requests
    select the least busy replica, with round robin ties. Each request is sent
    once; HTTP errors and response bytes pass through without generation retries.
    """

    def __init__(self, upstream_api_bases: list[str]):
        if not upstream_api_bases:
            raise ValueError("Audio benchmark routing requires at least one upstream endpoint")
        self._upstreams = [str(base).rstrip("/") for base in upstream_api_bases]
        for base in self._upstreams:
            parsed = URL(base)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.host
                or not parsed.path.endswith("/v1")
                or parsed.query_string
                or parsed.fragment
            ):
                raise ValueError(f"Expected an HTTP audio API base ending in /v1: {base!r}")
        self._inflight = [0] * len(self._upstreams)
        self._next_upstream = 0
        self._api_base: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stopping = threading.Event()
        self._serving = threading.Event()
        self._startup_error: BaseException | None = None
        self._runner: web.AppRunner | None = None
        self._session: aiohttp.ClientSession | None = None

    @property
    def api_base(self) -> str:
        if self._api_base is None:
            raise RuntimeError("Audio benchmark router is available only inside its context")
        return self._api_base

    def __enter__(self) -> AudioBenchmarkRouter:
        if self._thread is not None:
            raise RuntimeError("Audio benchmark router is already running")
        self._ready.clear()
        self._stopping.clear()
        self._serving.clear()
        self._startup_error = None
        self._inflight = [0] * len(self._upstreams)
        self._next_upstream = 0
        self._thread = threading.Thread(target=self._run, name="audio-benchmark-router", daemon=True)
        self._thread.start()
        try:
            self._ready.wait()
            if self._startup_error is not None:
                raise RuntimeError("Could not start the audio benchmark router") from self._startup_error
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        thread, loop = self._thread, self._loop
        self._stopping.set()
        if thread is not None:
            if loop is not None and self._serving.is_set():
                try:
                    # Also queue the stop if startup has completed but the
                    # background thread has not entered run_forever yet.
                    loop.call_soon_threadsafe(loop.stop)
                except RuntimeError:
                    # Startup failure may already have closed the loop.
                    pass
            thread.join()
        self._thread = None
        self._loop = None
        self._api_base = None

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            try:
                loop.run_until_complete(self._start())
                self._serving.set()
            except BaseException as error:
                self._startup_error = error
                return
            finally:
                self._ready.set()
            if self._stopping.is_set():
                loop.call_soon(loop.stop)
            loop.run_forever()
        finally:
            self._serving.clear()
            try:
                loop.run_until_complete(self._close())
            finally:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                finally:
                    loop.close()

    async def _start(self) -> None:
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0),
            timeout=aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=None),
            auto_decompress=False,
            trust_env=False,
            skip_auto_headers={"Accept", "Accept-Encoding", "Content-Type", "User-Agent"},
        )
        # Bodies contain base64 audio and routinely exceed aiohttp's 1 MiB
        # default. Both directions stream instead of buffering whole samples.
        app = web.Application(client_max_size=0)
        app.router.add_route("*", "/v1", self._forward)
        app.router.add_route("*", "/v1/{path:.*}", self._forward)
        self._runner = web.AppRunner(
            app,
            access_log=None,
            auto_decompress=False,
            handler_cancellation=True,
            # A zero aiohttp timeout means unlimited waiting; keep shutdown
            # bounded even when the benchmark child exits mid-generation.
            shutdown_timeout=0.1,
        )
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = self._runner.addresses[0][1]
        self._api_base = f"http://127.0.0.1:{port}/v1"

    async def _close(self) -> None:
        try:
            if self._runner is not None:
                await self._runner.cleanup()
        finally:
            self._runner = None
            if self._session is not None:
                await self._session.close()
            self._session = None

    def _select_upstream(self) -> int:
        # All handlers run on one loop; selection/increment has no await and
        # therefore cannot race with another handler's selection.
        minimum = min(self._inflight)
        for offset in range(len(self._upstreams)):
            index = (self._next_upstream + offset) % len(self._upstreams)
            if self._inflight[index] == minimum:
                self._inflight[index] += 1
                self._next_upstream = (index + 1) % len(self._upstreams)
                return index
        raise AssertionError("No audio benchmark upstream available")

    async def _forward(self, request: web.Request) -> web.StreamResponse:
        index = self._select_upstream()
        response: web.StreamResponse | None = None
        try:
            # raw_path retains percent escapes and the original query string.
            target = URL(self._upstreams[index] + request.raw_path[len("/v1") :], encoded=True)
            assert self._session is not None
            async with self._session.request(
                request.method,
                target,
                headers=_forward_headers(request.headers),
                data=request.content.iter_chunked(64 * 1024) if request.can_read_body else None,
                allow_redirects=False,
            ) as upstream:
                response = web.StreamResponse(
                    status=upstream.status,
                    reason=upstream.reason,
                    headers=_forward_headers(upstream.headers),
                )
                await response.prepare(request)
                async for chunk in upstream.content.iter_chunked(64 * 1024):
                    await response.write(chunk)
                await response.write_eof()
                return response
        except (aiohttp.ClientError, OSError, asyncio.TimeoutError):
            if response is not None and response.prepared:
                # Once headers are sent, replacing the status would corrupt a
                # streamed response. Closing exposes the interrupted stream.
                if request.transport is not None:
                    request.transport.close()
                return response
            return web.Response(status=502, text="Audio benchmark upstream connection failed\n")
        finally:
            self._inflight[index] -= 1
