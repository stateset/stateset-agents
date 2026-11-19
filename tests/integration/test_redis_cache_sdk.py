"""Optional real Redis contracts using an isolated Unix socket and no TCP port."""

import asyncio
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from stateset_agents.api.distributed_cache import CacheConfig, HybridCache, RedisCache

pytestmark = pytest.mark.integration


class Server:
    def __init__(self, binary, path):
        self.binary = binary
        self.path = path
        self.socket = path / "redis.sock"
        self.process = None
        self.log = (path / "redis.log").open("ab")

    def start(self):
        assert self.process is None or self.process.poll() is not None
        self.process = subprocess.Popen(
            [
                self.binary,
                "--port",
                "0",
                "--unixsocket",
                str(self.socket),
                "--unixsocketperm",
                "700",
                "--save",
                "",
                "--appendonly",
                "no",
                "--dir",
                str(self.path),
                "--loglevel",
                "warning",
            ],
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 5
        while not self.socket.exists():
            if self.process.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError((self.path / "redis.log").read_text())
            time.sleep(0.01)

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


@pytest.fixture
def server():
    pytest.importorskip("redis.asyncio")
    if os.name != "posix":
        pytest.skip("The Unix-socket Redis contract requires POSIX")
    binary = shutil.which("redis-server")
    if binary is None:
        pytest.skip("Optional Redis server binary is not installed")
    # Short socket paths also work on systems with a small sockaddr_un limit.
    with tempfile.TemporaryDirectory(prefix="stateset-redis-") as directory:
        instance = Server(binary, Path(directory))
        try:
            instance.start()
            yield instance
        finally:
            instance.stop()
            instance.log.close()


def cache(server):
    return RedisCache(
        CacheConfig(
            redis_url="unix://" + str(server.socket),
            health_check_interval=0,
            redis_connect_timeout_seconds=0.5,
        )
    )


def released(pool):
    assert not pool._in_use_connections
    assert all(
        not connection.is_connected for connection in pool._available_connections
    )


@pytest.mark.asyncio
async def test_real_redis_pool_is_shared_by_connectors_and_drained_on_close(server):
    owner = cache(server)
    try:
        assert all(await asyncio.gather(*(owner.connect() for _ in range(12))))
        assert owner.get_status()["generation"] == 1
        pool = owner._client.connection_pool
        assert all(
            await asyncio.gather(*(owner.set(str(i), {"index": i}) for i in range(24)))
        )
        values = await asyncio.gather(*(owner.get(str(i)) for i in range(24)))
        assert values == [{"index": i} for i in range(24)]
    finally:
        await owner.close()
    released(pool)
    assert owner.get_status()["state"] == "closed"


@pytest.mark.asyncio
async def test_real_redis_connection_error_retires_pool_before_explicit_recovery(
    server,
):
    owner = cache(server)
    try:
        assert await owner.connect()
        pool = owner._client.connection_pool
        assert await owner.set("old", "discarded by server restart")
        server.stop()
        assert not await owner.health_check()
        assert owner.get_status()["error_type"] == "ConnectionError"
        assert owner.get_status()["state"] == "unavailable"
        released(pool)
        server.start()
        assert await owner.health_check()
        assert owner.get_status()["generation"] == 2
        assert await owner.get("old") is None
        assert await owner.set("new", "recovered")
        assert await owner.get("new") == "recovered"
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_real_redis_cancelled_command_releases_borrowed_connection_before_close(
    server, monkeypatch
):
    owner = cache(server)
    request = None
    try:
        assert await owner.connect()
        client = owner._client
        pool = client.connection_pool

        async def blocking_get(key):
            return await client.blpop(key, timeout=0)

        monkeypatch.setattr(client, "get", blocking_get)
        request = asyncio.create_task(owner.get("missing-list"))

        async def admitted():
            while not pool._in_use_connections:
                await asyncio.sleep(0.001)

        await asyncio.wait_for(admitted(), 2)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert owner.get_status()["active_operations"] == 0
        assert not pool._in_use_connections
    finally:
        if request is not None and not request.done():
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
        await owner.close()
    released(pool)


@pytest.mark.asyncio
async def test_real_hybrid_refill_expires_during_outage_with_remote_ttl(server):
    owner = HybridCache(
        CacheConfig(
            redis_url="unix://" + str(server.socket),
            health_check_interval=0,
            redis_connect_timeout_seconds=0.5,
        )
    )
    try:
        assert await owner.connect()
        client = owner._redis._client
        await client.set("stateset:short", b'"value"', px=300)
        assert await owner.get("short") == "value"
        entry = owner._memory._cache["stateset:short"]
        assert 0 < entry.monotonic_expires_at - time.monotonic() <= 0.3
        server.stop()
        assert await owner.get("short") in {"value", None}
        await asyncio.sleep(
            max(0, entry.monotonic_expires_at - time.monotonic()) + 0.01
        )
        assert await owner.get("short") is None
        assert not await owner.exists("short")
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("denied", ["multi", "exec", "pttl"])
async def test_real_hybrid_restricted_acl_can_read_without_extending_fallback(
    server, denied
):
    import redis.asyncio as redis

    url = "unix://" + str(server.socket)
    admin = redis.from_url(url)
    owner = HybridCache(CacheConfig(redis_url=url, health_check_interval=0))
    restricted = None
    try:
        await admin.set("stateset:key", b'"readable"')
        await admin.execute_command(
            "ACL",
            "SETUSER",
            "reader",
            "on",
            "nopass",
            "~stateset:*",
            "+get",
            "+ping",
            "+multi",
            "+exec",
            "+pttl",
            "-" + denied,
        )
        restricted = redis.from_url(url, username="reader")
        owner._redis._client = restricted
        owner._redis._connected = True
        await owner._memory.set("key", "stale")
        assert await owner.get("key") == "readable"
        assert await owner._memory.get("key") is None
        assert not restricted.connection_pool._in_use_connections
    finally:
        await owner.close()
        if restricted is not None:
            await restricted.aclose()
        await admin.aclose()


@pytest.mark.asyncio
async def test_real_hybrid_cancelled_snapshot_drains_pipeline_before_pool_close(
    server, monkeypatch
):
    owner = HybridCache(CacheConfig(redis_url="unix://" + str(server.socket)))
    request = closing = None
    entered, cleaning, release = (asyncio.Event() for _ in range(3))
    try:
        assert await owner.connect()
        client = owner._redis._client
        await client.set("stateset:key", b'"value"', ex=30)
        pool = client.connection_pool
        connection = await pool.get_connection()
        original = connection.read_response

        async def read_response(*args, **kwargs):
            response = await original(*args, **kwargs)
            if isinstance(response, list) and response[0] == b'"value"':
                entered.set()
                try:
                    await asyncio.Future()
                finally:
                    cleaning.set()
                    await release.wait()
            return response

        monkeypatch.setattr(connection, "read_response", read_response)
        await pool.release(connection)
        request = asyncio.create_task(owner.get("key"))
        await asyncio.wait_for(entered.wait(), 2)
        request.cancel()
        await asyncio.wait_for(cleaning.wait(), 2)
        closing = asyncio.create_task(owner.close())
        for _ in range(3):
            request.cancel()
            await asyncio.sleep(0)
            assert not request.done() and not closing.done()
            assert pool._in_use_connections
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await request
        await closing
        released(pool)
        assert await owner._memory.get("key") is None
        assert owner.get_status()["entry_ordering"]["active_keys"] == 0
    finally:
        release.set()
        if request is not None and not request.done():
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await owner.close()
