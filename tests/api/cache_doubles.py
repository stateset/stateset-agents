"""Small Redis read-transaction double shared by cache contract tests."""


class ReadPipeline:
    def __init__(self, client):
        self.client = client
        self.key = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def get(self, key):
        self.key = key
        return self

    def pttl(self, key):
        assert key == self.key
        return self

    async def execute(self, *, raise_on_error):
        assert raise_on_error is False
        value = await self.client.get(self.key)
        return [value, getattr(self.client, "ttl_ms", 300_000)]
