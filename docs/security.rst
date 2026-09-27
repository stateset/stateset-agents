Security
========

Secret configuration
--------------------

``SecureConfig.set_secret`` encrypts values by default. Set
``CONFIG_ENCRYPTION_KEY`` and install ``cryptography`` before storing or reading
encrypted values. The operation raises ``RuntimeError`` if either requirement
is missing; it never substitutes base64 encoding for encryption. Use
``encrypt=False`` only when plaintext storage is explicitly intended. Secrets
are held in memory; ``save_config`` writes their names but not their values.

Gateway auth is controlled via environment variables such as:

- `API_REQUIRE_AUTH`
- `API_KEYS`
- `API_JWT_SECRET` (required in production)
