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

Gateway authentication
----------------------

Gateway auth is controlled via environment variables such as:

- `API_REQUIRE_AUTH`
- `API_KEYS`
- `API_JWT_SECRET` (required in production)

Security event retention
------------------------

``SecurityMonitor`` retains the most recent 10,000 events by default. Pass a
positive ``max_events`` value to choose another in-memory limit. Recent-event
queries and anomaly detection operate on the retained events.
