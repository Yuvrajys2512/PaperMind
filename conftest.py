# conftest.py — pytest configuration

import os

# ingestion.llm_client raises at import time if no provider API key is present.
# Tests that exercise modules importing it (e.g. off_topic_guard) mock the actual
# network call, so a dummy key is enough to let the import succeed in CI / any
# environment without real credentials. setdefault never overrides a real key.
os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")
