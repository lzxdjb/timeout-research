"""Exact quality-policy revisions with equivalent decisions during relocation."""

LEGACY_SOURCE_SHA256 = "ecd6c2e02af8ada7d0a5ae65501d59bb4b575f3b3f3a739111ee89a613349e8a"
# Pin the target module too: arbitrary later policy edits must invalidate pools.
SOURCE_HASH_MIGRATIONS = {'88b392b4acdf819b8b67360b9253b194803bc0b8a32ab1f74b032d9f28657a6f': LEGACY_SOURCE_SHA256}


def policy_source_identity(source_sha256: str) -> str:
    return SOURCE_HASH_MIGRATIONS.get(source_sha256, source_sha256)
