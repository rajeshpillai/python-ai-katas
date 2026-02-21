import uuid
from datetime import datetime, timezone

_versions_store: dict[str, list[dict]] = {}


async def save_version(user_id: str, kata_id: str, code: str) -> dict:
    key = f"{user_id}:{kata_id}"
    if key not in _versions_store:
        _versions_store[key] = []
    version = {
        "id": str(uuid.uuid4()),
        "code": code,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _versions_store[key].append(version)
    return version


async def list_versions(user_id: str, kata_id: str) -> list[dict]:
    key = f"{user_id}:{kata_id}"
    return _versions_store.get(key, [])


async def get_version(user_id: str, kata_id: str, version_id: str) -> dict | None:
    versions = await list_versions(user_id, kata_id)
    return next((v for v in versions if v["id"] == version_id), None)
