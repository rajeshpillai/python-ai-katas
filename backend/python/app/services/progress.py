_progress_store: dict[str, dict[str, str]] = {}


async def get_progress(user_id: str, track_id: str) -> dict[str, str]:
    key = f"{user_id}:{track_id}"
    return _progress_store.get(key, {})


async def save_progress(user_id: str, kata_id: str, status: str) -> None:
    key = f"{user_id}:foundational-ai"
    if key not in _progress_store:
        _progress_store[key] = {}
    _progress_store[key][kata_id] = status
