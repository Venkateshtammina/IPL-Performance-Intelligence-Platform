import os
import pickle
import tempfile
import time


def load_cached_plan(cache_path, cache_key, max_age_seconds=86400):
    try:
        with open(cache_path, "rb") as cache_file:
            cache = pickle.load(cache_file)
        entry = cache.get(cache_key)
        if entry is None:
            return None
        if time.time() - entry["created_at"] > max_age_seconds:
            return None
        return entry["value"]
    except (FileNotFoundError, OSError, pickle.PickleError, EOFError, AttributeError):
        return None


def store_cached_plan(cache_path, cache_key, value, max_entries=32):
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            with open(cache_path, "rb") as cache_file:
                cache = pickle.load(cache_file)
        except (FileNotFoundError, OSError, pickle.PickleError, EOFError):
            cache = {}
        cache[cache_key] = {"created_at": time.time(), "value": value}
        ordered_entries = sorted(
            cache.items(),
            key=lambda item: item[1].get("created_at", 0),
            reverse=True,
        )[:max_entries]
        cache = dict(ordered_entries)
        file_descriptor, temporary_path = tempfile.mkstemp(
            dir=os.path.dirname(cache_path),
            prefix=".bowling-plan-",
            suffix=".tmp",
        )
        try:
            with os.fdopen(file_descriptor, "wb") as cache_file:
                pickle.dump(cache, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary_path, cache_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        return True
    except (OSError, pickle.PickleError, TypeError):
        return False
