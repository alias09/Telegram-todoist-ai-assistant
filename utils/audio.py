from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Tuple


def ensure_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

async def download_to_temp_async(file_obj, suffix: str) -> str:
    """Download a Telegram File object to a temp path (async), trying multiple method signatures.

    Supports python-telegram-bot v20/v21 variations.
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    last_err: Exception | None = None
    # Try several method signatures to maximize compatibility
    for method_name, kwargs in [
        ("download_to_drive", {"custom_path": path}),
        ("download_to_drive", {"out": path}),
        ("download", {"custom_path": path}),
        ("download", {"out": path}),
    ]:
        try:
            method = getattr(file_obj, method_name, None)
            if method is None:
                continue
            await method(**kwargs)
            return path
        except Exception as e:
            last_err = e
            continue
    # If we reach here, all attempts failed
    if last_err:
        try:
            os.remove(path)
        except Exception:
            pass
        raise last_err
    return path


def ogg_to_wav(ogg_path: str) -> Tuple[bool, str]:
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg", "-y", "-i", ogg_path,
        "-ac", "1", "-ar", "16000", "-vn",
        wav_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, wav_path
    except subprocess.CalledProcessError:
        try:
            os.remove(wav_path)
        except Exception:
            pass
        return False, ""
