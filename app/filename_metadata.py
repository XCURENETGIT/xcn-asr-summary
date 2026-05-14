from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class AudioFilenameMetadata:
    caller: str
    extension_number: str
    call_started_at: datetime
    call_ended_at: datetime


def parse_collected_audio_filename(filename: str | Path) -> AudioFilenameMetadata | None:
    stem = Path(filename).stem
    parts = [part.strip() for part in stem.split("_") if part.strip()]
    if len(parts) != 13:
        return None

    caller, extension_number = parts[0], parts[1]
    date_parts = parts[2:]
    if not caller or not extension_number or not all(part.isdigit() for part in date_parts):
        return None

    try:
        year = int(date_parts[0])
        if year < 100:
            year += 2000
        start = datetime(
            year,
            int(date_parts[1]),
            int(date_parts[2]),
            int(date_parts[3]),
            int(date_parts[4]),
            int(date_parts[5]),
        )
        end = datetime(
            year,
            int(date_parts[6]),
            int(date_parts[7]),
            int(date_parts[8]),
            int(date_parts[9]),
            int(date_parts[10]),
        )
    except ValueError:
        return None

    if end < start:
        try:
            end = end.replace(year=end.year + 1)
        except ValueError:
            return None

    return AudioFilenameMetadata(
        caller=caller,
        extension_number=extension_number,
        call_started_at=start,
        call_ended_at=end,
    )
