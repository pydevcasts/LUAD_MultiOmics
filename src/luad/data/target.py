"""Target definition, cleaning, and grouping utilities.

This module does not train models. It only helps confirm and clean target labels.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

DEFAULT_STAGE_MAPPING: Dict[str, str] = {
    "STAGE IA": "STAGE I",
    "STAGE IB": "STAGE I",
    "STAGE I": "STAGE I",
    "STAGE IIA": "STAGE II",
    "STAGE IIB": "STAGE II",
    "STAGE II": "STAGE II",
    "STAGE IIIA": "STAGE III",
    "STAGE IIIB": "STAGE III",
    "STAGE IIIC": "STAGE III",
    "STAGE III": "STAGE III",
    "STAGE IV": "STAGE IV",
    "STAGE IVA": "STAGE IV",
    "STAGE IVB": "STAGE IV",
}

MISSING_LIKE_VALUES = {
    "",
    "NA",
    "N/A",
    "NULL",
    "NONE",
    "NOT AVAILABLE",
    "[NOT AVAILABLE]",
    "UNKNOWN",
    "UKNOWN",
}


def normalize_categorical_value(
    value: Any,
    uppercase: bool = True,
    strip: bool = True,
) -> Optional[str]:
    """
    Normalize a categorical value.

    - Converts NaN to None
    - Strips whitespace
    - Converts to uppercase if requested
    - Converts missing-like strings to None
    """
    if pd.isna(value):
        return None

    text_value = str(value)

    if strip:
        text_value = text_value.strip()

    if uppercase:
        text_value = text_value.upper()

    if text_value in MISSING_LIKE_VALUES:
        return None

    if text_value == "":
        return None

    return text_value


def normalize_mapping_keys(
    mapping: Dict[str, str],
    uppercase: bool = True,
    strip: bool = True,
) -> Dict[str, str]:
    """
    Normalize mapping keys using the same rules as values.
    """
    normalized_mapping: Dict[str, str] = {}

    for key, value in mapping.items():
        normalized_key = normalize_categorical_value(
            value=key,
            uppercase=uppercase,
            strip=strip,
        )

        if normalized_key is None:
            continue

        normalized_mapping[normalized_key] = str(value)

    return normalized_mapping


def map_target_series(
    series: pd.Series,
    mapping: Optional[Dict[str, str]] = None,
    uppercase: bool = True,
    strip: bool = True,
) -> pd.Series:
    """
    Map raw target values to cleaned/grouped target values.
    """
    if mapping is None:
        mapping = DEFAULT_STAGE_MAPPING

    normalized_mapping = normalize_mapping_keys(
        mapping=mapping,
        uppercase=uppercase,
        strip=strip,
    )

    def _map_value(value: Any) -> Optional[str]:
        normalized_value = normalize_categorical_value(
            value=value,
            uppercase=uppercase,
            strip=strip,
        )

        if normalized_value is None:
            return None

        return normalized_mapping.get(normalized_value, normalized_value)

    return series.map(_map_value)


def summarize_categorical_series(series: pd.Series) -> Dict[str, Any]:
    """
    Summarize a categorical target series.
    """
    value_counts = series.value_counts(dropna=True).sort_values(ascending=False)

    return {
        "missing_count": int(series.isna().sum()),
        "missing_ratio": float(series.isna().mean()),
        "unique_count": int(series.nunique(dropna=True)),
        "counts": {str(key): int(count) for key, count in value_counts.items()},
    }