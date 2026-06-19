"""Slot-based World Cup 2026 knockout bracket helpers.

This module is intentionally slot-based. It does not reseed teams by rating or
group-stage performance after qualification.
"""

from __future__ import annotations

import pandas as pd


class MissingThirdPlaceMappingError(ValueError):
    """Raised when an advancing third-place combination is not mapped yet."""


# Editable Round of 32 bracket slots.
#
# These slots follow the published 2026 Round of 32 match order. Third-place
# labels include the eligible source groups for that bracket position.
ROUND_OF_32_SLOT_MATCHES = [
    ("2A", "2B"),
    ("1E", "3_ABCDF"),
    ("1F", "2C"),
    ("1C", "2F"),
    ("1I", "3_CDFGH"),
    ("2E", "2I"),
    ("1A", "3_CEFHI"),
    ("1L", "3_EHIJK"),
    ("1D", "3_BEFIJ"),
    ("1G", "3_AEHIJ"),
    ("2K", "2L"),
    ("1H", "2J"),
    ("1B", "3_EFGIJ"),
    ("1J", "2H"),
    ("1K", "3_DEIJL"),
    ("2D", "2G"),
]


ROUND_OF_32_MATCH_NUMBERS = list(range(73, 89))


ROUND_OF_16_MATCHES = [
    (89, 74, 77),
    (90, 73, 75),
    (91, 76, 78),
    (92, 79, 80),
    (93, 83, 84),
    (94, 81, 82),
    (95, 86, 88),
    (96, 85, 87),
]


QUARTERFINAL_MATCHES = [
    (97, 89, 90),
    (98, 93, 94),
    (99, 91, 92),
    (100, 95, 96),
]


SEMIFINAL_MATCHES = [
    (101, 97, 98),
    (102, 99, 100),
]


FINAL_MATCH = (104, 101, 102)


# Third-place mappings for known official/provided advancing third-place group
# combinations. Add official FIFA mappings here as they become available.
#
# Values should map third-place slot labels such as "3_CEFHI" to concrete
# finisher labels such as "3C". Empty by default because the dynamic fallback
# below already respects the allowed group letters encoded in each slot label.
THIRD_PLACE_MAPPING = {}


def _standings_records(standings: object) -> list[dict[str, object]]:
    """Convert a standings object to records."""
    if isinstance(standings, pd.DataFrame):
        return standings.to_dict("records")

    if isinstance(standings, list):
        return standings

    raise TypeError("group standings must be a pandas DataFrame or list of records")


def get_group_finishers(group_results: dict) -> dict[str, str]:
    """Return labels such as 1A, 2A, and 3A mapped to team names."""
    finishers: dict[str, str] = {}

    for group, result in group_results.items():
        if group == "advancing_third_place_groups":
            continue

        standings = _standings_records(result["standings"])

        if len(standings) < 3:
            raise ValueError(f"group {group} does not have at least three finishers")

        group_letter = str(group).upper()

        finishers[f"1{group_letter}"] = str(standings[0]["team"])
        finishers[f"2{group_letter}"] = str(standings[1]["team"])
        finishers[f"3{group_letter}"] = str(standings[2]["team"])

    return finishers


def _resolve_slot(slot: str, third_place_slot_mapping: dict[str, str]) -> str:
    """Resolve a fixed bracket slot to a concrete group finisher label."""
    if slot.startswith("3_"):
        if slot not in third_place_slot_mapping:
            raise MissingThirdPlaceMappingError(
                f"third-place slot {slot} is missing from the mapping"
            )
        return third_place_slot_mapping[slot]

    return slot


def _third_place_slots() -> list[str]:
    """Return third-place slot labels in Round of 32 bracket order."""
    return [
        slot
        for match in ROUND_OF_32_SLOT_MATCHES
        for slot in match
        if slot.startswith("3_")
    ]


def _allowed_groups_for_third_place_slot(slot: str) -> set[str]:
    """Return group letters allowed by a third-place slot label."""
    if not slot.startswith("3_"):
        raise ValueError(f"{slot} is not a third-place slot")
    return set(slot.split("_", 1)[1])


def _build_provisional_third_place_mapping(third_place_groups: tuple[str, ...]) -> dict[str, str]:
    """Build a deterministic official-slot-compatible mapping.

    This is not the full official FIFA lookup table. It keeps simulations
    moving while respecting each slot's allowed third-place group letters.
    """
    if len(third_place_groups) != len(_third_place_slots()):
        raise MissingThirdPlaceMappingError(
            "Expected exactly eight advancing third-place groups, got "
            f"{len(third_place_groups)}: {third_place_groups}"
        )

    available_groups = set(third_place_groups)
    mapping: dict[str, str] = {}
    slots = _third_place_slots()

    def assign(slot_index: int) -> bool:
        if slot_index == len(slots):
            return True

        slot = slots[slot_index]
        allowed_groups = _allowed_groups_for_third_place_slot(slot)
        candidate_groups = sorted(
            group for group in available_groups if group in allowed_groups
        )

        for group in candidate_groups:
            available_groups.remove(group)
            mapping[slot] = f"3{group}"
            if assign(slot_index + 1):
                return True
            del mapping[slot]
            available_groups.add(group)

        return False

    if not assign(0):
        raise MissingThirdPlaceMappingError(
            "Could not assign advancing third-place groups "
            f"{third_place_groups} to the available Round of 32 third-place slots. "
            "Add an explicit mapping in THIRD_PLACE_MAPPING."
        )

    return mapping


def build_round_of_32_matches(
    group_finishers: dict[str, str],
    advancing_third_place_groups: list[str],
) -> list[tuple[str, str]]:
    """Build Round of 32 team matchups from fixed slot labels."""
    third_place_key = tuple(sorted(group.upper() for group in advancing_third_place_groups))

    third_place_slot_mapping = THIRD_PLACE_MAPPING.get(third_place_key)
    if third_place_slot_mapping is None:
        third_place_slot_mapping = _build_provisional_third_place_mapping(third_place_key)

    matches: list[tuple[str, str]] = []

    for slot_a, slot_b in ROUND_OF_32_SLOT_MATCHES:
        resolved_a = _resolve_slot(slot_a, third_place_slot_mapping)
        resolved_b = _resolve_slot(slot_b, third_place_slot_mapping)

        try:
            team_a = group_finishers[resolved_a]
            team_b = group_finishers[resolved_b]
        except KeyError as exc:
            raise ValueError(
                f"bracket slot {exc.args[0]} has no matching group finisher"
            ) from exc

        matches.append((team_a, team_b))

    return matches
