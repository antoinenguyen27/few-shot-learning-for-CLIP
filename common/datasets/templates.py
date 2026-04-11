"""Dataset-specific text templates for CLIP text classifiers."""

from __future__ import annotations


TEMPLATES: dict[str, tuple[str, ...]] = {
    "eurosat": (
        "a centered satellite photo of {}.",
        "a satellite photo of {}.",
    ),
    "flowers102": (
        "a photo of a {}, a type of flower.",
        "a close-up photo of a {} flower.",
    ),
    "stanford_cars": (
        "a photo of a {}.",
        "a photo of the car model {}.",
    ),
}


def get_templates(dataset: str) -> tuple[str, ...]:
    try:
        return TEMPLATES[dataset]
    except KeyError as exc:
        known = ", ".join(sorted(TEMPLATES))
        raise KeyError(f"No templates registered for '{dataset}'. Known datasets: {known}") from exc

