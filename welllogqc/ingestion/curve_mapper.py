"""
curve_mapper.py
---------------
Resolves raw vendor LAS curve mnemonics to canonical curve names using
configured alias definitions.
"""

from pathlib import Path
from typing import Any
import yaml


class CurveMapper:
    """Maps vendor LAS curve mnemonics to canonical petrophysical curve names.

    Reads alias definitions from a YAML configuration file where each canonical
    curve name maps to an ordered list of aliases. For each canonical group,
    aliases are searched top-to-bottom and the first matching mnemonic present
    in the LAS file is assigned.

    Attributes:
        alias_path: Path to the YAML file containing alias mappings.
        aliases: Dictionary of canonical curve names mapped to ordered alias lists.
    """

    def __init__(self, alias_path: str | Path | None = None) -> None:
        """Initialises the CurveMapper with alias configurations.

        Args:
            alias_path: Optional path to curve_aliases.yaml. If omitted,
                defaults to welllogqc/config/curve_aliases.yaml relative to
                this package.

        Raises:
            FileNotFoundError: If the specified or default alias YAML file cannot be found.
            ValueError: If the YAML file is empty or cannot be parsed as a dictionary.
        """
        if alias_path is None:
            default_path = Path(__file__).resolve().parent.parent / "config" / "curve_aliases.yaml"
            if not default_path.exists():
                fallback = Path("welllogqc/config/curve_aliases.yaml").resolve()
                if fallback.exists():
                    default_path = fallback
                else:
                    raise FileNotFoundError(
                        f"Curve aliases YAML configuration file not found at: {default_path}"
                    )
            self.alias_path = default_path
        else:
            self.alias_path = Path(alias_path).resolve()
            if not self.alias_path.exists():
                raise FileNotFoundError(
                    f"Specified curve aliases YAML file does not exist: {self.alias_path}"
                )

        with open(self.alias_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid alias configuration in {self.alias_path}: root must be a dict.")

        self.aliases: dict[str, list[str]] = {}
        for canonical, alias_list in data.items():
            if isinstance(alias_list, list):
                self.aliases[str(canonical).strip().upper()] = [
                    str(a).strip().upper() for a in alias_list
                ]
            else:
                self.aliases[str(canonical).strip().upper()] = [str(canonical).strip().upper()]

    def map(self, mnemonics: list[str]) -> dict[str, str | None]:
        """Maps a list of raw LAS mnemonics to canonical curve names.

        For each canonical group defined in the alias configuration, the method searches
        its aliases top-to-bottom. The first alias matching any mnemonic in the input list
        is mapped to that canonical name. Any mnemonics that do not match or are secondary
        aliases for an already-mapped canonical group receive None.

        Args:
            mnemonics: List of raw curve mnemonic strings from a LAS file.

        Returns:
            Dictionary mapping each raw mnemonic (original casing preserved) to its
            canonical name string, or None if unmapped.

        Raises:
            ValueError: If mnemonics is not an iterable collection.
        """
        if not hasattr(mnemonics, "__iter__"):
            raise ValueError(f"Expected iterable of mnemonics, got {type(mnemonics).__name__}")

        mapping: dict[str, str | None] = {m: None for m in mnemonics}
        assigned_mnemonics: set[str] = set()

        # Build lookup from upper-case mnemonic to original mnemonic
        upper_to_raw: dict[str, str] = {}
        for m in mnemonics:
            u = str(m).strip().upper()
            if u not in upper_to_raw:
                upper_to_raw[u] = m

        for canonical, alias_list in self.aliases.items():
            for alias in alias_list:
                if alias in upper_to_raw:
                    raw_m = upper_to_raw[alias]
                    if raw_m not in assigned_mnemonics:
                        mapping[raw_m] = canonical
                        assigned_mnemonics.add(raw_m)
                        break

        return mapping

    def unmapped(self, mnemonics: list[str]) -> list[str]:
        """Identifies raw mnemonics that have no canonical mapping.

        Args:
            mnemonics: List of raw curve mnemonic strings.

        Returns:
            List of mnemonic strings for which no canonical mapping could be resolved.

        Raises:
            ValueError: If mnemonics is not an iterable collection.
        """
        mapping = self.map(mnemonics)
        return [m for m in mnemonics if mapping.get(m) is None]
