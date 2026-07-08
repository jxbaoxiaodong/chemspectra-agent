"""
Deterministic local cross-validation rules for ChemSpectra Agent.

This module is a compact reimplementation of selected FTIR.fun production
reasoning ideas. The production tree is read-only; this file has no dependency
on Django, databases, or FTIR.fun internals.

Rule sources read from the FTIR.fun production codebase (read-only study,
reimplemented here without any production import):
- search_reasoning.py: evaluate_feature_rules, detect_background_flags,
  assess_quality_flags, build_library_axis_assessment
- material_family.py: POLYMER_NAME_PATTERNS and organic functional group names
- verdict.py: layer_agreement material-family/group compatibility
- fastapi_server/ftir_service.py: _build_uncertainty lead-score gap wording
"""

from __future__ import annotations

import re
from typing import Any


# Source: fastapi_server/ftir_service.py::_build_uncertainty treats gap <= 0.02
# as close enough to require fingerprint-region review.
CLOSE_LEAD_GAP_MAX = 0.02

# Source: IMPROVEMENT_PLAN.md. The production verdict threshold is config-driven
# and not exposed by the REST response, so the hackathon agent uses this named
# review threshold instead of embedding the number inline.
MATCH_QUALITY_MIN_SCORE = 0.75

# Source: user execution decision on 2026-07-03: when Top-1 < 0.85,
# Qwen must make the direction arbitration visible as a separate tool call.
ENTITY_STRONG_MATCH_MIN_SCORE = 0.85

# Direction arbitration uses majority support, not a hidden model score.
DIRECTION_MIN_WEIGHTED_SHARE = 0.50
DIRECTION_MIN_SUPPORTING_CANDIDATES = 2

# Source: IMPROVEMENT_PLAN.md deterministic confidence formula.
MIN_CROSS_VALIDATION_MULTIPLIER = 0.60

# Source: IMPROVEMENT_PLAN.md peak coverage check.
MIN_PEAK_EXPLANATION_COVERAGE = 0.60

# Source: search_reasoning.py::assess_quality_flags checks whether enough
# spectral signal exists before quality assessment. The API exposes peaks, not
# y-values, so this compact version uses a named minimum peak count.
MIN_PEAKS_FOR_BASIC_QUALITY_CHECK = 3

# Source: search_reasoning.py::detect_background_flags bands, adapted from
# intensity-based y-value checks to peak-position-only advisories.
BACKGROUND_BANDS_CM1 = {
    "co2": (2310.0, 2360.0),
    "moisture": (3400.0, 3700.0),
    "aliphatic": (2800.0, 2960.0),
}


# Source: material_family.py::POLYMER_NAME_PATTERNS, copied as regex data.
POLYMER_NAME_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("polyamide_imide", (r"\bpolyamide[\s-]*imide\b",)),
    ("polyether_ketone", (r"\bpoly[\s-]*ether[\s-]*ketone\b", r"\bpeek\b", r"\bpekk\b")),
    ("polyether_sulfone", (r"\bpoly[\s-]*ether[\s-]*sulfone\b", r"\bpes\b", r"\bpe[\s-]*sulfone\b")),
    ("polybenzimidazole", (r"\bpolybenzimidazole\b", r"\bpbi\b")),
    ("polyphenylene_sulfide", (r"\bpolyphenylene[\s-]*sulfide\b", r"\bpps\b")),
    ("polyphenylene_oxide", (r"\bpolyphenylene[\s-]*oxide\b", r"\bppo\b")),
    ("polyphosphazene", (r"\bpolyphosphazene\b",)),
    ("polysulfone_imide", (r"\bpoly[\s-]*sulfone[\s-]*imide\b",)),
    ("polyimide", (r"\bpolyimide\b", r"\bpi\b")),
    ("polyamide", (r"\bpolyamide\b", r"\bnylon[\s-]?\d+\b", r"\bnylon\b")),
    ("polyurethane", (r"\bpolyurethane\b", r"\bpu\b")),
    ("polyurea", (r"\bpolyurea\b",)),
    ("polyacetal", (r"\bpolyacetal\b", r"\bpolyoxymethylene\b", r"\bpom\b")),
    ("polycarbonate", (r"\bpolycarbonate\b", r"\bpc\b")),
    ("polyester", (r"\bpolyester\b", r"\bpet\b", r"\bpbt\b", r"\bpolycaprolactone\b", r"\bpcl\b")),
    ("polyarylate", (r"\bpolyarylate\b",)),
    ("alkyd_resin", (r"\balkyd\b",)),
    ("epoxy_resin", (r"\bepoxy\b",)),
    ("phenolic_resin", (r"\bphenolic\b", r"\bphenol[\s-]*formaldehyde\b", r"\bbakelite\b")),
    ("amino_resin", (r"\bmelamine[\s-]*formaldehyde\b", r"\burea[\s-]*formaldehyde\b", r"\bamino resin\b")),
    ("cellulose_derivative", (r"\bcellulose\b", r"\bcellulosic\b", r"\bcellulose acetate\b", r"\bnitrocellulose\b")),
    ("fluoropolymer", (r"\bpolytetrafluoroethylene\b", r"\bptfe\b", r"\bfluoropolymer\b", r"\bpolyvinylidene fluoride\b", r"\bpvdf\b", r"\bpolyvinyl fluoride\b")),
    ("polysiloxane", (r"\bpolysiloxane\b", r"\bsilicone rubber\b", r"\bpdms\b", r"\bpolydimethylsiloxane\b")),
    ("polysulfide", (r"\bpolysulfide\b",)),
    ("polysulfone", (r"\bpolysulfone\b", r"\bpsu\b")),
    ("polyacrylonitrile", (r"\bpolyacrylonitrile\b", r"\bpan\b")),
    ("polyacrylamide", (r"\bpolyacrylamide\b", r"\bpam\b")),
    ("polyacrylate", (r"\bpolyacrylate\b", r"\bpoly\(.*acrylate", r"\bacrylic resin\b")),
    ("polyvinyl_butyral", (r"\bpolyvinyl butyral\b", r"\bpvb\b")),
    ("polyvinyl_alcohol", (r"\bpolyvinyl alcohol\b", r"\bpva\b")),
    ("polyvinyl_acetate", (r"\bpolyvinyl acetate\b", r"\bpvac\b", r"\bpvAc\b")),
    ("polyvinyl_ether", (r"\bpolyvinyl ether\b",)),
    ("polyvinyl_chloride", (r"\bpolyvinyl chloride\b", r"\bpvc\b", r"\bvinyl chloride polymer\b")),
    ("polyether", (r"\bpolyether\b", r"\bpolyethylene oxide\b", r"\bpeo\b", r"\bpolyethylene glycol\b", r"\bpeg\b", r"\bpolypropylene glycol\b")),
    # styrenic groups all styrene-aromatic materials (PS, ABS, SAN, SBR, and
    # bare "styrene"/"styrene resin"/"K glue" library names) into one direction.
    # FTIR shares the same aromatic ring + backbone signature across these, so
    # splitting them into competing families wrongly dilutes the direction share.
    ("styrenic", (
        r"\bpolystyrene\b", r"\bps\b",
        r"\bacrylonitrile butadiene styrene\b", r"\babs\b", r"\bsan\b",
        r"\bstyrene\b", r"\bstyrenic\b",
        r"\bstyrene[\s-].*(copolymer|resin|rubber)\b",
        r"\bstyrene-butadiene\b", r"\bsbr\b", r"\bk glue\b",
    )),
    ("polyolefin", (r"\bpolyethylene\b", r"\bhigh density polyethylene\b", r"\bhdpe\b", r"\bldpe\b", r"\bpolypropylene\b", r"\bpp\b", r"\bpolyolefin\b")),
    ("polydiene", (r"\bpolybutadiene\b", r"\bpolyisoprene\b", r"\bpolydiene\b", r"\brubber\b")),
    ("polyoxazolidone", (r"\bpolyoxazolidone\b",)),
)

# Additional non-polymer families used by the demo sample and by common FTIR
# library hits. Names are derived from material_family.py::ORGANIC_SMARTS group
# labels plus the existing amino_resin polymer rule.
#
# protein_biopolymer is placed first so proteinaceous library hits (gelatin,
# collagen, silk, enzymes) are classified as a single biopolymer direction
# instead of leaking into polyamide/amide or staying unmapped. FTIR cannot
# distinguish individual proteins, so the correct resolution for these samples
# is a material direction, not a specific entity.
ORGANIC_NAME_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("protein_biopolymer", (
        r"\bgelatin\b", r"\bgelatine\b", r"\bcollagen\b", r"\bkeratin\b",
        r"\belastin\b", r"\bfibroin\b", r"\bsericin\b", r"\balbumin\b",
        r"\bcasein\b", r"\bmucin\b", r"\bpepsin\b", r"\btrypsin\b",
        r"\bchymotrypsin\b", r"\bprotease\b", r"\benzyme\b",
        r"\bleather\b", r"\bsilk\b", r"\bwool\b", r"\bfeather\b",
        r"\bprotein\b", r"\bpeptide\b", r"\bpolypeptide\b",
    )),
    ("melamine_triazine", (r"\bmelamine\b", r"\btriazine\b", r"\bdiamino\b", r"\bformamide\b")),
    ("ester", (r"\bester\b", r"\bacrylate\b", r"\bmethacrylate\b", r"\b\w+oate\b")),
    ("carboxylic_acid", (r"\bcarboxylic acid\b", r"\w+carboxylic\b", r"\bdicarboxylic\b", r"\bfatty acid\b")),
    ("amide", (r"\bamide\b", r"\bformamide\b", r"\burea\b")),
    ("carbonate_ester", (r"\bcarbonate\b",)),
    ("sulfone", (r"\bsulfone\b", r"\bsulfonyl\b")),
    ("epoxide", (r"\bepoxide\b", r"\bepoxy\b")),
    ("nitrile", (r"\bnitrile\b", r"\bcyano\b")),
)

MATERIAL_NAME_PATTERNS = POLYMER_NAME_PATTERNS + ORGANIC_NAME_PATTERNS


# Functional group expectations are a compact, API-local equivalent of the
# production feature-rule + layer-agreement checks.
FAMILY_EXPECTED_GROUPS: dict[str, dict[str, Any]] = {
    "polyester": {
        "required": ("ester", "carbonyl"),
        "supporting": ("c_o", "aromatic"),
        "reason": "Polyester requires ester C=O and C-O support.",
    },
    "polyamide": {
        "required": ("amide",),
        "supporting": ("n_h", "carbonyl", "c_n"),
        "reason": "Polyamide requires amide evidence, usually N-H/C=O/C-N bands.",
    },
    "polyurethane": {
        "required": ("urethane", "carbonyl"),
        "supporting": ("n_h", "c_o", "isocyanate"),
        "reason": "Polyurethane requires urethane carbonyl and N-H/C-O support.",
    },
    "polyolefin": {
        "required": ("aliphatic_ch",),
        "supporting": ("methylene", "methyl"),
        "reason": "Polyolefins require strong aliphatic C-H/CH2/CH3 evidence.",
    },
    "styrenic": {
        "required": ("aromatic",),
        "supporting": ("aliphatic_ch", "phenyl"),
        "reason": "Styrenic materials (PS/ABS/SAN/SBR) require aromatic ring evidence.",
    },
    "polycarbonate": {
        "required": ("carbonate", "carbonyl"),
        "supporting": ("aromatic", "c_o"),
        "reason": "Polycarbonate requires carbonate C=O and C-O support.",
    },
    "polyacrylate": {
        "required": ("ester", "carbonyl"),
        "supporting": ("c_o", "aliphatic_ch"),
        "reason": "Acrylic polymers require ester carbonyl evidence.",
    },
    "polyacrylonitrile": {
        "required": ("nitrile",),
        "supporting": ("aliphatic_ch",),
        "reason": "PAN requires nitrile C#N evidence.",
    },
    "polyvinyl_chloride": {
        "required": ("c_cl",),
        "supporting": ("aliphatic_ch",),
        "reason": "PVC requires C-Cl evidence plus polymer C-H bands.",
    },
    "polysiloxane": {
        "required": ("siloxane",),
        "supporting": ("si_c", "aliphatic_ch"),
        "reason": "Silicone materials require Si-O-Si/siloxane evidence.",
    },
    "cellulose_derivative": {
        "required": ("hydroxyl", "c_o"),
        "supporting": ("aliphatic_ch",),
        "reason": "Cellulosics require O-H and C-O carbohydrate evidence.",
    },
    "epoxy_resin": {
        "required": ("epoxide", "c_o"),
        "supporting": ("aromatic", "hydroxyl"),
        "reason": "Epoxy systems require epoxide/C-O evidence, often aromatic/O-H support.",
    },
    "phenolic_resin": {
        "required": ("phenol", "aromatic"),
        "supporting": ("hydroxyl", "c_o"),
        "reason": "Phenolic resins require phenolic/aromatic evidence.",
    },
    "amino_resin": {
        "required": ("amine", "triazine"),
        "supporting": ("urea", "formaldehyde", "c_n", "n_h"),
        "reason": "Melamine/urea amino resins require amine/triazine or urea-type evidence.",
    },
    "melamine_triazine": {
        "required": ("amine", "triazine"),
        "supporting": ("amide", "c_n", "n_h", "carbonyl"),
        "reason": "Melamine/triazine derivatives require triazine ring and amine/amide evidence.",
    },
    "protein_biopolymer": {
        "required": ("amide",),
        "supporting": ("n_h", "carbonyl", "hydroxyl", "c_n"),
        "reason": "Proteinaceous materials require amide I/II (C=O/N-H) backbone evidence.",
    },
    "amide": {
        "required": ("amide",),
        "supporting": ("carbonyl", "n_h", "c_n"),
        "reason": "Amide compounds require amide C=O/N-H/C-N evidence.",
    },
    "ester": {
        "required": ("ester", "carbonyl"),
        "supporting": ("c_o",),
        "reason": "Ester compounds require ester C=O/C-O evidence.",
    },
    "carboxylic_acid": {
        "required": ("carbonyl",),
        "supporting": ("hydroxyl", "c_o"),
        "reason": "Carboxylic acids require C=O plus broad O-H evidence.",
    },
    "carbonate_ester": {
        "required": ("carbonate", "carbonyl"),
        "supporting": ("c_o",),
        "reason": "Carbonates require carbonate C=O/C-O evidence.",
    },
    "sulfone": {
        "required": ("sulfone",),
        "supporting": ("s_o",),
        "reason": "Sulfones require S=O evidence.",
    },
    "epoxide": {
        "required": ("epoxide",),
        "supporting": ("c_o",),
        "reason": "Epoxide compounds require epoxide/C-O evidence.",
    },
    "nitrile": {
        "required": ("nitrile",),
        "supporting": (),
        "reason": "Nitriles require C#N evidence.",
    },
}

FAMILY_FORBIDDEN_GROUPS: dict[str, tuple[str, ...]] = {
    "polyolefin": ("carbonyl", "ester", "amide", "urethane", "carbonate", "nitrile", "sulfone"),
    # styrenic includes ABS, whose acrylonitrile contributes a real C#N band,
    # so nitrile is intentionally NOT forbidden here.
    "styrenic": ("ester", "amide", "urethane", "carbonate"),
    "polyvinyl_chloride": ("ester", "amide", "urethane", "carbonate", "nitrile"),
    "polysiloxane": ("ester", "amide", "urethane", "carbonate", "nitrile", "sulfone"),
    "fluoropolymer": ("carbonyl", "ester", "amide", "urethane", "carbonate", "nitrile"),
    "polyacrylonitrile": ("ester", "amide", "urethane", "carbonate"),
    "nitrile": ("ester", "amide", "urethane", "carbonate"),
}

GROUP_ALIASES: dict[str, tuple[str, ...]] = {
    "aliphatic_ch": ("aliphatic", "c-h", "c h", "ch2", "ch3", "methylene", "methyl", "alkyl"),
    "amide": ("amide", "formamide", "urea", "carbamide"),
    "amine": ("amine", "amino", "n-h", "n h", "nh2", "primary amine", "secondary amine"),
    "aromatic": ("aromatic", "benzene", "phenyl", "aryl"),
    "c_cl": ("c-cl", "c cl", "chloride", "chloro"),
    "c_n": ("c-n", "c n", "c=n", "c = n", "triazine"),
    "c_o": ("c-o", "c o", "ether", "alcohol", "ester c-o"),
    "carbonate": ("carbonate", "o-c=o", "o c o"),
    "carbonyl": ("c=o", "c = o", "carbonyl", "aldehyde", "ketone", "amide i"),
    "epoxide": ("epoxide", "epoxy", "oxirane"),
    "ester": ("ester", "acrylate", "methacrylate"),
    "formaldehyde": ("formaldehyde", "methylene bridge"),
    "hydroxyl": ("o-h", "o h", "hydroxyl", "alcohol", "phenol"),
    "isocyanate": ("isocyanate", "n=c=o", "n c o"),
    "methyl": ("methyl", "ch3"),
    "methylene": ("methylene", "ch2"),
    "n_h": ("n-h", "n h", "nh", "amine", "amide"),
    "nitrile": ("nitrile", "c#n", "cyano"),
    "phenol": ("phenol", "phenolic"),
    "phenyl": ("phenyl", "benzene", "aromatic"),
    "s_o": ("s=o", "s = o", "sulfonyl"),
    "si_c": ("si-c", "si c"),
    "siloxane": ("si-o", "si o", "siloxane", "si-o-si", "silicone"),
    "sulfone": ("sulfone", "sulfonyl", "so2"),
    "triazine": ("triazine", "melamine", "1,3,5-triazine"),
    "urea": ("urea", "carbamide"),
    "urethane": ("urethane", "carbamate"),
}


def cross_validate(search_result: dict[str, Any]) -> dict[str, Any]:
    """Run deterministic local checks on a search_library response."""
    matches = _list(search_result.get("matches"))
    peak_explanations = _list(search_result.get("peak_explanations"))
    evidence = _list(search_result.get("evidence"))
    peak_positions = _extract_peak_positions(peak_explanations, evidence)
    evidence_text = _normalize_text(" ".join(str(item) for item in evidence + peak_explanations))

    top_match = matches[0] if matches else {}
    top_name = str(top_match.get("name") or "").strip()
    top_score = _match_score(top_match)
    detected_family = classify_material_family(top_name)
    observed_groups = sorted(_detect_observed_groups(evidence_text))

    checks: list[dict[str, Any]] = []
    checks.append(_score_gap_check(matches))
    checks.append(_match_quality_check(top_score))
    checks.append(_family_group_check(detected_family, observed_groups, top_name))
    checks.append(_hard_forbidden_group_check(detected_family, observed_groups, top_name))
    checks.append(_peak_coverage_check(peak_explanations))
    checks.append(_background_check(peak_positions))
    checks.append(_basic_quality_check(peak_positions, peak_explanations))

    applicable_checks = [item for item in checks if item.get("applicable", True)]
    passed = sum(1 for item in applicable_checks if item.get("passed") is True)
    total = len(applicable_checks)
    cv_ratio = (passed / total) if total else 0.0
    confidence_multiplier = round(
        MIN_CROSS_VALIDATION_MULTIPLIER
        + (1.0 - MIN_CROSS_VALIDATION_MULTIPLIER) * cv_ratio,
        4,
    )

    return {
        "success": True,
        "engine": "local_deterministic_rules",
        "rules_source": [
            "search_reasoning.py background/quality/feature-rule concepts",
            "material_family.py name-pattern family classification",
            "verdict.py material-family/group agreement",
            "ftir_service.py lead-score gap uncertainty",
        ],
        "top_match": {
            "name": top_name,
            "cas": top_match.get("cas") or top_match.get("cas_number") or top_match.get("CAS NUMBER"),
            "score": top_score,
        },
        "detected_family": detected_family,
        "observed_groups": observed_groups,
        "observed_peak_positions": peak_positions,
        "checks": checks,
        "passed": passed,
        "total": total,
        "all_passed": total > 0 and passed == total,
        "confidence_multiplier": confidence_multiplier,
        "summary": f"{passed}/{total} applicable deterministic checks passed",
    }


def assess_direction(search_result: dict[str, Any]) -> dict[str, Any]:
    """Assess Top-N material-family consensus from search_library matches."""
    matches = _list(search_result.get("matches"))
    if not matches:
        return {
            "success": True,
            "engine": "local_direction_arbitration",
            "resolved_level": "uncertain_direction",
            "decision_status": "red",
            "top_match": {"name": None, "cas": None, "score": None},
            "entity_share": 0.0,
            "dominant_direction": None,
            "direction_confidence": 0.0,
            "supporting_candidates": 0,
            "competing_directions": [],
            "reason": "No library candidates were available for direction arbitration.",
        }

    scored_matches = [
        {
            "rank": index,
            "name": str(match.get("name") or "Unknown"),
            "cas": match.get("cas") or match.get("cas_number") or match.get("CAS NUMBER"),
            "score": _match_score(match),
            "family": classify_material_family(str(match.get("name") or "")),
        }
        for index, match in enumerate(matches, 1)
    ]
    groups: dict[str, dict[str, Any]] = {}
    unknown_candidates = []
    for item in scored_matches:
        family = item["family"]
        if not family:
            unknown_candidates.append(item)
            continue
        score = item["score"] or 0.0
        bucket = groups.setdefault(
            family,
            {"direction": family, "weighted_score": 0.0, "supporting_candidates": []},
        )
        bucket["weighted_score"] += score
        bucket["supporting_candidates"].append(item)

    total_weight = sum((item["score"] or 0.0) for item in scored_matches)
    ranked_groups = sorted(
        groups.values(),
        key=lambda row: (row["weighted_score"], len(row["supporting_candidates"])),
        reverse=True,
    )

    top = scored_matches[0]
    top_score = top["score"] or 0.0
    top_family = top["family"]
    top_gap = _lead_gap(matches)

    # Compute dominant_share early so both entity paths can use it.
    _pre_dominant = ranked_groups[0] if ranked_groups else None
    _pre_dominant_share = (
        _pre_dominant["weighted_score"] / total_weight
        if _pre_dominant and total_weight > 0
        else 0.0
    )
    # Direction-fully-converges: every candidate is the same family as Top-1.
    # Source: production entity_confidence formula gives 0.865 for abs_styrenic
    # (0.15 + 0.78×0.9137 + 0.35×0.0045), well above the 0.80 threshold, so
    # near-tied same-family candidates must resolve to entity/green even when
    # the raw score gap is smaller than CLOSE_LEAD_GAP_MAX.
    _direction_fully_converges = bool(
        _pre_dominant_share >= 0.95
        and top_family
        and _pre_dominant
        and top_family == _pre_dominant["direction"]
    )

    if (
        top_score >= ENTITY_STRONG_MATCH_MIN_SCORE
        and top_family
        and (
            (top_gap is not None and top_gap > CLOSE_LEAD_GAP_MAX)
            or _direction_fully_converges
        )
    ):
        gap_note = (
            f"gap {top_gap:.4f} > threshold" if top_gap is not None and top_gap > CLOSE_LEAD_GAP_MAX
            else f"all candidates converge on '{top_family}' direction (share {_pre_dominant_share:.0%})"
        )
        return {
            "success": True,
            "engine": "local_direction_arbitration",
            "resolved_level": "entity",
            "decision_status": "green",
            "top_match": {"name": top["name"], "cas": top["cas"], "score": round(top_score, 4)},
            "entity_share": round(top_score / total_weight, 4) if total_weight > 0 else 0.0,
            "dominant_direction": top_family,
            "direction_confidence": round(_pre_dominant_share if _direction_fully_converges else top_score, 4),
            "supporting_candidates": len(_pre_dominant["supporting_candidates"]) if _direction_fully_converges else 1,
            "supporting_candidate_names": (
                [item["name"] for item in _pre_dominant["supporting_candidates"][:5]]
                if _direction_fully_converges else [top["name"]]
            ),
            "competing_directions": _competing_directions(ranked_groups, top_family),
            "reason": (
                f"Top candidate score {top_score:.4f} meets the entity threshold; {gap_note}."
            ),
        }

    if not ranked_groups or total_weight <= 0:
        return {
            "success": True,
            "engine": "local_direction_arbitration",
            "resolved_level": "uncertain_direction",
            "decision_status": "red",
            "top_match": {"name": top["name"], "cas": top["cas"], "score": round(top_score, 4)},
            "entity_share": round(top_score / total_weight, 4) if total_weight > 0 else 0.0,
            "dominant_direction": None,
            "direction_confidence": 0.0,
            "supporting_candidates": 0,
            "unknown_candidates": [item["name"] for item in unknown_candidates[:5]],
            "competing_directions": [],
            "reason": "Candidates did not map to known material-family direction rules.",
        }

    dominant = ranked_groups[0]
    dominant_share = dominant["weighted_score"] / total_weight
    support_count = len(dominant["supporting_candidates"])
    direction_confidence = round(dominant_share, 4)
    # entity_share is the normalized confidence of locking the single Top-1
    # library entry against the whole candidate pool. When Top-2/3 scores are
    # close it is deliberately low, which is the honest signal that the specific
    # entity is not identifiable and the result should resolve to a direction.
    entity_share = round(top_score / total_weight, 4) if total_weight > 0 else 0.0
    passed_direction = (
        dominant_share >= DIRECTION_MIN_WEIGHTED_SHARE
        and support_count >= DIRECTION_MIN_SUPPORTING_CANDIDATES
    )
    resolved_level = "library_direction" if passed_direction else "uncertain_direction"
    decision_status = "yellow" if passed_direction else "red"

    return {
        "success": True,
        "engine": "local_direction_arbitration",
        "resolved_level": resolved_level,
        "decision_status": decision_status,
        "top_match": {"name": top["name"], "cas": top["cas"], "score": round(top_score, 4)},
        "entity_share": entity_share,
        "dominant_direction": dominant["direction"] if passed_direction else None,
        "direction_confidence": direction_confidence,
        "supporting_candidates": support_count,
        "supporting_candidate_names": [
            item["name"] for item in dominant["supporting_candidates"][:5]
        ],
        "competing_directions": _competing_directions(
            ranked_groups,
            dominant["direction"],
        ),
        "unknown_candidates": [item["name"] for item in unknown_candidates[:5]],
        "reason": (
            f"Top-1 score {top_score:.4f}, entity share {entity_share:.0%} — "
            f"near-tied candidates prevent entity-level lock-in, "
            f"but the '{dominant['direction']}' direction holds "
            f"{dominant_share:.0%} of weighted Top-N support across {support_count} "
            f"candidate(s), so the defensible level is material direction."
        ),
    }


def classify_material_family(name: str) -> str | None:
    """Classify a material name using production-derived regex patterns."""
    normalized = _normalize_text(name)
    if not normalized:
        return None
    for family_key, patterns in MATERIAL_NAME_PATTERNS:
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns):
            return family_key
    return None


# Family-specific discriminating techniques. These map "what would move a
# library_direction result down to a specific entity" onto concrete lab
# actions. The domain knowledge (which orthogonal method separates members of a
# family) is reimplemented here for the hackathon, not imported from production.
FAMILY_DISCRIMINATORS: dict[str, str] = {
    "styrenic": (
        "run a nitrogen/elemental check or DSC glass-transition scan to separate "
        "neat polystyrene from ABS/SAN (acrylonitrile) and SBR (butadiene)"
    ),
    "polyolefin": (
        "measure the DSC melting point and the 1378/1168 cm-1 methyl pattern to "
        "separate PP, PE grades, and PP/PE copolymers"
    ),
    "polyamide": (
        "compare N-H stretch position and DSC melting point against Nylon 6, 6/6, "
        "6/9 and 6/12 references to fix the specific polyamide grade"
    ),
    "protein_biopolymer": (
        "use amide I band shape plus a complementary method (thermal denaturation "
        "or a protein assay) to separate gelatin, collagen, silk and enzyme proteins"
    ),
    "polyester": (
        "compare the carbonyl position and aromatic bands against PET, PBT and "
        "aliphatic-polyester references to pin the specific polyester"
    ),
    "ester": (
        "acquire a reference spectrum of the suspected specific ester and compare "
        "the C-O and carbonyl fingerprint region directly"
    ),
    "carboxylic_acid": (
        "confirm the broad O-H and dimeric acid bands, then compare against the "
        "suspected specific acid reference under identical sampling"
    ),
}


def recommend_verification_plan(
    direction_result: dict[str, Any],
    cross_validation_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce a deterministic, human-facing next-step verification plan.

    Domain logic is reimplemented from FTIR.fun's public recommendation ideas
    (search_reasoning.py::_public_recommendation_lines) but restructured for the
    hackathon narrative: it makes the entity <-> direction level shift explicit
    and states what each step would confirm. Same inputs always yield the same
    plan, so it is safe for the deterministic/reproducible contract.
    """
    level = direction_result.get("resolved_level", "uncertain_direction")
    direction = direction_result.get("dominant_direction")
    top_match = direction_result.get("top_match") or {}
    top_name = top_match.get("name") if isinstance(top_match, dict) else None

    cv = cross_validation_result or {}
    failed_checks = [
        item for item in cv.get("checks", [])
        if item.get("applicable", True) and item.get("passed") is False
    ]
    has_conflict = any(item.get("check") == "hard_forbidden_groups" for item in failed_checks)
    has_quality_flag = any(
        item.get("check") in {"basic_quality", "background_interference", "peak_coverage"}
        for item in failed_checks
    )

    steps: list[str] = []

    if level == "entity":
        steps.append(
            f"Confirm the entity assignment of {top_name or 'the top candidate'} by "
            "comparing directly against a verified reference spectrum acquired under "
            "the same sampling conditions."
        )
        goal = "confirm_entity"
    elif level == "library_direction":
        steps.append(
            f"Treat '{direction}' as the defensible result level for now: the Top-N "
            "candidates agree on this material direction but not on a single library entry."
        )
        discriminator = FAMILY_DISCRIMINATORS.get(direction or "")
        if discriminator:
            steps.append(
                f"To narrow '{direction}' down to a specific material, {discriminator}."
            )
        else:
            steps.append(
                f"To narrow '{direction}' down to a specific material, compare the "
                "sample against reference spectra of the individual family members."
            )
        steps.append(
            "Provide sample prior information (origin, processing history, expected "
            "additives, known substrate) to constrain the candidate set further."
        )
        goal = "narrow_direction_to_entity"
    else:  # uncertain_direction
        steps.append(
            "Do not assign an identity yet: neither a single entity nor a single "
            "material direction is defensible from the current library evidence."
        )
        steps.append(
            "Re-measure with improved sample preparation and provide sample context "
            "(source, appearance, suspected class) before re-running the search."
        )
        goal = "insufficient_evidence"

    if has_conflict:
        steps.append(
            "Because observed functional groups conflict with the leading candidate, "
            "run an orthogonal measurement (Raman, DSC/TGA thermal analysis, or "
            "elemental analysis) before accepting any assignment."
        )
    if has_quality_flag:
        steps.append(
            "Repeat the acquisition after improving baseline, background subtraction, "
            "or moisture control; the current spectrum quality limits a firm conclusion."
        )

    return {
        "success": True,
        "engine": "local_verification_planner",
        "resolved_level": level,
        "goal": goal,
        "human_action_required": level != "entity" or bool(failed_checks),
        "steps": steps,
    }


def _score_gap_check(matches: list[dict[str, Any]]) -> dict[str, Any]:
    if len(matches) < 2:
        return _check(
            "score_gap",
            False,
            "Only one library candidate returned; top-1/top-2 ambiguity cannot be assessed.",
            applicable=False,
            severity="info",
        )
    s1 = _match_score(matches[0])
    s2 = _match_score(matches[1])
    if s1 is None or s2 is None:
        return _check(
            "score_gap",
            False,
            "Top candidate scores are missing; lead-score gap cannot be assessed.",
            severity="high",
        )
    gap = round(s1 - s2, 4)
    passed = gap > CLOSE_LEAD_GAP_MAX
    return _check(
        "score_gap",
        passed,
        (
            f"Top-1 vs Top-2 score gap = {gap:.4f}; "
            f"{'separated' if passed else 'close candidates, fingerprint review required'}."
        ),
        severity="medium" if not passed else "info",
        gap=gap,
        threshold=CLOSE_LEAD_GAP_MAX,
        candidate_1=matches[0].get("name"),
        candidate_2=matches[1].get("name"),
    )


def _match_quality_check(top_score: float | None) -> dict[str, Any]:
    if top_score is None:
        return _check(
            "match_quality",
            False,
            "Top match score is missing.",
            severity="high",
            threshold=MATCH_QUALITY_MIN_SCORE,
        )
    passed = top_score >= MATCH_QUALITY_MIN_SCORE
    return _check(
        "match_quality",
        passed,
        (
            f"Top match score = {top_score:.4f}; "
            f"{'meets' if passed else 'below'} the named review threshold."
        ),
        severity="medium" if not passed else "info",
        score=round(top_score, 4),
        threshold=MATCH_QUALITY_MIN_SCORE,
    )


def _family_group_check(
    detected_family: str | None,
    observed_groups: list[str],
    top_name: str,
) -> dict[str, Any]:
    if not detected_family:
        return _check(
            "family_group_agreement",
            False,
            f"No production-derived family pattern matched top candidate: {top_name or 'Unknown'}.",
            applicable=False,
            severity="info",
        )
    expected = FAMILY_EXPECTED_GROUPS.get(detected_family)
    if not expected:
        return _check(
            "family_group_agreement",
            False,
            f"Family {detected_family} has no local expected-group rule.",
            applicable=False,
            severity="info",
            family=detected_family,
        )
    observed = set(observed_groups)
    required = set(expected["required"])
    supporting = set(expected["supporting"])
    missing_required = sorted(required - observed)
    found_required = sorted(required & observed)
    found_supporting = sorted(supporting & observed)
    passed = not missing_required
    return _check(
        "family_group_agreement",
        passed,
        expected["reason"],
        severity="high" if not passed else "info",
        family=detected_family,
        found_required=found_required,
        missing_required=missing_required,
        found_supporting=found_supporting,
    )


def _hard_forbidden_group_check(
    detected_family: str | None,
    observed_groups: list[str],
    top_name: str,
) -> dict[str, Any]:
    if not detected_family:
        return _check(
            "hard_forbidden_groups",
            False,
            f"No family rule matched top candidate: {top_name or 'Unknown'}.",
            applicable=False,
            severity="info",
        )
    forbidden = set(FAMILY_FORBIDDEN_GROUPS.get(detected_family, ()))
    if not forbidden:
        return _check(
            "hard_forbidden_groups",
            False,
            f"Family {detected_family} has no local forbidden-group rule.",
            applicable=False,
            severity="info",
            family=detected_family,
        )
    observed = set(observed_groups)
    hits = sorted(forbidden & observed)
    passed = not hits
    return _check(
        "hard_forbidden_groups",
        passed,
        (
            "No candidate-specific forbidden functional groups detected."
            if passed else
            f"Observed group(s) conflict with {detected_family}: {', '.join(hits)}."
        ),
        severity="high" if not passed else "info",
        family=detected_family,
        forbidden_groups=sorted(forbidden),
        observed_forbidden_groups=hits,
    )


def _peak_coverage_check(peak_explanations: list[Any]) -> dict[str, Any]:
    if not peak_explanations:
        return _check(
            "peak_coverage",
            False,
            "No peak explanations were returned by search_library.",
            severity="medium",
            coverage=0.0,
            threshold=MIN_PEAK_EXPLANATION_COVERAGE,
        )
    explained = 0
    for item in peak_explanations:
        text = _normalize_text(item)
        if "unknown" not in text and "unassigned" not in text and "unexplained" not in text:
            explained += 1
    coverage = explained / len(peak_explanations)
    passed = coverage >= MIN_PEAK_EXPLANATION_COVERAGE
    return _check(
        "peak_coverage",
        passed,
        f"{explained}/{len(peak_explanations)} peak explanations are assigned ({coverage:.0%}).",
        severity="medium" if not passed else "info",
        coverage=round(coverage, 4),
        threshold=MIN_PEAK_EXPLANATION_COVERAGE,
    )


def _background_check(peak_positions: list[float]) -> dict[str, Any]:
    if not peak_positions:
        return _check(
            "background_interference",
            False,
            "No peak positions available for background-band screening.",
            applicable=False,
            severity="info",
        )
    flags = []
    for key, (low, high) in BACKGROUND_BANDS_CM1.items():
        if any(low <= pos <= high for pos in peak_positions):
            flags.append({"type": key, "band_cm1": [low, high]})
    # Aliphatic bands are often true polymer evidence, so they are advisory only
    # unless accompanied by CO2/moisture positions.
    blocking_flags = [item for item in flags if item["type"] in {"co2", "moisture"}]
    passed = not blocking_flags
    return _check(
        "background_interference",
        passed,
        (
            "No CO2/moisture background peaks detected in simplified peak-position screen."
            if passed else "Possible CO2/moisture background peaks detected."
        ),
        severity="medium" if not passed else "info",
        flags=flags,
    )


def _basic_quality_check(peak_positions: list[float], peak_explanations: list[Any]) -> dict[str, Any]:
    peak_count = len(peak_positions) or len(peak_explanations)
    passed = peak_count >= MIN_PEAKS_FOR_BASIC_QUALITY_CHECK
    return _check(
        "basic_quality",
        passed,
        (
            f"{peak_count} usable peak item(s) available for deterministic checks; "
            f"minimum required = {MIN_PEAKS_FOR_BASIC_QUALITY_CHECK}."
        ),
        severity="medium" if not passed else "info",
        peak_count=peak_count,
        threshold=MIN_PEAKS_FOR_BASIC_QUALITY_CHECK,
    )


def _detect_observed_groups(text: str) -> set[str]:
    found: set[str] = set()
    for group, aliases in GROUP_ALIASES.items():
        if any(alias in text for alias in aliases):
            found.add(group)
    return found


def _extract_peak_positions(*collections: list[Any]) -> list[float]:
    values: set[float] = set()
    for collection in collections:
        for item in collection:
            values.update(_numbers_from_any(item))
    return sorted(values, reverse=True)


def _numbers_from_any(value: Any) -> set[float]:
    rows: set[float] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in {"peak", "peaks", "position", "wavenumber", "wavenumbers", "cm1", "cm-1"}:
                rows.update(_numbers_from_any(item))
            elif isinstance(item, (dict, list, tuple)):
                rows.update(_numbers_from_any(item))
        return rows
    if isinstance(value, (list, tuple)):
        for item in value:
            rows.update(_numbers_from_any(item))
        return rows
    if isinstance(value, (int, float)):
        number = float(value)
        if _is_ftir_wavenumber(number):
            rows.add(round(number, 2))
        return rows
    for match in re.findall(r"(?<!\d)([4-9]\d{2}|[1-3]\d{3})(?:\.\d+)?", str(value)):
        number = float(match)
        if _is_ftir_wavenumber(number):
            rows.add(round(number, 2))
    return rows


def _is_ftir_wavenumber(number: float) -> bool:
    return 400.0 <= number <= 4000.0


def _match_score(match: dict[str, Any]) -> float | None:
    for key in ("similarity", "score", "match_score", "peak_match_score"):
        value = match.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _lead_gap(matches: list[dict[str, Any]]) -> float | None:
    if len(matches) < 2:
        return None
    s1 = _match_score(matches[0])
    s2 = _match_score(matches[1])
    if s1 is None or s2 is None:
        return None
    return round(s1 - s2, 4)


def check_spectrum_quality(
    spectrum_points: list[dict[str, Any]],
    peaks: list[float],
) -> dict[str, Any]:
    """Local quality assessment on parsed spectrum x/y data.

    Source: reimplemented from search_reasoning.py::detect_background_flags
    and assess_quality_flags using x/y band-mask approach instead of
    production grid slicing (no Django/DB dependency).

    Args:
        spectrum_points: list of {"x": wavenumber, "y": intensity} from /parse-spectrum
        peaks: detected peak positions in cm-1 from /parse-spectrum
    """
    warnings: list[str] = []
    flags: list[str] = []

    # ── Peak-position background screen (no y needed) ──────────────────────
    peaks_set = set(peaks or [])
    co2_peaks = [p for p in peaks_set if 2310.0 <= p <= 2360.0]
    moisture_peaks = [p for p in peaks_set if 3400.0 <= p <= 3700.0]
    if co2_peaks:
        flags.append("co2_background")
        warnings.append(
            f"CO₂ background peaks detected at {co2_peaks} cm⁻¹. "
            "This may indicate atmospheric contamination. Consider purging the sample compartment."
        )
    if moisture_peaks:
        flags.append("moisture_background")
        warnings.append(
            f"Moisture/O–H background peaks at {moisture_peaks} cm⁻¹. "
            "May indicate residual water. Consider drying the sample or ATR plate."
        )

    n_peaks = len(peaks or [])
    if n_peaks < MIN_PEAKS_FOR_BASIC_QUALITY_CHECK:
        flags.append("too_few_peaks")
        warnings.append(
            f"Only {n_peaks} peak(s) detected. Reliable identification requires "
            f"≥{MIN_PEAKS_FOR_BASIC_QUALITY_CHECK} peaks."
        )

    # ── y-value quality checks using band mask ──────────────────────────────
    if spectrum_points:
        xs = [float(pt.get("x", 0)) for pt in spectrum_points]
        ys = [float(pt.get("y", 0)) for pt in spectrum_points]
        n = len(ys)

        if n >= 10:
            max_y = max(ys)
            min_y = min(ys)

            # Curve-level background screen: mask y by wavenumber band so broad
            # background absorption is caught even when peak detection missed it.
            # Source thresholds: search_reasoning.py::detect_background_flags
            # (CO2 band max > 0.15, moisture band max > 0.2).
            co2_band_ys = [y for x, y in zip(xs, ys) if 2310.0 <= x <= 2360.0]
            moisture_band_ys = [y for x, y in zip(xs, ys) if 3400.0 <= x <= 3700.0]
            if co2_band_ys and max(co2_band_ys) > 0.15 and "co2_background" not in flags:
                flags.append("co2_background")
                warnings.append(
                    f"Elevated absorbance in the CO₂ band (max {max(co2_band_ys):.3f} "
                    "in 2310–2360 cm⁻¹). This may indicate atmospheric contamination."
                )
            if moisture_band_ys and max(moisture_band_ys) > 0.2 and "moisture_background" not in flags:
                flags.append("moisture_background")
                warnings.append(
                    f"Elevated absorbance in the moisture band (max {max(moisture_band_ys):.3f} "
                    "in 3400–3700 cm⁻¹). May indicate residual water."
                )

            if max_y > 1.25:
                flags.append("saturation")
                warnings.append(
                    f"Maximum intensity {max_y:.3f} exceeds 1.25 — spectrum may be saturated. "
                    "Reduce sample thickness or concentration."
                )
            if min_y < -0.02:
                flags.append("baseline_offset")
                warnings.append(
                    f"Minimum intensity {min_y:.3f} is negative — baseline correction may be needed."
                )

            # Noise check via first-difference standard deviation.
            diffs = [abs(ys[i + 1] - ys[i]) for i in range(n - 1)]
            noise_std = (sum(d * d for d in diffs) / len(diffs)) ** 0.5
            if noise_std > 0.05:
                flags.append("high_noise")
                warnings.append(
                    f"High spectral noise (diff-std {noise_std:.3f} > 0.05). "
                    "Results may be unreliable — consider signal averaging."
                )

            # Edge truncation: front/back 50 points mean > 0.25
            edge_n = min(50, n // 4)
            front_mean = sum(ys[:edge_n]) / edge_n if edge_n else 0.0
            back_mean = sum(ys[-edge_n:]) / edge_n if edge_n else 0.0
            if front_mean > 0.25 or back_mean > 0.25:
                flags.append("edge_truncation")
                warnings.append(
                    "Spectrum edges show elevated intensity — possible truncation or baseline drift. "
                    "Check wavenumber range coverage."
                )

    quality_ok = len(flags) == 0
    summary = (
        "Spectrum quality acceptable — no significant issues detected."
        if quality_ok
        else f"{len(flags)} quality issue(s) detected: {', '.join(flags)}."
    )

    return {
        "success": True,
        "quality_ok": quality_ok,
        "n_peaks_detected": n_peaks,
        "flags": flags,
        "warnings": warnings,
        "summary": summary,
    }


def _competing_directions(
    ranked_groups: list[dict[str, Any]],
    dominant_direction: str | None,
) -> list[dict[str, Any]]:
    competitors = []
    for group in ranked_groups:
        if group["direction"] == dominant_direction:
            continue
        competitors.append({
            "direction": group["direction"],
            "supporting_candidates": len(group["supporting_candidates"]),
            "weighted_score": round(group["weighted_score"], 4),
        })
    return competitors[:5]


def _check(
    check: str,
    passed: bool,
    detail: str,
    *,
    applicable: bool = True,
    severity: str = "info",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "check": check,
        "passed": passed,
        "applicable": applicable,
        "severity": severity,
        "detail": detail,
        **extra,
    }


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    replacements = {
        "\u207b": "-",
        "\u2212": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2261": "#",
        "\uff1d": "=",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
