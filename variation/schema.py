"""The canonical variation schema — three tables, one recipe, and the validators.

This module is the productisation hinge described in the build spec (§3.4). Every
client formats their spreadsheet differently; intake normalises whatever arrives into
the shapes defined here, and **everything downstream consumes only these shapes**.

Three tables, because the client's sheet mixes two kinds of data in one:

    VARIANTS   one row per output image   "which cells exist, what is each called"
    SPECS      one row per axis value     "what is each material or colour actually"
    PRODUCT    one row per product/plate  "plates, naming pattern, colour profile"

The consequence worth noticing: a 9x9 product has 81 variant rows but only 18 specs.
The client supplies 18 specifications and the system generates the 81 combinations.

NOTHING PRODUCT-SPECIFIC LIVES HERE. No region names, no axis names, no counts, no
filename patterns. Those are all recipe fields. A grep of this package for a product
noun should come back empty — that is the generalisation test (spec §0.5.3).
"""

from __future__ import annotations

import hashlib
import json
import re

SCHEMA_VERSION = "1.0"

# A column named `axis:something` declares a variation axis called `something`. Prefixing
# is what lets the axis COUNT be free while the structure stays fixed — a product may
# have one axis or six, and the column set changes without the schema changing. This
# mirrors WooCommerce's own `meta:` convention, which clients doing store imports have
# already met.
AXIS_PREFIX = "axis:"

VARIANT_SYSTEM_COLUMNS = ("status", "output_url")
SPEC_TYPES = ("hex", "reference_image", "word", "resolved_word")

# Tokens a naming pattern may use that are NOT axis lookups. A filename built only from
# axis values is a valid primary key and unreadable at a glance — a running number makes
# a folder of 35 sortable and quotable — so the pattern needs a way to say "the position
# of this cell" without the validator hunting for an axis called `index`.
NAMING_RESERVED = ("index", "n")

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$")


class ValidationError(ValueError):
    """Intake or recipe validation failed. The message names the offending row."""


def canonical(text) -> str:
    """Normalise an axis value to the canonical lowercase-hyphenated id.

    Sheet values arrive as `Grigio Talami Marble`, `grigio-talami-marble`, or with
    stray whitespace from a copy-paste. Every one of those must resolve to one id, or
    a VARIANTS row silently fails to find its SPECS row.
    """
    text = str(text or "").strip().lower()
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"[^a-z0-9\-]+", "", text)
    return re.sub(r"-{2,}", "-", text).strip("-")


def normalise_hex(value):
    """`E78820` / `#e78820` / `#abc` -> `#E78820`, or None if it is not a hex colour."""
    if value is None:
        return None
    match = _HEX_RE.match(str(value).strip())
    if not match:
        return None
    digits = match.group(1)
    if len(digits) == 3:                       # #abc -> #aabbcc
        digits = "".join(c * 2 for c in digits)
    return "#" + digits.upper()


def sha256_text(text) -> str:
    return "sha256:" + hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def sha256_bytes(data) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def stable_json(obj) -> str:
    """Deterministic JSON — sorted keys, no incidental whitespace. Used for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ── The three tables ─────────────────────────────────────────────────────────

def axis_names(variants) -> list:
    """Axis names discovered from the `axis:` prefixed columns, in first-seen order.

    Discovered, never declared: the axis COUNT is a property of the client's product,
    so anything that iterates a fixed number of axes is a defect (spec §0.5.2).
    """
    names, seen = [], set()
    for row in variants:
        for key in row:
            if key.startswith(AXIS_PREFIX):
                name = key[len(AXIS_PREFIX):].strip()
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
    return names


def specs_index(specs) -> dict:
    """`(axis, value) -> spec row`, with multi-reference rows merged into `refs`.

    A value may legitimately carry several reference images with distinct roles — a
    flat material sample plus a lit in-situ example. Those arrive as repeated rows and
    are collected here rather than averaged: averaging three references produces a
    fourth material matching none of them (spec §6).
    """
    index = {}
    for row in specs:
        axis = canonical(row.get("axis"))
        value = canonical(row.get("value"))
        if not axis or not value:
            continue
        key = (axis, value)
        entry = index.get(key)
        if entry is None:
            entry = {
                "axis": axis,
                "value": value,
                "display": str(row.get("display") or "").strip() or value,
                "filename_token": canonical(row.get("filename_token")) or value,
                "hex": normalise_hex(row.get("hex")),
                "description": str(row.get("description") or "").strip(),
                "paints": str(row.get("paints") or "").strip() or None,
                "refs": [],
            }
            index[key] = entry
        url = str(row.get("ref_url") or "").strip()
        if url:
            entry["refs"].append({
                "url": url,
                "role": str(row.get("ref_role") or "").strip() or "material",
            })
        # A later row may fill a field the first row left blank.
        for field in ("display", "description"):
            if not entry[field] and row.get(field):
                entry[field] = str(row[field]).strip()
        if entry["hex"] is None:
            entry["hex"] = normalise_hex(row.get("hex"))
        if not entry["paints"] and row.get("paints"):
            entry["paints"] = str(row["paints"]).strip()
    return index


def spec_type_of(entry) -> str:
    """Route a resolved spec entry to its modality (spec §6).

    `hex` is the best case: an exact target that also enables automatic colour
    verification and opens the non-generative path. `word` is the worst — a bare word
    is not a specification but a request for an opinion, and the model gives a
    different opinion every time it is asked. It must be resolved to a swatch before
    any generation happens.
    """
    if entry.get("refs"):
        return "reference_image"
    if entry.get("hex"):
        return "hex"
    return "word"


# ── Intake validation (spec §3.4.7) ──────────────────────────────────────────

def validate_intake(variants, specs, product, plates=None) -> list:
    """Every deterministic check that can run before any generation spend.

    Returns a list of human-readable problems, each naming the offending row. An empty
    list means intake passed. These are cheap and they run first, because the
    alternative is discovering a missing material at cell 380 of 720.
    """
    problems = []
    if not variants:
        problems.append("VARIANTS is empty — no output images are declared.")
    if not specs:
        problems.append("SPECS is empty — no axis value has a specification.")
    if not product:
        problems.append("PRODUCT is empty — no product row was found.")

    names = axis_names(variants)
    if not names:
        problems.append(
            "No `%s` columns in VARIANTS — at least one variation axis is required."
            % AXIS_PREFIX)

    index = specs_index(specs)
    spec_axes = {axis for axis, _ in index}

    # Every axis column has matching SPECS rows.
    for name in names:
        if canonical(name) not in spec_axes:
            problems.append(
                "Axis '%s' has a column in VARIANTS but no rows in SPECS." % name)

    # Every axis value used in VARIANTS exists in SPECS.
    missing = set()
    for i, row in enumerate(variants, start=1):
        for name in names:
            raw = row.get(AXIS_PREFIX + name)
            value = canonical(raw)
            if not value:
                problems.append(
                    "VARIANTS row %d ('%s') has no value for axis '%s'. The run MUST "
                    "stop: a guessed material produces a plausible image that is wrong."
                    % (i, row.get("filename", "?"), name))
                continue
            if (canonical(name), value) not in index:
                missing.add((name, value))
    for name, value in sorted(missing):
        problems.append(
            "Axis '%s' value '%s' is used in VARIANTS but has no SPECS row." % (name, value))

    # THE SINGLE MOST IMPORTANT VALIDATION RULE: every spec is a hex or a reference.
    # Enforcing it here converts "the client said amber" from a post-delivery argument
    # into a pre-production question.
    for (axis, value), entry in sorted(index.items()):
        if not entry["hex"] and not entry["refs"]:
            problems.append(
                "SPECS row '%s / %s' has neither a hex nor a ref_url. A bare word is not "
                "a specification — resolve it to a swatch first (spec §6.3)." % (axis, value))

    # filename is required, unique, and must parse back to its own axis values.
    seen = {}
    for i, row in enumerate(variants, start=1):
        name = str(row.get("filename") or "").strip()
        if not name:
            problems.append("VARIANTS row %d has no filename. It is the primary key." % i)
            continue
        if name in seen:
            problems.append(
                "Duplicate filename '%s' on VARIANTS rows %d and %d." % (name, seen[name], i))
        seen[name] = i

    # Cell-count reconciliation. A shortfall usually means the client omitted
    # combinations, and catching that here rather than at delivery is worth the check.
    plate_count = max(1, len(plates or []))
    expected = plate_count
    for name in names:
        values = {v for a, v in index if a == canonical(name)}
        expected *= max(1, len(values))
    if names and expected != len(variants):
        problems.append(
            "VARIANTS has %d rows but the axes x plates product is %d. The client may "
            "have omitted combinations — confirm before generating."
            % (len(variants), expected))

    return problems


# ── The recipe (spec §5.2) ───────────────────────────────────────────────────
# The recipe is the compiled, frozen configuration for one product. It is produced once
# per product by a vision LLM (Tier 0), read once by a human, then frozen — and all N
# images inherit it. Everything the run needs is in here.

RECIPE_REQUIRED = ("schema_version", "product", "plates", "regions", "axes",
                   "templates", "invariants", "lock_product", "lock_scene",
                   "naming", "verification")

# Fields the Tier 0 LLM MUST NOT populate. The locks and the tolerances are constants
# supplied by the system: an LLM that rewrites a lock will occasionally phrase it
# weakly, and that one cell is the one that drifts — a failure that is effectively
# undiscoverable by reading prompts. Tolerances must come from measurement (§7.5),
# never from a model's guess.
LLM_FORBIDDEN = ("lock_product", "lock_scene", "verification")


def validate_recipe(recipe) -> list:
    """Structural validation of a compiled recipe. Empty list means valid."""
    problems = []
    if not isinstance(recipe, dict):
        return ["Recipe is not a JSON object."]

    for key in RECIPE_REQUIRED:
        if key not in recipe:
            problems.append("Recipe is missing required key '%s'." % key)

    regions = recipe.get("regions") or []
    if not isinstance(regions, list) or not regions:
        problems.append("Recipe 'regions' must be a non-empty list.")

    axes = recipe.get("axes") or []
    if not isinstance(axes, list) or not axes:
        problems.append("Recipe 'axes' must be a non-empty list.")
        axes = []

    seen_axis = set()
    for i, axis in enumerate(axes):
        if not isinstance(axis, dict):
            problems.append("Recipe axis %d is not an object." % i)
            continue
        name = axis.get("name")
        if not name:
            problems.append("Recipe axis %d has no name." % i)
        elif name in seen_axis:
            problems.append("Recipe declares axis '%s' twice." % name)
        else:
            seen_axis.add(name)

        paints = axis.get("paints")
        if paints in (None, "", "null"):
            # Not fatal here — it is exactly what the human gate exists to resolve.
            problems.append(
                "Axis '%s' has no 'paints' region. The Tier 0 model was not confident; "
                "the operator MUST resolve it before generating (spec §5.6)." % name)
        elif regions and paints not in regions:
            problems.append(
                "Axis '%s' paints region '%s', which is not in the regions list %s."
                % (name, paints, regions))

        stype = axis.get("spec_type")
        if stype not in SPEC_TYPES:
            problems.append(
                "Axis '%s' has spec_type '%s'; expected one of %s."
                % (name, stype, list(SPEC_TYPES)))

        values = axis.get("values") or []
        if not isinstance(values, list) or not values:
            problems.append("Axis '%s' has no values." % name)
            continue
        for value in values:
            if not isinstance(value, dict) or not value.get("id"):
                problems.append("Axis '%s' has a value with no id." % name)
                continue
            vid = value["id"]
            # Each value is judged by ITS OWN type. A single axis routinely mixes
            # formats — a hex for one finish, a photograph of a sample for the next —
            # and judging every value by the axis's summary type failed whichever half
            # did not match it.
            vtype = value.get("spec_type") or stype
            if vtype not in SPEC_TYPES:
                problems.append(
                    "Axis '%s' value '%s' has spec_type '%s'; expected one of %s."
                    % (name, vid, vtype, list(SPEC_TYPES)))
            if vtype == "hex" and not normalise_hex(value.get("hex")):
                problems.append(
                    "Axis '%s' value '%s' is spec_type hex but carries no valid hex."
                    % (name, vid))
            if vtype == "reference_image" and not value.get("ref"):
                problems.append(
                    "Axis '%s' value '%s' is spec_type reference_image but has no ref."
                    % (name, vid))
            if vtype == "word":
                problems.append(
                    "Axis '%s' value '%s' is still spec_type 'word'. Resolve it to a "
                    "swatch before generating (spec §6.3)." % (name, vid))

    templates = recipe.get("templates") or {}
    if not isinstance(templates, dict):
        problems.append("Recipe 'templates' must be an object.")
    else:
        # A template is needed for every type any VALUE actually uses. Checking only the
        # axis's summary type let a mixed axis reach generation with no template for its
        # minority half, which surfaces as a hard failure per cell rather than here.
        for axis in axes:
            used = {v.get("spec_type") for v in (axis.get("values") or [])
                    if isinstance(v, dict) and v.get("spec_type")}
            used.add(axis.get("spec_type"))
            for stype in sorted(t for t in used if t):
                if stype not in templates:
                    problems.append(
                        "No template for spec_type '%s', used by axis '%s'."
                        % (stype, axis.get("name")))

    invariants = recipe.get("invariants") or []
    if not isinstance(invariants, list) or not invariants:
        problems.append(
            "Recipe 'invariants' is empty. Per-product invariants are the strongest "
            "identity anchor available and cost nothing to produce (spec §5.4.5).")

    naming = recipe.get("naming")
    if not naming or not isinstance(naming, str):
        problems.append("Recipe 'naming' pattern is missing. It is never invented (§2.4).")
    else:
        # Every axis referenced by the pattern must exist, or filenames silently
        # collapse to the same string and outputs overwrite each other.
        for token in re.findall(r"\{([^}]+)\}", naming):
            axis_part = token.split(".", 1)[0]
            if axis_part in NAMING_RESERVED:      # a running number, not an axis
                continue
            if axis_part not in seen_axis:
                problems.append(
                    "Naming pattern references '{%s}' but there is no axis '%s'."
                    % (token, axis_part))

    for key in ("lock_product", "lock_scene"):
        if not str(recipe.get(key) or "").strip():
            problems.append(
                "Recipe '%s' is empty. Both locks are constants appended verbatim to "
                "every prompt and neither is optional (spec §5.4.2)." % key)

    verification = recipe.get("verification") or {}
    if not isinstance(verification, dict):
        problems.append("Recipe 'verification' must be an object.")
    else:
        for key in ("frame_tolerance", "colour_tolerance_de", "max_retries"):
            if key not in verification:
                problems.append("Recipe verification is missing '%s'." % key)

    return problems


# Bookkeeping that changes without changing a single pixel. Excluded from the hash, or
# approving a recipe would invalidate every job already completed under it — and a job
# store keyed on recipe_hash would then regenerate work that was already paid for.
HASH_EXCLUDED = ("approved_by", "approved_at", "compiled_at", "recipe_hash", "_path")


def recipe_hash(recipe) -> str:
    """Hash of everything that affects output, ignoring bookkeeping fields."""
    payload = {k: v for k, v in (recipe or {}).items() if k not in HASH_EXCLUDED}
    return sha256_text(stable_json(payload))


def axis_by_name(recipe, name):
    for axis in recipe.get("axes") or []:
        if axis.get("name") == name:
            return axis
    return None


def value_by_id(axis, value_id):
    for value in axis.get("values") or []:
        if value.get("id") == value_id:
            return value
    return None


def render_filename(recipe, axes_values, index=None, total=None) -> str:
    """Build one output filename from the recipe's naming pattern.

    Supports `{axis}` and `{axis.field}`, where field is any key on the value —
    `filename_token` in practice. The pattern is lossy relative to the axis ids on
    purpose: a value id of `amber-transparent` may appear in filenames as `amber`,
    which is why a `filename_token` is carried per value rather than derived by string
    manipulation.
    """
    out = recipe.get("naming") or ""

    # `{index}` / `{n}` is a running number rather than an axis lookup, so it is
    # substituted first and removed from the token list the axis resolver then sees.
    if index is not None:
        width = max(2, len(str(int(total or 0)))) if total else 2
        number = str(int(index)).zfill(width)
        out = out.replace("{index}", number).replace("{n}", number)

    for token in set(re.findall(r"\{([^}]+)\}", out)):
        axis_name, _, field = token.partition(".")
        field = field or "filename_token"
        axis = axis_by_name(recipe, axis_name)
        value = value_by_id(axis, axes_values.get(axis_name)) if axis else None
        if value is None:
            raise ValidationError(
                "Naming pattern token '{%s}' cannot be resolved for cell %s."
                % (token, axes_values))
        replacement = value.get(field) or value.get("filename_token") or value.get("id")
        out = out.replace("{%s}" % token, str(replacement))
    return out
