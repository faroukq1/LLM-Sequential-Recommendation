"""
PromptCraft-SeqRec: 6 item description strategies for LLM embedding generation.
Each strategy is a function: item_dict -> str

Field names are for Amazon Beauty dataset (asin, title, brand, categories, description, price).
"""

from __future__ import annotations


def _safe_get(d: dict, key: str, default: str = "unknown") -> str:
    val = d.get(key, default)
    if not val or val == "" or val == []:
        return default
    if isinstance(val, list):
        val = val[0] if val else default
    return str(val).strip()


def _get_categories(d: dict, n: int = 3) -> str:
    cats = d.get("categories", d.get("category", []))
    if not cats:
        return "general"
    if cats and isinstance(cats[0], list):
        flat = [c for sub in cats for c in sub]
    else:
        flat = list(cats)
    meaningful = [c for c in flat if len(str(c)) > 2][1 : n + 1]
    return ", ".join(meaningful) if meaningful else ", ".join(str(c) for c in flat[:n])


def _get_description(d: dict, max_len: int = 200) -> str:
    desc = d.get("description", "")
    if isinstance(desc, list):
        desc = " ".join(desc)
    return str(desc)[:max_len].strip()


def _get_related_titles(d: dict, meta_lookup: dict, n: int = 3) -> str:
    related = d.get("related", {})
    also_bought = related.get("also_bought", related.get("bought_together", []))
    titles = []
    for asin in also_bought[:n]:
        m = meta_lookup.get(asin, {})
        t = _safe_get(m, "title")
        if t != "unknown":
            titles.append(t)
    return ", ".join(titles) if titles else "comparable products in this category"


def build_prompt(strategy: str, d: dict, meta_lookup: dict | None = None) -> str:
    """
    Build a text prompt for an item using the given strategy.

    Args:
        strategy: one of the 6 strategy names
        d: item metadata dict (fields: title, brand, categories, description, price, related, ...)
        meta_lookup: optional dict {asin: metadata} for type5_comparative related-item lookup
    """
    meta_lookup = meta_lookup or {}
    title = _safe_get(d, "title", d.get("asin", "unknown product"))
    brand = _safe_get(d, "brand")
    cats = _get_categories(d)
    desc = _get_description(d)
    price = _safe_get(d, "price")

    if strategy == "type1_title_only":
        return title

    elif strategy == "type2_structured":
        parts = [title]
        if brand != "unknown":
            parts.append(f"Brand: {brand}")
        if cats != "general":
            parts.append(f"Category: {cats}")
        if price != "unknown":
            parts.append(f"Price: {price}")
        return " | ".join(parts)

    elif strategy == "type3_rich_prose":
        base = title
        if brand != "unknown":
            base += f" by {brand}"
        if cats != "general":
            base += f", a {cats} product"
        if desc:
            base += f". {desc}"
        return base.strip()

    elif strategy == "type4_user_centric":
        text = f"Users who like {title} enjoy"
        if cats != "general":
            text += f": {cats}"
        if brand != "unknown":
            text += f" products from {brand}"
        return text

    elif strategy == "type5_comparative":
        similar = _get_related_titles(d, meta_lookup, n=3)
        text = f"{title} is similar to: {similar}"
        if cats != "general":
            text += f". Appeals to fans of: {cats}"
        return text

    elif strategy == "type6_hybrid":
        parts = [title]
        if cats != "general":
            parts.append(f"Category: {cats}")
        if brand != "unknown":
            parts.append(f"Brand: {brand}")
        return " | ".join(parts)

    return title


STRATEGY_NAMES = [
    "type1_title_only",
    "type2_structured",
    "type3_rich_prose",
    "type4_user_centric",
    "type5_comparative",
    "type6_hybrid",
]

STRATEGY_LABELS = {
    "type1_title_only": "T1: Title Only (Baseline)",
    "type2_structured": "T2: Structured Attrs",
    "type3_rich_prose": "T3: Rich Prose",
    "type4_user_centric": "T4: User Centric",
    "type5_comparative": "T5: Comparative",
    "type6_hybrid": "T6: Hybrid (Best)",
}


def apply_strategy(strategy: str, items: dict, meta_lookup: dict | None = None) -> tuple[list, list[str]]:
    """
    Apply a prompt strategy to all items.

    Args:
        strategy: strategy name
        items: dict of {item_id: metadata_dict}
        meta_lookup: optional full metadata dict for comparative strategy

    Returns:
        (item_ids, texts) — parallel lists
    """
    meta_lookup = meta_lookup or items
    item_ids = list(items.keys())
    texts = [build_prompt(strategy, items[iid], meta_lookup) for iid in item_ids]
    return item_ids, texts
