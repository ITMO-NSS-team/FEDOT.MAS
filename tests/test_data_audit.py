from scripts.data_audit import build_manifest, card_key, normalize_filename


def test_normalize_filename_is_unicode_and_separator_stable() -> None:
    assert normalize_filename("Монтаж__Й  окон") == "монтаж й окон"
    assert normalize_filename("Монтаж\u00a0окон") == "монтаж окон"


def test_manifest_matches_extension_independent_card_names() -> None:
    key = card_key("Категория", "Монтаж__окон")
    markdown = {key: [{"path": "data/cards_md/Категория/Монтаж__окон.md", "category": "Категория"}]}
    raw = {key: [{"path": "data/raw/Категория/Монтаж окон.pdf", "category": "Категория", "raw_extension": ".pdf"}]}

    row = build_manifest(markdown, raw)[0]

    assert row["has_markdown"] is True
    assert row["has_raw"] is True
    assert row["raw_extension"] == ".pdf"
    assert row["normalized_title"] == "монтаж окон"
