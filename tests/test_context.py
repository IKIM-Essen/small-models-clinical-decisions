import pytest

from kevmed import context, tokens


@pytest.fixture(scope="module")
def tok():
    return tokens.load_tokenizer()


def test_truncate_keeps_prefix_and_flags(tok):
    text = " ".join(f"w{i}" for i in range(500))
    out, cut = context.truncate(text, 50, tok)
    assert cut and out.startswith("w0 w1") and len(tok(out)["input_ids"]) <= 50
    same, cut2 = context.truncate("short", 50, tok)
    assert same == "short" and not cut2


def test_section_first_moves_named_sections_to_front(tok):
    text = "Anamnese\nlong history " * 30 + "\nDiagnosen\nI21.0 STEMI\n\nTherapie\nPCI done\n"
    out, cut = context.section_first(text, ["Diagnosen", "Therapie"], 60, tok)
    assert out.startswith("Diagnosen") and "PCI done" in out and cut


def test_chunks_cover_text_with_overlap(tok):
    text = " ".join(f"w{i}" for i in range(300))
    cs = context.chunks(text, 100, tok, overlap=10)
    assert len(cs) >= 3 and all(len(tok(c)["input_ids"]) <= 100 for c in cs)
    assert "w0" in cs[0] and "w299" in cs[-1]


def test_section_first_recognises_markdown_and_multiword_headings(tok):
    text = ("## Anamnese\n" + "long history text " * 40 + "\n**Therapie und Verlauf:**\nPCI durchgeführt.\n\n"
            "## Diagnosen:\nHypertonie\nDiabetes mellitus Typ 2\n\nZusammenfassung und Beurteilung\nStabil entlassen.\n")
    out, cut = context.section_first(text, ["Diagnosen:", "Therapie und Verlauf"], 80, tok)
    assert out.startswith("Diagnosen") and "Diabetes mellitus Typ 2" in out and "PCI durchgeführt" in out
    assert out.index("Diabetes") < out.index("PCI") and cut


def test_section_first_returns_text_unchanged_when_no_heading_matches(tok):
    text = "Plain paragraph one.\nPlain paragraph two."
    out, cut = context.section_first(text, ["Diagnosen"], 100, tok)
    assert out == text and not cut


def test_markdown_headings_always_split_even_when_named(tok):
    text = "## Anamnese\nA text.\n## Diagnosen\nD text.\n## Therapie\nT text.\n"
    secs = context._sections(text, {"diagnosen"})
    assert [h for h, _ in secs if h] == ["anamnese", "diagnosen", "therapie"]
