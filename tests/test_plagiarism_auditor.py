"""
Unit tests for the overlap / plagiarism auditor (ingestion/plagiarism_auditor.py).

Pure-function tests — no Chroma, no network, no LLM — covering the deterministic
shingle layer, which is where every verdict this module reports comes from. The
embedding/paraphrase tier is gated behind an LLM call and is exercised manually.

The module's one real failure mode is a false accusation, so most of these tests
assert the *absence* of a flag: independent writing on a shared topic, field
boilerplate, stopword runs, and the same document appearing twice in a library
must all come back clean. The attribution tests pin down the sharpest rule here —
a citation copied along with the text is the source's citation, not this author's
attribution, and must not launder reused text.
"""

from ingestion.plagiarism_auditor import (
    LONG_MATCH_WORDS,
    MIN_MATCH_WORDS,
    _Doc,
    _body_chunks,
    _classify,
    _content_words,
    _has_citation,
    build_index,
    find_matches,
    split_same_documents,
)


def _doc(paper_id, title, texts, section="Introduction", page=1, section_type="text"):
    d = _Doc(paper_id, title)
    for t in texts:
        d.add_chunk({"text": t, "metadata": {
            "section": section, "section_type": section_type, "page_num": page,
        }}, 100_000)
    return d


def _match(draft, sources):
    return find_matches(draft, sources, build_index(sources))


# 22 words: over MIN_MATCH_WORDS, under LONG_MATCH_WORDS.
STOLEN = ("The transformer architecture relies entirely on self attention mechanisms to draw "
          "global dependencies between input and output sequences without recurrence")
LONGER = (STOLEN + " and the model achieves substantially better translation quality "
          "across every language pair we measured")


# --- the match itself --------------------------------------------------------

def test_reused_passage_is_found_and_quoted_from_both_sides():
    src = _doc("s1", "Source Paper A", ["Preamble here. " + STOLEN + ". More text."])
    draft = _doc("d1", "draft", ["We begin. " + STOLEN + ". Our contribution differs."])

    matches = _match(draft, [src])
    assert len(matches) == 1

    entry = _classify(draft, [src], matches[0])
    assert entry["words"] >= MIN_MATCH_WORDS
    assert entry["source"]["title"] == "Source Paper A"
    # Excerpts are sliced from the original text, not rebuilt from tokens, so
    # the author sees their own words back.
    assert "self attention" in entry["excerpt"]
    assert "self attention" in entry["source"]["excerpt"]


def test_severity_tiers_key_off_run_length():
    src_s = _doc("s2", "A", ["Lead. " + STOLEN + ". Tail."])
    d_s   = _doc("d2", "draft", ["Open. " + STOLEN + ". Close."])
    short = _classify(d_s, [src_s], _match(d_s, [src_s])[0])
    assert short["verdict"] == "NEAR_VERBATIM" and short["severity"] == "med"

    src_l = _doc("s3", "A", ["Lead. " + LONGER + ". Tail."])
    d_l   = _doc("d3", "draft", ["Open. " + LONGER + ". Close."])
    long_ = _classify(d_l, [src_l], _match(d_l, [src_l])[0])
    assert long_["words"] >= LONG_MATCH_WORDS
    assert long_["verdict"] == "VERBATIM" and long_["severity"] == "high"


def test_independent_writing_on_the_same_topic_is_not_a_match():
    src = _doc("s4", "A", ["Preamble. " + STOLEN + ". More."])
    draft = _doc("d4", "draft", [
        "We study attention based models for translation. Our encoder uses stacked "
        "layers and we report BLEU on newstest2014 for every configuration."])
    assert _match(draft, [src]) == []


# --- attribution: credit counts only when the author gave it -----------------

def test_citation_written_beside_the_passage_downgrades_to_attributed():
    src = _doc("s5", "A", ["Preamble. " + STOLEN + ". More."])
    draft = _doc("d5", "draft", ["As shown by Vaswani et al. (2017), " + STOLEN + ". We build on this."])
    assert _classify(draft, [src], _match(draft, [src])[0])["verdict"] == "ATTRIBUTED"


def test_quotation_marks_downgrade_to_attributed():
    src = _doc("s6", "A", ["Preamble. " + STOLEN + ". More."])
    draft = _doc("d6", "draft", ['They write: "' + STOLEN + '." We agree.'])
    assert _classify(draft, [src], _match(draft, [src])[0])["verdict"] == "ATTRIBUTED"


def test_citations_copied_inside_the_passage_do_not_count_as_attribution():
    """The sharpest rule in the module.

    Reused related-work prose is always dense with citation markers — but those
    are the *source's* citations, carried over with the text. Counting them as
    credit would wave through exactly the passages most worth flagging.
    """
    inner = ("prior approaches such as the Extended Neural GPU [16], ByteNet [18] and "
             "ConvS2S [9] all use convolutional networks as their basic building block")
    src = _doc("s7", "A", ["Background. " + inner + ". More."])
    draft = _doc("d7", "draft", ["Our setting. " + inner + ". We differ."])
    assert _classify(draft, [src], _match(draft, [src])[0])["verdict"] != "ATTRIBUTED"


# --- guards against false accusations ----------------------------------------

def test_string_shared_by_three_papers_is_field_boilerplate():
    boiler = ("we evaluate our approach on the standard benchmark using the usual train "
              "development and test split reported in prior work")
    sources = [_doc(f"b{i}", f"Paper {i}", ["Intro. " + boiler + ". More."]) for i in range(3)]
    draft = _doc("d8", "draft", ["Setup. " + boiler + ". Results."])

    assert _match(draft, sources) == []
    # …but the same string in a single paper is still reported.
    assert len(_match(draft, sources[:1])) == 1


def test_a_run_of_stopwords_carries_no_evidence():
    filler = "of the results that we have to be able to do this in the case of the"
    assert _content_words(filler.split()) < 6
    src = _doc("s9", "A", ["Text. " + filler + " thing. End."])
    draft = _doc("d9", "draft", ["Other. " + filler + " thing. Done."])
    assert _match(draft, [src]) == []


SELF_BODY = " ".join(
    f"section {i} discusses the calibrated ensemble routing procedure in detail "
    f"with quantitative ablations over five random seeds" for i in range(12))


def test_same_document_in_the_library_is_excluded_not_reported():
    src = _doc("s10", "My Paper (published)", [SELF_BODY])
    draft = _doc("d10", "draft", [SELF_BODY])

    comparable, excluded = split_same_documents(draft, [src])
    assert comparable == []
    assert len(excluded) == 1
    assert excluded[0]["paper_id"] == "s10"
    assert "same document" in excluded[0]["reason"]


def test_every_duplicate_copy_is_excluded_not_just_the_first():
    """Regression from the first browser run, which had three copies of the
    draft in the library.

    The guard used to measure coverage from the completed match list, but the
    matcher is greedy — each draft span is assigned to one best source, so copy
    #1 absorbed all the coverage and copies #2 and #3 scored ~0 and survived.
    They then came back as "possible rewrite" cards of byte-identical text.
    Measuring each source independently, before indexing, catches all three.
    """
    draft = _doc("d13", "draft", [SELF_BODY])
    copies = [_doc(f"c{i}", f"Copy {i}", [SELF_BODY]) for i in range(3)]

    comparable, excluded = split_same_documents(draft, copies)
    assert comparable == []
    assert sorted(e["paper_id"] for e in excluded) == ["c0", "c1", "c2"]


def test_duplicates_are_removed_before_the_boilerplate_guard_can_see_them():
    """The other half of the same bug: three identical copies in the index also
    trip COMMON_SOURCE_LIMIT, so every genuine match would be discarded as
    "standard field phrasing". Splitting them out first is what prevents it."""
    src = _doc("s14", "Source Paper A", ["Preamble. " + STOLEN + ". More."])
    copies = [_doc(f"k{i}", f"Copy {i}", [SELF_BODY]) for i in range(3)]
    draft = _doc("d14", "draft", [SELF_BODY + " " + STOLEN + "."])

    comparable, excluded = split_same_documents(draft, copies + [src])
    assert len(excluded) == 3
    assert [s.paper_id for s in comparable] == ["s14"]
    # With the copies gone, the real reuse is still found.
    assert len(_match(draft, comparable)) == 1


def test_partial_reuse_does_not_trip_the_same_document_guard():
    src = _doc("s11", "A", ["Preamble. " + STOLEN + ". More."])
    draft = _doc("d11", "draft", [
        "Our framing is entirely different and shares nothing with the source. " + STOLEN + ". "
        + " ".join(f"unique sentence number {i} about our own distinct approach" for i in range(40))])

    comparable, excluded = split_same_documents(draft, [src])
    assert [s.paper_id for s in comparable] == ["s11"]
    assert excluded == []
    assert len(_match(draft, comparable)) == 1


# --- section filtering -------------------------------------------------------

def test_references_bibliography_acknowledgements_and_tables_are_excluded():
    """Two papers in a field share reference lists verbatim and share benchmark
    tables; neither is reuse. The abstract stays in — reused text there is
    exactly what this check is for."""
    chunks = [
        {"text": "body", "metadata": {"section": "3 Model Architecture", "section_type": "text"}},
        {"text": "refs", "metadata": {"section": "References", "section_type": "text"}},
        {"text": "bib",  "metadata": {"section": "Bibliography", "section_type": "text"}},
        {"text": "ack",  "metadata": {"section": "Acknowledgements", "section_type": "text"}},
        {"text": "tbl",  "metadata": {"section": "5 Results", "section_type": "table"}},
        {"text": "abs",  "metadata": {"section": "Abstract", "section_type": "text"}},
    ]
    kept = [c["metadata"]["section"] for c in _body_chunks(chunks)]
    assert kept == ["3 Model Architecture", "Abstract"]


# --- citation marker detection -----------------------------------------------

def test_citation_marker_detection():
    assert _has_citation("as in [12] we do")
    assert _has_citation("(Smith et al., 2020) shows")
    assert _has_citation("Devlin and Chang 2019 report")
    assert not _has_citation("plain prose with no marker")
    # A year inside a table reference is not a citation.
    assert not _has_citation("results in Table 3 show 2019 numbers")
