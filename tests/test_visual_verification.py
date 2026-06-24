"""Required test: verified VISUAL recall rejects an unsupported visual claim.

The no-confabulation guarantee extended to images: a claim is judged against the actual
pixels (qwen-vl-plus), and the raw image is the arbiter. Makes real vision calls, so it
skips without a key. We render a clearly DECREASING revenue chart and check that the
claim "revenue increased" is NOT entailed by the image."""
from __future__ import annotations

import pytest

from eidetic.config import get_settings
from eidetic.models import Modality, NLILabel


def _need_key():
    if not get_settings().has_api_key:
        pytest.skip("No DASHSCOPE_API_KEY: visual verification needs a real vision call.")


def _make_decreasing_chart(path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Q1", "Q2", "Q3", "Q4"], [100, 70, 40, 18], color="tab:red")
    ax.set_title("Quarterly Revenue 2026")
    ax.set_ylabel("Revenue ($M)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def test_visual_verification_rejects_unsupported_claim(engine, tmp_path):
    _need_key()
    img = tmp_path / "chart.png"
    _make_decreasing_chart(str(img))
    # The chart shows revenue FALLING. This claim is the opposite -> must NOT be entailed.
    label, conf = engine.client.verify_visual(str(img), "This chart shows revenue increasing every quarter.")
    assert label != NLILabel.ENTAILMENT.value


def test_visual_verification_accepts_supported_claim(engine, tmp_path):
    _need_key()
    img = tmp_path / "chart.png"
    _make_decreasing_chart(str(img))
    label, _ = engine.client.verify_visual(str(img), "This chart shows revenue declining across the quarters.")
    assert label != NLILabel.CONTRADICTION.value   # the pixels support a downward trend


def test_image_ingest_feeds_the_graph(engine, tmp_path):
    """Vision feeds the graph: an image ingest produces real entities/edges, not just a vector."""
    _need_key()
    img = tmp_path / "chart.png"
    _make_decreasing_chart(str(img))
    rec = engine.ingest_bytes(img.read_bytes(), "chart.png", source="vision-test")
    assert rec.modality == Modality.IMAGE
    # The visual extractor turned the chart into graph entities.
    assert len(rec.entities) >= 1
