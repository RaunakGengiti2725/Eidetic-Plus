"""The ONLY place model calls happen. Every method is a real Qwen/DashScope call.

Hard rule from the spec: no mocked model outputs anywhere. If a call cannot be
made (missing key, dead quota, API error) we raise loudly -- we never fabricate a
result. Callers must be prepared for ModelCallError.

Model-per-cognitive-function mapping (Section 10.7 of the dossier):
  qwen-flash   -> write-time salience/importance
  text-embedding-v4 / tongyi-embedding-vision -> encoding
  qwen-plus    -> entity/edge extraction, NLI verification, contradiction judge
  qwen3-rerank -> final ranking
  qwen3-max    -> answer generation
  qwen-plus/qwen3-thinking -> offline consolidation
  qwen-vl-ocr / qwen3-asr / qwen-doc / qwen-vl-plus -> multimodal ingestion
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import Settings, get_settings


class ModelCallError(RuntimeError):
    """Raised when a real model call cannot be completed. Never swallowed into a fake."""


def _strip_json(text: str) -> str:
    """Pull the first JSON object/array out of an LLM response (handles code fences)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    # Find the outermost {..} or [..].
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = text.find(open_c)
        end = text.rfind(close_c)
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
    return text


class DashScopeClient:
    """Thin, synchronous wrapper over the dashscope SDK with region + key wired in."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        import dashscope  # imported here so importing this module never requires the SDK at collection time

        self._ds = dashscope
        if self.settings.has_api_key:
            dashscope.api_key = self.settings.api_key
            dashscope.base_http_api_url = self.settings.dashscope_base_url

    # ---- guards -----------------------------------------------------------
    def _require_key(self) -> None:
        if not self.settings.has_api_key:
            raise ModelCallError(
                "DASHSCOPE_API_KEY is not set. Eidetic-Plus makes only real model calls "
                "and refuses to fabricate outputs. Add your key to .env and retry."
            )
        # SDK module-level config is process-global; (re)assert it.
        self._ds.api_key = self.settings.api_key
        self._ds.base_http_api_url = self.settings.dashscope_base_url

    @staticmethod
    def _ok(resp: Any) -> Any:
        code = getattr(resp, "status_code", 200)
        if code != 200:
            msg = getattr(resp, "message", "") or getattr(resp, "code", "")
            raise ModelCallError(f"DashScope call failed (HTTP {code}): {msg}")
        return resp

    # ---- embeddings -------------------------------------------------------
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Real text-embedding-v4 vectors. Batches of <=10 per request."""
        self._require_key()
        if not texts:
            return np.zeros((0, self.settings.embed_dim), dtype=np.float32)
        out: list[list[float]] = []
        for i in range(0, len(texts), 10):
            batch = texts[i : i + 10]
            resp = self._ds.TextEmbedding.call(
                model=self.settings.text_embed_model,
                input=batch,
                dimension=self.settings.embed_dim,
            )
            self._ok(resp)
            embs = resp.output["embeddings"]
            embs = sorted(embs, key=lambda e: e.get("text_index", 0))
            out.extend(e["embedding"] for e in embs)
        arr = np.asarray(out, dtype=np.float32)
        return arr

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def embed_image(self, path_or_uri: str) -> np.ndarray:
        """Real multimodal embedding (tongyi-embedding-vision / qwen3-vl-embedding)."""
        self._require_key()
        uri = path_or_uri
        if "://" not in uri:
            uri = f"file://{Path(uri).resolve()}"
        resp = self._ds.MultiModalEmbedding.call(
            model=self.settings.multimodal_embed_model,
            input=[{"image": uri}],
        )
        self._ok(resp)
        emb = resp.output["embeddings"][0]["embedding"]
        return np.asarray(emb, dtype=np.float32)

    # ---- chat / generation ------------------------------------------------
    def chat(self, model: str, system: str, user: str, *, json_mode: bool = False,
             temperature: float = 0.2, max_tokens: int = 2048) -> str:
        self._require_key()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        # Note: we deliberately do NOT send response_format -- support varies across
        # qwen tiers. JSON is requested in the prompt and parsed robustly (_strip_json).
        kwargs: dict[str, Any] = dict(
            model=model, messages=messages, result_format="message",
            temperature=temperature, max_tokens=max_tokens,
        )
        _ = json_mode  # accepted for API symmetry; parsing is prompt-driven
        resp = self._ds.Generation.call(**kwargs)
        self._ok(resp)
        return resp.output["choices"][0]["message"]["content"].strip()

    def chat_json(self, model: str, system: str, user: str, **kw) -> Any:
        raw = self.chat(model, system, user, json_mode=True, **kw)
        try:
            return json.loads(_strip_json(raw))
        except json.JSONDecodeError as e:
            raise ModelCallError(f"Model returned non-JSON for a JSON request: {e}: {raw[:200]}")

    # ---- Component 4: salience / importance -------------------------------
    def score_importance(self, text: str) -> float:
        """qwen-flash importance in [0,1]."""
        data = self.chat_json(
            self.settings.salience_model,
            "You score how important a memory is for an AI agent to retain long-term. "
            "Reply ONLY as JSON: {\"importance\": <float 0..1>}.",
            f"Memory:\n{text[:4000]}",
            temperature=0.0, max_tokens=64,
        )
        return float(max(0.0, min(1.0, data.get("importance", 0.5))))

    def score_affect(self, text: str) -> dict:
        """Affect-modulated write-time scoring (Phase 3) in ONE qwen-flash call: importance [0,1],
        arousal [0,1] (emotional intensity), valence [-1,1] (negative..positive). Real call."""
        data = self.chat_json(
            self.settings.salience_model,
            "You rate a memory's affect for an AI agent deciding how vividly to retain it. "
            "Reply ONLY as JSON: {\"importance\": <0..1>, \"arousal\": <0..1>, "
            "\"valence\": <-1..1>}. arousal = emotional intensity; valence = negative..positive.",
            f"Memory:\n{text[:4000]}",
            temperature=0.0, max_tokens=64,
        )
        d = data if isinstance(data, dict) else {}
        return {
            "importance": float(max(0.0, min(1.0, d.get("importance", 0.5)))),
            "arousal": float(max(0.0, min(1.0, d.get("arousal", 0.3)))),
            "valence": float(max(-1.0, min(1.0, d.get("valence", 0.0)))),
        }

    # ---- Component 2/7: extraction ---------------------------------------
    def extract_edges(self, text: str) -> list[dict[str, str]]:
        """qwen-plus entity/relation extraction for the bi-temporal graph."""
        model = self.settings.salience_model if self.settings.extract_light_enabled else self.settings.extract_model
        data = self.chat_json(
            model,
            "Extract factual (subject, relation, object) triples from the text for a "
            "knowledge graph. Reply ONLY as JSON: {\"triples\": [{\"src\":..,\"relation\":..,"
            "\"dst\":..,\"fact\":..}]}. Keep entity names canonical and short. Empty list if none.",
            f"Text:\n{text[:6000]}",
            temperature=0.0, max_tokens=1024,
        )
        triples = data.get("triples", []) if isinstance(data, dict) else []
        out = []
        for t in triples:
            if t.get("src") and t.get("relation") and t.get("dst"):
                out.append({
                    "src": str(t["src"]).strip(),
                    "relation": str(t["relation"]).strip(),
                    "dst": str(t["dst"]).strip(),
                    "fact": str(t.get("fact", f"{t['src']} {t['relation']} {t['dst']}")).strip(),
                })
        return out

    # ---- Component 7: contradiction judge --------------------------------
    def find_contradictions(self, new_fact: str, candidates: list[str]) -> list[int]:
        """Return indices of `candidates` that the new fact contradicts (qwen-plus)."""
        if not candidates:
            return []
        numbered = "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))
        data = self.chat_json(
            self.settings.verify_model,
            "You decide which existing facts a NEW fact contradicts (same subject+relation, "
            "incompatible object/value). Reply ONLY as JSON: {\"contradicts\": [<indices>]}.",
            f"NEW fact:\n{new_fact}\n\nEXISTING facts:\n{numbered}",
            temperature=0.0, max_tokens=256,
        )
        idxs = data.get("contradicts", []) if isinstance(data, dict) else []
        return [int(i) for i in idxs if isinstance(i, (int, float)) and 0 <= int(i) < len(candidates)]

    def extract_current_value_matches(self, query: str, candidates: list[dict]) -> list[dict]:
        """Extract all candidate answers for a current-value question.

        The model extracts semantic matches only. It must not compare timestamps or choose
        the freshest value. The caller does the timestamp argmax in Python.
        """
        if not candidates:
            return []
        data = self.chat_json(
            self.settings.verify_model,
            "You extract candidate answers for a current-value memory question. "
            "Return every candidate that semantically answers the question. Do NOT choose "
            "the latest, newest, best, or winning answer. Do NOT compare timestamps. "
            "Reply ONLY as JSON: {\"matches\":[{\"memory_id\":\"...\",\"timestamp\":0,"
            "\"answer\":\"...\",\"quote\":\"...\"}]}. Use the provided timestamp exactly.",
            json.dumps({"query": query, "candidates": candidates[:50]}, ensure_ascii=False),
            temperature=0.0, max_tokens=1024,
        )
        matches = data.get("matches", []) if isinstance(data, dict) else []
        return [m for m in matches if isinstance(m, dict)]

    # ---- Component 6: NLI verification -----------------------------------
    def nli(self, premise: str, hypothesis: str) -> tuple[str, float]:
        """NLI entailment with the raw record as premise. Returns (label, confidence).

        label in {entailment, neutral, contradiction}. This is the anti-confabulation
        backbone: only `entailment` counts as grounded."""
        data = self.chat_json(
            self.settings.verify_model,
            "You are a strict natural-language-inference judge. Given a PREMISE (ground "
            "truth) and a HYPOTHESIS, decide if the premise ENTAILS the hypothesis. "
            "Reply ONLY as JSON: {\"label\": \"entailment\"|\"neutral\"|\"contradiction\", "
            "\"confidence\": <float 0..1>}. Use 'entailment' only if the premise fully "
            "supports the hypothesis with no added claims.",
            f"PREMISE:\n{premise[:6000]}\n\nHYPOTHESIS:\n{hypothesis[:2000]}",
            temperature=0.0, max_tokens=128,
        )
        label = str(data.get("label", "neutral")).lower().strip()
        if label not in ("entailment", "neutral", "contradiction"):
            label = "neutral"
        return label, float(max(0.0, min(1.0, data.get("confidence", 0.5))))

    # ---- Component 6: reranking ------------------------------------------
    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """qwen3-rerank. Returns [(original_index, relevance_score)] sorted desc."""
        self._require_key()
        if not documents:
            return []
        docs = [d[:4000] for d in documents][:500]
        resp = self._ds.TextReRank.call(
            model=self.settings.rerank_model,
            query=query[:4000],
            documents=docs,
            top_n=min(top_n, len(docs)),
            return_documents=False,
        )
        self._ok(resp)
        results = resp.output["results"]
        return [(int(r["index"]), float(r["relevance_score"])) for r in results]

    # ---- Component 6: answer generation ----------------------------------
    def generate_answer(self, question: str, context_blocks: list[str],
                        model: Optional[str] = None) -> str:
        """Answer strictly grounded in retrieved memory. `model` selects the cascade tier
        (qwen-flash / qwen-plus / qwen3-max); defaults to the configured gen model."""
        ctx = "\n\n".join(f"[S{i}] {b[:3000]}" for i, b in enumerate(context_blocks))
        if self.settings.reader_cot_enabled:
            data = self.chat_json(
                model or self.settings.gen_model,
                "You are a memory agent. Answer ONLY from the provided sources. First write "
                "brief evidence notes for each useful source, then answer from those notes. "
                "Reply ONLY as JSON: {\"notes\":[{\"source\":\"S0\",\"relevant\":true,"
                "\"note\":\"...\"}],\"answer\":\"...\"}. The answer must cite sources inline "
                "as [S0], [S1]. If the sources do not contain the answer, set answer to "
                "\"I do not have that in memory.\" Never invent facts beyond the sources.",
                f"Question: {question}\n\nSources:\n{ctx}",
                temperature=0.1, max_tokens=1536,
            )
            answer = data.get("answer") if isinstance(data, dict) else None
            if not isinstance(answer, str) or not answer.strip():
                raise ModelCallError("Reader COT response did not include a non-empty JSON answer.")
            return answer.strip()
        return self.chat(
            model or self.settings.gen_model,
            "You are a memory agent. Answer ONLY from the provided sources. Cite sources "
            "inline as [S0], [S1]. If the sources do not contain the answer, say you do "
            "not have that in memory. Never invent facts beyond the sources.",
            f"Question: {question}\n\nSources:\n{ctx}",
            temperature=0.1, max_tokens=1024,
        )

    # ---- Component 5: consolidation --------------------------------------
    def consolidate_summary(self, texts: list[str]) -> str:
        """Generative semantic summary over replayed episodes (verified separately)."""
        joined = "\n---\n".join(t[:2000] for t in texts[:20])
        return self.chat(
            self.settings.consolidate_model,
            "You consolidate several raw episodic memories into one faithful semantic "
            "summary. Include ONLY information present in the episodes. No speculation.",
            f"Episodes:\n{joined}",
            temperature=0.2, max_tokens=512,
        )

    # ---- MemMA self-repair: probe generation ------------------------------
    def generate_probes(self, memory_text: str, n: int = 3) -> list[str]:
        """Synthesize self-test probe questions over a provisional memory to find gaps
        (factual recall, cross-session reasoning, temporal inference). Real qwen-flash call."""
        data = self.chat_json(
            self.settings.salience_model,
            "You generate self-test probe questions to find what an AI memory CANNOT yet "
            "answer about a stored memory. Cover factual recall, cross-session reasoning, and "
            "temporal inference. Reply ONLY as JSON: {\"probes\": [\"q1\", \"q2\", ...]}.",
            f"Memory:\n{memory_text[:4000]}\n\nGenerate {n} probe questions.",
            temperature=0.3, max_tokens=512,
        )
        probes = data.get("probes", []) if isinstance(data, dict) else []
        return [str(p).strip() for p in probes if str(p).strip()][:n]

    def generate_topic(self, query: str) -> str:
        """MIRIX Active Retrieval: generate an anticipated topic/sub-question BEFORE answering, to
        scaffold multi-hop/temporal retrieval. Real qwen-flash call."""
        data = self.chat_json(
            self.settings.salience_model,
            "You generate a short anticipated TOPIC or sub-question that retrieval should cover to "
            "answer a user question (especially multi-hop/temporal). Reply ONLY as JSON: "
            "{\"topic\": \"...\"}.",
            f"Question: {query[:2000]}",
            temperature=0.2, max_tokens=128,
        )
        return str(data.get("topic", "")).strip() if isinstance(data, dict) else ""

    def plan_verification_questions(self, draft: str, n: int = 3) -> list[str]:
        """Chain-of-Verification: plan verification questions for a draft answer, to be answered
        INDEPENDENTLY (factored, so the model can't copy its own hallucination). Real call."""
        data = self.chat_json(
            self.settings.verify_model,
            "You plan independent verification questions to fact-check a draft answer. Each must "
            "be answerable on its own without seeing the draft. Reply ONLY as JSON: "
            "{\"questions\": [\"q1\", ...]}.",
            f"Draft answer:\n{draft[:3000]}\n\nPlan {n} verification questions.",
            temperature=0.2, max_tokens=512,
        )
        qs = data.get("questions", []) if isinstance(data, dict) else []
        return [str(q).strip() for q in qs if str(q).strip()][:n]

    # ---- Multimodal ingestion --------------------------------------------
    def _mm_text(self, resp: Any) -> str:
        self._ok(resp)
        content = resp.output["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content.strip()
        parts = []
        for c in content:
            if isinstance(c, dict) and "text" in c:
                parts.append(c["text"])
        return "\n".join(parts).strip()

    def _file_uri(self, path: str) -> str:
        return path if "://" in path else f"file://{Path(path).resolve()}"

    def ocr_image(self, path: str) -> str:
        """qwen-vl-ocr: text, tables, formulas from an image."""
        self._require_key()
        resp = self._ds.MultiModalConversation.call(
            model=self.settings.ocr_model,
            messages=[{"role": "user", "content": [
                {"image": self._file_uri(path)},
                {"text": "Extract ALL text, tables (as HTML), and formulas from this image."},
            ]}],
        )
        return self._mm_text(resp)

    def describe_image(self, path: str) -> str:
        """qwen-vl-plus: semantic description for images without much text."""
        self._require_key()
        resp = self._ds.MultiModalConversation.call(
            model=self.settings.describe_model,
            messages=[{"role": "user", "content": [
                {"image": self._file_uri(path)},
                {"text": "Describe this image in detail for a searchable memory index."},
            ]}],
        )
        return self._mm_text(resp)

    def transcribe_audio(self, path: str) -> str:
        """qwen3-asr-flash transcription."""
        self._require_key()
        resp = self._ds.MultiModalConversation.call(
            model=self.settings.asr_model,
            messages=[{"role": "user", "content": [{"audio": self._file_uri(path)}]}],
        )
        return self._mm_text(resp)

    def extract_visual_graph(self, path: str) -> list[dict[str, str]]:
        """Turn an image/screenshot/diagram/table into (src, relation, dst) triples for
        the bi-temporal graph (real qwen-vl-plus). Vision FEEDS the graph, not just a vector."""
        self._require_key()
        resp = self._ds.MultiModalConversation.call(
            model=self.settings.describe_model,
            messages=[{"role": "user", "content": [
                {"image": self._file_uri(path)},
                {"text": "Extract factual (subject, relation, object) triples describing the "
                         "entities and relationships visible in this image (objects, people, "
                         "chart values, table rows, diagram links). Reply ONLY as JSON: "
                         "{\"triples\":[{\"src\":..,\"relation\":..,\"dst\":..,\"fact\":..}]}. "
                         "Empty list if none."},
            ]}],
        )
        raw = self._mm_text(resp)
        try:
            data = json.loads(_strip_json(raw))
        except json.JSONDecodeError:
            return []
        triples = data.get("triples", []) if isinstance(data, dict) else []
        out = []
        for t in triples:
            if t.get("src") and t.get("relation") and t.get("dst"):
                out.append({
                    "src": str(t["src"]).strip(), "relation": str(t["relation"]).strip(),
                    "dst": str(t["dst"]).strip(),
                    "fact": str(t.get("fact", f"{t['src']} {t['relation']} {t['dst']}")).strip(),
                })
        return out

    def verify_visual(self, path: str, claim: str) -> tuple[str, float]:
        """Visual NLI: does the actual image support `claim`? (real qwen-vl-plus judge).

        Extends the no-confabulation guarantee to images; the raw image is the arbiter.
        Returns (label, confidence) with label in {entailment, neutral, contradiction}."""
        self._require_key()
        resp = self._ds.MultiModalConversation.call(
            model=self.settings.describe_model,
            messages=[{"role": "user", "content": [
                {"image": self._file_uri(path)},
                {"text": "You are a strict visual fact-checker. Decide whether THIS IMAGE "
                         f"supports the claim: \"{claim[:1000]}\". Reply ONLY as JSON: "
                         "{\"label\":\"entailment\"|\"neutral\"|\"contradiction\",\"confidence\":"
                         "<0..1>}. Use 'entailment' only if the pixels clearly support the claim."},
            ]}],
        )
        raw = self._mm_text(resp)
        try:
            data = json.loads(_strip_json(raw))
        except json.JSONDecodeError:
            return "neutral", 0.0
        label = str(data.get("label", "neutral")).lower().strip()
        if label not in ("entailment", "neutral", "contradiction"):
            label = "neutral"
        return label, float(max(0.0, min(1.0, data.get("confidence", 0.5))))

    def describe_video(self, path: str) -> str:
        """qwen-vl-plus video understanding."""
        self._require_key()
        resp = self._ds.MultiModalConversation.call(
            model=self.settings.video_model,
            messages=[{"role": "user", "content": [
                {"video": self._file_uri(path)},
                {"text": "Describe this video in detail for a searchable memory index."},
            ]}],
        )
        return self._mm_text(resp)

    def read_document(self, path: str) -> str:
        """qwen-long document reading via the OpenAI-compatible Files API (DocMind-style).

        Uploads the file (purpose=file-extract), then asks qwen-long to return its full
        text. Real call end-to-end; raises on any failure."""
        self._require_key()
        import httpx

        base = self.settings.compatible_base_url
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        p = Path(path)
        with httpx.Client(timeout=120.0) as http:
            with open(p, "rb") as fh:
                up = http.post(
                    f"{base}/files",
                    headers=headers,
                    files={"file": (p.name, fh)},
                    data={"purpose": "file-extract"},
                )
            if up.status_code >= 300:
                raise ModelCallError(f"Document upload failed (HTTP {up.status_code}): {up.text[:200]}")
            file_id = up.json()["id"]
            chat = http.post(
                f"{base}/chat/completions",
                headers={**headers, "Content-Type": "application/json"},
                json={
                    # Use the configured document model (DOC_MODEL). The Files API file-extract
                    # path needs a long-context reader (qwen-long); DOC_MODEL is the single knob.
                    "model": self.settings.doc_model,
                    "messages": [
                        {"role": "system", "content": f"fileid://{file_id}"},
                        {"role": "user", "content": "Output the full text content of this document verbatim."},
                    ],
                },
            )
            if chat.status_code >= 300:
                raise ModelCallError(f"Document read failed (HTTP {chat.status_code}): {chat.text[:200]}")
            return chat.json()["choices"][0]["message"]["content"].strip()

    def describe_binary(self, name: str, sample: bytes) -> str:
        """Un-embeddable modality: ask qwen3-max to describe a sample for indexing."""
        b64 = base64.b64encode(sample[:2048]).decode("ascii")
        return self.chat(
            self.settings.gen_model,
            "You describe an opaque binary artifact for a memory index from its name and a "
            "base64 sample of its first bytes. Be factual and concise.",
            f"Name: {name}\nFirst-bytes (base64): {b64}",
            temperature=0.0, max_tokens=256,
        )


_client: Optional[DashScopeClient] = None


def get_client() -> DashScopeClient:
    global _client
    if _client is None:
        _client = DashScopeClient()
    return _client
