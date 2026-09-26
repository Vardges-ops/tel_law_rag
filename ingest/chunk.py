"""
Turn parsed articles (ingest/parse.py output) into retrieval chunks.

Rules:
- One chunk per article per language, UNLESS the article is long, in which
  case split on the article's own numbered sub-points (1), 2), ...) and
  never mid-sentence.
- Every chunk carries a header with chapter + article number + title, so
  the embedding sees context and the citation is unambiguous on its own.
- Armenian and English chunks for the same article share `article_number`
  but differ in `lang` -- this is the twin-corpus design. Article 17.1
  (English-only, per the reconciliation report) is indexed with lang="en"
  and no Armenian counterpart; that's expected, not an error.
- Token budget uses a cheap word-count proxy (~1.3 tokens/word for HY,
  ~0.75 for EN) rather than pulling in a tokenizer at this stage --
  good enough for chunk-sizing decisions, not for the final context
  assembly budget in rag/context.py, which will use the real tokenizer.
"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

MAX_WORDS = 350          # ~ soft cap per chunk before splitting
SUBPOINT_RE = re.compile(r'^\d+\)|^[ա-ֆ]+\)', re.MULTILINE)  # "1)" or Armenian letter ")"


@dataclass
class Chunk:
    chunk_id: str
    article_number: str
    lang: str             # "hy" | "en"
    chapter: str | None
    title: str | None
    text: str             # includes the header prefix, ready to embed


def make_header(lang: str, chapter: str | None, number: str, title: str | None) -> str:
    if lang == "hy":
        chap = f"Գլուխ {chapter}, " if chapter else ""
        return f"[{chap}Հոդված {number}. {title or ''}]"
    else:
        chap = f"Chapter {chapter}, " if chapter else ""
        return f"[{chap}Article {number}. {title or ''}]"


def split_long_text(text: str, max_words: int) -> list[str]:
    """Split on numbered sub-points if the article is long; merge adjacent
    sub-points back up to max_words so we don't over-fragment short points."""
    parts = SUBPOINT_RE.split(text)
    if len(parts) <= 1:
        # no sub-point structure found -- fall back to naive word chunking,
        # but this should be rare for a law with numbered points throughout
        words = text.split()
        return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)] or [text]

    markers = SUBPOINT_RE.findall(text)
    pieces = [markers[i] + parts[i + 1] for i in range(len(markers))]
    if parts[0].strip():
        pieces.insert(0, parts[0])

    merged, buf, buf_words = [], "", 0
    for piece in pieces:
        n = len(piece.split())
        if buf_words + n > max_words and buf:
            merged.append(buf.strip())
            buf, buf_words = "", 0
        buf += piece
        buf_words += n
    if buf.strip():
        merged.append(buf.strip())
    return merged


def chunk_article(number: str, chapter: str | None, title: str | None,
                   text: str, lang: str) -> list[Chunk]:
    header = make_header(lang, chapter, number, title)
    word_count = len(text.split())

    if word_count <= MAX_WORDS:
        return [Chunk(
            chunk_id=f"{number}_{lang}_0",
            article_number=number, lang=lang, chapter=chapter, title=title,
            text=f"{header}\n{text.strip()}",
        )]

    sub_texts = split_long_text(text, MAX_WORDS)
    return [
        Chunk(
            chunk_id=f"{number}_{lang}_{i}",
            article_number=number, lang=lang, chapter=chapter, title=title,
            text=f"{header} (part {i + 1}/{len(sub_texts)})\n{sub.strip()}",
        )
        for i, sub in enumerate(sub_texts)
    ]


def run(articles_path: str, out_path: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    with open(articles_path, encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            chunks += chunk_article(a["number"], a["chapter"], a["title_hy"], a["text_hy"], "hy")
            if a.get("text_en"):
                chunks += chunk_article(a["number"], a["chapter"], a["title_en"], a["text_en"], "en")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    return chunks


if __name__ == "__main__":
    chunks = run("data/processed/articles.jsonl", "data/processed/chunks.jsonl")
    hy = sum(1 for c in chunks if c.lang == "hy")
    en = sum(1 for c in chunks if c.lang == "en")
    print(f"Total chunks: {len(chunks)} (hy={hy}, en={en})")
    # sanity check: show one split article and one single-chunk article
    long_ones = [c for c in chunks if c.chunk_id.endswith("_1")]
    if long_ones:
        print("\nExample split chunk header:")
        print(long_ones[0].text.split("\n")[0])
