"""
Parse the Armenian and English PDFs of the Law on Electronic Communications
into article-level records, and reconcile the two by article number.

Design decision (locked in after inspecting the actual PDFs): the two
documents are NOT a matched translation pair -- the English PDF reflects
amendments (e.g. Article 4 point 10 repealed by HO-160-N, Article 17.1
added by HO-207-N) that are absent from the Armenian PDF body text.

Armenian is therefore the sole source of truth / indexing backbone.
English text is attached only where the article number aligns cleanly.
Anything that doesn't align is flagged, not guessed at.
"""

import re
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pdfplumber

ARTICLE_RE_HY = re.compile(r'^Հոդված\s+(\d+\.?\d*)\.\s*(.*)$')
ARTICLE_RE_EN = re.compile(r'^Article\s+(\d+\.?\d*)\.\s*(.*)$')

# Chapter headers are spaced-out caps in both PDFs, e.g. "Գ Լ ՈՒ Խ 1" / "C H A P T E R 1"
CHAPTER_RE_HY = re.compile(r'^Գ\s*Լ\s*ՈՒ\s*Խ\s+(\d+)')
CHAPTER_RE_EN = re.compile(r'^C\s*H\s*A\s*P\s*T\s*E\s*R\s+(\d+)')


@dataclass
class Article:  
    number: str
    chapter: str | None = None
    title_hy: str | None = None
    text_hy: str = ""
    title_en: str | None = None
    text_en: str | None = None


def extract_text(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def split_articles(full_text: str, article_re: re.Pattern, chapter_re: re.Pattern) -> list[Article]:
    """Walk the text line by line. A line matching article_re starts a new
    article; everything until the next match is that article's body."""
    articles: list[Article] = []
    current: Article | None = None
    current_chapter: str | None = None

    for raw_line in full_text.split("\n"):
        line = raw_line.strip()

        if m := chapter_re.match(line):
            current_chapter = m.group(1)
            continue

        if m := article_re.match(line):
            if current:
                articles.append(current)
            current = Article(number=m.group(1), chapter=current_chapter, title_hy=m.group(2))
        elif current:
            current.text_hy += raw_line + "\n"

    if current:
        articles.append(current)
    return articles


def reconcile(hy_articles: list[Article], en_articles: list[Article]) -> tuple[list[Article], dict]:
    """Attach English text to the matching Armenian article by number.
    Never invent an alignment -- report what doesn't match instead."""
    en_by_num = {a.number: a for a in en_articles}
    hy_numbers = {a.number for a in hy_articles}

    hy_only, en_only = [], []

    for a in hy_articles:
        match = en_by_num.get(a.number)
        if match:
            a.title_en = match.title_hy  # field reused for the EN title line
            a.text_en = match.text_hy
        else:
            hy_only.append(a.number)

    en_only = sorted(set(en_by_num) - hy_numbers, key=lambda n: [int(p) for p in n.split(".")])

    report = {
        "total_hy_articles": len(hy_articles),
        "total_en_articles": len(en_articles),
        "aligned": len(hy_articles) - len(hy_only),
        "hy_only": hy_only,   # Armenian articles with no English counterpart found
        "en_only": en_only,   # English articles with no Armenian counterpart (e.g. later amendments)
    }
    return hy_articles, report


def run(hy_pdf: str, en_pdf: str, out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    hy_text = extract_text(hy_pdf)
    en_text = extract_text(en_pdf)

    hy_articles = split_articles(hy_text, ARTICLE_RE_HY, CHAPTER_RE_HY)
    en_articles = split_articles(en_text, ARTICLE_RE_EN, CHAPTER_RE_EN)

    merged, report = reconcile(hy_articles, en_articles)

    with open(out / "articles.jsonl", "w", encoding="utf-8") as f:
        for a in merged:
            f.write(json.dumps(asdict(a), ensure_ascii=False) + "\n")

    with open(out / "reconciliation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    report = run(
        hy_pdf="ingest/law_hy.pdf",
        en_pdf="ingest/law_en.pdf",
        out_dir="data/processed",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
