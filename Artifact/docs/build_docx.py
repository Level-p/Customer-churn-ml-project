"""Build the Word version of the lab manual.

Two problems the naive `pandoc manual.md -o manual.docx` does not solve:

1. The hand-written Table of Contents uses GitHub-style anchors, which do not
   survive the docx writer. It is stripped and replaced with a native Word TOC
   field, which Word will paginate and keep up to date itself.

2. Several screenshots are full-page captures -- the model card is 2880 x 7596
   pixels. Scaled to the page width of an A4 document that is roughly seventeen
   inches tall, which Word cannot lay out sensibly. Any figure taller than 1.45x
   its width is therefore sliced into parts, each of which is given an explicit
   width so that it fits inside one page.
"""

from __future__ import annotations

import re
from pathlib import Path

import pypandoc
from PIL import Image

ROOT = Path(r"c:\Users\cpati\Downloads\B01042273\Artifact")
DOCS = ROOT / "docs"
SRC = DOCS / "LAB_MANUAL.md"
OUT = DOCS / "LAB_MANUAL.docx"
SPLIT_DIR = DOCS / "figures" / "print"

# A4 with 1 inch margins: 6.27in of text width, ~9.7in of text height. Leave room
# for the caption underneath, so cap a figure at 8.4in tall.
MAX_W_IN = 6.2
MAX_H_IN = 8.4
MAX_RATIO = MAX_H_IN / MAX_W_IN  # 1.35 -- beyond this a figure gets sliced
OVERLAP_PX = 60  # slices overlap slightly so no row of text is lost on a cut


def slice_image(path: Path) -> list[Path]:
    """Cut an over-tall screenshot into page-sized horizontal bands."""
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as image:
        width, height = image.size
        ratio = height / width
        if ratio <= MAX_RATIO:
            return [path]

        parts = int(-(-ratio // MAX_RATIO))  # ceiling division
        band = height // parts
        outputs: list[Path] = []
        for index in range(parts):
            top = max(0, index * band - (OVERLAP_PX if index else 0))
            bottom = min(height, (index + 1) * band + OVERLAP_PX)
            crop = image.crop((0, top, width, bottom))
            target = SPLIT_DIR / f"{path.stem}-{chr(ord('a') + index)}.png"
            crop.save(target, optimize=True)
            outputs.append(target)
        return outputs


def width_for(path: Path) -> float:
    """Widest this figure can be drawn without exceeding the page height."""
    with Image.open(path) as image:
        width, height = image.size
    by_height = MAX_H_IN * (width / height)
    return round(min(MAX_W_IN, by_height), 2)


IMAGE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<path>figures/[^)]+)\)\s*$", re.MULTILINE)


def rewrite_images(text: str) -> str:
    """Replace each figure with sized -- and if necessary, sliced -- versions."""

    def replace(match: re.Match) -> str:
        alt = match.group("alt")
        source = DOCS / match.group("path")
        pieces = slice_image(source)

        lines = []
        for index, piece in enumerate(pieces):
            relative = piece.relative_to(DOCS).as_posix()
            suffix = f" (part {index + 1} of {len(pieces)})" if len(pieces) > 1 else ""
            lines.append(
                f"![{alt}{suffix}]({relative}){{width={width_for(piece)}in}}"
            )
        if len(pieces) > 1:
            print(f"  {source.name}: sliced into {len(pieces)} parts")
        return "\n\n".join(lines)

    return IMAGE.sub(replace, text)


def strip_toc(text: str) -> str:
    """Drop the hand-written contents list; Word will build its own."""
    start = text.index("## Table of Contents")
    end = text.index("## 1. Introduction")
    return text[:start] + text[end:]


METADATA = """---
title: "Retention BDSS --- Lab Manual"
subtitle: "A Business Decision-Support System for Interpretable Churn Prediction"
date: "July 2026"
lang: en-GB
toc: true
toc-depth: 3
---

"""


def main() -> None:
    text = SRC.read_text(encoding="utf-8")

    # The title block at the top of the markdown is replaced by real document
    # metadata, so Word gets a proper title page rather than a big heading.
    text = text.split("---\n\n## Table of Contents", 1)
    header, body = text[0], "## Table of Contents" + text[1]
    body = strip_toc(body)
    body = rewrite_images(body)

    # Keep the stack/paper lines from the original header as an opening block.
    preamble = "\n".join(
        line for line in header.splitlines()
        if line.startswith("**")
    )

    document = METADATA + preamble + "\n\n" + body
    staged = DOCS / "_docx_source.md"
    staged.write_text(document, encoding="utf-8")

    pypandoc.convert_file(
        str(staged),
        "docx",
        format="markdown+pipe_tables+link_attributes+fenced_code_blocks",
        outputfile=str(OUT),
        extra_args=[
            "--toc",
            "--toc-depth=3",
            f"--resource-path={DOCS}",
            "--highlight-style=tango",
            "-V", "papersize=a4",
        ],
    )
    staged.unlink()
    print(f"\nwrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
