#!/usr/bin/env python3
"""Render the article PDF with Pandoc and Typst (Python standard library only).

The Markdown remains authoritative. The print edition uses two columns, a
full-width title and abstract, numbered equations, and linked references.
Run from any directory; optionally pass an output PDF path.
"""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
ARTICLE = ROOT / "docs/gcd-in-hf2q.md"
STYLE = Path(__file__).with_name("publication_style.typ")
BACK_MATTER = {"Acknowledgments", "Artifacts and reproducibility", "References"}


def words(node):
    if isinstance(node, list):
        return " ".join(filter(None, (words(child) for child in node)))
    if isinstance(node, dict):
        return node["c"] if node.get("t") == "Str" else words(node.get("c", []))
    return ""


def source_number(node):
    if isinstance(node, dict) and node.get("t") == "Note":
        match = re.search(r"Source\s+(\d+)", words(node["c"]))
        if match:
            return match.group(1)
    return None


def print_nodes(node):
    if isinstance(node, list):
        result = []
        index = 0
        while index < len(node):
            if source_number(node[index]):
                numbers = []
                while index < len(node) and (number := source_number(node[index])):
                    numbers.append(number)
                    index += 1
                # Move sentence punctuation after the reference group.
                punctuation = ""
                if result and isinstance(result[-1], dict) and result[-1].get("t") == "Str":
                    if result[-1]["c"].endswith((".", ";", ",")):
                        punctuation = result[-1]["c"][-1]
                        result[-1]["c"] = result[-1]["c"][:-1]
                result.extend([{"t": "Space"}, {"t": "Str", "c": "["}])
                for position, number in enumerate(numbers):
                    if position:
                        result.extend([{"t": "Str", "c": ","}, {"t": "Space"}])
                    result.append({"t": "Link", "c": [
                        ["", [], []], [{"t": "Str", "c": number}],
                        [f"#source-{number}", ""]]})
                result.append({"t": "Str", "c": "]" + punctuation})
            else:
                result.append(print_nodes(node[index]))
                index += 1
        return result
    if not isinstance(node, dict):
        return node
    node = {key: print_nodes(value) for key, value in node.items()}
    if node.get("t") == "Table":
        for column in node["c"][2]:
            column[0] = {"t": "AlignLeft"}
    return node


def raw(text):
    return {"t": "RawBlock", "c": ["typst", text]}


def caption(block):
    content = block["c"]
    if len(content) == 1 and content[0]["t"] == "Emph":
        content = content[0]["c"]
    return {"t": "Plain", "c": [{"t": "Strong", "c": content[:3]}] + content[3:]}


def prepare(document):
    document = print_nodes(document)
    blocks = document["blocks"]
    if blocks[0]["t"] != "Header" or words(blocks[2]) != "Abstract":
        raise ValueError("Expected article title, byline, and Abstract heading")
    title = words(blocks[0]["c"][2])
    author, affiliation = (part.strip() for part in words(blocks[1]).split("·", 1))
    result = [raw(f"#set document(title: {json.dumps(title)}, author: {json.dumps(author + ', ' + affiliation)})"),
              raw('#place(top + center, scope: "parent", float: true, clearance: 1.2em)['),
              raw(f"#paper-title({json.dumps(title)}, {json.dumps(author)}, {json.dumps(affiliation)})"),
              raw("#paper-abstract[")]
    index = 3
    while blocks[index]["t"] != "Header":
        result.append(blocks[index])
        index += 1
    result.append(raw("]]"))
    in_sources = False
    keep_back_matter = False
    while index < len(blocks):
        block = blocks[index]
        if block["t"] == "Header":
            if keep_back_matter:
                result.append(raw("]"))
                keep_back_matter = False
            title = words(block["c"][2])
            block["c"][0] -= 1
            if title in BACK_MATTER:
                if title in {"Acknowledgments", "Artifacts and reproducibility"}:
                    result.append(raw("#block(breakable: false)["))
                    keep_back_matter = True
                if title == "Artifacts and reproducibility":
                    result.append(raw("#set par(justify: false, first-line-indent: 0pt)"))
                result.extend([raw("#heading(level: 1, numbering: none)["),
                               {"t": "Plain", "c": block["c"][2]}, raw("]")])
                in_sources = title == "References"
                index += 1
                continue
        if in_sources and block["t"] == "OrderedList":
            for number, item in enumerate(block["c"][1], 1):
                item.insert(0, raw(f"#metadata(none) <source-{number}>"))
            result.append(raw('#set enum(numbering: "[1]")'))
        if (block["t"] == "Para" and len(block["c"]) == 1
                and block["c"][0]["t"] == "Image" and index + 1 < len(blocks)
                and words(blocks[index + 1]).startswith("Figure ")):
            result.extend([raw("#paper-figure(["), block, raw("], ["),
                           caption(blocks[index + 1]), raw("])")])
            index += 2
        elif (block["t"] == "Para" and words(block).startswith("Table ")
              and index + 1 < len(blocks) and blocks[index + 1]["t"] == "Table"):
            table = blocks[index + 1]
            number = int(re.match(r"Table (\d+)", words(block)).group(1))
            widths = {1: [.2, .4, .4], 2: [.25, .35, .4],
                      3: [.24, .34, .42], 4: [.22, .25, .13, .27, .13]}[number]
            for column, width in zip(table["c"][2], widths, strict=True):
                column[1] = {"t": "ColWidth", "c": width}
            wide = "true" if number in {3, 4} else "false"
            result.extend([raw("#paper-table(["), table, raw("], ["),
                           caption(block), raw(f"], wide: {wide})")])
            index += 2
        else:
            result.append(block)
            index += 1
    document["blocks"] = result
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=ARTICLE.with_suffix(".pdf"))
    args = parser.parse_args()
    for program in ["pandoc", "typst"]:
        if shutil.which(program) is None:
            parser.error(f"{program} is required on PATH")
    output = args.output.resolve()
    document = json.loads(subprocess.check_output([
        "pandoc", str(ARTICLE), "--from=markdown-implicit_figures", "--to=json",
    ], text=True))
    with tempfile.TemporaryDirectory(prefix="hf2q-gcd-pdf-") as directory:
        temp = Path(directory)
        metadata = temp / "metadata.json"
        metadata.write_text(json.dumps({
            "margin": {"x": "18mm", "y": "20mm"}, "papersize": "a4",
            "fontsize": "10pt", "mainfont": "Times New Roman",
            "codefont": "DejaVu Sans Mono", "mathfont": "STIX Two Math",
            "page-numbering": "1", "columns": 2,
        }))
        subprocess.run([
            "pandoc", "--from=json", "--pdf-engine=typst", "--resource-path=.",
            f"--metadata-file={metadata}", f"--include-before-body={STYLE}",
            f"--output={output}",
        ], input=json.dumps(prepare(document)), text=True, check=True, cwd=ARTICLE.parent)
    print(output)


if __name__ == "__main__":
    main()
