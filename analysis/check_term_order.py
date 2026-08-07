"""Flag any term that reaches the reader before the article defines it.

Written after shipping a draft that used "joint model" as a table heading one
whole section before defining it. That slipped through review because the phrase
does not appear in index.html at all - it is injected by widgets.js, so reading
the HTML top to bottom never shows it.

So position is computed the way a reader experiences it: prose is located by its
line in index.html, and widget text inherits the line of the element it writes
into.

    python analysis/check_term_order.py
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: term (prefix "CS:" to match case-sensitively)  ->  literal text that defines it
CHECKS = [
    (r"\btank rate\b|\btanking\b", "a rate of losing on purpose"),
    (r"\bElo\b", "Elo, in ninety seconds"),
    (r"\bsequential test\b|\bSPRT\b", "score each match as it arrives"),
    (r"likelihood ratio", "the <b>log-likelihood ratio</b>"),
    (r"CS:\btau\*", "we'll call <b>tau*</b>"),
    (r"CS:\bH0\b|\bH1\b", "<b>H0</b>, innocent"),
    (r"\bwhale\b", "the casino word for"),
    (r"\btilter\b", "chases losses"),
    (r"\bjoint\b", "That is what a <b>joint distribution</b> is"),
    (r"\bDirichlet\b|phantom match", "phantom matches"),
    (r"\bposterior\b", "a cloud of parameter settings"),
    (r"\bMCMC\b|\bNUTS\b", "guided random walk"),
    (r"\bAUC\b", "draw one shark and one honest player at random"),
    (r"\brake\b|house cut", "the house keeps a"),
    (r"bet tier|tiers <b>min</b>", "tiers <b>min</b>"),
]

# A quoted run of JS source is not display text. Real captions are single-line
# prose; anything carrying these markers is code the regex accidentally spanned.
CODE_MARKERS = ("\n", ";", "{", "}", "function", "return", "=>", "var ")


def display_strings(block):
    """The quoted fragments in a widget block that actually reach the page."""
    found = re.findall(r"'([^'\n]{4,})'", block) + re.findall(r'"([^"\n]{4,})"', block)
    return " ".join(s for s in found if not any(m in s for m in CODE_MARKERS))


def load():
    html = (ROOT / "docs/index.html").read_text().split("\n")
    js = (ROOT / "docs/widgets.js").read_text()
    # Comments explain the code, not the article. Keep block headers for splitting.
    js = re.sub(r"/\*.*?\*/",
                lambda m: "/* ====== " if m.group().startswith("/* ======") else " ",
                js, flags=re.S)
    js = re.sub(r"(?m)^\s*//.*$", "", js)

    id_line = {}
    for i, line in enumerate(html, 1):
        for m in re.finditer(r'id="([^"]+)"', line):
            id_line.setdefault(m.group(1), i)

    widgets = []
    for block in re.split(r"/\* ====== ", js)[1:]:
        ids = re.findall(r"getElementById\(['\"]([^'\"]+)['\"]\)", block)
        lines = [id_line[i] for i in ids if i in id_line]
        if lines:
            widgets.append((min(lines), display_strings(block)))
    return html, widgets


def first_seen(term, html, widgets):
    flags = 0 if term.startswith("CS:") else re.I
    pattern = term[3:] if term.startswith("CS:") else term
    hits = [
        i for i, line in enumerate(html, 1)
        if re.search(pattern, re.sub(r"<!--.*?-->", "", line), flags)
        and not line.lstrip().startswith((".", "#", "@"))  # skip CSS rules
    ]
    hits += [ln for ln, text in widgets if re.search(pattern, text, flags)]
    return min(hits) if hits else None


def main():
    html, widgets = load()
    in_style = False
    body = []
    for line in html:
        if "<style>" in line:
            in_style = True
        if not in_style:
            body.append(line)
        else:
            body.append("")
        if "</style>" in line:
            in_style = False

    problems = 0
    print(f"{'term':40}{'1st seen':>9}{'defined':>9}   verdict")
    for term, anchor in CHECKS:
        seen = first_seen(term, body, widgets)
        defined = next((i for i, l in enumerate(html, 1) if anchor in l), None)
        if defined is None:
            print(f"{term[:38]:40}{'?':>9}{'MISSING':>9}   definition anchor not found")
            problems += 1
            continue
        ok = seen is not None and seen >= defined
        problems += not ok
        print(f"{term[:38]:40}{seen if seen else '-':>9}{defined:>9}   "
              f"{'ok' if ok else 'USED BEFORE DEFINED'}")

    print()
    print("clean" if not problems else f"{problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
