#!/usr/bin/env python3
"""Methods document LaTeX -> GitHub Markdown: figures as PNG images followed
by a "Figure N." caption paragraph, table captions as "Table N." paragraphs,
\\ref replaced by the numbers of the LaTeX .aux file, numbered headings, then
pandoc to GFM with $...$ math."""
import re
import subprocess
import sys


def labels(aux):
    out = {}
    for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", open(aux).read()):
        out[m.group(1)] = m.group(2)
    return out


def balanced(s, i):
    """index after the {...} group starting at s[i] == '{'"""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError("unbalanced")


def caption_of(block):
    i = block.find(r"\caption{")
    if i < 0:
        return "", block
    j = balanced(block, i + len(r"\caption"))
    return block[i + len(r"\caption{"):j - 1], block[:i] + block[j:]


def main(tex, aux, out):
    lab = labels(aux)
    s = open(tex).read()
    s = s.replace(r"\fitbox{", "{")
    # figures
    def fig(m):
        block = m.group(1)
        cap, _ = caption_of(block)
        g = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{methods/([a-z_]+)\.pdf\}", block)
        lb = re.search(r"\\label\{([^}]*)\}", block)
        num = lab.get(lb.group(1), "") if lb else ""
        cap = re.sub(r"\\label\{[^}]*\}", "", cap)
        imgs = "".join(f"\n\n\\includegraphics{{methods/{x}.png}}\n\n" for x in g)
        return f"{imgs}\\noindent\\textbf{{Figure {num}.}} {cap}\n\n"
    s = re.sub(r"\\begin\{figure\}(?:\[[^]]*\])?(.*?)\\end\{figure\}", fig, s, flags=re.S)
    # tables: the caption becomes a paragraph after the tabular
    def tab(m):
        block = m.group(1)
        cap, rest = caption_of(block)
        lb = re.search(r"\\label\{([^}]*)\}", block)
        num = lab.get(lb.group(1), "") if lb else ""
        rest = re.sub(r"\\label\{[^}]*\}", "", rest)
        rest = rest.replace(r"\centering", "").replace(r"\small", "")
        return f"\n\n{rest}\n\n\\noindent\\textbf{{Table {num}.}} {cap}\n\n"
    s = re.sub(r"\\begin\{table\}(?:\[[^]]*\])?(.*?)\\end\{table\}", tab, s, flags=re.S)
    # references -> numbers
    s = re.sub(r"Figures~\\ref\{([^}]*)\}--\\ref\{([^}]*)\}",
               lambda m: f"Figures {lab.get(m.group(1), '?')}--{lab.get(m.group(2), '?')}", s)
    s = re.sub(r"\\ref\{([^}]*)\}", lambda m: lab.get(m.group(1), "?"), s)
    s = s.replace("~", " ")
    tmp = out + ".tex"
    open(tmp, "w").write(s)
    md = subprocess.run(["pandoc", "-f", "latex", "-t", "gfm+tex_math_dollars-tex_math_gfm", "--wrap=none", "-N",
                         "--shift-heading-level-by=1", tmp], capture_output=True, text=True, check=True).stdout
    title = ("# PCA robust to close relatives in PCAone: the `--robust` methods\n\n"
             "*relatePCA working notes. Markdown version of "
             "[pcaone_robust_methods.pdf](pcaone_robust_methods.pdf); figures in [methods/](methods/). "
             "Talk: [slides/relatedness_pca.pdf](slides/relatedness_pca.pdf).*\n\n")
    # numbered headings as in the PDF (\subsection* stays unnumbered)
    unnumbered = {m.strip() for m in re.findall(r"\\subsection\*\{([^}]*)\}", open(tex).read())}
    lines, sec, sub = [], 0, 0
    for line in md.split("\n"):
        if line.startswith("## "):
            sec, sub = sec + 1, 0
            line = f"## {sec} {line[3:]}"
        elif line.startswith("### ") and line[4:].strip() not in unnumbered:
            sub += 1
            line = f"### {sec}.{sub} {line[4:]}"
        lines.append(line)
    md = "\n".join(lines)
    # display math on lines of its own (GitHub renders $$ blocks reliably only there)
    md = re.sub(r"[ \t]*\$\$(.+?)\$\$[ \t]*", lambda m: "\n\n$$\n" + m.group(1).strip() + "\n$$\n\n", md,
                flags=re.S)
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = re.sub(r"<div id=\"[^\"]*\">\n\n", "", md)
    md = md.replace("\n</div>\n", "\n")
    open(out, "w").write(title + md)


if __name__ == "__main__":
    main(*sys.argv[1:4])
