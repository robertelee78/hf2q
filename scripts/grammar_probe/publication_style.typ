// Compact two-column ML-paper layout. Content lives in Markdown.
#set text(fill: black, hyphenate: true)
#set columns(gutter: 6mm)
#set par(justify: true, leading: 0.35em, spacing: 0.45em, first-line-indent: 1em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): set text(size: 11.5pt)
#show heading.where(level: 2): set text(size: 10pt)
#show heading: set block(above: 1em, below: 0.5em)
#set math.equation(numbering: "(1)")
#set table(inset: (x: 3pt, y: 3pt))
#show table: set text(size: 8pt, hyphenate: false)
#show table: set par(justify: false, first-line-indent: 0pt)
#show table.cell.where(y: 0): strong
#show table: it => block(stroke: (top: 0.5pt, bottom: 0.5pt), inset: (y: 2pt), it)
#show enum: set text(size: 8.5pt)
#show enum: set par(justify: false, first-line-indent: 0pt)
#set enum(spacing: 0.35em)
#show raw.where(block: false): set text(size: 7.7pt)
#show link: set text(fill: black)
#show figure.caption: set text(size: 8.5pt)
#show figure.caption: set par(first-line-indent: 0pt, justify: true)
#show figure.caption: it => align(left, it)
#set figure(gap: 6pt)

#let paper-title(title, author, affiliation) = {
  set par(first-line-indent: 0pt)
  align(center)[
    #v(0.4em)
    #text(size: 18pt, weight: "bold", title)
    #v(0.9em)
    #text(size: 12pt, author)
    #linebreak()
    #text(size: 11pt, affiliation)
    #v(0.8em)
  ]
}

#let paper-abstract(body) = block(
  inset: (x: 10mm), above: 0.4em, below: 0.4em, breakable: false,
)[
  #set text(size: 9.5pt)
  #set align(left)
  #set par(first-line-indent: 0pt, leading: 0.35em)
  #align(center)[#strong[Abstract]]
  #body
]

#let paper-figure(body, caption) = figure(
  body, caption: caption, kind: image, numbering: none,
  placement: top, scope: "parent",
)

#let paper-table(body, caption, wide: false) = figure(
  body, caption: caption, kind: table, numbering: none,
  placement: if wide { top } else { none },
  scope: if wide { "parent" } else { "column" },
)
