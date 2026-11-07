// Helper components for the technical documentation

#let callout(title, body, color: rgb("#1d4e89"), bg: rgb("#eef4fb"), icon: "") = {
  block(
    fill: bg,
    inset: (x: 12pt, y: 10pt),
    radius: 3pt,
    stroke: (left: 3pt + color),
    width: 100%,
    breakable: true,
    [
      #text(weight: "bold", fill: color)[#if icon != "" [#icon ] #title] \
      #v(0.3em)
      #text(size: 9.5pt)[#body]
    ],
  )
}

#let warning-box(title, body) = {
  callout(title, body, color: rgb("#c84b31"), bg: rgb("#fdf2f0"))
}

#let tip-box(title, body) = {
  callout(title, body, color: rgb("#2a9d8f"), bg: rgb("#eef8f6"))
}

#let status(state) = {
  let (fill, txt) = if state == "implemented" {
    (rgb("#e6f4ea"), text(size: 8pt, weight: "bold", fill: rgb("#1e7e34"))[✓ Implémenté])
  } else if state == "partial" {
    (rgb("#fff4e5"), text(size: 8pt, weight: "bold", fill: rgb("#b57c1e"))[◐ Partiel])
  } else if state == "development" {
    (rgb("#f3e8fd"), text(size: 8pt, weight: "bold", fill: rgb("#6f42c1"))[◆ Axe de développement])
  } else {
    (luma(240), text(size: 8pt)[#state])
  }
  box(
    fill: fill,
    inset: (x: 6pt, y: 3pt),
    radius: 2pt,
    outset: 0pt,
    baseline: 0%,
    txt,
  )
}
