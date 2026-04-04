"""HTML and SVG rendering helpers for scientific-validation reports."""

from __future__ import annotations

from html import escape

from validation.scientific_validation import ScientificValidationReport, ValidationSeries

_PALETTE = (
    "#0f766e",
    "#1d4ed8",
    "#b45309",
    "#be123c",
    "#4c1d95",
    "#166534",
)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha:.3f})"


def _format_value(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _chart_svg(title: str, subtitle: str, series_specs: list[tuple[str, ValidationSeries, str]]) -> str:
    width = 920
    height = 300
    pad_left = 64
    pad_right = 18
    pad_top = 28
    pad_bottom = 44
    chart_width = width - pad_left - pad_right
    chart_height = height - pad_top - pad_bottom

    steps = [point.step for _, series, _ in series_specs for point in series.points]
    values = [
        value
        for _, series, _ in series_specs
        for point in series.points
        for value in (point.minimum, point.maximum)
    ]
    x_min = min(steps)
    x_max = max(steps)
    y_min = min(values)
    y_max = max(values)
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        padding = 1.0 if y_min == 0.0 else abs(y_min) * 0.1
        y_min -= padding
        y_max += padding

    def x_position(step: int) -> float:
        return pad_left + (step - x_min) * chart_width / (x_max - x_min)

    def y_position(value: float) -> float:
        return pad_top + chart_height - (value - y_min) * chart_height / (y_max - y_min)

    grid_lines: list[str] = []
    tick_labels: list[str] = []
    for tick_index in range(5):
        fraction = tick_index / 4
        y_value = y_max - fraction * (y_max - y_min)
        y = y_position(y_value)
        grid_lines.append(
            f'<line x1="{pad_left}" y1="{y:.2f}" x2="{width - pad_right}" y2="{y:.2f}" '
            'stroke="#d6dde7" stroke-width="1" />'
        )
        tick_labels.append(
            f'<text x="{pad_left - 10}" y="{y + 4:.2f}" text-anchor="end" fill="#4b5563" '
            'font-size="12" font-family="Menlo, monospace">'
            f'{escape(_format_value(y_value, 2))}</text>'
        )

    x_ticks: list[str] = []
    step_values = sorted(set(steps))
    target_ticks = 6
    if len(step_values) > target_ticks:
        stride = max(1, len(step_values) // (target_ticks - 1))
        step_values = step_values[::stride]
        if step_values[-1] != x_max:
            step_values.append(x_max)
    for step in step_values:
        x = x_position(step)
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{height - pad_bottom}" x2="{x:.2f}" y2="{height - pad_bottom + 6}" '
            'stroke="#4b5563" stroke-width="1" />'
        )
        x_ticks.append(
            f'<text x="{x:.2f}" y="{height - pad_bottom + 22}" text-anchor="middle" fill="#4b5563" '
            'font-size="12" font-family="Menlo, monospace">'
            f"{step}</text>"
        )

    series_paths: list[str] = []
    legend_items: list[str] = []
    for index, (label, series, color) in enumerate(series_specs):
        upper = " ".join(
            f"{x_position(point.step):.2f},{y_position(point.maximum):.2f}"
            for point in series.points
        )
        lower = " ".join(
            f"{x_position(point.step):.2f},{y_position(point.minimum):.2f}"
            for point in reversed(series.points)
        )
        mean_path = " ".join(
            ("M" if point_index == 0 else "L") + f" {x_position(point.step):.2f} {y_position(point.mean):.2f}"
            for point_index, point in enumerate(series.points)
        )
        series_paths.append(
            f'<polygon points="{upper} {lower}" fill="{_hex_to_rgba(color, 0.12)}" stroke="none" />'
        )
        series_paths.append(
            f'<path d="{mean_path}" fill="none" stroke="{color}" stroke-width="2.5" '
            'stroke-linecap="round" stroke-linejoin="round" />'
        )
        last_point = series.points[-1]
        series_paths.append(
            f'<circle cx="{x_position(last_point.step):.2f}" cy="{y_position(last_point.mean):.2f}" r="3.5" '
            f'fill="{color}" />'
        )
        legend_y = 18 + index * 18
        legend_items.append(
            f'<circle cx="{width - 210}" cy="{legend_y}" r="5" fill="{color}" />'
            f'<text x="{width - 198}" y="{legend_y + 4}" fill="#111827" font-size="12" '
            'font-family="Inter, system-ui, sans-serif">'
            f"{escape(label)}</text>"
        )

    return (
        f'<section class="chart-card">'
        f'<div class="chart-heading"><h3>{escape(title)}</h3><p>{escape(subtitle)}</p></div>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="18" fill="#ffffff" />'
        f'{"".join(grid_lines)}'
        f'<line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" '
        'stroke="#1f2937" stroke-width="1.4" />'
        f'{"".join(tick_labels)}'
        f'{"".join(x_ticks)}'
        f'{"".join(series_paths)}'
        f'{"".join(legend_items)}'
        f'<text x="{width / 2:.2f}" y="{height - 10}" text-anchor="middle" fill="#4b5563" '
        'font-size="12" font-family="Inter, system-ui, sans-serif">Sampled Simulation Step</text>'
        f'</svg></section>'
    )


def render_scientific_validation_report(report: ScientificValidationReport) -> str:
    """Render a scientific-validation report to a standalone HTML artifact."""

    charts = [
        (
            "Assembly Trajectory",
            "Mean/min/max assembly score across sampled validation trajectories.",
            [("Assembly Score", report.series_for_label("Assembly Score"), _PALETTE[0])],
        ),
        (
            "Atomistic Alignment Drift",
            "How far the coarse trajectory remains from the local 6M0J-derived atomistic centroid frame.",
            [("Atomistic Centroid RMSD", report.series_for_label("Atomistic Centroid RMSD"), _PALETTE[1])],
        ),
        (
            "Interface Contact Recovery",
            "Recovered reference-interface contacts after similarity alignment into the local atomistic frame.",
            [("Contact Recovery", report.series_for_label("Contact Recovery"), _PALETTE[2])],
        ),
        (
            "Shadow Energy Error",
            "Baseline vs shadow-corrected absolute energy error relative to the trusted target.",
            [
                ("Baseline", report.series_for_label("Energy Error Baseline"), _PALETTE[3]),
                ("Shadow", report.series_for_label("Energy Error Shadow"), _PALETTE[0]),
            ],
        ),
        (
            "Shadow Force RMS Error",
            "Baseline vs shadow-corrected force RMS error against the trusted target.",
            [
                ("Baseline", report.series_for_label("Force RMS Error Baseline"), _PALETTE[3]),
                ("Shadow", report.series_for_label("Force RMS Error Shadow"), _PALETTE[1]),
            ],
        ),
        (
            "Shadow Max Force Error",
            "Worst-component force error before and after the shadow correction path.",
            [
                ("Baseline", report.series_for_label("Max Force Error Baseline"), _PALETTE[3]),
                ("Shadow", report.series_for_label("Max Force Error Shadow"), _PALETTE[2]),
            ],
        ),
    ]
    if report.benchmark_series:
        charts.append(
            (
                "Architecture Timing",
                "Average seconds per sampled benchmark slice for the main architecture pathways.",
                [
                    (series.label.replace(" Time", ""), series, _PALETTE[index % len(_PALETTE)])
                    for index, series in enumerate(report.benchmark_series)
                ],
            )
        )

    benchmark_rows = "".join(
        "<tr>"
        f"<th>{escape(label.replace('_', ' ').title())}</th>"
        f"<td>{escape(_format_value(float(value), 6))}</td>"
        "</tr>"
        for label, value in report.summary.mean_benchmark_case_seconds.to_dict().items()
    )
    chart_html = "".join(
        _chart_svg(title, subtitle, series_specs)
        for title, subtitle, series_specs in charts
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{escape(report.title)} | {escape(report.scenario_name)}</title>
    <style>
      :root {{
        color-scheme: light;
        --paper: #f5f1e8;
        --ink: #101828;
        --muted: #526071;
        --card: #fffdf8;
        --line: #d7d1c7;
        --accent: #0f766e;
        --accent-2: #1d4ed8;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top right, rgba(29, 78, 216, 0.10), transparent 28%),
          radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 24%),
          linear-gradient(180deg, #fcfaf4 0%, var(--paper) 100%);
      }}
      main {{
        max-width: 1240px;
        margin: 0 auto;
        padding: 40px 24px 56px;
      }}
      .hero {{
        background: linear-gradient(135deg, rgba(255,255,255,0.88), rgba(250,247,240,0.94));
        border: 1px solid rgba(16, 24, 40, 0.08);
        border-radius: 28px;
        padding: 28px;
        box-shadow: 0 28px 60px rgba(16, 24, 40, 0.08);
      }}
      .hero h1 {{
        margin: 0;
        font-size: clamp(2rem, 4vw, 3.4rem);
        line-height: 1.02;
      }}
      .hero p {{
        margin: 12px 0 0;
        max-width: 900px;
        color: var(--muted);
        font-family: "Inter", system-ui, sans-serif;
      }}
      .meta {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
      }}
      .pill {{
        border: 1px solid rgba(16, 24, 40, 0.12);
        border-radius: 999px;
        padding: 8px 12px;
        font-family: "Inter", system-ui, sans-serif;
        font-size: 0.92rem;
        background: rgba(255,255,255,0.75);
      }}
      .cards {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 16px;
        margin: 24px 0 30px;
      }}
      .card {{
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 18px;
        box-shadow: 0 12px 28px rgba(16, 24, 40, 0.06);
      }}
      .card h2 {{
        margin: 0 0 8px;
        font-size: 0.92rem;
        font-family: "Inter", system-ui, sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
      }}
      .card strong {{
        font-size: 2rem;
        line-height: 1;
      }}
      .card p {{
        margin: 10px 0 0;
        color: var(--muted);
        font-family: "Inter", system-ui, sans-serif;
      }}
      .chart-grid {{
        display: grid;
        gap: 18px;
      }}
      .chart-card {{
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(16, 24, 40, 0.08);
        border-radius: 26px;
        padding: 18px;
        box-shadow: 0 20px 44px rgba(16, 24, 40, 0.07);
      }}
      .chart-heading {{
        margin-bottom: 10px;
      }}
      .chart-heading h3 {{
        margin: 0;
        font-size: 1.2rem;
      }}
      .chart-heading p {{
        margin: 6px 0 0;
        font-family: "Inter", system-ui, sans-serif;
        color: var(--muted);
      }}
      .summary-grid {{
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(0, 0.8fr);
        gap: 18px;
        margin-top: 24px;
      }}
      .panel {{
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(16, 24, 40, 0.08);
        border-radius: 26px;
        padding: 22px;
        box-shadow: 0 20px 44px rgba(16, 24, 40, 0.07);
      }}
      .panel h3 {{
        margin: 0 0 12px;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        font-family: "Inter", system-ui, sans-serif;
      }}
      th, td {{
        padding: 10px 0;
        border-bottom: 1px solid rgba(16, 24, 40, 0.08);
        text-align: left;
      }}
      td {{
        font-feature-settings: "tnum" 1;
      }}
      .note {{
        margin-top: 14px;
        color: var(--muted);
        font-family: "Inter", system-ui, sans-serif;
      }}
      @media (max-width: 900px) {{
        .summary-grid {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <h1>{escape(report.title)}</h1>
        <p>This report plots how the current architecture performs over repeated sampled trajectories. It keeps the scientific boundary honest by showing assembly, atomistic-reference agreement, shadow-correction error, and architecture timing side by side.</p>
        <div class="meta">
          <div class="pill">Scenario: {escape(report.scenario_name)}</div>
          <div class="pill">Replicates: {report.replicates}</div>
          <div class="pill">Steps / Replicate: {report.steps_per_replicate}</div>
          <div class="pill">Sample Interval: {report.sample_interval}</div>
          <div class="pill">Samples: {len(report.samples)}</div>
          <div class="pill">Classification: {escape(report.classification)}</div>
        </div>
      </section>

      <section class="cards">
        <article class="card">
          <h2>Final Assembly</h2>
          <strong>{escape(_format_value(report.summary.final_assembly_score_mean))}</strong>
          <p>Mean final assembly score across replicate trajectories.</p>
        </article>
        <article class="card">
          <h2>Final RMSD</h2>
          <strong>{escape(_format_value(report.summary.final_atomistic_centroid_rmsd_mean))}</strong>
          <p>Mean final atomistic-centroid RMSD against the local 6M0J frame.</p>
        </article>
        <article class="card">
          <h2>Contact Recovery</h2>
          <strong>{escape(_format_percent(report.summary.final_contact_recovery_mean))}</strong>
          <p>Mean final recovered interface-contact fraction after similarity fitting.</p>
        </article>
        <article class="card">
          <h2>Full Shadow Wins</h2>
          <strong>{escape(_format_percent(report.summary.full_shadow_improvement_rate))}</strong>
          <p>Fraction of sampled states where all three shadow-fidelity error metrics improved.</p>
        </article>
        <article class="card">
          <h2>Mean Timing</h2>
          <strong>{escape(_format_value(report.summary.mean_benchmark_total_seconds, 6))}</strong>
          <p>Average total benchmark time per sampled architecture slice.</p>
        </article>
      </section>

      <section class="chart-grid">{chart_html}</section>

      <section class="summary-grid">
        <article class="panel">
          <h3>Shadow Improvement Rates</h3>
          <table>
            <tbody>
              <tr><th>Energy Absolute Error Improved</th><td>{escape(_format_percent(report.summary.energy_improvement_rate))}</td></tr>
              <tr><th>Force RMS Error Improved</th><td>{escape(_format_percent(report.summary.force_rms_improvement_rate))}</td></tr>
              <tr><th>Max Force Component Error Improved</th><td>{escape(_format_percent(report.summary.max_force_component_improvement_rate))}</td></tr>
              <tr><th>All Three Improved Together</th><td>{escape(_format_percent(report.summary.full_shadow_improvement_rate))}</td></tr>
            </tbody>
          </table>
          <p class="note">This is the most direct architecture-level readout of whether the shadow pathway is consistently helping instead of just adding motion.</p>
        </article>
        <article class="panel">
          <h3>Mean Benchmark Timing by Case</h3>
          <table>
            <tbody>{benchmark_rows}</tbody>
          </table>
          <p class="note">These timings come from the existing foundation benchmark suite, so they reflect the same force, graph, qcloud, residual, and controller pathways used elsewhere in the program.</p>
        </article>
      </section>

      <section class="panel">
        <h3>Scientific Boundary</h3>
        <p class="note">The current report evaluates a coarse-grained live benchmark against an atomistic reference frame and a trusted shadow target. It does not claim full all-atom binding physics, calibrated thermodynamics, or production kinetics yet.</p>
      </section>
    </main>
  </body>
</html>
"""


__all__ = ["render_scientific_validation_report"]
