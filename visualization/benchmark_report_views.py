"""HTML rendering helpers for small-protein benchmark reports."""

from __future__ import annotations

from html import escape

from benchmarks.small_protein import SmallProteinBenchmarkReport


def _format_seconds(value: float) -> str:
    return f"{value:.6f} s"


def _format_multiplier(value: float) -> str:
    return f"{value:.2f}x"


def _comparison_chart(report: SmallProteinBenchmarkReport) -> str:
    engine_modes = report.engine_mode_summary()
    classical = engine_modes["classical_only"]
    production = engine_modes["hybrid_production"]
    values = (
        classical["average_seconds_per_iteration"],
        production["average_seconds_per_iteration"],
    )
    maximum = max(values) if max(values) > 0.0 else 1.0
    width = 860
    height = 280
    bar_width = 190
    base_y = 220
    scale = 150.0 / maximum

    classical_height = values[0] * scale
    production_height = values[1] * scale

    return f"""
    <section class="chart-card">
      <div class="chart-heading">
        <h3>Engine Modes</h3>
        <p>Top-line benchmark view for the same engine in `classical_only` and `hybrid_production` mode.</p>
      </div>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="Small protein engine mode comparison">
        <rect x="0" y="0" width="{width}" height="{height}" rx="24" fill="#ffffff" />
        <line x1="90" y1="{base_y}" x2="{width - 70}" y2="{base_y}" stroke="#334155" stroke-width="1.5" />
        <rect x="170" y="{base_y - classical_height:.2f}" width="{bar_width}" height="{classical_height:.2f}" rx="18" fill="#0f766e" />
        <rect x="500" y="{base_y - production_height:.2f}" width="{bar_width}" height="{production_height:.2f}" rx="18" fill="#1d4ed8" />
        <text x="265" y="246" text-anchor="middle" fill="#0f172a" font-size="15" font-family="Inter, system-ui, sans-serif">classical_only</text>
        <text x="595" y="246" text-anchor="middle" fill="#0f172a" font-size="15" font-family="Inter, system-ui, sans-serif">hybrid_production</text>
        <text x="265" y="{base_y - classical_height - 12:.2f}" text-anchor="middle" fill="#0f172a" font-size="14" font-family="Menlo, monospace">{escape(_format_seconds(values[0]))}</text>
        <text x="595" y="{base_y - production_height - 12:.2f}" text-anchor="middle" fill="#0f172a" font-size="14" font-family="Menlo, monospace">{escape(_format_seconds(values[1]))}</text>
        <text x="595" y="{base_y - production_height - 30:.2f}" text-anchor="middle" fill="#475569" font-size="12" font-family="Inter, system-ui, sans-serif">{escape(_format_multiplier(float(production["relative_to_classical_only"]))) } vs classical_only</text>
      </svg>
    </section>
    """


def render_small_protein_benchmark_report(report: SmallProteinBenchmarkReport) -> str:
    """Render one standalone HTML report for the small-protein benchmark."""

    engine_modes = report.engine_mode_summary()
    diagnostics = "".join(
        "<tr>"
        f"<th>{escape(case.name)}</th>"
        f"<td>{escape(_format_seconds(case.average_seconds_per_iteration()))}</td>"
        f"<td>{escape(case.metadata.to_dict().get('engine_mode', case.metadata.to_dict().get('diagnostic_role', 'diagnostic')))}</td>"
        "</tr>"
        for case in report.benchmark_report.cases
    )
    parity_rows = "".join(
        "<tr>"
        f"<th>{escape(metric.label)}</th>"
        f"<td>{metric.passed}</td>"
        f"<td>{escape(_format_seconds(metric.absolute_error) if 'energy' in metric.label else f'{metric.absolute_error:.6f}')}</td>"
        f"<td>{escape(_format_seconds(metric.tolerance) if 'energy' in metric.label else f'{metric.tolerance:.6f}')}</td>"
        "</tr>"
        for metric in report.parity_report.metrics
    )

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{escape(report.spec.name)} | Small Protein Benchmark</title>
    <style>
      :root {{
        color-scheme: light;
        --paper: #f6f3ed;
        --ink: #0f172a;
        --muted: #475569;
        --line: #d9d4ca;
        --card: rgba(255,255,255,0.88);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: "Iowan Old Style", Georgia, serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top right, rgba(29, 78, 216, 0.10), transparent 26%),
          radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 24%),
          linear-gradient(180deg, #fcfaf6 0%, var(--paper) 100%);
      }}
      main {{
        max-width: 1160px;
        margin: 0 auto;
        padding: 36px 22px 56px;
      }}
      .hero, .chart-card, .table-card {{
        background: var(--card);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 24px;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.06);
      }}
      .hero {{
        padding: 28px 30px;
        margin-bottom: 22px;
      }}
      .hero h1 {{
        margin: 0 0 8px;
        font-size: 2.2rem;
      }}
      .hero p {{
        margin: 0;
        color: var(--muted);
        font-size: 1.02rem;
      }}
      .hero-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
        margin-top: 22px;
      }}
      .metric {{
        padding: 14px 16px;
        border-radius: 18px;
        background: rgba(248, 250, 252, 0.82);
        border: 1px solid rgba(15, 23, 42, 0.06);
      }}
      .metric span {{
        display: block;
        font-size: 0.82rem;
        color: var(--muted);
        margin-bottom: 6px;
      }}
      .metric strong {{
        font-size: 1.02rem;
      }}
      .chart-card, .table-card {{
        padding: 22px;
        margin-bottom: 20px;
      }}
      .chart-heading h3, .table-card h3 {{
        margin: 0 0 6px;
        font-family: Inter, system-ui, sans-serif;
      }}
      .chart-heading p {{
        margin: 0 0 10px;
        color: var(--muted);
        font-family: Inter, system-ui, sans-serif;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        font-family: Inter, system-ui, sans-serif;
      }}
      th, td {{
        text-align: left;
        padding: 12px 10px;
        border-top: 1px solid var(--line);
        font-size: 0.93rem;
      }}
      th {{
        color: var(--ink);
        font-weight: 600;
      }}
      td {{
        color: var(--muted);
      }}
      code {{
        font-family: Menlo, monospace;
        font-size: 0.92em;
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <h1>Small Protein Benchmark</h1>
        <p>One real-structure compact benchmark for the same engine, shown as top-line modes first and detailed ablations underneath.</p>
        <div class="hero-grid">
          <div class="metric"><span>Protein</span><strong>{escape(report.entity_id)}</strong></div>
          <div class="metric"><span>Source Structure</span><strong>{escape(report.structure_id)}</strong></div>
          <div class="metric"><span>Residues to Beads</span><strong>{report.residue_count} residues -> {report.bead_count} beads</strong></div>
          <div class="metric"><span>Backend</span><strong>{escape(str(report.execution_plan.selection.selected_backend))}</strong></div>
          <div class="metric"><span>Execution Mode</span><strong>{escape(report.execution_plan.execution_mode)}</strong></div>
          <div class="metric"><span>Backend Parity</span><strong>{'passed' if report.parity_report.all_passed() else 'failed'}</strong></div>
          <div class="metric"><span>classical_only</span><strong>{escape(_format_seconds(float(engine_modes['classical_only']['average_seconds_per_iteration'])))}</strong></div>
          <div class="metric"><span>hybrid_production</span><strong>{escape(_format_seconds(float(engine_modes['hybrid_production']['average_seconds_per_iteration'])))}</strong></div>
        </div>
      </section>
      {_comparison_chart(report)}
      <section class="table-card">
        <h3>Detailed Diagnostic Slices</h3>
        <table>
          <thead>
            <tr><th>Case</th><th>Avg Time / Iter</th><th>Role</th></tr>
          </thead>
          <tbody>
            {diagnostics}
          </tbody>
        </table>
      </section>
      <section class="table-card">
        <h3>Backend Parity Metrics</h3>
        <table>
          <thead>
            <tr><th>Metric</th><th>Passed</th><th>Absolute Error</th><th>Tolerance</th></tr>
          </thead>
          <tbody>
            {parity_rows}
          </tbody>
        </table>
      </section>
    </main>
  </body>
</html>"""


__all__ = ["render_small_protein_benchmark_report"]
