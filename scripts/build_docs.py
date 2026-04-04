"""Build the NeuroCGMD documentation website.

Generates a multi-page static site with shared sidebar navigation,
tutorial with embedded plots, reference manual, and contact pages.

Usage:
    python3 scripts/build_docs.py
    # Then open docs/website/index.html in a browser

Deploys to GitHub Pages by pushing docs/website/ to gh-pages branch.
"""

from __future__ import annotations
import shutil
from pathlib import Path
from textwrap import dedent

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "docs" / "website"
PLOTS = REPO / "outputs" / "barnase_barstar_20ns" / "plots"

# ── Sidebar definition ──────────────────────────────────────────────────────

NAV = [
    ("Getting Started", [
        ("index.html", "Home"),
        ("install.html", "Installation"),
        ("quickstart.html", "Quickstart"),
    ]),
    ("Tutorial", [
        ("tutorial/barnase-barstar.html", "Barnase-Barstar"),
        ("tutorial/output-gallery.html", "Output Gallery"),
    ]),
    ("User Guide", [
        ("manual/configuration.html", "Configuration"),
        ("manual/running.html", "Running Simulations"),
        ("manual/analysis.html", "Analysis Pipeline"),
        ("manual/back-mapping.html", "CG-to-AA Back-mapping"),
    ]),
    ("Reference", [
        ("reference/architecture.html", "Architecture"),
        ("reference/forcefields.html", "Force Fields"),
        ("reference/integrators.html", "Integrators"),
        ("reference/qcloud.html", "QCloud Layer"),
        ("reference/ml-residual.html", "ML Residual"),
        ("reference/observables.html", "Observables"),
    ]),
    ("Community", [
        ("compare.html", "Comparison"),
        ("contact.html", "Contact & Cite"),
    ]),
]


def _sidebar_html(active_path: str) -> str:
    """Generate sidebar HTML with the given page marked active."""
    # Compute relative prefix based on depth
    depth = active_path.count("/")
    prefix = "../" * depth if depth > 0 else ""

    lines = []
    lines.append('<div class="sidebar">')
    lines.append(f'  <div class="logo"><a href="{prefix}index.html">Neuro<span>CGMD</span></a>')
    lines.append('    <span class="version">v1.0.0 Documentation</span></div>')

    for section_title, links in NAV:
        lines.append(f'  <div class="section-title">{section_title}</div>')
        for href, label in links:
            active = ' class="nav-link active"' if href == active_path else ' class="nav-link"'
            lines.append(f'  <a{active} href="{prefix}{href}">{label}</a>')

    lines.append('</div>')
    return "\n".join(lines)


INTERACTIVE_CSS = """
<style>
.layer{position:relative;margin:0 0 6px;border-radius:12px;padding:20px;transition:all .3s}
.layer-title{font-size:13px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:10px;cursor:pointer;user-select:none}
.layer-title .tg{font-size:11px;opacity:.5;transition:transform .3s}
.layer-title .tg.open{transform:rotate(90deg)}
.layer-body{overflow:hidden;transition:max-height .5s ease,opacity .3s}
.layer-body.collapsed{max-height:0!important;opacity:0;padding:0}
.ng{display:grid;gap:10px}
.ng.c4{grid-template-columns:repeat(4,1fr)}.ng.c3{grid-template-columns:repeat(3,1fr)}.ng.c2{grid-template-columns:repeat(2,1fr)}
.nd{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:14px 16px;cursor:pointer;transition:all .25s;position:relative;overflow:hidden}
.nd:hover{transform:translateY(-3px);box-shadow:0 8px 24px #00000060}
.nd::before{content:'';position:absolute;top:0;left:0;right:0;height:3px}
.nd.bl::before{background:var(--blue)}.nd.bl:hover{border-color:var(--blue)}
.nd.pu::before{background:var(--purple)}.nd.pu:hover{border-color:var(--purple)}
.nd.go::before{background:var(--gold)}.nd.go:hover{border-color:var(--gold)}
.nd.te::before{background:var(--teal)}.nd.te:hover{border-color:var(--teal)}
.nd.gr::before{background:var(--green)}.nd.gr:hover{border-color:var(--green)}
.nd.or::before{background:var(--orange)}.nd.or:hover{border-color:var(--orange)}
.nd.pk::before{background:var(--pink)}.nd.pk:hover{border-color:var(--pink)}
.nd .nm{font-size:13px;font-weight:600;margin-bottom:4px}
.nd .ds{font-size:11px;color:var(--text-muted);line-height:1.5}
.nd .dt{display:none;margin-top:10px;padding-top:10px;border-top:1px solid var(--border);font-size:12px;color:var(--text-muted);line-height:1.7}
.nd.ex .dt{display:block;animation:fi .3s}
.nd .fm{font-family:'Times New Roman',Georgia,serif;font-size:13px;color:var(--gold);margin:6px 0;font-style:italic}
.eqbar{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:14px 20px;margin:12px 0;text-align:center;transition:border-color .3s;cursor:pointer}
.eqbar:hover{border-color:var(--gold)}
.eqbar .eq{color:var(--green);font-family:'Times New Roman',Georgia,serif}
.eqbar .nt{font-size:12px;color:var(--text-muted);margin-top:6px;font-style:italic}
.ecols{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
.ecol{display:flex;flex-direction:column;gap:8px}
.ch{text-align:center;font-size:12px;font-weight:700;letter-spacing:.06em;padding:8px;border-radius:6px;margin-bottom:4px}
.df{display:flex;align-items:center;justify-content:center;gap:8px;margin:8px 0;font-size:11px;color:var(--text-muted)}
.df .dots{display:flex;gap:4px}
.df .d{width:4px;height:4px;border-radius:50%;animation:pu 1.5s ease-in-out infinite}
.df .d:nth-child(2){animation-delay:.3s}.df .d:nth-child(3){animation-delay:.6s}
.df .d.bl{background:var(--blue)}.df .d.pu{background:var(--purple)}.df .d.go{background:var(--gold)}
.df .d.te{background:var(--teal)}.df .d.gr{background:var(--green)}.df .d.pk{background:var(--pink)}
.al{list-style:none;padding:0;margin:8px 0 0}
.al li{font-size:12px;padding:5px 0;display:flex;justify-content:space-between;border-bottom:1px solid #30363D40}
.al li:last-child{border-bottom:none}
.al .n{font-weight:600}.al .d{color:var(--text-muted);font-size:11px}
.chips{display:flex;gap:10px;flex-wrap:wrap;justify-content:center;margin:20px 0}
.chip{padding:10px 18px;border-radius:20px;font-size:12px;font-weight:600;text-align:center;border:1px solid;transition:transform .2s}
.chip:hover{transform:scale(1.05)}
.chip.bl{color:var(--blue);border-color:#58A6FF30;background:#58A6FF08}
.chip.pu{color:var(--purple);border-color:#BC8CFF30;background:#BC8CFF08}
.chip.go{color:var(--gold);border-color:#F0C75E30;background:#F0C75E08}
.chip.te{color:var(--teal);border-color:#3FB9A230;background:#3FB9A208}
.chip.pk{color:var(--pink);border-color:#F778BA30;background:#F778BA08}
.chip.or{color:var(--orange);border-color:#F0883E30;background:#F0883E08}
.chip.gr{color:var(--green);border-color:#56D36430;background:#56D36408}
.gi{background:var(--bg-card);border:1px solid var(--border);border-radius:10px;overflow:hidden;transition:border-color .3s}
.gi:hover{border-color:var(--blue)}.gi img{width:100%;display:block}
.gi .cap{padding:12px 14px}.gi .cap h4{font-size:13px;margin-bottom:3px}.gi .cap p{font-size:12px;color:var(--text-muted);margin:0}
.gg{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin:24px 0}
.gi.wide{grid-column:span 2}
.contact-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin:24px 0}
.cc{background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:18px;text-align:center;transition:border-color .3s;display:block}
.cc:hover{border-color:var(--teal)}.cc .tp{font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
.cc .sj{font-family:var(--mono);font-size:11px;color:var(--gold);margin-top:8px}
@keyframes pu{0%,100%{opacity:.3}50%{opacity:1}}
@keyframes fi{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:translateY(0)}}
@media(max-width:768px){.ecols{grid-template-columns:1fr}.ng.c4,.ng.c3{grid-template-columns:1fr}.gi.wide{grid-column:span 1}}
</style>
"""

INTERACTIVE_JS = """
<script>
function N(e){e.classList.toggle('ex')}
function T(t){var b=t.nextElementSibling,g=t.querySelector('.tg');b.classList.toggle('collapsed');g.classList.toggle('open')}
</script>
"""

def _page(path: str, title: str, content: str) -> str:
    """Wrap content in the full interactive page template."""
    depth = path.count("/")
    css_path = "../" * depth + "style.css" if depth > 0 else "style.css"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} - NeuroCGMD Documentation</title>
<meta name="description" content="NeuroCGMD: Quantum-Classical-CG-ML Cooperative Molecular Dynamics Engine documentation.">
<link rel="stylesheet" href="{css_path}">
{INTERACTIVE_CSS}
</head>
<body>
{_sidebar_html(path)}
<div class="main">
<div class="content">
{content}
</div>
<div class="footer">
  NeuroCGMD v1.0.0 &middot; MIT License &middot;
  <a href="mailto:bessuman.academia@gmail.com">bessuman.academia@gmail.com</a><br>
  500 KB of Python that speaks the language of atoms.
</div>
</div>
{INTERACTIVE_JS}
</body>
</html>"""


def _img(name: str, depth: int = 0) -> str:
    """Image tag referencing assets/plots/."""
    prefix = "../" * depth
    return f'{prefix}assets/plots/{name}'


def _write(path: str, title: str, content: str) -> None:
    """Write a page to disk."""
    full = SITE / path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(_page(path, title, content), encoding="utf-8")
    print(f"  {path}")


# ── Interactive component helpers ────────────────────────────────────────────

def layer(color: str, title: str, body: str, subtitle: str = "", collapsed: bool = False) -> str:
    """Expandable layer section with toggle."""
    cls = "collapsed" if collapsed else ""
    oc = "" if collapsed else " open"
    mh = "0" if collapsed else "3000px"
    sub = f' <span style="font-size:11px;font-weight:400;color:var(--text-muted);text-transform:none;letter-spacing:0">&mdash; {subtitle}</span>' if subtitle else ""
    return f'''<div class="layer" style="background:var(--bg-card);border:1px solid {color}40">
<div class="layer-title" style="color:{color}" onclick="T(this)"><span class="tg{oc}">&#9654;</span> {title}{sub}</div>
<div class="layer-body {cls}" style="max-height:{mh}px">
{body}
</div></div>'''


def node(color: str, name: str, desc: str, detail: str = "", formula: str = "") -> str:
    """Clickable card with expandable detail."""
    fm = f'<div class="fm">{formula}</div>' if formula else ""
    dt = f'<div class="dt">{detail}{fm}</div>' if detail or formula else ""
    return f'<div class="nd {color}" onclick="N(this)"><div class="nm">{name}</div><div class="ds">{desc}</div>{dt}</div>'


def flow(color: str, text: str) -> str:
    """Animated data-flow dots with label."""
    return f'<div class="df"><div class="dots"><div class="d {color}"></div><div class="d {color}"></div><div class="d {color}"></div></div><span>{text}</span><div class="dots"><div class="d {color}"></div><div class="d {color}"></div><div class="d {color}"></div></div></div>'


def grid(cols: int, *nodes: str) -> str:
    return f'<div class="ng c{cols}">{"".join(nodes)}</div>'


def gallery_item(img_path: str, title: str, desc: str = "", wide: bool = False) -> str:
    w = ' wide' if wide else ""
    d = f"<p>{desc}</p>" if desc else ""
    return f'<div class="gi{w}"><img src="{img_path}" alt="{title}"><div class="cap"><h4>{title}</h4>{d}</div></div>'


def eq(formula: str, note: str = "") -> str:
    nt = f'<div class="nt">{note}</div>' if note else ""
    return f'<div class="eqbar"><div class="eq" style="font-size:17px">{formula}</div>{nt}</div>'


# ── Page content generators ─────────────────────────────────────────────────

def build_home():
    nav_cards = grid(4,
        node("bl", "Install", "pip install neurocgmd", "One command, 500 KB, no compilation. <a href='install.html'>Read more &rarr;</a>"),
        node("pu", "Quickstart", "PDB to plots in one command", "Run your first simulation in 3 steps. <a href='quickstart.html'>Read more &rarr;</a>"),
        node("go", "Tutorial", "Barnase-barstar walkthrough", "Full step-by-step with every output shown. <a href='tutorial/barnase-barstar.html'>Read more &rarr;</a>"),
        node("te", "Architecture", "4 cooperative layers", "Interactive diagram of the engine. <a href='reference/architecture.html'>Explore &rarr;</a>"),
    )

    layers_body = grid(4,
        node("bl", "CG Dynamics", "Speed: ~1000 steps/min", "Langevin integration with classical forcefields. BAOAB splitting, velocity-Verlet, O(N) neighbor lists."),
        node("pu", "QCloud Quantum", "Accuracy: AA-level physics", "Force corrections on priority regions with adaptive focusing. Event detection for bond forming/breaking."),
        node("go", "ML Residual", "Efficiency: fewer QCloud calls", "Neural network learns correction patterns on-the-fly from QCloud. Reduces quantum compute over time."),
        node("te", "Back-mapping", "Detail: real residues", "Reconstructs full all-atom coordinates from CG trajectory. Carries CG + QCloud + ML information through."),
    )

    _write("index.html", "Home", f"""
<h1><span class="accent">NeuroCGMD</span> Documentation</h1>
<p class="page-desc">
  Quantum-Classical-CG-ML Cooperative Molecular Dynamics Engine.<br>
  One simulation. One information stream. All-atom accuracy at coarse-grained speed.
</p>

{nav_cards}

{flow("bl", "four layers fuse into one cooperative simulation")}

{layer("var(--orange)", "THE FOUR LAYERS", layers_body, "click any card to expand")}

{flow("gr", "the result")}

{eq("<b>F</b><sub>total</sub> = <b>F</b><sub>CG</sub> + &Delta;<b>F</b><sub>QCloud</sub> + &alpha; &Delta;<b>F</b><sub>ML</sub> &nbsp;&rarr;&nbsp; integrator &nbsp;&rarr;&nbsp; CG positions &nbsp;&rarr;&nbsp; back-map &nbsp;&rarr;&nbsp; AA",
    "500 KB of pure Python. No compilation. No Fortran. No MPI.")}

<div class="chips">
  <div class="chip bl">pip install neurocgmd</div>
  <div class="chip pu">Python 3.11+</div>
  <div class="chip go">numpy + matplotlib</div>
  <div class="chip te">MIT License</div>
</div>
""")


def build_install():
    # Method 1: pip
    m_pip = node("gr", "Option 1: pip install (Recommended)", "The simplest way — one command, everything is handled for you",
        "<strong>Step 1.</strong> Open a terminal<br>"
        "&bull; On macOS: search for \"Terminal\" in Spotlight<br>"
        "&bull; On Windows: open \"Command Prompt\" or \"PowerShell\"<br>"
        "&bull; On Linux: open any terminal application<br><br>"
        "<strong>Step 2.</strong> Type this command and press Enter:<br>"
        "<pre><code>pip install neurocgmd</code></pre>"
        "<strong>Step 3.</strong> Wait for it to finish. It downloads NeuroCGMD and its dependencies (NumPy and Matplotlib) automatically.<br><br>"
        "<strong>Step 4.</strong> That's it. NeuroCGMD is now installed.<br><br>"
        "<strong>Getting an error?</strong><br>"
        "&bull; Try <code>pip3 install neurocgmd</code> instead<br>"
        "&bull; Or try <code>python3 -m pip install neurocgmd</code><br>"
        "&bull; Permission denied? Add <code>--user</code>: <code>pip install --user neurocgmd</code>")

    # Method 2: Wheel
    m_whl = node("bl", "Option 2: Download Wheel (.whl)", "A pre-built package file you download and install",
        "<strong>Step 1.</strong> Go to: <a href='https://pypi.org/project/neurocgmd/1.0.0/#files' target='_blank'>pypi.org/project/neurocgmd/1.0.0/#files</a><br><br>"
        "<strong>Step 2.</strong> Click on <strong>neurocgmd-1.0.0-py3-none-any.whl</strong> to download it<br><br>"
        "<strong>Step 3.</strong> Open a terminal and go to your Downloads folder:<br>"
        "<pre><code>cd ~/Downloads</code></pre>"
        "<strong>Step 4.</strong> Install it:<br>"
        "<pre><code>pip install neurocgmd-1.0.0-py3-none-any.whl</code></pre>")

    # Method 3: Tarball
    m_tar = node("go", "Option 3: Download Tarball (.tar.gz)", "A compressed archive containing the source code",
        "<strong>Step 1.</strong> Go to: <a href='https://pypi.org/project/neurocgmd/1.0.0/#files' target='_blank'>pypi.org/project/neurocgmd/1.0.0/#files</a><br><br>"
        "<strong>Step 2.</strong> Click on <strong>neurocgmd-1.0.0.tar.gz</strong> to download it<br><br>"
        "<strong>Step 3.</strong> Open a terminal and go to your Downloads folder:<br>"
        "<pre><code>cd ~/Downloads</code></pre>"
        "<strong>Step 4.</strong> Extract the archive:<br>"
        "<pre><code>tar xzf neurocgmd-1.0.0.tar.gz</code></pre>"
        "<strong>Step 5.</strong> Go into the extracted folder:<br>"
        "<pre><code>cd neurocgmd-1.0.0</code></pre>"
        "<strong>Step 6.</strong> Install (the dot at the end is important):<br>"
        "<pre><code>pip install .</code></pre>")

    # Method 4: From GitHub
    m_github = node("pu", "Option 4: From GitHub (source code)", "Get the latest source code — with git or as a ZIP download",
        "<strong>With git:</strong><br><br>"
        "<strong>Step 1.</strong> Open a terminal and type:<br>"
        "<pre><code>git clone https://github.com/sciencemaths-collab/neurocgmd.git</code></pre>"
        "<strong>Step 2.</strong> Go into the folder:<br>"
        "<pre><code>cd neurocgmd</code></pre>"
        "<strong>Step 3.</strong> Install:<br>"
        "<pre><code>pip install .</code></pre>"
        "<br><strong>Without git (ZIP download):</strong><br><br>"
        "<strong>Step 1.</strong> Download: <a href='https://github.com/sciencemaths-collab/neurocgmd/archive/refs/heads/main.zip'>neurocgmd-main.zip</a><br><br>"
        "<strong>Step 2.</strong> Unzip the file (double-click on macOS/Windows)<br><br>"
        "<strong>Step 3.</strong> Open a terminal and go into the folder:<br>"
        "<pre><code>cd neurocgmd-main</code></pre>"
        "<strong>Step 4.</strong> Install:<br>"
        "<pre><code>pip install .</code></pre>")

    # Prerequisites
    prereq = node("go", "Prerequisites", "What you need before installing NeuroCGMD",
        "<strong>Python 3.11 or newer</strong> is required. To check your Python version:<br>"
        "<pre><code>python3 --version</code></pre>"
        "If you see 3.11 or higher, you are ready. If not, download Python from <a href=\"https://python.org/downloads\" target=\"_blank\">python.org/downloads</a>.<br><br>"
        "<strong>pip</strong> (Python's package installer) comes with Python. To check:<br>"
        "<pre><code>pip --version</code></pre>"
        "If this command is not found, install pip:<br>"
        "<pre><code>python3 -m ensurepip --upgrade</code></pre>")

    deps = grid(3,
        node("go", "Python", "&ge; 3.11", "The only prerequisite. Download from python.org if needed."),
        node("bl", "NumPy", "&ge; 1.24", "Installed automatically. Handles numerical computation."),
        node("pu", "Matplotlib", "&ge; 3.7", "Installed automatically. Generates all analysis plots."),
    )

    platforms = grid(3,
        node("te", "macOS", "Apple Silicon &amp; Intel", "Fully supported. Works on M1/M2/M3 and older Intel Macs."),
        node("te", "Linux", "x86_64 &amp; aarch64", "All major distributions: Ubuntu, CentOS, Fedora, Debian."),
        node("te", "Windows", "10 / 11", "Full support. Use Command Prompt, PowerShell, or WSL."),
    )

    _write("install.html", "Installation", f"""
<h1>Download &amp; <span class="accent">Install</span></h1>
<p class="page-desc">NeuroCGMD is a pure Python package. No compilation, no Fortran, no MPI, no GPU drivers required.</p>

<table>
<tr><th>Method</th><th>Command / Link</th></tr>
<tr><td><strong>pip</strong> (recommended)</td><td><code>pip install neurocgmd</code></td></tr>
<tr><td><strong>Wheel</strong> (.whl)</td><td><a href="https://pypi.org/project/neurocgmd/1.0.0/#files">neurocgmd-1.0.0-py3-none-any.whl</a></td></tr>
<tr><td><strong>Tarball</strong> (.tar.gz)</td><td><a href="https://pypi.org/project/neurocgmd/1.0.0/#files">neurocgmd-1.0.0.tar.gz</a></td></tr>
<tr><td><strong>Source</strong> (GitHub)</td><td><a href="https://github.com/sciencemaths-collab/neurocgmd">github.com/sciencemaths-collab/neurocgmd</a></td></tr>
<tr><td><strong>ZIP</strong> (no git needed)</td><td><a href="https://github.com/sciencemaths-collab/neurocgmd/archive/refs/heads/main.zip">Download ZIP</a></td></tr>
</table>

{layer("var(--gold)", "BEFORE YOU START", prereq, "check that Python 3.11+ is installed")}

{flow("gr", "choose your installation method")}

<div class="ng" style="grid-template-columns:1fr">
{m_pip}
{m_whl}
{m_tar}
{m_github}
</div>

{flow("gr", "verify your installation")}

{layer("var(--green)", "VERIFY INSTALLATION", '''After installing with any method above, open a terminal and run:<br><br>
<pre><code>neurocgmd --version</code></pre>
You should see: <code>neurocgmd 1.0.0</code><br><br>
Then run the full capabilities display:<br>
<pre><code>neurocgmd info</code></pre>
This shows a color-coded overview of all integrators, force fields, analysis tools, and supported systems.<br><br>
<strong>If the command is not found</strong>, your Python scripts directory may not be in your PATH. Try running it as a module instead:<br>
<pre><code>python3 -m neurocgmd info</code></pre>''', "make sure everything works")}

{flow("go", "dependencies")}

{layer("var(--gold)", "DEPENDENCIES", str(deps) + '<p style="color:var(--text-muted); font-size:13px; margin-top:12px;">NumPy and Matplotlib are installed automatically when you run <code>pip install neurocgmd</code>. You do not need to install them separately. There are no other dependencies &mdash; no CUDA, FFTW, OpenCL, MPI, or Fortran compiler required.</p>', "automatically installed &mdash; you don't need to do anything")}

{flow("te", "runs everywhere")}

{layer("var(--teal)", "SUPPORTED PLATFORMS", str(platforms), "any system that runs Python 3.11+")}

{flow("or", "need help?")}

{layer("var(--orange)", "TROUBLESHOOTING", grid(1,
    node("or", "Common Issues", "Click to expand solutions",
        "<strong>\"pip: command not found\"</strong><br>"
        "Try <code>pip3</code> instead of <code>pip</code>, or <code>python3 -m pip install neurocgmd</code>.<br><br>"
        "<strong>\"Permission denied\"</strong><br>"
        "Add <code>--user</code> flag: <code>pip install --user neurocgmd</code><br><br>"
        "<strong>\"Python version too old\"</strong><br>"
        "NeuroCGMD requires Python 3.11+. Download the latest from <a href=\"https://python.org/downloads\">python.org</a>.<br><br>"
        "<strong>\"neurocgmd: command not found\" after install</strong><br>"
        "Try: <code>python3 -m neurocgmd info</code><br>"
        "Or add your Python scripts directory to PATH.<br><br>"
        "<strong>Still stuck?</strong><br>"
        "Email <a href=\"mailto:bessuman.academia@gmail.com?subject=[NeuroCGMD] Installation Help\">bessuman.academia@gmail.com</a> with your OS, Python version, and the error message.")
), "click to see common solutions")}

{layer("var(--red)", "UNINSTALL", '<pre><code>pip uninstall neurocgmd</code></pre><p style="color:var(--text-muted);font-size:13px;">This removes NeuroCGMD but keeps your simulation output files and configurations.</p>', collapsed=True)}
""")


def build_quickstart():
    steps = "".join([
        node("gr", "1. Install", "pip install neurocgmd",
             '<pre><code>pip install neurocgmd</code></pre>'),
        node("bl", "2. Run", "neurocgmd run config.toml",
             'Single command: import PDB &rarr; equilibrate &rarr; produce &rarr; QCloud &rarr; ML &rarr; back-map &rarr; analyze &rarr; plot.<pre><code>neurocgmd run examples/barnase_barstar.toml</code></pre>'),
        node("pu", "3. Examine", "20+ plots + CG &amp; AA trajectories",
             '<pre><code>outputs/barnase_barstar/\n  plots/                        # 20+ PNG analysis plots\n  cg_trajectory.pdb             # CG multi-model\n  aa_backmapped_trajectory.pdb  # Full AA trajectory\n  energies.csv                  # Energy time series\n  run_summary.json              # Run metadata</code></pre>'),
        node("te", "4. Re-analyze", "neurocgmd analyze config.toml",
             'Run only analysis on existing trajectory. Useful for tweaking or running on partial data while simulation continues.<pre><code>neurocgmd analyze examples/barnase_barstar.toml</code></pre>'),
    ])
    _write("quickstart.html", "Quickstart", f"""
<h1>Quickstart</h1>
<p class="page-desc">From PDB structure to publication-ready analysis in minutes.</p>

<div class="ng" style="grid-template-columns:1fr">{steps}</div>
""")


def build_tutorial_barnase():
    d = 1
    I = lambda name: _img(name, d)
    G = lambda name, title, desc="", wide=False: gallery_item(I(name), title, desc, wide)

    sys_info = grid(3,
        node("bl", "Barnase (A,B,C)", "110 residues each", "RNase enzyme, the inhibited partner."),
        node("pu", "Barstar (D,E,F)", "89 residues each", "Barnase inhibitor, the binding partner."),
        node("te", "CG Mapping", "4640 atoms &rarr; 76 beads", "6 entities, 3 biological pairs: D-A, E-B, F-C."),
    )

    config_body = '''<pre><code>[system]
name = "barnase_barstar"
pdb_source = "benchmarks/reference_cases/data/1BRS.pdb"
output_dir = "outputs/barnase_barstar_2ns"

[dynamics]
stages = ["nvt", "npt", "production"]

[dynamics.nvt]
steps = 5000
time_step = 0.02
temperature = 300.0
ensemble = "NVT"

[dynamics.npt]
steps = 5000
time_step = 0.02
temperature = 300.0
ensemble = "NPT"

[dynamics.production]
steps = 100000
time_step = 0.02
temperature = 300.0
eval_stride = 50</code></pre>'''

    outputs_body = grid(3,
        node("bl", "cg_trajectory.pdb", "223-frame CG trajectory"),
        node("te", "aa_backmapped_trajectory.pdb", "102-frame AA trajectory"),
        node("go", "energies.csv + run_summary.json", "Data + metadata"),
    )

    # What you get — summarized as interactive cards
    analysis_summary = grid(3,
        node("bl", "Energy &amp; Thermodynamics", "energy_timeseries.png", "Potential, kinetic, and total energy time series confirming equilibration across NVT, NPT, and production stages."),
        node("bl", "Structural Stability", "rmsd.png, rmsf.png", "RMSD vs time shows structural drift plateaus. RMSF per bead identifies flexible loops and rigid core regions across all 6 chains."),
        node("bl", "Pair Correlations", "rdf.png", "Radial distribution function g(r) with characteristic peaks at bonded distances and LJ equilibrium."),
        node("bl", "Surface &amp; Compactness", "sasa_rg.png", "Solvent-accessible surface area and radius of gyration evolution over the trajectory."),
        node("bl", "Free Energy", "pmf.png, free_energy_landscape_2d.png", "Potential of mean force from Boltzmann inversion of COM distance. 2D free energy landscape (COM distance vs Rg) with trajectory path overlay."),
        node("bl", "Reaction Coordinate", "reaction_coordinate.png", "Center-of-mass distance between binding partners tracking binding progress over the full 2 ns."),
        node("te", "AA Residue Contacts", "aa_contact_map_*.png", "87&times;108 residue-residue contact frequency map at full amino acid resolution from back-mapped coordinates."),
        node("te", "Interface Pairs &amp; H-bonds", "aa_top_residue_pairs_*.png", "Top 30 interacting residue pairs ranked by contact frequency with H-bond overlay. Real residue names: GLY31, TYR29, ASP39, ARG83."),
        node("te", "Per-Residue Binding", "aa_residue_binding_contribution_*.png", "Binding contribution per amino acid with hotspot residues auto-labeled. Contact score (blue) + H-bond score (red)."),
        node("pu", "Energy Decomposition", "energy_decomposition.png", "Per-bead bonded, nonbonded, and total energy contribution across all CG beads."),
        node("pk", "Binding Dashboard", "binding_dashboard.png", "4-panel overview: COM distance, interface contacts, interaction energy, and binding correlation scatter."),
        node("or", "Structure Snapshots", "structure_snapshots.png", "XY projections at 6 timepoints showing complex evolution. Red = barnase, blue = barstar."),
    )

    _write("tutorial/barnase-barstar.html", "Tutorial: Barnase-Barstar", f"""
<h1>Tutorial: <span class="accent">Barnase-Barstar</span> Binding</h1>
<p class="page-desc">Complete walkthrough &mdash; crystal structure to publication-ready binding analysis.</p>

{layer("var(--blue)", "SYSTEM OVERVIEW", sys_info, "PDB: 1BRS &mdash; canonical protein-protein binding")}

{flow("or", "step 1: configure")}

{layer("var(--orange)", "CONFIGURATION", config_body, "eval_stride=50: full QCloud+ML every 50 steps, ~1000 steps/min")}

{flow("gr", "step 2: run")}

{layer("var(--green)", "RUN", '<pre><code>neurocgmd run examples/barnase_20ns.toml</code></pre><p style="color:var(--text-muted);font-size:13px">Import PDB &rarr; NVT &rarr; NPT &rarr; production (QCloud+ML) &rarr; back-map &rarr; analyze &rarr; plot</p>')}

{flow("te", "step 3: sample output")}

<div class="gg">
{G("aa_interface_hbonds_E_B.png", "Inter-Chain H-bonds: Barstar (E) &mdash; Barnase (B)", "18 persistent backbone H-bonds at the binding interface, detected with distance (&lt;3.5&Aring;) and angle (&gt;120&deg;) criteria on back-mapped AA structures. GLY31-HIS102 is the strongest at 90% occupancy.", True)}
{G("contact_map.png", "Inter-Chain Contact Map", "Bead-bead contact frequency across all 6 chains (A-F). Dashed lines mark entity boundaries. Dark regions indicate persistent binding interfaces between barnase-barstar pairs.", True)}
</div>

{flow("go", "what you get: 20+ publication-quality plots")}

{layer("var(--gold)", "COMPLETE ANALYSIS OUTPUT", analysis_summary, "click any card to see details &mdash; all generated automatically from one command")}

{flow("go", "trajectory files")}

{layer("var(--gold)", "OUTPUT FILES", str(outputs_body), "all PDB files load in Chimera, VMD, or PyMOL")}
""")


def build_tutorial_gallery():
    d = 1
    _write("tutorial/output-gallery.html", "Output Gallery", f"""
<h1>Output Gallery</h1>
<p class="page-desc">Every plot generated automatically from a single <code>neurocgmd run</code> command.</p>

<h2>Summary dashboard</h2>
<div class="gallery">
<div class="gallery-item wide">
  <img src="{_img('dashboard.png', d)}" alt="Dashboard">
  <div class="caption"><h4>8-Panel Summary Dashboard</h4><p>Energy, RMSD, RDF, RMSF, SASA, Rg, reaction coordinate, PMF.</p></div>
</div>
</div>

<h2>CG-level analysis</h2>
<div class="gallery">
<div class="gallery-item"><img src="{_img('energy_timeseries.png', d)}"><div class="caption"><h4>Energy</h4></div></div>
<div class="gallery-item"><img src="{_img('rmsd.png', d)}"><div class="caption"><h4>RMSD</h4></div></div>
<div class="gallery-item"><img src="{_img('rmsf.png', d)}"><div class="caption"><h4>RMSF</h4></div></div>
<div class="gallery-item"><img src="{_img('rdf.png', d)}"><div class="caption"><h4>RDF</h4></div></div>
<div class="gallery-item"><img src="{_img('sasa_rg.png', d)}"><div class="caption"><h4>SASA &amp; Rg</h4></div></div>
<div class="gallery-item"><img src="{_img('pmf.png', d)}"><div class="caption"><h4>PMF</h4></div></div>
<div class="gallery-item"><img src="{_img('reaction_coordinate.png', d)}"><div class="caption"><h4>Reaction Coordinate</h4></div></div>
<div class="gallery-item"><img src="{_img('free_energy_landscape_2d.png', d)}"><div class="caption"><h4>Free Energy Landscape</h4></div></div>
</div>

<h2>AA-level analysis</h2>
<div class="gallery">
<div class="gallery-item"><img src="{_img('hbonds_timeseries.png', d)}"><div class="caption"><h4>H-bond Count</h4></div></div>
<div class="gallery-item"><img src="{_img('hbond_occupancy.png', d)}"><div class="caption"><h4>H-bond Occupancy</h4></div></div>
<div class="gallery-item"><img src="{_img('aa_contact_map_D_A.png', d)}"><div class="caption"><h4>AA Contact Map</h4></div></div>
<div class="gallery-item"><img src="{_img('aa_top_residue_pairs_D_A.png', d)}"><div class="caption"><h4>Top Pairs</h4></div></div>
<div class="gallery-item"><img src="{_img('aa_interface_hbonds_D_A.png', d)}"><div class="caption"><h4>Interface H-bonds</h4></div></div>
<div class="gallery-item"><img src="{_img('aa_residue_binding_contribution_D_A.png', d)}"><div class="caption"><h4>Per-Residue</h4></div></div>
</div>

<h2>Binding analysis</h2>
<div class="gallery">
<div class="gallery-item"><img src="{_img('contact_map.png', d)}"><div class="caption"><h4>CG Contact Map</h4></div></div>
<div class="gallery-item"><img src="{_img('binding_energy_pairs.png', d)}"><div class="caption"><h4>Binding Energy</h4></div></div>
<div class="gallery-item"><img src="{_img('binding_energy_summary.png', d)}"><div class="caption"><h4>Mean Energies</h4></div></div>
<div class="gallery-item"><img src="{_img('interface_residues.png', d)}"><div class="caption"><h4>Interface Residues</h4></div></div>
<div class="gallery-item"><img src="{_img('interaction_energy.png', d)}"><div class="caption"><h4>Interaction Energy</h4></div></div>
<div class="gallery-item wide"><img src="{_img('binding_dashboard.png', d)}"><div class="caption"><h4>Binding Dashboard</h4></div></div>
</div>

<h2>Structure</h2>
<div class="gallery">
<div class="gallery-item wide"><img src="{_img('structure_snapshots.png', d)}"><div class="caption"><h4>Structure Snapshots</h4></div></div>
<div class="gallery-item wide"><img src="{_img('architecture.png', d)}"><div class="caption"><h4>Architecture Diagram</h4></div></div>
</div>
""")


def build_manual_pages():
    """Build user guide / manual pages — interactive style."""
    sys_keys = grid(3,
        node("or", "name", "string, required", "System name. Used for the output directory name."),
        node("or", "pdb_source", "path, required", "Path to the input PDB structure file."),
        node("or", "output_dir", "path, optional", "Output directory. Defaults to <code>outputs/{name}</code>."),
    )
    dyn_keys = grid(2,
        node("bl", "steps", "int, required", "Number of integration steps for this stage."),
        node("bl", "time_step", "float, default 0.02", "Integration timestep in picoseconds."),
        node("bl", "temperature", "float, default 300.0", "Target temperature in Kelvin."),
        node("bl", "ensemble", 'string, default "NVT"', "Thermodynamic ensemble: NVT or NPT."),
        node("gr", "eval_stride", "int, default 10", "Steps between full QCloud+ML evaluations. Key performance parameter: 10 (accurate), 50 (fast), 100 (very fast)."),
        node("bl", "stages", "list, required", 'Ordered stage names: <code>["nvt","npt","production"]</code>'),
    )
    _write("manual/configuration.html", "Configuration", f"""
<h1>Configuration <span class="accent">Reference</span></h1>
<p class="page-desc">Complete reference for the TOML configuration file format. Click any parameter to expand.</p>

{layer("var(--orange)", "[SYSTEM] SECTION", sys_keys)}

{flow("bl", "dynamics stages")}

{layer("var(--blue)", "[DYNAMICS.STAGE] PARAMETERS", dyn_keys, "eval_stride is the key performance knob")}

{flow("go", "examples")}

{layer("var(--gold)", "MINIMAL CONFIG", '<pre><code>[system]\\nname = "my_protein"\\npdb_source = "structure.pdb"\\n\\n[dynamics]\\nstages = ["production"]\\n\\n[dynamics.production]\\nsteps = 10000\\ntime_step = 0.02\\ntemperature = 300.0</code></pre>')}

{layer("var(--gold)", "FULL PRODUCTION CONFIG", '<pre><code>[system]\\nname = "barnase_barstar"\\npdb_source = "benchmarks/reference_cases/data/1BRS.pdb"\\noutput_dir = "outputs/barnase_barstar_2ns"\\n\\n[dynamics]\\nstages = ["nvt", "npt", "production"]\\n\\n[dynamics.nvt]\\nsteps = 5000\\ntime_step = 0.02\\ntemperature = 300.0\\nensemble = "NVT"\\n\\n[dynamics.npt]\\nsteps = 5000\\ntime_step = 0.02\\ntemperature = 300.0\\nensemble = "NPT"\\n\\n[dynamics.production]\\nsteps = 100000\\ntime_step = 0.02\\ntemperature = 300.0\\neval_stride = 50</code></pre>', collapsed=True)}
""")

    cmds = "".join([
        node("gr", "neurocgmd run config.toml", "Full pipeline", "Prepare + simulate + back-map + analyze + plot. One command does everything."),
        node("bl", "neurocgmd prepare config.toml", "Preparation only", "Import PDB, build CG topology, generate prepared_bundle.json."),
        node("pu", "neurocgmd analyze config.toml", "Analyze existing data", "Run analysis pipeline on existing trajectory. Can run on partial data while simulation continues."),
        node("te", "neurocgmd info", "Platform capabilities", "Color-coded display of all layers, integrators, analysis tools, supported systems."),
    ])
    pipeline = "".join([
        node("or", "1. Import", "Parse PDB, identify chains, map to CG beads"),
        node("or", "2. Topology", "Build bonds, angles, nonbonded lists"),
        node("bl", "3. NVT", "Thermostat coupling at constant volume"),
        node("bl", "4. NPT", "Pressure coupling at constant pressure"),
        node("gr", "5. Production", "Hybrid dynamics: CG + QCloud + ML"),
        node("te", "6. Back-map", "CG trajectory to AA coordinates"),
        node("pk", "7. Analyze", "Compute all observables, generate 20+ plots"),
    ])
    _write("manual/running.html", "Running Simulations", f"""
<h1>Running <span class="accent">Simulations</span></h1>
<p class="page-desc">The NeuroCGMD command-line interface. Click any command to expand.</p>

{layer("var(--green)", "COMMANDS", '<div class="ng" style="grid-template-columns:1fr">' + cmds + '</div>')}

{flow("or", "the run pipeline")}

{layer("var(--orange)", "PIPELINE STAGES", '<div class="ng" style="grid-template-columns:1fr">' + pipeline + '</div>', "neurocgmd run executes all 7 stages in order")}

{flow("go", "output directory")}

{layer("var(--gold)", "OUTPUT STRUCTURE", '<pre><code>outputs/my_system/\\n  prepared_bundle.json          # Topology &amp; parameters\\n  energies.csv                  # Energy time series (live)\\n  traj.jsonl                    # CG trajectory (JSONL)\\n  run_summary.json              # Run metadata\\n  cg_trajectory.pdb             # CG multi-model PDB\\n  aa_backmapped_trajectory.pdb  # AA multi-model PDB\\n  aa_reference.pdb              # Crystal structure\\n  plots/                        # 20+ analysis PNGs</code></pre>')}
""")

    cg_obs = grid(2,
        node("bl","Energy","Direct from integrator","PE, KE, total energy. Output: <code>energy_timeseries.png</code>"),
        node("bl","RMSD / RMSF","Structural drift &amp; fluctuation","Mass-weighted deviation from reference. Output: <code>rmsd.png</code>, <code>rmsf.png</code>"),
        node("bl","RDF g(r)","Pair correlation","Volume-normalized radial distribution. Output: <code>rdf.png</code>"),
        node("bl","SASA + Rg","Surface &amp; compactness","Shrake-Rupley SASA + mass-weighted Rg. Output: <code>sasa_rg.png</code>"),
        node("bl","PMF","Boltzmann inversion","W(r) = &minus;kT ln[P(r)]. Output: <code>pmf.png</code>"),
        node("bl","Free Energy","2D landscape","F(x,y) = &minus;kT ln[P(x,y)/P_max]. Output: <code>free_energy_landscape_2d.png</code>"),
    )
    aa_obs = grid(2,
        node("te","H-bonds","Distance &lt; 3.5&Aring; + angle &gt; 120&deg;","Backbone + side-chain. PRO excluded as donor. ARG, LYS, SER, THR, TYR, TRP, ASN, GLN, HIS donors. ASP, GLU, ASN, GLN, HIS, SER, THR, TYR acceptors."),
        node("te","Residue Contacts","Heavy-atom closest approach","Cutoff 4.5&Aring;. CA pre-filter at 12&Aring;. Per amino-acid resolution."),
        node("te","Interface Pairs","Ranked by contact + H-bond","Top 30 pairs with frequency bars. Blue = contact, red = H-bond overlay."),
        node("te","Per-Residue Binding","Hotspot profiles","Sum of contacts per residue. Hotspot residues auto-labeled."),
        node("te","Inter-Chain H-bonds","Backbone N-O network","Angle-validated. Top 20 most persistent pairs."),
        node("te","AA Trajectory","PDB export","Multi-model back-mapped trajectory for Chimera/VMD/PyMOL."),
    )
    qc_obs = grid(2,
        node("pu","Structural Events","Bond forming/breaking","Classified from correction magnitude spikes above baseline."),
        node("pu","Event Timeline","Magnitudes over time","Scatter plot of correction events colored by type."),
        node("pu","Energy Decomposition","Per-bead breakdown","Bonded, nonbonded, total per CG bead."),
        node("pu","Interaction Energy","Inter-entity LJ","Total LJ interaction between all entity pairs over time."),
    )
    _write("manual/analysis.html", "Analysis Pipeline", f"""
<h1>Analysis <span class="accent">Pipeline</span></h1>
<p class="page-desc">All observables computed automatically. Click any to expand method details.</p>

{layer("var(--blue)", "CG-LEVEL OBSERVABLES", cg_obs, "collective dynamics &mdash; same physics at both levels")}

{flow("te", "AA-level from back-mapped structures")}

{layer("var(--teal)", "AA-LEVEL OBSERVABLES", aa_obs, "requires actual amino acid identity and geometry")}

{flow("pu", "quantum correction insights")}

{layer("var(--purple)", "QCLOUD ANALYSIS", qc_obs, "specific to the quantum correction feedback loop")}

{flow("pk", "auto-detected for multi-chain systems")}

{layer("var(--pink)", "BINDING ANALYSIS", grid(2,
    node("pk","Entity Detection","Groups by protein type","Same bead count = same protein. Auto-pairs: D-A, E-B, F-C."),
    node("pk","Pairwise Energy","LJ binding per pair","Time series of interaction energy between biological binding pairs."),
    node("pk","Contact Maps","CG + AA level","Frequency heatmaps at both bead and residue resolution."),
    node("pk","Binding Dashboard","4-panel overview","COM distance, contacts, energy, correlation scatter."),
), "automatically detected for multi-chain systems")}
""")

    info_sources = grid(3,
        node("bl","CG Forces","Classical dynamics","Drive the base motion of all beads."),
        node("pu","QCloud Corrections","Quantum physics","Refine forces on priority regions."),
        node("go","ML Residual","Learned patterns","Fill in corrections between QCloud evaluations."),
    )
    method_steps = grid(2,
        node("te","1. Bead-to-Atom Mapping","Labels &rarr; residue ranges","Each CG bead label encodes chain + residue range. Resolves to all constituent AA atoms from reference PDB."),
        node("te","2. Interpolation Weights","Distance-weighted, same-chain",
             '<div class="fm">w<sub>i</sub> = (1/d<sub>i</sub>) / &sum;(1/d<sub>j</sub>)</div>Nearest 2 beads on same chain. Snap to weight 1.0 within 0.01&Aring;.'),
        node("te","3. Displacement Interpolation","Apply weighted CG deltas",
             '<div class="fm">r<sub>AA</sub><sup>(a)</sup>(t) = r<sub>AA</sub><sup>(a)</sup>(0) + &sum;<sub>i</sub> w<sub>i</sub> [r<sub>CG</sub><sup>(i)</sup>(t) &minus; r<sub>CG</sub><sup>(i)</sup>(0)]</div>At t=0, back-mapped = original crystal structure (&Delta;=0).'),
        node("te","4. Bond Relaxation","8 iterations of constraint fixing",
             'C-N target 1.33&Aring; (45% push/pull). CA-CA target 3.80&Aring; (30% pull). Entire residues dragged to preserve geometry.'),
    )
    outputs = grid(2,
        node("te","aa_backmapped_trajectory.pdb","Multi-model AA trajectory"),
        node("te","aa_backmapped_initial/final.pdb","First and last AA frames"),
        node("te","aa_backmapped_combined.pdb","Initial + final in one file"),
        node("bl","aa_reference.pdb","Original crystal structure"),
    )
    _write("manual/back-mapping.html", "CG-to-AA Back-mapping", f"""
<h1>CG &rarr; AA <span class="accent">Back-mapping</span></h1>
<p class="page-desc">How all-atom coordinates are reconstructed from coarse-grained trajectories.</p>

{layer("var(--blue)", "THE COOPERATIVE PRINCIPLE", info_sources, "AA positions carry all three information layers")}

{flow("te", "four-step method")}

{layer("var(--teal)", "METHOD", method_steps, "click each step to see the math")}

{eq("<b>r</b><sub>AA</sub><sup>(a)</sup>(t) = <b>r</b><sub>AA</sub><sup>(a)</sup>(0) + &sum;<sub>i</sub> w<sub>i</sub> [<b>r</b><sub>CG</sub><sup>(i)</sup>(t) &minus; <b>r</b><sub>CG</sub><sup>(i)</sup>(0)]",
    "AA coordinates carry CG dynamics + QCloud quantum corrections + ML residual")}

{flow("go", "output files")}

{layer("var(--gold)", "OUTPUT FILES", outputs, "all PDB files load in Chimera, VMD, or PyMOL")}
""")


def build_reference_pages():
    """Build reference manual pages — substantive, like real MD documentation."""
    # Architecture page is the handcrafted interactive one — skip regenerating it
    pass  # reference/architecture.html is already built interactively

    # Force Fields
    bonded = grid(2,
        node("bl", "Harmonic Bonds", "V(r) = &frac12;k(r &minus; r<sub>0</sub>)&sup2;",
             "The standard harmonic bond potential between connected CG beads. Provides the bonded backbone connectivity that maintains chain integrity during dynamics.<br><br>"
             "Default parameters derived from MARTINI-class coarse-grained force fields:<br>"
             "<div class='fm'>k = 1250 kJ/(mol&middot;nm&sup2;)</div>"
             "<div class='fm'>r<sub>0</sub> = 0.37 nm</div>"
             "The high spring constant ensures bonds oscillate around equilibrium rather than stretch significantly. For typical CG timesteps of 0.02 ps, this provides stable dynamics without constraints."),
        node("bl", "Harmonic Angles", "V(&theta;) = &frac12;k<sub>&theta;</sub>(&theta; &minus; &theta;<sub>0</sub>)&sup2;",
             "Angular potential between three consecutive bonded beads. Controls chain stiffness and local geometry. Angle parameters can be system-specific or derived from mapped AA distributions."),
    )
    nonbonded = grid(2,
        node("pu", "Lennard-Jones (Shifted)", "V(r) = 4&epsilon;[(&sigma;/r)<sup>12</sup>&minus;(&sigma;/r)<sup>6</sup>] &minus; V(r<sub>cut</sub>)",
             "The standard 12-6 Lennard-Jones potential with a shift at the cutoff distance. The r<sup>&minus;12</sup> term models Pauli repulsion at short range. The r<sup>&minus;6</sup> term models van der Waals dispersion attraction.<br><br>"
             "The potential is shifted so V(r<sub>cut</sub>) = 0, ensuring energy continuity at the cutoff and preventing discontinuous jumps that cause energy drift.<br><br>"
             "<div class='fm'>&epsilon; = 2.0 kJ/mol &mdash; well depth (interaction strength)</div>"
             "<div class='fm'>&sigma; = 0.47 nm &mdash; effective bead diameter</div>"
             "<div class='fm'>r<sub>min</sub> = 2<sup>1/6</sup>&sigma; &asymp; 0.528 nm &mdash; equilibrium distance</div>"
             "The force is:<br>"
             "<div class='fm'>F(r) = (24&epsilon;/r)[2(&sigma;/r)<sup>12</sup> &minus; (&sigma;/r)<sup>6</sup>]</div>"),
        node("pu", "Coulomb Electrostatics", "V(r) = q<sub>i</sub>q<sub>j</sub> / (4&pi;&epsilon;<sub>0</sub>r)",
             "Optional real-space Coulomb potential for charged beads. Applied when the force field includes electrostatic interactions (e.g., charged amino acid side-chains in CG representation). Can be combined with reaction-field or shifted cutoff methods."),
    )
    neighbor = node("gr", "O(N) Cell-List Neighbor Search", "Spatial hashing for efficient pair finding",
        "Divides the simulation box into cubic cells sized to the interaction cutoff. Each particle is assigned to a cell based on position. Pair interactions are computed only between particles in the same or 26 adjacent cells.<br><br>"
        "Computational complexity: O(N) vs O(N&sup2;) for brute-force. For a 76-bead system with 2.5 nm cutoff, this reduces pair evaluations from ~2900 to ~200 per step.<br><br>"
        "The list is rebuilt every step in the current implementation. Future versions may add Verlet list buffering with skin distance for reduced rebuild frequency.")

    _write("reference/forcefields.html", "Force Fields", f"""
<h1>Force Field <span class="accent">Reference</span></h1>
<p class="page-desc">Interaction potentials, functional forms, parameters, and implementation details. NeuroCGMD uses a MARTINI-derived coarse-grained force field with quantum corrections from the QCloud layer.</p>

{layer("var(--blue)", "BONDED INTERACTIONS", bonded, "maintain chain connectivity and local geometry")}

{flow("pu", "nonbonded interactions")}

{layer("var(--purple)", "NONBONDED INTERACTIONS", nonbonded, "van der Waals and electrostatics")}

{eq("<b>V</b><sub>total</sub> = &sum;<sub>bonds</sub> &frac12;k(r&minus;r<sub>0</sub>)&sup2; + &sum;<sub>pairs</sub> 4&epsilon;[(&sigma;/r)<sup>12</sup>&minus;(&sigma;/r)<sup>6</sup>] + &sum;<sub>charges</sub> q<sub>i</sub>q<sub>j</sub>/r",
    "Classical CG energy &mdash; corrected by QCloud and ML during simulation")}

{flow("gr", "efficient pair search")}

{layer("var(--green)", "NEIGHBOR LIST", neighbor, "O(N) spatial hashing")}
""")

    # Integrators
    vv = node("bl", "Velocity-Verlet Langevin", "Primary integrator &mdash; two force evaluations per step",
        "The workhorse integrator for production dynamics. Implements the velocity-Verlet algorithm with an optional Langevin thermostat for NVT sampling.<br><br>"
        "<strong>Algorithm (per timestep):</strong><br>"
        "<div class='fm'>1. v<sub>n+&frac12;</sub> = v<sub>n</sub> + (&Delta;t/2m) F<sub>n</sub></div>"
        "<div class='fm'>2. Apply thermostat: v<sub>damped</sub> = e<sup>&minus;&gamma;&Delta;t</sup> v + &radic;((1&minus;e<sup>&minus;2&gamma;&Delta;t</sup>) kT/m) &middot; &xi;</div>"
        "<div class='fm'>3. x<sub>n+1</sub> = x<sub>n</sub> + &Delta;t &middot; v<sub>n+&frac12;</sub></div>"
        "<div class='fm'>4. Evaluate F<sub>n+1</sub> at new positions</div>"
        "<div class='fm'>5. v<sub>n+1</sub> = v<sub>n+&frac12;</sub> + (&Delta;t/2m) F<sub>n+1</sub></div><br>"
        "The thermostat (step 2) is only applied when <code>stochastic=True</code>. The noise term &xi; is drawn from a standard normal distribution, ensuring correct Boltzmann sampling at the target temperature.<br><br>"
        "Kinetic energy is computed at each step: E<sub>k</sub> = &frac12; &sum; m<sub>i</sub> |v<sub>i</sub>|&sup2;")

    baoab = node("pu", "BAOAB Langevin Splitting", "Symmetric splitting &mdash; one force evaluation per step",
        "An alternative integrator based on the BAOAB splitting scheme (Leimkuhler &amp; Matthews, 2013). Splits the Liouvillian into kick (B), drift (A), and Ornstein-Uhlenbeck (O) substeps in the symmetric pattern B-A-O-A-B.<br><br>"
        "Advantages: better configurational sampling accuracy for equivalent timestep; only one force evaluation per step (vs two for velocity-Verlet). Commonly used in enhanced sampling applications.")

    params = grid(2,
        node("go", "time_step", "Default: 0.02 ps", "Integration timestep. For CG dynamics, timesteps of 0.01&ndash;0.04 ps are typical. Stability depends on the fastest vibrational mode (bond frequency)."),
        node("go", "friction_coefficient", "Default: from state", "Langevin coupling constant &gamma; (ps<sup>&minus;1</sup>). Controls the strength of thermostat coupling. Higher values: stronger coupling, faster equilibration. Lower values: more natural dynamics."),
        node("go", "stochastic", "Default: False", "Enable the stochastic Langevin noise term. Required for NVT sampling. When False, the integrator runs as deterministic velocity-Verlet (NVE-like)."),
        node("go", "thermal_energy_scale", "Default: 1.0", "Scaling factor for kT in the noise term. Normally 1.0 for standard simulations. Can be adjusted for replica exchange or simulated tempering protocols."),
    )

    _write("reference/integrators.html", "Integrators", f"""
<h1>Integrator <span class="accent">Reference</span></h1>
<p class="page-desc">Time integration algorithms for propagating the equations of motion. NeuroCGMD implements velocity-Verlet and BAOAB Langevin integrators with stochastic thermostats for canonical (NVT) sampling.</p>

{layer("var(--blue)", "VELOCITY-VERLET LANGEVIN", vv, "primary integrator for production dynamics")}

{eq("<b>v</b><sub>n+&frac12;</sub> = <b>v</b><sub>n</sub> + <sup>&Delta;t</sup>&frasl;<sub>2m</sub> <b>F</b><sub>total</sub> &nbsp;&rarr;&nbsp; <b>x</b><sub>n+1</sub> = <b>x</b><sub>n</sub> + &Delta;t&middot;<b>v</b><sub>n+&frac12;</sub> &nbsp;&rarr;&nbsp; <b>v</b><sub>n+1</sub> = <b>v</b><sub>n+&frac12;</sub> + <sup>&Delta;t</sup>&frasl;<sub>2m</sub> <b>F</b><sub>n+1</sub>",
    "F<sub>total</sub> includes classical, QCloud, and ML contributions")}

{flow("pu", "alternative integrator")}

{layer("var(--purple)", "BAOAB LANGEVIN SPLITTING", baoab, "symmetric splitting, one force eval per step", collapsed=True)}

{flow("go", "parameters")}

{layer("var(--gold)", "PARAMETERS", params, "click to expand details")}
""")

    # QCloud
    qcloud_components = grid(2,
        node("pu", "Region Selector", "Priority-based molecular subgraph selection",
             "At each evaluation step, the region selector identifies which parts of the system should receive quantum-level corrections. Selection criteria include:<br>"
             "- Structural importance (binding interfaces, catalytic sites)<br>"
             "- Prior correction magnitudes (particles that received large corrections previously)<br>"
             "- Event analyzer priority scores (regions where structural events are occurring)<br>"
             "- Computational budget (total QCloud compute is bounded per step)<br><br>"
             "Regions are molecular subgraphs &mdash; connected sets of CG beads that form a physically meaningful unit for quantum evaluation."),
        node("pu", "Quantum Correction Model", "AA-level force deltas on selected regions",
             "For each selected region, the correction model evaluates quantum-level forces and computes the difference from classical forces. The output is a set of per-particle force deltas:<br><br>"
             "<div class='fm'>&Delta;F<sub>QCloud</sub>(i) = F<sub>quantum</sub>(i) &minus; F<sub>classical</sub>(i)</div><br>"
             "Each force delta is bounded to prevent numerical instability:<br>"
             "<div class='fm'>|&Delta;F<sub>component</sub>| &le; 5.0 kJ/(mol&middot;nm)</div><br>"
             "Energy deltas are also bounded to prevent QCloud from dominating the total energy."),
        node("pu", "Event Analyzer", "Structural event detection from correction patterns",
             "Maintains per-particle statistics of correction magnitudes using exponential moving averages. When a correction spike exceeds baseline + n&sigma;, it classifies the event:<br><br>"
             "<strong>Bond forming</strong> &mdash; large correction on particles moving closer together<br>"
             "<strong>Bond breaking</strong> &mdash; large correction on particles moving apart<br>"
             "<strong>Conformational shift</strong> &mdash; sustained elevated correction above baseline<br>"
             "<strong>Interface rearrangement</strong> &mdash; correlated correction spikes across a binding interface<br><br>"
             "Detection threshold: correction > baseline + spike_threshold_sigma &times; std"),
        node("pu", "Adaptive Feedback Loop", "Corrections inform next region selection",
             "The feedback loop is the key architectural innovation. Event detection generates per-particle priority scores that feed directly back into the next region selection cycle:<br><br>"
             "<div class='fm'>priority(i) = base_score(i) + min(0.5, |&Delta;F(i)| / baseline(i))</div><br>"
             "This creates an <strong>adaptive focus mechanism</strong>: quantum compute automatically concentrates on the most physically interesting regions &mdash; where chemistry is happening. As a system equilibrates, QCloud naturally shifts focus from initial contacts to the evolving binding interface."),
    )

    _write("reference/qcloud.html", "QCloud Layer", f"""
<h1>QCloud <span class="accent">Quantum</span> Correction Layer</h1>
<p class="page-desc">The QCloud layer provides quantum-informed force corrections that refine the classical CG dynamics. It operates at evaluation steps (controlled by <code>eval_stride</code>) and produces per-particle force deltas that are additively coupled to the classical forces. The adaptive feedback loop ensures quantum compute is focused where it matters most.</p>

{layer("var(--purple)", "COMPONENTS", qcloud_components, "click any component to expand implementation details")}

{flow("pu", "force coupling")}

{eq("<b>F</b><sub>total</sub>(i) = <b>F</b><sub>classical</sub>(i) + bounded(&Delta;<b>F</b><sub>QCloud</sub>(i))",
    "additive coupling &mdash; QCloud corrections refine, never replace, classical forces")}

{flow("pu", "propagation chain")}

{layer("var(--purple)", "HOW CORRECTIONS REACH ATOMIC COORDINATES", grid(1,
    node("pu", "QCloud &rarr; Forces &rarr; Velocities &rarr; Positions &rarr; Trajectory &rarr; Back-map &rarr; AA",
         "Complete propagation path",
         "The QCloud force correction &Delta;F on particle i at step t affects the position at step t+1 through the integrator:<br><br>"
         "<div class='fm'>&Delta;x(QCloud) = &frac12; &Delta;t&sup2; &middot; &Delta;F<sub>QCloud</sub> / m</div><br>"
         "This position change is stored in the trajectory. When back-mapped to AA, the all-atom coordinates reflect the quantum correction. Every AA residue contact, H-bond, and binding interaction computed on the back-mapped structure therefore <strong>carries quantum-corrected physics</strong>.")
), "the full chain from quantum correction to atomic-level analysis")}
""")

    # ML Residual
    _write("reference/ml-residual.html", "ML Residual", f"""
<h1>ML Residual <span class="accent">Learning</span> Layer</h1>
<p class="page-desc">A neural network that learns the mapping from molecular configurations to quantum correction forces, trained on-the-fly during simulation. Over time, the ML model learns the correction landscape and can predict corrections between expensive QCloud evaluations, progressively reducing quantum compute cost.</p>

{layer("var(--gold)", "MODEL ARCHITECTURE", grid(1,
    node("go", "Multi-Layer Perceptron (MLP)", "Maps particle features to force correction vectors",
         "<strong>Input layer:</strong> particle positions (3D), velocities (3D), local environment descriptors derived from the neighbor list<br><br>"
         "<strong>Hidden layers:</strong> 2&ndash;3 fully-connected layers with ReLU activation. Width scales with system size.<br><br>"
         "<strong>Output layer:</strong> per-particle force correction vectors (&Delta;F<sub>x</sub>, &Delta;F<sub>y</sub>, &Delta;F<sub>z</sub>)<br><br>"
         "The model learns a residual &mdash; the difference between classical CG forces and the true (quantum-corrected) forces. This is mathematically equivalent to learning the QCloud correction function.")
), "neural residual model")}

{flow("go", "training protocol")}

{layer("var(--gold)", "ON-THE-FLY TRAINING", grid(2,
    node("go", "Training Loop", "QCloud corrections as ground truth",
         "At each full evaluation step (every <code>eval_stride</code> steps), the QCloud layer produces force corrections. These become training targets:<br><br>"
         "<div class='fm'>Loss = &sum;<sub>i</sub> |&Delta;F<sub>predicted</sub>(i) &minus; &Delta;F<sub>QCloud</sub>(i)|&sup2;</div><br>"
         "The model updates via stochastic gradient descent (SGD) with momentum. Learning rate is adaptive. The training set grows continuously throughout the simulation."),
    node("go", "Mixing Strategy", "&alpha; = 0.35 when QCloud active",
         "When QCloud corrections are available (full evaluation step), ML predictions are scaled by &alpha; = 0.35 to avoid double-counting:<br><br>"
         "<div class='fm'>F<sub>total</sub> = F<sub>CG</sub> + &Delta;F<sub>QCloud</sub> + 0.35 &middot; &Delta;F<sub>ML</sub></div><br>"
         "Between evaluation steps (lightweight path), ML predictions run at full scale (&alpha; = 1.0):<br><br>"
         "<div class='fm'>F<sub>total</sub> = F<sub>CG</sub> + 1.0 &middot; &Delta;F<sub>ML</sub></div><br>"
         "This allows the ML model to provide quantum-like corrections even on lightweight steps."),
), "the model improves throughout the simulation")}

{flow("go", "safety mechanisms")}

{layer("var(--gold)", "DRIFT CONTROL &amp; UNCERTAINTY", grid(2,
    node("go", "Energy Drift Monitor", "Rolling-window conservation check",
         "Monitors total energy over a rolling window of states. If the ML corrections introduce excessive drift (energy trending non-physically), the model predictions are automatically scaled down to prevent simulation instability."),
    node("go", "Ensemble Uncertainty", "Variance from multiple predictions",
         "An ensemble of ML models can provide uncertainty estimates. When predictions disagree (high variance), this signals that the current configuration is out-of-distribution and QCloud should be called for a direct correction."),
), "preventing ML-induced artifacts")}
""")

    # Observables
    structural = grid(2,
        node("bl", "RMSD", "Root Mean Square Deviation",
             "Mass-weighted structural deviation from a reference configuration. Measures overall structural drift over the trajectory.<br><br>"
             "<div class='fm'>RMSD = &radic;(&sum; m<sub>i</sub> |r<sub>i</sub> &minus; r<sub>ref,i</sub>|&sup2; / &sum; m<sub>i</sub>)</div><br>"
             "Units: nm. Computed at CG level. A plateau in RMSD indicates the system has reached structural equilibrium."),
        node("bl", "RMSF", "Root Mean Square Fluctuation",
             "Per-particle time-averaged positional fluctuation. Identifies flexible loops, hinge regions, and rigid structural cores.<br><br>"
             "<div class='fm'>RMSF(i) = &radic;(&lt;|r<sub>i</sub> &minus; &lt;r<sub>i</sub>&gt;|&sup2;&gt;)</div><br>"
             "Units: nm. High RMSF beads correspond to flexible surface loops; low RMSF beads form the structural core."),
        node("bl", "Radius of Gyration (Rg)", "Mass-weighted compactness measure",
             "Quantifies the overall size and compactness of the molecular system.<br><br>"
             "<div class='fm'>R<sub>g</sub> = &radic;(&sum; m<sub>i</sub> |r<sub>i</sub> &minus; r<sub>COM</sub>|&sup2; / &sum; m<sub>i</sub>)</div><br>"
             "Units: nm. Decreasing Rg indicates compaction (e.g., binding). Stable Rg indicates structural integrity."),
        node("bl", "SASA", "Solvent Accessible Surface Area",
             "Shrake-Rupley algorithm with configurable probe radius (default 0.14 nm). Distributes test points uniformly on a sphere around each particle and counts exposed points.<br><br>"
             "Units: nm&sup2;. Decreasing SASA upon binding indicates interface burial. Tracks solvent exposure changes during conformational transitions."),
    )
    thermo = grid(2,
        node("bl", "RDF g(r)", "Radial Distribution Function",
             "Pair correlation function describing the probability of finding a particle at distance r from another, normalized by ideal gas density.<br><br>"
             "<div class='fm'>g(r) = &lt;n(r)&gt; / (4&pi;r&sup2; &Delta;r &rho;)</div><br>"
             "Units: dimensionless. First peak position reveals equilibrium inter-particle distance. g(r) &rarr; 1 at large r indicates proper normalization. Computed from fixed simulation volume."),
        node("bl", "Potential of Mean Force", "Free energy from Boltzmann inversion",
             "Converts the probability distribution of a collective variable (typically COM distance) into a free energy profile:<br><br>"
             "<div class='fm'>W(r) = &minus;k<sub>B</sub>T ln[P(r) / P<sub>max</sub>]</div><br>"
             "Units: kT. Minima correspond to metastable states. Barriers indicate transition state energetics. Limited by sampling &mdash; regions not visited have undefined PMF."),
        node("bl", "2D Free Energy Landscape", "Two-dimensional PMF",
             "Joint probability distribution of two collective variables (COM distance and R<sub>g</sub>) converted to free energy:<br><br>"
             "<div class='fm'>F(x,y) = &minus;k<sub>B</sub>T ln[P(x,y) / P<sub>max</sub>]</div><br>"
             "Displayed as a heat map with trajectory overlay showing start and end points. Identifies binding pathways and metastable basins."),
        node("bl", "Reaction Coordinate", "Center-of-mass distance",
             "Mass-weighted center-of-mass distance between binding partners over time. Serves as the primary reaction coordinate for binding processes.<br><br>"
             "Units: nm. Decreasing distance indicates binding progression. Fluctuations reveal the dynamics of approach, contact, and separation events."),
    )
    hbond = grid(2,
        node("te", "H-bond Detection", "Geometric criteria with angle validation",
             "H-bonds are detected on back-mapped AA structures using combined distance and angle criteria to minimize false positives:<br><br>"
             "<strong>Distance:</strong> donor-acceptor &lt; 3.5 &Aring;<br>"
             "<strong>Angle:</strong> D-H&middot;&middot;&middot;A &gt; 120&deg; (approximated from CA&rarr;donor direction)<br>"
             "<strong>Proline:</strong> excluded as donor (secondary amine, no H)<br><br>"
             "<strong>Backbone:</strong> N (donor) and O (acceptor)<br>"
             "<strong>Side-chain donors:</strong> ARG (NE, NH1, NH2), LYS (NZ), ASN (ND2), GLN (NE2), HIS (ND1, NE2), TRP (NE1), SER (OG), THR (OG1), TYR (OH), CYS (SG)<br>"
             "<strong>Side-chain acceptors:</strong> ASP (OD1, OD2), GLU (OE1, OE2), ASN (OD1), GLN (OE1), HIS (ND1, NE2), SER (OG), THR (OG1), TYR (OH)"),
        node("te", "Contact Analysis", "Heavy-atom closest approach",
             "Residue-residue contacts detected from back-mapped AA heavy-atom distances:<br><br>"
             "<strong>Contact cutoff:</strong> 4.5 &Aring; (heavy atoms, excludes H)<br>"
             "<strong>CA pre-filter:</strong> skip pairs with CA-CA &gt; 12 &Aring; (efficiency)<br>"
             "<strong>Persistent contact:</strong> present in &gt; 50% of analyzed frames<br><br>"
             "Contacts are computed per-amino-acid (e.g., GLY31-HIS102) using actual residue names from the PDB, providing publication-ready interface maps."),
    )

    _write("reference/observables.html", "Observables", f"""
<h1>Observable <span class="accent">Reference</span></h1>
<p class="page-desc">Complete reference for all molecular observables computed by the NeuroCGMD analysis pipeline. Each observable is computed at the level where it is most physically meaningful &mdash; CG for collective properties, AA for atomic-detail properties.</p>

{layer("var(--blue)", "STRUCTURAL OBSERVABLES (CG)", structural, "computed on CG bead positions")}

{flow("bl", "thermodynamic observables")}

{layer("var(--blue)", "THERMODYNAMIC OBSERVABLES (CG)", thermo, "free energy, pair correlations, reaction coordinates")}

{flow("te", "AA-level observables from back-mapped structures")}

{layer("var(--teal)", "H-BOND &amp; CONTACT ANALYSIS (AA)", hbond, "requires actual amino acid identity and atomic geometry")}
""")


def build_compare():
    def row(feat, ncg, grm, omm, nmd):
        g = lambda v: f'<span style="color:var(--green)">{v}</span>' if "Native" in v or "Built-in" in v or "On-the-fly" in v or "Integrated" in v or "Auto" in v or "pip" in v or "500" in v or "Angle" in v else v
        return f"<tr><td><strong>{feat}</strong></td><td>{g(ncg)}</td><td>{grm}</td><td>{omm}</td><td>{nmd}</td></tr>"

    engines = grid(4,
        node("gr", "NeuroCGMD", "500 KB, pip install", "Pure Python. CG+QM+ML cooperative. Auto analysis. Integrated back-mapping."),
        node("bl", "GROMACS", "~50 MB, compile", "C/C++/CUDA. Industry standard. Excellent performance. Requires separate tools for analysis."),
        node("pu", "OpenMM", "~200 MB, conda", "C++/Python hybrid. GPU-accelerated. Plugin architecture for ML."),
        node("go", "NAMD", "~100 MB, compile", "C++/Charm++. Scalable parallel. MPI required."),
    )

    _write("compare.html", "Comparison", f"""
<h1>How NeuroCGMD <span class="accent">Compares</span></h1>
<p class="page-desc">Feature comparison with established molecular dynamics engines.</p>

{engines}

{flow("gr", "feature-by-feature comparison")}

{layer("var(--green)", "COMPARISON TABLE", '''<table>
<tr><th></th><th>NeuroCGMD</th><th>GROMACS</th><th>OpenMM</th><th>NAMD</th></tr>
''' + row("Language","Pure Python","C/C++/CUDA","C++/Python","C++/Charm++") + row("Install","<span style='color:var(--green)'>pip install</span>","Compile","conda","Compile") + row("Size","<span style='color:var(--green)'>500 KB</span>","~50 MB","~200 MB","~100 MB") + row("Dependencies","numpy, matplotlib","FFTW, MPI, CUDA...","OpenCL, CUDA...","MPI, Charm++...") + row("CG+QM coupling","<span style='color:var(--green)'>Native cooperative</span>","Separate tools","Via plugins","Not built-in") + row("ML corrections","<span style='color:var(--green)'>On-the-fly training</span>","External","Via OpenMM-ML","External") + row("Auto analysis","<span style='color:var(--green)'>Built-in (20+ plots)</span>","Separate tools","Manual","Separate tools") + row("Back-mapping","<span style='color:var(--green)'>Integrated</span>","Third-party","Third-party","Third-party") + row("Binding analysis","<span style='color:var(--green)'>Auto-detected</span>","Manual setup","Manual setup","Manual setup") + '''</table>''')
}
""")


def build_contact():
    cards = '''<div class="contact-grid">
<a href="mailto:bessuman.academia@gmail.com?subject=%5BNeuroCGMD%5D%20Academic%20Collaboration%20Inquiry&body=Hi%2C%0A%0AI%20am%20writing%20regarding%20a%20potential%20academic%20collaboration%20involving%20NeuroCGMD.%0A%0AInstitution%3A%20%0AResearch%20area%3A%20%0A%0ADetails%3A%0A" class="cc"><div class="tp">Research</div><h4>Academic Collaboration</h4><p style="font-size:12px;color:var(--text-muted)">Joint research, shared datasets, publications</p><div class="sj">[NeuroCGMD] Academic Collaboration</div></a>
<a href="mailto:bessuman.academia@gmail.com?subject=%5BNeuroCGMD%5D%20Bug%20Report%20%E2%80%94%20v1.0.0&body=NeuroCGMD%20version%3A%201.0.0%0APython%20version%3A%20%0AOS%3A%20%0A%0ADescription%3A%0A%0ASteps%20to%20reproduce%3A%0A1.%20%0A2.%20%0A3.%20%0A%0AExpected%3A%0AActual%3A%0A" class="cc"><div class="tp">Bugs</div><h4>Bug Reports</h4><p style="font-size:12px;color:var(--text-muted)">Unexpected behavior, crashes, errors</p><div class="sj">[NeuroCGMD] Bug Report</div></a>
<a href="mailto:bessuman.academia@gmail.com?subject=%5BNeuroCGMD%5D%20Technical%20Support%20Request&body=Hi%2C%0A%0AI%20need%20help%20with%20NeuroCGMD.%0A%0AWhat%20I%20am%20trying%20to%20do%3A%0A%0AWhat%20happened%3A%0A%0AConfig%20file%3A%0A" class="cc"><div class="tp">Support</div><h4>Technical Help</h4><p style="font-size:12px;color:var(--text-muted)">Setup, configuration, workflows</p><div class="sj">[NeuroCGMD] Support Request</div></a>
<a href="mailto:bessuman.academia@gmail.com?subject=%5BNeuroCGMD%5D%20Commercial%20Licensing%20%26%20Partnership&body=Hi%2C%0A%0AWe%20are%20interested%20in%20exploring%20commercial%20use%20of%20NeuroCGMD.%0A%0AOrganization%3A%20%0AUse%20case%3A%20%0A%0ADetails%3A%0A" class="cc"><div class="tp">Commercial</div><h4>Licensing &amp; Partnership</h4><p style="font-size:12px;color:var(--text-muted)">Enterprise, consulting, custom dev</p><div class="sj">[NeuroCGMD] Commercial Inquiry</div></a>
</div>'''

    _write("contact.html", "Contact & Cite", f"""
<h1>Contact &amp; <span class="accent">Citation</span></h1>
<p class="page-desc">Each link opens a pre-filled email with the appropriate subject and body template.</p>

{cards}

{flow("go", "citing NeuroCGMD")}

{layer("var(--gold)", "BIBTEX CITATION", '<pre><code>@software{{neurocgmd2026,\\n  author  = {{Essuman, Bernard}},\\n  title   = {{NeuroCGMD: Quantum-Classical-CG-ML Cooperative\\n             Molecular Dynamics Engine}},\\n  year    = {{2026}},\\n  version = {{1.0.0}},\\n  url     = {{https://github.com/sciencemaths-collab/neurocgmd}}\\n}}</code></pre>')}

<div class="chips">
  <div class="chip te">MIT License</div>
  <div class="chip bl">Free for all use</div>
  <div class="chip pu">Including commercial</div>
</div>
""")


def main():
    print("Building NeuroCGMD documentation website...")
    print(f"  Output: {SITE}/")
    print()

    # Copy plot assets
    asset_dir = SITE / "assets" / "plots"
    asset_dir.mkdir(parents=True, exist_ok=True)
    if PLOTS.exists():
        needed = [
            "architecture.png", "dashboard.png", "energy_timeseries.png",
            "rmsd.png", "rmsf.png", "rdf.png", "sasa_rg.png", "pmf.png",
            "reaction_coordinate.png", "structure_snapshots.png",
            "hbonds_timeseries.png", "hbond_occupancy.png",
            "energy_decomposition.png", "interaction_energy.png",
            "free_energy_landscape_2d.png", "contact_map.png",
            "binding_energy_pairs.png", "binding_energy_summary.png",
            "binding_dashboard.png", "interface_residues.png",
            "aa_contact_map_D_A.png", "aa_top_residue_pairs_D_A.png",
            "aa_residue_binding_contribution_D_A.png",
            "aa_interface_hbonds_D_A.png",
            "aa_interface_hbonds_E_B.png",
        ]
        for name in needed:
            src = PLOTS / name
            if src.exists():
                shutil.copy2(src, asset_dir / name)
        print(f"  Copied {len(list(asset_dir.glob('*.png')))} plot images")
    else:
        print("  Warning: no plots directory found — images will be missing")

    # Build pages
    print("\n  Pages:")
    build_home()
    build_install()
    build_quickstart()
    build_tutorial_barnase()
    build_tutorial_gallery()
    build_manual_pages()
    build_reference_pages()
    build_compare()
    build_contact()

    n_pages = len(list(SITE.rglob("*.html")))
    print(f"\n  {n_pages} pages built.")
    print(f"\n  Open in browser: file://{SITE}/index.html")
    print(f"  Deploy to GitHub Pages: push docs/website/ to gh-pages branch")


if __name__ == "__main__":
    main()
