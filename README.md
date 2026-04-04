<p align="center">
  <img src="https://img.shields.io/badge/NeuroCGMD-v1.0.0-blue?style=for-the-badge&labelColor=0D1117&color=58A6FF" alt="version"/>
  <img src="https://img.shields.io/badge/python-3.11+-teal?style=for-the-badge&labelColor=0D1117&color=3FB9A2" alt="python"/>
  <img src="https://img.shields.io/badge/license-MIT-purple?style=for-the-badge&labelColor=0D1117&color=BC8CFF" alt="license"/>
  <img src="https://img.shields.io/badge/deps-numpy%20%2B%20matplotlib-gold?style=for-the-badge&labelColor=0D1117&color=F0C75E" alt="deps"/>
</p>

<h1 align="center">
  NeuroCGMD
</h1>

<h4 align="center">
  Quantum-Classical-CG-ML Cooperative Molecular Dynamics Engine
</h4>

<p align="center">
  <em>One simulation. One information stream. All-atom accuracy at coarse-grained speed.</em>
</p>

<p align="center">
  <a href="#install">Install</a> &nbsp;&bull;&nbsp;
  <a href="#quickstart">Quickstart</a> &nbsp;&bull;&nbsp;
  <a href="#architecture">Architecture</a> &nbsp;&bull;&nbsp;
  <a href="#analysis">Analysis</a> &nbsp;&bull;&nbsp;
  <a href="#contact">Contact</a>
</p>

---

## What is NeuroCGMD?

NeuroCGMD is a **next-generation molecular dynamics engine** that fuses four layers of physics into a single cooperative simulation:

| Layer | What it does | Why it matters |
|:------|:-------------|:---------------|
| **CG Dynamics** | Langevin integration with classical forcefields | Speed: ~1000 steps/min |
| **QCloud** | Quantum-informed force corrections on priority regions | Accuracy: AA-level physics where it counts |
| **ML Residual** | Neural network learns correction patterns on-the-fly | Efficiency: reduces QCloud calls over time |
| **Back-mapping** | Reconstructs full all-atom coordinates from CG | Detail: real residue names, H-bonds, contacts |

The result: **a 500 KB pure-Python package** that does what legacy codes need millions of lines of Fortran/C++ to achieve.

---

<a id="install"></a>
## Download & Install

> **Before you begin:** You need Python 3.11 or newer installed on your computer.
> To check, open a terminal and type `python3 --version`. If you see 3.11 or higher, you're ready.
> If not, download Python from [python.org/downloads](https://python.org/downloads).

### Quick reference

| Method | Command / Link |
|:-------|:---------------|
| **pip** (recommended) | `pip install neurocgmd` |
| **Wheel** (.whl) | [neurocgmd-1.0.0-py3-none-any.whl](https://pypi.org/project/neurocgmd/1.0.0/#files) |
| **Tarball** (.tar.gz) | [neurocgmd-1.0.0.tar.gz](https://pypi.org/project/neurocgmd/1.0.0/#files) |
| **Source** (GitHub) | [github.com/sciencemaths-collab/neurocgmd](https://github.com/sciencemaths-collab/neurocgmd) |
| **ZIP** (no git needed) | [Download ZIP](https://github.com/sciencemaths-collab/neurocgmd/archive/refs/heads/main.zip) |

---

### Option 1: pip install (recommended)

This is the easiest method. It works on macOS, Linux, and Windows.

1. Open a terminal (on macOS: search for "Terminal"; on Windows: open "Command Prompt" or "PowerShell")
2. Type the following command and press Enter:
```bash
pip install neurocgmd
```
3. Wait for it to finish. It will download NeuroCGMD and its dependencies (NumPy and Matplotlib) automatically.
4. That's it. NeuroCGMD is now installed.

> **Getting an error?** Try `pip3 install neurocgmd` or `python3 -m pip install neurocgmd` instead.
> If you see "Permission denied", try `pip install --user neurocgmd`.

---

### Option 2: Download the wheel file (.whl)

A wheel file is a pre-built package you download and install manually.

1. Go to the download page: **[pypi.org/project/neurocgmd/1.0.0/#files](https://pypi.org/project/neurocgmd/1.0.0/#files)**
2. Click on **neurocgmd-1.0.0-py3-none-any.whl** to download it
3. Open a terminal and navigate to where the file was saved (usually your Downloads folder):
```bash
cd ~/Downloads
```
4. Install it:
```bash
pip install neurocgmd-1.0.0-py3-none-any.whl
```

---

### Option 3: Download the tarball (.tar.gz)

A tarball is a compressed archive containing the source code.

1. Go to the download page: **[pypi.org/project/neurocgmd/1.0.0/#files](https://pypi.org/project/neurocgmd/1.0.0/#files)**
2. Click on **neurocgmd-1.0.0.tar.gz** to download it
3. Open a terminal and navigate to where the file was saved:
```bash
cd ~/Downloads
```
4. Extract the archive:
```bash
tar xzf neurocgmd-1.0.0.tar.gz
```
5. Go into the extracted folder:
```bash
cd neurocgmd-1.0.0
```
6. Install:
```bash
pip install .
```

> **Note:** The dot (`.`) at the end is important — it tells pip to install from the current folder.

---

### Option 4: Download from GitHub (source code)

This gives you the latest source code and is ideal if you want to browse or modify the code.

**Using git:**
```bash
git clone https://github.com/sciencemaths-collab/neurocgmd.git
cd neurocgmd
pip install .
```

**Without git** (download ZIP):
1. Go to: **[Download ZIP](https://github.com/sciencemaths-collab/neurocgmd/archive/refs/heads/main.zip)**
2. Your browser will download a file called `neurocgmd-main.zip`
3. Unzip it (double-click on macOS/Windows, or `unzip neurocgmd-main.zip` in terminal)
4. Open a terminal and go into the folder:
```bash
cd neurocgmd-main
```
5. Install:
```bash
pip install .
```

---

### Verify your installation

After installing with any method above, open a terminal and run:

```bash
neurocgmd --version
```

You should see:
```
neurocgmd 1.0.0
```

Then run the full capabilities display:
```bash
neurocgmd info
```

This shows a color-coded overview of all integrators, force fields, analysis tools, and supported systems.

> **If `neurocgmd` command is not found**, try: `python3 -m neurocgmd info`

---

### Run your first simulation

Once installed, you can run a simulation with a single command:

```bash
neurocgmd run examples/barnase_barstar.toml
```

This will:
- Import a protein structure from a PDB file
- Run equilibration and production molecular dynamics
- Apply quantum corrections and ML learning
- Back-map the trajectory to all-atom resolution
- Generate 30+ publication-quality analysis plots
- Export CG and AA trajectory files

All output is saved to the `outputs/` folder.

---

### Requirements

| Dependency | Version | Notes |
|:-----------|:--------|:------|
| Python | 3.11 or newer | [Download here](https://python.org/downloads) |
| NumPy | 1.24+ | Installed automatically by pip |
| Matplotlib | 3.7+ | Installed automatically by pip |

No CUDA, MPI, Fortran, GPU drivers, or compilation required. If you can run Python, you can run NeuroCGMD.

---

<a id="quickstart"></a>
## Quickstart

### 1. Run a simulation

```bash
neurocgmd run examples/barnase_barstar.toml
```

This single command:
- Imports the PDB structure and maps to coarse-grained beads
- Runs NVT equilibration, NPT equilibration, and production dynamics
- Applies QCloud quantum corrections with adaptive region focusing
- Trains the ML residual model on-the-fly
- Back-maps the CG trajectory to full all-atom coordinates
- Generates 20+ publication-quality analysis plots
- Exports CG and AA trajectory PDB files

### 2. Analyze an existing trajectory

```bash
neurocgmd analyze examples/barnase_barstar.toml
```

### 3. Write your own config

```toml
[system]
name = "my_protein"
pdb_source = "structures/my_protein.pdb"

[dynamics]
stages = ["nvt", "npt", "production"]

[dynamics.production]
steps = 100000
time_step = 0.02
temperature = 300.0
eval_stride = 50        # Full QCloud+ML every 50 steps
```

---

<a id="architecture"></a>
## Architecture

```
                    +--------------------------------------------------+
                    |            SIMULATION ENGINE                      |
                    |                                                  |
                    |   CG Dynamics    QCloud Layer    ML Residual     |
                    |   +---------+   +-----------+   +-----------+   |
                    |   | Langevin|-->| Region    |-->| Residual  |   |
                    |   | Integr. |   | Selector  |   | Predictor |   |
                    |   +---------+   +-----------+   +-----------+   |
                    |   | Force   |   | Quantum   |   | On-the-fly|   |
                    |   | Field   |   | Correct.  |   | Training  |   |
                    |   +---------+   +-----------+   +-----------+   |
                    |                                                  |
                    |   F_total = F_CG + F_QCloud + F_ML --> integrator|
                    +----------------------|---------------------------+
                                           |
                                           v
                    +--------------------------------------------------+
                    |              BACK-MAPPING                         |
                    |   CG positions --> interpolation --> AA coords    |
                    |   (carries CG + QCloud + ML information)         |
                    +----------------------|---------------------------+
                                           |
                              +------------+------------+
                              |            |            |
                              v            v            v
                        CG Analysis   AA Analysis   QCloud Analysis
                        (collective)  (atomic)      (corrections)
```

### The cooperative principle

Each CG position at every timestep encodes three sources of information:

1. **Classical CG forces** drive the base dynamics
2. **QCloud quantum corrections** refine forces on priority regions (adaptive focus from correction feedback)
3. **ML residual predictions** fill in learned correction patterns between QCloud evaluations

When we back-map CG to AA, the all-atom coordinates inherit all three layers. The AA-level analysis (residue contacts, H-bonds, binding hotspots) therefore reflects **quantum-corrected physics at atomic resolution**.

### Intelligent analysis routing

Each analysis goes to the level where it is most meaningful:

| Level | Analyses | Why this level |
|:------|:---------|:---------------|
| **CG** | RMSD, RMSF, Rg, SASA, PMF, free energy landscape, RDF | Collective variables don't benefit from AA resolution |
| **AA** | Residue-residue contacts, H-bonds (angle+distance), interface hotspots, binding decomposition | Needs actual amino acid identity and geometry |
| **QCloud** | Structural events, correction timeline, adaptive focus regions | Specific to the quantum correction feedback |

---

<a id="analysis"></a>
## Analysis Output

A single `neurocgmd run` produces:

### Dynamics & Thermodynamics
- Energy time series (PE, KE, total)
- RMSD / RMSF structural analysis
- Radial distribution function g(r)
- SASA and radius of gyration

### Free Energy
- Potential of mean force (Boltzmann inversion)
- 2D free energy landscape (COM distance vs Rg)
- Reaction coordinate trajectory

### Binding Interactions (auto-detected)
- Inter-chain contact maps (CG and AA level)
- Pairwise binding energy decomposition
- Top interacting residue pairs (with H-bond overlay)
- Per-residue binding contribution profiles
- Interface H-bond network with angle validation

### Quantum Correction Insights
- QCloud structural event detection
- Event timeline with correction magnitudes
- Per-bead energy decomposition

### Trajectory Export
- CG trajectory PDB (multi-model)
- AA back-mapped trajectory PDB
- Initial/final snapshots (CG and AA)
- Reference crystal structure

---

## How it compares

| | NeuroCGMD | GROMACS | OpenMM | NAMD |
|:--|:---------|:--------|:-------|:-----|
| **Language** | Pure Python | C/C++/CUDA | C++/Python | C++/Charm++ |
| **Install** | `pip install` | Compile from source | conda | Compile |
| **Package size** | 500 KB | ~50 MB | ~200 MB | ~100 MB |
| **Dependencies** | numpy, matplotlib | FFTW, MPI, CUDA... | OpenCL, CUDA... | MPI, Charm++... |
| **CG + QM coupling** | Native cooperative | Separate tools | Via plugins | Not built-in |
| **ML corrections** | On-the-fly training | External | Via OpenMM-ML | External |
| **Auto analysis** | Built-in (20+ plots) | Separate tools | Manual | Separate tools |
| **Back-mapping** | Integrated | Third-party | Third-party | Third-party |

---

## Project Layout

```
core/           State models, provenance, lifecycle registry
physics/        Neighbor lists, force kernels, cell lists
forcefields/    Hybrid engine: classical + QCloud + ML composition
integrators/    BAOAB Langevin, Velocity-Verlet Langevin
qcloud/         Quantum corrections, region selection, event analysis
ml/             Neural residual model, online training, uncertainty
sampling/       Production engine, stage runner, eval stride control
validation/     Observables: SASA, Rg, H-bonds, contacts, RDF, RMSD
scripts/        CLI, plotting, back-mapping, binding analysis
topology/       System topology, bead mapping, bond graphs
config/         TOML manifest parsing, protein mapping tables
chemistry/      Residue semantics, interface logic
spring/         SPRING universal optimizer + NeuroCGMD bridge
```

---

<a id="contact"></a>
## Contact

<table>
<tr><td><strong>Academic collaboration</strong></td><td><a href="mailto:bessuman.academia@gmail.com?subject=%5BNeuroCGMD%5D%20Academic%20Collaboration%20Inquiry&body=Hi%2C%0A%0AI%20am%20writing%20regarding%20a%20potential%20academic%20collaboration%20involving%20NeuroCGMD.%0A%0AInstitution%3A%20%0AResearch%20area%3A%20%0A%0ADetails%3A%0A">bessuman.academia@gmail.com</a></td></tr>
<tr><td><strong>Bug reports</strong></td><td><a href="mailto:bessuman.academia@gmail.com?subject=%5BNeuroCGMD%5D%20Bug%20Report%20%E2%80%94%20v1.0.0&body=NeuroCGMD%20version%3A%201.0.0%0APython%20version%3A%20%0AOS%3A%20%0A%0ADescription%3A%0A%0ASteps%20to%20reproduce%3A%0A1.%20%0A2.%20%0A3.%20%0A%0AExpected%20behavior%3A%0AActual%20behavior%3A%0A">bessuman.academia@gmail.com</a></td></tr>
<tr><td><strong>Technical support</strong></td><td><a href="mailto:bessuman.academia@gmail.com?subject=%5BNeuroCGMD%5D%20Technical%20Support%20Request&body=Hi%2C%0A%0AI%20need%20help%20with%20NeuroCGMD.%0A%0AWhat%20I%20am%20trying%20to%20do%3A%0A%0AWhat%20happened%3A%0A%0AConfig%20file%20(if%20relevant)%3A%0A">bessuman.academia@gmail.com</a></td></tr>
<tr><td><strong>Commercial licensing</strong></td><td><a href="mailto:bessuman.academia@gmail.com?subject=%5BNeuroCGMD%5D%20Commercial%20Licensing%20%26%20Partnership&body=Hi%2C%0A%0AWe%20are%20interested%20in%20exploring%20commercial%20use%20of%20NeuroCGMD.%0A%0AOrganization%3A%20%0AUse%20case%3A%20%0A%0ADetails%3A%0A">bessuman.academia@gmail.com</a></td></tr>
</table>

Each link opens a pre-filled email with the appropriate subject line and template.

---

## Citation

If you use NeuroCGMD in your research, please cite:

```bibtex
@software{neurocgmd2026,
  author  = {Essuman, Bernard},
  title   = {NeuroCGMD: Quantum-Classical-CG-ML Cooperative Molecular Dynamics Engine},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/sciencemaths-collab/neurocgmd}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with cooperative intelligence. 500 KB of Python that speaks the language of atoms.</sub>
</p>
