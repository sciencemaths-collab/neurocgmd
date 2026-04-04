# IO Boundary

The repository keeps an `io/` top-level folder because it is part of the required
platform architecture. It is intentionally **not** a Python package in Section 1
to avoid colliding with Python's standard-library `io` module.

The serialization and export layer is now implemented in `io/export_registry.py`.
It is still kept outside a Python package boundary, so callers should access it
via explicit file-path loading rather than `import io.export_registry`.

Current Section 14 export artifact:

- `export_registry.py`
  - writes `dashboard.json` and `index.html` bundles for the local live dashboard
  - remains import-safe by avoiding package-style `io` imports

Classification: `[adapted]`
