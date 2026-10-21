from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

@dataclass
class EDAReport:
    overview: Dict[str, Any]
    missing: Dict[str, Any]
    numeric: Dict[str, Any]
    categorical: Dict[str, Any]
    correlation: Dict[str, Any]
    sample_size: int
    elapsed_seconds: float
    original_shape: tuple
    warnings: list[str] = field(default_factory=list)
    target: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    sql_source: Any = field(default=None, repr=False)
    recommended_engine: str = "pandas"
    sample: Optional[pd.DataFrame] = field(default=None, repr=False, compare=False)

    # ──────────────────────────────────────────────────────────────────────
    # Console
    # ──────────────────────────────────────────────────────────────────────

    def show(self) -> None:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        from rich.panel import Panel

        console = Console()
        orig_rows = self.original_shape[0]
        ncols = self.original_shape[1]

        console.print(
            Panel(
                f"[bold cyan]FlashEDA Report[/bold cyan]\n"
                f"Dataset: [green]{orig_rows:,}[/green] rows × [green]{ncols}[/green] cols  "
                f"| Analysed on [yellow]{self.sample_size:,}[/yellow] sampled rows  "
                f"| ⏱  [white]{self.elapsed_seconds:.2f}s[/white]",
                expand=False,
            )
        )

        if self.warnings:
            for w in self.warnings:
                console.print(f"  [bold yellow]⚠  {w}[/bold yellow]")
            console.print()

        miss = self.missing
        console.print(
            f"[bold]Missing values[/bold]  overall: [red]{miss['overall_null_pct']}%[/red]  "
            f"| rows with any null: [red]{miss['rows_with_any_null_pct']}%[/red]"
        )
        if miss["critical_columns"]:
            console.print(f"  Critical (>50% null): [red]{', '.join(miss['critical_columns'])}[/red]")
        if miss["warn_columns"]:
            console.print(f"  Warn    (>20% null): [yellow]{', '.join(miss['warn_columns'])}[/yellow]")
        console.print()

        num_cols = self.numeric.get("numeric_col_names", [])
        if num_cols:
            t = Table(title="Numeric columns", box=box.SIMPLE, show_lines=False)
            for h in ["Column", "Mean", "Std", "Min", "Median", "Max", "Outlier %", "Skew"]:
                t.add_column(h, style="white")
            for col in num_cols:
                s = self.numeric["columns"].get(col, {})
                if "error" in s:
                    continue
                flag = "[yellow]⚠[/yellow] " if s.get("skewed") else ""
                t.add_row(
                    col,
                    str(s.get("mean", "")),
                    str(s.get("std", "")),
                    str(s.get("min", "")),
                    str(s.get("median", "")),
                    str(s.get("max", "")),
                    f"{s.get('outlier_pct', 0)}%",
                    f"{flag}{s.get('skewness', '')}",
                )
            console.print(t)

        cat_cols = self.categorical.get("categorical_col_names", [])
        if cat_cols:
            t = Table(title="Categorical columns", box=box.SIMPLE, show_lines=False)
            for h in ["Column", "Unique", "Cardinality ratio", "Mode", "Mode %", "Flag"]:
                t.add_column(h)
            for col in cat_cols:
                s = self.categorical["columns"].get(col, {})
                if "error" in s:
                    continue
                flag = ""
                if s.get("likely_id_column"):
                    flag = "[red]likely ID[/red]"
                elif s.get("high_cardinality"):
                    flag = "[yellow]high cardinality[/yellow]"
                t.add_row(
                    col,
                    str(s.get("unique_count", "")),
                    str(s.get("cardinality_ratio", "")),
                    str(s.get("mode", ""))[:30],
                    f"{s.get('mode_pct', '')}%",
                    flag,
                )
            console.print(t)

        if self.target:
            _show_target(console, self.target)

        strong = self.correlation.get("numeric_strong_pairs", [])
        if strong:
            console.print("[bold]Strong numeric correlations (|r| ≥ 0.8)[/bold]")
            for pair in strong[:10]:
                console.print(f"  {pair['col_a']} ↔ {pair['col_b']}  r={pair['pearson_r']}")
            console.print()

    # ──────────────────────────────────────────────────────────────────────
    # HTML
    # ──────────────────────────────────────────────────────────────────────

    def to_html(self) -> str:
        return _render_html(self)

    def save_html(self, path: str = "flasheda_report.html") -> Path:
        p = Path(path)
        p.write_text(self.to_html(), encoding="utf-8")
        print(f"HTML report saved → {p.resolve()}")
        return p

    # ──────────────────────────────────────────────────────────────────────
    # PDF  ← THE METHOD THAT WAS MISSING
    # ──────────────────────────────────────────────────────────────────────

    def save_pdf(self, path: str = "flasheda_report.pdf") -> Path:
        """
        Save a full EDA report as a PDF file.
        Requires: pip install fpdf2
        """
        try:
            from fpdf import FPDF
        except ImportError:
            raise ImportError(
                "fpdf2 is required for PDF output. "
                "Install it with: pip install fpdf2"
            )

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        orig_rows, orig_cols = self.original_shape

        # ── Title bar ──────────────────────────────────────────────────────
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_fill_color(26, 26, 46)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 14, "FlashEDA Report",
                 new_x="LMARGIN", new_y="NEXT", fill=True, align="C")

        pdf.set_font("Helvetica", "", 9)
        pdf.set_fill_color(40, 40, 70)
        pdf.cell(
            0, 8,
            f"Rows: {orig_rows:,}  |  Columns: {orig_cols}  |  "
            f"Sample: {self.sample_size:,}  |  Time: {self.elapsed_seconds:.2f}s",
            new_x="LMARGIN", new_y="NEXT", fill=True, align="C"
        )
        pdf.ln(4)
        pdf.set_text_color(0, 0, 0)

        # ── Warnings ───────────────────────────────────────────────────────
        if self.warnings:
            _pdf_section_title(pdf, "Warnings")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(180, 100, 0)
            for w in self.warnings:
                pdf.cell(0, 6, _ascii(f"  ! {w}"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)

        # ── Overview ───────────────────────────────────────────────────────
        _pdf_section_title(pdf, "Dataset Overview")
        _pdf_kv_table(pdf, [
            ["Total rows",              f"{orig_rows:,}"],
            ["Total columns",           str(orig_cols)],
            ["Sample size used",        f"{self.sample_size:,}"],
            ["Analysis time",           f"{self.elapsed_seconds:.3f}s"],
            ["Overall null %",          f"{self.missing.get('overall_null_pct', 0)}%"],
            ["Duplicate rows (sample)", f"{self.overview.get('duplicate_pct_in_sample', 0)}%"],
        ])

        # ── Missing values ─────────────────────────────────────────────────
        _pdf_section_title(pdf, "Missing Values")
        miss_rows = [
            [col,
             str(info.get("null_count", 0)),
             f"{info.get('null_pct', 0)}%",
             info.get("severity", "ok").upper()]
            for col, info in self.missing.get("per_column", {}).items()
        ]
        _pdf_table(pdf, ["Column", "Null Count", "Null %", "Status"], miss_rows)

        # ── Numeric columns ────────────────────────────────────────────────
        num_cols = self.numeric.get("numeric_col_names", [])
        if num_cols:
            _pdf_section_title(pdf, "Numeric Columns")
            num_rows = []
            for col in num_cols:
                s = self.numeric["columns"].get(col, {})
                if "error" in s:
                    continue
                num_rows.append([
                    col,
                    str(s.get("mean", "")),
                    str(s.get("std", "")),
                    str(s.get("min", "")),
                    str(s.get("median", "")),
                    str(s.get("max", "")),
                    f"{s.get('outlier_pct', 0)}%",
                    str(s.get("skewness", "")),
                ])
            _pdf_table(pdf,
                       ["Column", "Mean", "Std", "Min", "Median",
                        "Max", "Outlier%", "Skew"],
                       num_rows)

        # ── Categorical columns ────────────────────────────────────────────
        cat_cols = self.categorical.get("categorical_col_names", [])
        if cat_cols:
            _pdf_section_title(pdf, "Categorical Columns")
            cat_rows = []
            for col in cat_cols:
                s = self.categorical["columns"].get(col, {})
                if "error" in s:
                    continue
                flag = ("LIKELY ID" if s.get("likely_id_column")
                        else "HIGH CARD" if s.get("high_cardinality") else "")
                cat_rows.append([
                    col,
                    str(s.get("unique_count", "")),
                    str(s.get("cardinality_ratio", "")),
                    str(s.get("mode", ""))[:20],
                    f"{s.get('mode_pct', '')}%",
                    flag,
                ])
            _pdf_table(pdf,
                       ["Column", "Unique", "Cardinality",
                        "Mode", "Mode %", "Flag"],
                       cat_rows)

        # ── Target ─────────────────────────────────────────────────────────
        if self.target:
            t = self.target
            _pdf_section_title(pdf, f"Target: {t['target']}")
            _pdf_kv_table(pdf, _target_summary(t))
            power = t.get("feature_power", [])[:15]
            if power:
                leaks = {l["column"] for l in t.get("leakage_columns", [])}
                _pdf_table(pdf, ["Feature", "Predictive power", "Flag"],
                           [[p["column"], str(p["score"]),
                             "POSSIBLE LEAK" if p["column"] in leaks else ""] for p in power])

        # ── Correlations ───────────────────────────────────────────────────
        strong = self.correlation.get("numeric_strong_pairs", [])
        if strong:
            _pdf_section_title(pdf, "Strong Correlations (|r| >= 0.8)")
            _pdf_table(pdf,
                       ["Column A", "Column B", "Pearson r"],
                       [[p["col_a"], p["col_b"], str(p["pearson_r"])]
                        for p in strong[:20]])

        # ── Footer ─────────────────────────────────────────────────────────
        pdf.ln(6)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 6,
                 "Generated by FlashEDA - constant-time EDA for any dataset size",
                 align="C", new_x="LMARGIN", new_y="NEXT")

        p = Path(path)
        pdf.output(str(p))
        print(f"PDF report saved -> {p.resolve()}")
        return p

    # ──────────────────────────────────────────────────────────────────────
    # Code generation
    # ──────────────────────────────────────────────────────────────────────

    def plan(self, features: Union[bool, Dict[str, Any]] = False, **options):
        """
        Engine-neutral cleaning plan built from this report's findings.

        features : True (or a dict of options for flasheda.features.plan_features)
                   to append model-ready features: date parts, encodings, ratios
        options  : passed to the cleaning planner, e.g. group_rare=False
        """
        from .codegen import plan_from_report
        plan = plan_from_report(self, **options)
        if features:
            from .features import plan_features
            plan = plan_features(self, plan, target=options.get("target"),
                                 **(features if isinstance(features, dict) else {}))
        return plan

    def codegen(
        self,
        engine: Optional[str] = None,
        *,
        output: str = "cleaned.parquet",
        validate: bool = True,
        features: Union[bool, Dict[str, Any]] = False,
        **plan_options,
    ):
        """
        Generate a cleaning script for the full dataset.

        engine        : "pandas" | "polars" | "duckdb" | "pyspark"
                        (default: the engine recommended for this dataset size)
        validate      : run the generated code on the sample and raise if it
                        doesn't match the reference result
        features      : also add model-ready features (see EDAReport.plan)
        plan_options  : passed to the planner, e.g. clip_outliers=False
        """
        from .codegen import generate

        code = generate(
            self.plan(features=features, **plan_options),
            engine or self.recommended_engine,
            source=self.source,
            sql=self.sql_source,
            output=output,
            sample=self.sample,
        )
        if validate and self.sample is not None:
            result = code.validate()
            if not result.ok:
                raise RuntimeError(f"Generated code failed validation: {result}")
        return code

    def contract(self, allow_extra_columns: bool = True):
        """Data contract learned from the sample (see flasheda.contract)."""
        from .contract import infer_contract
        return infer_contract(self, allow_extra_columns)

    def train(self, engine: Optional[str] = None, **options):
        """Compare models on the sample and generate prep + training scripts
        (see flasheda.modeling.train_from_report)."""
        from .modeling import train_from_report
        return train_from_report(self, engine, **options)

    # ──────────────────────────────────────────────────────────────────────
    # Serialisation
    # ──────────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overview":         self.overview,
            "missing":          self.missing,
            "numeric":          self.numeric,
            "categorical":      self.categorical,
            "correlation":      self.correlation,
            "sample_size":      self.sample_size,
            "elapsed_seconds":  self.elapsed_seconds,
            "original_shape":   list(self.original_shape),
            "warnings":         self.warnings,
            "target":           self.target,
            "source":           self.source or (self.sql_source.display if self.sql_source else None),
            "recommended_engine": self.recommended_engine,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __repr__(self) -> str:
        rows, cols = self.original_shape
        return (
            f"EDAReport(rows={rows:,}, cols={cols}, "
            f"sample={self.sample_size:,}, elapsed={self.elapsed_seconds:.2f}s, "
            f"engine={self.recommended_engine!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Target helpers
# ─────────────────────────────────────────────────────────────────────────────

def _target_summary(t: Dict[str, Any]) -> list:
    rows = [["Task", t["task"]], ["Rows with target", f"{t['count']:,}"]]
    if t.get("null_count"):
        rows.append(["Target nulls", f"{t['null_pct']}%"])
    if t["task"] == "regression":
        s = t["target_stats"]
        rows.append(["Mean / std", f"{s['mean']} / {s['std']}"])
        rows.append(["Skewness", str(s["skewness"])])
    else:
        top = ", ".join(f"{c['value']} {c['pct']}%" for c in t["classes"][:5])
        rows.append(["Classes", f"{t['n_classes']} ({top})"])
        rows.append(["Imbalance ratio", f"{t['imbalance_ratio']}:1"])
    rows.append(["Metric", t["recommended_metric"]])
    rows.append(["Validation", t["recommended_split"]])
    return rows


def _show_target(console, t: Dict[str, Any]) -> None:
    from rich.table import Table
    from rich import box

    console.print(f"[bold]Target[/bold] [cyan]{t['target']}[/cyan]")
    for key, val in _target_summary(t):
        console.print(f"  {key}: {val}")
    for leak in t.get("leakage_columns", []):
        console.print(f"  [bold red]Possible leakage:[/bold red] {leak['column']} — {leak['reason']}")
    power = t.get("feature_power", [])[:10]
    if power:
        table = Table(title="Predictive power (single feature, 0–1)", box=box.SIMPLE)
        table.add_column("Feature")
        table.add_column("Power")
        for p in power:
            table.add_row(str(p["column"]), str(p["score"]))
        console.print(table)
    console.print()


def _ascii(value: Any) -> str:
    """The PDF uses a core font, which only covers Latin-1."""
    return str(value).replace("—", "-").replace("–", "-").replace("²", "2").encode(
        "latin-1", "replace").decode("latin-1")


# ─────────────────────────────────────────────────────────────────────────────
# PDF helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pdf_section_title(pdf, title: str) -> None:
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(230, 235, 245)
    pdf.set_text_color(26, 26, 46)
    pdf.cell(0, 8, _ascii(f"  {title}"),
             new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def _pdf_kv_table(pdf, rows: list) -> None:
    pdf.set_font("Helvetica", "", 9)
    col_w = 70
    for key, val in rows:
        pdf.set_fill_color(248, 249, 252)
        pdf.cell(col_w, 7, _ascii(f"  {key}"), border=1, fill=True)
        pdf.cell(col_w, 7, _ascii(f"  {val}")[:60], border=1,
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)


def _pdf_table(pdf, headers: list, rows: list) -> None:
    if not rows:
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 6, "  No data.",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        return

    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    col_w = page_w / len(headers)

    # Header
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(26, 26, 46)
    pdf.set_text_color(255, 255, 255)
    for h in headers:
        pdf.cell(col_w, 7, _ascii(f" {h}"), border=1, fill=True)
    pdf.ln()

    # Rows
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(0, 0, 0)
    for i, row in enumerate(rows):
        pdf.set_fill_color(255, 255, 255) if i % 2 == 0 \
            else pdf.set_fill_color(245, 247, 252)
        for cell in row:
            pdf.cell(col_w, 6, _ascii(f" {str(cell)[:22]}"), border=1, fill=True)
        pdf.ln()

    pdf.ln(4)


# ─────────────────────────────────────────────────────────────────────────────
# HTML renderer
# ─────────────────────────────────────────────────────────────────────────────

def _esc(value: Any) -> str:
    import html
    return html.escape(str(value))


def _render_html(r: EDAReport) -> str:
    num_stats   = r.numeric.get("numeric_col_names", [])
    cat_stats   = r.categorical.get("categorical_col_names", [])
    strong_corr = r.correlation.get("numeric_strong_pairs", [])
    miss_per_col = r.missing.get("per_column", {})

    num_rows_html = ""
    for col in num_stats:
        s = r.numeric["columns"].get(col, {})
        if "error" in s:
            continue
        skew_badge = '<span class="badge warn">skewed</span>' if s.get("skewed") else ""
        out_badge  = '<span class="badge warn">outliers</span>' if s.get("outlier_pct", 0) > 5 else ""
        num_rows_html += f"""
        <tr>
          <td><strong>{col}</strong></td>
          <td>{s.get('mean','')}</td><td>{s.get('std','')}</td>
          <td>{s.get('min','')}</td><td>{s.get('median','')}</td>
          <td>{s.get('max','')}</td>
          <td>{s.get('outlier_pct','')}%</td>
          <td>{s.get('skewness','')} {skew_badge}</td>
          <td>{out_badge}</td>
        </tr>"""

    cat_rows_html = ""
    for col in cat_stats:
        s = r.categorical["columns"].get(col, {})
        if "error" in s:
            continue
        flag = ""
        if s.get("likely_id_column"):
            flag = '<span class="badge danger">likely ID</span>'
        elif s.get("high_cardinality"):
            flag = '<span class="badge warn">high cardinality</span>'
        top     = s.get("top_values", [])[:5]
        top_str = " · ".join(f"{t['value']} ({t['pct']}%)" for t in top)
        cat_rows_html += f"""
        <tr>
          <td><strong>{col}</strong></td>
          <td>{s.get('unique_count','')}</td>
          <td>{s.get('cardinality_ratio','')}</td>
          <td>{str(s.get('mode',''))[:40]}</td>
          <td>{s.get('mode_pct','')}%</td>
          <td class="muted">{top_str[:80]}</td>
          <td>{flag}</td>
        </tr>"""

    miss_rows_html = ""
    for col, info in miss_per_col.items():
        sev   = info.get("severity", "ok")
        color = {"ok": "#27ae60", "warn": "#f39c12",
                 "critical": "#e74c3c"}.get(sev, "#888")
        bar_w = min(100, int(info.get("null_pct", 0)))
        miss_rows_html += f"""
        <tr>
          <td>{col}</td>
          <td>{info.get('null_count', 0)}</td>
          <td>
            <div class="bar-bg">
              <div class="bar-fill" style="width:{bar_w}%;background:{color}"></div>
            </div>
            {info.get('null_pct', 0)}%
          </td>
          <td><span style="color:{color};font-weight:500">{sev}</span></td>
        </tr>"""

    corr_html = ""
    for p in strong_corr[:15]:
        corr_html += (f"<tr><td>{p['col_a']}</td>"
                      f"<td>{p['col_b']}</td>"
                      f"<td>{p['pearson_r']}</td></tr>")

    target_html = ""
    if r.target:
        t = r.target
        leaks = {l["column"] for l in t.get("leakage_columns", [])}
        summary = "".join(
            f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in _target_summary(t))
        rows = []
        for p in t.get("feature_power", [])[:15]:
            leak = p["column"] in leaks
            color = "#e74c3c" if leak else "#2f5bd3"
            badge = '<span class="badge danger">possible leak</span>' if leak else ""
            rows.append(
                f"<tr><td>{_esc(p['column'])}</td><td>"
                f'<div class="bar-bg"><div class="bar-fill" '
                f'style="width:{int(p["score"] * 100)}%;background:{color}"></div></div>'
                f"{p['score']}</td><td>{badge}</td></tr>"
            )
        power = "".join(rows)
        target_html = f"""
  <div class="card">
    <h2>Target: {_esc(t['target'])}</h2>
    <table><tbody>{summary}</tbody></table>
    <h2 style="margin-top:18px">Predictive power (single feature, 0–1)</h2>
    <table>
      <thead><tr><th>Feature</th><th>Power</th><th>Flags</th></tr></thead>
      <tbody>{power}</tbody>
    </table>
  </div>"""

    warn_html = "".join(
        f'<div class="warn-banner">⚠  {w}</div>'
        for w in r.warnings
    )

    orig_rows, orig_cols = r.original_shape

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FlashEDA Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:system-ui,sans-serif;background:#f5f7fa;color:#222;font-size:14px}}
  .container{{max-width:1100px;margin:0 auto;padding:24px 16px}}
  header{{background:#1a1a2e;color:#fff;padding:24px 32px;border-radius:10px;margin-bottom:24px}}
  header h1{{font-size:22px;font-weight:600;margin-bottom:6px}}
  header .meta{{font-size:13px;color:#aac}}
  .card{{background:#fff;border-radius:10px;padding:20px 24px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,.07)}}
  .card h2{{font-size:15px;font-weight:600;margin-bottom:14px;border-bottom:1px solid #eee;padding-bottom:8px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{text-align:left;padding:6px 10px;background:#f0f2f5;font-weight:500}}
  td{{padding:6px 10px;border-top:1px solid #f0f0f0;vertical-align:middle}}
  .badge{{display:inline-block;padding:2px 7px;border-radius:12px;font-size:11px;font-weight:600}}
  .badge.warn{{background:#fff3cd;color:#856404}}
  .badge.danger{{background:#f8d7da;color:#842029}}
  .muted{{color:#888;font-size:12px}}
  .bar-bg{{display:inline-block;width:80px;height:8px;background:#eee;border-radius:4px;vertical-align:middle;margin-right:6px}}
  .bar-fill{{height:8px;border-radius:4px}}
  .warn-banner{{background:#fff3cd;color:#856404;padding:10px 14px;border-radius:8px;margin-bottom:10px;font-size:13px}}
  .stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:4px}}
  .stat-box{{background:#f8f9fc;border-radius:8px;padding:14px 16px}}
  .stat-box .label{{font-size:11px;color:#888;margin-bottom:4px}}
  .stat-box .value{{font-size:20px;font-weight:600;color:#1a1a2e}}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>FlashEDA Report</h1>
    <div class="meta">
      {orig_rows:,} rows x {orig_cols} columns &nbsp;|&nbsp;
      Analysed on {r.sample_size:,} sampled rows &nbsp;|&nbsp;
      Completed in {r.elapsed_seconds:.2f}s
    </div>
  </header>
  {warn_html}
  <div class="card">
    <h2>Dataset overview</h2>
    <div class="stats-grid">
      <div class="stat-box"><div class="label">Total rows</div><div class="value">{orig_rows:,}</div></div>
      <div class="stat-box"><div class="label">Columns</div><div class="value">{orig_cols}</div></div>
      <div class="stat-box"><div class="label">Sample size</div><div class="value">{r.sample_size:,}</div></div>
      <div class="stat-box"><div class="label">Analysis time</div><div class="value">{r.elapsed_seconds:.2f}s</div></div>
      <div class="stat-box"><div class="label">Overall null %</div><div class="value">{r.missing.get('overall_null_pct',0)}%</div></div>
      <div class="stat-box"><div class="label">Duplicates (sample)</div><div class="value">{r.overview.get('duplicate_pct_in_sample',0)}%</div></div>
    </div>
  </div>
  <div class="card">
    <h2>Missing values</h2>
    <table>
      <thead><tr><th>Column</th><th>Null count</th><th>Null %</th><th>Status</th></tr></thead>
      <tbody>{miss_rows_html}</tbody>
    </table>
  </div>
  <div class="card">
    <h2>Numeric columns</h2>
    <table>
      <thead><tr><th>Column</th><th>Mean</th><th>Std</th><th>Min</th><th>Median</th><th>Max</th><th>Outlier %</th><th>Skewness</th><th>Flags</th></tr></thead>
      <tbody>{num_rows_html}</tbody>
    </table>
  </div>
  <div class="card">
    <h2>Categorical columns</h2>
    <table>
      <thead><tr><th>Column</th><th>Unique</th><th>Cardinality</th><th>Mode</th><th>Mode %</th><th>Top values</th><th>Flags</th></tr></thead>
      <tbody>{cat_rows_html}</tbody>
    </table>
  </div>
  {target_html}
  {"" if not corr_html else f'''
  <div class="card">
    <h2>Strong correlations (|r| >= 0.8)</h2>
    <table>
      <thead><tr><th>Column A</th><th>Column B</th><th>Pearson r</th></tr></thead>
      <tbody>{corr_html}</tbody>
    </table>
  </div>'''}
  <p class="muted" style="text-align:center;margin-top:16px">
    Generated by FlashEDA
  </p>
</div>
</body>
</html>"""