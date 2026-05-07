#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api/app.py — FastAPI serving layer for the Coredrill rhizome classifier.

Startup:
    uvicorn api.app:app --host 0.0.0.0 --port 8000

Environment variables:
    COREDRILL_PATH   path to coredrill_hierarchical.json  (required)
    MODEL_NAME       sentence-transformers model name      (default: sentence-transformers/LaBSE)
    TAU              softmax temperature                   (default: from coredrill json)
    API_KEY          bearer token for write endpoints      (optional; if unset, all endpoints are open)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Security, status, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

log = logging.getLogger("coredrill.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Lazy global state — loaded once on first request
# ---------------------------------------------------------------------------
_coredrill: dict | None = None
_model = None
_coredrill_path: str = os.environ.get("COREDRILL_PATH", "coredrill/coredrill_hierarchical.json")
_model_name: str = os.environ.get("MODEL_NAME", "sentence-transformers/LaBSE")
_api_key: str | None = os.environ.get("API_KEY")


def _get_coredrill() -> dict:
    global _coredrill
    if _coredrill is None:
        p = Path(_coredrill_path)
        if not p.exists():
            raise RuntimeError(
                f"Coredrill file not found: {_coredrill_path}\n"
                "Set the COREDRILL_PATH environment variable to the correct path.\n"
                "The coredrill file is produced by: python scripts/rhizome_engine.py build ..."
            )
        log.info(f"Loading coredrill from {p} ...")
        _coredrill = json.loads(p.read_text(encoding="utf-8"))
        log.info(f"Coredrill loaded. Nodes: {len(_coredrill.get('tree', {}))}")
    return _coredrill


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info(f"Loading embedding model: {_model_name} ...")
        _model = SentenceTransformer(_model_name)
        log.info("Model ready.")
    return _model


# ---------------------------------------------------------------------------
# Import predict / analyze from engine — the engine is the single source of truth
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from rhizome_engine import (  # type: ignore
    predict,
    analyze_text_rhizome,
    analyze_text_rhizome_v2,
    gram_inverse_scores,
    SKIP_NODES,
)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Coredrill — Rhizome Text Classifier",
    description=(
        "Geometry-native multilingual text classification over a 238-node rhizomatic taxonomy. "
        "Based on LaBSE embedding space structure. No trained classifiers. "
        "See the paper: *Geometry-Native Text Classification via High-Dimensional Embedding Space Analysis* (Lucas, 2026)."
    ),
    version="1.0.0",
    contact={"name": "Coredrill", "url": "https://github.com/your-username/coredrill"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)


def _check_key(credentials: HTTPAuthorizationCredentials | None):
    """If API_KEY is set, enforce bearer token auth."""
    if _api_key is None:
        return  # open mode
    if credentials is None or credentials.credentials != _api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000, description="Text to classify.")
    tau: Optional[float] = Field(None, ge=0.01, le=2.0, description="Softmax temperature. Defaults to coredrill tau.")
    full: bool = Field(False, description="If true, return full ranked table and active overlap set.")


class NodeResult(BaseModel):
    node: str
    level: int
    cosine_similarity: float
    mass: float  # relative probability mass, 100 = best
    production_score: float
    sigma_dist: float


class PredictionResponse(BaseModel):
    text: str
    best_node: str
    best_level: int
    confidence: float
    lineage: List[str]
    flow_type: str
    flow_label: str
    flow_intensities: dict
    active_overlap: List[NodeResult]
    ranked: Optional[List[NodeResult]] = None  # only if full=True
    pca_xyz: List[float]
    latency_ms: float


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=64)
    tau: Optional[float] = Field(None, ge=0.01, le=2.0)


class BatchItem(BaseModel):
    text: str
    best_node: str
    best_level: int
    confidence: float
    lineage: List[str]
    flow_type: str
    latency_ms: float


class BatchPredictResponse(BaseModel):
    results: List[BatchItem]
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    coredrill_nodes: int
    model: str
    coredrill_path: str


class InfoResponse(BaseModel):
    taxonomy_nodes: int
    taxonomy_levels: dict
    embedding_model: str
    coredrill_version: str
    skip_nodes: List[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_predict(text: str, tau: float | None) -> dict:
    """Run the full rhizome analysis pipeline. Returns raw analyze dict."""
    cd = _get_coredrill()
    model = _get_model()
    t0 = time.perf_counter()
    result = analyze_text_rhizome_v2(cd, text, model, tau=tau or cd.get("tau", 0.1), print_output=False)
    result["_latency_ms"] = (time.perf_counter() - t0) * 1000
    return result


def _build_response(raw: dict, full: bool = False) -> PredictionResponse:
    pred = raw["prediction"]
    active = raw.get("active_overlap_set", [])
    results_all = raw.get("results", [])
    cd = _get_coredrill()
    tree = cd.get("tree", {})
    gram_winner = raw.get("gram_winner")
    if gram_winner and gram_winner not in SKIP_NODES:
        best_node = gram_winner
    else:
        rec = raw.get("recommended_winner")
        if rec and isinstance(rec, dict) and rec.get("node"):
            best_node = rec["node"]
        elif results_all:
            best_node = results_all[0]["node"]
        else:
            best_node = pred.get("best_node", "unknown")
    best_level = int(tree.get(best_node, {}).get("level", pred.get("best_level", -1)))
    best_result = next((r for r in results_all if r["node"] == best_node), None)
    confidence = round(best_result["conf"], 4) if best_result else round(pred.get("top_level_confidence", 0.0), 4)
    backbone = cd.get("taxonomy_backbone", {})
    lineage_map = backbone.get("lineage_paths", {})
    paths = lineage_map.get(best_node, [])
    paths = [[n for n in p if n not in SKIP_NODES] for p in paths]
    paths = [p for p in paths if p]
    lineage = paths[0] if paths else [best_node]
    from rhizome_engine import _deleuzian_flow  # type: ignore
    flow = _deleuzian_flow(active, results_all, pred)
    def _node_result(r: dict) -> NodeResult:
        return NodeResult(
            node=r["node"],
            level=r["level"],
            cosine_similarity=round(r["score"], 4),
            mass=round(r.get("mass", 0.0), 2),
            production_score=round(r["production_score"], 4),
            sigma_dist=round(r["sigma_dist"], 3),
        )
    active_results = [_node_result(r) for r in active if r["node"] not in SKIP_NODES]
    ranked = None
    if full:
        ranked = [_node_result(r) for r in results_all[:40] if r["node"] not in SKIP_NODES]
    return PredictionResponse(
        text=raw["text"],
        best_node=best_node,
        best_level=best_level,
        confidence=confidence,
        lineage=lineage,
        flow_type=flow["flow_type"],
        flow_label=flow["flow_label"],
        flow_intensities=flow["intensities"],
        active_overlap=active_results,
        ranked=ranked,
        pca_xyz=pred.get("pca_xyz", [0.0, 0.0, 0.0]),
        latency_ms=round(raw["_latency_ms"], 1),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Coredrill API. See /docs for the interactive interface."}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    """Liveness / readiness check. Triggers model and coredrill loading."""
    cd = _get_coredrill()
    _get_model()
    return HealthResponse(
        status="ok",
        coredrill_nodes=len(cd.get("tree", {})),
        model=_model_name,
        coredrill_path=_coredrill_path,
    )


@app.get("/info", response_model=InfoResponse, tags=["meta"])
def info():
    """Taxonomy structure, model details, and skip-node list."""
    cd = _get_coredrill()
    tree = cd.get("tree", {})
    level_counts: dict[str, int] = {}
    for stats in tree.values():
        lvl = str(stats.get("level", "?"))
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
    return InfoResponse(
        taxonomy_nodes=len(tree),
        taxonomy_levels=level_counts,
        embedding_model=_model_name,
        coredrill_version=cd.get("version", "unknown"),
        skip_nodes=sorted(SKIP_NODES),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["classify"])
def predict_endpoint(
    req: PredictRequest,
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    """
    Classify a single text against the 238-node rhizome taxonomy.

    Returns the best-matching node, its lineage path, confidence, Deleuzian flow
    topology, and the active overlap set (co-activated semantic territories).

    Set `full=true` to receive the full ranked table (top 40 nodes).
    """
    _check_key(credentials)
    try:
        raw = _run_predict(req.text, req.tau)
        return _build_response(raw, full=req.full)
    except Exception as e:
        log.exception("predict failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["classify"])
def predict_batch(
    req: BatchPredictRequest,
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    """
    Classify up to 64 texts in a single request.
    Returns lightweight results (best node, lineage, flow type, confidence).
    """
    _check_key(credentials)
    t_total = time.perf_counter()
    results: list[BatchItem] = []
    for text in req.texts:
        try:
            raw = _run_predict(text, req.tau)
            resp = _build_response(raw, full=False)
            results.append(BatchItem(
                text=text[:120],
                best_node=resp.best_node,
                best_level=resp.best_level,
                confidence=resp.confidence,
                lineage=resp.lineage,
                flow_type=resp.flow_type,
                latency_ms=resp.latency_ms,
            ))
        except Exception as e:
            log.warning(f"batch item failed: {text[:60]!r} — {e}")
            results.append(BatchItem(
                text=text[:120],
                best_node="error",
                best_level=-1,
                confidence=0.0,
                lineage=[],
                flow_type="error",
                latency_ms=0.0,
            ))
    total_ms = (time.perf_counter() - t_total) * 1000
    return BatchPredictResponse(results=results, total_latency_ms=round(total_ms, 1))



@app.post("/predict/text", response_class=PlainTextResponse, tags=["classify"])
def predict_text(
    req: PredictRequest,
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    """
    Classify a text and return a human-readable plain-text report
    matching the terminal output style of the rhizome engine.
    Useful for curl or browser testing.
    """
    _check_key(credentials)
    try:
        raw = _run_predict(req.text, req.tau)
        resp = _build_response(raw, full=False)
        active = raw.get("active_overlap_set", [])
        results_all = raw.get("results", [])
        gram_scores = raw.get("gram_scores", {})
        gram_winner = raw.get("gram_winner", "n/a")

        W = 72
        lines = []
        lines.append("=" * W)
        lines.append("COREDRILL  —  RHIZOME PROFILE")
        lines.append("=" * W)
        text_short = req.text[:W-4] + ("..." if len(req.text) > W-4 else "")
        lines.append(f"  {text_short}")
        lines.append("")

        # ── GRAM-INVERSE first ───────────────────────────────────────────
        if gram_scores:
            GW = 88
            gram_state = raw.get("gram_state", "ok")
            state_tag = "  [VOID — L0 fallback]" if gram_state == "void" else ""
            lines.append(f"  GRAM-INVERSE  score = alpha * cos / sigma^0.5{state_tag}")
            lines.append(f"  winner: {gram_winner}")
            lines.append(f"  {'NODE':<28} {'score':>9} {'cos':>7} {'sigma':>6}  {'z':>5}  {'lap':>7}  {'gau':>7}  {'man':>7}  {'MASS':>6}")
            lines.append("  " + "-" * GW)
            result_nodes = {r["node"]: r for r in results_all if r["node"] not in SKIP_NODES}
            gram_ranked = sorted(gram_scores.items(), key=lambda x: -x[1])
            shown = 0
            for node, gs in gram_ranked:
                if shown >= 12 or node in SKIP_NODES:
                    continue
                r = result_nodes.get(node)
                if not r:
                    continue
                cos  = r["score"]
                sig  = r.get("sigma_eff", 0.0)
                z    = r["sigma_dist"]
                lap  = r.get("laplace_score", 0.0)
                gau  = r.get("gaussian_score", 0.0)
                man  = r.get("manifold_score", 0.0)
                mass = r.get("mass", 0.0)
                marker = " <<" if node == gram_winner else ""
                lines.append(
                    f"  {node:<28} {gs:+9.4f} {cos:7.4f} {sig:6.4f}  {z:5.2f}s  {lap:7.4f}  {gau:7.4f}  {man:7.4f}  {mass:5.1f}%{marker}"
                )
                shown += 1
        else:
            lines.append(f"  GRAM-INVERSE: not available")
        lines.append("")

        # ── Active territory profile ──────────────────────────────────────
        lines.append(f"  {'NODE':<28}  {'sigma':>6}  {'z':>5}  {'lap':>7}  {'gau':>7}  {'man':>7}  {'MASS':>6}")
        lines.append("  " + "-" * (W - 2))
        for r in active[:6]:
            if r["node"] in SKIP_NODES:
                continue
            z    = r["sigma_dist"]
            mass = r.get("mass", 0.0)
            sig  = r.get("sigma_eff", r.get("sigma_dist", 0.0))
            lap  = r.get("laplace_score", 0.0)
            gau  = r.get("gaussian_score", 0.0)
            man  = r.get("manifold_score", 0.0)
            filled = int((mass / 100.0) * 10)
            bar = "|" * filled + "." * (10 - filled)
            lines.append(f"  {r['node']:<28}  {sig:6.4f}  {z:5.2f}s  {lap:7.4f}  {gau:7.4f}  {man:7.4f}  [{bar}] {mass:5.1f}%")
        lines.append("")

        # ── Recommended and lineage ───────────────────────────────────────
        lines.append(f"  RECOMMENDED: {resp.best_node}  (level {resp.best_level})")
        lines.append("")
        if resp.lineage:
            lines.append("  CLADISTIC PATH:")
            lines.append(f"  {' -> '.join(resp.lineage)}")
            lines.append("")

        # ── Flow topology ─────────────────────────────────────────────────
        lines.append("  FLOW TOPOLOGY:")
        lines.append("")
        for r in active[:5]:
            if r["node"] in SKIP_NODES:
                continue
            z    = r["sigma_dist"]
            mass = r.get("mass", 0.0)
            stem = "===>" if mass >= 80 else ("==- " if mass >= 40 else ("--- " if mass >= 20 else "... "))
            pos  = "in  " if z < 1.0 else ("edge" if z < 1.5 else "out ")
            lines.append(f"    {stem} {r['node']:<28} [{pos}] z={z:.2f}s  {mass:5.1f}%")
        lines.append("")
        lines.append(f"  PRIMARY FLOW: {resp.flow_label}")
        lines.append("")
        for fname, fval in sorted(resp.flow_intensities.items(), key=lambda x: -x[1]):
            bar_len = int(fval * 28)
            bar = "#" * bar_len + "." * (28 - bar_len)
            marker = " <<" if fname == resp.flow_type else ""
            label = fname.replace("_", " ").upper()
            lines.append(f"    {label:<22} [{bar}] {fval:.2f}{marker}")
        lines.append("")

        # ── Full ranked table ─────────────────────────────────────────────
        lines.append("  FULL RANKED TABLE (top 20)")
        lines.append(f"  {'NODE':<28} {'sigma':>6} {'cos':>7} {'z':>6} {'prod':>9} {'mass':>7} {'lvl':>4}")
        lines.append("  " + "-" * (W - 2))
        for r in results_all[:20]:
            if r["node"] in SKIP_NODES:
                continue
            lines.append(
                f"  {r['node']:<28} {r.get('sigma_eff', 0):6.4f} {r['score']:7.4f} "
                f"{r['sigma_dist']:5.2f}s {r['production_score']:9.4f} "
                f"{r.get('mass', 0):6.1f}% {r['level']:4d}"
            )
        lines.append("")
        lines.append("=" * W)

        return "\n".join(lines)
    except Exception as e:
        import traceback
        return f"ERROR: {e}\n{traceback.format_exc()}"


@app.post("/predict/file", response_model=PredictionResponse, tags=["classify"])
async def predict_file(
    file: UploadFile,
    tau: Optional[float] = None,
    full: bool = False,
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    """
    Classify a plain-text file (.txt). Upload via multipart/form-data.
    Returns the same JSON response as /predict.

    Example:
        curl -X POST https://your-space.hf.space/predict/file \
             -H "Authorization: Bearer TOKEN" \
             -F "file=@mytext.txt"
    """
    _check_key(credentials)
    try:
        content = await file.read()
        text = content.decode("utf-8", errors="replace").strip()
        if not text:
            raise HTTPException(status_code=422, detail="File is empty.")
        if len(text) > 8000:
            text = text[:8000]
        raw = _run_predict(text, tau)
        return _build_response(raw, full=full)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("predict/file failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/file/text", response_class=PlainTextResponse, tags=["classify"])
async def predict_file_text(
    file: UploadFile,
    tau: Optional[float] = None,
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    """
    Classify a plain-text file (.txt) and return human-readable output.
    Upload via multipart/form-data.

    Example:
        curl -X POST https://your-space.hf.space/predict/file/text \
             -H "Authorization: Bearer TOKEN" \
             -F "file=@mytext.txt"
    """
    _check_key(credentials)
    try:
        content = await file.read()
        text = content.decode("utf-8", errors="replace").strip()
        if not text:
            return "ERROR: File is empty."
        if len(text) > 8000:
            text = text[:8000]
        # Reuse the predict/text logic by calling _run_predict directly
        from fastapi import Request
        raw = _run_predict(text, tau)
        resp = _build_response(raw, full=False)
        active = raw.get("active_overlap_set", [])
        results_all = raw.get("results", [])
        gram_scores = raw.get("gram_scores", {})
        gram_winner = raw.get("gram_winner", "n/a")

        W = 72
        lines = []
        lines.append("=" * W)
        lines.append(f"FILE: {file.filename}")
        lines.append("COREDRILL  —  RHIZOME PROFILE")
        lines.append("=" * W)
        text_short = text[:W-4] + ("..." if len(text) > W-4 else "")
        lines.append(f"  {text_short}")
        lines.append("")

        if gram_scores:
            GW = 88
            gram_state = raw.get("gram_state", "ok")
            state_tag = "  [VOID — L0 fallback]" if gram_state == "void" else ""
            lines.append(f"  GRAM-INVERSE  score = alpha * cos / sigma^0.5{state_tag}")
            lines.append(f"  winner: {gram_winner}")
            lines.append(f"  {'NODE':<28} {'score':>9} {'cos':>7} {'sigma':>6}  {'z':>5}  {'lap':>7}  {'gau':>7}  {'man':>7}  {'MASS':>6}")
            lines.append("  " + "-" * GW)
            result_nodes = {r["node"]: r for r in results_all if r["node"] not in SKIP_NODES}
            gram_ranked = sorted(gram_scores.items(), key=lambda x: -x[1])
            shown = 0
            for node, gs in gram_ranked:
                if shown >= 12 or node in SKIP_NODES:
                    continue
                r = result_nodes.get(node)
                if not r:
                    continue
                cos  = r["score"]
                sig  = r.get("sigma_eff", 0.0)
                z    = r["sigma_dist"]
                lap  = r.get("laplace_score", 0.0)
                gau  = r.get("gaussian_score", 0.0)
                man  = r.get("manifold_score", 0.0)
                mass = r.get("mass", 0.0)
                marker = " <<" if node == gram_winner else ""
                lines.append(
                    f"  {node:<28} {gs:+9.4f} {cos:7.4f} {sig:6.4f}  {z:5.2f}s  {lap:7.4f}  {gau:7.4f}  {man:7.4f}  {mass:5.1f}%{marker}"
                )
                shown += 1
        lines.append("")

        lines.append(f"  {'NODE':<28}  {'sigma':>6}  {'z':>5}  {'lap':>7}  {'gau':>7}  {'man':>7}  {'MASS':>6}")
        lines.append("  " + "-" * (W - 2))
        for r in active[:6]:
            if r["node"] in SKIP_NODES:
                continue
            z    = r["sigma_dist"]
            mass = r.get("mass", 0.0)
            sig  = r.get("sigma_eff", r.get("sigma_dist", 0.0))
            lap  = r.get("laplace_score", 0.0)
            gau  = r.get("gaussian_score", 0.0)
            man  = r.get("manifold_score", 0.0)
            filled = int((mass / 100.0) * 10)
            bar = "|" * filled + "." * (10 - filled)
            lines.append(f"  {r['node']:<28}  {sig:6.4f}  {z:5.2f}s  {lap:7.4f}  {gau:7.4f}  {man:7.4f}  [{bar}] {mass:5.1f}%")
        lines.append("")
        lines.append(f"  RECOMMENDED: {resp.best_node}  (level {resp.best_level})")
        lines.append("")
        if resp.lineage:
            lines.append(f"  CLADISTIC: {' -> '.join(resp.lineage)}")
            lines.append("")
        lines.append(f"  PRIMARY FLOW: {resp.flow_label}")
        for fname, fval in sorted(resp.flow_intensities.items(), key=lambda x: -x[1]):
            bar_len = int(fval * 28)
            bar = "#" * bar_len + "." * (28 - bar_len)
            marker = " <<" if fname == resp.flow_type else ""
            lines.append(f"    {fname.replace('_',' ').upper():<22} [{bar}] {fval:.2f}{marker}")
        lines.append("")
        lines.append("=" * W)
        return "\n".join(lines)
    except Exception as e:
        import traceback
        return f"ERROR: {e}\n{traceback.format_exc()}"

@app.get("/taxonomy/nodes", tags=["taxonomy"])
def taxonomy_nodes(level: Optional[int] = None):
    """
    List all taxonomy nodes, optionally filtered by level (0–5).
    """
    cd = _get_coredrill()
    tree = cd.get("tree", {})
    nodes = []
    for name, stats in tree.items():
        if name in SKIP_NODES:
            continue
        lvl = int(stats.get("level", -1))
        if level is not None and lvl != level:
            continue
        nodes.append({
            "name": name,
            "level": lvl,
            "has_centroid": stats.get("centroid") is not None or stats.get("rmc") is not None,
            "n": stats.get("n", 0),
            "sigma": round(float(stats.get("sigma") or 0), 4),
        })
    nodes.sort(key=lambda x: (x["level"], x["name"]))
    return {"count": len(nodes), "nodes": nodes}


@app.get("/taxonomy/node/{name}", tags=["taxonomy"])
def taxonomy_node_detail(name: str):
    """
    Detail for a single taxonomy node: sigma, n, source_counts, related nodes.
    """
    cd = _get_coredrill()
    tree = cd.get("tree", {})
    if name not in tree:
        raise HTTPException(status_code=404, detail=f"Node '{name}' not found.")
    stats = dict(tree[name])
    # Strip large arrays from the response
    stats.pop("centroid", None)
    stats.pop("rmc", None)
    stats.pop("W2", None)
    return {"name": name, **stats}
