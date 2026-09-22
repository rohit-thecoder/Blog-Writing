from __future__ import annotations

import json
import os
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Import your compiled LangGraph app
# -----------------------------
from bwa_backend import app


# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Stream graph progress if available; else invoke.
    Yields ("updates"/"values"/"final", payload).
    """
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


# -----------------------------
# Markdown renderer that supports local images
# -----------------------------
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    return Path(src).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))

        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)

        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or (alt or None), use_container_width=True)
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")

        i += 1


# -----------------------------
# ✅ NEW: Past blogs helpers
# -----------------------------
def list_past_blogs() -> List[Path]:
    """
    Returns .md files in current working directory, newest first.
    Filters out obvious non-blog markdown files if needed.
    """
    cwd = Path(".")
    files = [p for p in cwd.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    """
    Use first '# ' heading as title if present.
    """
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback



# -----------------------------
# Premium Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Blog Writing | LangGraph Blog Writer",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Premium styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #080b12;
        --panel: #0f1420;
        --panel-2: #131a28;
        --border: rgba(255,255,255,.09);
        --text: #f5f7fb;
        --muted: #8d98aa;
        --accent: #8b5cf6;
        --accent-2: #22d3ee;
        --success: #34d399;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 0%, rgba(139,92,246,.12), transparent 28%),
            radial-gradient(circle at 10% 10%, rgba(34,211,238,.07), transparent 24%),
            var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e17 0%, #0d121d 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    .block-container {
        max-width: 1480px;
        padding: 2rem 3rem 4rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -.025em;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.1rem 2.3rem;
        border: 1px solid var(--border);
        border-radius: 24px;
        background:
            linear-gradient(135deg, rgba(139,92,246,.16), rgba(34,211,238,.05) 50%, rgba(255,255,255,.025)),
            rgba(15,20,32,.82);
        box-shadow: 0 24px 80px rgba(0,0,0,.28);
        margin-bottom: 1.4rem;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -100px;
        top: -130px;
        border-radius: 50%;
        background: rgba(139,92,246,.16);
        filter: blur(8px);
    }

    .eyebrow {
        color: #a78bfa;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: .65rem;
    }

    .hero h1 {
        margin: 0;
        font-size: clamp(2rem, 4vw, 3.5rem);
        line-height: 1.02;
        color: #fff;
    }

    .hero p {
        max-width: 760px;
        color: #aeb8c9;
        font-size: 1rem;
        line-height: 1.7;
        margin: .9rem 0 0;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: .8rem;
        margin: .2rem 0 1.5rem;
    }

    .brand-mark {
        width: 42px;
        height: 42px;
        border-radius: 13px;
        display: grid;
        place-items: center;
        color: white;
        font-weight: 800;
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(135deg, #8b5cf6, #06b6d4);
        box-shadow: 0 10px 30px rgba(139,92,246,.28);
    }

    .brand-name {
        color: #fff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
    }

    .brand-sub {
        color: var(--muted);
        font-size: .74rem;
        margin-top: 2px;
    }

    .section-label {
        color: #cbd5e1;
        font-family: 'Space Grotesk', sans-serif;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .1em;
        text-transform: uppercase;
        margin: 1rem 0 .65rem;
    }

    .metric-card {
        min-height: 112px;
        padding: 1rem 1.15rem;
        border-radius: 18px;
        border: 1px solid var(--border);
        background: linear-gradient(145deg, rgba(19,26,40,.95), rgba(12,16,26,.95));
        box-shadow: 0 12px 35px rgba(0,0,0,.15);
    }

    .metric-label {
        color: var(--muted);
        font-size: .78rem;
        margin-bottom: .35rem;
    }

    .metric-value {
        color: #fff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.55rem;
        font-weight: 700;
    }

    .metric-accent {
        color: #a78bfa;
    }

    .empty-state {
        text-align: center;
        padding: 4rem 1rem;
        border: 1px dashed rgba(255,255,255,.12);
        border-radius: 22px;
        background: rgba(255,255,255,.018);
    }

    .empty-icon {
        font-size: 2rem;
        margin-bottom: .6rem;
    }

    .empty-title {
        color: #fff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
    }

    .empty-copy {
        color: var(--muted);
        margin-top: .35rem;
    }

    .result-title {
        color: #fff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.55rem;
        font-weight: 700;
        margin-bottom: .2rem;
    }

    .result-meta {
        color: var(--muted);
        font-size: .86rem;
        margin-bottom: 1.2rem;
    }

    .stTextArea textarea, .stTextInput input {
        background: rgba(255,255,255,.035) !important;
        border: 1px solid rgba(255,255,255,.10) !important;
        border-radius: 13px !important;
        color: #fff !important;
    }

    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: rgba(139,92,246,.7) !important;
        box-shadow: 0 0 0 1px rgba(139,92,246,.2) !important;
    }

    .stButton > button {
        border-radius: 12px;
        min-height: 42px;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,.10);
        background: rgba(255,255,255,.045);
        color: #eef2ff;
        transition: all .18s ease;
    }

    .stButton > button:hover {
        border-color: rgba(139,92,246,.55);
        background: rgba(139,92,246,.12);
        transform: translateY(-1px);
    }

    [data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #8b5cf6, #6d4aff) !important;
        border: none !important;
        box-shadow: 0 10px 28px rgba(124,58,237,.28);
    }

    [data-testid="stBaseButton-primary"]:hover {
        box-shadow: 0 14px 35px rgba(124,58,237,.38);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: .35rem;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        padding: .75rem 1rem;
        color: #8f9bad;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        color: #fff !important;
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, #8b5cf6, #22d3ee);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow: hidden;
    }

    .sidebar-note {
        padding: .85rem 1rem;
        border: 1px solid var(--border);
        border-radius: 13px;
        background: rgba(255,255,255,.025);
        color: #8995a8;
        font-size: .76rem;
        line-height: 1.55;
        margin-top: 1rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: .35rem;
        padding: .35rem .7rem;
        border-radius: 999px;
        background: rgba(52,211,153,.08);
        border: 1px solid rgba(52,211,153,.18);
        color: #6ee7b7;
        font-size: .75rem;
        font-weight: 700;
    }

    .source-card {
        padding: 1rem 1.1rem;
        margin-bottom: .7rem;
        border: 1px solid var(--border);
        border-radius: 16px;
        background: rgba(255,255,255,.025);
    }

    .source-title {
        color: #f8fafc;
        font-weight: 700;
        margin-bottom: .3rem;
    }

    .source-meta {
        color: #7f8a9d;
        font-size: .75rem;
    }

    .source-url {
        color: #a78bfa;
        font-size: .78rem;
        overflow-wrap: anywhere;
    }

    /* Hide Streamlit's default menu/footer for a cleaner app shell */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helpers for UI ----------
def _as_dict(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return {}


def metric_card(label: str, value: Any, accent: bool = False):
    cls = "metric-value metric-accent" if accent else "metric-value"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="{cls}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_empty(icon: str, title: str, copy: str):
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-icon">{icon}</div>
            <div class="empty-title">{title}</div>
            <div class="empty-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">✦</div>
            <div>
                <div class="brand-name">Blog Writing</div>
                <div class="brand-sub">LangGraph Blog Writer</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Create</div>', unsafe_allow_html=True)

    topic = st.text_area(
        "What should I write about?",
        height=125,
        placeholder="e.g. How Self-RAG improves reliability in production RAG systems...",
        key="topic_input",
        label_visibility="visible",
    )

    as_of = st.date_input("Research date", value=date.today())

    run_btn = st.button(
        "✦  Generate Blog",
        type="primary",
        use_container_width=True,
    )

    st.markdown('<div class="section-label">Library</div>', unsafe_allow_html=True)

    past_files = list_past_blogs()
    selected_md_file = None

    if not past_files:
        st.caption("No saved Markdown blogs in this workspace.")
    else:
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}

        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem

            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.selectbox(
            "Saved blogs",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("Open selected blog", use_container_width=True):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                st.session_state["last_out"] = {
                    "plan": None,
                    "evidence": [],
                    "image_specs": [],
                    "final": md_text,
                }
                st.rerun()

    st.markdown(
        """
        <div class="sidebar-note">
            <strong>Workflow</strong><br>
            Describe a topic → LangGraph plans and researches it → review evidence → preview the final Markdown → export.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Page header ----------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI CONTENT WORKSPACE</div>
        <h1>Research. Plan. Write.</h1>
        <p>
            A premium workspace for generating research-backed blogs with your LangGraph agent.
            Inspect the plan, evidence, images, execution logs, and final Markdown from one place.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- State ----------
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

if "logs" not in st.session_state:
    st.session_state["logs"] = []

logs: List[str] = []


def log(msg: str):
    logs.append(msg)


# ---------- Generation ----------
if run_btn:
    if not topic.strip():
        st.warning("Enter a topic before generating the blog.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    status = st.status("Building your blog...", expanded=True)
    progress_area = st.empty()

    current_state: Dict[str, Any] = {}
    last_node = None

    try:
        for kind, payload in try_stream(app, inputs):
            if kind in ("updates", "values"):
                node_name = None

                if (
                    isinstance(payload, dict)
                    and len(payload) == 1
                    and isinstance(next(iter(payload.values())), dict)
                ):
                    node_name = next(iter(payload.keys()))

                if node_name and node_name != last_node:
                    status.write(f"Running `{node_name}`")
                    last_node = node_name

                current_state = extract_latest_state(current_state, payload)

                summary = {
                    "mode": current_state.get("mode"),
                    "needs_research": current_state.get("needs_research"),
                    "queries": (
                        current_state.get("queries", [])[:5]
                        if isinstance(current_state.get("queries"), list)
                        else []
                    ),
                    "evidence_count": len(current_state.get("evidence", []) or []),
                    "tasks": (
                        len((current_state.get("plan") or {}).get("tasks", []))
                        if isinstance(current_state.get("plan"), dict)
                        else None
                    ),
                    "images": len(current_state.get("image_specs", []) or []),
                    "sections_done": len(current_state.get("sections", []) or []),
                }

                progress_area.json(summary)
                log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

            elif kind == "final":
                st.session_state["last_out"] = payload
                status.update(
                    label="Generation complete",
                    state="complete",
                    expanded=False,
                )
                log("[final] received final state")

    except Exception as exc:
        status.update(label="Generation failed", state="error")
        st.error(f"Something went wrong while running the graph: {exc}")

    if logs:
        st.session_state["logs"].extend(logs)


# ---------- Results ----------
out = st.session_state.get("last_out")

if not out:
    show_empty(
        "✦",
        "Your workspace is ready",
        "Enter a topic in the sidebar and generate your first research-backed blog.",
    )
    st.stop()

plan_dict = _as_dict(out.get("plan"))
evidence = out.get("evidence") or []
final_md = out.get("final") or ""
image_specs = out.get("image_specs") or []

# ---------- Result summary ----------
blog_title = (
    plan_dict.get("blog_title")
    or extract_title_from_md(final_md, "Untitled Blog")
    if final_md
    else plan_dict.get("blog_title", "Untitled Blog")
)

st.markdown(
    f"""
    <div class="result-title">{blog_title}</div>
    <div class="result-meta">
        <span class="status-pill">● Ready</span>
        &nbsp; Generated with LangGraph
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    metric_card("Research sources", len(evidence), True)
with m2:
    metric_card("Planned tasks", len(plan_dict.get("tasks", []) or []))
with m3:
    metric_card("Image assets", len(image_specs))
with m4:
    metric_card("Sections", len(out.get("sections") or []))

st.markdown("<br>", unsafe_allow_html=True)

tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["✦ Plan", "⌕ Evidence", "◉ Preview", "▧ Images", "≡ Logs"]
)

# ---------- Plan ----------
with tab_plan:
    if not plan_dict:
        show_empty("◇", "No plan available", "This run did not return a structured plan.")
    else:
        top1, top2, top3 = st.columns(3)
        with top1:
            st.markdown("**Audience**")
            st.caption(str(plan_dict.get("audience") or "Not specified"))
        with top2:
            st.markdown("**Tone**")
            st.caption(str(plan_dict.get("tone") or "Not specified"))
        with top3:
            st.markdown("**Blog type**")
            st.caption(str(plan_dict.get("blog_kind") or "Not specified"))

        tasks = plan_dict.get("tasks", []) or []

        if tasks:
            st.markdown("### Content architecture")

            rows = []
            for t in tasks:
                t = _as_dict(t)
                rows.append(
                    {
                        "#": t.get("id"),
                        "Section": t.get("title"),
                        "Words": t.get("target_words"),
                        "Research": "Yes" if t.get("requires_research") else "No",
                        "Citations": "Yes" if t.get("requires_citations") else "No",
                        "Code": "Yes" if t.get("requires_code") else "No",
                        "Tags": ", ".join(t.get("tags") or []),
                    }
                )

            st.dataframe(
                pd.DataFrame(rows).sort_values("#"),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("View raw task details"):
                st.json(tasks)

# ---------- Evidence ----------
with tab_evidence:
    if not evidence:
        show_empty(
            "⌕",
            "No research evidence",
            "No sources were returned for this run. This can happen in closed-book mode or when Tavily returns no results.",
        )
    else:
        st.caption(f"{len(evidence)} sources collected")

        for e in evidence:
            e = _as_dict(e)
            title = e.get("title") or "Untitled source"
            url = e.get("url") or ""
            published = e.get("published_at") or "Publication date unavailable"
            source = e.get("source") or "Unknown source"

            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">{title}</div>
                    <div class="source-meta">{source} · {published}</div>
                    <div class="source-url">{url}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------- Preview ----------
with tab_preview:
    if not final_md:
        show_empty("◉", "No Markdown yet", "The graph did not return a final Markdown document.")
    else:
        c1, c2 = st.columns([1, 1])

        with c1:
            st.caption("Rendered article")

        with c2:
            st.download_button(
                "Download Markdown",
                data=final_md.encode("utf-8"),
                file_name=f"{safe_slug(blog_title)}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        render_markdown_with_local_images(final_md)

        bundle = bundle_zip(
            final_md,
            f"{safe_slug(blog_title)}.md",
            Path("images"),
        )

        st.download_button(
            "Download bundle · Markdown + images",
            data=bundle,
            file_name=f"{safe_slug(blog_title)}_bundle.zip",
            mime="application/zip",
            use_container_width=True,
        )

# ---------- Images ----------
with tab_images:
    images_dir = Path("images")

    if not image_specs and not images_dir.exists():
        show_empty(
            "▧",
            "No image assets",
            "No image specifications or generated image files were returned.",
        )
    else:
        if image_specs:
            st.markdown("### Image plan")
            st.json(image_specs)

        if images_dir.exists():
            files = [p for p in images_dir.iterdir() if p.is_file()]

            if not files:
                st.info("The images directory exists but contains no files.")
            else:
                cols = st.columns(3)

                for idx, p in enumerate(sorted(files)):
                    with cols[idx % 3]:
                        st.image(str(p), caption=p.name, use_container_width=True)

                z = images_zip(images_dir)
                if z:
                    st.download_button(
                        "Download all images",
                        data=z,
                        file_name="images.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )

# ---------- Logs ----------
with tab_logs:
    st.caption("Latest execution events")

    stored_logs = st.session_state.get("logs", [])
    if not stored_logs:
        show_empty("≡", "No logs yet", "Execution events will appear here after a generation run.")
    else:
        st.text_area(
            "Event log",
            value="\n\n".join(stored_logs[-100:]),
            height=520,
            label_visibility="collapsed",
        )
