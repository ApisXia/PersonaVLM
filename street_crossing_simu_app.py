import os
import sys
import json

import gradio as gr
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from street_crossing_decision_command import StreetCrossingDecisionSystem

# ── Diagram layout constants ───────────────────────────────────────────────────
DX = 2.5
DY = 2.2
NODE_W = 1.8
NODE_H = 1.3
# Road zone: positions 0-4 are on sidewalk; road starts above pos4
_ROAD_LOWER = 4.7  # DY factor — lower curb (pos4=4.0 + gap 0.7)
_ROAD_UPPER = 5.9  # DY factor — upper curb (road width = 1.2)
_CROSS_Y = 6.6  # DY factor — crossing node (road_upper + gap 0.7, symmetric)
_Y_MAX = 7.2  # DY factor — top of diagram
_LINE_GAP = 0.12  # y-units gap on each side of centre for double solid lines

# ── Decision color palette (used in diagram + UI labels) ───────────────────────
DECISION_COLOR = {
    "forward": {
        "arrow": "#27ae60",
        "arrow_faint": "rgba(39,174,96,0.25)",
        "label_bg": "#d5f5e3",
    },
    "backward": {
        "arrow": "#e74c3c",
        "arrow_faint": "rgba(231,76,60,0.25)",
        "label_bg": "#fdecea",
    },
    "stop": {
        "arrow": "#e67e22",
        "arrow_faint": "rgba(230,126,34,0.25)",
        "label_bg": "#fef5e7",
    },
}
_DC_DEFAULT = {
    "arrow": "#888888",
    "arrow_faint": "rgba(136,136,136,0.25)",
    "label_bg": "#e8e8e8",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_persona_files():
    return (
        sorted(f for f in os.listdir("personas") if f.endswith(".json"))
        if os.path.isdir("personas")
        else []
    )


def get_scenarios():
    base = "data/250722_real_sim"
    return (
        sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        if os.path.exists(base)
        else []
    )


def load_personas(pfile: str) -> dict:
    with open(f"personas/{pfile}", "r", encoding="utf-8") as f:
        return json.load(f)


def step_label(step: dict) -> str:
    icon = (
        "🟢"
        if step["decision"] == "forward"
        else "🔴" if step["decision"] == "backward" else "🟡"
    )
    return (
        f"{icon} Step {step['time']}: {step['decision']}"
        f"  ({step['old_position']} → {step['new_position']})"
    )


# ── Decision flow diagram (Plotly) ─────────────────────────────────────────────
def render_diagram(history: list, video_folder: str, selected_idx=None) -> go.Figure:
    if not history:
        _ym = _Y_MAX * DY
        _rl = _ROAD_LOWER * DY
        _ru = _ROAD_UPPER * DY
        _x0 = -NODE_W * 0.7
        _x1 = 5.0
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(visible=False, range=[_x0, _x1]),
            yaxis=dict(visible=False, range=[-NODE_H * 1.5, _ym]),
            plot_bgcolor="#fafafa",
            paper_bgcolor="#fafafa",
            height=520,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        # Sidewalk (bottom)
        fig.add_shape(
            type="rect",
            x0=_x0,
            y0=-NODE_H * 1.5,
            x1=_x1,
            y1=_rl,
            fillcolor="#f5f0e8",
            line_width=0,
            layer="below",
        )
        # Road (middle)
        fig.add_shape(
            type="rect",
            x0=_x0,
            y0=_rl,
            x1=_x1,
            y1=_ru,
            fillcolor="#b0b0b0",
            line_width=0,
            layer="below",
        )
        # Far Roadside (top)
        fig.add_shape(
            type="rect",
            x0=_x0,
            y0=_ru,
            x1=_x1,
            y1=_ym,
            fillcolor="#f5f0e8",
            line_width=0,
            layer="below",
        )
        # Double solid lines — lower curb
        for dy in (-_LINE_GAP, +_LINE_GAP):
            fig.add_shape(
                type="line",
                x0=_x0,
                y0=_rl + dy,
                x1=_x1,
                y1=_rl + dy,
                line=dict(color="white", width=2.5),
            )
        # Double solid lines — upper curb
        for dy in (-_LINE_GAP, +_LINE_GAP):
            fig.add_shape(
                type="line",
                x0=_x0,
                y0=_ru + dy,
                x1=_x1,
                y1=_ru + dy,
                line=dict(color="white", width=2.5),
            )
        # Dashed center line (yellow)
        _center = (_rl + _ru) / 2
        fig.add_shape(
            type="line",
            x0=_x0,
            y0=_center,
            x1=_x1,
            y1=_center,
            line=dict(color="#f0c040", width=1.8, dash="dash"),
        )
        _lx = _x0 + 0.18
        fig.add_annotation(
            x=_lx,
            y=_rl / 2,
            text="Sidewalk",
            showarrow=False,
            font=dict(size=9, color="#9a8878"),
            textangle=-90,
            xref="x",
            yref="y",
            xanchor="center",
            yanchor="middle",
        )
        fig.add_annotation(
            x=_lx,
            y=(_rl + _ru) / 2,
            text="Road",
            showarrow=False,
            font=dict(size=9, color="#555"),
            textangle=-90,
            xref="x",
            yref="y",
            xanchor="center",
            yanchor="middle",
        )
        fig.add_annotation(
            x=_lx,
            y=(_ru + _ym) / 2,
            text="Roadside",
            showarrow=False,
            font=dict(size=9, color="#9a8878"),
            textangle=-90,
            xref="x",
            yref="y",
            xanchor="center",
            yanchor="middle",
        )
        fig.add_annotation(
            x=(_x0 + _x1) / 2,
            y=_rl / 2,
            xref="x",
            yref="y",
            text="Run the simulation to see the decision flow.",
            showarrow=False,
            font=dict(size=13, color="#aaa"),
        )
        return fig

    max_t = max(s["time"] for s in history) + 2
    x_max = max_t * DX + NODE_W
    y_max = _Y_MAX * DY
    road_l = _ROAD_LOWER * DY
    road_u = _ROAD_UPPER * DY

    fig = go.Figure()
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False, range=[-NODE_W * 0.7, x_max]),
        yaxis=dict(visible=False, range=[-NODE_H * 1.5, y_max]),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#fafafa",
        margin=dict(l=10, r=10, t=10, b=10),
        height=520,
    )

    # ── Background zones ───────────────────────────────────────────────────────
    x0_bg = -NODE_W * 0.7
    # Sidewalk (bottom)
    fig.add_shape(
        type="rect",
        x0=x0_bg,
        y0=-NODE_H * 1.5,
        x1=x_max,
        y1=road_l,
        fillcolor="#f5f0e8",
        line_width=0,
        layer="below",
    )
    # Road (middle)
    fig.add_shape(
        type="rect",
        x0=x0_bg,
        y0=road_l,
        x1=x_max,
        y1=road_u,
        fillcolor="#b0b0b0",
        line_width=0,
        layer="below",
    )
    # Far Roadside (top)
    fig.add_shape(
        type="rect",
        x0=x0_bg,
        y0=road_u,
        x1=x_max,
        y1=y_max,
        fillcolor="#f5f0e8",
        line_width=0,
        layer="below",
    )
    # Double solid lines — lower curb
    for dy in (-_LINE_GAP, +_LINE_GAP):
        fig.add_shape(
            type="line",
            x0=x0_bg,
            y0=road_l + dy,
            x1=x_max,
            y1=road_l + dy,
            line=dict(color="white", width=2.5),
        )
    # Double solid lines — upper curb
    for dy in (-_LINE_GAP, +_LINE_GAP):
        fig.add_shape(
            type="line",
            x0=x0_bg,
            y0=road_u + dy,
            x1=x_max,
            y1=road_u + dy,
            line=dict(color="white", width=2.5),
        )
    # Dashed center line (yellow)
    center_y = (road_l + road_u) / 2
    fig.add_shape(
        type="line",
        x0=x0_bg,
        y0=center_y,
        x1=x_max,
        y1=center_y,
        line=dict(color="#f0c040", width=1.8, dash="dash"),
    )
    # Zone labels (rotated, left margin)
    lx = x0_bg + 0.18
    fig.add_annotation(
        x=lx,
        y=road_l / 2,
        text="Sidewalk",
        showarrow=False,
        font=dict(size=9, color="#9a8878"),
        textangle=-90,
        xref="x",
        yref="y",
        xanchor="center",
        yanchor="middle",
    )
    fig.add_annotation(
        x=lx,
        y=(road_l + road_u) / 2,
        text="Road",
        showarrow=False,
        font=dict(size=9, color="#555"),
        textangle=-90,
        xref="x",
        yref="y",
        xanchor="center",
        yanchor="middle",
    )
    fig.add_annotation(
        x=lx,
        y=(road_u + y_max) / 2,
        text="Roadside",
        showarrow=False,
        font=dict(size=9, color="#9a8878"),
        textangle=-90,
        xref="x",
        yref="y",
        xanchor="center",
        yanchor="middle",
    )

    def add_arrow(x0, y0, x1, y1, color, width):
        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.0,
            arrowwidth=width,
            arrowcolor=color,
            text="",
        )

    def add_label(x0, y0, x1, y1, text, faint=False):
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2 + (0.28 if abs(y1 - y0) > 0.3 else 0.22)
        bg = DECISION_COLOR.get(text.lower(), _DC_DEFAULT)["label_bg"]
        fig.add_annotation(
            x=mx,
            y=my,
            text=text,
            showarrow=False,
            font=dict(size=9, color="#333"),
            bgcolor=bg,
            borderpad=3,
            opacity=0.2 if faint else 0.95,
            xref="x",
            yref="y",
            xanchor="center",
            yanchor="middle",
        )

    # Draw arrows between nodes
    for i, step in enumerate(history):
        t, p = step["time"], step["old_position"]
        dec, new_p = step["decision"], step["new_position"]
        crossing = step.get("is_crossing", False)
        cx, cy = t * DX, p * DY
        ncx = (t + 1) * DX
        tgt_y = (_CROSS_Y if crossing else float(new_p)) * DY
        arrow_clr = DECISION_COLOR.get(dec, _DC_DEFAULT)["arrow"]
        add_arrow(cx, cy, ncx, tgt_y, arrow_clr, 2.5)
        add_label(cx, cy, ncx, tgt_y, dec)
        # Always draw all three options as faint alternatives (skip chosen one)
        all_alts = [
            ("forward", (_CROSS_Y if p >= 4 else float(p + 1)) * DY),
            ("stop", float(p) * DY),
            ("backward", float(p - 1) * DY),
        ]
        for alt_dec, alt_y in all_alts:
            if alt_dec == dec:
                continue
            if alt_dec == "backward" and p <= 0:
                continue  # can't go below position 0
            alt_clr = DECISION_COLOR.get(alt_dec, _DC_DEFAULT)["arrow_faint"]
            add_arrow(cx, cy, ncx, alt_y, alt_clr, 1.0)
            add_label(cx, cy, ncx, alt_y, alt_dec, faint=True)

    # Build node data for scatter (all step nodes, then last result node)
    node_xs, node_ys, node_texts, hover_texts = [], [], [], []
    fill_colors, border_colors = [], []

    for i, step in enumerate(history):
        t, p = step["time"], step["old_position"]
        is_sel = i == selected_idx
        node_xs.append(t * DX)
        node_ys.append(p * DY)
        node_texts.append(f"T{t}<br>P{p}")
        fill_colors.append("#d5f5e3" if is_sel else "white")
        border_colors.append("#1a8a4a" if is_sel else "#888888")
        hover_texts.append(
            f"Step {t} | Pos {p}→{step['new_position']}<br>" f"{step['decision']}"
        )

    last = history[-1]
    ft, fp = last["time"] + 1, last["new_position"]
    is_cross_last = last.get("is_crossing", False)
    node_xs.append(ft * DX)
    node_ys.append(_CROSS_Y * DY if is_cross_last else float(fp) * DY)
    node_texts.append(f"T{ft}<br>{'cross' if is_cross_last else f'P{fp}'}")
    fill_colors.append("#aef6e5" if is_cross_last else "white")
    border_colors.append("#27ae60" if is_cross_last else "#888888")
    hover_texts.append("Cross the street" if is_cross_last else f"Time {ft} | Pos {fp}")

    fig.add_trace(
        go.Scatter(
            x=node_xs,
            y=node_ys,
            mode="markers+text",
            marker=dict(
                size=54,
                color=fill_colors,
                line=dict(width=3, color=border_colors),
                symbol="circle",
            ),
            text=node_texts,
            textposition="middle center",
            textfont=dict(size=8, color="#111"),
            hovertext=hover_texts,
            hoverinfo="text",
            name="",
        )
    )

    return fig


def render_sparkline(history: list) -> go.Figure:
    fig = go.Figure()
    if history:
        times = [h["time"] for h in history]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[h.get("confidence", 0) for h in history],
                mode="lines+markers",
                name="Confidence",
                line=dict(color="#2980b9", width=2),
                marker=dict(size=8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[h.get("trust", 0) for h in history],
                mode="lines+markers",
                name="Trust",
                line=dict(color="#e74c3c", width=2),
                marker=dict(size=8, symbol="square"),
            )
        )
        fig.update_layout(
            yaxis=dict(range=[0.5, 5.5], tickvals=[1, 2, 3, 4, 5], title="Score / 5"),
            xaxis=dict(title="Time step", tickvals=times),
            legend=dict(
                orientation="h",
                x=0.5,
                y=1.0,
                xanchor="center",
                yanchor="bottom",
            ),
        )
    fig.update_layout(
        height=260,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(
            text="Confidence & Trust Evolution", font=dict(color="rgba(0,0,0,0)")
        ),
    )
    return fig


# ── JS bridge: Plotly click → hidden textbox → Python callback ─────────────────
_CLICK_JS = """
() => {
    function attach() {
        var container = document.querySelector('#diagram-plot');
        if (!container) { setTimeout(attach, 400); return; }
        var plot = container.querySelector('.js-plotly-plot');
        if (!plot) { setTimeout(attach, 400); return; }
        if (plot.__grClickAttached) return;
        plot.__grClickAttached = true;
        plot.on('plotly_click', function(d) {
            if (!d || !d.points || !d.points.length) return;
            var idx = d.points[0].pointIndex;
            var inp = document.querySelector('#clicked-node-idx textarea');
            if (!inp) return;
            inp.value = String(idx);
            inp.dispatchEvent(new Event('input', {bubbles: true}));
        });
        new MutationObserver(function() {
            var p2 = container.querySelector('.js-plotly-plot');
            if (p2 && p2 !== plot) { plot.__grClickAttached = false; attach(); }
        }).observe(container, {childList: true, subtree: true});
    }
    attach();
}
"""

# ── Gradio app ─────────────────────────────────────────────────────────────────
persona_files = get_persona_files()
scenarios = get_scenarios()
default_pfile = (
    "persona_improvetransfer_v04.json"
    if "persona_improvetransfer_v04.json" in persona_files
    else (persona_files[0] if persona_files else "")
)

_CSS = """
#step-radio { overflow-y: auto; max-height: 160px; }
"""

with gr.Blocks(title="PersonaVLM", theme=gr.themes.Soft(), css=_CSS) as demo:

    history_state = gr.State([])
    vfolder_state = gr.State("")

    gr.Markdown("# 🚦 PersonaVLM — Street Crossing Decision Simulator")

    with gr.Row():

        # ── Left: configuration ────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=260):
            pfile_dd = gr.Dropdown(
                choices=persona_files, value=default_pfile, label="Persona file"
            )
            persona_dd = gr.Dropdown(choices=[], label="Persona")
            persona_desc = gr.Textbox(
                label="Persona description", lines=5, interactive=False
            )
            gr.Markdown("---")
            scenario_dd = gr.Dropdown(
                choices=scenarios,
                value=scenarios[0] if scenarios else None,
                label="Video scenario",
            )
            ehmi_dd = gr.Dropdown(
                choices=["eye", "lightbar", "no"], value="eye", label="eHMI type"
            )
            max_steps_sl = gr.Number(
                value=12, label="Max time steps", precision=0, minimum=1
            )
            temp_sl = gr.Slider(0.0, 2.0, value=1.2, step=0.1, label="Temperature")
            gr.Markdown("---")
            run_btn = gr.Button("▶ Run Simulation", variant="primary", size="lg")
            status_md = gr.Markdown("")

        # ── Right: main ────────────────────────────────────────────────────────
        with gr.Column(scale=3):
            gr.Markdown("### Simulation")
            clicked_node_idx = gr.Textbox(
                visible=False, value="", elem_id="clicked-node-idx"
            )
            with gr.Row():
                diagram_plot = gr.Plot(label="Decision Flow", elem_id="diagram-plot")
                with gr.Column():
                    step_radio = gr.Radio(
                        choices=[],
                        value=None,
                        label="Step History — click a step to replay",
                        interactive=True,
                        elem_id="step-radio",
                    )
                    video_out = gr.Video(label="Agent's View", height=340)

            decision_md = gr.Markdown("")
            with gr.Row():
                conf_txt = gr.Textbox(label="Confidence", interactive=False)
                trust_txt = gr.Textbox(label="Trust", interactive=False)
            reason_txt = gr.Textbox(label="Reasoning", lines=6, interactive=False)

            sparkline_plot = gr.Plot(label="Confidence & Trust")

            gr.Markdown("---")
            gr.Markdown("### Overall Result")
            with gr.Row():
                overall_video = gr.Video(label="Overall Video")
                assessment_txt = gr.Textbox(
                    label="Assessment", lines=10, interactive=False
                )

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def on_pfile_change(pfile):
        personas = load_personas(pfile)
        keys = list(personas.keys())
        first = keys[0] if keys else None
        desc = personas[first].get("description", "") if first else ""
        return gr.update(choices=keys, value=first), desc

    def on_persona_change(pfile, persona):
        if not persona:
            return ""
        return load_personas(pfile).get(persona, {}).get("description", "")

    def on_scenario_change(scenario):
        return (
            "eye"
            if "eye" in scenario
            else "lightbar" if "lightbar" in scenario else "no"
        )

    def on_step_select(choice, history, vfolder):
        empty = (None, "", "", "", "", render_diagram(history, vfolder or ""))
        if not choice or not history:
            return empty
        idx = next((i for i, h in enumerate(history) if step_label(h) == choice), None)
        if idx is None:
            return empty
        step = history[idx]
        icon = (
            "🟢"
            if step["decision"] == "forward"
            else "🔴" if step["decision"] == "backward" else "🟡"
        )
        dec_md = (
            f"### {icon} {step['decision'].upper()}\n"
            f"Position **{step['old_position']}** → **{step['new_position']}**"
        )
        conf = f"{step.get('confidence','?')}/5 — {step.get('confidence_reason','')}"
        trust = f"{step.get('trust','?')}/5 — {step.get('trust_reason','')}"
        vpath = step["video"]
        return (
            vpath if os.path.exists(vpath) else None,
            dec_md,
            conf,
            trust,
            step["reason"],
            render_diagram(history, vfolder or "", idx),
        )

    def on_diagram_click(idx_str, history, vfolder):
        empty = (None, "", "", "", "", render_diagram(history, vfolder or ""))
        if not idx_str or not history:
            return empty
        try:
            idx = int(idx_str)
        except (ValueError, TypeError):
            return empty
        if idx < 0 or idx >= len(history):
            return empty
        step = history[idx]
        icon = (
            "🟢"
            if step["decision"] == "forward"
            else "🔴" if step["decision"] == "backward" else "🟡"
        )
        dec_md = (
            f"### {icon} {step['decision'].upper()}\n"
            f"Position **{step['old_position']}** → **{step['new_position']}**"
        )
        conf = f"{step.get('confidence','?')}/5 — {step.get('confidence_reason','')}"
        trust = f"{step.get('trust','?')}/5 — {step.get('trust_reason','')}"
        vpath = step["video"]
        return (
            vpath if os.path.exists(vpath) else None,
            dec_md,
            conf,
            trust,
            step["reason"],
            render_diagram(history, vfolder or "", idx),
        )

    def run_simulation(pfile, persona, scenario, ehmi, max_steps, temp):
        no_update = (
            gr.update(),
            gr.update(),
        )  # placeholder for overall_video, assessment

        if not persona:
            yield (
                render_diagram([], ""),
                "⚠️ Please select a persona first.",
                [],
                gr.update(choices=[]),
                render_sparkline([]),
                "",
                *no_update,
            )
            return

        vfolder = f"data/250722_real_sim/{scenario}/split"

        system = StreetCrossingDecisionSystem(
            persona_type=persona,
            temperature=temp,
            include_distance=True,
            video_folder=vfolder,
            ehmi_type=ehmi,
            personas_file=f"personas/{pfile}",
            max_time_steps=int(max_steps),
        )

        history = []

        yield (
            render_diagram([], vfolder),
            "🚀 Starting simulation…",
            history,
            gr.update(choices=[]),
            render_sparkline([]),
            vfolder,
            *no_update,
        )

        while not system.is_crossing and system.current_time < system.max_time_steps:
            vp = system.get_next_video_path()

            yield (
                render_diagram(history, vfolder),
                f"⏳ Step {system.current_time} — querying VLM "
                f"(position {system.current_position})…",
                history,
                gr.update(choices=[step_label(h) for h in history]),
                render_sparkline(history),
                vfolder,
                *no_update,
            )

            decision = system.make_decision(vp)
            old_pos = system.current_position
            system.update_position(decision["decision"])

            entry = {
                "time": system.current_time,
                "video": os.path.abspath(vp),
                "old_position": old_pos,
                "new_position": system.current_position,
                "decision": decision["decision"],
                "reason": decision["reason"],
                "confidence": decision.get("confidence", 3),
                "confidence_reason": decision.get("confidence_reason", ""),
                "trust": decision.get("trust", 3),
                "trust_reason": decision.get("trust_reason", ""),
                "status": system.get_position_status(),
                "is_crossing": system.is_crossing,
            }
            system.history.append(entry)
            system.all_status.append(
                {
                    "time": system.current_time,
                    "old_position": old_pos,
                    "new_position": system.current_position,
                    "decision": decision["decision"],
                    "status": entry["status"],
                    "is_crossing": system.is_crossing,
                }
            )
            history = list(system.history)

            if system.is_crossing:
                system.save_results()
                combined = system.combine_videos()
                assessment = (
                    system.get_safety_outcome_description()
                    + "\n\n"
                    + system.get_confidence_trust_evolution()
                )
                yield (
                    render_diagram(history, vfolder),
                    f"✅ Crossed at step {system.current_time}! "
                    f"Saved to `{system.output_folder}`",
                    history,
                    gr.update(choices=[step_label(h) for h in history]),
                    render_sparkline(history),
                    vfolder,
                    combined,
                    assessment,
                )
                return

            system.current_time += 1

        system.save_results()
        combined = system.combine_videos()
        assessment = (
            system.get_safety_outcome_description()
            + "\n\n"
            + system.get_confidence_trust_evolution()
        )
        yield (
            render_diagram(history, vfolder),
            f"✅ Complete. Saved to `{system.output_folder}`",
            history,
            gr.update(choices=[step_label(h) for h in history]),
            render_sparkline(history),
            vfolder,
            combined,
            assessment,
        )

    # ── Wire up events ─────────────────────────────────────────────────────────
    pfile_dd.change(
        on_pfile_change, inputs=[pfile_dd], outputs=[persona_dd, persona_desc]
    )

    persona_dd.change(
        on_persona_change, inputs=[pfile_dd, persona_dd], outputs=[persona_desc]
    )

    scenario_dd.change(on_scenario_change, inputs=[scenario_dd], outputs=[ehmi_dd])

    step_radio.change(
        on_step_select,
        inputs=[step_radio, history_state, vfolder_state],
        outputs=[video_out, decision_md, conf_txt, trust_txt, reason_txt, diagram_plot],
    )

    clicked_node_idx.change(
        on_diagram_click,
        inputs=[clicked_node_idx, history_state, vfolder_state],
        outputs=[video_out, decision_md, conf_txt, trust_txt, reason_txt, diagram_plot],
    )

    run_btn.click(
        run_simulation,
        inputs=[pfile_dd, persona_dd, scenario_dd, ehmi_dd, max_steps_sl, temp_sl],
        outputs=[
            diagram_plot,
            status_md,
            history_state,
            step_radio,
            sparkline_plot,
            vfolder_state,
            overall_video,
            assessment_txt,
        ],
    )

    # Populate persona dropdown on load
    demo.load(
        lambda: on_pfile_change(default_pfile), outputs=[persona_dd, persona_desc]
    )

    # Attach Plotly click → hidden textbox bridge
    demo.load(fn=None, js=_CLICK_JS)


if __name__ == "__main__":
    demo.launch()
