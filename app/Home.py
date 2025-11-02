import streamlit as st

st.set_page_config(page_title="License Plate OCR Pipeline", layout="centered")

st.title("ANPR Pipeline")

st.write(
    """
    Welcome! You can explore our **features**, try out the **full ANPR model**,  
    or test **individual modules** — just select a page from the sidebar.
    """
)

graph = """
digraph LicensePlateOCR {
    rankdir=TB;
    fontname="Helvetica";

    node [shape=rectangle, style="rounded,filled", fontname="Helvetica", fontsize=13, width=2.8, height=0.7];
    edge [fontname="Helvetica", fontsize=12, color="#444444", penwidth=1.4];

    // --- Colors and Clusters ---
    subgraph cluster_detection {
        label="1. Detection Stage";
        color="#c8e6c9";
        style="rounded,filled";
        fillcolor="#e8f5e9";
        fontsize=16;
        Start [shape=ellipse, fillcolor="#81c784", fontsize=13, label="Start"];
        Input [fillcolor="#a5d6a7", label="Input Image"];
        Detect [fillcolor="#a5d6a7", label="Object Detection Model"];
        Region [fillcolor="#a5d6a7", label="Detected License Plate Region"];
        Crop [fillcolor="#a5d6a7", label="Crop Detected License Plate"];
    }

    subgraph cluster_ocr {
        label="2. OCR & Confidence Evaluation";
        color="#bbdefb";
        style="rounded,filled";
        fillcolor="#e3f2fd";
        fontsize=16;
        OCR [fillcolor="#90caf9", label="OCR Model\\n(Predict Text + Confidence)"];
        DecisionHigh [shape=diamond, fillcolor="#64b5f6", label="Confidence > 0.8"];
        OutputHigh [fillcolor="#bbdefb", label="Output OCR Result"];
        DecisionMid [shape=diamond, fillcolor="#64b5f6", label="0.6 < Confidence ≤ 0.8"];
        ProcUpscale [fillcolor="#bbdefb", label="Resize & Upscale → Binarize → Grayscale → OCR Again"];
        ProcFull [fillcolor="#bbdefb", label="Full Image Processing:\\nRescale → Binarize → Grayscale → Sharpen → CLAHE → Corrode → OCR Again"];
    }

    subgraph cluster_post {
        label="3. Post-Processing & Validation";
        color="#ffe0b2";
        style="rounded,filled";
        fillcolor="#fff3e0";
        fontsize=16;
        Vote [fillcolor="#ffcc80", label="Vote / Compare Against Previous Predictions\\n(Select Most Frequent or Highest Confidence)"];
        FinalOCR [fillcolor="#ffe0b2", label="Final OCR Result"];
        Regex [fillcolor="#ffe0b2", label="Apply Regex Filtering"];
        Valid [shape=diamond, fillcolor="#ffb74d", label="Valid License Plate Format?"];
        Yes [fillcolor="#fff3e0", label="Yes → Return Final Plate String"];
        No [fillcolor="#fff3e0", label="No → Flag as Invalid / Reprocess / Reject"];
        End [shape=ellipse, fillcolor="#ffcc80", fontsize=13, label="End"];
    }

    // --- Connections ---
    Start -> Input -> Detect -> Region -> Crop -> OCR -> DecisionHigh;
    DecisionHigh -> OutputHigh [label="Yes", color="#388e3c"];
    DecisionHigh -> DecisionMid [label="No", color="#d32f2f"];
    DecisionMid -> ProcUpscale [label="0.6 < c ≤ 0.8", color="#1976d2"];
    DecisionMid -> ProcFull [label="c ≤ 0.6", color="#1976d2"];
    OutputHigh -> Vote;
    ProcUpscale -> Vote;
    ProcFull -> Vote;
    Vote -> FinalOCR -> Regex -> Valid;
    Valid -> Yes [label="Yes", color="#388e3c"];
    Valid -> No [label="No", color="#d32f2f"];
    Yes -> End;
    No -> End;
}
"""

st.graphviz_chart(graph)
