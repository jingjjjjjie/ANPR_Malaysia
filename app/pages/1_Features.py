import streamlit as st

st.title("Features")

st.markdown(
    "Advanced AI-powered license plate detection built for real-world reliability while keeping it "
    "<span style='background-color: #FFF3B0; padding: 0.2em 0.4em; border-radius: 4px;'>lightweight</span> "
    "and efficient.",
    unsafe_allow_html=True
)
st.caption("Note: Certain edge cases may still produce false detections or missed recognitions. Capabilities are powered by our **data-driven training curriculum**.")

st.subheader("1. Powerful Detection")
st.write("Accurately detects license plates at various distances and sizes, including those that are partially obscured.")


col1, col2 = st.columns(2)

with col1:
    st.image("images/demo_predictions/a.jpg", caption="Distant plates", use_container_width=True)

with col2:
    st.image("images/demo_predictions/b.jpg", caption="Partially Obscured", use_container_width=True)

st.subheader("2. All-Weather Performance")
st.write("Designed to maintain high detection accuracy under challenging conditions such as glare, low illumination, nighttime environments, and rain.")

col1, col2 = st.columns(2)

with col1:
    st.image("images/demo_predictions/c.jpg", caption="Rainy Weather", use_container_width=True)

with col2:
    st.image("images/demo_predictions/d.jpg", caption="Glare", use_container_width=True)

st.subheader("3. Smart Differentiation")
st.write("Effectively differentiates genuine license plates from signboards and other plate-like text patterns.")

col1, col2 = st.columns(2)

with col1:
    st.image("images/demo_predictions/e.jpg", caption="Plate-Like Signboards", use_container_width=True)

with col2:
    st.image("images/demo_predictions/f.jpg", caption="Signboards", use_container_width=True)

st.markdown('---')