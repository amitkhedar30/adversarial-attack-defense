import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Adversarial Defense Dashboard", layout="wide")

st.title("🛡️ Adversarial Machine Learning: Defense & Certification")
st.markdown("Explore the empirical robustness, certified boundaries, and loss landscapes of our trained ResNet18 models on CIFAR-10.")

# --- Tab Layout ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Empirical Benchmarks", 
    "📈 Certified Safe Zone", 
    "🏔️ Loss Landscapes", 
    "🔀 Transferability Matrix"
])

# --- Tab 1: Empirical Benchmarks ---
with tab1:
    st.header("Baseline vs. PGD Adversarial Training")
    st.markdown("How the models perform under direct White-Box gradient attacks.")
    
    # We can hardcode the final empirical results from Phase 1 here for a clean display
    metrics = {
        "Attack Type": ["Clean (No Attack)", "FGSM (L_inf)", "PGD (L_inf)"],
        "Baseline ResNet18": ["94.44%", "33.77%", "0.03%"],
        "Robust ResNet18 (PGD-AT)": ["79.63%", "53.83%", "49.29%"]
    }
    st.table(pd.DataFrame(metrics))

# --- Tab 2: Certified Defense ---
with tab2:
    st.header("Randomized Smoothing: Certified $L_2$ Radius")
    st.markdown("Mathematical proof that the smoothed model cannot be fooled within specific noise thresholds.")
    
    try:
        df_radii = pd.read_csv("certification_radii.csv")
        st.dataframe(df_radii.head(15)) # Show a preview of the raw math
        
        img_curve = Image.open("certified_accuracy_curve.png")
        st.image(img_curve, caption="Certified Accuracy vs. L2 Radius", use_container_width=True)
    except FileNotFoundError:
        st.warning("Please ensure 'certification_radii.csv' and 'certified_accuracy_curve.png' are in the same folder as app.py.")

# --- Tab 3: Loss Landscapes ---
with tab3:
    st.header("2D Loss Landscape Visualizations")
    st.markdown("Visual proof of why PGD Adversarial Training works. Notice the steep cliff on the left versus the flat plateau on the right.")
    
    try:
        img_landscape = Image.open("loss_landscape_comparison.png")
        st.image(img_landscape, use_container_width=True)
    except FileNotFoundError:
        st.warning("Please ensure 'loss_landscape_comparison.png' is in the same folder as app.py.")

# --- Tab 4: Transferability ---
with tab4:
    st.header("Black-Box Transferability Matrix")
    st.markdown("Cross-architecture attacks between ResNet18 and MobileNetV2.")
    
    try:
        df_matrix = pd.read_csv("transferability_matrix.csv", index_col=0)
        st.table(df_matrix)
    except FileNotFoundError:
        st.warning("Please ensure 'transferability_matrix.csv' is in the same folder as app.py.")