import streamlit as st
from utils.config.Configuration import set_page_config
# Set the page configuration
set_page_config()

# Title and description
st.title("ML Risk Toolkit 📊")
st.write("Welcome to the ML Risk Toolkit – your platform for learning Machine Learning algorithms using Financial Risk Datasets.")
st.write("Explore different ML techniques applied to risk management.")

# Load images
classification_img = "img\Classification.png"  
regression_img = "img\Regression.png"  
clustering_img = "img\Clustering.png"  

# Three sections
col1, col2, col3 = st.columns(3)

with col1:
    st.image(classification_img, caption="Classification", use_column_width=True)
    st.write("Learn how classification models predict financial risks, fraud detection, and credit scoring.")
    if st.button("Go to Classification"):
        st.switch_page("pages/1_Classification.py")

with col2:
    st.image(regression_img, caption="Regression", use_column_width=True)
    st.write("Explore regression models for financial forecasting, credit risk modeling, and market predictions.")
    if st.button("Go to Regression"):
        st.switch_page("pages/2_Regression.py")

with col3:
    st.image(clustering_img, caption="Clustering", use_column_width=True)
    st.write("Understand clustering methods for risk segmentation, anomaly detection, and portfolio analysis.")
    if st.button("Go to Clustering"):
        st.switch_page("pages/3_Clustering.py")


# Footer
st.write("---")
st.write("📌 **ML Risk Toolkit** - Bringing Machine Learning to Financial Risk Analysis.")
