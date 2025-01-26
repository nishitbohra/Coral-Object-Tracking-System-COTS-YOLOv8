import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Enhanced UI Configuration with Dark Theme
st.set_page_config(
    page_title="🌊 Advanced Marine Object Detection", 
    page_icon="🐠", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Advanced Custom CSS for Professional Dark Theme
def load_custom_css():
    st.markdown("""
    <style>
    :root {
        --primary-color: #1e88e5;
        --secondary-color: #00acc1;
        --background-dark: #121212;
        --card-dark: #1e1e1e;
    }
    
    .stApp {
        background-color: var(--background-dark);
        color: #ffffff;
    }
    
    .stCard {
        background-color: var(--card-dark);
        border-radius: 12px;
        border: 1px solid var(--primary-color);
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
        color: #ffffff;
        transition: all 0.3s ease;
    }
    
    .stCard:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.7);
    }
    
    h1, h2, h3 {
        color: var(--primary-color) !important;
        font-family: 'Roboto', sans-serif;
    }
    
    .stMetric {
        background-color: var(--card-dark);
        border-radius: 10px;
        padding: 15px;
        color: #ffffff;
    }
    
    .stTextInput > div > div > input {
        background-color: var(--card-dark);
        color: #ffffff;
        border: 1px solid var(--secondary-color);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model_path = 'E:/COTS/aquarium_pretrain/src/runs/cots/weights/pruned_best.pt'
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Model Loading Error: {e}")
        return None

def detect_objects(model, image):
    if model is None:
        st.error("Model not loaded.")
        return None
    
    try:
        image_np = np.array(image)
        results = model(image_np)
        result = results[0]
        
        processed_results = {
            'names': [result.names[int(cls)] for cls in result.boxes.cls],
            'boxes': result.boxes.xyxy.tolist(),
            'confidences': result.boxes.conf.tolist(),
            'annotated_image': result.plot(),
            'count_per_class': {},
            'confidence_stats': {}
        }
        
        # Enhanced Object Counting and Confidence Analysis
        for name, conf in zip(processed_results['names'], processed_results['confidences']):
            processed_results['count_per_class'][name] = processed_results['count_per_class'].get(name, 0) + 1
            processed_results['confidence_stats'][name] = {
                'count': processed_results['count_per_class'][name],
                'avg_confidence': np.mean([c for n, c in zip(processed_results['names'], processed_results['confidences']) if n == name])
            }
        
        return processed_results
    
    except Exception as e:
        st.error(f"Detection Error: {e}")
        return None

def create_advanced_visualizations(results):
    # Confidence Heatmap
    conf_df = pd.DataFrame.from_dict(results['confidence_stats'], orient='index')
    
    # Confidence vs Count Scatter Plot
    plt.figure(figsize=(10, 6), facecolor='#121212')
    plt.style.use('dark_background')
    plt.scatter(conf_df['count'], conf_df['avg_confidence'], 
                c=conf_df['avg_confidence'], cmap='viridis', 
                alpha=0.7, s=100)
    plt.title('Object Detection: Count vs Confidence', color='white')
    plt.xlabel('Object Count', color='lightgray')
    plt.ylabel('Average Confidence', color='lightgray')
    plt.colorbar(label='Confidence Level')
    plt.tight_layout()
    
    return plt

def main():
    load_custom_css()
    
    # Professional Sidebar
    st.sidebar.title("🌊 Marine Intelligence Platform")
    st.sidebar.markdown("### Advanced Object Detection System")
    
    st.sidebar.info("""
    🔬 System Capabilities:
    - Precision Marine Object Detection
    - Advanced AI Analytics
    - Environmental Monitoring
    
    Developed with ❤️ by Nishit Bohra
    """)

    # Main Content
    st.title("🐠 Marine Object Detection System")
    st.subheader("Intelligent Underwater Environment Analysis")
    
    # Model Loading
    model = load_model()
    
    # Enhanced File Uploader
    uploaded_file = st.file_uploader(
        "Upload Marine Image", 
        type=["jpg", "jpeg", "png"],
        help="Analyze marine environments with advanced AI"
    )
    
    if uploaded_file:
        # Visualization Columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🖼️ Detailed Detection Analysis")
            image = Image.open(uploaded_file)
            results = detect_objects(model, image)
        
        if results:
            # Annotated Image
            with col1:
                st.image(results['annotated_image'], 
                         caption="Advanced COTS Analyzed Image", 
                         use_container_width=True)
            
            # Detailed Analysis
            with col2:
                st.subheader("🔍 Intelligent Insights")
                
                total_objects = len(results['names'])
                st.metric("Detected Marine Objects", total_objects)
                
                # Object Distribution Pie Chart
                if results['count_per_class']:
                    fig = px.pie(
                        values=list(results['count_per_class'].values()),
                        names=list(results['count_per_class'].keys()),
                        title="Marine Object Distribution",
                        color_discrete_sequence=px.colors.sequential.Viridis
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Advanced Visualizations
            advanced_viz = create_advanced_visualizations(results)
            st.pyplot(advanced_viz)
        
        # Professional Project Summary
        st.markdown("## 🌍 Advanced Tracking Methodology")
        st.info(f"""
        **Marine Intelligence Insights:**
        - Detected Objects: {total_objects}
        - AI-Powered Precision Detection
        - Comprehensive Marine Environment Analysis
        
        **Developed with Advanced Machine Learning Technologies**
        """)

if __name__ == "__main__":
    main()