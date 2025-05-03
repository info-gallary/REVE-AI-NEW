import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from textwrap import dedent
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from dotenv import load_dotenv
from agno.models.groq import Groq
from agno.agent import Agent, RunResponse
from agno.tools.csv_toolkit import CsvTools
from agno.tools.reasoning import ReasoningTools
import tempfile
import re
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import tempfile

import streamlit as st
import requests
import streamlit as st
import requests
import re
from PIL import Image
from io import BytesIO
import streamlit as st
import requests
import re
from PIL import Image
from io import BytesIO
from predict_d import predict_d
from predict_c import predict_c
from PIL import Image
from fastapi import FastAPI, HTTPException,File,Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from io import BytesIO
from textwrap import dedent
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from agno.models.groq import Groq
import numpy as np

from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.arxiv import ArxivTools
from agno.tools.googlesearch import GoogleSearchTools

from agno.media import Image as AgnoImage
from fastapi import UploadFile, File
from typing import Optional
from typing import Dict, Optional, List
import asyncio
import requests
import os


# Set page config
st.set_page_config(
        page_title="REVE-AI",
        page_icon="reve_logo.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )
# Create tabs
tab1, tab2 = st.tabs(["Soil Health Analysis", "Skin Diagnosis Dashboard"])
st.sidebar.image("reve_logo.png", use_container_width=True, caption="REVE AI")
st.sidebar.markdown("---")







with tab1:
    # Title with styling
    st.markdown("""
    # 🌱 Soil Health Analysis Dashboard
    Analyze soil properties for better agricultural outcomes
    """)
    
    # Upload CSV
    uploaded_file = st.file_uploader("Upload your soil data CSV file", type=["csv"])
    

    if uploaded_file is not None:
        # Save the uploaded file to a temporary CSV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        # Read the CSV file
        df = pd.read_csv(temp_path)

        # Show a success message and a preview of the raw data
        st.success("CSV uploaded successfully!")
        with st.expander("Preview Raw Data"):
            st.dataframe(df)
            st.caption(f"CSV saved to temporary path: {temp_path}")
            st.subheader("Column Names")
            st.write(df.columns.tolist())

        # Strip whitespace from column names and convert them to lower-case for ease of matching
        df.columns = [col.strip() for col in df.columns]
        df.columns = [col.lower() for col in df.columns]

        # Define the wavelength columns (as strings) that are required.
        required_wavelengths = [str(w) for w in [410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940]]

        # Define the required chemical property columns.
        required_chem = ["capacitive", "temp", "moist", "ec", "ph", "nitro", "posh", "pota"]

        # Only keep the columns that exist in the dataset
        existing_wavelengths = [col for col in required_wavelengths if col in df.columns]
        existing_chem = [col for col in required_chem if col in df.columns]
        required_columns = existing_wavelengths + existing_chem

        # Create a new dataframe with only the required columns
        df_required = df[required_columns].copy()

        # Sidebar for visualization selection
        st.sidebar.title("📊 Soil Data Analysis Controls")
        viz_type = st.sidebar.radio(
            "Choose Visualization Type",
            ["Spectral Analysis", "Chemical Properties", "Comprehensive View"]
        )

        # Data summary metrics
        st.markdown("## 📈 Key Metrics")
        metric_cols = st.columns(4)
        if not df_required.empty:
            with metric_cols[0]:
                try:
                    st.metric("Average pH", f"{df_required['ph'].mean():.2f}")
                except Exception:
                    st.metric("Average pH", "N/A")
            with metric_cols[1]:
                try:
                    st.metric("Temperature", f"{df_required['temp'].mean():.1f}°C")
                except Exception:
                    st.metric("Temperature", "N/A")
            with metric_cols[2]:
                st.metric("Samples Count", f"{len(df_required)}")
            with metric_cols[3]:
                try:
                    st.metric("Avg Capacitive", f"{df_required['capacitive'].mean():.0f}")
                except Exception:
                    st.metric("Avg Capacitive", "N/A")

        # Visualization based on selection
        if viz_type == "Spectral Analysis":
            st.markdown("## 🔬 Soil Spectral Analysis")
            spec_tab1, spec_tab2, spec_tab3 = st.tabs(["Line Plot", "Heatmap", "3D View"])

            with spec_tab1:
                # Line plot: each row's spectral data across the available wavelength columns
                fig = go.Figure()
                for i, row in df_required.iterrows():
                    fig.add_trace(go.Scatter(
                        x=existing_wavelengths,
                        y=[row[w] for w in existing_wavelengths],
                        mode='lines+markers',
                        name=f"Sample {i+1}"
                    ))
                fig.update_layout(
                    title="Spectral Response per Sample",
                    xaxis_title="Wavelength (nm)",
                    yaxis_title="Absorbance",
                    legend_title="Sample Number",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

            with spec_tab2:
                # Heatmap of spectral data
                heatmap_data = df_required[existing_wavelengths].values
                fig = px.imshow(
                    heatmap_data,
                    labels=dict(x="Wavelength (nm)", y="Sample Number", color="Absorbance"),
                    x=existing_wavelengths,
                    y=[f"Sample {i+1}" for i in range(len(df_required))],
                    color_continuous_scale="Viridis"
                )
                fig.update_layout(height=500, title="Spectral Absorbance Heatmap")
                st.plotly_chart(fig, use_container_width=True)

            with spec_tab3:
                # 3D Surface plot of spectral data
                z_data = df_required[existing_wavelengths].values
                fig = go.Figure(data=[go.Surface(
                    z=z_data,
                    x=existing_wavelengths,
                    y=[f"Sample {i+1}" for i in range(len(df_required))],
                    colorscale='Viridis'
                )])
                fig.update_layout(
                    title='3D Spectral Surface',
                    scene=dict(
                        xaxis_title='Wavelength (nm)',
                        yaxis_title='Sample',
                        zaxis_title='Absorbance'
                    ),
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Chemical Properties":
            st.markdown("## 🧪 Chemical Properties Analysis")
            chem_tab1, chem_tab2 = st.tabs(["Data Table", "Property Charts"])

            with chem_tab1:
                st.subheader("Chemical Properties Overview")
                st.dataframe(df_required[existing_chem])

            with chem_tab2:
                col1, col2 = st.columns(2)
                with col1:
                    try:
                        fig = px.bar(
                            df_required,
                            x=df_required.index,
                            y="ph",
                            title="pH Distribution Across Samples",
                            color="ph",
                            color_continuous_scale="RdYlGn",
                            text_auto='.2f'
                        )
                        fig.add_hline(y=7.0, line_dash="dash", line_color="gray", annotation_text="Neutral pH")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.error("Error plotting pH distribution.")
                with col2:
                    try:
                        fig = px.scatter(
                            df_required,
                            x="temp",
                            y="capacitive",
                            title="Temperature vs Capacitive",
                            color=df_required.index,
                            size=[10] * len(df_required)
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.error("Error plotting Temperature vs Capacitive.")

                if (df_required["nitro"].sum() > 0 or df_required["posh"].sum() > 0 or df_required["pota"].sum() > 0):
                    npk_cols = ["nitro", "posh", "pota"]
                    fig = go.Figure()
                    for i, row in df_required.iterrows():
                        fig.add_trace(go.Scatterpolar(
                            r=[row[col] for col in npk_cols],
                            theta=['Nitrogen', 'Phosphorous', 'Potassium'],
                            fill='toself',
                            name=f"Sample {i+1}"
                        ))
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                            )
                        ),
                        showlegend=True,
                        title="NPK Distribution (Radar Chart)",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("NPK data is not available in this dataset (all values are zero).")

        else:  # Comprehensive View
            st.markdown("## 🔍 Comprehensive Soil Analysis")
            col1, col2 = st.columns(2)

            with col1:
                fig = go.Figure()
                for i, row in df_required.iterrows():
                    fig.add_trace(go.Scatter(
                        x=existing_wavelengths,
                        y=[row[w] for w in existing_wavelengths],
                        mode='lines',
                        name=f"Sample {i+1}"
                    ))
                fig.update_layout(
                    title="Spectral Response",
                    xaxis_title="Wavelength (nm)",
                    yaxis_title="Absorbance",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)

                fig = px.bar(
                    df_required,
                    x=df_required.index,
                    y="ph",
                    title="pH Values",
                    color="ph",
                    color_continuous_scale="RdBu_r",
                    text_auto='.2f'
                )
                fig.add_hline(y=7.0, line_dash="dash", line_color="gray")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Chemical Properties")
                st.dataframe(df_required[existing_chem], height=350)

                fig = px.histogram(
                    df_required,
                    x="temp",
                    nbins=10,
                    title="Temperature Distribution",
                    color_discrete_sequence=['indianred']
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        # AI Analysis with Groq
        st.markdown("## 🤖 AI-Powered Soil Analysis")
        
        analysis_tab1, analysis_tab2 = st.tabs(["Raw Analysis", "Comprehensive Report"])
        
        with analysis_tab1:
            if st.button("Generate Basic Analysis"):
                with st.spinner("Analyzing soil data..."):
                    # Load environment variables
                    load_dotenv()
                    
                    agent = Agent(
                        model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
                        # model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
                        # model=Gemini(id="gemini-2.0-flash-exp"),    
                        tools=[CsvTools(csvs=[temp_path],row_limit=11,read_csvs=True)],
                        markdown=True,
                        show_tool_calls=True,
                        instructions=[
                            """
                        Read all the rows from the dataset and generate a detailed soil health report. Use the following parameters from the CSV:
                
                        Wavelengths: [410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940]
                        Capacity Moisture 
                        Moisture
                        Temperature
                        Electric conductivity (μS/10 gram)
                        pH
                        Nitrogen (mg/10 g)
                        Phosphorous (mg/10 g)
                        Potassium (mg/10 g)
                
                        For each record combine for one soil, 
                        analyze the soil's physical and chemical properties and summarize whether the soil condition is excellent, good, moderate, or poor. 
                        Provide observations and possible recommendations for improvement, such as fertilizer use, irrigation suggestions, or pH balancing if needed.
                    """
                        ]
                    )
                
                    response: RunResponse = agent.run("give analysis of soil health?")
                    st.markdown(response.content)
        
        with analysis_tab2:
            if st.button("Generate Comprehensive Report"):
                with st.spinner("Creating comprehensive soil health report..."):
                    # Load environment variables
                    load_dotenv()
                    
                    # First agent to get initial analysis
                    agent = Agent(
                        model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
                        # model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
                        # model=Gemini(id="gemini-2.0-flash-exp"),    
                        tools=[CsvTools(csvs=[temp_path],row_limit=11,read_csvs=True)],
                        markdown=True,
                        show_tool_calls=True,
                        instructions=[
                            """
                        Read all the rows from the dataset and generate a detailed soil health report. Use the following parameters from the CSV:
                
                        Wavelengths: [410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940]
                        Capacity Moisture 
                        Moisture
                        Temperature
                        Electric conductivity (μS/10 gram)
                        pH
                        Nitrogen (mg/10 g)
                        Phosphorous (mg/10 g)
                        Potassium (mg/10 g)
                
                        For each record combine for one soil, 
                        analyze the soil's physical and chemical properties and summarize whether the soil condition is excellent, good, moderate, or poor. 
                        Provide observations and possible recommendations for improvement, such as fertilizer use, irrigation suggestions, or pH balancing if needed.
                    """
                        ]
                    )
                
                    response: RunResponse = agent.run("give analysis of soil health?")
                    
                    # Second agent for comprehensive report
                    report_agent = Agent(
                        name="Academic Paper Researcher",
                        # model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
                        model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
                        model=Gemini(id="gemini-2.0-flash-exp"),    
                        role="Research academic papers and scholarly content",
                        tools=[ReasoningTools(
                                think=True,
                                analyze=True,
                                add_instructions=True,
                                add_few_shot=True,
                            )],
                        context={"report": response.content},
                        add_context=True,
                        add_name_to_instructions=True,
                        instructions=dedent("""
                                            
                                            
                    You are given structured analysis {report} of data across multiple samples for the same soil. Each sample includes:
                
                    - Spectral data across wavelengths [410–940 nm]
                    - Physical properties: Moisture (Capacity and Actual), Temperature
                    - Chemical properties: pH, Electric Conductivity (EC), Nitrogen, Phosphorous, Potassium
                    - A health classification (Excellent/Good/Moderate/Poor)
                    - Observations and suggested improvements for each sample
                
                    Your task is to generate a **Soil Health Report** that:
                
                    1. **Aggregates all insights** to provide a holistic understanding of the soil's condition.
                    2. **Analyzes trends** in the spectral and chemical data across the different samples.
                    3. **Identifies consistency or variation** in parameters like moisture, pH, nutrient levels, etc.
                    4. **Provides an overall soil health rating** with reasoning (based on the average and variation).
                    5. **Summarizes common observations** and **consolidates the best recommendations** into a final action plan.                      
                    6. Describes:
                    - Suitability of this soil for common crops (based on NPK, pH, EC, etc.)
                    - Fertilizer or organic amendment suggestions
                    - Irrigation or drainage needs
                    - Warnings about salinity, acidity/alkalinity, or imbalances
                
                    Use clear, informative, and farmer-friendly language. Avoid technical jargon unless necessary. Ensure that a non-expert (e.g., a farmer or agriculture officer) can easily understand the report and act on it.
                    """
                    ),
                    )
                    res: RunResponse=report_agent.run("generate soil health report of given data?")
                    st.markdown(res.content)
    
    else:
        st.info("Please upload a CSV file to begin analysis.")
        
        # Add sample data and instructions
        with st.expander("How to use this dashboard"):
            st.write("""
            1. Upload your soil data CSV file using the file uploader above
            2. The dashboard will automatically process and visualize the data
            3. Use the sidebar to filter data and choose visualization types
            4. Generate AI-powered analysis using the buttons in the Analysis section
            """)
            
            st.markdown("#### Required CSV Format")
            st.write("""
            Your CSV should contain the following columns:
            - Records: in the format 'X_Yml-Z' where X is soil index, Y is water level, Z is sample number
            - Wavelength columns: 410, 435, 460, etc. (spectral data)
            - Chemical properties: pH, EC, Nitrogen, Phosphorous, Potassium
            - Physical properties: Moisture, Temperature
            """)

with tab2:
    # Custom CSS
        

    # Configure the page


    # Custom CSS with additional styling
    st.markdown("""
    <style>
        /* Existing styles */
        .upload-box {
            border: 2px dashed #4e8cff;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 2rem;
            background-color: rgba(0, 128, 0, 0.1);
            transition: all 0.3s ease;
        }
        .upload-box:hover {
            background-color: rgba(0, 128, 0, 0.2);
            transform: scale(1.01);
        }
        .report-section {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-top: 1.5rem;
        }
        .header {
            background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        /* New styles */
        .sidebar .sidebar-content {
            background-image: linear-gradient(180deg, #2b5876 0%, #4e4376 100%);
        }
        .sidebar .sidebar-content .block-container {
            padding-top: 2rem;
        }
        .stButton>button {
            background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
            color: white;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .stat-box {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
            margin: 1rem 0;
        }
        .stat-number {
            font-size: 1rem;
            font-weight: bold;
            color: #2b5876;
        }
    </style>
    """, unsafe_allow_html=True)


    # Main content
    st.markdown("<div class='header'><h1>🏥 REVE AI</h1><h3>Advanced Skin Disease Diagnosis</h3></div>", unsafe_allow_html=True)

    # Create tabs for different sections
    def classify_image(
            file: UploadFile = File(...),
            age: int = Form(...),
            gender: str = Form(...),
            notes: Optional[str] = Form(None)
        ):

            try:
                print(patient_id,age,gender,notes)
                image = Image.open(file).convert("RGB")
                img_bytes = BytesIO()
                image.save(img_bytes, format="PNG")
                image_bytes = img_bytes.getvalue()
                agno_image = AgnoImage(content=image_bytes, format="png")
                verify_med_agent = Agent(
                    name="Medical Imaging Expert",
                    model=Gemini(id="gemini-2.0-flash-exp"),
                    tools=[DuckDuckGoTools()],
                    markdown=True,
                    instructions=dedent(
                        f"""Analyze the given skin image as an very good and expert dermatologist and expert to determine if the skin is healthy or unhealthy and also check that to mislead the model is there any artificial marks or not that should be notice by you.
                            - confidence percentage should be between 90 to 100 and you can use the decimal value also.
                            - If healthy, classify it as 'Healthy' and provide the confidence level in percentage.
                            - If unhealthy, classify it as 'Unhealthy' and provide the confidence level in percentage.
                            - Additionally, determine the skin type as one of the following: 'Dry', 'Oily', or 'Normal'.
                            - give answer in strictly <classification>,<confidence score in percent>,<skin type>,<remarks : give some remarks that is in one to two lines> format only.
                            """
                    )
                )

                result: RunResponse = verify_med_agent.run("Please analyze this medical image.", images=[agno_image])
                result_c = predict_c(image)
                result_d = predict_d(image)
                if result_c["confidence"] > result_d["confidence"]:
                    result_pred = result_c
                    minor_result = result_d
                elif result_d["confidence"] > result_c["confidence"]:
                    result_pred = result_d
                    minor_result = result_c
                else:
                    result_pred = result_c        
                unhealthy_skin_agent = Agent(
                    name="Medical Imaging Analysis Expert",
                    model=Gemini(id="gemini-2.0-flash-exp"),
                    tools=[DuckDuckGoTools()],
                    context={"verify": result.content,"notes":notes,"age":age,"gender":gender},
                    add_context=True,
                    markdown=True,
                    instructions=dedent(
                        f"""Analyze the given skin image as an expert dermatologist.If the skin appears healthy, classify the prediction as it 'Healthy' and provide the confidence level. If unhealthy, use the model output to determine the disease. The prediction is by deep learning model is {result_pred}. If classified as one of the following: 'Actinic Keratosis', 'Atopic Dermatitis', 'Benign Keratosis', 'Dermatofibroma', 'Melanocytic Nevus', 'Melanoma', 'Squamous Cell Carcinoma', 'Tinea Ringworm Candidiasis', or 'Vascular Lesion', assess the likelihood of skin cancer other wise its a diesease. Provide the disease name, confidence level, and remarks. Additionally, include possible symptoms that might be present for further diagnostic evaluation.
                        - give answer in strictly <disease>,<confidence score in percent>,<remarks in two to three lines> format only.
                        - If the skin appears healthy, classify it as 'Healthy' and provide the confidence level in percentage.
                        """
                    )
                )
                print("diagnosis....")
                pred: RunResponse = unhealthy_skin_agent.run("Please analyze this medical image.", images=[agno_image])
                    # model=Groq(id="llama-3.1-8b-instant"),


                report_agent = Agent(
                    name="Medical Imaging Analysis and report generator Expert",
                    model=Gemini(id="gemini-2.0-flash-exp"),
                    tools=[GoogleSearchTools()],
                    context={"pred": pred.content,"notes":notes,"age":age,"gender":gender},
                    add_context=True,
                    markdown=True,
                    instructions=dedent(f"""# Skin Disease Diagnosis Report 🏥  
                                        If the skin classification is unhealthy then in report also add the our model predicted that  {pred.content}  but it also give two answer                         

        Using search and provide relevent links and based on the context give complete diagnosis report for dermatologist to understand the case :  
                                        
        ## Step 1: Image Technical Assessment  

        ### 1.1 Imaging & Quality Review  
        - Imaging Modality Identification: (Dermatoscopic, Clinical, Histopathological, etc.)  
        - Anatomical Region & Patient Positioning: (Specify if available)  
        - Image Quality Evaluation: (Contrast, Clarity, Presence of Artifacts)  
        - Technical Adequacy for Diagnostic Purposes: (Yes/No, with reasoning)  

        ### 1.2 Professional Dermatological Analysis  
        - Systematic Anatomical Review  
        - Primary Findings: (Lesion Size, Shape, Texture, Color, etc.)  
        - Secondary Observations (if applicable)  
        - Anatomical Variants or Incidental Findings  
        - Severity Assessment: (Normal / Mild / Moderate / Severe)  

        ---

        ## Step 2: Context-Specific Diagnosis & Clinical Interpretation  
        - Primary Diagnosis: (Detailed interpretation based on the given disease context)  
        - Secondary Condition (if suspected): (Mention briefly without shifting focus)  

        ---

        ## Step 3: Recommended Next Steps  
        - Home Remedies & Skincare: (Moisturizing, Avoiding Triggers, Hydration)  
        - Medications & Treatments: (Antifungal, Antibiotic, Steroid Creams, Oral Medications)  
        - When to See a Doctor: (Persistent Symptoms, Spreading, Bleeding, Painful Lesions)  
        - Diagnostic Tests (if required): (Skin Biopsy, Allergy Tests, Blood Tests)  

        ---

        ## Step 4: Patient Education  
        - Clear, Jargon-Free Explanation of Findings  
        - Visual Analogies & Simple Diagrams (if helpful)  
        - Common Questions Addressed  
        - Lifestyle Implications (if any)  

        ---

        ## Step 5: Ayurvedic or Home Solutions  
        (Applied only if the condition is non-cancerous or mild and use web search)  
        - Dry & Irritated Skin: Apply Aloe Vera gel, **Coconut oil, or **Ghee for deep moisturization.  
        - Inflammation & Redness: Use a paste of Sandalwood (Chandan) and Rose water for cooling effects.  
        - Fungal & Bacterial Infections: Apply Turmeric (Haldi) paste with honey or Neem leaves for antimicrobial benefits.  
        - Eczema & Psoriasis: Drink Giloy (Guduchi) juice and use a paste of Manjistha & Licorice (Yashtimadhu) for skin detox.  

        ---

        ## Step 6: Evidence-Based Context & References  
        - Recent relevant medical literature  
        - Standard treatment guidelines  
        - Similar case studies  
        - Technological advances in imaging/treatment  
        - 2-3 authoritative medical references
        - give related links also with references.  

        ---

        ## Final Summary & Conclusion  
        📌 Key Takeaways:  
        - Most Likely Diagnosis: (Brief summary)  
        - Recommended Actions: (Main steps for treatment and next consultation)  
        The most likely condition the patient could have is *{result_pred['class']}* with a confidence of {result_pred['confidence']:.2f}. 
        Additionally, there is a minor possibility of *{minor_result['class']}* with a confidence of {minor_result['confidence']:.2f}. 

        *Remarks:*  
        - *{result_pred['class']}* (Confidence: {result_pred['confidence']:.2f}) is the primary concern and should be prioritized for diagnosis and treatment.  
        - *{minor_result['class']}* (Confidence: {minor_result['confidence']:.2f}) may be a secondary condition or share similar symptoms. Further medical evaluation is recommended to rule it out.
        Note: This report is AI-generated and should not replace professional medical consultation. Always consult a dermatologist for a confirmed diagnosis and personalized treatment.  
        - give answer in proper markdown format.

        ---
        """)
                )
                print("report generating....")
                report: RunResponse = report_agent.run("Please analyze this skin image output context and generate a proper diagnosis report for Dermatologist to understand.", images=[agno_image])
                print("report completed....")
                net_agent = Agent(
                    name="Medical Imaging Expert",
                    model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
                    # model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
                    # model=Gemini(id="gemini-2.0-flash-exp"),
                    tools=[ArxivTools()],  
                    context={"pred": pred.content,"notes":notes,"age":age,"gender":gender},
                    add_context=True,
                    markdown=True,  
                    instructions=dedent(
                        f"""You are an AI-powered Dermatology Voice Assistant, designed to provide expert-level support to dermatologists. Your role is to analyze report {report.content} recommend evidence-based treatments, and guide doctors on the next steps using the latest research and drug discoveries.  
                        The most likely condition the patient could have is *{result_pred['class']}* with a confidence of {result_pred['confidence']:.2f}. 
                        Additionally, there is a minor possibility of *{minor_result['class']}* with a confidence of {minor_result['confidence']:.2f}. 
                        use arxiv tool to provide accurate relevent links.

                        *Remarks:*  
                        - *{result_pred['class']}* (Confidence: {result_pred['confidence']:.2f}) is the primary concern and should be prioritized for diagnosis and treatment.  
                        - *{minor_result['class']}* (Confidence: {minor_result['confidence']:.2f}) may be a secondary condition or share similar symptoms. Further medical evaluation is recommended to rule it out.

                        ### 1️⃣ Understand & Analyze the Case  
                        - Listen to the doctor’s query about a patient’s condition.  
                        - Identify the disease or condition being discussed.  
                        - Analyze symptoms, affected areas, and disease progression based on the given context or medical report.  

                        ### 2️⃣ Provide the Latest Treatment Recommendations  
                        - Fetch current treatment guidelines, FDA-approved drugs, and clinical trials using web sources.  
                        - Explain the best available treatment options, including **topical, oral, biologic, and advanced therapies.  
                        - Compare traditional treatments with newly discovered therapies (e.g., AI-assisted skin diagnostics, gene therapy, biologics).  

                        ### 3️⃣ Generate a Complete Prescription Plan  
                        - Suggest medications, dosages, frequency, and possible side effects.  
                        - Recommend adjunct therapies, such as lifestyle modifications and skincare routines.  
                        - Warn about contraindications or potential drug interactions.  

                        ### 4️⃣ Guide the Doctor on the Next Steps  
                        - Recommend further diagnostic tests (e.g., biopsy, dermoscopy, blood tests, genetic markers).  
                        - Suggest patient follow-up intervals and monitoring plans.  
                        - Provide guidelines for managing severe or resistant cases.  

                        ### 5️⃣ Provide Reliable Medical Sources & Links  
                        - Fetch research-backed insights from trusted sources such as PubMed, JAMA Dermatology, The Lancet, FDA, and WHO.  
                        - Offer links to the latest studies, treatment guidelines, and clinical trials for validation.  
                        ---

                        Instructions should be understandable by Dermatologists not for layman audience and make it like a proffesional advice to doctor like doctor is giving advice to the other doctor and make complete instruction summarize and in 4 to 5 lines pointwise.
                        - give answer in proper markdown format.

                        """)

                )
                print("jarvis...")
                jarvis: RunResponse = net_agent.run("Please analyze this skin based diagnostics report and give instructions to doctor")
                data = {"Verify": result.content, "Prediction": pred.content,"Report":report.content,"Jarvis": jarvis.content}
                print("completed....")
            

                return data
            except requests.exceptions.RequestException as e:
                raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")
    def clean_report_content(report_text):
            """Extract only the second occurrence of the report content"""
            # Split by the header pattern
            parts = report_text.split("# Skin Disease Diagnosis Report 🏥")
            
            if len(parts) > 2:
                # Reconstruct with just the second report
                return "# Skin Disease Diagnosis Report 🏥" + parts[2]
            return report_text  # fallback to original if no duplicates

    # Upload section
    st.markdown("<div class='upload-box'><h3>Upload Skin Image for Analysis</h3></div>", unsafe_allow_html=True)

    # Create two columns for patient info
    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input("Patient ID")
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        notes = st.text_area("Notes (optional)")

    # Image upload with preview
    uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    # Your existing image processing and analysis code here...
    cond = None
    diag = None
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Skin Image", width=300)
        with col2:
            st.markdown("""
                <div class="stat-box" style="color: black;">
                    <h4>Image Details</h4>
                    <p>Size: {} x {}</p>
                    <p>Format: {}</p>
                </div>
            """.format(image.size[0], image.size[1], image.format), unsafe_allow_html=True)


        if st.button("Analyze Image", type="primary"):
                with st.spinner("🔍 Analyzing image with our AI models. This may take 30-60 seconds..."):
                    try:

                        results = classify_image(file=uploaded_file,age=age,gender=gender,notes=notes)
                    
                        
                        # Clean the report content
                        cleaned_report = clean_report_content(results["Report"])
                        
                        # Display results in expandable sections
                        st.success("✅ Analysis Complete!")
                        
                        # Quick summary cards
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            with st.expander("Initial Assessment", expanded=True):
                                verify_parts = results['Verify'].strip('`').split(',')
                                st.write(f"**Classification:** {verify_parts[0]}")
                                
                                confidence = verify_parts[1].replace('%','')
                                conf_class = "severity-high" if float(confidence) > 80 else "severity-medium" if float(confidence) > 50 else "severity-low"
                                st.write(f"**Confidence:** <span class='{conf_class}'>{verify_parts[1]}</span>", unsafe_allow_html=True)
                                
                                st.write(f"**Skin Type:** {verify_parts[2]}")
                                st.write(f"**Remarks:** {verify_parts[3]}")
                        
                        with col2:
                            with st.expander("Disease Prediction", expanded=True):
                                pred_parts = results['Prediction'].split(',')
                                diag = pred_parts[0]
                                st.write(f"**Condition:** {pred_parts[0]}")
                                
                                confidence = pred_parts[1].replace('%','')
                                cond=confidence
                                conf_class = "severity-high" if float(confidence) > 80 else "severity-medium" if float(confidence) > 50 else "severity-low"
                                st.write(f"**Confidence:** <span class='{conf_class}'>{pred_parts[1]}</span>", unsafe_allow_html=True)
                                
                                if len(pred_parts) > 2:
                                    st.write(f"**Remarks:** {pred_parts[2]}")
                        
                        # Full cleaned report
                        with st.expander("📋 Full Diagnosis Report", expanded=True):
                            st.markdown("<div class='report-section'>", unsafe_allow_html=True)
                            st.markdown(cleaned_report)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Clinical recommendations
                        with st.expander("💊 Clinical Recommendations", expanded=False):
                            st.markdown("<div class='report-section'>", unsafe_allow_html=True)
                            st.markdown(results['Jarvis'])
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                            
                    except Exception as e:
                        st.error(f"⚠️ An error occurred: {str(e)}")

        with st.sidebar:
            st.markdown("---")
            st.markdown("💊 Skin Diagnosis Controls")
            st.markdown("### Filters")
            diagnosis_type = st.selectbox(
                "Diagnosis Type",
                ["All", "Skin Cancer", "Infections", "Inflammatory Conditions"]
            )

            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0,
                max_value=100,
                value=50,
                help="Filter results based on confidence score"
            )

            st.markdown("---")
            st.markdown("### Quick Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-number">{cond}</div>
                        <div style="color: black;">Confidence</div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-number" >{diag}</div>
                        <div style="color: black;">Diagnoses</div>
                    </div>
                """, unsafe_allow_html=True)

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Support")
        st.sidebar.markdown("📧 Dr. MadhuKant Patel")
        st.sidebar.markdown("📞 +91 99250 27534")



        # Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p>© 2025 REVE AI | Prototype/POC </p>
    <small>Version 1.0</small>
</div>
""", unsafe_allow_html=True)
