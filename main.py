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
from agno.tools.reasoning import ReasoningTools
from agno.tools.googlesearch import GoogleSearchTools
import mymodel_utils

from agno.media import Image as AgnoImage
from fastapi import UploadFile, File
from typing import Optional
from typing import Dict, Optional, List

import requests
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="REVE-AI",
    description="API SERVICES FOR AL AI/ML MODELS.",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialization ---
initialization_successful = mymodel_utils.initialize_application()
if not initialization_successful:
    print("WARNING: Application failed to initialize properly.")

# --- Pydantic Models ---

@app.get("/")
def read_root():
    return {"message": "Welcome to REVE-AI"}

    
@app.post("/predict")
async def classify_image(
    file: UploadFile = File(...),
    patient_id: int = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    notes: Optional[str] = Form(None)
):
    try:
        print(patient_id,age,gender,notes)
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
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
        data = {"PatientId": patient_id, "Verify": result.content, "Prediction": pred.content,"Report":report.content,"Jarvis": jarvis.content}
        #store in mariaDB
        # url = "http://192.168.1.10/ReveDataAPI/api/SkinOutput/insert"
        url="http://192.168.1.10/ReveDataAPI/api/SkinOutput/insert"
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()  # Raise an exception for 4xx/5xx responses
            print("Status Code:", response.status_code)
            print("Response:", response.json())

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to the server.")
        except requests.exceptions.Timeout:
            print("Error: Request timed out.")
        except requests.exceptions.RequestException as err:
            print(f"An error occurred: {err}")
            

        return data
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")

class AnalyzeRequest(BaseModel):
    waterLevel: int = Field(..., description="Water level in ml")
    wavelengths: Dict[str, float] = Field(..., description="Spectral wavelength values")

class AnalyzeResponse(BaseModel):
    Prediction_Status: str
    Input_Water_Level: int
    Provided_Features: List[str]
    Imputed_Features: List[str]
    pH: Optional[float]
    nitro: Optional[float]
    phosphorus: Optional[float]
    potassium: Optional[float]
    capacityMoist: Optional[float]
    temperature: Optional[float]
    moisture: Optional[float]
    electricalConductivity: Optional[float]

@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_soil(data: AnalyzeRequest):
    # if not mymodel_utils.get_status():
    #     raise HTTPException(status_code=503, detail="Service not ready, initialization failed.")

    water_level = data.waterLevel
    wavelengths = data.wavelengths

    if water_level not in mymodel_utils.WATER_LEVELS_TO_PROCESS:
        raise HTTPException(status_code=400, detail=f"'waterLevel' must be one of {mymodel_utils.WATER_LEVELS_TO_PROCESS}")

    valid_keys = set(mymodel_utils.SPECTRAL_COLS)
    provided_keys = set(wavelengths.keys())
    invalid_keys = provided_keys - valid_keys
    if invalid_keys:
        raise HTTPException(status_code=400, detail=f"Invalid spectral keys: {list(invalid_keys)}. Valid keys: {mymodel_utils.SPECTRAL_COLS}")

    if not (2 <= len(provided_keys) <= len(mymodel_utils.SPECTRAL_COLS)):
        raise HTTPException(status_code=400, detail=f"Must provide 2–{len(mymodel_utils.SPECTRAL_COLS)} wavelengths. Provided: {len(provided_keys)}")

    try:
        processed_wavelengths = {k: float(v) for k, v in wavelengths.items()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid numeric value in wavelengths: {str(e)}")

    status_info, predictions = mymodel_utils.run_prediction(processed_wavelengths, water_level)
    response_data = {**status_info, **predictions}

    frontend_key_map = {
        'Ph': 'pH',
        'Nitro': 'nitro',
        'Posh Nitro': 'phosphorus',
        'Pota Nitro': 'potassium',
        'Capacitity Moist': 'capacityMoist',
        'Temp': 'temperature',
        'Moist': 'moisture',
        'EC': 'electricalConductivity'
    }

    formatted = {
        'Prediction_Status': response_data.get('Prediction_Status', 'Unknown Error'),
        'Input_Water_Level': response_data.get('Input_Water_Level', water_level),
        'Provided_Features': response_data.get('Provided_Features', list(processed_wavelengths.keys())),
        'Imputed_Features': response_data.get('Imputed_Features', [])
    }

    for model_key, frontend_key in frontend_key_map.items():
        val = response_data.get(model_key)
        formatted[frontend_key] = None if (val is None or (isinstance(val, float) and np.isnan(val))) else val

    if "Error" in formatted["Prediction_Status"] or "Failed" in formatted["Prediction_Status"]:
        raise HTTPException(status_code=500, detail=formatted)
    
    return formatted

"""
example input request body for analyze endpoint:

{
  "waterLevel": 0,
  "wavelengths": {
    "410": 720.93,
    "435": 147.35,
    "460": 366.269,
    "485": 93.037,
    "510": 176.81,
    "535": 158.097,
    "560": 27.952,
    "585": 32.928,
    "610": 137.657,
    "645": 47.346,
    "680": 90.863,
    "705": 8.622,
    "730": 16.666,
    "760": 15.699,
    "810": 34.915,
    "860": 112.237,
    "900": 14.632,
    "940": 6.464
  }
}

"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run('main:app', host="127.0.0.1", port=6701, reload=True)