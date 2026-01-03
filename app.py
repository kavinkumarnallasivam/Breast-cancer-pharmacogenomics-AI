import pandas as pd
import numpy as np
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ==============================
# Generate Synthetic Dataset
# ==============================
np.random.seed(42)
n = 500

data = pd.DataFrame({
    'patient_id': [f'P{i:03d}' for i in range(1, n+1)],
    'age': np.random.randint(30, 75, n),
    'menopausal_status': np.random.choice(['pre', 'post'], n),
    'tumor_size_cm': np.round(np.random.normal(3.5, 1.0, n), 2),
    'lymph_node_status': np.random.choice([0, 1], n, p=[0.6, 0.4]),
    'histologic_grade': np.random.choice([1, 2, 3], n, p=[0.2, 0.5, 0.3]),
    'ER_status': np.random.choice([0, 1], n, p=[0.3, 0.7]),
    'PR_status': np.random.choice([0, 1], n, p=[0.4, 0.6]),
    'HER2_status': np.random.choice([0, 1], n, p=[0.7, 0.3]),
    'CYP2D6_genotype': np.random.choice(['*1/*1', '*1/*4', '*4/*4'], n),
    'TPMT_genotype': np.random.choice(['normal', 'intermediate', 'poor'], n),
    'DPYD_genotype': np.random.choice(['normal', 'reduced'], n),
    'drug_regimen': np.random.choice(['AC-T', 'FEC', 'CMF'], n),
    'dose_intensity': np.random.randint(70, 100, n),
    'num_cycles': np.random.randint(4, 8, n),
    'baseline_Hb': np.round(np.random.normal(12.5, 1.2, n), 1),
    'baseline_ALT': np.round(np.random.normal(25, 8, n), 1),
    'baseline_creatinine': np.round(np.random.normal(0.9, 0.2, n), 2),
    'adverse_effect_grade': np.random.choice([0,1,2,3,4], n)
})

data['response_status'] = np.where(
    (data['ER_status']==1) &
    (data['CYP2D6_genotype']=='*1/*1') &
    (data['dose_intensity']>85) &
    (data['TPMT_genotype']=='normal'), 1, 0
)

# ==============================
# Preprocessing
# ==============================
df = data.copy()
target = 'response_status'
X = df.drop(columns=['patient_id', target])
y = df[target]

encoders = {}
for col in X.select_dtypes('object').columns:
    enc = LabelEncoder()
    X[col] = enc.fit_transform(X[col])
    encoders[col] = enc

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# Train Model
# ==============================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ==============================
# Prediction Function
# ==============================
def predict_response(*vals):
    df_input = pd.DataFrame([dict(zip(X.columns, vals))])
    for col, enc in encoders.items():
        df_input[col] = enc.transform([df_input[col][0]])

    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]

    return (
        f"Predicted Response: {'Responder' if pred else 'Non-Responder'}\n"
        f"Probability of Response: {prob:.2f}"
    )

# ==============================
# Gradio UI
# ==============================
inputs = [
    gr.Number(label="Age"),
    gr.Radio(["pre","post"], label="Menopausal Status"),
    gr.Number(label="Tumor Size (cm)"),
    gr.Radio([0,1], label="Lymph Node Status"),
    gr.Slider(1,3,step=1,label="Histologic Grade"),
    gr.Radio([0,1], label="ER Status"),
    gr.Radio([0,1], label="PR Status"),
    gr.Radio([0,1], label="HER2 Status"),
    gr.Dropdown(["*1/*1","*1/*4","*4/*4"], label="CYP2D6"),
    gr.Dropdown(["normal","intermediate","poor"], label="TPMT"),
    gr.Dropdown(["normal","reduced"], label="DPYD"),
    gr.Dropdown(["AC-T","FEC","CMF"], label="Drug Regimen"),
    gr.Slider(70,100,label="Dose Intensity"),
    gr.Slider(4,8,label="Number of Cycles"),
    gr.Number(label="Hemoglobin"),
    gr.Number(label="ALT"),
    gr.Number(label="Creatinine"),
    gr.Slider(0,4,label="Adverse Effect Grade")
]

demo = gr.Interface(
    fn=predict_response,
    inputs=inputs,
    outputs="textbox",
    title="AI-Driven Pharmacogenomics Model for Breast Cancer",
    description="Predict chemotherapy response using clinical and pharmacogenomic features."
)

if __name__ == "__main__":
    demo.launch()
