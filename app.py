import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Diabetes AI Report", page_icon="🩺", layout="wide")

# --- ปรับ CSS ให้ปุ่มใน Sidebar เป็นการ์ดมนๆ สวยๆ แบบในรูปเพื่อน ---
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    [data-testid="stSidebar"] {
        background-color: #1e1b4b; /* โทนสีม่วงเข้มหรูหราแบบในภาพ */
    }
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h2 {
        color: #f8fafc !important;
    }
    /* ปรับแต่งปุ่มใน Sidebar ให้เป็นทรงโค้งมนเหมือนการ์ด */
    [data-testid="stSidebar"] .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.2em;
        background-color: #ffffff;
        color: #1e1b4b !important;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: 0.3s;
        text-align: left;
        margin-bottom: 8px;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #e2e8f0;
        color: #0f172a !important;
    }
    .card {
        background-color: #1e293b;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        border: 1px solid #334155;
        color: #f8fafc !important;
    }
    .card p, .card li, .card b {
        color: #f8fafc !important;
    }
    .stButton>button {
        border-radius: 8px;
        height: 3em;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        font-weight: bold;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# โหลดข้อมูล
@st.cache_data
def load_data():
    names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv('diabetes.csv', names=names)
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols] = df[cols].replace(0, np.nan)
    df.fillna(df.mean(), inplace=True)
    return df

df = load_data()
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Regression (Logistic)": LogisticRegression(max_iter=500),
    "SVM": SVC(kernel='linear', probability=True),
    "Ensemble (Random Forest)": RandomForestClassifier(random_state=42)
}

# --- จัดการสถานะหน้า (Session State) สำหรับปุ่มเมนู ---
if 'menu' not in st.session_state:
    st.session_state.menu = "หน้าหลัก"

# --- Sidebar แบบปุ่มกดการ์ด ---
with st.sidebar:
    # รูปโปรไฟล์ (ใช้อิโมจิหรือลิงก์รูปก็ได้ครับ ตรงนี้ผมทำวงกลมจำลองสไตล์เพื่อนให้)
    st.markdown("""
        <div style="text-align: center; padding-bottom: 10px;">
            <div style="width: 100px; height: 100px; border-radius: 50%; background-color: #38bdf8; display: inline-flex; align-items: center; justify-content: center; font-size: 40px; margin-bottom: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
                👨‍💻
            </div>
            <h3 style="margin: 0; color: white;">นายคณิศร จันทรสูตร</h3>
            <p style="color: #94a3b8; font-size: 14px; margin: 5px 0;">Computer Science Student</p>
            <p style="color: #cbd5e1; font-size: 13px;">รหัส 664245019 • หมู่ 66/43</p>
        </div>
        <hr style="border-color: #334155;">
        <h4 style="color: #38bdf8; font-size: 16px; margin-bottom: 10px;">📌 เมนูโครงงาน</h4>
    """, unsafe_allow_html=True)
    
    # สร้างปุ่มเมนูเรียงลงมาแทน Radio
    if st.button("🏠 ภาพรวมโครงงาน", use_container_width=True):
        st.session_state.menu = "หน้าหลัก"
    if st.button("📊 ปัญหาและ Dataset", use_container_width=True):
        st.session_state.menu = "1. ปัญหาและ Dataset"
    if st.button("🧹 Data Preprocessing", use_container_width=True):
        st.session_state.menu = "2. Data Preprocessing"
    if st.button("🧠 ทฤษฎีของโมเดล", use_container_width=True):
        st.session_state.menu = "3. สร้างโมเดล ML"
    if st.button("📈 ประเมินและเปรียบเทียบ", use_container_width=True):
        st.session_state.menu = "4. ประเมินโมเดล"
    if st.button("🔮 ทดลองทำนาย", use_container_width=True):
        st.session_state.menu = "5. เว็บแอปใช้งาน"

menu = st.session_state.menu

# --- หน้า Content ---
if menu == "หน้าหลัก":
    st.title("🩺 Diabetes Prediction & AI Analytics")
    st.markdown("#### ระบบวิเคราะห์และทำนายโรคเบาหวานด้วย Machine Learning สมัยใหม่")
    st.write("ยินดีต้อนรับสู่รายงานสรุปโปรเจ็คเชิงวิเคราะห์ ออกแบบด้วยธีม Modern Dashboard ทันสมัยและอ่านง่าย")
    st.image("https://images.unsplash.com/photo-1576091160550-2173dba999ef?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80", use_container_width=True)

elif menu == "1. ปัญหาและ Dataset":
    st.header("1. การกำหนดปัญหาและ Dataset")
    st.markdown("""
    <div class='card'>
    <b>📌 ปัญหาที่พบ:</b> โรคเบาหวานเป็นภัยเงียบระดับโลก การตรวจคัดกรองเบื้องต้นด้วยระบบอัจฉริยะช่วยให้ผู้ป่วยเข้ารับการรักษาได้ทันท่วงที<br><br>
    <b>📊 ข้อมูลที่ใช้ (Dataset):</b> Pima Indians Diabetes Database
    <ul>
        <li><b>จำนวนตัวอย่างทั้งหมด:</b> 768 รายการ</li>
        <li><b>ตัวแปรต้น (Features):</b> 8 ปัจจัยด้านสุขภาพ (เช่น ระดับน้ำตาล, BMI, ความดันโลหิต)</li>
        <li><b>ตัวแปรเป้าหมาย (Target):</b> Outcome (0 = ปกติ, 1 = เสี่ยงเป็นเบาหวาน)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif menu == "2. Data Preprocessing":
    st.header("2. Data Preprocessing")
    st.write("ขั้นตอนการเตรียมและทำความสะอาดข้อมูลดิบก่อนนำไปฝึกสอนโมเดล:")
    st.table(pd.DataFrame({
        "ขั้นตอนสำคัญ": ["จัดการ Missing Values", "จัดการค่า 0 ที่ผิดปกติ", "Feature Scaling"],
        "วิธีการทางสถิติ": ["ใช้การเติมค่าเฉลี่ย (Mean Imputation)", "แปลงค่า 0 ในคอลัมน์สุขภาพเป็น NaN", "ปรับสเกลมาตรฐานด้วย StandardScaler"],
        "ประโยชน์": ["ป้องกันข้อมูลสูญหาย", "ขจัดข้อมูลขยะทางการแพทย์", "ลดความเหลื่อมล้ำของช่วงตัวเลข"]
    }))
    st.markdown("#### 🔍 ตัวอย่างข้อมูลหลังผ่านกระบวนการ Preprocessing:")
    st.dataframe(df.head(10), use_container_width=True)

elif menu == "3. สร้างโมเดล ML":
    st.header("3. อัลกอริทึม Machine Learning ที่ศึกษา")
    st.markdown("""
    <div class='card'>
    โปรเจ็คนี้ได้รวบรวมอัลกอริทึมตามบทเรียนทั้งหมด 5 โมเดลหลัก:
    <ol>
        <li><b>KNN (K-Nearest Neighbors):</b> จำแนกกลุ่มตามความใกล้เคียงของข้อมูลรอบข้าง</li>
        <li><b>Decision Tree:</b> วิเคราะห์เงื่อนไขแยกกลุ่มข้อมูลในรูปแบบโครงสร้างต้นไม้</li>
        <li><b>Regression (Logistic Regression):</b> โมเดลรีเกรสชันสำหรับงานจำแนกประเภท (Binary Classification)</li>
        <li><b>SVM (Support Vector Machine):</b> ค้นหาเส้นแบ่ง Hyperplane ที่สร้างระยะ Margin กว้างที่สุด</li>
        <li><b>Ensemble (Random Forest):</b> ผสานพลังต้นไม้หลายต้นเพื่อเพิ่มความแม่นยำสูงสุด</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

elif menu == "4. ประเมินโมเดล":
    st.header("4. การประเมินและเปรียบเทียบโมเดล")
    st.write("ตารางเปรียบเทียบค่าความแม่นยำ (Accuracy) ของแต่ละอัลกอริทึม:")
    
    results = []
    for name, m in models.items():
        m.fit(X_train_scaled, y_train)
        score = accuracy_score(y_test, m.predict(X_test_scaled))
        results.append({"Model": name, "Accuracy (%)": round(score * 100, 2)})
    
    res_df = pd.DataFrame(results)
    st.table(res_df)
    
    st.write("📈 **Confusion Matrix ของโมเดลหลัก (SVM):**")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(pd.crosstab(y_test, models["SVM"].predict(X_test_scaled)), annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig)

elif menu == "5. เว็บแอปใช้งาน":
    st.header("5. Streamlit Application (ระบบทำนายผลอัจฉริยะ)")
    st.write("เลือกโมเดลที่ต้องการใช้งาน (หรือเลือก **รันทุกโมเดลพร้อมกัน**) และกรอกข้อมูลสุขภาพด้านล่างนี้:")
    
    model_options = ["KNN", "Decision Tree", "Regression (Logistic)", "SVM", "Ensemble (Random Forest)", "🚀 รันทุกโมเดลพร้อมกัน"]
    model_choice = st.radio("เลือกโหมดการทำนาย:", model_options, horizontal=True)
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            p = st.number_input('ตั้งครรภ์ (Pregnancies)', 0, 20, 1)
            g = st.number_input('ระดับน้ำตาล (Glucose)', 0, 200, 110)
            bp = st.number_input('ความดันโลหิต (BloodPressure)', 0, 140, 70)
            skin = st.number_input('ความหนาผิวหนัง (SkinThickness)', 0, 100, 20)
        with col2:
            ins = st.number_input('ระดับอินซูลิน (Insulin)', 0, 900, 79)
            bmi = st.number_input('ดัชนีมวลกาย (BMI)', 0.0, 70.0, 25.0)
            dpf = st.number_input('ประวัติครอบครัว (DPF)', 0.0, 2.5, 0.5)
            age = st.number_input('อายุ (Age)', 1, 100, 30)
            
        submit = st.form_submit_button(f"🚀 ประเมินความเสี่ยง ({model_choice})", use_container_width=True)
        
    if submit:
        input_data = pd.DataFrame([[p, g, bp, skin, ins, bmi, dpf, age]], columns=X.columns)
        scaled_input = scaler.transform(input_data)
        
        st.markdown("---")
        
        if model_choice == "🚀 รันทุกโมเดลพร้อมกัน":
            st.subheader("📊 ผลการทำนายและระดับความมั่นใจจากทุกโมเดล:")
            compare_results = []
            for name, m in models.items():
                m.fit(X_train_scaled, y_train)
                pred = m.predict(scaled_input)[0]
                prob = m.predict_proba(scaled_input)[0]
                confidence = prob[pred] * 100
                
                status = "🚨 พบความเสี่ยง" if pred == 1 else "✅ ปกติ"
                compare_results.append({
                    "Model": name, 
                    "Prediction": status, 
                    "Confidence (%)": f"{confidence:.2f}%"
                })
            
            res_table = pd.DataFrame(compare_results)
            st.table(res_table)
            st.info("💡 สังเกตว่าโมเดลแต่ละตัวจะมีเปอร์เซ็นต์ความมั่นใจ (Confidence Score) ต่างกันตามตรรกะการประมวลผลของอัลกอริทึมนั้นๆ")
        
        else:
            active_model = models[model_choice]
            active_model.fit(X_train_scaled, y_train)
            prediction = active_model.predict(scaled_input)[0]
            prob = active_model.predict_proba(scaled_input)[0]
            confidence = prob[prediction] * 100
            
            if prediction == 1:
                st.error(f"🚨 **ผลการทำนายด้วย {model_choice}:** พบความเสี่ยงเป็นโรคเบาหวาน (ระดับความมั่นใจ: **{confidence:.2f}%**)")
            else:
                st.success(f"✅ **ผลการทำนายด้วย {model_choice}:** ไม่พบความเสี่ยง / สุขภาพปกติ (ระดับความมั่นใจ: **{confidence:.2f}%**)")