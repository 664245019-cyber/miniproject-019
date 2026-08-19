# --- หน้า Content ที่เพิ่มเนื้อหาให้อัดแน่น ---
if menu == "หน้าหลัก":
    st.title("🩺 Diabetes Prediction Project")
    st.markdown("### รายงานสรุปผลโปรเจ็ค Machine Learning เพื่อการคัดกรองเบาหวาน")
    st.write("โปรเจ็คนี้พัฒนาขึ้นเพื่อประยุกต์ใช้โมเดล SVM ในการทำนายความเสี่ยงโรคเบาหวานโดยใช้ข้อมูลสุขภาพที่เป็นมาตรฐานสากล")
    st.image("https://images.unsplash.com/photo-1576091160550-2173dba999ef?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80", use_container_width=True)

elif menu == "1. ปัญหาและ Dataset":
    st.header("1. การกำหนดปัญหาและ Dataset")
    st.markdown("""
    **ปัญหาที่พบ:** โรคเบาหวานเป็นภัยเงียบที่ไม่แสดงอาการในช่วงแรก การตรวจพบเร็วช่วยลดค่าใช้จ่ายในการรักษา
    **Dataset ที่เลือกใช้:** Pima Indians Diabetes Dataset
    * **จำนวนตัวอย่าง:** 768 รายการ
    * **ตัวแปรต้น (Features):** 8 ตัวแปรสุขภาพ ได้แก่ Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    * **ตัวแปรตาม (Target):** Outcome (0 = ปกติ, 1 = เป็นเบาหวาน)
    **เหตุผล:** ข้อมูลชุดนี้มีความสมบูรณ์ทางสถิติและเป็นที่ยอมรับในการเรียนรู้ Machine Learning ทั่วโลก
    """)

elif menu == "2. Data Preprocessing":
    st.header("2. Data Preprocessing")
    st.write("ข้อมูลดิบผ่านกระบวนการทำความสะอาดอย่างเป็นระบบ:")
    st.table(pd.DataFrame({
        "ขั้นตอน": ["จัดการ Missing Values", "จัดการค่า 0 ผิดปกติ", "Feature Scaling"],
        "วิธีการ": ["ใช้การเติมค่าเฉลี่ย (Mean Imputation)", "เปลี่ยนค่า 0 ในคอลัมน์สุขภาพเป็น NaN", "ใช้ StandardScaler ปรับสเกลข้อมูลให้เป็นมาตรฐาน"],
        "ผลลัพธ์": ["ข้อมูลไม่สูญหาย", "ข้อมูลสะอาดขึ้น", "ลดความเหลื่อมล้ำของข้อมูล"]
    }))
    st.write("ตัวอย่างข้อมูลที่ผ่านการทำ Preprocessing แล้ว:")
    st.dataframe(df.head(10), use_container_width=True)

elif menu == "3. สร้างโมเดล ML":
    st.header("3. การสร้างโมเดล Machine Learning")
    st.markdown("""
    **อัลกอริทึม:** Support Vector Machine (SVM)
    * **ทำไมต้อง SVM:** เพราะมีประสิทธิภาพสูงในการแยกกลุ่มข้อมูล (Classification) ที่มีความซับซ้อนและข้อมูลไม่เป็นเส้นตรง
    * **Linear Kernel:** เราเลือกใช้ Linear Kernel เนื่องจากความสัมพันธ์ของตัวแปรมีความเป็นเส้นตรงที่ค่อนข้างชัดเจน
    * **Hyperplane:** หลักการคือการหาเส้นแบ่งที่ดีที่สุดระหว่างกลุ่ม Positive และ Negative โดยการรักษาระยะห่าง (Margin) ให้มากที่สุด
    """)

elif menu == "4. ประเมินโมเดล":
    st.header("4. การประเมินและเปรียบเทียบโมเดล")
    acc = accuracy_score(y, model.predict(scaler.transform(X)))
    st.metric("Accuracy ของโมเดล SVM", f"{acc*100:.2f}%")
    st.write("Confusion Matrix เพื่อดูการทายถูก/ผิด:")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pd.crosstab(y, model.predict(scaler.transform(X))), annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
    st.pyplot(fig)
    st.write("จากผลลัพธ์พบว่าโมเดลมีความแม่นยำสูง สามารถใช้เป็นเครื่องมือคัดกรองเบื้องต้นได้จริง")

elif menu == "5. เว็บแอปใช้งาน":
    # ... (ส่วนเดิมของข้อ 5)