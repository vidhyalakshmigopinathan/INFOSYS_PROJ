# ============================================
# SkillGapAI - Milestone 4: Dashboard & Report Export
# Weeks 7–8
# ============================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF

st.set_page_config(page_title="SkillGapAI - Dashboard", layout="wide")

# -------------------------------------------
# HEADER
# -------------------------------------------
st.markdown("""
<h2 style='color:white; background-color:#0074D9; padding:14px; border-radius:10px'>
📊 SkillGapAI - Milestone 4: Dashboard & Report Export (Weeks 7–8)
</h2>
<p>Final comparison dashboard with interactive charts and downloadable reports.</p>
""", unsafe_allow_html=True)


# ------------------------------------------------
# Dummy data (Replace with Milestone 3 outputs later)
# ------------------------------------------------
resume_skills = {
    "Python": 92,
    "Machine Learning": 88,
    "TensorFlow": 85,
    "SQL": 65,
    "Statistics": 89,
    "Communication": 70,
    "AWS": 30,
    "Project Management": 40
}

job_requirements = {
    "Python": 95,
    "Machine Learning": 90,
    "TensorFlow": 88,
    "SQL": 75,
    "Statistics": 90,
    "Communication": 85,
    "AWS": 80,
    "Project Management": 75
}

skills_df = pd.DataFrame({
    "Skill": resume_skills.keys(),
    "Resume Score": resume_skills.values(),
    "Job Requirement Score": job_requirements.values()
})

matched = 6
missing = 4
overall_match = 72


# -------------------------------------------
# Skill Match Overview
# -------------------------------------------
st.subheader("📌 Skill Match Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Overall Match", f"{overall_match}%")
col2.metric("Matched Skills", matched)
col3.metric("Missing Skills", missing)
col4.metric("Total Skills", len(resume_skills))


# -------------------------------------------
# Bar Chart – Resume vs Job Requirements
# -------------------------------------------
st.markdown("### 📊 Resume vs Job Requirement Comparison")

fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(skills_df))
width = 0.35

ax.bar(x - width/2, skills_df["Resume Score"], width, label="Resume Skills")
ax.bar(x + width/2, skills_df["Job Requirement Score"], width, label="Job Requirements")

ax.set_xticks(x)
ax.set_xticklabels(skills_df["Skill"], rotation=45)
ax.set_ylabel("Match Percentage")
ax.legend()

st.pyplot(fig)


# -------------------------------------------
# Radar Chart – Role View
# -------------------------------------------
st.markdown("### 🎯 Role View Comparison")

labels = list(resume_skills.keys())[:5]
resume_values = [resume_skills[k] for k in labels]
job_values = [job_requirements[k] for k in labels]

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
resume_values += resume_values[:1]
job_values += job_values[:1]
angles += angles[:1]

fig = plt.figure(figsize=(5, 5))
ax = plt.subplot(111, polar=True)

ax.plot(angles, resume_values, linewidth=2, label="Current Profile")
ax.fill(angles, resume_values, alpha=0.25)

ax.plot(angles, job_values, linewidth=2, label="Job Requirements")
ax.fill(angles, job_values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_title("Role View Radar Chart")
ax.legend(loc="upper right")

st.pyplot(fig)


# -------------------------------------------
# Missing Skills Section
# -------------------------------------------
st.markdown("### 🚀 Upskilling Recommendations")

recommended = [
    ("AWS Cloud Services", "Complete AWS Solutions Architect Course"),
    ("Advanced Statistics", "Enroll in Data Science Statistics Program"),
    ("Project Management", "Consider PMP Certification")
]

for skill, advice in recommended:
    st.info(f"**{skill}** — {advice}")


# -------------------------------------------------------
# CSV Download
# -------------------------------------------------------
st.download_button(
    label="📥 Download Skill Report (CSV)",
    data=skills_df.to_csv(index=False),
    file_name="skillgap_report.csv",
    mime="text/csv"
)


# -------------------------------------------------------
# PDF Report Export
# -------------------------------------------------------
def generate_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="SkillGapAI - Skill Gap Report", ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Overall Match: {overall_match}%", ln=True)
    pdf.cell(200, 10, txt=f"Matched Skills: {matched}", ln=True)
    pdf.cell(200, 10, txt=f"Missing Skills: {missing}", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt="Skill Comparison:", ln=True)

    for i, row in skills_df.iterrows():
        pdf.cell(200, 10, txt=f"{row['Skill']} - Resume: {row['Resume Score']}%, Job: {row['Job Requirement Score']}%", ln=True)

    return pdf.output(dest='S').encode('latin1')


pdf_data = generate_pdf()

st.download_button(
    label="📄 Download Full Report (PDF)",
    data=pdf_data,
    file_name="skillgap_report.pdf",
    mime="application/pdf"
)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Milestone 4 • Dashboard & Reporting • SkillGapAI • Developed by Suriya Varshan</p>",
    unsafe_allow_html=True
)
