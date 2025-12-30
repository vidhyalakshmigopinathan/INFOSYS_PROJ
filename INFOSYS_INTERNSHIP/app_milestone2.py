# ==========================================
# SkillGapAI - Milestone 2: Skill Extraction using NLP
# Weeks 3–4
# ==========================================

import streamlit as st
import spacy
import re
import matplotlib.pyplot as plt
from io import BytesIO

# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------
st.set_page_config(page_title="SkillGapAI - Milestone 2", layout="wide")

st.markdown(
    """
    <h2 style='color:white; background-color:#117A65; padding:15px; border-radius:10px'>
    🧠 SkillGapAI - Milestone 2: Skill Extraction using NLP
    </h2>
    <p><b>Objective:</b> Extract and classify technical & soft skills separately 
    from both Resume and Job Description using spaCy-based NLP pipelines. 
    Display structured tags, wanted job skills, and skill distribution charts.</p>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------
# LOAD SPACY MODEL
# ------------------------------------------
@st.cache_resource
def load_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_model()

# ------------------------------------------
# SKILL LISTS
# ------------------------------------------
technical_skills = [
    "python", "java", "c++", "sql", "html", "css", "javascript", "react", "node.js",
    "tensorflow", "pytorch", "machine learning", "data analysis", "data visualization",
    "aws", "azure", "gcp", "power bi", "tableau", "django", "flask", "scikit-learn", "nlp"
]

soft_skills = [
    "communication", "leadership", "teamwork", "problem solving", "time management",
    "adaptability", "critical thinking", "creativity", "collaboration", "decision making"
]

# ------------------------------------------
# CLEAN & EXTRACT FUNCTIONS
# ------------------------------------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

def extract_skills(text):
    text = clean_text(text)
    found_tech = [skill.title() for skill in technical_skills if skill in text]
    found_soft = [skill.title() for skill in soft_skills if skill in text]
    return list(set(found_tech)), list(set(found_soft))

# ------------------------------------------
# LAYOUT: SIDE-BY-SIDE INPUTS
# ------------------------------------------
col_resume, col_jd = st.columns(2)

with col_resume:
    st.markdown("### 👨‍💻 Resume Text")
    resume_text = st.text_area("Paste Resume Content Here:", "", height=250)

with col_jd:
    st.markdown("### 🏢 Job Description Text")
    jd_text = st.text_area("Paste Job Description Content Here:", "", height=250)

# ------------------------------------------
# PROCESS BOTH INPUTS
# ------------------------------------------
if resume_text or jd_text:
    st.markdown("---")
    st.markdown("## 🔍 Skill Extraction Results")

    # Resume extraction
    if resume_text:
        tech_resume, soft_resume = extract_skills(resume_text)
        total_resume = len(tech_resume) + len(soft_resume)

        st.markdown("### 📄 Resume Skill Extraction")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ⚙️ Technical Skills (Candidate Possesses)")
            st.write(", ".join(tech_resume) if tech_resume else "None found.")
        with col2:
            st.markdown("#### 💬 Soft Skills (Candidate Possesses)")
            st.write(", ".join(soft_resume) if soft_resume else "None found.")

        # Resume Skill Chart
        fig, ax = plt.subplots(figsize=(3, 3))
        labels = ["Technical", "Soft"]
        sizes = [len(tech_resume), len(soft_resume)]
        colors = ["#1F77B4", "#2ECC71"]
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
        ax.axis("equal")
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)
        st.caption(f"Total Resume Skills: {total_resume}")

    # Job Description extraction (Wanted Skills)
    if jd_text:
        tech_jd, soft_jd = extract_skills(jd_text)
        total_jd = len(tech_jd) + len(soft_jd)

        st.markdown("### 🏢 Wanted Skills for Job Description")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### ⚙️ Required Technical Skills (From Job Post)")
            st.write(", ".join(tech_jd) if tech_jd else "No technical skills mentioned.")
        with col4:
            st.markdown("#### 💬 Required Soft Skills (From Job Post)")
            st.write(", ".join(soft_jd) if soft_jd else "No soft skills mentioned.")

        # JD Skill Chart
        fig2, ax2 = plt.subplots(figsize=(3, 3))
        labels2 = ["Technical", "Soft"]
        sizes2 = [len(tech_jd), len(soft_jd)]
        colors2 = ["#1F77B4", "#2ECC71"]
        ax2.pie(sizes2, labels=labels2, autopct="%1.1f%%", colors=colors2, startangle=90)
        ax2.axis("equal")
        buf2 = BytesIO()
        fig2.savefig(buf2, format="png")
        st.image(buf2)
        st.caption(f"Total Required Skills: {total_jd}")

else:
    st.info("Please paste resume and/or job description text to extract skills.")

# ------------------------------------------
# FOOTER
# ------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Milestone 2 • Skill Extraction using NLP • SkillGapAI Project • Developed by Suriya Varshan</p>",
    unsafe_allow_html=True
)
