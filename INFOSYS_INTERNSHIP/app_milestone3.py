# ==========================================
# SkillGapAI - Milestone 3: Skill Gap Analysis & Similarity Matching
# Weeks 5–6
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util

# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------
st.set_page_config(page_title="SkillGapAI - Milestone 3", layout="wide")

st.markdown(
    """
    <h2 style='color:white; background-color:#5B2C6F; padding:15px; border-radius:10px'>
    🧠 SkillGapAI - Milestone 3: Skill Gap Analysis & Similarity Matching
    </h2>
    <p><b>Objective:</b> Compare candidate and job skills using BERT embeddings to find matched, partial, 
    and missing skills. Display a skill similarity matrix and summary dashboard.</p>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------
# LOAD MODEL
# ------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ------------------------------------------
# SKILL EXTRACTION INPUTS
# ------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👨‍💻 Resume Skills")
    resume_skills_input = st.text_area("Enter or paste extracted resume skills (comma-separated):", "Python, SQL, Machine Learning, Tableau")

with col2:
    st.markdown("### 🏢 Job Description Skills")
    jd_skills_input = st.text_area("Enter or paste required job skills (comma-separated):", "Python, Data Visualization, Deep Learning, Communication, AWS")

resume_skills = [s.strip() for s in resume_skills_input.split(",") if s.strip()]
jd_skills = [s.strip() for s in jd_skills_input.split(",") if s.strip()]

# ------------------------------------------
# SIMILARITY COMPUTATION
# ------------------------------------------
if resume_skills and jd_skills:
    st.markdown("---")
    st.markdown("## 🔍 Skill Gap Analysis")

    resume_embeddings = model.encode(resume_skills, convert_to_tensor=True)
    jd_embeddings = model.encode(jd_skills, convert_to_tensor=True)

    similarity_matrix = util.cos_sim(resume_embeddings, jd_embeddings).cpu().numpy()

    # Convert to DataFrame for heatmap
    sim_df = pd.DataFrame(similarity_matrix, index=resume_skills, columns=jd_skills)

    # --------------------------------------
    # Skill Classification
    # --------------------------------------
    matched_skills = []
    partial_skills = []
    missing_skills = []

    for j_skill in jd_skills:
        max_sim = sim_df[j_skill].max()
        if max_sim >= 0.8:
            matched_skills.append(j_skill)
        elif 0.5 <= max_sim < 0.8:
            partial_skills.append(j_skill)
        else:
            missing_skills.append(j_skill)

    total_skills = len(jd_skills)
    overall_match = round(((len(matched_skills) + 0.5 * len(partial_skills)) / total_skills) * 100, 2)

    # --------------------------------------
    # VISUALIZATIONS
    # --------------------------------------
    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown("### 📈 Skill Similarity Matrix (BERT-based Cosine Similarity)")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(sim_df, annot=True, cmap="YlGnBu", cbar=True, fmt=".2f")
        plt.title("Skill Similarity Heatmap")
        st.pyplot(fig)

    with c2:
        st.markdown("### 📊 Skill Match Overview")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("✅ Matched Skills", len(matched_skills))
        col_b.metric("🟡 Partial Matches", len(partial_skills))
        col_c.metric("❌ Missing Skills", len(missing_skills))
        st.metric("📊 Overall Match", f"{overall_match}%")

        # Pie Chart
        labels = ["Matched", "Partial", "Missing"]
        sizes = [len(matched_skills), len(partial_skills), len(missing_skills)]
        colors = ["#2ECC71", "#F1C40F", "#E74C3C"]
        fig2, ax2 = plt.subplots(figsize=(3, 3))
        ax2.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
        ax2.axis("equal")
        st.pyplot(fig2)

    # --------------------------------------
    # MISSING SKILLS LIST
    # --------------------------------------
    st.markdown("---")
    st.markdown("### ❌ Missing Skills from Resume (Needed for Job)")
    if missing_skills:
        for skill in missing_skills:
            st.error(f"🚫 {skill}")
    else:
        st.success("🎯 No missing skills! Great match.")

    # --------------------------------------
    # MATCHED + PARTIAL DETAILS
    # --------------------------------------
    st.markdown("---")
    st.subheader("📋 Detailed Skill Comparison")
    comp_data = []
    for jd_skill in jd_skills:
        max_sim = sim_df[jd_skill].max()
        best_resume_skill = sim_df[jd_skill].idxmax()
        comp_data.append({
            "Job Skill": jd_skill,
            "Closest Resume Skill": best_resume_skill,
            "Similarity Score": round(max_sim * 100, 2)
        })

    st.dataframe(pd.DataFrame(comp_data))

else:
    st.info("Please enter both Resume and Job Description skills to start the analysis.")

# ------------------------------------------
# FOOTER
# ------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Milestone 3 • Skill Gap Analysis & Similarity Matching • SkillGapAI Project • Developed by Suriya Varshan</p>",
    unsafe_allow_html=True
)
