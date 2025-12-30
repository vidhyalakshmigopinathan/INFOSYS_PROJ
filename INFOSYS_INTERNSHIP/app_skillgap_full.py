# app_skillgap_full.py
# SkillGapAI — Full app: Milestone 1-4 combined with DB, NLP, embeddings, similarity, exports

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import re
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from sentence_transformers import SentenceTransformer, util
import spacy
import docx2txt
import PyPDF2
from datetime import datetime

# -----------------------
# CONFIG
# -----------------------
DB_PATH = "skillgap.db"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # lightweight & fast
SKILL_KEYWORDS = [
    # technical
    "python","pandas","numpy","scikit-learn","tensorflow","pytorch","keras","sql","nosql","hadoop",
    "spark","aws","azure","gcp","docker","kubernetes","ci/cd","jenkins","terraform","react","angular",
    "html","css","javascript","django","flask","rest api","api","microservices","spring boot","java",
    "c++","c#","ruby","tableau","power bi","nlp","computer vision","deep learning","data visualization",
    # soft
    "communication","leadership","teamwork","problem solving","time management","stakeholder management",
    "agile","scrum","kanban","project management","critical thinking"
]
SKILL_KEYWORDS = [s.lower() for s in SKILL_KEYWORDS]

# -----------------------
# UTIL: DB
# -----------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS resumes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    text TEXT,
                    created_at TEXT
                )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS job_descriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    text TEXT,
                    created_at TEXT
                )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_id INTEGER,
                    jd_id INTEGER,
                    matched_skills TEXT,
                    partial_skills TEXT,
                    missing_skills TEXT,
                    overall_match REAL,
                    created_at TEXT
                )""")
    conn.commit()
    conn.close()

def save_resume(filename, text):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO resumes (filename, text, created_at) VALUES (?, ?, ?)",
                (filename, text, datetime.utcnow().isoformat()))
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid

def save_jd(title, text):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO job_descriptions (title, text, created_at) VALUES (?, ?, ?)",
                (title, text, datetime.utcnow().isoformat()))
    conn.commit()
    jid = cur.lastrowid
    conn.close()
    return jid

def save_analysis(resume_id, jd_id, matched, partial, missing, overall):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""INSERT INTO analyses (resume_id, jd_id, matched_skills, partial_skills, missing_skills, overall_match, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (resume_id, jd_id, ",".join(matched), ",".join(partial), ",".join(missing), overall, datetime.utcnow().isoformat()))
    conn.commit()
    aid = cur.lastrowid
    conn.close()
    return aid

def load_resumes():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM resumes ORDER BY id DESC", conn)
    conn.close()
    return df

def load_jds():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM job_descriptions ORDER BY id DESC", conn)
    conn.close()
    return df

def load_analyses():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM analyses ORDER BY id DESC", conn)
    conn.close()
    return df

# -----------------------
# UTIL: Parsing
# -----------------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_file(uploaded_file):
    # uploaded_file: streamlit file-like
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            out = []
            for p in reader.pages:
                txt = p.extract_text()
                if txt:
                    out.append(txt)
            return clean_text("\n".join(out))
        except Exception:
            return ""
    elif name.endswith(".docx"):
        try:
            return clean_text(docx2txt.process(uploaded_file))
        except Exception:
            return ""
    elif name.endswith(".txt"):
        try:
            return clean_text(uploaded_file.read().decode("utf-8"))
        except Exception:
            return ""
    else:
        return ""

# -----------------------
# NLP: spaCy model & simple NER-ish extraction
# -----------------------
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

# Note: do not call load_spacy() at import time (avoids blocking Streamlit startup)

def ner_skill_extract(text):
    """Use spaCy noun chunks + keyword match to extract candidate skills"""
    # ensure spaCy model is loaded (cached)
    nlp = load_spacy()
    text_low = text.lower()
    found = set()
    # keyword matching (fast & reliable)
    for sk in SKILL_KEYWORDS:
        if re.search(r'\b' + re.escape(sk) + r'\b', text_low):
            found.add(sk)
    # noun-chunk based candidates (two-word phrases)
    doc = nlp(text)
    for nc in doc.noun_chunks:
        chunk = nc.text.strip().lower()
        if 2 <= len(chunk.split()) <= 3:
            # if chunk contains known technical word
            for kw in ["machine", "learning", "deep", "data", "visualization", "cloud", "tensorflow","pytorch","project"]:
                if kw in chunk:
                    found.add(chunk)
    # normalize: title case
    return sorted({s.title() for s in found})

# -----------------------
# Embeddings model
# -----------------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

# Note: do not call load_embed_model() at import time (load lazily inside compute_similarity)

def compute_similarity(resume_skills, jd_skills):
    if not resume_skills or not jd_skills:
        return None, None
    # ensure embedding model is loaded (cached)
    embed_model = load_embed_model()
    resume_emb = embed_model.encode(resume_skills, convert_to_tensor=True)
    jd_emb = embed_model.encode(jd_skills, convert_to_tensor=True)
    sim = util.cos_sim(resume_emb, jd_emb).cpu().numpy()  # shape (R, J)
    sim_df = pd.DataFrame(sim, index=resume_skills, columns=jd_skills)
    return sim_df, sim

def classify_skills(sim_df):
    matched, partial, missing = [], [], []
    for j in sim_df.columns:
        max_sim = sim_df[j].max()
        if max_sim >= 0.80:
            matched.append(j)
        elif max_sim >= 0.50:
            partial.append(j)
        else:
            missing.append(j)
    total = len(sim_df.columns)
    overall = round(((len(matched) + 0.5*len(partial)) / total) * 100,2) if total else 0.0
    return matched, partial, missing, overall

# -----------------------
# Upskilling recommender (simple rule-based)
# -----------------------
UPSKILL_LINKS = {
    "Aws": "https://aws.amazon.com/training/",
    "Azure": "https://learn.microsoft.com/en-us/training/azure/",
    "Gcp": "https://cloud.google.com/training",
    "Machine Learning": "https://www.coursera.org/learn/machine-learning",
    "Deep Learning": "https://www.deeplearning.ai/",
    "Sql": "https://www.udemy.com/topic/sql/",
    "Project Management": "https://www.pmi.org/certifications",
    "Data Visualization": "https://www.coursera.org/specializations/data-visualization"
}

def recommend_upskilling(missing_skills):
    recs = []
    for ms in missing_skills:
        key = ms.split()[0]  # simple match on first token
        link = UPSKILL_LINKS.get(key.title()) or None
        recs.append({"skill": ms, "course": link or "Search online courses for " + ms})
    return recs

# -----------------------
# PDF export helper
# -----------------------
def make_pdf_report(resume_text, jd_text, matched, partial, missing, overall):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, "SkillGapAI - Skill Gap Analysis Report", ln=True, align="C")
    pdf.ln(6)
    pdf.cell(0, 6, f"Overall Match Score: {overall} %", ln=True)
    pdf.ln(4)
    pdf.cell(0, 6, "Matched Skills: " + (", ".join(matched) if matched else "None"), ln=True)
    pdf.cell(0, 6, "Partial Matches: " + (", ".join(partial) if partial else "None"), ln=True)
    pdf.cell(0, 6, "Missing Skills: " + (", ".join(missing) if missing else "None"), ln=True)
    pdf.ln(8)
    pdf.cell(0, 6, "Resume (parsed):", ln=True)
    pdf.multi_cell(0, 6, (resume_text[:1200] + "...") if len(resume_text) > 1200 else resume_text)
    pdf.ln(4)
    pdf.cell(0, 6, "Job Description (parsed):", ln=True)
    pdf.multi_cell(0, 6, (jd_text[:1200] + "...") if len(jd_text) > 1200 else jd_text)
    return pdf.output(dest='S').encode('latin1')

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="SkillGapAI - Full", layout="wide")
st.title("SkillGapAI — Full Pipeline (Upload → NLP → Match → Dashboard → Export)")
init_db()

menu = st.sidebar.selectbox("Navigation", [
    "Upload & Parse",
    "Skill Extraction",
    "Skill Gap Analysis",
    "Dashboard & Exports",
    "Database"
])

# ---------------- Upload & Parse ----------------
if menu == "Upload & Parse":
    st.header("Milestone 1: Upload & Parse Resumes / Job Descriptions")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Upload Resume")
        uploaded_resume = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf","docx","txt"], key="res")
        if uploaded_resume:
            resume_text = extract_text_from_file(uploaded_resume)
            st.success("Parsed Resume Text")
            st.text_area("Parsed Resume", resume_text, height=300)
            if st.button("Save Resume to DB"):
                rid = save_resume(uploaded_resume.name, resume_text)
                st.success(f"Saved resume id: {rid}")
    with col2:
        st.subheader("Upload Job Description")
        uploaded_jd = st.file_uploader("Upload Job Description (PDF, DOCX, TXT)", type=["pdf","docx","txt"], key="jd")
        jd_title = st.text_input("Job Title / Role", value="Job Role")
        if uploaded_jd:
            jd_text = extract_text_from_file(uploaded_jd)
            st.success("Parsed Job Description Text")
            st.text_area("Parsed Job Description", jd_text, height=300)
            if st.button("Save Job Description to DB"):
                jid = save_jd(jd_title, jd_text)
                st.success(f"Saved JD id: {jid}")

    st.markdown("---")
    st.info("You may paste text directly in Skill Extraction tab for faster testing.")

# ---------------- Skill Extraction ----------------
elif menu == "Skill Extraction":
    st.header("Milestone 2: Skill Extraction (Resume & Job Description)")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Resume Text (paste or select from DB)")
        resumes = load_resumes()
        resume_choice = None
        if not resumes.empty:
            choice = st.selectbox("Select saved resume", options=["-- none --"] + [f"{r['id']}:{r['filename']}" for i,r in resumes.iterrows()])
            if choice and choice != "-- none --":
                rid = int(choice.split(":")[0])
                resume_choice = resumes[resumes['id']==rid].iloc[0]['text']
                st.write(f"Loaded resume id {rid}")
        resume_text = st.text_area("Resume Text", resume_choice or "", height=240)

    with col2:
        st.subheader("Job Description Text (paste or select from DB)")
        jds = load_jds()
        jd_choice = None
        if not jds.empty:
            choice2 = st.selectbox("Select saved JD", options=["-- none --"] + [f"{r['id']}:{r['title']}" for i,r in jds.iterrows()])
            if choice2 and choice2 != "-- none --":
                jid = int(choice2.split(":")[0])
                jd_choice = jds[jds['id']==jid].iloc[0]['text']
                st.write(f"Loaded JD id {jid}")
        jd_text = st.text_area("Job Description Text", jd_choice or "", height=240)

    if st.button("Extract Skills"):
        with st.spinner("Extracting skills using spaCy + keyword matching..."):
            resume_skills = ner_skill_extract(resume_text or "")
            jd_skills = ner_skill_extract(jd_text or "")
        st.success("Extraction complete")
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Candidate Skills")
            st.write(", ".join(resume_skills) if resume_skills else "None found")
        with colB:
            st.subheader("Job Required Skills (Wanted)")
            st.write(", ".join(jd_skills) if jd_skills else "None found")

# ---------------- Skill Gap Analysis ----------------
elif menu == "Skill Gap Analysis":
    st.header("Milestone 3: Skill Gap Analysis & Similarity Matching")
    st.info("Paste or choose resume and job skills to compute similarity (BERT-based).")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Resume Skills (comma-separated)")
        rtext = st.text_area("Resume skills", value="Python, SQL, Machine Learning")
        resume_skills = [s.strip() for s in rtext.split(",") if s.strip()]
    with col2:
        st.subheader("Job Skills (comma-separated)")
        jtext = st.text_area("Job skills", value="Python, Data Visualization, Deep Learning, Communication, AWS")
        jd_skills = [s.strip() for s in jtext.split(",") if s.strip()]

    if st.button("Compute Similarity"):
        if not resume_skills or not jd_skills:
            st.error("Provide both resume and job skills.")
        else:
            with st.spinner("Computing embeddings and similarity..."):
                sim_df, _ = compute_similarity(resume_skills, jd_skills)
                matched, partial, missing, overall = classify_skills(sim_df)
            st.success("Analysis complete")
            st.subheader("Similarity Matrix (heatmap)")
            fig, ax = plt.subplots(figsize=(7,4))
            sns.heatmap(sim_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
            st.pyplot(fig)
            st.markdown("### Summary")
            st.write(f"Matched: {matched}")
            st.write(f"Partial: {partial}")
            st.write(f"Missing: {missing}")
            st.metric("Overall Match %", f"{overall}%")
            # Save analysis optionally
            if st.button("Save Analysis to DB"):
                # Save minimal resume/jd entries if not present (we save raw text entries)
                rid = save_resume("inline_resume", " , ".join(resume_skills))
                jid = save_jd("inline_jd", " , ".join(jd_skills))
                aid = save_analysis(rid, jid, matched, partial, missing, overall)
                st.success(f"Saved analysis id {aid}")
            # Recommendations
            recs = recommend_upskilling(missing)
            st.markdown("### Upskilling Recommendations for Missing Skills")
            for r in recs:
                st.info(f"{r['skill']} — {r['course']}")

# ---------------- Dashboard & Exports ----------------
elif menu == "Dashboard & Exports":
    st.header("Milestone 4: Dashboard & Reporting")
    st.info("View recent analyses and download CSV/PDF reports")
    analyses = load_analyses()
    if analyses.empty:
        st.warning("No analyses saved yet. Run analysis first and save to DB.")
    else:
        st.dataframe(analyses)
        sel = st.selectbox("Select analysis id to view", options=analyses['id'].tolist())
        row = analyses[analyses['id'] == sel].iloc[0]
        st.subheader(f"Analysis ID {sel}")
        st.write("Matched:", row['matched_skills'].split(",") if row['matched_skills'] else [])
        st.write("Partial:", row['partial_skills'].split(",") if row['partial_skills'] else [])
        st.write("Missing:", row['missing_skills'].split(",") if row['missing_skills'] else [])
        st.metric("Overall Match %", f"{row['overall_match']}%")
        # Create CSV and PDF for this analysis
        csv_bytes = "JobSkill,ClosestResumeSkill,Score\n"
        # We can re-create a simple CSV: list missing/partial/matched
        for ms in (row['matched_skills'].split(",") if row['matched_skills'] else []):
            csv_bytes += f"{ms.strip()},Matched,100\n"
        for ps in (row['partial_skills'].split(",") if row['partial_skills'] else []):
            csv_bytes += f"{ps.strip()},Partial,65\n"
        for m in (row['missing_skills'].split(",") if row['missing_skills'] else []):
            csv_bytes += f"{m.strip()},Missing,10\n"
        st.download_button("Download CSV Report", data=csv_bytes, file_name=f"analysis_{sel}.csv", mime="text/csv")
        # PDF
        pdf_bytes = make_pdf_report("Resume text not stored", "JD text not stored", 
                                    (row['matched_skills'].split(",") if row['matched_skills'] else []),
                                    (row['partial_skills'].split(",") if row['partial_skills'] else []),
                                    (row['missing_skills'].split(",") if row['missing_skills'] else []),
                                    row['overall_match'])
        st.download_button("Download PDF Report", data=pdf_bytes, file_name=f"analysis_{sel}.pdf", mime="application/pdf")

# ---------------- Database viewing ----------------
elif menu == "Database":
    st.header("Database: Raw Tables")
    st.subheader("Resumes")
    st.dataframe(load_resumes())
    st.subheader("Job Descriptions")
    st.dataframe(load_jds())
    st.subheader("Analyses")
    st.dataframe(load_analyses())

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("<small style='color:gray;'>SkillGapAI • End-to-end pipeline • Developed by Suriya Varshan</small>", unsafe_allow_html=True)
