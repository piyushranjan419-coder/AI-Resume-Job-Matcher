
import streamlit as st
import re
import math
from io import BytesIO
from collections import Counter
from xml.sax.saxutils import escape

from pypdf import PdfReader
from docx import Document

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Resume & Job Matcher",
    page_icon="📄",
    layout="wide"
)

# ==========================================
# PROFESSIONAL UI DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0B1120;
    color: #F1F5F9;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263449;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

h1, h2, h3 {
    color: #F8FAFC !important;
    font-weight: 700 !important;
}

p, label, li {
    color: #CBD5E1;
}

.stButton > button {
    background: linear-gradient(90deg, #2563EB, #7C3AED);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1D4ED8, #6D28D9);
    border: none;
    color: white;
}

.stDownloadButton > button {
    background-color: #059669;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
}

.stTextArea textarea {
    background-color: #1E293B;
    color: white;
    border: 1px solid #334155;
    border-radius: 10px;
}

[data-testid="stFileUploader"] {
    background-color: #1E293B;
    border: 1px dashed #64748B;
    border-radius: 12px;
    padding: 15px;
}

[data-testid="stMetric"] {
    background-color: #172338;
    border: 1px solid #334155;
    padding: 20px;
    border-radius: 12px;
}

[data-testid="stMetricValue"] {
    color: #38BDF8;
}

hr {
    border-color: #334155;
}

</style>
""", unsafe_allow_html=True)



# ==========================================
# SKILL DATABASE
# ==========================================

SKILL_ALIASES = {
    "Python": ["python"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning"],
    "SQL": ["sql", "mysql", "postgresql"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Scikit-Learn": ["scikit-learn", "sklearn"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "NLP": ["nlp", "natural language processing"],
    "Java": ["java"],
    "C++": ["c++"],
    "HTML": ["html"],
    "CSS": ["css"],
    "JavaScript": ["javascript"],
    "React": ["react"],
    "GitHub": ["github"],
    "Docker": ["docker"],
    "AWS": ["aws", "amazon web services"],
    "Power BI": ["power bi"],
    "Excel": ["excel", "ms excel"],
    "Communication": ["communication"],
    "Leadership": ["leadership"],
    "Problem Solving": ["problem solving"],
    "Data Structures": ["data structures", "dsa"],
    "Algorithms": ["algorithms"],
    "Computer Networks": ["computer networks"],
    "Operating Systems": ["operating systems"],
    "MongoDB": ["mongodb"],
    "Flask": ["flask"],
    "Django": ["django"],
    "Streamlit": ["streamlit"],
    "Git": ["git"],
    "REST API": ["rest api", "restful api"]
}


# ==========================================
# LEARNING RESOURCES
# ==========================================

LEARNING_RESOURCES = {
    "Python": "Practice Python basics, functions, OOP and projects.",
    "Machine Learning": "Learn regression, classification and model evaluation.",
    "Deep Learning": "Study neural networks and backpropagation.",
    "SQL": "Practice SELECT, JOIN, GROUP BY and subqueries.",
    "Data Analysis": "Practice data cleaning and data visualization.",
    "Pandas": "Learn DataFrames, filtering and data cleaning.",
    "NumPy": "Learn arrays, indexing and mathematical operations.",
    "Scikit-Learn": "Learn preprocessing, model training and evaluation.",
    "TensorFlow": "Learn tensors, layers and neural networks.",
    "PyTorch": "Learn tensors, autograd and neural networks.",
    "NLP": "Learn tokenization and text classification.",
    "Java": "Learn object-oriented programming and core Java.",
    "JavaScript": "Learn functions, DOM and asynchronous JavaScript.",
    "React": "Learn components, props, state and hooks.",
    "Docker": "Learn containers, images and Dockerfiles.",
    "AWS": "Learn EC2, S3, IAM and cloud fundamentals.",
    "Power BI": "Learn Power Query, visuals and dashboards.",
    "Excel": "Practice formulas, lookup functions and pivot tables.",
    "HTML": "Learn semantic HTML and page structure.",
    "CSS": "Learn selectors, flexbox and grid.",
    "Data Structures": "Practice arrays, linked lists, stacks and queues.",
    "Algorithms": "Study sorting, searching and time complexity.",
    "MongoDB": "Learn collections, documents and CRUD operations.",
    "Flask": "Build small Python web applications.",
    "Django": "Learn models, views, templates and APIs.",
    "Streamlit": "Build interactive Python applications.",
    "Git": "Practice commits, branches and version control.",
    "REST API": "Learn HTTP methods, JSON and API integration."
}


# ==========================================
# RESUME TEXT EXTRACTION
# ==========================================

def extract_resume_text(uploaded_file):

    extension = uploaded_file.name.split(".")[-1].lower()
    file_bytes = uploaded_file.getvalue()

    try:

        if extension == "pdf":

            reader = PdfReader(BytesIO(file_bytes))

            text = []

            for page in reader.pages:
                text.append(page.extract_text() or "")

            return "\n".join(text)

        elif extension == "docx":

            document = Document(BytesIO(file_bytes))

            return "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
            )

        elif extension == "txt":

            return file_bytes.decode(
                "utf-8",
                errors="replace"
            )

        return ""

    except Exception as error:

        raise ValueError(
            f"Resume extraction failed: {error}"
        )


# ==========================================
# SKILL DETECTION
# ==========================================

def detect_skills(text):

    found_skills = []
    text = text.lower()

    for skill, aliases in SKILL_ALIASES.items():

        for alias in aliases:

            pattern = (
                r"(?<!\w)"
                + re.escape(alias.lower())
                + r"(?!\w)"
            )

            if re.search(pattern, text):

                found_skills.append(skill)
                break

    return found_skills


# ==========================================
# TF-IDF AND COSINE SIMILARITY
# PURE PYTHON - NO SCIKIT-LEARN
# ==========================================

def calculate_match(resume_text, job_text):

    if not resume_text.strip() or not job_text.strip():
        return 0

    def tokenize(text):

        return re.findall(
            r"[a-zA-Z0-9+#.]+",
            text.lower()
        )

    resume_words = tokenize(resume_text)
    job_words = tokenize(job_text)

    if not resume_words or not job_words:
        return 0

    resume_tf = Counter(resume_words)
    job_tf = Counter(job_words)

    vocabulary = set(resume_words + job_words)

    idf = {}

    for word in vocabulary:

        document_count = (
            int(word in resume_tf)
            + int(word in job_tf)
        )

        idf[word] = (
            math.log(3 / (document_count + 1)) + 1
        )

    resume_vector = {}
    job_vector = {}

    for word in vocabulary:

        resume_vector[word] = (
            resume_tf[word] * idf[word]
        )

        job_vector[word] = (
            job_tf[word] * idf[word]
        )

    dot_product = sum(
        resume_vector[word] * job_vector[word]
        for word in vocabulary
    )

    resume_magnitude = math.sqrt(
        sum(value ** 2 for value in resume_vector.values())
    )

    job_magnitude = math.sqrt(
        sum(value ** 2 for value in job_vector.values())
    )

    if resume_magnitude == 0 or job_magnitude == 0:
        return 0

    similarity = dot_product / (
        resume_magnitude * job_magnitude
    )

    return round(similarity * 100, 2)


# ==========================================
# RESUME IMPROVEMENT SUGGESTIONS
# ==========================================

def generate_suggestions(
    resume_text,
    missing_skills
):

    suggestions = []
    text = resume_text.lower()

    if len(resume_text.split()) < 100:

        suggestions.append(
            "Add more relevant information to your resume."
        )

    if not re.search(r"\b(project|projects)\b", text):

        suggestions.append(
            "Add an academic or personal projects section."
        )

    if not re.search(r"\b(experience|internship)\b", text):

        suggestions.append(
            "Add internship or practical experience if available."
        )

    if not re.search(r"\b(education|qualification)\b", text):

        suggestions.append(
            "Include your educational qualifications."
        )

    if not re.search(
        r"\b(achievement|achievements|award|certification)\b",
        text
    ):

        suggestions.append(
            "Include relevant achievements and certifications."
        )

    if missing_skills:

        suggestions.append(
            "Add relevant missing skills after learning and practicing them."
        )

    if not suggestions:

        suggestions.append(
            "Your resume has a good basic structure. "
            "Add measurable achievements to improve it further."
        )

    return suggestions


# ==========================================
# LEARNING RECOMMENDATIONS
# ==========================================

def generate_learning_recommendations(missing_skills):

    recommendations = []

    for skill in missing_skills:

        resource = LEARNING_RESOURCES.get(
            skill,
            f"Learn {skill} fundamentals and build a practical project."
        )

        recommendations.append(
            f"{skill}: {resource}"
        )

    return recommendations


# ==========================================
# PDF REPORT GENERATOR
# ==========================================

def generate_pdf_report(result):

    buffer = BytesIO()

    pdf = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "AI Resume & Job Matcher Report",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "<b>Resume:</b> "
            + escape(result["filename"]),
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            "<b>Job Match Score:</b> "
            + str(result["match_score"])
            + "%",
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 15))

    sections = [
        ("Resume Skills", result["resume_skills"]),
        ("Job Required Skills", result["job_skills"]),
        ("Matched Skills", result["matched_skills"]),
        ("Missing Skills", result["missing_skills"]),
        ("Improvement Suggestions", result["suggestions"]),
        ("Learning Recommendations", result["recommendations"])
    ]

    for heading, items in sections:

        story.append(
            Paragraph(
                escape(heading),
                styles["Heading2"]
            )
        )

        story.append(Spacer(1, 6))

        if items:

            for item in items:

                story.append(
                    Paragraph(
                        "&#8226; " + escape(str(item)),
                        styles["BodyText"]
                    )
                )

                story.append(Spacer(1, 5))

        else:

            story.append(
                Paragraph(
                    "None identified.",
                    styles["BodyText"]
                )
            )

        story.append(Spacer(1, 10))

    pdf.build(story)

    buffer.seek(0)

    return buffer.getvalue()



# ==========================================
# PROFESSIONAL APPLICATION HEADER
# ==========================================

st.caption("✨ AI POWERED CAREER TOOL")

st.title("📄 AI Resume & Job Matcher")

st.write(
    "Analyze your resume. Discover skill gaps. "
    "Improve your career opportunities."
)

st.info(
    "Upload your resume and compare your skills "
    "with your target job description."
)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🎯 Navigation")

    st.write("AI Resume & Job Matcher")

    st.info(
        "Upload your resume and enter a job description "
        "to start your analysis."
    )

    st.caption(
        "Built using Python and Streamlit."
    )

# ==========================================
# SKILL COVERAGE ANALYSIS
# ==========================================

if "analysis" in st.session_state:

    coverage_result = st.session_state["analysis"]

    st.subheader("📈 Skill Coverage Analysis")

    total_skills = (
        len(coverage_result["matched_skills"])
        + len(coverage_result["missing_skills"])
    )

    if total_skills > 0:

        matched_count = len(
            coverage_result["matched_skills"]
        )

        coverage = (matched_count / total_skills) * 100

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Required Skills Coverage",
                f"{coverage:.1f}%"
            )

        with col2:
            st.metric(
                "Total Required Skills",
                total_skills
            )

        st.progress(coverage / 100)

        st.caption(
            f"You matched {matched_count} out of "
            f"{total_skills} required skills."
        )

    else:
        st.info("No recognized job skills found.")

 # ==========================================
# RESUME UPLOAD
# ==========================================

st.subheader("1. Upload Your Resume")

uploaded_file = st.file_uploader(
    "Choose your resume",
    type=["pdf", "docx", "txt"]
)


# ==========================================
# JOB DESCRIPTION
# ==========================================

st.subheader("2. Enter Job Description")

job_description = st.text_area(
    "Paste the job description here",
    height=200,
    placeholder=(
        "Example: We are looking for a Python developer "
        "with SQL, Machine Learning and communication skills."
    )
)


# ==========================================
# ANALYZE BUTTON
# ==========================================

analyze_button = st.button(
    "🚀 Analyze My Resume",
    type="primary",
    use_container_width=True
)


# ==========================================
# ANALYSIS
# ==========================================

if analyze_button:

    if uploaded_file is None:

        st.error("Please upload your resume first.")

    elif not job_description.strip():

        st.error("Please enter a job description.")

    else:

        try:

            with st.spinner("Analyzing your resume..."):

                resume_text = extract_resume_text(
                    uploaded_file
                )

                if not resume_text.strip():

                    st.error(
                        "No readable text found in your resume. "
                        "Please try another file."
                    )

                    st.stop()

                resume_skills = detect_skills(
                    resume_text
                )

                job_skills = detect_skills(
                    job_description
                )

                matched_skills = [
                    skill for skill in job_skills
                    if skill in resume_skills
                ]

                missing_skills = [
                    skill for skill in job_skills
                    if skill not in resume_skills
                ]

                match_score = calculate_match(
                    resume_text,
                    job_description
                )

                suggestions = generate_suggestions(
                    resume_text,
                    missing_skills
                )

                recommendations = (
                    generate_learning_recommendations(
                        missing_skills
                    )
                )

                st.session_state["analysis"] = {

                    "filename": uploaded_file.name,

                    "match_score": match_score,

                    "resume_skills": resume_skills,

                    "job_skills": job_skills,

                    "matched_skills": matched_skills,

                    "missing_skills": missing_skills,

                    "suggestions": suggestions,

                    "recommendations": recommendations
                }

            st.success("Resume analysis completed!")

        except Exception as error:

            st.error(f"Analysis failed: {error}")


# ==========================================
# RESULTS DASHBOARD
# ==========================================

if "analysis" in st.session_state:

    result = st.session_state["analysis"]

    st.divider()

    st.header("📊 Resume Analysis Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Job Match Score",
        f"{result['match_score']}%"
    )

    col2.metric(
        "Matched Skills",
        len(result["matched_skills"])
    )

    col3.metric(
        "Missing Skills",
        len(result["missing_skills"])
    )

    st.progress(
        min(result["match_score"] / 100, 1.0)
    )

    st.caption(
        "Match score represents text similarity, "
        "not the probability of getting hired."
    )

    # SKILL ANALYSIS

    st.subheader("💡 Skill Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### ✅ Matched Skills")

        if result["matched_skills"]:

            for skill in result["matched_skills"]:
                st.success(skill)

        else:

            st.info("No matching skills detected.")

    with col2:

        st.markdown("### ❌ Missing Skills")

        if result["missing_skills"]:

            for skill in result["missing_skills"]:
                st.warning(skill)

        else:

            st.success("No missing skills detected!")

    # RESUME SKILLS

    with st.expander("View all detected resume skills"):

        if result["resume_skills"]:

            st.write(
                ", ".join(result["resume_skills"])
            )

        else:

            st.info("No skills detected.")

    # IMPROVEMENT SUGGESTIONS

    st.subheader("📝 Resume Improvement Suggestions")

    for suggestion in result["suggestions"]:

        st.info(suggestion)

    # LEARNING RECOMMENDATIONS

    st.subheader("📚 Learning Recommendations")

    if result["recommendations"]:

        for recommendation in result["recommendations"]:

            st.write("🎓", recommendation)

    else:

        st.success(
            "No missing skills detected from our skill database."
        )

    # PDF REPORT

    st.divider()

    st.subheader("📥 Download Your Report")

    pdf_data = generate_pdf_report(result)

    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_data,
        file_name="AI_Resume_Analysis_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "AI Resume & Job Matcher | Python | Streamlit | TF-IDF"
)
