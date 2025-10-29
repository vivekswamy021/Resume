import streamlit as st
import os
import pdfplumber
import docx
import openpyxl
import json
import tempfile
from groq import Groq
from gtts import gTTS # Although gTTS is not used in the final flow, keeping for completeness if it was part of original
import traceback
import re
from dotenv import load_dotenv 
from datetime import date 

# -------------------------
# CONFIGURATION & API SETUP
# -------------------------

# CRITICAL FIX: Using the currently supported Groq model.
GROQ_MODEL = "llama-3.1-8b-instant"

# Options for LLM functions
section_options = ["name", "email", "phone", "skills", "education", "experience", "certifications", "projects", "strength", "personal_details", "github", "linkedin", "full resume"]
question_section_options = ["skills","experience", "certifications", "education", "projects"]
answer_types = [("Point-wise", "points"), ("Detailed", "detailed"), ("Key Points", "key")]


# Load environment variables from .env file
load_dotenv()

# Ensure GROQ_API_KEY is defined in your environment or .env file
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GROQ_API_KEY:
    # Fail early with a clear message if the key is missing.
    st.error(
        "🚨 FATAL ERROR: GROQ_API_KEY environment variable not set. "
        "Please ensure a '.env' file exists in the script directory with this line: "
        "GROQ_API_KEY=\"YOUR_KEY_HERE\""
    )
    st.stop()

# Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)


# -------------------------
# Utility: Navigation Manager
# -------------------------
def go_to(page_name):
    """Changes the current page in Streamlit's session state."""
    st.session_state.page = page_name

# -------------------------
# CORE LOGIC: FILE HANDLING AND EXTRACTION
# -------------------------

def get_file_type(file_path):
    """Identifies the file type based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return 'pdf'
    elif ext == '.docx':
        return 'docx'
    else:
        # Assuming other file types like .txt, .json are treated as plain text
        return 'txt' 

def extract_content(file_type, file_path):
    """Extracts text content from PDF or DOCX files using robust libraries."""
    try:
        if file_type == 'pdf':
            text = ''
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
            if not text.strip():
                return "Error: PDF extraction failed. The file might be a scanned image without searchable text or is empty."
            return text
        
        elif file_type == 'docx':
            doc = docx.Document(file_path)
            text = '\n'.join([para.text for para in doc.paragraphs])
            if not text.strip():
                return "Error: DOCX content extraction failed. The file appears to be empty."
            return text
        
        elif file_type == 'txt':
            # Handle plain text, .json, or other unrecognized formats as text
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        else:
            return "Error: Unsupported file type."
    
    except Exception as e:
        return f"Fatal Extraction Error: Failed to read file content. Error details: {e}"

# -------------------------
# LLM & Extraction Functions
# -------------------------

@st.cache_data(show_spinner="Analyzing content with Groq LLM...")
def parse_with_llm(text, return_type='json'):
    """Sends resume text to the LLM for structured information extraction."""
    if text.startswith("Error"):
        return {"error": text, "raw_output": ""}

    prompt = f"""Extract the following information from the resume in structured JSON.
    Ensure all relevant details for each category are captured.
    - Name, - Email, - Phone, - Skills, - Education (list of degrees/institutions/dates), 
    - Experience (list of job roles/companies/dates/responsibilities), - Certifications (list), 
    - Projects (list of project names/descriptions/technologies), - Strength (list of personal strengths/qualities), 
    - Personal Details (e.g., address, date of birth, nationality), - Github (URL), - LinkedIn (URL)
    
    Resume Text:
    {text}
    
    Provide the output strictly as a JSON object.
    """
    content = ""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()

        # Robust JSON extraction
        json_str = content
        if json_str.startswith('```json'):
            json_str = json_str[len('```json'):]
        if json_str.endswith('```'):
            json_str = json_str[:-len('```')]
        json_str = json_str.strip()

        json_start = json_str.find('{')
        json_end = json_str.rfind('}') + 1

        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = json_str[json_start:json_end]

        parsed = json.loads(json_str)

    except json.JSONDecodeError as e:
        error_msg = f"JSON decoding error from LLM. LLM returned malformed JSON. Error: {e}"
        parsed = {"error": error_msg, "raw_output": content}
    except Exception as e:
        error_msg = f"LLM API interaction error: {e}"
        parsed = {"error": error_msg, "raw_output": "No LLM response due to API error."}

    if return_type == 'json':
        return parsed
    elif return_type == 'markdown':
        if "error" in parsed:
            return f"**Error:** {parsed.get('error', 'Unknown parsing error')}\nRaw output:\n```\n{parsed.get('raw_output','')}\n```"
        
        md = ""
        for k, v in parsed.items():
            if v:
                md += f"**{k.replace('_', ' ').title()}**:\n"
                if isinstance(v, list):
                    for item in v:
                        if item: 
                            md += f"- {item}\n"
                elif isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if sub_v:
                            md += f"  - {sub_k.replace('_', ' ').title()}: {sub_v}\n"
                else:
                    md += f"  {v}\n"
                md += "\n"
        return md
    return {"error": "Invalid return_type"}


def extract_jd_from_linkedin_url(url: str) -> str:
    """
    Simulates JD content extraction from a LinkedIn URL.
    This simulation is used for robustness in a pure Streamlit environment.
    """
    try:
        job_title = "Data Scientist"
        try:
            match = re.search(r'/jobs/view/([^/]+)', url)
            if match:
                job_title = match.group(1).replace('-', ' ').title()
        except:
            pass

        if "linkedin.com/jobs/" not in url:
             return f"[Error: Not a valid LinkedIn Job URL format: {url}]"

        
        # Simulated synthesized JD content 
        jd_text = f"""
        --- Simulated JD for: {job_title} ---
        
        **Company:** Quantum Analytics Inc.
        **Role:** {job_title}
        
        **Responsibilities:**
        - Develop and implement machine learning models to solve complex business problems.
        - Clean, transform, and analyze large datasets using Python/R and SQL.
        - Collaborate with engineering teams to deploy models into production environments.
        - Communicate findings and model performance to non-technical stakeholders.
        
        **Requirements:**
        - MS/PhD in Computer Science, Statistics, or a quantitative field.
        - 3+ years of experience as a Data Scientist.
        - Expertise in Python (Pandas, Scikit-learn, TensorFlow/PyTorch).
        - Experience with cloud platforms (AWS, Azure, or GCP).
        
        --- End Simulated JD ---
        """
        
        return jd_text.strip()
            
    except Exception as e:
        return f"[Fatal Extraction Error: Simulation failed for URL {url}. Error: {e}]"


def evaluate_jd_fit(job_description, parsed_json):
    """Evaluates how well a resume fits a given job description, including section-wise scores."""
    if not job_description.strip(): return "Please paste a job description."
    
    relevant_resume_data = {
        'Skills': parsed_json.get('skills', 'Not found or empty'),
        'Experience': parsed_json.get('experience', 'Not found or empty'),
        'Education': parsed_json.get('education', 'Not found or empty'),
    }
    resume_summary = json.dumps(relevant_resume_data, indent=2)

    prompt = f"""Evaluate how well the following resume content matches the provided job description.
    
    Job Description: {job_description}
    
    Resume Sections for Analysis:
    {resume_summary}
    
    Provide a detailed evaluation structured as follows:
    1.  **Overall Fit Score:** A score out of 10.
    2.  **Section Match Percentages:** A percentage score for the match in the key sections (Skills, Experience, Education).
    3.  **Strengths/Matches:** Key points where the resume aligns well with the JD.
    4.  **Gaps/Areas for Improvement:** Key requirements in the JD that are missing or weak in the resume.
    5.  **Overall Summary:** A concise summary of the fit.
    
    **Format the output strictly as follows, ensuring the scores are easily parsable (use brackets or no brackets around scores):**
    Overall Fit Score: [Score]/10
    
    --- Section Match Analysis ---
    Skills Match: [XX]%
    Experience Match: [YY]%
    Education Match: [ZZ]%
    
    Strengths/Matches:
    - Point 1
    - Point 2
    
    Gaps/Areas for Improvement:
    - Point 1
    - Point 2
    
    Overall Summary: [Concise summary]
    """

    response = client.chat.completions.create(
        model=GROQ_MODEL, 
        messages=[{"role": "user", "content": prompt}], 
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


# -------------------------
# Utility Functions
# -------------------------
def dump_to_excel(parsed_json, filename):
    """Dumps parsed JSON data to an Excel file."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Profile Data"
    ws.append(["Category", "Details"])
    
    section_order = ['name', 'email', 'phone', 'github', 'linkedin', 'experience', 'education', 'skills', 'projects', 'certifications', 'strength', 'personal_details']
    
    for section_key in section_order:
        if section_key in parsed_json and parsed_json[section_key]:
            content = parsed_json[section_key]
            
            if section_key in ['name', 'email', 'phone', 'github', 'linkedin']:
                ws.append([section_key.replace('_', ' ').title(), str(content)])
            else:
                ws.append([])
                ws.append([section_key.replace('_', ' ').title()])
                
                if isinstance(content, list):
                    for item in content:
                        if item:
                            ws.append(["", str(item)])
                elif isinstance(content, dict):
                    for k, v in content.items():
                        if v:
                            ws.append(["", f"{k.replace('_', ' ').title()}: {v}"])
                else:
                    ws.append(["", str(content)])

    wb.save(filename)
    with open(filename, "rb") as f:
        return f.read()

def parse_and_store_resume(uploaded_file, file_name_key='default'):
    """Handles file upload, parsing, and stores results in session state."""
    
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_type = get_file_type(temp_path)
    text = extract_content(file_type, temp_path)
    
    if text.startswith("Error"):
        return {"error": text, "full_text": text}

    parsed = parse_with_llm(text, return_type='json')
    
    if not parsed or "error" in parsed:
        return {"error": parsed.get('error', 'Unknown parsing error'), "full_text": text}

    # Generate Excel data for download if needed (only for single resume upload in Candidate dashboard)
    excel_data = None
    if file_name_key == 'single_resume_candidate':
        try:
            name = parsed.get('name', 'candidate').replace(' ', '_').strip()
            name = "".join(c for c in name if c.isalnum() or c in ('_', '-')).rstrip()
            if not name: name = "candidate"
            excel_filename = os.path.join(tempfile.gettempdir(), f"{name}_parsed_data.xlsx")
            excel_data = dump_to_excel(parsed, excel_filename)
        except Exception as e:
            st.warning(f"Could not generate Excel file for single resume: {e}")
    
    return {
        "parsed": parsed,
        "full_text": text,
        "excel_data": excel_data,
        "name": parsed.get('name', uploaded_file.name.split('.')[0])
        # Add placeholders for JD/Date which will be updated later
    }


def qa_on_resume(question):
    """Chatbot for Resume (Q&A) using LLM."""
    parsed_json = st.session_state.parsed
    full_text = st.session_state.full_text
    prompt = f"""Given the following resume information:
    Resume Text: {full_text}
    Parsed Resume Data (JSON): {json.dumps(parsed_json, indent=2)}
    Answer the following question about the resume concisely and directly.
    If the information is not present, state that clearly.
    Question: {question}
    """
    response = client.chat.completions.create(model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.4)
    return response.choices[0].message.content.strip()

def generate_interview_questions(parsed_json, section):
    """Generates categorized interview questions using LLM."""
    section_title = section.replace("_", " ").title()
    section_content = parsed_json.get(section, "")
    if isinstance(section_content, (list, dict)):
        section_content = json.dumps(section_content, indent=2)
    elif not isinstance(section_content, str):
        section_content = str(section_content)

    if not section_content.strip():
        return f"No significant content found for the '{section_title}' section in the parsed resume. Please select a section with relevant data to generate questions."

    prompt = f"""Based on the following {section_title} section from the resume: {section_content}
Generate 3 interview questions each for these levels: Generic, Basic, Intermediate, Difficult.
**IMPORTANT: Format the output strictly as follows, with level headers and questions starting with 'Qx:':**
[Generic]
Q1: Question text...
...
[Difficult]
Q1: Question text...
    """
    response = client.chat.completions.create(
        model=GROQ_MODEL, 
        messages=[{"role": "user", "content": prompt}], 
        temperature=0.5
    )
    return response.choices[0].message.content.strip()


# -------------------------
# UI PAGES: Authentication
# -------------------------
def login_page():
    st.title("🌐 PragyanAI Job Portal")
    st.header("Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login", use_container_width=True):
        if email and password:
            # Simulate successful login and go to role selection
            st.success("Login successful!")
            go_to("role_selection")
        else:
            st.error("Please enter both email and password")

    st.markdown("---")
    
    if st.button("Don't have an account? Sign up here"):
        go_to("signup")

def signup_page():
    st.header("Create an Account")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up", use_container_width=True):
        if password == confirm and email:
            st.success("Signup successful! Please login.")
            go_to("login")
        else:
            st.error("Passwords do not match or email is empty")

    if st.button("Already have an account? Login here"):
        go_to("login")

def role_selection_page():
    st.header("Select Your Role")
    role = st.selectbox(
        "Choose a Dashboard",
        ["Select Role", "Admin Dashboard", "Candidate Dashboard", "Hiring Company Dashboard"]
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("Continue", use_container_width=True):
            if role == "Admin Dashboard":
                go_to("admin_dashboard")
            elif role == "Candidate Dashboard":
                go_to("candidate_dashboard")
            elif role == "Hiring Company Dashboard":
                go_to("hiring_dashboard")
            else:
                st.warning("Please select a role first")

    with col2:
        if st.button("⬅️ Go Back to Login"):
            go_to("login")

# -------------------------
# UI PAGES: Dashboards
# -------------------------

def update_resume_status(resume_name, new_status, applied_jd, submitted_date, resume_list_index):
    """
    Callback function to update the status and metadata of a specific resume.
    """
    # 1. Update status in the separate status dictionary
    st.session_state.resume_statuses[resume_name] = new_status
    
    # 2. Update metadata in the resumes_to_analyze list
    # Use the index to safely update the specific resume object
    if 0 <= resume_list_index < len(st.session_state.resumes_to_analyze):
        st.session_state.resumes_to_analyze[resume_list_index]['applied_jd'] = applied_jd
        st.session_state.resumes_to_analyze[resume_list_index]['submitted_date'] = submitted_date
        st.success(f"Status and metadata for **{resume_name}** updated to **{new_status}**.")
    else:
        st.error(f"Error: Could not find resume index {resume_list_index} for update.")
        

def admin_dashboard():
    st.header("🧑‍💼 Admin Dashboard")
    
    # --- MODIFIED NAVIGATION BLOCK ---
    nav_col1, nav_col2 = st.columns([1, 1])

    with nav_col1:
        # Button to go back to role selection
        if st.button("⬅️ Go Back to Role Selection", use_container_width=True):
            go_to("role_selection")
            
    with nav_col2:
        # Log Out button directing to the login page
        if st.button("🚪 Log Out", use_container_width=True):
            go_to("login") 
    # --- END MODIFIED NAVIGATION BLOCK ---
    
    # Initialize Admin session state variables
    if "admin_jd_list" not in st.session_state:
        st.session_state.admin_jd_list = []
    if "resumes_to_analyze" not in st.session_state:
        st.session_state.resumes_to_analyze = []
    if "admin_match_results" not in st.session_state:
        st.session_state.admin_match_results = []
    if "resume_statuses" not in st.session_state:
        st.session_state.resume_statuses = {}
        
    # --- VENDOR INITIALIZATION ---
    if "vendors" not in st.session_state:
        st.session_state.vendors = []
    if "vendor_statuses" not in st.session_state:
        st.session_state.vendor_statuses = {}
    # --- END VENDOR INITIALIZATION ---

    
    # --- UPDATED TAB ORDER ---
    tab_jd, tab_analysis, tab_approval, tab_vendors, tab_statistics = st.tabs([
        "📄 JD Management", 
        "📊 Resume Analysis", 
        "👤 Candidate Approval", 
        "🤝 Vendor Approval", 
        "📈 Statistics" 
    ])
    # -------------------------

    # --- TAB 1: JD Management ---
    with tab_jd:
        st.subheader("Add and Manage Job Descriptions (JD)")
        
        jd_type = st.radio("Select JD Type", ["Single JD", "Multiple JD"], key="jd_type_admin")
        st.markdown("### Add JD by:")
        
        # Options for adding JD 
        method = st.radio("Choose Method", ["Upload File", "Paste Text", "LinkedIn URL"], key="jd_add_method_admin") 

        # URL
        if method == "LinkedIn URL":
            url_list = st.text_area(
                "Enter one or more URLs (comma separated)" if jd_type == "Multiple JD" else "Enter URL", key="url_list_admin"
            )
            if st.button("Add JD(s) from URL", key="add_jd_url_btn_admin"):
                if url_list:
                    urls = [u.strip() for u in url_list.split(",")] if jd_type == "Multiple JD" else [url_list.strip()]
                    
                    count = 0
                    for url in urls:
                        if not url: continue
                        
                        with st.spinner(f"Attempting JD extraction for: {url}"):
                            jd_text = extract_jd_from_linkedin_url(url)
                        
                        # Use a cleaner name for display
                        name_base = url.split('/jobs/view/')[-1].split('/')[0] if '/jobs/view/' in url else f"URL {count+1}"
                        st.session_state.admin_jd_list.append({"name": f"JD from URL: {name_base}", "content": jd_text})
                        if not jd_text.startswith("[Error"):
                            count += 1
                            
                    if count > 0:
                        st.success(f"✅ {count} JD(s) added successfully! Check the display below for the extracted content.")
                    else:
                        st.error("No JDs were added successfully.")


        # Paste Text
        elif method == "Paste Text":
            text_list = st.text_area(
                "Paste one or more JD texts (separate by '---')" if jd_type == "Multiple JD" else "Paste JD text here", key="text_list_admin"
            )
            if st.button("Add JD(s) from Text", key="add_jd_text_btn_admin"):
                if text_list:
                    texts = [t.strip() for t in text_list.split("---")] if jd_type == "Multiple JD" else [text_list.strip()]
                    for i, text in enumerate(texts):
                         if text:
                            # Use the first line as a name
                            name_base = text.splitlines()[0].strip()
                            if len(name_base) > 30: name_base = f"{name_base[:27]}..."
                            if not name_base: name_base = f"Pasted JD {len(st.session_state.admin_jd_list) + i + 1}"
                            
                            st.session_state.admin_jd_list.append({"name": name_base, "content": text})
                    st.success(f"✅ {len(texts)} JD(s) added successfully!")

        # Upload File
        elif method == "Upload File":
            uploaded_files = st.file_uploader(
                "Upload JD file(s)",
                type=["pdf", "txt", "docx"],
                accept_multiple_files=True if jd_type == "Multiple JD" and uploaded_files else False,
                key="jd_file_uploader_admin"
            )
            if st.button("Add JD(s) from File", key="add_jd_file_btn_admin"):
                files_to_process = uploaded_files if jd_type == "Multiple JD" and uploaded_files else [uploaded_files]
                count = 0
                for file in files_to_process:
                    if file:
                        temp_dir = tempfile.mkdtemp()
                        temp_path = os.path.join(temp_dir, file.name)
                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())
                            
                        file_type = get_file_type(temp_path)
                        jd_text = extract_content(file_type, temp_path)
                        
                        if not jd_text.startswith("Error"):
                            st.session_state.admin_jd_list.append({"name": file.name, "content": jd_text})
                            count += 1
                if count > 0:
                    st.success(f"✅ {count} JD(s) added successfully!")
                else:
                    st.error("No valid JD files were uploaded or content extraction failed.")

        # Display Added JDs
        if st.session_state.admin_jd_list:
            
            # --- CLEAR BUTTON LOGIC ---
            col_display_header, col_clear_button = st.columns([3, 1])
            
            with col_display_header:
                st.markdown("### ✅ Current JDs Added:")
                
            with col_clear_button:
                # Button to clear the entire JD list
                if st.button("🗑️ Clear All JDs", key="clear_jds_admin", use_container_width=True, help="Removes all currently loaded JDs."):
                    st.session_state.admin_jd_list = []
                    # Also clear existing match results if any JD is removed
                    st.session_state.admin_match_results = [] 
                    st.success("All JDs and associated match results have been cleared.")
                    st.rerun() 
            # --- END CLEAR BUTTON LOGIC ---

            for idx, jd_item in enumerate(st.session_state.admin_jd_list, 1):
                title = jd_item['name']
                # Clean up simulated title prefix for display
                display_title = title.replace("--- Simulated JD for: ", "")
                with st.expander(f"JD {idx}: {display_title}"):
                    st.text(jd_item['content'])
        else:
            st.info("No Job Descriptions added yet.")


    # --- TAB 2: Resume Analysis (INCLUDES ALL ROBUSTNESS FIXES) --- 
    with tab_analysis:
        st.subheader("Analyze Resumes Against Job Descriptions")

        # 1. Resume Upload
        st.markdown("#### 1. Upload Resumes")
        
        resume_upload_type = st.radio("Upload Type", ["Single Resume", "Multiple Resumes"], key="resume_upload_type_admin")

        uploaded_files = st.file_uploader(
            "Choose files to analyze",
            type=["pdf", "docx", "txt", "json", "rtf"], 
            accept_multiple_files=(resume_upload_type == "Multiple Resumes"),
            key="resume_file_uploader_admin"
        )
        
        col_parse, col_clear = st.columns([3, 1])
        
        with col_parse:
            if st.button("Load and Parse Resume(s) for Analysis", key="parse_resumes_admin", use_container_width=True):
                if uploaded_files:
                    files_to_process = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
                    
                    count = 0
                    with st.spinner("Parsing resume(s)... This may take a moment."):
                        for file in files_to_process:
                            if file:
                                result = parse_and_store_resume(file, file_name_key='admin_analysis')
                                
                                if "error" not in result:
                                    # --- ADD PLACEHOLDER METADATA HERE ---
                                    result['applied_jd'] = "N/A (Pending Assignment)"
                                    result['submitted_date'] = date.today().strftime("%Y-%m-%d")
                                    
                                    st.session_state.resumes_to_analyze.append(result)
                                    
                                    # Initialize status for the resume using its unique name
                                    resume_id = result['name']
                                    if resume_id not in st.session_state.resume_statuses:
                                        st.session_state.resume_statuses[resume_id] = "Pending"
                                    # --- END ADD PLACEHOLDER METADATA HERE ---
                                    
                                    count += 1
                                else:
                                    st.error(f"Failed to parse {file.name}: {result['error']}")

                    if count > 0:
                        st.success(f"Successfully loaded and parsed {count} resume(s) for analysis.")
                        st.rerun() 
                    elif not st.session_state.resumes_to_analyze:
                        st.warning("No resumes were successfully loaded and parsed.")
                else:
                    st.warning("Please upload one or more resume files.")
        
        with col_clear:
            # --- NEW CLEAR BUTTON LOGIC ---
            if st.button("🗑️ Clear All Resumes", key="clear_resumes_admin", use_container_width=True, help="Removes all currently loaded resumes and match results."):
                st.session_state.resumes_to_analyze = []
                st.session_state.admin_match_results = []
                st.session_state.resume_statuses = {} 
                st.success("All resumes and associated match results have been cleared.")
                st.rerun() 
            # --- END NEW CLEAR BUTTON LOGIC ---


        st.markdown("---")

        # 2. JD Selection and Analysis
        st.markdown("#### 2. Select JD and Run Analysis")

        if not st.session_state.resumes_to_analyze:
            st.info("Upload and parse resumes first to enable analysis.")

        if not st.session_state.admin_jd_list:
            st.error("Please add at least one Job Description in the 'JD Management' tab before running an analysis.")
            return

        # --- RESUME SELECTION ---
        resume_names = [r['name'] for r in st.session_state.resumes_to_analyze]
        selected_resume_names = st.multiselect(
            "Select Resume(s) for Matching",
            options=resume_names,
            default=resume_names, 
            key="select_resumes_admin"
        )
        
        # Filter the resumes to be analyzed based on selection
        resumes_to_match = [
            r for r in st.session_state.resumes_to_analyze 
            if r['name'] in selected_resume_names
        ]
        # --- END RESUME SELECTION ---

        jd_options = {item['name']: item['content'] for item in st.session_state.admin_jd_list}
        selected_jd_name = st.selectbox("Select JD for Matching", list(jd_options.keys()), key="select_jd_admin")
        selected_jd_content = jd_options.get(selected_jd_name, "")


        if st.button(f"Run Match Analysis on {len(resumes_to_match)} Selected Resume(s)", key="run_match_analysis_admin"):
            st.session_state.admin_match_results = []
            
            if not selected_jd_content:
                st.error("Selected JD content is empty.")
                return
            
            if not resumes_to_match:
                st.warning("No resumes were selected for matching.")
                return

            with st.spinner(f"Matching {len(resumes_to_match)} resumes against '{selected_jd_name}'..."):
                for resume_data in resumes_to_match: # Use the FILTERED list
                    
                    resume_name = resume_data['name']
                    parsed_json = resume_data['parsed']

                    try:
                        fit_output = evaluate_jd_fit(selected_jd_content, parsed_json)
                        
                        # --- ENHANCED EXTRACTION LOGIC (FIXED FOR ROBUSTNESS: Overall Score) ---
                        # New FIX: Allows any whitespace/non-digit character sequence between the label and the score (number)
                        overall_score_match = re.search(r'Overall Fit Score:\s*[^\d]*(\d+)\s*/10', fit_output, re.IGNORECASE)
                        
                        # Look for the section analysis block between delimiters
                        section_analysis_match = re.search(
                             r'--- Section Match Analysis ---\s*(.*?)\s*Strengths/Matches:', 
                             fit_output, re.DOTALL
                        )

                        skills_percent = 'N/A'
                        experience_percent = 'N/A'
                        education_percent = 'N/A'
                        
                        if section_analysis_match:
                            section_text = section_analysis_match.group(1)
                            
                            # Look for "Skills Match: [XX]%" - handles optional brackets [ ] around the percentage
                            skills_match = re.search(r'Skills Match:\s*\[?(\d+)%\]?', section_text, re.IGNORECASE)
                            experience_match = re.search(r'Experience Match:\s*\[?(\d+)%\]?', section_text, re.IGNORECASE)
                            education_match = re.search(r'Education Match:\s*\[?(\d+)%\]?', section_text, re.IGNORECASE)
                            
                            if skills_match:
                                skills_percent = skills_match.group(1)
                            if experience_match:
                                experience_percent = experience_match.group(1)
                            if education_match:
                                education_percent = education_match.group(1)
                        
                        overall_score = overall_score_match.group(1) if overall_score_match else 'N/A'
                        # --- END ENHANCED EXTRACTION LOGIC ---

                        st.session_state.admin_match_results.append({
                            "resume_name": resume_name,
                            "jd_name": selected_jd_name,
                            "overall_score": overall_score,
                            "skills_percent": skills_percent,
                            "experience_percent": experience_percent, 
                            "education_percent": education_percent,   
                            "full_analysis": fit_output
                        })
                    except Exception as e:
                        st.session_state.admin_match_results.append({
                            "resume_name": resume_name,
                            "jd_name": selected_jd_name,
                            "overall_score": "Error",
                            "skills_percent": "Error",
                            "experience_percent": "Error", 
                            "education_percent": "Error",   
                            "full_analysis": f"Error running analysis: {e}\n{traceback.format_exc()}"
                        })
                st.success("Analysis complete!")


        # 3. Display Results
        if st.session_state.get('admin_match_results'):
            st.markdown("#### 3. Match Results")
            results_df = st.session_state.admin_match_results
            
            # Create a simple table/summary of results
            display_data = []
            for item in results_df:
                # Get status from the separate status dictionary using the resume name as the key
                status = st.session_state.resume_statuses.get(item["resume_name"], 'Pending') 
                
                display_data.append({
                    "Resume": item["resume_name"],
                    "JD": item["jd_name"],
                    "Fit Score (out of 10)": item["overall_score"],
                    "Skills (%)": item.get("skills_percent", "N/A"),
                    "Experience (%)": item.get("experience_percent", "N/A"), 
                    "Education (%)": item.get("education_percent", "N/A"),
                    "Approval Status": status
                })

            st.dataframe(display_data, use_container_width=True)

            # Display detailed analysis in expanders
            st.markdown("##### Detailed Reports")
            for item in results_df:
                # Ensure the display reflects the extracted score/percentage, even if it's 'N/A' or 'Error'
                header_text = f"Report for **{item['resume_name']}** against {item['jd_name']} (Score: **{item['overall_score']}/10** | S: **{item.get('skills_percent', 'N/A')}%** | E: **{item.get('experience_percent', 'N/A')}%** | Edu: **{item.get('education_percent', 'N/A')}%**)"
                with st.expander(header_text):
                    st.markdown(item['full_analysis'])

                    
    # --- TAB 3: Candidate Approval (FIXED to include JD/Date inputs) ---
    with tab_approval:
        st.header("Review and Approve Candidate Resumes")
        st.markdown("### Resume Status List")
        
        if not st.session_state.resumes_to_analyze:
            st.info("No resumes have been uploaded and parsed in the 'Resume Analysis' tab yet.")
            return
            
        # Get list of uploaded JD names for the dropdown
        jd_options = [item['name'].replace("--- Simulated JD for: ", "") for item in st.session_state.admin_jd_list]
        jd_options.insert(0, "Select JD") # Add a default placeholder

        # Use the list of parsed resumes to drive the approval interface
        for idx, resume_data in enumerate(st.session_state.resumes_to_analyze):
            resume_name = resume_data['name']
            current_status = st.session_state.resume_statuses.get(resume_name, "Pending")
            
            # Extract current metadata for pre-filling the inputs
            current_applied_jd = resume_data.get('applied_jd', 'N/A (Pending Assignment)')
            current_submitted_date = resume_data.get('submitted_date', date.today().strftime("%Y-%m-%d"))

            # Use a container to group the elements for a single resume
            with st.container(border=True):
                
                # --- METADATA INPUT AND DISPLAY (FIXED SECTION) ---
                st.markdown(f"**Resume:** **{resume_name}**")
                
                # Inputs for metadata
                col_jd_input, col_date_input = st.columns(2)
                
                with col_jd_input:
                    # Determine the default index for the selectbox
                    try:
                        # Find the index of the currently saved JD name in the options list
                        default_value = current_applied_jd if current_applied_jd != "N/A (Pending Assignment)" else "Select JD"
                        jd_default_index = jd_options.index(default_value)
                    except ValueError:
                        jd_default_index = 0
                        
                    # --- Selectbox for Applied JD ---
                    new_applied_jd = st.selectbox(
                        "Applied for JD Title", 
                        options=jd_options,
                        index=jd_default_index,
                        key=f"jd_select_{resume_name}_{idx}",
                    )
                    
                with col_date_input:
                    # Try to parse the stored string date, fall back to today's date if 'N/A' or failed
                    try:
                        # Ensure the input date is an actual date object
                        date_obj = date.fromisoformat(current_submitted_date)
                    except (ValueError, TypeError):
                        date_obj = date.today()
                        
                    new_submitted_date = st.date_input(
                        "Submitted Date", 
                        value=date_obj,
                        key=f"date_input_{resume_name}_{idx}"
                    )
                    
                st.markdown(f"**Current Status:** **{current_status}**")
                
                st.markdown("---")
                
                # --- STATUS SELECTOR AND UPDATE BUTTON ---
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("Set Status:")
                    # Key needs to be unique for each selectbox
                    new_status = st.selectbox(
                        "Set Status",
                        ["Pending", "Approved", "Rejected", "Shortlisted"],
                        index=["Pending", "Approved", "Rejected", "Shortlisted"].index(current_status),
                        key=f"status_select_{resume_name}_{idx}",
                        label_visibility="collapsed"
                    )

                with col2:
                    # Pass the index and new metadata values to the update function
                    if st.button("Update", key=f"update_btn_{resume_name}_{idx}"):
                        
                        # Logic to save the correct JD name or placeholder
                        if new_applied_jd == "Select JD" and len(jd_options) > 1:
                            # If the user selects the placeholder, save the official placeholder text
                            jd_to_save = "N/A (Pending Assignment)"
                        else:
                            # Otherwise, save the selected JD name
                            jd_to_save = new_applied_jd
                            
                        update_resume_status(
                            resume_name, 
                            new_status, 
                            jd_to_save, 
                            new_submitted_date.strftime("%Y-%m-%d"),
                            idx
                        )
                        st.rerun() 
                
        st.markdown("---")
                
        # Optional: Display a summary table of all statuses
        
        # Create a richer summary table including the new metadata
        summary_data = []
        for resume_data in st.session_state.resumes_to_analyze:
            name = resume_data['name']
            summary_data.append({
                "Resume": name, 
                "Applied JD": resume_data.get('applied_jd', 'N/A'),
                "Submitted Date": resume_data.get('submitted_date', 'N/A'),
                "Status": st.session_state.resume_statuses.get(name, "Pending")
            })
            
        st.subheader("Summary of All Resumes")
        st.dataframe(summary_data, use_container_width=True)

    # --- TAB 4: Vendor Approval ---
    with tab_vendors:
        st.header("🤝 Vendor Management") 
        
        # --- SECTION 1: Add New Vendor ---
        st.markdown("### 1. Add New Vendor")
        with st.form("add_vendor_form"):
            col1, col2 = st.columns(2)
            with col1:
                vendor_name = st.text_input("Vendor Name", key="new_vendor_name")
            with col2:
                vendor_domain = st.text_input("Service / Domain Name", key="new_vendor_domain")
                
            col3, col4 = st.columns(2)
            with col3:
                submitted_date = st.date_input("Submitted Date", value=date.today(), key="new_vendor_date")
            with col4:
                initial_status = st.selectbox(
                    "Set Status", 
                    ["Pending Review", "Approved", "Rejected"],
                    key="new_vendor_status"
                )
            
            add_vendor_button = st.form_submit_button("Add Vendor", use_container_width=True)

            if add_vendor_button:
                if vendor_name and vendor_domain:
                    # Create a unique ID/key for the vendor
                    vendor_id = vendor_name.strip() 
                    
                    # Check for duplicates (by name)
                    if vendor_id in st.session_state.vendor_statuses:
                        st.warning(f"Vendor '{vendor_name}' already exists.")
                    else:
                        new_vendor = {
                            'name': vendor_name.strip(),
                            'domain': vendor_domain.strip(),
                            'submitted_date': submitted_date.strftime("%Y-%m-%d")
                        }
                        st.session_state.vendors.append(new_vendor)
                        st.session_state.vendor_statuses[vendor_id] = initial_status
                        st.success(f"Vendor **{vendor_name}** added successfully with status **{initial_status}**.")
                        st.rerun() 
                else:
                    st.error("Please fill in both Vendor Name and Service / Domain Name.")

        st.markdown("---")
        
        # --- SECTION 2: Update Existing Vendor Status (NEW/Restored Logic) ---
        st.markdown("### 2. Update Existing Vendor Status")
        
        if not st.session_state.vendors:
            st.info("No vendors have been added yet.")
        else:
            for idx, vendor in enumerate(st.session_state.vendors):
                vendor_name = vendor['name']
                vendor_id = vendor_name # Key in vendor_statuses is the name
                current_status = st.session_state.vendor_statuses.get(vendor_id, "Unknown")
                
                with st.container(border=True):
                    
                    col_info, col_status_input, col_update_btn = st.columns([3, 2, 1])
                    
                    with col_info:
                        st.markdown(f"**Vendor:** {vendor_name} (`{vendor['domain']}`) - *Submitted: {vendor['submitted_date']}*")
                        st.markdown(f"**Current Status:** **{current_status}**")
                        
                    with col_status_input:
                        # Selectbox to choose new status
                        new_status = st.selectbox(
                            "Set Status",
                            ["Pending Review", "Approved", "Rejected"],
                            index=["Pending Review", "Approved", "Rejected"].index(current_status),
                            key=f"vendor_status_select_{idx}",
                            label_visibility="collapsed"
                        )

                    with col_update_btn:
                        # Add some vertical space to align the button
                        st.markdown("##") 
                        if st.button("Update", key=f"vendor_update_btn_{idx}", use_container_width=True):
                            
                            # Update the status in the session state dictionary
                            st.session_state.vendor_statuses[vendor_id] = new_status
                            
                            st.success(f"Status for **{vendor_name}** updated to **{new_status}**.")
                            st.rerun()
                            
            st.markdown("---")
            
            # --- Summary Table (Optional, but helpful for quick glance) ---
            summary_data = []
            for vendor in st.session_state.vendors:
                name = vendor['name']
                summary_data.append({
                    "Vendor Name": name,
                    "Domain": vendor['domain'],
                    "Submitted Date": vendor['submitted_date'],
                    "Status": st.session_state.vendor_statuses.get(name, "Unknown")
                })
            
            st.subheader("Summary of All Vendors")
            st.dataframe(summary_data, use_container_width=True)


    # --- TAB 5: Statistics ---
    with tab_statistics:
        st.header("System Statistics")
        st.markdown("---")

        # 1. Calculate Metrics
        total_candidates = len(st.session_state.resumes_to_analyze)
        total_jds = len(st.session_state.admin_jd_list)
        total_vendors = len(st.session_state.vendors)
        no_of_applications = total_candidates 
        
        # 2. Display Metrics using st.metric
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(label="Total Candidates", value=total_candidates, delta="Resumes Submitted")

        with col2:
            st.metric(label="Total JDs", value=total_jds, delta_color="off")
        
        with col3:
            st.metric(label="Total Vendors", value=total_vendors, delta_color="off")

        with col4:
            st.metric(label="No. of Applications", value=no_of_applications, delta_color="off")
            
        st.markdown("---")
        
        # Optional: Add a breakdown of resume status
        status_counts = {}
        for status in st.session_state.resume_statuses.values():
            status_counts[status] = status_counts.get(status, 0) + 1
            
        st.subheader("Candidate Status Breakdown")
        
        status_cols = st.columns(len(status_counts) or 1)
        
        if status_counts:
            col_count = len(status_cols)
            for i, (status, count) in enumerate(status_counts.items()):
                with status_cols[i % col_count]:
                    st.metric(label=f"{status}", value=count)
        else:
            st.info("No resumes loaded to calculate status breakdown.")


def candidate_dashboard():
    st.header("👩‍🎓 Candidate Dashboard")
    st.markdown("Welcome! Use the tabs below to upload your resume and access AI preparation tools.")

    # --- MODIFIED NAVIGATION BLOCK ---
    nav_col1, nav_col2 = st.columns([1, 1])

    with nav_col1:
        if st.button("⬅️ Go Back to Role Selection", use_container_width=True):
            go_to("role_selection")
            
    with nav_col2:
        if st.button("🚪 Log Out", key="candidate_logout_btn", use_container_width=True):
            go_to("login") 
    # --- END MODIFIED NAVIGATION BLOCK ---
    
    # Sidebar for Resume Upload (Centralized Upload)
    with st.sidebar:
        st.header("Upload Your Resume")
        uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])
        
        if uploaded_file is not None:
            if st.button("Parse Resume", use_container_width=True):
                # Centralized upload logic for Candidate Dashboard
                result = parse_and_store_resume(uploaded_file, file_name_key='single_resume_candidate')
                
                if "error" not in result:
                    st.session_state.parsed = result['parsed']
                    st.session_state.full_text = result['full_text']
                    st.session_state.excel_data = result['excel_data'] 
                    st.success("Resume parsed successfully!")
                else:
                    st.error(f"Parsing failed: {result['error']}")

        
        st.markdown("---")
        if st.session_state.parsed.get("name"):
            st.success(f"Resume for **{st.session_state.parsed['name']}** is loaded.")
        elif st.session_state.full_text:
            st.warning("Resume file loaded, but parsing may have errors.")
        else:
            st.info("Please upload a resume to begin.")

    # Main Content Tabs (AI Resume Parser Features)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Resume Parsing", 
        "💬 Resume Chatbot (Q&A)", 
        "❓ Interview Prep", 
        "📚 JD Management", 
        "🎯 Batch JD Match" 
    ])
    
    # --- TAB 1: Resume Parsing ---
    with tab1:
        st.header("Resume Parsing")
        if not st.session_state.full_text:
            st.warning("Please upload and parse a resume in the sidebar first.")
            return

        col1, col2 = st.columns(2)
        with col1:
            output_format = st.radio('Output Format', ['json', 'markdown'], key='format_radio_c')
        with col2:
            section = st.selectbox('Select Section to View', section_options, key='section_select_c')

        parsed = st.session_state.parsed
        full_text = st.session_state.full_text

        if "error" in parsed:
            st.error(parsed.get("error", "An unknown error occurred during parsing."))
            return

        # Display Main Parsed Output
        if output_format == 'json':
            output_str = json.dumps(parsed, indent=2)
            st.text_area("Parsed Output (JSON)", output_str, height=350)
        else:
            output_str = parse_with_llm(full_text, return_type='markdown')
            st.markdown("### Parsed Output (Markdown)")
            st.markdown(output_str)

        # Download Buttons
        if st.session_state.excel_data:
            st.download_button(
                label="Download Parsed Data (Excel)",
                data=st.session_state.excel_data,
                file_name=f"{parsed.get('name', 'candidate').replace(' ', '_')}_parsed_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Section View
        section_content_str = ""
        if section == "full resume":
            section_content_str = full_text
        elif section in parsed:
            section_val = parsed[section]
            section_content_str = json.dumps(section_val, indent=2) if isinstance(section_val, (list, dict)) else str(section_val)
        else:
            section_content_str = f"Section '{section}' not found or is empty."

        st.text_area("Selected Section Content", section_content_str, height=200)

    # --- TAB 2: Resume Chatbot (Q&A) ---
    with tab2:
        st.header("Resume Chatbot (Q&A)")
        st.markdown("### Ask any question about the uploaded resume.")
        if not st.session_state.full_text:
            st.warning("Please upload and parse a resume first.")
            return

        question = st.text_input("Your Question", placeholder="e.g., What are the candidate's key skills?")
        
        if st.button("Get Answer", key="qa_btn"):
            with st.spinner("Generating answer..."):
                try:
                    answer = qa_on_resume(question)
                    st.session_state.qa_answer = answer
                except Exception as e:
                    st.error(f"Error during Q&A: {e}")
                    st.session_state.qa_answer = "Could not generate an answer."

        if st.session_state.get('qa_answer'):
            st.text_area("Answer", st.session_state.qa_answer, height=150)

    # --- TAB 3: Interview Prep ---
    with tab3:
        st.header("Interview Preparation Tools")
        if not st.session_state.parsed or "error" in st.session_state.parsed:
            st.warning("Please upload and successfully parse a resume first.")
            return

        st.subheader("Generate Interview Questions")
        section_choice = st.selectbox("Select Section", question_section_options, key='iq_section_c')
        
        if st.button("Generate Interview Questions", key='iq_btn_c'):
            with st.spinner("Generating questions..."):
                try:
                    raw_questions_response = generate_interview_questions(st.session_state.parsed, section_choice)
                    st.session_state.iq_output = raw_questions_response
                except Exception as e:
                    st.error(f"Error generating questions: {e}")
                    st.session_state.iq_output = "Error generating questions."

        if st.session_state.get('iq_output'):
            st.text_area("Generated Interview Questions (by difficulty level)", st.session_state.iq_output, height=400)
            
    # --- TAB 4: JD Management (Candidate) ---
    with tab4:
        st.header("📚 Manage Job Descriptions for Matching")
        st.markdown("Add multiple JDs here to compare your resume against them in the next tab.")
        
        # Initialize JD list specific to candidate if not present (to avoid mixing with admin's list)
        if "candidate_jd_list" not in st.session_state:
             st.session_state.candidate_jd_list = []
        
        jd_type = st.radio("Select JD Type", ["Single JD", "Multiple JD"], key="jd_type_candidate")
        st.markdown("### Add JD by:")
        
        # Options for adding JD 
        method = st.radio("Choose Method", ["Upload File", "Paste Text", "LinkedIn URL"], key="jd_add_method_candidate") 

        # URL
        if method == "LinkedIn URL":
            url_list = st.text_area(
                "Enter one or more URLs (comma separated)" if jd_type == "Multiple JD" else "Enter URL", key="url_list_candidate"
            )
            if st.button("Add JD(s) from URL", key="add_jd_url_btn_candidate"):
                if url_list:
                    urls = [u.strip() for u in url_list.split(",")] if jd_type == "Multiple JD" else [url_list.strip()]
                    
                    count = 0
                    for url in urls:
                        if not url: continue
                        
                        with st.spinner(f"Attempting JD extraction for: {url}"):
                            jd_text = extract_jd_from_linkedin_url(url)
                        
                        name_base = url.split('/jobs/view/')[-1].split('/')[0] if '/jobs/view/' in url else f"URL {count+1}"
                        st.session_state.candidate_jd_list.append({"name": f"JD from URL: {name_base}", "content": jd_text})
                        if not jd_text.startswith("[Error"):
                            count += 1
                            
                    if count > 0:
                        st.success(f"✅ {count} JD(s) added successfully! Check the display below for the extracted content.")
                    else:
                        st.error("No JDs were added successfully.")


        # Paste Text
        elif method == "Paste Text":
            text_list = st.text_area(
                "Paste one or more JD texts (separate by '---')" if jd_type == "Multiple JD" else "Paste JD text here", key="text_list_candidate"
            )
            if st.button("Add JD(s) from Text", key="add_jd_text_btn_candidate"):
                if text_list:
                    texts = [t.strip() for t in text_list.split("---")] if jd_type == "Multiple JD" else [text_list.strip()]
                    for i, text in enumerate(texts):
                         if text:
                            name_base = text.splitlines()[0].strip()
                            if len(name_base) > 30: name_base = f"{name_base[:27]}..."
                            if not name_base: name_base = f"Pasted JD {len(st.session_state.candidate_jd_list) + i + 1}"
                            
                            st.session_state.candidate_jd_list.append({"name": name_base, "content": text})
                    st.success(f"✅ {len(texts)} JD(s) added successfully!")

        # Upload File
        elif method == "Upload File":
            uploaded_files = st.file_uploader(
                "Upload JD file(s)",
                type=["pdf", "txt", "docx"],
                accept_multiple_files=True if jd_type == "Multiple JD" and uploaded_files else False,
                key="jd_file_uploader_candidate"
            )
            if st.button("Add JD(s) from File", key="add_jd_file_btn_candidate"):
                files_to_process = uploaded_files if jd_type == "Multiple JD" and uploaded_files else [uploaded_files]
                count = 0
                for file in files_to_process:
                    if file:
                        temp_dir = tempfile.mkdtemp()
                        temp_path = os.path.join(temp_dir, file.name)
                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())
                            
                        file_type = get_file_type(temp_path)
                        jd_text = extract_content(file_type, temp_path)
                        
                        if not jd_text.startswith("Error"):
                            st.session_state.candidate_jd_list.append({"name": file.name, "content": jd_text})
                            count += 1
                if count > 0:
                    st.success(f"✅ {count} JD(s) added successfully!")
                else:
                    st.error("No valid JD files were uploaded or content extraction failed.")

        # Display Added JDs
        if st.session_state.candidate_jd_list:
            
            # --- CLEAR BUTTON LOGIC (Candidate) ---
            col_display_header, col_clear_button = st.columns([3, 1])
            
            with col_display_header:
                st.markdown("### ✅ Current JDs Added:")
                
            with col_clear_button:
                # Button to clear the entire JD list
                if st.button("🗑️ Clear All JDs", key="clear_jds_candidate", use_container_width=True, help="Removes all currently loaded JDs."):
                    st.session_state.candidate_jd_list = []
                    # Also clear existing match results
                    st.session_state.candidate_match_results = [] 
                    st.success("All JDs and associated match results have been cleared.")
                    st.rerun() 
            # --- END CLEAR BUTTON LOGIC (Candidate) ---

            for idx, jd_item in enumerate(st.session_state.candidate_jd_list, 1):
                title = jd_item['name']
                display_title = title.replace("--- Simulated JD for: ", "")
                with st.expander(f"JD {idx}: {display_title}"):
                    st.text(jd_item['content'])
        else:
            st.info("No Job Descriptions added yet.")

    # --- TAB 5: Batch JD Match (Candidate) (INCLUDES ROBUSTNESS FIXES) ---
    with tab5:
        st.header("🎯 Batch JD Match")
        st.markdown("Compare your current resume against all saved job descriptions.")

        if not st.session_state.parsed:
            st.warning("Please **upload and parse your resume** in the sidebar first.")
            return

        if not st.session_state.candidate_jd_list:
            st.error("Please **add Job Descriptions** in the 'JD Management' tab (Tab 4) before running batch analysis.")
            return
            
        # Initialize results list for the candidate dashboard
        if "candidate_match_results" not in st.session_state:
            st.session_state.candidate_match_results = []

        if st.button(f"Run Batch Match Against {len(st.session_state.candidate_jd_list)} JDs"):
            st.session_state.candidate_match_results = []
            
            resume_name = st.session_state.parsed.get('name', 'Uploaded Resume')
            parsed_json = st.session_state.parsed

            with st.spinner(f"Matching {resume_name}'s resume against {len(st.session_state.candidate_jd_list)} JDs..."):
                for jd_item in st.session_state.candidate_jd_list:
                    
                    jd_name = jd_item['name']
                    jd_content = jd_item['content']

                    try:
                        fit_output = evaluate_jd_fit(jd_content, parsed_json)
                        
                        # --- ENHANCED EXTRACTION LOGIC (FIXED FOR ROBUSTNESS: Overall Score) ---
                        # New FIX: Allows any whitespace/non-digit character sequence between the label and the score (number)
                        overall_score_match = re.search(r'Overall Fit Score:\s*[^\d]*(\d+)\s*/10', fit_output, re.IGNORECASE)
                        
                        # Look for the section analysis block between delimiters
                        section_analysis_match = re.search(
                             r'--- Section Match Analysis ---\s*(.*?)\s*Strengths/Matches:', 
                             fit_output, re.DOTALL
                        )

                        skills_percent = 'N/A'
                        experience_percent = 'N/A'
                        education_percent = 'N/A'
                        
                        if section_analysis_match:
                            section_text = section_analysis_match.group(1)
                            
                            # Look for "Skills Match: [XX]%" - handles optional brackets [ ] around the percentage
                            skills_match = re.search(r'Skills Match:\s*\[?(\d+)%\]?', section_text, re.IGNORECASE)
                            experience_match = re.search(r'Experience Match:\s*\[?(\d+)%\]?', section_text, re.IGNORECASE)
                            education_match = re.search(r'Education Match:\s*\[?(\d+)%\]?', section_text, re.IGNORECASE)
                            
                            if skills_match:
                                skills_percent = skills_match.group(1)
                            if experience_match:
                                experience_percent = experience_match.group(1)
                            if education_match:
                                education_percent = education_match.group(1)
                        
                        overall_score = overall_score_match.group(1) if overall_score_match else 'N/A'
                        # --- END ENHANCED EXTRACTION LOGIC ---

                        st.session_state.candidate_match_results.append({
                            "jd_name": jd_name,
                            "overall_score": overall_score,
                            "skills_percent": skills_percent,
                            "experience_percent": experience_percent, 
                            "education_percent": education_percent,   
                            "full_analysis": fit_output
                        })
                    except Exception as e:
                        st.session_state.candidate_match_results.append({
                            "jd_name": jd_name,
                            "overall_score": "Error",
                            "skills_percent": "Error",
                            "experience_percent": "Error", 
                            "education_percent": "Error",   
                            "full_analysis": f"Error running analysis for {jd_name}: {e}\n{traceback.format_exc()}"
                        })
                st.success("Batch analysis complete!")


        # 3. Display Results
        if st.session_state.get('candidate_match_results'):
            st.markdown("#### Match Results for Your Resume")
            results_df = st.session_state.candidate_match_results
            
            # Create a simple table/summary of results
            display_data = []
            for item in results_df:
                display_data.append({
                    "Job Description": item["jd_name"].replace("--- Simulated JD for: ", ""),
                    "Fit Score (out of 10)": item["overall_score"],
                    "Skills (%)": item.get("skills_percent", "N/A"),
                    "Experience (%)": item.get("experience_percent", "N/A"), 
                    "Education (%)": item.get("education_percent", "N/A"),   
                })

            st.dataframe(display_data, use_container_width=True)

            # Display detailed analysis in expanders
            st.markdown("##### Detailed Reports")
            for item in results_df:
                header_text = f"Report for **{item['jd_name'].replace('--- Simulated JD for: ', '')}** (Score: **{item['overall_score']}/10** | S: **{item.get('skills_percent', 'N/A')}%** | E: **{item.get('experience_percent', 'N/A')}%** | Edu: **{item.get('education_percent', 'N/A')}%**)"
                with st.expander(header_text):
                    st.markdown(item['full_analysis'])


def hiring_dashboard():
    st.header("🏢 Hiring Company Dashboard")
    st.write("Manage job postings and view candidate applications. (Placeholder for future features)")
    
    # --- MODIFIED NAVIGATION BLOCK ---
    nav_col1, nav_col2 = st.columns([1, 1])

    with nav_col1:
        if st.button("⬅️ Go Back to Role Selection", use_container_width=True):
            go_to("role_selection")
            
    with nav_col2:
        if st.button("🚪 Log Out", key="hiring_logout_btn", use_container_width=True):
            go_to("login") 
    # --- END MODIFIED NAVIGATION BLOCK ---

# -------------------------
# Main App Initialization
# -------------------------
def main():
    st.set_page_config(layout="wide", page_title="PragyanAI Job Portal")

    # --- Session State Initialization ---
    if 'page' not in st.session_state:
        st.session_state.page = "login"
    
    # Initialize session state for AI features
    if 'parsed' not in st.session_state:
        st.session_state.parsed = {}
        st.session_state.full_text = ""
        st.session_state.excel_data = None
        st.session_state.qa_answer = ""
        st.session_state.iq_output = ""
        st.session_state.jd_fit_output = ""
        
        # Admin Dashboard specific lists
        st.session_state.admin_jd_list = [] 
        st.session_state.resumes_to_analyze = [] 
        st.session_state.admin_match_results = [] 
        st.session_state.resume_statuses = {} 
        
        # --- VENDOR STATE INIT ---
        st.session_state.vendors = []
        st.session_state.vendor_statuses = {}
        # --- END VENDOR STATE INIT ---
        
        # Candidate Dashboard specific lists
        st.session_state.candidate_jd_list = []
        st.session_state.candidate_match_results = []


    # --- Page Routing ---
    if st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "signup":
        signup_page()
    elif st.session_state.page == "role_selection":
        role_selection_page()
    elif st.session_state.page == "admin_dashboard":
        admin_dashboard()
    elif st.session_state.page == "candidate_dashboard":
        candidate_dashboard()
    elif st.session_state.page == "hiring_dashboard":
        hiring_dashboard()

if __name__ == '__main__':
    main()
