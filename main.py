# main.py

import os
import time
import datetime
import sqlite3
import base64
import uuid
from dotenv import load_dotenv
from typing import TypedDict, List

# --- Pydantic for structured output ---
from pydantic import BaseModel, Field

# --- Core LangGraph and LangChain components ---
from langgraph.graph import StateGraph, END
from langchain_google_vertexai import ChatVertexAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
import requests

# --- RAG specific components ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- Database Configuration ---
DB_PATH = "sahayak_memory.db"

# --- 1. Environment and Tool Setup ---
load_dotenv()
llm = ChatVertexAI(model_name="gemini-2.5-pro") # Note: User specified gemini-2.5-pro
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
tavily_tool = TavilySearchResults(max_results=3)

# --- Database Setup ---
def setup_memory_database():
    """Initializes the SQLite database and creates the 'interactions' table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            topic TEXT NOT NULL,
            grade_level TEXT NOT NULL,
            clarity_score INTEGER,
            engagement_score INTEGER,
            educational_value_score INTEGER,
            lesson_file TEXT,
            quiz_file TEXT
        );
    """)
    conn.commit()
    conn.close()
    print("✅ Database setup complete.")


# --- NEW: Image Generation Logic based on your provided tool ---
def generate_image_with_imagen(prompt: str) -> str:
    """Generates an image using the Google Imagen 4.0 API and saves it locally."""
    print("🎨 Generating image with Google Imagen 4.0...")
    
    # Securely get API key from environment variable
    api_key = os.getenv("IMAGEN_API_KEY")
    if not api_key:
        print("❌ ERROR: IMAGEN_API_KEY environment variable not set.")
        return "Image generation failed: API key not configured."

    url = "https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-generate-preview-06-06:predict"
    headers = {'x-goog-api-key': api_key, 'Content-Type': 'application/json'}
    data = {"instances": [{"prompt": prompt}], "parameters": {"sampleCount": 1}}

    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)

        if response.status_code == 200:
            result = response.json()
            if 'predictions' in result and result['predictions']:
                prediction = result['predictions'][0]
                if 'bytesBase64Encoded' in prediction:
                    image_b64 = prediction['bytesBase64Encoded']
                    image_data = base64.b64decode(image_b64)

                    # Save the image locally
                    filename = f"generated_image_{uuid.uuid4().hex[:8]}.png"
                    image_dir = "outputs/images"
                    os.makedirs(image_dir, exist_ok=True)
                    local_path = os.path.join(image_dir, filename)

                    with open(local_path, 'wb') as f:
                        f.write(image_data)
                    
                    print(f"✅ Image saved successfully to {local_path}")
                    return local_path
            
            print("⚠️ No image data found in Imagen response.")
            return "No image data found in Imagen response."
        else:
            print(f"❌ Imagen API request failed with status {response.status_code}: {response.text}")
            return f"Imagen API request failed: {response.status_code}"

    except Exception as e:
        print(f"❌ An error occurred during image generation: {e}")
        return f"Image generation failed: {e}"


# --- 2. State and Intent/Evaluation Definition ---
class Intent(BaseModel):
    topic: str = Field(description="The main subject or topic for the lesson plan.")
    grade_level: str = Field(description="The target grade level for the lesson.")

class EvaluationReport(BaseModel):
    clarity_score: int = Field(description="Clarity score (1-5), is it easy to understand for the grade level?")
    clarity_feedback: str = Field(description="Qualitative feedback on clarity.")
    engagement_score: int = Field(description="Engagement score (1-5), is the analogy and content interesting?")
    engagement_feedback: str = Field(description="Qualitative feedback on engagement.")
    educational_value_score: int = Field(description="Educational value score (1-5), does it meet learning objectives?")
    educational_value_feedback: str = Field(description="Qualitative feedback on educational value.")

class GraphState(TypedDict):
    # Input fields
    user_request: str
    topic: str
    grade_level: str
    retriever: object
    # Content generation fields
    grounded_content: str
    supplemental_content: str
    lesson_prompt: str
    quiz_prompt: str
    enhanced_image_prompt: str # New field for the enhanced prompt
    # Generated content fields
    lesson_plan: str
    compiled_lesson: str
    quiz: str
    verification_report: str
    image_url: str
    # Evaluation fields
    evaluation_report: EvaluationReport
    # Metadata and control flow
    execution_time: float
    error: str
    compilation_complete: bool
    # Memory and output fields
    lesson_filename: str
    quiz_filename: str

# --- 3. Agent Nodes ---
def intent_parser_node(state: GraphState):
    """Parses the user's request into a structured format."""
    # (No changes to this node)
    print("---NODE: INTENT PARSER---")
    structured_llm = llm.with_structured_output(Intent)
    prompt = f"Parse the following user request to extract the lesson topic and grade level.\n\nRequest: \"{state['user_request']}\""
    try:
        parsed_intent = structured_llm.invoke(prompt)
        print(f"✅ Intent Parsed: Topic='{parsed_intent.topic}', Grade='{parsed_intent.grade_level}'")
        return {"topic": parsed_intent.topic, "grade_level": parsed_intent.grade_level}
    except Exception as e:
        return {"error": f"Failed to parse user request: {e}"}

def rag_agent_node(state: GraphState):
    """Retrieves content from source material and verifies topic relevance."""
    # (No changes to this node)
    print("---NODE: RAG AGENT---")
    topic, retriever = state["topic"], state["retriever"]
    docs = retriever.invoke(topic)
    content = "\n\n".join([d.page_content for d in docs])
    topic_check_prompt = f"Analyze if the following text contains substantial information about '{topic}'. Look for direct discussions, not brief mentions. Return ONLY 'yes' or 'no'.\n\nText: {content[:2000]}"
    has_topic = llm.invoke(topic_check_prompt).content.lower().strip()
    if has_topic != "yes":
        print(f"⚠️ Warning: The source material does not contain information about '{topic}'")
        return {"error": f"I apologize, but my source material does not cover '{topic}' in detail."}
    return {"grounded_content": "\n\n---\n\n".join([f"Source: Page {d.metadata.get('page', 'N/A')}\n{d.page_content}" for d in docs])}

# --- NEW: Image Prompt Enhancer Node ---
def image_prompt_enhancer_node(state: GraphState):
    """Enhances a basic image prompt with educational context."""
    print("---NODE: IMAGE PROMPT ENHANCER---")
    topic = state['topic']
    grade_level = state['grade_level']
    
    enhanced_prompt = f"""
    Educational image for a lesson on '{topic}' for {grade_level} students.
    
    Style: Clear, simple, educational diagram or illustration with clean lines. High contrast and easy to understand.
    Quality: High resolution, focus on educational accuracy and visual clarity.
    Format: Minimalist, suitable for a textbook or worksheet.
    """
    return {"enhanced_image_prompt": enhanced_prompt.strip()}


# --- UPDATED: Image Generator Node ---
def imagen_generator_node(state: GraphState):
    """Generates an image using the enhanced prompt and Imagen 4.0."""
    print("---NODE: IMAGEN GENERATOR---")
    enhanced_prompt = state.get("enhanced_image_prompt")
    if not enhanced_prompt:
        return {"error": "Cannot generate image without an enhanced prompt."}
    
    image_path = generate_image_with_imagen(enhanced_prompt)
    return {"image_url": image_path}

def creative_assistant_node(state: GraphState):
    """Generates a culturally relevant analogy."""
    # (No changes to this node)
    print("---NODE: CREATIVE ASSISTANT---")
    topic, grade = state["topic"], state["grade_level"]
    query_prompt = f"Generate a search query for culturally relevant analogies to teach '{topic}' to {grade} students in India."
    query = llm.invoke(query_prompt).content.strip()
    search_results = tavily_tool.invoke(query)
    formatted_results = [res.get('content', '') for res in search_results if isinstance(res, dict) and res.get('content')]
    results_content = "\n\n".join(formatted_results) if formatted_results else "No relevant results."
    synthesis_prompt = f"Create a simple, two-sentence analogy to explain '{topic}' to {grade} students in India, using these search results or common Indian life if results are unhelpful:\n\n{results_content}"
    synthesis = llm.invoke(synthesis_prompt).content.strip()
    return {"supplemental_content": synthesis}

# ... (The following nodes remain the same: enhanced_prompt_composer_node, lesson_generator_node, quiz_generator_node, hallucination_guard_node, final_compiler_node, evaluation_agent_node, save_outputs_node, memory_agent_node) ...

def enhanced_prompt_composer_node(state: GraphState):
    print("---NODE: ENHANCED PROMPT COMPOSER---")
    grounded_content, supplemental_content, topic, grade_level = state["grounded_content"], state["supplemental_content"], state["topic"], state["grade_level"]
    lesson_prompt = f"""Create a lesson plan about '{topic}'. 

Primary Source Material (Facts):
---
{grounded_content}
---

Creative Element (Analogy):
---
{supplemental_content}
---

Task: Create a detailed lesson plan for {grade_level} with the following structure:
1. Topic
2. Target Grade ({grade_level})
3. Objectives
4. Materials
5. Introduction (incorporate the creative analogy)
6. Activities
7. Assessment

Important Guidelines:
- Use only facts explicitly stated in the source material
- Maintain professional, academic language throughout
- Format in clean markdown
- Start directly with the topic, no introductory text
"""
    quiz_prompt = f"""Create a worksheet about '{topic}' for {grade_level}.

Primary Source Material:
---
{grounded_content}
---

Requirements:
- Create 3-4 questions appropriate for a {grade_level} understanding of the topic
- Include an answer key
- Base all questions strictly on the source material
- Format in clean markdown
- Start directly with the worksheet title
"""
    return {"lesson_prompt": lesson_prompt, "quiz_prompt": quiz_prompt}

def lesson_generator_node(state: GraphState):
    print("---NODE: LESSON GENERATOR---")
    prompt = f"""You are a professional curriculum writer who produces clean, direct markdown content. Start directly with the content. No meta-commentary or explanations. Pure markdown content only.
    
    {state["lesson_prompt"]}"""
    lesson_plan = llm.invoke(prompt).content.strip()
    return {"lesson_plan": lesson_plan}

def quiz_generator_node(state: GraphState):
    print("---NODE: QUIZ GENERATOR---")
    quiz = llm.invoke(state["quiz_prompt"]).content
    return {"quiz": quiz}

def hallucination_guard_node(state: GraphState):
    print("---NODE: HALLUCINATION GUARD---")
    lesson_plan, grounded_content = state["lesson_plan"], state["grounded_content"]
    verification_prompt = f"Fact-check the 'Lesson Plan' against the 'Source Text'. If all claims are supported, respond with 'All claims verified.'. Otherwise, list unsupported claims.\n\nSource Text:\n{grounded_content}\n\nLesson Plan:\n{lesson_plan}"
    return {"verification_report": llm.invoke(verification_prompt).content}

def final_compiler_node(state: GraphState):
    print("---NODE: FINAL COMPILER---")
    if state.get("compilation_complete"): return state
    if state.get("error"): return state
    required_inputs = {"lesson_plan": state.get("lesson_plan"), "quiz": state.get("quiz"), "image_url": state.get("image_url"), "verification_report": state.get("verification_report")}
    missing_inputs = [k for k, v in required_inputs.items() if not v or (isinstance(v, str) and v.strip() == "")]
    if missing_inputs: return state
    print("✅ All inputs received, proceeding with compilation")
    try:
        compiled_lesson = required_inputs["lesson_plan"]
        if required_inputs["image_url"] and required_inputs["image_url"] != "No image generated.":
            compiled_lesson += f"\n\n### Visual Aid Suggestion\n\n![{state.get('topic')}]({required_inputs['image_url']})\n"
        return {"compilation_complete": True, "compiled_lesson": compiled_lesson}
    except Exception as e: return {"error": f"Compilation failed: {e}"}

def evaluation_agent_node(state: GraphState):
    print("---NODE: EVALUATION AGENT---")
    if state.get("evaluation_report"): return {}
    if not state.get("compilation_complete"): return {}
    lesson_plan = state.get("compiled_lesson", state.get("lesson_plan"))
    quiz, grade_level = state.get("quiz"), state.get("grade_level")
    if not all([lesson_plan, quiz, grade_level]): return {"evaluation_report": {"error": "Missing content for evaluation."}}
    evaluator_llm = llm.with_structured_output(EvaluationReport)
    prompt = f"You are an expert curriculum reviewer. Evaluate the following educational content created for {grade_level} students based on Clarity, Engagement, and Educational Value (1-5 scale).\n\nLesson Plan:\n{lesson_plan}\n\nQuiz:\n{quiz}"
    try:
        report = evaluator_llm.invoke(prompt)
        print(f"✅ Evaluation Complete: Clarity={report.clarity_score}/5, Engagement={report.engagement_score}/5")
        return {"evaluation_report": report}
    except Exception as e: return {"evaluation_report": {"error": f"Failed to generate evaluation report: {e}"}}

def save_outputs_node(state: GraphState):
    print("---NODE: SAVING OUTPUTS---")
    if state.get("error"): return state
    output_dir, timestamp = "outputs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    topic_slug = state['topic'].lower().replace(" ", "_").replace(":", "")
    report = state.get("evaluation_report")
    evaluation_content = ""
    if report and not (isinstance(report, dict) and report.get("error")):
        evaluation_content = f"\n---\n### Evaluation Report\n---\n- **Clarity Score**: {report.clarity_score}/5\n- **Feedback**: {report.clarity_feedback}\n- **Engagement Score**: {report.engagement_score}/5\n- **Feedback**: {report.engagement_feedback}\n- **Educational Value Score**: {report.educational_value_score}/5\n- **Feedback**: {report.educational_value_feedback}"
    verification_report = f"\n\n---\n### Verification Report\n---\n{state['verification_report']}"
    perf_report = f"\n\n---\n### Performance Report\n---\nTotal Execution Time: {state['execution_time']:.2f} seconds"
    lesson_content = f"{state.get('compiled_lesson', state['lesson_plan'])}{evaluation_content}{verification_report}{perf_report}"
    lesson_filename = os.path.join(output_dir, f"{timestamp}_{topic_slug}_lesson.md")
    quiz_filename = os.path.join(output_dir, f"{timestamp}_{topic_slug}_quiz.md")
    try:
        with open(lesson_filename, "w", encoding="utf-8") as f: f.write(lesson_content)
        print(f"✅ Lesson plan saved to {lesson_filename}")
        with open(quiz_filename, "w", encoding="utf-8") as f: f.write(state['quiz'])
        print(f"✅ Quiz saved to {quiz_filename}")
        return {"lesson_filename": lesson_filename, "quiz_filename": quiz_filename}
    except Exception as e: return {"error": f"Error saving files: {e}"}

def memory_agent_node(state: GraphState):
    print("---NODE: MEMORY AGENT---")
    if not state.get("compilation_complete"): return {}
    try:
        topic, grade_level, report = state.get("topic"), state.get("grade_level"), state.get("evaluation_report")
        if not all([topic, grade_level, report]) or (isinstance(report, dict) and report.get("error")): return {}
        lesson_file, quiz_file = state.get("lesson_filename"), state.get("quiz_filename")
        if not all([lesson_file, quiz_file]): return {} # Ensure filenames are present
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO interactions (topic, grade_level, clarity_score, engagement_score, educational_value_score, lesson_file, quiz_file) VALUES (?, ?, ?, ?, ?, ?, ?);",
                       (topic, grade_level, report.clarity_score, report.engagement_score, report.educational_value_score, lesson_file, quiz_file))
        conn.commit()
        conn.close()
        print(f"✅ Memory Agent: Saved to database.")
    except Exception as e: print(f"❌ Memory Agent: Error saving to database: {e}")
    return {}

# --- 4. Graph Construction ---
builder = StateGraph(GraphState)
builder.add_node("Intent_Parser", intent_parser_node)
builder.add_node("RAG_Agent", rag_agent_node)
builder.add_node("Creative_Assistant", creative_assistant_node)
builder.add_node("Enhanced_Prompt_Composer", enhanced_prompt_composer_node)
builder.add_node("Lesson_Generator", lesson_generator_node)
builder.add_node("Quiz_Generator", quiz_generator_node)
# --- UPDATED GRAPH NODES ---
builder.add_node("Image_Prompt_Enhancer", image_prompt_enhancer_node)
builder.add_node("Imagen_Generator", imagen_generator_node)
# ---
builder.add_node("hallucination_guard", hallucination_guard_node)
builder.add_node("Final_Compiler", final_compiler_node)
builder.add_node("Evaluation_Agent", evaluation_agent_node)
builder.add_node("Memory_Agent", memory_agent_node)
builder.add_node("save_outputs_node", save_outputs_node)

builder.set_entry_point("Intent_Parser")
builder.add_edge("Intent_Parser", "RAG_Agent")

def should_continue(state):
    return "error" not in state or state["error"] is None

builder.add_conditional_edges("RAG_Agent", should_continue, {True: "Enhanced_Prompt_Composer", False: "save_outputs_node"})

builder.add_edge("Enhanced_Prompt_Composer", "Creative_Assistant")
builder.add_edge("Enhanced_Prompt_Composer", "Lesson_Generator")
builder.add_edge("Enhanced_Prompt_Composer", "Quiz_Generator")
# --- UPDATED GRAPH FLOW FOR IMAGES ---
builder.add_edge("Enhanced_Prompt_Composer", "Image_Prompt_Enhancer")
builder.add_edge("Image_Prompt_Enhancer", "Imagen_Generator")
# ---
builder.add_edge("Creative_Assistant", "Final_Compiler") # Creative assistant feeds into compiler
builder.add_edge("Lesson_Generator", "hallucination_guard")
builder.add_edge("hallucination_guard", "Final_Compiler")
builder.add_edge("Quiz_Generator", "Final_Compiler")
builder.add_edge("Imagen_Generator", "Final_Compiler")

builder.add_edge("Final_Compiler", "Evaluation_Agent")
builder.add_edge("Evaluation_Agent", "save_outputs_node")
builder.add_edge("save_outputs_node", "Memory_Agent")
builder.add_edge("Memory_Agent", END)

graph = builder.compile()

# --- 5. RAG Pipeline Setup ---
def setup_rag_pipeline(source_document_path: str):
    """Sets up the RAG pipeline from a single source document."""
    loader = PyPDFLoader(source_document_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(docs_split, embedding_model)
    return vectorstore.as_retriever()

# --- 6. Invocation and Testing ---
# (Removing voice logic as per focus on core functionality)
if __name__ == "__main__":
    setup_memory_database()
    pdf_path = "source_material.pdf"
    try:
        rag_retriever = setup_rag_pipeline(pdf_path)
    except Exception as e:
        print(f"❌ Error setting up RAG pipeline: {e}")
        exit(1)

    print("\n\n--- Welcome to Project Sahayak ---")
    user_input = input("Hello! What lesson can I prepare for you today?\n> ")

    if not user_input.strip():
        print("No input received. Exiting.")
        exit(1)

    initial_state = {
        "user_request": user_input,
        "retriever": rag_retriever,
    }

    print("\n🚀 Starting Graph Execution... 🚀")
    start_time = time.time()
    final_state = graph.invoke(initial_state)
    end_time = time.time()

    if "error" in final_state and final_state["error"]:
        print(f"\n--- ❗ERROR OCCURRED ---\n{final_state['error']}")
    else:
        # Manually set final time and save
        final_state['execution_time'] = end_time - start_time
        save_outputs_node(final_state)
        print("\n\n✅✅✅ --- Content Generation Complete --- ✅✅✅")