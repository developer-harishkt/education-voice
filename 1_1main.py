import os
import time
import datetime
import uuid
import base64
from dotenv import load_dotenv
from typing import TypedDict

# --- Pydantic for structured output ---
from pydantic import BaseModel, Field

# --- Core LangGraph and LangChain components ---
from langgraph.graph import StateGraph, END
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_tavily import TavilySearch
import requests

# --- RAG specific components ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- Firebase & Cloud Function Framework ---
import functions_framework
import firebase_admin
from firebase_admin import credentials, firestore, _apps

# ==============================================================================
# --- 1. INITIALIZATION (GLOBAL SCOPE) ---
# ==============================================================================
load_dotenv()

# --- Initialize Firebase Admin SDK ---
db = None
try:
    if not _apps:
        firebase_admin.initialize_app()
    db = firestore.client()
    print("✅ Firebase Admin SDK is ready.")
except Exception as e:
    print(f"❌ FATAL: Firebase Admin SDK initialization failed: {e}")
    db = None

# --- Initialize LLM and Tools ---
llm = ChatVertexAI(model_name="gemini-1.5-pro")

# --- MEMORY OPTIMIZATION: Switched to serverless VertexAIEmbeddings ---
try:
    embedding_model = VertexAIEmbeddings(model_name="gemini-embedding-001")
except Exception as e:
    print(f"Warning: Could not initialize VertexAI embeddings: {e}")
    # Fallback to a simpler embedding model
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

tavily_tool = TavilySearch(max_results=3)

# --- Initialize RAG Retriever ---
PDF_PATH = "source_material.pdf"
RAG_RETRIEVER = None
try:
    print("Setting up RAG pipeline...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(docs_split, embedding_model)
    RAG_RETRIEVER = vectorstore.as_retriever()
    print("✅ RAG pipeline setup complete.")
except Exception as e:
    print(f"❌ FATAL: Could not set up RAG pipeline during initialization: {e}")

# --- Progress Tracking Mapping ---
NODE_PROGRESS = {
    "Intent_Parser": 10, "RAG_Agent": 20, "Creative_Assistant": 30,
    "Enhanced_Prompt_Composer": 40, "Lesson_Generator": 55, "Quiz_Generator": 55,
    "Image_Prompt_Enhancer": 55, "hallucination_guard": 70, "Imagen_Generator": 80,
    "Final_Compiler": 85, "Evaluation_Agent": 90, "save_to_firestore_node": 100
}

# ==============================================================================
# --- 2. STATE AND DATA MODELS ---
# ==============================================================================
class Intent(BaseModel):
    topic: str = Field(description="The main subject of the lesson plan.")
    grade_level: str = Field(description="The target grade level for the lesson.")

class EvaluationReport(BaseModel):
    clarity_score: int
    clarity_feedback: str
    engagement_score: int
    engagement_feedback: str
    educational_value_score: int
    educational_value_feedback: str

class GraphState(TypedDict):
    run_id: str; user_id: str; thread_id: str
    user_request: str; topic: str; grade_level: str
    retriever: object; grounded_content: str; supplemental_content: str
    lesson_prompt: str; quiz_prompt: str; enhanced_image_prompt: str
    lesson_plan: str; compiled_lesson: str; quiz: str
    verification_report: str; image_url: str
    evaluation_report: EvaluationReport
    error: str; compilation_complete: bool

# ==============================================================================
# --- 3. HELPER FUNCTIONS ---
# ==============================================================================
def update_progress(state: GraphState, current_node: str):
    run_id, user_id, thread_id = state['run_id'], state['user_id'], state['thread_id']
    if not db or not run_id: return
    try:
        progress = NODE_PROGRESS.get(current_node, 0)
        db.collection('progress_tracking').document(run_id).update({
            "status": "In Progress", "current_node": current_node, "progress": progress,
            "user_id": user_id, "thread_id": thread_id, "last_updated": firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        print(f"❌ Failed to update progress for run {run_id}: {e}")

def generate_image_with_imagen(prompt: str) -> str:
    print("🎨 Simulating image generation...")
    return "https://storage.googleapis.com/your-bucket-name/placeholder.png"

# ==============================================================================
# --- 4. GRAPH AGENT NODES ---
# ==============================================================================
def intent_parser_node(state: GraphState):
    update_progress(state, "Intent_Parser")
    structured_llm = llm.with_structured_output(Intent)
    prompt = f"Parse the user request: \"{state['user_request']}\""
    return structured_llm.invoke(prompt)

def rag_agent_node(state: GraphState):
    update_progress(state, "RAG_Agent")
    topic = state["topic"]
    docs = state["retriever"].invoke(topic)
    content = "\n\n".join([d.page_content for d in docs])
    topic_check_prompt = f"Analyze if the text contains substantial info about '{topic}'. Return ONLY 'yes' or 'no'.\n\nText:{content[:2000]}"
    has_topic = llm.invoke(topic_check_prompt).content.lower().strip()
    if has_topic != "yes":
        return {"error": f"I apologize, but my source material does not cover '{topic}' in detail."}
    return {"grounded_content": "\n\n---\n\n".join([f"Source: Page {d.metadata.get('page', 'N/A')}\n{d.page_content}" for d in docs])}

def creative_assistant_node(state: GraphState):
    update_progress(state, "Creative_Assistant")
    synthesis_prompt = f"Create a simple, two-sentence analogy to explain '{state['topic']}' to {state['grade_level']} students in India."
    return {"supplemental_content": llm.invoke(synthesis_prompt).content.strip()}

def enhanced_prompt_composer_node(state: GraphState):
    update_progress(state, "Enhanced_Prompt_Composer")
    lesson_prompt = f"""Create a lesson plan about '{state['topic']}'. 

Primary Source Material (Facts):
---
{state['grounded_content']}
---

Creative Element (Analogy):
---
{state['supplemental_content']}
---

Task: Create a detailed lesson plan for {state['grade_level']} with a standard educational structure."""
    quiz_prompt = f"""Create a worksheet with 3-4 questions and an answer key about '{state['topic']}' for {state['grade_level']}.

Primary Source Material:
---
{state['grounded_content']}
---
"""
    return {"lesson_prompt": lesson_prompt, "quiz_prompt": quiz_prompt}

def lesson_generator_node(state: GraphState):
    update_progress(state, "Lesson_Generator")
    return {"lesson_plan": llm.invoke(state["lesson_prompt"]).content.strip()}

def quiz_generator_node(state: GraphState):
    update_progress(state, "Quiz_Generator")
    return {"quiz": llm.invoke(state["quiz_prompt"]).content.strip()}

def image_prompt_enhancer_node(state: GraphState):
    update_progress(state, "Image_Prompt_Enhancer")
    return {"enhanced_image_prompt": f"Educational illustration for a lesson on '{state['topic']}'..."}

def imagen_generator_node(state: GraphState):
    update_progress(state, "Imagen_Generator")
    return {"image_url": generate_image_with_imagen(state["enhanced_image_prompt"])}

def hallucination_guard_node(state: GraphState):
    update_progress(state, "hallucination_guard")
    verification_prompt = f"Fact-check this lesson plan. If supported, respond only with 'All claims verified.'.\n\nSource:{state['grounded_content']}\n\nLesson:{state['lesson_plan']}"
    return {"verification_report": llm.invoke(verification_prompt).content.strip()}

def final_compiler_node(state: GraphState):
    update_progress(state, "Final_Compiler")
    verification_report = state.get("verification_report", "Verification pending.")
    print(f"Final Compiler: Ack verification report ('{verification_report[:30]}...').")
    compiled_lesson = state.get("lesson_plan", "")
    if state.get("image_url"):
        compiled_lesson += f"\n\n![Visual Aid]({state['image_url']})"
    return {"compilation_complete": True, "compiled_lesson": compiled_lesson}

def evaluation_agent_node(state: GraphState):
    update_progress(state, "Evaluation_Agent")
    evaluator_llm = llm.with_structured_output(EvaluationReport)
    return {"evaluation_report": evaluator_llm.invoke("Evaluate this lesson...")}

def save_to_firestore_node(state: GraphState):
    update_progress(state, "save_to_firestore_node")
    run_id, user_id, thread_id = state['run_id'], state['user_id'], state['thread_id']
    final_status = "Complete"
    if not db: return {}
    try:
        report = state.get("evaluation_report")
        doc = {
            "user_id": user_id, "thread_id": thread_id, "run_id": run_id,
            "topic": state.get("topic", ""), "gradeLevel": state.get("grade_level", ""),
            "createdAt": firestore.SERVER_TIMESTAMP,
            "lessonMarkdown": state.get("compiled_lesson", ""),
            "quizMarkdown": state.get("quiz", ""),
            "analogy": state.get("supplemental_content", ""),
            "imageUrl": state.get("image_url", ""),
            "evaluation": report.dict() if isinstance(report, BaseModel) else {}
        }
        db.collection('users').document(user_id).collection('threads').document(thread_id).collection('lessons').document(run_id).set(doc)
    except Exception as e: final_status = "Error"
    try:
        db.collection('progress_tracking').document(run_id).update({"status": final_status, "progress": 100})
    except Exception: pass
    return {}

# ==============================================================================
# --- 5. GRAPH CONSTRUCTION ---
# ==============================================================================
builder = StateGraph(GraphState)

builder.add_node("Intent_Parser", intent_parser_node)
builder.add_node("RAG_Agent", rag_agent_node)
builder.add_node("Creative_Assistant", creative_assistant_node)
builder.add_node("Enhanced_Prompt_Composer", enhanced_prompt_composer_node)
builder.add_node("Lesson_Generator", lesson_generator_node)
builder.add_node("Quiz_Generator", quiz_generator_node)
builder.add_node("Image_Prompt_Enhancer", image_prompt_enhancer_node)
builder.add_node("Imagen_Generator", imagen_generator_node)
builder.add_node("hallucination_guard", hallucination_guard_node)
builder.add_node("Final_Compiler", final_compiler_node)
builder.add_node("Evaluation_Agent", evaluation_agent_node)
builder.add_node("save_to_firestore_node", save_to_firestore_node)

builder.set_entry_point("Intent_Parser")
builder.add_edge("Intent_Parser", "RAG_Agent")

def should_continue(state):
    if state.get("error"):
        print(f"Stopping graph due to RAG agent error: {state['error']}")
        return "end"
    return "continue"

builder.add_conditional_edges(
    "RAG_Agent", should_continue,
    {"continue": "Creative_Assistant", "end": "save_to_firestore_node"}
)
builder.add_edge("Creative_Assistant", "Enhanced_Prompt_Composer")
builder.add_edge("Enhanced_Prompt_Composer", "Lesson_Generator")
builder.add_edge("Enhanced_Prompt_Composer", "Quiz_Generator")
builder.add_edge("Enhanced_Prompt_Composer", "Image_Prompt_Enhancer")
builder.add_edge("Image_Prompt_Enhancer", "Imagen_Generator")
builder.add_edge("Lesson_Generator", "hallucination_guard")
builder.add_edge("hallucination_guard", "Final_Compiler")
builder.add_edge("Quiz_Generator", "Final_Compiler")
builder.add_edge("Imagen_Generator", "Final_Compiler")
builder.add_edge("Final_Compiler", "Evaluation_Agent")
builder.add_edge("Evaluation_Agent", "save_to_firestore_node")
builder.add_edge("save_to_firestore_node", END)

graph = builder.compile()

# ==============================================================================
# --- 6. CLOUD FUNCTION ENTRY POINT ---
# ==============================================================================
@functions_framework.http
def handler(request):
    request_json = request.get_json(silent=True)
    if not request_json or 'user_request' not in request_json or 'user_id' not in request_json:
        return ("Invalid request: JSON body must include 'user_request' and 'user_id'.", 400)

    user_request = request_json['user_request']
    user_id = request_json['user_id']
    thread_id = request_json.get('thread_id', str(uuid.uuid4()))
    run_id = str(uuid.uuid4())

    if not db or not RAG_RETRIEVER:
        return ("Internal Server Error: Backend not ready.", 500)

    if db:
        db.collection('progress_tracking').document(run_id).set({
            "status": "Accepted", "progress": 0, "current_node": "None", "user_id": user_id,
            "thread_id": thread_id, "received_at": firestore.SERVER_TIMESTAMP
        })

    initial_state = {
        "run_id": run_id, "user_id": user_id, "thread_id": thread_id,
        "user_request": user_request, "retriever": RAG_RETRIEVER,
    }

    try:
        graph.invoke(initial_state)
        return ({"status": "processing_started", "run_id": run_id, "thread_id": thread_id}, 202)
    except Exception as e:
        if db:
            db.collection('progress_tracking').document(run_id).update({"status": "Failed", "error": str(e)})
        return ({"status": "error", "run_id": run_id, "message": str(e)}, 500)