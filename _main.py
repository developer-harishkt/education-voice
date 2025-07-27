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
from langchain_google_vertexai import ChatVertexAI

# --- Firebase & Cloud Function Framework ---
import functions_framework
import firebase_admin
from firebase_admin import credentials, firestore

# ==============================================================================
# --- 1. INITIALIZATION (GLOBAL SCOPE) ---
# ==============================================================================
load_dotenv()

# Set environment variables to prevent memory issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# Memory management
import gc
gc.collect()

# --- Initialize Firebase Admin SDK ---
db = None
try:
    firebase_admin.initialize_app()
    db = firestore.client()
    print("✅ Firebase Admin SDK initialized successfully.")
except Exception:
    db = None

# --- Initialize LLM only (no embeddings to save memory) ---
llm = ChatVertexAI(model_name="gemini-2.5-pro")

# --- Mock RAG Retriever for testing ---
class MockRetriever:
    def invoke(self, query):
        return [type('obj', (object,), {
            'page_content': f"Mock educational content about {query}. This is sample content for testing purposes.",
            'metadata': {'page': 1}
        })()]

RAG_RETRIEVER = MockRetriever()
print("✅ Mock RAG pipeline setup complete.")

# --- Progress Tracking Mapping ---
NODE_PROGRESS = {
    "Intent_Parser": 15, "Creative_Assistant": 30, "Enhanced_Prompt_Composer": 45,
    "Lesson_Generator": 65, "Quiz_Generator": 65, "Image_Prompt_Enhancer": 65,
    "Imagen_Generator": 80, "Final_Compiler": 90, "save_to_firestore_node": 100
}

# ==============================================================================
# --- 2. STATE AND DATA MODELS ---
# ==============================================================================
class Intent(BaseModel):
    topic: str = Field(description="The main subject of the lesson plan.")
    grade_level: str = Field(description="The target grade level for the lesson.")

class GraphState(TypedDict):
    run_id: str; user_id: str; thread_id: str
    user_request: str; topic: str; grade_level: str
    supplemental_content: str; lesson_prompt: str; quiz_prompt: str
    enhanced_image_prompt: str; lesson_plan: str; compiled_lesson: str
    quiz: str; image_url: str; error: str; compilation_complete: bool

# ==============================================================================
# --- 3. HELPER FUNCTIONS ---
# ==============================================================================
def update_progress(state: GraphState, current_node: str):
    run_id = state.get('run_id', 'local')
    if not db or not run_id: 
        print(f"🔄 Progress (local): Node={current_node}")
        return
    try:
        progress = NODE_PROGRESS.get(current_node, 0)
        db.collection('progress_tracking').document(run_id).update({
            "status": "In Progress", "current_node": current_node, "progress": progress,
            "user_id": state['user_id'], "thread_id": state['thread_id'], 
            "last_updated": firestore.SERVER_TIMESTAMP
        })
        print(f"🔄 Progress (run:{run_id}): Node={current_node}, Progress={progress}%")
    except Exception as e:
        print(f"❌ Failed to update progress for run {run_id}: {e}")

def generate_image_placeholder(prompt: str) -> str:
    """Return a placeholder image URL to avoid memory-intensive image generation."""
    print(f"🎨 Using placeholder image for prompt: {prompt[:50]}...")
    return "https://via.placeholder.com/400x300/4CAF50/FFFFFF?text=Educational+Image"

# ==============================================================================
# --- 4. GRAPH AGENT NODES ---
# ==============================================================================
def intent_parser_node(state: GraphState):
    update_progress(state, "Intent_Parser")
    try:
        structured_llm = llm.with_structured_output(Intent)
        prompt = f"Parse this request and extract topic and grade level: '{state['user_request']}'"
        result = structured_llm.invoke(prompt)
        return {"topic": result.topic, "grade_level": result.grade_level}
    except Exception as e:
        print(f"❌ Intent parser error: {e}")
        # Fallback parsing
        request = state['user_request'].lower()
        topic = "general science" if "science" in request else "general topic"
        grade_level = "5th grade" if "5" in request else "elementary"
        return {"topic": topic, "grade_level": grade_level}

def creative_assistant_node(state: GraphState):
    update_progress(state, "Creative_Assistant")
    try:
        prompt = f"Create a simple analogy to explain '{state['topic']}' to {state['grade_level']} students."
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        return {"supplemental_content": content.strip()}
    except Exception as e:
        print(f"❌ Creative assistant error: {e}")
        return {"supplemental_content": f"Think of {state['topic']} like something familiar to students."}

def enhanced_prompt_composer_node(state: GraphState):
    update_progress(state, "Enhanced_Prompt_Composer")
    
    lesson_prompt = f"""Create a lesson plan about '{state['topic']}' for {state['grade_level']}.

Include:
1. Topic: {state['topic']}
2. Grade Level: {state['grade_level']}
3. Learning Objectives (2-3 clear goals)
4. Materials Needed
5. Lesson Activities (3-4 activities)
6. Assessment Method

Use this analogy to help explain: {state['supplemental_content']}

Keep it educational and age-appropriate."""

    quiz_prompt = f"""Create a 4-question worksheet about '{state['topic']}' for {state['grade_level']}.

Include:
- 2 multiple choice questions
- 1 short answer question  
- 1 drawing/diagram question
- Answer key at the end

Make questions appropriate for {state['grade_level']} level."""

    return {"lesson_prompt": lesson_prompt, "quiz_prompt": quiz_prompt}

def lesson_generator_node(state: GraphState):
    update_progress(state, "Lesson_Generator")
    try:
        response = llm.invoke(state["lesson_prompt"])
        content = response.content if hasattr(response, 'content') else str(response)
        return {"lesson_plan": content.strip()}
    except Exception as e:
        print(f"❌ Lesson generator error: {e}")
        return {"lesson_plan": f"# {state['topic']} Lesson Plan\n\nBasic lesson content for {state['grade_level']} students."}

def quiz_generator_node(state: GraphState):
    update_progress(state, "Quiz_Generator")
    try:
        response = llm.invoke(state["quiz_prompt"])
        content = response.content if hasattr(response, 'content') else str(response)
        return {"quiz": content.strip()}
    except Exception as e:
        print(f"❌ Quiz generator error: {e}")
        return {"quiz": f"# {state['topic']} Worksheet\n\nBasic quiz questions for {state['grade_level']} students."}

def image_prompt_enhancer_node(state: GraphState):
    update_progress(state, "Image_Prompt_Enhancer")
    enhanced_prompt = f"Educational diagram showing {state['topic']} concepts for {state['grade_level']} students"
    return {"enhanced_image_prompt": enhanced_prompt}

def imagen_generator_node(state: GraphState):
    update_progress(state, "Imagen_Generator")
    image_url = generate_image_placeholder(state.get("enhanced_image_prompt", ""))
    return {"image_url": image_url}

def final_compiler_node(state: GraphState):
    update_progress(state, "Final_Compiler")
    compiled_lesson = state.get("lesson_plan", "")
    if state.get("image_url"):
        compiled_lesson += f"\n\n![Visual Aid]({state['image_url']})"
    return {"compilation_complete": True, "compiled_lesson": compiled_lesson}

def save_to_firestore_node(state: GraphState):
    update_progress(state, "save_to_firestore_node")
    run_id, user_id, thread_id = state['run_id'], state['user_id'], state['thread_id']
    
    if not db:
        print(f"✅ Final output ready (local mode) for run: {run_id}")
        print(f"📝 Topic: {state.get('topic', 'N/A')}")
        print(f"📚 Grade: {state.get('grade_level', 'N/A')}")
        return {}
    
    try:
        doc = {
            "user_id": user_id, "thread_id": thread_id, "run_id": run_id,
            "topic": state.get("topic", ""), "gradeLevel": state.get("grade_level", ""),
            "createdAt": firestore.SERVER_TIMESTAMP,
            "lessonMarkdown": state.get("compiled_lesson", ""),
            "quizMarkdown": state.get("quiz", ""),
            "analogy": state.get("supplemental_content", ""),
            "imageUrl": state.get("image_url", ""),
            "evaluation": {"status": "completed"}
        }
        db.collection('users').document(user_id).collection('threads').document(thread_id).collection('lessons').document(run_id).set(doc)
        db.collection('progress_tracking').document(run_id).update({"status": "Complete", "progress": 100})
        print(f"✅ Saved to Firestore: {run_id}")
    except Exception as e:
        print(f"❌ Firestore save failed: {e}")
        if db:
            try:
                db.collection('progress_tracking').document(run_id).update({"status": "Error", "error": str(e)})
            except Exception:
                pass
    return {}

# ==============================================================================
# --- 5. GRAPH CONSTRUCTION ---
# ==============================================================================
builder = StateGraph(GraphState)

# Add nodes
for name, func in [
    ("Intent_Parser", intent_parser_node),
    ("Creative_Assistant", creative_assistant_node), 
    ("Enhanced_Prompt_Composer", enhanced_prompt_composer_node),
    ("Lesson_Generator", lesson_generator_node),
    ("Quiz_Generator", quiz_generator_node),
    ("Image_Prompt_Enhancer", image_prompt_enhancer_node),
    ("Imagen_Generator", imagen_generator_node),
    ("Final_Compiler", final_compiler_node),
    ("save_to_firestore_node", save_to_firestore_node)
]:
    builder.add_node(name, func)

# Define flow
builder.set_entry_point("Intent_Parser")
builder.add_edge("Intent_Parser", "Creative_Assistant")
builder.add_edge("Creative_Assistant", "Enhanced_Prompt_Composer")

# Parallel execution
builder.add_edge("Enhanced_Prompt_Composer", "Lesson_Generator")
builder.add_edge("Enhanced_Prompt_Composer", "Quiz_Generator")
builder.add_edge("Enhanced_Prompt_Composer", "Image_Prompt_Enhancer")
builder.add_edge("Image_Prompt_Enhancer", "Imagen_Generator")

# Compilation
builder.add_edge("Lesson_Generator", "Final_Compiler")
builder.add_edge("Quiz_Generator", "Final_Compiler")
builder.add_edge("Imagen_Generator", "Final_Compiler")

# Finish
builder.add_edge("Final_Compiler", "save_to_firestore_node")
builder.add_edge("save_to_firestore_node", END)

graph = builder.compile()

# ==============================================================================
# --- 6. CLOUD FUNCTION ENTRY POINT ---
# ==============================================================================
@functions_framework.http
def handler(request):
    try:
        # Force garbage collection before processing
        gc.collect()
        
        request_json = request.get_json(silent=True)
        if not request_json or 'user_request' not in request_json or 'user_id' not in request_json:
            return ("Invalid request: JSON body must include 'user_request' and 'user_id'.", 400)

        user_request = request_json['user_request']
        user_id = request_json['user_id']
        thread_id = request_json.get('thread_id', str(uuid.uuid4()))
        run_id = str(uuid.uuid4())

        print(f"🚀 Processing request: {user_request[:50]}...")

        # Create initial progress document
        if db:
            try:
                db.collection('progress_tracking').document(run_id).set({
                    "status": "Accepted", "progress": 0, "current_node": "None", 
                    "user_id": user_id, "thread_id": thread_id, 
                    "received_at": firestore.SERVER_TIMESTAMP
                })
            except Exception as e:
                print(f"❌ Failed to create progress document: {e}")

        # Prepare initial state
        initial_state = {
            "run_id": run_id, "user_id": user_id, "thread_id": thread_id,
            "user_request": user_request
        }

        # Execute graph
        try:
            result = graph.invoke(initial_state)
            print(f"✅ Graph execution completed for run: {run_id}")
            return ({"status": "processing_started", "run_id": run_id, "thread_id": thread_id}, 202)
        except Exception as e:
            print(f"❌ Graph execution failed: {e}")
            if db:
                try:
                    db.collection('progress_tracking').document(run_id).update({
                        "status": "Failed", "error": str(e)
                    })
                except Exception:
                    pass
            return ({"status": "error", "run_id": run_id, "message": str(e)}, 500)
            
    except Exception as e:
        print(f"❌ Handler error: {e}")
        return ({"status": "error", "message": "Internal server error"}, 500)
    finally:
        # Clean up memory after each request
        gc.collect()
