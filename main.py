# main.py

import os
import time
import datetime
import sqlite3
from dotenv import load_dotenv
from typing import TypedDict, List

# --- Pydantic for structured output ---
from pydantic import BaseModel, Field

# --- Core LangGraph and LangChain components ---
from langgraph.graph import StateGraph, END
from langchain_google_vertexai import ChatVertexAI, VertexAIImageGeneratorChat
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
import requests
import re
import json

# --- Firestore imports ---
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    from google.cloud.firestore import Client
    FIRESTORE_AVAILABLE = True
except ImportError:
    print("⚠️ Firebase Admin SDK not available. Firestore will be disabled.")
    FIRESTORE_AVAILABLE = False

# --- RAG specific components ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- 1. Environment Setup ---
load_dotenv()

# --- Database Configuration ---
DB_PATH = "sahayak_memory.db"

# --- Firestore Configuration ---
FIRESTORE_ENABLED = os.getenv('FIRESTORE_ENABLED', 'false').lower() == 'true'
FIRESTORE_PROJECT_ID = os.getenv('FIRESTORE_PROJECT_ID', '')
FIRESTORE_COLLECTION = os.getenv('FIRESTORE_COLLECTION', 'lessons')

# --- Tool Setup ---
llm = ChatVertexAI(model_name="gemini-2.5-pro")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
tavily_tool = TavilySearchResults(max_results=3)

# Initialize image generation with fallback options
try:
    image_generation_tool = VertexAIImageGeneratorChat(model="imagegeneration@006")
    print("✅ Vertex AI Image Generation initialized")
except Exception as e:
    print(f"⚠️ Vertex AI Image Generation not available: {e}")
    print("ℹ️ Will use Hugging Face Stable Diffusion API as fallback")
    image_generation_tool = None

# --- NEW: Database Setup ---
def setup_memory_database():
    """Initializes the SQLite database and creates the 'interactions' table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create the interactions table if it doesn't exist
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


# --- NEW: JSON Data Models for Direct LLM Generation ---
class LessonContent(BaseModel):
    """Structured lesson content for frontend consumption."""
    title: str = Field(description="The title of the lesson")
    objectives: List[str] = Field(description="List of learning objectives")
    materials: List[str] = Field(description="List of required materials")
    introduction: str = Field(description="Introduction text for the lesson")
    activities: List[dict] = Field(description="List of activities with title and description")
    assessment: str = Field(description="Assessment description")
    visualAid: dict = Field(description="Visual aid information with imageUrl and description")


class QuizContent(BaseModel):
    """Structured quiz content for frontend consumption."""
    title: str = Field(description="The title of the quiz")
    questions: List[dict] = Field(description="List of questions with options and correct answer")
    answerKey: List[dict] = Field(description="Answer key with question index and correct answer")





def generate_image_with_fallback(prompt: str) -> str:
    # Function body remains the same as previous version...
    if image_generation_tool:
        try:
            print("Attempting Vertex AI image generation...")
            urls = image_generation_tool.invoke(prompt)
            if urls and urls[0]: return urls[0]
        except Exception as e: print(f"⚠️ Vertex AI Image Generation failed: {e}")
    try:
        print("Attempting Hugging Face image generation...")
        models = ["CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1", "runwayml/stable-diffusion-v1-5"]
        headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}", "Content-Type": "application/json"}
        formatted_prompt = f"educational textbook illustration, black and white line drawing, {prompt}, simple clean lines, labeled diagram style, high contrast, minimalist, technical drawing"
        for model in models:
            try:
                print(f"Trying model: {model}")
                api_url = f"https://api-inference.huggingface.co/models/{model}"
                response = requests.post(api_url, headers=headers, json={"inputs": formatted_prompt, "options": {"wait_for_model": True}, "parameters": {"negative_prompt": "color, photorealistic, complex, messy, blurry", "num_inference_steps": 30, "guidance_scale": 7.5,}}, timeout=30)
                if response.status_code == 200:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_dir = "outputs/images"
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = f"{image_dir}/{timestamp}_{model.split('/')[-1]}_generated.png"
                    with open(image_path, "wb") as f: f.write(response.content)
                    print(f"✅ Image generated using {model} and saved to {image_path}")
                    return image_path
                else:
                    print(f"⚠️ Model {model} failed with status {response.status_code}")
                    if response.text: print(f"Error details: {response.text}")
            except Exception as e: print(f"⚠️ Error with model {model}: {e}")
        return "No image generated."
    except Exception as e:
        print(f"⚠️ Hugging Face Image Generation failed: {e}")
        return "No image generated."


# --- 2. State and Intent/Evaluation Definition ---

class Intent(BaseModel):
    """The structured output for the user's request."""
    topic: str = Field(description="The main subject or topic for the lesson plan.")
    grade_level: str = Field(description="The target grade level for the lesson.")

# --- Evaluation Agent Components ---
class EvaluationReport(BaseModel):
    """A structured evaluation of the generated educational content."""
    clarity_score: int = Field(description="Clarity score (1-5), is it easy to understand for the grade level?")
    clarity_feedback: str = Field(description="Qualitative feedback on clarity.")
    engagement_score: int = Field(description="Engagement score (1-5), is the analogy and content interesting?")
    engagement_feedback: str = Field(description="Qualitative feedback on engagement.")
    educational_value_score: int = Field(description="Educational value score (1-5), does it meet learning objectives?")
    educational_value_feedback: str = Field(description="Qualitative feedback on educational value.")

class GraphState(TypedDict):
    """The state of the graph."""
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
    
    # Generated content fields
    lesson_plan: str
    compiled_lesson: str
    quiz: str
    verification_report: str
    image_url: str
    
    # Structured JSON content for frontend
    lesson_content_json: LessonContent
    quiz_content_json: QuizContent
    
    # Evaluation fields
    evaluation_report: EvaluationReport
    
    # Metadata fields
    execution_time: float
    error: str
    
    # Memory and output fields
    lesson_filename: str
    quiz_filename: str
    
    # Control flow fields
    compilation_complete: bool


# --- NEW: Firestore Functions ---
def init_firestore():
    """Initialize Firestore connection."""
    if not FIRESTORE_ENABLED:
        return None
    
    try:
        if not firebase_admin._apps:
            firebase_admin.initialize_app()
        db = firestore.client()
        print("✅ Firestore initialized successfully")
        return db
    except Exception as e:
        print(f"⚠️ Firestore initialization failed: {e}")
        return None


def create_firestore_document(state: GraphState) -> dict:
    """Create Firestore document from state data."""
    try:
        # Extract data from state
        topic = state.get("topic", "")
        grade_level = state.get("grade_level", "")
        evaluation_report = state.get("evaluation_report")
        lesson_content = state.get("lesson_content_json")
        quiz_content = state.get("quiz_content_json")
        analogy = state.get("supplemental_content", "")
        execution_time = state.get("execution_time", 0.0)
        lesson_filename = state.get("lesson_filename", "")
        quiz_filename = state.get("quiz_filename", "")
        
        # Create document structure
        document = {
            "topic": topic,
            "gradeLevel": grade_level,
            "createdAt": datetime.datetime.now().isoformat(),
            "analogy": analogy,
            "metadata": {
                "executionTime": execution_time,
                "lessonFilename": lesson_filename,
                "quizFilename": quiz_filename
            }
        }
        
        # Add lesson content if available
        if lesson_content:
            document["lessonContent"] = lesson_content.model_dump()
        
        # Add quiz content if available
        if quiz_content:
            document["quizContent"] = quiz_content.model_dump()
        
        # Add evaluation if available
        if evaluation_report and not isinstance(evaluation_report, dict):
            document["evaluation"] = {
                "clarity_score": evaluation_report.clarity_score,
                "clarity_feedback": evaluation_report.clarity_feedback,
                "engagement_score": evaluation_report.engagement_score,
                "engagement_feedback": evaluation_report.engagement_feedback,
                "educational_value_score": evaluation_report.educational_value_score,
                "educational_value_feedback": evaluation_report.educational_value_feedback
            }
        
        return document
        
    except Exception as e:
        print(f"⚠️ Error creating Firestore document: {e}")
        return None


def save_to_firestore(state: GraphState) -> bool:
    """Save lesson data to Firestore."""
    if not FIRESTORE_ENABLED:
        return False
    
    try:
        db = init_firestore()
        if not db:
            return False
        
        document = create_firestore_document(state)
        if not document:
            return False
        
        # Save to Firestore
        doc_ref = db.collection(FIRESTORE_COLLECTION).add(document)
        print(f"✅ Saved to Firestore with ID: {doc_ref[1].id}")
        return True
        
    except Exception as e:
        print(f"⚠️ Error saving to Firestore: {e}")
        return False


# --- 3. Agent Nodes ---

def intent_parser_node(state: GraphState):
    """Parses the user's request into a structured format."""
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
    print("---NODE: RAG AGENT---")
    topic, retriever = state["topic"], state["retriever"]
    docs = retriever.invoke(topic)
    
    # Check if we got meaningful content about the topic
    content = "\n\n".join([d.page_content for d in docs])
    topic_check_prompt = f"""Analyze if the following text contains substantial information about '{topic}'.
    
    Text: {content[:2000]}  # Using first 2000 chars for better context
    
    Requirements:
    - Look for direct mentions or clear discussions of the topic
    - Do not count metaphorical or analogical uses
    - Do not count brief mentions without substantial information
    - Consider only factual content, not interpretations
    
    Return ONLY 'yes' or 'no'."""
    
    has_topic = llm.invoke(topic_check_prompt).content.lower().strip()
    
    if has_topic != "yes":
        print(f"⚠️ Warning: The source material does not contain information about '{topic}'")
        return {
            "error": f"I apologize, but I don't have enough reliable information about '{topic}' in my source material. "
                    "I can help you with topics related to plant nutrition, microorganisms, and their role in agriculture."
        }
    
    return {"grounded_content": "\n\n---\n\n".join([f"Source: Page {d.metadata.get('page', 'N/A')}\n{d.page_content}" for d in docs])}

def image_generator_node(state: GraphState):
    print("---NODE: IMAGE GENERATOR---")
    try: return {"image_url": generate_image_with_fallback(f"Educational diagram about {state['topic']}.")}
    except Exception as e: return {"image_url": "No image generated."}

def creative_assistant_node(state: GraphState):
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
- Do not include statistics or percentages unless directly quoted
- Maintain professional, academic language throughout
- Avoid conversational phrases or greetings
- Format in clean markdown
- Start directly with the topic, no introductory text
- Do not include any meta-commentary or notes about the lesson plan
- Keep language concise and direct"""

    quiz_prompt = f"""Create a worksheet about '{topic}' for {grade_level}.

Primary Source Material:
---
{grounded_content}
---

Requirements:
- Create 3-4 questions appropriate for a {grade_level} understanding of the topic
- Include an answer key
- Base all questions strictly on the source material
- Do not include any conversational language
- Format in clean markdown
- Start directly with the worksheet title
- No introductory text or meta-commentary
- Keep language concise and direct"""

    return {"lesson_prompt": lesson_prompt, "quiz_prompt": quiz_prompt}

def lesson_generator_node(state: GraphState):
    """Generates the lesson plan with strict formatting."""
    print("---NODE: LESSON GENERATOR---")
    
    # Add a system message to enforce clean output
    prompt = f"""You are a professional curriculum writer who produces clean, direct markdown content.
    
    Important:
    - Start directly with the content
    - No meta-commentary or explanations
    - No "Here's", "I've created", or similar phrases
    - No notes about the lesson plan
    - Pure markdown content only
    - Return content as a single string, not a list
    
    {state["lesson_prompt"]}"""
    
    lesson_plan = llm.invoke(prompt).content
    
    # Handle case where content might be a list
    if isinstance(lesson_plan, list):
        lesson_plan = "\n".join(str(item) for item in lesson_plan)
    
    # Ensure we have a string
    lesson_plan = str(lesson_plan).strip()
    
    # Remove any common conversational starters
    if lesson_plan.lower().startswith(("here", "i have", "i've", "sure", "certainly", "this lesson")):
        lesson_plan = "\n".join(lesson_plan.split("\n")[1:]).strip()
    
    return {"lesson_plan": lesson_plan}


def lesson_json_generator_node(state: GraphState):
    """Generates structured JSON lesson content for frontend."""
    print("---NODE: LESSON JSON GENERATOR---")
    
    # Get the markdown lesson content
    lesson_plan = state.get("lesson_plan", "")
    topic = state.get("topic", "")
    grade_level = state.get("grade_level", "")
    
    if not lesson_plan:
        print("⚠️ No lesson plan available for JSON generation")
        return {}
    
    # Create structured prompt for JSON generation
    json_prompt = f"""Convert the following lesson plan into structured JSON format for frontend consumption.

Lesson Plan:
{lesson_plan}

Topic: {topic}
Grade Level: {grade_level}

Generate a structured JSON object with the following format:
{{
  "title": "Lesson title extracted from the content",
  "objectives": ["objective 1", "objective 2", "objective 3"],
  "materials": ["material 1", "material 2", "material 3"],
  "introduction": "Introduction text from the lesson",
  "activities": [
    {{
      "title": "Activity title",
      "description": "Activity description"
    }}
  ],
  "assessment": "Assessment description",
  "visualAid": {{
    "imageUrl": "",
    "description": "Visual aid description if mentioned"
  }}
}}

Important:
- Extract objectives from the lesson plan
- Extract materials from the lesson plan
- Extract activities with their titles and descriptions
- Extract assessment information
- Keep the JSON structure clean and valid
- If any field is not found in the lesson, use appropriate default values
"""
    
    try:
        # Use structured output to generate JSON
        structured_llm = llm.with_structured_output(LessonContent)
        lesson_content_json = structured_llm.invoke(json_prompt)
        
        print("✅ Structured lesson JSON generated successfully")
        return {"lesson_content_json": lesson_content_json}
        
    except Exception as e:
        print(f"⚠️ Error generating lesson JSON: {e}")
        return {}


def quiz_json_generator_node(state: GraphState):
    """Generates structured JSON quiz content for frontend."""
    print("---NODE: QUIZ JSON GENERATOR---")
    
    # Get the markdown quiz content
    quiz = state.get("quiz", "")
    topic = state.get("topic", "")
    
    if not quiz:
        print("⚠️ No quiz available for JSON generation")
        return {}
    
    # Create structured prompt for JSON generation
    json_prompt = f"""Convert the following quiz into structured JSON format for frontend consumption.

Quiz:
{quiz}

Topic: {topic}

Generate a structured JSON object with the following format:
{{
  "title": "Quiz title",
  "questions": [
    {{
      "question": "Question text",
      "type": "multiple_choice",
      "options": ["option A", "option B", "option C", "option D"],
      "answer": "correct answer"
    }}
  ],
  "answerKey": [
    {{
      "questionIndex": 0,
      "correctAnswer": "correct answer",
      "explanation": "explanation for the answer"
    }}
  ]
}}

Important:
- Extract questions and their options
- Identify correct answers
- Create answer key with explanations
- Keep the JSON structure clean and valid
- If any field is not found in the quiz, use appropriate default values
"""
    
    try:
        # Use structured output to generate JSON
        structured_llm = llm.with_structured_output(QuizContent)
        quiz_content_json = structured_llm.invoke(json_prompt)
        
        print("✅ Structured quiz JSON generated successfully")
        return {"quiz_content_json": quiz_content_json}
        
    except Exception as e:
        print(f"⚠️ Error generating quiz JSON: {e}")
        return {}

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
    """Compiles the final lesson plan by embedding the image URL."""
    print("---NODE: FINAL COMPILER---")
    
    # Check if we've already compiled
    if state.get("compilation_complete"):
        print("⏭️ Compilation already complete, skipping...")
        return state
    
    # Check for any error state
    if state.get("error"):
        print(f"❌ Cannot compile due to error: {state['error']}")
        return state
    
    # Wait for all required components
    required_inputs = {
        "lesson_plan": state.get("lesson_plan"),
        "quiz": state.get("quiz"),
        "image_url": state.get("image_url"),
        "verification_report": state.get("verification_report")
    }
    
    # If any required input is missing or empty
    missing_inputs = [k for k, v in required_inputs.items() if not v or (isinstance(v, str) and v.strip() == "")]
    if missing_inputs:
        print(f"⏳ Waiting for inputs: {missing_inputs}")
        return state
    
    print("✅ All inputs received, proceeding with compilation")
    
    try:
        lesson_plan = required_inputs["lesson_plan"]
        image_url = required_inputs["image_url"]
        topic = state.get("topic")
        
        # Check verification report
        verification_report = required_inputs["verification_report"]
        if verification_report.lower() != "all claims verified.":
            print("⚠️ Warning: Some claims could not be verified")
            print(f"Verification Report: {verification_report}")
        
        # Create a compiled version instead of modifying lesson_plan directly
        compiled_lesson = lesson_plan
        
        # Embed the image link into the lesson plan using Markdown
        if image_url and image_url != "No image generated.":
            visual_aid_section = f"\n\n### Visual Aid Suggestion\n\n![{topic}]({image_url})\n"
            compiled_lesson += visual_aid_section
        
        return {
            "compilation_complete": True,
            "compiled_lesson": compiled_lesson  # New key for the compiled version
        }
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        return {"error": f"Compilation failed: {e}"}

def evaluation_agent_node(state: GraphState):
    """The 'critic' agent that assesses the generated content against a rubric."""
    print("---NODE: EVALUATION AGENT---")
    
    # Skip if we already have an evaluation report
    if state.get("evaluation_report"):
        print("⏭️ Evaluation already complete, skipping...")
        return {}  # Return empty dict if no changes
    
    # Only proceed if compilation is complete
    if not state.get("compilation_complete"):
        print("⏳ Waiting for compilation to complete...")
        return {}  # Return empty dict if waiting
    
    # Ensure all required content is present
    lesson_plan = state.get("compiled_lesson", state.get("lesson_plan"))  # Use compiled version if available
    quiz = state.get("quiz")
    grade_level = state.get("grade_level")

    if not all([lesson_plan, quiz, grade_level]):
        print("❌ Missing required content for evaluation")
        return {"evaluation_report": {"error": "Missing content for evaluation."}}

    evaluator_llm = llm.with_structured_output(EvaluationReport)
    
    prompt = f"""
    You are an expert curriculum reviewer. Evaluate the following educational content created for {grade_level} students.
    
    **Rubric:**
    1.  **Clarity:** Is the language and structure clear and appropriate for the grade level?
    2.  **Engagement:** Is the content, especially the analogy and activities, engaging and interesting?
    3.  **Educational Value:** Does the lesson plan effectively teach the topic and meet its objectives?

    **Content to Evaluate:**
    ---
    **Lesson Plan:**
    {lesson_plan}
    ---
    **Quiz:**
    {quiz}
    ---
    
    Provide a score (1-5, 5 being best) and concise feedback for each category.
    """
    
    try:
        report = evaluator_llm.invoke(prompt)
        print(f"✅ Evaluation Complete: Clarity={report.clarity_score}/5, Engagement={report.engagement_score}/5")
        return {"evaluation_report": report}  # Only return the evaluation report
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return {"evaluation_report": {"error": f"Failed to generate evaluation report: {e}"}}

# --- NEW: Save and Memory Nodes ---
def save_outputs_node(state: GraphState):
    """Saves the lesson plan and quiz to files and adds their paths to the state."""
    print("---NODE: SAVING OUTPUTS---")
    
    # Skip if there's an error
    if state.get("error"):
        print(f"❌ Cannot save outputs due to error: {state['error']}")
        return state
        
    output_dir = "outputs"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    topic_slug = state['topic'].lower().replace(" ", "_").replace(":", "")
    
    report = state.get("evaluation_report")
    evaluation_content = "### Evaluation Report\n---\nNo evaluation was generated."
    if report and not (isinstance(report, dict) and report.get("error")):
        evaluation_content = f"\n---\n### Evaluation Report\n---\n- **Clarity Score**: {report.clarity_score}/5\n- **Feedback**: {report.clarity_feedback}\n- **Engagement Score**: {report.engagement_score}/5\n- **Feedback**: {report.engagement_feedback}\n- **Educational Value Score**: {report.educational_value_score}/5\n- **Feedback**: {report.educational_value_feedback}"
    
    verification_report = f"\n\n---\n### Verification Report\n---\n{state['verification_report']}"
    perf_report = f"\n\n---\n### Performance Report\n---\nTotal Execution Time: {state['execution_time']:.2f} seconds"
    
    # Use compiled_lesson instead of lesson_plan
    lesson_content = f"{state.get('compiled_lesson', state['lesson_plan'])}{evaluation_content}{verification_report}{perf_report}"
    lesson_filename = os.path.join(output_dir, f"{timestamp}_{topic_slug}_lesson.md")
    quiz_filename = os.path.join(output_dir, f"{timestamp}_{topic_slug}_quiz.md")
    
    try:
        with open(lesson_filename, "w", encoding="utf-8") as f: f.write(lesson_content)
        print(f"✅ Lesson plan saved to {lesson_filename}")
        with open(quiz_filename, "w", encoding="utf-8") as f: f.write(state['quiz'])
        print(f"✅ Quiz saved to {quiz_filename}")
        return {
            **state,  # Preserve existing state
            "lesson_filename": lesson_filename,
            "quiz_filename": quiz_filename
        }
    except Exception as e:
        print(f"❌ Error saving files: {e}")
        return {
            **state,
            "error": f"Error saving files: {e}"
        }

def memory_agent_node(state: GraphState):
    """Saves the results of the run to the SQLite database."""
    print("---NODE: MEMORY AGENT---")
    
    # Skip if compilation isn't complete
    if not state.get("compilation_complete"):
        print("⏳ Memory Agent: Waiting for compilation to complete...")
        return {}  # Return empty dict if waiting
    
    try:
        # Extract data from state
        topic = state.get("topic")
        grade_level = state.get("grade_level")
        evaluation_report = state.get("evaluation_report")
        
        # Skip if we don't have all required data
        missing_fields = []
        if not topic: missing_fields.append("topic")
        if not grade_level: missing_fields.append("grade_level")
        if not evaluation_report: missing_fields.append("evaluation_report")
        
        if missing_fields:
            print(f"⏳ Memory Agent: Waiting for required fields: {', '.join(missing_fields)}")
            return {}  # Return empty dict if missing data
        
        # Skip if there was an evaluation error
        if isinstance(evaluation_report, dict) and evaluation_report.get("error"):
            print(f"❌ Memory Agent: Skipping database save due to evaluation error: {evaluation_report['error']}")
            return {}  # Return empty dict if there's an error
        
        # Get the latest output filenames
        output_dir = "outputs"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_slug = topic.lower().replace(" ", "_").replace(":", "")
        lesson_file = f"{timestamp}_{topic_slug}_lesson.md"
        quiz_file = f"{timestamp}_{topic_slug}_quiz.md"
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if we already have this entry
        cursor.execute("""
            SELECT id FROM interactions 
            WHERE topic = ? AND grade_level = ? AND lesson_file = ? AND quiz_file = ?
        """, (topic, grade_level, lesson_file, quiz_file))
        
        if cursor.fetchone():
            print("⏭️ Memory Agent: Entry already exists in database, skipping...")
            conn.close()
            return {}  # Return empty dict if entry exists
        
        # Insert the interaction record
        cursor.execute("""
            INSERT INTO interactions (
                topic,
                grade_level,
                clarity_score,
                engagement_score,
                educational_value_score,
                lesson_file,
                quiz_file
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (
            topic,
            grade_level,
            evaluation_report.clarity_score,
            evaluation_report.engagement_score,
            evaluation_report.educational_value_score,
            lesson_file,
            quiz_file
        ))
        
        conn.commit()
        conn.close()
        print(f"""✅ Memory Agent: Saved to SQLite database:
- Topic: {topic}
- Grade: {grade_level}
- Scores: Clarity={evaluation_report.clarity_score}/5, Engagement={evaluation_report.engagement_score}/5, Educational Value={evaluation_report.educational_value_score}/5
- Files: {lesson_file}, {quiz_file}""")
        
        # --- NEW: Save to Firestore ---
        firestore_success = save_to_firestore(state)
        if firestore_success:
            print("✅ Memory Agent: Also saved to Firestore")
        else:
            print("⚠️ Memory Agent: Firestore save failed, but SQLite save succeeded")
        
        # Return only the filenames that were created
        return {
            "lesson_filename": lesson_file,
            "quiz_filename": quiz_file
        }
        
    except Exception as e:
        print(f"❌ Memory Agent: Error saving to database: {e}")
        return {}  # Return empty dict on error
    
    return {}

# --- 4. Graph Construction ---
builder = StateGraph(GraphState)

# Add all nodes
builder.add_node("Intent_Parser", intent_parser_node)
builder.add_node("RAG_Agent", rag_agent_node)
builder.add_node("Creative_Assistant", creative_assistant_node)
builder.add_node("Enhanced_Prompt_Composer", enhanced_prompt_composer_node)
builder.add_node("Lesson_Generator", lesson_generator_node)
builder.add_node("Lesson_JSON_Generator", lesson_json_generator_node)
builder.add_node("Quiz_Generator", quiz_generator_node)
builder.add_node("Quiz_JSON_Generator", quiz_json_generator_node)
builder.add_node("Image_Generator", image_generator_node)
builder.add_node("hallucination_guard", hallucination_guard_node)
builder.add_node("Final_Compiler", final_compiler_node)
builder.add_node("Evaluation_Agent", evaluation_agent_node)
builder.add_node("Memory_Agent", memory_agent_node)

# --- Define the graph's flow ---
builder.set_entry_point("Intent_Parser")

# After parsing, start with RAG agent
builder.add_edge("Intent_Parser", "RAG_Agent")

# Define conditional edges based on RAG results
def should_continue(state):
    """Determine if we should continue based on RAG results"""
    if "error" in state:
        return False
    return True

# Add conditional edges from RAG_Agent
builder.add_conditional_edges(
    "RAG_Agent",
    should_continue,
    {
        True: "Creative_Assistant",  # Continue with parallel processing
        False: END  # Stop if topic not found
    }
)

# Connect Creative Assistant to Prompt Composer
builder.add_edge("Creative_Assistant", "Enhanced_Prompt_Composer")

# Once prompts are ready, generate lesson, quiz, and image in parallel
builder.add_edge("Enhanced_Prompt_Composer", "Lesson_Generator")
builder.add_edge("Enhanced_Prompt_Composer", "Quiz_Generator")
builder.add_edge("Enhanced_Prompt_Composer", "Image_Generator")

# Generate structured JSON content after markdown content is ready
builder.add_edge("Lesson_Generator", "Lesson_JSON_Generator")
builder.add_edge("Quiz_Generator", "Quiz_JSON_Generator")

# The lesson must be fact-checked by the guard
builder.add_edge("Lesson_Generator", "hallucination_guard")

# First, wait for hallucination check to complete
builder.add_edge("hallucination_guard", "Final_Compiler")

# Then wait for quiz, image, and JSON content
builder.add_edge("Quiz_Generator", "Final_Compiler")
builder.add_edge("Image_Generator", "Final_Compiler")
builder.add_edge("Lesson_JSON_Generator", "Final_Compiler")
builder.add_edge("Quiz_JSON_Generator", "Final_Compiler")

# Add evaluation and memory after compilation
builder.add_edge("Final_Compiler", "Evaluation_Agent")
builder.add_edge("Evaluation_Agent", "Memory_Agent")
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

# --- NEW: Voice Agent Components ---
import speech_recognition as sr

def listen_for_voice_command(language="en-IN"):
    """
    Listens for a voice command from the microphone and transcribes it to text.
    
    Args:
        language (str): The language code for transcription (e.g., 'en-IN', 'hi-IN').
    
    Returns:
        str: The transcribed text, or None if an error occurs.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Calibrating for ambient noise, please wait...")
        r.adjust_for_ambient_noise(source, duration=1)
        print(f"Listening in {language}... Please speak your request.")
        
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            print("Recognizing...")
            # Transcribe audio using Google's Web Speech API
            text = r.recognize_google(audio, language=language)
            print(f"✅ Voice Input Recognized: '{text}'")
            return text
        except sr.WaitTimeoutError:
            print("⚠️ Listening timed out while waiting for phrase to start.")
            return None
        except sr.UnknownValueError:
            print("❌ I'm sorry, I could not understand the audio. Please try again.")
            return None
        except sr.RequestError as e:
            print(f"❌ Could not request results from the speech recognition service; {e}")
            return None
        except Exception as e:
            print(f"❌ An unexpected error occurred during voice recognition: {e}")
            return None


# --- 6. Invocation and Testing ---
if __name__ == "__main__":
    import sys
    
    # ... (the --test block remains the same) ...
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        from test_graph import run_all_tests
        success = run_all_tests()
        if not success:
            print("\n❌ Graph structure tests failed")
            exit(1)
        print("\n✅ All graph structure tests passed")
        exit(0)
    
    # Initialize the database
    setup_memory_database()
    
    pdf_path = "source_material.pdf"
    try:
        rag_retriever = setup_rag_pipeline(pdf_path)
    except Exception as e:
        print(f"❌ Error setting up RAG pipeline: {e}")
        exit(1)

    print("\n\n--- Welcome to Project Sahayak ---")
    
    # --- MODIFIED: User input selection (Voice or Text) ---
    user_input = None
    while not user_input:
        choice = input("Press 'V' for voice input or 'T' for text input, then Enter: ").strip().upper()
        if choice == 'T':
            user_input = input("Hello! What lesson can I prepare for you today? (e.g., 'a lesson on the water cycle for 4th graders')\n> ")
        elif choice == 'V':
            lang_choice = input("Select language for voice input ('en' for English, 'hi' for Hindi): ").strip().lower()
            lang_code = "hi-IN" if lang_choice == "hi" else "en-IN"
            user_input = listen_for_voice_command(language=lang_code)
        else:
            print("Invalid choice. Please press 'V' or 'T'.")
    
    # --- The rest of the execution logic remains the same ---
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

    if final_state.get("error"):
        print(f"\n--- ❗ERROR OCCURRED ---\n{final_state['error']}")
    else:
        final_state['execution_time'] = end_time - start_time
        save_outputs_node(final_state)
        print("\n\n✅✅✅ --- Content Generation Complete --- ✅✅✅")