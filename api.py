# api.py

import sys
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

# Import necessary components from your sahayak package
from sahayak.utils import initialize_firebase, setup_rag_pipeline
from sahayak.graph import api_graph

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    print("🚀 API starting up...")
    try:
        # Initialize Firebase
        initialize_firebase()
        # Setup the RAG pipeline retriever globally (will be updated per request)
        app.state.rag_retriever = setup_rag_pipeline()
        print("✅ RAG pipeline retriever is ready.")
    except Exception as e:
        print(f"❌ Critical startup error: {e}")
        sys.exit(1)
    
    yield
    
    # Shutdown (if needed)
    print("🛑 API shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Project Sahayak API",
    description="API for generating educational lesson plans.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Health Check Endpoint ---
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Project Sahayak API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "message": "Project Sahayak API is running!",
        "version": "1.0.0"
    }

# --- Request Model ---
class LessonRequest(BaseModel):
    user_request: str
    user_uuid: str

# --- Background Task Function ---
def run_lesson_generation(initial_state: dict):
    """The function that will be run in the background."""
    print(f"🚀 Starting background task for user: {initial_state.get('user_uuid')}")
    start_time = time.time()
    
    user_uuid = initial_state.get('user_uuid')
    
    # Initialize Firebase status tracker
    try:
        from sahayak.firebase_status_tracker import create_status_tracker
        status_tracker = create_status_tracker(user_uuid)
        initial_state['status_tracker'] = status_tracker
    except Exception as e:
        print(f"⚠️ Failed to initialize status tracker: {e}")
        status_tracker = None

    # Don't setup RAG pipeline here - let the agentic validator decide first
    # RAG will be set up in the Intent_Parser node after validation passes
    print(f"🔍 Request received: '{user_request}' - will validate before RAG setup")

    # The final state is not strictly needed here as the result is published to Firebase
    final_state = {}
    error_occurred = False
    
    for event in api_graph.stream(initial_state):
        for key, value in event.items():
            print(f"--- Finished Node: {key} ---")
            
            # Update status tracker with node completion
            if status_tracker:
                status_tracker.mark_node_complete(key, f"Completed: {key}")
            
            # Check for errors
            if value is not None and isinstance(value, dict) and "error" in value:
                error_msg = value["error"]
                print(f"❌ Error in {key}: {error_msg}")
                error_occurred = True
                
                # Update status tracker with error
                if status_tracker:
                    status_tracker.mark_failed(error_msg, key)
                
                # Don't continue processing if there's an error
                break
            
            if value is not None and isinstance(value, dict):
                final_state.update(value)
        
        # Break out of outer loop if error occurred
        if error_occurred:
            break

    end_time = time.time()
    
    if error_occurred:
        print(f"❌ Background task failed in {end_time - start_time:.2f} seconds.")
    else:
        print(f"✅ Background task finished in {end_time - start_time:.2f} seconds.")
        # Mark as successful
        if status_tracker:
            status_tracker.mark_success("Lesson generation completed successfully")


# --- API Endpoint ---
@app.post("/generate_lesson_plan")
async def generate_lesson_plan(request: LessonRequest, background_tasks: BackgroundTasks):
    """
    Accepts a user request and starts the lesson generation process in the background.
    """
    if not request.user_request or not request.user_uuid:
        raise HTTPException(status_code=400, detail="user_request and user_uuid are required.")

    # Prepare the initial state for the graph
    initial_state = {
        "user_request": request.user_request,
        "user_uuid": request.user_uuid,
        "retriever": app.state.rag_retriever,  # This will be updated in the background task
    }

    # Add the graph execution as a background task
    background_tasks.add_task(run_lesson_generation, initial_state)

    # Immediately return a response to the client
    return {"message": "Lesson plan generation started. The result will be delivered to your app via Firebase."}

# To run this API, use the command:
# uvicorn api:app --reload
