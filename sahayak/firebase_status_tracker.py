#!/usr/bin/env python3
"""
Firebase Status Tracker - Real-time progress tracking for lesson generation
"""

import time
import datetime
from typing import Dict, Any, Optional
from firebase_admin import db

class FirebaseStatusTracker:
    """Tracks and updates lesson generation status in Firebase."""
    
    def __init__(self, user_uuid: str):
        """Initialize status tracker for a specific user."""
        self.user_uuid = user_uuid
        self.lesson_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.total_nodes = 12  # Total nodes in the graph
        self.completed_nodes = 0
        self.current_node = None
        
        # Initialize status in Firebase
        self._initialize_status()
    
    def _initialize_status(self):
        """Initialize the status entry in Firebase."""
        try:
            status_ref = db.reference(f"/lesson_status/{self.user_uuid}/{self.lesson_id}")
            initial_status = {
                "status": "acknowledged",
                "message": "Request received and acknowledged",
                "progress": 0,
                "current_node": "initializing",
                "start_time": self.start_time,
                "last_updated": time.time(),
                "total_nodes": self.total_nodes,
                "completed_nodes": 0,
                "estimated_completion": None,
                "error": None,
                "warning": None
            }
            status_ref.set(initial_status)
            print(f"✅ Status initialized for user {self.user_uuid}, lesson {self.lesson_id}")
        except Exception as e:
            print(f"❌ Failed to initialize Firebase status: {e}")
    
    def update_status(self, status: str, message: str, node_name: str = None, 
                     error: str = None, warning: str = None):
        """Update the status in Firebase."""
        try:
            # Update progress
            if node_name and node_name != self.current_node:
                self.completed_nodes += 1
                self.current_node = node_name
            
            progress = min(100, int((self.completed_nodes / self.total_nodes) * 100))
            
            # Calculate estimated completion
            elapsed_time = time.time() - self.start_time
            if self.completed_nodes > 0:
                avg_time_per_node = elapsed_time / self.completed_nodes
                remaining_nodes = self.total_nodes - self.completed_nodes
                estimated_completion = time.time() + (avg_time_per_node * remaining_nodes)
            else:
                estimated_completion = None
            
            status_data = {
                "status": status,
                "message": message,
                "progress": progress,
                "current_node": node_name or self.current_node,
                "last_updated": time.time(),
                "completed_nodes": self.completed_nodes,
                "total_nodes": self.total_nodes,
                "estimated_completion": estimated_completion,
                "elapsed_time": elapsed_time
            }
            
            if error:
                status_data["error"] = error
                status_data["status"] = "failed"
            
            if warning:
                status_data["warning"] = warning
            
            # Update Firebase
            status_ref = db.reference(f"/lesson_status/{self.user_uuid}/{self.lesson_id}")
            status_ref.update(status_data)
            
            print(f"📊 Status updated: {status} - {message} (Progress: {progress}%)")
            
        except Exception as e:
            print(f"❌ Failed to update Firebase status: {e}")
    
    def mark_node_complete(self, node_name: str, message: str = None):
        """Mark a node as completed."""
        if not message:
            message = f"Completed: {node_name}"
        
        self.update_status(
            status="processing",
            message=message,
            node_name=node_name
        )
    
    def mark_success(self, final_message: str = "Lesson generation completed successfully"):
        """Mark the process as successfully completed."""
        self.completed_nodes = self.total_nodes
        self.update_status(
            status="completed",
            message=final_message,
            node_name="completed"
        )
    
    def mark_failed(self, error_message: str, node_name: str = None):
        """Mark the process as failed."""
        self.update_status(
            status="failed",
            message=f"Failed at {node_name or 'unknown step'}: {error_message}",
            node_name=node_name,
            error=error_message
        )
    
    def mark_warning(self, warning_message: str, node_name: str = None):
        """Mark a warning during processing."""
        self.update_status(
            status="processing",
            message=f"Warning at {node_name or 'current step'}: {warning_message}",
            node_name=node_name,
            warning=warning_message
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status from Firebase."""
        try:
            status_ref = db.reference(f"/lesson_status/{self.user_uuid}/{self.lesson_id}")
            return status_ref.get() or {}
        except Exception as e:
            print(f"❌ Failed to get status: {e}")
            return {}

def create_status_tracker(user_uuid: str) -> FirebaseStatusTracker:
    """Factory function to create a status tracker."""
    return FirebaseStatusTracker(user_uuid)

# Node completion tracking
NODE_PROGRESS_MAP = {
    "Agentic_Validator": 1,
    "Intent_Parser": 2,
    "RAG_Agent": 3,
    "LLM_Reranker": 4,
    "Creative_Assistant": 5,
    "Enhanced_Prompt_Composer": 6,
    "Lesson_Generator": 7,
    "Quiz_Generator": 8,
    "Image_Generator": 9,
    "hallucination_guard": 10,
    "Final_Compiler": 11,
    "Evaluation_Agent": 12,
    "Firebase_Publisher": 13
} 