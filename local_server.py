#!/usr/bin/env python3
"""
Simplified local server for testing without Google Cloud dependencies
"""
import json
from flask import Flask, request, jsonify
import uuid

app = Flask(__name__)

@app.route('/', methods=['POST'])
def handler():
    """HTTP handler that mimics the Cloud Function response"""
    try:
        request_json = request.get_json(silent=True)
        
        if not request_json or 'user_request' not in request_json or 'user_id' not in request_json:
            return jsonify({"error": "Invalid request: JSON body must include 'user_request' and 'user_id'."}), 400

        # Extract required fields
        user_request = request_json['user_request']
        user_id = request_json['user_id']
        thread_id = request_json.get('thread_id', str(uuid.uuid4()))
        run_id = str(uuid.uuid4())

        print(f"📥 Received request:")
        print(f"   User ID: {user_id}")
        print(f"   Thread ID: {thread_id}")
        print(f"   Request: {user_request}")

        # Simulate processing (this would normally be the LangGraph execution)
        response = {
            "status": "processing_started",
            "run_id": run_id,
            "thread_id": thread_id,
            "message": "Mock response - your request is being processed",
            "user_request": user_request
        }

        return jsonify(response), 202

    except Exception as e:
        print(f"❌ Error processing request: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Local education-voice server is running"
    })

if __name__ == '__main__':
    print("🚀 Starting local education-voice server...")
    print("🔧 This is a simplified version for testing without Google Cloud")
    print("📍 Server will run on http://localhost:8080")
    
    app.run(host='0.0.0.0', port=8080, debug=False)
