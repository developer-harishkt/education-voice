#!/usr/bin/env python3
"""
Ultra-simple HTTP server for testing
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import uuid

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {
            "status": "healthy",
            "message": "Simple server is running"
        }
        self.wfile.write(json.dumps(response).encode())

    def do_POST(self):
        """Handle POST requests"""
        try:
            # Read the request body
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            if not post_data:
                self.send_error(400, "Empty request body")
                return
            
            # Parse JSON
            try:
                request_json = json.loads(post_data.decode())
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
                return
            
            # Validate required fields
            if 'user_request' not in request_json or 'user_id' not in request_json:
                self.send_error(400, "Invalid request: JSON body must include 'user_request' and 'user_id'.")
                return
            
            # Extract fields
            user_request = request_json['user_request']
            user_id = request_json['user_id']
            thread_id = request_json.get('thread_id', str(uuid.uuid4()))
            run_id = str(uuid.uuid4())
            
            print(f"📥 Received POST request:")
            print(f"   User ID: {user_id}")
            print(f"   Thread ID: {thread_id}")  
            print(f"   Request: {user_request}")
            
            # Send response
            self.send_response(202)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                "status": "processing_started",
                "run_id": run_id,
                "thread_id": thread_id,
                "message": "Request received and processing started",
                "user_request": user_request
            }
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            print(f"❌ Error processing POST request: {e}")
            self.send_error(500, f"Internal server error: {str(e)}")

if __name__ == '__main__':
    port = 8080
    server = HTTPServer(('localhost', port), SimpleHandler)
    print(f"🚀 Starting simple HTTP server on port {port}")
    print(f"📍 Server accessible at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\n🛑 Server stopped")
        server.server_close()
