#!/usr/bin/env python3
"""
Check Firebase Realtime Database for lesson data
"""

import firebase_admin
from firebase_admin import credentials, db
import os

def check_firebase_data():
    """Check if lesson data exists in Firebase"""
    print("🔍 CHECKING FIREBASE REALTIME DATABASE")
    print("="*50)
    
    try:
        # Initialize Firebase
        if not firebase_admin._apps:
            cred = credentials.Certificate("firebase-service-account.json")
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://agentic-ai-645f6-default-rtdb.firebaseio.com/'
            })
            print("✅ Firebase initialized")
        
        # Get reference to lesson_plans
        ref = db.reference("lesson_plans")
        data = ref.get()
        
        if data:
            print(f"✅ Found {len(data)} lesson entries in Firebase:")
            for user_id, lesson_data in data.items():
                print(f"\n👤 User: {user_id}")
                print(f"   Topic: {lesson_data.get('topic', 'N/A')}")
                print(f"   Grade Level: {lesson_data.get('grade_level', 'N/A')}")
                print(f"   Status: {lesson_data.get('status', 'N/A')}")
                print(f"   Timestamp: {lesson_data.get('timestamp', 'N/A')}")
                
                # Check if lesson content exists
                if lesson_data.get('lesson_plan'):
                    print(f"   ✅ Lesson plan: {len(lesson_data['lesson_plan'])} characters")
                if lesson_data.get('quiz'):
                    print(f"   ✅ Quiz: {len(lesson_data['quiz'])} characters")
                if lesson_data.get('evaluation'):
                    eval_data = lesson_data['evaluation']
                    print(f"   📊 Evaluation: Clarity={eval_data.get('clarity_score', 'N/A')}, Engagement={eval_data.get('engagement_score', 'N/A')}")
        else:
            print("ℹ️ No lesson data found in Firebase yet")
            
    except Exception as e:
        print(f"❌ Error checking Firebase: {e}")

if __name__ == "__main__":
    check_firebase_data() 