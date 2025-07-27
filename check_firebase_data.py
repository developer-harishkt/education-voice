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
            for user_id, user_data in data.items():
                print(f"\n👤 User: {user_id}")
                
                # Check if user_data is a dict (new structure) or direct lesson data (old structure)
                if isinstance(user_data, dict) and 'latest' in user_data:
                    # New structure with multiple lessons
                    latest_lesson = user_data.get('latest', {})
                    print(f"   📚 Latest Lesson:")
                    print(f"      Topic: {latest_lesson.get('topic', 'N/A')}")
                    print(f"      Grade Level: {latest_lesson.get('grade_level', 'N/A')}")
                    print(f"      Status: {latest_lesson.get('status', 'N/A')}")
                    print(f"      Timestamp: {latest_lesson.get('timestamp', 'N/A')}")
                    print(f"      Lesson ID: {latest_lesson.get('lesson_id', 'N/A')}")
                    
                    # Count total lessons for this user
                    lesson_count = len([k for k in user_data.keys() if k not in ['latest', 'errors']])
                    print(f"   📊 Total Lessons: {lesson_count}")
                    
                    # Check if lesson content exists
                    if latest_lesson.get('lesson_plan'):
                        print(f"      ✅ Lesson plan: {len(latest_lesson['lesson_plan'])} characters")
                    if latest_lesson.get('quiz'):
                        print(f"      ✅ Quiz: {len(latest_lesson['quiz'])} characters")
                    if latest_lesson.get('evaluation'):
                        eval_data = latest_lesson['evaluation']
                        print(f"      📊 Evaluation: Clarity={eval_data.get('clarity_score', 'N/A')}, Engagement={eval_data.get('engagement_score', 'N/A')}")
                    
                    # List all lessons for this user
                    print(f"   📋 All Lessons:")
                    for lesson_key, lesson_data in user_data.items():
                        if lesson_key not in ['latest', 'errors'] and isinstance(lesson_data, dict):
                            print(f"      - {lesson_data.get('lesson_id', lesson_key)}: {lesson_data.get('topic', 'N/A')} ({lesson_data.get('grade_level', 'N/A')})")
                            
                else:
                    # Old structure (single lesson per user)
                    print(f"   Topic: {user_data.get('topic', 'N/A')}")
                    print(f"   Grade Level: {user_data.get('grade_level', 'N/A')}")
                    print(f"   Status: {user_data.get('status', 'N/A')}")
                    print(f"   Timestamp: {user_data.get('timestamp', 'N/A')}")
                    
                    # Check if lesson content exists
                    if user_data.get('lesson_plan'):
                        print(f"   ✅ Lesson plan: {len(user_data['lesson_plan'])} characters")
                    if user_data.get('quiz'):
                        print(f"   ✅ Quiz: {len(user_data['quiz'])} characters")
                    if user_data.get('evaluation'):
                        eval_data = user_data['evaluation']
                        print(f"   📊 Evaluation: Clarity={eval_data.get('clarity_score', 'N/A')}, Engagement={eval_data.get('engagement_score', 'N/A')}")
        else:
            print("ℹ️ No lesson data found in Firebase yet")
            
    except Exception as e:
        print(f"❌ Error checking Firebase: {e}")

if __name__ == "__main__":
    check_firebase_data() 