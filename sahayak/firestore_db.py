
# sahayak/firestore_db.py

import datetime
from google.cloud import firestore
from typing import Dict, List, Optional

class FirestoreDatabase:
    """Firestore database wrapper for the education voice application."""
    
    def __init__(self, collection_name: str = "interactions"):
        self.db = firestore.Client()
        self.collection = self.db.collection(collection_name)
    
    def add_interaction(self, topic: str, grade_level: str, language: str = "English",
                       clarity_score: Optional[int] = None,
                       engagement_score: Optional[int] = None,
                       educational_value_score: Optional[int] = None,
                       lesson_file: Optional[str] = None,
                       quiz_file: Optional[str] = None) -> str:
        """Add a new interaction record."""
        doc_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'topic': topic,
            'grade_level': grade_level,
            'language': language,
            'clarity_score': clarity_score,
            'engagement_score': engagement_score,
            'educational_value_score': educational_value_score,
            'lesson_file': lesson_file,
            'quiz_file': quiz_file
        }
        
        doc_ref = self.collection.add(doc_data)
        return doc_ref[1].id
    
    def get_interaction(self, doc_id: str) -> Optional[Dict]:
        """Get an interaction by document ID."""
        doc = self.collection.document(doc_id).get()
        if doc.exists:
            return doc.to_dict()
        return None
    
    def get_recent_interactions(self, limit: int = 10) -> List[Dict]:
        """Get recent interactions."""
        docs = self.collection.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit).stream()
        return [doc.to_dict() for doc in docs]
    
    def update_scores(self, doc_id: str, clarity_score: int, engagement_score: int, educational_value_score: int):
        """Update evaluation scores for an interaction."""
        self.collection.document(doc_id).update({
            'clarity_score': clarity_score,
            'engagement_score': engagement_score,
            'educational_value_score': educational_value_score
        })
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        docs = list(self.collection.stream())
        total_interactions = len(docs)
        
        # Calculate average scores
        total_clarity = sum(doc.to_dict().get('clarity_score', 0) or 0 for doc in docs if doc.to_dict().get('clarity_score'))
        total_engagement = sum(doc.to_dict().get('engagement_score', 0) or 0 for doc in docs if doc.to_dict().get('engagement_score'))
        total_educational = sum(doc.to_dict().get('educational_value_score', 0) or 0 for doc in docs if doc.to_dict().get('educational_value_score'))
        
        clarity_count = sum(1 for doc in docs if doc.to_dict().get('clarity_score'))
        engagement_count = sum(1 for doc in docs if doc.to_dict().get('engagement_score'))
        educational_count = sum(1 for doc in docs if doc.to_dict().get('educational_value_score'))
        
        return {
            'total_interactions': total_interactions,
            'average_clarity': total_clarity / clarity_count if clarity_count > 0 else 0,
            'average_engagement': total_engagement / engagement_count if engagement_count > 0 else 0,
            'average_educational_value': total_educational / educational_count if educational_count > 0 else 0
        }
