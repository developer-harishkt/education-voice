# sahayak/utils.py

import datetime
import os
import sqlite3
import speech_recognition as sr
import requests
import glob
import re
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIImageGeneratorChat
from langdetect import detect, LangDetectException

# Import Gemini embeddings (replacing HuggingFace)
try:
    from .gemini_embeddings import gemini_embeddings
    if gemini_embeddings is None or not gemini_embeddings.is_available():
        raise ImportError("Gemini embeddings not available")
    USE_GEMINI_EMBEDDINGS = True
except ImportError as e:
    USE_GEMINI_EMBEDDINGS = False

# Always import HuggingFace embeddings as fallback
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    print("❌ HuggingFace embeddings not available")
    HUGGINGFACE_AVAILABLE = False

# Import Firestore database (new)
try:
    from .firestore_db import FirestoreDatabase
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False

# --- Environment Setup ---
load_dotenv()
DB_PATH = "sahayak_memory.db"

# Initialize image generation tool once
try:
    image_generation_tool = VertexAIImageGeneratorChat(model="imagegeneration@006")
except Exception as e:
    image_generation_tool = None

# Initialize database (Firestore primary, SQLite fallback)
if FIRESTORE_AVAILABLE:
    try:
        firestore_db = FirestoreDatabase()
        USE_FIRESTORE = True
    except Exception as e:
        USE_FIRESTORE = False
else:
    USE_FIRESTORE = False

# --- Language Detection and Support ---
def detect_document_language(text_sample: str) -> str:
    """Detect the primary language of a document."""
    try:
        # Take first 1000 characters for language detection
        sample = text_sample[:1000] if len(text_sample) > 1000 else text_sample
        language = detect(sample)
        
        # Map language codes to common names
        language_map = {
            'en': 'English',
            'hi': 'Hindi', 
            'kn': 'Kannada',
            'ta': 'Tamil',
            'te': 'Telugu',
            'ml': 'Malayalam',
            'mr': 'Marathi',
            'bn': 'Bengali',
            'gu': 'Gujarati',
            'pa': 'Punjabi',
            'or': 'Odia'
        }
        
        detected_lang = language_map.get(language, language.upper())
        print(f"🌐 Detected language: {detected_lang} ({language})")
        return detected_lang
        
    except LangDetectException as e:
        print(f"⚠️ Language detection failed: {e}")
        return "English"  # Default fallback

def detect_request_language(user_request: str) -> str:
    """Detect the language of the user request or desired output language."""
    try:
        # First, check for explicit language instructions in English
        language_instructions = {
            r'in\s+tamil': 'Tamil',
            r'in\s+hindi': 'Hindi',
            r'in\s+kannada': 'Kannada',
            r'in\s+telugu': 'Telugu',
            r'in\s+marathi': 'Marathi',
            r'in\s+bengali': 'Bengali',
            r'tamil\s+language': 'Tamil',
            r'hindi\s+language': 'Hindi',
            r'kannada\s+language': 'Kannada',
            r'telugu\s+language': 'Telugu',
            r'marathi\s+language': 'Marathi',
            r'bengali\s+language': 'Bengali',
            r'தமிழில்': 'Tamil',
            r'हिंदी में': 'Hindi',
            r'ಕನ್ನಡದಲ್ಲಿ': 'Kannada',
            r'తెలుగులో': 'Telugu',
            r'మరాఠీలో': 'Marathi',
            r'বাংলায়': 'Bengali'
        }
        
        for pattern, language in language_instructions.items():
            if re.search(pattern, user_request, re.IGNORECASE):
                print(f"🌐 Language instruction detected: {language}")
                return language
        
        # If no explicit instruction, detect from the text content
        language = detect(user_request)
        language_map = {
            'en': 'English',
            'hi': 'Hindi', 
            'kn': 'Kannada',
            'ta': 'Tamil',
            'te': 'Telugu',
            'ml': 'Malayalam',
            'mr': 'Marathi',
            'bn': 'Bengali',
            'gu': 'Gujarati',
            'pa': 'Punjabi',
            'or': 'Odia'
        }
        detected_lang = language_map.get(language, 'English')
        print(f"🌐 Request language detected: {detected_lang}")
        return detected_lang
    except:
        return "English"

def extract_grade_from_request(user_request: str) -> str:
    """Extract grade level from user request."""
    grade_patterns = {
        r'(\d+)(?:st|nd|rd|th)\s*grade': lambda m: f"class{m.group(1)}",
        r'class\s*(\d+)': lambda m: f"class{m.group(1)}",
        r'grade\s*(\d+)': lambda m: f"class{m.group(1)}",
        r'(\d+)ஆம்\s*வகுப்பு': lambda m: f"class{m.group(1)}",  # Tamil: 6ஆம் வகுப்பு
        r'(\d+)\s*வகுப்பு': lambda m: f"class{m.group(1)}",      # Tamil: 6 வகுப்பு
        r'(\d+)ஆம்\s*தரம்': lambda m: f"class{m.group(1)}",      # Tamil: 6ஆம் தரம்
        r'(\d+)\s*कक्षा': lambda m: f"class{m.group(1)}",         # Hindi
        r'(\d+)\s*ತರಗತಿ': lambda m: f"class{m.group(1)}",         # Kannada
    }
    
    for pattern, formatter in grade_patterns.items():
        match = re.search(pattern, user_request, re.IGNORECASE)
        if match:
            grade = formatter(match)
            print(f"📚 Grade extracted: {grade}")
            return grade
    
    # Default to class8 if no grade found
    print("📚 No grade detected, defaulting to class8")
    return "class8"

def extract_subject_from_request(user_request: str) -> str:
    """Extract subject from user request."""
    subject_patterns = {
        r'science|அறிவியல்|विज्ञान|ವಿಜ್ಞಾನ': 'science',
        r'math|mathematics|கணிதம்|गणित|ಗಣಿತ': 'mathematics',
        r'social|social studies|சமூக அறிவியல்|सामाजिक विज्ञान|ಸಾಮಾಜಿಕ ವಿಜ್ಞಾನ': 'social_studies'
    }
    
    for pattern, subject in subject_patterns.items():
        if re.search(pattern, user_request, re.IGNORECASE):
            print(f"📖 Subject extracted: {subject}")
            return subject
    
    # Default to science if no subject found
    print("📖 No subject detected, defaulting to science")
    return "science"

def find_best_source_material(user_request: str) -> dict:
    """Find the best matching source material based on user request."""
    print(f"🔍 Finding best source material for: '{user_request}'")
    
    # Detect language and extract grade/subject
    language = detect_request_language(user_request)
    grade = extract_grade_from_request(user_request)
    subject = extract_subject_from_request(user_request)
    
    # Map language names to directory names
    language_dir_map = {
        'English': 'english',
        'Hindi': 'hindi',
        'Kannada': 'kannada',
        'Tamil': 'tamil',
        'Telugu': 'telugu',
        'Marathi': 'marathi',
        'Bengali': 'bengali'
    }
    
    language_dir = language_dir_map.get(language, 'english')
    
    # Construct the expected path
    expected_path = f"source_materials/{language_dir}/{grade}/{subject}"
    print(f"📁 Looking for materials in: {expected_path}")
    
    # Find PDFs in the expected directory
    if os.path.exists(expected_path):
        pdf_files = glob.glob(f"{expected_path}/*.pdf")
        if pdf_files:
            # Return the first available PDF
            selected_pdf = pdf_files[0]
            print(f"✅ Found exact source material: {selected_pdf}")
            return {
                "path": selected_pdf,
                "match_type": "exact",
                "requested_language": language,
                "requested_grade": grade,
                "requested_subject": subject,
                "found_language": language,
                "found_grade": grade,
                "found_subject": subject
            }
    
    # Smart Fallback Strategy:
    # 1. Try same subject, different grade
    # 2. Try same grade, different subject (but only if subject is related)
    # 3. Try same language, different grade and subject
    # 4. Try English equivalent
    
    print(f"🔍 Implementing smart fallback strategy...")
    
    # Fallback 1: Same subject, different grade
    subject_pattern = f"source_materials/{language_dir}/*/{subject}/*.pdf"
    subject_files = glob.glob(subject_pattern)
    
    if subject_files:
        # Sort by grade to find closest match
        subject_files.sort()
        selected_pdf = subject_files[0]
        path_parts = selected_pdf.split('/')
        found_grade = path_parts[2]
        
        # Check for significant grade downgrade (more than 2 grades down)
        requested_grade_num = int(grade.replace('class', ''))
        found_grade_num = int(found_grade.replace('class', ''))
        grade_difference = requested_grade_num - found_grade_num
        
        if grade_difference > 2:
            print(f"❌ Significant grade downgrade detected: {grade} → {found_grade} (difference: {grade_difference})")
            print(f"⚠️ Skipping this fallback to avoid educational injustice")
        else:
            print(f"✅ Found same subject ({subject}) in grade {found_grade}")
            print(f"⚠️ Using fallback: {selected_pdf}")
            
            return {
                "path": selected_pdf,
                "match_type": "fallback_same_subject",
                "requested_language": language,
                "requested_grade": grade,
                "requested_subject": subject,
                "found_language": language,
                "found_grade": found_grade,
                "found_subject": subject,
                "warning": f"Requested {language} {grade} {subject} not available. Using {language} {found_grade} {subject} as fallback."
            }
    
    # Fallback 2: Same grade, related subject (only for science/mathematics)
    if subject in ['science', 'mathematics']:
        related_subjects = ['science', 'mathematics'] if subject == 'science' else ['mathematics', 'science']
        
        for related_subject in related_subjects:
            if related_subject != subject:
                grade_pattern = f"source_materials/{language_dir}/{grade}/{related_subject}/*.pdf"
                grade_files = glob.glob(grade_pattern)
                
                if grade_files:
                    selected_pdf = grade_files[0]
                    print(f"✅ Found related subject ({related_subject}) in same grade ({grade})")
                    print(f"⚠️ Using fallback: {selected_pdf}")
                    
                    return {
                        "path": selected_pdf,
                        "match_type": "fallback_related_subject",
                        "requested_language": language,
                        "requested_grade": grade,
                        "requested_subject": subject,
                        "found_language": language,
                        "found_grade": grade,
                        "found_subject": related_subject,
                        "warning": f"Requested {language} {grade} {subject} not available. Using {language} {grade} {related_subject} as fallback."
                    }
    
    # Fallback 3: Same language, any grade and subject
    language_pattern = f"source_materials/{language_dir}/**/*.pdf"
    language_files = glob.glob(language_pattern, recursive=True)
    
    if language_files:
        selected_pdf = language_files[0]
        path_parts = selected_pdf.split('/')
        found_grade = path_parts[2] if len(path_parts) > 2 else grade
        found_subject = path_parts[3] if len(path_parts) > 3 else subject
        
        print(f"⚠️ Using language fallback: {selected_pdf}")
        print(f"⚠️ Mismatch: Requested {language} {grade} {subject}, Found {language} {found_grade} {found_subject}")
        
        return {
            "path": selected_pdf,
            "match_type": "fallback_language_only",
            "requested_language": language,
            "requested_grade": grade,
            "requested_subject": subject,
            "found_language": language,
            "found_grade": found_grade,
            "found_subject": found_subject,
            "warning": f"Requested {language} {grade} {subject} not available. Using {language} {found_grade} {found_subject} as fallback."
        }
    
    # Fallback 4: English equivalent (same subject and grade)
    english_pattern = f"source_materials/english/{grade}/{subject}/*.pdf"
    english_files = glob.glob(english_pattern)
    
    if english_files:
        selected_pdf = english_files[0]
        print(f"✅ Found English equivalent: {selected_pdf}")
        
        return {
            "path": selected_pdf,
            "match_type": "fallback_english_equivalent",
            "requested_language": language,
            "requested_grade": grade,
            "requested_subject": subject,
            "found_language": "English",
            "found_grade": grade,
            "found_subject": subject,
            "warning": f"Requested {language} {grade} {subject} not available. Using English {grade} {subject} as fallback."
        }
    
    # Final fallback: use default English material
    default_path = "source_materials/english/class8/science/KARNATAKA_SCIENCE_CLASS8_ENGLISH_2023.pdf"
    print(f"❌ No matching source material found, using default: {default_path}")
    return {
        "path": default_path,
        "match_type": "default",
        "requested_language": language,
        "requested_grade": grade,
        "requested_subject": subject,
        "found_language": "English",
        "found_grade": "class8",
        "found_subject": "science",
        "warning": f"Requested {language} {grade} {subject} not available. Using English class8 science as default."
    }

def get_language_specific_prompt(language: str, topic: str, grade_level: str) -> str:
    """Generate language-specific prompts for lesson generation."""
    
    language_prompts = {
        'Hindi': {
            'lesson': f"'{topic}' के बारे में एक विस्तृत पाठ योजना बनाएं।\nकक्षा: {grade_level}\n\nनिम्नलिखित संरचना का पालन करें:\n1. विषय\n2. लक्ष्य कक्षा\n3. उद्देश्य\n4. सामग्री\n5. परिचय\n6. गतिविधियां\n7. मूल्यांकन\n\nमहत्वपूर्ण दिशानिर्देश:\n- सरल और स्पष्ट हिंदी का प्रयोग करें\n- शैक्षणिक भाषा बनाए रखें\n- मार्कडाउन प्रारूप में प्रस्तुत करें",
            'quiz': f"'{topic}' के लिए {grade_level} कक्षा का वर्कशीट बनाएं।\n\nनिम्नलिखित शामिल करें:\n- 3-4 प्रश्न जो {grade_level} के स्तर के अनुरूप हों\n- बहुविकल्पीय और लघु उत्तरीय प्रश्न\n- स्पष्ट निर्देश\n- उत्तर कुंजी"
        },
        'Kannada': {
            'lesson': f"'{topic}' ಬಗ್ಗೆ ವಿವರವಾದ ಪಾಠ ಯೋಜನೆಯನ್ನು ರಚಿಸಿ।\nತರಗತಿ: {grade_level}\n\nಈ ಕೆಳಗಿನ ರಚನೆಯನ್ನು ಅನುಸರಿಸಿ:\n1. ವಿಷಯ\n2. ಗುರಿ ತರಗತಿ\n3. ಉದ್ದೇಶಗಳು\n4. ಸಾಮಗ್ರಿಗಳು\n5. ಪರಿಚಯ\n6. ಚಟುವಟಿಕೆಗಳು\n7. ಮೌಲ್ಯಮಾಪನ\n\nಮುಖ್ಯ ಮಾರ್ಗದರ್ಶನಗಳು:\n- ಸರಳ ಮತ್ತು ಸ್ಪಷ್ಟ ಕನ್ನಡ ಬಳಸಿ\n- ಶೈಕ್ಷಣಿಕ ಭಾಷೆಯನ್ನು ಕಾಯ್ದುಕೊಳ್ಳಿ\n- ಮಾರ್ಕ್ಡೌನ್ ರೂಪದಲ್ಲಿ ಪ್ರಸ್ತುತಪಡಿಸಿ",
            'quiz': f"'{topic}' ಗಾಗಿ {grade_level} ತರಗತಿಯ ವರ್ಕ್‌ಶೀಟ್ ರಚಿಸಿ।\n\nಇವುಗಳನ್ನು ಸೇರಿಸಿ:\n- {grade_level} ಮಟ್ಟಕ್ಕೆ ಸೂಕ್ತವಾದ 3-4 ಪ್ರಶ್ನೆಗಳು\n- ಬಹು ಆಯ್ಕೆ ಮತ್ತು ಸಣ್ಣ ಉತ್ತರ ಪ್ರಶ್ನೆಗಳು\n- ಸ್ಪಷ್ಟ ಸೂಚನೆಗಳು\n- ಉತ್ತರ ಕೀ"
        },
        'Tamil': {
            'lesson': f"'{topic}' பற்றி விரிவான பாடத் திட்டத்தை உருவாக்கவும்।\nவகுப்பு: {grade_level}\n\nபின்வரும் கட்டமைப்பைப் பின்பற்றவும்:\n1. தலைப்பு\n2. இலக்கு வகுப்பு\n3. நோக்கங்கள்\n4. பொருட்கள்\n5. அறிமுகம்\n6. செயல்பாடுகள்\n7. மதிப்பீடு\n\nமுக்கிய வழிகாட்டுதல்கள்:\n- எளிமையான மற்றும் தெளிவான தமிழைப் பயன்படுத்தவும்\n- கல்வி மொழியை பராமரிக்கவும்\n- மார்க்‌டவுன் வடிவத்தில் வழங்கவும்",
            'quiz': f"'{topic}' க்கான {grade_level} வகுப்பு வரைவு தாளை உருவாக்கவும்।\n\nபின்வருவனவற்றைச் சேர்க்கவும்:\n- {grade_level} நிலைக்கு ஏற்ற 3-4 கேள்விகள்\n- பல தேர்வு மற்றும் குறுகிய பதில் கேள்விகள்\n- தெளிவான வழிகாட்டுதல்கள்\n- பதில் விசை"
        },
        'English': {
            'lesson': f"Create a detailed lesson plan about '{topic}'.\nGrade: {grade_level}\n\nFollow this structure:\n1. Topic\n2. Target Grade\n3. Objectives\n4. Materials\n5. Introduction\n6. Activities\n7. Assessment\n\nImportant Guidelines:\n- Use clear and simple language\n- Maintain academic tone\n- Format in markdown",
            'quiz': f"Create a worksheet about '{topic}' for {grade_level}.\n\nInclude:\n- 3-4 questions appropriate for {grade_level} level\n- Multiple choice and short answer questions\n- Clear instructions\n- Answer key"
        }
    }
    
    return language_prompts.get(language, language_prompts['English'])

# --- Firebase Initialization ---
def initialize_firebase():
    """Initializes the Firebase Admin SDK."""
    try:
        firebase_admin.get_app()
        print("🔥 Firebase app already initialized.")
    except ValueError:
        cred = credentials.Certificate("firebase-service-account.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://agentic-ai-11-default-rtdb.firebaseio.com/' # IMPORTANT: Replace with your actual URL
        })
        print("🔥 Firebase app initialized successfully.")

# --- Database and RAG Setup ---
def setup_memory_database():
    """Initializes the database (Firestore primary, SQLite fallback)."""
    if USE_FIRESTORE:
        print("✅ Using Firestore database")
        return
    
    # Fallback to SQLite
    print("⚠️ Using SQLite database (fallback)")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            topic TEXT NOT NULL,
            grade_level TEXT NOT NULL,
            language TEXT DEFAULT 'English',
            clarity_score INTEGER,
            engagement_score INTEGER,
            educational_value_score INTEGER,
            lesson_file TEXT,
            quiz_file TEXT
        );
    """)
    conn.commit()
    conn.close()
    print("✅ SQLite database setup complete.")

def add_interaction_to_database(topic: str, grade_level: str, language: str = "English",
                               clarity_score: int = None, engagement_score: int = None,
                               educational_value_score: int = None, lesson_file: str = None,
                               quiz_file: str = None) -> str:
    """Add interaction to database (Firestore or SQLite)."""
    global USE_FIRESTORE
    
    if USE_FIRESTORE:
        try:
            doc_id = firestore_db.add_interaction(
                topic=topic,
                grade_level=grade_level,
                language=language,
                clarity_score=clarity_score,
                engagement_score=engagement_score,
                educational_value_score=educational_value_score,
                lesson_file=lesson_file,
                quiz_file=quiz_file
            )
            print(f"✅ Interaction added to Firestore: {doc_id}")
            return doc_id
        except Exception as e:
            print(f"⚠️ Firestore failed, falling back to SQLite: {e}")
            USE_FIRESTORE = False
    
    # SQLite fallback
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO interactions (topic, grade_level, language, clarity_score, 
                                engagement_score, educational_value_score, lesson_file, quiz_file)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (topic, grade_level, language, clarity_score, engagement_score, 
          educational_value_score, lesson_file, quiz_file))
    interaction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    print(f"✅ Interaction added to SQLite: {interaction_id}")
    return str(interaction_id)

def get_database_statistics():
    """Get database statistics (Firestore or SQLite)."""
    global USE_FIRESTORE
    
    if USE_FIRESTORE:
        try:
            stats = firestore_db.get_statistics()
            print("✅ Retrieved statistics from Firestore")
            return stats
        except Exception as e:
            print(f"⚠️ Firestore failed, falling back to SQLite: {e}")
            USE_FIRESTORE = False
    
    # SQLite fallback
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM interactions")
    total_interactions = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(clarity_score), AVG(engagement_score), AVG(educational_value_score) FROM interactions WHERE clarity_score IS NOT NULL")
    averages = cursor.fetchone()
    conn.close()
    
    return {
        'total_interactions': total_interactions,
        'average_clarity': averages[0] or 0,
        'average_engagement': averages[1] or 0,
        'average_educational_value': averages[2] or 0
    }

def setup_rag_pipeline(source_document_path: str = None, user_request: str = None):
    """Sets up the RAG pipeline with intelligent source selection."""
    
    source_info = None
    
    if source_document_path is None and user_request:
        # Use intelligent source selection based on user request
        source_info = find_best_source_material(user_request)
        source_document_path = source_info["path"]
    elif source_document_path is None:
        # Default to the organized source material
        source_document_path = "source_materials/english/class8/science/KARNATAKA_SCIENCE_CLASS8_ENGLISH_2023.pdf"
        source_info = {
            "path": source_document_path,
            "match_type": "default",
            "found_language": "English",
            "found_grade": "class8",
            "found_subject": "science"
        }
    
    print(f"📚 Loading source material: {source_document_path}")
    
    loader = PyPDFLoader(source_document_path)
    docs = loader.load()
    
    # Detect language from first few documents
    if docs:
        sample_text = docs[0].page_content[:1000]
        detected_language = detect_document_language(sample_text)
        print(f"🌐 Source document language: {detected_language}")
    else:
        detected_language = "English"
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = splitter.split_documents(docs)
    
    # Use Gemini embeddings if available, otherwise use HuggingFace
    embedding_success = False
    
    if USE_GEMINI_EMBEDDINGS and gemini_embeddings is not None:
        try:
            print("🔍 Using Gemini embeddings for vector creation...")
            # Use the wrapper methods that include retry logic
            texts = [doc.page_content for doc in docs_split]
            embeddings_list = gemini_embeddings.embed_documents(texts)
            
            # Create FAISS vectorstore manually with embeddings
            import numpy as np
            from langchain_community.vectorstores import FAISS
            from langchain_core.embeddings import Embeddings
            
            # Create a temporary embedding class for FAISS
            class TempEmbeddings(Embeddings):
                def __init__(self, embeddings_list):
                    self.embeddings_list = embeddings_list
                    self.current_index = 0
                
                def embed_documents(self, texts):
                    # Return pre-computed embeddings
                    result = self.embeddings_list[self.current_index:self.current_index + len(texts)]
                    self.current_index += len(texts)
                    return result
                
                def embed_query(self, text):
                    # For queries, we'll need to compute new embeddings
                    return gemini_embeddings.embed_query(text)
            
            temp_embeddings = TempEmbeddings(embeddings_list)
            vectorstore = FAISS.from_documents(docs_split, temp_embeddings)
            print("✅ Vector store created with Gemini embeddings")
            embedding_success = True
            
        except Exception as e:
            print(f"⚠️ Gemini embeddings failed, falling back to HuggingFace: {e}")
    
    # Fallback to HuggingFace if Gemini failed or is not available
    if not embedding_success:
        if HUGGINGFACE_AVAILABLE:
            try:
                embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = FAISS.from_documents(docs_split, embedding_model)
                print("✅ Vector store created with HuggingFace embeddings")
                embedding_success = True
            except Exception as e:
                print(f"⚠️ HuggingFace embeddings also failed: {e}")
        else:
            print("❌ HuggingFace embeddings not available")
    
    if not embedding_success:
        raise Exception("No embedding model available - both Gemini and HuggingFace failed")
    
    retriever = vectorstore.as_retriever(search_kwargs={'k': 10})
    
    # Store metadata in a way that doesn't conflict with the retriever object
    retriever._document_language = detected_language
    retriever._source_path = source_document_path
    retriever._source_info = source_info
    return retriever

# --- Image Generation ---
def generate_image_with_fallback(prompt: str) -> str:
    """Generates an image using Vertex AI with a Hugging Face fallback."""
    if image_generation_tool:
        try:
            print("Attempting Vertex AI image generation...")
            urls = image_generation_tool.invoke(prompt)
            if urls and urls[0]:
                return urls[0]
        except Exception as e:
            print(f"⚠️ Vertex AI Image Generation failed: {e}")

    print("Attempting Hugging Face image generation...")
    models = ["CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1"]
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}", "Content-Type": "application/json"}
    formatted_prompt = f"educational textbook illustration, black and white line drawing, {prompt}, simple clean lines"

    for model in models:
        try:
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            response = requests.post(api_url, headers=headers, json={"inputs": formatted_prompt, "options": {"wait_for_model": True}}, timeout=45)
            if response.status_code == 200:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                image_dir = "outputs/images"
                os.makedirs(image_dir, exist_ok=True)
                image_path = f"{image_dir}/{timestamp}_generated.png"
                with open(image_path, "wb") as f:
                    f.write(response.content)
                print(f"✅ Image generated using {model} and saved to {image_path}")
                return image_path
            else:
                print(f"⚠️ Model {model} failed with status {response.status_code}")
        except Exception as e:
            print(f"⚠️ Error with model {model}: {e}")
    return "No image generated."

# --- Voice Input ---
def listen_for_voice_command(language="en-IN"):
    """Listens for a voice command and transcribes it."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Calibrating for ambient noise...")
        r.adjust_for_ambient_noise(source, duration=1)
        print(f"Listening in {language}... Please speak.")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            print("Recognizing...")
            text = r.recognize_google(audio, language=language)
            print(f"✅ Voice Input Recognized: '{text}'")
            return text
        except sr.WaitTimeoutError:
            print("⚠️ Listening timed out.")
        except sr.UnknownValueError:
            print("❌ Could not understand the audio.")
        except sr.RequestError as e:
            print(f"❌ Recognition service error; {e}")
    return None
