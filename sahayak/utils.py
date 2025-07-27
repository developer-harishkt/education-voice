# sahayak/utils.py

import datetime
import os
import sqlite3
import speech_recognition as sr
import requests
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIImageGeneratorChat
from langdetect import detect, LangDetectException

# --- Environment Setup ---
load_dotenv()
DB_PATH = "sahayak_memory.db"

# Initialize image generation tool once
try:
    image_generation_tool = VertexAIImageGeneratorChat(model="imagegeneration@006")
    print("✅ Vertex AI Image Generation initialized")
except Exception as e:
    print(f"⚠️ Vertex AI Image Generation not available: {e}. Using fallback.")
    image_generation_tool = None

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
    """Initializes the SQLite database."""
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
    print("✅ Database setup complete.")

def setup_rag_pipeline(source_document_path: str = None):
    """Sets up the RAG pipeline with automatic source selection."""
    
    if source_document_path is None:
        # Default to the organized source material
        source_document_path = "source_materials/english/class8/science/KARNATAKA_SCIENCE_CLASS8_ENGLISH_2023.pdf"
    
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
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs_split, embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 10})
    
    # Store metadata in a way that doesn't conflict with the retriever object
    retriever._document_language = detected_language
    retriever._source_path = source_document_path
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
