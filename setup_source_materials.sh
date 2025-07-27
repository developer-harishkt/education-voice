#!/bin/bash
# setup_source_materials.sh
# Script to create the organized directory structure for educational PDFs

echo "📚 Setting up source materials directory structure..."
echo "="*60

# Create main directory
mkdir -p source_materials

# Define languages and grades
languages=("english" "hindi" "kannada" "tamil" "telugu" "marathi" "bengali")
grades=("class3" "class4" "class5" "class6" "class7" "class8" "class9" "class10")
subjects=("science" "mathematics" "social_studies")

# Create language directories
for lang in "${languages[@]}"; do
    echo "📁 Creating directory for $lang..."
    mkdir -p source_materials/$lang
    
    # Create grade directories
    for grade in "${grades[@]}"; do
        mkdir -p source_materials/$lang/$grade
        
        # Create subject directories
        for subject in "${subjects[@]}"; do
            mkdir -p source_materials/$lang/$grade/$subject
        done
    done
done

echo ""
echo "✅ Source materials directory structure created successfully!"
echo ""
echo "📁 Directory structure:"
echo "source_materials/"
echo "├── english/"
echo "│   ├── class3/"
echo "│   │   ├── science/"
echo "│   │   ├── mathematics/"
echo "│   │   └── social_studies/"
echo "│   ├── class4/"
echo "│   └── ... (and so on for all grades)"
echo "├── hindi/"
echo "├── kannada/"
echo "├── tamil/"
echo "├── telugu/"
echo "├── marathi/"
echo "└── bengali/"
echo ""
echo "📝 Naming convention for PDFs:"
echo "BOARD_SUBJECT_GRADE_LANGUAGE_YEAR.pdf"
echo ""
echo "Examples:"
echo "• NCERT_SCIENCE_CLASS8_ENGLISH_2023.pdf"
echo "• KARNATAKA_SCIENCE_CLASS8_KANNADA_2023.pdf"
echo "• UPBOARD_SCIENCE_CLASS8_HINDI_2023.pdf"
echo ""
echo "💡 Next steps:"
echo "1. Download PDFs using the Perplexity Pro guide"
echo "2. Rename files according to the naming convention"
echo "3. Place PDFs in the appropriate folders"
echo "4. Test the system with different grade/language combinations"
echo ""
echo "🎯 Quality checklist:"
echo "• Complete textbooks (not just chapters)"
echo "• High-quality scans (300+ DPI)"
echo "• Recent editions (2018-2024)"
echo "• Official board publications"
echo "• Free/open access"
echo "• Proper text extraction possible" 