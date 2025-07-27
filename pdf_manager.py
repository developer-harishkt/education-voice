#!/usr/bin/env python3
"""
PDF Manager Utility
Helps manage and validate the educational PDF collection
"""

import os
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple

class PDFManager:
    def __init__(self, source_dir: str = "source_materials"):
        self.source_dir = source_dir
        self.naming_pattern = re.compile(r'^([A-Z]+)_([A-Z_]+)_([A-Z0-9]+)_([A-Z]+)_(\d{4})\.pdf$')
    
    def validate_naming_convention(self, filename: str) -> Dict:
        """Validate if a filename follows the naming convention."""
        match = self.naming_pattern.match(filename)
        if match:
            board, subject, grade, language, year = match.groups()
            return {
                'valid': True,
                'board': board,
                'subject': subject,
                'grade': grade,
                'language': language,
                'year': year
            }
        else:
            return {'valid': False, 'error': 'Does not match naming convention'}
    
    def scan_pdfs(self) -> Dict:
        """Scan all PDFs in the source materials directory."""
        pdf_files = glob.glob(f"{self.source_dir}/**/*.pdf", recursive=True)
        
        inventory = {
            'total_files': len(pdf_files),
            'valid_files': 0,
            'invalid_files': 0,
            'by_language': {},
            'by_grade': {},
            'by_subject': {},
            'invalid_files_list': []
        }
        
        for pdf_file in pdf_files:
            filename = os.path.basename(pdf_file)
            relative_path = os.path.relpath(pdf_file, self.source_dir)
            
            validation = self.validate_naming_convention(filename)
            
            if validation['valid']:
                inventory['valid_files'] += 1
                
                # Group by language
                lang = validation['language'].lower()
                if lang not in inventory['by_language']:
                    inventory['by_language'][lang] = []
                inventory['by_language'][lang].append({
                    'file': pdf_file,
                    'relative_path': relative_path,
                    'board': validation['board'],
                    'subject': validation['subject'],
                    'grade': validation['grade'],
                    'year': validation['year']
                })
                
                # Group by grade
                grade = validation['grade'].lower()
                if grade not in inventory['by_grade']:
                    inventory['by_grade'][grade] = []
                inventory['by_grade'][grade].append({
                    'file': pdf_file,
                    'relative_path': relative_path,
                    'board': validation['board'],
                    'subject': validation['subject'],
                    'language': validation['language'],
                    'year': validation['year']
                })
                
                # Group by subject
                subject = validation['subject'].lower()
                if subject not in inventory['by_subject']:
                    inventory['by_subject'][subject] = []
                inventory['by_subject'][subject].append({
                    'file': pdf_file,
                    'relative_path': relative_path,
                    'board': validation['board'],
                    'grade': validation['grade'],
                    'language': validation['language'],
                    'year': validation['year']
                })
            else:
                inventory['invalid_files'] += 1
                inventory['invalid_files_list'].append({
                    'file': pdf_file,
                    'relative_path': relative_path,
                    'error': validation['error']
                })
        
        return inventory
    
    def print_inventory(self, inventory: Dict):
        """Print a formatted inventory report."""
        print("📚 PDF INVENTORY REPORT")
        print("="*60)
        print(f"📊 Total Files: {inventory['total_files']}")
        print(f"✅ Valid Files: {inventory['valid_files']}")
        print(f"❌ Invalid Files: {inventory['invalid_files']}")
        print()
        
        # By Language
        print("🌐 BY LANGUAGE:")
        print("-" * 30)
        for lang, files in inventory['by_language'].items():
            print(f"  {lang.upper()}: {len(files)} files")
            for file_info in files:
                print(f"    • {file_info['board']} {file_info['subject']} {file_info['grade']} ({file_info['year']})")
        print()
        
        # By Grade
        print("📖 BY GRADE:")
        print("-" * 30)
        for grade in sorted(inventory['by_grade'].keys()):
            files = inventory['by_grade'][grade]
            print(f"  {grade.upper()}: {len(files)} files")
            for file_info in files:
                print(f"    • {file_info['board']} {file_info['subject']} {file_info['language']} ({file_info['year']})")
        print()
        
        # By Subject
        print("📚 BY SUBJECT:")
        print("-" * 30)
        for subject in sorted(inventory['by_subject'].keys()):
            files = inventory['by_subject'][subject]
            print(f"  {subject.upper()}: {len(files)} files")
            for file_info in files:
                print(f"    • {file_info['board']} {file_info['grade']} {file_info['language']} ({file_info['year']})")
        print()
        
        # Invalid Files
        if inventory['invalid_files_list']:
            print("❌ INVALID FILES:")
            print("-" * 30)
            for file_info in inventory['invalid_files_list']:
                print(f"  • {file_info['relative_path']}")
                print(f"    Error: {file_info['error']}")
            print()
    
    def suggest_renaming(self, filename: str) -> str:
        """Suggest a proper filename based on the current filename."""
        # Remove common extensions
        name = filename.replace('.pdf', '').replace('.PDF', '')
        
        # Try to extract components
        parts = name.replace('_', ' ').replace('-', ' ').split()
        
        if len(parts) >= 4:
            # Assume format: Board Subject Grade Language Year
            board = parts[0].upper()
            subject = parts[1].upper()
            grade = parts[2].upper()
            language = parts[3].upper()
            year = parts[4] if len(parts) > 4 else "2023"
            
            return f"{board}_{subject}_{grade}_{language}_{year}.pdf"
        else:
            return f"UNKNOWN_SUBJECT_CLASS8_ENGLISH_2023.pdf"
    
    def check_coverage(self, inventory: Dict) -> Dict:
        """Check coverage of different grades, languages, and subjects."""
        languages = ['english', 'hindi', 'kannada', 'tamil', 'telugu', 'marathi', 'bengali']
        grades = ['class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'class10']
        subjects = ['science', 'mathematics', 'social_studies']
        
        coverage = {
            'languages': {},
            'grades': {},
            'subjects': {},
            'recommendations': []
        }
        
        # Check language coverage
        for lang in languages:
            coverage['languages'][lang] = len(inventory['by_language'].get(lang, []))
        
        # Check grade coverage
        for grade in grades:
            coverage['grades'][grade] = len(inventory['by_grade'].get(grade, []))
        
        # Check subject coverage
        for subject in subjects:
            coverage['subjects'][subject] = len(inventory['by_subject'].get(subject, []))
        
        # Generate recommendations
        for lang in languages:
            if coverage['languages'][lang] == 0:
                coverage['recommendations'].append(f"Add {lang} textbooks")
        
        for grade in grades:
            if coverage['grades'][grade] == 0:
                coverage['recommendations'].append(f"Add {grade} textbooks")
        
        for subject in subjects:
            if coverage['subjects'][subject] == 0:
                coverage['recommendations'].append(f"Add {subject} textbooks")
        
        return coverage
    
    def print_coverage_report(self, coverage: Dict):
        """Print a coverage report."""
        print("📊 COVERAGE REPORT")
        print("="*60)
        
        print("\n🌐 Language Coverage:")
        for lang, count in coverage['languages'].items():
            status = "✅" if count > 0 else "❌"
            print(f"  {status} {lang.upper()}: {count} files")
        
        print("\n📖 Grade Coverage:")
        for grade, count in coverage['grades'].items():
            status = "✅" if count > 0 else "❌"
            print(f"  {status} {grade.upper()}: {count} files")
        
        print("\n📚 Subject Coverage:")
        for subject, count in coverage['subjects'].items():
            status = "✅" if count > 0 else "❌"
            print(f"  {status} {subject.upper()}: {count} files")
        
        if coverage['recommendations']:
            print("\n💡 Recommendations:")
            for rec in coverage['recommendations']:
                print(f"  • {rec}")

def main():
    """Main function to run the PDF manager."""
    manager = PDFManager()
    
    print("🔍 Scanning PDF collection...")
    inventory = manager.scan_pdfs()
    
    if inventory['total_files'] == 0:
        print("📁 No PDFs found in source_materials/ directory")
        print("💡 Use the Perplexity Pro guide to find and download PDFs")
        return
    
    # Print inventory
    manager.print_inventory(inventory)
    
    # Check coverage
    coverage = manager.check_coverage(inventory)
    manager.print_coverage_report(coverage)
    
    # Summary
    print("\n" + "="*60)
    print("📋 SUMMARY:")
    print(f"• Total PDFs: {inventory['total_files']}")
    print(f"• Valid naming: {inventory['valid_files']}")
    print(f"• Invalid naming: {inventory['invalid_files']}")
    
    if inventory['valid_files'] > 0:
        print("✅ Your PDF collection is ready for multi-grade, multi-language lesson generation!")
    else:
        print("⚠️ No valid PDFs found. Please rename files according to the naming convention.")

if __name__ == "__main__":
    main() 