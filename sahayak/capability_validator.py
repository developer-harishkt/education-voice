#!/usr/bin/env python3
"""
Capability Validator - Deterministic Pre-Validation Tool
Acts as a cache and decision maker for the agentic system.
"""

import os
import glob
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of capability validation."""
    can_fulfill: bool
    reason: str
    suggested_alternatives: List[str]
    available_materials: List[str]
    grade_downgrade_detected: bool
    language_mismatch: bool
    subject_mismatch: bool

class CapabilityValidator:
    """Deterministic validator for lesson generation capabilities."""
    
    def __init__(self):
        """Initialize with scanned material inventory."""
        self.available_materials = self._scan_materials()
        self.material_cache = self._build_material_cache()
    
    def _scan_materials(self) -> Dict[str, List[str]]:
        """Scan all available source materials."""
        materials = {}
        
        # Scan all language directories
        language_dirs = ['english', 'hindi', 'tamil', 'telugu', 'kannada', 'marathi', 'bengali']
        
        for lang in language_dirs:
            lang_path = f"source_materials/{lang}"
            if os.path.exists(lang_path):
                materials[lang] = []
                
                # Find all PDFs in this language
                pdf_pattern = f"{lang_path}/**/*.pdf"
                pdf_files = glob.glob(pdf_pattern, recursive=True)
                
                for pdf in pdf_files:
                    materials[lang].append(pdf)
        
        return materials
    
    def _build_material_cache(self) -> Dict[str, Dict]:
        """Build a structured cache of available materials."""
        cache = {}
        
        for language, pdf_files in self.available_materials.items():
            cache[language] = {
                'grades': set(),
                'subjects': set(),
                'grade_subject_combinations': set(),
                'files': pdf_files
            }
            
            for pdf in pdf_files:
                # Parse path: source_materials/language/grade/subject/filename.pdf
                path_parts = pdf.split('/')
                if len(path_parts) >= 4:
                    grade = path_parts[2]
                    subject = path_parts[3]
                    
                    cache[language]['grades'].add(grade)
                    cache[language]['subjects'].add(subject)
                    cache[language]['grade_subject_combinations'].add(f"{grade}/{subject}")
        
        return cache
    
    def validate_request(self, language: str, grade: str, subject: str, topic: str = None) -> ValidationResult:
        """
        Validate if the system can fulfill a lesson request.
        
        Args:
            language: Requested language
            grade: Requested grade
            subject: Requested subject
            topic: Optional topic for additional validation
            
        Returns:
            ValidationResult with detailed analysis
        """
        # Normalize language
        language_map = {
            'English': 'english',
            'Hindi': 'hindi', 
            'Tamil': 'tamil',
            'Telugu': 'telugu',
            'Kannada': 'kannada',
            'Marathi': 'marathi',
            'Bengali': 'bengali'
        }
        normalized_lang = language_map.get(language, language.lower())
        
        # Check if language is available
        if normalized_lang not in self.material_cache:
            return ValidationResult(
                can_fulfill=False,
                reason=f"Language '{language}' is not supported. Available: {list(self.material_cache.keys())}",
                suggested_alternatives=[f"Use English instead of {language}"],
                available_materials=[],
                grade_downgrade_detected=False,
                language_mismatch=True,
                subject_mismatch=False
            )
        
        lang_cache = self.material_cache[normalized_lang]
        
        # Check exact match
        exact_combination = f"{grade}/{subject}"
        if exact_combination in lang_cache['grade_subject_combinations']:
            return ValidationResult(
                can_fulfill=True,
                reason=f"Exact match found: {language} {grade} {subject}",
                suggested_alternatives=[],
                available_materials=[f for f in lang_cache['files'] if f"{grade}/{subject}" in f],
                grade_downgrade_detected=False,
                language_mismatch=False,
                subject_mismatch=False
            )
        
        # Check for alternatives
        alternatives = []
        grade_downgrade = False
        language_mismatch = False
        subject_mismatch = False
        
        # Check same subject, different grade
        if subject in lang_cache['subjects']:
            available_grades = [g for g in lang_cache['grades'] if f"{g}/{subject}" in lang_cache['grade_subject_combinations']]
            if available_grades:
                # Check for grade downgrade
                try:
                    requested_grade_num = int(grade.replace('class', ''))
                    for available_grade in available_grades:
                        available_grade_num = int(available_grade.replace('class', ''))
                        if available_grade_num < requested_grade_num:
                            grade_downgrade = True
                            alternatives.append(f"⚠️ {language} {available_grade} {subject} (downgrade)")
                        else:
                            alternatives.append(f"✅ {language} {available_grade} {subject}")
                except ValueError:
                    alternatives.append(f"⚠️ {language} {available_grades[0]} {subject}")
        
        # Check English equivalent
        if normalized_lang != 'english' and 'english' in self.material_cache:
            english_cache = self.material_cache['english']
            if f"{grade}/{subject}" in english_cache['grade_subject_combinations']:
                alternatives.append(f"✅ English {grade} {subject} (language alternative)")
        
        # Check related subjects (science ↔ mathematics)
        if subject in ['science', 'mathematics']:
            related_subject = 'mathematics' if subject == 'science' else 'science'
            if related_subject in lang_cache['subjects']:
                related_combinations = [f"{g}/{related_subject}" for g in lang_cache['grades'] 
                                      if f"{g}/{related_subject}" in lang_cache['grade_subject_combinations']]
                if related_combinations:
                    alternatives.append(f"⚠️ {language} {grade} {related_subject} (related subject)")
                    subject_mismatch = True
        
        # Determine if we can fulfill
        can_fulfill = len(alternatives) > 0 and not grade_downgrade
        
        if not can_fulfill:
            reason = f"Requested {language} {grade} {subject} not available"
            if grade_downgrade:
                reason += " and would require educational downgrade"
            if not alternatives:
                reason += ". No suitable alternatives found."
        else:
            reason = f"Can fulfill with alternatives: {', '.join(alternatives)}"
        
        return ValidationResult(
            can_fulfill=can_fulfill,
            reason=reason,
            suggested_alternatives=alternatives,
            available_materials=lang_cache['files'],
            grade_downgrade_detected=grade_downgrade,
            language_mismatch=language_mismatch,
            subject_mismatch=subject_mismatch
        )
    
    def get_available_combinations(self, language: str = None) -> Dict[str, List[str]]:
        """Get all available language/grade/subject combinations."""
        combinations = {}
        
        for lang, cache in self.material_cache.items():
            if language and lang != language.lower():
                continue
            combinations[lang] = list(cache['grade_subject_combinations'])
        
        return combinations
    
    def get_system_capabilities(self) -> Dict:
        """Get comprehensive system capabilities report."""
        capabilities = {
            'supported_languages': list(self.material_cache.keys()),
            'total_materials': sum(len(files) for files in self.available_materials.values()),
            'language_breakdown': {},
            'grade_coverage': {},
            'subject_coverage': {}
        }
        
        all_grades = set()
        all_subjects = set()
        
        for lang, cache in self.material_cache.items():
            capabilities['language_breakdown'][lang] = {
                'materials_count': len(cache['files']),
                'grades': list(cache['grades']),
                'subjects': list(cache['subjects']),
                'combinations': list(cache['grade_subject_combinations'])
            }
            all_grades.update(cache['grades'])
            all_subjects.update(cache['subjects'])
        
        capabilities['grade_coverage'] = list(all_grades)
        capabilities['subject_coverage'] = list(all_subjects)
        
        return capabilities

# Global instance for caching
capability_validator = CapabilityValidator() 