#!/usr/bin/env python3
"""
Agentic Validator Node - Makes intelligent decisions using deterministic tools
"""

from typing import Dict, Any
from .capability_validator import capability_validator
from .state import GraphState

def agentic_validator_node(state: GraphState) -> Dict[str, Any]:
    """
    Agentic node that validates capabilities before expensive operations.
    
    This node acts as an intelligent gatekeeper that:
    1. Uses deterministic tools to check capabilities
    2. Makes agentic decisions about proceeding
    3. Provides clear alternatives to users
    4. Prevents educational injustice
    """
    print("---NODE: AGENTIC VALIDATOR---")
    
    # Extract request parameters
    user_request = state.get("user_request", "")
    topic = state.get("topic", "")
    grade_level = state.get("grade_level", "")
    
    # Parse language, grade, and subject from request
    from .utils import detect_request_language, extract_grade_from_request, extract_subject_from_request
    
    language = detect_request_language(user_request)
    grade = extract_grade_from_request(user_request)
    subject = extract_subject_from_request(user_request)
    
    print(f"🔍 Agentic validation for: {language} {grade} {subject}")
    print(f"📝 Topic: {topic}")
    
    # Use deterministic validator
    validation_result = capability_validator.validate_request(
        language=language,
        grade=grade,
        subject=subject,
        topic=topic
    )
    
    print(f"✅ Validation result: {validation_result.can_fulfill}")
    print(f"📋 Reason: {validation_result.reason}")
    
    # Agentic decision making
    if not validation_result.can_fulfill:
        # Agent decides to exit early with clear explanation
        error_message = f"""
I apologize, but I cannot fulfill your request for the following reasons:

{validation_result.reason}

**Available Alternatives:**
"""
        
        if validation_result.suggested_alternatives:
            for alt in validation_result.suggested_alternatives:
                error_message += f"- {alt}\n"
        else:
            error_message += "- No suitable alternatives available at this time\n"
        
        error_message += f"""
**System Capabilities:**
- Supported Languages: {', '.join(capability_validator.get_system_capabilities()['supported_languages'])}
- Total Materials: {capability_validator.get_system_capabilities()['total_materials']}

Please try a different combination or contact support for additional materials.
"""
        
        print(f"❌ Agentic decision: Exit early - {validation_result.reason}")
        
        # Update status tracker if available
        status_tracker = state.get('status_tracker')
        if status_tracker:
            status_tracker.mark_failed(validation_result.reason, "Agentic_Validator")
        
        return {"error": error_message.strip()}
    
    # If we can fulfill, proceed with the best available option
    if validation_result.suggested_alternatives:
        print(f"✅ Agentic decision: Proceed with alternatives")
        print(f"📚 Available alternatives: {validation_result.suggested_alternatives}")
        
        # Choose the best alternative (prefer exact match, then English equivalent)
        best_alternative = None
        for alt in validation_result.suggested_alternatives:
            if "✅" in alt and "English" in alt:
                best_alternative = alt
                break
            elif "✅" in alt:
                best_alternative = alt
                break
        
        if best_alternative:
            print(f"🎯 Selected alternative: {best_alternative}")
            
            # Update state with the selected alternative
            return {
                "validation_passed": True,
                "selected_alternative": best_alternative,
                "available_materials": validation_result.available_materials,
                "warning": f"Using alternative: {best_alternative}" if "⚠️" in best_alternative else None
            }
    
    # Exact match found
    print(f"✅ Agentic decision: Proceed with exact match")
    return {
        "validation_passed": True,
        "exact_match": True,
        "available_materials": validation_result.available_materials
    }

def get_system_capabilities() -> Dict:
    """Get system capabilities for agentic decision making."""
    return capability_validator.get_system_capabilities()

def get_available_combinations(language: str = None) -> Dict[str, list]:
    """Get available combinations for agentic suggestions."""
    return capability_validator.get_available_combinations(language) 