# sahayak/graph.py

from langgraph.graph import StateGraph, END
from .state import GraphState
from .nodes import (
    intent_parser_node,
    rag_agent_node,
    llm_reranker_node,
    creative_assistant_node,
    enhanced_prompt_composer_node,
    lesson_generator_node,
    quiz_generator_node,
    image_generator_node,
    hallucination_guard_node,
    final_compiler_node,
    evaluation_agent_node,
    firebase_publish_node
)
from .agentic_validator_node import agentic_validator_node

def build_api_graph():
    """Builds and compiles the LangGraph agent for the API endpoint."""
    builder = StateGraph(GraphState)

    # Add all nodes
    builder.add_node("Agentic_Validator", agentic_validator_node)
    builder.add_node("Intent_Parser", intent_parser_node)
    builder.add_node("RAG_Agent", rag_agent_node)
    builder.add_node("LLM_Reranker", llm_reranker_node)
    builder.add_node("Creative_Assistant", creative_assistant_node)
    builder.add_node("Enhanced_Prompt_Composer", enhanced_prompt_composer_node)
    builder.add_node("Lesson_Generator", lesson_generator_node)
    builder.add_node("Quiz_Generator", quiz_generator_node)
    builder.add_node("Image_Generator", image_generator_node)
    builder.add_node("hallucination_guard", hallucination_guard_node)
    builder.add_node("Final_Compiler", final_compiler_node)
    builder.add_node("Evaluation_Agent", evaluation_agent_node)
    builder.add_node("Firebase_Publisher", firebase_publish_node)

    # Define the graph's flow
    builder.set_entry_point("Agentic_Validator")
    
    def should_proceed_after_validation(state):
        return "continue" if not state.get("error") else "end"
    
    builder.add_conditional_edges("Agentic_Validator", should_proceed_after_validation, {"continue": "Intent_Parser", "end": END})
    builder.add_edge("Intent_Parser", "RAG_Agent")
    builder.add_edge("RAG_Agent", "LLM_Reranker")

    def should_continue(state):
        return "continue" if not state.get("error") else "end"

    builder.add_conditional_edges("LLM_Reranker", should_continue, {"continue": "Creative_Assistant", "end": END})

    builder.add_edge("Creative_Assistant", "Enhanced_Prompt_Composer")
    builder.add_edge("Enhanced_Prompt_Composer", "Lesson_Generator")
    builder.add_edge("Enhanced_Prompt_Composer", "Quiz_Generator")
    builder.add_edge("Enhanced_Prompt_Composer", "Image_Generator")

    builder.add_edge("Lesson_Generator", "hallucination_guard")
    builder.add_edge("hallucination_guard", "Final_Compiler")
    builder.add_edge("Quiz_Generator", "Final_Compiler")
    builder.add_edge("Image_Generator", "Final_Compiler")

    builder.add_edge("Final_Compiler", "Evaluation_Agent")
    builder.add_edge("Evaluation_Agent", "Firebase_Publisher")
    builder.add_edge("Firebase_Publisher", END)

    return builder.compile()

# Compile the API graph when the module is loaded
api_graph = build_api_graph()
