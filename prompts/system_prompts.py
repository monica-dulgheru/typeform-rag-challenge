"""
System prompts and templates for the Typeform RAG prototype.

This module defines the prompt templates used for the RAG system,
including system prompts, context formatting, and citation instructions.
"""

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are a Typeform customer support assistant. Your goal is to help users with questions about Typeform features and functionality.

IMPORTANT INSTRUCTIONS:
1. Answer questions using ONLY the information provided in this prompt and the CONTEXT section below.
2. Do not use any prior knowledge or information not present in the context.
3. If the provided CONTEXT is not relevant or does not contain enough information to confidently answer the user's question, you must respond with: "I cannot find enough information in the available Help Center articles to answer this question."
4. Provide your complete answer first, then list all citations at the end in the format [Source: chunk_id] on separate lines.
5. Be helpful, clear, and concise in your responses.
6. If the user asks about features not covered in the context, politely explain that you don't have information about that topic in the available Help Center articles.

Example of proper citation format:
"The Typeform Help Center offers articles on creating multi-language forms. A multi-language form is a form where questions can be displayed in multiple languages depending on the user's locale.

[Source: art_123_chunk_001]
[Source: art_123_chunk_004]
[Source: art_345_chunk_010]"

Remember: Only use information from the provided context. Do not make up or assume any information."""

# =============================================================================
# RAG TEMPLATES
# =============================================================================

RAG_TEMPLATE = """CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

# Alternative template with more explicit instructions
RAG_TEMPLATE_DETAILED = """Based on the following context from Typeform Help Center articles, please answer the user's question.

CONTEXT:
{context}

USER QUESTION: {question}

Please provide a helpful answer using only the information from the context above. If you cannot find sufficient information to answer the question, please say so clearly. Provide your complete answer first, then list all citations at the end in the format [Source: chunk_id] on separate lines."""

# =============================================================================
# CONTEXT FORMATTING FUNCTIONS
# =============================================================================

def format_context_for_prompt(retrieved_chunks: list) -> str:
    """
    Format retrieved chunks into a context string for the prompt.
    
    Args:
        retrieved_chunks: List of chunk dictionaries with metadata
        
    Returns:
        str: Formatted context string
    """
    if not retrieved_chunks:
        return "No relevant information found."
    
    context_parts = []
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        chunk_id = chunk.get('chunk_id', f'chunk_{i}')
        chunk_text = chunk.get('chunk_text', '')
        
        # Format each chunk with clear separation
        chunk_context = f"[Source: {chunk_id}] {chunk_text}"
        context_parts.append(chunk_context)
    
    return "\n\n".join(context_parts)

def format_context_with_metadata(retrieved_chunks: list) -> str:
    """
    Format retrieved chunks with additional metadata for better context.
    
    Args:
        retrieved_chunks: List of chunk dictionaries with metadata
        
    Returns:
        str: Formatted context string with metadata
    """
    if not retrieved_chunks:
        return "No relevant information found."
    
    context_parts = []
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        chunk_id = chunk.get('chunk_id', f'chunk_{i}')
        chunk_text = chunk.get('chunk_text', '')
        doc_title = chunk.get('doc_title', 'Unknown Document')
        
        # Format with document title for better context
        chunk_context = f"From '{doc_title}' [Source: {chunk_id}]:\n{chunk_text}"
        context_parts.append(chunk_context)
    
    return "\n\n".join(context_parts)


# =============================================================================
# PROMPT VERSIONS FOR EXPERIMENTATION
# =============================================================================

# Version 1: Basic prompt (default)
PROMPT_VERSION_1 = {
    'name': 'basic',
    'system_prompt': SYSTEM_PROMPT,
    'template': RAG_TEMPLATE,
    'context_formatter': format_context_for_prompt
}

# Version 2: Detailed prompt with metadata
PROMPT_VERSION_2 = {
    'name': 'detailed',
    'system_prompt': SYSTEM_PROMPT,
    'template': RAG_TEMPLATE_DETAILED,
    'context_formatter': format_context_with_metadata
}

# Version 3: Minimal prompt (for testing)
PROMPT_VERSION_3 = {
    'name': 'minimal',
    'system_prompt': "You are a helpful assistant. Answer the question based on the provided context.",
    'template': RAG_TEMPLATE,
    'context_formatter': format_context_for_prompt
}

# Available prompt versions
PROMPT_VERSIONS = {
    'basic': PROMPT_VERSION_1,
    'detailed': PROMPT_VERSION_2,
    'min': PROMPT_VERSION_3
}


# =============================================================================
# PROMPT CONSTRUCTION FUNCTIONS
# =============================================================================

def get_prompt_version(version: str) -> dict:
    """
    Get a specific prompt version.
    
    Args:
        version: Version identifier (must be one of the keys in PROMPT_VERSIONS)
        
    Returns:
        dict: Prompt version configuration
        
    Raises:
        ValueError: If version is not found in PROMPT_VERSIONS
    """
    if version not in PROMPT_VERSIONS:
        available_versions = list(PROMPT_VERSIONS.keys())
        raise ValueError(f"Invalid prompt version '{version}'. Available versions: {available_versions}")
    return PROMPT_VERSIONS[version]


def build_prompt_with_version(question: str, retrieved_chunks: list, version: str) -> str:
    """
    Build a RAG prompt using a specific prompt version.
    
    Args:
        question: User's question
        retrieved_chunks: List of retrieved chunks
        version: Prompt version to use
        
    Returns:
        str: Complete prompt for the LLM
    """
    prompt_config = get_prompt_version(version)
    
    # Format context using the version's formatter
    context = prompt_config['context_formatter'](retrieved_chunks)
    
    # Build prompt
    template = prompt_config['template']
    full_prompt = f"{prompt_config['system_prompt']}\n\n{template.format(context=context, question=question)}"
    
    return full_prompt


# =============================================================================
# CITATION EXTRACTION FUNCTIONS
# =============================================================================

def extract_citations_from_response(response: str) -> list:
    """
    Extract citations from LLM response.
    
    Args:
        response: LLM response text
        
    Returns:
        list: List of chunk IDs mentioned in citations
    """
    import re
    
    # Pattern to match [Source: chunk_id] citations
    citation_pattern = r'\[Source:\s*([^\]]+)\]'
    citations = re.findall(citation_pattern, response)
    
    return citations

def validate_citations(response: str, retrieved_chunks: list) -> dict:
    """
    Validate that citations in response are in the retrieved chunks.
    
    Args:
        response: LLM response text
        retrieved_chunks: List of retrieved chunks
        
    Returns:
        dict: Validation results
    """
    citations = extract_citations_from_response(response)
    retrieved_chunk_ids = [chunk.get('chunk_id', '') for chunk in retrieved_chunks]
    
    valid_citations = []
    invalid_citations = []
    
    for citation in citations:
        if citation in retrieved_chunk_ids:
            valid_citations.append(citation)
        else:
            invalid_citations.append(citation)
    
    return {
        'total_citations': len(citations),
        'valid_citations': valid_citations,
        'invalid_citations': invalid_citations,
        'citation_coverage': len(valid_citations) / len(retrieved_chunk_ids) if retrieved_chunk_ids else 0
    }

