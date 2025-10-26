from typing import List, Dict, Any, Optional
from prompts.system_prompts import extract_citations_from_response, validate_citations

# =============================================================================
# RESPONSE ANALYSIS FUNCTIONS
# =============================================================================

def analyze_response_quality(response_data: Dict[str, Any], retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the quality of a generated response.
    
    Args:
        response_data: Response data from generate_response
        retrieved_chunks: List of retrieved chunks used for context
        
    Returns:
        dict: Quality analysis
    """
    answer = response_data.get('answer', '')
    
    # Extract and validate citations directly from the answer
    citations = extract_citations_from_response(answer)
    citation_validation = validate_citations(answer, retrieved_chunks)
    
    # Basic quality metrics
    analysis = {
        'answer_length': len(answer),
        'word_count': len(answer.split()),
        'num_citations': len(citations),
        'citation_coverage': citation_validation.get('citation_coverage', 0.0),
        'valid_citations': len(citation_validation.get('valid_citations', [])),
        'invalid_citations': len(citation_validation.get('invalid_citations', [])),
        'generation_time': response_data.get('generation_time_seconds', 0.0),
        'tokens_used': response_data.get('tokens_used', 0)
    }
    
    # Check for fallback response
    fallback_indicators = [
        "cannot find enough information",
        "not enough information",
        "unable to find",
        "no information available"
    ]
    
    is_fallback = any(indicator in answer.lower() for indicator in fallback_indicators)
    analysis['is_fallback_response'] = is_fallback
    
    # Check for proper citation format
    has_citations = len(citations) > 0
    analysis['has_citations'] = has_citations
    
    return analysis