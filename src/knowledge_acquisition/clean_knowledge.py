"""
Knowledge acquisition and cleaning module for Typeform RAG prototype.

This module processes raw HTML files from the Typeform Help Center,
extracts and cleans the content, and saves structured documents with metadata.
"""

import os
import json
import re
import unicodedata
from datetime import datetime
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import config and utils
import sys
from config import (
    RAW_DOCS_DIR, CLEAN_DOCS_ARTIFACTS_DIR, CLEAN_DOCS_SUMMARY_DIR,
    construct_path
)
from src.utils import analyze_documents, save_document_profile

# =============================================================================
# TEXT NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_text(text: str) -> str:
    """
    Apply comprehensive text normalization while preserving structural markers.
    
    Args:
        text: Raw text to normalize
        
    Returns:
        str: Normalized text with preserved structure
    """
    if not text:
        return ""
    
    # Unicode normalization (NFC)
    text = unicodedata.normalize('NFC', text)
    
    # Preserve structural markers before normalization
    # Replace structural markers with placeholders
    text = re.sub(r'^#{1,6}\s+', '[HEADING]', text, flags=re.MULTILINE)
    text = re.sub(r'^(Step \d+:)', '[STEP_START]\\1[STEP_END]', text, flags=re.MULTILINE)
    text = re.sub(r'^\[NOTE\]', '[NOTE_PLACEHOLDER]', text, flags=re.MULTILINE)
    text = re.sub(r'^-\s+', '[BULLET]', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '[NUMBERED]', text, flags=re.MULTILINE)
    
    # Case normalization (lowercase) - but preserve placeholders
    text = text.lower()
    
    # Restore structural markers
    text = re.sub(r'\[heading\]', '##', text, flags=re.MULTILINE)
    text = re.sub(r'\[step_start\](.*?)\[step_end\]', '\\1', text, flags=re.MULTILINE)
    text = re.sub(r'\[note_placeholder\]', '[NOTE]', text, flags=re.MULTILINE)
    text = re.sub(r'\[bullet\]', '-', text, flags=re.MULTILINE)
    text = re.sub(r'\[numbered\]', '1.', text, flags=re.MULTILINE)
    
    # Whitespace standardization - preserve paragraph breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Preserve double newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces/tabs with single space
    text = text.strip()
    
    # Standardize bullet points (for any remaining ones)
    text = re.sub(r'[•·▪▫‣⁃]', '-', text)
    
    # Placeholder substitution for URLs
    text = re.sub(r'https?://[^\s]+', '[URL]', text)
    
    # Placeholder substitution for dates (basic patterns)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', text)
    text = re.sub(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', '[DATE]', text)
    
    # Placeholder substitution for phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    
    return text

def extract_structured_content(soup: BeautifulSoup) -> str:
    """
    Extract main content from Typeform Help Center HTML while preserving structure.
    
    Args:
        soup: BeautifulSoup object of the HTML
        
    Returns:
        str: Extracted main content text with preserved structure
    """
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "header", "footer"]):
        script.decompose()
    
    # Try to find the main article content
    main_content = None
    
    # Look for common article content selectors
    selectors = [
        'article',
        '.article-body',
        '.article-content',
        '.content.article-content',
        '.content',
        '.main-content',
        '[role="main"]'
    ]
    
    for selector in selectors:
        main_content = soup.select_one(selector)
        if main_content:
            break
    
    # If no specific article container found, use body
    if not main_content:
        main_content = soup.find('body')
    
    if not main_content:
        logger.warning("Could not find main content container")
        return ""
    
    # Extract all text content with proper spacing
    structured_text = main_content.get_text(separator=' ', strip=True)
    
    # Clean up common navigation and boilerplate text
    boilerplate_patterns = [
        r'help center',
        r'search.*help',
        r'contact.*support',
        r'privacy.*policy',
        r'terms.*service',
        r'cookie.*policy',
        r'©.*typeform',
        r'all rights reserved',
        r'was this article helpful\?',
        r'yes.*no',
        r'feedback',
        r'related articles',
        r'see also',
        r'next.*previous',
        r'back to.*top'
    ]
    
    # Apply targeted boilerplate removal (more conservative)
    for pattern in boilerplate_patterns:
        # Only remove if it's at the end of the text (likely footer content)
        structured_text = re.sub(pattern + r'$', '', structured_text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Apply minimal post-processing for readability
    structured_text = _clean_extracted_text(structured_text)
    
    return structured_text

def _clean_extracted_text(text: str) -> str:
    """
    Clean extracted text with minimal post-processing for readability.
    
    Args:
        text: Raw extracted text from HTML
        
    Returns:
        str: Cleaned text with normalized whitespace
    """
    if not text:
        return ""
    
    # Normalize whitespace - multiple spaces to single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize newlines - multiple newlines to max double newline
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove excessive punctuation artifacts
    text = re.sub(r'[.]{3,}', '...', text)  # Multiple dots to ellipsis
    text = re.sub(r'[-]{3,}', '---', text)  # Multiple dashes to triple dash
    
    # Clean up common HTML artifacts
    text = re.sub(r'\s+([.!?])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Ensure space after sentence
    
    return text.strip()


def extract_metadata(soup: BeautifulSoup, file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from HTML document.
    
    Args:
        soup: BeautifulSoup object of the HTML
        file_path: Path to the source file
        
    Returns:
        dict: Extracted metadata
    """
    metadata = {
        'read_timestamp': datetime.now().isoformat(),
        'source_file': os.path.basename(file_path)
    }
    
    # Extract title
    title_tag = soup.find('title')
    if title_tag:
        title = title_tag.get_text().strip()
        # Clean up title (remove "– Help Center" suffix)
        title = re.sub(r'\s*–\s*Help Center.*$', '', title)
        metadata['doc_title'] = title
    else:
        metadata['doc_title'] = 'Unknown Title'
    
    # Extract URL from canonical link or meta tags
    canonical_link = soup.find('link', rel='canonical')
    if canonical_link and canonical_link.get('href'):
        metadata['full_doc_url'] = canonical_link['href']
    else:
        # Try to extract from og:url meta tag
        og_url = soup.find('meta', property='og:url')
        if og_url and og_url.get('content'):
            metadata['full_doc_url'] = og_url['content']
        else:
            metadata['full_doc_url'] = ''
    
    # Extract description
    description_tag = soup.find('meta', attrs={'name': 'description'})
    if description_tag and description_tag.get('content'):
        metadata['description'] = description_tag['content']
    
    # Extract language
    html_tag = soup.find('html')
    if html_tag and html_tag.get('lang'):
        metadata['language'] = html_tag['lang']
    else:
        metadata['language'] = 'en'
    
    # Generate document ID from title or URL
    if metadata['full_doc_url']:
        # Extract article ID from URL
        match = re.search(r'articles/(\d+)', metadata['full_doc_url'])
        if match:
            metadata['doc_id'] = f"art_{match.group(1)}"
        else:
            # Fallback to filename
            metadata['doc_id'] = os.path.splitext(os.path.basename(file_path))[0]
    else:
        metadata['doc_id'] = os.path.splitext(os.path.basename(file_path))[0]
    
    return metadata

# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def process_html_file(file_path: str) -> Dict[str, Any]:
    """
    Process a single HTML file and extract cleaned content.
    
    Args:
        file_path: Path to the HTML file
        
    Returns:
        dict: Processed document with text and metadata
    """
    try:
        logger.info(f"Processing file: {file_path}")
        
        # Read HTML file
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract metadata
        metadata = extract_metadata(soup, file_path)
        
        # Extract main content
        raw_text = extract_structured_content(soup)
        
        # Normalize text
        normalized_text = normalize_text(raw_text)
        
        # Add text statistics
        metadata['num_words'] = len(normalized_text.split())
        metadata['num_characters'] = len(normalized_text)
        
        # Create document
        document = {
            **metadata,
            'text': normalized_text
        }
        
        logger.info(f"Processed {metadata['doc_id']}: {metadata['num_words']} words, {metadata['num_characters']} characters")
        return document
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        raise

def process_all_html_files() -> List[Dict[str, Any]]:
    """
    Process all HTML files in the raw_docs directory.
    
    Returns:
        list: List of processed documents
    """
    documents = []
    
    # Get all HTML files in raw_docs directory
    html_files = [f for f in os.listdir(RAW_DOCS_DIR) if f.endswith('.html')]
    
    if not html_files:
        logger.warning(f"No HTML files found in {RAW_DOCS_DIR}")
        return documents
    
    logger.info(f"Found {len(html_files)} HTML files to process")
    
    for html_file in html_files:
        file_path = os.path.join(RAW_DOCS_DIR, html_file)
        try:
            document = process_html_file(file_path)
            documents.append(document)
        except Exception as e:
            logger.error(f"Failed to process {html_file}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(documents)} documents")
    return documents

def save_cleaned_documents(documents: List[Dict[str, Any]]) -> None:
    """
    Save cleaned documents to individual JSON files.
    
    Args:
        documents: List of processed documents
    """
    try:
        # Ensure output directory exists (using centralized path management)
        artifacts_dir = construct_path('clean_docs_artifacts_dir')
        if artifacts_dir:
            os.makedirs(artifacts_dir, exist_ok=True)
        
        for doc in documents:
            doc_id = doc['doc_id']
            output_file = os.path.join(artifacts_dir, f"{doc_id}.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved document: {output_file}")
        
        logger.info(f"Saved {len(documents)} cleaned documents")
        
    except Exception as e:
        logger.error(f"Error saving cleaned documents: {e}")
        raise

def generate_document_profile(documents: List[Dict[str, Any]]) -> None:
    """
    Generate and save document profile statistics.
    
    Args:
        documents: List of processed documents
    """
    try:
        # Ensure summary directory exists (using centralized path management)
        summary_dir = construct_path('clean_docs_summary_dir')
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        
        # Analyze documents
        analysis = analyze_documents(documents)
        
        # Save document profile CSV
        doc_stats_path = os.path.join(summary_dir, "doc_stats.csv")
        save_document_profile(documents, doc_stats_path)
        
        # Save analysis summary
        analysis_path = os.path.join(summary_dir, "doc_analysis.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated document profile: {doc_stats_path}")
        logger.info(f"Generated analysis summary: {analysis_path}")
        
        # Log summary statistics
        logger.info("=== DOCUMENT ANALYSIS SUMMARY ===")
        logger.info(f"Total documents: {analysis.get('total_documents', 0)}")
        if 'text_length_stats' in analysis:
            stats = analysis['text_length_stats']
            logger.info(f"Text length - Mean: {stats['mean']:.0f}, Min: {stats['min']}, Max: {stats['max']}")
        if 'word_count_stats' in analysis:
            stats = analysis['word_count_stats']
            logger.info(f"Word count - Mean: {stats['mean']:.0f}, Min: {stats['min']}, Max: {stats['max']}")
        
    except Exception as e:
        logger.error(f"Error generating document profile: {e}")
        raise

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run the knowledge acquisition and cleaning pipeline.
    """
    try:
        logger.info("Starting knowledge acquisition and cleaning pipeline")
        
        # Process all HTML files
        documents = process_all_html_files()
        
        if not documents:
            logger.error("No documents were processed successfully")
            return
        
        # Save cleaned documents
        save_cleaned_documents(documents)
        
        # Generate document profile
        generate_document_profile(documents)
        
        logger.info("Knowledge acquisition and cleaning pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
