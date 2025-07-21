import json
import re

def clean_json_response(response_text):
    """Clean and extract JSON from LLM response"""
    try:
        # Try to parse as-is first
        return json.loads(response_text)
    except:
        pass
    
    # Try to find JSON block in response
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass
    
    # Return None if no valid JSON found
    return None

async def simple_analyze_pdf(text: str, file_id: str, file_path) -> tuple:
    """Simple PDF analysis with robust JSON handling"""
    
    # Basic text analysis
    words = text.split()
    word_count = len(words)
    sentences = text.split('.')
    sentence_count = len(sentences)
    
    # Simple LLM prompt
    prompt = f"""
    Analyze this document text and respond with ONLY a valid JSON object:

    Text: {text[:2000]}...

    {{
        "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
        "document_summary": "Summary of the document",
        "main_topics": ["Topic 1", "Topic 2"],
        "recommendations": ["Recommendation 1"],
        "executive_summary": "Executive summary"
    }}
    """
    
    try:
        # Get LLM response
        response = await call_local_llm(prompt)
        
        # Clean and parse JSON
        insights = clean_json_response(response)
        
        if insights and isinstance(insights, dict):
            return insights, []
        else:
            # Fallback to statistical analysis
            return {
                "key_findings": [
                    f"Document contains {word_count} words",
                    f"Analysis shows {sentence_count} sentences",
                    "AI analysis unavailable - showing basic statistics"
                ],
                "document_summary": f"Document with {word_count} words analyzed. AI processing unavailable.",
                "main_topics": ["Document Analysis", "Text Processing"],
                "recommendations": ["Review document manually", "Try AI analysis again later"],
                "executive_summary": f"Basic analysis completed: {word_count} words, {sentence_count} sentences. AI insights unavailable."
            }, []
            
    except Exception as e:
        # Return honest error message
        return {
            "key_findings": ["AI analysis unavailable"],
            "document_summary": "Document uploaded successfully, but AI analysis failed.",
            "main_topics": ["Service Error"],
            "recommendations": ["Try again later", "Contact support"],
            "executive_summary": "Document processed, AI analysis unavailable."
        }, []

# Test function
import asyncio

async def test():
    result = await simple_analyze_pdf("This is a test document about AI technology trends.", "test", None)
    print(json.dumps(result[0], indent=2))

if __name__ == "__main__":
    asyncio.run(test())
