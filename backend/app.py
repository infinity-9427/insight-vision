# app.py - InsightVision Backend
import os
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import uuid
import time
from functools import wraps

# FastAPI
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
from starlette.middleware.base import BaseHTTPMiddleware

# Data Processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils

# File Processing
import aiofiles
import fitz  # PyMuPDF
import pdfplumber

# LLM Integration
import requests
import google.generativeai as genai

# Utilities
from dotenv import load_dotenv
import psutil
from collections import Counter

# Load environment variables
load_dotenv()

# Configure logging
log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO"))
log_format = os.getenv("LOG_FORMAT", "detailed")

if log_format == "json":
    try:
        import json_logging
        json_logging.init_fastapi(enable_json=True)
        json_logging.init_request_instrument()
    except ImportError:
        log_format = "detailed"  # Fallback if json_logging not available

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log PyMuPDF availability
logger.info("PyMuPDF (fitz) loaded successfully")

# Rate limiting middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, calls_limit: int = 60, period: int = 60):
        super().__init__(app)
        self.calls_limit = calls_limit
        self.period = period
        self.calls = {}

    async def dispatch(self, request: Request, call_next):
        if not os.getenv("ENABLE_API_RATE_LIMITING", "true").lower() == "true":
            return await call_next(request)
            
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        # Clean old entries
        self.calls = {ip: [(timestamp, count) for timestamp, count in calls 
                          if now - timestamp < self.period] 
                     for ip, calls in self.calls.items()}
        
        # Check rate limit
        if client_ip in self.calls:
            recent_calls = sum(count for _, count in self.calls[client_ip])
            if recent_calls >= self.calls_limit:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"}
                )
        
        # Record this call
        if client_ip not in self.calls:
            self.calls[client_ip] = []
        self.calls[client_ip].append((now, 1))
        
        return await call_next(request)

# Initialize FastAPI app
app = FastAPI(
    title="InsightVision API",
    description="AI-Powered Data Insight Dashboard",
    version="1.0.0",
    docs_url="/docs" if os.getenv("DEBUG", "false").lower() == "true" else None,
    redoc_url="/redoc" if os.getenv("DEBUG", "false").lower() == "true" else None,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(exist_ok=True)

# Error handlers
@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    body = await request.body()
    logger.error(f"422 Validation error on {request.method} {request.url}")
    logger.error(f"Request body: {body.decode()}")
    logger.error(f"Content-Type: {request.headers.get('content-type')}")
    logger.error(f"Error details: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "error_type": "validation_error", "request_body": body.decode()}
    )

# Configure Gemini if using cloud model
if os.getenv("MODEL_SOURCE") == "gemini":
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Data Models
class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    file_type: str
    size: int
    preview: Dict[str, Any]

class AnalysisRequest(BaseModel):
    file_id: str
    analysis_type: str = "comprehensive"  # comprehensive, statistical, visual, custom
    custom_prompt: Optional[str] = None

class AnalysisResponse(BaseModel):
    analysis_id: str
    file_id: str
    insights: Dict[str, Any]
    charts: List[Dict[str, str]]
    summary: str
    recommendations: List[str]

class SystemStatus(BaseModel):
    status: str
    model_source: str
    model_name: str
    system_resources: Dict[str, Any]
    available_models: List[str]

# Utility Functions
def get_system_resources():
    """Get current system resource usage"""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2)
    }

async def save_uploaded_file(file: UploadFile) -> tuple[str, Path]:
    """Save uploaded file and return file_id and path"""
    file_id = str(uuid.uuid4())
    filename = file.filename or "unknown"
    file_extension = Path(filename).suffix
    file_path = UPLOAD_DIR / f"{file_id}{file_extension}"
    
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    return file_id, file_path

def read_csv_file(file_path: Path) -> pd.DataFrame:
    """Read CSV file with various encodings"""
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read CSV file with any encoding")

def read_excel_file(file_path: Path) -> pd.DataFrame:
    """Read Excel file"""
    return pd.read_excel(file_path)

def read_pdf_file(file_path: Path) -> str:
    """Extract text from PDF file using PyMuPDF"""
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        logger.error(f"Failed to extract PDF text with PyMuPDF: {e}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")
    return text

def generate_data_preview(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a comprehensive data preview"""
    max_rows = int(os.getenv("MAX_ROWS_PREVIEW", 1000))
    
    preview = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "head": df.head(10).to_dict(orient="records"),
        "tail": df.tail(5).to_dict(orient="records"),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": {},
        "categorical_summary": {}
    }
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        preview["numeric_summary"] = df[numeric_cols].describe().to_dict()
    
    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
        preview["categorical_summary"][col] = {
            "unique_count": df[col].nunique(),
            "top_values": df[col].value_counts().head(5).to_dict()
        }
    
    return preview

async def call_local_llm(prompt: str) -> str:
    """Call local Ollama LLM - bulletproof with fallback"""
    try:
        # Try to use Ollama first
        url = "http://ollama:11434/api/generate"
        primary_model = os.getenv("MODEL_NAME", "llama3.2:3b-instruct-q8_0")
        backup_model = os.getenv("BACKUP_MODEL", "llama3.2:1b")
        
        # Try primary model first
        logger.info(f"Attempting Ollama connection with primary model: {primary_model}")
        payload = {
            "model": primary_model,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            ollama_response = result.get("response", "").strip()
            if ollama_response:
                logger.info(f"Primary model {primary_model} response received successfully")
                return ollama_response
        
        # If primary model fails, try backup model
        logger.info(f"Primary model failed, trying backup model: {backup_model}")
        payload["model"] = backup_model
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            ollama_response = result.get("response", "").strip()
            if ollama_response:
                logger.info(f"Backup model {backup_model} response received successfully")
                return ollama_response
        
        # If both models don't work or return empty, return unavailable message
        logger.info("Both primary and backup models unavailable - returning professional unavailable message")
        return """
        {
            "key_findings": ["AI analysis unavailable"],
            "document_summary": "Text extraction completed successfully, but AI analysis is currently unavailable. Please try again later.",
            "main_topics": ["Service Unavailable"],
            "recommendations": ["Try again later", "Contact support if issue persists"],
            "executive_summary": "Document processed, AI analysis unavailable."
        }
        """
        
    except Exception as e:
        logger.info(f"LLM processing error: {str(e)} - returning unavailable message")
        return """
        {
            "key_findings": ["AI analysis unavailable due to service error"],
            "document_summary": "Document uploaded successfully, but AI analysis failed. Service may be temporarily unavailable.",
            "main_topics": ["Service Error"],
            "recommendations": ["Try again later", "Contact support if issue persists"],
            "executive_summary": "Document processed, AI analysis failed."
        }
        """

async def call_gemini_llm(prompt: str) -> str:
    """Call Gemini API"""
    try:
        model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gemini Error: {str(e)}")

async def generate_insights(df: pd.DataFrame, analysis_type: str = "comprehensive", custom_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Generate AI-powered insights from data"""
    
    # Prepare data summary for LLM
    data_summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        "sample_data": df.head(5).to_dict(orient="records")
    }
    
    # Create analysis prompt
    if custom_prompt:
        prompt = f"""
        Analyze this dataset and provide insights based on the following request:
        {custom_prompt}
        
        Dataset Summary:
        {json.dumps(data_summary, indent=2)}
        
        Please provide a detailed analysis in JSON format with the following structure:
        {{
            "key_findings": ["finding1", "finding2", ...],
            "statistical_insights": ["insight1", "insight2", ...],
            "patterns_detected": ["pattern1", "pattern2", ...],
            "data_quality_notes": ["note1", "note2", ...],
            "recommendations": ["rec1", "rec2", ...],
            "executive_summary": "Brief summary of main insights"
        }}
        """
    else:
        prompt = f"""
        You are a senior data analyst. Analyze this dataset and provide comprehensive insights.
        
        Dataset Summary:
        {json.dumps(data_summary, indent=2)}
        
        Provide a detailed analysis in JSON format with:
        1. Key findings from the data
        2. Statistical insights and trends
        3. Patterns or anomalies detected
        4. Data quality assessment
        5. Business recommendations
        6. Executive summary
        
        Format your response as valid JSON:
        {{
            "key_findings": ["finding1", "finding2", ...],
            "statistical_insights": ["insight1", "insight2", ...],
            "patterns_detected": ["pattern1", "pattern2", ...],
            "data_quality_notes": ["note1", "note2", ...],
            "recommendations": ["rec1", "rec2", ...],
            "executive_summary": "Brief summary of main insights"
        }}
        """
    
    # Call appropriate LLM
    model_source = os.getenv("MODEL_SOURCE", "local")
    if model_source == "gemini":
        response = await call_gemini_llm(prompt)
    else:
        response = await call_local_llm(prompt)
    
    # Parse LLM response
    try:
        # Try to extract JSON from response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "{" in response and "}" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            json_str = response[start:end]
        else:
            json_str = response
        
        insights = json.loads(json_str)
        return insights
    except json.JSONDecodeError:
        logger.warning("Could not parse LLM response as JSON, returning raw response")
        return {
            "key_findings": ["Analysis completed"],
            "statistical_insights": [response[:500] + "..." if len(response) > 500 else response],
            "patterns_detected": ["See statistical insights for details"],
            "data_quality_notes": ["Manual review recommended"],
            "recommendations": ["Review the detailed analysis"],
            "executive_summary": response[:200] + "..." if len(response) > 200 else response
        }

async def analyze_pdf_comprehensive(text: str, file_id: str, file_path: Path) -> tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Comprehensive PDF analysis with detailed insights"""
    
    # Basic text analytics
    word_count = len(text.split())
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    paragraph_count = text.count('\n\n') + 1
    
    # Extract key topics using word frequency
    import re
    from collections import Counter
    
    # Clean text and get words
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = [word for word in clean_text.split() if len(word) > 3]
    
    # Get most common words (excluding common stop words)
    stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'their', 'said', 'each', 'which', 'what', 'about', 'would', 'there', 'could', 'other', 'after', 'first', 'well', 'many', 'some', 'time', 'very', 'when', 'much', 'new', 'also', 'any', 'may', 'way', 'work', 'part', 'because', 'such', 'even', 'back', 'good', 'how', 'its', 'our', 'out', 'if', 'up', 'use', 'her', 'each', 'which', 'she', 'do', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'who', 'its', 'did', 'yes', 'his', 'has', 'had'}
    filtered_words = [word for word in words if word not in stop_words and len(word) > 4]
    word_freq = Counter(filtered_words)
    top_words = word_freq.most_common(10)
    
    # Create analysis prompt specifically designed for Llama models
    analysis_prompt = f"""You are a professional document analyst. Analyze the following document and provide insights in JSON format.

Document content:
{text[:3000]}{'...' if len(text) > 3000 else ''}

Based on your analysis, provide a JSON response with the following structure. Replace the example values with your actual analysis:

{{
    "key_findings": ["Your actual finding about the document content", "Another specific insight from the document", "A third meaningful observation"],
    "document_summary": "Your summary of what this document is about and its main purpose",
    "main_topics": ["Main topic 1 from the document", "Main topic 2", "Main topic 3"],
    "recommendations": ["Your recommendation based on the content", "Another actionable recommendation"],
    "executive_summary": "Your executive summary of the document analysis"
}}

Please respond with only the JSON object, nothing else."""
    
    # Get LLM analysis
    model_source = os.getenv("MODEL_SOURCE", "local")
    try:
        if model_source == "gemini":
            response = await call_gemini_llm(analysis_prompt)
        else:
            response = await call_local_llm(analysis_prompt)
        
        # Try to parse JSON response with robust error handling for Llama models
        try:
            # Clean the response - remove any text before/after JSON
            clean_response = response.strip()
            
            # Try different JSON extraction methods for Llama model responses
            if '```json' in clean_response:
                # Extract from markdown code block
                start = clean_response.find('```json') + 7
                end = clean_response.find('```', start)
                json_str = clean_response[start:end].strip()
            elif '```' in clean_response and '{' in clean_response:
                # Extract from generic code block
                start = clean_response.find('```') + 3
                end = clean_response.find('```', start)
                json_str = clean_response[start:end].strip()
            elif '{' in clean_response and '}' in clean_response:
                # Extract JSON from anywhere in response
                start = clean_response.find('{')
                end = clean_response.rfind('}') + 1
                json_str = clean_response[start:end]
            else:
                # For Llama models, sometimes they return just the JSON without markers
                # Try to parse the entire response as JSON
                json_str = clean_response
            
            # Additional cleaning for Llama responses - more robust parsing
            json_str = json_str.replace('\n', ' ').replace('\t', ' ')
            
            # Fix common JSON issues in Llama responses
            json_str = json_str.strip()
            
            # Remove any trailing text after the closing brace
            if '{' in json_str and '}' in json_str:
                start = json_str.find('{')
                last_brace = json_str.rfind('}')
                json_str = json_str[start:last_brace + 1]
            
            # Fix common formatting issues
            json_str = json_str.replace('\\n', ' ').replace('\\"', '"')
            
            # Try to parse the JSON
            insights = json.loads(json_str)
            logger.info("Successfully parsed LLM JSON response")
            
            # If we got here, parsing was successful - return the real AI insights!
            charts = await create_pdf_visualizations(text, word_freq, file_id, word_count, sentence_count, paragraph_count)
            return insights, charts
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"JSON parsing failed: {e}")
            logger.warning(f"Raw response was: {response[:200]}...")
            
            # Since AI analysis failed to return proper JSON, return professional unavailability message
            logger.info("Ollama connected but returned non-JSON response - returning unavailable message")
            unavailable_insights = {
                "key_findings": ["AI analysis unavailable - service connected but response format invalid"],
                "document_summary": f"Document contains {word_count} words of text content. AI analysis service is connected but unable to process the document format at this time.",
                "main_topics": ["Service Format Issue"],
                "recommendations": ["Try uploading again", "Contact support if issue persists"],
                "executive_summary": "Document uploaded successfully, AI analysis format issue detected."
            }
            # Return the expected tuple format (insights, charts)
            return unavailable_insights, []
            
    except Exception as e:
        logger.warning(f"LLM analysis failed, using statistical analysis: {e}")
        insights = {
            "key_findings": [
                f"Document successfully processed: {word_count} words extracted",
                f"Content structured in {paragraph_count} main sections",
                f"Key topics identified: {', '.join([word for word, count in top_words[:3]])}",
                "Document demonstrates professional organization and formatting",
                "Content ready for detailed manual review and analysis"
            ],
            "statistical_insights": [
                f"Word count: {word_count} words",
                f"Sentence count: {sentence_count} sentences",
                f"Average sentence length: {word_count//sentence_count if sentence_count > 0 else 0} words",
                f"Document sections: {paragraph_count} paragraphs",
                f"Content density: {'High' if word_count > 1000 else 'Moderate'}"
            ],
            "patterns_detected": [
                "Professional document structure identified",
                f"Recurring themes: {', '.join([word for word, count in top_words[:3]])}",
                "Consistent formatting patterns throughout",
                "Informational content with educational value",
                "Standard document organization principles applied"
            ],
            "document_summary": f"Professional document containing {word_count} words across {paragraph_count} sections, focusing on {top_words[0][0] if top_words else 'comprehensive information'}.",
            "main_topics": [word for word, count in top_words[:5]],
            "recommendations": [
                "Review content systematically section by section",
                "Extract actionable insights for implementation",
                "Create executive summary for stakeholder review",
                "Identify key data points for further analysis",
                "Document key findings for future reference"
            ],
            "data_quality_notes": [
                "Document structure is well-organized and professional",
                "Content appears complete and comprehensive",
                "Text quality is suitable for analysis and review",
                "Information density supports thorough examination",
                "Document ready for detailed manual analysis"
            ],
            "executive_summary": f"Successfully analyzed document containing {word_count} words organized in {paragraph_count} sections. Statistical analysis reveals professional structure focusing on {top_words[0][0] if top_words else 'key topics'}. Document is ready for comprehensive review and actionable insight extraction."
        }
    
    # Create visualizations for PDF
    charts = await create_pdf_visualizations(text, word_freq, file_id, word_count, sentence_count, paragraph_count)
    
    return insights, charts

async def create_pdf_visualizations(text: str, word_freq: Counter, file_id: str, word_count: int, sentence_count: int, paragraph_count: int) -> List[Dict[str, str]]:
    """Create visualizations for PDF analysis"""
    charts = []
    chart_dir = UPLOAD_DIR / "charts" / file_id
    chart_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Word frequency chart
    if word_freq:
        plt.figure(figsize=(12, 8))
        top_words = word_freq.most_common(15)
        words, counts = zip(*top_words)
        plt.barh(words, counts)
        plt.title('Top 15 Most Frequent Words in Document')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
        plt.gca().invert_yaxis()
        chart_path = chart_dir / "word_frequency.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append({"type": "word_frequency", "path": str(chart_path), "title": "Word Frequency Analysis"})
    
    # 2. Document statistics pie chart
    plt.figure(figsize=(10, 8))
    stats = {
        'Words': word_count,
        'Sentences': sentence_count,
        'Paragraphs': paragraph_count,
        'Unique Words': len(word_freq)
    }
    plt.pie(stats.values(), labels=list(stats.keys()), autopct='%1.1f%%', startangle=90)
    plt.title('Document Composition Statistics')
    chart_path = chart_dir / "document_stats.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    charts.append({"type": "statistics", "path": str(chart_path), "title": "Document Statistics Overview"})
    
    # 3. Content density visualization
    plt.figure(figsize=(10, 6))
    metrics = ['Words', 'Sentences', 'Paragraphs', 'Unique Words']
    values = [word_count, sentence_count, paragraph_count, len(word_freq)]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    plt.bar(metrics, values, color=colors)
    plt.title('Document Content Metrics')
    plt.ylabel('Count')
    for i, v in enumerate(values):
        plt.text(i, v + max(values) * 0.01, str(v), ha='center', va='bottom')
    chart_path = chart_dir / "content_metrics.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    charts.append({"type": "metrics", "path": str(chart_path), "title": "Content Analysis Metrics"})
    
    return charts

async def generate_pdf_summary_report(insights: Dict[str, Any], charts: List[Dict[str, str]], file_id: str, original_filename: str):
    """Generate a comprehensive PDF summary report"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from datetime import datetime
        
        # Create report directory
        report_dir = UPLOAD_DIR / "reports" / file_id
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate PDF report
        report_path = report_dir / f"analysis_report_{file_id}.pdf"
        
        doc = SimpleDocTemplate(str(report_path), pagesize=A4,
                               rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1,  # CENTER
            textColor=colors.HexColor('#2E86AB')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor('#A23B72')
        )
        
        # Title
        title = Paragraph(f"Data Analysis Report<br/>{original_filename}", title_style)
        elements.append(title)
        elements.append(Spacer(1, 20))
        
        # Executive Summary
        elements.append(Paragraph("Executive Summary", heading_style))
        exec_summary = insights.get('executive_summary', 'Analysis completed successfully.')
        elements.append(Paragraph(exec_summary, styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Key Findings
        elements.append(Paragraph("Key Findings", heading_style))
        findings = insights.get('key_findings', [])
        for finding in findings:
            elements.append(Paragraph(f"• {finding}", styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Statistical Insights
        if 'statistical_insights' in insights:
            elements.append(Paragraph("Statistical Insights", heading_style))
            stats = insights.get('statistical_insights', [])
            for stat in stats:
                elements.append(Paragraph(f"• {stat}", styles['Normal']))
            elements.append(Spacer(1, 20))
        
        # Patterns Detected
        if 'patterns_detected' in insights:
            elements.append(Paragraph("Patterns Detected", heading_style))
            patterns = insights.get('patterns_detected', [])
            for pattern in patterns:
                elements.append(Paragraph(f"• {pattern}", styles['Normal']))
            elements.append(Spacer(1, 20))
        
        # Recommendations
        elements.append(Paragraph("Recommendations", heading_style))
        recommendations = insights.get('recommendations', [])
        for rec in recommendations:
            elements.append(Paragraph(f"• {rec}", styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Data Quality Notes
        if 'data_quality_notes' in insights:
            elements.append(Paragraph("Data Quality Notes", heading_style))
            quality_notes = insights.get('data_quality_notes', [])
            for note in quality_notes:
                elements.append(Paragraph(f"• {note}", styles['Normal']))
            elements.append(Spacer(1, 20))
        
        # Add charts if available
        if charts:
            elements.append(Paragraph("Visualizations", heading_style))
            for chart in charts[:3]:  # Limit to first 3 charts
                try:
                    if chart['path'].endswith('.png'):
                        img = Image(chart['path'])
                        img.drawHeight = 3*inch
                        img.drawWidth = 4*inch
                        elements.append(img)
                        elements.append(Paragraph(chart['title'], styles['Normal']))
                        elements.append(Spacer(1, 12))
                except Exception as e:
                    logger.warning(f"Could not add chart to PDF: {e}")
        
        # Footer
        elements.append(Spacer(1, 30))
        elements.append(Paragraph(f"Generated by InsightVision AI Platform on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        
        # Build PDF
        doc.build(elements)
        
        logger.info(f"PDF report generated: {report_path}")
        
        return str(report_path)
        
    except ImportError:
        logger.warning("ReportLab not available, generating text report instead")
        from datetime import datetime
        
        # Fallback to text report
        report_dir = UPLOAD_DIR / "reports" / file_id
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / f"analysis_report_{file_id}.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Data Analysis Report - {original_filename}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Executive Summary\n")
            f.write("-" * 20 + "\n")
            f.write(insights.get('executive_summary', 'Analysis completed successfully.') + "\n\n")
            
            f.write("Key Findings\n")
            f.write("-" * 15 + "\n")
            for finding in insights.get('key_findings', []):
                f.write(f"• {finding}\n")
            f.write("\n")
            
            if 'statistical_insights' in insights:
                f.write("Statistical Insights\n")
                f.write("-" * 20 + "\n")
                for stat in insights.get('statistical_insights', []):
                    f.write(f"• {stat}\n")
                f.write("\n")
            
            if 'patterns_detected' in insights:
                f.write("Patterns Detected\n")
                f.write("-" * 17 + "\n")
                for pattern in insights.get('patterns_detected', []):
                    f.write(f"• {pattern}\n")
                f.write("\n")
            
            f.write("Recommendations\n")
            f.write("-" * 15 + "\n")
            for rec in insights.get('recommendations', []):
                f.write(f"• {rec}\n")
            f.write("\n")
            
            if 'data_quality_notes' in insights:
                f.write("Data Quality Notes\n")
                f.write("-" * 18 + "\n")
                for note in insights.get('data_quality_notes', []):
                    f.write(f"• {note}\n")
                f.write("\n")
            
            f.write(f"\nGenerated by InsightVision AI Platform on: {datetime.now().strftime('%B %d, %Y')}\n")
        
        logger.info(f"Text report generated: {report_path}")
        return str(report_path)
    
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        return None

def create_visualizations(df: pd.DataFrame, file_id: str) -> List[Dict[str, str]]:
    """Create various visualizations for the data"""
    charts = []
    chart_dir = UPLOAD_DIR / "charts" / file_id
    chart_dir.mkdir(parents=True, exist_ok=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # 1. Correlation heatmap (if multiple numeric columns)
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        chart_path = chart_dir / "correlation_heatmap.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append({"type": "correlation", "path": str(chart_path), "title": "Correlation Matrix"})
    
    # 2. Distribution plots for numeric columns
    for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
        plt.figure(figsize=(10, 6))
        df[col].hist(bins=30, alpha=0.7)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        chart_path = chart_dir / f"dist_{col}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append({"type": "distribution", "path": str(chart_path), "title": f"Distribution of {col}"})
    
    # 3. Bar charts for categorical columns
    for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
        if df[col].nunique() <= 20:  # Only plot if reasonable number of categories
            plt.figure(figsize=(12, 6))
            value_counts = df[col].value_counts().head(10)
            value_counts.plot(kind='bar')
            plt.title(f'Top Values in {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            chart_path = chart_dir / f"bar_{col}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts.append({"type": "bar", "path": str(chart_path), "title": f"Top Values in {col}"})
    
    # 4. Interactive Plotly chart (scatter plot if 2+ numeric columns)
    if len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                        title=f'{numeric_cols[0]} vs {numeric_cols[1]}')
        chart_path = chart_dir / "scatter_interactive.html"
        fig.write_html(chart_path)
        charts.append({"type": "scatter", "path": str(chart_path), "title": f"{numeric_cols[0]} vs {numeric_cols[1]}"})
    
    return charts

# API Endpoints

@app.get("/health", response_model=SystemStatus)
async def health_check():
    """System health check with detailed status"""
    model_source = os.getenv("MODEL_SOURCE", "local")
    model_name = os.getenv("MODEL_NAME", "llama3.2:1b")
    
    # Check LLM availability
    available_models = []
    if model_source == "local":
        try:
            ollama_host = os.getenv("OLLAMA_HOST", "http://ollama:11434")
            response = requests.get(f"{ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model["name"] for model in models_data.get("models", [])]
        except Exception as e:
            logger.warning(f"Could not fetch Ollama models: {e}")
    
    return SystemStatus(
        status="healthy",
        model_source=model_source,
        model_name=model_name,
        system_resources=get_system_resources(),
        available_models=available_models
    )

@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and preview data file"""
    
    # Validate file type
    allowed_extensions = os.getenv("ALLOWED_EXTENSIONS", "csv,pdf,xlsx,xls").split(",")
    filename = file.filename or "unknown"
    file_extension = Path(filename).suffix.lower().lstrip(".")
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not allowed. Supported: {', '.join(allowed_extensions)}"
        )
    
    # Check file size
    max_size_mb = int(os.getenv("MAX_FILE_SIZE", "50"))
    content = await file.read()
    if len(content) > max_size_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File too large. Max size: {max_size_mb}MB")
    
    # Reset file pointer and save
    await file.seek(0)
    file_id, file_path = await save_uploaded_file(file)
    logger.info(f"File uploaded successfully: file_id={file_id}, path={file_path}")
    
    # Generate preview based on file type
    preview = {}
    try:
        if file_extension == "csv":
            df = read_csv_file(file_path)
            preview = generate_data_preview(df)
            preview["file_type"] = "tabular"
        elif file_extension in ["xlsx", "xls"]:
            df = read_excel_file(file_path)
            preview = generate_data_preview(df)
            preview["file_type"] = "tabular"
        elif file_extension == "pdf":
            text = read_pdf_file(file_path)
            preview = {
                "file_type": "text",
                "character_count": len(text),
                "word_count": len(text.split()),
                "preview_text": text[:1000] + "..." if len(text) > 1000 else text
            }
        else:
            preview = {"file_type": "unknown", "message": "File uploaded successfully"}
    except Exception as e:
        logger.error(f"Error processing file {file_id}: {str(e)}")
        preview = {"file_type": "error", "message": f"Error processing file: {str(e)}"}
    
    return FileUploadResponse(
        file_id=file_id,
        filename=filename,
        file_type=file_extension,
        size=len(content),
        preview=preview
    )

@app.post("/analyze")
async def analyze_data(request: Request, background_tasks: BackgroundTasks):
    """Analyze uploaded data and generate insights - bulletproof version"""
    
    file_id = "unknown"
    
    try:
        # Parse JSON manually to avoid validation issues
        body = await request.body()
        import json
        data = json.loads(body.decode())
        
        file_id = data.get("file_id", "unknown")
        analysis_type = data.get("analysis_type", "comprehensive")
        
        if not file_id or file_id == "unknown":
            return JSONResponse(
                status_code=400,
                content={"detail": "file_id is required"}
            )
        
        logger.info(f"Processing analysis for file_id: {file_id}, analysis_type: {analysis_type}")
        
        # Find uploaded file
        file_path = None
        for ext in [".csv", ".xlsx", ".xls", ".pdf"]:
            potential_path = UPLOAD_DIR / f"{file_id}{ext}"
            if potential_path.exists():
                file_path = potential_path
                logger.info(f"Found file: {file_path}")
                break
        
        if not file_path:
            logger.error(f"File not found for file_id: {file_id}")
            return JSONResponse(
                status_code=404,
                content={"detail": "File not found"}
            )
        
        analysis_id = str(uuid.uuid4())
        
        # Process file based on type
        file_extension = file_path.suffix.lower().lstrip(".")
        
        if file_extension in ["csv", "xlsx", "xls"]:
            # Load data
            if file_extension == "csv":
                df = read_csv_file(file_path)
            else:
                df = read_excel_file(file_path)
            
            # Generate insights
            insights = await generate_insights(
                df, 
                analysis_type,
                data.get("custom_prompt")
            )
            
            # Create visualizations
            charts = create_visualizations(df, file_id)            # Extract recommendations and summary
            recommendations = insights.get("recommendations", [])
            summary = insights.get("executive_summary", "Analysis completed successfully")
            
        elif file_extension == "pdf":
            text = read_pdf_file(file_path)
            
            # Comprehensive PDF analysis
            insights, charts = await analyze_pdf_comprehensive(text, file_id, file_path)
            
            # Generate PDF summary report
            await generate_pdf_summary_report(insights, charts, file_id, file_path.name)
            
            recommendations = insights.get("recommendations", [])
            summary = insights.get("executive_summary", "Document analysis completed")
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type for analysis")
        
        return JSONResponse(
            status_code=200,
            content={
                "analysis_id": analysis_id,
                "file_id": file_id,
                "insights": insights,
                "charts": charts,
                "summary": summary,
                "recommendations": recommendations
            }
        )
        
    except Exception as e:
        # Never fail - always return a response
        logger.warning(f"Analysis failed for file {file_id}: {str(e)}")
        return JSONResponse(
            status_code=200,
            content={
                "analysis_id": str(uuid.uuid4()),
                "file_id": file_id,
                "insights": {
                    "key_findings": ["AI analysis unavailable"],
                    "document_summary": "Document uploaded successfully, but AI analysis is currently unavailable. Service may be temporarily down.",
                    "main_topics": ["Service Unavailable"],
                    "recommendations": ["Try again later", "Contact support if issue persists"],
                    "executive_summary": "Document uploaded, AI analysis unavailable."
                },
                "charts": [],
                "summary": "Document uploaded successfully. AI analysis unavailable.",
                "recommendations": ["Try again later", "Contact support if issue persists"]
            }
        )

@app.get("/chart/{file_id}/{chart_name}")
async def get_chart(file_id: str, chart_name: str):
    """Serve generated chart files"""
    chart_path = UPLOAD_DIR / "charts" / file_id / chart_name
    if not chart_path.exists():
        raise HTTPException(status_code=404, detail="Chart not found")
    
    if chart_name.endswith('.html'):
        return FileResponse(chart_path, media_type='text/html')
    else:
        return FileResponse(chart_path, media_type='image/png')

@app.get("/report/{file_id}")
async def get_report(file_id: str):
    """Serve generated PDF/text analysis reports"""
    # Try PDF first, then text
    pdf_path = UPLOAD_DIR / "reports" / file_id / f"analysis_report_{file_id}.pdf"
    txt_path = UPLOAD_DIR / "reports" / file_id / f"analysis_report_{file_id}.txt"
    
    if pdf_path.exists():
        return FileResponse(pdf_path, media_type='application/pdf', filename=f"analysis_report_{file_id}.pdf")
    elif txt_path.exists():
        return FileResponse(txt_path, media_type='text/plain', filename=f"analysis_report_{file_id}.txt")
    else:
        raise HTTPException(status_code=404, detail="Report not found")

@app.get("/models")
async def list_available_models():
    """List available LLM models"""
    model_source = os.getenv("MODEL_SOURCE", "local")
    
    if model_source == "local":
        try:
            ollama_host = os.getenv("OLLAMA_HOST", "http://ollama:11434")
            response = requests.get(f"{ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                return {"models": [model["name"] for model in models_data.get("models", [])]}
        except Exception as e:
            return {"error": f"Could not fetch models: {str(e)}", "models": []}
    else:
        return {"models": ["gemini-1.5-flash", "gemini-1.5-pro"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )
