"""
Utility functions and configuration constants for Sentiment-to-Price Correlation Analyzer.
"""

import os
import sys
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Project Directory Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "output" / "figures"
OUTPUT_REPORTS_DIR = PROJECT_ROOT / "output" / "reports"

# Ensure required directories exist
for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, OUTPUT_FIGURES_DIR, OUTPUT_REPORTS_DIR]:
    os.makedirs(path, exist_ok=True)

# Universe of 20 NSE Stocks
NSE_STOCKS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", 
    "ITC", "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", 
    "TATAMOTORS", "AXISBANK", "LT", "WIPRO", "HCLTECH", 
    "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "NTPC"
]

def get_symbol_with_suffix(symbol: str) -> str:
    """Appends .NS suffix for Yahoo Finance if not already present."""
    symbol = symbol.strip().upper()
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"

def setup_logger(name: str = "sentiment_analyzer") -> logging.Logger:
    """Configures and returns a logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        log_file = PROJECT_ROOT / "pipeline.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def save_figure(fig, filename: str) -> str:
    """Saves a matplotlib/plotly figure to the output/figures directory."""
    filepath = OUTPUT_FIGURES_DIR / filename
    if hasattr(fig, "savefig"):
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    elif hasattr(fig, "write_image"):
        fig.write_image(str(filepath))
    return str(filepath)

def export_markdown_to_pdf(markdown_text: str, pdf_filename: str) -> str:
    """
    Converts a Markdown string to a PDF report using ReportLab.
    Falls back cleanly if ReportLab formatting hits issues.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    
    pdf_path = OUTPUT_REPORTS_DIR / pdf_filename
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    # Title style
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        spaceAfter=12
    )

    lines = markdown_text.split('\n')
    for line in lines:
        line_str = line.strip()
        if not line_str:
            story.append(Spacer(1, 6))
            continue
            
        # Clean formatting tags
        clean_text = line_str.replace('**', '').replace('`', '')
        if clean_text.startswith('# '):
            story.append(Paragraph(clean_text[2:], title_style))
        elif clean_text.startswith('## '):
            story.append(Paragraph(clean_text[3:], styles['Heading2']))
        elif clean_text.startswith('### '):
            story.append(Paragraph(clean_text[4:], styles['Heading3']))
        elif clean_text.startswith('- ') or clean_text.startswith('* '):
            story.append(Paragraph(f"• {clean_text[2:]}", styles['Normal']))
        else:
            story.append(Paragraph(clean_text, styles['Normal']))
        story.append(Spacer(1, 4))
        
    doc.build(story)
    return str(pdf_path)
