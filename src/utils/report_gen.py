"""
Automated Forensic Audit PDF & HTML Report Generator
Creates academic and courtroom-ready media authentication certificates.
"""

import os
import io
import hashlib
import datetime
from pathlib import Path
from typing import Dict, Any
from PIL import Image
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


def generate_forensic_pdf(
    image: Image.Image,
    analysis_results: Dict[str, Any],
    output_path: Path,
    filename: str = "analyzed_media.jpg"
) -> Path:
    """
    Generates a PDF Forensic Authentication Report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute SHA-256 Hash of Image
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    media_sha256 = hashlib.sha256(img_bytes).hexdigest()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1E293B'),
        alignment=TA_CENTER
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748B'),
        alignment=TA_CENTER
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )

    elements = []

    # Title & Header
    elements.append(Paragraph("AI-DRIVEN SYNTHETIC MEDIA DETECTION ENGINE", title_style))
    elements.append(Paragraph("FORENSIC AUTHENTICITY AUDIT REPORT & INTEGRITY CERTIFICATE", subtitle_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284C7'), spaceBefore=2, spaceAfter=10))

    # Metadata & Timestamp
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    meta_data = [
        [Paragraph(f"<b>Target Filename:</b> {filename}", body_style), Paragraph(f"<b>Inspection Date:</b> {now_str}", body_style)],
        [Paragraph(f"<b>Image Dimensions:</b> {image.width} x {image.height} px", body_style), Paragraph(f"<b>Color Mode:</b> {image.mode}", body_style)],
        [Paragraph(f"<b>SHA-256 Media Hash:</b> <font size=7>{media_sha256[:32]}...{media_sha256[-8:]}</font>", body_style), Paragraph("<b>Classification Engine:</b> EfficientNet-B4 + Multi-Spectral Fusion", body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 12))

    # Verdict Box
    verdict = analysis_results["verdict"]
    synth_prob = analysis_results["synthetic_probability"]
    confidence = analysis_results["confidence"]
    badge_col = colors.HexColor('#DC2626') if "Synthetic" in verdict else (colors.HexColor('#16A34A') if "Authentic" in verdict else colors.HexColor('#D97706'))

    verdict_style = ParagraphStyle(
        'VerdictBox',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=badge_col,
        alignment=TA_CENTER
    )
    verdict_data = [
        [Paragraph(f"FINAL FORENSIC VERDICT: {verdict.upper()}", verdict_style)],
        [Paragraph(f"Synthetic Probability: <b>{synth_prob:.1f}%</b>  |  Overall Model Confidence: <b>{confidence:.1f}%</b>  |  Authenticity Index: <b>{100 - synth_prob:.1f}%</b>", ParagraphStyle('SubVerdict', parent=body_style, alignment=TA_CENTER))]
    ]
    verdict_table = Table(verdict_data, colWidths=[540])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FEF2F2') if "Synthetic" in verdict else (colors.HexColor('#F0FDF4') if "Authentic" in verdict else colors.HexColor('#FFFBEB'))),
        ('BOX', (0, 0), (-1, -1), 1.5, badge_col),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    elements.append(verdict_table)
    elements.append(Spacer(1, 12))

    # Multi-Evidence Signal Breakdown
    elements.append(Paragraph("Orthogonal Forensic Signal Analysis", section_heading))
    sig = analysis_results["signals"]
    sig_data = [
        ["Forensic Signal", "Methodology / Theory", "Weight", "Score", "Severity Rating"],
        ["Deep Learning Features", "EfficientNet-B4 Spatial Anomaly Classifier", sig["deep_learning"]["weight"], f"{sig['deep_learning']['score']:.1f}%", sig["deep_learning"]["severity"]],
        ["2D FFT Spectral Analysis", "Azimuthal High-Frequency Decay & Upsampling Peaks", sig["fft_spectral"]["weight"], f"{sig['fft_spectral']['score']:.1f}%", sig["fft_spectral"]["severity"]],
        ["Error Level Analysis (ELA)", "JPEG Compression Consistency & Residual Variance", sig["ela_compression"]["weight"], f"{sig['ela_compression']['score']:.1f}%", sig["ela_compression"]["severity"]],
        ["Sensor Noise Residuals", "High-Frequency PRNU Variance & Spatial Homogeneity", sig["sensor_noise"]["weight"], f"{sig['sensor_noise']['score']:.1f}%", sig["sensor_noise"]["severity"]],
        ["Chrominance Dispersion", "YCbCr & HSV Color Space Correlation Discrepancy", "Advisory", f"{sig['color_chrominance']['score']:.1f}%", sig["color_chrominance"]["severity"]],
    ]
    sig_table = Table(sig_data, colWidths=[120, 210, 50, 60, 100])
    sig_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('ALIGN', (2, 0), (3, -1), 'CENTER'),
    ]))
    elements.append(sig_table)
    elements.append(Spacer(1, 12))

    # Visual Evidence Images (Original, Grad-CAM, FFT, ELA)
    elements.append(Paragraph("Visual Forensic Evidence Artifacts", section_heading))
    
    # Save temporary images to buffer for ReportLab
    temp_imgs = []
    
    # 1. Original
    orig_thumb = image.copy()
    orig_thumb.thumbnail((200, 200))
    b_orig = io.BytesIO()
    orig_thumb.save(b_orig, format='JPEG')
    b_orig.seek(0)
    
    # 2. Grad-CAM
    if analysis_results["gradcam"]["overlay_rgb"] is not None:
        cam_pil = Image.fromarray(analysis_results["gradcam"]["overlay_rgb"])
        cam_pil.thumbnail((200, 200))
        b_cam = io.BytesIO()
        cam_pil.save(b_cam, format='JPEG')
        b_cam.seek(0)
    else:
        b_cam = b_orig

    # 3. FFT
    fft_pil = Image.fromarray(analysis_results["signals"]["fft_spectral"]["visualization"])
    fft_pil.thumbnail((200, 200))
    b_fft = io.BytesIO()
    fft_pil.save(b_fft, format='JPEG')
    b_fft.seek(0)

    # 4. ELA
    ela_pil = Image.fromarray(analysis_results["signals"]["ela_compression"]["visualization"])
    ela_pil.thumbnail((200, 200))
    b_ela = io.BytesIO()
    ela_pil.save(b_ela, format='JPEG')
    b_ela.seek(0)

    img_grid = [
        [RLImage(b_orig, width=1.7*inch, height=1.7*inch), RLImage(b_cam, width=1.7*inch, height=1.7*inch), RLImage(b_fft, width=1.7*inch, height=1.7*inch)],
        [Paragraph("<b>Original Upload</b>", ParagraphStyle('c1', parent=body_style, alignment=TA_CENTER)),
         Paragraph("<b>Grad-CAM Heatmap</b>", ParagraphStyle('c2', parent=body_style, alignment=TA_CENTER)),
         Paragraph("<b>2D FFT Spectrum</b>", ParagraphStyle('c3', parent=body_style, alignment=TA_CENTER))]
    ]
    img_table = Table(img_grid, colWidths=[180, 180, 180])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 2),
    ]))
    elements.append(img_table)
    elements.append(Spacer(1, 10))

    # Academic & Forensic Note
    note_text = "<b>Methodological Note:</b> This report is generated by a multi-evidence synthetic media detection architecture combining convolutional deep feature representations with signal-level physical consistency metrics (Fourier analysis, PRNU residuals, and compression error levels). Visual heatmaps highlight pixel regions contributing most heavily to classification logits."
    elements.append(Paragraph(note_text, ParagraphStyle('NoteStyle', parent=body_style, fontSize=7.5, textColor=colors.HexColor('#64748B'))))

    doc.build(elements)
    return output_path
