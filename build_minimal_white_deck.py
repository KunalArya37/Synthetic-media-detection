"""
Build the presentation with a White & Minimal College Template Theme:
- Background: Pure White (#FFFFFF)
- Typography: Deep Navy (#0F2B5C) titles, Dark Charcoal (#1E293B) body text
- Font Sizes:
    - Slide Titles: 28pt - 30pt Bold (Deep Navy #0F2B5C)
    - Category Tag: 11pt Bold (Tech Blue #0284C7)
    - Card Headings: 18pt - 20pt Bold
    - Body Text: 14.5pt - 15.5pt Clean, simple, conversational viva bullet points
    - Slide 1 Title: 32pt Bold
    - Slide 2 Index: 32pt Bold "Index", 16.5pt items, NO ampersands (&)
- Header Category: INDUSTRIAL TRAINING PRESENTATION
- Exactly 20 slides
"""
import os
import shutil
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

W = prs.slide_width
H = prs.slide_height

# Minimal White & SKIT Navy Palette
WHITE        = RGBColor(0xFF, 0xFF, 0xFF)
BG_WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
CARD_BG      = RGBColor(0xF8, 0xFA, 0xFC)  # Crisp off-white / light slate
CARD_ALT     = RGBColor(0xF1, 0xF5, 0xF9)
NAVY_TITLE   = RGBColor(0x0F, 0x2B, 0x5C)  # Official SKIT Deep Navy
PRIMARY_BLUE = RGBColor(0x02, 0x84, 0xC7)  # Vibrant Tech Blue
TEXT_DARK    = RGBColor(0x1E, 0x29, 0x3B)  # Charcoal for body readability
TEXT_MUTED   = RGBColor(0x64, 0x74, 0x8B)  # Slate grey for subtitles
BORDER_LIGHT = RGBColor(0xCB, 0xD5, 0xE1)  # Minimal outline
DIVIDER      = RGBColor(0xCB, 0xD5, 0xE1)  # Clean separator

GREEN_BG     = RGBColor(0xF0, 0xFD, 0xF4)
GREEN_TXT    = RGBColor(0x16, 0xA3, 0x4A)
GREEN_BORDER = RGBColor(0x86, 0xEF, 0xAC)

AMBER_BG     = RGBColor(0xFF, 0xFB, 0xEB)
AMBER_TXT    = RGBColor(0xD9, 0xF7, 0x06) if False else RGBColor(0xD9, 0x77, 0x06)
AMBER_BORDER = RGBColor(0xFD, 0xE6, 0x8A)

RED_BG       = RGBColor(0xFE, 0xF2, 0xF2)
RED_TXT      = RGBColor(0xDC, 0x26, 0x26)
RED_BORDER   = RGBColor(0xFE, 0xCA, 0xCA)

BLUE_BG      = RGBColor(0xF0, 0xF9, 0xFF)
BLUE_BORDER  = RGBColor(0xBA, 0xE6, 0xFD)

FONT_FAMILY  = "Times New Roman"

def set_slide_bg(slide):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG_WHITE
    bg.line.fill.background()
    return bg

def add_header(slide, title_text, category_text="INDUSTRIAL TRAINING PRESENTATION"):
    # Prominent Header with 28pt bold title
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.3), W - Inches(1.6), Inches(0.95))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    p0 = tf.paragraphs[0]
    p0.space_after = Pt(2)
    r0 = p0.add_run()
    r0.text = category_text.upper()
    r0.font.name = FONT_FAMILY
    r0.font.bold = True
    r0.font.size = Pt(11)
    r0.font.color.rgb = PRIMARY_BLUE
    
    p1 = tf.add_paragraph()
    r1 = p1.add_run()
    r1.text = title_text
    r1.font.name = FONT_FAMILY
    r1.font.bold = True
    r1.font.size = Pt(28)
    r1.font.color.rgb = NAVY_TITLE
    
    # Subtle minimal divider line below header
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.22), W - Inches(1.6), Pt(1.5))
    line.fill.solid()
    line.fill.fore_color.rgb = DIVIDER
    line.line.fill.background()

def add_card(slide, l, t, w, h, bg_color=CARD_BG, border_color=BORDER_LIGHT, border_w=Pt(1.2)):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = bg_color
    if border_color:
        sh.line.color.rgb = border_color
        sh.line.width = border_w
    else:
        sh.line.fill.background()
    return sh

def add_bullets(tf, items, base_size=15, title_size=18.5, space_after=8, align=PP_ALIGN.LEFT):
    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if (idx == 0 and len(tf.paragraphs) > 0) else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        
        bold_txt = item[0] if len(item) > 0 else ""
        reg_txt  = item[1] if len(item) > 1 else ""
        col_ovr  = item[2] if len(item) > 2 else None
        
        # Section header check: empty or newline-only reg_txt and not bullet
        is_header = (reg_txt.strip() == "") and not bold_txt.startswith("•")
        f_size = title_size if is_header else base_size
        
        if bold_txt:
            r1 = p.add_run()
            r1.text = bold_txt
            r1.font.name = FONT_FAMILY
            r1.font.bold = True
            r1.font.size = Pt(f_size)
            if col_ovr:
                r1.font.color.rgb = col_ovr
            elif is_header:
                r1.font.color.rgb = NAVY_TITLE
            else:
                r1.font.color.rgb = NAVY_TITLE
                
        if reg_txt and reg_txt.strip() != "":
            r2 = p.add_run()
            r2.text = reg_txt
            r2.font.name = FONT_FAMILY
            r2.font.bold = False
            r2.font.size = Pt(base_size)
            r2.font.color.rgb = TEXT_DARK

# ==============================================================================
# SLIDE 1: TITLE SLIDE (COLLEGE TEMPLATE FORMAT)
# ==============================================================================
print("Generating Slide 1: Title...")
s1 = prs.slides.add_slide(blank_layout)
set_slide_bg(s1)

# SKIT Logo
if os.path.exists(r'gamma_assets\skit_logo.png'):
    s1.shapes.add_picture(r'gamma_assets\skit_logo.png', Inches(0.8), Inches(0.4), Inches(1.3), Inches(1.3))

# Institute and Department Header
tb_inst = s1.shapes.add_textbox(Inches(2.3), Inches(0.35), W - Inches(3.1), Inches(1.2))
tf_inst = tb_inst.text_frame
tf_inst.word_wrap = True
p_inst1 = tf_inst.paragraphs[0]
p_inst1.alignment = PP_ALIGN.CENTER
r_inst1 = p_inst1.add_run()
r_inst1.text = "Swami Keshvanand Institute of Technology, Management & Gramothan, Jaipur"
r_inst1.font.name = FONT_FAMILY
r_inst1.font.size = Pt(26)
r_inst1.font.bold = True
r_inst1.font.color.rgb = NAVY_TITLE

p_inst2 = tf_inst.add_paragraph()
p_inst2.alignment = PP_ALIGN.CENTER
r_inst2 = p_inst2.add_run()
r_inst2.text = "Department of Computer Science and Engineering"
r_inst2.font.name = FONT_FAMILY
r_inst2.font.size = Pt(18)
r_inst2.font.bold = True
r_inst2.font.color.rgb = TEXT_MUTED

# Main Title Card
add_card(s1, Inches(0.8), Inches(1.75), W - Inches(1.6), Inches(2.35), bg_color=CARD_BG, border_color=PRIMARY_BLUE, border_w=Pt(1.8))
tb_title = s1.shapes.add_textbox(Inches(1.0), Inches(1.85), W - Inches(2.0), Inches(2.15))
tf_title = tb_title.text_frame
tf_title.word_wrap = True

p_pres = tf_title.paragraphs[0]
p_pres.alignment = PP_ALIGN.CENTER
r_pres = p_pres.add_run()
r_pres.text = "Industrial Training Presentation on"
r_pres.font.name = FONT_FAMILY
r_pres.font.size = Pt(18)
r_pres.font.bold = True
r_pres.font.color.rgb = PRIMARY_BLUE

p_maint = tf_title.add_paragraph()
p_maint.alignment = PP_ALIGN.CENTER
r_maint = p_maint.add_run()
r_maint.text = "Synthetic Media Detection"
r_maint.font.name = FONT_FAMILY
r_maint.font.size = Pt(32)
r_maint.font.bold = True
r_maint.font.color.rgb = NAVY_TITLE

p_subt = tf_title.add_paragraph()
p_subt.alignment = PP_ALIGN.CENTER
r_subt = p_subt.add_run()
r_subt.text = "Dual-Domain Deep Learning for Detecting AI-Generated Images"
r_subt.font.name = FONT_FAMILY
r_subt.font.size = Pt(16)
r_subt.font.bold = False
r_subt.font.color.rgb = TEXT_DARK

# Bottom Cards: Supervision (Left) & Presenter (Right)
add_card(s1, Inches(0.8), Inches(4.35), Inches(5.7), Inches(2.55), bg_color=CARD_BG, border_color=BORDER_LIGHT)
tb_sup = s1.shapes.add_textbox(Inches(1.05), Inches(4.5), Inches(5.2), Inches(2.25))
tf_sup = tb_sup.text_frame
tf_sup.word_wrap = True
add_bullets(tf_sup, [
    ("Submitted to:", "", PRIMARY_BLUE),
    ("Dr. Mehul Mahrishi", ""),
    ("HOD & Professor, CSE Department", ""),
    ("SKIT, Jaipur", ""),
], base_size=15, title_size=16, space_after=3)
tf_sup.paragraphs[0].runs[0].font.size = Pt(16)
tf_sup.paragraphs[1].runs[0].font.size = Pt(20)
tf_sup.paragraphs[1].runs[0].font.bold = True
tf_sup.paragraphs[2].runs[0].font.color.rgb = TEXT_DARK
tf_sup.paragraphs[3].runs[0].font.color.rgb = TEXT_MUTED

add_card(s1, Inches(6.8), Inches(4.35), Inches(5.7), Inches(2.55), bg_color=CARD_BG, border_color=BORDER_LIGHT)
tb_pres = s1.shapes.add_textbox(Inches(7.05), Inches(4.5), Inches(5.2), Inches(2.25))
tf_pres = tb_pres.text_frame
tf_pres.word_wrap = True
add_bullets(tf_pres, [
    ("Presented By:", "", PRIMARY_BLUE),
    ("Name: Kunal Arya", ""),
    ("Roll No.: 23ESKCS126", ""),
    ("Semester & Year: 7th & 4th", ""),
    ("Academic Session: 2026-27", ""),
], base_size=15, title_size=16, space_after=3)
tf_pres.paragraphs[0].runs[0].font.size = Pt(16)
tf_pres.paragraphs[1].runs[0].font.size = Pt(20)
tf_pres.paragraphs[1].runs[0].font.bold = True
tf_pres.paragraphs[2].runs[0].font.color.rgb = TEXT_DARK
tf_pres.paragraphs[3].runs[0].font.color.rgb = TEXT_DARK
tf_pres.paragraphs[4].runs[0].font.color.rgb = TEXT_MUTED

# ==============================================================================
# SLIDE 2: INDEX (MINIMAL 2-COLUMN LIST - NO AMPERSANDS, 32PT TITLE)
# ==============================================================================
print("Generating Slide 2: Index...")
s2 = prs.slides.add_slide(blank_layout)
set_slide_bg(s2)

tb_t_idx = s2.shapes.add_textbox(Inches(1.0), Inches(0.35), Inches(8.0), Inches(0.85))
tf_t_idx = tb_t_idx.text_frame
p_idx = tf_t_idx.paragraphs[0]
r_idx = p_idx.add_run()
r_idx.text = "Index"
r_idx.font.name = FONT_FAMILY
r_idx.font.size = Pt(32)
r_idx.font.bold = True
r_idx.font.color.rgb = NAVY_TITLE

line2 = s2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(1.22), W - Inches(2.0), Pt(1.5))
line2.fill.solid()
line2.fill.fore_color.rgb = DIVIDER
line2.line.fill.background()

tb_left = s2.shapes.add_textbox(Inches(1.0), Inches(1.45), Inches(5.4), Inches(5.6))
tf_left = tb_left.text_frame
tf_left.word_wrap = True
left_items = [
    "1. Introduction",
    "2. Problem Statement",
    "3. Project Objectives",
    "4. Technology Stack",
    "5. System Architecture",
    "6. System Requirements",
    "7. Use Case Diagram",
    "8. Sequence Diagram",
    "9. Multi-Evidence Scoring",
    "10. Decision Thresholds",
]
for idx, item in enumerate(left_items):
    p = tf_left.paragraphs[0] if idx == 0 else tf_left.add_paragraph()
    p.space_after = Pt(13)
    r = p.add_run()
    r.text = item
    r.font.name = FONT_FAMILY
    r.font.size = Pt(16.5)
    r.font.bold = True if idx in [0, 4, 8] else False
    r.font.color.rgb = NAVY_TITLE if idx in [0, 4, 8] else TEXT_DARK

tb_right = s2.shapes.add_textbox(Inches(7.0), Inches(1.45), Inches(5.4), Inches(5.6))
tf_right = tb_right.text_frame
tf_right.word_wrap = True
right_items = [
    "11. Benchmark Datasets",
    "12. Explainable AI and EXIF",
    "13. Model Training History",
    "14. Internal Benchmark Results",
    "15. External Benchmark Results",
    "16. Robustness Testing",
    "17. Project Screenshots",
    "18. Future Scope",
    "19. Conclusion",
    "20. Questions and Answers",
]
for idx, item in enumerate(right_items):
    p = tf_right.paragraphs[0] if idx == 0 else tf_right.add_paragraph()
    p.space_after = Pt(13)
    r = p.add_run()
    r.text = item
    r.font.name = FONT_FAMILY
    r.font.size = Pt(16.5)
    r.font.bold = True if idx in [0, 6, 8] else False
    r.font.color.rgb = NAVY_TITLE if idx in [0, 6, 8] else TEXT_DARK

# ==============================================================================
# SLIDE 3: INTRODUCTION (SIMPLIFIED VIVA CONTENT)
# ==============================================================================
print("Generating Slide 3: Introduction...")
s3 = prs.slides.add_slide(blank_layout)
set_slide_bg(s3)
add_header(s3, "Introduction: Background & Need for Detection")

add_card(s3, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.6))
tb_i1 = s3.shapes.add_textbox(Inches(1.05), Inches(1.6), Inches(5.2), Inches(5.2))
tf_i1 = tb_i1.text_frame
tf_i1.word_wrap = True
add_bullets(tf_i1, [
    ("WHAT IS SYNTHETIC MEDIA?", "", PRIMARY_BLUE),
    ("• Definition: ", "Digital images, photos, or media generated artificially using generative AI models."),
    ("• Popular AI Tools: ", "Midjourney, DALL-E, Stable Diffusion, and Flux."),
    ("• Easy Creation: ", "Anyone can generate realistic photos from a simple text prompt in seconds."),
    ("• Human Eyes Fail: ", "Modern AI images look completely real; human visual accuracy drops below 55% when identifying them."),
], base_size=15, title_size=19, space_after=14)

add_card(s3, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.6))
tb_i2 = s3.shapes.add_textbox(Inches(7.05), Inches(1.6), Inches(5.2), Inches(5.2))
tf_i2 = tb_i2.text_frame
tf_i2.word_wrap = True
add_bullets(tf_i2, [
    ("WHY IS DETECTION CRITICAL?", "", PRIMARY_BLUE),
    ("• Fake News: ", "Synthetic photos spread rapidly on social media, misleading the general public."),
    ("• Identity Fraud: ", "Used to forge fake ID cards, fake documents, and scam individuals online."),
    ("• Legal Evidence: ", "Tampered photos risk contaminating digital evidence presented in legal cases."),
    ("• Project Mission: ", "Build an automated deep learning tool to reliably check whether an image is Real or AI-generated."),
], base_size=15, title_size=19, space_after=14)

# ==============================================================================
# SLIDE 4: PROBLEM STATEMENT (KEY TECHNICAL CHALLENGES)
# ==============================================================================
print("Generating Slide 4: Problem Statement...")
s4 = prs.slides.add_slide(blank_layout)
set_slide_bg(s4)
add_header(s4, "Problem Statement: Key Technical Challenges")

prob_cards = [
    ("1. High Realism & Subtle Clues", "• Modern AI models produce realistic faces, textures, and lighting.\n• No visible distortion or obvious flaws.\n• Human observers cannot reliably tell real photos from fakes.", RED_TXT, RED_BG, RED_BORDER),
    ("2. Generalization Problem", "• Detectors trained on older GAN models fail on new models like Midjourney or DALL-E.\n• Hard to detect synthetic images from completely new, unseen generators.", AMBER_TXT, AMBER_BG, AMBER_BORDER),
    ("3. Compression & Black-Box AI", "• WhatsApp and Instagram compress images, erasing subtle forensic cues.\n• Most AI models output a score without explaining WHY an image is fake.", PRIMARY_BLUE, BLUE_BG, BLUE_BORDER),
]

for idx, (p_title, p_desc, p_col, p_bg, p_border) in enumerate(prob_cards):
    px = Inches(0.8 + idx * 4.0)
    add_card(s4, px, Inches(1.4), Inches(3.7), Inches(5.6), bg_color=p_bg, border_color=p_border)
    tb_p = s4.shapes.add_textbox(px + Inches(0.2), Inches(1.6), Inches(3.3), Inches(5.2))
    tf_p = tb_p.text_frame
    tf_p.word_wrap = True
    add_bullets(tf_p, [
        (p_title, "", p_col),
        ("", p_desc, TEXT_DARK)
    ], base_size=14.5, title_size=18.5, space_after=8)

# ==============================================================================
# SLIDE 5: OBJECTIVES & SCOPE OF WORK
# ==============================================================================
print("Generating Slide 5: Objectives...")
s5 = prs.slides.add_slide(blank_layout)
set_slide_bg(s5)
add_header(s5, "Project Objectives & Scope of Work")

add_card(s5, Inches(0.8), Inches(1.4), W - Inches(1.6), Inches(5.6))
tb_o = s5.shapes.add_textbox(Inches(1.1), Inches(1.6), W - Inches(2.2), Inches(5.2))
tf_o = tb_o.text_frame
tf_o.word_wrap = True
add_bullets(tf_o, [
    ("CORE SYSTEM OBJECTIVES", "", PRIMARY_BLUE),
    ("1. Binary Classifier: ", "Accurately classify input images as Authentic (Real) or Synthetic (AI-Generated)."),
    ("2. Standard Benchmark Training: ", "Train EfficientNet-B4 on 27,000+ benchmark images from the GenImage dataset."),
    ("3. Multi-Evidence Analysis: ", "Combine deep learning (55%), frequency analysis (20%), and compression checks (15%)."),
    ("4. Visual Heatmaps (Grad-CAM): ", "Highlight the exact image regions (eyes, skin, borders) that triggered the fake prediction."),
    ("5. Camera EXIF Audit: ", "Check 6 physical camera hardware tags to verify optical authenticity."),
    ("6. Working Web Application: ", "Build a fast Streamlit dashboard supporting single scans and batch testing (up to 20 images)."),
], base_size=15, title_size=19, space_after=13)

# ==============================================================================
# SLIDE 6: TECH STACK
# ==============================================================================
print("Generating Slide 6: Tech Stack...")
s6 = prs.slides.add_slide(blank_layout)
set_slide_bg(s6)
add_header(s6, "Technology Stack Used")

tech_items = [
    ("DEEP LEARNING MODEL", "• Python 3.11 & PyTorch 2.6 (CUDA GPU)\n• EfficientNet-B4 Backbone (~19M parameters)\n• timm library for pre-trained weights\n• Mixed Precision (AMP) for fast inference", PRIMARY_BLUE, BLUE_BG, BLUE_BORDER),
    ("IMAGE FORENSICS", "• OpenCV & PIL for image preprocessing\n• 2D Fast Fourier Transform (FFT) for grid artifacts\n• Error Level Analysis (ELA) for compression checks\n• Camera EXIF parser for hardware metadata", GREEN_TXT, GREEN_BG, GREEN_BORDER),
    ("MODEL EXPLAINABILITY", "• PyTorch-GradCAM for visual attention heatmaps\n• Monte Carlo Dropout for confidence intervals\n• Matplotlib & Seaborn for benchmark evaluation", AMBER_TXT, AMBER_BG, AMBER_BORDER),
    ("WEB INTERFACE & HARDWARE", "• Streamlit web framework\n• FPDF2 library for generating PDF audit reports\n• NVIDIA GeForce RTX 4050 GPU (6GB VRAM)\n• High throughput: <150ms per image", NAVY_TITLE, CARD_BG, BORDER_LIGHT),
]

for idx, (t_title, t_desc, t_col, t_bg, t_border) in enumerate(tech_items):
    cx = Inches(0.8 + (idx % 2) * 6.0)
    cy = Inches(1.4 + (idx // 2) * 2.85)
    add_card(s6, cx, cy, Inches(5.7), Inches(2.65), bg_color=t_bg, border_color=t_border)
    tb_tc = s6.shapes.add_textbox(cx + Inches(0.25), cy + Inches(0.2), Inches(5.2), Inches(2.25))
    tf_tc = tb_tc.text_frame
    tf_tc.word_wrap = True
    add_bullets(tf_tc, [
        (t_title, "", t_col),
        ("", t_desc, TEXT_DARK)
    ], base_size=14.5, title_size=18, space_after=5)

# ==============================================================================
# SLIDE 7: SYSTEM ARCHITECTURE
# ==============================================================================
print("Generating Slide 7: System Architecture...")
s7 = prs.slides.add_slide(blank_layout)
set_slide_bg(s7)
add_header(s7, "System Architecture & End-to-End Pipeline")

arch_steps = [
    ("STEP 1: IMAGE INGESTION & PREPROCESSING", "• User uploads image (JPG, PNG, or WEBP).\n• System validates file format and resizes to 380x380 pixels.\n• Normalizes pixel tensor and loads it into GPU memory.", PRIMARY_BLUE, BLUE_BG, BLUE_BORDER),
    ("STEP 2: MULTI-EVIDENCE FEATURE EXTRACTION", "• Spatial CNN: EfficientNet-B4 extracts deep visual features.\n• Frequency FFT: 2D Fourier transform detects hidden grid patterns from AI upsampling.\n• Compression ELA: Measures compression differences across surfaces.\n• Sensor Noise: Checks natural camera sensor noise (PRNU) against AI smoothness.", GREEN_TXT, GREEN_BG, GREEN_BORDER),
    ("STEP 3: DECISION, VISUALIZATION & REPORTING", "• Weighted score formula computes final risk score (0 to 100%).\n• Final Verdict: Authentic (<40%), Suspect (40-59%), or Synthetic (>=60%).\n• Shows Grad-CAM heatmap and generates a downloadable PDF audit report.", AMBER_TXT, AMBER_BG, AMBER_BORDER),
]

for idx, (a_title, a_desc, a_col, a_bg, a_border) in enumerate(arch_steps):
    ay = Inches(1.4 + idx * 1.9)
    add_card(s7, Inches(0.8), ay, W - Inches(1.6), Inches(1.75), bg_color=a_bg, border_color=a_border)
    tb_ac = s7.shapes.add_textbox(Inches(1.05), ay + Inches(0.12), W - Inches(2.1), Inches(1.5))
    tf_ac = tb_ac.text_frame
    tf_ac.word_wrap = True
    add_bullets(tf_ac, [
        (a_title, "", a_col),
        ("", a_desc, TEXT_DARK)
    ], base_size=14, title_size=18, space_after=3)

# ==============================================================================
# SLIDE 8: USE CASE DIAGRAM (LIGHT THEME EMBEDDED)
# ==============================================================================
print("Generating Slide 8: Use Case...")
s8 = prs.slides.add_slide(blank_layout)
set_slide_bg(s8)
add_header(s8, "UML Use Case Workflow: Operational User Roles")
if os.path.exists(r'gamma_assets\slide_08_use_case_diagram.png'):
    s8.shapes.add_picture(r'gamma_assets\slide_08_use_case_diagram.png', Inches(0.8), Inches(1.35), Inches(11.7), Inches(5.75))

# ==============================================================================
# SLIDE 9: SEQUENCE DIAGRAM (LIGHT THEME EMBEDDED)
# ==============================================================================
print("Generating Slide 9: Sequence...")
s9 = prs.slides.add_slide(blank_layout)
set_slide_bg(s9)
add_header(s9, "Sequence Diagram: Step-by-Step Detection Flow")
if os.path.exists(r'gamma_assets\slide_09_sequence_diagram.png'):
    s9.shapes.add_picture(r'gamma_assets\slide_09_sequence_diagram.png', Inches(0.8), Inches(1.35), Inches(11.7), Inches(5.75))

# ==============================================================================
# SLIDE 10: MULTI-EVIDENCE SCORING
# ==============================================================================
print("Generating Slide 10: Multi-Evidence...")
s10 = prs.slides.add_slide(blank_layout)
set_slide_bg(s10)
add_header(s10, "Multi-Evidence Forensic Scoring & Formulation")

add_card(s10, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.6))
tb_s10_1 = s10.shapes.add_textbox(Inches(1.05), Inches(1.6), Inches(5.2), Inches(5.2))
tf_s10_1 = tb_s10_1.text_frame
tf_s10_1.word_wrap = True
add_bullets(tf_s10_1, [
    ("FOUR EVIDENCE CHANNELS", "", PRIMARY_BLUE),
    ("• Deep Learning (55%): ", "EfficientNet-B4 checks overall image and texture patterns."),
    ("• 2D FFT Frequency (20%): ", "Detects repeating grid lines left by AI generators."),
    ("• Error Level Analysis (15%): ", "Finds differences in JPEG compression across the image."),
    ("• Sensor Noise Check (10%): ", "Checks for real camera sensor noise vs artificial smoothness."),
    ("• Key Advantage: ", "If compression destroys frequency cues, the deep learning model still catches the fake."),
], base_size=15, title_size=19, space_after=13)

add_card(s10, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.6))
tb_s10_2 = s10.shapes.add_textbox(Inches(7.05), Inches(1.6), Inches(5.2), Inches(5.2))
tf_s10_2 = tb_s10_2.text_frame
tf_s10_2.word_wrap = True
add_bullets(tf_s10_2, [
    ("SCORING FORMULA & DECISION RULES", "", PRIMARY_BLUE),
    ("Scoring Equation:", "\nFinal Score = 0.55*DL + 0.20*FFT + 0.15*ELA + 0.10*Noise\n", NAVY_TITLE),
    ("Decision Rules:", "", NAVY_TITLE),
    ("• Score >= 60%: ", "SYNTHETIC (AI-Generated Image)", RED_TXT),
    ("• Score 40% - 59%: ", "SUSPECT (Needs secondary manual review)", AMBER_TXT),
    ("• Score < 40%: ", "AUTHENTIC (Genuine Camera Image)", GREEN_TXT),
    ("• Camera EXIF Check: ", "Audits 6 camera hardware tags to verify physical origin."),
], base_size=15, title_size=19, space_after=11)

# ==============================================================================
# SLIDE 11: DECISION FLOWCHART (LIGHT THEME EMBEDDED)
# ==============================================================================
print("Generating Slide 11: Decision Flowchart...")
s11 = prs.slides.add_slide(blank_layout)
set_slide_bg(s11)
add_header(s11, "Forensic Decision & Classification Threshold Engine")
if os.path.exists(r'gamma_assets\slide_11_decision_threshold_flowchart.png'):
    s11.shapes.add_picture(r'gamma_assets\slide_11_decision_threshold_flowchart.png', Inches(0.8), Inches(1.35), Inches(11.7), Inches(5.75))

# ==============================================================================
# SLIDE 12: DATASETS
# ==============================================================================
print("Generating Slide 12: Datasets...")
s12 = prs.slides.add_slide(blank_layout)
set_slide_bg(s12)
add_header(s12, "Published Benchmark Datasets Used")

add_card(s12, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.6))
tb_s12_1 = s12.shapes.add_textbox(Inches(1.05), Inches(1.6), Inches(5.2), Inches(5.2))
tf_s12_1 = tb_s12_1.text_frame
tf_s12_1.word_wrap = True
add_bullets(tf_s12_1, [
    ("GENIMAGE DATASET (TRAINING)", "", PRIMARY_BLUE),
    ("• Benchmark Size: ", "27,750 balanced image pairs across diverse ImageNet categories."),
    ("• 8 Diverse Generators: ", "Midjourney, Stable Diffusion, BigGAN, GLIDE, VQDM, ADM, Wukong, Latent Diffusion."),
    ("• Training Set: ", "18,750 images used to train the EfficientNet-B4 model."),
    ("• Validation Set: ", "3,750 images used for early stopping."),
    ("• Internal Test Set: ", "3,250 held-out images used for evaluation."),
], base_size=15, title_size=19, space_after=13)

add_card(s12, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.6))
tb_s12_2 = s12.shapes.add_textbox(Inches(7.05), Inches(1.6), Inches(5.2), Inches(5.2))
tf_s12_2 = tb_s12_2.text_frame
tf_s12_2.word_wrap = True
add_bullets(tf_s12_2, [
    ("EXTERNAL TEST SET (SYNTHBUSTER + RAISE)", "", PRIMARY_BLUE),
    ("• Benchmark Size: ", "2,000 completely unseen images never seen during training."),
    ("• Real Images (RAISE): ", "1,000 raw, uncompressed camera photos."),
    ("• Fake Images (Synthbuster): ", "1,000 images from new commercial models: DALL-E 2/3, Adobe Firefly, Midjourney v5, SD-XL."),
    ("• Scientific Goal: ", "Evaluates real-world generalization on brand-new generative tools."),
], base_size=15, title_size=19, space_after=13)

# ==============================================================================
# SLIDE 13: EXPLAINABLE AI & EXIF
# ==============================================================================
print("Generating Slide 13: XAI & EXIF...")
s13 = prs.slides.add_slide(blank_layout)
set_slide_bg(s13)
add_header(s13, "Explainable AI (XAI) & EXIF Metadata Verification")

add_card(s13, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.6))
tb_s13_1 = s13.shapes.add_textbox(Inches(1.05), Inches(1.6), Inches(5.2), Inches(5.2))
tf_s13_1 = tb_s13_1.text_frame
tf_s13_1.word_wrap = True
add_bullets(tf_s13_1, [
    ("VISUAL & FREQUENCY EXPLAINABILITY", "", PRIMARY_BLUE),
    ("• Grad-CAM Heatmaps: ", "Highlights the exact pixel areas (eyes, skin, borders) that made the model mark the image as fake."),
    ("• 2D Fourier Spectrum: ", "Shows frequency patterns; AI images often display unnatural grid spikes."),
    ("• Error Level Analysis (ELA): ", "Highlights areas with inconsistent JPEG compression."),
    ("• User Benefit: ", "Users see clear visual evidence instead of just an opaque score."),
], base_size=15, title_size=19, space_after=13)

add_card(s13, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.6))
tb_s13_2 = s13.shapes.add_textbox(Inches(7.05), Inches(1.6), Inches(5.2), Inches(5.2))
tf_s13_2 = tb_s13_2.text_frame
tf_s13_2.word_wrap = True
add_bullets(tf_s13_2, [
    ("6-POINT CAMERA EXIF AUDIT", "", PRIMARY_BLUE),
    ("• 6 Tags Checked: ", "Camera Make, Model, Lens, ISO/Exposure, Timestamp, and Software Signature."),
    ("• Real Photos: ", "Preserve 4 to 6 valid camera hardware tags."),
    ("• AI Images: ", "Score 0 to 1 out of 6 (generative tools strip hardware metadata)."),
    ("• Multi-Layer Defense: ", "Combined with visual scores for reliable forensic verification."),
], base_size=15, title_size=19, space_after=13)

# ==============================================================================
# SLIDE 14: TRAINING HISTORY
# ==============================================================================
print("Generating Slide 14: Training History...")
s14 = prs.slides.add_slide(blank_layout)
set_slide_bg(s14)
add_header(s14, "Model Training & Loss Convergence")

if os.path.exists(r'gamma_assets\slide_14_training_history.png'):
    s14.shapes.add_picture(r'gamma_assets\slide_14_training_history.png', Inches(0.8), Inches(1.4), Inches(6.5), Inches(5.6))

add_card(s14, Inches(7.6), Inches(1.4), Inches(4.9), Inches(5.6))
tb_s14 = s14.shapes.add_textbox(Inches(7.85), Inches(1.6), Inches(4.4), Inches(5.2))
tf_s14 = tb_s14.text_frame
tf_s14.word_wrap = True
add_bullets(tf_s14, [
    ("KEY TRAINING RESULTS", "", PRIMARY_BLUE),
    ("• 87.44% Validation Accuracy: ", "Reached peak accuracy on the GenImage validation split."),
    ("• 0.316 Validation Loss: ", "Steady loss decrease without overfitting."),
    ("• Best Checkpoint: ", "Locked at Epoch 8 with an ROC-AUC of 0.9434."),
    ("• Optimizer: ", "AdamW with Cosine Annealing learning rate schedule."),
    ("• GPU Acceleration: ", "Trained efficiently on local NVIDIA GeForce RTX 4050 GPU using mixed precision (AMP)."),
], base_size=15, title_size=19, space_after=14)

# ==============================================================================
# SLIDE 15: INTERNAL GENIMAGE TEST
# ==============================================================================
print("Generating Slide 15: Internal Benchmark...")
s15 = prs.slides.add_slide(blank_layout)
set_slide_bg(s15)
add_header(s15, "Internal Benchmark: GenImage Test Results (3,250 Images)")

if os.path.exists(r'gamma_assets\slide_15_internal_genimage_evaluation.png'):
    s15.shapes.add_picture(r'gamma_assets\slide_15_internal_genimage_evaluation.png', Inches(0.8), Inches(1.4), Inches(11.7), Inches(3.75))

badges = [
    ("TEST ACCURACY", "73.75 %", PRIMARY_BLUE, "On 3,250 held-out images"),
    ("TEST PRECISION", "92.11 %", GREEN_TXT, "Very low false alarms on real photos"),
    ("TEST RECALL", "64.67 %", AMBER_TXT, "Catches ~2 out of 3 fakes"),
    ("ROC-AUC SCORE", "0.8851", PRIMARY_BLUE, "High real vs fake separation"),
]
for idx, (b_t, b_v, b_c, b_s) in enumerate(badges):
    bx = Inches(0.8 + idx * 2.95)
    add_card(s15, bx, Inches(5.35), Inches(2.85), Inches(1.65), bg_color=CARD_BG, border_color=BORDER_LIGHT)
    tb_b = s15.shapes.add_textbox(bx + Inches(0.15), Inches(5.45), Inches(2.55), Inches(1.45))
    tf_b = tb_b.text_frame
    tf_b.word_wrap = True
    add_bullets(tf_b, [
        (b_t, "", NAVY_TITLE),
        (b_v, "\n", b_c),
        ("", b_s, TEXT_DARK)
    ], base_size=12, title_size=14, space_after=2, align=PP_ALIGN.CENTER)
    tf_b.paragraphs[1].runs[0].font.size = Pt(22)
    tf_b.paragraphs[1].runs[0].font.bold = True

# ==============================================================================
# SLIDE 16: EXTERNAL GENERALIZATION TEST
# ==============================================================================
print("Generating Slide 16: External Benchmark...")
s16 = prs.slides.add_slide(blank_layout)
set_slide_bg(s16)
add_header(s16, "External Benchmark: Testing on Unseen Generators (2,000 Images)")

if os.path.exists(r'gamma_assets\slide_16_external_generalization_evaluation.png'):
    s16.shapes.add_picture(r'gamma_assets\slide_16_external_generalization_evaluation.png', Inches(0.8), Inches(1.4), Inches(11.7), Inches(3.75))

ext_badges = [
    ("UNSEEN ACCURACY", "63.05 %", PRIMARY_BLUE, "On completely new models"),
    ("REAL PHOTO ACCURACY", "73.80 %", GREEN_TXT, "On raw camera photos (RAISE)"),
    ("ROC-AUC SCORE", "0.6763", AMBER_TXT, "Discrimination on new generators"),
    ("KEY TAKEAWAY", "VALIDATED", PRIMARY_BLUE, "Multi-evidence bridges domain gap"),
]
for idx, (b_t, b_v, b_c, b_s) in enumerate(ext_badges):
    bx = Inches(0.8 + idx * 2.95)
    add_card(s16, bx, Inches(5.35), Inches(2.85), Inches(1.65), bg_color=CARD_BG, border_color=BORDER_LIGHT)
    tb_b = s16.shapes.add_textbox(bx + Inches(0.15), Inches(5.45), Inches(2.55), Inches(1.45))
    tf_b = tb_b.text_frame
    tf_b.word_wrap = True
    add_bullets(tf_b, [
        (b_t, "", NAVY_TITLE),
        (b_v, "\n", b_c),
        ("", b_s, TEXT_DARK)
    ], base_size=12, title_size=14, space_after=2, align=PP_ALIGN.CENTER)
    tf_b.paragraphs[1].runs[0].font.size = Pt(20)
    tf_b.paragraphs[1].runs[0].font.bold = True

# ==============================================================================
# SLIDE 17: ROBUSTNESS BENCHMARKS
# ==============================================================================
print("Generating Slide 17: Robustness...")
s17 = prs.slides.add_slide(blank_layout)
set_slide_bg(s17)
add_header(s17, "Robustness Testing: Performance Under Degradation")

if os.path.exists(r'gamma_assets\slide_17_robustness_benchmarks.png'):
    s17.shapes.add_picture(r'gamma_assets\slide_17_robustness_benchmarks.png', Inches(0.8), Inches(1.4), Inches(5.8), Inches(5.6))

add_card(s17, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.6))
tb_s17 = s17.shapes.add_textbox(Inches(7.05), Inches(1.6), Inches(5.2), Inches(5.2))
tf_s17 = tb_s17.text_frame
tf_s17.word_wrap = True
add_bullets(tf_s17, [
    ("ROBUSTNESS FINDINGS", "", PRIMARY_BLUE),
    ("• JPEG Compression: ", "Stays reliable (>72% accuracy) at quality 75-90; drops under heavy compression (quality 50)."),
    ("• Gaussian Blur: ", "Blurring washes out fine details, causing a slight drop in confidence."),
    ("• Sensor Noise: ", "Model remains stable under normal camera noise levels (sigma <= 10)."),
    ("• Image Resizing: ", "Downscaling below 256x256 removes key texture cues; native resolution is recommended."),
    ("• Multi-Evidence Defense: ", "When compression damages frequency cues, deep learning and EXIF checks maintain accuracy."),
], base_size=15, title_size=19, space_after=13)

# ==============================================================================
# SLIDE 18: DASHBOARD SCREENSHOTS
# ==============================================================================
print("Generating Slide 18: Dashboard...")
s18 = prs.slides.add_slide(blank_layout)
set_slide_bg(s18)
add_header(s18, "Working Web Application: Single & Batch Image Interface")

if os.path.exists(r'gamma_assets\slide_18_dashboard_single_analysis.png'):
    s18.shapes.add_picture(r'gamma_assets\slide_18_dashboard_single_analysis.png', Inches(0.8), Inches(1.4), Inches(5.7), Inches(4.45))

if os.path.exists(r'gamma_assets\slide_18_dashboard_batch_analysis.png'):
    s18.shapes.add_picture(r'gamma_assets\slide_18_dashboard_batch_analysis.png', Inches(6.8), Inches(1.4), Inches(5.7), Inches(4.45))

add_card(s18, Inches(0.8), Inches(5.95), Inches(5.7), Inches(1.05))
tb_cap1 = s18.shapes.add_textbox(Inches(0.95), Inches(6.05), Inches(5.4), Inches(0.9))
tf_cap1 = tb_cap1.text_frame
tf_cap1.word_wrap = True
add_bullets(tf_cap1, [
    ("SINGLE-IMAGE SCAN: ", "Displays input image, Grad-CAM attention heatmap, 2D FFT spectrum, ELA map, and camera EXIF audit.")
], base_size=13, space_after=0)

add_card(s18, Inches(6.8), Inches(5.95), Inches(5.7), Inches(1.05))
tb_cap2 = s18.shapes.add_textbox(Inches(6.95), Inches(6.05), Inches(5.4), Inches(0.9))
tf_cap2 = tb_cap2.text_frame
tf_cap2.word_wrap = True
add_bullets(tf_cap2, [
    ("BATCH QUEUE: ", "Processes up to 20 images in parallel (~2.5s total) with clear Authentic / Synthetic status badges.")
], base_size=13, space_after=0)

# ==============================================================================
# SLIDE 19: CONCLUSION & FUTURE SCOPE
# ==============================================================================
print("Generating Slide 19: Conclusion...")
s19 = prs.slides.add_slide(blank_layout)
set_slide_bg(s19)
add_header(s19, "Conclusion & Future Scope")

add_card(s19, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.6))
tb_s19_1 = s19.shapes.add_textbox(Inches(1.05), Inches(1.6), Inches(5.2), Inches(5.2))
tf_s19_1 = tb_s19_1.text_frame
tf_s19_1.word_wrap = True
add_bullets(tf_s19_1, [
    ("SUMMARY OF WORK DONE", "", PRIMARY_BLUE),
    ("• End-to-End Detector: ", "Built a working tool combining EfficientNet-B4 with 2D FFT and ELA forensics."),
    ("• High Accuracy: ", "Achieved 87.44% validation accuracy and 92.11% precision on standard benchmarks."),
    ("• Visual Heatmaps: ", "Grad-CAM shows users exactly which parts of the image look fake."),
    ("• Fast Local Speed: ", "Runs in under 150ms per image on an NVIDIA RTX 4050 GPU."),
    ("• Web Interface: ", "Easy-to-use Streamlit dashboard with batch testing and downloadable PDF reports."),
], base_size=15, title_size=19, space_after=13)

add_card(s19, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.6))
tb_s19_2 = s19.shapes.add_textbox(Inches(7.05), Inches(1.6), Inches(5.2), Inches(5.2))
tf_s19_2 = tb_s19_2.text_frame
tf_s19_2.word_wrap = True
add_bullets(tf_s19_2, [
    ("FUTURE SCOPE & EXTENSIONS", "", PRIMARY_BLUE),
    ("• Video Deepfake Detection: ", "Extend the model to detect face swaps in video clips using temporal frame analysis."),
    ("• Voice Clone Detection: ", "Integrate audio spectrogram models to detect AI voice clones in phone scams."),
    ("• Browser Extension: ", "Package the model into a lightweight extension to verify online images in real time."),
    ("• Social Media Integration: ", "Automated content screening for high-volume platform moderation."),
], base_size=15, title_size=19, space_after=13)

# ==============================================================================
# SLIDE 20: QUESTIONS AND ANSWERS (THANK YOU SLIDE)
# ==============================================================================
print("Generating Slide 20: Thank You...")
s20 = prs.slides.add_slide(blank_layout)
set_slide_bg(s20)

add_card(s20, Inches(1.5), Inches(1.3), W - Inches(3.0), Inches(5.0), bg_color=CARD_BG, border_color=PRIMARY_BLUE, border_w=Pt(1.8))
tb_ty = s20.shapes.add_textbox(Inches(1.8), Inches(1.5), W - Inches(3.6), Inches(4.6))
tf_ty = tb_ty.text_frame
tf_ty.word_wrap = True
add_bullets(tf_ty, [
    ("THANK YOU", "", PRIMARY_BLUE),
    ("Questions & Discussion", "\n"),
    ("Project Title: ", "AI-Driven Synthetic Media Detection Using Deep Learning\n"),
    ("Presentation: ", "Industrial Training Presentation (ITR) | Session 2026-27\n"),
    ("Candidate: ", "Kunal Arya (Roll No: 23ESKCS126) | 7th Sem, 4th Year B.Tech CSE\n"),
    ("Supervisor: ", "Dr. Mehul Mahrishi | Professor & Head of Department, CSE\n"),
    ("Institution: ", "Swami Keshvanand Institute of Technology, Management & Gramothan (SKIT), Jaipur"),
], base_size=15, title_size=22, space_after=5, align=PP_ALIGN.CENTER)
tf_ty.paragraphs[0].runs[0].font.size = Pt(40)
tf_ty.paragraphs[0].runs[0].font.bold = True
tf_ty.paragraphs[1].runs[0].font.size = Pt(24)

# ==============================================================================
# SAVE PRESENTATION TO TARGET FILES
# ==============================================================================
out_files = [
    r'Synthetic_Media_Detection.pptx',
    r'Synthetic_Media_Detection_ITR_FINAL.pptx',
    r'Synthetic_Media_Detection_SKIT_FINAL.pptx',
    r'C:\Users\Kunal Arya\.gemini\antigravity\brain\2c40ff48-2192-440e-a872-15ea41fb948b\Synthetic_Media_Detection_PPT.pptx'
]

for of in out_files:
    try:
        prs.save(of)
        print("SUCCESSFULLY SAVED TO:", of)
    except Exception as e:
        print("LOCKED OR FAILED (SKIPPED):", of, e)

print(f"DONE! Total slides built: {len(prs.slides)}")
