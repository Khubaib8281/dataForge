from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  
from datetime import datetime

def add_title_page(story, uploaded_file=None, project_name=None):
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(   
        "TitleStyle",
        parent=styles["Title"],
        fontSize=28,
        alignment=1,  # center
        spaceAfter=30
    )
    subtitle_style = ParagraphStyle(
        "SubTitleStyle",
        parent=styles["Normal"],
        fontSize=14,
        alignment=1,
        spaceAfter=20
    )
    section_style = ParagraphStyle(
        "SectionStyle",
        parent=styles["Normal"],
        fontSize=12,
        leading=16,
        alignment=1
    )


    # Decide dataset/project name
    dataset_name = project_name if project_name else (uploaded_file.name if uploaded_file else "Dataset")

    # Report date
    report_date = datetime.now().strftime("%B %d, %Y")

    # Build title page
    story.append(Paragraph("Exploratory Data Analysis Report", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"<b>Dataset:</b> {dataset_name}", subtitle_style))
    story.append(Paragraph(f"<b>Generated on:</b> {report_date}", subtitle_style))
    story.append(Paragraph(f"<b>Prepared by:</b> DataForge", subtitle_style))
    story.append(Spacer(1, 40))
    
    story.append(Paragraph("Auto-generated EDA report with plot-wise explanations", styles["Heading2"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "This report provides an automated exploratory data analysis "
        "including data quality checks, structured insights, and "
        "visualizations with professional interpretations.",
        section_style
    ))

    story.append(PageBreak())
