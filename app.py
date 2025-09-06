import streamlit as st
from io import BytesIO
import plotly.io as pio
import streamlit.components.v1 as components
from streamlit_agraph import agraph, Node, Edge, Config
import pandas as pd
import os, sys
import io
import contextlib
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
import plotly.express as px
from fpdf import FPDF
import statsmodels.api as sm
import tempfile
import sqlite3  
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  
from reportlab.pdfgen import canvas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eda_utils import clean_data, data_info,generate_basic_visuals, generate_visuals_from_suggestions, _df_signature, _enforce_readable_layout,_generate_single_figure,_strongen_bar_family_colors,_vis_key, generate_fig_cached, generate_static_matplotlib
from LLM import llm_response, get_viz_suggestions, nl_to_sql, check_data_quality, explain_plot
from sql_conn import init_db
from styling import add_footer, apply_dataportal_css, apply_custom_theme, inject_footer
from pdf_func import add_title_page  


st.set_page_config(page_title="DataForge", layout = 'wide')
# st.title("DataForge: AI-Powered EDA & Reporting Tool 🛠️")
apply_dataportal_css()   
# App Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>DataForge</h1>", unsafe_allow_html=True)

# Subtitle / Description
st.markdown(
    "<h4 style='text-align: center; color: #ffffff;'>AI-Powered Data Cleaning, Visualization & Reporting</h4>",   
    unsafe_allow_html=True
)

st.markdown("---")   

with st.expander("🔎 How DataForge Works?"):        
    with st.expander("Step 1: Data Quality Checker"):
        st.info("Check if the dataset has missing values, inconsistent categories, or anomalies.")

    with st.expander("Step 2: Data Cleaning Agent"):
        st.info("Automatically clean the data using LLM-generated instructions.")

    with st.expander("Step 3: Visualization Agent"):
        st.info("Generate interactive charts and summaries for quick insights.")

    with st.expander("Step 4: Analyst Agent"):
        st.info("Run automated SQL queries or ask manual SQL questions.")

    with st.expander("Step 5: Reporting Agent"):
        st.info("Create explainable AI-powered reports from cleaned data.")


uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])


for key in ["show_raw_data", "show_data_info", "clean_data", "show_clean_data", "generate_basic_visuals", "visual_suggestions", "show_clean_data_info", 'cleaned_info']:       
    if key not in st.session_state:
        st.session_state[key] = False   

def apply_llm_cleaning(df, code):   
    local_vars = {"df": df.copy(), "pd": pd, "np": np}
    try:
        exec(code, local_vars, local_vars)  # same dict for globals and locals
        if "df_cleaned" in local_vars:
            return local_vars["df_cleaned"]
        elif "df" in local_vars:  # fallback if LLM edited df in place
            return local_vars["df"]
        return df
    except Exception as e:
        st.error(f"Error running LLM cleaning code: {e}")  
        return df
  

def display_visual_report(figures, cols_per_row=2, prefix="report"):
    """
    Displays interactive Plotly figures in a grid layout using Streamlit.
    Guarantees unique keys by combining prefix + index + UUID.
    """
    for idx, fig in enumerate(figures):
        # make a stable but globally unique key
        unique_key = f"{prefix}_plotly_fig_{idx}_{uuid.uuid4().hex}"
        
        if idx % cols_per_row == 0:
            cols = st.columns(cols_per_row)

        cols[idx % cols_per_row].plotly_chart(
            fig,
            use_container_width=True,
            key=unique_key
        ) 


def display_data_info(info):
    if "error" in info:
        st.error(info["error"])
        return

    # st.subheader("Data Shape")
    # st.write(f"Rows: {info['shape'][0]}, Columns: {info['shape'][1]}")

    # st.subheader("Columns")
    # st.write(", ".join(info['columns']))

    # st.subheader("Data Types")
    # for col, dtype in info['dtypes'].items():
    #     st.write(f"**{col}**: {dtype}")

    # st.subheader("Missing Values")
    # for col, missing in info['missing_values'].items():
    #     st.write(f"**{col}**: {missing}")

    st.subheader("Summary Statistics")
    st.dataframe(info["summary_stats"])  # <- nice interactive table instead of text

    # st.subheader("Dataset Overview")
    # st.text(info["summary_text"])        


def export_html_report(figures, filename="visual_report.html"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
        with open(tmpfile.name, "w", encoding="utf-8") as f:
            for fig in figures:
                f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        return tmpfile.name
    

def export_pdf_report(figures, filename="visual_report.pdf"):
    """
    Export a list of plotly figures to a PDF report.
    Each figure is converted to a temporary PNG and inserted into the PDF,
    with its title printed above the plot.
    """

    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter    

    for idx, fig in enumerate(figures, 1):
        # Get the title from Plotly figure
        title = fig.layout.title.text if fig.layout.title else f"Figure {idx}"

        # Create a temp file safely for Windows
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            tmp_name = tmpfile.name

        # Save figure to temp file
        fig.write_image(tmp_name)  # Plotly figure

        # Draw title above the plot
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(width / 2, height - 40, title)

        # Draw image below the title
        img_width = width - 100
        img_height = height - 150  
        c.drawImage(tmp_name, 50, 80, width=img_width, height=img_height, preserveAspectRatio=True)

        c.showPage()

        # Clean up temp file
        os.remove(tmp_name)    

    c.save()
    return filename



if uploaded_file:   
    df = pd.read_csv(uploaded_file)
    project_name = st.text_input("Enter Project Name (optional)")  
    d_info = data_info(df)
    # if "llm_res" not in st.session_state:
    #     # Show spinner while LLM generates response
    #     with st.spinner("Generating data cleaning instructions..."):  
    #         st.session_state.llm_res = llm_response(df)  

    # # Show stored result (fast)   
    # st.write("### LLM Response for Data Cleaning Instructions")
    # st.code(st.session_state.llm_res, language='python')

    

# Create tabs      
data_quality_checker ,tab_cleaning, tab_visualization, sql_playground, reporting_agent = st.tabs(["Data Quality Checker","Data Cleaning Agent", "Data Visualization Agent", "Analyst Agent", "Reporting Agent"])  

# ----------------------- Data Quality checker ------------------------- 

with data_quality_checker:
    st.subheader("Data Quality Checker")   

    if not uploaded_file:   
        st.warning("Upload your dataset")
    else:
        if st.button("Check Quality"):
            with st.spinner("Checking data quality and identifying problems.Please wait..."):
                raw_report = check_data_quality(df)

                # Always ensure it's a dict
                if isinstance(raw_report, str):
                    try:
                        quality_report = json.loads(raw_report)
                    except json.JSONDecodeError:
                        st.error("LLM returned invalid JSON.")
                        st.stop()
                elif isinstance(raw_report, dict):
                    quality_report = raw_report
                else:
                    st.error("Unexpected format from Data Quality Checker")
                    st.stop()

                st.session_state.quality_report = quality_report   # <-- changed
    
                if quality_report.get("needs_cleaning"):
                    st.session_state.quality_issues = quality_report.get("issues", [])  # <-- changed
                    st.session_state.cleaned = False   # not yet cleaned  # <-- changed
                else:
                    st.session_state.df_cleaned = df
                    st.session_state.quality_issues = []   # <-- changed
                    st.session_state.cleaned = True    # mark as cleaned  # <-- changed

        # Always show status if it exists
        if st.session_state.get("cleaned") is True:   # <-- changed
            st.success("Dataset looks clean. No need to visit 'Data Cleaning Agent'")  # <-- moved outside
        elif st.session_state.get("cleaned") is False:   # <-- changed
            st.warning("Data quality issues detected: Visit 'Data Cleaning Agent'")  # <-- moved outside

        # Always persist problems until next Check Quality
        if st.session_state.get("quality_issues"):   # <-- changed
            st.subheader("Detected Issues")   # <-- moved outside button
            for i, issue in enumerate(st.session_state.quality_issues, 1):   # <-- changed
                st.write(f"**{i}.** {issue}")   # <-- changed


## --------------------------- data cleaning agent --------------------------------
with tab_cleaning:
    st.header("Data Cleaning Agent")

    if not uploaded_file:
        st.warning("Upload your dataset")
    else:

        label = "Hide Raw Data Preview" if st.session_state.show_raw_data else "Show Raw Data Preview"   
        if st.button(label):
            st.session_state.show_raw_data = not st.session_state.show_raw_data
        if st.session_state.show_raw_data:
            st.write("### Raw Data Preview")
            st.dataframe(df.head())   


        label = "Hide Raw Data Info" if st.session_state.show_data_info else "Show Raw Data Info"
        if st.button(label):
            st.session_state.show_data_info = not st.session_state.show_data_info
        if st.session_state.show_data_info:
            st.write("### Data Info")
            # st.json(d_info)
            display_data_info(d_info)

    # Button: triggers cleaning only once
        if st.button("Clean Data"):
            # if not uploaded_file:
            #     st.warning("Upload your data First")
            if "quality_report" not in st.session_state:
                st.warning("Go to quality checker")  
            else:
                # Only generate if not already generated   
                if "llm_res" not in st.session_state:   
                    with st.spinner("Cleaning Data.Please wait..."):
                        st.session_state.llm_res = llm_response(df, st.session_state.quality_report)

                # Show stored result
                # st.write("### LLM Response for Data Cleaning Instructions")
                # st.code(st.session_state.llm_res, language='python')

                # Apply cleaning
                code = st.session_state.llm_res
                df_cleaned = apply_llm_cleaning(df, code)
                st.session_state.df_cleaned = df_cleaned        

                #  set flag
                st.session_state.cleaned = True   # <-- moved outside

        # persistent message (runs on every rerun)
        if st.session_state.get("cleaned", False):   # <-- changed
            st.success("Data cleaned successfully!")   # <-- changed


        if "df_cleaned" in st.session_state:  
            label = "Hide Cleaned Data Preview" if st.session_state.show_clean_data else "Show Cleaned Data Preview"
            if st.button(label):
                st.session_state.show_clean_data = not st.session_state.show_clean_data
            if st.session_state.show_clean_data:
                st.write("### Cleaned Data Preview")
                st.dataframe(st.session_state.df_cleaned.head())

            
            label = "Hide Cleaned Data Info" if st.session_state.show_clean_data_info else "Show Cleaned Data Info"   
            if st.button(label):
                st.session_state.show_clean_data_info = not st.session_state.show_clean_data_info
                if st.session_state.show_clean_data_info:
                    st.session_state.cleaned_info = data_info(st.session_state.df_cleaned)
                    st.write("### Cleaned Data Info")
                    # st.json(cleaned_info)     
                    display_data_info(st.session_state.cleaned_info)   

            csv_buffer = io.StringIO()
            st.session_state.df_cleaned.to_csv(csv_buffer, index = False)
            csv_buffer.seek(0)

            st.download_button(
                label = "Download Cleaned Data as .csv",
                file_name="cleaned_data.csv", 
                mime="text/csv",
                key = "download_csv",  
                data = csv_buffer.getvalue()
            )

# ------------------------ visualization section -------------------------

with tab_visualization:
    st.header("Visualization Agent")

    if not uploaded_file:
        st.warning("Upload your dataset")  
    elif "df_cleaned" not in st.session_state:   
        st.warning("Please clean your data first.")  
    else:
        df_to_use = st.session_state.df_cleaned

        # --- Mode selection ---
        mode = st.radio("Select visualization mode", ["Static Plots", "Interactive Plots"])    

        # --- Step 1: Generate suggestions ---
        if st.button("Generate Visualization Suggestions"):
            with st.spinner("Analyzing dataset and generating suggestions..."):
                suggestions = get_viz_suggestions(df_to_use)
                st.session_state.visual_suggestions = suggestions

        # --- Step 2: Pick ONE suggestion ---
        if "visual_suggestions" in st.session_state and st.session_state.visual_suggestions:
            st.subheader("Visualization Suggestions")

            # Build radio options
            options = []
            for i, v in enumerate(st.session_state.visual_suggestions):
                desc = f"📊 {v['type'].capitalize()} on `{v.get('x','')}`"
                if v.get('y'):
                    desc += f" vs `{v['y']}`"
                options.append((i, v, desc))

            choice = st.radio(
                "Select a visualization:",
                options,
                format_func=lambda x: x[2],
                key="plot_radio"
            )

            idx, v, _ = choice

            # --- Step 3: Generate and display plot ---
            fig = None
            if st.button("Generate Visualization"):
                with st.spinner("Generating visualization..."):

                    if mode == "Interactive Plots":
                        fig = _generate_single_figure(df_to_use, v)

                        # Handle xticks for readability
                        x_col = v.get("x")
                        if isinstance(x_col, str) and x_col in df_to_use.columns:
                            if df_to_use[x_col].nunique() > 30:   
                                fig.update_xaxes(showticklabels=True, nticks=30)

                        # Layout styling
                        if v["type"].lower() == "bar":
                            fig.update_layout(template="plotly", barmode="group")
                        else:
                            fig.update_layout(template="plotly_white")

                        # storing session_state for displaying figures consistently 
                        plot_key = f"plot_{idx}"

                        # --- Store figure once ---
                        if plot_key not in st.session_state:
                            st.session_state[plot_key] = fig   # only the first time

                        # --- Always use session_state for display ---
                        st.subheader("Interactive Visualization")
                        st.plotly_chart(st.session_state[plot_key], use_container_width=True, key=f"plotly_single_{idx}")

                        # --- Download button ---
                        html = pio.to_html(st.session_state[plot_key], include_plotlyjs="cdn", full_html=True)
                        st.download_button(
                            "Download HTML",
                            data=html,   
                            file_name=f"plot_{idx}.html",
                            mime="text/html",
                            key=f"download_{idx}"  # important: unique key
                        )  
    

                    elif mode == "Static Plots":
                        fig = generate_static_matplotlib(df_to_use, v)

                        st.subheader("Static Visualization")
                        st.pyplot(fig)    

                        # --- Export single static fig as PDF ---
                        from reportlab.platypus import SimpleDocTemplate, Image, Spacer
                        from reportlab.lib.pagesizes import A4
                        from reportlab.lib.units import inch
                        import tempfile, os

                        buffer = BytesIO()
                        doc = SimpleDocTemplate(buffer, pagesize=A4)
                        elements = []

                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                            tmp_path = tmp.name
                            fig.savefig(tmp_path, dpi=200, bbox_inches="tight")
                            elements.append(Image(tmp_path, width=6*inch, height=4*inch))
                            elements.append(Spacer(1, 0.25*inch))

                        doc.build(elements)
                        buffer.seek(0)

                        st.download_button(
                            "Download PDF",
                            data=buffer,
                            file_name="visualization.pdf",   
                            mime="application/pdf"
                        )
    


# ------------------------ sql playground section -------------------------

with sql_playground:
    st.subheader("🛠 Analyst Agent")  

    if not uploaded_file:
        st.warning("Upload your dataset")
    else:

        if "df_cleaned" not in st.session_state:
            st.warning("Please visit cleaning/data quality checker agent")   
        else:
            conn = init_db(st.session_state.df_cleaned)        
            # Mode Selection
            mode = st.radio("Choose Query Mode:", ["Manual SQL", "AI-powered SQL"])
    
            # manual sql mode
            if mode == "Manual SQL":
                st.subheader("✍️ Write SQL Query")
                query = st.text_area("SQL:", "SELECT * FROM df_cleaned LIMIT 5")

                if st.button("Run Query"):
                    try:
                        result = pd.read_sql_query(query, conn)
                        st.dataframe(result)

                        # Only try to plot if we have at least 2 columns
                        if not result.empty and result.shape[1] >= 2:
                            try:
                                # Use the first column as index, plot the rest
                                st.bar_chart(result.set_index(result.columns[0]))
                            except Exception:
                                st.warning("Query result not suitable for bar chart.")
                    except Exception as e:
                        st.error(f"SQL error: {e}\nQuery was: {query}")


            # ai-powered sql mode    
            else:
                st.subheader("🤖 Ask in natural language")
                user_prompt = st.text_input("Your question:", "Show average age by gender")

                if st.button("Generate Query"):
                    schema = str(st.session_state.df_cleaned.dtypes.to_dict())
                    sql_query = nl_to_sql(user_prompt, schema)
                    # storing in session_state
                    st.session_state.sql_query = sql_query

                if "sql_query" in st.session_state:    
                    st.code(st.session_state.sql_query, language='sql')
                    sql_query = st.text_area("Edit SQL query:", st.session_state.sql_query)
        
                    if st.button("Run Query"):
                        try:
                            result = pd.read_sql_query(sql_query, conn)
                            st.dataframe(result)

                            # Only plot if result has at least 2 columns
                            if not result.empty and result.shape[1] >= 2:
                                try:
                                    # Try plotting with first column as index
                                    st.bar_chart(result.set_index(result.columns[0]))
                                except Exception:
                                    st.warning("Query returned non-numeric values, skipping chart.")
                        except Exception as e:
                            st.error(f"AI-generated query failed: {e}\nSQL was: {sql_query}")
            

# ------------------------- Reporting agent ----------------------------
   

with reporting_agent:
    st.subheader("Reporting Agent")


    if not uploaded_file:   
        st.warning("Upload your dataset")   
    else:    
        all_reports = []
        if not st.session_state.visual_suggestions:
            st.warning("Go to Visualization Agent to generate visual suggestions first")
        else:
            visual_suggestions = st.session_state.get("visual_suggestions", [])

            if isinstance(visual_suggestions, bool):
                visual_suggestions = []    
    
            # st.write("visual_suggestions:", visual_suggestions)
            # st.write("all_reports before PDF:", all_reports)
            if not st.session_state.cleaned_info:
                st.session_state.cleaned_info = data_info(st.session_state.df_cleaned)      

            # st.write("Dataset info:", st.session_state.cleaned_info)       
    
    # --------------------- reporting agent's report -------------------------



            def create_pdf_report(all_reports, output_file="EDA_Report.pdf"):
                doc = SimpleDocTemplate(output_file)
                story = []
                styles = getSampleStyleSheet()

                add_title_page(story, uploaded_file, project_name)

                # Data Quality Issues (before cleaning)
                if "quality_report" in st.session_state and st.session_state.quality_report:
                    qr = st.session_state.quality_report
                    story.append(Paragraph("Data Quality Issues (Before Cleaning)", styles['Heading1']))

                    if "issues" in qr and qr["issues"]:
                        story.append(Paragraph("Issues Found:", styles['Heading2']))
                        for issue in qr["issues"]:
                            story.append(Paragraph(f"• {issue}", styles['Normal']))
                        story.append(Spacer(1, 12))

                    if "recommendations" in qr and qr["recommendations"]:
                        story.append(Paragraph("Recommendations:", styles['Heading2']))
                        for rec in qr["recommendations"]:
                            story.append(Paragraph(f"• {rec}", styles['Normal']))
                        story.append(Spacer(1, 12))

                    story.append(PageBreak())


                # Cleaned Dataset Info
                if "cleaned_info" in st.session_state and st.session_state.cleaned_info:
                    cleaned_info = st.session_state.cleaned_info

                    story.append(Paragraph("Cleaned Dataset Info", styles['Heading1']))
                    story.append(Paragraph(f"Rows: {cleaned_info['shape'][0]}, Columns: {cleaned_info['shape'][1]}", styles['Normal']))
                    story.append(Spacer(1, 12))

                    story.append(Paragraph("Columns & Data Types", styles['Heading2']))
                    for col, dtype in cleaned_info['dtypes'].items():
                        story.append(Paragraph(f"• {col}: {dtype}", styles['Normal']))



                    story.append(Spacer(1, 12))
                    story.append(Paragraph("Missing Values (After Cleaning)", styles['Heading2']))
                    for col, missing in cleaned_info['missing_values'].items():
                        story.append(Paragraph(f"• {col}: {missing}", styles['Normal']))
                    story.append(PageBreak())

                # Add each plot + explanation (structured format)
                for report in all_reports:
                    explanation = explain_plot(report["metadata"], st.session_state.cleaned_info)

                    # st.write("Explanation for", report["plot_id"], ":", explanation)
    
                    # Plot title
                    story.append(Paragraph(f"Plot {report['plot_id']+1}", styles['Heading2']))
                    story.append(Image(report["image_path"], width=400, height=300))
                    story.append(Spacer(1, 12))

                    if isinstance(explanation, dict):
                        if "what_it_shows" in explanation:
                            story.append(Paragraph("1. What this plot shows:", styles['Heading3']))
                            for point in explanation["what_it_shows"]:
                                story.append(Paragraph(f"• {point}", styles['Normal']))
                            story.append(Spacer(1, 6))

                        if "key_insights" in explanation:
                            story.append(Paragraph("2. Key Insights:", styles['Heading3']))
                            for point in explanation["key_insights"]:
                                story.append(Paragraph(f"• {point}", styles['Normal']))
                            story.append(Spacer(1, 6))

                        if "how_to_use" in explanation:
                            story.append(Paragraph("3. How to use this:", styles['Heading3']))
                            for point in explanation["how_to_use"]:
                                story.append(Paragraph(f"• {point}", styles['Normal']))
                            story.append(Spacer(1, 6))
                    else:
                        story.append(Paragraph(explanation, styles['Normal']))

                    story.append(PageBreak())
        
                doc.build(story)
                return output_file
        


            # === Inside Streamlit app ===
            all_reports = []
            for idx, vis in enumerate(visual_suggestions):
                # Generate static figure
                fig = generate_static_matplotlib(st.session_state.df_cleaned, vis)   

                # Save plot to temp image
                img_path = f"temp_plot_{idx}.png"
                fig.savefig(img_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                # Store metadata for LLM
                report_item = {
                    "plot_id": idx,
                    "metadata": vis,
                    "image_path": img_path
                }
                all_reports.append(report_item)

            # Generate + Download button
            if st.button("Generate PDF Report"):
                with st.spinner("Generating PDF report..."):    
                    pdf_path = create_pdf_report(all_reports)
                with open(pdf_path, "rb") as f:
                    st.download_button(    
                        "Download EDA Report",
                        data=f,
                        file_name="EDA_Report.pdf",
                        mime="application/pdf"
                    )

# apply_custom_theme()             
inject_footer()    
  