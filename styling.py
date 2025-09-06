import streamlit as st
from datetime import datetime

def add_footer():
    st.markdown(
        """
        <hr style="margin-top:2rem; margin-bottom:1rem; border: 1px solid #ddd;">
        <div style="text-align: center; color: gray; font-size: 14px;">
            <p><b>DataForge</b> — Automated Exploratory Data Analysis</p>
            <p>Developed by <b>Khubaib Ahmad</b> | AI/ML Engineer & Data Scientist</p>
        </div>
        """,
        unsafe_allow_html=True
    )        

def inject_footer():   
    st.markdown(   
        """
        <style>
        /* Footer container */

            .stDownloadButton>button {
            background-color: #dc3545; /* red background */
            color: #ffffff;            /* white text */
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            border: none;
            cursor: pointer;
            transition: background-color 0.2s ease;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3); /* subtle text shadow */
        }

        .stDownloadButton>button:hover {
            background-color: #c82333; /* slightly darker red on hover */
        }    

        
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: rgba(33, 37, 41, 0.95); /* Dark semi-transparent */
            color: #f1f1f1;
            font-family: 'Roboto', sans-serif;
            font-size: 0.9em;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.7em 2em;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.3);
            z-index: 9999;
        }

        /* Footer text sections */
        .footer-left, .footer-right {
            display: flex;
            flex-direction: column;   
        }

        .footer-left span, .footer-right span {
            margin: 0;
            line-height: 1.2;
        }

        /* Footer links hover */
        .footer a {
            color: #4da6ff;
            text-decoration: none;
            transition: color 0.3s ease;
        }
        .footer a:hover {
            color: #1a8cff;
            text-decoration: underline;
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
            .footer {
                flex-direction: column;
                text-align: center;
                padding: 1em;
            }
            .footer-left, .footer-right {
                margin: 0.3em 0;
            }
        }
        </style>

        <div class="footer">
            <div class="footer-left">
                <span><b>DataForge</b> &copy; 2025 | AI-Powered EDA Tool</span>
                <span>Developed by <a href="https://www.linkedin.com/in/muhammad-khubaib-ahmad-" target="_blank">Muhammad Khubaib Ahmad</a></span>
            </div>
            <div class="footer-right">
                <span>Version 1.0.0</span> 
                <span>All rights reserved</span>    
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# <div class="footer-left">
#                 <span><b>DataForge</b> &copy; 2025 | AI-Powered EDA Tool</span>
#                 <span>Developed by <a href="https://www.linkedin.com/in/muhammad-khubaib-ahmad-" target="_blank">Khubaib Ahmad</a></span>
#             </div>


import streamlit as st

def apply_dataportal_css():
    """
    Apply professional custom CSS to Streamlit app while respecting default theme colors
    and ensuring proper contrast in both light and dark mode.
    """
    st.markdown(
        """
        <style>
        /* --- Import font --- */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
        }

        /* --- General text color & background --- */
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }

        /* --- Titles / Headings --- */
        .stMarkdown h1 {
            color: var(--primary-color);
            font-size: 2.8em;
            font-weight: 700;
            margin-bottom: 0.3em;
        }
        .stMarkdown h2 {
            color: var(--primary-color);
            font-size: 2em;
            font-weight: 600;
            margin-top: 1em;
        }
        .stMarkdown h3 {
            color: var(--text-color);
            font-size: 1.6em;
            font-weight: 500;
        }

        /* Expander container */
        div[role="button"].stExpander {
            background-color: #ffffff; /* white background */
            border: 1px solid #d1d1d1; /* subtle border */
            border-radius: 8px;
            padding: 0.5rem 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            transition: background-color 0.2s ease, box-shadow 0.2s ease;
            color: #111111; /* text color */
        }

        /* Expander header hover effect */
        div[role="button"].stExpander:hover {
            background-color: #f0f0f0;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            cursor: pointer;
        }

        /* Expander content */
        div.stExpanderContent {
            background-color: #fafafa; /* slightly lighter background */
            border-top: 1px solid #d1d1d1;
            padding: 1rem;
            color: #111111; /* text inside expander */
        }


        /* --- Sidebar --- */
        .css-1d391kg {
            background-color: var(--sidebar-background-color);
            padding: 1rem;
            border-radius: 8px;
        }

        /* --- Buttons --- */
        div.stButton > button {
            background-color: #000000; /* black background */
            color: #ffffff; /* white text */
            border-radius: 8px;
            padding: 0.5em 1em;
            font-weight: 600;
            border: none;
            transition: background-color 0.2s ease;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* subtle shadow */
        }

        div.stButton > button:hover {
            background-color: #222222; /* slightly lighter black on hover */
            cursor: pointer;
        }        
    

        /* --- Tables --- */
        .stTable, table {
            border-collapse: collapse;
            color: var(--text-color);
        }
        table th {
            background-color: var(--primary-color);
            color: var(--text-color-on-primary);
        }
        table td, table th {
            padding: 8px;
            text-align: left;
        }

        /* --- Plots (Plotly / Altair) --- */
        .stPlotlyChart > div, .stAltairChart > div {
            background-color: var(--background-color);
            border-radius: 8px;
            padding: 0.5em;
        }

        
        """,
        unsafe_allow_html=True
    )       
    

import streamlit as st

def apply_custom_theme():
    custom_css = """

    /* Expander header container */
    div[role="button"].stExpander {
        background-color: #e8e8e8 !important;  /* light gray */
        border: 1px solid #d1d1d1 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        color: #000000 !important;
        font-weight: 500 !important;
    }

    /* Ensure header text stays black */
    div[role="button"].stExpander span,
    div[role="button"].stExpander div {
        color: #000000 !important;
    }

    /* Hover effect on header */
    div[role="button"].stExpander:hover {
        background-color: #f2f2f2 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08) !important;
        cursor: pointer;
    }

    /* Content inside expander */
    div.stExpanderContent {
        background-color: #f9f9f9 !important;
        border-top: 1px solid #d1d1d1 !important;
        padding: 1rem !important;
        color: #111111 !important;
    }

    /* All nested text in content forced to black */
    div.stExpanderContent * {
        color: #111111 !important;
    }

    /* Arrow rotation on open */
    div[role="button"].stExpander svg {
        transition: transform 0.2s ease !important;
    }

    div[role="button"].stExpander[aria-expanded="true"] svg {
        transform: rotate(90deg) !important;
    }
    
    /* Overall background */
    .stApp {
        background-color: #f9f9f9;
        color: #000000;   
    }

    /* Main container and sections */
    .main .block-container {
        padding: 2rem 2rem 2rem 2rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    /* Titles, headers, and subheaders */
    h1, h2, h3, h4 {
        color: #111111;
    }    


    /* Subtitles */
    .stMarkdown h4, .stMarkdown h5 {
        color: #333333;
    }

    /* Buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: #00000;   
        border-radius: 6px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #000000;      
    }

    /* Download button styling */
    .stDownloadButton>button {
        background-color: #dc3545; /* red background */
        color: #ffffff;            /* white text */
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        cursor: pointer;
        transition: background-color 0.2s ease;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3); /* subtle text shadow */
    }

    .stDownloadButton>button:hover {
        background-color: #c82333; /* slightly darker red on hover */
    }
    
    /* Text input field */
/* Text input field styling */
    div.stTextInput>div>div>input {
        background-color: #ffffff !important; /* white input box */
        color: #000000 !important;            /* black input text */
        border: 1px solid #d1d1d1 !important; /* subtle border */
        border-radius: 6px !important;
        padding: 0.4rem 0.6rem !important;
    }

    /* Placeholder text color */
    div.stTextInput>div>div>input::placeholder {
        color: #666666 !important; /* gray placeholder */
    }
    

    /* Tabs */
    [data-baseweb="tab-list"] button {
        background-color: #e8e8e8;
        color: #111111;
        border-radius: 6px 6px 0 0;
        font-weight: 500;
    }

    [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #4CAF50;
    }

    /* Info, warning, and error boxes */
    /* All alerts default to dark red background with white text */
    /* All alerts general styling */
/* All alerts general styling with forced black text */
    .stAlert {
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #ffffff;  /* soft light red/pink */
        border-left: 6px solid #000000; /* red accent */
        color: #000000 !important;      /* force black text */
    }

    /* Force all text inside the alert to black */
    .stAlert * {
        color: #000000 !important;
    }

    /* Info alerts */
    .stAlert[data-testid="stAlert-info"] {
        background-color: #e6f0fa;  
        border-left: 4px solid #1a73e8; 
    }

    /* Warning alerts */
    .stAlert[data-testid="stAlert-warning"] {
        background-color: #fff8e1;  
        border-left: 4px solid #ffc107; 
    }

    /* Error alerts */
    .stAlert[data-testid="stAlert-error"] {
        background-color: #ffffff;    
        border-left: 4px solid #000000;    
    }
    
    

    /* File uploader container */
    .stFileUploader > div {
        border: 2px dashed #d1d1d1;
        border-radius: 8px;
        padding: 1rem;
        background-color: #fafafa; /* light neutral background */
        color: #111111; /* default text color */
    }

    /* Browse button inside file uploader */
    .stFileUploader button {
        background-color: #000000; /* black button */
        color: #ffffff; /* white text */
        border-radius: 6px;
        padding: 0.4rem 0.8rem;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.2s ease;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5); /* subtle shadow */
    }

    .stFileUploader button:hover {
        background-color: #222222; /* slightly lighter black on hover */
    }
     
    /* Loaders / spinners */
    .stSpinner>div {
        border: 4px solid #e0e0e0;
        border-top: 4px solid #4CAF50;
    }
    """
    st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

    