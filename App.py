"""
Mann Ki Baat Analysis - Main App
Multi-page Streamlit Application
"""

import streamlit as st

st.set_page_config(
    page_title="Mann Ki Baat Analysis",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎙️ Mann Ki Baat Analysis Platform")
st.markdown("---")

st.markdown("""
## Welcome to the Mann Ki Baat Topic & Stock Impact Analysis

This platform provides comprehensive analysis of PM Modi's Mann Ki Baat speeches:

### 📊 **Page 1: Topic Modeling**
- Discover key themes and topics across episodes
- Analyze topic distribution over time
- Explore individual episodes
- Export topic data for further analysis

### 📈 **Page 2: Stock Market Impact**
- Correlate topics with sector performance
- Analyze temporal trends
- Visualize topic-stock relationships
- Export correlation data

---

### 🚀 Getting Started

1. **Navigate** to "Topic Modeling" from the sidebar
2. **Configure** model parameters
3. **Train** the topic model
4. **Export** quarterly and episode data
5. **Switch** to "Stock Market Impact" for correlation analysis

### 📁 Required Files

- **For Topic Modeling**: Transcript files in `mann_ki_baat_transcripts/` folder
- **For Stock Impact**: Exported CSV files from Topic Modeling in `data/` folder

---

**Select a page from the sidebar to begin →**
""")

st.sidebar.success("Select a page above to get started")