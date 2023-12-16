import streamlit as st
from multiapp import MultiApp
from scripts import utils
from pages import overview_analysis, engagement_analysis, experience_analysis, satisfaction_analysis, model_implementation

# load the data from the slq dump to a postgress db and read it to a pandas df then save as pkl in the data/ folder
utils.load_data_to_df()

st.set_page_config(page_title="TellCo Telecom Analytics", layout="wide")

app = MultiApp()

st.sidebar.markdown("""
# TellCo's User Analytics
### Multi-Page App
This multi-page app is using the [streamlit-multiapps](https://github.com/upraneelnihar/streamlit-multiapps) framework developed by [Praneel Nihar](https://medium.com/@u.praneel.nihar). Also check out his [Medium article](https://medium.com/@u.praneel.nihar/building-multi-page-web-app-using-streamlit-7a40d55fa5b4).
### Modifications
\t- Page Folder Based Access
\t- Presentation changed to SideBar
""")

# Add all your application here
app.add_app("User Overview Analysis", overview_analysis.app)
app.add_app("User Engagement Analysis", engagement_analysis.app)
app.add_app("User Experience Analysis", experience_analysis.app)
app.add_app("User Satisfaction Analysis", satisfaction_analysis.app)
app.add_app("Predict Satisfaction", model_implementation.app)

# The main app
app.run()