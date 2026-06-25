# Entry point for Streamlit Cloud
# Streamlit Cloud runs this file directly, so we exec krishi_mitra.py
# in the current module context so all st.* calls render correctly.

exec(open("krishi_mitra.py", encoding="utf-8").read())
