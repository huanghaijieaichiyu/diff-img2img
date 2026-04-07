import os
import sys
import streamlit.web.cli as stcli

def resolve_path(path):
    if getattr(sys, "frozen", False):
        basedir = sys._MEIPASS
    else:
        basedir = os.path.dirname(__file__)
    return os.path.join(basedir, path)

if __name__ == "__main__":
    # Set environment variable to help app find resources if needed
    os.environ["FROZEN_APP_PATH"] = resolve_path(".")
    
    # Path to your streamlit app
    app_path = resolve_path(os.path.join("ui", "app.py"))
    
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())
