"""Entry point for Streamlit Cloud deployment.

This file is required at the repository root for Streamlit Cloud to
discover and run the app. It simply imports and runs the main app from
the dimtensor.web package.
"""

from src.dimtensor.web.app import main

if __name__ == "__main__":
    main()
