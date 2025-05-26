# RAQ

RAQ is a Streamlit application that allows you to upload documents and ask questions about their content. It uses Retrieval-Augmented Generation (RAG) with Groq's fast AI models to provide contextual and accurate answers.

## Features

- Upload documents in PDF format.
- Ask questions about the content of uploaded documents.
- Uses Retrieval-Augmented Generation (RAG) with Groq's fast AI models for contextual and accurate answers.

## Requirements

- Python 3.12 or higher
- Streamlit
- PyPDF2

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   ```

2. Navigate to the project directory:

   ```bash
   cd QuickRAG
   ```

3. Create and activate a virtual environment:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:

   ```bash
   streamlit run streamlit_app.py
   ```

2. Open the provided URL in your browser to interact with the application.

## Troubleshooting

- Ensure the virtual environment is activated before running the application.
- If `PyPDF2` import issues persist, verify the Python interpreter in your editor is set to the virtual environment.
