# Smart Assistant for Research Summarization

This GenAI assistant processes uploaded PDF/TXT documents to:

- Generate a 150-word summary.
- Answer any user question (Ask Anything) with citation from document.
- Pose 3 logic-based questions (Challenge Me) and evaluate user's answers with justification.

## Features

- Context-aware answers grounded in the uploaded document.
- Logic question generation and grading.
- Clean UI using Gradio.
- Strict reference-based prompting.

## Setup Instructions

```bash
git clone https://github.com/yourusername/docwise-assistant
cd docwise-assistant
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Add your OpenAI API key to `.env`:

```
OPENAI_API_KEY=your-api-key
```

Run the app:

```bash
python app.py
```

