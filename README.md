# Edukat-InsightAgent

Edukat-InsightAgent is an intelligent database query assistant designed for educational institutions. It leverages Generative AI to translate natural language queries into SQL commands, allowing staff and administrators to interact with their school database using plain English.

The project features a dual-interface architecture:
- **Streamlit UI**: An interactive web-based chat interface for direct user interaction.
- **FastAPI Backend**: A RESTful API service to integrate the chat functionality into other platforms.

## Features

- **Natural Language to SQL**: Converts user questions (e.g., "How many students are in class 5A?") into executable SQL queries.
- **Intent Classification**:Intelligently categorizes user queries to determine the best processing flow.
- **Entity Extraction**: Automatically identifies key details like class names, sections, and subjects from the query.
- **Context Awareness**: Maintains conversation history for follow-up questions.
- **Dual Interface**: Run as a standalone web app or as a backend microservice.

## Tech Stack

- **Language**: Python 3.9+
- **LLM Integration**: LangChain, OpenAI (GPT-4o-mini)
- **Database**: PostgreSQL
- **Vector Search**: FAISS (for schema search)
- **Web Frameworks**: Streamlit, FastAPI
- **Libraries**: SQLAlchemy, Pydantic, Sentence-Transformers

## Prerequisites

- Python 3.9 or higher
- PostgreSQL Database (with schema loaded)
- OpenAI API Key

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Edukat-InsightAgent
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the root directory and configure your credentials:

```env
# Database Configuration
user=your_db_username
password=your_db_password
host=localhost
port=5432
database=your_database_name

# AI Configuration
OPENAI_API_KEY=sk-your-openai-api-key
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
```

## Usage

### Option 1: Run the Streamlit UI
This launches the interactive chat interface in your browser.

```bash
streamlit run prod_main.py
```

### Option 2: Run the Chatbot API
This starts the FastAPI server, accessible via HTTP endpoints.

```bash
uvicorn prod_service:app --reload
```

- **Swagger Documentation**: Visit `http://127.0.0.1:8000/docs` to test the API endpoints interactively.
- **Health Check**: `GET /chatbot`

## API Endpoints

- `POST /chatbot/chat_initialize`: Start a new chat session.
- `POST /chatbot/chat_interact`: Send a message and get a response.
- `POST /chatbot/chat_history`: Retrieve conversation history.
- `POST /chatbot/chat_end`: End a session.

## For cost analytics dashboard:
This launches the cost analytics dashboard in your browser. To track the cost of the API calls, you can use the cost analytics dashboard.

```bash
streamlit run cost_analytics.py
```