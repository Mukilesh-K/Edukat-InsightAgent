import streamlit as st
from decimal import Decimal
import os
import json
import faiss
import pickle
import uuid
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_classic.chains import LLMChain
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import logging
from datetime import datetime 
from db_operations import database
from intent_classify import IntentClassification

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Retrieve credentials
user = os.getenv("user")
password = os.getenv("password")
host = os.getenv("host")
port = os.getenv("port")
database_name = os.getenv("database")
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")

# Page configuration
st.set_page_config(page_title="Database Query Assistant", page_icon=":mag:", layout="centered")

# Initialize models
@st.cache_resource
def get_sentence_transformer():
    return SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

model = get_sentence_transformer()
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)

# Helper Functions
def init_database():
    try:
        pg_uri = f"postgresql://{user}:{password}@{host}:{port}/{database_name}"
        db = SQLDatabase.from_uri(pg_uri)
        logger.info("Database initialized successfully.")
        return db
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return None

def load_yaml_config(file_path):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load YAML file '{file_path}': {e}")
        return {}

def load_faiss_index_and_schema(load_embeddings=False):
    """Load FAISS index and schema texts."""
    try:
        faiss_index = faiss.read_index("schema_index.faiss")
        with open("schema_texts.pkl", "rb") as f:
            schema_texts = pickle.load(f)
        
        if load_embeddings:
            schema_embeddings = model.encode(list(schema_texts.values()), normalize_embeddings=True)
            return faiss_index, schema_texts, np.array(schema_embeddings)
        
        return faiss_index, schema_texts
    except Exception as e:
        logger.error(f"Error loading FAISS resources: {e}")
        return None, None, None if load_embeddings else (None, None)

def search_schema(query, top_k=7):
    """Search schema using FAISS."""
    faiss_index, schema_texts, schema_embeddings = load_faiss_index_and_schema(load_embeddings=True)
    if not faiss_index:
        return []

    try:
        query_embedding = model.encode([query], normalize_embeddings=True)
        distances, indices = faiss_index.search(np.array(query_embedding), top_k)
        
        similar_embeddings = schema_embeddings[indices[0]]
        similarities = cosine_similarity(query_embedding, similar_embeddings)[0]
        
        schema_values = list(schema_texts.values())
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(schema_values):
                results.append((schema_values[idx], float(similarities[i])))

        # Mandatory tables
        mandatory_tables = ['staff_details', 'student_details', 'class_details']
        final_results = []
        
        for table in mandatory_tables:
            for s_name, s_text in schema_texts.items():
                if s_name == table:
                    final_results.append((s_text, 1.0))
        
        mandatory_texts = {r[0] for r in final_results}
        for res in results:
            if res[0] not in mandatory_texts:
                final_results.append(res)
                
        return sorted(final_results, key=lambda x: x[1], reverse=True)
    except Exception as e:
        logger.error(f"Schema search error: {e}")
        return []

class QueryProcessor:
    def __init__(self):
        self.intent_classifier = IntentClassification()
        self.intents_config = load_yaml_config('intents.yml')
        self.entities_config = load_yaml_config('entities.yml')

    def classify_intent(self, text):
        return self.intent_classifier.intent_classification(text)

    def extract_entities(self, query, intent):
        """Extract entities based on intent configuration using LLM."""
        if intent not in self.intents_config:
            return {}
            
        required_entities = self.intents_config[intent].get('entities', [])
        if not required_entities:
            return {}

        # Prepare context from entities.yml for the required entities
        entity_definitions = {k: self.entities_config.get(k, []) for k in required_entities}
        
        prompt = ChatPromptTemplate.from_template(
            """Extract the following entities from the user query: {entities_list}.
            
            Entity Definitions/Examples:
            {entity_defs}
            
            User Query: "{query}"
            
            Return the result as a valid JSON object where keys are the entity names and values are the extracted values (or null if not found).
            Scale/normalize values if needed (e.g., "5th grade" -> "5", "maths" -> "Mathematics").
            JSON:
            """
        )
        
        chain = prompt | llm | JsonOutputParser()
        try:
            return chain.invoke({
                "entities_list": required_entities,
                "entity_defs": json.dumps(entity_definitions, indent=2),
                "query": query
            })
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {}

    def is_related_query(self, current_query, previous_query, previous_response):
        if not previous_query:
            return False
            
        prompt = ChatPromptTemplate.from_template(
            """Determine if the Current Query is related to the Previous Query or Response.
            Previous Query: "{last_query}"
            Previous Response: "{last_response}"
            Current Query: "{current_query}"
            
            Is it a follow-up, refinement, or related question? 
            Return strictly "YES" or "NO".
            """
        )
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({
            "last_query": previous_query,
            "last_response": previous_response,
            "current_query": current_query
        })
        
        is_related_llm = "YES" in result.strip().upper()
        
        # Semantic check as backup/confirmation
        embeddings = model.encode([current_query, previous_query])
        similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        
        return is_related_llm or (similarity > 0.6)

    def generate_contextual_response(self, query, previous_response):
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant. The user is asking a follow-up question.
            Previous Context (Answer to previous question):
            {context}
            
            User's Follow-up Question:
            {query}
            
            Answer the user's question using the provided context. If the answer isn't in the context, politely say you don't have that information.
            """
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": previous_response, "query": query})

    def get_sql_chain(self, query, entities, schema_context):
        template = """
        You are an expert PostgreSQL data analyst for a school.
        Generate a SQL query to answer the user's question.
        
        Schema Context:
        {schema}
        
        User Question: {query}
        Extracted Entities: {entities}
        
        Rules:
        1. Return ONLY the SQL query. No markdown, no explanations.
        2. Use the provided schema.
        3. Handle case sensitivity (use ILIKE if unsure).
        4. Join tables correctly based on IDs (e.g., student_id, staff_id, class_id).
        
        ### Example SQL Queries:
        
        - **Question**: "which staff is taking english for 5b"
        - **SQL Query**:
            ```sql
            SELECT DISTINCT
                s.staff_code,
                s.first_name,
                s.last_name,
                s.designation,
                s.department
            FROM subject_details sd
            JOIN staff_details s 
                ON sd.staff_id = s.staff_id
            WHERE LOWER(sd.subject_name) = 'english'
            AND sd.class = '9'
            AND sd.section = 'B';
            ```
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        
        schema_str = "\n".join([f"{s[0]}" for s in schema_context])
        return chain.invoke({
            "schema": schema_str,
            "query": query,
            "entities": entities
        }).replace("```sql", "").replace("```", "").strip()

    def generate_response(self, query, sql_result, executed_sql):
        template = """
    You are an intelligent assistant generating user-friendly responses based on query results, document details and extracted columns, identified entities like school, class, and section. Use the data as is, without assuming additional information.
        User Question: {query}
        Executed SQL: {sql}
        Query Result: {result}
        
        Provide a natural language answer based on the result.
        If the result is empty, say "I couldn't find any information matching your request."
        Format list items clearly.
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({
            "query": query,
            "sql": executed_sql,
            "result": sql_result
        })

# Main Application Flow
def main():
    st.markdown("<h1 style='text-align: center;'>Database Query Assistant</h1>", unsafe_allow_html=True)
    
    # Initialize DB
    db = init_database()
    if not db:
        st.error("Failed to connect to the database. Please check your configuration.")
        return

    # Processor
    processor = QueryProcessor()

    # Session State
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'last_q' not in st.session_state:
        st.session_state.last_q = ""
    if 'last_r' not in st.session_state:
        st.session_state.last_r = ""

    # Chat UI
    for q, r in st.session_state.history:
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(r)

    query = st.chat_input("Ask a question about the school database...")
    
    if query:
        st.chat_message("user").write(query)
        
        with st.spinner("Processing..."):
            response = ""
            
            # # 1. Check Related
            # if processor.is_related_query(query, st.session_state.last_q, st.session_state.last_r):
            #     logger.info("Query detected as related to previous context.")
            #     response = processor.generate_contextual_response(query, st.session_state.last_r)
            # else:
            # 1. New Query Flow
            # Intent
            intent = processor.classify_intent(query)
            logger.info(f"Target Intent: {intent}")
            
            if not intent:
                response = "I couldn't understand your intent. Please try asking differently."
            else:
                # Entities
                entities = processor.extract_entities(query, intent)
                logger.info(f"Extracted Entities: {entities}")
                
                # Schema
                schema_results = search_schema(query)
                
                # SQL
                sql_query = processor.get_sql_chain(query, entities, schema_results)
                logger.info(f"Generated SQL: {sql_query}")
                    
                # Execute
                try:
                    sql_result = db.run(sql_query)
                    logger.info(f"SQL Result: {sql_result}")
                        
                    # Response
                    response = processor.generate_response(query, sql_result, sql_query)
                except Exception as e:
                    logger.error(f"SQL Execution Error: {e}")
                    response = f"I generated a query but it failed to run. \nSQL: `{sql_query}`\nError: {str(e)}"

            # Update State
            st.session_state.last_q = query
            st.session_state.last_r = response
            st.session_state.history.append((query, response))
            
            st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()