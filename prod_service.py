import os
from datetime import datetime
import json
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
import asyncio
import logging
from contextlib import asynccontextmanager
from db_operations import end_chatbot_session, SessionLocal, database, ChatbotMaster, get_or_create_chatbot_master, store_query_details
from sqlalchemy.orm import Session
import uuid
from dotenv import load_dotenv
from prod_main import QueryProcessor, load_faiss_index_and_schema, search_schema
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.callbacks.manager import get_openai_callback
from typing import Union, List, Dict, Any
from datetime import timezone

# Load environment variables from .env file
load_dotenv()

class LangChainMemoryManager:
    def __init__(self):
        self.session_memories = {}  # {session_id: {'chat_history': [], 'metadata': {}}}
        self.storage = {} # Simple key-value store for compatibility
        
    def get(self, key):
        return self.storage.get(key)
        
    def set(self, key, value):
        self.storage[key] = value

    def _ensure_session(self, session_id: str):
        if session_id not in self.session_memories:
            self.session_memories[session_id] = {
                'chat_history': ChatMessageHistory(),
                'metadata': {
                    'created_at': datetime.now(timezone.utc),
                    'updated_at': datetime.now(timezone.utc),
                    'status': 'active'  #set initial status to active
                }
            }
    
    def add_message(self, session_id: str, message: Union[HumanMessage, AIMessage]):
        self._ensure_session(session_id)
        history = self.session_memories[session_id]['chat_history']
        
        if isinstance(history, ChatMessageHistory):
            if isinstance(message, HumanMessage):
                history.add_user_message(message.content)
            elif isinstance(message, AIMessage):
                history.add_ai_message(message.content)
        else:
            history.append(message)  # fallback if using plain list

        self.session_memories[session_id]['metadata']['updated_at'] = datetime.now(timezone.utc)
            
    def get_chat_history(self, session_id: str) -> List[Union[HumanMessage, AIMessage]]:
        if session_id not in self.session_memories:
            return []
        history = self.session_memories[session_id]['chat_history']
        return history.messages if isinstance(history, ChatMessageHistory) else history
        
    def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.session_memories:
            return {}
        return self.session_memories[session_id]['metadata']
    
    def clear_session(self, session_id: str):
        if session_id in self.session_memories:
            del self.session_memories[session_id]

    def get_metadata(self, session_id: str, key: str, default=None):
        if session_id not in self.session_memories:
            return default
        return self.session_memories[session_id]['metadata'].get(key, default)

    def set_metadata(self, session_id: str, key: str, value: Any):
        if session_id in self.session_memories:
            self.session_memories[session_id]['metadata'][key] = value


memory = LangChainMemoryManager()

SAMPLE_QUESTIONS = [
    "Show me the school details",
    "How many students are there?",
    "List all staff members",
    "What is the schedule for Class 5A?"
]

logger = logging.getLogger(__name__)

def get_session_id():
    """
    Generate a new, unique session ID using UUID4.
    """
    return str(uuid.uuid4())

def get_session_data(session_id: str):
    data = memory.get(f"session:{session_id}")
    if data:
        return json.loads(data)
    return None

def save_session_data(session_id: str, data: dict):
    memory.set(f"session:{session_id}", json.dumps(data))

def init_session_data(session_id: str):
    data = {
        "session_id": session_id,
        "chat_history": [],
        "last_q": None,
        "last_r": None,
        "created_at": str(datetime.now())
    }
    save_session_data(session_id, data)
    return data

class QueryRequest(BaseModel):
    session_id: str
    user_query: str

class ChatResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    data: dict

class SessionRequest(BaseModel):
    session_id: str

class ChatMessage(BaseModel):
    timestamp: str
    message: str
    message_type: str  # 'HumanMessage' or 'AIMessage'

class ChatHistoryResponse(BaseModel):
    session_id: str
    chat_history: List[ChatMessage]
    success: bool
    message: str

class StreamlitManager:
    def __init__(self):
        self.process = None
        self.status = "stopped"
        
    async def start_app(self):
        if self.process is None:
            try:
                self.process = await asyncio.create_subprocess_exec(
                    "streamlit", "run", "main.py",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                self.status = "running"
                asyncio.create_task(self._monitor_process())
            except Exception as e:
                logger.error(f"Failed to start Streamlit: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def _monitor_process(self):
        try:
            await self.process.wait()
        except Exception as e:
            logger.error(f"Streamlit process error: {e}")
        finally:
            self.status = "stopped"
            self.process = None

    async def stop_app(self):
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
                logger.info("Streamlit process terminated gracefully.")
            except asyncio.TimeoutError:
                logger.warning("Streamlit process did not terminate gracefully. Force killing it.")
                self.process.kill()
                await self.process.wait()
            except Exception as e:
                logger.error(f"Unexpected error while stopping Streamlit: {e}")
                raise HTTPException(status_code=500, detail="Failed to stop Streamlit process")
            finally:
                self.status = "stopped"
                self.process = None
                logger.info("Streamlit process cleanup complete.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load resources
    try:
        faiss_index, schema_texts = load_faiss_index_and_schema()
        if faiss_index is None or schema_texts is None:
            logger.warning("FAISS index or schema texts could not be loaded. Search functionality will be limited.")
        
        app.state.faiss_index = faiss_index
        app.state.schema_texts = schema_texts
        
        # Initialize QueryProcessor
        # Note: QueryProcessor in prod_main initializes its own resources (IntentClassifier, etc.)
        app.state.query_processor = QueryProcessor()
        logger.info("Application resources initialized successfully.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # We don't raise here to allow app to start, but functionality will be broken.
    
    yield
    
    # Cleanup if needed

app = FastAPI(lifespan=lifespan)

@app.get("/chatbot")
def health_check():
    return "Application Health Status Good"

def get_db():
    try:
        db_instance = SessionLocal()
        try:
            yield db_instance
        finally:
            db_instance.close()
    except Exception as e:
        logger.error(f"Database session creation failed: {e}")
        raise HTTPException(status_code=500, detail="Database session error")

@app.post("/chatbot/chat_initialize", response_model=ChatResponse)
async def start_session():
    try:
        session_id = get_session_id()
        init_session_data(session_id)

        return ChatResponse(
            success=True,
            message="New session started successfully.",
            session_id=session_id,  
            data={
                "status": "running",
                "sample_questions": SAMPLE_QUESTIONS,
                "is_clickable": True
            }
        )

    except Exception as e:
        logger.error(f"Error starting new session: {e}")
        return ChatResponse(
            success=False,
            message="Failed to start session",
            session_id="",           
            data={"error": str(e)}   
        )

@app.post("/chatbot/chat_interact", response_model=ChatResponse)
async def interact_with_chatbot(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    db: Any = Depends(get_db)
):
    try:
        session_data = get_session_data(request.session_id)
        if session_data is None:
            # Try to init if missing (resilience)
            logger.warning(f"Session {request.session_id} not found, initializing new session.")
            session_data = init_session_data(request.session_id)

        user_query = request.user_query
        current_time = str(datetime.now())

        # Log User Message
        human_message = ChatMessage(
            timestamp=current_time,
            message=user_query,
            message_type="HumanMessage"
        ).dict()
        session_data["chat_history"].append(human_message)
        
        # Processor
        processor = app.state.query_processor
        
        response_text = ""
        intent = None
        sql_query = None
        
        # Get or create chatbot master record
        chatbot_master_id = get_or_create_chatbot_master(db, request.session_id, user_type="user")
        
        # Initialize token tracking variables
        total_tokens = 0
        total_cost = 0.0

        try:
            with get_openai_callback() as cb:
                # 1. Intent Classification
                intent = processor.classify_intent(user_query)
                logger.info(f"Identified Intent: {intent}")
                
                if not intent:
                    response_text = "I couldn't understand your intent. Please try asking differently."
                else:
                    # 2. Entity Extraction
                    entities = processor.extract_entities(user_query, intent)
                    logger.info(f"Entities: {entities}")
                    
                    # 3. Schema Search
                    schema_results = search_schema(user_query)
                    
                    # 4. SQL Generation
                    sql_query = processor.get_sql_chain(user_query, entities, schema_results)
                    logger.info(f"Generated SQL: {sql_query}")
                    
                    # 5. Execution
                    from sqlalchemy import text
                    result_proxy = db.execute(text(sql_query))
                    
                    # Fetch results appropriately
                    if result_proxy.returns_rows:
                        sql_result = [tuple(row) for row in result_proxy.fetchall()]
                    else:
                        db.commit() 
                        sql_result = "Query executed successfully."

                    logger.info(f"SQL Result: {sql_result}")
                    
                    # 6. Response Generation
                    response_text = processor.generate_response(user_query, sql_result, sql_query)
                
                # Capture token usage
                total_tokens = cb.total_tokens
                total_cost = cb.total_cost

        except Exception as e:
            logger.error(f"Error during query processing pipeline: {e}")
            response_text = f"An error occurred while processing your request: {str(e)}"
            # Even if error, we want to log the attempt

        # Log AI Message
        ai_message = ChatMessage(
            timestamp=str(datetime.now()),
            message=str(response_text),
            message_type="AIMessage"
        ).dict()
        session_data["chat_history"].append(ai_message)
        
        # Save session
        save_session_data(request.session_id, session_data)
        
        # Store query details in DB
        store_query_details(
            session=db,
            chatbot_master_id=chatbot_master_id,
            intent_type=intent,
            query=user_query,
            response=response_text,
            token_usage=total_tokens,
            token_cost=total_cost,
            status="success" if intent else "failed",
            remarks=None,
            generated_query=sql_query
        )
        
        return ChatResponse(
            success=True,
            message="Query processed successfully.",
            session_id=request.session_id,
            data={
                "status": "running",
                "session_id": request.session_id,
                "chat_history": session_data["chat_history"],
                "is_clickable": True
            }
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return ChatResponse(
            success=False,
            message=f"Failed to process query: {str(e)}",
            session_id=request.session_id,
            data={}
        )

@app.post("/chatbot/chat_end", response_model=ChatResponse)
async def end_application(request: SessionRequest):
    try:
        session_data = get_session_data(request.session_id)
        if session_data:
            # We don't delete, just mark or leave as is. 
            pass
        
        return ChatResponse(
            success=True,
            message="Session ended successfully.",
            session_id=request.session_id,
            data={
                "status": "stopped",
                "session_id": request.session_id,
                "chat_history": session_data["chat_history"] if session_data else [],
                "is_clickable": False
            }
        )
    except Exception as e:
        logger.error(f"Error ending session: {e}")
        return ChatResponse(
            success=False,
            message="Failed to end session",
            session_id=request.session_id,
            error=str(e),
            data={}
        )

@app.post("/chatbot/chat_history", response_model=ChatHistoryResponse)
async def get_chat_history(request: SessionRequest):
    try:
        session_data = get_session_data(request.session_id)
        if not session_data:
            return ChatHistoryResponse(
                session_id=request.session_id,
                chat_history=[],
                success=False,
                message="Session not found"
            )
            
        return ChatHistoryResponse(
            session_id=request.session_id,
            chat_history=session_data["chat_history"],
            success=True,
            message="Chat history retrieved successfully"
        )
    
    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        return ChatHistoryResponse(
             session_id=request.session_id,
             chat_history=[],
             success=False,
             message=f"Error: {str(e)}"
        )

@app.post("/chatbot/chat_status", response_model=ChatResponse)
async def get_status(request: SessionRequest):
    try:
        session_data = get_session_data(request.session_id)
        if not session_data:
             return ChatResponse(
                 success=False, 
                 message="Invalid session", 
                 session_id=request.session_id,
                 data={"status": "unknown"}
            )

        return ChatResponse(
            success=True,
            message="Status retrieved",
            session_id=request.session_id,
            data={
                "status": "active",
                "session_id": request.session_id
            }
        )
    except Exception as e:
        logger.error(f"Error retrieving session status: {e}")
        return ChatResponse(
            success=False, 
            message=str(e), 
            session_id=request.session_id,
            data={}
        )