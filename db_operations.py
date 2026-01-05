from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from sqlalchemy.dialects.postgresql import ARRAY
from datetime import datetime, timezone
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve database credentials from environment variables
DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_PORT = os.getenv("port")
DB_NAME = os.getenv("database")

Base = declarative_base()

class ChatbotMaster(Base):
    __tablename__ = 'chatbot_master'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    user_type = Column(Text, nullable=False)
    session_start_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    session_end_at = Column(DateTime, nullable=True)
    status = Column(String, nullable=False)

class ChatbotUserQuery(Base):
    __tablename__ = 'chatbot_user_query'

    id = Column(Integer, primary_key=True, autoincrement=True)
    chatbot_master_id = Column(Integer, ForeignKey('chatbot_master.id'), nullable=False)
    intent_type = Column(String, nullable=True)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    token_usage = Column(Integer, nullable=True)
    token_cost = Column(Float, nullable=True)
    status = Column(String, nullable=True)
    remarks = Column(Text, nullable=True)
    generated_query = Column(Text, nullable=True)

    chatbot_master = relationship("ChatbotMaster", back_populates="user_queries")

ChatbotMaster.user_queries = relationship("ChatbotUserQuery", order_by=ChatbotUserQuery.id, back_populates="chatbot_master")

# Construct the database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
print("database:", DATABASE_URL)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def database():
    """
    Initializes the database by creating tables if they do not exist.
    """
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully.")

def get_or_create_chatbot_master(session, session_id, user_type):
    """
    Get or create a ChatbotMaster record and return its ID.
    """
    chatbot_master = session.query(ChatbotMaster).filter_by(session_id=session_id).first()
    if not chatbot_master:
        # Create a new ChatbotMaster record if it doesn't exist
        chatbot_master = ChatbotMaster(
            session_id=session_id,
            user_type= user_type,         
            status="active",
        )
        session.add(chatbot_master)
        session.commit()  # Save the record to the database
    return chatbot_master.id

def end_chatbot_session(session: Session, session_id: str):
    """
    Update session_end_at and status in ChatbotMaster when the session ends.
    """
    try:
        chatbot_master = session.query(ChatbotMaster).filter_by(session_id=session_id).first()
        if chatbot_master:
            chatbot_master.session_end_at = datetime.now(timezone.utc)
            chatbot_master.status = "inactive"
            session.commit()
            print(f"Session {session_id} ended successfully in database.")
        else:
            print(f"No ChatbotMaster record found for session_id: {session_id}")
            raise ValueError("Invalid session_id")
    except Exception as e:
        print(f"Error ending chatbot session: {e}")
        session.rollback()
        raise

def store_query_details(
    session, chatbot_master_id, intent_type, query,
    response, token_usage, token_cost, status, remarks, generated_query
):
    """
    Inserts query-related details into the appropriate table based on user type.
    """
    try:
        record = ChatbotUserQuery(
                chatbot_master_id=chatbot_master_id,
                intent_type=intent_type,
                query=query,
                response=response,
                token_usage=token_usage,
                token_cost=token_cost,
                status=status,
                remarks=remarks,
                generated_query = generated_query
            )        

        session.add(record)
        session.commit()
        print("Chatbot User Query details stored successfully.")
    except Exception as e:
        print(f"Error storing query details: {e}")
        session.rollback()
        raise