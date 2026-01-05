import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variable for model if set, else default
load_dotenv()
model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
model = SentenceTransformer(model_name)

# Define the schema
schema_texts = {
    "staff_details": """CREATE TABLE staff_details (
    staff_id SERIAL PRIMARY KEY,
    staff_code VARCHAR(10) UNIQUE,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    gender CHAR(1),
    date_of_birth DATE,
    email VARCHAR(100) UNIQUE,
    phone VARCHAR(15),
    designation VARCHAR(50),
    department VARCHAR(50),
    date_of_joining DATE
);""",
    "student_details": """CREATE TABLE student_details (
    student_id SERIAL PRIMARY KEY,
    admission_no VARCHAR(15) UNIQUE,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    gender CHAR(1),
    date_of_birth DATE,
    email VARCHAR(100),
    phone VARCHAR(15),
    class VARCHAR(5),
    section CHAR(1),
    admission_date DATE
);""",
    "subject_details": """CREATE TABLE subject_details (
    subject_id SERIAL PRIMARY KEY,
    subject_code VARCHAR(10) UNIQUE,
    subject_name VARCHAR(100),
    staff_id INT REFERENCES staff_details(staff_id),
    class VARCHAR(5),
    section CHAR(1),
    weekly_hours INT,
    academic_year VARCHAR(9),
    created_at TIMESTAMP
);""",
    "class_details": """CREATE TABLE class_details (
    class_detail_id SERIAL PRIMARY KEY,
    class VARCHAR(5),
    section CHAR(1),
    student_id INT REFERENCES student_details(student_id),
    class_teacher_id INT REFERENCES staff_details(staff_id),
    academic_year VARCHAR(9),
    room_number VARCHAR(10),
    shift VARCHAR(20),
    created_at TIMESTAMP
);""",
    "period_activity": """CREATE TABLE period_activity (
    period_activity_id SERIAL PRIMARY KEY,
    class VARCHAR(5),
    section CHAR(1),
    period_number INT,
    subject_id INT REFERENCES subject_details(subject_id),
    staff_id INT REFERENCES staff_details(staff_id),
    day_of_week VARCHAR(10),
    start_time TIME,
    end_time TIME,
    academic_year VARCHAR(9),
    created_at TIMESTAMP
);""",
    "attendance_pct": """CREATE TABLE attendance_pct (
    attendance_id SERIAL PRIMARY KEY,
    student_id INT REFERENCES student_details(student_id),
    month VARCHAR(15),
    academic_year VARCHAR(9),
    total_working_days INT,
    days_present INT,
    days_absent INT,
    leave_reason VARCHAR(100),
    attendance_percentage NUMERIC(5,2),
    recorded_on DATE
);"""
}

# Generate embeddings
print("Generating embeddings...")
schema_descriptions = list(schema_texts.values())
embeddings = model.encode(schema_descriptions, normalize_embeddings=True)
embeddings = np.array(embeddings).astype('float32')
print(f"Embeddings shape: {embeddings.shape}")

# Create and populate FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner Product (Cosine Similarity since normalized)
index.add(embeddings)
print(f"Number of vectors in index: {index.ntotal}")

# Save index and schema texts
faiss.write_index(index, "schema_index.faiss")

with open("schema_texts.pkl", "wb") as f:
    pickle.dump(schema_texts, f)

print("Saved schema_index.faiss and schema_texts.pkl successfully.")
