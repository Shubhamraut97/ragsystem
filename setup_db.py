import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"))

# Create tables
with engine.connect() as conn:
    conn.execute(
        text("""
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            filename VARCHAR(255) NOT NULL,
            file_size INTEGER,
            chunking_strategy VARCHAR(50),
            total_chunks INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    )

    conn.execute(
        text("""
        CREATE TABLE IF NOT EXISTS chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER,
            content TEXT,
            token_count INTEGER,
            vector_id VARCHAR(255),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    )

    conn.execute(
        text("""
        CREATE TABLE IF NOT EXISTS bookings (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID,
            name VARCHAR(100),
            email VARCHAR(255),
            booking_date DATE,
            booking_time TIME,
            status VARCHAR(50) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    )

    conn.commit()
    print(" Database tables created successfully!")
