from sqlalchemy import create_engine

def test_connection():
    database_url = "postgresql://postgres:password@postgres_db:5432/mydatabase"
    engine = create_engine(database_url)

    try:
        with engine.connect() as connection:
            result = connection.execute("SELECT 1")
            print("Database connection successful:", result.fetchone())
    except Exception as e:
        print("Database connection failed:", e)

if __name__ == "__main__":
    test_connection()
