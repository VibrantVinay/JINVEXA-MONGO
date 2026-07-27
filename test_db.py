# D:\Jinvexa\test_db.py
import asyncio
from Database.MongoHandler import MongoHandler

async def test_connection():
    print("🔄 Attempting to connect to MongoDB Atlas...")
    try:
        db = MongoHandler()
        # Perform a test write
        await db.save_user_profile("test_user_001", {"status": "Atlas is working!", "ai": "Jinvexa"})
        print("✅ SUCCESS! Connected to Atlas and wrote a test profile.")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())