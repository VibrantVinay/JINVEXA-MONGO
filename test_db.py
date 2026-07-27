# D:\Jinvexa\test_db.py
from Agents.MemoryHandler import MemoryHandler
from Models.UserProfile import UserProfile

def test_connection():
    print("🔄 Initializing MemoryHandler and checking storage backend...")
    try:
        # Initialize handler (it will automatically read STORAGE_TYPE from .env)
        memory = MemoryHandler(storage_dir="memory_storage")
        
        print(f"📊 Active Storage Type: {memory.storage_type.upper()}")
        
        # Create a dummy user profile for testing
        test_profile = UserProfile(
            user_id="test_user_atlas_001",
            goals=["Verify MongoDB Atlas connection", "Master AI"],
            preferred_depth="professional"
        )
        
        print("🔄 Attempting to write test profile to database...")
        memory.save_profile(test_profile)
        
        print("🔄 Attempting to read test profile back from database...")
        loaded_profile = memory.load_profile("test_user_atlas_001")
        
        if loaded_profile and loaded_profile.user_id == "test_user_atlas_001":
            print("✅ SUCCESS! Connected to Atlas, wrote a profile, and successfully read it back!")
        else:
            print("⚠️ Profile write succeeded, but could not read it back.")
            
    except Exception as e:
        print(f"❌ Connection or Test Failed: {e}")

if __name__ == "__main__":
    test_connection()
