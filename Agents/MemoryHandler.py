# D:\Jinvexa\Agents\MemoryHandler.py

import json
import os
import sqlite3
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import hashlib
from dataclasses import dataclass, asdict, field

from Models.UserProfile import UserProfile
from Models.LearningPlan import LearningPlan

# ==================== MONGODB DEPENDENCY CHECK ====================
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    import certifi
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False


@dataclass
class SessionMemory:
    """Represents a complete session memory"""
    session_id: str
    user_id: str
    mode: str  # "goal" or "reference"
    created_at: str
    last_accessed: str
    conversation_history: List[Dict]
    extracted_data: Optional[Dict] = None
    concepts: Optional[Dict] = None
    knowledge_graph: Optional[Dict] = None
    gap_analysis: Optional[Dict] = None
    learning_plan: Optional[Dict] = None
    user_profile: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "mode": self.mode,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "conversation_history": self.conversation_history,
            "extracted_data": self.extracted_data,
            "concepts": self.concepts,
            "knowledge_graph": self.knowledge_graph,
            "gap_analysis": self.gap_analysis,
            "learning_plan": self.learning_plan,
            "user_profile": self.user_profile,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SessionMemory":
        return cls(**data)


class MemoryHandler:
    """
    Handles persistent memory storage for the Jinvexa Learning Platform.
    Supports both local JSON file storage and MongoDB Atlas cloud storage.
    """
    
    def __init__(self, storage_dir: str = "memory_storage", storage_type: str = None):
        """
        Initialize the Memory Handler
        
        Args:
            storage_dir: Directory for storing local memory files
            storage_type: "json" or "mongodb" (defaults to environment variable STORAGE_TYPE)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_type = (storage_type or os.getenv("STORAGE_TYPE", "json")).lower()
        
        # Create storage directories for local fallback or file-based outputs
        self.storage_dir.mkdir(exist_ok=True)
        (self.storage_dir / "sessions").mkdir(exist_ok=True)
        (self.storage_dir / "profiles").mkdir(exist_ok=True)
        
        # In-memory cache for fast access
        self._cache: Dict[str, SessionMemory] = {}
        self._profile_cache: Dict[str, UserProfile] = {}
        
        # Initialize MongoDB if selected
        self.mongo_client = None
        self.db = None
        if self.storage_type == "mongodb":
            if not MONGODB_AVAILABLE:
                print("⚠️ MongoDB libraries (pymongo/certifi) not installed. Falling back to JSON storage.")
                self.storage_type = "json"
            else:
                self._init_mongodb()
                
        print(f"💾 MemoryHandler initialized using backend: [{self.storage_type.upper()}]")

    def _init_mongodb(self):
        """Establish connection to MongoDB cluster with SSL handshake protections."""
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_DB_NAME", "jinvexa_db")
        try:
            # Using relaxed SSL parameters to prevent Windows handshake/TLS errors
            self.mongo_client = MongoClient(
                uri,
                tls=True if "mongodb+srv://" in uri else False,
                tlsCAFile=certifi.where() if "mongodb+srv://" in uri else None,
                tlsAllowInvalidCertificates=True,
                tlsAllowInvalidHostnames=True,
                serverSelectionTimeoutMS=5000
            )
            self.db = self.mongo_client[db_name]
            # Verify connection
            self.mongo_client.admin.command('ping')
            print(f"✅ Successfully connected to MongoDB Atlas cluster: {db_name}")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}. Falling back to JSON storage.")
            self.storage_type = "json"

    # ==================== SESSION MANAGEMENT ====================
    
    def create_session(
        self,
        user_id: str,
        mode: str,
        user_profile: Optional[UserProfile] = None
    ) -> str:
        """Create a new session and save it to the active backend."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hash_suffix = hashlib.md5(f"{user_id}{datetime.now()}".encode()).hexdigest()[:8]
        session_id = f"{user_id}_{timestamp}_{hash_suffix}"
        
        session = SessionMemory(
            session_id=session_id,
            user_id=user_id,
            mode=mode,
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            conversation_history=[],
            user_profile=user_profile.to_dict() if user_profile else None
        )
        
        self._cache[session_id] = session
        self._save_session(session)
        print(f"✅ Session created: {session_id[:20]}...")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionMemory]:
        """Get a session by ID."""
        if session_id in self._cache:
            return self._cache[session_id]
        
        session = self._load_session(session_id)
        if session:
            self._cache[session_id] = session
        return session
    
    def update_session(self, session: SessionMemory):
        """Update an existing session."""
        session.last_accessed = datetime.now().isoformat()
        self._cache[session.session_id] = session
        self._save_session(session)
    
    def delete_session(self, session_id: str):
        """Delete a session."""
        if session_id in self._cache:
            del self._cache[session_id]
        self._delete_session(session_id)
    
    def get_user_sessions(self, user_id: str, limit: int = 10) -> List[SessionMemory]:
        """Get all sessions for a user."""
        sessions = []
        
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                cursor = self.db.sessions.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1).limit(limit)
                for doc in cursor:
                    parsed_data = self._parse_session_data(doc)
                    sessions.append(SessionMemory.from_dict(parsed_data))
                return sessions
            except Exception as e:
                print(f"⚠️ MongoDB get_user_sessions error: {e}")
                return []
        
        # Fallback for JSON File Storage
        session_dir = self.storage_dir / "sessions"
        if session_dir.exists():
            for file_path in session_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if data.get("user_id") == user_id:
                        parsed_data = self._parse_session_data(data)
                        sessions.append(SessionMemory.from_dict(parsed_data))
                except Exception as e:
                    print(f"⚠️ Error loading session {file_path}: {e}")
                    continue
        
        sessions.sort(key=lambda x: x.created_at, reverse=True)
        return sessions[:limit]
    
    def _parse_session_data(self, data: Dict) -> Dict:
        """Parse session data, converting JSON strings back to objects if needed."""
        parsed = data.copy()
        json_keys = ["conversation_history", "extracted_data", "concepts", 
                     "knowledge_graph", "gap_analysis", "learning_plan", 
                     "user_profile", "metadata"]
        
        for key in json_keys:
            if key in parsed and parsed[key] is not None and isinstance(parsed[key], str):
                try:
                    parsed[key] = json.loads(parsed[key])
                except:
                    parsed[key] = None
            elif key in parsed and parsed[key] is None:
                parsed[key] = None
        return parsed
    
    def add_conversation_message(self, session_id: str, role: str, message: str):
        """Add a message to session conversation history."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session.conversation_history.append({
            "role": role,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        self.update_session(session)
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session."""
        session = self.get_session(session_id)
        return session.conversation_history if session else []
    
    def clear_conversation_history(self, session_id: str):
        """Clear conversation history for a session."""
        session = self.get_session(session_id)
        if session:
            session.conversation_history = []
            self.update_session(session)
    
    # ==================== PROFILE MANAGEMENT ====================
    
    def save_profile(self, profile: UserProfile):
        """Save a user profile."""
        self._profile_cache[profile.user_id] = profile
        self._save_profile(profile)
    
    def load_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load a user profile."""
        if user_id in self._profile_cache:
            return self._profile_cache[user_id]
        
        profile = self._load_profile(user_id)
        if profile:
            self._profile_cache[user_id] = profile
        return profile
    
    def delete_profile(self, user_id: str):
        """Delete a user profile."""
        if user_id in self._profile_cache:
            del self._profile_cache[user_id]
        self._delete_profile(user_id)
    
    def list_profiles(self) -> List[str]:
        """List all user IDs with profiles."""
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                return [doc["user_id"] for doc in self.db.profiles.find({}, {"user_id": 1, "_id": 0})]
            except Exception as e:
                print(f"⚠️ MongoDB list_profiles error: {e}")
                return []
                
        profile_dir = self.storage_dir / "profiles"
        if profile_dir.exists():
            return [f.stem for f in profile_dir.glob("*.json")]
        return []

    # ==================== PARSED SOURCE CACHING ====================

    def save_parsed_source(self, source_id: str, source_type: str, parsed_data: Dict[str, Any]):
        """Save parsed text from YouTubeTranscript, WebsiteParser, or DocumentParser."""
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                document = {
                    "source_id": source_id,
                    "source_type": source_type,
                    "data": parsed_data,
                    "updated_at": datetime.now().isoformat()
                }
                self.db.parsed_sources.update_one(
                    {"source_id": source_id},
                    {"$set": document},
                    upsert=True
                )
                print(f"💾 Cached parsed source in MongoDB: {source_id[:30]}... ({source_type})")
                return
            except Exception as e:
                print(f"⚠️ MongoDB save_parsed_source error: {e}")
                
        # Local fallback cache for parsed sources
        cache_dir = self.storage_dir / "parsed_sources"
        cache_dir.mkdir(exist_ok=True)
        safe_name = hashlib.md5(source_id.encode()).hexdigest() + ".json"
        with open(cache_dir / safe_name, 'w', encoding='utf-8') as f:
            json.dump({"source_id": source_id, "source_type": source_type, "data": parsed_data}, f, indent=2)

    def get_parsed_source(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve previously parsed source data to avoid re-extracting."""
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                doc = self.db.parsed_sources.find_one({"source_id": source_id}, {"_id": 0})
                return doc["data"] if doc else None
            except Exception as e:
                print(f"⚠️ MongoDB get_parsed_source error: {e}")
                return None
                
        cache_dir = self.storage_dir / "parsed_sources"
        safe_name = hashlib.md5(source_id.encode()).hexdigest() + ".json"
        cache_file = cache_dir / safe_name
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f).get("data")
            except:
                pass
        return None
    
    # ==================== STATISTICS AND METRICS ====================
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get learning statistics for a user."""
        sessions = self.get_user_sessions(user_id, limit=100)
        stats = {
            "user_id": user_id,
            "total_sessions": len(sessions),
            "active_sessions": len([s for s in sessions if s.learning_plan]),
            "total_conversations": sum(len(s.conversation_history) for s in sessions),
            "learning_plans": [],
            "knowledge_progress": {}
        }
        
        for session in sessions:
            if session.learning_plan:
                plan = session.learning_plan
                stats["learning_plans"].append({
                    "topic": plan.get("main_topic", "Unknown"),
                    "goal": plan.get("goal", "Unknown"),
                    "estimated_hours": plan.get("estimated_time_hours", 0)
                })
        
        profile = self.load_profile(user_id)
        if profile:
            stats["knowledge_progress"] = {
                "known_concepts": len(profile.known_concepts),
                "topics": list(profile.known_concepts.keys())[:20]
            }
        return stats
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics across all users."""
        profiles = self.list_profiles()
        stats = {
            "total_users": len(profiles),
            "total_sessions": 0,
            "total_conversations": 0,
            "learning_plans_generated": 0
        }
        
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                stats["total_sessions"] = self.db.sessions.count_documents({})
                stats["total_conversations"] = stats["total_sessions"] * 5
                return stats
            except Exception as e:
                pass
                
        session_dir = self.storage_dir / "sessions"
        if session_dir.exists():
            stats["total_sessions"] = len(list(session_dir.glob("*.json")))
            stats["total_conversations"] = stats["total_sessions"] * 5
        return stats
    
    # ==================== STORAGE BACKEND METHODS ====================
    
    def _save_session(self, session: SessionMemory):
        """Save session to active storage backend."""
        data = session.to_dict()
        
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                self.db.sessions.update_one(
                    {"session_id": session.session_id},
                    {"$set": data},
                    upsert=True
                )
                return
            except Exception as e:
                print(f"⚠️ MongoDB save_session failed: {e}. Falling back to file.")
        
        # Fallback JSON storage
        for key in ["conversation_history", "extracted_data", "concepts", 
                    "knowledge_graph", "gap_analysis", "learning_plan", 
                    "user_profile", "metadata"]:
            if data.get(key) is not None:
                try:
                    data[key] = json.dumps(data[key], ensure_ascii=False)
                except:
                    data[key] = None
                    
        session_file = self.storage_dir / "sessions" / f"{session.session_id}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_session(self, session_id: str) -> Optional[SessionMemory]:
        """Load session from active storage backend."""
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                data = self.db.sessions.find_one({"session_id": session_id}, {"_id": 0})
                if data:
                    data = self._parse_session_data(data)
                    return SessionMemory.from_dict(data)
            except Exception as e:
                print(f"⚠️ MongoDB load_session failed: {e}")
                
        session_file = self.storage_dir / "sessions" / f"{session_id}.json"
        if session_file.exists():
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data = self._parse_session_data(data)
                return SessionMemory.from_dict(data)
            except Exception as e:
                print(f"⚠️ Error loading session {session_id}: {e}")
        return None
    
    def _delete_session(self, session_id: str):
        """Delete session from active storage backend."""
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                self.db.sessions.delete_one({"session_id": session_id})
            except Exception as e:
                pass
        session_file = self.storage_dir / "sessions" / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
    
    def _save_profile(self, profile: UserProfile):
        """Save profile to active storage backend."""
        data = {
            "user_id": profile.user_id,
            "profile_data": profile.to_dict(),
            "created_at": profile.created_at,
            "updated_at": profile.updated_at
        }
        
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                self.db.profiles.update_one(
                    {"user_id": profile.user_id},
                    {"$set": data},
                    upsert=True
                )
                return
            except Exception as e:
                print(f"⚠️ MongoDB save_profile failed: {e}. Falling back to file.")
                
        profile_file = self.storage_dir / "profiles" / f"{profile.user_id}.json"
        with open(profile_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load profile from active storage backend."""
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                data = self.db.profiles.find_one({"user_id": user_id}, {"_id": 0})
                if data:
                    return UserProfile.from_dict(data.get("profile_data", {}))
            except Exception as e:
                print(f"⚠️ MongoDB load_profile failed: {e}")
                
        profile_file = self.storage_dir / "profiles" / f"{user_id}.json"
        if profile_file.exists():
            try:
                with open(profile_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return UserProfile.from_dict(data.get("profile_data", {}))
            except Exception as e:
                print(f"⚠️ Error loading profile {user_id}: {e}")
        return None
    
    def _delete_profile(self, user_id: str):
        """Delete profile from active storage backend."""
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                self.db.profiles.delete_one({"user_id": user_id})
            except Exception as e:
                pass
        profile_file = self.storage_dir / "profiles" / f"{user_id}.json"
        if profile_file.exists():
            profile_file.unlink()
    
    # ==================== LESSON MANAGEMENT (TeachingAgent) ====================

    def save_lesson(self, session_id: str, topic: str, phase_title: str, lesson_text: str) -> str:
        """Save a lesson text file and record metadata in MongoDB."""
        safe_topic = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')
        safe_topic = re.sub(r'[-\s]+', '_', safe_topic)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{session_id}_{timestamp}_{safe_topic}.txt"
        filepath = Path("learn_files/lessons") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        content_string = f"Session ID: {session_id}\nTopic: {topic}\nPhase: {phase_title}\nGenerated: {datetime.now().isoformat()}\n{'='*60}\n\n{lesson_text}"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content_string)
            
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                self.db.lessons.update_one(
                    {"file_path": str(filepath)},
                    {"$set": {"session_id": session_id, "topic": topic, "phase": phase_title, "content": content_string, "generated_at": datetime.now().isoformat()}},
                    upsert=True
                )
            except Exception:
                pass
        return str(filepath)

    def save_manifest(self, session_id: str, manifest: list, main_topic: str):
        """Save manifest JSON/TXT and sync watch order to MongoDB."""
        manifest_dir = Path("learn_files/manifests")
        manifest_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_data = {
            "session_id": session_id,
            "main_topic": main_topic,
            "generated_at": datetime.now().isoformat(),
            "total_lessons": len(manifest),
            "watch_order": manifest
        }
        
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                self.db.manifests.update_one({"session_id": session_id}, {"$set": manifest_data}, upsert=True)
            except Exception:
                pass
                
        manifest_file = manifest_dir / f"{session_id}_manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)
        
        readable_file = manifest_dir / f"{session_id}_watch_order.txt"
        with open(readable_file, 'w', encoding='utf-8') as f:
            f.write(f"Course: {main_topic}\nSession: {session_id}\nGenerated: {datetime.now().isoformat()}\n{'='*60}\n\n📚 WATCH ORDER\n\n")
            for item in manifest:
                order = item.get("order", 0)
                phase = item.get("phase", "")
                topic = item.get("topic", "")
                content_type = item.get("content_type", "text")
                gender = item.get("gender", "")
                icon = f"🎧 ({gender})" if content_type == "audio" else "📄"
                f.write(f"{order}. {icon} {topic}\n   Phase: {phase}\n   Type: {content_type.upper()}\n\n")

    def save_audio_metadata(self, session_id: str, topic: str, audio_file: str, gender: str):
        """Save audio metadata locally and to MongoDB."""
        audio_dir = Path("learn_files/audio")
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_metadata_file = audio_dir / f"{session_id}_audio_metadata.json"
        
        existing = {}
        if audio_metadata_file.exists():
            try:
                with open(audio_metadata_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except:
                pass
        
        if session_id not in existing:
            existing[session_id] = []
        
        entry = {"topic": topic, "generated_at": datetime.now().isoformat(), "gender": gender, "file": audio_file}
        existing[session_id].append(entry)
        
        with open(audio_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
            
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                self.db.audio_metadata.update_one({"session_id": session_id}, {"$push": {"audio_files": entry}}, upsert=True)
            except Exception:
                pass

    def save_course_metadata(self, session_id: str, course_content: dict):
        """Save course metadata locally and to MongoDB."""
        lessons_dir = Path("learn_files/lessons")
        lessons_dir.mkdir(parents=True, exist_ok=True)
        course_metadata_file = lessons_dir / f"{session_id}_course_metadata.json"
        with open(course_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(course_content, f, indent=2, ensure_ascii=False)
            
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                self.db.course_metadata.update_one({"session_id": session_id}, {"$set": course_content}, upsert=True)
            except Exception:
                pass

    def get_lesson_content(self, lesson_file: str) -> str:
        """Read lesson content from file or MongoDB."""
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                doc = self.db.lessons.find_one({"file_path": str(lesson_file)}, {"_id": 0})
                if doc and "content" in doc:
                    return doc["content"]
            except Exception:
                pass
        try:
            filepath = Path(lesson_file)
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
        except:
            pass
        return ""

    def get_manifest(self, session_id: str) -> dict:
        """Get manifest data for a session from MongoDB or local file."""
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                doc = self.db.manifests.find_one({"session_id": session_id}, {"_id": 0})
                if doc:
                    return doc
            except Exception:
                pass
        manifest_file = Path(f"learn_files/manifests/{session_id}_manifest.json")
        if manifest_file.exists():
            try:
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def list_sessions_with_plans(self) -> list:
        """List all sessions that have learning plans."""
        sessions = []
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                cursor = self.db.sessions.find({"learning_plan": {"$ne": None}}, {"_id": 0})
                for data in cursor:
                    lp = data.get('learning_plan')
                    if isinstance(lp, str):
                        try:
                            lp = json.loads(lp)
                        except:
                            continue
                    sessions.append({
                        "session_id": data.get('session_id', ''),
                        "user_id": data.get('user_id', ''),
                        "mode": data.get('mode', ''),
                        "created_at": data.get('created_at', ''),
                        "main_topic": lp.get('main_topic', 'Unknown'),
                        "goal": lp.get('goal', ''),
                        "total_hours": lp.get('estimated_time_hours', 0),
                        "phase_count": len(lp.get('roadmap', [])),
                        "learning_plan": lp
                    })
                return sessions
            except Exception:
                pass
                
        session_dir = self.storage_dir / "sessions"
        if session_dir.exists():
            for file_path in session_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    learning_plan = data.get('learning_plan')
                    if learning_plan:
                        if isinstance(learning_plan, str):
                            try:
                                learning_plan = json.loads(learning_plan)
                            except:
                                pass
                        sessions.append({
                            "session_id": data.get('session_id', ''),
                            "user_id": data.get('user_id', ''),
                            "mode": data.get('mode', ''),
                            "created_at": data.get('created_at', ''),
                            "main_topic": learning_plan.get('main_topic', 'Unknown'),
                            "goal": learning_plan.get('goal', ''),
                            "total_hours": learning_plan.get('estimated_time_hours', 0),
                            "phase_count": len(learning_plan.get('roadmap', [])),
                            "learning_plan": learning_plan
                        })
                except Exception:
                    continue
        return sessions

    def list_generated_lessons(self, session_id: str) -> list:
        """List generated lessons for a session."""
        lessons = []
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                cursor = self.db.lessons.find({"session_id": session_id}, {"_id": 0})
                for doc in cursor:
                    lessons.append({"file": doc.get("file_path", ""), "topic": doc.get("topic", "Unknown"), "phase": doc.get("phase", "Unknown"), "generated_at": doc.get("generated_at", "Unknown")})
                if lessons:
                    return lessons
            except Exception:
                pass
                
        lessons_dir = Path("learn_files/lessons")
        if lessons_dir.exists():
            for file_path in lessons_dir.glob(f"{session_id}*.txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()[:10]
                    metadata = {}
                    for line in lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            metadata[key.strip()] = value.strip()
                    lessons.append({"file": str(file_path), "topic": metadata.get('Topic', 'Unknown'), "phase": metadata.get('Phase', 'Unknown'), "generated_at": metadata.get('Generated', 'Unknown')})
                except:
                    pass
        return lessons

    # ==================== ASSIGNMENT MANAGEMENT ====================

    def save_assignment(self, assignment: dict) -> str:
        """Save assignment to file and MongoDB."""
        session_id = assignment.get("session_id", "unknown")
        assignment_id = assignment.get("assignment_id", "unknown")
        session_dir = Path("learn_files/assignments/sessions") / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        assignment_file = session_dir / f"{assignment_id}.json"
        with open(assignment_file, 'w', encoding='utf-8') as f:
            json.dump(assignment, f, indent=2, ensure_ascii=False)
            
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                self.db.assignments.update_one({"assignment_id": assignment_id}, {"$set": assignment}, upsert=True)
            except Exception:
                pass
        return str(assignment_file)

    def get_assignment(self, assignment_id: str) -> dict:
        """Get assignment by ID."""
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                doc = self.db.assignments.find_one({"assignment_id": assignment_id}, {"_id": 0})
                if doc:
                    return doc
            except Exception:
                pass
                
        sessions_dir = Path("learn_files/assignments/sessions")
        if sessions_dir.exists():
            for session_dir in sessions_dir.iterdir():
                if session_dir.is_dir():
                    assignment_file = session_dir / f"{assignment_id}.json"
                    if assignment_file.exists():
                        with open(assignment_file, 'r', encoding='utf-8') as f:
                            return json.load(f)
        return None

    def list_assignments(self, session_id: str) -> list:
        """List all assignments for a session."""
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                cursor = self.db.assignments.find({"session_id": session_id}, {"_id": 0}).sort("generated_at", -1)
                return [{"assignment_id": doc.get("assignment_id", ""), "generated_at": doc.get("generated_at", ""), "total_questions": doc.get("total_questions", 0), "difficulty": doc.get("difficulty", "intermediate")} for doc in cursor]
            except Exception:
                pass
                
        assignments = []
        session_dir = Path("learn_files/assignments/sessions") / session_id
        if session_dir.exists():
            for file_path in session_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    assignments.append({"assignment_id": data.get("assignment_id", ""), "generated_at": data.get("generated_at", ""), "total_questions": data.get("total_questions", 0), "difficulty": data.get("difficulty", "intermediate")})
                except:
                    pass
        return sorted(assignments, key=lambda x: x.get("generated_at", ""), reverse=True)

    def save_assignment_result(self, user_id: str, assignment_id: str, result: dict) -> str:
        """Save evaluation result and update profile."""
        user_dir = Path("learn_files/assignments/results") / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        result_file = user_dir / f"{assignment_id}_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                result["user_id"] = user_id
                result["assignment_id"] = assignment_id
                self.db.assignment_results.update_one({"user_id": user_id, "assignment_id": assignment_id}, {"$set": result}, upsert=True)
            except Exception:
                pass
                
        profile = self.load_profile(user_id)
        if profile:
            profile.assignments.append({
                "assignment_id": assignment_id,
                "score": result.get("scores", {}).get("total", {}).get("percentage", 0),
                "grade": result.get("scores", {}).get("total", {}).get("grade", "N/A"),
                "date": datetime.now().isoformat()
            })
            self.save_profile(profile)
        return str(result_file)

    def get_user_results(self, user_id: str) -> list:
        """Get all results for a user."""
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                return list(self.db.assignment_results.find({"user_id": user_id}, {"_id": 0}))
            except Exception:
                pass
                
        results = []
        user_dir = Path("learn_files/assignments/results") / user_id
        if user_dir.exists():
            for file_path in sorted(user_dir.glob("*.json"), reverse=True):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        results.append(json.load(f))
                except:
                    pass
        return results

    def save_progress_summary(self, user_id: str, progress: dict) -> str:
        """Save progress summary."""
        progress_file = Path("learn_files/assignments/progress") / f"{user_id}_progress.json"
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
            
        if self.storage_type == "mongodb" and self.db is not None:
            try:
                self.db.progress_summaries.update_one({"user_id": user_id}, {"$set": progress}, upsert=True)
            except Exception:
                pass
        return str(progress_file)

    # ==================== MENTORING (SQLite) MANAGEMENT ====================

    def init_mentoring_db(self):
        """Initialize mentoring SQLite database."""
        db_path = Path("learn_files/mentoring/mentoring_memory.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.mentoring_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.mentoring_cursor = self.mentoring_conn.cursor()
        
        self.mentoring_cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_count INTEGER DEFAULT 0,
                token_count INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 1
            )
        """)
        self.mentoring_cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                token_count INTEGER DEFAULT 0,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        self.mentoring_cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_content_cache (
                session_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                topics TEXT,
                phases TEXT,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.mentoring_cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)")
        self.mentoring_cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id)")
        self.mentoring_cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)")
        self.mentoring_cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
        self.mentoring_conn.commit()

    def _ensure_mentoring_db(self):
        """Ensure mentoring DB is initialized."""
        if not hasattr(self, 'mentoring_conn'):
            self.init_mentoring_db()

    def create_mentoring_conversation(self, user_id: str, session_id: str = None, mode: str = "session") -> str:
        """Create a new conversation."""
        self._ensure_mentoring_db()
        if mode == "full" and session_id is None:
            session_id = f"all_sessions_{user_id}"
            
        if session_id:
            self.mentoring_cursor.execute("""
                SELECT id FROM conversations 
                WHERE user_id = ? AND session_id = ? AND is_active = 1
                ORDER BY last_accessed DESC LIMIT 1
            """, (user_id, session_id))
            result = self.mentoring_cursor.fetchone()
            if result:
                return str(result[0])
                
        self.mentoring_cursor.execute("""
            INSERT INTO conversations (session_id, user_id, mode, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, user_id, mode, datetime.now().isoformat(), datetime.now().isoformat()))
        self.mentoring_conn.commit()
        return str(self.mentoring_cursor.lastrowid)

    def add_mentoring_message(self, conversation_id: str, role: str, content: str):
        """Add a message to a conversation."""
        self._ensure_mentoring_db()
        token_count = len(content) // 4
        self.mentoring_cursor.execute("""
            INSERT INTO messages (conversation_id, role, content, timestamp, token_count)
            VALUES (?, ?, ?, ?, ?)
        """, (conversation_id, role, content, datetime.now().isoformat(), token_count))
        self.mentoring_cursor.execute("""
            UPDATE conversations 
            SET last_accessed = ?, message_count = message_count + 1, token_count = token_count + ?
            WHERE id = ?
        """, (datetime.now().isoformat(), token_count, conversation_id))
        self.mentoring_conn.commit()

    def get_mentoring_conversation_history(self, conversation_id: str, limit: int = 20) -> list:
        """Get conversation history."""
        self._ensure_mentoring_db()
        self.mentoring_cursor.execute("""
            SELECT role, content, timestamp FROM messages 
            WHERE conversation_id = ? 
            ORDER BY timestamp DESC LIMIT ?
        """, (conversation_id, limit))
        rows = self.mentoring_cursor.fetchall()
        return [{"role": row[0], "content": row[1], "timestamp": row[2]} for row in rows[::-1]]

    def get_mentoring_conversation_info(self, conversation_id: str) -> dict:
        """Get conversation metadata."""
        self._ensure_mentoring_db()
        self.mentoring_cursor.execute("""
            SELECT id, session_id, user_id, mode, created_at, last_accessed, message_count, token_count
            FROM conversations WHERE id = ?
        """, (conversation_id,))
        row = self.mentoring_cursor.fetchone()
        if row:
            return {"id": row[0], "session_id": row[1], "user_id": row[2], "mode": row[3], "created_at": row[4], "last_accessed": row[5], "message_count": row[6], "token_count": row[7]}
        return {}

    def list_mentoring_conversations(self, user_id: str) -> list:
        """List all conversations for a user."""
        self._ensure_mentoring_db()
        self.mentoring_cursor.execute("""
            SELECT id, session_id, mode, created_at, last_accessed, message_count
            FROM conversations 
            WHERE user_id = ? AND is_active = 1
            ORDER BY last_accessed DESC
        """, (user_id,))
        rows = self.mentoring_cursor.fetchall()
        conversations = []
        for row in rows:
            session_id = row[1]
            topic = "Unknown"
            if session_id:
                manifest_file = Path(f"learn_files/manifests/{session_id}_manifest.json")
                if manifest_file.exists():
                    try:
                        with open(manifest_file, 'r', encoding='utf-8') as f:
                            topic = json.load(f).get("main_topic", "Unknown")
                    except:
                        pass
            conversations.append({"id": row[0], "session_id": row[1], "mode": row[2], "created_at": row[3], "last_accessed": row[4], "message_count": row[5], "topic": topic})
        return conversations

    def get_mentoring_session_content(self, session_id: str) -> dict:
        """Get cached session content or load from manifest."""
        self._ensure_mentoring_db()
        self.mentoring_cursor.execute("SELECT content, topics, phases FROM session_content_cache WHERE session_id = ?", (session_id,))
        result = self.mentoring_cursor.fetchone()
        if result:
            return {"content": result[0], "topics": json.loads(result[1]) if result[1] else [], "phases": json.loads(result[2]) if result[2] else []}
            
        manifest_file = Path(f"learn_files/manifests/{session_id}_manifest.json")
        if not manifest_file.exists():
            return {"content": "", "topics": [], "phases": []}
            
        try:
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            watch_order = manifest.get("watch_order", [])
            topics = [item.get("topic", "") for item in watch_order if item.get("topic")]
            phases = list(set([item.get("phase", "") for item in watch_order if item.get("phase")]))
            
            content_parts = []
            for item in watch_order[:10]:
                text_file = item.get("text_file", "")
                if text_file:
                    try:
                        filepath = Path(text_file)
                        if filepath.exists():
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content_parts.append(f"Topic: {item.get('topic', '')}")
                                content_parts.append(f.read()[:1000])
                    except:
                        pass
            full_content = "\n\n".join(content_parts)
            self.mentoring_cursor.execute("""
                INSERT OR REPLACE INTO session_content_cache 
                (session_id, content, topics, phases, cached_at)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, full_content[:50000], json.dumps(topics[:20]), json.dumps(phases[:10]), datetime.now().isoformat()))
            self.mentoring_conn.commit()
            return {"content": full_content[:50000], "topics": topics[:20], "phases": phases[:10]}
        except Exception as e:
            return {"content": "", "topics": [], "phases": []}

    def get_mentoring_all_user_content(self, user_id: str) -> dict:
        """Get all content from all sessions of a user."""
        all_topics, all_phases, all_content = [], [], []
        for session in self.get_user_sessions(user_id):
            content = self.get_mentoring_session_content(session.session_id)
            if content.get("content"):
                all_content.append(f"Session: {session.session_id[:20]}...")
                all_content.append(content["content"])
                all_topics.extend(content.get("topics", []))
                all_phases.extend(content.get("phases", []))
        return {"content": "\n\n".join(all_content)[:100000], "topics": list(set(all_topics))[:30], "phases": list(set(all_phases))[:15]}

    def manage_mentoring_history(self, conversation_id: str):
        """Manage conversation history by trimming old messages."""
        self._ensure_mentoring_db()
        try:
            self.mentoring_cursor.execute("SELECT id FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC", (conversation_id,))
            messages = self.mentoring_cursor.fetchall()
            if len(messages) > 20:
                keep_ids = [m[0] for m in messages[-15:]]
                self.mentoring_cursor.execute(f"DELETE FROM messages WHERE conversation_id = ? AND id NOT IN ({','.join('?'*len(keep_ids))})", [conversation_id] + keep_ids)
                self.mentoring_cursor.execute("UPDATE conversations SET message_count = ? WHERE id = ?", (len(keep_ids), conversation_id))
                self.mentoring_conn.commit()
        except Exception:
            pass

    def garbage_collect_mentoring(self, max_conversation_age_days: int = 7, max_messages_per_session: int = 50):
        """Clean up old conversations."""
        self._ensure_mentoring_db()
        try:
            cutoff_date = (datetime.now() - timedelta(days=max_conversation_age_days)).isoformat()
            self.mentoring_cursor.execute("DELETE FROM conversations WHERE last_accessed < ? AND is_active = 0", (cutoff_date,))
            self.mentoring_cursor.execute("SELECT id, message_count FROM conversations WHERE message_count > ?", (max_messages_per_session,))
            for conv_id, _ in self.mentoring_cursor.fetchall():
                keep_count = min(30, max_messages_per_session // 2)
                self.mentoring_cursor.execute("""
                    DELETE FROM messages WHERE conversation_id = ? AND id NOT IN (
                        SELECT id FROM messages WHERE conversation_id = ? ORDER BY timestamp DESC LIMIT ?
                    )
                """, (conv_id, conv_id, keep_count))
                self.mentoring_cursor.execute("UPDATE conversations SET message_count = ? WHERE id = ?", (keep_count, conv_id))
            self.mentoring_cursor.execute("DELETE FROM messages WHERE conversation_id NOT IN (SELECT id FROM conversations)")
            self.mentoring_conn.commit()
        except Exception:
            pass

    def get_mentoring_session_topic(self, session_id: str) -> str:
        """Get the topic of a session."""
        if not session_id:
            return "Unknown"
        manifest_file = Path(f"learn_files/manifests/{session_id}_manifest.json")
        if manifest_file.exists():
            try:
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    return json.load(f).get("main_topic", "Unknown")
            except:
                pass
        return "Unknown"

    def close_mentoring_db(self):
        """Close mentoring database connection."""
        if hasattr(self, 'mentoring_conn'):
            self.mentoring_conn.close()

    def export_data(self, user_id: str, export_dir: str = "exports") -> str:
        """Export all data for a user."""
        export_path = Path(export_dir) / user_id
        export_path.mkdir(parents=True, exist_ok=True)
        profile = self.load_profile(user_id)
        if profile:
            with open(export_path / "profile.json", 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
        sessions = self.get_user_sessions(user_id)
        with open(export_path / "sessions.json", 'w') as f:
            json.dump([s.to_dict() for s in sessions], f, indent=2)
        return str(export_path)
