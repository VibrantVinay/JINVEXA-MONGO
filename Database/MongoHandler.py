# D:\Jinvexa\Database\MongoHandler.py

import logging
from typing import Dict, Any, List, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
from Config.Config import Config  # Import your centralized Config class

logger = logging.getLogger(__name__)


class MongoHandler:
    """
    Asynchronous MongoDB Handler powered by centralized Config.py.
    Manages connections and CRUD operations for Profiles, Plans, Graphs, and Parsed Data.
    """
    def __init__(self):
        self.config = Config()
        self.uri = self.config.get_mongo_uri()
        self.db_name = self.config.get_mongo_db_name()
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self._connect()

    def _connect(self):
        """Establish connection to MongoDB cluster."""
        try:
            self.client = AsyncIOMotorClient(self.uri)
            self.db = self.client[self.db_name]
            logger.info(f"Connected successfully to MongoDB database: {self.db_name}")
        except ConnectionFailure as e:
            logger.error(f"Could not connect to MongoDB: {e}")
            raise

    # ==================== USER PROFILES ====================
    async def save_user_profile(self, user_id: str, profile_data: Dict[str, Any]):
        """Upsert a UserProfile document."""
        collection = self.db.user_profiles
        await collection.update_one(
            {"user_id": user_id},
            {"$set": profile_data},
            upsert=True
        )
        logger.info(f"Saved UserProfile to MongoDB: {user_id}")

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a UserProfile document."""
        collection = self.db.user_profiles
        return await collection.find_one({"user_id": user_id}, {"_id": 0})

    # ==================== LEARNING PLANS ====================
    async def save_learning_plan(self, session_id: str, plan_data: Dict[str, Any]):
        """Save or update a LearningPlan."""
        collection = self.db.learning_plans
        plan_data["session_id"] = session_id
        await collection.update_one(
            {"session_id": session_id},
            {"$set": plan_data},
            upsert=True
        )
        logger.info(f"Saved LearningPlan to MongoDB for session: {session_id}")

    async def get_learning_plan(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a LearningPlan by session_id."""
        collection = self.db.learning_plans
        return await collection.find_one({"session_id": session_id}, {"_id": 0})

    # ==================== KNOWLEDGE GRAPHS ====================
    async def save_knowledge_graph(self, graph_id: str, graph_data: Dict[str, Any]):
        """Save a serialized KnowledgeGraph."""
        collection = self.db.knowledge_graphs
        graph_data["graph_id"] = graph_id
        await collection.update_one(
            {"graph_id": graph_id},
            {"$set": graph_data},
            upsert=True
        )
        logger.info(f"Saved KnowledgeGraph to MongoDB: {graph_id}")

    async def get_knowledge_graph(self, graph_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a KnowledgeGraph by ID."""
        collection = self.db.knowledge_graphs
        return await collection.find_one({"graph_id": graph_id}, {"_id": 0})

    # ==================== PARSED DATA / EXTRACTS ====================
    async def save_parsed_source(self, source_id: str, source_type: str, parsed_data: Dict[str, Any]):
        """
        Save parsed text and metadata from YouTubeTranscript, WebsiteParser, or DocumentParser.
        """
        collection = self.db.parsed_sources
        document = {
            "source_id": source_id,
            "source_type": source_type,  # e.g., 'youtube', 'pdf', 'website'
            "data": parsed_data
        }
        await collection.update_one(
            {"source_id": source_id},
            {"$set": document},
            upsert=True
        )
        logger.info(f"Cached parsed source in MongoDB: {source_id} ({source_type})")

    async def get_parsed_source(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve previously parsed source data to avoid re-extracting."""
        collection = self.db.parsed_sources
        return await collection.find_one({"source_id": source_id}, {"_id": 0})

    # ==================== ASSIGNMENTS & RESULTS ====================
    async def save_assignment(self, assignment_data: Dict[str, Any]):
        """Save a generated assignment."""
        collection = self.db.assignments
        assignment_id = assignment_data.get("assignment_id")
        await collection.update_one(
            {"assignment_id": assignment_id},
            {"$set": assignment_data},
            upsert=True
        )
        logger.info(f"Saved assignment to MongoDB: {assignment_id}")

    async def get_assignment(self, assignment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an assignment by ID."""
        collection = self.db.assignments
        return await collection.find_one({"assignment_id": assignment_id}, {"_id": 0})

    async def save_assignment_result(self, user_id: str, assignment_id: str, result_data: Dict[str, Any]):
        """Save an evaluated assignment result."""
        collection = self.db.assignment_results
        result_data["user_id"] = user_id
        result_data["assignment_id"] = assignment_id
        await collection.update_one(
            {"user_id": user_id, "assignment_id": assignment_id},
            {"$set": result_data},
            upsert=True
        )
        logger.info(f"Saved assignment result to MongoDB for user {user_id}")

    async def get_user_results(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all evaluated assignment results for a specific user."""
        collection = self.db.assignment_results
        cursor = collection.find({"user_id": user_id}, {"_id": 0})
        return await cursor.to_list(length=None)

    async def save_progress_summary(self, user_id: str, progress_data: Dict[str, Any]):
        """Save aggregated progress summary for a user."""
        collection = self.db.progress_summaries
        await collection.update_one(
            {"user_id": user_id},
            {"$set": progress_data},
            upsert=True
        )
        logger.info(f"Saved progress summary to MongoDB for user {user_id}")
