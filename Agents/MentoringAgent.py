# D:\Jinvexa\Agents\MentoringAgent.py

import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import re
import asyncio
from Agents.BaseAgent import BaseAgent


class MentoringAgent(BaseAgent):
    """
    Intelligent mentoring chatbot with two modes:
    1. Session-specific mentoring
    2. Full context mentoring (all sessions)
    Uses MemoryHandler for all data operations (SQLite conversation memory).
    """

    def __init__(
        self,
        llm_client: Any,
        memory_handler: Any,
        config: Optional[Dict] = None
    ):
        super().__init__("MentoringAgent", llm_client)
        
        self.llm_client = llm_client
        self.memory = memory_handler
        self.config = config or {}
        
        # Configuration
        self.max_history_tokens = self.config.get("max_history_tokens", 4000)
        self.max_conversation_age_days = self.config.get("max_conversation_age_days", 7)
        self.max_messages_per_session = self.config.get("max_messages_per_session", 50)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and delegate to appropriate handler."""
        action = input_data.get("action", "chat")
        
        if action == "chat":
            return await self.chat(
                user_id=input_data.get("user_id", ""),
                message=input_data.get("message", ""),
                conversation_id=input_data.get("conversation_id"),
                session_id=input_data.get("session_id"),
                mode=input_data.get("mode", "session")
            )
        elif action == "list_conversations":
            conversations = self.memory.list_mentoring_conversations(input_data.get("user_id", ""))
            return {"conversations": conversations}
        elif action == "get_conversation_info":
            info = self.memory.get_mentoring_conversation_info(input_data.get("conversation_id", ""))
            return {"info": info}
        elif action == "get_session_topic":
            topic = self.memory.get_mentoring_session_topic(input_data.get("session_id", ""))
            return {"topic": topic}
        
        return {"error": f"Unknown action: {action}"}

    # ==================== SESSION CONTENT MANAGEMENT ====================

    def _get_session_content(self, session_id: str) -> Dict[str, Any]:
        """
        Get cached session content or load from manifest. Delegates to MemoryHandler.
        """
        return self.memory.get_mentoring_session_content(session_id)

    def _get_all_user_content(self, user_id: str) -> Dict[str, Any]:
        """
        Get all content from all sessions of a user. Delegates to MemoryHandler.
        """
        return self.memory.get_mentoring_all_user_content(user_id)

    # ==================== CONVERSATION MANAGEMENT ====================

    def create_conversation(self, user_id: str, session_id: str = None, mode: str = "session") -> str:
        """
        Create a new conversation session. Delegates to MemoryHandler.
        """
        return self.memory.create_mentoring_conversation(user_id, session_id, mode)

    def add_message(self, conversation_id: str, role: str, content: str):
        """
        Add a message to a conversation. Delegates to MemoryHandler.
        """
        self.memory.add_mentoring_message(conversation_id, role, content)
        
        # Check if we need history management
        conv_info = self.memory.get_mentoring_conversation_info(conversation_id)
        if conv_info:
            token_count_total = conv_info.get("token_count", 0)
            message_count = conv_info.get("message_count", 0)
            
            if token_count_total > self.max_history_tokens or message_count > self.max_messages_per_session:
                self.memory.manage_mentoring_history(conversation_id)

    def get_conversation_history(self, conversation_id: str, limit: int = 20) -> List[Dict]:
        """
        Get conversation history. Delegates to MemoryHandler.
        """
        return self.memory.get_mentoring_conversation_history(conversation_id, limit)

    def get_conversation_info(self, conversation_id: str) -> Dict:
        """
        Get conversation metadata. Delegates to MemoryHandler.
        """
        return self.memory.get_mentoring_conversation_info(conversation_id)

    def list_conversations(self, user_id: str) -> List[Dict]:
        """
        List all conversations for a user. Delegates to MemoryHandler.
        """
        return self.memory.list_mentoring_conversations(user_id)

    # ==================== CHAT FUNCTIONALITY ====================

    async def chat(
        self,
        user_id: str,
        message: str,
        conversation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        mode: str = "session"
    ) -> Dict[str, Any]:
        """
        Main chat function.
        Mode 1: session-specific
        Mode 2: full context (all sessions)
        """
        # Get or create conversation
        self.log_reasoning("Starting mentoring session...", f"Mode: {mode.upper()}, User: {user_id}", "thinking")
        if not conversation_id:
            conversation_id = self.create_conversation(user_id, session_id, mode)
        
        # Get conversation info
        conv_info = self.get_conversation_info(conversation_id)
        
        # Get relevant content based on mode
        self.log_reasoning("Loading context...", f"Loading {'all sessions' if mode == 'full' else 'session content'}", "thinking")
        if mode == "full":
            # Mode 2: All sessions
            context_data = self._get_all_user_content(user_id)
            context_label = "All your learning sessions"
        else:
            # Mode 1: Specific session
            if not session_id:
                return {"error": "Session ID required for session mode"}
            context_data = self._get_session_content(session_id)
            context_label = f"Session: {session_id[:20]}..."
        
        # Get conversation history
        history = self.get_conversation_history(conversation_id, limit=10)
        
        # Build system prompt
        system_prompt = self._build_system_prompt(
            mode=mode,
            context_label=context_label,
            context_data=context_data,
            history=history
        )
        
        # Build user prompt
        user_prompt = self._build_user_prompt(message, history)
        
        # Get response from LLM
        self.log_reasoning("Generating response...", "Using LLM to formulate helpful response", "thinking")
        try:
            response = await self.llm_client.complete(
                prompt=user_prompt,
                system_prompt=system_prompt
            )
        except Exception as e:
            response = f"I apologize, but I encountered an error: {e}. Please try again."

        # Add messages to history
        self.add_message(conversation_id, "user", message)
        self.add_message(conversation_id, "assistant", response)
        
        # Run garbage collection if needed
        self.memory.garbage_collect_mentoring(
            max_conversation_age_days=self.max_conversation_age_days,
            max_messages_per_session=self.max_messages_per_session
        )
        
        self.log_reasoning("Response generated", f"{len(response)} characters", "success")
        
        return {
            "response": response,
            "conversation_id": conversation_id,
            "message_count": conv_info.get("message_count", 0) + 2,
            "mode": mode
        }

    def _build_system_prompt(
        self,
        mode: str,
        context_label: str,
        context_data: Dict,
        history: List[Dict]
    ) -> str:
        """
        Build system prompt with context.
        """
        topics = context_data.get("topics", [])
        phases = context_data.get("phases", [])
        content = context_data.get("content", "")
        
        # Summarize content for the prompt
        content_summary = content[:3000] if content else "No specific content available."
        
        prompt = f"""You are Jinvexa Mentor, an intelligent and caring AI learning mentor for the Jinvexa Learning Platform.

Mode: {mode.upper()} - {context_label}

## Your Role
You are a supportive mentor who helps learners understand concepts, answers questions, provides guidance, and encourages learning. You are knowledgeable, patient, and explain things clearly.

## Course Context
Topics: {', '.join(topics[:10])}
Phases: {', '.join(phases[:5])}

Content Summary:
{content_summary}

## Guidelines
1. Be conversational and warm, like a friendly tutor
2. Use examples and analogies to explain concepts
3. Ask clarifying questions when needed
4. Encourage and motivate the learner
5. Reference the course content when relevant
6. If you don't know something, be honest and suggest looking it up

## Style
- Use emojis occasionally for engagement
- Break down complex topics simply
- Celebrate learning progress
- Be patient and encouraging

## Important Notes
- You ONLY have knowledge about the course content provided above
- If asked about something outside this scope, politely redirect to the course
- For coding questions, provide clean, well-explained examples
- Always be helpful and supportive"""
        
        return prompt

    def _build_user_prompt(self, message: str, history: List[Dict]) -> str:
        """
        Build user prompt with conversation history context.
        """
        # Build history context
        history_context = ""
        if history and len(history) > 1:
            recent = history[-5:]  # Last 5 messages for context
            history_context = "\nPrevious conversation:\n"
            for msg in recent:
                role = "Student" if msg["role"] == "user" else "Mentor"
                history_context += f"{role}: {msg['content']}\n"
        
        prompt = f"""{history_context}

Current question/input:
{message}

Please provide a helpful, informative response as the Jinvexa Mentor."""
        
        return prompt

    def get_session_topic(self, session_id: str) -> str:
        """
        Get the topic of a session for display. Delegates to MemoryHandler.
        """
        return self.memory.get_mentoring_session_topic(session_id)

    def close(self):
        """Close database connection. Delegates to MemoryHandler."""
        self.memory.close_mentoring_db()