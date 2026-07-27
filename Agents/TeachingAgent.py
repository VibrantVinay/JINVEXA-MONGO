# D:\Jinvexa\Agents\TeachingAgent.py

import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import your existing TTS
import sys
sys.path.append(str(Path(__file__).parent.parent))
from output.tts import TextToSpeech

from Agents.BaseAgent import BaseAgent


class TeachingAgent(BaseAgent):
    """
    Teaching Layer - Generates detailed lessons with multithreading.
    Outputs structured lessons with JSON manifest for watch order.
    """
    
    def __init__(
        self,
        llm_client: Any,
        memory_handler: Any,
        config: Optional[Dict] = None
    ):
        super().__init__("TeachingAgent", llm_client)
        
        self.llm_client = llm_client
        self.memory = memory_handler
        self.config = config or {}
        self.tts = TextToSpeech()
        
        # Main output directory
        self.learn_files_dir = Path("learn_files")
        self.lessons_dir = self.learn_files_dir / "lessons"
        self.audio_dir = self.learn_files_dir / "audio"
        self.manifest_dir = self.learn_files_dir / "manifests"
        
        # Create all directories
        self.learn_files_dir.mkdir(exist_ok=True)
        self.lessons_dir.mkdir(exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
        self.manifest_dir.mkdir(exist_ok=True)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Track generated content
        self.generated_content: Dict[str, Dict] = {}
        self._lock = threading.Lock()
    
    # ==================== IMPLEMENT ABSTRACT METHOD ====================
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data - required by BaseAgent.
        """
        action = input_data.get("action", "list_sessions")
        
        if action == "generate_course":
            session_id = input_data.get("session_id")
            if not session_id:
                return {"error": "session_id is required"}
            return await self.generate_course_from_session(session_id)
        
        elif action == "list_sessions":
            sessions = self.list_available_sessions()
            return {"sessions": sessions, "count": len(sessions)}
        
        elif action == "get_status":
            session_id = input_data.get("session_id")
            if not session_id:
                return {"error": "session_id is required"}
            return self.get_course_status(session_id)
        
        return {"error": f"Unknown action: {action}"}
    
    # ==================== MAIN ENTRY POINTS ====================
    
    def list_available_sessions(self) -> List[Dict]:
        """List all sessions that have learning plans"""
        if self.memory:
            return self.memory.list_sessions_with_plans()
        return []
    
    async def generate_course_from_session(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Generate a complete course with multithreading.
        LLM decides format for each topic.
        """
        # Get session data
        self.log_reasoning("Starting course generation...", f"Loading session: {session_id[:20]}...", "thinking")
        session_data = self._get_session_data(session_id)
        if not session_data:
            return {"error": f"Session {session_id} not found or has no learning plan"}
        
        learning_plan = session_data.get("learning_plan", {})
        if isinstance(learning_plan, str):
            try:
                learning_plan = json.loads(learning_plan)
            except:
                return {"error": "Invalid learning plan format"}
        
        main_topic = learning_plan.get("main_topic", "Untitled Course")
        roadmap = learning_plan.get("roadmap", [])
        
        if not roadmap:
            return {"error": "No roadmap found in learning plan"}
        
        self.log_reasoning("Session loaded", f"Course: {main_topic} with {len(roadmap)} phases", "success")
        
        # Get user profile
        user_profile = None
        if self.memory:
            user_profile = self.memory.load_profile(session_data.get("user_id", ""))
        
        # First, let LLM decide the format for each topic
        self.log_reasoning("AI deciding formats...", "Analyzing each topic to choose best format (text/audio)", "thinking")
        format_decisions = await self._decide_formats(roadmap, main_topic, user_profile)
        self.log_reasoning("Format decisions made", f"{len(format_decisions)} topics analyzed", "success")
        
        # Prepare all lesson tasks
        lesson_tasks = []
        lesson_counter = 0
        
        for phase in roadmap:
            phase_number = phase.get("phase_number", 0)
            phase_title = phase.get("title", f"Phase {phase_number}")
            topics = phase.get("topics", [])
            
            for topic in topics:
                if not topic or topic.strip() == "":
                    continue
                
                lesson_counter += 1
                format_decision = self._get_format_for_topic(format_decisions, topic, phase_title)
                
                lesson_tasks.append({
                    "lesson_id": lesson_counter,
                    "topic": topic,
                    "phase_number": phase_number,
                    "phase_title": phase_title,
                    "main_topic": main_topic,
                    "output_format": format_decision.get("output_format", "text"),
                    "gender": format_decision.get("gender", "female"),
                    "reason": format_decision.get("reason", ""),
                    "user_profile": user_profile,
                    "session_id": session_id
                })
        
        # Generate lessons in parallel using multithreading
        self.log_reasoning("Generating lessons...", f"Using {self.executor._max_workers} parallel threads for {len(lesson_tasks)} lessons", "thinking")
        print(f"\n🚀 Generating {len(lesson_tasks)} lessons in parallel...")
        
        # Run tasks in parallel
        results = await asyncio.gather(*[
            self._generate_lesson_async(task) for task in lesson_tasks
        ])
        
        # Collect results
        course_content = {
            "session_id": session_id,
            "main_topic": main_topic,
            "generated_at": datetime.now().isoformat(),
            "phases": [],
            "total_lessons": 0,
            "format_decisions": format_decisions,
            "manifest": []
        }
        
        # Group results by phase
        phases_dict = {}
        for result in results:
            if result and result.get("success"):
                phase_key = f"{result['phase_number']}_{result['phase_title']}"
                if phase_key not in phases_dict:
                    phases_dict[phase_key] = {
                        "phase_number": result["phase_number"],
                        "phase_title": result["phase_title"],
                        "topics": [],
                        "lessons": []
                    }
                
                phases_dict[phase_key]["topics"].append(result["topic"])
                phases_dict[phase_key]["lessons"].append(result)
                course_content["total_lessons"] += 1
        
        # Add to course content
        for phase_key, phase_data in sorted(phases_dict.items()):
            course_content["phases"].append(phase_data)
        
        # Build manifest with proper watch order
        manifest = self._build_manifest(course_content)
        course_content["manifest"] = manifest
        
        self.log_reasoning("Course generation complete", f"Generated {course_content['total_lessons']} lessons across {len(course_content['phases'])} phases", "success")
        
        # Save manifest
        self._save_manifest(session_id, manifest, main_topic)
        
        # Save course metadata
        self._save_course_metadata(session_id, course_content)
        
        return course_content
    
    async def _generate_lesson_async(self, task: Dict) -> Dict:
        """
        Generate a single lesson with multithreading support.
        """
        try:
            # Run LLM generation in thread pool
            loop = asyncio.get_event_loop()
            
            # Generate lesson text using LLM (in thread pool)
            lesson_text = await loop.run_in_executor(
                self.executor,
                self._generate_lesson_text_sync,
                task
            )
            
            if not lesson_text:
                return {"success": False, "error": "Failed to generate text"}
            
            # Generate audio if needed (directly await async method)
            audio_file = None
            if task["output_format"] == "audio":
                audio_file = await self._generate_audio_async(
                    task["session_id"],
                    task["topic"],
                    lesson_text,
                    task["gender"]
                )
            
            # Save text lesson (in thread pool)
            text_file = await loop.run_in_executor(
                self.executor,
                self._save_text_lesson_sync,
                task["session_id"],
                task["topic"],
                task["phase_title"],
                lesson_text
            )
            
            result = {
                "success": True,
                "lesson_id": task["lesson_id"],
                "topic": task["topic"],
                "phase_number": task["phase_number"],
                "phase_title": task["phase_title"],
                "output_format": task["output_format"],
                "gender": task["gender"],
                "reason": task["reason"],
                "text_file": text_file,
                "audio_file": audio_file,
                "lesson_text": lesson_text[:500] + "..." if len(lesson_text) > 500 else lesson_text
            }
            
            print(f"✅ Lesson {task['lesson_id']}/{task['phase_title']}: {task['topic']} ({task['output_format']})")
            return result
            
        except Exception as e:
            print(f"❌ Error generating lesson for '{task.get('topic', 'Unknown')}': {e}")
            return {"success": False, "error": str(e), "topic": task.get("topic", "Unknown")}
    
    def _generate_lesson_text_sync(self, task: Dict) -> str:
        """
        Synchronous version of lesson text generation (for thread pool).
        """
        topic = task["topic"]
        phase_title = task["phase_title"]
        main_topic = task["main_topic"]
        user_profile = task.get("user_profile")
        output_format = task.get("output_format", "text")
        
        user_level = "beginner"
        if user_profile:
            user_level = user_profile.preferred_depth or "beginner"
        
        tone = "conversational and warm" if output_format == "audio" else "clear and structured"
        
        # Generate prompt
        prompt = f"""
You are an expert teacher creating a detailed, engaging lesson.

Course: {main_topic}
Phase: {phase_title}
Topic: {topic}
Learner Level: {user_level}
Tone: {tone}

Create a comprehensive lesson that includes:
1. **Introduction** - Hook the learner, explain why this topic matters
2. **Core Concepts** - Clear explanations with simple analogies
3. **Practical Examples** - Real-world applications
4. **Key Takeaways** - Summary of important points
5. **Practice Exercise** - Simple exercise to reinforce learning
6. **Next Steps** - What to learn next

Format with markdown headers.

Lesson:
"""
        
        try:
            # Run LLM completion (this is synchronous in this context)
            import asyncio
            response = asyncio.run(self.llm_client.complete(prompt))
            return response.strip()
        except Exception as e:
            print(f"❌ LLM error for '{topic}': {e}")
            return self._create_fallback_lesson(topic, main_topic, user_level)
    
    async def _generate_audio_async(self, session_id: str, topic: str, lesson_text: str, gender: str) -> Optional[str]:
        """
        Async audio generation (runs directly on the event loop).
        """
        try:
            clean_text = self._clean_text_for_tts(lesson_text)
            if not clean_text or len(clean_text.strip()) < 10:
                print(f"⚠️ TTS skipped for '{topic}': text too short after cleaning")
                return None

            safe_topic = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')
            safe_topic = re.sub(r'[-\s]+', '_', safe_topic)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            audio_filename = f"{session_id}_{timestamp}_{safe_topic}_{gender}.mp3"
            audio_filepath = self.audio_dir / audio_filename

            await self.tts.speak_async(
                txt=clean_text[:5000],
                gender=gender,
                output=str(audio_filepath)
            )

            # Verify file was created
            if not audio_filepath.exists() or audio_filepath.stat().st_size == 0:
                print(f"❌ TTS error for '{topic}': output file is empty")
                return None

            self._save_audio_metadata(session_id, topic, str(audio_filepath), gender)
            print(f"🎧 Audio generated: {audio_filename}")
            return str(audio_filepath)

        except Exception as e:
            print(f"❌ TTS error for '{topic}': {e}")
            return None

    def _generate_audio_sync(self, session_id: str, topic: str, lesson_text: str, gender: str) -> Optional[str]:
        """
        Synchronous audio generation (for thread pool).
        """
        try:
            clean_text = self._clean_text_for_tts(lesson_text)
            
            safe_topic = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')
            safe_topic = re.sub(r'[-\s]+', '_', safe_topic)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            audio_filename = f"{session_id}_{timestamp}_{safe_topic}_{gender}.mp3"
            audio_filepath = self.audio_dir / audio_filename
        
            self.tts.speak(
                txt=clean_text[:5000],
                gender=gender,
                output=str(audio_filepath)
            )
            
            self._save_audio_metadata(session_id, topic, str(audio_filepath), gender)
            return str(audio_filepath)
            
        except Exception as e:
            print(f"❌ TTS error for '{topic}': {e}")
            return None
    
    def _save_text_lesson_sync(self, session_id: str, topic: str, phase_title: str, lesson_text: str) -> str:
        """
        Synchronous text lesson saving (for thread pool).
        Delegates to MemoryHandler.
        """
        return self.memory.save_lesson(session_id, topic, phase_title, lesson_text)
    
    def _build_manifest(self, course_content: Dict) -> List[Dict]:
        """
        Build a JSON manifest with watch order and content type.
        """
        manifest = []
        lesson_counter = 0
        
        for phase in course_content.get("phases", []):
            for lesson in phase.get("lessons", []):
                lesson_counter += 1
                
                # Determine content type
                output_format = lesson.get("output_format", "text")
                if output_format == "audio":
                    content_type = "audio"
                    file_path = lesson.get("audio_file", "")
                else:
                    content_type = "text"
                    file_path = lesson.get("text_file", "")
                
                manifest.append({
                    "order": lesson_counter,
                    "phase": lesson.get("phase_title", ""),
                    "phase_number": lesson.get("phase_number", 0),
                    "topic": lesson.get("topic", ""),
                    "content_type": content_type,
                    "gender": lesson.get("gender", ""),
                    "file_path": file_path,
                    "text_file": lesson.get("text_file", ""),
                    "audio_file": lesson.get("audio_file", ""),
                    "format_reason": lesson.get("reason", "")
                })
        
        return manifest
    
    def _save_manifest(self, session_id: str, manifest: List[Dict], main_topic: str):
        """
        Save the manifest JSON file.
        Delegates to MemoryHandler.
        """
        self.memory.save_manifest(session_id, manifest, main_topic)
    
    # ==================== FORMAT DECISION METHODS ====================
    
    async def _decide_formats(
        self,
        roadmap: List[Dict],
        main_topic: str,
        user_profile: Optional[Any] = None
    ) -> List[Dict]:
        """Use LLM to decide the best format for each topic."""
        topic_list = []
        for phase in roadmap:
            phase_title = phase.get("title", "Phase")
            for topic in phase.get("topics", []):
                if topic and topic.strip():
                    topic_list.append({
                        "phase": phase_title,
                        "topic": topic
                    })
        
        if not topic_list:
            return []
        
        user_level = "beginner"
        user_goals = ""
        if user_profile:
            user_level = user_profile.preferred_depth or "beginner"
            if user_profile.goals:
                user_goals = f"\nUser goals: {', '.join(user_profile.goals)}"
        
        prompt = f"""
You are an expert educational content strategist. For each topic in this course, decide the BEST delivery format.

Course: {main_topic}
User Level: {user_level}{user_goals}

For each topic, choose ONE of these formats:
- "text" - Best for self-paced reading, reference material, code-heavy content
- "female_voice" - Best for warm, engaging explanations, beginner-friendly topics
- "male_voice" - Best for technical, professional, or advanced topics

Topics:
{json.dumps(topic_list, indent=2)}

Return a JSON array where each object has:
- topic: the topic name
- phase: the phase title
- output_format: "text", "female_voice", or "male_voice"
- reason: brief explanation of why this format is best
"""
        
        try:
            response = await self.llm_client.complete(prompt)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                decisions = json.loads(json_match.group())
                if isinstance(decisions, list) and len(decisions) > 0:
                    return decisions
        except Exception as e:
            self.logger.error(f"Format decision LLM error: {e}")
        
        return self._fallback_format_decisions(topic_list)
    
    def _fallback_format_decisions(self, topic_list: List[Dict]) -> List[Dict]:
        """Fallback format decisions based on topic characteristics."""
        decisions = []
        
        text_keywords = ["algorithm", "structure", "code", "syntax", "implementation", "architecture", "design pattern", "framework", "library", "api", "database", "sql", "nosql", "security", "performance", "optimization", "memory", "concurrency", "threading"]
        female_keywords = ["introduction", "overview", "basics", "fundamental", "concept", "idea", "principle", "beginner", "welcome", "getting started", "thought", "why", "what is", "explanation", "simple", "easy"]
        male_keywords = ["advanced", "professional", "technical", "deep", "complex", "expert", "master", "production", "enterprise", "scaling", "architecture", "system design", "optimization", "performance"]
        
        for item in topic_list:
            topic = item.get("topic", "").lower()
            phase = item.get("phase", "").lower()
            
            if any(kw in topic for kw in text_keywords):
                output_format = "text"
                reason = "Technical/reference content best for self-paced reading"
            elif any(kw in topic for kw in male_keywords):
                output_format = "male_voice"
                reason = "Advanced/technical topic benefits from professional narration"
            elif any(kw in topic for kw in female_keywords):
                output_format = "female_voice"
                reason = "Introductory/conceptual topic benefits from warm audio guidance"
            else:
                if "beginner" in phase or "foundation" in phase:
                    output_format = "female_voice"
                    reason = "Foundation topic suited for warm audio explanation"
                elif "advanced" in phase or "capstone" in phase:
                    output_format = "male_voice"
                    reason = "Advanced topic suited for professional narration"
                else:
                    output_format = "text"
                    reason = "Balanced content suitable for self-paced reading"
            
            decisions.append({
                "topic": item.get("topic"),
                "phase": item.get("phase"),
                "output_format": output_format,
                "reason": reason
            })
        
        return decisions
    
    def _get_format_for_topic(
        self,
        format_decisions: List[Dict],
        topic: str,
        phase_title: str
    ) -> Dict:
        """Get format decision for a specific topic.
        
        Uses fuzzy phase matching because the LLM may shorten phase titles
        (e.g. returns 'Fundamentals' instead of the full descriptive title).
        """
        for decision in format_decisions:
            if decision.get("topic") == topic:
                # Fuzzy phase match: LLM may shorten phase titles
                decision_phase = decision.get("phase", "")
                if decision_phase in phase_title or phase_title in decision_phase:
                    output_format = decision.get("output_format", "text")
                    if output_format == "female_voice":
                        return {"output_format": "audio", "gender": "female", "reason": decision.get("reason", "")}
                    elif output_format == "male_voice":
                        return {"output_format": "audio", "gender": "male", "reason": decision.get("reason", "")}
                    else:
                        return {"output_format": "text", "gender": None, "reason": decision.get("reason", "")}
        
        return {"output_format": "text", "gender": None, "reason": "Default format"}
    
    # ==================== UTILITY METHODS ====================
    
    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for TTS."""
        text = re.sub(r'#+\s*', '', text)
        text = re.sub(r'\*\*|\*|__|_', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        text = re.sub(r'`[^`]*`', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        return text.strip()[:5000]
    
    def _create_fallback_lesson(self, topic: str, main_topic: str, level: str) -> str:
        """Create fallback lesson."""
        return f"""
# {topic}

## Introduction
Welcome to this lesson on **{topic}** as part of **{main_topic}**.

This lesson is designed for a {level} level learner.

## Core Concepts
Understanding {topic} is essential for mastering {main_topic}.

### Key Points
- {topic} is a fundamental concept
- It builds on previous knowledge
- Practice is essential

## Practice Exercise
1. Research one example of {topic}
2. Write down what you learned
3. Apply to a small project

## Next Steps
Continue to the next lesson.
"""
    
    def _save_audio_metadata(self, session_id: str, topic: str, audio_file: str, gender: str):
        """Save audio metadata. Delegates to MemoryHandler."""
        self.memory.save_audio_metadata(session_id, topic, audio_file, gender)
    
    def _save_course_metadata(self, session_id: str, course_content: Dict):
        """Save course metadata. Delegates to MemoryHandler."""
        self.memory.save_course_metadata(session_id, course_content)
        
        if self.memory:
            session = self.memory.get_session(session_id)
            if session:
                if not session.metadata:
                    session.metadata = {}
                session.metadata["course_generated"] = True
                session.metadata["course_generated_at"] = datetime.now().isoformat()
                session.metadata["total_lessons"] = course_content.get("total_lessons", 0)
                session.metadata["manifest"] = course_content.get("manifest", [])
                self.memory.update_session(session)
    
    def _get_session_data(self, session_id: str) -> Optional[Dict]:
        """Get session data from memory."""
        if not self.memory:
            return None
        
        session = self.memory.get_session(session_id)
        if not session:
            return None
        
        learning_plan = session.learning_plan
        if isinstance(learning_plan, str):
            try:
                learning_plan = json.loads(learning_plan)
            except:
                learning_plan = None
        
        if not learning_plan:
            return None
        
        return {
            "session_id": session_id,
            "user_id": session.user_id,
            "learning_plan": learning_plan
        }
    
    def get_course_status(self, session_id: str) -> Dict:
        """Get course status."""
        return self.memory.get_manifest(session_id) if self.memory else {"error": "No course found for this session"}
    
    def get_lesson_content(self, lesson_file: str) -> Optional[str]:
        """Get lesson content. Delegates to MemoryHandler."""
        return self.memory.get_lesson_content(lesson_file) if self.memory else None
    
    def list_generated_lessons(self, session_id: str) -> List[Dict]:
        """List generated lessons. Delegates to MemoryHandler."""
        return self.memory.list_generated_lessons(session_id) if self.memory else []
    
    def __del__(self):
        """Cleanup thread pool."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)