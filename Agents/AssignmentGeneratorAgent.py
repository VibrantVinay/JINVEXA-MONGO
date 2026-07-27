# D:\Jinvexa\Agents\AssignmentGeneratorAgent.py

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import random
from Agents.BaseAgent import BaseAgent


class AssignmentGeneratorAgent(BaseAgent):
    """
    Generates personalized assignments from lesson content.
    Automatically configures based on course complexity and user level.
    """

    def __init__(
        self,
        llm_client: Any,
        memory_handler: Any,
        config: Optional[Dict] = None
    ):
        super().__init__("AssignmentGeneratorAgent", llm_client)
        
        self.llm_client = llm_client
        self.memory = memory_handler
        self.config = config or {}
        
        # Storage directories
        self.assignments_dir = Path("learn_files/assignments")
        self.sessions_dir = self.assignments_dir / "sessions"
        self.results_dir = self.assignments_dir / "results"
        self.templates_dir = self.assignments_dir / "templates"
        
        # Create directories
        self.assignments_dir.mkdir(exist_ok=True)
        self.sessions_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.templates_dir.mkdir(exist_ok=True)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and generate an assignment."""
        session_id = input_data.get("session_id", "")
        user_id = input_data.get("user_id", "user_1")
        return await self.generate_assignment(session_id, user_id)

    def get_session_lessons(self, session_id: str) -> List[Dict]:
        """Get all lessons for a session from the manifest."""
        manifest = self.memory.get_manifest(session_id) if self.memory else {}
        return manifest.get("watch_order", [])

    def get_lesson_content(self, lesson_file: str) -> str:
        """Get the full content of a lesson file."""
        return self.memory.get_lesson_content(lesson_file) if self.memory else ""

    async def generate_assignment(
        self,
        session_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete assignment with auto-configuration.
        """
        self.log_reasoning("Starting assignment generation...", f"Generating for session: {session_id[:20]}...", "thinking")
        
        # Get lessons from manifest
        lessons = self.get_session_lessons(session_id)
        
        if not lessons:
            return {"error": f"No lessons found for session {session_id}"}
        
        # Get lesson content
        lesson_contents = []
        total_content_length = 0
        
        for lesson in lessons:
            text_file = lesson.get("text_file", "")
            if text_file:
                content = self.get_lesson_content(text_file)
                if content:
                    lesson_contents.append({
                        "topic": lesson.get("topic", ""),
                        "phase": lesson.get("phase", ""),
                        "content": content,
                        "content_length": len(content)
                    })
                    total_content_length += len(content)
        
        if not lesson_contents:
            return {"error": "Could not extract lesson content"}
        
        # Get user profile for personalization
        user_profile = None
        if self.memory and user_id:
            user_profile = self.memory.load_profile(user_id)
        
        # Auto-configure assignment using LLM
        self.log_reasoning("AI configuring assignment...", "Analyzing course complexity and user level", "thinking")
        config = await self._auto_configure_assignment(
            lesson_contents=lesson_contents,
            total_lessons=len(lesson_contents),
            total_content_length=total_content_length,
            user_profile=user_profile
        )
        self.log_reasoning("Configuration complete", f"MCQ: {config.get('num_mcq')}, Written: {config.get('num_written')}, Difficulty: {config.get('difficulty')}", "success")
        
        # Generate questions using LLM
        self.log_reasoning("Generating MCQ questions...", f"Creating {config.get('num_mcq')} multiple-choice questions", "thinking")
        mcq_questions = await self._generate_mcq_questions(
            lesson_contents, 
            config.get("num_mcq", 5),
            config.get("difficulty", "intermediate")
        )
        self.log_reasoning("MCQ questions ready", f"Generated {len(mcq_questions)} MCQ questions", "success")
        
        self.log_reasoning("Generating written questions...", f"Creating {config.get('num_written')} written questions", "thinking")
        written_questions = await self._generate_written_questions(
            lesson_contents,
            config.get("num_written", 2),
            config.get("difficulty", "intermediate")
        )
        self.log_reasoning("Written questions ready", f"Generated {len(written_questions)} written questions", "success")
        
        # Build assignment
        assignment_id = f"assign_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        assignment = {
            "assignment_id": assignment_id,
            "session_id": session_id,
            "user_id": user_id,
            "generated_at": datetime.now().isoformat(),
            "configuration": config,
            "total_questions": config.get("num_mcq", 5) + config.get("num_written", 2),
            "questions": {
                "mcq": mcq_questions,
                "written": written_questions
            },
            "time_limit_minutes": self._calculate_time_limit(
                config.get("num_mcq", 5),
                config.get("num_written", 2)
            ),
            "passing_score": config.get("passing_score", 70)
        }
        
        # Save assignment
        self._save_assignment(assignment)
        
        return assignment

    async def _auto_configure_assignment(
        self,
        lesson_contents: List[Dict],
        total_lessons: int,
        total_content_length: int,
        user_profile: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to automatically configure assignment based on course.
        """
        # Get user level if available
        user_level = "beginner"
        user_goals = ""
        previous_performance = ""
        
        if user_profile:
            user_level = user_profile.preferred_depth or "beginner"
            if user_profile.goals:
                user_goals = f"User goals: {', '.join(user_profile.goals)}"
            
            # Check previous assignment performance
            if hasattr(user_profile, 'assignments') and user_profile.assignments:
                last_scores = [a.get('score', 0) for a in user_profile.assignments[-3:]]
                if last_scores:
                    avg_last = sum(last_scores) / len(last_scores)
                    previous_performance = f"Previous average score: {avg_last:.1f}%"
        
        # Prepare course summary
        topics = [l.get("topic", "") for l in lesson_contents if l.get("topic")]
        phases = list(set([l.get("phase", "") for l in lesson_contents if l.get("phase")]))
        
        prompt = f"""
You are an expert educational assessment designer. Configure an assignment for this course.

Course Summary:
- Total Lessons: {total_lessons}
- Topics: {', '.join(topics[:10])}
- Phases: {', '.join(phases)}
- Total Content Length: {total_content_length} characters
- User Level: {user_level}
{user_goals}
{previous_performance}

Based on this information, determine the optimal assignment configuration:

1. Number of MCQ questions (3-10): More for complex courses, fewer for simple ones
2. Number of written questions (1-4): More for advanced/deep topics
3. Difficulty level: beginner, intermediate, or advanced
4. Passing score (60-80%): Based on difficulty and user level

Consider:
- Course complexity (more phases → more questions)
- User level (beginner → more MCQs, advanced → more written)
- Content depth (deep content → more written questions)
- Previous performance (if available)

Return ONLY a JSON object in this exact format:
{{
    "num_mcq": 5,
    "num_written": 2,
    "difficulty": "intermediate",
    "passing_score": 70,
    "reasoning": "Brief explanation of why this configuration was chosen"
}}

JSON:
"""
        
        try:
            response = await self.llm_client.complete_with_json(prompt)
            if response and isinstance(response, dict):
                # Validate and sanitize
                return {
                    "num_mcq": max(3, min(10, response.get("num_mcq", 5))),
                    "num_written": max(1, min(4, response.get("num_written", 2))),
                    "difficulty": response.get("difficulty", "intermediate") if response.get("difficulty") in ["beginner", "intermediate", "advanced"] else "intermediate",
                    "passing_score": max(60, min(80, response.get("passing_score", 70))),
                    "reasoning": response.get("reasoning", "Auto-configured based on course content.")
                }
        except Exception as e:
            print(f"⚠️ Auto-configuration error: {e}")
        
        # Fallback: intelligent default based on content
        return self._fallback_configuration(total_lessons, total_content_length, user_level)

    def _fallback_configuration(
        self,
        total_lessons: int,
        total_content_length: int,
        user_level: str
    ) -> Dict[str, Any]:
        """
        Intelligent fallback configuration based on course metrics.
        """
        # More lessons → more questions
        if total_lessons >= 10:
            num_mcq = 8
            num_written = 3
        elif total_lessons >= 5:
            num_mcq = 6
            num_written = 2
        else:
            num_mcq = 4
            num_written = 1
        
        # More content → more questions
        if total_content_length > 50000:
            num_mcq = min(10, num_mcq + 2)
            num_written = min(4, num_written + 1)
        
        # Difficulty based on user level
        difficulty_map = {
            "beginner": "beginner",
            "intermediate": "intermediate",
            "advanced": "advanced"
        }
        difficulty = difficulty_map.get(user_level, "intermediate")
        
        # Passing score based on difficulty
        passing_score_map = {
            "beginner": 65,
            "intermediate": 70,
            "advanced": 75
        }
        passing_score = passing_score_map.get(difficulty, 70)
        
        return {
            "num_mcq": num_mcq,
            "num_written": num_written,
            "difficulty": difficulty,
            "passing_score": passing_score,
            "reasoning": f"Auto-configured: {total_lessons} lessons, {total_content_length} chars, user level: {user_level}"
        }

    async def _generate_mcq_questions(
        self,
        lesson_contents: List[Dict],
        num_questions: int,
        difficulty: str
    ) -> List[Dict]:
        """
        Generate MCQ questions using LLM with proper topic assignment.
        """
        # Prepare lesson summaries with clear topic mapping
        lesson_summaries = []
        for lesson in lesson_contents[:5]:
            topic = lesson.get("topic", "")
            phase = lesson.get("phase", "")
            content = lesson.get("content", "")[:2000]
            lesson_summaries.append(f"Topic: {topic}\nPhase: {phase}\nContent: {content[:500]}...")
        
        combined_content = "\n\n".join(lesson_summaries)
        
        prompt = f"""
You are a expert question generator. Based on the following course content, generate {num_questions} multiple-choice questions.

Course Content:
{combined_content}

Difficulty Level: {difficulty}

CRITICAL: For each question, you MUST include the exact topic name from the content above.

For each question, provide:
1. A clear question
2. 4 options (A, B, C, D)
3. The correct answer index (0-3)
4. The EXACT topic name from the content (this is MANDATORY)
5. Brief explanation of why the answer is correct

Return ONLY a JSON array in this exact format:
[
    {{
        "id": "mcq_1",
        "question": "Your question here?",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correct_answer": 0,
        "topic": "EXACT TOPIC NAME FROM CONTENT",
        "explanation": "Why this is correct"
    }}
]

The correct_answer should be the index (0-3) of the correct option.

JSON:
"""
        
        try:
            response = await self.llm_client.complete(prompt)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
                if isinstance(questions, list):
                    # Ensure each question has a topic
                    for q in questions:
                        if not q.get("topic") or q.get("topic") == "Unknown":
                            # Try to infer topic from question content
                            q_text = q.get("question", "").lower()
                            for lesson in lesson_contents:
                                topic = lesson.get("topic", "")
                                if topic and topic.lower() in q_text:
                                    q["topic"] = topic
                                    break
                            if not q.get("topic"):
                                # Use first topic as fallback
                                q["topic"] = lesson_contents[0].get("topic", "General") if lesson_contents else "General"
                    return questions[:num_questions]
        except Exception as e:
            print(f"❌ MCQ generation error: {e}")
        
        # Fallback questions with proper topics
        return self._fallback_mcq_questions(lesson_contents, num_questions)

    async def _generate_written_questions(
        self,
        lesson_contents: List[Dict],
        num_questions: int,
        difficulty: str
    ) -> List[Dict]:
        """
        Generate written/essay questions using LLM.
        """
        lesson_summaries = []
        for lesson in lesson_contents[:5]:
            topic = lesson.get("topic", "")
            content = lesson.get("content", "")[:2000]
            lesson_summaries.append(f"Topic: {topic}\nContent: {content[:500]}...")
        
        combined_content = "\n\n".join(lesson_summaries)
        
        prompt = f"""
You are a expert question generator. Based on the following course content, generate {num_questions} written/essay questions.

Course Content:
{combined_content}

Difficulty Level: {difficulty}

For each question, provide:
1. A clear, open-ended question
2. The topic it relates to
3. Maximum score (out of 10)
4. Grading rubric with criteria

Return ONLY a JSON array in this exact format:
[
    {{
        "id": "written_1",
        "question": "Your open-ended question here?",
        "topic": "Related Topic",
        "max_score": 10,
        "rubric": {{
            "clarity": 3,
            "correctness": 4,
            "examples": 3
        }}
    }}
]

JSON:
"""
        
        try:
            response = await self.llm_client.complete(prompt)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
                if isinstance(questions, list):
                    return questions[:num_questions]
        except Exception as e:
            print(f"❌ Written question generation error: {e}")
        
        return self._fallback_written_questions(lesson_contents, num_questions)

    def _fallback_mcq_questions(self, lesson_contents: List[Dict], num_questions: int) -> List[Dict]:
        """Fallback MCQ questions with proper topics."""
        questions = []
        topics = [l.get("topic", "General") for l in lesson_contents if l.get("topic")]
        
        if not topics:
            topics = ["General"]
        
        for i in range(min(num_questions, len(topics))):
            topic = topics[i % len(topics)]
            questions.append({
                "id": f"mcq_{i+1}",
                "question": f"What is a key concept in {topic}?",
                "options": ["Option A - Correct", "Option B", "Option C", "Option D"],
                "correct_answer": 0,
                "topic": topic,
                "explanation": f"This is the fundamental concept of {topic}."
            })
        
        return questions

    def _fallback_written_questions(self, lesson_contents: List[Dict], num_questions: int) -> List[Dict]:
        """Fallback written questions."""
        questions = []
        topics = [l.get("topic", "Unknown") for l in lesson_contents if l.get("topic")]
        
        for i in range(min(num_questions, len(topics))):
            topic = topics[i % len(topics)]
            questions.append({
                "id": f"written_{i+1}",
                "question": f"Explain the importance of {topic} in detail. Provide examples.",
                "topic": topic,
                "max_score": 10,
                "rubric": {
                    "clarity": 3,
                    "correctness": 4,
                    "examples": 3
                }
            })
        
        return questions

    def _extract_topics_from_lessons(self, lesson_contents: List[Dict]) -> List[str]:
        """
        Extract unique topics from lesson contents with fallback.
        """
        topics = []
        
        for lesson in lesson_contents:
            topic = lesson.get("topic", "")
            if topic and topic not in topics:
                topics.append(topic)
        
        # If no topics found, try to extract from content
        if not topics:
            for lesson in lesson_contents:
                content = lesson.get("content", "")
                # Look for common topic patterns
                lines = content.split('\n')[:20]
                for line in lines:
                    if '#' in line:  # Markdown headers
                        clean = re.sub(r'#+\s*', '', line).strip()
                        if clean and len(clean) < 50 and clean not in topics:
                            topics.append(clean)
                            break
                    if 'Topic:' in line:
                        clean = line.replace('Topic:', '').strip()
                        if clean and clean not in topics:
                            topics.append(clean)
                            break
        
        return topics if topics else ["General"]

    def _calculate_time_limit(self, num_mcq: int, num_written: int) -> int:
        """Calculate time limit based on question count."""
        # 1 minute per MCQ, 5 minutes per written
        return (num_mcq * 1) + (num_written * 5) + 5  # +5 minutes buffer

    def _save_assignment(self, assignment: Dict):
        """Save assignment to file. Delegates to MemoryHandler."""
        return self.memory.save_assignment(assignment)

    def get_assignment(self, assignment_id: str) -> Optional[Dict]:
        """Get assignment by ID. Delegates to MemoryHandler."""
        return self.memory.get_assignment(assignment_id) if self.memory else None

    def list_assignments(self, session_id: str) -> List[Dict]:
        """List all assignments for a session. Delegates to MemoryHandler."""
        return self.memory.list_assignments(session_id) if self.memory else []