# D:\Jinvexa\Agents\LearningDiscoveryAgent.py

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
from collections import defaultdict

# Change from relative to absolute imports
from Agents.BaseAgent import BaseAgent
from Models.UserProfile import UserProfile
from Models.LearningPlan import LearningPlan
from Models.KnowledgeGraph import KnowledgeGraph


class LearningDiscoveryAgent(BaseAgent):
    """
    The main conversational agent that:
    1. Understands the source (document or goal)
    2. Understands the learner through adaptive questioning
    3. Discovers learning goals
    4. Estimates knowledge with confidence
    5. Generates personalized learning plans
    """
    
    def __init__(
        self,
        data_extractor: Any,
        concept_extractor: Any,
        dependency_agent: Any,
        knowledge_gap_agent: Any,
        llm_client: Any,
        config: Optional[Dict] = None,
        memory_handler: Optional[Any] = None  # Add memory handler parameter
    ):
        super().__init__("LearningDiscoveryAgent", llm_client)
        
        self.data_extractor = data_extractor
        self.concept_extractor = concept_extractor
        self.dependency_agent = dependency_agent
        self.knowledge_gap_agent = knowledge_gap_agent
        self.config = config or {}
        
        # Initialize Memory Handler
        self.memory = memory_handler
        
        # Session storage (for backward compatibility)
        self.sessions: Dict[str, Dict] = {}
        
        # Question templates
        self.question_templates = self._load_question_templates()
    
    # ==================== IMPLEMENT ABSTRACT METHOD ====================
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data - Main entry point for the agent.
        This implements the abstract method from BaseAgent.
        
        Input data can contain:
        - mode: "goal" or "reference"
        - user_id: str
        - source: str (for reference mode)
        - goal: str (for goal mode)
        - user_profile: UserProfile (optional)
        """
        
        mode = input_data.get("mode", "goal")
        user_id = input_data.get("user_id", "default_user")
        
        if mode == "reference":
            source = input_data.get("source")
            if not source:
                return {"error": "Source is required for reference mode"}
            
            user_profile = input_data.get("user_profile")
            
            # Use memory-enabled method if available
            if self.memory:
                conversation, plan, session_id = await self.process_reference_mode_with_memory(
                    user_id=user_id,
                    source=source,
                    user_profile=user_profile
                )
            else:
                conversation, plan = await self.process_reference_mode(
                    user_id=user_id,
                    source=source,
                    user_profile=user_profile
                )
                session_id = f"{user_id}_{datetime.now().timestamp()}"
            
            return {
                "mode": "reference",
                "conversation": conversation,
                "plan": plan.to_dict() if plan else None,
                "session_id": session_id
            }
        
        else:  # goal mode
            goal = input_data.get("goal")
            if not goal:
                return {"error": "Goal is required for goal mode"}
            
            user_profile = input_data.get("user_profile")
            
            # Use memory-enabled method if available
            if self.memory:
                conversation, plan, session_id = await self.process_goal_mode_with_memory(
                    user_id=user_id,
                    goal_statement=goal,
                    user_profile=user_profile
                )
            else:
                conversation, plan = await self.process_goal_mode(
                    user_id=user_id,
                    goal_statement=goal,
                    user_profile=user_profile
                )
                session_id = f"{user_id}_{datetime.now().timestamp()}"
            
            return {
                "mode": "goal",
                "conversation": conversation,
                "plan": plan.to_dict() if plan else None,
                "session_id": session_id
            }
    
    def _load_question_templates(self) -> Dict[str, List[str]]:
        """Load adaptive question templates"""
        return {
            "general": [
                "What's your current experience level with this topic?",
                "What's your primary goal? (understand, build, interview, research)",
                "How much time can you dedicate per week?",
            ],
            "python": [
                "How comfortable are you with Python?",
                "Have you built any Python projects?",
                "Do you understand Python's async/await?",
                "Are you familiar with Python decorators and context managers?",
            ],
            "statistics": [
                "Do you understand basic statistics?",
                "Are you comfortable with probability concepts?",
                "Do you know what a standard deviation is?",
            ],
            "linear_algebra": [
                "Are you familiar with matrices and matrix operations?",
                "Do you understand vector spaces?",
                "Can you explain what eigenvalues are?",
            ],
            "machine_learning": [
                "Have you implemented any ML models before?",
                "Do you understand supervised vs unsupervised learning?",
                "Are you familiar with gradient descent?",
                "Do you know what overfitting is?",
            ],
            "deep_learning": [
                "Do you understand neural networks?",
                "Are you familiar with backpropagation?",
                "Do you know what activation functions are?",
                "Have you used any deep learning frameworks?",
            ],
            "web_development": [
                "Are you comfortable with HTML and CSS?",
                "Do you understand JavaScript?",
                "Have you built any web applications?",
                "Are you familiar with REST APIs?",
            ],
            "devops": [
                "Are you comfortable with the command line?",
                "Do you understand Linux basics?",
                "Have you used Docker before?",
                "Are you familiar with CI/CD pipelines?",
            ],
        }
    
    # ==================== MAIN ENTRY POINTS (Without Memory) ====================
    
    async def process_reference_mode(
        self,
        user_id: str,
        source: str,
        user_profile: Optional[UserProfile] = None
    ) -> Tuple[List[Dict], LearningPlan]:
        """
        Mode 2: User provides a reference (document, video, repo, etc.)
        Returns: (conversation_history, learning_plan)
        """
        session_id = f"{user_id}_{datetime.now().timestamp()}"
        
        # Initialize session
        self.sessions[session_id] = {
            "user_id": user_id,
            "mode": "reference",
            "source": source,
            "profile": user_profile or UserProfile(user_id=user_id),
            "history": [],
            "extracted_data": None,
            "concepts": None,
            "knowledge_graph": None,
            "gap_analysis": None,
            "learning_plan": None,
            "question_index": 0,
            "pending_questions": [],
            "phase": "extraction"
        }
        
        # Step 1: Extract data from source
        self._log(f"Extracting data from: {source}")
        extracted_data = await self._extract_data(source)
        self.sessions[session_id]["extracted_data"] = extracted_data
        
        # Step 2: Get the text content from extracted data
        text_content = self._get_text_from_extracted(extracted_data)
        
        # Step 3: Analyze source and extract concepts
        self._log("Extracting concepts...")
        concepts = await self._extract_concepts(text_content, source)
        self.sessions[session_id]["concepts"] = concepts
        
        # Step 4: Start adaptive conversation
        self._log("Starting discovery conversation...")
        conversation = await self._conduct_discovery_conversation(
            session_id,
            concepts,
            mode="reference"
        )
        
        # Step 5: Build knowledge graph
        self._log("Building knowledge graph...")
        knowledge_graph = await self._build_knowledge_graph(concepts)
        self.sessions[session_id]["knowledge_graph"] = knowledge_graph
        
        # Step 6: Identify gaps
        self._log("Analyzing knowledge gaps...")
        gaps = await self._identify_gaps(session_id, knowledge_graph)
        self.sessions[session_id]["gap_analysis"] = gaps
        
        # Step 7: Generate learning plan
        self._log("Generating learning plan...")
        plan = await self._generate_learning_plan(session_id)
        self.sessions[session_id]["learning_plan"] = plan
        
        return conversation, plan
    
    async def process_goal_mode(
        self,
        user_id: str,
        goal_statement: str,
        user_profile: Optional[UserProfile] = None
    ) -> Tuple[List[Dict], LearningPlan]:
        """
        Mode 1: User states what they want to become/learn
        Returns: (conversation_history, learning_plan)
        """
        session_id = f"{user_id}_{datetime.now().timestamp()}"
        
        self.sessions[session_id] = {
            "user_id": user_id,
            "mode": "goal",
            "goal_statement": goal_statement,
            "profile": user_profile or UserProfile(user_id=user_id),
            "history": [],
            "concepts": None,
            "knowledge_graph": None,
            "gap_analysis": None,
            "learning_plan": None,
            "question_index": 0,
            "pending_questions": [],
            "phase": "goal_analysis"
        }
        
        # Step 1: Extract concepts from goal
        self._log(f"Analyzing goal: {goal_statement}")
        concepts = await self._extract_concepts_from_goal(goal_statement)
        self.sessions[session_id]["concepts"] = concepts
        
        # Step 2: Start adaptive conversation
        conversation = await self._conduct_discovery_conversation(
            session_id,
            concepts,
            mode="goal"
        )
        
        # Step 3: Build knowledge graph
        knowledge_graph = await self._build_knowledge_graph(concepts)
        self.sessions[session_id]["knowledge_graph"] = knowledge_graph
        
        # Step 4: Identify gaps
        gaps = await self._identify_gaps(session_id, knowledge_graph)
        self.sessions[session_id]["gap_analysis"] = gaps
        
        # Step 5: Generate plan
        plan = await self._generate_learning_plan(session_id)
        self.sessions[session_id]["learning_plan"] = plan
        
        return conversation, plan
    
    # ==================== MEMORY-ENABLED METHODS ====================
    
    async def process_goal_mode_with_memory(
        self,
        user_id: str,
        goal_statement: str,
        user_profile: Optional[UserProfile] = None
    ) -> Tuple[List[Dict], LearningPlan, str]:
        """
        Mode 1 with memory: User states what they want to become/learn
        Returns: (conversation_history, learning_plan, session_id)
        """
        
        # Load or create profile from memory
        if not user_profile and self.memory:
            user_profile = self.memory.load_profile(user_id)
        
        if not user_profile:
            user_profile = UserProfile(user_id=user_id)
        
        # Create session in memory
        if self.memory:
            session_id = self.memory.create_session(
                user_id=user_id,
                mode="goal",
                user_profile=user_profile
            )
        else:
            session_id = f"{user_id}_{datetime.now().timestamp()}"
        
        # Process goal
        self._log(f"Analyzing goal: {goal_statement}")
        concepts = await self._extract_concepts_from_goal(goal_statement)
        
        # Store concepts in session
        if self.memory:
            session = self.memory.get_session(session_id)
            if session:
                session.concepts = concepts
                session.user_profile = user_profile.to_dict()
                self.memory.update_session(session)
        
        # Conduct conversation
        conversation = await self._conduct_discovery_conversation_with_memory(
            session_id,
            concepts,
            mode="goal",
            profile=user_profile
        )
        
        # Build knowledge graph
        knowledge_graph = await self._build_knowledge_graph(concepts)
        
        # Identify gaps
        gaps = await self._identify_gaps_with_profile(user_profile, knowledge_graph)
        
        # Generate plan
        plan = await self._generate_learning_plan_from_data(
            user_profile, concepts, gaps, knowledge_graph
        )
        
        # Update session with results
        if self.memory:
            session = self.memory.get_session(session_id)
            if session:
                session.knowledge_graph = knowledge_graph.to_dict() if knowledge_graph else None
                session.gap_analysis = gaps
                session.learning_plan = plan.to_dict()
                session.user_profile = user_profile.to_dict()
                self.memory.update_session(session)
            
            # Save profile
            self.memory.save_profile(user_profile)
        
        return conversation, plan, session_id
    
    async def process_reference_mode_with_memory(
        self,
        user_id: str,
        source: str,
        user_profile: Optional[UserProfile] = None
    ) -> Tuple[List[Dict], LearningPlan, str]:
        """
        Mode 2 with memory: User provides a reference
        Returns: (conversation_history, learning_plan, session_id)
        """
        
        # Load or create profile from memory
        if not user_profile and self.memory:
            user_profile = self.memory.load_profile(user_id)
        
        if not user_profile:
            user_profile = UserProfile(user_id=user_id)
        
        # Create session in memory
        if self.memory:
            session_id = self.memory.create_session(
                user_id=user_id,
                mode="reference",
                user_profile=user_profile
            )
        else:
            session_id = f"{user_id}_{datetime.now().timestamp()}"
        
        # Extract data
        self._log(f"Extracting data from: {source}")
        extracted_data = await self._extract_data(source)
        
        # Get text content
        text_content = self._get_text_from_extracted(extracted_data)
        
        # Extract concepts
        self._log("Extracting concepts...")
        concepts = await self._extract_concepts(text_content, source)
        
        # Store in session
        if self.memory:
            session = self.memory.get_session(session_id)
            if session:
                session.extracted_data = extracted_data
                session.concepts = concepts
                session.user_profile = user_profile.to_dict()
                self.memory.update_session(session)
        
        # Conduct conversation
        conversation = await self._conduct_discovery_conversation_with_memory(
            session_id,
            concepts,
            mode="reference",
            profile=user_profile,
            extracted_data=extracted_data
        )
        
        # Build knowledge graph
        knowledge_graph = await self._build_knowledge_graph(concepts)
        
        # Identify gaps
        gaps = await self._identify_gaps_with_profile(user_profile, knowledge_graph)
        
        # Generate plan
        plan = await self._generate_learning_plan_from_data(
            user_profile, concepts, gaps, knowledge_graph
        )
        
        # Update session with results
        if self.memory:
            session = self.memory.get_session(session_id)
            if session:
                session.knowledge_graph = knowledge_graph.to_dict() if knowledge_graph else None
                session.gap_analysis = gaps
                session.learning_plan = plan.to_dict()
                session.user_profile = user_profile.to_dict()
                self.memory.update_session(session)
            
            # Save profile
            self.memory.save_profile(user_profile)
        
        return conversation, plan, session_id
    
    # ==================== CONVERSATION METHODS WITH MEMORY ====================
    
    async def _conduct_discovery_conversation_with_memory(
        self,
        session_id: str,
        concepts: Dict,
        mode: str,
        profile: UserProfile,
        extracted_data: Optional[Dict] = None
    ) -> List[Dict]:
        """Conduct adaptive discovery conversation with memory"""
        
        history = []
        
        # Get conversation history from memory if available
        if self.memory:
            session = self.memory.get_session(session_id)
            if session:
                history = session.conversation_history
        
        # Get main topic
        main_topic = concepts.get("main_topic", "this topic")
        
        # Generate intro based on mode
        if mode == "reference":
            source_type = extracted_data.get("type", "document") if extracted_data else "document"
            source_names = {
                "youtube": "YouTube video",
                "website": "website",
                "document": "document",
                "image": "image",
                "error": "reference"
            }
            source_name = source_names.get(source_type, "reference")
            
            intro = f"""📚 I've analyzed your {source_name} about **{main_topic}**.

🎯 Before I create your personalized learning plan, I'd like to understand your background and goals.

Let me ask you a few questions to help me build the perfect curriculum for you.

💡 **Quick Tip:** The more you tell me about what you already know, the better I can customize your learning path."""
        else:
            intro = f"""🎯 I understand you want to learn about **{main_topic}**. That's a great goal!

🧠 Before I create your learning plan, let me understand your current knowledge and what you want to achieve.

💡 **Quick Tip:** Be honest about your experience level—this helps me create the most effective learning path for you."""
        
        # Add intro if history is empty
        if not history:
            history.append({
                "role": "agent",
                "message": intro,
                "timestamp": datetime.now().isoformat()
            })
        
        # Generate questions based on existing profile knowledge
        questions = await self._generate_initial_questions_with_profile(
            session_id, concepts, profile
        )
        
        # Add questions if history is empty
        if not history or len(history) <= 1:
            for question in questions:
                history.append({
                    "role": "agent",
                    "message": question,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Update session
        if self.memory:
            session = self.memory.get_session(session_id)
            if session:
                session.conversation_history = history
                self.memory.update_session(session)
        
        return history
    
    async def _generate_initial_questions_with_profile(
        self,
        session_id: str,
        concepts: Dict,
        profile: UserProfile
    ) -> List[str]:
        """Generate initial questions based on existing profile knowledge"""
        
        main_topic = concepts.get("main_topic", "")
        subtopics = concepts.get("subtopics", [])
        
        questions = []
        
        # Check if we already know some information
        known_topics = [k for k in profile.known_concepts.keys() if k in subtopics]
        
        if not known_topics:
            # General questions
            questions.append(f"📊 How would you rate your current experience with **{main_topic}**? (beginner, intermediate, or advanced)")
        else:
            questions.append(f"📊 I see you already have some knowledge in {', '.join(known_topics[:3])}. How would you rate your overall experience with **{main_topic}**?")
        
        questions.append(f"🎯 What's your primary goal? (understand concepts, build projects, prepare for interviews, or do research)")
        questions.append("⏰ How many hours per week can you dedicate to learning?")
        
        # Ask about unknown subtopics
        unknown_subtopics = [t for t in subtopics[:3] if t.lower() not in [k.lower() for k in profile.known_concepts.keys()]]
        for topic in unknown_subtopics[:3]:
            questions.append(f"🤔 Are you already familiar with **{topic}**? (yes/no/partially)")
        
        questions.append(f"💭 Is there anything specific within **{main_topic}** that you're most interested in or concerned about?")
        
        return questions
    
    async def _identify_gaps_with_profile(
        self,
        profile: UserProfile,
        knowledge_graph: KnowledgeGraph
    ) -> Dict:
        """Identify knowledge gaps using profile"""
        
        return await self.knowledge_gap_agent.analyze_gaps(profile, knowledge_graph)
    
    async def _generate_learning_plan_from_data(
        self,
        profile: UserProfile,
        concepts: Dict,
        gap_analysis: Dict,
        knowledge_graph: KnowledgeGraph
    ) -> LearningPlan:
        """Generate learning plan from existing data with fallback"""
        
        main_topic = concepts.get("main_topic", "Learning Topic")
        goal_type = profile.goals[0] if profile.goals else "understand"
        
        # Extract strengths and gaps - ensure they exist
        strengths = [
            k.get("concept", "") for k in gap_analysis.get("known", [])
        ][:5]
        
        knowledge_gaps = [
            u.get("concept", "") for u in gap_analysis.get("unknown", [])
        ][:10]
        
        # Remove duplicates from knowledge gaps
        knowledge_gaps = list(dict.fromkeys(knowledge_gaps))
        
        # If no gaps found, use the main topic as a gap
        if not knowledge_gaps:
            knowledge_gaps = [main_topic]
        
        # Build roadmap with unique titles - pass gap_analysis directly
        roadmap = await self._build_roadmap_with_data(main_topic, gap_analysis, knowledge_graph)
        
        # If roadmap is still empty, create a fallback
        if not roadmap:
            roadmap = [{
                "phase_number": 1,
                "title": f"📖 Getting Started with {main_topic}",
                "description": f"Learn the fundamentals of {main_topic}",
                "topics": [main_topic],
                "estimated_hours": 4,
                "projects": ["Practice exercises"],
                "difficulty": "beginner"
            }]
        
        # Generate projects with fallback
        projects = await self._generate_projects_with_data(main_topic, gap_analysis)
        if not projects:
            projects = [{
                "title": f"🚀 Getting Started with {main_topic}",
                "description": f"A beginner-friendly project for {main_topic}",
                "difficulty": "beginner",
                "estimated_hours": 4
            }]
        
        # Generate quizzes with fallback
        quizzes = await self._generate_quizzes_with_data(gap_analysis)
        
        # Generate resources with fallback
        resources = await self._generate_resources_with_data(main_topic)
        
        # Calculate confidence scores
        confidence_scores = {}
        for concept in gap_analysis.get("unknown", []):
            confidence_scores[concept.get("concept", "")] = concept.get("confidence", 0.0)
        
        # Calculate estimated time with fallback
        total_hours = sum(
            u.get("estimated_time_hours", 4) 
            for u in gap_analysis.get("unknown", [])
        )
        if total_hours == 0:
            total_hours = 10
        
        return LearningPlan(
            main_topic=main_topic,
            goal=profile.goals[0] if profile.goals else "Master " + main_topic,
            goal_type=goal_type,
            current_level=profile.preferred_depth or "beginner",
            strengths=strengths,
            knowledge_gaps=knowledge_gaps,
            estimated_time_hours=total_hours,
            roadmap=roadmap,
            projects=projects,
            quizzes=quizzes,
            resources=resources,
            confidence_scores=confidence_scores
        )
    
    async def _build_roadmap_with_data(
        self,
        main_topic: str,
        gap_analysis: Dict,
        knowledge_graph: KnowledgeGraph
    ) -> List[Dict]:
        """
        Build roadmap with unique, meaningful phase titles.
        No duplicate names or topics.
        """
        
        unknown = gap_analysis.get("unknown", [])
        partially_known = gap_analysis.get("partially_known", [])
        
        roadmap = []
        phase_num = 1
        used_titles = set()
        used_topics = set()
        
        # Helper to generate unique title
        def get_unique_title(base_title: str, topics: List[str]) -> str:
            title = base_title
            counter = 1
            while title in used_titles:
                # Add the first topic to differentiate
                if topics and counter == 1:
                    title = f"{base_title}: {topics[0]}"
                else:
                    title = f"{base_title} (Part {counter})"
                counter += 1
            used_titles.add(title)
            return title
        
        # Phase 1: Foundation (if there are partially known concepts)
        if partially_known:
            topic_names = [p.get("concept", "") for p in partially_known[:3]]
            topic_names = [t for t in topic_names if t and t not in used_topics]
            
            if topic_names:
                title = "🔧 Strengthening Your Foundation"
                if len(topic_names) >= 2:
                    title = f"🔧 Mastering {topic_names[0]} & {topic_names[1]}"
                elif len(topic_names) == 1:
                    title = f"🔧 Deepening {topic_names[0]}"
                
                roadmap.append({
                    "phase_number": phase_num,
                    "title": get_unique_title(title, topic_names),
                    "description": f"Reinforce your knowledge of {', '.join(topic_names)} to build a solid foundation",
                    "topics": topic_names,
                    "estimated_hours": sum(2 for _ in topic_names),
                    "projects": ["Concept review and practice exercises"],
                    "difficulty": "beginner"
                })
                phase_num += 1
                used_topics.update(topic_names)
        
        # Phases for unknown concepts - group by logical categories
        if unknown:
            # Sort unknown by difficulty or importance (use order from gap analysis)
            # Group concepts into logical phases (3-4 concepts per phase)
            phase_size = 4
            for i in range(0, len(unknown), phase_size):
                chunk = unknown[i:i+phase_size]
                topic_names = [u.get("concept", "") for u in chunk]
                # Filter out empty topics and already used topics
                topic_names = [t for t in topic_names if t and t not in used_topics]
                
                if not topic_names:
                    continue
                
                # Determine phase category based on position
                if i == 0:
                    # First core phase - fundamentals
                    if len(topic_names) >= 3:
                        title = f"📖 Fundamentals: {topic_names[0]}, {topic_names[1]} & {topic_names[2]}"
                    elif len(topic_names) == 2:
                        title = f"📖 Understanding {topic_names[0]} & {topic_names[1]}"
                    else:
                        title = f"📖 Getting Started with {topic_names[0]}"
                elif i >= len(unknown) - phase_size:
                    # Last phase - advanced/integration
                    if len(topic_names) >= 3:
                        title = f"🚀 Advanced: {topic_names[0]}, {topic_names[1]} & {topic_names[2]}"
                    elif len(topic_names) == 2:
                        title = f"🚀 Deep Dive: {topic_names[0]} & {topic_names[1]}"
                    else:
                        title = f"🚀 Advanced {topic_names[0]}"
                else:
                    # Middle phases
                    if len(topic_names) >= 3:
                        title = f"📚 Building Skills: {topic_names[0]}, {topic_names[1]} & {topic_names[2]}"
                    elif len(topic_names) == 2:
                        title = f"📚 Exploring {topic_names[0]} & {topic_names[1]}"
                    else:
                        title = f"📚 Mastering {topic_names[0]}"
                
                # Generate unique title
                unique_title = get_unique_title(title, topic_names)
                
                # Generate description
                if len(topic_names) > 3:
                    description = f"Learn essential concepts including {', '.join(topic_names[:3])}, and more"
                elif len(topic_names) == 2:
                    description = f"Learn {topic_names[0]} and {topic_names[1]} in depth"
                elif len(topic_names) == 1:
                    description = f"Master {topic_names[0]} with practical understanding"
                else:
                    description = f"Learn essential concepts for understanding {main_topic}"
                
                # Calculate hours
                hours = sum(u.get("estimated_time_hours", 4) for u in chunk)
                
                roadmap.append({
                    "phase_number": phase_num,
                    "title": unique_title,
                    "description": description,
                    "topics": topic_names,
                    "estimated_hours": hours if hours > 0 else 4,
                    "projects": [f"Practice: {topic_names[0]}" for _ in range(min(1, len(topic_names)))] if topic_names else [],
                    "difficulty": "intermediate" if i > 0 else "beginner"
                })
                phase_num += 1
                used_topics.update(topic_names)
        
        # Final Phase: Capstone - only if we have enough topics
        if len(used_topics) >= 3:
            # Get the most advanced topics not used yet (or use last few)
            advanced_topics = list(used_topics)[-3:] if len(used_topics) >= 3 else list(used_topics)
            
            # Get actual concept objects for these topics
            advanced_concepts = []
            for topic in advanced_topics:
                for u in unknown:
                    if u.get("concept", "") == topic:
                        advanced_concepts.append(u)
                        break
            
            if advanced_concepts:
                topic_names = [u.get("concept", "") for u in advanced_concepts]
                
                if len(topic_names) >= 3:
                    title = f"🏆 Capstone: {topic_names[0]}, {topic_names[1]} & {topic_names[2]}"
                elif len(topic_names) == 2:
                    title = f"🏆 Capstone: {topic_names[0]} & {topic_names[1]}"
                else:
                    title = f"🏆 Capstone: {topic_names[0] if topic_names else 'Final Project'}"
                
                roadmap.append({
                    "phase_number": phase_num,
                    "title": get_unique_title(title, topic_names),
                    "description": f"Apply everything you've learned by building a comprehensive {main_topic} project",
                    "topics": topic_names,
                    "estimated_hours": 8,
                    "projects": ["🎯 Capstone Project: End-to-end implementation"],
                    "difficulty": "advanced"
                })
                phase_num += 1
        
        # If we still have only one phase, add a meaningful title
        if len(roadmap) == 1 and roadmap[0].get("title") in ["📖 Core Concepts", "📖 Advanced Core Concepts"]:
            roadmap[0]["title"] = f"📖 Getting Started with {main_topic}"
        
        return roadmap
    
    async def _generate_projects_with_data(self, main_topic: str, gap_analysis: Dict) -> List[Dict]:
        """Generate projects with main topic"""
        return [
            {
                "title": f"🚀 Getting Started with {main_topic}",
                "description": f"A beginner-friendly project to understand the basics of {main_topic}",
                "difficulty": "beginner",
                "estimated_hours": 4
            },
            {
                "title": f"💡 {main_topic} Application",
                "description": f"Build a practical application using {main_topic} concepts",
                "difficulty": "intermediate",
                "estimated_hours": 8
            },
            {
                "title": f"🏆 Advanced {main_topic} Project",
                "description": f"Build a production-ready solution with {main_topic}",
                "difficulty": "advanced",
                "estimated_hours": 16
            }
        ]
    
    async def _generate_quizzes_with_data(self, gap_analysis: Dict) -> List[Dict]:
        """Generate quizzes from gaps"""
        unknown = gap_analysis.get("unknown", [])[:3]
        return [{
            "topic": concept["concept"],
            "questions": [
                f"What is the key concept of {concept['concept']}?",
                f"How does {concept['concept']} work in practice?",
                f"When would you use {concept['concept']}?",
            ],
            "difficulty": concept.get("difficulty", "intermediate")
        } for concept in unknown]
    
    async def _generate_resources_with_data(self, main_topic: str) -> List[Dict]:
        """Generate resources with main topic"""
        cleaned_topic = main_topic.lower().replace(" ", "+")
        return [
            {
                "title": f"📚 Official Documentation for {main_topic}",
                "url": f"https://docs.{main_topic.lower().replace(' ', '')}.com",
                "type": "documentation"
            },
            {
                "title": f"🎥 Video Tutorials for {main_topic}",
                "url": f"https://www.youtube.com/results?search_query={cleaned_topic}+tutorial",
                "type": "video"
            },
            {
                "title": f"💻 GitHub Projects for {main_topic}",
                "url": f"https://github.com/search?q={cleaned_topic}",
                "type": "code"
            }
        ]
    
    def continue_conversation_with_memory(
        self,
        session_id: str,
        user_message: str
    ) -> Dict:
        """Continue conversation with memory"""
        
        if not self.memory:
            return {"error": "Memory handler not initialized"}
        
        session = self.memory.get_session(session_id)
        if not session:
            return {"error": f"Session {session_id} not found"}
        
        # Add user message to history
        session.conversation_history.append({
            "role": "user",
            "message": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Process response and generate acknowledgment
        # Extract knowledge from the response
        import asyncio
        asyncio.create_task(self._extract_knowledge_from_response_async(session_id, user_message))
        
        # Add agent acknowledgment
        session.conversation_history.append({
            "role": "agent",
            "message": "✅ I've recorded your response. Let me analyze it to refine your learning plan.",
            "timestamp": datetime.now().isoformat()
        })
        
        self.memory.update_session(session)
        
        return {
            "session_id": session_id,
            "conversation_history": session.conversation_history
        }
    
    async def _extract_knowledge_from_response_async(self, session_id: str, response: str):
        """Async wrapper for extracting knowledge from response"""
        try:
            # Get session
            session = self.memory.get_session(session_id)
            if not session:
                return
            
            # Load profile
            profile = self.memory.load_profile(session.user_id)
            if not profile:
                return
            
            # Extract concepts using LLM
            prompt = f"""
            Parse this user response and extract knowledge information.
            
            Response: {response}
            
            Return a JSON object with:
            - concepts: List of concepts mentioned
            - confidence: How confident they seem (0-1)
            - experience: beginner | intermediate | advanced
            - goal: understand | build | interview | research | teach | certification
            - time_available: hours per week (number)
            """
            
            try:
                result = await self.llm.complete(prompt)
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    
                    # Update profile with concepts
                    for concept in data.get("concepts", []):
                        profile.add_knowledge(
                            concept=concept,
                            confidence=data.get("confidence", 0.5),
                            evidence=[f"User mentioned: {response}"]
                        )
                    
                    # Update goal if specified
                    if data.get("goal") and data.get("goal") not in profile.goals:
                        profile.goals.append(data.get("goal"))
                    
                    # Update preferences
                    if data.get("experience"):
                        profile.preferred_depth = data.get("experience")
                    
                    # Save updated profile
                    self.memory.save_profile(profile)
                    
                    # Update session
                    session.user_profile = profile.to_dict()
                    self.memory.update_session(session)
                    
            except Exception as e:
                self._log(f"Failed to extract knowledge: {e}", "debug")
                
        except Exception as e:
            self._log(f"Error in _extract_knowledge_from_response_async: {e}", "error")
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get a summary of a session"""
        
        if not self.memory:
            return {"error": "Memory handler not initialized"}
        
        session = self.memory.get_session(session_id)
        if not session:
            return {"error": f"Session {session_id} not found"}
        
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "mode": session.mode,
            "created_at": session.created_at,
            "last_accessed": session.last_accessed,
            "message_count": len(session.conversation_history),
            "has_learning_plan": session.learning_plan is not None,
            "topics": session.concepts.get("subtopics", []) if session.concepts else [],
            "main_topic": session.concepts.get("main_topic") if session.concepts else None
        }
    
    # ==================== CORE AGENT METHODS (Original) ====================
    
    async def _extract_data(self, source: str) -> Dict:
        """
        Extract data from source using your DataExtractor
        """
        try:
            result = self.data_extractor.extract(source)
            source_type = result.get("type", "unknown")
            self._log(f"Extracted {source_type} from source")
            return result
        except Exception as e:
            self._log(f"Extraction failed: {e}", "error")
            return {
                "type": "error",
                "data": {
                    "content": f"Failed to extract: {str(e)}"
                },
                "metadata": {
                    "source": source,
                    "error": str(e)
                }
            }
    
    def _get_text_from_extracted(self, extracted_data: Dict) -> str:
        """
        Extract text content from your DataExtractor's output format
        Handles all the different output types from your extractor
        """
        # Get the data section
        data = extracted_data.get("data", {})
        
        # Handle different extraction types
        source_type = extracted_data.get("type", "")
        
        if source_type == "youtube":
            # YouTube transcript - look for content or full_text
            content = data.get("content", "")
            if content:
                return content
            # Fallback to full_text if content is empty
            return data.get("full_text", "")
        
        elif source_type == "website":
            # Website content
            content = data.get("content", "")
            if content:
                return content
            # Fallback to full_text
            return data.get("full_text", "")
        
        elif source_type == "document":
            # Document content
            content = data.get("content", "")
            if content:
                return content
            # Fallback to full_text
            return data.get("full_text", "")
        
        elif source_type == "image":
            # Image extracted text
            content = data.get("content", "")
            if content:
                return content
            return data.get("text", "")
        
        else:
            # Generic fallback
            if isinstance(data, dict):
                # Try common keys
                for key in ["content", "text", "full_text", "transcript"]:
                    if key in data:
                        val = data[key]
                        if isinstance(val, str) and len(val) > 10:
                            return val
                # If all else fails, convert dict to string
                return json.dumps(data, ensure_ascii=False)
            elif isinstance(data, str):
                return data
        
        return ""
    
    async def _extract_concepts(self, text: str, source: str) -> Dict:
        """Extract concepts from text"""
        if not text or len(text.strip()) < 10:
            return {
                "main_topic": self._infer_topic_from_source(source),
                "subtopics": [],
                "domain": "General",
                "difficulty": "intermediate",
                "keywords": []
            }
        return await self.concept_extractor.extract(text)
    
    def _infer_topic_from_source(self, source: str) -> str:
        """Infer topic from source URL or filename"""
        import re
        from pathlib import Path
        
        if source.startswith(('http://', 'https://')):
            match = re.search(r'/([^/]+)/?$', source)
            if match:
                topic = match.group(1)
                topic = re.sub(r'[_-]', ' ', topic)
                topic = re.sub(r'\..+$', '', topic)
                return topic.title()
        else:
            path = Path(source)
            if path.exists():
                topic = path.stem
                topic = re.sub(r'[_-]', ' ', topic)
                return topic.title()
        return "Unknown Topic"
    
    async def _extract_concepts_from_goal(self, goal: str) -> Dict:
        """Extract concepts from a goal statement"""
        prompt = f"""
        Analyze this learning goal and extract the main topics and subtopics:
        
        Goal: {goal}
        
        Return a JSON object with:
        - main_topic: The primary subject
        - subtopics: List of related subtopics
        - domain: The broader field
        - difficulty: beginner | intermediate | advanced
        - keywords: Important keywords
        """
        
        try:
            response = await self.llm.complete(prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        words = goal.split()
        return {
            "main_topic": " ".join(words[:3]) if words else goal,
            "subtopics": [],
            "domain": "General",
            "difficulty": "intermediate",
            "keywords": [w for w in words if len(w) > 3][:5]
        }
    
    async def _conduct_discovery_conversation(
        self,
        session_id: str,
        concepts: Dict,
        mode: str
    ) -> List[Dict]:
        """Conduct adaptive discovery conversation"""
        
        session = self.sessions[session_id]
        profile = session["profile"]
        history = []
        
        main_topic = concepts.get("main_topic", "this topic")
        
        if mode == "reference":
            extracted = session.get("extracted_data", {})
            source_type = extracted.get("type", "document")
            source_names = {
                "youtube": "YouTube video",
                "website": "website",
                "document": "document",
                "image": "image",
                "error": "reference"
            }
            source_name = source_names.get(source_type, "reference")
            
            intro = f"""📚 I've analyzed your {source_name} about **{main_topic}**.

🎯 Before I create your personalized learning plan, I'd like to understand your background and goals.

Let me ask you a few questions to help me build the perfect curriculum for you.

💡 **Quick Tip:** The more you tell me about what you already know, the better I can customize your learning path."""
        else:
            intro = f"""🎯 I understand you want to learn about **{main_topic}**. That's a great goal!

🧠 Before I create your learning plan, let me understand your current knowledge and what you want to achieve.

💡 **Quick Tip:** Be honest about your experience level—this helps me create the most effective learning path for you."""
        
        history.append({
            "role": "agent",
            "message": intro,
            "timestamp": datetime.now().isoformat()
        })
        
        session["history"] = history
        
        questions = await self._generate_initial_questions(session_id, concepts)
        
        for question in questions:
            history.append({
                "role": "agent",
                "message": question,
                "timestamp": datetime.now().isoformat()
            })
        
        session["pending_questions"] = questions
        session["history"] = history
        
        return history
    
    async def _generate_initial_questions(self, session_id: str, concepts: Dict) -> List[str]:
        """Generate initial questions based on concepts"""
        
        session = self.sessions[session_id]
        main_topic = concepts.get("main_topic", "")
        subtopics = concepts.get("subtopics", [])
        
        questions = []
        questions.append(f"📊 How would you rate your current experience with **{main_topic}**? (beginner, intermediate, or advanced)")
        questions.append(f"🎯 What's your primary goal? (understand concepts, build projects, prepare for interviews, or do research)")
        questions.append("⏰ How many hours per week can you dedicate to learning?")
        
        relevant_subtopics = subtopics[:3]
        if relevant_subtopics:
            for topic in relevant_subtopics:
                questions.append(f"🤔 Are you already familiar with **{topic}**? (yes/no/partially)")
        
        questions.append(f"💭 Is there anything specific within **{main_topic}** that you're most interested in or concerned about?")
        
        return questions
    
    async def _process_user_response(self, session_id: str, response: str) -> Dict:
        """Process user response and extract information"""
        
        session = self.sessions[session_id]
        profile = session["profile"]
        history = session["history"]
        
        history.append({
            "role": "user",
            "message": response,
            "timestamp": datetime.now().isoformat()
        })
        
        await self._extract_knowledge_from_response(session_id, response)
        
        if session.get("pending_questions"):
            session["pending_questions"].pop(0)
        
        return {"processed": True}
    
    async def _extract_knowledge_from_response(self, session_id: str, response: str):
        """Extract knowledge information from user response"""
        
        session = self.sessions[session_id]
        profile = session["profile"]
        
        prompt = f"""
        Parse this user response and extract knowledge information.
        
        Response: {response}
        
        Return a JSON object with:
        - concepts: List of concepts mentioned
        - confidence: How confident they seem (0-1)
        - experience: beginner | intermediate | advanced
        - goal: understand | build | interview | research | teach | certification
        - time_available: hours per week (number)
        """
        
        try:
            result = await self.llm.complete(prompt)
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                for concept in data.get("concepts", []):
                    profile.add_knowledge(
                        concept=concept,
                        confidence=data.get("confidence", 0.5),
                        evidence=[f"User mentioned: {response}"]
                    )
                
                if data.get("goal"):
                    profile.goals.append(data.get("goal"))
                
                if data.get("experience"):
                    profile.preferred_depth = data.get("experience")
                
                if data.get("time_available"):
                    hours = float(data.get("time_available"))
                    if hours <= 2:
                        profile.learning_pace = "slow"
                    elif hours <= 5:
                        profile.learning_pace = "moderate"
                    else:
                        profile.learning_pace = "fast"
        except Exception as e:
            self._log(f"Failed to parse response: {e}", "debug")
    
    async def _update_profile_from_conversation(self, session_id: str):
        """Update user profile based on entire conversation"""
        
        session = self.sessions[session_id]
        profile = session["profile"]
        history = session["history"]
        
        user_messages = [h["message"] for h in history if h["role"] == "user"]
        if not user_messages:
            return
        
        combined = "\n".join(user_messages)
        
        prompt = f"""
        Extract all knowledge concepts from these user responses:
        
        {combined}
        
        Return a JSON object mapping concept names to confidence levels (0-1).
        Example: {{"Python": 0.9, "Docker": 0.3, "Statistics": 0.7}}
        """
        
        try:
            response = await self.llm.complete(prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for concept, confidence in data.items():
                    if isinstance(confidence, (int, float)):
                        profile.add_knowledge(
                            concept=concept,
                            confidence=min(1.0, confidence),
                            evidence=["Extracted from conversation"]
                        )
        except Exception as e:
            self._log(f"Failed to update profile: {e}", "debug")
    
    async def _build_knowledge_graph(self, concepts: Dict) -> KnowledgeGraph:
        """Build knowledge graph from concepts"""
        result = await self.dependency_agent.process({"concepts": concepts})
        return result.get("graph", KnowledgeGraph())
    
    async def _identify_gaps(self, session_id: str, knowledge_graph: KnowledgeGraph) -> Dict:
        """Identify knowledge gaps"""
        session = self.sessions[session_id]
        profile = session["profile"]
        return await self.knowledge_gap_agent.analyze_gaps(profile, knowledge_graph)
    
    async def _generate_learning_plan(self, session_id: str) -> LearningPlan:
        """Generate personalized learning plan"""
        
        session = self.sessions[session_id]
        profile = session["profile"]
        concepts = session["concepts"]
        gap_analysis = session["gap_analysis"]
        knowledge_graph = session["knowledge_graph"]
        
        main_topic = concepts.get("main_topic", "Learning Topic")
        goal_type = profile.goals[0] if profile.goals else "understand"
        
        # Extract strengths and gaps
        strengths = [
            k["concept"] for k in gap_analysis.get("known", [])
        ][:5]
        
        knowledge_gaps = [
            u["concept"] for u in gap_analysis.get("unknown", [])
        ][:10]
        
        # Build roadmap - pass concepts for better titles
        roadmap = await self._build_roadmap(session_id, gap_analysis, knowledge_graph)
        
        # Generate projects
        projects = await self._generate_projects(session_id, concepts, gap_analysis)
        
        # Generate quizzes
        quizzes = await self._generate_quizzes(session_id, concepts, gap_analysis)
        
        # Generate resources
        resources = await self._generate_resources(session_id, concepts)
        
        # Calculate confidence scores
        confidence_scores = {}
        for concept in gap_analysis.get("unknown", []):
            confidence_scores[concept["concept"]] = concept.get("confidence", 0.0)
        
        # Calculate estimated time
        total_hours = sum(
            u.get("estimated_time_hours", 4) 
            for u in gap_analysis.get("unknown", [])
        )
        
        return LearningPlan(
            main_topic=main_topic,
            goal=profile.goals[0] if profile.goals else "Master " + main_topic,
            goal_type=goal_type,
            current_level=profile.preferred_depth,
            strengths=strengths,
            knowledge_gaps=knowledge_gaps,
            estimated_time_hours=total_hours or 10,
            roadmap=roadmap,
            projects=projects,
            quizzes=quizzes,
            resources=resources,
            confidence_scores=confidence_scores
        )
    
    async def _build_roadmap(
        self,
        session_id: str,
        gap_analysis: Dict,
        knowledge_graph: KnowledgeGraph
    ) -> List[Dict]:
        """Build learning roadmap with meaningful phase titles"""
        
        # Get the session to access concepts
        session = self.sessions.get(session_id, {})
        concepts = session.get("concepts", {})
        main_topic = concepts.get("main_topic", "the topic")
        
        unknown = gap_analysis.get("unknown", [])
        partially_known = gap_analysis.get("partially_known", [])
        
        roadmap = []
        phase_num = 1
        
        # Phase 1: Foundation (if there are partially known concepts)
        if partially_known:
            # Generate a meaningful title based on the concepts
            topic_names = [p["concept"] for p in partially_known[:2]]
            phase_title = f"🔧 Strengthening {', '.join(topic_names)}" if topic_names else "🔧 Foundation Building"
            
            roadmap.append({
                "phase_number": phase_num,
                "title": phase_title,
                "description": f"Reinforce your existing knowledge of {', '.join(topic_names) if topic_names else 'key concepts'} to build a solid foundation",
                "topics": [p["concept"] for p in partially_known[:3]],
                "estimated_hours": sum(2 for _ in partially_known[:3]),
                "projects": ["Concept review and practice exercises"],
                "difficulty": "beginner"
            })
            phase_num += 1
        
        # Phase 2-3: Core concepts with meaningful names
        if unknown:
            # Group unknown concepts into phases of 3-4
            for i in range(0, len(unknown), 4):
                chunk = unknown[i:i+4]
                
                # Generate meaningful phase title based on topics
                topic_names = [u["concept"] for u in chunk]
                
                if i == 0:
                    # First core phase
                    if len(topic_names) >= 3:
                        phase_title = f"📖 Mastering {topic_names[0]}, {topic_names[1]}, and {topic_names[2]}"
                    elif len(topic_names) == 2:
                        phase_title = f"📖 Understanding {topic_names[0]} and {topic_names[1]}"
                    else:
                        phase_title = f"📖 Getting Started with {topic_names[0] if topic_names else main_topic}"
                else:
                    # Subsequent phases
                    if len(topic_names) >= 3:
                        phase_title = f"🚀 Advanced: {topic_names[0]}, {topic_names[1]}, and {topic_names[2]}"
                    elif len(topic_names) == 2:
                        phase_title = f"🚀 Deep Dive into {topic_names[0]} and {topic_names[1]}"
                    else:
                        phase_title = f"🚀 Exploring {topic_names[0] if topic_names else 'Advanced Concepts'}"
                
                # Generate description
                if len(topic_names) > 3:
                    description = f"Learn essential concepts including {', '.join(topic_names[:3])}, and more"
                else:
                    description = f"Learn essential concepts for understanding {main_topic}"
                
                roadmap.append({
                    "phase_number": phase_num,
                    "title": phase_title,
                    "description": description,
                    "topics": topic_names,
                    "estimated_hours": sum(u.get("estimated_time_hours", 4) for u in chunk),
                    "projects": [f"Practice: {u['concept']}" for u in chunk[:1]] if chunk else [],
                    "difficulty": "intermediate"
                })
                phase_num += 1
        
        # Final Phase: Integration
        if unknown:
            # Get the most advanced topics for capstone
            advanced_topics = unknown[-3:] if len(unknown) >= 3 else unknown
            topic_names = [u["concept"] for u in advanced_topics]
            
            if len(topic_names) >= 3:
                phase_title = f"🏆 Capstone: {topic_names[0]}, {topic_names[1]}, and {topic_names[2]}"
            elif len(topic_names) == 2:
                phase_title = f"🏆 Capstone: {topic_names[0]} and {topic_names[1]}"
            else:
                phase_title = f"🏆 Capstone: {topic_names[0] if topic_names else 'Final Project'}"
            
            roadmap.append({
                "phase_number": phase_num,
                "title": phase_title,
                "description": f"Apply everything you've learned by building a comprehensive {main_topic} project",
                "topics": [u["concept"] for u in advanced_topics],
                "estimated_hours": 8,
                "projects": ["🎯 Capstone Project: End-to-end implementation"],
                "difficulty": "advanced"
            })
        
        return roadmap
    
    async def _generate_projects(self, session_id: str, concepts: Dict, gap_analysis: Dict) -> List[Dict]:
        """Generate project ideas"""
        main_topic = concepts.get("main_topic", "")
        return [
            {
                "title": f"🚀 Getting Started with {main_topic}",
                "description": f"A beginner-friendly project to understand the basics of {main_topic}",
                "difficulty": "beginner",
                "estimated_hours": 4
            },
            {
                "title": f"💡 {main_topic} Application",
                "description": f"Build a practical application using {main_topic} concepts",
                "difficulty": "intermediate",
                "estimated_hours": 8
            },
            {
                "title": f"🏆 Advanced {main_topic} Project",
                "description": f"Build a production-ready solution with {main_topic}",
                "difficulty": "advanced",
                "estimated_hours": 16
            }
        ]
    
    async def _generate_quizzes(self, session_id: str, concepts: Dict, gap_analysis: Dict) -> List[Dict]:
        """Generate quiz questions"""
        unknown = gap_analysis.get("unknown", [])[:3]
        quizzes = []
        for concept in unknown:
            quizzes.append({
                "topic": concept["concept"],
                "questions": [
                    f"What is the key concept of {concept['concept']}?",
                    f"How does {concept['concept']} work in practice?",
                    f"When would you use {concept['concept']}?",
                ],
                "difficulty": concept.get("difficulty", "intermediate")
            })
        return quizzes
    
    async def _generate_resources(self, session_id: str, concepts: Dict) -> List[Dict]:
        """Generate resource recommendations"""
        main_topic = concepts.get("main_topic", "")
        cleaned_topic = main_topic.lower().replace(" ", "+")
        return [
            {
                "title": f"📚 Official Documentation for {main_topic}",
                "url": f"https://docs.{main_topic.lower().replace(' ', '')}.com",
                "type": "documentation"
            },
            {
                "title": f"🎥 Video Tutorials for {main_topic}",
                "url": f"https://www.youtube.com/results?search_query={cleaned_topic}+tutorial",
                "type": "video"
            },
            {
                "title": f"💻 GitHub Projects for {main_topic}",
                "url": f"https://github.com/search?q={cleaned_topic}",
                "type": "code"
            }
        ]
    
    async def _generate_next_question(self, session_id: str) -> str:
        """Generate the next adaptive question"""
        
        session = self.sessions[session_id]
        concepts = session["concepts"]
        pending = session.get("pending_questions", [])
        
        if pending:
            return pending[0]
        
        profile = session["profile"]
        main_topic = concepts.get("main_topic", "")
        gap_analysis = session.get("gap_analysis", {})
        unknown = gap_analysis.get("unknown", [])
        
        if unknown:
            concept = unknown[0]["concept"]
            return f"🔍 Let's dive deeper. Do you have any experience with **{concept}**? If yes, can you tell me more about what you know?"
        
        return f"💭 Are there any specific areas within **{main_topic}** that you're most excited to learn about?"

    async def start_goal_conversation(
        self,
        user_id: str,
        goal_statement: str,
        user_profile: Optional[UserProfile] = None
    ) -> Tuple[List[Dict], List[Dict], str]:
        """
        Start interactive conversation for goal-based learning.
        Returns: (conversation_history, questions, session_id)
        """
        
        # Load or create profile
        self.log_reasoning("Loading user profile...", f"User: {user_id}", "thinking")
        if not user_profile and self.memory:
            user_profile = self.memory.load_profile(user_id)
        if not user_profile:
            user_profile = UserProfile(user_id=user_id)
        
        # Create session
        if self.memory:
            session_id = self.memory.create_session(
                user_id=user_id,
                mode="goal",
                user_profile=user_profile
            )
        else:
            session_id = f"{user_id}_{datetime.now().timestamp()}"
        
        # Extract concepts from goal using LLM
        self.log_reasoning("Analyzing user request...", f"User wants to learn: {goal_statement}", "thinking")
        self.log_reasoning("Extracting concepts...", "Using LLM to identify main topics and subtopics", "thinking")
        self._log(f"Analyzing goal: {goal_statement}")
        concepts = await self._extract_concepts_from_goal(goal_statement)
        self.log_reasoning("Concepts extracted", f"Found: {concepts.get('main_topic', 'Unknown')} with {len(concepts.get('subtopics', []))} subtopics", "success")
        
        # Initialize session with interactive state
        self.sessions[session_id] = {
            "user_id": user_id,
            "mode": "goal",
            "goal_statement": goal_statement,
            "profile": user_profile,
            "history": [],
            "concepts": concepts,
            "knowledge_graph": None,
            "gap_analysis": None,
            "learning_plan": None,
            "phase": "questioning",
            "answers": [],
            "question_index": 0,
            "conversation_complete": False
        }
        
        # Generate dynamic questions using LLM
        self.log_reasoning("Generating personalized questions...", "Creating questions based on your profile", "thinking")
        questions = await self._generate_dynamic_questions(session_id, concepts, user_profile)
        self.sessions[session_id]["questions"] = questions
        self.log_reasoning("Questions ready", f"Generated {len(questions)} questions for you", "success")
        
        # Build knowledge graph
        self.log_reasoning("Building knowledge graph...", "Mapping dependencies between concepts", "thinking")
        knowledge_graph = await self._build_knowledge_graph(concepts)
        self.sessions[session_id]["knowledge_graph"] = knowledge_graph
        self.log_reasoning("Knowledge graph ready", f"Created {len(knowledge_graph.nodes) if knowledge_graph else 0} nodes with dependencies", "success")
        
        # Create intro
        main_topic = concepts.get("main_topic", "this topic")
        intro = f"""🎯 I understand you want to learn about **{main_topic}**. That's a great goal!

🧠 Let me ask you a few questions to understand your background and create the perfect learning plan for you.

💡 **Quick Tip:** Be honest about your knowledge - this helps me build the right path for you!"""
        
        history = [{
            "role": "agent",
            "message": intro,
            "timestamp": datetime.now().isoformat()
        }]
        
        # Add first question
        if questions:
            first_q = questions[0]
            history.append({
                "role": "agent",
                "message": first_q["question"],
                "timestamp": datetime.now().isoformat()
            })
        
        self.sessions[session_id]["history"] = history
        
        return history, questions, session_id

    async def start_reference_conversation(
        self,
        user_id: str,
        source: str,
        user_profile: Optional[UserProfile] = None
    ) -> Tuple[List[Dict], List[Dict], str]:
        """
        Start interactive conversation for reference-based learning.
        Returns: (conversation_history, questions, session_id)
        """
        
        # Load or create profile
        self.log_reasoning("Loading user profile...", f"User: {user_id}", "thinking")
        if not user_profile and self.memory:
            user_profile = self.memory.load_profile(user_id)
        if not user_profile:
            user_profile = UserProfile(user_id=user_id)
        
        # Create session
        if self.memory:
            session_id = self.memory.create_session(
                user_id=user_id,
                mode="reference",
                user_profile=user_profile
            )
        else:
            session_id = f"{user_id}_{datetime.now().timestamp()}"
        
        # Extract data from source
        self.log_reasoning("Extracting source content...", f"Source: {source}", "thinking")
        self.log_reasoning("Parsing content...", "Extracting text from the provided source", "thinking")
        self._log(f"Extracting data from: {source}")
        extracted_data = await self._extract_data(source)
        self.log_reasoning("Content extracted", f"Type: {extracted_data.get('type', 'unknown')}", "success")
        
        # Get text content
        text_content = self._get_text_from_extracted(extracted_data)
        
        # Extract concepts
        self.log_reasoning("Extracting concepts...", "Using LLM to identify main topics from content", "thinking")
        self._log("Extracting concepts...")
        concepts = await self._extract_concepts(text_content, source)
        self.log_reasoning("Concepts extracted", f"Found: {concepts.get('main_topic', 'Unknown')} with {len(concepts.get('subtopics', []))} subtopics", "success")
        
        # Initialize session with interactive state
        self.sessions[session_id] = {
            "user_id": user_id,
            "mode": "reference",
            "source": source,
            "extracted_data": extracted_data,
            "profile": user_profile,
            "history": [],
            "concepts": concepts,
            "knowledge_graph": None,
            "gap_analysis": {},
            "learning_plan": None,
            "phase": "questioning",
            "answers": [],
            "question_index": 0,
            "conversation_complete": False,
            "questions": []
        }
        
        # Build knowledge graph
        self.log_reasoning("Building knowledge graph...", "Mapping dependencies between concepts", "thinking")
        knowledge_graph = await self._build_knowledge_graph(concepts)
        self.sessions[session_id]["knowledge_graph"] = knowledge_graph
        self.log_reasoning("Knowledge graph ready", f"Created {len(knowledge_graph.nodes) if knowledge_graph else 0} nodes with dependencies", "success")
        
        # Generate dynamic questions using LLM (based on the extracted content)
        self.log_reasoning("Generating personalized questions...", "Creating questions based on extracted content", "thinking")
        questions = await self._generate_dynamic_questions_for_reference(
            session_id, concepts, user_profile, extracted_data
        )
        self.sessions[session_id]["questions"] = questions
        self.log_reasoning("Questions ready", f"Generated {len(questions)} questions for you", "success")
        
        # Create intro with source info
        main_topic = concepts.get("main_topic", "this topic")
        source_type = extracted_data.get("type", "document")
        source_names = {
            "youtube": "YouTube video",
            "website": "website",
            "document": "document",
            "image": "image"
        }
        source_name = source_names.get(source_type, "reference")
        
        intro = f"""📚 I've analyzed your {source_name} about **{main_topic}**.

🎯 Let me understand your background and goals to create a personalized learning plan.

💡 **Quick Tip:** Be honest about your knowledge - this helps me build the right path for you!"""
        
        history = [{
            "role": "agent",
            "message": intro,
            "timestamp": datetime.now().isoformat()
        }]
        
        # Add first question
        if questions and len(questions) > 0:
            first_q = questions[0]
            history.append({
                "role": "agent",
                "message": first_q.get("question", ""),
                "timestamp": datetime.now().isoformat()
            })
        
        self.sessions[session_id]["history"] = history
        
        return history, questions, session_id

    async def _generate_dynamic_questions(
        self,
        session_id: str,
        concepts: Dict,
        profile: UserProfile
    ) -> List[Dict]:
        """
        Generate dynamic questions using LLM based on concepts and user profile.
        No predefined templates - everything is generated by the brain.
        """
        main_topic = concepts.get("main_topic", "")
        subtopics = concepts.get("subtopics", [])
        
        prompt = f"""
        You are a learning discovery agent. Based on the topic '{main_topic}' and its subtopics: {', '.join(subtopics[:5]) if subtopics else 'None'}, 
        generate 5-7 personalized questions to understand the user's current knowledge level, goals, and interests.

        The questions should be:
        1. Adaptive based on the topic
        2. Mix of multiple choice and open-ended
        3. Helpful for creating a personalized learning plan
        4. Natural and conversational

        Return a JSON array where each question has:
        - id: unique identifier
        - type: "mcq" or "text"
        - question: the actual question
        - options: list of options (only for mcq type)
        - key: what this question helps understand

        Example format:
        [
            {{
                "id": "exp_1",
                "type": "mcq",
                "question": "How would you rate your experience with neural networks?",
                "options": ["Beginner", "Intermediate", "Advanced", "Expert"],
                "key": "experience_level"
            }}
        ]
        """
        
        try:
            response = await self.llm.complete_with_json(prompt)
            if response and isinstance(response, list):
                return response
        except Exception as e:
            self._log(f"Error generating dynamic questions: {e}", "error")
        
        # Fallback questions if LLM fails
        return [
            {
                "id": "exp_1",
                "type": "mcq",
                "question": f"How would you rate your current experience with {main_topic}?",
                "options": ["Beginner", "Intermediate", "Advanced"],
                "key": "experience_level"
            },
            {
                "id": "goal_1",
                "type": "mcq",
                "question": "What's your primary learning goal?",
                "options": ["Build projects", "Career transition", "Academic research", "Personal interest", "Interview preparation"],
                "key": "learning_goal"
            },
            {
                "id": "time_1",
                "type": "text",
                "question": "How many hours per week can you dedicate to learning?",
                "key": "time_commitment"
            },
            {
                "id": "open_1",
                "type": "text",
                "question": f"Is there anything specific within {main_topic} that excites you or you're concerned about?",
                "key": "specific_interest"
            }
        ]

    async def _generate_dynamic_questions_for_reference(
        self,
        session_id: str,
        concepts: Dict,
        profile: UserProfile,
        extracted_data: Dict
    ) -> List[Dict]:
        """
        Generate dynamic questions based on the reference content.
        """
        main_topic = concepts.get("main_topic", "")
        subtopics = concepts.get("subtopics", [])
        source_type = extracted_data.get("type", "document")
        
        # Custom prompt for reference mode
        prompt = f"""
        You are a learning discovery agent. A user has shared a {source_type} about '{main_topic}'.
        The content covers these topics: {', '.join(subtopics[:5]) if subtopics else 'various aspects'}.
        
        Generate 4-6 personalized questions to understand the user's:
        1. Current knowledge level related to this content
        2. What they hope to learn or achieve
        3. Specific areas they want to focus on
        4. Their background and time commitment
        
        The questions should be:
        1. Specific to the topic
        2. Mix of multiple choice and open-ended
        3. Natural and conversational
        4. Helpful for creating a personalized learning plan

        Return ONLY a JSON array where each question has:
        - id: unique identifier
        - type: "mcq" or "text"
        - question: the actual question
        - options: list of options (only for mcq type)
        - key: what this question helps understand

        Example: [{{"id": "exp_1", "type": "mcq", "question": "How would you rate your experience?", "options": ["Beginner", "Intermediate", "Advanced"], "key": "experience_level"}}]
        """
        
        try:
            if hasattr(self.llm, 'complete_with_json'):
                response = await self.llm.complete_with_json(prompt)
                if response and isinstance(response, list) and len(response) > 0:
                    return response
        except Exception as e:
            self._log(f"Error generating dynamic questions for reference: {e}", "error")
        
        # Fallback questions
        return [
            {
                "id": "exp_1",
                "type": "mcq",
                "question": f"How would you rate your current experience with {main_topic}?",
                "options": ["Beginner", "Intermediate", "Advanced"],
                "key": "experience_level"
            },
            {
                "id": "goal_1",
                "type": "mcq",
                "question": "What's your primary learning goal?",
                "options": ["Understand concepts", "Build projects", "Career transition", "Research", "Interview preparation"],
                "key": "learning_goal"
            },
            {
                "id": "time_1",
                "type": "text",
                "question": "How many hours per week can you dedicate to learning?",
                "key": "time_commitment"
            },
            {
                "id": "focus_1",
                "type": "text",
                "question": f"What specific aspects of {main_topic} are you most interested in?",
                "key": "focus_area"
            }
        ]

    async def process_answer(
        self,
        session_id: str,
        answer: str,
        question_index: int
    ) -> Dict:
        """
        Process user's answer to a question and return next question or plan.
        """
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found", "type": "error"}
        
        questions = session.get("questions", [])
        
        # Store answer
        if "answers" not in session:
            session["answers"] = []
        
        current_q = questions[question_index] if question_index < len(questions) else None
        
        if current_q:
            session["answers"].append({
                "question": current_q,
                "answer": answer,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update profile based on answer
            await self._update_profile_from_answer_llm(session, current_q, answer)
        
        # Check if all questions answered
        if question_index + 1 >= len(questions):
            return await self._finalize_plan(session_id)  # Add await here
        
        # Return next question
        next_q = questions[question_index + 1]
        return {
            "type": "next_question",
            "question": next_q.get("question", ""),
            "options": next_q.get("options", []),
            "question_id": next_q.get("id", ""),
            "progress": f"{question_index + 1}/{len(questions)}"
        }

    async def _update_profile_from_answer_llm(self, session: Dict, question: Dict, answer: str):
        """
        Use LLM to extract knowledge from user's answer and update profile.
        """
        profile = session["profile"]
        
        prompt = f"""
        Analyze this user's answer to extract knowledge information.
        
        Question: {question.get('question')}
        Answer: {answer}
        
        Return a JSON object with:
        - concepts: List of concepts mentioned in the answer
        - confidence: Overall confidence level (0-1)
        - experience: "beginner" | "intermediate" | "advanced"
        - goal: "build" | "career" | "research" | "interest" | "interview"
        - time_available: hours per week (number)
        """
        
        try:
            response = await self.llm.complete_with_json(prompt)
            if response:
                # Update concepts
                for concept in response.get("concepts", []):
                    profile.add_knowledge(
                        concept=concept,
                        confidence=response.get("confidence", 0.5),
                        evidence=[f"User mentioned in answer: {answer[:100]}"]
                    )
                
                # Update goal
                if response.get("goal"):
                    profile.goals.append(response.get("goal"))
                
                # Update experience level
                if response.get("experience"):
                    profile.preferred_depth = response.get("experience")
                
                # Save profile
                if self.memory:
                    self.memory.save_profile(profile)
        except Exception as e:
            self._log(f"Error updating profile from answer: {e}", "debug")

    async def _finalize_plan(self, session_id: str) -> Dict:
        """
        Finalize the conversation and generate learning plan.
        """
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found", "type": "error"}
        
        profile = session.get("profile")
        concepts = session.get("concepts", {})
        knowledge_graph = session.get("knowledge_graph")
        
        if not profile:
            return {"error": "Missing profile", "type": "error"}
        
        # Get gap analysis first
        if knowledge_graph:
            gap_analysis = await self._identify_gaps_with_profile(profile, knowledge_graph)
            session["gap_analysis"] = gap_analysis
        else:
            gap_analysis = session.get("gap_analysis", {})
        
        # Generate final plan using the existing method - properly await it
        try:
            plan = await self._generate_learning_plan_from_data(
                profile, 
                concepts, 
                gap_analysis, 
                knowledge_graph
            )
            
            session["learning_plan"] = plan
            session["conversation_complete"] = True
            
            # Save to memory
            if self.memory:
                mem_session = self.memory.get_session(session_id)
                if mem_session:
                    mem_session.learning_plan = plan.to_dict() if plan else None
                    mem_session.user_profile = profile.to_dict()
                    mem_session.gap_analysis = gap_analysis
                    self.memory.update_session(mem_session)
            
            return {
                "type": "plan_ready",
                "plan": plan.to_dict() if plan else None
            }
        except Exception as e:
            self._log(f"Error generating plan: {e}", "error")
            return {
                "type": "error",
                "error": str(e)
            }
    
    def _enough_information(self, session_id: str) -> bool:
        """Check if we have enough information to generate a plan"""
        session = self.sessions[session_id]
        profile = session["profile"]
        
        if len(profile.known_concepts) < 3:
            return False
        if not profile.goals:
            return False
        if "knowledge_graph" not in session or not session["knowledge_graph"].nodes:
            return False
        return True
    
    def get_session_state(self, session_id: str) -> Dict:
        """Get current session state"""
        session = self.sessions.get(session_id, {})
        return {
            "session_id": session_id,
            "mode": session.get("mode"),
            "phase": session.get("phase"),
            "profile": session.get("profile", {}).to_dict() if session.get("profile") else None,
            "has_plan": session.get("learning_plan") is not None,
            "conversation_length": len(session.get("history", []))
        }
    
    def clear_session(self, session_id: str):
        """Clear session data"""
        if session_id in self.sessions:
            del self.sessions[session_id]