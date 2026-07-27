# D:\Jinvexa\Agents\KnowledgeGapAgent.py

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass

# Change from relative to absolute imports
from Agents.BaseAgent import BaseAgent
from Models.UserProfile import UserProfile
from Models.KnowledgeGraph import KnowledgeGraph


@dataclass
class KnowledgeGap:
    concept: str
    confidence: float
    required: bool
    dependencies: List[str]
    difficulty: str
    estimated_time_hours: int


class KnowledgeGapAgent(BaseAgent):
    """
    Agent responsible for identifying knowledge gaps by comparing
    user profile with required knowledge.
    """
    
    def __init__(self, llm_client: Any, config: Optional[Dict] = None):
        super().__init__("KnowledgeGapAgent", llm_client)
        self.config = config or {}
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify knowledge gaps"""
        user_profile = input_data.get("user_profile")
        knowledge_graph = input_data.get("knowledge_graph")
        
        if not user_profile or not knowledge_graph:
            return {
                "error": "Missing user_profile or knowledge_graph",
                "gaps": [],
                "known": [],
                "unknown": []
            }
        
        return await self.analyze_gaps(user_profile, knowledge_graph)
    
    async def analyze_gaps(
        self,
        user_profile: UserProfile,
        knowledge_graph: KnowledgeGraph
    ) -> Dict[str, Any]:
        """Analyze knowledge gaps between user and required knowledge"""
        
        required_concepts = list(knowledge_graph.nodes.keys())
        
        known_concepts = []
        unknown_concepts = []
        partially_known_concepts = []
        
        for concept_id in required_concepts:
            node = knowledge_graph.nodes.get(concept_id)
            if not node:
                continue
            
            concept_name = node.name
            confidence = user_profile.get_confidence(concept_name)
            
            if confidence >= 0.7:
                known_concepts.append({
                    "concept": concept_name,
                    "confidence": confidence,
                    "node_id": concept_id
                })
            elif confidence >= 0.3:
                partially_known_concepts.append({
                    "concept": concept_name,
                    "confidence": confidence,
                    "node_id": concept_id
                })
            else:
                unknown_concepts.append({
                    "concept": concept_name,
                    "confidence": confidence,
                    "node_id": concept_id,
                    "dependencies": node.prerequisites,
                    "difficulty": node.difficulty
                })
        
        # Estimate learning time for gaps
        for gap in unknown_concepts:
            gap["estimated_time_hours"] = await self._estimate_learning_time(
                gap["concept"],
                gap["difficulty"]
            )
        
        return {
            "known": known_concepts,
            "partially_known": partially_known_concepts,
            "unknown": unknown_concepts,
            "total_required": len(required_concepts),
            "known_count": len(known_concepts),
            "partially_known_count": len(partially_known_concepts),
            "unknown_count": len(unknown_concepts),
            "completion_percentage": (len(known_concepts) / len(required_concepts)) * 100 if required_concepts else 0
        }
    
    async def _estimate_learning_time(self, concept: str, difficulty: str) -> int:
        """Estimate learning time in hours"""
        
        # Time estimates based on difficulty
        time_map = {
            "beginner": 2,
            "intermediate": 4,
            "advanced": 8,
            "expert": 15
        }
        
        return time_map.get(difficulty, 4)
    
    async def generate_gap_summary(self, gap_analysis: Dict) -> str:
        """Generate a human-readable summary of gaps"""
        
        known = gap_analysis.get("known", [])
        partial = gap_analysis.get("partially_known", [])
        unknown = gap_analysis.get("unknown", [])
        
        summary = f"""
Knowledge Gap Analysis:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Already Known: {len(known)} concepts
   {', '.join([k['concept'] for k in known])}

⚠️ Partially Known: {len(partial)} concepts (needs reinforcement)
   {', '.join([p['concept'] for p in partial])}

❌ Need to Learn: {len(unknown)} concepts
   {', '.join([u['concept'] for u in unknown])}

Completion: {gap_analysis.get('completion_percentage', 0):.1f}%
"""
        
        if unknown:
            summary += f"\nEstimated time for gaps: {sum(u.get('estimated_time_hours', 0) for u in unknown)} hours"
        
        return summary