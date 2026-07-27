# D:\Jinvexa\Models\UserProfile.py

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

@dataclass
class UserProfile:
    """Stores comprehensive user knowledge profile with confidence tracking"""
    
    user_id: str
    known_concepts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    learning_history: List[Dict] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    assignments: List[Dict] = field(default_factory=list)
    preferred_depth: str = "intermediate"  # overview, beginner, intermediate, professional, research
    preferred_language: str = "en"
    learning_pace: str = "moderate"  # slow, moderate, fast
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_knowledge(self, concept: str, confidence: float, evidence: List[str]):
        """Add or update knowledge about a concept"""
        concept_key = concept.lower().strip()
        self.known_concepts[concept_key] = {
            "concept": concept,
            "confidence": min(1.0, max(0.0, confidence)),
            "evidence": evidence,
            "last_updated": datetime.now().isoformat()
        }
        self.updated_at = datetime.now().isoformat()
    
    def get_confidence(self, concept: str) -> float:
        """Get confidence level for a concept, default 0.0 if unknown"""
        concept_key = concept.lower().strip()
        return self.known_concepts.get(concept_key, {}).get("confidence", 0.0)
    
    def is_concept_known(self, concept: str, threshold: float = 0.6) -> bool:
        """Check if concept is known with confidence above threshold"""
        return self.get_confidence(concept) >= threshold
    
    def get_known_concepts_list(self, min_confidence: float = 0.5) -> List[str]:
        """Get list of known concepts with confidence above threshold"""
        return [
            data["concept"] 
            for data in self.known_concepts.values() 
            if data["confidence"] >= min_confidence
        ]
    
    def get_weak_concepts(self, max_confidence: float = 0.4) -> List[str]:
        """Get list of weakly known concepts"""
        return [
            data["concept"] 
            for data in self.known_concepts.values() 
            if data["confidence"] <= max_confidence
        ]
    
    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "known_concepts": self.known_concepts,
            "learning_history": self.learning_history,
            "goals": self.goals,
            "assignments": self.assignments,
            "preferred_depth": self.preferred_depth,
            "preferred_language": self.preferred_language,
            "learning_pace": self.learning_pace,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "UserProfile":
        return cls(**data)
    
    def save(self, filepath: str):
        """Save profile to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "UserProfile":
        """Load profile from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)