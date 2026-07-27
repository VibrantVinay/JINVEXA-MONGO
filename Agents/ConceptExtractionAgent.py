# D:\Jinvexa\Agents\ConceptExtractionAgent.py

import json
import re
from typing import Dict, List, Any, Optional

# Change from relative to absolute imports
from Agents.BaseAgent import BaseAgent


class ConceptExtractionAgent(BaseAgent):
    """
    Agent responsible for extracting concepts, topics, and subtopics
    from any text content.
    """
    
    def __init__(self, llm_client: Any, config: Optional[Dict] = None):
        super().__init__("ConceptExtractionAgent", llm_client)
        self.config = config or {}
        
        # Common stop words for filtering
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'for', 'nor', 'on', 'at', 'to', 'by',
            'in', 'of', 'with', 'without', 'about', 'through', 'during', 'between'
        }
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract concepts from input text"""
        text = input_data.get("text", "")
        
        # Try to get from LLM if available
        if self.llm:
            return await self._extract_with_llm(text)
        else:
            return self._extract_with_regex(text)
    
    async def extract(self, text: str) -> Dict[str, Any]:
        """Extract concepts from text"""
        return await self.process({"text": text})
    
    async def _extract_with_llm(self, text: str) -> Dict[str, Any]:
        """Extract concepts using LLM"""
        
        # Truncate text if too long
        if len(text) > 8000:
            text = text[:8000] + "..."
        
        prompt = f"""
        Analyze the following text and extract all learning concepts, topics, and subtopics.
        
        Text:
        {text}
        
        Return a JSON object with:
        1. main_topic: The primary subject/topic
        2. subtopics: List of all subtopics mentioned
        3. domain: The broader domain/field
        4. difficulty: beginner | intermediate | advanced | expert
        5. keywords: Important keywords and terms
        6. prerequisites: Topics that are assumed knowledge
        7. related_topics: Related topics not directly mentioned
        
        Example output format:
        {{
            "main_topic": "Machine Learning",
            "subtopics": ["Linear Regression", "Neural Networks", "Deep Learning"],
            "domain": "Artificial Intelligence",
            "difficulty": "intermediate",
            "keywords": ["algorithm", "data", "training", "model"],
            "prerequisites": ["Python", "Statistics"],
            "related_topics": ["Data Science", "Big Data"]
        }}
        """
        
        try:
            response = await self.llm.complete(prompt)
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._extract_with_regex(text)
        except Exception as e:
            self._log(f"LLM extraction failed: {e}", "error")
            return self._extract_with_regex(text)
    
    def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        """Extract concepts using regex and heuristics"""
        
        # Simple keyword extraction
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Filter stop words
        concepts = [w for w in words if w.lower() not in self.stop_words]
        
        # Remove duplicates and limit
        concepts = list(dict.fromkeys(concepts))[:20]
        
        return {
            "main_topic": concepts[0] if concepts else "Unknown Topic",
            "subtopics": concepts[1:],
            "domain": "General",
            "difficulty": "intermediate",
            "keywords": concepts[:10],
            "prerequisites": [],
            "related_topics": []
        }