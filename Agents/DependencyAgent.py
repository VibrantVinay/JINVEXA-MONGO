# D:\Jinvexa\Agents\DependencyAgent.py

import json
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

# Change from relative to absolute imports
from Agents.BaseAgent import BaseAgent
from Models.KnowledgeGraph import KnowledgeGraph, KnowledgeNode


class DependencyAgent(BaseAgent):
    """
    Agent responsible for mapping dependencies between concepts.
    Builds a knowledge graph showing what concepts depend on what.
    """
    
    def __init__(self, llm_client: Any, config: Optional[Dict] = None):
        super().__init__("DependencyAgent", llm_client)
        self.config = config or {}
        
        # Pre-defined dependency mappings for common topics
        self.known_dependencies = self._load_known_dependencies()
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build dependency graph from concepts"""
        concepts = input_data.get("concepts", {})
        main_topic = concepts.get("main_topic", "")
        subtopics = concepts.get("subtopics", [])
        
        # Build the graph
        graph = await self.build_dependency_graph(main_topic, subtopics)
        
        return {
            "main_topic": main_topic,
            "graph": graph,
            "nodes": len(graph.nodes),
            "edges": sum(len(edges) for edges in graph.adjacency.values())
        }
    
    async def build_dependency_graph(
        self,
        main_topic: str,
        subtopics: List[str]
    ) -> KnowledgeGraph:
        """Build a complete dependency graph"""
        
        graph = KnowledgeGraph()
        all_concepts = [main_topic] + subtopics
        
        # Add all concepts as nodes
        for concept in all_concepts:
            if concept not in graph.nodes:
                node = KnowledgeNode(
                    id=concept.lower().replace(" ", "_"),
                    name=concept,
                    difficulty="intermediate"
                )
                graph.add_node(node)
        
        # Get dependencies for each concept
        for concept in all_concepts:
            deps = await self._get_dependencies(concept)
            
            # Add dependency nodes
            for dep in deps:
                dep_id = dep.lower().replace(" ", "_")
                if dep_id not in graph.nodes:
                    node = KnowledgeNode(
                        id=dep_id,
                        name=dep,
                        difficulty="intermediate"
                    )
                    graph.add_node(node)
                
                # Add edge: concept depends on dep
                concept_id = concept.lower().replace(" ", "_")
                graph.add_edge(concept_id, dep_id)
                
                # Recursively get deeper dependencies
                if self.config.get("recursive", True):
                    deeper_deps = await self._get_dependencies(dep)
                    for deeper_dep in deeper_deps:
                        deeper_id = deeper_dep.lower().replace(" ", "_")
                        if deeper_id not in graph.nodes:
                            node = KnowledgeNode(
                                id=deeper_id,
                                name=deeper_dep,
                                difficulty="intermediate"
                            )
                            graph.add_node(node)
                        graph.add_edge(dep_id, deeper_id)
        
        return graph
    
    async def _get_dependencies(self, concept: str) -> List[str]:
        """Get dependencies for a concept"""
        
        # Check known dependencies first
        if concept in self.known_dependencies:
            return self.known_dependencies[concept]
        
        # Use LLM if available
        if self.llm:
            return await self._get_dependencies_with_llm(concept)
        
        # Fallback to common patterns
        return self._get_dependencies_heuristic(concept)
    
    async def _get_dependencies_with_llm(self, concept: str) -> List[str]:
        """Get dependencies using LLM"""
        
        prompt = f"""
        What are the prerequisite concepts needed to understand "{concept}"?
        
        Return a list of concepts in JSON array format.
        Example: ["Python", "Statistics", "Linear Algebra"]
        
        Only include direct prerequisites, not everything related.
        """
        
        try:
            response = await self.llm.complete(prompt)
            # Parse JSON list
            deps = json.loads(response)
            if isinstance(deps, list):
                return deps
        except:
            pass
        
        return []
    
    def _get_dependencies_heuristic(self, concept: str) -> List[str]:
        """Get dependencies using heuristics"""
        
        # Common dependency patterns
        patterns = {
            "Machine Learning": ["Python", "Statistics", "Linear Algebra", "Calculus"],
            "Deep Learning": ["Machine Learning", "Python", "Linear Algebra", "Calculus"],
            "Neural Networks": ["Linear Algebra", "Calculus", "Python"],
            "NLP": ["Machine Learning", "Python", "Linguistics"],
            "Computer Vision": ["Machine Learning", "Python", "Image Processing"],
            "Transformers": ["Deep Learning", "Neural Networks", "Attention", "Self-Attention"],
            "LLM": ["Transformers", "Deep Learning", "NLP"],
            "LangGraph": ["Python", "FastAPI", "Agents", "LLMs"],
            "FastAPI": ["Python", "REST", "JSON", "Async Programming"],
            "Docker": ["Linux", "Containers"],
            "Kubernetes": ["Docker", "Linux", "Networking"],
            "React": ["JavaScript", "HTML", "CSS"],
            "Node.js": ["JavaScript"],
            "Python": ["Programming Basics"],
            "Java": ["Programming Basics"],
            "Rust": ["Programming Basics", "Memory Management"],
            "Statistics": ["Mathematics", "Probability"],
            "Linear Algebra": ["Mathematics"],
            "Calculus": ["Mathematics"],
            "Attention": ["Matrix Multiplication", "Vectors", "Softmax"],
            "RAG": ["Vector Databases", "Embeddings", "LLMs", "Retrieval"],
            "Agents": ["LLMs", "Function Calling", "Memory", "Planning"],
        }
        
        return patterns.get(concept, [f"{concept} Basics"])
    
    def _load_known_dependencies(self) -> Dict[str, List[str]]:
        """Load known dependency mappings"""
        return {
            "Machine Learning": ["Python", "Statistics", "Linear Algebra", "Calculus"],
            "Deep Learning": ["Machine Learning", "Python", "Linear Algebra", "Calculus"],
            "Neural Networks": ["Linear Algebra", "Calculus", "Python"],
            "NLP": ["Machine Learning", "Python"],
            "Computer Vision": ["Machine Learning", "Python", "Image Processing"],
            "Transformers": ["Deep Learning", "Attention"],
            "LLM": ["Transformers", "Deep Learning", "NLP"],
            "LangGraph": ["Python", "FastAPI", "Agents", "LLMs"],
            "FastAPI": ["Python", "REST", "JSON"],
            "Docker": ["Linux"],
            "Kubernetes": ["Docker", "Linux"],
            "React": ["JavaScript", "HTML", "CSS"],
            "Python": ["Programming Basics"],
            "Statistics": ["Mathematics", "Probability"],
            "Linear Algebra": ["Mathematics"],
            "Calculus": ["Mathematics"],
            "Attention": ["Matrix Multiplication", "Vectors", "Softmax"],
            "RAG": ["Vector Databases", "Embeddings", "LLMs", "Retrieval"],
            "Agents": ["LLMs", "Function Calling", "Memory", "Planning"],
        }