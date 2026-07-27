# D:\Jinvexa\Models\KnowledgeGraph.py

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
import json

@dataclass
class KnowledgeNode:
    """A node in the knowledge graph"""
    id: str
    name: str
    description: str = ""
    difficulty: str = "intermediate"
    prerequisites: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    is_learned: bool = False
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)

class KnowledgeGraph:
    """Represents a graph of knowledge dependencies"""
    
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)
    
    def add_node(self, node: KnowledgeNode):
        """Add a node to the graph"""
        self.nodes[node.id] = node
        self.adjacency[node.id] = set()
        self.reverse_adjacency[node.id] = set()
    
    def add_edge(self, from_id: str, to_id: str):
        """Add a dependency edge: from_id depends on to_id"""
        if from_id in self.nodes and to_id in self.nodes:
            self.adjacency[from_id].add(to_id)
            self.reverse_adjacency[to_id].add(from_id)
    
    def get_prerequisites(self, node_id: str) -> List[str]:
        """Get all prerequisites for a node"""
        if node_id not in self.nodes:
            return []
        return list(self.adjacency.get(node_id, set()))
    
    def get_dependents(self, node_id: str) -> List[str]:
        """Get all nodes that depend on this node"""
        if node_id not in self.nodes:
            return []
        return list(self.reverse_adjacency.get(node_id, set()))
    
    def get_learning_path(self, target_node: str) -> List[str]:
        """Get the learning path (topological order) to reach target"""
        # BFS/DFS to find all prerequisites
        visited = set()
        path = []
        
        def dfs(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            
            # First, learn all prerequisites
            for prereq in self.get_prerequisites(node_id):
                dfs(prereq)
            
            # Then learn this node
            path.append(node_id)
        
        dfs(target_node)
        return path
    
    def get_depth(self, node_id: str) -> int:
        """Get the depth of a node in the dependency tree"""
        max_depth = 0
        for prereq in self.get_prerequisites(node_id):
            max_depth = max(max_depth, 1 + self.get_depth(prereq))
        return max_depth
    
    def to_dict(self) -> Dict:
        return {
            "nodes": {
                node_id: {
                    "id": node.id,
                    "name": node.name,
                    "description": node.description,
                    "difficulty": node.difficulty,
                    "prerequisites": node.prerequisites,
                    "dependencies": node.dependencies,
                    "is_learned": node.is_learned,
                    "confidence": node.confidence,
                    "metadata": node.metadata
                }
                for node_id, node in self.nodes.items()
            },
            "edges": {
                from_id: list(to_set)
                for from_id, to_set in self.adjacency.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "KnowledgeGraph":
        graph = cls()
        
        for node_id, node_data in data.get("nodes", {}).items():
            node = KnowledgeNode(**node_data)
            graph.add_node(node)
        
        for from_id, to_ids in data.get("edges", {}).items():
            for to_id in to_ids:
                graph.add_edge(from_id, to_id)
        
        return graph
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)