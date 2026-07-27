# D:\Jinvexa\Agents\BaseAgent.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name: str, llm_client: Optional[Any] = None):
        self.name = name
        self.llm = llm_client
        self.logger = logging.getLogger(f"Agent.{name}")
        self.reasoning_log: List[Dict] = []
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return output"""
        pass
    
    def _log(self, message: str, level: str = "info"):
        """Log a message"""
        if level == "info":
            self.logger.info(f"[{self.name}] {message}")
        elif level == "error":
            self.logger.error(f"[{self.name}] {message}")
        elif level == "debug":
            self.logger.debug(f"[{self.name}] {message}")
    
    def log_reasoning(self, step: str, detail: str = "", status: str = "info", indent: int = 1) -> Dict:
        """
        Log a reasoning step with structured [INFO] format.
        
        Output: [INFO] Agent.AgentName: Step: Detail
        
        Status: "info", "success", "warning", "error"
        Stores structured data in reasoning_log for future GUI use.
        Does NOT change agent behavior - only logs what the agent is doing.
        """
        # Build log message
        log_message = step
        if detail and detail.strip():
            log_message = f"{step}: {detail}"
        
        # Log to console with structured format
        if status == "info":
            self.logger.info(log_message)
        elif status == "success":
            self.logger.info(f"[OK] {log_message}")
        elif status == "warning":
            self.logger.warning(f"[WARN] {log_message}")
        elif status == "error":
            self.logger.error(f"[ERR] {log_message}")
        else:
            self.logger.info(log_message)
        
        # Store structured data for future GUI use
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "level": status,
            "step": step,
            "detail": detail,
            "message": log_message
        }
        self.reasoning_log.append(entry)
        
        return entry
    
    def get_reasoning_log(self) -> List[Dict]:
        """Get all reasoning steps."""
        return self.reasoning_log
    
    def clear_reasoning_log(self):
        """Clear reasoning log."""
        self.reasoning_log = []