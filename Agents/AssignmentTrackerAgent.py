# D:\Jinvexa\Agents\AssignmentTrackerAgent.py

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from Agents.BaseAgent import BaseAgent


class AssignmentTrackerAgent(BaseAgent):
    """
    Tracks user progress across assignments.
    """

    def __init__(
        self,
        llm_client: Any,
        memory_handler: Any,
        config: Optional[Dict] = None
    ):
        super().__init__("AssignmentTrackerAgent", llm_client)
        
        self.llm_client = llm_client
        self.memory = memory_handler
        self.config = config or {}
        
        self.results_dir = Path("learn_files/assignments/results")
        self.progress_dir = Path("learn_files/assignments/progress")
        self.progress_dir.mkdir(exist_ok=True)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return user progress."""
        user_id = input_data.get("user_id", "user_1")
        action = input_data.get("action", "progress")
        if action == "progress":
            return self.get_user_progress(user_id)
        elif action == "certificate":
            return self.get_certificate_eligibility(user_id)
        elif action == "history":
            limit = input_data.get("limit", 10)
            return {"history": self.get_assignment_history(user_id, limit)}
        elif action == "weak_areas":
            return {"weak_areas": self.get_weak_areas(user_id)}
        return {"error": f"Unknown action: {action}"}

    def get_user_progress(self, user_id: str) -> Dict[str, Any]:
        """
        Get complete progress for a user.
        """
        # Get profile from memory
        profile = None
        if self.memory:
            profile = self.memory.load_profile(user_id)
        
        # Get all results
        results = self._get_user_results(user_id)
        
        if not results:
            return {
                "user_id": user_id,
                "total_assignments": 0,
                "average_score": 0,
                "grades": [],
                "performance": "No assignments completed yet"
            }
        
        # Calculate stats
        scores = [r.get("scores", {}).get("total", {}).get("percentage", 0) for r in results]
        grades = [r.get("scores", {}).get("total", {}).get("grade", "N/A") for r in results]
        
        avg_score = sum(scores) / len(scores) if scores else 0
        latest = results[0] if results else {}
        
        # Calculate improvement trend
        trend = self._calculate_trend(scores)
        
        return {
            "user_id": user_id,
            "total_assignments": len(results),
            "average_score": round(avg_score, 2),
            "best_score": max(scores) if scores else 0,
            "latest_score": scores[0] if scores else 0,
            "latest_grade": grades[0] if grades else "N/A",
            "grades": grades[:5],
            "trend": trend,
            "performance": self._get_performance_level(avg_score),
            "recommendations": self._get_recommendations(avg_score, latest),
            "history": results[:5]  # Last 5 assignments
        }

    def get_assignment_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """
        Get assignment history for a user.
        """
        results = self._get_user_results(user_id)
        return results[:limit]

    def get_weak_areas(self, user_id: str) -> List[str]:
        """
        Identify weak areas based on assignment results.
        """
        results = self._get_user_results(user_id)
        weak_areas = []
        
        for result in results:
            improvement_areas = result.get("improvement_areas", [])
            weak_areas.extend(improvement_areas)
        
        # Count occurrences
        from collections import Counter
        counter = Counter(weak_areas)
        
        # Return most frequent weak areas
        return [area for area, count in counter.most_common(5)]

    def _get_user_results(self, user_id: str) -> List[Dict]:
        """
        Get all results for a user. Delegates to MemoryHandler.
        """
        return self.memory.get_user_results(user_id) if self.memory else []

    def _calculate_trend(self, scores: List[float]) -> str:
        """
        Calculate performance trend.
        """
        if len(scores) < 2:
            return "Insufficient data"
        
        recent_avg = sum(scores[:3]) / min(3, len(scores)) if len(scores) >= 3 else scores[0]
        older_avg = sum(scores[3:]) / max(1, len(scores) - 3) if len(scores) > 3 else scores[-1]
        
        if recent_avg > older_avg + 10:
            return "📈 Improving"
        elif recent_avg < older_avg - 10:
            return "📉 Needs attention"
        else:
            return "➡️ Consistent"

    def _get_performance_level(self, avg_score: float) -> str:
        """
        Get performance level based on average score.
        """
        if avg_score >= 85:
            return "🌟 Excellent"
        elif avg_score >= 70:
            return "👍 Good"
        elif avg_score >= 55:
            return "📚 Satisfactory"
        else:
            return "📖 Needs Improvement"

    def _get_recommendations(self, avg_score: float, latest: Dict) -> List[str]:
        """
        Generate recommendations based on performance.
        """
        recommendations = []
        
        if avg_score >= 85:
            recommendations.append("You're doing great! Consider taking more advanced courses.")
        elif avg_score >= 70:
            recommendations.append("Good progress! Review concepts you scored low on.")
        else:
            recommendations.append("We recommend reviewing the course content and trying again.")
        
        # Add specific recommendations from latest assignment
        if latest:
            feedback = latest.get("feedback", {})
            if feedback.get("recommendation"):
                recommendations.append(feedback.get("recommendation"))
        
        # Add weak areas
        weak_areas = latest.get("improvement_areas", [])
        if weak_areas:
            recommendations.append(f"Focus on: {', '.join(weak_areas[:3])}")
        
        return recommendations[:5]

    def get_certificate_eligibility(self, user_id: str) -> Dict[str, Any]:
        """
        Check if user is eligible for a certificate.
        """
        progress = self.get_user_progress(user_id)
        
        total_assignments = progress.get("total_assignments", 0)
        avg_score = progress.get("average_score", 0)
        
        # Requirements: At least 3 assignments with average >= 70%
        eligible = total_assignments >= 3 and avg_score >= 70
        
        return {
            "eligible": eligible,
            "total_assignments": total_assignments,
            "average_score": avg_score,
            "requirement": "3 assignments with average >= 70%",
            "status": "✅ Eligible" if eligible else "❌ Not yet eligible"
        }

    def save_progress_summary(self, user_id: str):
        """
        Save progress summary to file. Delegates to MemoryHandler.
        """
        progress = self.get_user_progress(user_id)
        return self.memory.save_progress_summary(user_id, progress) if self.memory else None