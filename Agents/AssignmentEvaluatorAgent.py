# D:\Jinvexa\Agents\AssignmentEvaluatorAgent.py

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from Agents.BaseAgent import BaseAgent


class AssignmentEvaluatorAgent(BaseAgent):
    """
    Evaluates assignment answers and calculates scores.
    Auto-grades MCQ and uses LLM for written answers.
    """

    def __init__(
        self,
        llm_client: Any,
        memory_handler: Any,
        assignment_generator: Any,
        config: Optional[Dict] = None
    ):
        super().__init__("AssignmentEvaluatorAgent", llm_client)
        
        self.llm_client = llm_client
        self.memory = memory_handler
        self.assignment_generator = assignment_generator
        self.config = config or {}
        
        # Storage directories
        self.results_dir = Path("learn_files/assignments/results")
        self.results_dir.mkdir(exist_ok=True)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and evaluate an assignment."""
        assignment_id = input_data.get("assignment_id", "")
        user_answers = input_data.get("user_answers", {})
        user_id = input_data.get("user_id", "user_1")
        return await self.evaluate_assignment(assignment_id, user_answers, user_id)

    async def evaluate_assignment(
        self,
        assignment_id: str,
        user_answers: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Evaluate user's answers for an assignment.
        """
        # Get assignment
        assignment = self.assignment_generator.get_assignment(assignment_id)
        
        if not assignment:
            return {"error": f"Assignment {assignment_id} not found"}
        
        # Get questions
        mcq_questions = assignment.get("questions", {}).get("mcq", [])
        written_questions = assignment.get("questions", {}).get("written", [])
        
        # Evaluate MCQ answers
        self.log_reasoning("Starting evaluation...", f"Evaluating assignment: {assignment_id[:20]}...", "thinking")
        self.log_reasoning("Checking MCQ answers...", f"Checking {len(mcq_questions)} multiple-choice questions", "thinking")
        mcq_results = self._evaluate_mcq(mcq_questions, user_answers)
        self.log_reasoning("MCQ evaluation complete", f"Score: {mcq_results.get('correct', 0)}/{mcq_results.get('total', 0)}", "success")
        
        # Evaluate written answers
        self.log_reasoning("Evaluating written answers...", f"Using LLM to grade {len(written_questions)} written answers", "thinking")
        written_results = await self._evaluate_written(
            written_questions, 
            user_answers
        )
        self.log_reasoning("Written evaluation complete", f"Score: {written_results.get('score', 0)}/{written_results.get('total', 0)}", "success")
        
        # Calculate scores
        total_score, max_score, grade = self._calculate_scores(
            mcq_results, 
            written_results
        )
        
        # Generate feedback
        feedback = self._generate_feedback(
            mcq_results,
            written_results,
            total_score,
            grade
        )
        
        # Build result
        result = {
            "assignment_id": assignment_id,
            "user_id": user_id,
            "session_id": assignment.get("session_id", ""),
            "evaluated_at": datetime.now().isoformat(),
            "scores": {
                "mcq": mcq_results,
                "written": written_results,
                "total": {
                    "score": total_score,
                    "max_score": max_score,
                    "percentage": round((total_score / max_score) * 100, 2) if max_score > 0 else 0,
                    "grade": grade
                }
            },
            "feedback": feedback,
            "wrong_answers": self._get_wrong_answers(mcq_results, written_results),
            "improvement_areas": self._get_improvement_areas(mcq_results, written_results)
        }
        
        # Save result
        self._save_result(user_id, assignment_id, result)
        
        self.log_reasoning("Evaluation complete", f"Grade: {grade}, Score: {round((total_score / max_score) * 100, 2) if max_score > 0 else 0}%", "success")
        
        return result

    def _evaluate_mcq(
        self,
        mcq_questions: List[Dict],
        user_answers: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate MCQ answers.
        """
        results = {
            "correct": 0,
            "total": len(mcq_questions),
            "details": []
        }
        
        for question in mcq_questions:
            q_id = question.get("id", "")
            correct = question.get("correct_answer", -1)
            user_choice = user_answers.get(q_id, -1)
            
            is_correct = user_choice == correct
            
            results["details"].append({
                "id": q_id,
                "question": question.get("question", ""),
                "user_answer": user_choice,
                "correct_answer": correct,
                "is_correct": is_correct,
                "explanation": question.get("explanation", "")
            })
            
            if is_correct:
                results["correct"] += 1
        
        results["percentage"] = round((results["correct"] / results["total"]) * 100, 2) if results["total"] > 0 else 0
        
        return results

    async def _evaluate_written(
        self,
        written_questions: List[Dict],
        user_answers: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate written/essay answers using LLM.
        """
        results = {
            "score": 0,
            "total": 0,
            "details": []
        }
        
        for question in written_questions:
            q_id = question.get("id", "")
            user_answer = user_answers.get(q_id, "")
            max_score = question.get("max_score", 10)
            rubric = question.get("rubric", {})
            
            results["total"] += max_score
            
            # Use LLM to evaluate if user provided an answer
            if user_answer and len(user_answer.strip()) > 10:
                score, feedback = await self._evaluate_written_with_llm(
                    question=question.get("question", ""),
                    user_answer=user_answer,
                    max_score=max_score,
                    rubric=rubric
                )
            else:
                score = 0
                feedback = "No answer provided."
            
            results["details"].append({
                "id": q_id,
                "question": question.get("question", ""),
                "user_answer": user_answer,
                "score": score,
                "max_score": max_score,
                "feedback": feedback
            })
            
            results["score"] += score
        
        results["percentage"] = round((results["score"] / results["total"]) * 100, 2) if results["total"] > 0 else 0
        
        return results

    async def _evaluate_written_with_llm(
        self,
        question: str,
        user_answer: str,
        max_score: int,
        rubric: Dict
    ) -> Tuple[int, str]:
        """
        Use LLM to evaluate a written answer.
        """
        prompt = f"""
You are a expert grader. Evaluate the following student answer.

Question: {question}
Student Answer: {user_answer}

Grading Rubric:
- Clarity (max {rubric.get('clarity', 3)}): How clear and well-structured is the answer?
- Correctness (max {rubric.get('correctness', 4)}): Is the information correct?
- Examples (max {rubric.get('examples', 3)}): Are relevant examples provided?

Return a JSON object with:
{{
    "score": total_score (0-{max_score}),
    "feedback": "Detailed feedback for the student"
}}

JSON:
"""
        
        try:
            response = await self.llm_client.complete_with_json(prompt)
            
            if response:
                score = response.get("score", 0)
                feedback = response.get("feedback", "Good effort!")
                return score, feedback
        except Exception as e:
            print(f"❌ Written evaluation error: {e}")
        
        # Fallback: simple evaluation
        word_count = len(user_answer.split())
        if word_count > 50:
            score = max_score
        elif word_count > 20:
            score = max_score // 2
        else:
            score = 0
        
        return score, f"Score based on length. {score}/{max_score}"

    def _calculate_scores(
        self,
        mcq_results: Dict,
        written_results: Dict
    ) -> Tuple[int, int, str]:
        """
        Calculate total scores and grade.
        """
        mcq_score = mcq_results.get("correct", 0)
        mcq_total = mcq_results.get("total", 1)
        
        written_score = written_results.get("score", 0)
        written_total = written_results.get("total", 1)
        
        total_score = mcq_score + written_score
        max_score = mcq_total + written_total
        
        if max_score == 0:
            return 0, 0, "N/A"
        
        percentage = (total_score / max_score) * 100
        
        if percentage >= 90:
            grade = "A"
        elif percentage >= 80:
            grade = "B"
        elif percentage >= 70:
            grade = "C"
        elif percentage >= 60:
            grade = "D"
        else:
            grade = "F"
        
        return total_score, max_score, grade

    def _generate_feedback(
        self,
        mcq_results: Dict,
        written_results: Dict,
        total_score: int,
        grade: str
    ) -> Dict:
        """
        Generate overall feedback.
        """
        return {
            "overall": f"Your score: {grade}. You answered {mcq_results.get('correct', 0)}/{mcq_results.get('total', 0)} MCQ questions correctly.",
            "mcq": f"MCQ Score: {mcq_results.get('percentage', 0)}%",
            "written": f"Written Score: {written_results.get('percentage', 0)}%",
            "recommendation": self._get_recommendation(grade)
        }

    def _get_recommendation(self, grade: str) -> str:
        """Get recommendation based on grade."""
        recommendations = {
            "A": "Excellent work! You have a strong understanding of the material.",
            "B": "Good job! Review a few key concepts to reach mastery.",
            "C": "Satisfactory. Consider reviewing the lessons and trying again.",
            "D": "Needs improvement. We recommend reviewing the course content.",
            "F": "Please review the material thoroughly and try the assignment again."
        }
        return recommendations.get(grade, "Keep learning!")

    def _get_wrong_answers(self, mcq_results: Dict, written_results: Dict) -> List[Dict]:
        """Get list of wrong answers with explanations."""
        wrong = []
        
        for detail in mcq_results.get("details", []):
            if not detail.get("is_correct", True):
                topic = detail.get("topic", "")
                if not topic or topic == "Unknown":
                    topic = "General Concept"
                
                # Get the correct answer text
                correct_idx = detail.get("correct_answer", -1)
                options = detail.get("options", [])
                correct_text = options[correct_idx] if 0 <= correct_idx < len(options) else str(correct_idx)
                
                wrong.append({
                    "type": "mcq",
                    "question": detail.get("question", ""),
                    "user_answer": detail.get("user_answer", ""),
                    "correct_answer": correct_text,
                    "explanation": detail.get("explanation", ""),
                    "topic": topic
                })
        
        for detail in written_results.get("details", []):
            if detail.get("score", 0) < detail.get("max_score", 10) * 0.6:
                topic = detail.get("topic", "")
                if not topic or topic == "Unknown":
                    topic = "Written Response"
                
                wrong.append({
                    "type": "written",
                    "question": detail.get("question", ""),
                    "score": detail.get("score", 0),
                    "max_score": detail.get("max_score", 10),
                    "feedback": detail.get("feedback", ""),
                    "topic": topic
                })
        
        return wrong

    def _get_improvement_areas(self, mcq_results: Dict, written_results: Dict) -> List[str]:
        """
        Get areas for improvement with proper topic names.
        """
        areas = []
        topic_count = {}
        
        # Check MCQ weak areas
        for detail in mcq_results.get("details", []):
            if not detail.get("is_correct", True):
                topic = detail.get("topic", "")
                
                # If topic is empty or "Unknown", try to extract from question
                if not topic or topic == "Unknown":
                    question = detail.get("question", "")
                    # Try to extract topic from question context
                    for word in question.split():
                        if len(word) > 5 and word.istitle():
                            topic = word
                            break
                    if not topic or topic == "Unknown":
                        topic = "General Concept"
                
                if topic not in topic_count:
                    topic_count[topic] = 0
                topic_count[topic] += 1
        
        # Check written weak areas
        for detail in written_results.get("details", []):
            if detail.get("score", 0) < detail.get("max_score", 10) * 0.6:
                topic = detail.get("topic", "")
                
                if not topic or topic == "Unknown":
                    question = detail.get("question", "")
                    # Try to extract topic from question
                    for word in question.split():
                        if len(word) > 5 and word.istitle():
                            topic = word
                            break
                    if not topic or topic == "Unknown":
                        topic = "Written Response"
                
                if topic not in topic_count:
                    topic_count[topic] = 0
                topic_count[topic] += 1
        
        # Sort by frequency and create recommendations
        sorted_topics = sorted(topic_count.items(), key=lambda x: x[1], reverse=True)
        
        for topic, count in sorted_topics[:5]:
            if topic and topic != "Unknown" and topic != "General Concept":
                areas.append(f"Review: {topic} ({count} incorrect)")
            else:
                # Try to make it more meaningful
                if topic == "General Concept":
                    areas.append("Review: Core Concepts (multiple incorrect)")
                else:
                    areas.append(f"Review: {topic}")
        
        # If no areas found, add a default message
        if not areas:
            areas.append("Great job! No major areas for improvement.")
        
        return areas[:5]

    def _save_result(self, user_id: str, assignment_id: str, result: Dict):
        """Save evaluation result to file. Delegates to MemoryHandler."""
        self.memory.save_assignment_result(user_id, assignment_id, result)