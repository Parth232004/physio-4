"""
Physio Intelligence v3 - Exercise Library
Day 2: Comprehensive exercise metrics and form analysis

Exercise Library:
- Shoulder raises
- Shoulder rotations
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime
try:
    from fpdf import FPDF
except ImportError:
    FPDF = None


class ExerciseType(Enum):
    """Supported exercise types - Upper body only."""
    SHOULDER_RAISE = "shoulder_raise"
    SHOULDER_ROTATION = "shoulder_rotation"


@dataclass
class AngleRange:
    """Angle range for a joint during exercise."""
    min_angle: float
    max_angle: float
    ideal_angle: float
    unit: str = "degrees"


@dataclass
class JointThreshold:
    """Joint position thresholds for proper form."""
    joint_name: str
    landmark_indices: Tuple[int, int]
    required_range: AngleRange
    warnings: List[str] = field(default_factory=list)


@dataclass
class FormMetrics:
    """Form metrics for an exercise."""
    exercise_name: str
    exercise_type: ExerciseType
    primary_joints: List[str]
    angle_thresholds: Dict[str, AngleRange]
    joint_thresholds: List[JointThreshold]
    phase_definitions: Dict[str, Dict]
    common_mistakes: Dict[str, List[str]]
    recommended_reps: int = 10
    recommended_sets: int = 3
    rest_between_sets: int = 60


class ExerciseLibrary:
    """
    Comprehensive exercise library with proper form metrics.
    Stores and provides exercise configurations for physiotherapy tracking.
    """
    
    def __init__(self):
        self.exercises: Dict[ExerciseType, FormMetrics] = {}
        self._initialize_exercise_library()
    
    def _initialize_exercise_library(self):
        """Initialize the exercise library with all exercises."""
        
        # Shoulder Raise (existing exercise)
        self.exercises[ExerciseType.SHOULDER_RAISE] = FormMetrics(
            exercise_name="Shoulder Raise",
            exercise_type=ExerciseType.SHOULDER_RAISE,
            primary_joints=["left_shoulder", "left_elbow", "left_wrist"],
            angle_thresholds={
                "shoulder_flexion": AngleRange(0, 180, 90, "degrees"),
                "elbow_flexion": AngleRange(160, 180, 170, "degrees")
            },
            joint_thresholds=[
                JointThreshold(
                    joint_name="left_arm",
                    landmark_indices=(11, 13, 15),  # shoulder, elbow, wrist
                    required_range=AngleRange(30, 180, 90, "degrees"),
                    warnings=[
                        "Keep slight bend in elbow",
                        "Don't lock your elbow at the top"
                    ]
                )
            ],
            phase_definitions={
                "idle": {"angle_min": 0, "angle_max": 30},
                "raise": {"angle_min": 30, "angle_max": 150},
                "hold": {"angle_min": 150, "angle_max": 180},
                "lower": {"angle_min": 30, "angle_max": 0}
            },
            common_mistakes={
                "raise": [
                    "Using momentum instead of controlled movement",
                    "Elbow locked at top position",
                    "Shoulder shrugging during raise"
                ],
                "hold": [
                    "Holding with bent elbow",
                    "Incomplete range of motion",
                    "Swaying or losing balance"
                ],
                "lower": [
                    "Dropping the arm too quickly",
                    "Not controlling the descent",
                    "Shoulder elevation"
                ]
            },
            recommended_reps=12,
            recommended_sets=3,
            rest_between_sets=45
        )
        
        # Shoulder Rotation
        self.exercises[ExerciseType.SHOULDER_ROTATION] = FormMetrics(
            exercise_name="Shoulder Rotation",
            exercise_type=ExerciseType.SHOULDER_ROTATION,
            primary_joints=["left_shoulder", "left_elbow", "left_wrist"],
            angle_thresholds={
                "shoulder_abduction": AngleRange(0, 90, 45, "degrees"),
                "shoulder_rotation": AngleRange(0, 180, 90, "degrees"),
                "elbow_flexion": AngleRange(70, 110, 90, "degrees")
            },
            joint_thresholds=[
                JointThreshold(
                    joint_name="shoulder_rotation",
                    landmark_indices=(11, 13, 15),
                    required_range=AngleRange(0, 180, 90, "degrees"),
                    warnings=[
                        "Keep elbow at 90 degrees",
                        "Rotate from shoulder, not wrist",
                        "Control the movement speed"
                    ]
                ),
                JointThreshold(
                    joint_name="arm_elevation",
                    landmark_indices=(23, 11, 13),
                    required_range=AngleRange(70, 90, 85, "degrees"),
                    warnings=[
                        "Keep arm at 90 degrees to body",
                        "Don't lower arm below horizontal"
                    ]
                )
            ],
            phase_definitions={
                "start": {"angle_min": 0, "angle_max": 10},
                "rotate_out": {"angle_min": 10, "angle_max": 170},
                "hold": {"angle_min": 170, "angle_max": 180},
                "rotate_back": {"angle_min": 170, "angle_max": 10}
            },
            common_mistakes={
                "rotate_out": [
                    "Bending the elbow",
                    "Rotating from wrist instead of shoulder",
                    "Moving too fast",
                    "Not controlling the return"
                ],
                "hold": [
                    "Insufficient hold time",
                    "Losing arm position"
                ]
            },
            recommended_reps=15,
            recommended_sets=2,
            rest_between_sets=45
        )
    
    def get_exercise(self, exercise_type: ExerciseType) -> Optional[FormMetrics]:
        """Get exercise metrics by type."""
        return self.exercises.get(exercise_type)
    
    def get_all_exercises(self) -> Dict[str, str]:
        """Get list of all available exercises."""
        return {
            ex_type.value: metrics.exercise_name 
            for ex_type, metrics in self.exercises.items()
        }
    
    def get_exercise_names(self) -> List[str]:
        """Get list of exercise names."""
        return [metrics.exercise_name for metrics in self.exercises.values()]
    
    def get_common_mistakes(self, exercise_type: ExerciseType, phase: str) -> List[str]:
        """Get common mistakes for an exercise phase."""
        metrics = self.exercises.get(exercise_type)
        if metrics:
            return metrics.common_mistakes.get(phase, [])
        return []
    
    def validate_form(self, exercise_type: ExerciseType, angles: Dict[str, float], 
                      phase: str) -> Dict:
        """
        Validate exercise form against thresholds.
        Returns validation results with any issues found.
        """
        metrics = self.exercises.get(exercise_type)
        if not metrics:
            return {"valid": False, "issues": ["Exercise not found"]}
        
        issues = []
        suggestions = []
        
        for threshold in metrics.joint_thresholds:
            joint_name = threshold.joint_name
            angle = angles.get(joint_name)
            if angle is not None:
                if angle < threshold.required_range.min_angle:
                    issues.append(f"{joint_name} angle too low")
                    suggestions.append(f"Increase {joint_name} range of motion")
                elif angle > threshold.required_range.max_angle:
                    issues.append(f"{joint_name} angle too high")
                    suggestions.append(f"Reduce {joint_name} extension")
        
        # Get phase-specific mistakes
        phase_mistakes = self.get_common_mistakes(exercise_type, phase)
        if phase_mistakes:
            suggestions.extend(phase_mistakes)
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions,
            "phase": phase
        }
    
    def export_to_json(self) -> str:
        """Export exercise library to JSON."""
        library_data = {
            "export_date": datetime.now().isoformat(),
            "exercises": {
                ex_type.value: {
                    "name": metrics.exercise_name,
                    "primary_joints": metrics.primary_joints,
                    "angle_thresholds": {
                        k: {
                            "min": v.min_angle,
                            "max": v.max_angle,
                            "ideal": v.ideal_angle,
                            "unit": v.unit
                        }
                        for k, v in metrics.angle_thresholds.items()
                    },
                    "phase_definitions": metrics.phase_definitions,
                    "common_mistakes": metrics.common_mistakes,
                    "recommended_reps": metrics.recommended_reps,
                    "recommended_sets": metrics.recommended_sets,
                    "rest_between_sets": metrics.rest_between_sets
                }
                for ex_type, metrics in self.exercises.items()
            }
        }
        return json.dumps(library_data, indent=2)
    
    def get_recommended_exercises(self, difficulty: str = "beginner") -> List[str]:
        """Get recommended exercises based on difficulty level."""
        if difficulty == "beginner":
            return ["Shoulder Raise"]
        elif difficulty == "intermediate":
            return ["Shoulder Raise", "Shoulder Rotation"]
        elif difficulty == "advanced":
            return ["Shoulder Raise", "Shoulder Rotation"]
        else:
            return ["Shoulder Raise"]


# ============== SESSION RECORDER ==============
class SessionRecorder:
    """
    Records exercise sessions with timestamps, rep counts, and transitions.
    """
    
    def __init__(self):
        self.current_session = None
        self.session_history = []
    
    def start_session(self, exercise_type: ExerciseType, user_id: Optional[str] = None):
        """Start a new recording session."""
        self.current_session = {
            "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "exercise_type": exercise_type.value,
            "user_id": user_id,
            "start_time": datetime.now().isoformat(),
            "data": [],
            "reps": 0,
            "transitions": [],
            "form_issues": [],
            "status": "active"
        }
        return self.current_session["session_id"]
    
    def record_frame(self, angles: Dict[str, float], phase: str, 
                     form_validation: Optional[Dict] = None):
        """Record a single frame of exercise data."""
        if not self.current_session:
            return
        
        frame_data = {
            "timestamp": datetime.now().isoformat(),
            "angles": angles,
            "phase": phase,
            "form_validation": form_validation
        }
        self.current_session["data"].append(frame_data)
        
        # Track transitions
        if self.current_session["data"]:
            last_phase = self.current_session["data"][-2].get("phase") if len(self.current_session["data"]) > 1 else None
            if last_phase and last_phase != phase:
                self.current_session["transitions"].append({
                    "from": last_phase,
                    "to": phase,
                    "timestamp": frame_data["timestamp"]
                })
                
                # Count reps when transitioning from hold to lower
                if last_phase == "hold" and phase == "lower":
                    self.current_session["reps"] += 1
        
        # Track form issues
        if form_validation and not form_validation["valid"]:
            self.current_session["form_issues"].append({
                "timestamp": frame_data["timestamp"],
                "issues": form_validation["issues"],
                "phase": phase
            })
    
    def end_session(self) -> Dict:
        """End the current session and return recorded data."""
        if not self.current_session:
            return {}
        
        self.current_session["end_time"] = datetime.now().isoformat()
        self.current_session["status"] = "completed"
        self.current_session["duration_seconds"] = (
            datetime.fromisoformat(self.current_session["end_time"]) -
            datetime.fromisoformat(self.current_session["start_time"])
        ).total_seconds()
        
        # Calculate form score
        total_frames = len(self.current_session["data"])
        issue_frames = len(self.current_session["form_issues"])
        self.current_session["form_score"] = round(
            ((total_frames - issue_frames) / total_frames * 100) if total_frames > 0 else 0, 2
        )
        
        # Save to history
        completed_session = self.current_session.copy()
        self.session_history.append(completed_session)
        self.current_session = None
        
        return completed_session
    
    def get_session_summary(self, session_id: Optional[str] = None) -> Dict:
        """Get summary of current or specified session."""
        if session_id:
            for session in self.session_history:
                if session["session_id"] == session_id:
                    return session
            return {}
        return self.current_session or {}
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all recorded sessions."""
        return self.session_history


# ============== PROGRESS DASHBOARD ==============
class ProgressDashboard:
    """
    User progress dashboard with historical tracking and trends.
    """
    
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.sessions: List[Dict] = []
        self.weekly_goals: Dict = {}
    
    def add_session(self, session_data: Dict):
        """Add a completed session to the dashboard."""
        session_data["user_id"] = self.user_id
        self.sessions.append(session_data)
        self._update_weekly_goals(session_data)
    
    def get_performance_summary(self, days: int = 7) -> Dict:
        """Get performance summary for the last N days."""
        from datetime import datetime, timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        recent_sessions = [
            s for s in self.sessions 
            if datetime.fromisoformat(s["start_time"]) >= cutoff
        ]
        
        if not recent_sessions:
            return {
                "total_sessions": 0,
                "total_reps": 0,
                "average_form_score": 0,
                "improvement_trend": "N/A"
            }
        
        total_reps = sum(s.get("reps", 0) for s in recent_sessions)
        form_scores = [s.get("form_score", 0) for s in recent_sessions]
        avg_form_score = sum(form_scores) / len(form_scores)
        
        # Calculate improvement trend
        if len(form_scores) >= 2:
            recent_avg = sum(form_scores[-3:]) / min(3, len(form_scores))
            older_avg = sum(form_scores[:-3]) / max(1, len(form_scores) - 3)
            trend = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "total_sessions": len(recent_sessions),
            "total_reps": total_reps,
            "average_form_score": round(avg_form_score, 2),
            "improvement_trend": trend,
            "sessions": recent_sessions
        }
    
    def get_weekly_summary(self) -> Dict:
        """Get week-over-week progress summary."""
        this_week = self.get_performance_summary(days=7)
        last_week = self.get_performance_summary(days=14)
        
        this_week_sessions = this_week["total_sessions"]
        last_week_sessions = last_week["total_sessions"]
        
        session_change = (
            ((this_week_sessions - last_week_sessions) / last_week_sessions * 100)
            if last_week_sessions > 0 else 100
        )
        
        this_week_reps = this_week["total_reps"]
        last_week_reps = last_week["total_reps"]
        
        rep_change = (
            ((this_week_reps - last_week_reps) / last_week_reps * 100)
            if last_week_reps > 0 else 100
        )
        
        return {
            "this_week": {
                "sessions": this_week_sessions,
                "reps": this_week_reps,
                "avg_form_score": this_week["average_form_score"]
            },
            "last_week": {
                "sessions": last_week_sessions,
                "reps": last_week_reps,
                "avg_form_score": last_week["average_form_score"]
            },
            "changes": {
                "sessions": round(session_change, 2),
                "reps": round(rep_change, 2)
            }
        }
    
    def get_rep_accuracy(self, exercise_type: str) -> Dict:
        """Calculate repetition accuracy for an exercise type."""
        exercise_sessions = [
            s for s in self.sessions 
            if s.get("exercise_type") == exercise_type
        ]
        
        if not exercise_sessions:
            return {"accuracy": 0, "total_reps": 0, "sessions": 0}
        
        total_reps = sum(s.get("reps", 0) for s in exercise_sessions)
        avg_form_score = sum(s.get("form_score", 0) for s in exercise_sessions) / len(exercise_sessions)
        
        return {
            "exercise_type": exercise_type,
            "total_reps": total_reps,
            "sessions": len(exercise_sessions),
            "accuracy_percentage": round(avg_form_score, 2)
        }
    
    def get_improvement_trends(self) -> Dict:
        """Get detailed improvement trends over time."""
        if len(self.sessions) < 2:
            return {"trend": "insufficient_data", "data_points": []}
        
        # Group by week
        trends = []
        for session in self.sessions:
            trends.append({
                "date": session["start_time"][:10],
                "form_score": session.get("form_score", 0),
                "reps": session.get("reps", 0),
                "exercise": session.get("exercise_type", "unknown")
            })
        
        return {
            "trend": "improving" if trends[-1]["form_score"] > trends[0]["form_score"] else "stable",
            "data_points": trends,
            "total_sessions": len(self.sessions)
        }
    
    def _update_weekly_goals(self, session_data: Dict):
        """Update weekly goals based on completed sessions."""
        exercise_type = session_data.get("exercise_type", "unknown")
        if exercise_type not in self.weekly_goals:
            self.weekly_goals[exercise_type] = {
                "target_reps": 30,
                "current_reps": 0,
                "target_sessions": 3,
                "completed_sessions": 0
            }
        
        self.weekly_goals[exercise_type]["current_reps"] += session_data.get("reps", 0)
        self.weekly_goals[exercise_type]["completed_sessions"] += 1
    
    def get_weekly_goals_status(self) -> Dict:
        """Get status of weekly goals."""
        return {
            "user_id": self.user_id,
            "goals": self.weekly_goals,
            "update_date": datetime.now().isoformat()
        }
    
    def export_progress_report(self) -> Dict:
        """Export comprehensive progress report."""
        return {
            "user_id": self.user_id,
            "report_date": datetime.now().isoformat(),
            "total_sessions": len(self.sessions),
            "performance_summary": self.get_performance_summary(),
            "weekly_summary": self.get_weekly_summary(),
            "improvement_trends": self.get_improvement_trends(),
            "weekly_goals": self.get_weekly_goals_status()
        }


# ============== FORM CORRECTION ASSISTANT ==============
class FormCorrectionAssistant:
    """
    Provides targeted form corrections and tracks recurring issues.
    """
    
    def __init__(self, exercise_library: ExerciseLibrary):
        self.exercise_library = exercise_library
        self.recurring_issues: Dict[str, List[Dict]] = {}
        self.correction_history: List[Dict] = []
    
    def analyze_form(self, exercise_type: ExerciseType, angles: Dict[str, float],
                     phase: str) -> Dict:
        """
        Analyze exercise form and provide corrections.
        """
        # Validate form
        validation = self.exercise_library.validate_form(exercise_type, angles, phase)
        
        # Get specific corrections
        corrections = self._generate_corrections(exercise_type, validation, phase)
        
        # Track recurring issues
        if not validation["valid"]:
            self._track_issue(exercise_type.value, validation["issues"], phase)
        
        # Log correction
        correction_entry = {
            "timestamp": datetime.now().isoformat(),
            "exercise_type": exercise_type.value,
            "phase": phase,
            "validation": validation,
            "corrections": corrections
        }
        self.correction_history.append(correction_entry)
        
        return {
            "validation": validation,
            "corrections": corrections,
            "severity": self._determine_severity(validation)
        }
    
    def _generate_corrections(self, exercise_type: ExerciseType, 
                              validation: Dict, phase: str) -> List[str]:
        """Generate specific correction messages."""
        corrections = []
        
        # Get phase-specific mistakes
        mistakes = self.exercise_library.get_common_mistakes(exercise_type, phase)
        
        # Add targeted suggestions
        for suggestion in validation.get("suggestions", []):
            if suggestion not in corrections:
                corrections.append(suggestion)
        
        # Add relevant mistakes as warnings
        for mistake in mistakes[:3]:  # Limit to 3 most relevant
            if mistake not in corrections:
                corrections.append(f"Tip: {mistake}")
        
        return corrections
    
    def _determine_severity(self, validation: Dict) -> str:
        """Determine severity of form issues."""
        issue_count = len(validation.get("issues", []))
        if issue_count >= 3:
            return "stop"
        elif issue_count >= 2:
            return "warning"
        elif issue_count >= 1:
            return "mild"
        return "none"
    
    def _track_issue(self, exercise_type: str, issues: List[str], phase: str):
        """Track recurring issues for a user."""
        key = f"{exercise_type}_{phase}"
        if key not in self.recurring_issues:
            self.recurring_issues[key] = []
        
        for issue in issues:
            self.recurring_issues[key].append({
                "issue": issue,
                "timestamp": datetime.now().isoformat(),
                "count": len([i for i in self.recurring_issues[key] if i["issue"] == issue]) + 1
            })
    
    def get_recurring_issues(self, exercise_type: Optional[str] = None) -> Dict:
        """Get recurring issues, optionally filtered by exercise type."""
        if exercise_type:
            return {
                k: v for k, v in self.recurring_issues.items() 
                if k.startswith(exercise_type)
            }
        return self.recurring_issues
    
    def get_correction_summary(self) -> Dict:
        """Get summary of all corrections made."""
        by_severity = {"mild": 0, "warning": 0, "stop": 0, "none": 0}
        by_exercise = {}
        
        for correction in self.correction_history:
            severity = self._determine_severity(correction["validation"])
            if severity not in by_severity:
                by_severity[severity] = 0
            by_severity[severity] += 1
            
            ex_type = correction["exercise_type"]
            if ex_type not in by_exercise:
                by_exercise[ex_type] = 0
            by_exercise[ex_type] += 1
        
        return {
            "total_corrections": len(self.correction_history),
            "by_severity": by_severity,
            "by_exercise": by_exercise,
            "recurring_issues_count": len(self.recurring_issues)
        }


# ============== PDF EXPORT ==============
class SessionReportExporter:
    """
    Exports session reports to PDF format.
    """
    
    def __init__(self, exercise_library: ExerciseLibrary):
        self.exercise_library = exercise_library
    
    def generate_report(self, session_data: Dict, progress_data: Dict,
                        recommendations: List[str]) -> Dict:
        """
        Generate a comprehensive session report.
        Returns report data structure (PDF generation would require additional library).
        """
        report = {
            "report_id": f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_date": datetime.now().isoformat(),
            "session_summary": {
                "session_id": session_data.get("session_id"),
                "exercise_type": session_data.get("exercise_type"),
                "duration_seconds": session_data.get("duration_seconds"),
                "total_reps": session_data.get("reps"),
                "form_score": session_data.get("form_score"),
                "start_time": session_data.get("start_time"),
                "end_time": session_data.get("end_time")
            },
            "form_quality": {
                "score": session_data.get("form_score"),
                "issues_count": len(session_data.get("form_issues", [])),
                "transitions_count": len(session_data.get("transitions", [])),
                "overall_rating": self._get_overall_rating(session_data.get("form_score", 0))
            },
            "performance_trends": progress_data.get("improvement_trends", {}),
            "recommendations": recommendations,
            "export_format": "PDF available",
            "pdf_available": True
        }
        
        return report
    
    def _get_overall_rating(self, form_score: float) -> str:
        """Get overall rating based on form score."""
        if form_score >= 90:
            return "Excellent"
        elif form_score >= 75:
            return "Good"
        elif form_score >= 60:
            return "Fair"
        elif form_score >= 40:
            return "Needs Improvement"
        else:
            return "Poor"
    
    def generate_recommendations(self, session_data: Dict, 
                                 form_score: float) -> List[str]:
        """Generate follow-up exercise recommendations."""
        recommendations = []
        exercise_type = session_data.get("exercise_type", "")
        
        # Based on form score
        if form_score < 60:
            recommendations.append(f"Focus on mastering {exercise_type} form before increasing reps")
            recommendations.append("Consider reducing weight or resistance")
        elif form_score < 80:
            recommendations.append(f"Good progress on {exercise_type}, continue practicing")
            recommendations.append("Focus on the specific issues noted during session")
        else:
            recommendations.append(f"Excellent form on {exercise_type}")
            recommendations.append("Ready to increase difficulty or try next exercise")
        
        # Based on exercise type
        if exercise_type == "shoulder_raise":
            recommendations.append("Next exercise: Shoulder Rotation")
        elif exercise_type == "shoulder_rotation":
            recommendations.append("Next exercise: Try Shoulder Raise with increased range of motion")
        
        # General recommendations
        recommendations.append("Always warm up before exercising")
        recommendations.append("Stay hydrated during workout")
        
        return recommendations
    
    def export_to_json(self, report: Dict) -> str:
        """Export report to JSON format."""
        return json.dumps(report, indent=2)
    
    def export_to_pdf(self, report: Dict, output_path: Optional[str] = None) -> str:
        """
        Export report to PDF format.
        Returns the path to the generated PDF file.
        """
        if FPDF is None:
            raise ImportError("fpdf library is required for PDF export. Install with: pip install fpdf")
        
        class PDFReport(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 16)
                self.cell(0, 10, 'Physio Intelligence - Session Report', 0, 1, 'C')
                self.ln(5)
            
            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
        pdf = PDFReport()
        pdf.add_page()
        
        # Session Summary
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Session Summary', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        session_summary = report.get("session_summary", {})
        pdf.cell(0, 7, f"Session ID: {session_summary.get('session_id', 'N/A')}", 0, 1)
        pdf.cell(0, 7, f"Exercise Type: {session_summary.get('exercise_type', 'N/A').replace('_', ' ').title()}", 0, 1)
        pdf.cell(0, 7, f"Duration: {session_summary.get('duration_seconds', 0)} seconds", 0, 1)
        pdf.cell(0, 7, f"Total Reps: {session_summary.get('total_reps', 0)}", 0, 1)
        pdf.cell(0, 7, f"Form Score: {session_summary.get('form_score', 0)}%", 0, 1)
        pdf.ln(5)
        
        # Form Quality
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Form Quality', 0, 1)
        pdf.set_font('Arial', '', 10)
        form_quality = report.get("form_quality", {})
        pdf.cell(0, 7, f"Overall Rating: {form_quality.get('overall_rating', 'N/A')}", 0, 1)
        pdf.cell(0, 7, f"Issues Count: {form_quality.get('issues_count', 0)}", 0, 1)
        pdf.ln(5)
        
        # Recommendations
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Recommendations', 0, 1)
        pdf.set_font('Arial', '', 10)
        for rec in report.get("recommendations", []):
            pdf.cell(0, 7, f"- {rec}", 0, 1)
        pdf.ln(5)
        
        # Export info
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 7, f"Report generated: {report.get('generated_date', 'N/A')}", 0, 1)
        pdf.cell(0, 7, f"Report ID: {report.get('report_id', 'N/A')}", 0, 1)
        
        # Save to file
        if output_path is None:
            output_path = f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(output_path)
        
        return output_path


# ============== MAIN INTEGRATION ==============
class PhysioIntelligenceDay2:
    """
    Day 2 integration of all new features.
    Combines exercise library, session recorder, progress dashboard,
    form correction assistant, and PDF export.
    """
    
    def __init__(self):
        self.exercise_library = ExerciseLibrary()
        self.session_recorder = SessionRecorder()
        self.progress_dashboard = ProgressDashboard()
        self.form_correction = FormCorrectionAssistant(self.exercise_library)
        self.report_exporter = SessionReportExporter(self.exercise_library)
    
    def start_exercise_session(self, exercise_type: str, user_id: Optional[str] = None) -> str:
        """Start an exercise session with full tracking."""
        ex_type = ExerciseType(exercise_type)
        return self.session_recorder.start_session(ex_type, user_id)
    
    def process_frame(self, exercise_type: str, angles: Dict[str, float], 
                      phase: str) -> Dict:
        """Process a frame with full analysis."""
        ex_type = ExerciseType(exercise_type)
        
        # Analyze form
        form_analysis = self.form_correction.analyze_form(ex_type, angles, phase)
        
        # Record frame
        self.session_recorder.record_frame(angles, phase, form_analysis["validation"])
        
        return {
            "angles": angles,
            "phase": phase,
            "form_analysis": form_analysis,
            "corrections": form_analysis["corrections"]
        }
    
    def end_exercise_session(self) -> Dict:
        """End session and generate comprehensive report."""
        session_data = self.session_recorder.end_session()
        
        if not session_data:
            return {}
        
        # Add to progress dashboard
        self.progress_dashboard.add_session(session_data)
        
        # Generate recommendations
        recommendations = self.report_exporter.generate_recommendations(
            session_data, session_data.get("form_score", 0)
        )
        
        # Generate report
        progress_data = self.progress_dashboard.export_progress_report()
        report = self.report_exporter.generate_report(
            session_data, progress_data, recommendations
        )
        
        return {
            "session_summary": session_data,
            "progress_report": progress_data,
            "session_report": report,
            "recommendations": recommendations
        }
    
    def get_dashboard_summary(self) -> Dict:
        """Get comprehensive dashboard summary."""
        return {
            "performance": self.progress_dashboard.get_performance_summary(),
            "weekly": self.progress_dashboard.get_weekly_summary(),
            "trends": self.progress_dashboard.get_improvement_trends(),
            "goals": self.progress_dashboard.get_weekly_goals_status(),
            "correction_summary": self.form_correction.get_correction_summary()
        }


# ============== TESTS ==============
def test_exercise_library():
    """Test exercise library functionality."""
    print("Testing Exercise Library...")
    
    library = ExerciseLibrary()
    
    # Test getting exercises
    assert ExerciseType.SHOULDER_RAISE in library.exercises
    assert ExerciseType.SHOULDER_ROTATION in library.exercises
    
    # Test exercise names
    names = library.get_exercise_names()
    assert "Shoulder Raise" in names
    assert "Shoulder Rotation" in names
    
    # Test form validation
    validation = library.validate_form(
        ExerciseType.SHOULDER_RAISE,
        {"shoulder_flexion": 90},
        "raise"
    )
    assert "valid" in validation
    
    print("✓ Exercise Library tests passed")


def test_session_recorder():
    """Test session recorder functionality."""
    print("Testing Session Recorder...")
    
    recorder = SessionRecorder()
    
    # Start session
    session_id = recorder.start_session(ExerciseType.SHOULDER_RAISE)
    assert session_id is not None
    assert recorder.current_session is not None
    
    # Record frames
    for i in range(10):
        recorder.record_frame({"shoulder": 30 + i * 10}, "raise", None)
    
    # End session
    session_data = recorder.end_session()
    assert session_data["status"] == "completed"
    assert len(session_data["data"]) == 10
    
    print("✓ Session Recorder tests passed")


def test_progress_dashboard():
    """Test progress dashboard functionality."""
    print("Testing Progress Dashboard...")
    
    dashboard = ProgressDashboard("test_user")
    
    # Add mock sessions
    for i in range(5):
        dashboard.add_session({
            "session_id": f"session_{i}",
            "exercise_type": "shoulder_raise",
            "reps": 10 + i,
            "form_score": 70 + i * 2,
            "start_time": datetime.now().isoformat()
        })
    
    # Test performance summary
    summary = dashboard.get_performance_summary()
    assert summary["total_sessions"] == 5
    assert summary["total_reps"] > 0
    
    # Test weekly summary
    weekly = dashboard.get_weekly_summary()
    assert "this_week" in weekly
    assert "last_week" in weekly
    
    print("✓ Progress Dashboard tests passed")


def test_form_correction():
    """Test form correction assistant."""
    print("Testing Form Correction...")
    
    library = ExerciseLibrary()
    correction = FormCorrectionAssistant(library)
    
    # Analyze form
    result = correction.analyze_form(
        ExerciseType.SHOULDER_RAISE,
        {"shoulder_flexion": 45},
        "raise"
    )
    
    assert "validation" in result
    assert "corrections" in result
    assert "severity" in result
    
    # Test recurring issues
    issues = correction.get_recurring_issues()
    assert isinstance(issues, dict)
    
    print("✓ Form Correction tests passed")


def test_report_exporter():
    """Test session report exporter."""
    print("Testing Report Exporter...")
    
    library = ExerciseLibrary()
    exporter = SessionReportExporter(library)
    
    # Generate mock session data
    session_data = {
        "session_id": "test_session",
        "exercise_type": "shoulder_raise",
        "duration_seconds": 120,
        "reps": 10,
        "form_score": 85,
        "start_time": datetime.now().isoformat(),
        "end_time": datetime.now().isoformat(),
        "form_issues": [],
        "transitions": []
    }
    
    # Generate recommendations
    recommendations = exporter.generate_recommendations(session_data, 85)
    assert len(recommendations) > 0
    
    # Generate report
    report = exporter.generate_report(session_data, {}, recommendations)
    assert report["session_summary"]["form_score"] == 85
    
    # Test PDF export
    try:
        pdf_path = exporter.export_to_pdf(report, "test_report.pdf")
        assert pdf_path == "test_report.pdf"
        print("  ✓ PDF export successful")
    except ImportError:
        print("  ⚠ fpdf not installed, skipping PDF test")
    
    print("✓ Report Exporter tests passed")


def test_day2_integration():
    """Test Day 2 integration."""
    print("Testing Day 2 Integration...")
    
    day2 = PhysioIntelligenceDay2()
    
    # Start session
    session_id = day2.start_exercise_session("shoulder_raise")
    assert session_id is not None
    
    # Process frames
    for i in range(5):
        result = day2.process_frame(
            "shoulder_raise",
            {"shoulder_flexion": 30 + i * 20},
            "raise"
        )
        assert "form_analysis" in result
    
    # End session
    end_result = day2.end_exercise_session()
    assert "session_summary" in end_result
    assert "progress_report" in end_result
    
    # Get dashboard
    dashboard = day2.get_dashboard_summary()
    assert "performance" in dashboard
    
    print("✓ Day 2 Integration tests passed")


def run_all_tests():
    """Run all Day 2 tests."""
    print("="*60)
    print("Physio Intelligence v3 - Day 2 Tests")
    print("="*60 + "\n")
    
    test_exercise_library()
    test_session_recorder()
    test_progress_dashboard()
    test_form_correction()
    test_report_exporter()
    test_day2_integration()
    
    print("\n" + "="*60)
    print("ALL DAY 2 TESTS PASSED ✓")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
