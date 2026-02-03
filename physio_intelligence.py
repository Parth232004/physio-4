"""
Real-Time Physio Intelligence v3
Live Demo Sprint - Day 1 Implementation

Features:
- Live Pose Reliability with MediaPipe webcam
- Phase Awareness (raise/hold/lower detection)
- Intelligent Corrections (severity-based feedback)
- Session Intelligence (start/stop, scores, logs)
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque
import json

# ============== CONFIGURATION ==============
class Config:
    # Camera settings
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    
    # Pose detection settings
    MIN_DETECTION_CONFIDENCE: float = 0.5
    MIN_TRACKING_CONFIDENCE: float = 0.5
    
    # Angle thresholds for exercises (example: shoulder raise)
    RAISE_START_ANGLE: float = 30.0  # Degrees from vertical
    RAISE_PEAK_ANGLE: float = 150.0  # Degrees from vertical
    HOLD_THRESHOLD: float = 5.0  # Degrees tolerance for hold detection
    
    # Phase timing
    MIN_HOLD_TIME: float = 1.0  # Minimum hold duration in seconds
    PHASE_CHANGE_COOLDOWN: float = 0.3  # Debounce phase changes
    
    # Feedback settings
    FEEDBACK_COOLDOWN: float = 2.0  # Seconds between same-type feedback
    SEVERITY_COOLDOWNS: Dict[str, float] = {
        "mild": 3.0,
        "warning": 5.0,
        "stop": 10.0
    }
    
    # Session settings
    SESSION_TIMEOUT: float = 30.0  # Timeout for no pose detection
    
    # Camera reconnection settings
    CAMERA_RECONNECT_ATTEMPTS: int = 3
    CAMERA_RECONNECT_DELAY: float = 1.0
    
    # Data limits
    MAX_ANGLE_HISTORY: int = 1000
    MAX_FEEDBACK_HISTORY: int = 100
    
    # Performance settings
    TARGET_FPS: int = 30


# ============== ANGLE CALCULATIONS ==============
def calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    Calculate angle at point b given three points a, b, c.
    Returns angle in degrees (0-180).
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)


def get_landmark_coords(landmarks, landmark_idx: int, frame_shape: Tuple[int, int]) -> Tuple[float, float]:
    """Extract normalized landmark coordinates and convert to pixel coordinates."""
    h, w = frame_shape
    lm = landmarks[landmark_idx]
    return (lm.x * w, lm.y * h)


def validate_angle(angle: float) -> bool:
    """Validate angle is within reasonable bounds."""
    return 0 <= angle <= 180 and not (np.isnan(angle) or np.isinf(angle))


# ============== PHASE DETECTION ==============
class PhaseDetector:
    """
    Detects exercise phases: raise, hold, lower
    """
    def __init__(self, config: type = Config):
        self.config = config
        self.current_phase: str = "idle"
        self.phase_start_time: float = 0
        self.last_phase_change: float = 0
        self.angle_history: deque = deque(maxlen=30)  # Store last 30 frames
        self.hold_start_angle: float = 0
        
    def update(self, angle: float, timestamp: float) -> str:
        """Update phase detection with new angle and return current phase."""
        self.angle_history.append((angle, timestamp))
        
        # Check phase change cooldown
        if timestamp - self.last_phase_change < self.config.PHASE_CHANGE_COOLDOWN:
            return self.current_phase
        
        # State machine for phase detection
        if self.current_phase == "idle" or self.current_phase == "lower":
            if angle > self.config.RAISE_START_ANGLE:
                self.current_phase = "raise"
                self.last_phase_change = timestamp
                self.phase_start_time = timestamp
                
        elif self.current_phase == "raise":
            if angle >= self.config.RAISE_PEAK_ANGLE - 10:
                self.current_phase = "hold"
                self.last_phase_change = timestamp
                self.hold_start_angle = angle
                
        elif self.current_phase == "hold":
            # Check if holding for minimum time
            hold_duration = timestamp - self.phase_start_time
            if hold_duration >= self.config.MIN_HOLD_TIME:
                # Check if still in hold range
                if abs(angle - self.hold_start_angle) < self.config.HOLD_THRESHOLD:
                    pass  # Stay in hold
                else:
                    self.current_phase = "lower"
                    self.last_phase_change = timestamp
            elif angle < self.config.RAISE_START_ANGLE:
                self.current_phase = "lower"
                self.last_phase_change = timestamp
                
        elif self.current_phase == "lower":
            if angle < self.config.RAISE_START_ANGLE:
                self.current_phase = "idle"
                self.last_phase_change = timestamp
                
        return self.current_phase
    
    def reset(self):
        """Reset phase detector to initial state."""
        self.current_phase = "idle"
        self.phase_start_time = 0
        self.last_phase_change = 0
        self.angle_history.clear()


# ============== FEEDBACK SYSTEM ==============
class FeedbackSystem:
    """
    Manages intelligent corrections with severity-based feedback.
    Prevents feedback spam using cooldowns and message queuing.
    Messages are displayed for minimum 2 seconds with queue support.
    """
    def __init__(self, config: type = Config):
        self.config = config
        self.last_feedback_time: float = 0
        self.last_feedback_type: str = ""
        self.severity_timers: Dict[str, float] = {}
        self.feedback_history: List[Dict] = []
        
        # Message queue for display with minimum duration
        self.message_queue: deque = deque(maxlen=10)
        self.current_message: Optional[Dict] = None
        self.message_display_start: float = 0
        self.MESSAGE_MIN_DISPLAY_TIME: float = 2.0  # 2 seconds minimum display
        
    def queue_feedback(self, phase: str, angle: float, issues: List[str], 
                       timestamp: float) -> Optional[Dict]:
        """
        Queue feedback for display. Messages are queued if current message
        hasn't been displayed for minimum 2 seconds.
        """
        # Check overall cooldown
        if timestamp - self.last_feedback_time < self.config.FEEDBACK_COOLDOWN:
            return None
        
        # Determine severity and message
        severity = "mild"
        message = ""
        
        for issue in issues:
            if "stop" in issue.lower():
                severity = "stop"
                message = issue
                break
            elif "warning" in issue.lower():
                severity = "warning"
                message = issue
            elif "mild" in issue.lower() and severity == "mild":
                message = issue
        
        # If no specific issue, generate phase-appropriate feedback
        if not message:
            if phase == "raise":
                message = f"Raise arm smoothly - {angle:.1f}°"
            elif phase == "hold":
                message = f"Hold position - {angle:.1f}°"
            elif phase == "lower":
                message = f"Lower arm controlled - {angle:.1f}°"
            else:
                return None
        
        # Check severity-specific cooldown
        if severity in self.config.SEVERITY_COOLDOWNS:
            if severity in self.severity_timers:
                if timestamp - self.severity_timers[severity] < self.config.SEVERITY_COOLDOWNS[severity]:
                    return None
            self.severity_timers[severity] = timestamp
        
        # Generate feedback
        feedback = {
            "timestamp": timestamp,
            "phase": phase,
            "angle": angle,
            "severity": severity,
            "message": message,
            "issues": issues
        }
        
        # Add to queue
        self.message_queue.append(feedback)
        self.feedback_history.append(feedback)
        self.last_feedback_time = timestamp
        self.last_feedback_type = severity
        
        return feedback
    
    def get_display_message(self, timestamp: float) -> Optional[Dict]:
        """
        Get the current message to display. Returns current message if within
        minimum display time, otherwise returns next queued message.
        """
        # If no current message, get from queue
        if self.current_message is None:
            if self.message_queue:
                self.current_message = self.message_queue.popleft()
                self.message_display_start = timestamp
            return self.current_message
        
        # Check if current message has been displayed long enough
        display_duration = timestamp - self.message_display_start
        if display_duration >= self.MESSAGE_MIN_DISPLAY_TIME:
            # Check for queued messages
            if self.message_queue:
                self.current_message = self.message_queue.popleft()
                self.message_display_start = timestamp
            else:
                # No more queued messages, clear current after a short grace period
                if display_duration >= self.MESSAGE_MIN_DISPLAY_TIME + 1.0:
                    self.current_message = None
            
        return self.current_message
    
    def get_queued_count(self) -> int:
        """Get number of messages waiting in queue."""
        return len(self.message_queue)
    
    def get_feedback_summary(self) -> Dict:
        """Get summary of all feedback given."""
        by_severity = {"mild": 0, "warning": 0, "stop": 0}
        for fb in self.feedback_history:
            if fb["severity"] in by_severity:
                by_severity[fb["severity"]] += 1
        return by_severity
    
    def clear_queue(self):
        """Clear all queued messages."""
        self.message_queue.clear()
        self.current_message = None


# ============== SESSION MANAGER ==============
class SessionManager:
    """
    Manages exercise session with scoring and safety tracking.
    """
    def __init__(self, config: type = Config):
        self.config = config
        self.is_active: bool = False
        self.session_start: float = 0
        self.session_end: float = 0
        self.last_pose_time: float = 0
        self.total_reps: int = 0
        self.completed_reps: int = 0
        self.safety_violations: int = 0
        self.phase_logs: List[Dict] = []
        self.feedback_logs: List[Dict] = []
        self.angle_stream: List[Tuple[float, float]] = []  # (timestamp, angle)
        
    def start_session(self):
        """Start a new exercise session."""
        self.is_active = True
        self.session_start = time.time()
        self.last_pose_time = self.session_start
        self.total_reps = 0
        self.completed_reps = 0
        self.safety_violations = 0
        self.phase_logs = []
        self.feedback_logs = []
        self.angle_stream = []
        self.log_event("session_start", {"timestamp": self.session_start})
    
    def stop_session(self):
        """End the current session."""
        if self.is_active:
            self.session_end = time.time()
            self.is_active = False
            self.log_event("session_end", {
                "timestamp": self.session_end,
                "duration": self.session_end - self.session_start
            })
    
    def update_pose(self, angle: float, phase: str, timestamp: float):
        """Update session with new pose data."""
        if not self.is_active:
            return
        
        self.last_pose_time = timestamp
        self.angle_stream.append((timestamp, angle))
        
        # Track reps
        if phase == "hold":
            self.total_reps += 1
            
        elif phase == "idle" and self.total_reps > self.completed_reps:
            # Check if it was a valid rep
            valid_rep = True
            for ts, ang in self.angle_stream[-30:]:  # Check recent angles
                if ang > self.config.RAISE_PEAK_ANGLE - 10:
                    valid_rep = True
                    break
            if valid_rep:
                self.completed_reps += 1
                self.log_event("rep_completed", {
                    "rep_number": self.completed_reps,
                    "timestamp": timestamp
                })
    
    def add_safety_violation(self, violation: str, timestamp: float):
        """Record a safety violation."""
        self.safety_violations += 1
        self.log_event("safety_violation", {
            "violation": violation,
            "count": self.safety_violations,
            "timestamp": timestamp
        })
    
    def log_event(self, event_type: str, data: Dict):
        """Log an event with timestamp."""
        log_entry = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data
        }
        if event_type in ["phase_change", "rep_completed"]:
            self.phase_logs.append(log_entry)
        elif event_type in ["safety_violation", "feedback"]:
            self.feedback_logs.append(log_entry)
    
    def check_timeout(self) -> bool:
        """Check if session has timed out due to no pose detection."""
        if self.is_active and (time.time() - self.last_pose_time > self.config.SESSION_TIMEOUT):
            self.stop_session()
            return True
        return False
    
    def get_summary(self) -> Dict:
        """Get session summary dictionary."""
        duration = self.session_end - self.session_start if self.session_end > 0 else 0
        
        return {
            "session_active": self.is_active,
            "duration_seconds": round(duration, 2),
            "total_reps": self.total_reps,
            "completed_reps": self.completed_reps,
            "consistency_score": round(
                (self.completed_reps / self.total_reps * 100) if self.total_reps > 0 else 0, 2
            ),
            "safety_violations": self.safety_violations,
            "angle_samples": len(self.angle_stream),
            "phase_logs_count": len(self.phase_logs),
            "feedback_logs_count": len(self.feedback_logs),
            "start_time": datetime.fromtimestamp(self.session_start).isoformat() if self.session_start else None,
            "end_time": datetime.fromtimestamp(self.session_end).isoformat() if self.session_end else None
        }
    
    def get_logs(self) -> List[Dict]:
        """Get all session logs."""
        return {
            "phase_logs": self.phase_logs,
            "feedback_logs": self.feedback_logs
        }


# ============== MAIN PHYSIO INTELLIGENCE ENGINE ==============
class PhysioIntelligence:
    """
    Main engine for Real-Time Physio Intelligence.
    Integrates pose detection, phase awareness, feedback, and session management.
    """
    def __init__(self, config: type = Config):
        self.config = config
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        self.phase_detector = PhaseDetector(config)
        self.feedback_system = FeedbackSystem(config)
        self.session_manager = SessionManager(config)
        
        self.cap = None
        self.is_running = False
        self.is_paused = False
        self.frame_count = 0
        self.dropped_frames = 0
        self.no_pose_frames = 0
        
    def initialize_camera(self) -> bool:
        """Initialize webcam connection."""
        self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
        
        if not self.cap.isOpened():
            print(f"[ERROR] Failed to open camera at index {self.config.CAMERA_INDEX}")
            return False
        
        print(f"[INFO] Camera initialized: {self.config.FRAME_WIDTH}x{self.config.FRAME_HEIGHT}")
        return True
    
    def _reconnect_camera(self) -> bool:
        """Attempt to reconnect camera with exponential backoff."""
        print("[WARN] Camera disconnected, attempting to reconnect...")
        
        if self.cap:
            self.cap.release()
        
        for attempt in range(self.config.CAMERA_RECONNECT_ATTEMPTS):
            delay = self.config.CAMERA_RECONNECT_DELAY * (2 ** attempt)
            time.sleep(delay)
            
            if self.initialize_camera():
                print(f"[INFO] Camera reconnected after {attempt + 1} attempt(s)")
                return True
        
        print("[ERROR] Failed to reconnect camera after {self.config.CAMERA_RECONNECT_ATTEMPTS} attempts")
        return False
    
    def pause_session(self):
        """Pause the current session."""
        self.is_paused = True
        print("[INFO] Session paused - press 'r' to resume")
    
    def resume_session(self):
        """Resume a paused session."""
        self.is_paused = False
        print("[INFO] Session resumed")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame and return annotated frame + pose data.
        """
        frame_height, frame_width, _ = frame.shape
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        pose_data = {
            "timestamp": time.time(),
            "landmarks": None,
            "angles": {},
            "phase": "no_pose",
            "feedback": None,
            "issues": []
        }
        
        if results.pose_landmarks:
            self.frame_count += 1
            
            # Extract key landmarks for shoulder raise exercise
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates (left side for demo)
            left_shoulder = get_landmark_coords(landmarks, 11, (frame_height, frame_width))
            left_elbow = get_landmark_coords(landmarks, 13, (frame_height, frame_width))
            left_wrist = get_landmark_coords(landmarks, 15, (frame_height, frame_width))
            
            # Calculate angles
            shoulder_angle = calculate_angle(left_elbow, left_shoulder, 
                                             (left_shoulder[0] + 50, left_shoulder[1]))
            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            pose_data["landmarks"] = results.pose_landmarks
            pose_data["angles"] = {
                "shoulder": shoulder_angle,
                "elbow": elbow_angle
            }
            
            # Update phase detection
            current_phase = self.phase_detector.update(shoulder_angle, pose_data["timestamp"])
            pose_data["phase"] = current_phase
            
            # Check for issues
            issues = self._check_issues(shoulder_angle, elbow_angle, current_phase)
            pose_data["issues"] = issues
            
            # Queue feedback (new messages are added to queue)
            self.feedback_system.queue_feedback(
                current_phase, shoulder_angle, issues, pose_data["timestamp"]
            )
            
            # Get display message (respects 2-second minimum display time)
            display_feedback = self.feedback_system.get_display_message(pose_data["timestamp"])
            pose_data["feedback"] = display_feedback
            pose_data["queue_count"] = self.feedback_system.get_queued_count()
            
            # Update session
            self.session_manager.update_pose(shoulder_angle, current_phase, 
                                             pose_data["timestamp"])
            
            # Draw pose on frame
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Draw angle info on frame
            self._draw_info(frame, shoulder_angle, current_phase, display_feedback, 
                          self.feedback_system.get_queued_count())
            
        else:
            self.no_pose_frames += 1
            pose_data["phase"] = "no_pose"
        
        return frame, pose_data
    
    def _check_issues(self, shoulder_angle: float, elbow_angle: float, 
                     phase: str) -> List[str]:
        """Check for exercise form issues."""
        issues = []
        
        # Check for locked elbow during raise
        if elbow_angle > 160 and phase == "raise":
            issues.append("warning: Keep slight bend in elbow")
        
        # Check for excessive range of motion
        if shoulder_angle > 170:
            issues.append("stop: Range of motion too extreme")
        
        # Check for too fast movement (based on recent angles would need history)
        if shoulder_angle > 100 and phase == "raise":
            issues.append("mild: Control the movement speed")
        
        # Check for adequate raise
        if phase == "hold" and shoulder_angle < 80:
            issues.append("warning: Raise arm higher")
        
        return issues
    
    def _draw_info(self, frame: np.ndarray, angle: float, phase: str, 
                  feedback: Optional[Dict], queue_count: int = 0):
        """Draw pose information on frame."""
        # Draw angle
        cv2.putText(frame, f"Angle: {angle:.1f}°", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw phase
        cv2.putText(frame, f"Phase: {phase.upper()}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Draw feedback if available
        if feedback:
            color = (0, 255, 0) if feedback["severity"] == "mild" else \
                    (0, 165, 255) if feedback["severity"] == "warning" else (0, 0, 255)
            cv2.putText(frame, feedback["message"], (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Show remaining display time
            display_time = time.time() - self.feedback_system.message_display_start
            remaining = max(0, self.feedback_system.MESSAGE_MIN_DISPLAY_TIME - display_time)
            if remaining > 0:
                cv2.putText(frame, f"({remaining:.1f}s)", (10, 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # Draw queue count if messages are queued
        if queue_count > 0:
            cv2.putText(frame, f"Queued: {queue_count}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
        
        # Draw no-pose feedback
        if phase == "no_pose":
            cv2.putText(frame, "Searching for pose...", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        
        # Draw pause status
        if self.is_paused:
            cv2.putText(frame, "PAUSED - Press 'p' to resume", (frame.shape[1] - 400, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw session info
        if self.session_manager.is_active:
            cv2.putText(frame, f"Reps: {self.session_manager.completed_reps}", 
                        (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    def run_session(self, duration_seconds: int = 60):
        """Run a live exercise session with pause/resume and camera reconnection."""
        if not self.initialize_camera():
            return
        
        self.session_manager.start_session()
        self.is_running = True
        self.is_paused = False
        
        print(f"[INFO] Session started. Duration: {duration_seconds}s")
        print(f"[INFO] Press 'q' to stop, 'p' to pause/resume, 's' to get session summary")
        
        start_time = time.time()
        
        while self.is_running and (time.time() - start_time) < duration_seconds:
            # Handle pause
            if self.is_paused:
                cv2.waitKey(100)  # Reduced frame rate during pause
                continue
            
            ret, frame = self.cap.read()
            
            if not ret:
                self.dropped_frames += 1
                print(f"[WARN] Dropped frame {self.dropped_frames}")
                # Attempt camera reconnection
                if not self._reconnect_camera():
                    break
                continue
            
            # Process frame
            annotated_frame, pose_data = self.process_frame(frame)
            
            # Display frame
            cv2.imshow('Physio Intelligence v3 - Live Demo', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.is_running = False
            elif key == ord('p'):
                if self.is_paused:
                    self.resume_session()
                else:
                    self.pause_session()
            elif key == ord('s'):
                summary = self.session_manager.get_summary()
                print(f"\n[SESSION SUMMARY]")
                print(json.dumps(summary, indent=2))
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.is_running = False
        self.session_manager.stop_session()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Print final stats
        self._print_final_stats()
    
    def _print_final_stats(self):
        """Print final session statistics."""
        summary = self.session_manager.get_summary()
        logs = self.session_manager.get_logs()
        
        print("\n" + "="*50)
        print("FINAL SESSION STATISTICS")
        print("="*50)
        print(f"Session Duration: {summary['duration_seconds']}s")
        print(f"Total Frames: {self.frame_count}")
        print(f"Dropped Frames: {self.dropped_frames}")
        print(f"No-Pose Frames: {self.no_pose_frames}")
        print(f"Total Reps: {summary['total_reps']}")
        print(f"Completed Reps: {summary['completed_reps']}")
        print(f"Consistency Score: {summary['consistency_score']}%")
        print(f"Safety Violations: {summary['safety_violations']}")
        print(f"Angle Samples: {summary['angle_samples']}")
        print("="*50)
        
        # Print feedback summary
        feedback_summary = self.feedback_system.get_feedback_summary()
        print(f"\nFeedback Summary:")
        print(f"  Mild: {feedback_summary['mild']}")
        print(f"  Warning: {feedback_summary['warning']}")
        print(f"  Stop: {feedback_summary['stop']}")
        
        # Save logs to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"session_{timestamp}.json"
        with open(log_file, 'w') as f:
            json.dump({
                "summary": summary,
                "feedback_summary": feedback_summary,
                "phase_logs": logs["phase_logs"],
                "feedback_logs": logs["feedback_logs"]
            }, f, indent=2)
        print(f"\nLogs saved to: {log_file}")


# ============== DEMO ENTRY POINT ==============
def main():
    """Main entry point for demo."""
    print("="*60)
    print("Physio Intelligence v3 - Real-Time Demo")
    print("="*60)
    print("\nControls:")
    print("  'q' - Quit session")
    print("  's' - Show session summary")
    print("\nStarting in 3 seconds...")
    print("="*60)
    
    time.sleep(3)
    
    # Initialize and run session
    engine = PhysioIntelligence()
    
    try:
        engine.run_session(duration_seconds=120)  # 2-minute demo session
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        engine.cleanup()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        engine.cleanup()


if __name__ == "__main__":
    main()
