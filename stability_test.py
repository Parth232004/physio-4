"""
Physio Intelligence v3 - 10-15 Minute Stability Test
Day 2: Production Readiness Validation

This script performs extended stress testing of the Physio Intelligence system
to verify stability, performance, and reliability over extended operation.

Usage:
    python stability_test.py [--duration MINUTES] [--output OUTPUT_DIR]

Features:
- Extended session testing (10-15 minutes)
- Performance monitoring (FPS, memory, frame drops)
- Automatic recovery testing (simulated pose loss)
- Comprehensive logging
- Session data export
"""

import sys
import time
import psutil
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add current directory to path
sys.path.insert(0, '.')

from physio_intelligence import (
    PhysioIntelligence, Config, SessionManager,
    PhaseDetector, FeedbackSystem
)
import cv2
import mediapipe as mp
import numpy as np


class StabilityTest:
    """
    Comprehensive stability test for Physio Intelligence v3.
    Validates system reliability over extended operation.
    """
    
    def __init__(self, duration_minutes: int = 10, output_dir: str = "stability_test_output"):
        self.config = Config
        self.duration_seconds = duration_minutes * 60
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Performance metrics
        self.frame_count = 0
        self.dropped_frames = 0
        self.no_pose_frames = 0
        self.successful_frames = 0
        self.fps_history: List[float] = []
        self.memory_history: List[float] = []
        self.fps = 0
        self.start_time = 0
        self.last_fps_update = 0
        
        # Test results
        self.test_results: Dict = {
            "test_start": None,
            "test_end": None,
            "duration_minutes": duration_minutes,
            "passed": False,
            "metrics": {},
            "events": [],
            "errors": []
        }
        
    def _log_event(self, event_type: str, message: str):
        """Log a test event with timestamp."""
        timestamp = datetime.now().isoformat()
        event = {
            "timestamp": timestamp,
            "type": event_type,
            "message": message,
            "elapsed": time.time() - self.start_time
        }
        self.test_results["events"].append(event)
        print(f"[{event_type.upper()}] {message} ({event['elapsed']:.1f}s elapsed)")
    
    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def _update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            frames_since_update = self.frame_count - getattr(self, '_last_frame_count', 0)
            self.fps = frames_since_update / (current_time - self.last_fps_update)
            self.fps_history.append(self.fps)
            self._last_frame_count = self.frame_count
            self.last_fps_update = current_time
    
    def _simulate_pose_loss(self, frame: np.ndarray, probability: float = 0.01) -> np.ndarray:
        """
        Simulate pose loss for testing recovery.
        Returns original frame or blurred frame to simulate poor detection.
        """
        if np.random.random() < probability:
            # Add slight blur to simulate poor conditions
            return cv2.GaussianBlur(frame, (5, 5), 0)
        return frame
    
    def run_test(self) -> Dict:
        """
        Run the complete stability test.
        Returns test results dictionary.
        """
        print("="*70)
        print("PHYSIO INTELLIGENCE v3 - STABILITY TEST")
        print("="*70)
        print(f"Duration: {self.duration_seconds // 60} minutes")
        print(f"Output Directory: {self.output_dir}")
        print("="*70)
        
        self.start_time = time.time()
        self.test_results["test_start"] = datetime.now().isoformat()
        self._log_event("info", "Starting stability test")
        
        # Initialize components
        try:
            engine = PhysioIntelligence()
            self._log_event("success", "PhysioIntelligence initialized")
        except Exception as e:
            self.test_results["errors"].append(str(e))
            self._log_event("error", f"Failed to initialize: {e}")
            return self.test_results
        
        # Initialize camera
        if not engine.initialize_camera():
            self._log_event("error", "Failed to initialize camera")
            self.test_results["errors"].append("Camera initialization failed")
            return self.test_results
        
        self._log_event("success", "Camera initialized")
        
        # Start session
        engine.session_manager.start_session()
        self._log_event("info", "Session started")
        
        print("\n" + "="*70)
        print("STABILITY TEST IN PROGRESS")
        print("="*70)
        print(f"Time | FPS | Memory | Frames | Dropped | No-Pose")
        print("-"*70)
        
        # Main test loop
        frame_times = []
        start_loop = time.time()
        
        try:
            while (time.time() - start_loop) < self.duration_seconds:
                loop_start = time.time()
                
                # Check for pause
                if engine.is_paused:
                    time.sleep(0.1)
                    continue
                
                # Read frame
                ret, frame = engine.cap.read()
                
                if not ret:
                    self.dropped_frames += 1
                    if self.dropped_frames % 30 == 0:
                        self._log_event("warn", f"Dropped frame #{self.dropped_frames}")
                    continue
                
                # Simulate occasional pose loss for recovery testing
                test_frame = self._simulate_pose_loss(frame, probability=0.005)
                
                # Process frame
                annotated_frame, pose_data = engine.process_frame(test_frame)
                
                # Update metrics
                self.frame_count += 1
                self.successful_frames += 1
                
                if pose_data["phase"] == "no_pose":
                    self.no_pose_frames += 1
                
                # Update FPS
                self._update_fps()
                self.memory_history.append(self._get_memory_mb())
                
                # Progress output every 30 seconds
                elapsed = time.time() - start_loop
                if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                    elapsed_min = elapsed / 60
                    progress = elapsed / self.duration_seconds * 100
                    print(f"{elapsed_min:5.1f}m | {self.fps:5.1f} | {self.memory_history[-1]:6.1f}MB | "
                          f"{self.frame_count:6} | {self.dropped_frames:6} | {self.no_pose_frames:6}")
                    self._log_event("info", f"Progress: {progress:.1f}% - FPS: {self.fps:.1f}")
                
                # Display frame (comment out for headless testing)
                cv2.imshow('Stability Test - Physio Intelligence v3', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self._log_event("info", "User requested stop")
                    break
                elif key == ord('p'):
                    if engine.is_paused:
                        engine.resume_session()
                        self._log_event("info", "Resumed from pause")
                    else:
                        engine.pause_session()
                        self._log_event("info", "Paused by user")
            
        except Exception as e:
            self.test_results["errors"].append(str(e))
            self._log_event("error", f"Unexpected error: {e}")
        
        finally:
            # Cleanup
            engine.cleanup()
            cv2.destroyAllWindows()
        
        # Calculate final metrics
        total_time = time.time() - start_loop
        self.test_results["test_end"] = datetime.now().isoformat()
        self.test_results["duration_actual_seconds"] = total_time
        
        # Calculate statistics
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        max_memory = max(self.memory_history) if self.memory_history else 0
        avg_memory = sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0
        drop_rate = (self.dropped_frames / self.frame_count * 100) if self.frame_count > 0 else 0
        no_pose_rate = (self.no_pose_frames / self.frame_count * 100) if self.frame_count > 0 else 0
        
        self.test_results["metrics"] = {
            "total_frames": self.frame_count,
            "successful_frames": self.successful_frames,
            "dropped_frames": self.dropped_frames,
            "no_pose_frames": self.no_pose_frames,
            "drop_rate_percent": round(drop_rate, 2),
            "no_pose_rate_percent": round(no_pose_rate, 2),
            "average_fps": round(avg_fps, 2),
            "max_fps": round(max(self.fps_history), 2) if self.fps_history else 0,
            "min_fps": round(min(self.fps_history), 2) if self.fps_history else 0,
            "memory_max_mb": round(max_memory, 2),
            "memory_avg_mb": round(avg_memory, 2),
            "duration_actual_minutes": round(total_time / 60, 2)
        }
        
        # Determine pass/fail
        passed = (
            total_time >= self.duration_seconds * 0.9 and  # Ran for 90% of target
            avg_fps >= 20 and  # At least 20 FPS average
            drop_rate < 5 and  # Less than 5% dropped frames
            len(self.test_results["errors"]) == 0  # No errors
        )
        self.test_results["passed"] = passed
        
        # Print final results
        self._print_final_results()
        
        # Save results
        self._save_results()
        
        return self.test_results
    
    def _print_final_results(self):
        """Print final test results."""
        m = self.test_results["metrics"]
        
        print("\n" + "="*70)
        print("STABILITY TEST RESULTS")
        print("="*70)
        print(f"Duration: {m['duration_actual_minutes']} minutes")
        print(f"Total Frames: {m['total_frames']}")
        print(f"Dropped Frames: {m['dropped_frames']} ({m['drop_rate_percent']}%)")
        print(f"No-Pose Frames: {m['no_pose_frames']} ({m['no_pose_rate_percent']}%)")
        print(f"Average FPS: {m['average_fps']}")
        print(f"Max Memory: {m['memory_max_mb']} MB")
        print(f"Average Memory: {m['memory_avg_mb']} MB")
        print(f"Errors: {len(self.test_results['errors'])}")
        
        status = "PASSED ✓" if self.test_results["passed"] else "FAILED ✗"
        print(f"\nTest Status: {status}")
        print("="*70)
    
    def _save_results(self):
        """Save test results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"stability_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")


class DemoVideoRecorder:
    """
    Helper class for recording demo videos.
    Note: Actual video recording requires user action with external tools.
    This provides setup instructions and configuration.
    """
    
    @staticmethod
    def get_setup_instructions() -> str:
        """Get instructions for recording demo video."""
        return """
DEMO VIDEO RECORDING INSTRUCTIONS
==================================

REQUIRED EQUIPMENT:
- Webcam (1080p recommended)
- Good lighting (face clearly visible)
- Quiet environment

SETUP STEPS:

1. Position Camera:
   - 3-5 feet from camera
   - Good lighting on face/upper body
   - Show left side to camera

2. Prepare Environment:
   - Clear background
   - Remove obstacles
   - Ensure stable internet (not required but good)

3. Recording Software Options:
   a) QuickTime (macOS): File > New Movie Recording
   b) OBS Studio: Free, cross-platform, professional
   c) iMovie (macOS): For editing
   d) Windows Game Bar: Win+G to record

4. Recommended Recording Flow:

   [0:00-0:30] Introduction
   - Show system startup
   - Explain controls (q, p, s)
   
   [0:30-2:00] Shoulder Raise Demo
   - Perform 10 reps with good form
   - Show real-time angle display
   - Show phase detection (RAISE/HOLD/LOWER)
   - Show feedback messages
   
   [2:00-3:00] Form Corrections
   - Deliberately make mistakes
   - Show warning messages
   - Show severity-based feedback
   
   [3:00-4:00] Pause/Resume Demo
   - Press 'p' to pause
   - Show pause indicator
   - Press 'r' to resume
   
   [4:00-5:00] Session Summary
   - Press 's' for summary
   - Show JSON output
   - Explain metrics

5. Post-Recording:
   - Trim silence/empty sections
   - Add title card
   - Add captions for clarity
   - Export as MP4, < 3 minutes

TECHNICAL CHECKLIST:
✓ System starts without errors
✓ Webcam activates immediately
✓ Pose detection starts within 3 seconds
✓ Angle updates in real-time (<100ms latency)
✓ Phase detection accurate (>90%)
✓ Feedback appears with 2-second minimum display
✓ Session summary generates correctly
✓ No crashes during 5+ minute recording

"""
    
    @staticmethod
    def get_demo_script() -> str:
        """Get a sample demo script."""
        return """
SAMPLE DEMO SCRIPT
==================

[INTRO - 30 seconds]
"Hello, I'm demonstrating Physio Intelligence v3, a real-time
physiotherapy exercise tracking system built with MediaPipe."

[SYSTEM START - 30 seconds]
"Let's start the application. You can see the camera initializing
and the session starting automatically."

[EXERCISE DEMO - 2 minutes]
"Now I'll perform shoulder raises. Notice how the system
tracks my arm angle in real-time.

As I raise my arm, the phase changes to RAISE.
When I reach the top, it switches to HOLD.
As I lower, it shows LOWER.

The feedback appears at the bottom of the screen, giving me
form corrections like 'Control the movement speed' or
'Keep slight bend in elbow'."

[FORM CORRECTION - 30 seconds]
"If I let my elbow lock, the system gives a warning.
This helps prevent injury and ensures proper form."

[PAUSE/RESUME - 30 seconds]
"I can press 'p' to pause the session at any time.
Press 'p' again to resume. Useful for breaks!"

[SESSION SUMMARY - 30 seconds]
"Press 's' to see the session summary. This shows total reps,
consistency score, safety violations, and more."

[OUTRO - 30 seconds]
"That's Physio Intelligence v3 - real-time exercise tracking
with intelligent feedback and comprehensive session analytics.
Ready for demo and submission."
"""


def main():
    """Main entry point for stability test."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Physio Intelligence v3 Stability Test')
    parser.add_argument('--duration', type=int, default=10, 
                        help='Test duration in minutes (default: 10)')
    parser.add_argument('--output', type=str, default='stability_test_output',
                        help='Output directory for results')
    parser.add_argument('--demo-video', action='store_true',
                        help='Show demo video instructions instead of running test')
    
    args = parser.parse_args()
    
    if args.demo_video:
        print(DemoVideoRecorder.get_setup_instructions())
        print(DemoVideoRecorder.get_demo_script())
        return
    
    # Run stability test
    test = StabilityTest(duration_minutes=args.duration, output_dir=args.output)
    results = test.run_test()
    
    # Exit with appropriate code
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
