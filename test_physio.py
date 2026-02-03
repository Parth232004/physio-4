"""
Test script for Physio Intelligence v3
Tests core components without webcam for validation
"""

import sys
import time
sys.path.insert(0, '.')

from physio_intelligence import (
    Config, calculate_angle, PhaseDetector, 
    FeedbackSystem, SessionManager, PhysioIntelligence
)
import numpy as np


def test_angle_calculation():
    """Test angle calculation function."""
    print("Testing angle calculation...")
    
    # Test cases: a, b, c points
    test_cases = [
        # Straight line (180 degrees at point b)
        ((0, 1), (0, 0), (0, -1)),
        # Right angle (90 degrees)
        ((1, 0), (0, 0), (0, 1)),
        # 45 degree angle
        ((1, 1), (0, 0), (1, 0)),
    ]
    
    expected_angles = [180, 90, 45]
    
    for i, (a, b, c) in enumerate(test_cases):
        angle = calculate_angle(a, b, c)
        print(f"  Test {i+1}: Calculated {angle:.1f}°, Expected {expected_angles[i]}°")
        assert abs(angle - expected_angles[i]) < 1.0, f"Angle mismatch: {angle} vs {expected_angles[i]}"
    
    print("✓ Angle calculation tests passed\n")


def test_phase_detection():
    """Test phase detection."""
    print("Testing phase detection...")
    
    detector = PhaseDetector()
    timestamp = time.time()
    
    # Simulate raise phase to peak
    angles = [20, 40, 60, 80, 100, 120, 140, 150, 160, 170]
    for i, angle in enumerate(angles):
        phase = detector.update(angle, timestamp + i * 0.1)
        print(f"  Angle {angle}° -> Phase: {phase}")
    
    # Verify we reached hold phase (should be at peak angle)
    assert detector.current_phase == "hold", f"Expected hold phase, got {detector.current_phase}"
    
    # Simulate hold phase
    for i in range(5):
        phase = detector.update(165, timestamp + 10 + i * 0.5)
    print(f"  After hold: Phase: {phase}")
    print("✓ Phase detection tests passed\n")


def test_feedback_system():
    """Test feedback system with queued display."""
    print("Testing feedback system...")
    
    feedback_sys = FeedbackSystem()
    timestamp = time.time()
    
    # Queue feedback
    issues = ["mild: Watch your form"]
    fb = feedback_sys.queue_feedback("raise", 45.0, issues, timestamp)
    assert fb is not None, "Feedback should be queued"
    print(f"  Queued feedback: {fb['message']} (severity: {fb['severity']})")
    
    # Get display message (should return the queued message)
    display = feedback_sys.get_display_message(timestamp)
    assert display is not None, "Display message should be available"
    print(f"  Display message: {display['message']}")
    
    # Try to queue more feedback immediately (should be blocked by cooldown)
    fb2 = feedback_sys.queue_feedback("raise", 50.0, issues, timestamp + 1)
    assert fb2 is None, "Feedback should be None due to cooldown"
    print("  Cooldown working correctly")
    
    # Queue different severity feedback
    fb3 = feedback_sys.queue_feedback("hold", 90.0, ["stop: Too extreme"], timestamp + 10)
    assert fb3 is not None, "High severity feedback should go through"
    print(f"  High severity feedback: {fb3['message']} (severity: {fb3['severity']})")
    
    # Test queue count
    queue_count = feedback_sys.get_queued_count()
    print(f"  Queued messages: {queue_count}")
    
    print("✓ Feedback system tests passed\n")


def test_session_manager():
    """Test session manager."""
    print("Testing session manager...")
    
    session = SessionManager()
    
    # Start session
    session.start_session()
    print(f"  Session started: {session.is_active}")
    
    # Simulate pose updates
    timestamp = time.time()
    for i in range(5):
        session.update_pose(30 + i * 20, "raise", timestamp + i * 0.5)
    
    session.update_pose(160, "hold", timestamp + 3)
    session.update_pose(30, "idle", timestamp + 4)
    
    # Check summary
    summary = session.get_summary()
    print(f"  Total reps: {summary['total_reps']}")
    print(f"  Completed reps: {summary['completed_reps']}")
    print(f"  Consistency score: {summary['consistency_score']}%")
    
    # Stop session
    session.stop_session()
    print(f"  Session ended: {session.is_active}")
    
    print("✓ Session manager tests passed\n")


def test_integration():
    """Integration test for full system initialization."""
    print("Testing system initialization...")
    
    try:
        engine = PhysioIntelligence()
        print("  ✓ PhysioIntelligence initialized successfully")
        
        # Check all components exist
        assert hasattr(engine, 'phase_detector')
        assert hasattr(engine, 'feedback_system')
        assert hasattr(engine, 'session_manager')
        print("  ✓ All components initialized")
        
        print("✓ Integration tests passed\n")
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        raise


def main():
    """Run all tests."""
    print("="*60)
    print("Physio Intelligence v3 - Component Tests")
    print("="*60 + "\n")
    
    try:
        test_angle_calculation()
        test_phase_detection()
        test_feedback_system()
        test_session_manager()
        test_integration()
        
        print("="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nSystem is ready for live webcam demo.")
        print("Run: python physio_intelligence.py")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
