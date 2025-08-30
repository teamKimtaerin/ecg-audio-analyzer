#!/usr/bin/env python3
"""
Test runner for ECG Audio Analysis
Comprehensive test execution with reporting and coverage
"""

import sys
import subprocess
from pathlib import Path
import argparse
from typing import List, Optional

def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status"""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run ECG Audio Analysis tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--gpu", action="store_true", help="Include GPU tests")
    parser.add_argument("--parallel", "-n", type=int, help="Number of parallel workers")
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.extend(["-v", "-s"])
    else:
        cmd.append("-q")
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term-missing"])
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    # Add test selection markers
    markers = []
    if args.unit:
        markers.append("unit")
    elif args.integration:
        markers.append("integration")
    elif args.performance:
        markers.append("performance")
    
    if args.fast:
        markers.append("not slow")
    
    if not args.gpu:
        markers.append("not gpu")
    
    if markers:
        cmd.extend(["-m", " and ".join(markers)])
    
    # Set test paths
    test_paths = []
    if args.unit:
        test_paths.append("tests/unit")
    elif args.integration:
        test_paths.append("tests/integration")
    elif args.performance:
        test_paths.append("tests/performance")
    else:
        test_paths.append("tests")
    
    cmd.extend(test_paths)
    
    # Run the tests
    print("🧪 Running ECG Audio Analysis Tests")
    print("=" * 60)
    
    success = run_command(cmd, "Running pytest")
    
    if success:
        print("\n✅ All tests passed!")
        
        if args.coverage:
            print("\n📊 Coverage report generated in htmlcov/index.html")
        
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())