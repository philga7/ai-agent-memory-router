#!/usr/bin/env python3
"""
Test runner script for AI Agent Memory Router.

This script provides a comprehensive test runner with various options for
running different types of tests, generating reports, and managing test execution.
"""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Optional


class TestRunner:
    """Test runner for AI Agent Memory Router."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / "tests"
        self.reports_dir = self.project_root / "artifacts" / "reports"
        self.logs_dir = self.project_root / "logs"
        
        # Ensure directories exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def run_command(self, command: List[str], description: str) -> bool:
        """Run a command and return success status."""
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(command)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=False,
                text=True,
                check=True
            )
            
            duration = time.time() - start_time
            print(f"\n✅ {description} completed successfully in {duration:.2f} seconds")
            return True
            
        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            print(f"\n❌ {description} failed after {duration:.2f} seconds")
            print(f"Exit code: {e.returncode}")
            return False
        except Exception as e:
            duration = time.time() - start_time
            print(f"\n❌ {description} failed with error after {duration:.2f} seconds")
            print(f"Error: {e}")
            return False
    
    def run_unit_tests(self, verbose: bool = False, coverage: bool = True) -> bool:
        """Run unit tests."""
        command = ["python", "-m", "pytest", "tests/unit/"]
        
        if verbose:
            command.append("-v")
        
        if coverage:
            command.extend([
                "--cov=app",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/unit",
                "--cov-report=xml:coverage-unit.xml"
            ])
        
        command.extend([
            "-m", "unit",
            "--tb=short",
            "--durations=10",
            "-W", "ignore::DeprecationWarning",
            "-W", "ignore::PendingDeprecationWarning",
            "-W", "ignore::UserWarning"
        ])
        
        return self.run_command(command, "Unit Tests")
    
    def run_integration_tests(self, verbose: bool = False, coverage: bool = True) -> bool:
        """Run integration tests."""
        command = ["python", "-m", "pytest", "tests/integration/"]
        
        if verbose:
            command.append("-v")
        
        if coverage:
            command.extend([
                "--cov=app",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/integration",
                "--cov-report=xml:coverage-integration.xml"
            ])
        
        command.extend([
            "-m", "integration",
            "--tb=short",
            "--durations=10",
            "-W", "ignore::DeprecationWarning",
            "-W", "ignore::PendingDeprecationWarning",
            "-W", "ignore::UserWarning"
        ])
        
        return self.run_command(command, "Integration Tests")
    
    def run_e2e_tests(self, verbose: bool = False, coverage: bool = True) -> bool:
        """Run end-to-end tests."""
        command = ["python", "-m", "pytest", "tests/e2e/"]
        
        if verbose:
            command.append("-v")
        
        if coverage:
            command.extend([
                "--cov=app",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/e2e",
                "--cov-report=xml:coverage-e2e.xml"
            ])
        
        command.extend([
            "-m", "e2e",
            "--tb=short",
            "--durations=10",
            "-W", "ignore::DeprecationWarning",
            "-W", "ignore::PendingDeprecationWarning",
            "-W", "ignore::UserWarning"
        ])
        
        return self.run_command(command, "End-to-End Tests")
    
    def run_all_tests(self, verbose: bool = False, coverage: bool = True) -> bool:
        """Run all tests."""
        command = ["python", "-m", "pytest", "tests/"]
        
        if verbose:
            command.append("-v")
        
        if coverage:
            command.extend([
                "--cov=app",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml:coverage.xml",
                "--cov-fail-under=80"
            ])
        
        command.extend([
            "--tb=short",
            "--durations=10",
            "--maxfail=5",
            "--junitxml=test-results.xml",
            "--html=test-results.html",
            "--self-contained-html",
            "-W", "ignore::DeprecationWarning",
            "-W", "ignore::PendingDeprecationWarning",
            "-W", "ignore::UserWarning"
        ])
        
        return self.run_command(command, "All Tests")
    
    def run_fast_tests(self, verbose: bool = False) -> bool:
        """Run only fast tests."""
        command = ["python", "-m", "pytest", "tests/", "-m", "fast"]
        
        if verbose:
            command.append("-v")
        
        command.extend([
            "--tb=short",
            "--durations=5"
        ])
        
        return self.run_command(command, "Fast Tests")
    
    def run_smoke_tests(self, verbose: bool = False) -> bool:
        """Run smoke tests."""
        command = ["python", "-m", "pytest", "tests/", "-m", "smoke"]
        
        if verbose:
            command.append("-v")
        
        command.extend([
            "--tb=short",
            "--durations=5"
        ])
        
        return self.run_command(command, "Smoke Tests")
    
    def run_api_tests(self, verbose: bool = False) -> bool:
        """Run API tests."""
        command = ["python", "-m", "pytest", "tests/", "-m", "api"]
        
        if verbose:
            command.append("-v")
        
        command.extend([
            "--tb=short",
            "--durations=10"
        ])
        
        return self.run_command(command, "API Tests")
    
    def run_performance_tests(self, verbose: bool = False) -> bool:
        """Run performance tests."""
        command = ["python", "-m", "pytest", "tests/", "-m", "performance"]
        
        if verbose:
            command.append("-v")
        
        command.extend([
            "--tb=short",
            "--durations=20"
        ])
        
        return self.run_command(command, "Performance Tests")
    
    def run_security_tests(self, verbose: bool = False) -> bool:
        """Run security tests."""
        command = ["python", "-m", "pytest", "tests/", "-m", "security"]
        
        if verbose:
            command.append("-v")
        
        command.extend([
            "--tb=short",
            "--durations=10"
        ])
        
        return self.run_command(command, "Security Tests")
    
    def run_coverage_report(self) -> bool:
        """Generate comprehensive coverage report."""
        command = [
            "python", "-m", "coverage", "combine",
            "coverage-unit.xml",
            "coverage-integration.xml",
            "coverage-e2e.xml"
        ]
        
        if not self.run_command(command, "Combine Coverage Reports"):
            return False
        
        command = [
            "python", "-m", "coverage", "report",
            "--show-missing",
            "--fail-under=80"
        ]
        
        if not self.run_command(command, "Generate Coverage Report"):
            return False
        
        command = [
            "python", "-m", "coverage", "html",
            "--directory=htmlcov",
            "--title=AI Agent Memory Router Coverage Report"
        ]
        
        return self.run_command(command, "Generate HTML Coverage Report")
    
    def lint_code(self) -> bool:
        """Run code linting."""
        command = ["python", "-m", "flake8", "app/", "tests/"]
        return self.run_command(command, "Code Linting (flake8)")
    
    def format_code(self) -> bool:
        """Format code."""
        command = ["python", "-m", "black", "app/", "tests/"]
        return self.run_command(command, "Code Formatting (black)")
    
    def type_check(self) -> bool:
        """Run type checking."""
        command = ["python", "-m", "mypy", "app/"]
        return self.run_command(command, "Type Checking (mypy)")
    
    def security_scan(self) -> bool:
        """Run security scan."""
        command = ["python", "-m", "bandit", "-r", "app/", "-f", "json", "-o", "security-report.json"]
        return self.run_command(command, "Security Scan (bandit)")
    
    def clean_reports(self) -> bool:
        """Clean up old reports."""
        import shutil
        
        try:
            # Clean up old coverage reports
            for pattern in ["htmlcov*", "coverage*.xml", "junit*.xml", "test-results.*"]:
                for path in self.project_root.glob(pattern):
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
            
            # Clean up old test artifacts
            artifacts_dir = self.project_root / "artifacts"
            if artifacts_dir.exists():
                shutil.rmtree(artifacts_dir)
            
            print("✅ Cleaned up old reports and artifacts")
            return True
            
        except Exception as e:
            print(f"❌ Failed to clean up reports: {e}")
            return False


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Test runner for AI Agent Memory Router",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all                    # Run all tests with coverage
  python run_tests.py --unit --verbose         # Run unit tests with verbose output
  python run_tests.py --integration --no-cov   # Run integration tests without coverage
  python run_tests.py --e2e --fast             # Run only fast e2e tests
  python run_tests.py --smoke                  # Run smoke tests
  python run_tests.py --api --performance      # Run API and performance tests
  python run_tests.py --coverage               # Generate coverage report only
  python run_tests.py --lint --format          # Run linting and formatting
  python run_tests.py --clean                  # Clean up old reports
        """
    )
    
    # Test type options
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests")
    parser.add_argument("--fast", action="store_true", help="Run only fast tests")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests")
    parser.add_argument("--api", action="store_true", help="Run API tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--security", action="store_true", help="Run security tests")
    
    # Coverage options
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--no-cov", action="store_true", help="Disable coverage collection")
    
    # Output options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--clean", action="store_true", help="Clean up old reports")
    
    # Code quality options
    parser.add_argument("--lint", action="store_true", help="Run code linting")
    parser.add_argument("--format", action="store_true", help="Format code")
    parser.add_argument("--type-check", action="store_true", help="Run type checking")
    parser.add_argument("--security-scan", action="store_true", help="Run security scan")
    
    args = parser.parse_args()
    
    # If no specific tests are requested, run all tests
    if not any([
        args.all, args.unit, args.integration, args.e2e, args.fast, args.smoke,
        args.api, args.performance, args.security, args.coverage, args.clean,
        args.lint, args.format, args.type_check, args.security_scan
    ]):
        args.all = True
    
    runner = TestRunner()
    success = True
    
    # Clean up old reports if requested
    if args.clean:
        success &= runner.clean_reports()
    
    # Run code quality checks
    if args.lint:
        success &= runner.lint_code()
    
    if args.format:
        success &= runner.format_code()
    
    if args.type_check:
        success &= runner.type_check()
    
    if args.security_scan:
        success &= runner.security_scan()
    
    # Run tests
    coverage = not args.no_cov
    
    if args.all:
        success &= runner.run_all_tests(verbose=args.verbose, coverage=coverage)
    else:
        if args.unit:
            success &= runner.run_unit_tests(verbose=args.verbose, coverage=coverage)
        
        if args.integration:
            success &= runner.run_integration_tests(verbose=args.verbose, coverage=coverage)
        
        if args.e2e:
            success &= runner.run_e2e_tests(verbose=args.verbose, coverage=coverage)
        
        if args.fast:
            success &= runner.run_fast_tests(verbose=args.verbose)
        
        if args.smoke:
            success &= runner.run_smoke_tests(verbose=args.verbose)
        
        if args.api:
            success &= runner.run_api_tests(verbose=args.verbose)
        
        if args.performance:
            success &= runner.run_performance_tests(verbose=args.verbose)
        
        if args.security:
            success &= runner.run_security_tests(verbose=args.verbose)
    
    # Generate coverage report if requested
    if args.coverage:
        success &= runner.run_coverage_report()
    
    # Print final results
    print(f"\n{'='*60}")
    if success:
        print("🎉 All operations completed successfully!")
        sys.exit(0)
    else:
        print("❌ Some operations failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
