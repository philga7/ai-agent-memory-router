# AI Agent Memory Router - Testing Guide

This comprehensive testing guide covers all aspects of testing the AI Agent Memory Router, including unit tests, integration tests, end-to-end tests, and quality assurance practices.

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Coverage Requirements](#coverage-requirements)
- [Quality Gates](#quality-gates)
- [Continuous Integration](#continuous-integration)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Overview

The AI Agent Memory Router uses a comprehensive testing strategy that includes:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions and API endpoints
- **End-to-End Tests**: Test complete workflows and user scenarios
- **Performance Tests**: Test system performance under various loads
- **Security Tests**: Test security features and vulnerability protection
- **Smoke Tests**: Quick tests for basic functionality verification

## Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_config.py      # Configuration tests
│   ├── test_fixtures.py    # Fixture tests
│   └── test_logging.py     # Logging tests
├── integration/            # Integration tests
│   ├── test_memory_api.py  # Memory API tests
│   ├── test_agents_api.py  # Agents API tests
│   ├── test_health_api.py  # Health API tests
│   └── test_context_api.py # Context API tests
├── e2e/                    # End-to-end tests
│   └── test_complete_workflows.py # Complete workflow tests
├── fixtures/               # Test fixtures and utilities
│   ├── database.py         # Database fixtures
│   ├── mocks.py           # Mock fixtures
│   └── data.py            # Test data fixtures
└── utils/                  # Test utilities
    ├── config.py          # Test configuration
    ├── database.py        # Database utilities
    └── helpers.py         # Test helpers
```

## Running Tests

### Quick Start

```bash
# Run all tests with coverage
python run_tests.py --all

# Run specific test categories
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --e2e

# Run with verbose output
python run_tests.py --all --verbose
```

### Using pytest directly

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m e2e

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test files
pytest tests/unit/test_config.py
pytest tests/integration/test_memory_api.py
```

### Test Runner Options

The `run_tests.py` script provides comprehensive test execution options:

```bash
# Test categories
--unit              # Unit tests
--integration       # Integration tests
--e2e              # End-to-end tests
--fast             # Fast tests only
--smoke            # Smoke tests
--api              # API tests
--performance      # Performance tests
--security         # Security tests

# Coverage options
--coverage         # Generate coverage report
--no-cov           # Disable coverage collection

# Output options
--verbose, -v      # Verbose output
--clean            # Clean up old reports

# Code quality
--lint             # Run code linting
--format           # Format code
--type-check       # Run type checking
--security-scan    # Run security scan
```

## Test Categories

### Unit Tests

Unit tests test individual components in isolation using mocks and fixtures.

**Location**: `tests/unit/`

**Examples**:
- Configuration validation
- Logging functionality
- Utility functions
- Model validation

**Running**:
```bash
python run_tests.py --unit
pytest -m unit
```

### Integration Tests

Integration tests test component interactions and API endpoints with real database operations.

**Location**: `tests/integration/`

**Examples**:
- Memory API endpoints
- Agent management API
- Health monitoring API
- Context management API
- Cipher integration

**Running**:
```bash
python run_tests.py --integration
pytest -m integration
```

### End-to-End Tests

End-to-end tests test complete workflows and user scenarios.

**Location**: `tests/e2e/`

**Examples**:
- Agent onboarding workflow
- Memory routing workflow
- Multi-agent collaboration
- System health monitoring
- Error handling and recovery

**Running**:
```bash
python run_tests.py --e2e
pytest -m e2e
```

### Performance Tests

Performance tests verify system performance under various loads.

**Markers**: `@pytest.mark.performance`

**Examples**:
- API response times
- Memory usage
- Concurrent request handling
- Database performance

**Running**:
```bash
python run_tests.py --performance
pytest -m performance
```

### Security Tests

Security tests verify security features and vulnerability protection.

**Markers**: `@pytest.mark.security`

**Examples**:
- SQL injection protection
- XSS protection
- Input validation
- Authentication and authorization

**Running**:
```bash
python run_tests.py --security
pytest -m security
```

### Smoke Tests

Smoke tests provide quick verification of basic functionality.

**Markers**: `@pytest.mark.smoke`

**Examples**:
- Basic health checks
- Simple API calls
- Database connectivity
- Service startup

**Running**:
```bash
python run_tests.py --smoke
pytest -m smoke
```

## Coverage Requirements

### Minimum Coverage Thresholds

- **Overall Coverage**: 80%
- **Critical Components**: 90%
- **API Endpoints**: 85%
- **Business Logic**: 90%

### Coverage Reports

Coverage reports are generated in multiple formats:

- **Terminal**: `--cov-report=term-missing`
- **HTML**: `--cov-report=html:htmlcov`
- **XML**: `--cov-report=xml:coverage.xml`
- **JUnit**: `--cov-report=junit:junit.xml`

### Viewing Coverage Reports

```bash
# Generate HTML coverage report
python run_tests.py --coverage

# Open HTML report
open htmlcov/index.html
```

## Quality Gates

### Pre-commit Checks

Before committing code, ensure all quality gates pass:

```bash
# Run all quality checks
python run_tests.py --lint --format --type-check --security-scan

# Run tests with coverage
python run_tests.py --all --coverage
```

### Quality Gate Requirements

1. **Code Coverage**: ≥ 80%
2. **Linting**: No flake8 errors
3. **Type Checking**: No mypy errors
4. **Security Scan**: No high/critical vulnerabilities
5. **All Tests**: Must pass
6. **Performance**: Response times within limits

### Continuous Integration

The CI/CD pipeline runs the following checks:

```yaml
# Example CI configuration
test:
  script:
    - python run_tests.py --all --coverage
    - python run_tests.py --lint
    - python run_tests.py --type-check
    - python run_tests.py --security-scan
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
```

## Continuous Integration

### GitHub Actions

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run tests
      run: python run_tests.py --all --coverage
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
```

### Local Development

For local development, use the test runner with appropriate options:

```bash
# Quick development cycle
python run_tests.py --fast --verbose

# Full test suite before commit
python run_tests.py --all --coverage --lint

# Debug specific test
pytest tests/integration/test_memory_api.py::TestMemoryRoutingAPI::test_route_memory_success -v
```

## Troubleshooting

### Common Issues

#### Test Database Issues

```bash
# Clean test database
rm -f data/test_*.db

# Recreate test database
python -c "from tests.fixtures.database import create_test_database; create_test_database()"
```

#### Import Errors

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run with module flag
python -m pytest tests/
```

#### Coverage Issues

```bash
# Clean coverage data
rm -f .coverage coverage*.xml

# Regenerate coverage
python run_tests.py --coverage
```

#### Performance Test Failures

```bash
# Run performance tests with more time
pytest -m performance --timeout=600

# Check system resources
python run_tests.py --performance --verbose
```

### Debug Mode

Enable debug mode for detailed test output:

```bash
# Set debug environment
export DEBUG=1
export LOG_LEVEL=DEBUG

# Run tests with debug output
python run_tests.py --all --verbose
```

### Test Isolation

Ensure tests run in isolation:

```bash
# Run tests in random order
pytest --random-order

# Run tests with fresh database
pytest --create-db
```

## Best Practices

### Writing Tests

1. **Test Naming**: Use descriptive test names that explain the scenario
2. **Arrange-Act-Assert**: Structure tests with clear sections
3. **One Assertion**: Each test should verify one specific behavior
4. **Independent Tests**: Tests should not depend on each other
5. **Mock External Dependencies**: Use mocks for external services

### Test Data

1. **Use Fixtures**: Leverage pytest fixtures for test data
2. **Realistic Data**: Use realistic test data that matches production
3. **Edge Cases**: Test boundary conditions and edge cases
4. **Clean Up**: Ensure tests clean up after themselves

### Performance Testing

1. **Baseline Metrics**: Establish performance baselines
2. **Load Testing**: Test under various load conditions
3. **Resource Monitoring**: Monitor memory and CPU usage
4. **Timeout Handling**: Set appropriate timeouts for tests

### Security Testing

1. **Input Validation**: Test all input validation
2. **Authentication**: Test authentication and authorization
3. **Injection Attacks**: Test for SQL injection, XSS, etc.
4. **Data Protection**: Test data encryption and protection

### Maintenance

1. **Regular Updates**: Keep test dependencies updated
2. **Test Review**: Review tests during code reviews
3. **Refactoring**: Refactor tests when code changes
4. **Documentation**: Keep test documentation updated

## Test Configuration

### Environment Variables

```bash
# Test environment
export TESTING=true
export DATABASE_URL=sqlite:///data/test_ai_agent_memory.db
export LOG_LEVEL=WARNING

# Coverage
export COVERAGE_PROCESS_START=.coveragerc

# Performance
export PERFORMANCE_TEST_TIMEOUT=300
```

### Configuration Files

- **pytest.ini**: Pytest configuration
- **.coveragerc**: Coverage configuration
- **conftest.py**: Test fixtures and configuration
- **run_tests.py**: Test runner script

## Reporting

### Test Reports

Test reports are generated in multiple formats:

- **JUnit XML**: For CI/CD integration
- **HTML**: For human-readable reports
- **Coverage**: For code coverage analysis
- **Performance**: For performance metrics

### Metrics

Key metrics tracked:

- **Test Coverage**: Code coverage percentage
- **Test Duration**: Time to run tests
- **Test Success Rate**: Percentage of passing tests
- **Performance Metrics**: Response times and throughput
- **Security Issues**: Number of security vulnerabilities

## Conclusion

This testing guide provides comprehensive coverage of the AI Agent Memory Router testing strategy. Follow these guidelines to ensure high-quality, reliable, and maintainable tests that support the development and deployment of the system.

For questions or issues, refer to the troubleshooting section or contact the development team.