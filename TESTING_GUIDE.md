# Testing Guide for Intelligent Memory Routing Engine

This guide explains how to test the Intelligent Memory Routing Engine to ensure it's working correctly.

## 🧪 **Testing Overview**

The routing system includes comprehensive testing at multiple levels:
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end system testing
- **Performance Tests**: Load and performance validation
- **Demo Scripts**: Interactive demonstrations

## 📋 **Prerequisites**

1. **Python Environment**: Ensure you have Python 3.8+ installed
2. **Dependencies**: Install required packages
3. **Database**: SQLite database will be created automatically for testing

## 🚀 **Quick Start Testing**

### Option 1: Run Basic Tests (Recommended for first-time testing)

```bash
# Run the basic test suite
python run_routing_tests.py
```

This will run a comprehensive test suite that validates:
- Content classification accuracy
- Routing decision logic
- Service integration
- Performance benchmarks

### Option 2: Interactive Demo

```bash
# Run the interactive demo
python demo_routing_system.py
```

This provides a visual demonstration of the routing system in action.

### Option 3: Full Test Suite with pytest

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest test_routing_engine.py -v

# Run with coverage
pytest test_routing_engine.py --cov=app --cov-report=html
```

## 🔍 **Test Categories**

### 1. Content Classification Tests

Tests the ability to correctly classify different types of content:

```python
# Example test content
conversation = "Hi there! How are you doing today?"
code = "def hello(): print('Hello, World!')"
knowledge = "Machine learning is a subset of AI..."
```

**What to expect:**
- Conversation content → High confidence classification as "conversation"
- Code content → High confidence classification as "code"
- Knowledge content → High confidence classification as "knowledge"

### 2. Routing Decision Tests

Tests the intelligent routing decisions:

```python
# Test scenarios
- High-priority conversation → Should route to Cipher or SQLite
- Code content → Should route to SQLite for fast access
- Large documents → Should route to Weaviate for distributed storage
- Critical content → Should route to SQLite for reliability
```

**What to expect:**
- Decisions should be made within 100ms
- Confidence scores should be > 0.5 for clear content types
- Reasons should be meaningful and explain the decision

### 3. Batch Processing Tests

Tests the ability to process multiple routing requests:

```python
# Batch of 10 requests
- Should process in parallel
- Success rate should be > 80%
- Total processing time should be < 5 seconds
```

**What to expect:**
- Parallel processing should be faster than sequential
- Error handling should be graceful
- All requests should be processed

### 4. Performance Tests

Tests system performance under load:

```python
# Performance benchmarks
- Single request: < 100ms
- Batch of 10 requests: < 5 seconds
- Memory usage: < 100MB
- Database operations: < 50ms
```

## 📊 **Expected Test Results**

### Successful Test Run Output

```
🧪 Running Basic Routing Engine Tests...

Testing Content Classifier...
✓ Content classified as: conversation (confidence: 0.85)

Testing Routing Engine...
✓ Routing decision: cipher (confidence: 0.78)
✓ Reason: Conversational content optimized for Cipher
✓ Processing time: 0.045s

Testing Routing Service...
✓ Single routing successful: cipher
✓ Batch routing successful: 100.0% success rate
✓ System health: healthy

Testing Performance...
✓ Performance test passed

🎉 All tests passed! The routing system is working correctly.
```

### Health Check Results

```
System Health Check:
  Status: HEALTHY
  Policies: 3
  Cache Status: {'cache_size': 5, 'cache_keys': [...]}
  Issues: []
  Recommendations: ['System is operating normally']
```

## 🐛 **Troubleshooting Common Issues**

### Issue 1: Import Errors

**Error**: `ModuleNotFoundError: No module named 'app'`

**Solution**:
```bash
# Ensure you're in the project root directory
cd /path/to/ai-agent-memory-router

# Add the app directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/app"
```

### Issue 2: Database Connection Errors

**Error**: `ConnectionError: Failed to get database connection`

**Solution**:
```bash
# Check if SQLite is available
python -c "import sqlite3; print('SQLite available')"

# Ensure write permissions in the project directory
ls -la data/
```

### Issue 3: Performance Issues

**Error**: Tests are running slowly

**Solution**:
```bash
# Check system resources
top -l 1 | grep Python

# Run tests with reduced load
python run_routing_tests.py --light
```

### Issue 4: Memory Issues

**Error**: `MemoryError` or high memory usage

**Solution**:
```bash
# Monitor memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"

# Reduce batch sizes in tests
# Edit test_routing_engine.py and reduce batch sizes
```

## 📈 **Performance Benchmarks**

### Expected Performance Metrics

| Metric | Target | Acceptable |
|--------|--------|------------|
| Single Request Latency | < 50ms | < 100ms |
| Batch Processing (10 items) | < 2s | < 5s |
| Memory Usage | < 50MB | < 100MB |
| Database Query Time | < 10ms | < 50ms |
| Cache Hit Rate | > 80% | > 60% |

### Load Testing

```bash
# Run load test with 100 requests
python -c "
import asyncio
from test_routing_engine import test_performance_under_load
asyncio.run(test_performance_under_load())
"
```

## 🔧 **Advanced Testing**

### Custom Test Scenarios

Create custom test scenarios by modifying the test files:

```python
# Add to test_routing_engine.py
async def test_custom_scenario():
    """Test a custom routing scenario."""
    context = RoutingContext(
        agent_id="custom-agent",
        project_id="custom-project",
        memory_type="custom",
        content="Your custom content here",
        priority=3,
        tags=["custom", "test"],
        metadata={"custom": True},
        source="custom-test",
        timestamp=datetime.utcnow()
    )
    
    decision = await routing_engine.make_routing_decision(context)
    assert decision.confidence > 0.5
```

### Stress Testing

```bash
# Run stress test
python -c "
import asyncio
import time
from test_routing_engine import TestRoutingService

async def stress_test():
    # Create 1000 requests
    requests = []
    for i in range(1000):
        # Create request...
        pass
    
    start_time = time.time()
    # Process requests
    end_time = time.time()
    
    print(f'Processed 1000 requests in {end_time - start_time:.2f}s')

asyncio.run(stress_test())
"
```

## 📝 **Test Reports**

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=app --cov-report=html

# View report
open htmlcov/index.html
```

### Performance Reports

```bash
# Generate performance report
python run_routing_tests.py --report

# View performance metrics
cat performance_report.json
```

## 🎯 **Validation Checklist**

Before considering the system production-ready, ensure:

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Performance benchmarks are met
- [ ] Health checks return "healthy"
- [ ] No memory leaks detected
- [ ] Database operations are fast
- [ ] Error handling is graceful
- [ ] Logging is comprehensive
- [ ] Documentation is complete

## 🚨 **When Tests Fail**

If tests fail, follow this debugging process:

1. **Check the error message** - Look for specific error details
2. **Review the logs** - Check for additional context
3. **Verify dependencies** - Ensure all packages are installed
4. **Check system resources** - Verify memory and CPU availability
5. **Run individual tests** - Isolate the failing component
6. **Check database state** - Ensure SQLite database is accessible
7. **Review configuration** - Verify all settings are correct

## 📞 **Getting Help**

If you encounter issues:

1. **Check the logs** in the `logs/` directory
2. **Review the error messages** carefully
3. **Run tests individually** to isolate issues
4. **Check system requirements** and dependencies
5. **Verify file permissions** and paths

## 🎉 **Success Criteria**

The routing system is working correctly when:

- ✅ All tests pass consistently
- ✅ Performance benchmarks are met
- ✅ Health checks return "healthy"
- ✅ No errors in logs
- ✅ Memory usage is stable
- ✅ Database operations are fast
- ✅ Routing decisions are logical and consistent

---

**Happy Testing!** 🧪

The Intelligent Memory Routing Engine is designed to be robust and reliable. These tests ensure it meets production standards and performs optimally under various conditions.
