#!/bin/bash

# Start Weaviate for testing the AI Agent Memory Router integration

echo "🚀 Starting Weaviate for AI Agent Memory Router testing..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Start Weaviate services (integrated with main Docker Compose)
echo "📦 Starting Weaviate and transformers services..."
docker-compose up -d weaviate t2v-transformers

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check if Weaviate is ready
echo "🔍 Checking Weaviate health..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -f http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; then
        echo "✅ Weaviate is ready!"
        break
    else
        echo "⏳ Attempt $attempt/$max_attempts - Weaviate not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    echo "❌ Weaviate failed to start within the expected time."
    echo "📋 Checking service logs..."
    docker-compose -f docker-compose.weaviate.yml logs weaviate
    exit 1
fi

# Check if transformers service is ready
echo "🔍 Checking transformers service health..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -f http://localhost:8080/health > /dev/null 2>&1; then
        echo "✅ Transformers service is ready!"
        break
    else
        echo "⏳ Attempt $attempt/$max_attempts - Transformers service not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    echo "⚠️  Transformers service may not be ready, but Weaviate is running."
    echo "📋 You can still test basic Weaviate functionality."
fi

echo ""
echo "🎉 Weaviate is ready for testing!"
echo ""
echo "📋 Service URLs:"
echo "   - Weaviate API: http://localhost:8080"
echo "   - Weaviate GraphQL: http://localhost:8080/v1/graphql"
echo "   - Weaviate REST: http://localhost:8080/v1"
echo ""
echo "🧪 To test the integration, run:"
echo "   python test_weaviate_integration.py"
echo ""
echo "🛑 To stop Weaviate, run:"
echo "   docker-compose stop weaviate t2v-transformers"
echo "   # Or stop all services:"
echo "   docker-compose down"
echo ""
