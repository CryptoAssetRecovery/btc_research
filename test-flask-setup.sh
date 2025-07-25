#!/bin/bash
# Simple test script to verify Flask setup works

echo "🧪 Testing BTC Research Flask Setup"
echo "=================================="

# Test 1: Check if Docker is running
echo "1. Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi
echo "✅ Docker is running"

# Test 2: Check if files exist
echo "2. Checking required files..."
if [ ! -f "Dockerfile.dev" ]; then
    echo "❌ Dockerfile.dev not found"
    exit 1
fi
if [ ! -f "docker-compose.dev.yml" ]; then
    echo "❌ docker-compose.dev.yml not found"
    exit 1
fi
if [ ! -f "pyproject.toml" ]; then
    echo "❌ pyproject.toml not found"
    exit 1
fi
echo "✅ All required files found"

# Test 3: Build the development image
echo "3. Building development Docker image..."
if ! docker build -f Dockerfile.dev -t btc-research-dev . > build.log 2>&1; then
    echo "❌ Docker build failed. Check build.log for details."
    exit 1
fi
echo "✅ Docker image built successfully"

# Test 4: Check if uvicorn is available in the container
echo "4. Testing uvicorn availability..."
if ! docker run --rm btc-research-dev which uvicorn > /dev/null 2>&1; then
    echo "❌ uvicorn not found in container"
    exit 1
fi
echo "✅ uvicorn is available"

# Test 5: Check if Flask app can be imported
echo "5. Testing Flask app import..."
if ! docker run --rm btc-research-dev python -c "from btc_research.web.app import app; print('Flask app imported successfully')" > /dev/null 2>&1; then
    echo "❌ Flask app import failed"
    exit 1
fi
echo "✅ Flask app can be imported"

echo ""
echo "🎉 All tests passed! Your setup is ready."
echo ""
echo "📋 Quick Start Commands:"
echo "========================"
echo "1. Start dashboard only:"
echo "   docker compose -f docker-compose.dev.yml up dashboard"
echo ""
echo "2. Start full stack (dashboard + API + Redis):"
echo "   docker compose -f docker-compose.dev.yml up"
echo ""
echo "3. View dashboard at: http://localhost:5000"
echo "4. View API docs at: http://localhost:8000/docs"
echo ""
echo "🔥 Hot Reloading:"
echo "=================="
echo "- Edit files in btc_research/web/templates/ - changes will be visible immediately"
echo "- Edit files in btc_research/web/static/ - refresh browser to see changes"
echo "- Edit Python files in btc_research/web/ - Flask will auto-reload"
echo ""
echo "📝 Template files are at: btc_research/web/templates/"
echo "   - base.html, dashboard.html, charts.html, etc."

# Clean up
rm -f build.log