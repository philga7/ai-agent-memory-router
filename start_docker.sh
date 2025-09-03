#!/bin/bash

# Docker startup script for AI Agent Memory Router
# This script helps you start, stop, and manage your Docker environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop or Docker daemon."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check if docker-compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_error "docker-compose is not installed or not in PATH"
        exit 1
    fi
    print_success "docker-compose is available"
}

# Function to start services
start_services() {
    print_status "Starting Docker services..."
    
    # Create necessary directories
    mkdir -p data logs
    
    # Start services
    docker-compose up -d
    
    print_success "Services started successfully!"
    print_status "Waiting for services to be ready..."
    
    # Wait for services to be healthy
    sleep 10
    
    # Show service status
    docker-compose ps
    
    print_status "Services are starting up. You can check logs with:"
    echo "  docker-compose logs -f <service_name>"
    echo ""
    print_status "Test your setup with:"
    echo "  python test_docker_setup.py"
}

# Function to stop services
stop_services() {
    print_status "Stopping Docker services..."
    docker-compose down
    print_success "Services stopped successfully!"
}

# Function to restart services
restart_services() {
    print_status "Restarting Docker services..."
    docker-compose down
    docker-compose up -d
    print_success "Services restarted successfully!"
}

# Function to show logs
show_logs() {
    if [ -z "$1" ]; then
        print_status "Showing logs for all services..."
        docker-compose logs -f
    else
        print_status "Showing logs for service: $1"
        docker-compose logs -f "$1"
    fi
}

# Function to show status
show_status() {
    print_status "Docker service status:"
    docker-compose ps
}

# Function to clean up
cleanup() {
    print_warning "This will remove all containers, networks, and volumes!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Cleaning up Docker environment..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        print_success "Cleanup completed!"
    else
        print_status "Cleanup cancelled"
    fi
}

# Function to show help
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     Start all Docker services"
    echo "  stop      Stop all Docker services"
    echo "  restart   Restart all Docker services"
    echo "  status    Show service status"
    echo "  logs      Show logs (all services or specific service)"
    echo "  cleanup   Remove all containers, networks, and volumes"
    echo "  test      Run the test script to verify setup"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start                    # Start all services"
    echo "  $0 logs api                 # Show logs for API service"
    echo "  $0 test                     # Run test script"
}

# Function to run tests
run_tests() {
    print_status "Running Docker setup tests..."
    if [ -f "test_docker_setup.py" ]; then
        python test_docker_setup.py
    else
        print_error "Test script not found: test_docker_setup.py"
        exit 1
    fi
}

# Main script logic
main() {
    case "${1:-start}" in
        start)
            check_docker
            check_docker_compose
            start_services
            ;;
        stop)
            check_docker
            stop_services
            ;;
        restart)
            check_docker
            restart_services
            ;;
        status)
            check_docker
            show_status
            ;;
        logs)
            check_docker
            show_logs "$2"
            ;;
        cleanup)
            check_docker
            cleanup
            ;;
        test)
            run_tests
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
