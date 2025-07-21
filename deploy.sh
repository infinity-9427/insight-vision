#!/bin/bash

# ================================
# 🚀 InsightVision Production Deployment Script
# ================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="insightvision-ai-platform"
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
LOG_FILE="./logs/deployment_$(date +%Y%m%d_%H%M%S).log"

# Create directories
mkdir -p logs backups nginx/certs nginx/logs

# Functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

# Pre-deployment checks
pre_deployment_checks() {
    log "Starting pre-deployment checks..."
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        error "Docker is not running. Please start Docker and try again."
    fi
    
    # Check if docker-compose is available
    if ! command -v docker-compose &> /dev/null; then
        error "docker-compose is not installed. Please install it and try again."
    fi
    
    # Check if .env.production exists
    if [ ! -f ".env.production" ]; then
        warn ".env.production not found. Creating from template..."
        cp .env .env.production
        warn "Please edit .env.production with your production settings before continuing."
        read -p "Press Enter to continue after editing .env.production..."
    fi
    
    # Check available disk space (need at least 10GB)
    AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
    if [ "$AVAILABLE_SPACE" -lt 10485760 ]; then  # 10GB in KB
        error "Insufficient disk space. At least 10GB required."
    fi
    
    # Check available memory (recommend at least 16GB)
    AVAILABLE_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$AVAILABLE_MEM" -lt 16 ]; then
        warn "Less than 16GB RAM available. Performance may be affected."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error "Deployment cancelled."
        fi
    fi
    
    log "Pre-deployment checks completed."
}

# Backup existing data
backup_data() {
    log "Creating backup of existing data..."
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup volumes if they exist
    if docker volume ls | grep -q "${PROJECT_NAME}_ollama_data"; then
        docker run --rm -v "${PROJECT_NAME}_ollama_data:/data" -v "$(pwd)/$BACKUP_DIR:/backup" alpine tar czf /backup/ollama_data.tar.gz -C /data .
        log "Ollama data backed up"
    fi
    
    if docker volume ls | grep -q "${PROJECT_NAME}_postgres_data"; then
        docker run --rm -v "${PROJECT_NAME}_postgres_data:/data" -v "$(pwd)/$BACKUP_DIR:/backup" alpine tar czf /backup/postgres_data.tar.gz -C /data .
        log "PostgreSQL data backed up"
    fi
    
    # Backup uploads directory
    if [ -d "backend/uploads" ]; then
        cp -r backend/uploads "$BACKUP_DIR/"
        log "Uploads directory backed up"
    fi
    
    log "Backup completed: $BACKUP_DIR"
}

# Pull latest images
pull_images() {
    log "Pulling latest Docker images..."
    
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.production pull
    
    log "Docker images updated."
}

# Deploy application
deploy() {
    log "Starting deployment..."
    
    # Stop existing containers
    info "Stopping existing containers..."
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.production down
    
    # Build and start new containers
    info "Building and starting containers..."
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.production up --build -d
    
    # Wait for services to be healthy
    info "Waiting for services to become healthy..."
    sleep 30
    
    # Check service health
    check_health
    
    log "Deployment completed successfully!"
}

# Check service health
check_health() {
    log "Checking service health..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        info "Health check attempt $attempt/$max_attempts"
        
        # Check backend health
        if curl -f -s http://localhost:8000/health > /dev/null; then
            log "Backend service is healthy"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            error "Backend service failed to become healthy"
        fi
        
        sleep 10
        ((attempt++))
    done
    
    # Check frontend
    if curl -f -s http://localhost:3000 > /dev/null; then
        log "Frontend service is healthy"
    else
        warn "Frontend service may not be ready yet"
    fi
    
    log "Health checks completed"
}

# Setup SSL certificates (Let's Encrypt)
setup_ssl() {
    log "Setting up SSL certificates..."
    
    if [ ! -f "nginx/certs/fullchain.pem" ]; then
        warn "SSL certificates not found. You need to obtain SSL certificates."
        warn "For Let's Encrypt, you can use:"
        warn "  certbot certonly --webroot -w nginx/certs -d your-domain.com"
        warn "Then copy the certificates to nginx/certs/"
    else
        log "SSL certificates found"
    fi
}

# Cleanup old Docker resources
cleanup() {
    log "Cleaning up Docker resources..."
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (be careful!)
    # docker volume prune -f
    
    log "Cleanup completed"
}

# Show deployment status
show_status() {
    log "Deployment Status:"
    echo
    
    info "Running containers:"
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.production ps
    echo
    
    info "Service URLs:"
    echo "  Frontend: http://localhost:3000"
    echo "  Backend API: http://localhost:8000"
    echo "  API Docs: http://localhost:8000/docs"
    echo "  Grafana: http://localhost:3001"
    echo "  Prometheus: http://localhost:9090"
    echo
    
    info "Logs:"
    echo "  View all logs: docker-compose -f docker-compose.yml -f docker-compose.prod.yml logs -f"
    echo "  View backend logs: docker logs insightvision-backend -f"
    echo "  View frontend logs: docker logs insightvision-frontend -f"
    echo
}

# Install Ollama model
install_model() {
    local model_name="${1:-llama3.2:1b}"
    
    log "Installing Ollama model: $model_name"
    
    docker exec insightvision-ollama ollama pull "$model_name"
    
    log "Model $model_name installed successfully"
}

# Main deployment function
main() {
    log "Starting InsightVision Production Deployment"
    
    case "${1:-deploy}" in
        "deploy")
            pre_deployment_checks
            backup_data
            pull_images
            deploy
            setup_ssl
            cleanup
            show_status
            ;;
        "backup")
            backup_data
            ;;
        "health")
            check_health
            ;;
        "status")
            show_status
            ;;
        "install-model")
            install_model "$2"
            ;;
        "logs")
            docker-compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.production logs -f
            ;;
        "stop")
            log "Stopping all services..."
            docker-compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.production down
            ;;
        "restart")
            log "Restarting all services..."
            docker-compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.production restart
            ;;
        *)
            echo "Usage: $0 {deploy|backup|health|status|install-model|logs|stop|restart}"
            echo
            echo "Commands:"
            echo "  deploy       - Full production deployment"
            echo "  backup       - Backup existing data"
            echo "  health       - Check service health"
            echo "  status       - Show deployment status"
            echo "  install-model MODEL - Install Ollama model"
            echo "  logs         - Show service logs"
            echo "  stop         - Stop all services"
            echo "  restart      - Restart all services"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
