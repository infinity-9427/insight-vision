# InsightVision AI Analytics Platform

A modern AI-powered data analytics platform with PDF processing capabilities, built with Next.js frontend, FastAPI backend, and Ollama LLM integration.

## 🚀 Purpose

InsightVision provides intelligent data analysis through:
- **PDF Document Processing** with PyMuPDF text extraction
- **AI-Powered Insights** using Ollama LLM (llama3.1:70b-instruct)
- **Interactive Dashboard** with data visualization
- **Multi-format Support** (CSV, PDF, Excel, JSON)

## 📁 Project Structure

```
├── backend/           # FastAPI backend with PyMuPDF integration
│   ├── app.py        # Main API application
│   ├── Dockerfile    # Backend container
│   ├── requirements.txt
│   ├── uploads/      # File upload directory
│   └── data/         # Data processing directory
├── frontend/         # Next.js dashboard application
├── docker-compose.yml # Multi-service orchestration
└── README.md         # This file
```

## 🛠 Quick Start

### Development Environment

#### Clone and Setup
```bash
git clone <repository-url>
cd insightvision-ai-platform

# Copy environment file and configure
cp .env .env.local
# Edit .env.local with your settings
```

#### Run Development Server
```bash
# Start all services (development mode)
docker-compose up --build -d

# 🎉 That's it! Models download automatically!
# No need to manually install Ollama models

# View logs
docker-compose logs -f

# Check model download progress
docker logs insightvision-ollama-models
```

> **🚀 NEW: Automatic Model Setup!** 
> Models are now downloaded automatically when you run `docker-compose up`. 
> No manual installation required! See [Model Setup Guide](docs/MODEL_SETUP.md) for details.

### Production Deployment

#### Prerequisites
- Docker and Docker Compose installed
- At least 16GB RAM (32GB+ recommended for larger models)
- 50GB+ free disk space
- Domain name configured (for SSL)

#### Quick Production Deployment
```bash
# Use the automated deployment script
./deploy.sh deploy

# Or manually:
cp .env.production .env.prod.local
# Edit .env.prod.local with your production settings

docker-compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.prod.local up -d
```

#### Production Features
- **Nginx reverse proxy** with SSL termination
- **PostgreSQL database** for data persistence
- **Redis** for session management and caching
- **Monitoring** with Prometheus and Grafana
- **Log aggregation** with Loki
- **Automated backups** and health checks
- **Rate limiting** and security headers
- **Resource limits** and auto-restart policies

### Access the Application

#### Development
- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Ollama LLM**: http://localhost:11434

#### Production
- **Frontend**: https://yourdomain.com
- **API**: https://yourdomain.com/api
- **Monitoring Dashboard**: https://yourdomain.com:3001 (Grafana)
- **Metrics**: https://yourdomain.com:9090 (Prometheus)

## 🧪 Test Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Upload File
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-document.pdf"
```

### Get System Info
```bash
curl http://localhost:8000/system-info
```

### Chat with AI
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Analyze the uploaded data"}'
```

## 🔧 Development

### Environment Setup
1. Ensure Docker/Podman is installed
2. Clone the repository
3. Copy `.env` to `.env.local` and configure
4. Run `docker-compose up --build`

### Production Deployment
1. Copy `.env.production` to `.env.prod.local`
2. Configure production settings (database, SSL, etc.)
3. Run `./deploy.sh deploy` for automated deployment
4. Set up SSL certificates (Let's Encrypt recommended)
5. Configure DNS to point to your server

### Architecture
- **Backend**: FastAPI with PyMuPDF 1.26.3, Ollama integration
- **Frontend**: Next.js 15 with TypeScript, Tailwind CSS
- **Database**: PostgreSQL (production) / SQLite (development)
- **LLM**: Ollama serving llama3.1:70b-instruct-q4_K_M (production)
- **Proxy**: Nginx with SSL termination and rate limiting
- **Monitoring**: Prometheus + Grafana + Loki stack
- **Caching**: Redis for sessions and API caching

### Security Features
- **Rate limiting** on API endpoints
- **CORS protection** with configurable origins
- **Security headers** (CSP, XSS protection, etc.)
- **JWT authentication** ready (optional)
- **File upload validation** and size limits
- **Docker security** with non-root users
- **SSL/TLS encryption** in production

### Features
- ✅ PDF text extraction and processing
- ✅ Multi-format file upload support
- ✅ AI-powered data insights
- ✅ Interactive charts and visualizations
- ✅ Real-time data analysis
- ✅ Responsive web interface

## 📊 Tech Stack

**Backend:**
- FastAPI (Python 3.12)
- PyMuPDF 1.26.3
- Ollama LLM Integration
- Docker containerization

**Frontend:**
- Next.js 15
- TypeScript
- Tailwind CSS
- React Components

**Infrastructure:**
- Docker Compose
- Podman/Docker
- Multi-service architecture

---

Built with ❤️ for intelligent data analysis and document processing.
