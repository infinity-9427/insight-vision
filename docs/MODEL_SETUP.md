# 🤖 Ollama Model Auto-Download

This section explains how the InsightVision AI Platform automatically downloads and manages Ollama models when you run Docker Compose.

## 🚀 How It Works

When you run `docker-compose up`, the system automatically:

1. **Starts Ollama service** - The core Ollama server starts and becomes healthy
2. **Downloads AI models** - A dedicated service downloads the required models
3. **Verifies models** - Ensures models are properly installed and available
4. **Starts application** - The main app starts only after models are ready

## 📋 Model Configuration

### Primary & Backup Models

The system uses a **primary + backup** strategy for reliability:

```bash
# Primary model (configured in .env)
MODEL_NAME=llama3.2:3b-instruct-q8_0    # ~3GB VRAM

# Backup model (if primary fails)
BACKUP_MODEL=llama3.2:1b                # ~1GB VRAM
```

### Available Model Options

Choose based on your system capabilities:

| Model | Size | RAM Needed | Use Case |
|-------|------|------------|----------|
| `llama3.2:1b` | ~1GB | 2GB+ | Development/Testing |
| `llama3.2:3b-instruct-q8_0` | ~3GB | 6GB+ | Balanced Performance |
| `llama3.1:8b-instruct-q8_0` | ~8GB | 16GB+ | Production Quality |
| `llama3.1:70b-instruct-q4_K_M` | ~40GB | 64GB+ | Maximum Quality |

## 🔧 Configuration

### Quick Setup

1. **Edit `.env` file**:
   ```bash
   # Choose your primary model
   MODEL_NAME=llama3.2:3b-instruct-q8_0
   
   # Set backup model
   BACKUP_MODEL=llama3.2:1b
   ```

2. **Start the platform**:
   ```bash
   docker-compose up
   ```

3. **Watch the magic happen** ✨:
   - Ollama starts
   - Models download automatically
   - App starts when ready

### Advanced Configuration

For custom setups, modify `docker-compose.yml`:

```yaml
ollama-models:
  environment:
    - MODEL_NAME=your-preferred-model
    - BACKUP_MODEL=your-backup-model
```

## 📊 Model Download Process

The initialization follows this flow:

```
🚀 Start Ollama Service
     ↓
⏳ Wait for Ollama Ready
     ↓
📥 Download Primary Model
     ↓ (if fails)
📥 Download Backup Model
     ↓
✅ Verify Model Available
     ↓
🎯 Start Application
```

## 🔍 Monitoring

### Check Model Status

```bash
# See downloaded models
docker exec insightvision-ollama ollama list

# Check model initialization logs
docker logs insightvision-ollama-models
```

### Model Health Check

The system includes automatic health checks:

- **Ollama service**: Checked every 30 seconds
- **Model availability**: Verified during startup
- **Fallback handling**: Automatic backup model usage

## 🛠️ Troubleshooting

### Common Issues

1. **Slow model download**:
   - First run takes longer (downloading models)
   - Use smaller models for testing
   - Check internet connection

2. **Insufficient memory**:
   ```bash
   # Use smaller model
   MODEL_NAME=llama3.2:1b
   ```

3. **Model download fails**:
   - Check Docker logs: `docker logs insightvision-ollama-models`
   - Try backup model: Already configured automatically
   - Check Ollama service: `docker logs insightvision-ollama`

### Manual Model Management

```bash
# Enter Ollama container
docker exec -it insightvision-ollama bash

# List available models
ollama list

# Pull specific model
ollama pull llama3.2:3b-instruct-q8_0

# Remove unused models
ollama rm old-model-name
```

## 🎯 Performance Tips

1. **Development**: Use `llama3.2:1b` for fast iteration
2. **Production**: Use `llama3.2:3b-instruct-q8_0` or larger
3. **Memory-constrained**: Stick with 1B models
4. **High-quality analysis**: Use 8B+ models

## 🔄 Updating Models

To update to newer models:

1. **Update `.env`**:
   ```bash
   MODEL_NAME=llama3.3:1b  # New version
   ```

2. **Restart services**:
   ```bash
   docker-compose down
   docker-compose up
   ```

The system will automatically download the new model while keeping the old one as backup until verified.

---

## 🎉 Ready to Go!

With this setup, you never need to manually download Ollama models again. Just run `docker-compose up` and the platform handles everything automatically! 🚀
