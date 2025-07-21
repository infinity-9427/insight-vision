#!/bin/bash

# =============================================================================
# 🚀 Ollama Model Initialization Script
# =============================================================================
# This script automatically downloads the required Ollama models for 
# InsightVision AI Platform when Docker Compose starts up.

set -e  # Exit on any error

# Configuration from environment variables
PRIMARY_MODEL="${MODEL_NAME:-llama3.2:3b-instruct-q8_0}"
BACKUP_MODEL="${BACKUP_MODEL:-llama3.2:1b}"
OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
MAX_RETRIES=3
RETRY_DELAY=10

echo "🚀 InsightVision AI - Model Initialization Started"
echo "======================================================="
echo "📋 Primary Model: $PRIMARY_MODEL"
echo "🔄 Backup Model:  $BACKUP_MODEL"
echo "🌐 Ollama Host:   $OLLAMA_HOST"
echo ""

# Function to check if Ollama is ready
wait_for_ollama() {
    echo "⏳ Waiting for Ollama service to be ready..."
    local retries=0
    while [ $retries -lt $MAX_RETRIES ]; do
        if ollama list >/dev/null 2>&1; then
            echo "✅ Ollama service is ready!"
            return 0
        fi
        retries=$((retries + 1))
        echo "   Attempt $retries/$MAX_RETRIES failed, retrying in ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
    done
    echo "❌ Failed to connect to Ollama after $MAX_RETRIES attempts"
    exit 1
}

# Function to pull a model
pull_model() {
    local model=$1
    local model_type=$2
    
    echo "📥 Downloading $model_type model: $model"
    echo "   This may take a while depending on model size and internet speed..."
    
    if ollama pull "$model"; then
        echo "✅ Successfully downloaded $model_type model: $model"
        return 0
    else
        echo "❌ Failed to download $model_type model: $model"
        return 1
    fi
}

# Function to verify model is available
verify_model() {
    local model=$1
    echo "🔍 Verifying model availability: $model"
    
    if ollama list | grep -q "$model"; then
        echo "✅ Model verified: $model"
        return 0
    else
        echo "❌ Model not found: $model"
        return 1
    fi
}

# Main execution
main() {
    # Wait for Ollama to be ready
    wait_for_ollama
    
    echo ""
    echo "📦 Starting model downloads..."
    echo "----------------------------------------"
    
    # Try to pull primary model
    if pull_model "$PRIMARY_MODEL" "primary"; then
        if verify_model "$PRIMARY_MODEL"; then
            echo ""
            echo "🎉 Primary model setup complete!"
            
            # Optionally pull backup model for redundancy
            echo ""
            echo "🔄 Setting up backup model for redundancy..."
            if pull_model "$BACKUP_MODEL" "backup"; then
                verify_model "$BACKUP_MODEL"
            else
                echo "⚠️  Backup model failed to download, but primary model is available"
            fi
        else
            echo "❌ Primary model verification failed"
            exit 1
        fi
    else
        echo ""
        echo "⚠️  Primary model failed, trying backup model..."
        
        if pull_model "$BACKUP_MODEL" "backup"; then
            if verify_model "$BACKUP_MODEL"; then
                echo ""
                echo "✅ Backup model setup complete! (Primary model unavailable)"
            else
                echo "❌ Backup model verification failed"
                exit 1
            fi
        else
            echo "❌ Both primary and backup models failed to download"
            exit 1
        fi
    fi
    
    echo ""
    echo "📋 Final model inventory:"
    echo "------------------------"
    ollama list
    
    echo ""
    echo "🎯 Model initialization complete!"
    echo "🚀 InsightVision AI is ready to provide intelligent insights!"
    echo ""
}

# Execute main function
main
