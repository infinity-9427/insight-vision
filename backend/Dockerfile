# Dockerfile - InsightVision Backend
FROM python:3.12.11-alpine3.22

# Install system dependencies for data processing and PDF handling
RUN apk update && apk add --no-cache \
    curl \
    gcc \
    g++ \
    make \
    cmake \
    libffi-dev \
    openssl-dev \
    libxml2-dev \
    libxslt-dev \
    jpeg-dev \
    libpng-dev \
    freetype-dev \
    pkgconfig \
    musl-dev \
    zlib-dev \
    linux-headers \
    python3-dev

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install dependencies with PyMuPDF wheel strategy
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    # First try to install from pre-built wheel
    pip install --no-cache-dir --only-binary=all PyMuPDF==1.23.8 || \
    # If wheel fails, try latest version which may have Alpine fixes
    pip install --no-cache-dir --only-binary=all PyMuPDF || \
    # If all wheel attempts fail, compile 1.23.8 with specific flags
    ( \
        export CFLAGS="-I/usr/include -I/usr/local/include" && \
        export LDFLAGS="-L/usr/lib -L/usr/local/lib" && \
        export CC=gcc && \
        export CXX=g++ && \
        export PYMUPDF_SETUP_MUPDF_BUILD_TYPE="shared" && \
        export PYMUPDF_INCLUDES="" && \
        pip install --no-cache-dir --no-build-isolation --verbose PyMuPDF==1.23.8 \
    ) && \
    # Install remaining requirements
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY *.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create non-root user for security
RUN adduser -D -u 1000 appuser

# Create necessary directories and set proper ownership
RUN mkdir -p uploads charts data logs && \
    chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
