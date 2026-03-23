# Dockerfile
# ------------------------------------------------------------
# Production container for Sage.
#
# Single container runs both the Streamlit UI and the
# APScheduler daily pipeline on the same process.
#
# In production at scale: split into two containers —
# one for the UI (stateless, horizontally scalable)
# one for the scheduler (single instance, stateful)
# Use Railway's multi-service deployment for that.
# ------------------------------------------------------------

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# libpq-dev needed for psycopg2 (Supabase Postgres driver)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker layer caching
# If requirements don't change, this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
# Never run production containers as root
RUN useradd -m -u 1000 sage && chown -R sage:sage /app
USER sage

# Streamlit config
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose port
EXPOSE 8501

# Health check — Railway uses this to verify the container
# is running correctly before routing traffic to it
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start command
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]