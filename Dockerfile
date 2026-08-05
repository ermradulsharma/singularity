# 🚀 The Independent AGI Production Image
FROM python:3.11-slim

# Set secure working directory
WORKDIR /app

# Copy source code (without transferring host OS secrets)
COPY src/ /app/src/
COPY main.py daemon.py /app/

# Install strictly necessary dependencies. No network exploitation tools.
RUN pip install --no-cache-dir torch tiktoken scipy sympy

# 🔒 EXTREME SECURITY: Run as a non-root user
# Even if the AGI evolves to break the python sandbox, it cannot gain root access.
RUN useradd -m agi_user
USER agi_user

# Default command: Run the autonomous background daemon
CMD ["python", "daemon.py"]
