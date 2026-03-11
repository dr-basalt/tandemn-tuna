FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    unzip \
    gnupg \
    lsb-release \
    apt-transport-https \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip

# Install Google Cloud SDK (official method)
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y \
        google-cloud-cli \
        google-cloud-cli-gke-gcloud-auth-plugin \
    && rm -rf /var/lib/apt/lists/*

# Ensure gcloud is in PATH for all users
ENV PATH="/usr/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies with GCP support
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -e .[all] && \
    pip install --no-cache-dir \
        google-cloud-storage \
        google-cloud-compute \
        google-auth \
        google-auth-oauthlib \
        google-auth-httplib2

# Copy the application
COPY . .

# Create a non-root user
RUN useradd --create-home --shell /bin/bash tuna
RUN chown -R tuna:tuna /app
USER tuna

# Expose the default port for the router
EXPOSE 8080

# Default command - can be overridden
CMD ["tuna", "--help"]