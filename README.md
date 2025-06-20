# 📚 StoryForge AI - Professional Story Generator

A sophisticated AI-powered story generation application built with Streamlit, featuring advanced language models and professional-grade user interface.

## ✨ Features

### 🎯 Core Functionality
- **AI Story Generation**: Powered by fine-tuned GPT-Neo models
- **Story Refinement**: Enhanced with Ollama/Mistral for professional editing
- **Multiple Genres**: Fantasy, Sci-Fi, Mystery, Horror, Adventure, Romance, Thriller, Historical
- **Customizable Parameters**: Adjustable creativity, focus, and length settings
- **Professional UI**: Modern, responsive design with intuitive controls

### 🛠️ Technical Features
- **Docker Deployment**: Containerized for easy deployment
- **Error Handling**: Comprehensive error handling and logging
- **Memory Management**: Optimized for various hardware configurations
- **Download Functionality**: Export stories in multiple formats
- **Session Management**: Track generation history and statistics
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 4GB+ RAM (8GB+ recommended)
- GPU support (optional but recommended)

### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/storyforge-ai.git
cd storyforge-ai
chmod +x deploy.sh
```

### 2. Configuration
```bash
# Copy and edit environment variables
cp .env.example .env
nano .env  # Edit with your settings
```

### 3. Deploy
```bash
# Automated deployment
./deploy.sh

# Or manual deployment
docker-compose up -d
```

### 4. Access
- **Web App**: http://localhost:8501
- **Ollama API**: http://localhost:11434

## 📁 Project Structure

```
storyforge-ai/
├── streamlit_app.py          # Main application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
├── docker-compose.yml       # Multi-service setup
├── deploy.sh               # Deployment script
├── .env.example            # Environment template
├── config.toml             # Streamlit configuration
├── gptneo-story-model/     # Fine-tuned model files
├── stories/                # Generated stories
└── logs/                   # Application logs
```

## ⚙️ Configuration

### Environment Variables
```bash
# Model configuration
MODEL_PATH=gptneo-story-model
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=mistral

# Optional: Hugging Face authentication
HUGGINGFACE_TOKEN=your_token_here
```

### Streamlit Configuration
```toml
[server]
runOnSave = true
port = 8501
enableCORS = false

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
```

## 🔧 Advanced Setup

### Custom Model Integration
1. **Place your fine-tuned model** in `gptneo-story-model/`
2. **Update MODEL_PATH** in `.env`
3. **Ensure compatibility** with Transformers library

### GPU Support
```yaml
# Add to docker-compose.yml
services:
  storyforge-ai:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Scaling Configuration
```yaml
# For high-traffic deployment
services:
  storyforge-ai:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

## 📊 Monitoring & Maintenance

### Health Checks
```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f storyforge-ai

# Monitor resource usage
docker stats
```

### Performance Optimization
1. **Memory Management**: Adjust `max_length` parameters
2. **GPU Utilization**: Monitor CUDA memory usage
3. **Cache Settings**: Configure Streamlit caching
4. **Load Balancing**: Use reverse proxy for production

## 🐳 Deployment Options

### Local Development
```bash
# Direct Python execution
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Production Deployment
```bash
# With reverse proxy (Nginx)
docker-compose -f docker-compose.pro