# Frontend - CapibaraGPT Web Application

Web frontend application to interact with the CapibaraGPT system.

## Features

- **Interactive chat** with AI models
- **RAG search** (Milvus + Nebula Graph)
- **TTS** (Text-to-Speech) integration
- **OAuth** (GitHub, Google)
- **Response evaluation** system
- **Prompt templates**
- **Entropy monitoring**
- **Dark/Light mode**

## Quick Start

### Local Development

```bash
# Serve static files
cd public
python3 -m http.server 8080

# Open in browser
# http://localhost:8080/chat.html
```

### With HTTP Server

```bash
# Use any HTTP server
# Nginx, Apache, etc.

# Example with nginx:
cp deployment/nginx.conf /etc/nginx/sites-available/capibara
ln -s /etc/nginx/sites-available/capibara /etc/nginx/sites-enabled/
systemctl reload nginx
```

## Structure

```
frontend/
├── public/               # Static files (HTML)
│   ├── index.html        # Landing page
│   ├── chat.html         # Chat application
│   ├── login.html        # Login page
│   └── assets/           # Images, fonts, etc.
├── src/                  # JavaScript source code
│   ├── config.js         # Main configuration
│   ├── chat-app.js       # Main chat application
│   ├── script.js         # Global scripts
│   ├── clients/          # API clients
│   │   ├── milvus-client.js    # Milvus client
│   │   ├── nebula-client.js    # Nebula Graph client
│   │   ├── rag-client.js       # Unified RAG client
│   │   └── api-client.js       # Generic API client
│   ├── components/       # UI components
│   │   ├── chatbot.js          # Chatbot component
│   │   ├── rating-system.js    # Evaluation system
│   │   ├── template-profiles.js # Templates
│   │   └── neural-animation.js # Animations
│   ├── integrations/     # Service integrations
│   │   ├── smart-mcp-integration.js  # MCP v2.0
│   │   ├── consensus-integration.js  # Consensus
│   │   ├── entropy-monitor.js        # Entropy monitor
│   │   └── tts-integration.js        # Text-to-Speech
│   └── utils/            # Utilities
│       ├── translations.js
│       └── helpers.js
├── styles/               # CSS
│   ├── main.css
│   └── chat.css
├── deployment/           # Deploy configs
│   ├── nginx.conf        # Nginx configuration
│   ├── Dockerfile        # Production Dockerfile
│   └── deploy.sh         # Deploy script
└── tests/                # Frontend tests
```

## Configuration

### config.js

Main configuration file at `src/config.js`:

```javascript
// Detects if you're on localhost or production
const isLocalhost = window.location.hostname === 'localhost';

const CHATBOT_CONFIG = {
    // Main backend (VM bounty2)
    BACKEND_URL: isLocalhost
        ? 'http://localhost:5001'
        : 'http://34.12.166.76:5001',

    // Services
    SERVICES: {
        MCP: {
            enabled: false,
            url: isLocalhost
                ? 'http://localhost:5003'
                : 'http://34.175.136.104:5003'
        },
        TTS: {
            enabled: true,
            url: isLocalhost
                ? 'http://localhost:5002'
                : 'http://34.175.136.104:5002'
        },
        RAG3_BRIDGE: {
            enabled: true,
            url: isLocalhost
                ? 'http://localhost:8000'
                : 'http://10.154.0.2:8000'
        }
    }
};
```

### Environment Variables

For production, configure in `deployment/.env`:

```bash
# VM URLs
VM_MODELS_URL=http://34.12.166.76
VM_SERVICES_URL=http://34.175.136.104
VM_RAG_URL=http://10.154.0.2

# OAuth
GITHUB_CLIENT_ID=your_id
GOOGLE_CLIENT_ID=your_id

# Features
ENABLE_TTS=true
ENABLE_MCP=false
ENABLE_RAG=true
```

## Main Components

### Chat App

Main chat application (`src/chat-app.js`):

```javascript
// Send message
async function sendMessage(message) {
    const response = await fetch(
        `${CHATBOT_CONFIG.BACKEND_URL}/api/v1/query`,
        {
            method: 'POST',
            headers: CHATBOT_CONFIG.HEADERS,
            body: JSON.stringify({ message })
        }
    );
    const data = await response.json();
    return data.response;
}
```

### RAG Client

Unified RAG client (`src/clients/rag-client.js`):

```javascript
const ragClient = new RAGClient({
    hybridWeight: 0.7,      // 70% vector, 30% graph
    enrichContext: true,    // Enrich with graph
    useTOON: true          // TOON optimization
});

// Hybrid search
const results = await ragClient.search("What is Python?");
console.log(results.context);  // Formatted context
console.log(results.stats);    // Statistics (tokens saved, etc.)
```

### Milvus Client

Client for vector search (`src/clients/milvus-client.js`):

```javascript
const milvusClient = new MilvusClient();

// Search by text
const results = await milvusClient.searchByText("machine learning", {
    top_k: 10
});

// Hybrid search with filters
const filteredResults = await milvusClient.hybridSearch(
    "deep learning",
    { category: "AI", timestamp: { $gte: "2025-01-01" } },
    { top_k: 5 }
);
```

### TTS Integration

Text-to-Speech (`src/integrations/tts-integration.js`):

```javascript
// Synthesize text to speech
const audio = await synthesizeText("Hello, I'm CapibaraGPT", {
    voice: 'default',
    speed: 1.0
});

// Play
audio.play();
```

## Advanced Features

### Evaluation System

Allows evaluating LLM responses:

```javascript
// Evaluate response
ratingSystem.rate(messageId, {
    accuracy: 5,
    relevance: 4,
    helpfulness: 5,
    comment: "Excellent response"
});
```

### Prompt Templates

Predefined templates for common queries:

```javascript
// Use template
const prompt = templateProfiles.apply("code_review", {
    language: "Python",
    code: "def hello(): print('hi')"
});
```

### Entropy Monitoring

Detects response degradation:

```javascript
// Calculate entropy
const entropy = entropyMonitor.calculate(response);

if (entropy < threshold) {
    console.log("Warning: Low entropy detected");
}
```

## Customization

### Styles

Modify `styles/chat.css`:

```css
/* Dark theme */
.dark-theme {
    --bg-color: #1a1a1a;
    --text-color: #ffffff;
    --accent-color: #00d4aa;
}

/* Light theme */
.light-theme {
    --bg-color: #ffffff;
    --text-color: #000000;
    --accent-color: #0066cc;
}
```

### Logo and Branding

Replace files in `public/assets/`:
- `logo.png` - Main logo
- `favicon.ico` - Site icon
- `banner.jpg` - Landing page banner

## Deployment

### Nginx

```bash
# Copy configuration
sudo cp deployment/nginx.conf /etc/nginx/sites-available/capibara

# Enable site
sudo ln -s /etc/nginx/sites-available/capibara /etc/nginx/sites-enabled/

# Copy files
sudo cp -r public/* /var/www/capibara/
sudo cp -r src /var/www/capibara/
sudo cp -r styles /var/www/capibara/

# Reload nginx
sudo systemctl reload nginx
```

### Docker

```bash
# Build image
docker build -f deployment/Dockerfile -t capibara-frontend .

# Run container
docker run -p 80:80 capibara-frontend

# With docker-compose
cd deployment
docker-compose up -d
```

## Debugging

### DevTools Console

Verify configuration:

```javascript
// See loaded configuration
console.log(CHATBOT_CONFIG);

// See enabled services
console.log(CHATBOT_CONFIG.SERVICES);

// Test backend connection
fetch(`${CHATBOT_CONFIG.BACKEND_URL}/health`)
    .then(r => r.json())
    .then(console.log);
```

### Network Tab

Verify requests:
- Backend should be port 5001
- TTS should be port 5002
- MCP should be port 5003
- RAG Bridge should be port 8000

## Tests

```bash
# Unit tests
npm run test

# E2E tests
npm run test:e2e

# Linting
npm run lint

# Build
npm run build
```

---

**Maintainer**: CapibaraGPT Team
**Version**: 3.0
