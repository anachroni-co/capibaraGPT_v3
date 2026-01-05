#!/bin/bash
# Frontend Installation Script for Capibara6
# Use this script to install and configure the frontend components on a new VM or local system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BLUE}################################################################${NC}"
    echo -e "${BLUE}## $1${NC}"
    echo -e "${BLUE}################################################################${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Variables
PROJECT_DIR="/opt/capibara6"
FRONTEND_DIR="/opt/capibara6/web"
USER="capibara6"
HTTP_PORT=${HTTP_PORT:-8000}

print_header "Installing Capibara6 Frontend Components"
print_info "Setting up frontend web interface for Capibara6"

# Function to install system dependencies
install_system_deps() {
    print_header "Installing System Dependencies"
    
    # Update package list
    sudo apt-get update

    # Install basic dependencies
    print_info "Installing basic system dependencies..."
    sudo apt-get install -y \
        curl \
        wget \
        git \
        python3 \
        python3-pip \
        python3-venv \
        nginx \
        nodejs \
        npm \
        net-tools \
        htop \
        vim

    print_success "System dependencies installed"
}

# Function to clone and setup capibara6
setup_capibara6() {
    print_header "Setting up Capibara6 Project"

    if [ -d "$PROJECT_DIR" ]; then
        print_warning "Project directory already exists, backing up..."
        sudo mv "$PROJECT_DIR" "${PROJECT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    fi

    # Create project directory
    sudo mkdir -p "$PROJECT_DIR"
    sudo chown -R "$USER:$USER" "$PROJECT_DIR"
    
    # Clone repository
    print_info "Cloning Capibara6 repository..."
    git clone https://github.com/anachroni-co/capibara6.git "$PROJECT_DIR"
    
    # Change ownership
    sudo chown -R "$USER:$USER" "$PROJECT_DIR"
    
    print_success "Capibara6 project cloned"
}

# Function to setup static file server
setup_static_server() {
    print_header "Setting up Static File Server"

    # Install Python dependencies for simple server
    sudo apt-get install -y python3-pip
    python3 -m pip install --user pipx
    python3 -m pipx ensurepath

    # Create a dedicated simple server script
    cat > "$FRONTEND_DIR/start_server.py" << 'EOF'
#!/usr/bin/env python3
import http.server
import socketserver
import os
import sys
from functools import partial

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

if __name__ == "__main__":
    port = int(os.environ.get("HTTP_PORT", 8000))
    directory = os.environ.get("SERVER_DIRECTORY", ".")
    
    # Change to the web directory
    os.chdir(directory)
    
    handler = partial(CORSRequestHandler, directory=directory)
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving at http://localhost:{port}")
        print(f"Serving directory: {directory}")
        httpd.serve_forever()
EOF

    chmod +x "$FRONTEND_DIR/start_server.py"
    
    print_success "Static file server script created"
}

# Function to configure frontend settings
configure_frontend() {
    print_header "Configuring Frontend Settings"

    cd "$FRONTEND_DIR"
    
    # Create/update config.js with default settings
    cat > config.js << 'EOF'
// Capibara6 Frontend Configuration
const CONFIG = {
    // Backend endpoints
    BACKEND_URL: 'http://localhost:5001',  // Adjust to your backend server
    BACKEND_MODELS_URL: 'http://34.12.166.76:5001',  // Bounty2 VM
    BACKEND_SERVICES_URL: 'http://34.175.136.104',   // Services VM
    
    // Service endpoints
    TTS_URL: 'http://34.175.136.104:5002',
    MCP_URL: 'http://34.175.136.104:5003',
    N8N_URL: 'http://34.175.136.104:5678',
    RAG_URL: 'http://10.154.0.2:8000',  // RAG3 VM
    
    // Model configuration
    DEFAULT_MODEL: 'gpt-oss-20b',
    TEMPERATURE: 0.7,
    MAX_TOKENS: 1000,
    
    // UI Settings
    ENABLE_RAG: true,
    ENABLE_TTS: true,
    ENABLE_MCP: true,
    ENABLE_CONSENSUS: true,
    
    // CORS proxy (for development)
    USE_CORS_PROXY: true,
    CORS_PROXY_URL: 'http://localhost:5001/proxy',  // Backend can act as proxy
    
    // WebSocket settings
    ENABLE_STREAMING: true,
    
    // Debug settings
    DEBUG: false,
    LOG_LEVEL: 'INFO'
};

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONFIG;
} else if (typeof window !== 'undefined') {
    window.CONFIG = CONFIG;
}
EOF

    print_success "Frontend configuration created"
}

# Function to setup Nginx (if needed)
setup_nginx() {
    print_header "Setting up Nginx (Optional)"

    read -p "Do you want to set up Nginx as a reverse proxy? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Configure Nginx
        sudo tee /etc/nginx/sites-available/capibara6-frontend > /dev/null <<EOF
server {
    listen 80;
    server_name _;

    root $FRONTEND_DIR;
    index index.html;

    # Set appropriate headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    
    location / {
        try_files $uri $uri/ =404;
    }

    # CORS headers for API calls
    location /api/ {
        proxy_pass http://localhost:5001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # CORS headers
        add_header Access-Control-Allow-Origin *;
        add_header Access-Control-Allow-Methods 'GET, POST, OPTIONS';
        add_header Access-Control-Allow-Headers *;
    }

    # Security headers
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

        # Enable the site
        sudo ln -sf /etc/nginx/sites-available/capibara6-frontend /etc/nginx/sites-enabled/
        sudo rm -f /etc/nginx/sites-enabled/default

        # Test and restart Nginx
        sudo nginx -t && sudo systemctl restart nginx
        print_success "Nginx configured as reverse proxy"
    else
        print_info "Skipping Nginx setup"
    fi
}

# Function to setup systemd service for file server
setup_systemd_service() {
    print_header "Setting up Systemd Service for Frontend"

    sudo tee /etc/systemd/system/capibara6-frontend.service > /dev/null <<EOF
[Unit]
Description=Capibara6 Frontend Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$FRONTEND_DIR
Environment=HTTP_PORT=$HTTP_PORT
Environment=SERVER_DIRECTORY=$FRONTEND_DIR
ExecStart=/usr/bin/python3 start_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Enable and start the service
    sudo systemctl daemon-reload
    sudo systemctl enable capibara6-frontend
    sudo systemctl start capibara6-frontend
    
    print_success "Frontend service configured and started"
}

# Function to run tests
run_tests() {
    print_header "Running Frontend Tests"

    cd "$FRONTEND_DIR"

    # Check if main files exist
    if [ -f "chat.html" ] && [ -f "index.html" ]; then
        print_success "Main HTML files found"
    else
        print_warning "Main HTML files not found, checking available files..."
        ls -la *.html
    fi

    # Check if the service is running
    if sudo systemctl is-active --quiet capibara6-frontend; then
        print_success "Frontend service is running"
        
        # Get the port from the service file (or default)
        ACTUAL_PORT=$(grep "HTTP_PORT" /etc/systemd/system/capibara6-frontend.service | grep -o "[0-9]*" | head -1)
        if [ -z "$ACTUAL_PORT" ]; then
            ACTUAL_PORT=8000
        fi
        
        print_info "Service accessible at: http://localhost:$ACTUAL_PORT"
    else
        print_error "Frontend service is not running"
    fi
}

# Function to display status
display_status() {
    print_header "Frontend System Status"

    print_info "Frontend location: $FRONTEND_DIR"
    print_info "User: $USER"
    print_info "Server port: $HTTP_PORT"
    
    print_info "Frontend service status:"
    sudo systemctl status capibara6-frontend --no-pager -l
    
    # Check if the server is responding
    if nc -z localhost $HTTP_PORT; then
        print_success "Frontend server is listening on port $HTTP_PORT"
    else
        print_warning "Frontend server may not be accessible yet"
    fi
}

# Function to show usage instructions
show_instructions() {
    print_header "Next Steps and Usage"

    echo "To manage the frontend service:"
    echo "  sudo systemctl start capibara6-frontend    # Start service"
    echo "  sudo systemctl stop capibara6-frontend     # Stop service"
    echo "  sudo systemctl restart capibara6-frontend  # Restart service"
    echo "  sudo systemctl status capibara6-frontend   # Check status"
    echo "  sudo journalctl -u capibara6-frontend -f   # View logs"
    echo ""
    echo "To access the frontend:"
    echo "  Chat interface: http://localhost:$HTTP_PORT/chat.html"
    echo "  Main page: http://localhost:$HTTP_PORT"
    echo ""
    echo "For development, you can run directly:"
    echo "  cd $FRONTEND_DIR"
    echo "  python3 -m http.server $HTTP_PORT"
    echo ""
    echo "Configuration files:"
    echo "  Frontend config: $FRONTEND_DIR/config.js"
    echo "  Environment: $FRONTEND_DIR/.env (create if needed)"
    echo ""
    echo "Note: Make sure backend services are accessible before using the frontend"
}

# Function to create a simple test endpoint
create_test_endpoint() {
    print_header "Creating Frontend Test Files"

    cd "$FRONTEND_DIR"

    # Create a simple test HTML file
    cat > test.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Capibara6 Frontend Test</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 50px auto; 
            padding: 20px; 
            background-color: #f5f5f5;
        }
        .container { 
            background: white; 
            padding: 30px; 
            border-radius: 10px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .status { 
            padding: 10px; 
            margin: 10px 0; 
            border-radius: 5px; 
        }
        .success { background-color: #d4edda; color: #155724; }
        .error { background-color: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🦫 Capibara6 Frontend Test</h1>
        <p>This page verifies that the Capibara6 frontend is properly serving files.</p>
        
        <div id="status-section">
            <h2>System Status</h2>
            <div id="status" class="status">Checking status...</div>
        </div>
        
        <div id="config-section">
            <h2>Configuration</h2>
            <div id="config"></div>
        </div>
        
        <div id="services-section">
            <h2>Backend Services</h2>
            <div id="services"></div>
        </div>
    </div>

    <script>
        // Test configuration loading
        async function testConfig() {
            try {
                // Try to load config.js
                const script = document.createElement('script');
                script.src = 'config.js';
                document.head.appendChild(script);
                
                script.onload = function() {
                    document.getElementById('config').innerHTML = '<div class="status success">Configuration loaded successfully</div>';
                    if (window.CONFIG) {
                        document.getElementById('config').innerHTML += '<p>Default model: ' + (window.CONFIG.DEFAULT_MODEL || 'Not set') + '</p>';
                    }
                };
                
                script.onerror = function() {
                    document.getElementById('config').innerHTML = '<div class="status error">Could not load config.js</div>';
                };
            } catch (e) {
                document.getElementById('config').innerHTML = '<div class="status error">Error loading config: ' + e.message + '</div>';
            }
        }

        // Test basic connectivity
        function testConnectivity() {
            document.getElementById('status').innerHTML = '<div class="status success">Frontend server is accessible</div>';
        }

        // Test services (asynchronously)
        async function testServices() {
            const servicesDiv = document.getElementById('services');
            servicesDiv.innerHTML = '<p>Testing backend services...</p>';
            
            // Test backend health (this may fail due to CORS without proper backend)
            try {
                const response = await fetch('http://localhost:5001/health', {
                    method: 'GET',
                    mode: 'no-cors'  // This may not work as expected
                });
                servicesDiv.innerHTML += '<div class="status success">Local backend health check initiated</div>';
            } catch (e) {
                servicesDiv.innerHTML += '<div class="status error">Could not reach local backend: ' + e.message + '</div>';
            }
        }

        // Run tests when page loads
        window.onload = function() {
            testConnectivity();
            testConfig();
            testServices();
        };
    </script>
</body>
</html>
EOF

    print_success "Test endpoint created at $FRONTEND_DIR/test.html"
}

# Main execution
main() {
    print_header "Starting Capibara6 Frontend Installation"

    install_system_deps
    setup_capibara6
    setup_static_server
    configure_frontend
    create_test_endpoint
    setup_nginx
    setup_systemd_service
    run_tests
    display_status
    show_instructions

    print_header "Frontend installation completed successfully!"
    print_success "Capibara6 frontend is now installed and configured."
    print_info "Review the status above and the next steps for usage instructions."
    print_info "The frontend test page is available at http://localhost:$HTTP_PORT/test.html"
}

# Run main function
main