// Chatbot capibara6 con GPT-OSS-20B
class Capibara6ChatGPTOSS {
    constructor() {
        this.toggle = document.getElementById('chatbot-toggle');
        this.window = document.getElementById('chatbot-window');
        this.close = document.getElementById('chatbot-close');
        this.input = document.getElementById('chatbot-input');
        this.send = document.getElementById('chatbot-send');
        this.messages = document.getElementById('chatbot-messages');
        this.isOpen = false;
        this.isLoading = false;
        
        // Configuración del backend
        this.backendUrl = this.getBackendUrl();
        
        // Datos del usuario
        this.userConversations = this.loadUserData();
        
        this.init();
    }
    
    init() {
        this.toggle.addEventListener('click', () => this.toggleChat());
        this.close.addEventListener('click', () => this.toggleChat());
        this.send.addEventListener('click', () => this.sendMessage());
        this.input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Mostrar mensaje de bienvenida con información del modelo
        this.addMessage('¡Hola! Soy capibara6, asistente de IA con modelo GPT-OSS-20B. ¿En qué puedo ayudarte?', 'bot');
    }
    
    getBackendUrl() {
        // Usar dominio correcto con SSL válido y www
        return 'http://localhost:5001';
    }
    
    toggleChat() {
        this.isOpen = !this.isOpen;
        this.window.classList.toggle('open', this.isOpen);
        if (this.isOpen) {
            this.input.focus();
        }
    }
    
    async sendMessage() {
        const message = this.input.value.trim();
        if (!message || this.isLoading) return;
        
        // Agregar mensaje del usuario
        this.addMessage(message, 'user');
        this.input.value = '';
        
        // Mostrar indicador de carga
        this.showLoading();
        
        try {
            // Llamar al backend con GPT-OSS-20B
            const response = await this.callGPTOSS(message);
            this.hideLoading();
            this.addMessage(response, 'bot');
            
            // Guardar conversación
            this.saveUserMessage(message, response);
            
        } catch (error) {
            this.hideLoading();
            console.error('Error:', error);
            this.addMessage('Lo siento, hubo un error al procesar tu mensaje. Por favor, inténtalo de nuevo.', 'bot');
        }
    }
    
    async callGPTOSS(message) {
        const response = await fetch(`${this.backendUrl}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                max_tokens: 500,
                temperature: 0.7
            })
        });
        
        if (!response.ok) {
            throw new Error(`Error ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        return data.response || 'No se recibió respuesta del modelo.';
    }
    
    async callGPTOSSStream(message) {
        const response = await fetch(`${this.backendUrl}/api/chat/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                max_tokens: 500,
                temperature: 0.7
            })
        });
        
        if (!response.ok) {
            throw new Error(`Error ${response.status}: ${response.statusText}`);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';
        
        // Crear mensaje de bot vacío para streaming
        const messageDiv = this.createMessageElement('', 'bot');
        const contentDiv = messageDiv.querySelector('.message-content p');
        this.messages.appendChild(messageDiv);
        
        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.content) {
                                fullResponse += data.content;
                                contentDiv.textContent = fullResponse;
                                this.messages.scrollTop = this.messages.scrollHeight;
                            }
                            if (data.done) {
                                return fullResponse;
                            }
                        } catch (e) {
                            // Ignorar líneas malformadas
                        }
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }
        
        return fullResponse;
    }
    
    addMessage(text, type) {
        const messageDiv = this.createMessageElement(text, type);
        this.messages.appendChild(messageDiv);
        this.messages.scrollTop = this.messages.scrollHeight;
    }
    
    createMessageElement(text, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${type}-message`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = type === 'bot' 
            ? '<i data-lucide="bot" style="width: 20px; height: 20px;"></i>' 
            : '<i data-lucide="user" style="width: 20px; height: 20px;"></i>';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        content.innerHTML = `<p>${text}</p>`;
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        
        // Inicializar iconos de Lucide
        if (typeof lucide !== 'undefined') {
            setTimeout(() => lucide.createIcons(), 0);
        }
        
        return messageDiv;
    }
    
    showLoading() {
        this.isLoading = true;
        this.send.disabled = true;
        this.input.disabled = true;
        
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'chat-message bot-message loading-message';
        loadingDiv.innerHTML = `
            <div class="message-avatar">
                <i data-lucide="bot" style="width: 20px; height: 20px;"></i>
            </div>
            <div class="message-content">
                <p>capibara6 está pensando...</p>
                <div class="loading-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        
        this.messages.appendChild(loadingDiv);
        this.messages.scrollTop = this.messages.scrollHeight;
        
        // Inicializar iconos
        if (typeof lucide !== 'undefined') {
            setTimeout(() => lucide.createIcons(), 0);
        }
    }
    
    hideLoading() {
        this.isLoading = false;
        this.send.disabled = false;
        this.input.disabled = false;
        
        const loadingMessage = this.messages.querySelector('.loading-message');
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }
    
    // Métodos para gestión de datos del usuario
    loadUserData() {
        const data = localStorage.getItem('capibara6_user_data');
        return data ? JSON.parse(data) : {
            conversations: [],
            timestamp: new Date().toISOString()
        };
    }
    
    saveUserMessage(userMessage, aiResponse) {
        const timestamp = new Date().toISOString();
        
        // Guardar la conversación
        this.userConversations.conversations.push({
            user_message: userMessage,
            ai_response: aiResponse,
            timestamp: timestamp
        });
        
        // Guardar en localStorage
        localStorage.setItem('capibara6_user_data', JSON.stringify(this.userConversations));
        
        // Enviar al backend si hay muchas conversaciones
        if (this.userConversations.conversations.length >= 5) {
            this.sendToBackend();
        }
    }
    
    async sendToBackend() {
        try {
            const response = await fetch(`${this.backendUrl}/api/save-conversation`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    conversations: this.userConversations.conversations,
                    email: 'anonymous@capibara6.com' // Usuario anónimo
                })
            });
            
            const result = await response.json();
            console.log('Conversaciones guardadas:', result);
            
        } catch (error) {
            console.warn('Error guardando conversaciones:', error);
        }
    }
}

// Inicializar chatbot
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new Capibara6ChatGPTOSS();
    });
} else {
    new Capibara6ChatGPTOSS();
}
