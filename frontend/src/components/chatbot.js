// Chatbot capibara6
class Capibara6Chat {
    constructor() {
        this.toggle = document.getElementById('chatbot-toggle');
        this.window = document.getElementById('chatbot-window');
        this.close = document.getElementById('chatbot-close');
        this.input = document.getElementById('chatbot-input');
        this.send = document.getElementById('chatbot-send');
        this.messages = document.getElementById('chatbot-messages');
        this.isOpen = false;
        
        // Estado de conversación
        this.conversationState = null;
        this.awaitingCongressResponse = false;
        this.awaitingExperienceResponse = false;
        this.hasAskedAboutCongress = localStorage.getItem('hasAskedAboutCongress') === 'true';
        
        // Datos del usuario
        this.userConversations = this.loadUserData();
        
        this.responses = this.getResponses();
        
        this.init();
    }
    
    init() {
        this.toggle.addEventListener('click', () => this.toggleChat());
        this.close.addEventListener('click', () => this.toggleChat());
        this.send.addEventListener('click', () => this.sendMessage());
        this.input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
    }
    
    toggleChat() {
        this.isOpen = !this.isOpen;
        this.window.classList.toggle('open', this.isOpen);
        if (this.isOpen) {
            this.input.focus();
        }
    }
    
    sendMessage() {
        const message = this.input.value.trim();
        if (!message) return;
        
        this.addMessage(message, 'user');
        this.input.value = '';
        
        // Guardar mensaje del usuario
        this.saveUserMessage(message);
        
        setTimeout(() => {
            const response = this.getResponse(message);
            this.addMessage(response.text, 'bot', response.quickReplies);
        }, 600);
    }
    
    addMessage(text, type, quickReplies = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${type}-message`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = type === 'bot' ? '<i data-lucide="bot" style="width: 20px; height: 20px;"></i>' : '<i data-lucide="user" style="width: 20px; height: 20px;"></i>';
        
        // Inicializar el icono de Lucide
        if (typeof lucide !== 'undefined') {
            setTimeout(() => lucide.createIcons(), 0);
        }
        
        const content = document.createElement('div');
        content.className = 'message-content';
        content.innerHTML = `<p>${text}</p>`;
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        this.messages.appendChild(messageDiv);
        
        // Agregar botones de respuesta rápida si existen
        if (quickReplies && quickReplies.length > 0 && type === 'bot') {
            this.addQuickReplies(quickReplies);
        }
        
        this.messages.scrollTop = this.messages.scrollHeight;
    }
    
    addQuickReplies(replies) {
        // Eliminar botones de respuesta rápida anteriores si existen
        const existingReplies = this.messages.querySelector('.quick-replies');
        if (existingReplies) {
            existingReplies.remove();
        }
        
        const quickRepliesDiv = document.createElement('div');
        quickRepliesDiv.className = 'quick-replies';
        
        replies.forEach(reply => {
            const button = document.createElement('button');
            button.className = 'quick-reply-btn';
            button.textContent = reply.text;
            button.onclick = () => {
                this.handleQuickReply(reply.value);
                quickRepliesDiv.remove();
            };
            quickRepliesDiv.appendChild(button);
        });
        
        this.messages.appendChild(quickRepliesDiv);
        this.messages.scrollTop = this.messages.scrollHeight;
    }
    
    handleQuickReply(value) {
        // Simular que el usuario escribió la respuesta
        this.addMessage(value, 'user');
        
        setTimeout(() => {
            const response = this.getResponse(value);
            this.addMessage(response.text, 'bot', response.quickReplies);
        }, 600);
    }
    
    getResponse(message) {
        const lang = document.documentElement.getAttribute('data-lang') || 'es';
        const responses = this.responses[lang];
        const lowerMessage = message.toLowerCase();
        
        // Manejar flujo de conversación sobre el congreso
        if (this.awaitingCongressResponse) {
            this.awaitingCongressResponse = false;
            
            // Verificar si dice que sí está en el congreso
            if (lowerMessage.includes('sí') || lowerMessage.includes('si') || 
                lowerMessage.includes('yes') || lowerMessage.includes('claro') || 
                lowerMessage.includes('por supuesto')) {
                this.awaitingExperienceResponse = true;
                return {
                    text: lang === 'es' ? '¡Genial! ¿Qué tal lo estás pasando?' : 'Great! How are you enjoying it?',
                    quickReplies: lang === 'es' 
                        ? [
                            { text: '😊 Bien', value: 'Bien' },
                            { text: '😐 Regular', value: 'Regular' },
                            { text: '😕 Mal', value: 'Mal' }
                        ]
                        : [
                            { text: '😊 Good', value: 'Good' },
                            { text: '😐 Okay', value: 'Okay' },
                            { text: '😕 Bad', value: 'Bad' }
                        ]
                };
            } else {
                return {
                    text: lang === 'es'
                        ? 'Entendido. Si tienes alguna pregunta sobre capibara6, estaré encantado de ayudarte. 😊'
                        : 'Understood. If you have any questions about capibara6, I\'ll be happy to help you. 😊',
                    quickReplies: null
                };
            }
        }
        
        // Manejar respuesta sobre cómo lo está pasando
        if (this.awaitingExperienceResponse) {
            this.awaitingExperienceResponse = false;
            
            if (lowerMessage.includes('bien') || lowerMessage.includes('good') || 
                lowerMessage.includes('great') || lowerMessage.includes('genial') || 
                lowerMessage.includes('excelente')) {
                return {
                    text: lang === 'es' ? '¡Me alegro mucho! 😊' : 'I\'m very glad! 😊',
                    quickReplies: null
                };
            } else {
                return {
                    text: lang === 'es'
                        ? 'Bueno, ¡por lo menos el catering no está mal no? XD'
                        : 'Well, at least the catering isn\'t bad, right? XD',
                    quickReplies: null
                };
            }
        }
        
        // Detectar si pregunta si somos capibara6 de verdad
        if ((lowerMessage.includes('eres') || lowerMessage.includes('are you')) && 
            (lowerMessage.includes('capibara6') || lowerMessage.includes('verdad') || 
             lowerMessage.includes('real') || lowerMessage.includes('de verdad'))) {
            return {
                text: lang === 'es'
                    ? 'Soy solo un chatbot, un asistente virtual para ayudarte con información sobre capibara6. 🤖'
                    : 'I\'m just a chatbot, a virtual assistant to help you with information about capibara6. 🤖',
                quickReplies: null
            };
        }
        
        // Buscar respuesta relevante para preguntas específicas
        for (const [keywords, response] of Object.entries(responses)) {
            if (keywords === 'default') continue; // Saltar el default en esta iteración
            
            const keywordList = keywords.split('|');
            if (keywordList.some(keyword => lowerMessage.includes(keyword))) {
                return response;
            }
        }
        
        // Si no hay pregunta específica, ofrecer contacto y preguntar por el congreso
        return responses.default();
    }
    
    getResponses() {
        return {
            es: {
                'hola|saludos|hey|hi': { 
                    text: '¡Hola! 👋 Soy el asistente de <strong>capibara6</strong>. ¿Te gustaría saber más sobre nuestra arquitectura híbrida, rendimiento o características?',
                    quickReplies: null
                },
                'precio|costo|coste|price': {
                    text: 'capibara6 es un proyecto de código abierto. Para uso enterprise, contáctanos en <a href="mailto:info@anachroni.co" style="color: var(--primary-light);">info@anachroni.co</a> para planes personalizados.',
                    quickReplies: null
                },
                'tpu|hardware|procesador': {
                    text: 'capibara6 está optimizado para <strong>Google TPU v5e/v6e-64</strong> (4,500+ tokens/sec) y <strong>Google ARM Axion</strong> (2,100+ tokens/sec). ¡Rendimiento enterprise-grade! ⚡',
                    quickReplies: null
                },
                'arquitectura|modelo|architecture': {
                    text: 'Usamos una arquitectura híbrida: <strong>70% Transformer</strong> (precisión) + <strong>30% Mamba SSM</strong> (velocidad O(n)). Lo mejor de ambos mundos! 🧠',
                    quickReplies: null
                },
                'contexto|tokens|ventana': {
                    text: '¡Tenemos la <strong>mayor ventana de contexto</strong> del mercado con más de <strong>10M tokens</strong>! Superamos a GPT-4 (128K), Claude (200K) y Gemini (1M). 🏆',
                    quickReplies: null
                },
                'compliance|gdpr|privacidad|seguridad': {
                    text: 'Cumplimos <strong>100%</strong> con GDPR, CCPA y AI Act de la UE. Certificado para empresas y <strong>administraciones públicas</strong>. 🔒',
                    quickReplies: null
                },
                'multimodal|imagen|video|audio': {
                    text: 'Sí! Procesamos <strong>texto, imagen y video</strong> con encoders especializados. También tenemos Text-to-Speech con contexto emocional. 🌐',
                    quickReplies: null
                },
                'mamba|transformer|moe': {
                    text: 'Nuestra arquitectura combina 32 expertos MoE con routing dinámico, más el balance Transformer/Mamba. Precisión del 97.8% con eficiencia O(n). 🎯',
                    quickReplies: null
                },
                'instalar|install|setup|comenzar': {
                    text: 'Para comenzar: <code>git clone https://github.com/anachroni-co/capibara6</code> y sigue nuestra <a href="#quickstart">guía rápida</a>. Necesitas Python 3.9+ y acceso a TPU/ARM Axion. 🚀',
                    quickReplies: null
                },
                'github|repo|repositorio|code': {
                    text: 'Nuestro repositorio está en <a href="https://github.com/anachroni-co/capibara6" target="_blank">github.com/anachroni-co/capibara6</a>. ¡Dale una ⭐ si te gusta!',
                    quickReplies: null
                },
                'anachroni|empresa|company': {
                    text: '<strong>Anachroni s.coop</strong> es una cooperativa española especializada en IA avanzada. Visita <a href="https://www.anachroni.co" target="_blank">www.anachroni.co</a> o escríbenos a info@anachroni.co 🇪🇸',
                    quickReplies: null
                },
                'demo|prueba|test': {
                    text: 'Estamos preparando demos interactivas. Mientras tanto, explora la <a href="#docs">documentación</a> o contacta con nosotros para un acceso anticipado. 🎪',
                    quickReplies: null
                },
                'default': () => {
                    if (!this.hasAskedAboutCongress) {
                        this.awaitingCongressResponse = true;
                        this.hasAskedAboutCongress = true;
                        localStorage.setItem('hasAskedAboutCongress', 'true');
                        return {
                            text: 'Si necesitas información más específica, déjanos tu email y te contactaremos pronto. Puedes escribirnos a <a href="mailto:info@anachroni.co" style="color: var(--primary-light);">info@anachroni.co</a> 📧<br><br>Por cierto, ¿estás en el Congreso ES2030?',
                            quickReplies: [
                                { text: '✅ Sí', value: 'Sí' },
                                { text: '❌ No', value: 'No' }
                            ]
                        };
                    } else {
                        return {
                            text: 'Si necesitas información más específica, déjanos tu email y te contactaremos pronto. Puedes escribirnos a <a href="mailto:info@anachroni.co" style="color: var(--primary-light);">info@anachroni.co</a> 📧',
                            quickReplies: null
                        };
                    }
                }
            },
            en: {
                'hello|hi|hey|greetings': {
                    text: 'Hello! 👋 I\'m the <strong>capibara6</strong> assistant. Would you like to know more about our hybrid architecture, performance, or features?',
                    quickReplies: null
                },
                'price|cost|pricing': {
                    text: 'capibara6 is an open-source project. For enterprise use, contact us at <a href="mailto:info@anachroni.co" style="color: var(--primary-light);">info@anachroni.co</a> for custom plans.',
                    quickReplies: null
                },
                'tpu|hardware|processor': {
                    text: 'capibara6 is optimized for <strong>Google TPU v5e/v6e-64</strong> (4,500+ tokens/sec) and <strong>Google ARM Axion</strong> (2,100+ tokens/sec). Enterprise-grade performance! ⚡',
                    quickReplies: null
                },
                'architecture|model': {
                    text: 'We use a hybrid architecture: <strong>70% Transformer</strong> (precision) + <strong>30% Mamba SSM</strong> (O(n) speed). Best of both worlds! 🧠',
                    quickReplies: null
                },
                'context|tokens|window': {
                    text: 'We have the <strong>largest context window</strong> in the market with over <strong>10M tokens</strong>! We surpass GPT-4 (128K), Claude (200K), and Gemini (1M). 🏆',
                    quickReplies: null
                },
                'compliance|gdpr|privacy|security': {
                    text: 'We comply <strong>100%</strong> with GDPR, CCPA, and EU AI Act. Certified for enterprises and <strong>public administrations</strong>. 🔒',
                    quickReplies: null
                },
                'multimodal|image|video|audio': {
                    text: 'Yes! We process <strong>text, image, and video</strong> with specialized encoders. We also have Text-to-Speech with emotional context. 🌐',
                    quickReplies: null
                },
                'mamba|transformer|moe': {
                    text: 'Our architecture combines 32 MoE experts with dynamic routing, plus the Transformer/Mamba balance. 97.8% accuracy with O(n) efficiency. 🎯',
                    quickReplies: null
                },
                'install|setup|start|begin': {
                    text: 'To start: <code>git clone https://github.com/anachroni-co/capibara6</code> and follow our <a href="#quickstart">quick guide</a>. You need Python 3.9+ and TPU/ARM Axion access. 🚀',
                    quickReplies: null
                },
                'github|repo|repository|code': {
                    text: 'Our repository is at <a href="https://github.com/anachroni-co/capibara6" target="_blank">github.com/anachroni-co/capibara6</a>. Give us a ⭐ if you like it!',
                    quickReplies: null
                },
                'anachroni|company': {
                    text: '<strong>Anachroni s.coop</strong> is a Spanish cooperative specialized in advanced AI. Visit <a href="https://www.anachroni.co" target="_blank">www.anachroni.co</a> or write to info@anachroni.co 🇪🇸',
                    quickReplies: null
                },
                'demo|trial|test': {
                    text: 'We\'re preparing interactive demos. Meanwhile, explore the <a href="#docs">documentation</a> or contact us for early access. 🎪',
                    quickReplies: null
                },
                'default': () => {
                    if (!this.hasAskedAboutCongress) {
                        this.awaitingCongressResponse = true;
                        this.hasAskedAboutCongress = true;
                        localStorage.setItem('hasAskedAboutCongress', 'true');
                        return {
                            text: 'If you need more specific information, leave us your email and we\'ll contact you soon. You can write to us at <a href="mailto:info@anachroni.co" style="color: var(--primary-light);">info@anachroni.co</a> 📧<br><br>By the way, are you at the ES2030 Congress?',
                            quickReplies: [
                                { text: '✅ Yes', value: 'Yes' },
                                { text: '❌ No', value: 'No' }
                            ]
                        };
                    } else {
                        return {
                            text: 'If you need more specific information, leave us your email and we\'ll contact you soon. You can write to us at <a href="mailto:info@anachroni.co" style="color: var(--primary-light);">info@anachroni.co</a> 📧',
                            quickReplies: null
                        };
                    }
                }
            }
        };
    }
    
    // Métodos para gestión de datos del usuario
    loadUserData() {
        const data = localStorage.getItem('capibara6_user_data');
        return data ? JSON.parse(data) : {
            emails: [],
            conversations: [],
            timestamp: new Date().toISOString()
        };
    }
    
    saveUserMessage(message) {
        const timestamp = new Date().toISOString();
        
        // Guardar el mensaje
        this.userConversations.conversations.push({
            message: message,
            timestamp: timestamp
        });
        
        // Detectar email
        const emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g;
        const emails = message.match(emailRegex);
        
        if (emails) {
            emails.forEach(email => {
                if (!this.userConversations.emails.includes(email)) {
                    this.userConversations.emails.push(email);
                }
            });
        }
        
        // Guardar en localStorage
        localStorage.setItem('capibara6_user_data', JSON.stringify(this.userConversations));
        
        // Enviar al backend si hay email
        if (emails && emails.length > 0) {
            this.sendToBackend(emails[0]);
        }
    }
    
    async sendToBackend(email) {
        const lang = document.documentElement.getAttribute('data-lang') || 'es';
        
        // Mostrar mensaje de confirmación inmediatamente
        setTimeout(() => {
            const confirmMsg = lang === 'es' 
                ? '✅ ¡Gracias! Hemos guardado tu email. Te contactaremos pronto. 📧'
                : '✅ Thank you! We\'ve saved your email. We\'ll contact you soon. 📧';
            this.addMessage(confirmMsg, 'bot', null);
        }, 800);
        
        // Intentar enviar al backend en segundo plano
        try {
            const backendUrl = typeof CHATBOT_CONFIG !== 'undefined'
                ? CHATBOT_CONFIG.BACKEND_URL + CHATBOT_CONFIG.ENDPOINTS.SAVE_CONVERSATION
                : (window.location.hostname === 'localhost'
                    ? 'http://localhost:5001/api/save-conversation'
                    : 'http://34.12.166.76:5001/api/save-conversation');
            
            console.log('Enviando email al backend:', email);
            console.log('URL del backend:', backendUrl);
            
            const response = await fetch(backendUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    email: email,
                    conversations: this.userConversations.conversations
                })
            });
            
            const result = await response.json();
            console.log('Respuesta del backend:', result);
            
            if (result.success) {
                console.log('✅ Email guardado y enviado correctamente');
            } else {
                console.warn('⚠️ El backend respondió pero hubo un error:', result.error);
            }
        } catch (error) {
            // Solo registrar el error en consola, no mostrar al usuario
            console.warn('⚠️ Backend no disponible:', error);
        }
    }
}

// Inicializar chatbot
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new Capibara6Chat();
    });
} else {
    new Capibara6Chat();
}



