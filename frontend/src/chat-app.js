// ============================================
// Chat Application - Capibara6
// ============================================

// Configuración del sistema de consenso (comentado por ahora, usar modelo original)
    const CONSENSUS_CONFIG = {
        serverUrl: 'http://localhost:5005/api/consensus/query',
    fallbackUrl: 'http://34.175.104.187:8080/completion',  // IP actualizada
    enabled: false, // Deshabilitado por ahora
    defaultTemplate: 'general',
    selectedModels: ['capibara6']
};

// Configuración del modelo original
    const MODEL_CONFIG = {
        // Usar proxy de Vercel para evitar problemas de Mixed Content
        serverUrl: 'http://localhost:5001/api/chat',  // Servidor local
    systemPrompt: 'Eres Capibara6, un asistente experto en tecnología, programación e IA. Responde de forma clara, estructurada y en español.',  // System prompt mejorado
    defaultParams: {
        n_predict: 200,  // Optimizado para respuestas completas pero no excesivas
        temperature: 0.8,  // Aumentado para más creatividad y diversidad
        top_p: 0.9,  // Aumentado para mejor diversidad de respuestas
        repeat_penalty: 1.1,  // Mantenido bajo para evitar repeticiones excesivas
        presence_penalty: 0.0,  // Deshabilitado (causaba respuestas extrañas)
        frequency_penalty: 0.0,  // Deshabilitado (causaba respuestas extrañas)
        stop: [
            "Usuario:",
            "Capibara6:",
            "<end_of_turn>",
            "<|end_of_turn|>",
            "<|im_end|>",
            "\n\n"
        ]
    }
};

// Estado de la aplicación
let currentChatId = null;
let chats = [];
let isTyping = false;
let attachedFiles = [];
let lastUserMessage = null;
let abortController = null;

// Estado del consenso (deshabilitado por ahora)
let availableModels = ['capibara6'];
let availableTemplates = ['general'];
let selectedTemplate = 'general';
let selectedModels = ['capibara6'];
let consensusEnabled = false;

// Elementos del DOM
const sidebar = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebar-overlay');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebarToggleHeader = document.getElementById('sidebar-toggle-header');
const newChatBtn = document.getElementById('new-chat-btn');
const emptyState = document.getElementById('empty-state');
const chatArea = document.getElementById('chat-area');
const messagesContainer = document.getElementById('messages-container');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const attachBtn = document.getElementById('attach-btn');
const fileInput = document.getElementById('file-input');
const attachmentsPreview = document.getElementById('attachments-preview');
const chatsToday = document.getElementById('chats-today');
const chatsPrevious = document.getElementById('chats-previous');
const statusLed = document.getElementById('status-led');
const statusText = document.getElementById('status-text');
const consensusIndicator = document.getElementById('consensus-indicator');
const consensusCount = document.getElementById('consensus-count');

// ============================================
// Inicialización
// ============================================
function init() {
    console.log('🚀 Iniciando Capibara6 Chat...');
    loadChatsFromStorage();
    setupEventListeners();
    console.log('✅ Event listeners configurados');
    updateChatsList();
    restoreSidebarState();
    
    // Aplicar idioma guardado
    const savedLang = localStorage.getItem('preferred-language') || 'es';
    if (typeof changeLanguage === 'function') {
        changeLanguage(savedLang);
    }
    
    // Verificar conexión al servidor
    checkServerConnection();
    
    // TTS ya inicializado automáticamente en tts-integration.js
    
    // Cargar configuración del consenso (comentado por ahora)
    // loadConsensusConfig();
    
    // Click en indicador de consenso para ver info
    if (consensusIndicator) {
        consensusIndicator.addEventListener('click', showConsensusInfo);
    }
    
    // Actualizar contador de consenso con 1 modelo
    updateConsensusCount(1);
    
    // Cerrar sidebar al hacer click en el overlay
    if (sidebarOverlay) {
        sidebarOverlay.addEventListener('click', () => {
            closeSidebar();
        });
    }
    
    // Manejar cambio de tamaño de ventana
    window.addEventListener('resize', () => {
        const isMobile = window.innerWidth <= 768;
        const isHidden = sidebar.classList.contains('hidden');
        
        // Si cambiamos de móvil a desktop y el sidebar está visible, quitar overlay
        if (!isMobile && !isHidden && sidebarOverlay) {
            sidebarOverlay.classList.remove('active');
        }
        
        // Si cambiamos de desktop a móvil y el sidebar está visible, mostrar overlay
        if (isMobile && !isHidden && sidebarOverlay) {
            sidebarOverlay.classList.add('active');
        }
    });
    
    console.log('✅ Chat app initialized');
}

// ============================================
// Event Listeners
// ============================================
function setupEventListeners() {
    // Sidebar toggle
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', toggleSidebar);
    }
    if (sidebarToggleHeader) {
        sidebarToggleHeader.addEventListener('click', toggleSidebar);
    }
    
    // Nueva conversación
    if (newChatBtn) {
        newChatBtn.addEventListener('click', createNewChat);
    }
    
    // Input de mensaje
    if (messageInput) {
        messageInput.addEventListener('input', handleInputChange);
        messageInput.addEventListener('keydown', handleKeyDown);
    }
    
    // Enviar mensaje o detener generación
    if (sendBtn) {
        sendBtn.addEventListener('click', () => {
            console.log('🖱️ Botón de envío clickeado');
            if (isTyping) {
                console.log('⏹️ Deteniendo generación...');
                stopGeneration();
            } else {
                console.log('📤 Enviando mensaje...');
                sendMessage();
            }
        });
    }
    
    // Adjuntar archivo
    if (attachBtn) {
        attachBtn.addEventListener('click', () => {
            fileInput.click();
        });
    }
    
    // Manejar selección de archivos
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    // Selector de plantillas
    const templateSelect = document.getElementById('template-select');
    if (templateSelect) {
        templateSelect.addEventListener('change', (e) => {
            const templateKey = e.target.value;
            if (window.applyTemplate) {
                window.applyTemplate(templateKey);
                
                // Actualizar descripción
                const template = window.TEMPLATE_PROFILES[templateKey];
                const descEl = document.getElementById('template-description');
                if (descEl && template) {
                    descEl.textContent = template.description;
                }
            }
        });
    }
    
    // Botón de estadísticas de plantillas (ahora abre modal detallado)
    const btnTemplateStats = document.getElementById('show-template-stats');
    if (btnTemplateStats) {
        btnTemplateStats.addEventListener('click', () => {
            if (window.showDetailedStats) {
                window.showDetailedStats();
            }
        });
    }
    
    // Tarjetas de sugerencia
    document.querySelectorAll('.suggestion-card').forEach(card => {
        card.addEventListener('click', () => {
            const prompt = card.getAttribute('data-prompt');
            if (prompt) {
                startNewChatWithPrompt(prompt);
            }
        });
    });
    
    // Botones de configuración
    const settingsBtn = document.querySelector('.chat-actions button[title="Configuración"]');
    if (settingsBtn) {
        settingsBtn.addEventListener('click', openSettingsModal);
    }
    
    // Selector de idioma en modal
    const languageSelect = document.getElementById('language-select');
    if (languageSelect) {
        languageSelect.addEventListener('change', (e) => {
            if (typeof changeLanguage === 'function') {
                changeLanguage(e.target.value);
            }
        });
    }
    
    // Pestañas del sidebar
    document.querySelectorAll('.sidebar-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTab = tab.getAttribute('data-tab');
            
            // Desactivar todas las pestañas
            document.querySelectorAll('.sidebar-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            // Activar la pestaña seleccionada
            tab.classList.add('active');
            const tabContent = document.getElementById(`tab-${targetTab}`);
            if (tabContent) {
                tabContent.classList.add('active');
            }
        });
    });
}

// ============================================
// Gestión de Sidebar
// ============================================
function toggleSidebar() {
    const isHidden = sidebar.classList.contains('hidden');
    
    if (isHidden) {
        openSidebar();
    } else {
        closeSidebar();
    }
}

function openSidebar() {
    sidebar.classList.remove('hidden');
    if (sidebarOverlay) {
        // En móvil o cuando el sidebar se superpone, mostrar overlay
        const isMobile = window.innerWidth <= 768;
        if (isMobile) {
            sidebarOverlay.classList.add('active');
        }
    }
    localStorage.setItem('sidebar-hidden', 'false');
}

function closeSidebar() {
    sidebar.classList.add('hidden');
    if (sidebarOverlay) {
        sidebarOverlay.classList.remove('active');
    }
    localStorage.setItem('sidebar-hidden', 'true');
}

// Restaurar estado del sidebar al cargar
function restoreSidebarState() {
    const sidebarHidden = localStorage.getItem('sidebar-hidden');
    
    // En desktop, mostrar sidebar por defecto
    // En mobile, ocultar por defecto
    const isMobile = window.innerWidth <= 768;
    
    if (sidebarHidden === 'true' || (sidebarHidden === null && isMobile)) {
        closeSidebar();
    } else if (sidebarHidden === 'false' && !isMobile) {
        openSidebar();
    } else {
        closeSidebar();
    }
}

// ============================================
// Gestión de Chats
// ============================================
function createNewChat() {
    const chat = {
        id: generateId(),
        title: 'Nueva Conversación',
        messages: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
    };
    
    chats.unshift(chat);
    currentChatId = chat.id;
    
    // Limpiar contexto anterior
    lastUserMessage = null;
    attachedFiles = [];
    
    saveChatsToStorage();
    updateChatsList();
    showChatArea();
    clearMessages();
    updateAttachmentsPreview();
    
    messageInput.focus();
    
    console.log('✅ Nuevo chat creado - Contexto limpio');
}

function loadChat(chatId) {
    const chat = chats.find(c => c.id === chatId);
    if (!chat) return;
    
    currentChatId = chatId;
    
    // Limpiar contexto anterior al cambiar de chat
    lastUserMessage = null;
    attachedFiles = [];
    
    showChatArea();
    clearMessages();
    updateAttachmentsPreview();
    
    // Cargar mensajes
    chat.messages.forEach(msg => {
        appendMessage(msg.role, msg.content, false);
    });
    
    // Recuperar el último mensaje del usuario para regenerar
    const userMessages = chat.messages.filter(m => m.role === 'user');
    if (userMessages.length > 0) {
        lastUserMessage = userMessages[userMessages.length - 1].content;
    }
    
    // Actualizar título
    document.getElementById('current-chat-title').textContent = chat.title;
    
    // Actualizar lista de chats
    updateChatsList();
    
    console.log('✅ Chat cargado:', chat.title);
}

function deleteChat(chatId) {
    if (confirm('¿Estás seguro de que quieres eliminar esta conversación?')) {
        chats = chats.filter(c => c.id !== chatId);
        
        if (currentChatId === chatId) {
            currentChatId = null;
            showEmptyState();
        }
        
        saveChatsToStorage();
        updateChatsList();
    }
}

function updateChatsList() {
    const today = new Date().toDateString();
    const todayChats = [];
    const previousChats = [];
    
    chats.forEach(chat => {
        const chatDate = new Date(chat.updatedAt).toDateString();
        if (chatDate === today) {
            todayChats.push(chat);
        } else {
            previousChats.push(chat);
        }
    });
    
    chatsToday.innerHTML = todayChats.length > 0 
        ? todayChats.map(renderChatItem).join('') 
        : '<div style="padding: 1rem; text-align: center; color: var(--text-muted); font-size: 0.9rem;">No hay conversaciones hoy</div>';
    
    chatsPrevious.innerHTML = previousChats.length > 0 
        ? previousChats.map(renderChatItem).join('') 
        : '<div style="padding: 1rem; text-align: center; color: var(--text-muted); font-size: 0.9rem;">No hay conversaciones anteriores</div>';
    
    // Reattach event listeners
    document.querySelectorAll('.chat-item').forEach(item => {
        item.addEventListener('click', (e) => {
            if (!e.target.closest('.btn-chat-action')) {
                loadChat(item.dataset.chatId);
            }
        });
    });
    
    document.querySelectorAll('.btn-delete-chat').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteChat(btn.dataset.chatId);
        });
    });
}

function renderChatItem(chat) {
    const isActive = chat.id === currentChatId;
    const preview = chat.messages.length > 0 
        ? chat.messages[chat.messages.length - 1].content.substring(0, 50) + '...'
        : 'Sin mensajes';
    
    return `
        <div class="chat-item ${isActive ? 'active' : ''}" data-chat-id="${chat.id}">
            <div class="chat-item-content">
                <div class="chat-item-title">${escapeHtml(chat.title)}</div>
                <div class="chat-item-preview">${escapeHtml(preview)}</div>
            </div>
            <div class="chat-item-actions">
                <button class="btn-chat-action btn-delete-chat" data-chat-id="${chat.id}" title="Eliminar">
                    <i data-lucide="x" style="width: 16px; height: 16px;"></i>
                </button>
            </div>
        </div>
    `;
}

// ============================================
// Gestión de Mensajes
// ============================================
function handleInputChange() {
    // Auto-resize textarea
    messageInput.style.height = 'auto';
    messageInput.style.height = messageInput.scrollHeight + 'px';
    
    updateSendButtonState();
}

function updateSendButtonState() {
    // Enable send button if there's text or files, or if is typing
    sendBtn.disabled = !isTyping && messageInput.value.trim() === '' && attachedFiles.length === 0;
}

function setSendButtonToStop() {
    sendBtn.disabled = false;
    sendBtn.classList.add('stop-mode');
    const icon = sendBtn.querySelector('i');
    if (icon) {
        icon.setAttribute('data-lucide', 'square');
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
    }
}

function setSendButtonToSend() {
    sendBtn.classList.remove('stop-mode');
    const icon = sendBtn.querySelector('i');
    if (icon) {
        icon.setAttribute('data-lucide', 'arrow-up');
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
    }
    updateSendButtonState();
}

function stopGeneration() {
    if (abortController) {
        abortController.abort();
        abortController = null;
    }
    
    isTyping = false;
    setSendButtonToSend();
    updateServerStatus('connected', 'Detenido');
    
    // Quitar mensaje de streaming si existe
    const streamingMsg = document.getElementById('streaming-message');
    if (streamingMsg) {
        streamingMsg.classList.remove('streaming');
    }
    
    console.log('⏹️ Generación detenida por el usuario');
}

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

async function sendMessage() {
    console.log('🔍 sendMessage() llamada');
    const content = messageInput.value.trim();
    console.log('📝 Contenido:', content);
    console.log('⏳ isTyping:', isTyping);
    
    if ((!content && attachedFiles.length === 0) || isTyping) {
        console.log('❌ Saliendo: sin contenido o está escribiendo');
        return;
    }
    
    // Crear chat si no existe
    if (!currentChatId) {
        createNewChat();
    }
    
    // Construir mensaje con archivos si hay
    let messageContent = content;
    if (attachedFiles.length > 0) {
        const filesList = attachedFiles.map(f => f.name).join(', ');
        messageContent = content ? `${content}\n\n📎 Archivos adjuntos: ${filesList}` : `📎 Archivos adjuntos: ${filesList}`;
    }
    
    // Guardar el último mensaje del usuario para regeneración
    lastUserMessage = content || 'Archivos adjuntos';

    // Guardar copia de los archivos antes de limpiarlos
    const filesToSend = [...attachedFiles];

    // Agregar mensaje del usuario
    console.log('📨 Agregando mensaje del usuario:', messageContent);
    appendMessage('user', messageContent);

    // Limpiar input y archivos (con pequeño delay para evitar flash visual)
    setTimeout(() => {
        messageInput.value = '';
        messageInput.style.height = 'auto';
        attachedFiles = [];
        updateAttachmentsPreview();
        updateSendButtonState();
    }, 50);
    
    // Guardar mensaje
    saveMessage('user', messageContent);
    console.log('💾 Mensaje guardado');
    
    // Actualizar título del chat si es el primer mensaje
    const chat = chats.find(c => c.id === currentChatId);
    if (chat && chat.messages.length === 1) {
        chat.title = content.substring(0, 50) + (content.length > 50 ? '...' : '');
        document.getElementById('current-chat-title').textContent = chat.title;
        updateChatsList();
    }
    
    // Simular respuesta del asistente
    console.log('🎬 Llamando a simulateAssistantResponse con:', lastUserMessage);
    await simulateAssistantResponse(lastUserMessage, filesToSend);
    console.log('✅ simulateAssistantResponse completada');
}

function appendMessage(role, content, save = true, stats = null) {
    console.log(`📝 appendMessage(${role}):`, content.substring(0, 50) + '...');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const isUser = role === 'user';
    const roleText = isUser ? 'Tú' : 'Capibara6';
    const iconName = isUser ? 'circle-user' : 'sparkles';
    
    // Generar HTML de estadísticas si están disponibles
    let statsHTML = '';
    if (!isUser && stats) {
        statsHTML = `
            <div class="message-stats">
                <div class="stat-item" title="Tiempo de generación">
                    <i data-lucide="clock" style="width: 12px; height: 12px;"></i>
                    <span>${stats.duration}s</span>
                </div>
                <div class="stat-item" title="Tokens generados">
                    <i data-lucide="message-square" style="width: 12px; height: 12px;"></i>
                    <span>${stats.tokens} gen</span>
                </div>
                <div class="stat-item" title="Tokens del prompt">
                    <i data-lucide="arrow-right" style="width: 12px; height: 12px;"></i>
                    <span>${stats.promptTokens} in</span>
                </div>
                <div class="stat-item" title="Total de tokens">
                    <i data-lucide="layers" style="width: 12px; height: 12px;"></i>
                    <span>${stats.totalTokens} total</span>
                </div>
                <div class="stat-item" title="Velocidad de generación">
                    <i data-lucide="zap" style="width: 12px; height: 12px;"></i>
                    <span>${stats.tokensPerSecond} tok/s</span>
                </div>
                <div class="stat-item" title="Modelo usado">
                    <i data-lucide="cpu" style="width: 12px; height: 12px;"></i>
                    <span>${stats.model}</span>
                </div>
            </div>
        `;
    }
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i data-lucide="${iconName}" style="width: 20px; height: 20px;"></i>
        </div>
        <div class="message-content">
            <div class="message-role">${roleText}</div>
            <div class="message-text">${formatMessage(content)}</div>
            ${statsHTML}
            ${!isUser ? `
                <div class="message-actions">
                    <button class="btn-message-action">
                        <i data-lucide="clipboard" style="width: 14px; height: 14px;"></i>
                        Copiar
                    </button>
                    <button class="btn-message-action">
                        <i data-lucide="star" style="width: 14px; height: 14px;"></i>
                        Evaluar
                    </button>
                    <button class="btn-message-action">
                        <i data-lucide="message-circle" style="width: 14px; height: 14px;"></i>
                        Regenerar
                    </button>
                </div>
            ` : ''}
        </div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Reinicializar iconos de Lucide
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
    
    // Agregar event listeners para acciones
    messageDiv.querySelectorAll('.btn-message-action').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const btnText = btn.textContent.trim();
            if (btnText.includes('Copiar')) {
                copyMessageContent(content);
            } else if (btnText.includes('Escuchar')) {
                // Leer el mensaje en voz alta
                if (window.speakText) {
                    const textDiv = messageDiv.querySelector('.message-text');
                    const textContent = textDiv ? textDiv.textContent : content;
                    window.speakText(textContent, btn);
                } else {
                    console.error('❌ TTS no está cargado');
                }
            } else if (btnText.includes('Regenerar')) {
                regenerateResponse();
            } else if (btnText.includes('Evaluar')) {
                console.log('⭐ Botón Evaluar clickeado');
                console.log('🔍 showRatingModal disponible:', typeof window.showRatingModal);
                
                // Abrir modal de evaluación detallada
                if (window.showRatingModal) {
                    console.log('✅ Abriendo modal de evaluación...');
                    window.showRatingModal(messageDiv);
                } else {
                    console.error('❌ window.showRatingModal no está disponible');
                    alert('El sistema de evaluación no está cargado. Recarga la página.');
                }
            }
        });
    });
    
    if (save) {
        saveMessage(role, content);
    }
}

function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant typing';
    typingDiv.id = 'typing-indicator';
    typingDiv.innerHTML = `
        <div class="message-avatar">
            <i data-lucide="sparkles" style="width: 20px; height: 20px;"></i>
        </div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
}

function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

function createStreamingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant streaming';
    messageDiv.id = 'streaming-message';
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i data-lucide="sparkles" style="width: 20px; height: 20px;"></i>
        </div>
        <div class="message-content">
            <div class="message-role">Capibara6</div>
            <div class="message-text"></div>
        </div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
    
    return messageDiv;
}

function addStatsToMessage(messageDiv, stats) {
    const statsHTML = `
        <div class="message-stats">
            <div class="stat-item" title="Tiempo de generación">
                <i data-lucide="clock"></i>
                <span>${stats.duration}s</span>
            </div>
            <div class="stat-item" title="Tokens generados">
                <i data-lucide="message-square"></i>
                <span>${stats.tokens} gen</span>
            </div>
            <div class="stat-item" title="Tokens del prompt">
                <i data-lucide="arrow-right"></i>
                <span>${stats.promptTokens} in</span>
            </div>
            <div class="stat-item" title="Total de tokens">
                <i data-lucide="layers"></i>
                <span>${stats.totalTokens} total</span>
            </div>
            <div class="stat-item" title="Velocidad de generación">
                <i data-lucide="zap"></i>
                <span>${stats.tokensPerSecond} tok/s</span>
            </div>
            <div class="stat-item" title="Modelo usado">
                <i data-lucide="cpu"></i>
                <span>${stats.model}</span>
            </div>
        </div>
        <div class="message-actions">
            <button class="btn-message-action">
                <i data-lucide="clipboard" style="width: 14px; height: 14px;"></i>
                Copiar
            </button>
            <button class="btn-message-action btn-speak">
                <i data-lucide="volume" style="width: 14px; height: 14px;"></i>
                <span class="btn-text">Escuchar</span>
            </button>
            <button class="btn-message-action">
                <i data-lucide="star" style="width: 14px; height: 14px;"></i>
                Evaluar
            </button>
            <button class="btn-message-action">
                <i data-lucide="message-circle" style="width: 14px; height: 14px;"></i>
                Regenerar
            </button>
        </div>
    `;
    
    const contentDiv = messageDiv.querySelector('.message-content');
    contentDiv.insertAdjacentHTML('beforeend', statsHTML);
    
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
    
    // Agregar event listeners
    messageDiv.querySelectorAll('.btn-message-action').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const btnText = btn.textContent.trim();
            const textDiv = messageDiv.querySelector('.message-text');
            const content = textDiv.textContent;
            
            if (btnText.includes('Copiar')) {
                copyMessageContent(content);
            } else if (btnText.includes('Escuchar')) {
                // Leer el mensaje en voz alta
                if (window.speakText) {
                    const textDiv = messageDiv.querySelector('.message-text');
                    const textContent = textDiv ? textDiv.textContent : content;
                    window.speakText(textContent, btn);
                } else {
                    console.error('❌ TTS no está cargado');
                }
            } else if (btnText.includes('Regenerar')) {
                regenerateResponse();
            } else if (btnText.includes('Evaluar')) {
                console.log('⭐ Botón Evaluar clickeado');
                console.log('🔍 showRatingModal disponible:', typeof window.showRatingModal);
                
                // Abrir modal de evaluación detallada
                if (window.showRatingModal) {
                    console.log('✅ Abriendo modal de evaluación...');
                    window.showRatingModal(messageDiv);
                } else {
                    console.error('❌ window.showRatingModal no está disponible');
                    alert('El sistema de evaluación no está cargado. Recarga la página.');
                }
            }
        });
    });
}

async function simulateAssistantResponse(userMessage, filesToSend = []) {
    console.log('🤖 simulateAssistantResponse() iniciada con:', userMessage);
    isTyping = true;
    hideTypingIndicator();
    updateServerStatus('connecting', 'Generando...');
    setSendButtonToStop();
    
    const startTime = performance.now();
    let streamingMessageDiv = null;
    let streamingTextDiv = null;
    let accumulatedText = '';
    let tokensGenerated = 0;
    
    // Crear nuevo AbortController para esta generación
    let abortController = new AbortController();
    
    try {
        // Obtener el historial de la conversación actual
        const chat = chats.find(c => c.id === currentChatId);
        
        // 🎯 SMART MCP: Analizar si necesita contexto adicional
        let augmentedMessage = userMessage;
        if (typeof window.smartMCPAnalyze === 'function') {
            try {
                const mcpResult = await window.smartMCPAnalyze(userMessage);
                augmentedMessage = mcpResult.prompt;
                
                if (mcpResult.needsContext) {
                    console.log('🎯 Smart MCP: Contexto agregado', {
                        contextsAdded: mcpResult.contextsAdded
                    });
                }
            } catch (error) {
                console.warn('⚠️ Smart MCP falló, usando mensaje original:', error);
            }
        }
        
        // Construir historial con system prompt corto
        let conversationHistory = `<bos>`;
        
        if (MODEL_CONFIG.systemPrompt) {
            conversationHistory += `<start_of_turn>system\n${MODEL_CONFIG.systemPrompt}<end_of_turn>\n`;
        }
        
        if (chat && chat.messages.length > 0) {
            // Construir el historial completo de la conversación (usando mensajes ORIGINALES, no augmentados)
            chat.messages.forEach(msg => {
                if (msg.role === 'user') {
                    conversationHistory += `<start_of_turn>user\n${msg.content}<end_of_turn>\n`;
                } else if (msg.role === 'assistant') {
                    conversationHistory += `<start_of_turn>model\n${msg.content}<end_of_turn>\n`;
                }
            });
        }
        
        // 🔥 Agregar el mensaje actual AUGMENTADO
        conversationHistory += `<start_of_turn>user\n${augmentedMessage}<end_of_turn>\n`;
        
        // Agregar el turno del modelo para que genere la respuesta
        conversationHistory += '<start_of_turn>model\n';
        
        // Crear mensaje vacío para streaming
        streamingMessageDiv = createStreamingMessage();
        streamingTextDiv = streamingMessageDiv.querySelector('.message-text');

        // Preparar datos para enviar
        let fetchOptions = {
            method: 'POST',
            signal: abortController.signal
        };

        // Si hay archivos adjuntos, usar FormData; sino, usar JSON
        if (filesToSend.length > 0) {
            const formData = new FormData();
            formData.append('message', userMessage);
            formData.append('max_tokens', MODEL_CONFIG.defaultParams.n_predict);
            formData.append('temperature', MODEL_CONFIG.defaultParams.temperature);

            // Agregar archivos
            filesToSend.forEach(file => {
                formData.append('files', file);
            });

            fetchOptions.body = formData;
            // NO establecer Content-Type para FormData, el navegador lo hace automáticamente
        } else {
            fetchOptions.headers = {
                'Content-Type': 'application/json',
            };
            fetchOptions.body = JSON.stringify({
                message: userMessage,
                max_tokens: MODEL_CONFIG.defaultParams.n_predict,
                temperature: MODEL_CONFIG.defaultParams.temperature
            });
        }

        // Conectar con nuestro backend que se conecta a GPT-OSS-20B
        const response = await fetch(MODEL_CONFIG.serverUrl, fetchOptions);
        
        if (!response.ok) {
            throw new Error(`Error del servidor: ${response.status}`);
        }
        
        // Leer la respuesta JSON de nuestro backend
        const data = await response.json();
        
        if (data.response || data.content) {
            // Simular streaming escribiendo la respuesta caracter por caracter
            const fullResponse = data.response || data.content;
            let currentText = '';
            
            for (let i = 0; i <= fullResponse.length; i++) {
                if (abortController.signal.aborted) {
                    console.log('🛑 Generación cancelada por el usuario');
                    break;
                }
                
                currentText = fullResponse.substring(0, i);
                streamingTextDiv.textContent = currentText;
                streamingMessageDiv.scrollIntoView({ behavior: 'smooth' });
                
                // Pequeña pausa para simular streaming
                if (i < fullResponse.length) {
                    await new Promise(resolve => setTimeout(resolve, 20));
                }
            }
            
            accumulatedText = fullResponse;
            
            // Usar los datos reales del modelo si están disponibles
            if (data.tokens) {
                tokensGenerated = data.tokens;
            }
        } else {
            throw new Error('No se recibió respuesta del modelo');
        }
        
        const endTime = performance.now();
        const duration = ((endTime - startTime) / 1000).toFixed(2);
        
        console.log('📊 Texto acumulado final:', accumulatedText.length, 'chars');
        if (accumulatedText.length === 0) {
            console.error('❌ ERROR: No se acumuló ningún texto');
        }
        
        updateServerStatus('connected', 'Conectado');
        
        // Si no recibió stream, intentar parsear respuesta completa
        if (!accumulatedText && buffer) {
            try {
                const data = JSON.parse(buffer);
                accumulatedText = data.content || '';
                tokensGenerated = data.tokens_predicted || 0;
                tokensEvaluated = data.tokens_evaluated || 0;
                streamingTextDiv.innerHTML = formatMessage(accumulatedText);
            } catch (e) {
                console.error('Error parseando respuesta:', e);
            }
        }
        
        // Limpiar la respuesta de tokens no deseados agresivamente (FINAL)
        accumulatedText = accumulatedText
            // Tokens de control y metadata
            .replace(/<\|im_end\|>/g, '')
            .replace(/<\|end_of_turn\|>/g, '')
            .replace(/<end_of_turn>/g, '')
            .replace(/<model[^>]*>/g, '')
            .replace(/<user[^>]*>/g, '')
            .replace(/<bot[^>]*>/g, '')
            .replace(/<system[^>]*>/g, '')
            .replace(/<im_persona[^>]*>/g, '')
            .replace(/\[INST\]/g, '')
            .replace(/\[\/INST\]/g, '')
            .replace(/\<\>/g, '')
            .replace(/\<s\>/g, '')
            .replace(/\<\/s\>/g, '')
            // Tags HTML
            .replace(/<\/?p>/gi, '')
            .replace(/<\/?div>/gi, '')
            .replace(/<\/?span>/gi, '')
            .replace(/<\/?br>/gi, '')
            .replace(/<img[^>]*>/gi, '')
            .replace(/<audio[^>]*>/gi, '')
            .replace(/<video[^>]*>/gi, '')
            // Metadata
            .replace(/Respuesta:/g, '')
            .replace(/Puntuación:/g, '')
            .replace(/Explicacion:/g, '')
            .replace(/\{-[^}]*\}/g, '')
            .replace(/\{--\}/g, '')
            .replace(/\\\{-\}/g, '')
            .replace(/\bChatGPT\b/g, 'Capibara6')
            .replace(/\bBing\b/g, 'Capibara6')
            // .replace(/\bMicrosoft\b/g, 'Anachroni s.coop')  // Deshabilitado
            // .replace(/\bOpenAI\b/g, 'Anachroni s.coop')  // Deshabilitado
            .replace(/\bClaude\b/g, 'Capibara6')
            .replace(/\bAnthropic\b/g, 'Anachroni s.coop')
            // Artefactos de código
            .replace(/`;$/gm, '')
            .replace(/^const query = `/gm, '')
            .replace(/^const /gm, '')
            .replace(/^let /gm, '')
            .replace(/^var /gm, '')
            // Instrucciones filtradas (solo si empiezan línea)
            .replace(/^Responde de forma clara.*/gim, '')
            .replace(/^Separa párrafos.*/gim, '')
            .replace(/^No uses acrónimos.*/gim, '')
            .replace(/^La respuesta debe tener.*/gim, '')
            .replace(/^\(Inglés estadounidense.*/gim, '')
            .replace(/^usando un estilo conversacional.*/gim, '')
            .replace(/^máximo\s*\d+\s*palabras.*/gim, '')
            .replace(/^Resumen:\s*Combina\s+\w+\s+para crear.*/gim, '')
            .replace(/html<!DOCTYPE/g, '<!DOCTYPE')
            .replace(/html</g, '<')
            .replace(/php</g, '<')
            .replace(/javascript</g, '<')
            .replace(/```\n```/g, '```')
            // Tags HTML incompletos
            .replace(/<\/us$/gi, '')
            .replace(/<\w+$/gi, '')
            .replace(/\(<$/gi, '')
            // Código LaTeX y matemático
            .replace(/\\textbackslash.*/gi, '')
            .replace(/\\begin\{.*?\}.*/gi, '')
            .replace(/\\end\{.*?\}.*/gi, '')
            .replace(/\\hline.*/gi, '')
            .replace(/\{-\}\s*\\.*/gi, '')
            // Palabras sin sentido repetidas
            .replace(/\b(\w{5,})\1+\b/gi, '')
            .replace(/[höøåäöüß]{3,}/gi, '')
            .trim();
        
        // Calcular estadísticas
        if (tokensGenerated === 0) {
            tokensGenerated = accumulatedText.split(' ').length; // Estimación simple si no tenemos datos del modelo
        }
        const tokensEvaluated = 0; // No disponible en nuestro formato
        const tokensPerSecond = duration > 0 ? (tokensGenerated / parseFloat(duration)).toFixed(1) : '0';
        const totalTokens = tokensGenerated + tokensEvaluated;
        const modelName = 'capibara6';
        
        // Quitar clase streaming (elimina cursor parpadeante)
        streamingMessageDiv.classList.remove('streaming');
        
        // Agregar estadísticas y botones
        addStatsToMessage(streamingMessageDiv, {
            duration: duration,
            tokens: tokensGenerated,
            tokensPerSecond: tokensPerSecond,
            promptTokens: tokensEvaluated,
            totalTokens: totalTokens,
            model: modelName
        });
        
        // Guardar mensaje
        saveMessage('assistant', accumulatedText);
        
    } catch (error) {
        // Si fue abortado por el usuario, no mostrar error
        if (error.name === 'AbortError') {
            console.log('⏹️ Generación cancelada por el usuario');
            updateServerStatus('connected', 'Cancelado');
            
            // Guardar lo que se generó hasta ahora
            if (accumulatedText && streamingMessageDiv) {
                streamingMessageDiv.classList.remove('streaming');
                addStatsToMessage(streamingMessageDiv, {
                    duration: ((performance.now() - startTime) / 1000).toFixed(2),
                    tokens: 0,
                    tokensPerSecond: '0',
                    promptTokens: 0,
                    totalTokens: 0,
                    model: 'capibara6'
                });
                saveMessage('assistant', accumulatedText.trim() + ' [Cancelado]');
            } else if (streamingMessageDiv) {
                streamingMessageDiv.remove();
            }
        } else {
            updateServerStatus('error', 'Error');
            console.error('Error al conectar con el modelo:', error);
            
            // Eliminar mensaje de streaming si existe
            if (streamingMessageDiv) {
                streamingMessageDiv.remove();
            }
            
            // Mensaje de error amigable
            const errorMessage = `Lo siento, no pude conectar con el modelo de IA. Error: ${error.message}\n\nPor favor, verifica que el servidor esté activo.`;
            appendMessage('assistant', errorMessage);
        }
    }
    
    isTyping = false;
    setSendButtonToSend();
    abortController = null;
}


/**
 * Detecta y elimina repeticiones de párrafos completos
 */
function removeRepetitions(text) {
    // Dividir en oraciones (aproximadamente)
    const sentences = text.split(/(?<=[.!?])\s+/);
    const seen = new Set();
    const unique = [];
    
    for (const sentence of sentences) {
        const normalized = sentence.trim().toLowerCase();
        // Si la oración es muy corta o ya la vimos, saltarla
        if (normalized.length < 20 || seen.has(normalized)) {
            continue;
        }
        seen.add(normalized);
        unique.push(sentence);
    }
    
    return unique.join(' ');
}

/**
 * Mejora automáticamente el formato si el modelo no lo hizo
 */
function autoImproveFormatting(text) {
    // Si ya tiene formato markdown, no tocar
    if (text.includes('**') || text.includes('##') || text.includes('\n-') || text.includes('\n1.')) {
        return text;
    }
    
    // Detectar listas implícitas (textos que empiezan con palabras enumerativas)
    const listPatterns = [
        /(\d+\.\s+[A-Z])/g,  // "1. Algo"
        /(Primero|Segundo|Tercero|Cuarto|Quinto)[,:]?\s/gi,
        /(En primer lugar|En segundo lugar|Por último)[,:]?\s/gi
    ];
    
    for (const pattern of listPatterns) {
        if (pattern.test(text)) {
            // Convertir a lista numerada
            text = text.replace(/(\d+)\.\s+([A-ZÁÉÍÓÚ])/g, '\n\n$1. **$2');
            break;
        }
    }
    
    // Separar párrafos largos (más de 300 caracteres sin punto y seguido)
    text = text.replace(/([.!?])\s+([A-ZÁÉÍÓÚ])/g, '$1\n\n$2');
    
    // Detectar conceptos técnicos y ponerlos en negrita
    const technicalTerms = [
        'Transformer', 'Mamba', 'arquitectura', 'algoritmo', 'modelo',
        'atención', 'secuencia', 'híbrido', 'eficiente', 'robusta',
        'transformer', 'mamba', 'seq2seq', 'preentrenamiento', 'GPT'
    ];
    
    for (const term of technicalTerms) {
        const regex = new RegExp(`\\b(${term})\\b`, 'gi');
        // Solo aplicar si la palabra no está ya en negrita
        if (!text.includes(`**${term}**`)) {
            let count = 0;
            text = text.replace(regex, (match) => {
                count++;
                // Solo en negrita la primera mención
                return count === 1 ? `**${match}**` : match;
            });
        }
    }
    
    // Resumen automático DESHABILITADO (genera texto sin sentido con Gemma)
    // const wordCount = text.split(/\s+/).length;
    // const paragraphCount = text.split(/\n\n+/).length;
    // 
    // if (wordCount > 400 || paragraphCount > 4) {
    //     if (!text.match(/\b(resumen|en resumen|resumiendo)\b/i)) {
    //         const summary = generateAutoSummary(text);
    //         if (summary && summary.length > 10) {
    //             text += '\n\n---\n\n**Resumen:** ' + summary;
    //         }
    //     }
    // }
    
    return text;
}

/**
 * Genera un resumen automático de la respuesta
 */
function generateAutoSummary(text) {
    // Extraer las primeras oraciones significativas
    const sentences = text.split(/(?<=[.!?])\s+/);
    const firstSentence = sentences[0] || '';
    
    // Buscar conceptos clave (palabras en negrita o términos técnicos)
    const keyConcepts = [];
    const boldMatches = text.match(/\*\*([^*]+)\*\*/g);
    if (boldMatches && boldMatches.length > 0) {
        // Extraer hasta 3 conceptos principales
        for (let i = 0; i < Math.min(3, boldMatches.length); i++) {
            keyConcepts.push(boldMatches[i].replace(/\*\*/g, ''));
        }
    }
    
    // Construir resumen breve
    if (keyConcepts.length > 0) {
        return `Combina ${keyConcepts.join(', ')} para crear una solución efectiva.`;
    } else {
        // Si no hay conceptos en negrita, usar primera oración resumida
        const words = firstSentence.split(/\s+/).slice(0, 20);
        return words.join(' ') + (firstSentence.split(/\s+/).length > 20 ? '...' : '');
    }
}

function formatMessage(content) {
    // Limpiar repeticiones antes de formatear
    content = removeRepetitions(content);
    
    // Auto-mejorar formato si el modelo no lo hizo
    content = autoImproveFormatting(content);
    
    // Auto-detectar código sin delimitadores y envolverlo
    content = autoFormatCode(content);
    
    // Configurar marked.js con opciones mejoradas para formateo
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            breaks: true,        // Convertir \n en <br>
            gfm: true,          // GitHub Flavored Markdown
            headerIds: false,   // No generar IDs en headers
            mangle: false,      // No mangle emails
            pedantic: false,    // No modo pedantic (permite mejor formateo)
            smartLists: true,   // Mejorar detección de listas
            smartypants: false, // No convertir comillas
            highlight: function(code, lang) {
                if (lang && typeof hljs !== 'undefined' && hljs.getLanguage(lang)) {
                    try {
                        return hljs.highlight(code, { language: lang }).value;
                    } catch (err) {
                        console.error(err);
                    }
                }
                return code;
            }
        });
        
        // Renderizar markdown
        const html = marked.parse(content);
        return html;
    }
    
    // Fallback si marked.js no está disponible
    let formatted = escapeHtml(content);
    formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    formatted = formatted.replace(/\*(.+?)\*/g, '<em>$1</em>');
    formatted = formatted.replace(/```(\w+)?\n([\s\S]+?)```/g, '<pre><code>$2</code></pre>');
    formatted = formatted.replace(/`(.+?)`/g, '<code>$1</code>');
    formatted = formatted.replace(/\n/g, '<br>');
    
    return formatted;
}

function autoFormatCode(content) {
    // Si ya tiene bloques de código, no hacer nada
    if (content.includes('```')) {
        return content;
    }
    
    // Detectar PHP (class, function __construct, $this->, etc.)
    if (content.includes('class ') && (content.includes('$this->') || content.includes('private $') || content.includes('public function'))) {
        // Buscar donde empieza y termina el código PHP
        const classStart = content.indexOf('class ');
        const lastBrace = content.lastIndexOf('}');
        
        if (classStart >= 0 && lastBrace > classStart) {
            const phpCode = content.substring(classStart, lastBrace + 1);
            const before = content.substring(0, classStart);
            const after = content.substring(lastBrace + 1);
            
            content = before + '\n```php\n' + phpCode + '\n```\n' + after;
            return content;
        }
    }
    
    // Detectar HTML (<!DOCTYPE, <html>, etc.)
    const htmlPattern = /<!DOCTYPE[^>]*>[\s\S]*<\/html>/gi;
    if (htmlPattern.test(content)) {
        const match = content.match(htmlPattern);
        if (match) {
            content = content.replace(match[0], '\n```html\n' + match[0] + '\n```\n');
            return content;
        }
    }
    
    // Detectar tags HTML sueltos (múltiples tags)
    const htmlTagPattern = /<[a-z]+[^>]*>[\s\S]*?<\/[a-z]+>/gi;
    const htmlMatches = content.match(htmlTagPattern);
    if (htmlMatches && htmlMatches.length > 2) {
        // Si hay más de 2 tags HTML, probablemente es código HTML
        const firstTag = content.indexOf('<');
        const lastCloseTag = content.lastIndexOf('>');
        if (firstTag >= 0 && lastCloseTag > firstTag) {
            const htmlCode = content.substring(firstTag, lastCloseTag + 1);
            content = content.substring(0, firstTag) + 
                     '\n```html\n' + htmlCode + '\n```\n' + 
                     content.substring(lastCloseTag + 1);
            return content;
        }
    }
    
    // Detectar bloques de código JavaScript
    const functionPattern = /function\s+\w+\s*\([^)]*\)\s*\{[^}]+\}/g;
    if (functionPattern.test(content)) {
        content = content.replace(functionPattern, (match) => {
            return '\n```javascript\n' + match + '\n```\n';
        });
    }
    
    // Detectar Python
    const pythonPattern = /def\s+\w+\s*\([^)]*\):[^\n]+/g;
    if (pythonPattern.test(content)) {
        content = content.replace(pythonPattern, (match) => {
            return '\n```python\n' + match + '\n```\n';
        });
    }
    
    return content;
}

function clearMessages() {
    messagesContainer.innerHTML = '';
}

function saveMessage(role, content) {
    if (!currentChatId) return;
    
    const chat = chats.find(c => c.id === currentChatId);
    if (!chat) return;
    
    chat.messages.push({ role, content, timestamp: new Date().toISOString() });
    chat.updatedAt = new Date().toISOString();
    
    saveChatsToStorage();
}

function copyMessageContent(content) {
    // Limpiar formato HTML antes de copiar
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = content;
    const textContent = tempDiv.textContent || tempDiv.innerText;
    
    navigator.clipboard.writeText(textContent).then(() => {
        // Mostrar feedback visual
        console.log('✅ Contenido copiado al portapapeles');
    }).catch(err => {
        console.error('Error al copiar:', err);
    });
}

async function regenerateResponse() {
    if (!lastUserMessage || isTyping) {
        console.log('No hay mensaje para regenerar o el modelo está generando');
        return;
    }
    
    // Encontrar y eliminar el último mensaje del asistente
    const messages = messagesContainer.querySelectorAll('.message.assistant');
    if (messages.length > 0) {
        const lastAssistantMessage = messages[messages.length - 1];
        lastAssistantMessage.remove();
        
        // Eliminar también del chat guardado
        const chat = chats.find(c => c.id === currentChatId);
        if (chat && chat.messages.length > 0) {
            // Eliminar el último mensaje si es del asistente
            const lastMsg = chat.messages[chat.messages.length - 1];
            if (lastMsg.role === 'assistant') {
                chat.messages.pop();
                saveChatsToStorage();
            }
        }
    }
    
    // Generar nueva respuesta
    await simulateAssistantResponse(lastUserMessage);
}

// ============================================
// Gestión de Estado
// ============================================
function showEmptyState() {
    emptyState.style.display = 'flex';
    chatArea.style.display = 'none';
}

function showChatArea() {
    emptyState.style.display = 'none';
    chatArea.style.display = 'flex';
}

function startNewChatWithPrompt(prompt) {
    createNewChat();
    messageInput.value = prompt;
    handleInputChange();
    messageInput.focus();
}

// ============================================
// Almacenamiento Local
// ============================================
function saveChatsToStorage() {
    try {
        localStorage.setItem('capibara6-chats', JSON.stringify(chats));
    } catch (e) {
        console.error('Error saving chats:', e);
    }
}

function loadChatsFromStorage() {
    try {
        const stored = localStorage.getItem('capibara6-chats');
        if (stored) {
            chats = JSON.parse(stored);
        }
    } catch (e) {
        console.error('Error loading chats:', e);
        chats = [];
    }
}

// ============================================
// Modal de Configuración
// ============================================
function openSettingsModal() {
    const modal = document.getElementById('settings-modal');
    if (modal) {
        modal.style.display = 'flex';
    }
}

function closeSettingsModal() {
    const modal = document.getElementById('settings-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

// Cerrar modal al hacer click fuera
document.addEventListener('click', (e) => {
    const modal = document.getElementById('settings-modal');
    if (modal && e.target === modal) {
        closeSettingsModal();
    }
});

// ============================================
// Manejo de Archivos
// ============================================
function handleFileSelect(event) {
    const files = Array.from(event.target.files);
    
    files.forEach(file => {
        // Validar tamaño (máximo 10MB)
        if (file.size > 10 * 1024 * 1024) {
            alert(`El archivo ${file.name} es demasiado grande. Máximo 10MB.`);
            return;
        }
        
        attachedFiles.push(file);
    });
    
    updateAttachmentsPreview();
    updateSendButtonState();
    
    // Limpiar input para permitir seleccionar el mismo archivo otra vez
    fileInput.value = '';
}

function updateAttachmentsPreview() {
    if (attachedFiles.length === 0) {
        attachmentsPreview.style.display = 'none';
        attachmentsPreview.innerHTML = '';
        return;
    }
    
    attachmentsPreview.style.display = 'flex';
    attachmentsPreview.innerHTML = '';
    
    attachedFiles.forEach((file, index) => {
        const isImage = file.type.startsWith('image/');
        const itemDiv = document.createElement('div');
        itemDiv.className = `attachment-item ${isImage ? 'image' : ''}`;
        
        if (isImage) {
            // Previsualización de imagen
            const reader = new FileReader();
            reader.onload = (e) => {
                itemDiv.innerHTML = `
                    <img src="${e.target.result}" alt="${file.name}" class="attachment-image">
                    <button class="attachment-remove" data-index="${index}">
                        <i data-lucide="x" style="width: 12px; height: 12px;"></i>
                    </button>
                `;
                if (typeof lucide !== 'undefined') {
                    lucide.createIcons();
                }
            };
            reader.readAsDataURL(file);
        } else {
            // Ícono para documentos
            const icon = getFileIcon(file.name);
            itemDiv.innerHTML = `
                <div class="attachment-icon">
                    <i data-lucide="${icon}" style="width: 16px; height: 16px;"></i>
                </div>
                <div class="attachment-info">
                    <div class="attachment-name">${file.name}</div>
                    <div class="attachment-size">${formatFileSize(file.size)}</div>
                </div>
                <button class="attachment-remove" data-index="${index}">
                    <i data-lucide="x" style="width: 12px; height: 12px;"></i>
                </button>
            `;
        }
        
        attachmentsPreview.appendChild(itemDiv);
        
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
    });
    
    // Agregar event listeners para eliminar archivos
    document.querySelectorAll('.attachment-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const index = parseInt(btn.getAttribute('data-index'));
            removeAttachment(index);
        });
    });
}

function removeAttachment(index) {
    attachedFiles.splice(index, 1);
    updateAttachmentsPreview();
    updateSendButtonState();
}

function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const iconMap = {
        'pdf': 'file-text',
        'doc': 'file-text',
        'docx': 'file-text',
        'txt': 'file-text',
        'csv': 'table',
        'xlsx': 'table',
        'xls': 'table',
        'pptx': 'presentation',
        'ppt': 'presentation',
        'zip': 'archive',
        'rar': 'archive',
    };
    return iconMap[ext] || 'file';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// ============================================
// Estado del Servidor
// ============================================
async function checkServerConnection() {
    try {
        updateServerStatus('connecting', 'Verificando...');
        
        const response = await fetch(MODEL_CONFIG.serverUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: '<bos><start_of_turn>user\ntest<end_of_turn>\n<start_of_turn>model\n',
                n_predict: 1
            })
        });
        
        if (response.ok) {
            updateServerStatus('connected', 'Conectado');
            console.log('✅ Servidor conectado');
        } else {
            updateServerStatus('error', 'Error');
            console.error('❌ Error de conexión:', response.status);
        }
    } catch (error) {
        updateServerStatus('error', 'Sin conexión');
        console.error('❌ No se pudo conectar al servidor:', error);
    }
}

function updateServerStatus(status, text) {
    if (!statusLed || !statusText) return;
    
    // Limpiar clases previas
    statusLed.classList.remove('connected', 'error', 'connecting');
    statusText.classList.remove('connected', 'error', 'connecting');
    
    // Agregar nueva clase
    statusLed.classList.add(status);
    statusText.classList.add(status);
    statusText.textContent = text;
}

function showConsensusInfo() {
    const info = `
╔═══════════════════════════════════╗
║   INFORMACIÓN DE CONSENSO         ║
╚═══════════════════════════════════╝

🌐 Modelos Activos: 1
📍 Servidor: 34.175.89.158:8080
🤖 Modelo: Capibara6 (base: Gemma3-12B)

⚙️ CONFIGURACIÓN ACTUAL:
├─ Temperature: 0.6
├─ Max Tokens: 100
├─ Repeat Penalty: 1.3
├─ Top-p: 0.85
└─ Streaming: Activado

💡 SISTEMA DE CONSENSO:
🟡 MODO BETA - Capibara6 Consensus
El sistema actualmente usa un solo modelo.
Para activar consenso entre múltiples modelos,
se pueden configurar servidores adicionales
en el futuro.

📊 CALIDAD DE RESPUESTAS:
Si las respuestas mezclan idiomas o son
incoherentes, considera:
• Reducir temperature
• Aumentar repeat_penalty
• Ajustar el system prompt
• Usar modo de consenso con múltiples modelos
    `.trim();
    
    alert(info);
    console.log('📊 Información de consenso:', {
        models: 1,
        server: MODEL_CONFIG.serverUrl,
        params: MODEL_CONFIG.defaultParams
    });
}

function updateConsensusCount(count) {
    if (consensusCount) {
        consensusCount.textContent = count;
        
        // Cambiar color según número de modelos
        if (count > 1) {
            consensusIndicator.style.borderColor = '#10a37f';
        } else {
            consensusIndicator.style.borderColor = '#3d3d3d';
        }
        
        // Actualizar tooltip con información de beta
        consensusIndicator.title = `🟡 MODO BETA - Capibara6 Consensus\nModelos activos: ${count}\nServidor: 34.175.89.158:8080\nModelo: Capibara6 (base: Gemma3-12B)`;
    }
}

// ============================================
// Utilidades
// ============================================
function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================
// Inicialización cuando el DOM está listo
// ============================================
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        // Verificar autenticación primero (DESHABILITADO por ahora)
        // if (!checkAuthentication()) {
        //     return; // Redirigir a login si no está autenticado
        // }
        
        init();
    });
} else {
    init();
}

// ============================================
// Authentication Functions
// ============================================
function checkAuthentication() {
    const token = localStorage.getItem('auth_token');
    const userData = localStorage.getItem('user_data');
    
    if (!token || !userData) {
        // No está autenticado, redirigir a login
        window.location.href = 'login.html';
        return false;
    }
    
    // Verificar si el token es válido (opcional, para desarrollo)
    try {
        const user = JSON.parse(userData);
        updateUserInterface(user);
        return true;
    } catch (error) {
        console.error('Error parsing user data:', error);
        localStorage.removeItem('auth_token');
        localStorage.removeItem('user_data');
        window.location.href = 'login.html';
        return false;
    }
}

function updateUserInterface(user) {
    // Actualizar la interfaz con información del usuario
    const userInfo = document.getElementById('user-info');
    if (userInfo) {
        userInfo.innerHTML = `
            <img src="${getUserAvatar(user)}" alt="${user.name}" class="user-avatar">
            <div class="user-details">
                <span class="user-name">${user.name}</span>
                <span class="user-provider">${user.provider}</span>
            </div>
        `;
    }
    
    // Agregar botón de logout
    const sidebar = document.querySelector('.sidebar');
    if (sidebar && !document.getElementById('logout-btn')) {
        const logoutBtn = document.createElement('button');
        logoutBtn.id = 'logout-btn';
        logoutBtn.className = 'logout-btn';
        logoutBtn.innerHTML = `
            <i data-lucide="log-out"></i>
            <span>Cerrar Sesión</span>
        `;
        logoutBtn.onclick = logout;
        sidebar.appendChild(logoutBtn);
        lucide.createIcons();
    }
}

function getUserAvatar(user) {
    if (user.provider === 'github') {
        return `https://github.com/${user.name}.png?size=32`;
    } else if (user.provider === 'google') {
        return `https://ui-avatars.com/api/?name=${encodeURIComponent(user.name)}&background=10a37f&color=fff&size=32`;
    } else {
        return `https://ui-avatars.com/api/?name=${encodeURIComponent(user.name)}&background=10a37f&color=fff&size=32`;
    }
}

function logout() {
    if (confirm('¿Estás seguro de que quieres cerrar sesión?')) {
        localStorage.removeItem('auth_token');
        localStorage.removeItem('user_data');
        window.location.href = 'login.html';
    }
}

// Exportar funciones para uso global
window.closeSettingsModal = closeSettingsModal;
window.logout = logout;

// ============================================
// Importar función de cambio de idioma si existe
// ============================================
if (typeof window.capibaraLanguage !== 'undefined') {
    window.changeLanguage = window.capibaraLanguage.switch;
}

