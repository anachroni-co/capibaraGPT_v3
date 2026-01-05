// chatbot.js - Funcionalidad para el chatbot de Capibara6

let chatMessages = [];

document.addEventListener('DOMContentLoaded', function() {
    const sendButton = document.getElementById('send-button');
    const messageInput = document.getElementById('message-input');
    const chatMessagesDiv = document.getElementById('chat-messages');
    
    if (sendButton && messageInput && chatMessagesDiv) {
        // Enviar mensaje al hacer clic en el botón
        sendButton.addEventListener('click', sendMessage);
        
        // Enviar mensaje al presionar Enter
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }
    
    // Cargar mensajes iniciales
    loadInitialMessages();
});

function loadInitialMessages() {
    const chatMessagesDiv = document.getElementById('chat-messages');
    if (chatMessagesDiv) {
        chatMessagesDiv.innerHTML = '<div class="message bot">¡Hola! Soy Capibara6, tu asistente de IA. ¿En qué puedo ayudarte hoy?</div>';
    }
}

async function sendMessage() {
    const messageInput = document.getElementById('message-input');
    const chatMessagesDiv = document.getElementById('chat-messages');
    
    if (!messageInput || !chatMessagesDiv) return;
    
    const message = messageInput.value.trim();
    if (!message) return;
    
    // Añadir mensaje del usuario al chat
    addMessageToChat(message, 'user');
    messageInput.value = '';
    
    try {
        // Mostrar indicador de "escribiendo..."
        const thinkingElement = document.createElement('div');
        thinkingElement.className = 'message bot';
        thinkingElement.id = 'thinking-indicator';
        thinkingElement.textContent = 'Pensando...';
        chatMessagesDiv.appendChild(thinkingElement);
        chatMessagesDiv.scrollTop = chatMessagesDiv.scrollHeight;
        
        // Hacer la solicitud al backend
        const response = await makeApiRequest('chat', { prompt: message });
        
        // Remover indicador de pensamiento
        const indicator = document.getElementById('thinking-indicator');
        if (indicator) indicator.remove();
        
        // Añadir la respuesta del bot
        if (response && response.response) {
            addMessageToChat(response.response, 'bot');
        } else {
            addMessageToChat('Lo siento, no pude procesar tu solicitud en este momento.', 'bot');
        }
    } catch (error) {
        console.error('Error al enviar mensaje:', error);
        
        // Remover indicador de pensamiento
        const indicator = document.getElementById('thinking-indicator');
        if (indicator) indicator.remove();
        
        addMessageToChat('Lo siento, ocurrió un error al procesar tu solicitud.', 'bot');
    }
}

function addMessageToChat(message, sender) {
    const chatMessagesDiv = document.getElementById('chat-messages');
    if (!chatMessagesDiv) return;
    
    const messageElement = document.createElement('div');
    messageElement.className = `message ${sender}`;
    messageElement.textContent = message;
    
    chatMessagesDiv.appendChild(messageElement);
    chatMessagesDiv.scrollTop = chatMessagesDiv.scrollHeight;
}