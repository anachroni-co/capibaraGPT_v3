// ============================================
// AUTO-INJECT ENTROPY TO MESSAGES
// ============================================
// Este script automáticamente agrega entropía a los mensajes cuando se crean

(function() {
    'use strict';
    
    console.log('🎯 Monitor de entropía cargado');
    
    // Esperar a que todo esté listo (después de chat-app.js)
    setTimeout(initEntropyInjection, 500);
    
    function initEntropyInjection() {
        console.log('🎯 Inicializando monitor de entropía...');
        
        // Buscar contenedor de mensajes con múltiples selectores
        const messagesContainer = document.getElementById('messages') || 
                                 document.querySelector('.messages') ||
                                 document.querySelector('#messages-container');
        
        if (!messagesContainer) {
            console.warn('⚠️ Contenedor de mensajes no encontrado, reintentando...');
            setTimeout(initEntropyInjection, 1000);
            return;
        }
        
        console.log('✅ Contenedor de mensajes encontrado');
        
        // Crear observer para detectar cambios en los mensajes
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                // Detectar nodos agregados (mensajes nuevos)
                mutation.addedNodes.forEach(function(node) {
                    if (node.nodeType === 1 && node.classList) {
                        if (node.classList.contains('assistant-message') || 
                            node.classList.contains('message')) {
                            // Esperar a que termine el streaming
                            waitForMessageComplete(node);
                        }
                    }
                });
                
                // Detectar cambios en atributos (cuando se remueve 'streaming')
                if (mutation.type === 'attributes' && 
                    mutation.attributeName === 'class' &&
                    mutation.target.classList.contains('assistant-message')) {
                    
                    // Si ya no tiene la clase 'streaming', procesar
                    if (!mutation.target.classList.contains('streaming')) {
                        console.log('✓ Mensaje terminó streaming, procesando...');
                        processMessage(mutation.target);
                    }
                }
            });
        });
        
        // Observar el contenedor
        observer.observe(messagesContainer, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['class']
        });
        
        // Procesar mensajes existentes
        document.querySelectorAll('.assistant-message, .message').forEach(processMessage);
        
        console.log('✅ Monitor de entropía activado');
    }
    
    function waitForMessageComplete(messageDiv) {
        // Esperar a que el mensaje deje de tener la clase 'streaming'
        const checkInterval = setInterval(() => {
            if (!messageDiv.classList.contains('streaming')) {
                clearInterval(checkInterval);
                console.log('✓ Mensaje completado, procesando entropía...');
                processMessage(messageDiv);
            }
        }, 100);
        
        // Timeout de seguridad (10 segundos)
        setTimeout(() => {
            clearInterval(checkInterval);
            console.log('⚠️ Timeout esperando fin de streaming');
        }, 10000);
    }
    
    function processMessage(messageDiv) {
        // Esperar un poco más para asegurar que el mensaje está completamente renderizado
        setTimeout(() => {
            try {
                console.log('🔍 Procesando mensaje...', messageDiv);
                
                // Buscar stats con múltiples selectores
                let statsDiv = messageDiv.querySelector('.message-stats');
                
                // Si no encuentra, buscar en todo el documento los stats más recientes
                if (!statsDiv) {
                    const allStats = document.querySelectorAll('.message-stats');
                    statsDiv = allStats[allStats.length - 1]; // El más reciente
                }
                
                if (!statsDiv) {
                    console.log('⚠️ No se encontró .message-stats');
                    return;
                }
                
                console.log('✓ Stats encontrados:', statsDiv);
                
                // Verificar si ya tiene entropía
                if (statsDiv.querySelector('.stat-entropy')) {
                    console.log('✓ Mensaje ya tiene entropía');
                    return;
                }
                
                // Buscar el texto del mensaje
                let textDiv = messageDiv.querySelector('.message-text');
                
                if (!textDiv) {
                    // Buscar en el documento completo el mensaje más reciente
                    const allTexts = document.querySelectorAll('.message-text');
                    textDiv = allTexts[allTexts.length - 1];
                }
                
                if (!textDiv) {
                    console.log('⚠️ No se encontró .message-text');
                    return;
                }
                
                // Obtener el texto del mensaje
                const text = textDiv.innerText || textDiv.textContent || '';
                if (!text || text.trim().length === 0) {
                    console.log('⚠️ Mensaje sin texto');
                    return;
                }
                
                console.log('✓ Texto encontrado:', text.substring(0, 50) + '...');
                
                // Calcular entropía
                const temperature = 0.6;
                const entropy = typeof calculateEntropy === 'function' 
                    ? calculateEntropy(text, temperature) 
                    : estimateEntropy(text);
                
                console.log(`📊 Entropía calculada: ${entropy.toFixed(2)} H`);
                
                // Crear HTML de entropía
                const entropyHTML = typeof createEntropyHTML === 'function'
                    ? createEntropyHTML(entropy)
                    : createBasicEntropyHTML(entropy);
                
                console.log('✓ HTML creado:', entropyHTML.substring(0, 100));
                
                // Insertar antes del último stat-item (el del modelo)
                const lastStat = statsDiv.querySelector('.stat-item:last-child');
                if (lastStat) {
                    lastStat.insertAdjacentHTML('beforebegin', entropyHTML);
                    console.log('✅ Entropía agregada antes del último stat');
                } else {
                    statsDiv.insertAdjacentHTML('beforeend', entropyHTML);
                    console.log('✅ Entropía agregada al final');
                }
                
                // Reinicializar iconos de Lucide
                if (typeof lucide !== 'undefined' && typeof lucide.createIcons === 'function') {
                    lucide.createIcons();
                    console.log('✓ Iconos reinicializados');
                }
                
                console.log('✅ Proceso completado');
                
            } catch (error) {
                console.error('❌ Error procesando mensaje:', error);
            }
        }, 500); // Aumentar delay para dar tiempo a que se renderice
    }
    
    // Función de estimación básica si no está disponible la principal
    function estimateEntropy(text) {
        const words = text.split(/\s+/).filter(w => w.length > 2);
        if (words.length === 0) return 0.8;
        
        const uniqueWords = new Set(words);
        const diversity = uniqueWords.size / words.length;
        
        // Estimación simple: diversidad léxica normalizada
        return Math.max(0.6, Math.min(2.5, diversity * 2.0));
    }
    
    // Función básica de creación de HTML si no está disponible la principal
    function createBasicEntropyHTML(entropy) {
        const classes = ['entropy-low', 'entropy-normal', 'entropy-medium', 'entropy-high'];
        const colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444'];
        const descriptions = ['Muy predecible', 'Balanceado', 'Creativo', 'Muy creativo'];
        
        let index = 0;
        if (entropy >= 1.0) index = 1;
        if (entropy >= 1.4) index = 2;
        if (entropy >= 1.8) index = 3;
        
        const entropyClass = classes[index];
        const description = descriptions[index];
        
        return `
            <span class="stat-item stat-entropy ${entropyClass}" 
                  title="Entropía: ${entropy.toFixed(2)} - ${description}
                  
Mide la diversidad léxica y creatividad:
• 0.6-1.0: Muy predecible
• 1.0-1.4: Balanceado
• 1.4-1.8: Creativo
• 1.8-2.5: Muy creativo"
                  style="color: ${colors[index]};">
                <i data-lucide="activity"></i>
                ${entropy.toFixed(2)} H
            </span>
        `;
    }
})();

