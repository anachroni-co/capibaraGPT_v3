/**
 * TTS Integration for Capibara6
 * Solo voz Sofia - Sin selector de voces
 * Prioridad 1: Coqui TTS (VITS neural, alta calidad español)
 * Fallback: Web Speech API del navegador
 */

const TTS_CONFIG = {
    enabled: true,
    // Deshabilitar Coqui TTS temporalmente (usar solo Web Speech API)
    useCoquiTTS: false,
    apiEndpoint: 'http://localhost:5001/api/tts/speak',  // Servidor local
    language: 'es',
    rate: 1.0,      // Velocidad (0.5 - 2.0)
    pitch: 1.0,     // Tono (0 - 2)
    volume: 1.0,    // Volumen (0 - 1)
    voice: 'sofia'  // Solo voz Sofia
};

// Estado global
let currentUtterance = null;
let isSpeaking = false;
let currentSpeakingButton = null;

/**
 * Obtiene la mejor voz en español disponible (solo para fallback Web Speech API)
 */
function getBestSpanishVoice() {
    const voices = window.speechSynthesis.getVoices();
    
    if (voices.length === 0) {
        console.warn('⚠️ No hay voces disponibles aún');
        return null;
    }
    
    // Buscar cualquier voz en español
    const spanishVoice = voices.find(v => v.lang.startsWith('es'));
    if (spanishVoice) {
        console.log(`✓ Voz español encontrada: ${spanishVoice.name} (${spanishVoice.lang})`);
        return spanishVoice;
    }
    
    // Último recurso: primera voz disponible
    console.warn(`⚠️ No se encontró voz en español. Usando: ${voices[0]?.name}`);
    return voices[0];
}

/**
 * Lee el texto usando Coqui TTS (si está disponible) o Web Speech API
 */
async function speakText(text, button) {
    // Si ya está hablando, detener
    if (isSpeaking) {
        stopSpeaking();
        return;
    }
    
    // Limpiar texto antes de leer
    let cleanText = text
        .replace(/```[\s\S]*?```/g, '')           // Eliminar bloques de código
        .replace(/`[^`]+`/g, '')                  // Eliminar código inline
        .replace(/In \[\d*\]:/g, '')              // Eliminar prompts de notebook
        .replace(/Out\[\d*\]:/g, '')              // Eliminar outputs de notebook
        .replace(/\*\*/g, '')                     // Eliminar negritas markdown
        .replace(/__/g, '')                       // Eliminar cursivas
        .replace(/[#\-\*\[\]]/g, '')              // Eliminar markdown y corchetes
        .replace(/https?:\/\/[^\s]+/g, '')       // Eliminar URLs
        .replace(/\n{3,}/g, '\n\n')               // Normalizar saltos
        .replace(/[^\w\s.,;:!?¿¡áéíóúñÁÉÍÓÚÑüÜ()\-]/g, ' ') // Solo caracteres válidos
        .replace(/\s+/g, ' ')                     // Normalizar espacios
        .trim();
    
    // Dividir en oraciones y tomar solo las primeras 2-3
    const sentences = cleanText.split(/[.!?]+/).filter(s => s.trim().length > 0);
    if (sentences.length > 0) {
        // Tomar máximo 2 oraciones completas
        cleanText = sentences.slice(0, 2).join('. ') + '.';
    }
    
    // Límite estricto de caracteres
    cleanText = cleanText.substring(0, 300);  // ✅ Solo 300 caracteres (Web Speech API es limitado)
    
    if (!cleanText) {
        console.warn('⚠️ No hay texto para leer');
        return;
    }
    
    console.log(`🎙️ Texto limpio para TTS (${cleanText.length} chars): "${cleanText.substring(0, 100)}..."`);
    
    // Intentar usar Coqui TTS primero (en producción)
    if (TTS_CONFIG.useCoquiTTS) {
        try {
            await speakWithCoquiTTS(cleanText, button);
            return;
        } catch (error) {
            console.warn('⚠️ Coqui TTS no disponible, usando Web Speech API:', error);
            // Continuar con fallback
        }
    }
    
    // Fallback: Web Speech API
    speakWithWebAPI(cleanText, button);
}

/**
 * Síntesis con Coqui TTS (XTTS v2) - Solo voz Sofia
 */
async function speakWithCoquiTTS(text, button) {
    isSpeaking = true;
    currentSpeakingButton = button;
    updateButtonState(button, 'speaking');
    
    // Siempre usar la voz de Sofía
    const voiceId = TTS_CONFIG.voice;
    
    console.log(`🎤 Usando voz: ${voiceId}`);
    
    try {
        const response = await fetch(TTS_CONFIG.apiEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                text,
                language: TTS_CONFIG.language,
                voice_id: voiceId
            })
        });
        
        if (!response.ok) {
            throw new Error('Coqui TTS API error');
        }
        
        // El servidor devuelve un archivo WAV binario directamente
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        
        audio.onplay = () => {
            console.log(`🔊 Coqui TTS reproduciendo... (audio WAV generado)`);
        };
        
        audio.onended = () => {
            isSpeaking = false;
            updateButtonState(button, 'idle');
            currentSpeakingButton = null;
            URL.revokeObjectURL(audioUrl); // Limpiar el objeto URL
            console.log('✅ Coqui TTS completado');
        };
        
        audio.onerror = (error) => {
            console.error('❌ Error reproduciendo audio:', error);
            isSpeaking = false;
            updateButtonState(button, 'idle');
            currentSpeakingButton = null;
            URL.revokeObjectURL(audioUrl); // Limpiar el objeto URL
        };
        
        await audio.play();
        
    } catch (error) {
        isSpeaking = false;
        updateButtonState(button, 'idle');
        currentSpeakingButton = null;
        throw error; // Propagar para fallback
    }
}

/**
 * Síntesis con Web Speech API (fallback)
 */
function speakWithWebAPI(text, button, retryCount = 0) {
    // Si el texto es muy largo y es un retry, acortar más
    if (retryCount > 0 && text.length > 200) {
        text = text.substring(0, 200);
        console.log(`🔄 Retry ${retryCount}: texto acortado a 200 chars`);
    }
    
    // Asegurar que las voces estén cargadas
    const voices = window.speechSynthesis.getVoices();
    if (voices.length === 0) {
        console.warn('⚠️ Voces aún no cargadas. Esperando...');
        
        // Esperar a que las voces se carguen
        setTimeout(() => {
            speakWithWebAPI(text, button, retryCount);
        }, 100);
        return;
    }
    
    // Crear utterance
    currentUtterance = new SpeechSynthesisUtterance(text);
    
    // Configurar voz
    const voice = getBestSpanishVoice();
    if (voice) {
        currentUtterance.voice = voice;
        currentUtterance.lang = voice.lang;
        console.log(`🎤 Usando voz: ${voice.name}`);
    } else {
        currentUtterance.lang = 'es-ES';
        console.warn('⚠️ No se encontró voz en español, usando por defecto');
    }
    
    // Configurar parámetros
    currentUtterance.rate = Math.max(0.5, Math.min(2.0, TTS_CONFIG.rate)); // Limitar rango
    currentUtterance.pitch = Math.max(0, Math.min(2, TTS_CONFIG.pitch)); // Limitar rango
    currentUtterance.volume = Math.max(0, Math.min(1, TTS_CONFIG.volume)); // Limitar rango
    
    // Event handlers
    currentUtterance.onstart = () => {
        isSpeaking = true;
        currentSpeakingButton = button;
        updateButtonState(button, 'speaking');
        console.log('🔊 Web Speech API iniciado');
    };
    
    currentUtterance.onend = () => {
        isSpeaking = false;
        updateButtonState(button, 'idle');
        currentSpeakingButton = null;
        currentUtterance = null;
        console.log('✅ Web Speech API completado');
    };
    
    currentUtterance.onerror = (event) => {
        const errorType = event.error || 'unknown';
        console.error('❌ Error TTS:', errorType);
        
        // Intentar detener cualquier speech en curso
        if (window.speechSynthesis.speaking) {
            window.speechSynthesis.cancel();
        }
        
        isSpeaking = false;
        updateButtonState(button, 'idle');
        currentSpeakingButton = null;
        currentUtterance = null;
        
        // Si el error es 'unknown' o 'undefined', es probable que el texto sea muy largo
        // Reintentar con texto más corto
        if ((errorType === 'unknown' || errorType === 'undefined' || errorType === 'synthesis-failed') && retryCount < 1) {
            console.warn(`⚠️ Error de síntesis (${errorType}). Intentando con texto más corto...`);
            
            // Acortar el texto a solo la primera oración
            const shortText = text.split(/[.!?]/)[0] + '.';
            
            if (shortText.length > 5 && shortText.length < 200) { // Rango válido
                setTimeout(() => {
                    speakWithWebAPI(shortText, button, retryCount + 1);
                }, 1000); // Esperar más tiempo
            } else {
                console.warn('⚠️ Texto no adecuado para síntesis.');
            }
        } else {
            console.warn('⚠️ No se pudo sintetizar el texto después de varios intentos.');
            console.log('💡 Web Speech API tiene limitaciones. TTS deshabilitado para este mensaje.');
        }
    };
    
    // Cancelar cualquier speech anterior (importante!)
    if (window.speechSynthesis.speaking) {
        console.log('🛑 Cancelando speech anterior...');
        window.speechSynthesis.cancel();
    }
    
    // Pequeño delay para asegurar que todo esté listo
    setTimeout(() => {
        if (!currentUtterance) return;
        
        try {
            console.log(`🔊 Iniciando síntesis: "${text.substring(0, 50)}..." (${text.length} chars)`);
            window.speechSynthesis.speak(currentUtterance);
        } catch (error) {
            console.error('❌ Error al llamar speak():', error);
            isSpeaking = false;
            updateButtonState(button, 'idle');
            currentSpeakingButton = null;
            currentUtterance = null;
        }
    }, 150);  // Aumentado a 150ms para mejor compatibilidad
}

/**
 * Detiene la lectura
 */
function stopSpeaking() {
    if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
    }
    
    if (currentSpeakingButton) {
        updateButtonState(currentSpeakingButton, 'idle');
    }
    
    isSpeaking = false;
    currentSpeakingButton = null;
    currentUtterance = null;
    console.log('⏹️ TTS detenido');
}

/**
 * Actualiza el estado visual del botón
 */
function updateButtonState(button, state) {
    // Validar que el botón existe
    if (!button) {
        console.warn('⚠️ updateButtonState: botón no encontrado');
        return;
    }

    const icon = button.querySelector('i');
    const text = button.querySelector('.btn-text');
    
    if (state === 'speaking') {
        if (icon) icon.setAttribute('data-lucide', 'volume-2');
        if (text) text.textContent = 'Detener';
        button.classList.add('speaking');
    } else {
        if (icon) icon.setAttribute('data-lucide', 'volume');
        if (text) text.textContent = 'Escuchar';
        button.classList.remove('speaking');
    }
    
    // Reinicializar iconos de Lucide
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
}

/**
 * Inicializar TTS
 */
function initTTS() {
    // Cargar voces (pueden tardar en cargar)
    if (!window.speechSynthesis) {
        console.warn('⚠️ Web Speech API no disponible en este navegador');
        return;
    }
    
    // Forzar carga de voces
    let voicesLoaded = false;
    
    function loadVoices() {
        const voices = window.speechSynthesis.getVoices();
        
        if (voices.length > 0 && !voicesLoaded) {
            voicesLoaded = true;
            
            const spanishVoices = voices.filter(v => v.lang.startsWith('es'));
            console.log('🔊 Voces disponibles:', voices.length);
            console.log('🇪🇸 Voces en español:', spanishVoices.length);
            
            if (spanishVoices.length > 0) {
                spanishVoices.forEach(v => {
                    console.log(`  - ${v.name} (${v.lang}) ${v.localService ? '[Local]' : '[Remote]'}`);
                });
            }
            
            const selectedVoice = getBestSpanishVoice();
            if (selectedVoice) {
                console.log(`🎯 Voz seleccionada: ${selectedVoice.name} (${selectedVoice.lang})`);
            } else {
                console.warn('⚠️ No se pudo seleccionar voz');
            }
        }
    }
    
    // Cargar voces inmediatamente
    loadVoices();
    
    // También escuchar el evento por si las voces cargan después
    if (window.speechSynthesis.onvoiceschanged !== undefined) {
        window.speechSynthesis.onvoiceschanged = loadVoices;
    }
    
    // Forzar carga en algunos navegadores (Chrome)
    setTimeout(loadVoices, 100);
    setTimeout(loadVoices, 500);
}

// Exportar funciones globalmente
window.speakText = speakText;
window.stopSpeaking = stopSpeaking;
window.initTTS = initTTS;

// Auto-inicializar
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTTS);
} else {
    initTTS();
}

console.log('🔊 TTS Integration cargado');
console.log('✅ Funciones disponibles: speakText, stopSpeaking');

