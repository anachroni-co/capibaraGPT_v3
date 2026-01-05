// ============================================
// MONITOR DE ENTROPÍA - CAPIBARA6
// ============================================

/**
 * Calcula la entropía estimada de una respuesta
 * La entropía mide la "sorpresa" o aleatoriedad en las predicciones
 * 
 * @param {string} text - Texto de la respuesta
 * @param {number} temperature - Temperatura usada en la generación
 * @returns {number} - Valor de entropía estimado (0-4+)
 */
function calculateEntropy(text, temperature = 0.6) {
    if (!text || text.length === 0) return 0;
    
    // Normalizar texto
    const cleanText = text.toLowerCase().trim();
    const words = cleanText.split(/\s+/).filter(w => w.length > 2);
    
    if (words.length === 0) return 0;
    if (words.length < 5) return 0.8; // Respuestas muy cortas = baja entropía
    
    // 1. Diversidad léxica (Type-Token Ratio) - más importante
    const uniqueWords = new Set(words);
    const lexicalDiversity = uniqueWords.size / words.length;
    
    // 2. Repetición de palabras (penaliza repeticiones)
    const wordCounts = {};
    words.forEach(w => {
        wordCounts[w] = (wordCounts[w] || 0) + 1;
    });
    
    const repeatedWords = Object.values(wordCounts).filter(count => count > 1).length;
    const repetitionRatio = repeatedWords / uniqueWords.size;
    
    // 3. Coherencia de bigramas
    const bigrams = new Set();
    for (let i = 0; i < words.length - 1; i++) {
        bigrams.add(words[i] + ' ' + words[i+1]);
    }
    const bigramDiversity = bigrams.size / Math.max(1, words.length - 1);
    
    // 4. Calcular entropía base (0-2.5)
    let entropy = 0;
    
    // Diversidad léxica: 0.7-1.0 es normal (contribuye 0.8-1.2)
    entropy += (lexicalDiversity - 0.6) * 2.0;
    
    // Diversidad de bigramas: contribuye 0.3-0.6
    entropy += bigramDiversity * 0.6;
    
    // Penalización por repetición: reduce 0-0.5
    entropy -= repetitionRatio * 0.5;
    
    // 5. Ajuste por temperatura (factor 0.9-1.1)
    const tempAdjustment = 0.9 + (temperature - 0.5) * 0.4;
    entropy *= tempAdjustment;
    
    // 6. Normalizar a rango 0.6 - 2.5
    // La mayoría de respuestas coherentes estarán entre 1.0-1.8
    entropy = Math.max(0.6, Math.min(2.5, entropy));
    
    return entropy;
}

/**
 * Obtiene la clase CSS según el nivel de entropía
 * @param {number} entropy - Valor de entropía
 * @returns {string} - Clase CSS
 */
function getEntropyClass(entropy) {
    if (entropy < 1.0) return 'entropy-low';      // Muy predecible (0.6-1.0)
    if (entropy < 1.4) return 'entropy-normal';   // Normal (1.0-1.4)
    if (entropy < 1.8) return 'entropy-medium';   // Creativo (1.4-1.8)
    return 'entropy-high';                         // Muy creativo (1.8-2.5)
}

/**
 * Obtiene descripción legible de la entropía
 * @param {number} entropy - Valor de entropía
 * @returns {string} - Descripción
 */
function getEntropyDescription(entropy) {
    if (entropy < 1.0) return 'Muy predecible';
    if (entropy < 1.4) return 'Balanceado';
    if (entropy < 1.8) return 'Creativo';
    return 'Muy creativo';
}

/**
 * Obtiene el emoji según el nivel de entropía
 * @param {number} entropy - Valor de entropía
 * @returns {string} - Emoji
 */
function getEntropyEmoji(entropy) {
    if (entropy < 1.0) return '🎯'; // Muy predecible
    if (entropy < 1.4) return '📊'; // Balanceado
    if (entropy < 1.8) return '🎨'; // Creativo
    return '🌟';                     // Muy creativo
}

/**
 * Calcula entropía desde la respuesta del servidor (si disponible)
 * @param {object} serverResponse - Respuesta del servidor
 * @returns {number|null} - Entropía del servidor o null si no disponible
 */
function getServerEntropy(serverResponse) {
    // Verificar si el servidor devuelve entropía directamente
    if (serverResponse && serverResponse.entropy !== undefined) {
        return serverResponse.entropy;
    }
    
    // Verificar en timings o stats
    if (serverResponse && serverResponse.timings && serverResponse.timings.entropy !== undefined) {
        return serverResponse.timings.entropy;
    }
    
    return null;
}

/**
 * Crea el elemento HTML para mostrar la entropía
 * @param {number} entropy - Valor de entropía
 * @returns {string} - HTML del elemento
 */
function createEntropyHTML(entropy) {
    const entropyClass = getEntropyClass(entropy);
    const description = getEntropyDescription(entropy);
    
    return `
        <span class="stat-item stat-entropy ${entropyClass}" 
              title="Entropía: ${entropy.toFixed(2)} - ${description}
              
Mide la diversidad léxica y creatividad de la respuesta:
• 0.6-1.0: Muy predecible (respuestas directas y consistentes)
• 1.0-1.4: Balanceado (respuestas normales)
• 1.4-1.8: Creativo (respuestas variadas y originales)
• 1.8-2.5: Muy creativo (máxima diversidad)">
            <i data-lucide="activity"></i>
            ${entropy.toFixed(2)} H
        </span>
    `;
}

// Exportar funciones para uso global
if (typeof window !== 'undefined') {
    window.calculateEntropy = calculateEntropy;
    window.getEntropyClass = getEntropyClass;
    window.getEntropyDescription = getEntropyDescription;
    window.getEntropyEmoji = getEntropyEmoji;
    window.getServerEntropy = getServerEntropy;
    window.createEntropyHTML = createEntropyHTML;
}
