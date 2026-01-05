/**
 * Plantillas de Configuración para Capibara6
 * 10 perfiles diferentes para probar y encontrar el mejor
 */

const TEMPLATE_PROFILES = {
    conservador: {
        name: "🛡️ Conservador",
        description: "Muy coherente y predecible, evita riesgos",
        systemPrompt: "Eres un asistente útil y preciso.",
        params: {
            n_predict: 150,
            temperature: 0.5,
            top_p: 0.75,
            repeat_penalty: 1.4,
            presence_penalty: 0.2,
            frequency_penalty: 0.2
        }
    },
    
    balanceado: {
        name: "⚖️ Balanceado",
        description: "Equilibrio entre coherencia y creatividad",
        systemPrompt: "Eres un asistente útil.",
        params: {
            n_predict: 200,
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.5,
            presence_penalty: 0.3,
            frequency_penalty: 0.3
        }
    },
    
    creativo: {
        name: "🎨 Creativo",
        description: "Más diversidad y originalidad en respuestas",
        systemPrompt: "Eres un asistente útil y creativo.",
        params: {
            n_predict: 250,
            temperature: 0.85,
            top_p: 0.95,
            repeat_penalty: 1.3,
            presence_penalty: 0.4,
            frequency_penalty: 0.4
        }
    },
    
    preciso: {
        name: "🎯 Preciso",
        description: "Respuestas cortas, directas y exactas",
        systemPrompt: "Responde de forma breve y precisa.",
        params: {
            n_predict: 100,
            temperature: 0.4,
            top_p: 0.7,
            repeat_penalty: 1.6,
            presence_penalty: 0.1,
            frequency_penalty: 0.1
        }
    },
    
    conversacional: {
        name: "💬 Conversacional",
        description: "Natural y amigable, como una conversación",
        systemPrompt: "Eres un asistente amigable y conversacional.",
        params: {
            n_predict: 180,
            temperature: 0.75,
            top_p: 0.92,
            repeat_penalty: 1.4,
            presence_penalty: 0.25,
            frequency_penalty: 0.25
        }
    },
    
    tecnico: {
        name: "🔧 Técnico",
        description: "Para explicaciones técnicas detalladas",
        systemPrompt: "Eres un asistente técnico experto.",
        params: {
            n_predict: 300,
            temperature: 0.6,
            top_p: 0.88,
            repeat_penalty: 1.5,
            presence_penalty: 0.3,
            frequency_penalty: 0.3
        }
    },
    
    conciso: {
        name: "⚡ Conciso",
        description: "Máxima brevedad, mínimo de palabras",
        systemPrompt: "Responde en máximo 2 oraciones.",
        params: {
            n_predict: 80,
            temperature: 0.55,
            top_p: 0.8,
            repeat_penalty: 1.7,
            presence_penalty: 0.15,
            frequency_penalty: 0.15
        }
    },
    
    detallado: {
        name: "📚 Detallado",
        description: "Explicaciones completas y exhaustivas",
        systemPrompt: "Proporciona explicaciones detalladas y completas.",
        params: {
            n_predict: 400,
            temperature: 0.65,
            top_p: 0.9,
            repeat_penalty: 1.4,
            presence_penalty: 0.35,
            frequency_penalty: 0.35
        }
    },
    
    experimental: {
        name: "🧪 Experimental",
        description: "Configuración única y diferente",
        systemPrompt: "Eres un asistente útil.",
        params: {
            n_predict: 220,
            temperature: 0.8,
            top_p: 0.85,
            repeat_penalty: 1.2,
            presence_penalty: 0.5,
            frequency_penalty: 0.2
        }
    },
    
    gemma_optimizado: {
        name: "✨ Capibara6 Optimizado",
        description: "Basado en pruebas reales con Capibara6",
        systemPrompt: "Eres un asistente útil.",
        params: {
            n_predict: 200,
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.5,
            presence_penalty: 0.3,
            frequency_penalty: 0.3
        }
    }
};

// Exportar para uso global
window.TEMPLATE_PROFILES = TEMPLATE_PROFILES;

// Plantilla activa (por defecto: balanceado)
window.activeTemplate = 'balanceado';

// Ratings de plantillas (guardados en localStorage)
window.templateRatings = JSON.parse(localStorage.getItem('templateRatings')) || {};

/**
 * Aplica una plantilla a MODEL_CONFIG
 */
function applyTemplate(templateKey) {
    if (!TEMPLATE_PROFILES[templateKey]) {
        console.error('❌ Plantilla no encontrada:', templateKey);
        return false;
    }
    
    const template = TEMPLATE_PROFILES[templateKey];
    
    // Actualizar MODEL_CONFIG
    MODEL_CONFIG.systemPrompt = template.systemPrompt;
    Object.assign(MODEL_CONFIG.defaultParams, template.params);
    
    // Actualizar plantilla activa
    window.activeTemplate = templateKey;
    
    // Guardar en localStorage
    localStorage.setItem('activeTemplate', templateKey);
    
    console.log('✅ Plantilla aplicada:', template.name);
    console.log('📊 Parámetros:', template.params);
    
    return true;
}

/**
 * Incrementa el rating de una plantilla cuando se da "Me gusta"
 */
function rateTemplate(templateKey, positive = true) {
    if (!window.templateRatings[templateKey]) {
        window.templateRatings[templateKey] = { likes: 0, dislikes: 0 };
    }
    
    if (positive) {
        window.templateRatings[templateKey].likes++;
        console.log('👍 +1 Like para:', TEMPLATE_PROFILES[templateKey].name);
    } else {
        window.templateRatings[templateKey].dislikes++;
        console.log('👎 +1 Dislike para:', TEMPLATE_PROFILES[templateKey].name);
    }
    
    // Guardar en localStorage
    localStorage.setItem('templateRatings', JSON.stringify(window.templateRatings));
    
    // Actualizar UI si existe
    updateTemplateStats();
}

/**
 * Obtiene las mejores plantillas por rating
 */
function getTopTemplates(limit = 3) {
    const ratings = window.templateRatings;
    const sorted = Object.keys(ratings)
        .map(key => ({
            key,
            name: TEMPLATE_PROFILES[key].name,
            likes: ratings[key].likes,
            dislikes: ratings[key].dislikes,
            score: ratings[key].likes - ratings[key].dislikes
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, limit);
    
    return sorted;
}

/**
 * Actualiza las estadísticas de plantillas en la UI
 */
function updateTemplateStats() {
    const statsContainer = document.getElementById('template-stats');
    if (!statsContainer) return;
    
    const top = getTopTemplates(3);
    
    let html = '<h4>🏆 Mejores Plantillas:</h4><div class="top-templates">';
    top.forEach((t, i) => {
        html += `
            <div class="top-template">
                <span class="rank">#${i + 1}</span>
                <span class="name">${t.name}</span>
                <span class="score">👍 ${t.likes} 👎 ${t.dislikes}</span>
            </div>
        `;
    });
    html += '</div>';
    
    statsContainer.innerHTML = html;
}

// Exportar funciones globalmente
window.applyTemplate = applyTemplate;
window.rateTemplate = rateTemplate;
window.getTopTemplates = getTopTemplates;
window.updateTemplateStats = updateTemplateStats;

// Cargar plantilla guardada al inicio
document.addEventListener('DOMContentLoaded', () => {
    const savedTemplate = localStorage.getItem('activeTemplate') || 'balanceado';
    applyTemplate(savedTemplate);
    console.log('📋 Sistema de plantillas cargado');
    console.log('🎯 Plantilla activa:', savedTemplate);
});

