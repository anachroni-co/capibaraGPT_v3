/**
 * Sistema de Evaluación Detallado para Plantillas
 * Permite puntuar cada aspecto de la respuesta
 */

// Criterios de evaluación
const RATING_CRITERIA = {
    coherencia: {
        name: "Coherencia",
        icon: "check-circle",
        description: "¿La respuesta tiene sentido y es lógica?",
        affectedParams: ["temperature", "top_p"]
    },
    longitud: {
        name: "Longitud",
        icon: "align-left",
        description: "¿La longitud es adecuada (ni muy corta ni muy larga)?",
        affectedParams: ["n_predict"]
    },
    formato: {
        name: "Formato",
        icon: "layout",
        description: "¿Usa párrafos, listas, negrita correctamente?",
        affectedParams: ["systemPrompt"]
    },
    precision: {
        name: "Precisión",
        icon: "target",
        description: "¿La información es correcta y precisa?",
        affectedParams: ["temperature", "repeat_penalty"]
    },
    limpieza: {
        name: "Limpieza",
        icon: "sparkles",
        description: "¿Sin basura (LaTeX, HTML, repeticiones)?",
        affectedParams: ["repeat_penalty", "presence_penalty", "frequency_penalty"]
    },
    naturalidad: {
        name: "Naturalidad",
        icon: "message-circle",
        description: "¿Suena natural y conversacional?",
        affectedParams: ["temperature", "top_p", "repeat_penalty"]
    }
};

// Almacenamiento de ratings detallados
window.detailedRatings = JSON.parse(localStorage.getItem('detailedRatings')) || {};

/**
 * Muestra el modal de rating detallado
 */
function showRatingModal(messageElement) {
    const template = window.activeTemplate;
    const templateInfo = window.TEMPLATE_PROFILES[template];
    
    if (!templateInfo) return;
    
    // Crear modal
    const modal = document.createElement('div');
    modal.className = 'rating-modal';
    modal.innerHTML = `
        <div class="rating-modal-content">
            <div class="rating-modal-header">
                <h3>📊 Evaluar Respuesta</h3>
                <p class="rating-template-name">${templateInfo.name}</p>
                <button class="rating-close" onclick="closeRatingModal()">
                    <i data-lucide="x" style="width: 20px; height: 20px;"></i>
                </button>
            </div>
            
            <div class="rating-modal-body">
                <p class="rating-instructions">Puntúa cada aspecto de 1 (malo) a 5 (excelente):</p>
                
                <div class="rating-criteria">
                    ${Object.entries(RATING_CRITERIA).map(([key, criteria]) => `
                        <div class="rating-criterion">
                            <div class="criterion-header">
                                <i data-lucide="${criteria.icon}" style="width: 16px; height: 16px;"></i>
                                <span class="criterion-name">${criteria.name}</span>
                            </div>
                            <p class="criterion-description">${criteria.description}</p>
                            <div class="rating-stars" data-criterion="${key}">
                                ${[1, 2, 3, 4, 5].map(star => `
                                    <button class="star-btn" data-value="${star}" onclick="setRating('${key}', ${star})">
                                        <i data-lucide="star" style="width: 20px; height: 20px;"></i>
                                    </button>
                                `).join('')}
                            </div>
                            <div class="affected-params">
                                <span class="params-label">Afecta:</span>
                                <span class="params-list">${criteria.affectedParams.join(', ')}</span>
                            </div>
                        </div>
                    `).join('')}
                </div>
                
                <div class="rating-notes">
                    <label for="rating-notes-input">Notas adicionales (opcional):</label>
                    <textarea id="rating-notes-input" class="rating-notes-input" 
                              placeholder="Ej: Se repite mucho, genera LaTeX, muy largo..."></textarea>
                </div>
                
                <div class="rating-modal-footer">
                    <button class="btn-rating-cancel" onclick="closeRatingModal()">Cancelar</button>
                    <button class="btn-rating-submit" onclick="submitRating()">Guardar Evaluación</button>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Reinicializar iconos de Lucide
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
    
    // Guardar referencia al mensaje
    window.currentRatingMessage = messageElement;
    window.currentRatingTemplate = template;
    window.currentRatings = {};
}

/**
 * Establece el rating de un criterio
 */
function setRating(criterion, value) {
    window.currentRatings[criterion] = value;
    
    // Actualizar visualmente las estrellas
    const starsContainer = document.querySelector(`.rating-stars[data-criterion="${criterion}"]`);
    if (starsContainer) {
        const buttons = starsContainer.querySelectorAll('.star-btn');
        buttons.forEach((btn, index) => {
            if (index < value) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }
}

/**
 * Cierra el modal de rating
 */
function closeRatingModal() {
    const modal = document.querySelector('.rating-modal');
    if (modal) {
        modal.remove();
    }
    window.currentRatingMessage = null;
    window.currentRatingTemplate = null;
    window.currentRatings = {};
}

/**
 * Guarda la evaluación detallada
 */
function submitRating() {
    const template = window.currentRatingTemplate;
    const ratings = window.currentRatings;
    const notes = document.getElementById('rating-notes-input')?.value || '';
    
    // Verificar que haya al menos 3 criterios evaluados
    const ratedCount = Object.keys(ratings).length;
    if (ratedCount < 3) {
        alert('Por favor, evalúa al menos 3 criterios antes de guardar.');
        return;
    }
    
    // Inicializar storage para esta plantilla si no existe
    if (!window.detailedRatings[template]) {
        window.detailedRatings[template] = {
            templateName: window.TEMPLATE_PROFILES[template].name,
            totalEvaluations: 0,
            averageScores: {},
            allRatings: [],
            recommendations: []
        };
    }
    
    // Guardar esta evaluación
    const evaluation = {
        timestamp: new Date().toISOString(),
        scores: ratings,
        notes: notes,
        totalScore: Object.values(ratings).reduce((a, b) => a + b, 0) / Object.keys(ratings).length
    };
    
    window.detailedRatings[template].allRatings.push(evaluation);
    window.detailedRatings[template].totalEvaluations++;
    
    // Calcular promedios
    Object.keys(RATING_CRITERIA).forEach(criterion => {
        const scores = window.detailedRatings[template].allRatings
            .map(r => r.scores[criterion])
            .filter(s => s !== undefined);
        
        if (scores.length > 0) {
            window.detailedRatings[template].averageScores[criterion] = 
                (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(2);
        }
    });
    
    // Generar recomendaciones automáticas
    generateRecommendations(template);
    
    // Guardar en localStorage
    localStorage.setItem('detailedRatings', JSON.stringify(window.detailedRatings));
    
    console.log('✅ Evaluación guardada:', template);
    console.log('📊 Promedio general:', evaluation.totalScore.toFixed(2));
    
    // Mostrar confirmación visual
    showRatingConfirmation(evaluation.totalScore);
    
    // Cerrar modal
    closeRatingModal();
}

/**
 * Genera recomendaciones automáticas basadas en los ratings
 */
function generateRecommendations(template) {
    const data = window.detailedRatings[template];
    const recommendations = [];
    
    // Analizar cada criterio
    Object.entries(data.averageScores).forEach(([criterion, avgScore]) => {
        const score = parseFloat(avgScore);
        const criteriaInfo = RATING_CRITERIA[criterion];
        
        if (score < 3) {
            // Puntuación baja - necesita mejora
            recommendations.push({
                criterion: criterion,
                score: score,
                action: 'increase',
                params: criteriaInfo.affectedParams,
                message: `⚠️ ${criteriaInfo.name} baja (${score}/5) - Ajustar: ${criteriaInfo.affectedParams.join(', ')}`
            });
        } else if (score >= 4.5) {
            // Puntuación alta - mantener
            recommendations.push({
                criterion: criterion,
                score: score,
                action: 'maintain',
                params: criteriaInfo.affectedParams,
                message: `✅ ${criteriaInfo.name} excelente (${score}/5) - Mantener configuración`
            });
        }
    });
    
    data.recommendations = recommendations;
}

/**
 * Muestra confirmación visual después de evaluar
 */
function showRatingConfirmation(score) {
    const color = score >= 4 ? '#10a37f' : score >= 3 ? '#f59e0b' : '#ef4444';
    const emoji = score >= 4 ? '🎉' : score >= 3 ? '👍' : '⚠️';
    
    const toast = document.createElement('div');
    toast.className = 'rating-toast';
    toast.style.background = color;
    toast.innerHTML = `
        <span>${emoji} Evaluación guardada</span>
        <span>Promedio: ${score.toFixed(1)}/5</span>
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('show');
    }, 10);
    
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 2000);
}

/**
 * Muestra estadísticas detalladas
 */
function showDetailedStats() {
    const statsModal = document.createElement('div');
    statsModal.className = 'stats-modal';
    
    // Ordenar plantillas por promedio general
    const sortedTemplates = Object.entries(window.detailedRatings)
        .map(([key, data]) => {
            const avgTotal = data.allRatings.length > 0
                ? data.allRatings.reduce((sum, r) => sum + r.totalScore, 0) / data.allRatings.length
                : 0;
            return { key, data, avgTotal };
        })
        .sort((a, b) => b.avgTotal - a.avgTotal);
    
    let html = `
        <div class="stats-modal-content">
            <div class="stats-modal-header">
                <h3>📊 Estadísticas Detalladas</h3>
                <button class="stats-close" onclick="closeStatsModal()">
                    <i data-lucide="x" style="width: 20px; height: 20px;"></i>
                </button>
            </div>
            
            <div class="stats-modal-body">
    `;
    
    if (sortedTemplates.length === 0) {
        html += '<p class="no-data">No hay evaluaciones aún. Prueba las plantillas y evalúalas.</p>';
    } else {
        sortedTemplates.forEach(({ key, data, avgTotal }, index) => {
            html += `
                <div class="template-stats-card">
                    <div class="template-stats-header">
                        <span class="rank">#${index + 1}</span>
                        <span class="template-name">${data.templateName}</span>
                        <span class="total-score ${avgTotal >= 4 ? 'good' : avgTotal >= 3 ? 'medium' : 'bad'}">
                            ${avgTotal.toFixed(1)}/5
                        </span>
                    </div>
                    
                    <div class="criteria-scores">
                        ${Object.entries(data.averageScores).map(([criterion, score]) => `
                            <div class="criterion-score">
                                <span class="criterion-label">${RATING_CRITERIA[criterion].name}:</span>
                                <div class="score-bar">
                                    <div class="score-fill" style="width: ${(score / 5) * 100}%"></div>
                                </div>
                                <span class="score-value">${score}/5</span>
                            </div>
                        `).join('')}
                    </div>
                    
                    ${data.recommendations && data.recommendations.length > 0 ? `
                        <div class="recommendations">
                            <h4>💡 Recomendaciones:</h4>
                            ${data.recommendations.map(rec => `
                                <p class="recommendation ${rec.action}">${rec.message}</p>
                            `).join('')}
                        </div>
                    ` : ''}
                    
                    <div class="evaluation-count">
                        Total de evaluaciones: ${data.totalEvaluations}
                    </div>
                </div>
            `;
        });
    }
    
    html += `
                <div class="stats-actions">
                    <button class="btn-export-ratings" onclick="exportRatings()">
                        <i data-lucide="download" style="width: 16px; height: 16px;"></i>
                        Exportar Resultados
                    </button>
                    <button class="btn-clear-ratings" onclick="clearRatings()">
                        <i data-lucide="trash-2" style="width: 16px; height: 16px;"></i>
                        Borrar Todo
                    </button>
                </div>
            </div>
        </div>
    `;
    
    statsModal.innerHTML = html;
    document.body.appendChild(statsModal);
    
    // Reinicializar iconos
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
}

/**
 * Cierra el modal de estadísticas
 */
function closeStatsModal() {
    const modal = document.querySelector('.stats-modal');
    if (modal) {
        modal.remove();
    }
}

/**
 * Exporta los resultados a JSON
 */
function exportRatings() {
    const data = window.detailedRatings;
    const json = JSON.stringify(data, null, 2);
    
    // Crear blob y descargar
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `capibara6-ratings-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    console.log('📥 Ratings exportados');
}

/**
 * Borra todos los ratings
 */
function clearRatings() {
    if (confirm('¿Estás seguro de borrar todas las evaluaciones?')) {
        window.detailedRatings = {};
        localStorage.removeItem('detailedRatings');
        closeStatsModal();
        console.log('🗑️ Ratings borrados');
    }
}

/**
 * Obtiene la mejor plantilla basada en criterio específico
 */
function getBestTemplateFor(criterion) {
    const templates = Object.entries(window.detailedRatings)
        .filter(([key, data]) => data.averageScores[criterion])
        .map(([key, data]) => ({
            key,
            name: data.templateName,
            score: parseFloat(data.averageScores[criterion])
        }))
        .sort((a, b) => b.score - a.score);
    
    return templates[0] || null;
}

/**
 * Genera sugerencias de configuración óptima
 */
function generateOptimalConfig() {
    const suggestions = {};
    
    // Para cada criterio, encontrar la mejor plantilla
    Object.keys(RATING_CRITERIA).forEach(criterion => {
        const best = getBestTemplateFor(criterion);
        if (best) {
            suggestions[criterion] = {
                template: best.name,
                score: best.score,
                params: RATING_CRITERIA[criterion].affectedParams
            };
        }
    });
    
    return suggestions;
}

// Exportar funciones globalmente
window.showRatingModal = showRatingModal;
window.closeRatingModal = closeRatingModal;
window.setRating = setRating;
window.submitRating = submitRating;
window.showDetailedStats = showDetailedStats;
window.closeStatsModal = closeStatsModal;
window.exportRatings = exportRatings;
window.clearRatings = clearRatings;
window.getBestTemplateFor = getBestTemplateFor;
window.generateOptimalConfig = generateOptimalConfig;

console.log('📊 Sistema de evaluación detallado cargado');
console.log('✅ Funciones exportadas:', {
    showRatingModal: typeof window.showRatingModal,
    showDetailedStats: typeof window.showDetailedStats
});

