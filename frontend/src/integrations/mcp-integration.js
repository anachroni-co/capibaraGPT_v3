// ============================================
// MCP INTEGRATION - CAPIBARA6
// Model Context Protocol para reducir alucinaciones
// ============================================

const MCP_CONFIG = {
    serverUrl: 'http://localhost:5003/api/mcp',
    enabled: false,  // DESACTIVADO temporalmente para debugging
    autoAugment: true,  // Aumentar automáticamente con contexto
    defaultContexts: ['company_info', 'current_date']
};

// Estado del MCP
let mcpAvailable = false;
let availableContexts = [];
let availableTools = [];

// ============================================
// FUNCIONES DE MCP
// ============================================

async function initMCP() {
    try {
        // Verificar salud del servidor MCP
        const healthResponse = await fetch(`${MCP_CONFIG.serverUrl}/health`);
        
        if (healthResponse.ok) {
            mcpAvailable = true;
            console.log('✅ MCP Server disponible');
            
            // Cargar contextos disponibles
            await loadMCPContexts();
            
            // Cargar herramientas disponibles
            await loadMCPTools();
        }
    } catch (error) {
        console.warn('⚠️ MCP Server no disponible:', error);
        mcpAvailable = false;
    }
}

async function loadMCPContexts() {
    try {
        const response = await fetch(`${MCP_CONFIG.serverUrl}/contexts`);
        if (response.ok) {
            const data = await response.json();
            availableContexts = data.contexts || [];
            console.log('📋 Contextos MCP cargados:', availableContexts.length);
        }
    } catch (error) {
        console.warn('Error cargando contextos MCP:', error);
    }
}

async function loadMCPTools() {
    try {
        const response = await fetch(`${MCP_CONFIG.serverUrl}/tools`);
        if (response.ok) {
            const data = await response.json();
            availableTools = data.tools || [];
            console.log('🛠️ Herramientas MCP cargadas:', availableTools.length);
        }
    } catch (error) {
        console.warn('Error cargando herramientas MCP:', error);
    }
}

async function augmentPromptWithContext(prompt, contexts = null) {
    if (!mcpAvailable || !MCP_CONFIG.enabled) {
        return prompt;
    }
    
    try {
        const response = await fetch(`${MCP_CONFIG.serverUrl}/augment`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prompt: prompt,
                contexts: contexts || MCP_CONFIG.defaultContexts
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log('📊 Prompt aumentado con contexto MCP');
            return data.augmented_prompt;
        }
    } catch (error) {
        console.warn('Error aumentando prompt con MCP:', error);
    }
    
    return prompt;
}

async function getContext(contextId) {
    if (!mcpAvailable) {
        return null;
    }
    
    try {
        const response = await fetch(`${MCP_CONFIG.serverUrl}/context/${contextId}`);
        if (response.ok) {
            return await response.json();
        }
    } catch (error) {
        console.warn(`Error obteniendo contexto ${contextId}:`, error);
    }
    
    return null;
}

async function calculateWithMCP(expression) {
    if (!mcpAvailable) {
        return null;
    }
    
    try {
        const response = await fetch(`${MCP_CONFIG.serverUrl}/calculate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                expression: expression
            })
        });
        
        if (response.ok) {
            return await response.json();
        }
    } catch (error) {
        console.warn('Error en cálculo MCP:', error);
    }
    
    return null;
}

async function verifyFactWithMCP(claim) {
    if (!mcpAvailable) {
        return null;
    }
    
    try {
        const response = await fetch(`${MCP_CONFIG.serverUrl}/verify`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                claim: claim
            })
        });
        
        if (response.ok) {
            return await response.json();
        }
    } catch (error) {
        console.warn('Error verificando hecho:', error);
    }
    
    return null;
}

// ============================================
// DETECCIÓN AUTOMÁTICA DE NECESIDAD DE CONTEXTO
// ============================================

function detectContextNeeds(prompt) {
    const contexts = [];
    const promptLower = prompt.toLowerCase();
    
    // Detectar preguntas sobre la empresa/producto
    if (promptLower.includes('capibara') || 
        promptLower.includes('anachroni') ||
        promptLower.includes('quién eres') ||
        promptLower.includes('qué eres') ||
        promptLower.includes('quién te creó')) {
        contexts.push('company_info');
    }
    
    // Detectar preguntas técnicas sobre el sistema
    if (promptLower.includes('modelo') ||
        promptLower.includes('parámetros') ||
        promptLower.includes('especificaciones') ||
        promptLower.includes('hardware')) {
        contexts.push('technical_specs');
    }
    
    // Detectar preguntas sobre fecha/tiempo
    if (promptLower.includes('qué día') ||
        promptLower.includes('qué fecha') ||
        promptLower.includes('hoy') ||
        promptLower.includes('actual')) {
        contexts.push('current_date');
    }
    
    return contexts.length > 0 ? contexts : MCP_CONFIG.defaultContexts;
}

function detectCalculation(prompt) {
    // Detectar si el prompt contiene una expresión matemática
    const mathPatterns = [
        /\d+\s*[\+\-\*\/]\s*\d+/,
        /cuánto es/i,
        /calcula/i,
        /resultado de/i
    ];
    
    return mathPatterns.some(pattern => pattern.test(prompt));
}

// ============================================
// INTERFAZ DE USUARIO
// ============================================

function createMCPIndicator() {
    const indicator = document.createElement('div');
    indicator.id = 'mcp-indicator';
    indicator.className = 'mcp-indicator';
    indicator.innerHTML = `
        <div class="mcp-status ${mcpAvailable ? 'active' : 'inactive'}">
            <i data-lucide="shield-check"></i>
            <span>MCP ${mcpAvailable ? 'Activo' : 'Inactivo'}</span>
        </div>
    `;
    
    indicator.title = mcpAvailable 
        ? `MCP Activo - Reducción de alucinaciones\n${availableContexts.length} contextos disponibles\n${availableTools.filter(t => t.enabled).length} herramientas activas`
        : 'MCP Inactivo';
    
    return indicator;
}

function updateMCPStatus() {
    const serverStatus = document.getElementById('server-status');
    if (serverStatus && mcpAvailable) {
        const mcpBadge = document.createElement('span');
        mcpBadge.className = 'mcp-badge';
        mcpBadge.innerHTML = '<i data-lucide="shield-check"></i> MCP';
        mcpBadge.title = 'Model Context Protocol activo - Reduciendo alucinaciones';
        serverStatus.appendChild(mcpBadge);
        
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
    }
}

// ============================================
// INTEGRACIÓN CON EL CHAT
// ============================================

// Hook para aumentar prompts antes de enviarlos al modelo
window.mcpAugmentPrompt = async function(prompt) {
    if (!MCP_CONFIG.enabled || !mcpAvailable) {
        return prompt;
    }
    
    let augmentedPrompt = prompt;
    
    // Detectar si necesita cálculo
    if (detectCalculation(prompt)) {
        console.log('🧮 Pregunta matemática detectada');
        
        // Convertir texto a expresión matemática
        let expression = prompt.toLowerCase()
            .replace(/multiplicado\s+por|multiplicar\s+por|por/gi, '*')
            .replace(/dividido\s+por|dividir\s+por|entre/gi, '/')
            .replace(/más|suma/gi, '+')
            .replace(/menos|resta/gi, '-');
        
        // Extraer la expresión numérica
        const match = expression.match(/(\d+\s*[\+\-\*\/]\s*\d+)/);
        if (match) {
            const result = await calculateWithMCP(match[1]);
            if (result && !result.error) {
                augmentedPrompt = `${prompt}\n\n[DATO VERIFICADO: ${match[1]} = ${result.result}. Usa este resultado exacto en tu respuesta.]`;
                console.log(`🧮 Cálculo verificado: ${match[1]} = ${result.result}`);
            }
        }
    }
    
    // Detectar contextos necesarios
    const neededContexts = detectContextNeeds(prompt);
    
    // Aumentar con contexto si es necesario
    if (MCP_CONFIG.autoAugment && neededContexts.length > 0) {
        augmentedPrompt = await augmentPromptWithContext(augmentedPrompt, neededContexts);
    }
    
    return augmentedPrompt;
};

// Inicializar MCP cuando el DOM esté listo
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        initMCP();
        setTimeout(updateMCPStatus, 1000);
    });
} else {
    initMCP();
    setTimeout(updateMCPStatus, 1000);
}

// Exportar funciones
window.mcpConfig = MCP_CONFIG;
window.mcpGetContext = getContext;
window.mcpCalculate = calculateWithMCP;
window.mcpVerifyFact = verifyFactWithMCP;
window.mcpAugmentPrompt = augmentPromptWithContext;
