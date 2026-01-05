/**
 * Smart MCP Integration v2.0
 * Basado en principios reales de MCP y RAG selectivo
 * Solo agrega contexto cuando es REALMENTE necesario
 */

const SMART_MCP_CONFIG = {
    // Usar el servidor integrado en la VM - HTTPS en producción
    serverUrl: 'http://localhost:5003/api/mcp/augment',  // Endpoint correcto MCP
    healthUrl: 'http://localhost:5003/api/mcp/health',   // Health check MCP
    enabled: false,  // ⚠️ DESHABILITADO por defecto - activar solo si MCP server está corriendo
    timeout: 2000, // 2 segundos máximo
    fallbackOnError: true
};

/**
 * Analiza la consulta y solo agrega contexto si es necesario
 */
async function smartMCPAnalyze(userQuery) {
    // Si está deshabilitado, devolver la query original
    if (!SMART_MCP_CONFIG.enabled) {
        return {
            needsContext: false,
            prompt: userQuery,
            lightweight: true
        };
    }

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), SMART_MCP_CONFIG.timeout);

        const response = await fetch(SMART_MCP_CONFIG.serverUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: userQuery }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`MCP server error: ${response.status}`);
        }

        const data = await response.json();
        
        // Log para debugging (solo si hay contexto agregado)
        if (data.needs_context) {
            console.log('🎯 Smart MCP: Contexto agregado', {
                contextsAdded: data.contexts_added,
                original: userQuery,
                augmented: data.augmented_prompt.substring(0, 100) + '...'
            });
        }

        return {
            needsContext: data.needs_context,
            prompt: data.augmented_prompt,
            contextsAdded: data.contexts_added,
            lightweight: data.lightweight
        };

    } catch (error) {
        if (error.name === 'AbortError') {
            console.warn('⏱️ Smart MCP timeout, usando query original');
        } else {
            console.error('❌ Smart MCP error:', error.message);
        }

        // Fallback: devolver query original
        if (SMART_MCP_CONFIG.fallbackOnError) {
            return {
                needsContext: false,
                prompt: userQuery,
                lightweight: true,
                error: error.message
            };
        }

        throw error;
    }
}

/**
 * Verifica el estado del servidor MCP
 */
async function checkSmartMCPHealth() {
    try {
        // En localhost: conectar directo a puerto 5003 (MCP Server)
        // En producción: usar proxy de Vercel que conecta a la VM
        const healthUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
            ? SMART_MCP_CONFIG.healthUrl
            : '/api/mcp-health';

        console.log(`🔍 Verificando Smart MCP en: ${healthUrl}`);
        
        const response = await fetch(healthUrl, {
            method: 'GET',
            signal: AbortSignal.timeout(3000)  // 3 segundos de timeout
        });

        console.log(`📡 Respuesta MCP: status=${response.status}, ok=${response.ok}`);

        if (response.ok) {
            const data = await response.json();
            console.log('📦 Datos MCP:', data);
            
            // Verificar que la respuesta tenga datos válidos
            if (data && (data.status === 'healthy' || data.service || data.approach)) {
                console.log('✅ Smart MCP ACTIVO:', data.service || data.approach || 'healthy');
                return true;
            }
        }
        
        console.log('⚠️ Smart MCP respondió pero con formato inesperado');
        return false;
    } catch (error) {
        console.log('ℹ️ Smart MCP no disponible (se usará modo directo)');
        console.log('🔍 Error:', error.message);
        return false;
    }
}

/**
 * Toggle para habilitar/deshabilitar MCP desde la UI
 */
function toggleSmartMCP(enabled) {
    SMART_MCP_CONFIG.enabled = enabled;
    console.log(`🔄 Smart MCP ${enabled ? 'activado' : 'desactivado'}`);
    
    // Actualizar indicador en UI si existe
    const mcpIndicator = document.querySelector('.mcp-indicator');
    if (mcpIndicator) {
        mcpIndicator.classList.toggle('active', enabled);
        mcpIndicator.title = enabled 
            ? 'Smart MCP: Activo (contexto selectivo)' 
            : 'Smart MCP: Desactivado';
    }
}

// Exportar funciones globalmente
window.smartMCPAnalyze = smartMCPAnalyze;
window.checkSmartMCPHealth = checkSmartMCPHealth;
window.toggleSmartMCP = toggleSmartMCP;

// Verificar estado al cargar
document.addEventListener('DOMContentLoaded', () => {
    checkSmartMCPHealth().then(isActive => {
        SMART_MCP_CONFIG.enabled = isActive;
        
        // Actualizar indicador visual
        const mcpIndicator = document.querySelector('.mcp-indicator');
        if (mcpIndicator) {
            mcpIndicator.classList.toggle('active', isActive);
            mcpIndicator.title = isActive 
                ? 'Smart MCP v2.0: Activo (RAG selectivo)' 
                : 'Smart MCP: No disponible';
        }
    });
});

console.log('🚀 Smart MCP Integration v2.0 cargado');
console.log('📊 Enfoque: Selective RAG - Solo contexto cuando es necesario');

