/**
 * Vercel Serverless Function - MCP Service Proxy Mejorado
 * Proxy HTTPS para el servicio MCP (Model Context Protocol) en VM services con mejoras
 */

// Simple cache en memoria para respuestas (limitado por la naturaleza serverless de Vercel)
const RESPONSE_CACHE = new Map();
const CACHE_TTL = 30000; // 30 segundos en ms para health checks y búsquedas comunes

function generateCacheKey(url, method, body) {
    return `${method}-${url}-${body ? JSON.stringify(body) : ''}`;
}

function cleanupCache() {
    const now = Date.now();
    for (const [key, { timestamp }] of RESPONSE_CACHE.entries()) {
        if (now - timestamp > CACHE_TTL) {
            RESPONSE_CACHE.delete(key);
        }
    }
}

export default async function handler(req, res) {
    // CORS headers (se aplican siempre)
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With');

    // Handle preflight (OPTIONS) immediately
    if (req.method === 'OPTIONS') {
        return res.status(200).end();
    }

    try {
        // URL base de MCP en la VM services
        const MCP_BASE_URL = process.env.MCP_URL || 'http://34.175.255.139:5003';
        
        // Obtener el path de la solicitud y construir URL completa
        const { path = '' } = req.query;
        const mcpPath = Array.isArray(path) ? path.join('/') : path;
        const fullMcpPath = mcpPath.startsWith('/') ? mcpPath : `/${mcpPath}`;
        const mcpUrl = `${MCP_BASE_URL}${fullMcpPath}`;
        
        // Para health checks, intentar usar cache
        const isHealthCheck = fullMcpPath.includes('/health') || fullMcpPath.includes('/status');
        if (isHealthCheck) {
            const cacheKey = generateCacheKey(mcpUrl, req.method, req.body);
            cleanupCache();
            
            const cachedResponse = RESPONSE_CACHE.get(cacheKey);
            if (cachedResponse && Date.now() - cachedResponse.timestamp < CACHE_TTL) {
                return res.status(cachedResponse.status).json(cachedResponse.data);
            }
        }
        
        // Preparar opciones para fetch con timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 20000); // 20 segundos timeout para MCP
        
        const fetchOptions = {
            method: req.method,
            headers: {
                'Content-Type': 'application/json',
                ...req.headers,
                // Asegurar que no se pasen encabezados problemáticos
                'host': undefined,
                'content-length': undefined,
                'connection': undefined,
            },
            signal: controller.signal
        };
        
        // Solo incluir body si hay datos y el método lo permite
        if ((req.method === 'POST' || req.method === 'PUT' || req.method === 'PATCH') && req.body) {
            fetchOptions.body = JSON.stringify(req.body);
        }
        
        try {
            // Hacer la solicitud a MCP
            const mcpResponse = await fetch(mcpUrl, fetchOptions);
            clearTimeout(timeoutId);
            
            // Obtener el cuerpo de la respuesta
            const responseText = await mcpResponse.text();
            
            // Para health checks, almacenar en cache
            if (isHealthCheck) {
                const cacheKey = generateCacheKey(mcpUrl, req.method, req.body);
                RESPONSE_CACHE.set(cacheKey, {
                    status: mcpResponse.status,
                    data: responseText ? JSON.parse(responseText) : null,
                    timestamp: Date.now()
                });
            }
            
            // Devolver la respuesta con el mismo status code
            res.status(mcpResponse.status);
            
            try {
                // Si la respuesta es JSON válida, devolver como JSON
                const jsonData = responseText ? JSON.parse(responseText) : {};
                res.json(jsonData);
            } catch (e) {
                // Si no es JSON, devolver como texto
                res.setHeader('Content-Type', 'text/plain');
                res.send(responseText);
            }
            
        } catch (fetchError) {
            clearTimeout(timeoutId);
            if (fetchError.name === 'AbortError') {
                console.error('❌ Timeout en solicitud a MCP:', mcpUrl);
                res.status(408).json({
                    error: 'Tiempo de espera agotado al conectar con MCP',
                    path: mcpPath
                });
            } else {
                console.error('❌ Error en solicitud a MCP:', fetchError.message);
                res.status(503).json({
                    error: 'Error al conectar con el servicio MCP',
                    details: fetchError.message,
                    path: mcpPath
                });
            }
        }
        
    } catch (error) {
        console.error('❌ Error general en MCP proxy:', error);
        res.status(500).json({
            error: 'Error interno en el proxy de MCP',
            details: error.message
        });
    }
}

export const config = {
    api: {
        bodyParser: {
            sizeLimit: '5mb',
        },
    },
};