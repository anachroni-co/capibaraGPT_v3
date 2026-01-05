/**
 * Vercel Serverless Function - N8N Proxy Mejorado
 * Proxy HTTPS para el servicio N8N en VM services con manejo avanzado de errores
 */

// Simple cache en memoria para respuestas (limitado por la naturaleza serverless de Vercel)
const RESPONSE_CACHE = new Map();
const CACHE_TTL = 60000; // 1 minuto en ms para health checks

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
        // URL base de N8N en la VM services
        const N8N_BASE_URL = process.env.N8N_URL || 'http://34.175.255.139:5678';
        
        // Obtener el path de la solicitud y construir URL completa
        const { path = '' } = req.query;
        const n8nPath = Array.isArray(path) ? path.join('/') : path;
        const fullN8NPath = n8nPath.startsWith('/') ? n8nPath : `/${n8nPath}`;
        const n8nUrl = `${N8N_BASE_URL}${fullN8NPath}`;
        
        // Para health checks, intentar usar cache
        const isHealthCheck = fullN8NPath.includes('/health');
        if (isHealthCheck) {
            const cacheKey = generateCacheKey(n8nUrl, req.method, req.body);
            cleanupCache();
            
            const cachedResponse = RESPONSE_CACHE.get(cacheKey);
            if (cachedResponse && Date.now() - cachedResponse.timestamp < CACHE_TTL) {
                return res.status(cachedResponse.status).json(cachedResponse.data);
            }
        }
        
        // Preparar opciones para fetch con timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 segundos timeout
        
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
            // Hacer la solicitud a N8N
            const n8nResponse = await fetch(n8nUrl, fetchOptions);
            clearTimeout(timeoutId);
            
            // Obtener el cuerpo de la respuesta
            const responseText = await n8nResponse.text();
            
            // Para health checks, almacenar en cache
            if (isHealthCheck) {
                const cacheKey = generateCacheKey(n8nUrl, req.method, req.body);
                RESPONSE_CACHE.set(cacheKey, {
                    status: n8nResponse.status,
                    data: responseText ? JSON.parse(responseText) : null,
                    timestamp: Date.now()
                });
            }
            
            // Devolver la respuesta con el mismo status code
            res.status(n8nResponse.status);
            
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
                console.error('❌ Timeout en solicitud a N8N:', n8nUrl);
                res.status(408).json({
                    error: 'Tiempo de espera agotado al conectar con N8N',
                    path: n8nPath
                });
            } else {
                console.error('❌ Error en solicitud a N8N:', fetchError.message);
                res.status(503).json({
                    error: 'Error al conectar con el servicio N8N',
                    details: fetchError.message,
                    path: n8nPath
                });
            }
        }
        
    } catch (error) {
        console.error('❌ Error general en N8N proxy:', error);
        res.status(500).json({
            error: 'Error interno en el proxy de N8N',
            details: error.message
        });
    }
}

export const config = {
    api: {
        bodyParser: {
            sizeLimit: '10mb',
        },
    },
};