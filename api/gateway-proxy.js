/**
 * Vercel Serverless Function - Gateway Proxy (Simplificado)
 * Proxy HTTPS para el servicio Gateway en VM services
 * Versión simplificada para evitar problemas de CORS
 */

export default async function handler(req, res) {
    // Asegurar CORS headers en todos los casos
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With');

    // Si es solicitud OPTIONS (preflight), responder inmediatamente
    if (req.method === 'OPTIONS') {
        res.status(200).setHeader('Content-Length', '0').end();
        return;
    }

    // Solo permitir POST
    if (req.method !== 'POST') {
        return res.status(405).json({ 
            error: 'Method not allowed',
            allowed: 'POST'
        });
    }

    try {
        // Datos requeridos
        const { message, prompt, messages } = req.body;
        if (!message && !prompt && (!messages || messages.length === 0)) {
            return res.status(400).json({ 
                error: 'Message, prompt, or messages are required' 
            });
        }

        // URL base del gateway en la VM services
        const GATEWAY_BASE_URL = process.env.GATEWAY_URL || 'http://34.175.255.139:8080';
        const GATEWAY_CHAT_URL = `${GATEWAY_BASE_URL}/api/chat`;
        
        console.log(`📡 Conectando al gateway: ${GATEWAY_CHAT_URL}`);
        
        // Hacer solicitud al gateway con timeout
        const response = await fetch(GATEWAY_CHAT_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message || prompt || (messages && messages[messages.length - 1]?.content),
                model: req.body.model || 'phi4_fast',
                use_semantic_router: true,
                temperature: req.body.temperature || 0.7,
                max_tokens: req.body.max_tokens || 200,
                // Incluir contexto si está disponible
                ...(messages ? { messages } : {})
            }),
            signal: AbortSignal.timeout(35000) // 35 segundos
        });

        console.log(`📡 Gateway respondió con status: ${response.status}`);

        // Obtener respuesta
        const responseText = await response.text();
        
        // Devolver respuesta con el mismo status
        res.status(response.status);
        
        if (responseText) {
            try {
                const jsonData = JSON.parse(responseText);
                res.json(jsonData);
            } catch (e) {
                // Si no es JSON válido, devolver como texto
                res.setHeader('Content-Type', 'application/json');
                res.send(JSON.stringify({ response: responseText }));
            }
        } else {
            res.json({ status: 'ok', message: 'Empty response from gateway' });
        }

    } catch (error) {
        console.error('❌ Error en gateway proxy:', error);
        
        // Asegurar headers CORS incluso en error
        res.setHeader('Access-Control-Allow-Origin', '*');
        res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
        res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With');

        if (error.name === 'AbortError') {
            res.status(408).json({
                error: 'Gateway timeout',
                details: 'La conexión con el servicio de IA tomó demasiado tiempo'
            });
        } else {
            res.status(503).json({
                error: 'Servicio no disponible',
                details: error.message
            });
        }
    }
}

export const config = {
    api: {
        bodyParser: {
            sizeLimit: '10mb',
        },
    },
};