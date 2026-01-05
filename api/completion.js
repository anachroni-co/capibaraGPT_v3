/**
 * Vercel Serverless Function
 * Proxy HTTPS para el modelo Gemma en la VM
 * Resuelve el problema de Mixed Content
 */

export default async function handler(req, res) {
    // Solo permitir POST
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    // CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    // Handle preflight
    if (req.method === 'OPTIONS') {
        return res.status(200).end();
    }

    try {
        // URL del gateway server en VM services - IP externa
        const CHAT_URL = 'http://34.175.255.139:8080/api/chat';
        
        // Preparar el payload para el gateway server
        const payload = {
            message: req.body.prompt || req.body.message || req.body.text || '',
            model: req.body.model || 'aya_expanse_multilingual',
            temperature: req.body.temperature || 0.7,
            max_tokens: req.body.max_tokens || req.body.n_predict || 200,
            use_semantic_router: true
        };

        // Reenviar la petición al gateway server
        const response = await fetch(CHAT_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        // Si la VM no está disponible, usar fallback
        if (!response.ok) {
            console.log('⚠️ Gateway server no disponible, usando fallback...');
            return res.status(200).json({
                content: "Lo siento, el modelo de IA no está disponible en este momento. Por favor, intenta más tarde o contacta al administrador.",
                stop: true
            });
        }

        // Recibir respuesta del gateway server
        const data = await response.json();

        // Formatear la respuesta para que sea compatible con el frontend
        if (data.choices && data.choices[0]) {
            // Formato OpenAI
            res.status(200).json({
                content: data.choices[0].message?.content || data.response || data.content,
                model: data.model,
                finish_reason: data.choices[0].finish_reason || 'stop'
            });
        } else {
            // Formato alternativo
            res.status(200).json({
                content: data.response || data.content || (data.choices && data.choices[0]?.message?.content),
                model: data.model,
                finish_reason: 'stop'
            });
        }

    } catch (error) {
        console.error('Proxy error:', error);
        
        // Si es error de conexión, usar fallback en lugar de error 500
        if (error.message.includes('fetch failed') || error.message.includes('ECONNREFUSED')) {
            console.log('⚠️ Error de conexión, usando fallback...');
            return res.status(200).json({
                content: "Lo siento, el modelo de IA no está disponible en este momento. Por favor, intenta más tarde o contacta al administrador.",
                stop: true
            });
        }
        
        res.status(500).json({ 
            error: 'Error al conectar con el modelo',
            details: error.message 
        });
    }
}

