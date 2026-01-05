// Proxy para conectar con la VM models-europe desde Vercel
export default async function handler(req, res) {
    // Configurar CORS más restrictivo
    const allowedOrigin = process.env.NODE_ENV === 'production'
        ? 'https://www.capibara6.com'
        : 'http://localhost:3000';

    res.setHeader('Access-Control-Allow-Origin', allowedOrigin);
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

    console.log('📡 Entrando en handler de /api/chat');

    if (req.method === 'OPTIONS') {
        console.log('🔄 Manejando preflight OPTIONS');
        return res.status(200).end();
    }

    if (req.method !== 'POST') {
        console.log('❌ Método no permitido:', req.method);
        return res.status(405).json({ error: 'Método no permitido' });
    }

    try {
        const { message, model, temperature, max_tokens, use_semantic_router } = req.body;
        console.log('📥 Datos recibidos:', { message: message ? message.substring(0, 50) + '...' : 'vacio', model, temperature, max_tokens, use_semantic_router });

        // Preparar payload para conectar con VM models-europe
        // El gateway server espera un campo 'message' único como string simple
        const userMessage = message || (req.body.messages && req.body.messages[0]?.content) || '';

        // Usar modelo más rápido por defecto para evitar timeouts con aya_expanse_multilingual
        const defaultModel = 'phi4_fast';  // Modelo más rápido como fallback

        const payload = {
            message: userMessage, // Campo requerido por el gateway server
            model: model || defaultModel,  // Cambiamos a un modelo más rápido por defecto
            temperature: temperature || 0.7,
            max_tokens: max_tokens || 200,
            use_semantic_router: use_semantic_router || false
        };

        console.log('📤 Enviando solicitud al gateway server:', 'http://34.175.255.139:8080/api/chat');
        console.log('📋 Payload:', { model: payload.model, content_len: userMessage.length, temp: payload.temperature });

        // Conectar al gateway server en VM services (IP externa) que enruta a la VM models-europe
        // Aumentar timeout para manejar modelos grandes
        // Primero intentamos con el modelo solicitado, pero usamos uno más rápido como fallback
        const targetModel = payload.model !== 'aya_expanse_multilingual' ? payload.model : 'phi4_fast';

        const attemptPayload = {
            ...payload,
            model: targetModel
        };

        console.log(`🔄 Usando modelo fallback: ${targetModel} en lugar de aya_expanse_multilingual`);

        const response = await fetch('http://34.175.255.139:8080/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(attemptPayload),
            signal: AbortSignal.timeout(90000) // 90 segundos para modelos grandes
        });

        console.log('📥 Recibiendo respuesta del gateway server:', response.status);

        const data = await response.json();
        console.log('📋 Datos recibidos:', { status_in_response: data.status, has_choices: !!data.choices, model: data.model });

        // Verificar si el backend ya devolvió una respuesta simulada
        if (data.status && (data.status.includes('simulated') || data.status.includes('fallback'))) {
            console.log('🔄 Gateway devolvió respuesta simulada, retornando tal cual');
            return res.status(200).json(data);
        }

        console.log('✅ Retornando respuesta exitosa:', response.status);
        return res.status(response.status).json(data);

    } catch (error) {
        console.error('❌ Error en proxy a VM services gateway:', error.message);
        console.error('📝 Detalles del error:', {
            name: error.name,
            message: error.message,
            stack: error.stack?.substring(0, 200)
        });

        // En lugar de devolver respuesta simulada, dejar que sea el gateway server quien la maneje
        // Si llegamos aquí es porque no pudimos conectar al gateway server en absoluto
        return res.status(503).json({
            error: 'Service unavailable',
            message: `No se puede conectar al gateway server: ${error.message}`,
            timestamp: new Date().toISOString()
        });
    }
}
