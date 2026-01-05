/**
 * Función de prueba para verificar el manejo de CORS
 */

export default async function handler(req, res) {
    // Siempre aplicar headers CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

    // Responder a OPTIONS inmediatamente
    if (req.method === 'OPTIONS') {
        res.status(200).end();
        return;
    }

    // Responder a otras solicitudes
    res.status(200).json({ 
        status: 'ok', 
        message: 'CORS test endpoint working',
        received: req.method,
        timestamp: new Date().toISOString()
    });
}