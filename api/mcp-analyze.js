/**
 * Vercel Serverless Function
 * Proxy HTTPS para Smart MCP en la VM
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
        // URL del Smart MCP en la VM services - IP externa
        const MCP_URL = process.env.SMART_MCP_URL || 'http://34.175.255.139:5010/analyze';
        
        // Reenviar la petición a la VM
        const response = await fetch(MCP_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(req.body),
            // Timeout de 2 segundos
            signal: AbortSignal.timeout(2000)
        });

        const data = await response.json();
        res.status(response.status).json(data);

    } catch (error) {
        console.error('MCP Proxy error:', error);
        
        // Si falla, devolver query original (fallback)
        res.status(200).json({
            needs_context: false,
            original_query: req.body?.query || '',
            augmented_prompt: req.body?.query || '',
            contexts_added: 0,
            lightweight: true,
            error: 'MCP no disponible (usando fallback)'
        });
    }
}

