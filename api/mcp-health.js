/**
 * Vercel Serverless Function - MCP Health Check
 * Verifica si Smart MCP está disponible en la VM
 */

export default async function handler(req, res) {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  try {
    // URL del Smart MCP en la VM (endpoint de health)
    // Obtener URL base sin el /analyze
    const SMART_MCP_BASE_URL = process.env.SMART_MCP_URL
      ? process.env.SMART_MCP_URL.replace('/analyze', '')
      : 'http://34.175.255.139:5010';
    
    const MCP_URL = `${SMART_MCP_BASE_URL}/health`;

    console.log(`🔍 Health check MCP: ${MCP_URL}`);

    // Check health de la VM
    const response = await fetch(MCP_URL, {
      method: 'GET',
      headers: {
        'Accept': 'application/json'
      },
      signal: AbortSignal.timeout(3000)
    });

    if (!response.ok) {
      throw new Error(`MCP responded with status ${response.status}`);
    }

    const data = await response.json();
    console.log('✅ MCP disponible');
    
    return res.status(200).json(data);

  } catch (error) {
    console.error('❌ MCP no disponible:', error.message);

    // Devolver respuesta que indica que MCP no está disponible
    return res.status(503).json({
      service: 'smart-mcp',
      status: 'unavailable',
      error: error.message
    });
  }
}

