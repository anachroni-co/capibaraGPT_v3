/**
 * Vercel Serverless Function - MCP Status
 * Alias para mcp-health.js para mantener compatibilidad con frontend
 *
 * Ruta: /api/mcp/status
 *
 * Actualizado: 2025-12-01
 */

export default async function handler(req, res) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  console.log('📡 Entrando en MCP status handler:', req.method, req.url);

  if (req.method === 'OPTIONS') {
    console.log('🔄 Manejando preflight OPTIONS para MCP');
    return res.status(200).end();
  }

  if (req.method !== 'GET') {
    console.log('❌ Método no permitido para MCP status:', req.method);
    return res.status(405).json({ error: 'Método no permitido para MCP status' });
  }

  try {
    // Intentar puerto 5003 primero (MCP API principal)
    const MCP_PRIMARY_URL = process.env.MCP_HEALTH_URL || 'http://34.175.255.139:5003/api/mcp/health';
    const MCP_FALLBACK_URL = process.env.MCP_FALLBACK_URL || 'http://34.175.255.139:5010/health';

    console.log(`🔍 MCP Status check puerto 5003: ${MCP_PRIMARY_URL}`);

    try {
      const response = await fetch(MCP_PRIMARY_URL, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        },
        signal: AbortSignal.timeout(5000) // 5 segundos timeout
      });

      console.log('📥 Respuesta del puerto 5003:', response.status);

      if (response.ok) {
        const data = await response.json();
        console.log('✅ MCP disponible en puerto 5003');

        return res.status(200).json({
          ...data,
          available: true,
          vm: 'services',
          port: 5003
        });
      } else {
        console.log('⚠️ Puerto 5003 respondió pero con error:', response.status);
      }
    } catch (primaryError) {
      console.log('⚠️ Puerto 5003 no responde:', primaryError.message);
    }

    // Fallback: Intentar puerto 5010
    console.log(`🔍 Intentando MCP puerto 5010: ${MCP_FALLBACK_URL}`);
    try {
      const fallbackResponse = await fetch(MCP_FALLBACK_URL, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        },
        signal: AbortSignal.timeout(5000)
      });

      console.log('📥 Respuesta del puerto 5010:', fallbackResponse.status);

      if (fallbackResponse.ok) {
        const data = await fallbackResponse.json();
        console.log('✅ MCP disponible en puerto 5010 (fallback)');

        return res.status(200).json({
          ...data,
          available: true,
          vm: 'services',
          port: 5010,
          note: 'Using fallback port'
        });
      } else {
        console.log('⚠️ Puerto 5010 respondió pero con error:', fallbackResponse.status);
      }
    } catch (fallbackError) {
      console.log('⚠️ Puerto 5010 tampoco responde:', fallbackError.message);
    }

    // Si ambos fallan
    console.log('❌ Ambos puertos MCP fallaron');
    throw new Error('Ambos puertos MCP no responden');

  } catch (error) {
    console.error('❌ MCP no disponible:', error.message);

    // Devolver respuesta que indica que MCP no está disponible
    return res.status(200).json({
      service: 'mcp-server',
      status: 'unavailable',
      available: false,
      error: error.message,
      vm: 'services',
      ports_tried: [5003, 5010],
      fallback_mode: true
    });
  }
}
