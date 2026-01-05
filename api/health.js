/**
 * Vercel Serverless Function - Health Check
 * Endpoint general de health para verificar estado del backend
 *
 * Actualizado: 2025-12-01
 */

export default async function handler(req, res) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle preflight
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  try {
    // Health check del backend
    const health = {
      status: 'healthy',
      service: 'Capibara6 Backend',
      timestamp: new Date().toISOString(),
      environment: 'production',
      version: '2.0.0',
      services: {
        vercel: 'operational',
        serverless: 'operational'
      },
      endpoints: {
        completion: '/api/completion',
        tts: '/api/tts',
        mcp: {
          health: '/api/mcp-health',
          status: '/api/mcp/status',
          analyze: '/api/mcp-analyze'
        }
      }
    };

    return res.status(200).json(health);

  } catch (error) {
    console.error('Health check error:', error);
    return res.status(500).json({
      status: 'error',
      error: error.message
    });
  }
}
