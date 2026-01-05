// api/mcp/analyze.js - Proxy a VM MCP
export default async function handler(request, response) {
  if (request.method !== 'POST') {
    return response.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Decidir entre el MCP integrado en el servidor principal o el standalone
    const backendResponse = await fetch('http://34.175.215.109:5010/api/mcp/analyze', {  // Usando el standalone
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request.body),
    });

    const data = await backendResponse.json();
    response.status(backendResponse.status).json(data);
  } catch (error) {
    console.error('Error proxy MCP:', error);
    response.status(500).json({ error: 'Error interno del servidor' });
  }
}