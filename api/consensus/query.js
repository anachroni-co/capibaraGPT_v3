// Proxy para consultas de consenso
export default async function handler(req, res) {
    // Configurar CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    
    if (req.method === 'OPTIONS') {
        return res.status(200).end();
    }
    
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Método no permitido' });
    }
    
    try {
        // Reenviar la petición a la VM
        const response = await fetch('http://34.175.255.139:5002/api/consensus/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(req.body)
        });
        
        const data = await response.json();
        
        return res.status(response.status).json(data);
        
    } catch (error) {
        console.error('Error en proxy de consenso:', error);
        return res.status(500).json({ error: 'Error interno del servidor' });
    }
}
