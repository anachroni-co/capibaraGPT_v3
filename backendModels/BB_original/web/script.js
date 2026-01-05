// script.js - Funciones generales para la plataforma Capibara6

document.addEventListener('DOMContentLoaded', function() {
    console.log('Capibara6 plataforma cargada');
    
    // Inicializar animaciones neuronales
    if (typeof initNeuralAnimation === 'function') {
        initNeuralAnimation();
    }
});

// Función para hacer solicitudes API a los servidores backend
async function makeApiRequest(endpoint, data) {
    try {
        const response = await fetch(`https://your-vercel-deployment.vercel.app/api/${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('Error en la solicitud API:', error);
        throw error;
    }
}