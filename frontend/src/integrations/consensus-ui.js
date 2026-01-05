/**
 * Capibara6 Consensus UI
 * Gestión del panel de servicios, selector de modelos y controles avanzados
 */

// Configuración de modelos disponibles en Ollama
const AVAILABLE_MODELS = [
    {
        id: 'gpt-oss:20b',
        name: 'GPT-OSS 20B',
        description: '20.9B parámetros - Modelo principal',
        params: '20.9B',
        icon: '🦫',
        contextWindow: '4K tokens',
        recommended: true
    },
    {
        id: 'mistral',
        name: 'Mistral',
        description: 'Modelo rápido y eficiente',
        params: '7B',
        icon: '🌬️',
        contextWindow: '32K tokens',
        recommended: false
    },
    {
        id: 'phi3',
        name: 'Phi-3',
        description: 'Modelo compacto de Microsoft',
        params: '3.8B',
        icon: '⚡',
        contextWindow: '128K tokens',
        recommended: false
    }
];

// Estado del panel de servicios
let servicesPanelOpen = false;
let currentModel = 'gpt-oss:20b';
let servicesStatus = {
    ollama: { online: false, checking: true },
    tts: { online: false, enabled: false },
    mcp: { online: false, enabled: true },
    n8n: { online: false, checking: false }
};

/**
 * Inicializar panel de servicios
 */
function initConsensusUI() {
    console.log('🎨 Inicializando Consensus UI...');

    // Crear panel de servicios si no existe
    if (!document.getElementById('services-panel')) {
        createServicesPanel();
    }

    // Setup event listeners
    setupServicesPanelListeners();

    // Cargar configuración guardada
    loadSavedConfiguration();

    // Verificar estado de servicios
    checkAllServicesStatus();

    // Actualizar UI inicial
    updateActiveModelDisplay();
    updateServiceBadges();

    console.log('✅ Consensus UI inicializado');
}

/**
 * Crear panel de servicios dinámicamente
 */
function createServicesPanel() {
    const panel = document.createElement('div');
    panel.id = 'services-panel';
    panel.className = 'services-panel';

    panel.innerHTML = `
        <div class="services-panel-header">
            <div class="services-panel-title">
                <i data-lucide="settings" style="width: 20px; height: 20px;"></i>
                <span>Panel de Control</span>
            </div>
            <button class="btn-icon" id="close-services-panel">
                <i data-lucide="x" style="width: 20px; height: 20px;"></i>
            </button>
        </div>

        <div class="services-panel-content">
            <!-- Modelo Actual -->
            <div class="current-model-card" id="current-model-card">
                <div class="current-model-header">
                    <div class="model-icon">
                        🦫
                    </div>
                    <div class="model-info">
                        <h3 id="current-model-name">GPT-OSS 20B</h3>
                        <p id="current-model-desc">20.9B parámetros</p>
                    </div>
                </div>
                <div class="model-specs">
                    <div class="model-spec-item">
                        <div class="model-spec-label">Parámetros</div>
                        <div class="model-spec-value" id="model-params">20.9B</div>
                    </div>
                    <div class="model-spec-item">
                        <div class="model-spec-label">Contexto</div>
                        <div class="model-spec-value" id="model-context">4K tokens</div>
                    </div>
                    <div class="model-spec-item">
                        <div class="model-spec-label">Hardware</div>
                        <div class="model-spec-value">VM Bounty2</div>
                    </div>
                    <div class="model-spec-item">
                        <div class="model-spec-label">Backend</div>
                        <div class="model-spec-value">Ollama</div>
                    </div>
                </div>
            </div>

            <!-- Selector de Modelos -->
            <div class="model-selector">
                <label class="model-selector-label">
                    <i data-lucide="cpu" style="width: 14px; height: 14px;"></i>
                    Modelo de IA
                </label>
                <div class="model-options" id="model-options">
                    <!-- Se llenarán dinámicamente -->
                </div>
            </div>

            <!-- Estado de Servicios -->
            <div class="services-status">
                <div class="services-status-title">
                    <i data-lucide="activity" style="width: 14px; height: 14px;"></i>
                    Estado de Servicios
                </div>

                <div class="service-item" id="service-ollama">
                    <div class="service-item-left">
                        <div class="service-icon ollama">🦙</div>
                        <div class="service-info">
                            <h4>Ollama</h4>
                            <p>Servidor de modelos</p>
                        </div>
                    </div>
                    <div class="service-status-badge checking">
                        <div class="service-status-led checking"></div>
                        <span>Verificando...</span>
                    </div>
                </div>

                <div class="service-item" id="service-tts">
                    <div class="service-item-left">
                        <div class="service-icon tts">🎙️</div>
                        <div class="service-info">
                            <h4>TTS (Coqui)</h4>
                            <p>Text-to-Speech</p>
                        </div>
                    </div>
                    <div class="service-status-badge offline">
                        <div class="service-status-led"></div>
                        <span>Offline</span>
                    </div>
                </div>

                <div class="service-item" id="service-mcp">
                    <div class="service-item-left">
                        <div class="service-icon mcp">🧠</div>
                        <div class="service-info">
                            <h4>Smart MCP</h4>
                            <p>Contexto verificado</p>
                        </div>
                    </div>
                    <div class="service-status-badge offline">
                        <div class="service-status-led"></div>
                        <span>Offline</span>
                    </div>
                </div>

                <div class="service-item" id="service-n8n">
                    <div class="service-item-left">
                        <div class="service-icon n8n">🔄</div>
                        <div class="service-info">
                            <h4>N8N</h4>
                            <p>Automatización</p>
                        </div>
                    </div>
                    <div class="service-status-badge offline">
                        <div class="service-status-led"></div>
                        <span>Offline</span>
                    </div>
                </div>
            </div>

            <!-- Controles de Servicios -->
            <div class="services-controls">
                <div class="service-toggle">
                    <span class="service-toggle-label">
                        <i data-lucide="volume-2" style="width: 14px; height: 14px;"></i>
                        TTS Activado
                    </span>
                    <div class="toggle-switch" id="toggle-tts" data-service="tts"></div>
                </div>

                <div class="service-toggle">
                    <span class="service-toggle-label">
                        <i data-lucide="shield-check" style="width: 14px; height: 14px;"></i>
                        MCP Activado
                    </span>
                    <div class="toggle-switch active" id="toggle-mcp" data-service="mcp"></div>
                </div>
            </div>

            <!-- Parámetros de Generación -->
            <div class="parameter-controls">
                <div class="parameter-controls-title">
                    <i data-lucide="sliders-horizontal" style="width: 14px; height: 14px;"></i>
                    Parámetros de Generación
                </div>

                <div class="parameter-item">
                    <div class="parameter-label">
                        <span class="parameter-name">Temperatura (Creatividad)</span>
                        <span class="parameter-value" id="param-temp-value">0.7</span>
                    </div>
                    <input type="range" class="parameter-slider" id="param-temp"
                           min="0" max="2" step="0.1" value="0.7">
                </div>

                <div class="parameter-item">
                    <div class="parameter-label">
                        <span class="parameter-name">Máximo de Tokens</span>
                        <span class="parameter-value" id="param-tokens-value">500</span>
                    </div>
                    <input type="range" class="parameter-slider" id="param-tokens"
                           min="100" max="2000" step="50" value="500">
                </div>

                <div class="parameter-item">
                    <div class="parameter-label">
                        <span class="parameter-name">Top P (Diversidad)</span>
                        <span class="parameter-value" id="param-topp-value">0.9</span>
                    </div>
                    <input type="range" class="parameter-slider" id="param-topp"
                           min="0" max="1" step="0.05" value="0.9">
                </div>
            </div>

            <!-- Info del Sistema -->
            <div class="info-card">
                <div class="info-card-title">
                    <i data-lucide="info" style="width: 14px; height: 14px;"></i>
                    Infraestructura
                </div>
                <div class="info-card-content">
                    <ul class="info-card-list">
                        <li><span>VM Modelos:</span> <strong>34.12.166.76</strong></li>
                        <li><span>VM Servicios:</span> <strong>34.175.136.104</strong></li>
                        <li><span>Región:</span> <strong>europe-west4</strong></li>
                    </ul>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(panel);

    // Reinicializar iconos de Lucide
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
}

/**
 * Setup event listeners del panel
 */
function setupServicesPanelListeners() {
    // Botón cerrar panel
    const closeBtn = document.getElementById('close-services-panel');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => toggleServicesPanel(false));
    }

    // Renderizar opciones de modelos
    renderModelOptions();

    // Event listeners para toggles de servicios
    document.querySelectorAll('.toggle-switch').forEach(toggle => {
        toggle.addEventListener('click', function() {
            this.classList.toggle('active');
            const service = this.dataset.service;
            handleServiceToggle(service, this.classList.contains('active'));
        });
    });

    // Event listeners para sliders de parámetros
    const tempSlider = document.getElementById('param-temp');
    const tokensSlider = document.getElementById('param-tokens');
    const toppSlider = document.getElementById('param-topp');

    if (tempSlider) {
        tempSlider.addEventListener('input', (e) => {
            document.getElementById('param-temp-value').textContent = e.target.value;
            if (typeof MODEL_CONFIG !== 'undefined') {
                MODEL_CONFIG.defaultParams.temperature = parseFloat(e.target.value);
            }
        });
    }

    if (tokensSlider) {
        tokensSlider.addEventListener('input', (e) => {
            document.getElementById('param-tokens-value').textContent = e.target.value;
            if (typeof MODEL_CONFIG !== 'undefined') {
                MODEL_CONFIG.defaultParams.n_predict = parseInt(e.target.value);
            }
        });
    }

    if (toppSlider) {
        toppSlider.addEventListener('input', (e) => {
            document.getElementById('param-topp-value').textContent = e.target.value;
            if (typeof MODEL_CONFIG !== 'undefined') {
                MODEL_CONFIG.defaultParams.top_p = parseFloat(e.target.value);
            }
        });
    }
}

/**
 * Renderizar opciones de modelos
 */
function renderModelOptions() {
    const container = document.getElementById('model-options');
    if (!container) return;

    container.innerHTML = '';

    AVAILABLE_MODELS.forEach(model => {
        const option = document.createElement('div');
        option.className = 'model-option' + (model.id === currentModel ? ' active' : '');
        option.dataset.modelId = model.id;

        option.innerHTML = `
            <div class="model-option-radio"></div>
            <div class="model-option-info">
                <div class="model-option-name">${model.icon} ${model.name}</div>
                <div class="model-option-desc">${model.description}</div>
            </div>
        `;

        option.addEventListener('click', () => selectModel(model.id));
        container.appendChild(option);
    });
}

/**
 * Seleccionar un modelo
 */
function selectModel(modelId) {
    currentModel = modelId;

    // Actualizar opciones visuales
    document.querySelectorAll('.model-option').forEach(opt => {
        opt.classList.toggle('active', opt.dataset.modelId === modelId);
    });

    // Actualizar tarjeta de modelo actual
    const model = AVAILABLE_MODELS.find(m => m.id === modelId);
    if (model) {
        document.getElementById('current-model-name').textContent = model.name;
        document.getElementById('current-model-desc').textContent = model.description;
        document.getElementById('model-params').textContent = model.params;
        document.getElementById('model-context').textContent = model.contextWindow;
        document.querySelector('.current-model-card .model-icon').textContent = model.icon;
    }

    // Guardar configuración
    localStorage.setItem('selected-model', modelId);

    // Actualizar indicador en header
    updateActiveModelDisplay();

    // Notificar al sistema
    console.log(`✅ Modelo seleccionado: ${modelId}`);

    // Actualizar MODEL_CONFIG si existe
    if (typeof MODEL_CONFIG !== 'undefined') {
        // Cambiar el modelo en la configuración
        // Nota: El backend debe soportar el cambio de modelo
        console.log('📝 Actualizando configuración de modelo...');
    }
}

/**
 * Toggle panel de servicios
 */
function toggleServicesPanel(open = null) {
    const panel = document.getElementById('services-panel');
    if (!panel) return;

    servicesPanelOpen = open !== null ? open : !servicesPanelOpen;
    panel.classList.toggle('open', servicesPanelOpen);
}

/**
 * Manejar toggle de servicio
 */
function handleServiceToggle(service, enabled) {
    console.log(`🔄 ${service} ${enabled ? 'activado' : 'desactivado'}`);

    switch(service) {
        case 'tts':
            if (typeof TTS_CONFIG !== 'undefined') {
                TTS_CONFIG.useCoquiTTS = enabled;
                servicesStatus.tts.enabled = enabled;
            }
            break;
        case 'mcp':
            if (typeof SMART_MCP_CONFIG !== 'undefined') {
                SMART_MCP_CONFIG.enabled = enabled;
                servicesStatus.mcp.enabled = enabled;
            }
            break;
    }

    // Guardar configuración
    localStorage.setItem(`service-${service}-enabled`, enabled);

    // Actualizar badges
    updateServiceBadges();
}

/**
 * Verificar estado de todos los servicios
 */
async function checkAllServicesStatus() {
    console.log('🔍 Verificando estado de servicios...');

    // Verificar Ollama
    await checkServiceStatus('ollama', 'http://34.12.166.76:11434/api/tags');

    // Verificar TTS (solo si está habilitado)
    if (window.CHATBOT_CONFIG?.SERVICES?.TTS_ENABLED !== false) {
        await checkServiceStatus('tts', 'http://34.175.136.104:5002/health');
    }

    // Verificar MCP (solo si está habilitado)
    if (window.CHATBOT_CONFIG?.SERVICES?.MCP_ENABLED) {
        await checkServiceStatus('mcp', 'http://34.175.136.104:5003/api/mcp/health');
    }

    // N8N deshabilitado por defecto - requiere VPN/túnel
    // await checkServiceStatus('n8n', 'http://34.175.136.104:5678/healthz');

    updateServiceBadges();
}

/**
 * Verificar estado de un servicio específico
 */
async function checkServiceStatus(serviceName, url) {
    const serviceElement = document.getElementById(`service-${serviceName}`);
    if (!serviceElement) return;

    const badge = serviceElement.querySelector('.service-status-badge');
    const led = serviceElement.querySelector('.service-status-led');

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000);

        const response = await fetch(url, {
            method: 'GET',
            signal: controller.signal,
            mode: 'cors'
        });

        clearTimeout(timeoutId);

        if (response.ok) {
            servicesStatus[serviceName].online = true;
            badge.className = 'service-status-badge online';
            badge.innerHTML = '<div class="service-status-led"></div><span>Online</span>';
            console.log(`✅ ${serviceName} online`);
        } else {
            throw new Error('Service not available');
        }
    } catch (error) {
        servicesStatus[serviceName].online = false;
        badge.className = 'service-status-badge offline';
        badge.innerHTML = '<div class="service-status-led"></div><span>Offline</span>';
        console.log(`❌ ${serviceName} offline`);
    } finally {
        servicesStatus[serviceName].checking = false;
    }
}

/**
 * Actualizar badges de servicios en header
 */
function updateServiceBadges() {
    const container = document.getElementById('service-badges-container');
    if (!container) return;

    container.innerHTML = '';

    // Badge MCP si está activo
    if (servicesStatus.mcp.enabled) {
        const badge = createServiceBadge('MCP', servicesStatus.mcp.online, 'shield-check');
        container.appendChild(badge);
    }

    // Badge TTS si está activo
    if (servicesStatus.tts.enabled) {
        const badge = createServiceBadge('TTS', servicesStatus.tts.online, 'volume-2');
        container.appendChild(badge);
    }
}

/**
 * Crear badge de servicio
 */
function createServiceBadge(name, online, icon) {
    const badge = document.createElement('div');
    badge.className = 'service-badge' + (online ? ' active' : '');
    badge.title = `${name}: ${online ? 'Online' : 'Offline'}`;
    badge.innerHTML = `
        <i data-lucide="${icon}" class="service-badge-icon"></i>
        <span>${name}</span>
    `;

    if (typeof lucide !== 'undefined') {
        setTimeout(() => lucide.createIcons(), 0);
    }

    return badge;
}

/**
 * Actualizar display del modelo activo en header
 */
function updateActiveModelDisplay() {
    const indicator = document.getElementById('active-model-indicator');
    if (!indicator) return;

    const model = AVAILABLE_MODELS.find(m => m.id === currentModel);
    if (!model) return;

    indicator.querySelector('.model-indicator-name').textContent = model.name;
    indicator.querySelector('.model-indicator-params').textContent = model.params;
    indicator.querySelector('.model-indicator-icon').textContent = model.icon;
}

/**
 * Cargar configuración guardada
 */
function loadSavedConfiguration() {
    // Modelo guardado
    const savedModel = localStorage.getItem('selected-model');
    if (savedModel && AVAILABLE_MODELS.find(m => m.id === savedModel)) {
        selectModel(savedModel);
    }

    // Estado de servicios
    const ttsEnabled = localStorage.getItem('service-tts-enabled') === 'true';
    const mcpEnabled = localStorage.getItem('service-mcp-enabled') !== 'false'; // default true

    const ttsToggle = document.getElementById('toggle-tts');
    const mcpToggle = document.getElementById('toggle-mcp');

    if (ttsToggle) {
        ttsToggle.classList.toggle('active', ttsEnabled);
        if (typeof TTS_CONFIG !== 'undefined') {
            TTS_CONFIG.useCoquiTTS = ttsEnabled;
        }
        servicesStatus.tts.enabled = ttsEnabled;
    }

    if (mcpToggle) {
        mcpToggle.classList.toggle('active', mcpEnabled);
        if (typeof SMART_MCP_CONFIG !== 'undefined') {
            SMART_MCP_CONFIG.enabled = mcpEnabled;
        }
        servicesStatus.mcp.enabled = mcpEnabled;
    }
}

// Auto-inicializar cuando el DOM esté listo
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initConsensusUI);
} else {
    initConsensusUI();
}

// Exportar funciones globales
window.toggleServicesPanel = toggleServicesPanel;
window.checkAllServicesStatus = checkAllServicesStatus;
window.servicesStatus = servicesStatus;
