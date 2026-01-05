/**
 * Cliente JavaScript para Nebula Graph Database
 * Accede a Nebula Graph a través de capibara6-api (bridge)
 *
 * Funcionalidades:
 * - Consultas de grafo (MATCH, GO, FETCH)
 * - Inserción de vértices y aristas
 * - Análisis de relaciones
 * - Traversal de grafos
 */

class NebulaClient {
    constructor(config = {}) {
        // Configuración desde config.js o valores por defecto
        this.config = {
            bridgeUrl: config.bridgeUrl || CHATBOT_CONFIG.SERVICES.RAG3_BRIDGE.url,
            nebulaConfig: config.nebulaConfig || CHATBOT_CONFIG.SERVICES.NEBULA_GRAPH.config,
            studioUrl: config.studioUrl || CHATBOT_CONFIG.SERVICES.NEBULA_GRAPH.studio_url,
            timeout: config.timeout || CHATBOT_CONFIG.SERVICES.NEBULA_GRAPH.timeout,
            useCache: config.useCache !== undefined ? config.useCache : true,
            cacheTTL: config.cacheTTL || 300000 // 5 minutos
        };

        // Cache de resultados de consultas
        this.queryCache = new Map();

        // Estadísticas
        this.stats = {
            queries: 0,
            inserts: 0,
            cache_hits: 0,
            cache_misses: 0,
            errors: 0
        };

        console.log('🕸️ NebulaClient initialized', this.config);
    }

    /**
     * Ejecutar consulta nGQL (Nebula Graph Query Language)
     * @param {string} query - Consulta nGQL
     * @param {Object} params - Parámetros para la consulta
     * @returns {Promise<Object>} Resultado de la consulta
     */
    async query(query, params = {}) {
        this.stats.queries++;

        // Generar clave de cache
        const cacheKey = this._generateCacheKey(query, params);

        // Verificar cache
        if (this.config.useCache && this.queryCache.has(cacheKey)) {
            const cached = this.queryCache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.config.cacheTTL) {
                this.stats.cache_hits++;
                console.log('✅ Cache hit for Nebula query');
                return cached.results;
            } else {
                this.queryCache.delete(cacheKey);
            }
        }

        this.stats.cache_misses++;

        try {
            const response = await this._makeRequest(
                CHATBOT_CONFIG.SERVICES.RAG3_BRIDGE.endpoints.NEBULA_QUERY,
                'POST',
                {
                    query: query,
                    params: params,
                    space: this.config.nebulaConfig.space_name
                }
            );

            const results = response.results || [];

            // Guardar en cache
            if (this.config.useCache) {
                this.queryCache.set(cacheKey, {
                    results: results,
                    timestamp: Date.now()
                });

                this._cleanCache();
            }

            console.log(`🕸️ Nebula query completed: ${results.length} results`);
            return results;

        } catch (error) {
            this.stats.errors++;
            console.error('❌ Nebula query error:', error);
            throw error;
        }
    }

    /**
     * Buscar vértices por propiedades
     * @param {string} tag - Tag del vértice (tipo)
     * @param {Object} properties - Propiedades a buscar
     * @param {number} limit - Límite de resultados
     * @returns {Promise<Array>} Vértices encontrados
     */
    async findVertices(tag, properties = {}, limit = 100) {
        const whereClause = this._buildWhereClause(properties);
        const query = `
            MATCH (v:${tag})
            ${whereClause ? 'WHERE ' + whereClause : ''}
            RETURN v
            LIMIT ${limit}
        `;

        return await this.query(query);
    }

    /**
     * Buscar relaciones (aristas) entre nodos
     * @param {string} fromId - ID del nodo origen
     * @param {string} edgeType - Tipo de relación
     * @param {number} depth - Profundidad de búsqueda
     * @returns {Promise<Array>} Relaciones encontradas
     */
    async findRelations(fromId, edgeType = null, depth = 1) {
        const edgePattern = edgeType ? `-[r:${edgeType}]-` : `-[r]-`;
        const query = `
            GO ${depth} STEPS FROM "${fromId}"
            OVER ${edgeType || '*'}
            YIELD src(edge) AS source, dst(edge) AS target, properties(edge) AS props
        `;

        return await this.query(query);
    }

    /**
     * Encontrar el camino más corto entre dos nodos
     * @param {string} fromId - ID del nodo origen
     * @param {string} toId - ID del nodo destino
     * @param {Object} options - Opciones de búsqueda
     * @returns {Promise<Array>} Camino encontrado
     */
    async findShortestPath(fromId, toId, options = {}) {
        const maxHops = options.maxHops || 5;
        const edgeType = options.edgeType || '*';

        const query = `
            FIND SHORTEST PATH FROM "${fromId}" TO "${toId}"
            OVER ${edgeType}
            UPTO ${maxHops} STEPS
            YIELD path AS p
        `;

        return await this.query(query);
    }

    /**
     * Obtener vecinos de un nodo
     * @param {string} nodeId - ID del nodo
     * @param {Object} options - Opciones de búsqueda
     * @returns {Promise<Array>} Vecinos encontrados
     */
    async getNeighbors(nodeId, options = {}) {
        const depth = options.depth || 1;
        const direction = options.direction || 'both'; // 'in', 'out', 'both'
        const tag = options.tag || null;

        let query;
        if (direction === 'out') {
            query = `GO ${depth} STEPS FROM "${nodeId}" OVER *`;
        } else if (direction === 'in') {
            query = `GO ${depth} STEPS FROM "${nodeId}" OVER * REVERSELY`;
        } else {
            query = `GO ${depth} STEPS FROM "${nodeId}" OVER * BIDIRECT`;
        }

        query += ` YIELD dst(edge) AS neighbor, properties($$) AS props`;

        if (tag) {
            query += ` WHERE properties($$).tag == "${tag}"`;
        }

        return await this.query(query);
    }

    /**
     * Análisis de comunidades (clustering de nodos)
     * @param {string} tag - Tag del vértice
     * @param {Object} options - Opciones de análisis
     * @returns {Promise<Array>} Comunidades detectadas
     */
    async analyzeCommunities(tag, options = {}) {
        const depth = options.depth || 2;
        const minSize = options.minSize || 3;

        const query = `
            MATCH (v:${tag})-[*1..${depth}]-(connected:${tag})
            WITH v, count(DISTINCT connected) AS connections
            WHERE connections >= ${minSize}
            RETURN v, connections
            ORDER BY connections DESC
        `;

        return await this.query(query);
    }

    /**
     * Insertar vértice
     * @param {string} tag - Tag del vértice
     * @param {string} id - ID del vértice
     * @param {Object} properties - Propiedades del vértice
     * @returns {Promise<Object>} Resultado de la inserción
     */
    async insertVertex(tag, id, properties) {
        this.stats.inserts++;

        try {
            const response = await this._makeRequest(
                CHATBOT_CONFIG.SERVICES.RAG3_BRIDGE.endpoints.NEBULA_INSERT,
                'POST',
                {
                    type: 'vertex',
                    tag: tag,
                    id: id,
                    properties: properties,
                    space: this.config.nebulaConfig.space_name
                }
            );

            console.log(`✅ Nebula vertex inserted: ${id}`);
            return response;

        } catch (error) {
            this.stats.errors++;
            console.error('❌ Nebula insert vertex error:', error);
            throw error;
        }
    }

    /**
     * Insertar arista (relación)
     * @param {string} edgeType - Tipo de arista
     * @param {string} fromId - ID del nodo origen
     * @param {string} toId - ID del nodo destino
     * @param {Object} properties - Propiedades de la arista
     * @returns {Promise<Object>} Resultado de la inserción
     */
    async insertEdge(edgeType, fromId, toId, properties = {}) {
        this.stats.inserts++;

        try {
            const response = await this._makeRequest(
                CHATBOT_CONFIG.SERVICES.RAG3_BRIDGE.endpoints.NEBULA_INSERT,
                'POST',
                {
                    type: 'edge',
                    edge_type: edgeType,
                    from_id: fromId,
                    to_id: toId,
                    properties: properties,
                    space: this.config.nebulaConfig.space_name
                }
            );

            console.log(`✅ Nebula edge inserted: ${fromId} -[${edgeType}]-> ${toId}`);
            return response;

        } catch (error) {
            this.stats.errors++;
            console.error('❌ Nebula insert edge error:', error);
            throw error;
        }
    }

    /**
     * Análisis de centralidad (nodos más importantes)
     * @param {string} tag - Tag del vértice
     * @param {number} limit - Límite de resultados
     * @returns {Promise<Array>} Nodos más centrales
     */
    async analyzeCentrality(tag, limit = 10) {
        const query = `
            MATCH (v:${tag})-[r]-()
            WITH v, count(r) AS degree
            RETURN v, degree
            ORDER BY degree DESC
            LIMIT ${limit}
        `;

        return await this.query(query);
    }

    /**
     * Obtener estadísticas del cliente
     * @returns {Object} Estadísticas de uso
     */
    getStats() {
        return {
            ...this.stats,
            cache_size: this.queryCache.size,
            cache_hit_rate: this.stats.queries > 0
                ? (this.stats.cache_hits / this.stats.queries * 100).toFixed(2) + '%'
                : '0%',
            studio_url: this.config.studioUrl
        };
    }

    /**
     * Limpiar cache
     */
    clearCache() {
        this.queryCache.clear();
        console.log('🗑️ Nebula cache cleared');
    }

    /**
     * Abrir Nebula Studio en nueva ventana
     */
    openStudio() {
        window.open(this.config.studioUrl, '_blank');
    }

    // ========== Métodos Privados ==========

    /**
     * Hacer request al bridge
     * @private
     */
    async _makeRequest(endpoint, method = 'GET', data = null) {
        const url = `${this.config.bridgeUrl}${endpoint}`;

        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            timeout: this.config.timeout
        };

        if (data && method !== 'GET') {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(url, options);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    }

    /**
     * Generar clave de cache
     * @private
     */
    _generateCacheKey(query, params) {
        const paramsStr = JSON.stringify(params);
        return `${query}:${paramsStr}`;
    }

    /**
     * Limpiar cache antigua
     * @private
     */
    _cleanCache() {
        const maxCacheSize = 100;
        if (this.queryCache.size > maxCacheSize) {
            const entries = Array.from(this.queryCache.entries());
            entries.sort((a, b) => a[1].timestamp - b[1].timestamp);

            const toDelete = entries.slice(0, entries.length - maxCacheSize);
            toDelete.forEach(([key]) => this.queryCache.delete(key));

            console.log(`🗑️ Cache cleaned: removed ${toDelete.length} old entries`);
        }
    }

    /**
     * Construir cláusula WHERE
     * @private
     */
    _buildWhereClause(properties) {
        if (!properties || Object.keys(properties).length === 0) {
            return '';
        }

        const conditions = [];

        for (const [key, value] of Object.entries(properties)) {
            if (typeof value === 'string') {
                conditions.push(`v.${key} == "${value}"`);
            } else if (typeof value === 'number') {
                conditions.push(`v.${key} == ${value}`);
            } else if (typeof value === 'boolean') {
                conditions.push(`v.${key} == ${value}`);
            }
        }

        return conditions.join(' AND ');
    }
}

// Exportar para uso global
if (typeof window !== 'undefined') {
    window.NebulaClient = NebulaClient;
}
