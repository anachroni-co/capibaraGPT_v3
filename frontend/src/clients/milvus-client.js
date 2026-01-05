/**
 * Cliente JavaScript para Milvus Vector Database
 * Accede a Milvus a través de capibara6-api (bridge)
 *
 * Funcionalidades:
 * - Búsqueda vectorial semántica
 * - Inserción de vectores
 * - Gestión de colecciones
 * - Cache de resultados
 */

class MilvusClient {
    constructor(config = {}) {
        // Configuración desde config.js o valores por defecto
        this.config = {
            bridgeUrl: config.bridgeUrl || CHATBOT_CONFIG.SERVICES.RAG3_BRIDGE.url,
            milvusConfig: config.milvusConfig || CHATBOT_CONFIG.SERVICES.MILVUS.config,
            searchParams: config.searchParams || CHATBOT_CONFIG.SERVICES.MILVUS.search_params,
            timeout: config.timeout || CHATBOT_CONFIG.SERVICES.MILVUS.timeout,
            useCache: config.useCache !== undefined ? config.useCache : true,
            cacheTTL: config.cacheTTL || 300000 // 5 minutos
        };

        // Cache de resultados de búsqueda
        this.searchCache = new Map();

        // Estadísticas
        this.stats = {
            searches: 0,
            cache_hits: 0,
            cache_misses: 0,
            inserts: 0,
            errors: 0
        };

        console.log('🔍 MilvusClient initialized', this.config);
    }

    /**
     * Búsqueda vectorial semántica
     * @param {Array<number>} vector - Vector de embedding (384 dimensiones)
     * @param {Object} options - Opciones de búsqueda
     * @returns {Promise<Array>} Resultados de búsqueda con scores
     */
    async search(vector, options = {}) {
        this.stats.searches++;

        // Generar clave de cache
        const cacheKey = this._generateCacheKey(vector, options);

        // Verificar cache
        if (this.config.useCache && this.searchCache.has(cacheKey)) {
            const cached = this.searchCache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.config.cacheTTL) {
                this.stats.cache_hits++;
                console.log('✅ Cache hit for Milvus search');
                return cached.results;
            } else {
                // Cache expirado
                this.searchCache.delete(cacheKey);
            }
        }

        this.stats.cache_misses++;

        const searchOptions = {
            collection_name: this.config.milvusConfig.collection_name,
            vector: vector,
            top_k: options.top_k || this.config.searchParams.top_k,
            nprobe: options.nprobe || this.config.searchParams.nprobe,
            offset: options.offset || this.config.searchParams.offset,
            output_fields: options.output_fields || ['id', 'text', 'metadata'],
            filter: options.filter || null
        };

        try {
            const response = await this._makeRequest(
                CHATBOT_CONFIG.SERVICES.RAG3_BRIDGE.endpoints.MILVUS_SEARCH,
                'POST',
                searchOptions
            );

            const results = response.results || [];

            // Guardar en cache
            if (this.config.useCache) {
                this.searchCache.set(cacheKey, {
                    results: results,
                    timestamp: Date.now()
                });

                // Limpiar cache antigua si es necesario
                this._cleanCache();
            }

            console.log(`🔍 Milvus search completed: ${results.length} results`);
            return results;

        } catch (error) {
            this.stats.errors++;
            console.error('❌ Milvus search error:', error);
            throw error;
        }
    }

    /**
     * Búsqueda semántica desde texto (genera embedding automáticamente)
     * @param {string} text - Texto a buscar
     * @param {Object} options - Opciones de búsqueda
     * @returns {Promise<Array>} Resultados de búsqueda
     */
    async searchByText(text, options = {}) {
        try {
            // Generar embedding del texto a través del bridge
            const embedding = await this._getEmbedding(text);

            // Realizar búsqueda vectorial
            return await this.search(embedding, options);

        } catch (error) {
            this.stats.errors++;
            console.error('❌ Milvus searchByText error:', error);
            throw error;
        }
    }

    /**
     * Insertar vectores en Milvus
     * @param {Array<Object>} data - Array de objetos con {id, vector, text, metadata}
     * @returns {Promise<Object>} Resultado de la inserción
     */
    async insert(data) {
        this.stats.inserts++;

        const insertData = {
            collection_name: this.config.milvusConfig.collection_name,
            data: data
        };

        try {
            const response = await this._makeRequest(
                CHATBOT_CONFIG.SERVICES.RAG3_BRIDGE.endpoints.MILVUS_INSERT,
                'POST',
                insertData
            );

            console.log(`✅ Milvus insert: ${data.length} vectors inserted`);
            return response;

        } catch (error) {
            this.stats.errors++;
            console.error('❌ Milvus insert error:', error);
            throw error;
        }
    }

    /**
     * Búsqueda híbrida (combina vector search con filtros)
     * @param {string} text - Texto a buscar
     * @param {Object} filters - Filtros adicionales
     * @param {Object} options - Opciones de búsqueda
     * @returns {Promise<Array>} Resultados filtrados
     */
    async hybridSearch(text, filters = {}, options = {}) {
        try {
            const embedding = await this._getEmbedding(text);

            // Agregar filtros a las opciones
            const searchOptions = {
                ...options,
                filter: this._buildFilterExpression(filters)
            };

            return await this.search(embedding, searchOptions);

        } catch (error) {
            this.stats.errors++;
            console.error('❌ Milvus hybridSearch error:', error);
            throw error;
        }
    }

    /**
     * Obtener información de la colección
     * @returns {Promise<Object>} Información de la colección
     */
    async getCollectionInfo() {
        try {
            const response = await this._makeRequest(
                `/api/v1/milvus/collection/${this.config.milvusConfig.collection_name}`,
                'GET'
            );

            console.log('📊 Collection info:', response);
            return response;

        } catch (error) {
            console.error('❌ Get collection info error:', error);
            throw error;
        }
    }

    /**
     * Obtener estadísticas del cliente
     * @returns {Object} Estadísticas de uso
     */
    getStats() {
        return {
            ...this.stats,
            cache_size: this.searchCache.size,
            cache_hit_rate: this.stats.searches > 0
                ? (this.stats.cache_hits / this.stats.searches * 100).toFixed(2) + '%'
                : '0%'
        };
    }

    /**
     * Limpiar cache
     */
    clearCache() {
        this.searchCache.clear();
        console.log('🗑️ Milvus cache cleared');
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
     * Obtener embedding de un texto
     * @private
     */
    async _getEmbedding(text) {
        const response = await this._makeRequest(
            '/api/v1/embeddings',
            'POST',
            { text: text }
        );

        return response.embedding;
    }

    /**
     * Generar clave de cache
     * @private
     */
    _generateCacheKey(vector, options) {
        const vectorStr = vector.slice(0, 10).join(','); // Primeros 10 elementos
        const optionsStr = JSON.stringify(options);
        return `${vectorStr}:${optionsStr}`;
    }

    /**
     * Limpiar cache antigua
     * @private
     */
    _cleanCache() {
        const maxCacheSize = 100;
        if (this.searchCache.size > maxCacheSize) {
            // Eliminar las entradas más antiguas
            const entries = Array.from(this.searchCache.entries());
            entries.sort((a, b) => a[1].timestamp - b[1].timestamp);

            const toDelete = entries.slice(0, entries.length - maxCacheSize);
            toDelete.forEach(([key]) => this.searchCache.delete(key));

            console.log(`🗑️ Cache cleaned: removed ${toDelete.length} old entries`);
        }
    }

    /**
     * Construir expresión de filtro para Milvus
     * @private
     */
    _buildFilterExpression(filters) {
        if (!filters || Object.keys(filters).length === 0) {
            return null;
        }

        const expressions = [];

        for (const [field, value] of Object.entries(filters)) {
            if (typeof value === 'string') {
                expressions.push(`${field} == "${value}"`);
            } else if (typeof value === 'number') {
                expressions.push(`${field} == ${value}`);
            } else if (Array.isArray(value)) {
                expressions.push(`${field} in [${value.map(v => typeof v === 'string' ? `"${v}"` : v).join(', ')}]`);
            }
        }

        return expressions.join(' && ');
    }
}

// Exportar para uso global
if (typeof window !== 'undefined') {
    window.MilvusClient = MilvusClient;
}
