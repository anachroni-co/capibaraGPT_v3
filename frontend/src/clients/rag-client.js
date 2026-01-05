/**
 * Cliente RAG Unificado
 * Combina búsqueda vectorial (Milvus) con knowledge graph (Nebula)
 * para proporcionar contexto enriquecido
 *
 * Funcionalidades:
 * - Búsqueda híbrida (vector + graph)
 * - Enriquecimiento de contexto
 * - Ranking de resultados
 * - Optimización con TOON
 */

class RAGClient {
    constructor(config = {}) {
        // Inicializar clientes de Milvus y Nebula
        this.milvusClient = new MilvusClient(config.milvus);
        this.nebulaClient = new NebulaClient(config.nebula);

        this.config = {
            bridgeUrl: config.bridgeUrl || CHATBOT_CONFIG.SERVICES.RAG3_BRIDGE.url,
            hybridWeight: config.hybridWeight || 0.7, // Peso de búsqueda vectorial vs grafo
            maxResults: config.maxResults || 10,
            enrichContext: config.enrichContext !== undefined ? config.enrichContext : true,
            useTOON: config.useTOON !== undefined ? config.useTOON : true
        };

        // Estadísticas
        this.stats = {
            searches: 0,
            hybrid_searches: 0,
            context_enrichments: 0,
            total_tokens_saved: 0
        };

        console.log('🎯 RAGClient initialized (Unified Milvus + Nebula)');
    }

    /**
     * Búsqueda RAG completa (vector + knowledge graph)
     * @param {string} query - Query del usuario
     * @param {Object} options - Opciones de búsqueda
     * @returns {Promise<Object>} Contexto enriquecido
     */
    async search(query, options = {}) {
        this.stats.searches++;
        this.stats.hybrid_searches++;

        console.log(`🔍 RAG Search: "${query}"`);

        try {
            // 1. Búsqueda vectorial en Milvus
            const vectorResults = await this.milvusClient.searchByText(query, {
                top_k: options.top_k || 10,
                output_fields: ['id', 'text', 'metadata', 'timestamp']
            });

            console.log(`  → Vector search: ${vectorResults.length} results`);

            // 2. Enriquecer con knowledge graph si está habilitado
            let enrichedResults = vectorResults;

            if (this.config.enrichContext && vectorResults.length > 0) {
                enrichedResults = await this._enrichWithGraph(vectorResults);
                this.stats.context_enrichments++;
                console.log(`  → Graph enrichment: +${enrichedResults.length - vectorResults.length} related items`);
            }

            // 3. Ranking y selección de mejores resultados
            const rankedResults = this._rankResults(enrichedResults, query);
            const finalResults = rankedResults.slice(0, this.config.maxResults);

            // 4. Formatear contexto para el LLM
            const context = await this._formatContext(finalResults, {
                useTOON: this.config.useTOON && finalResults.length >= CHATBOT_CONFIG.OPTIMIZATION.TOON.min_sources,
                query: query
            });

            return {
                context: context,
                results: finalResults,
                stats: {
                    vector_results: vectorResults.length,
                    enriched_results: enrichedResults.length,
                    final_results: finalResults.length,
                    format: context.format,
                    tokens_saved: context.tokens_saved || 0
                }
            };

        } catch (error) {
            console.error('❌ RAG search error:', error);
            throw error;
        }
    }

    /**
     * Búsqueda contextual (incluye historial de conversación)
     * @param {string} query - Query actual
     * @param {Array} conversationHistory - Historial de mensajes
     * @param {Object} options - Opciones
     * @returns {Promise<Object>} Contexto enriquecido
     */
    async contextualSearch(query, conversationHistory = [], options = {}) {
        // Construir query expandida con contexto de conversación
        const expandedQuery = this._buildExpandedQuery(query, conversationHistory);

        // Realizar búsqueda RAG
        const results = await this.search(expandedQuery, options);

        // Agregar contexto de conversación
        results.conversation_context = conversationHistory.slice(-3); // Últimos 3 mensajes

        return results;
    }

    /**
     * Búsqueda con filtros específicos
     * @param {string} query - Query del usuario
     * @param {Object} filters - Filtros (timestamp, tipo, etc.)
     * @param {Object} options - Opciones
     * @returns {Promise<Object>} Contexto filtrado
     */
    async filteredSearch(query, filters = {}, options = {}) {
        // Búsqueda híbrida con filtros en Milvus
        const vectorResults = await this.milvusClient.hybridSearch(query, filters, options);

        // Formatear contexto
        const context = await this._formatContext(vectorResults, {
            useTOON: this.config.useTOON && vectorResults.length >= 5,
            query: query
        });

        return {
            context: context,
            results: vectorResults,
            filters_applied: filters
        };
    }

    /**
     * Análisis de relaciones (usando solo Nebula Graph)
     * @param {string} entityId - ID de la entidad
     * @param {Object} options - Opciones de análisis
     * @returns {Promise<Object>} Análisis de relaciones
     */
    async analyzeRelations(entityId, options = {}) {
        const depth = options.depth || 2;

        // Obtener vecinos
        const neighbors = await this.nebulaClient.getNeighbors(entityId, {
            depth: depth,
            direction: options.direction || 'both'
        });

        // Analizar centralidad
        const centralityAnalysis = await this.nebulaClient.analyzeCentrality(
            options.tag || 'entity',
            options.limit || 10
        );

        return {
            entity_id: entityId,
            neighbors: neighbors,
            centrality: centralityAnalysis,
            depth: depth
        };
    }

    /**
     * Obtener estadísticas combinadas
     * @returns {Object} Estadísticas de todos los clientes
     */
    getStats() {
        return {
            rag: this.stats,
            milvus: this.milvusClient.getStats(),
            nebula: this.nebulaClient.getStats(),
            optimization: {
                toon_enabled: this.config.useTOON,
                hybrid_weight: this.config.hybridWeight,
                context_enrichment: this.config.enrichContext
            }
        };
    }

    /**
     * Limpiar todos los caches
     */
    clearAllCaches() {
        this.milvusClient.clearCache();
        this.nebulaClient.clearCache();
        console.log('🗑️ All RAG caches cleared');
    }

    // ========== Métodos Privados ==========

    /**
     * Enriquecer resultados vectoriales con knowledge graph
     * @private
     */
    async _enrichWithGraph(vectorResults) {
        const enriched = [...vectorResults];

        try {
            // Para cada resultado vectorial, buscar relaciones en el grafo
            for (const result of vectorResults.slice(0, 5)) { // Solo top 5 para no sobrecargar
                if (result.metadata && result.metadata.entity_id) {
                    // Obtener vecinos del nodo
                    const neighbors = await this.nebulaClient.getNeighbors(
                        result.metadata.entity_id,
                        { depth: 1 }
                    );

                    // Agregar vecinos relevantes a los resultados
                    if (neighbors.length > 0) {
                        enriched.push(...neighbors.slice(0, 2).map(n => ({
                            id: n.neighbor,
                            text: n.props?.text || '',
                            metadata: { ...n.props, source: 'graph', related_to: result.id },
                            score: result.score * 0.8 // Menor score para resultados del grafo
                        })));
                    }
                }
            }

        } catch (error) {
            console.warn('⚠️ Graph enrichment failed, using vector results only:', error);
        }

        return enriched;
    }

    /**
     * Ranking de resultados (combina scores vectoriales y de grafo)
     * @private
     */
    _rankResults(results, query) {
        return results
            .map(result => ({
                ...result,
                final_score: this._calculateFinalScore(result, query)
            }))
            .sort((a, b) => b.final_score - a.final_score);
    }

    /**
     * Calcular score final (híbrido)
     * @private
     */
    _calculateFinalScore(result, query) {
        const vectorScore = result.score || 0;
        const graphBonus = result.metadata?.source === 'graph' ? 0.2 : 0;
        const recencyBonus = this._calculateRecencyBonus(result.metadata?.timestamp);

        return (vectorScore * this.config.hybridWeight) +
               (graphBonus * (1 - this.config.hybridWeight)) +
               recencyBonus;
    }

    /**
     * Bonus por recencia (preferir contenido más reciente)
     * @private
     */
    _calculateRecencyBonus(timestamp) {
        if (!timestamp) return 0;

        const now = Date.now();
        const age = now - new Date(timestamp).getTime();
        const dayInMs = 24 * 60 * 60 * 1000;

        // Bonus decae exponencialmente con el tiempo
        if (age < dayInMs) return 0.1;
        if (age < dayInMs * 7) return 0.05;
        if (age < dayInMs * 30) return 0.02;
        return 0;
    }

    /**
     * Formatear contexto para el LLM
     * @private
     */
    async _formatContext(results, options = {}) {
        if (results.length === 0) {
            return {
                text: 'No hay información relevante disponible.',
                format: 'plain',
                tokens_saved: 0
            };
        }

        // Usar TOON si está habilitado y hay suficientes fuentes
        if (options.useTOON) {
            return await this._formatWithTOON(results, options.query);
        }

        // Formato JSON tradicional
        return await this._formatWithJSON(results, options.query);
    }

    /**
     * Formatear con TOON (optimización de tokens)
     * @private
     */
    async _formatWithTOON(results, query) {
        const header = `Información relevante para: "${query}" (formato TOON)\n\n`;

        const toonData = `sources[${results.length}]{id,text,score,timestamp,source}:\n` +
            results.map(r =>
                `  ${r.id},${r.text.slice(0, 100)}...,${r.final_score?.toFixed(3) || r.score?.toFixed(3)},${r.metadata?.timestamp || 'N/A'},${r.metadata?.source || 'vector'}`
            ).join('\n');

        const text = header + toonData;

        // Calcular tokens ahorrados (estimado)
        const jsonSize = JSON.stringify(results).length;
        const toonSize = text.length;
        const tokensSaved = jsonSize - toonSize;

        this.stats.total_tokens_saved += tokensSaved;

        console.log(`  💡 TOON: ${((tokensSaved / jsonSize) * 100).toFixed(1)}% tokens saved`);

        return {
            text: text,
            format: 'toon',
            tokens_saved: tokensSaved,
            original_size: jsonSize,
            optimized_size: toonSize
        };
    }

    /**
     * Formatear con JSON tradicional
     * @private
     */
    async _formatWithJSON(results, query) {
        const contextText = `Información relevante para: "${query}"\n\n` +
            results.map((r, i) =>
                `${i + 1}. ${r.text}\n   (Score: ${r.final_score?.toFixed(3) || r.score?.toFixed(3)}, Source: ${r.metadata?.source || 'vector'})`
            ).join('\n\n');

        return {
            text: contextText,
            format: 'json',
            tokens_saved: 0
        };
    }

    /**
     * Construir query expandida con contexto de conversación
     * @private
     */
    _buildExpandedQuery(query, conversationHistory) {
        if (conversationHistory.length === 0) {
            return query;
        }

        // Tomar últimos 2 mensajes como contexto
        const recentContext = conversationHistory
            .slice(-2)
            .map(msg => msg.content)
            .join(' ');

        return `${recentContext} ${query}`;
    }
}

// Exportar para uso global
if (typeof window !== 'undefined') {
    window.RAGClient = RAGClient;
}
