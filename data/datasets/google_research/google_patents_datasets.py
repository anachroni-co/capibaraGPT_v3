"""
Google Patents Datasets Manager for CapibaraGPT v3

Specialized manager for Google Patents Public Datasets including:
- 90+ million patent publications from 17+ countries
- USPTO full text and bibliographic data
- Patent research data with translations and similarity vectors
- BigQuery integration for large-scale patent analysis
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import datetime
import json

logger = logging.getLogger(__name__)

class GooglePatentsDatasets:
    """Manager for Google Patents Public Datasets."""

    def __init__(self):
        """Initialize the Google Patents datasets manager."""
        self.dataset_info = {
            "google_patents_public_data": {
                "name": "Google Patents Public Data",
                "description": "90+ million patent publications from 17 countries with US full text",
                "size": "Multi-TB",
                "provider": "IFI CLAIMS Patent Services & Google",
                "bigquery_dataset": "patents-public-data:patents",
                "update_frequency": "Quarterly",
                "coverage": {
                    "countries": 17,
                    "publications": 90000000,
                    "time_range": "1834-present"
                },
                "license": "CC BY 4.0",
                "url": "https://console.cloud.google.com/marketplace/details/google_patents_public_datasets/google-patents-public-data"
            },
            "google_patents_research_data": {
                "name": "Google Patents Research Data",
                "description": "Enhanced patent data with translations, similarity vectors, and extracted terms",
                "size": "Multi-TB",
                "provider": "Google Research",
                "bigquery_dataset": "patents-public-data:patents_research",
                "features": [
                    "english_translations",
                    "similarity_vectors",
                    "extracted_terms",
                    "word2vec_embeddings",
                    "lstm_model"
                ],
                "coverage": {
                    "abstracts_translated": 6000000,
                    "embedding_dimensions": 300
                }
            },
            "patents_view": {
                "name": "PatentsView",
                "description": "USPTO patents with detailed inventor, assignee, and citation data",
                "provider": "USPTO & Research Partners",
                "bigquery_dataset": "patents-public-data:patentsview",
                "features": [
                    "inventor_data",
                    "assignee_data",
                    "citation_networks",
                    "cpc_classifications",
                    "government_interest"
                ]
            },
            "usitc_investigations": {
                "name": "USITC 337 Investigations",
                "description": "US International Trade Commission patent infringement investigations",
                "provider": "USITC",
                "bigquery_dataset": "patents-public-data:usitc",
                "features": ["trade_complaints", "patent_disputes", "industry_classifications"]
            },
            "fda_orange_book": {
                "name": "FDA Orange Book",
                "description": "FDA approved drugs and their associated patents",
                "provider": "FDA",
                "bigquery_dataset": "patents-public-data:fda_orange_book",
                "features": ["approved_drugs", "drug_patents", "exclusivity_data"]
            }
        }

        # Patent classification systems
        self.classification_systems = {
            "cpc": {
                "name": "Cooperative Patent Classification",
                "description": "Joint classification system by EPO and USPTO",
                "sections": {
                    "A": "Human Necessities",
                    "B": "Performing Operations; Transporting",
                    "C": "Chemistry; Metallurgy",
                    "D": "Textiles; Paper",
                    "E": "Fixed Constructions",
                    "F": "Mechanical Engineering; Lighting; Heating",
                    "G": "Physics",
                    "H": "Electricity"
                }
            },
            "ipc": {
                "name": "International Patent Classification",
                "description": "WIPO administered classification system",
                "levels": ["section", "class", "subclass", "group", "subgroup"]
            },
            "uspc": {
                "name": "United States Patent Classification",
                "description": "Legacy USPTO classification system",
                "status": "deprecated"
            }
        }

        # BigQuery table schemas
        self.table_schemas = {
            "publications": {
                "publication_number": "STRING",
                "application_number": "STRING",
                "filing_date": "DATE",
                "publication_date": "DATE",
                "grant_date": "DATE",
                "title": "STRING",
                "abstract": "STRING",
                "claims": "STRING",
                "inventor": "REPEATED RECORD",
                "assignee": "REPEATED RECORD",
                "cpc": "REPEATED RECORD",
                "uspc": "REPEATED RECORD",
                "citation": "REPEATED RECORD"
            },
            "research_data": {
                "publication_number": "STRING",
                "title_translated": "STRING",
                "abstract_translated": "STRING",
                "similarity_vector": "REPEATED FLOAT",
                "top_terms": "REPEATED STRING",
                "embedding_vector": "REPEATED FLOAT"
            }
        }

        # Common search fields and operators
        self.search_fields = {
            "title": "Title text search",
            "abstract": "Abstract text search",
            "claims": "Claims text search",
            "inventor": "Inventor name search",
            "assignee": "Assignee/company search",
            "cpc_code": "CPC classification code",
            "filing_date": "Patent filing date",
            "publication_date": "Publication date",
            "grant_date": "Grant date",
            "citation_count": "Number of citations"
        }

    def get_available_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Get all available Google Patents datasets."""
        return self.dataset_info

    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific dataset."""
        return self.dataset_info.get(dataset_id)

    def get_bigquery_tables(self) -> Dict[str, str]:
        """Get BigQuery dataset references for all patent datasets."""
        return {
            dataset_id: info.get("bigquery_dataset", "")
            for dataset_id, info in self.dataset_info.items()
            if "bigquery_dataset" in info
        }

    def generate_bigquery_query(self, search_params: Dict[str, Any]) -> str:
        """
        Generate BigQuery SQL query for patent search.

        Args:
            search_params: Dictionary with search parameters

        Returns:
            SQL query string
        """
        base_table = "patents-public-data.patents.publications"

        # Base SELECT clause
        select_fields = [
            "publication_number",
            "title",
            "abstract",
            "filing_date",
            "publication_date",
            "inventor",
            "assignee",
            "cpc"
        ]

        query = f"SELECT {', '.join(select_fields)} FROM `{base_table}`"

        # Build WHERE clause with parameterized values
        where_conditions = []
        query_params = []

        if "title" in search_params:
            where_conditions.append("LOWER(title) LIKE CONCAT('%', LOWER(@title_param), '%')")
            query_params.append(("title_param", search_params["title"]))

        if "abstract" in search_params:
            where_conditions.append("LOWER(abstract) LIKE CONCAT('%', LOWER(@abstract_param), '%')")
            query_params.append(("abstract_param", search_params["abstract"]))

        if "assignee" in search_params:
            where_conditions.append("EXISTS(SELECT 1 FROM UNNEST(assignee) AS a WHERE LOWER(a.name) LIKE CONCAT('%', LOWER(@assignee_param), '%'))")
            query_params.append(("assignee_param", search_params["assignee"]))

        if "inventor" in search_params:
            where_conditions.append("EXISTS(SELECT 1 FROM UNNEST(inventor) AS i WHERE LOWER(i.name) LIKE CONCAT('%', LOWER(@inventor_param), '%'))")
            query_params.append(("inventor_param", search_params["inventor"]))

        if "cpc_code" in search_params:
            where_conditions.append("EXISTS(SELECT 1 FROM UNNEST(cpc) AS c WHERE c.code LIKE CONCAT(@cpc_param, '%'))")
            query_params.append(("cpc_param", search_params["cpc_code"]))

        if "filing_date_start" in search_params:
            where_conditions.append("filing_date >= @filing_start_param")
            query_params.append(("filing_start_param", search_params["filing_date_start"]))

        if "filing_date_end" in search_params:
            where_conditions.append("filing_date <= @filing_end_param")
            query_params.append(("filing_end_param", search_params["filing_date_end"]))

        if where_conditions:
            query += " WHERE " + " AND ".join(where_conditions)

        # Add ORDER BY (whitelist allowed columns to prevent injection)
        _ALLOWED_ORDER_COLUMNS = {"publication_date", "filing_date", "title"}
        if "order_by" in search_params:
            order_col = search_params["order_by"]
            if order_col in _ALLOWED_ORDER_COLUMNS:
                query += f" ORDER BY {order_col}"
            else:
                query += " ORDER BY publication_date DESC"
        else:
            query += " ORDER BY publication_date DESC"

        if "limit" in search_params:
            limit_val = int(search_params["limit"])  # ensure integer
            query += f" LIMIT {limit_val}"
        else:
            query += " LIMIT 1000"

        return query, query_params

    def get_patent_landscape_query(self, technology_area: str,
                                  years: Optional[List[int]] = None) -> str:
        """
        Generate query for patent landscape analysis.

        Args:
            technology_area: CPC code or technology description
            years: Optional list of years to analyze

        Returns:
            BigQuery SQL for landscape analysis
        """
        base_table = "patents-public-data.patents.publications"

        query = f"""
        SELECT
            EXTRACT(YEAR FROM filing_date) as filing_year,
            assignee_name,
            COUNT(*) as patent_count,
            ARRAY_AGG(DISTINCT cpc_code) as cpc_codes
        FROM (
            SELECT
                filing_date,
                a.name as assignee_name,
                c.code as cpc_code
            FROM `{base_table}`,
            UNNEST(assignee) AS a,
            UNNEST(cpc) AS c
            WHERE c.code LIKE '{technology_area}%'
        """

        if years:
            year_list = ",".join(map(str, years))
            query += f" AND EXTRACT(YEAR FROM filing_date) IN ({year_list})"

        query += """
        )
        GROUP BY filing_year, assignee_name
        HAVING patent_count >= 5
        ORDER BY filing_year DESC, patent_count DESC
        """

        return query

    def get_citation_analysis_query(self, patent_numbers: List[str]) -> str:
        """
        Generate query for citation analysis.

        Args:
            patent_numbers: List of patent numbers to analyze

        Returns:
            BigQuery SQL for citation analysis
        """
        patent_list = "','".join(patent_numbers)

        query = f"""
        WITH target_patents AS (
            SELECT publication_number, title, assignee
            FROM `patents-public-data.patents.publications`
            WHERE publication_number IN ('{patent_list}')
        ),
        forward_citations AS (
            SELECT
                c.publication_number as citing_patent,
                c.title as citing_title,
                ca.name as citing_assignee,
                cc.code as citing_cpc,
                t.publication_number as cited_patent,
                t.title as cited_title
            FROM `patents-public-data.patents.publications` c,
            UNNEST(c.citation) as cite,
            UNNEST(c.assignee) as ca,
            UNNEST(c.cpc) as cc
            JOIN target_patents t ON cite.publication_number = t.publication_number
        )
        SELECT
            cited_patent,
            cited_title,
            citing_assignee,
            citing_cpc,
            COUNT(*) as citation_count
        FROM forward_citations
        GROUP BY cited_patent, cited_title, citing_assignee, citing_cpc
        ORDER BY citation_count DESC
        """

        return query

    def get_inventor_analysis_query(self, inventor_name: str) -> str:
        """
        Generate query for inventor productivity analysis.

        Args:
            inventor_name: Name of inventor to analyze

        Returns:
            BigQuery SQL for inventor analysis
        """
        query = f"""
        SELECT
            EXTRACT(YEAR FROM filing_date) as filing_year,
            i.name as inventor_name,
            a.name as assignee_name,
            c.code as cpc_code,
            COUNT(*) as patent_count,
            ARRAY_AGG(DISTINCT title) as patent_titles
        FROM `patents-public-data.patents.publications`,
        UNNEST(inventor) AS i,
        UNNEST(assignee) AS a,
        UNNEST(cpc) AS c
        WHERE LOWER(i.name) LIKE '%{inventor_name.lower()}%'
        GROUP BY filing_year, inventor_name, assignee_name, cpc_code
        ORDER BY filing_year DESC, patent_count DESC
        """

        return query

    def get_technology_trend_query(self, cpc_codes: List[str],
                                  start_year: int = 2000) -> str:
        """
        Generate query for technology trend analysis.

        Args:
            cpc_codes: List of CPC codes to analyze
            start_year: Starting year for analysis

        Returns:
            BigQuery SQL for trend analysis
        """
        cpc_list = "','".join(cpc_codes)

        query = f"""
        SELECT
            EXTRACT(YEAR FROM filing_date) as filing_year,
            c.code as cpc_code,
            COUNT(*) as patent_count,
            COUNT(DISTINCT assignee_name) as unique_assignees,
            COUNT(DISTINCT inventor_name) as unique_inventors
        FROM (
            SELECT
                filing_date,
                c.code,
                a.name as assignee_name,
                i.name as inventor_name
            FROM `patents-public-data.patents.publications`,
            UNNEST(cpc) AS c,
            UNNEST(assignee) AS a,
            UNNEST(inventor) AS i
            WHERE c.code IN ('{cpc_list}')
            AND EXTRACT(YEAR FROM filing_date) >= {start_year}
        )
        GROUP BY filing_year, cpc_code
        ORDER BY filing_year DESC, patent_count DESC
        """

        return query

    def get_automated_landscaping_example(self) -> Dict[str, Any]:
        """
        Get example of automated patent landscaping setup.

        Returns:
            Dictionary with landscaping methodology
        """
        return {
            "methodology": "Automated Patent Landscaping (Abood, Feltenberger, 2016)",
            "approach": "Semi-supervised machine learning",
            "model": {
                "lstm": "Long Short-Term Memory neural networks",
                "word2vec": "Word embeddings trained on 6M patent abstracts",
                "embedding_dimensions": 300
            },
            "process": [
                "1. Define seed set of representative patents",
                "2. Extract text features (title, abstract, claims)",
                "3. Generate word2vec embeddings",
                "4. Train LSTM classifier on seed set",
                "5. Apply model to find similar patents",
                "6. Iterate and refine results"
            ],
            "github_repo": "https://github.com/google/patents-public-data",
            "example_notebook": "automated_patent_landscaping.ipynb"
        }

    def get_bigquery_costs(self) -> Dict[str, str]:
        """Get information about BigQuery usage costs."""
        return {
            "query_pricing": "Based on data processed (TB)",
            "storage_pricing": "Monthly storage costs",
            "free_tier": "1 TB queries + 10 GB storage per month",
            "estimated_cost_small_query": "$5-50 per TB processed",
            "optimization_tips": [
                "Use SELECT only needed columns",
                "Add date range filters",
                "Use LIMIT for testing",
                "Cache results for repeated queries",
                "Consider materialized views for complex queries"
            ]
        }

    def generate_sample_queries(self) -> Dict[str, str]:
        """Generate sample BigQuery queries for common use cases."""
        return {
            "basic_search": """
                SELECT publication_number, title, filing_date, assignee
                FROM `patents-public-data.patents.publications`
                WHERE LOWER(title) LIKE '%artificial intelligence%'
                AND EXTRACT(YEAR FROM filing_date) >= 2020
                LIMIT 100
            """,

            "technology_landscape": """
                SELECT
                    EXTRACT(YEAR FROM filing_date) as year,
                    a.name as company,
                    COUNT(*) as patents
                FROM `patents-public-data.patents.publications`,
                UNNEST(assignee) AS a,
                UNNEST(cpc) AS c
                WHERE c.code LIKE 'G06N%'  -- AI/Machine Learning
                AND EXTRACT(YEAR FROM filing_date) >= 2015
                GROUP BY year, company
                HAVING patents >= 10
                ORDER BY year DESC, patents DESC
            """,

            "citation_network": """
                SELECT
                    p.publication_number,
                    p.title,
                    cite.publication_number as cited_patent,
                    COUNT(*) as citation_count
                FROM `patents-public-data.patents.publications` p,
                UNNEST(p.citation) as cite
                WHERE p.assignee[OFFSET(0)].name LIKE '%Google%'
                GROUP BY p.publication_number, p.title, cited_patent
                ORDER BY citation_count DESC
            """,

            "inventor_productivity": """
                SELECT
                    i.name as inventor,
                    COUNT(*) as total_patents,
                    MIN(filing_date) as first_patent,
                    MAX(filing_date) as latest_patent,
                    COUNT(DISTINCT a.name) as companies_worked_with
                FROM `patents-public-data.patents.publications`,
                UNNEST(inventor) AS i,
                UNNEST(assignee) AS a
                GROUP BY inventor
                HAVING total_patents >= 50
                ORDER BY total_patents DESC
            """
        }

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about Google Patents datasets."""
        return {
            "total_publications": 90000000,
            "countries_covered": 17,
            "time_span": "1834-present",
            "update_frequency": "Quarterly",
            "full_text_coverage": "USPTO patents",
            "translation_coverage": "6M abstracts in English",
            "classification_systems": list(self.classification_systems.keys()),
            "bigquery_dataset_size": "Multi-TB",
            "estimated_query_cost": "$5-50 per TB processed",
            "key_features": [
                "Full patent text and metadata",
                "Citation networks",
                "Inventor and assignee data",
                "CPC/IPC classifications",
                "Machine translations",
                "Similarity vectors",
                "Government interest statements",
                "FDA drug-patent linkages"
            ]
        }

# Factory function
def get_google_patents_datasets() -> GooglePatentsDatasets:
    """Get Google Patents datasets manager."""
    return GooglePatentsDatasets()


class GooglePatentsDataset(GooglePatentsDatasets):
    """Backward-compatible alias for GooglePatentsDatasets."""
