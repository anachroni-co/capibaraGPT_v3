"""
N8N Templates Manager para Capibara6
Gestiona plantillas de workflows pre-configuradas
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path
import requests

class N8NTemplatesManager:
    """Gestor de plantillas de workflows n8n"""

    def __init__(self, templates_dir: str = None):
        if templates_dir is None:
            templates_dir = os.path.join(
                os.path.dirname(__file__),
                'data/n8n/workflows/templates'
            )
        self.templates_dir = Path(templates_dir)
        self.catalog_file = self.templates_dir / 'catalog.json'
        self.catalog = self._load_catalog()

    def _load_catalog(self) -> Dict:
        """Carga el catálogo de plantillas"""
        try:
            with open(self.catalog_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'version': '1.0.0',
                'categories': [],
                'statistics': {}
            }

    def get_catalog(self) -> Dict:
        """Obtiene el catálogo completo de plantillas"""
        return self.catalog

    def get_categories(self) -> List[Dict]:
        """Obtiene todas las categorías"""
        return self.catalog.get('categories', [])

    def get_category(self, category_id: str) -> Optional[Dict]:
        """Obtiene una categoría específica"""
        for category in self.catalog.get('categories', []):
            if category['id'] == category_id:
                return category
        return None

    def get_template(self, template_id: str) -> Optional[Dict]:
        """Obtiene información de una plantilla específica"""
        for category in self.catalog.get('categories', []):
            for template in category.get('templates', []):
                if template['id'] == template_id:
                    return template
        return None

    def get_recommended_templates(self) -> List[Dict]:
        """Obtiene plantillas recomendadas para Capibara6"""
        recommended = []
        for category in self.catalog.get('categories', []):
            for template in category.get('templates', []):
                if template.get('capibara6_integration', {}).get('recommended', False):
                    recommended.append({
                        **template,
                        'category_id': category['id'],
                        'category_name': category['name']
                    })

        # Ordenar por prioridad
        recommended.sort(
            key=lambda x: x.get('capibara6_integration', {}).get('priority', 99)
        )
        return recommended

    def get_templates_by_priority(self, priority: int = 1) -> List[Dict]:
        """Obtiene plantillas por nivel de prioridad"""
        templates = []
        for category in self.catalog.get('categories', []):
            for template in category.get('templates', []):
                template_priority = template.get('capibara6_integration', {}).get('priority', 99)
                if template_priority == priority:
                    templates.append({
                        **template,
                        'category_id': category['id'],
                        'category_name': category['name']
                    })
        return templates

    def download_template(self, template_id: str) -> Optional[Dict]:
        """
        Descarga una plantilla desde GitHub o carga desde local

        Args:
            template_id: ID de la plantilla

        Returns:
            Dict con el workflow JSON o None si falla
        """
        template = self.get_template(template_id)
        if not template:
            return None

        source_url = template.get('source_url')

        # Si es una plantilla custom, cargar desde archivo local
        if template.get('capibara6_integration', {}).get('custom', False):
            local_file = self.templates_dir / f"capibara6-{template_id}.json"
            if local_file.exists():
                with open(local_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None

        # Si no, descargar desde GitHub
        if source_url and source_url != 'custom':
            try:
                response = requests.get(source_url, timeout=10)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                print(f"Error descargando plantilla {template_id}: {e}")
                return None

        return None

    def get_template_requirements(self, template_id: str) -> List[str]:
        """Obtiene los requisitos de una plantilla"""
        template = self.get_template(template_id)
        if template:
            return template.get('requires', [])
        return []

    def search_templates(self, query: str) -> List[Dict]:
        """
        Busca plantillas por palabra clave

        Args:
            query: Término de búsqueda

        Returns:
            Lista de plantillas que coinciden
        """
        query_lower = query.lower()
        results = []

        for category in self.catalog.get('categories', []):
            for template in category.get('templates', []):
                # Buscar en nombre, descripción y casos de uso
                if (query_lower in template['name'].lower() or
                    query_lower in template['description'].lower() or
                    any(query_lower in use_case.lower()
                        for use_case in template.get('use_cases', []))):
                    results.append({
                        **template,
                        'category_id': category['id'],
                        'category_name': category['name']
                    })

        return results

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas del catálogo"""
        return self.catalog.get('statistics', {})

    def import_template_to_n8n(
        self,
        template_id: str,
        n8n_url: str = 'http://n8n:5678',
        api_key: Optional[str] = None
    ) -> Dict:
        """
        Importa una plantilla directamente a n8n

        Args:
            template_id: ID de la plantilla
            n8n_url: URL de la instancia n8n
            api_key: API key de n8n (si está habilitada la autenticación)

        Returns:
            Dict con resultado de la importación
        """
        workflow = self.download_template(template_id)
        if not workflow:
            return {
                'success': False,
                'error': 'Plantilla no encontrada o error al descargar'
            }

        # Preparar headers
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['X-N8N-API-KEY'] = api_key

        try:
            # Importar workflow a n8n
            response = requests.post(
                f"{n8n_url}/api/v1/workflows",
                json=workflow,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()

            return {
                'success': True,
                'workflow_id': response.json().get('id'),
                'message': f'Workflow "{workflow["name"]}" importado exitosamente'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error importando a n8n: {str(e)}'
            }


# Instancia global
templates_manager = N8NTemplatesManager()


def get_templates_catalog():
    """Obtiene el catálogo completo"""
    return templates_manager.get_catalog()


def get_recommended_templates():
    """Obtiene plantillas recomendadas"""
    return templates_manager.get_recommended_templates()


def get_template_details(template_id: str):
    """Obtiene detalles de una plantilla"""
    template = templates_manager.get_template(template_id)
    if template:
        # Añadir información de requisitos
        template['requirements_details'] = {
            'apis': [req for req in template.get('requires', []) if 'API' in req],
            'credentials': [req for req in template.get('requires', []) if 'API' not in req],
            'integration_endpoints': template.get('capibara6_integration', {}).get('endpoints', [])
        }
    return template


def search_templates(query: str):
    """Busca plantillas"""
    return templates_manager.search_templates(query)


def download_template_json(template_id: str):
    """Descarga el JSON de una plantilla"""
    return templates_manager.download_template(template_id)


def import_template(template_id: str, n8n_url: str = None, api_key: str = None):
    """Importa una plantilla a n8n"""
    if n8n_url is None:
        # Intentar detectar la URL de n8n según el entorno
        n8n_url = os.getenv('N8N_URL', 'http://n8n:5678')

    return templates_manager.import_template_to_n8n(
        template_id=template_id,
        n8n_url=n8n_url,
        api_key=api_key
    )
