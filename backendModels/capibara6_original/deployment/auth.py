#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authentication & Authorization - Sistema de autenticación y autorización.
"""

import logging
import os
import time
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """Roles de usuario."""
    ADMIN = "admin"
    USER = "user"
    DEVELOPER = "developer"
    READONLY = "readonly"

class Permission(Enum):
    """Permisos del sistema."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    API_ACCESS = "api_access"
    GRAPHQL_ACCESS = "graphql_access"

@dataclass
class User:
    """Usuario del sistema."""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: List[Permission]
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool
    api_key: Optional[str] = None

@dataclass
class APIKey:
    """Clave API."""
    key_id: str
    key_hash: str
    user_id: str
    name: str
    permissions: List[Permission]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    is_active: bool

class AuthenticationManager:
    """Gestor de autenticación."""
    
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY", "capibara6_secret_key_change_in_production")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
        
        # Contexto de encriptación de contraseñas
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Almacenamiento en memoria (en producción usar base de datos)
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.refresh_tokens: Dict[str, str] = {}  # token -> user_id
        
        # Inicializar usuarios por defecto
        self._create_default_users()
        
        logger.info("AuthenticationManager inicializado")
    
    def _create_default_users(self):
        """Crea usuarios por defecto."""
        # Usuario admin
        admin_user = User(
            user_id="admin_001",
            username="admin",
            email="admin@capibara6.com",
            role=UserRole.ADMIN,
            permissions=[p for p in Permission],
            created_at=datetime.now(),
            last_login=None,
            is_active=True
        )
        self.users[admin_user.user_id] = admin_user
        
        # Usuario developer
        dev_user = User(
            user_id="dev_001",
            username="developer",
            email="dev@capibara6.com",
            role=UserRole.DEVELOPER,
            permissions=[Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.API_ACCESS, Permission.GRAPHQL_ACCESS],
            created_at=datetime.now(),
            last_login=None,
            is_active=True
        )
        self.users[dev_user.user_id] = dev_user
        
        # Usuario readonly
        readonly_user = User(
            user_id="readonly_001",
            username="readonly",
            email="readonly@capibara6.com",
            role=UserRole.READONLY,
            permissions=[Permission.READ, Permission.API_ACCESS],
            created_at=datetime.now(),
            last_login=None,
            is_active=True
        )
        self.users[readonly_user.user_id] = readonly_user
        
        logger.info(f"Creados {len(self.users)} usuarios por defecto")
    
    def hash_password(self, password: str) -> str:
        """Encripta una contraseña."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verifica una contraseña."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """Crea un token de acceso JWT."""
        to_encode = {
            "sub": user_id,
            "type": "access",
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + (expires_delta or timedelta(minutes=self.access_token_expire_minutes))
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, user_id: str) -> str:
        """Crea un token de refresh."""
        to_encode = {
            "sub": user_id,
            "type": "refresh",
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        self.refresh_tokens[encoded_jwt] = user_id
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verifica un token JWT."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Autentica un usuario."""
        # En un entorno real, esto buscaría en la base de datos
        for user in self.users.values():
            if user.username == username and user.is_active:
                # Para usuarios por defecto, usar contraseñas simples
                if username == "admin" and password == "admin123":
                    return user
                elif username == "developer" and password == "dev123":
                    return user
                elif username == "readonly" and password == "readonly123":
                    return user
        
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Obtiene un usuario por ID."""
        return self.users.get(user_id)
    
    def create_api_key(self, user_id: str, name: str, permissions: List[Permission], expires_days: Optional[int] = None) -> str:
        """Crea una nueva clave API."""
        # Generar clave API
        api_key = f"cap6_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Crear objeto APIKey
        key_id = f"key_{int(time.time() * 1000)}"
        expires_at = None
        if expires_days:
            expires_at = datetime.now() + timedelta(days=expires_days)
        
        api_key_obj = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            permissions=permissions,
            created_at=datetime.now(),
            expires_at=expires_at,
            last_used=None,
            is_active=True
        )
        
        self.api_keys[key_hash] = api_key_obj
        
        logger.info(f"API key creada para usuario {user_id}: {name}")
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[Tuple[User, APIKey]]:
        """Verifica una clave API."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash not in self.api_keys:
            return None
        
        api_key_obj = self.api_keys[key_hash]
        
        # Verificar si está activa
        if not api_key_obj.is_active:
            return None
        
        # Verificar expiración
        if api_key_obj.expires_at and datetime.now() > api_key_obj.expires_at:
            return None
        
        # Obtener usuario
        user = self.get_user_by_id(api_key_obj.user_id)
        if not user or not user.is_active:
            return None
        
        # Actualizar último uso
        api_key_obj.last_used = datetime.now()
        
        return user, api_key_obj
    
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoca una clave API."""
        for key_hash, api_key_obj in self.api_keys.items():
            if api_key_obj.key_id == key_id:
                api_key_obj.is_active = False
                logger.info(f"API key revocada: {key_id}")
                return True
        return False
    
    def check_permission(self, user: User, permission: Permission) -> bool:
        """Verifica si un usuario tiene un permiso."""
        return permission in user.permissions
    
    def get_user_permissions(self, user_id: str) -> List[Permission]:
        """Obtiene los permisos de un usuario."""
        user = self.get_user_by_id(user_id)
        if user:
            return user.permissions
        return []


class RateLimiter:
    """Sistema de rate limiting."""
    
    def __init__(self):
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        self.default_limits = {
            "api": {"requests": 100, "window": 60},  # 100 requests por minuto
            "graphql": {"requests": 200, "window": 60},  # 200 requests por minuto
            "batch": {"requests": 10, "window": 60},  # 10 batches por minuto
        }
        
        logger.info("RateLimiter inicializado")
    
    def is_allowed(self, identifier: str, endpoint: str = "api") -> Tuple[bool, Dict[str, Any]]:
        """Verifica si una request está permitida."""
        current_time = time.time()
        window = self.default_limits.get(endpoint, self.default_limits["api"])["window"]
        max_requests = self.default_limits.get(endpoint, self.default_limits["api"])["requests"]
        
        # Limpiar entradas antiguas
        self._cleanup_old_entries(current_time, window)
        
        # Obtener o crear entrada para el identificador
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = {
                "requests": [],
                "endpoint": endpoint
            }
        
        # Agregar request actual
        self.rate_limits[identifier]["requests"].append(current_time)
        
        # Contar requests en la ventana
        window_start = current_time - window
        recent_requests = [
            req_time for req_time in self.rate_limits[identifier]["requests"]
            if req_time >= window_start
        ]
        
        self.rate_limits[identifier]["requests"] = recent_requests
        
        # Verificar límite
        is_allowed = len(recent_requests) <= max_requests
        
        # Calcular información de rate limit
        remaining = max(0, max_requests - len(recent_requests))
        reset_time = int(current_time + window)
        
        rate_limit_info = {
            "limit": max_requests,
            "remaining": remaining,
            "reset": reset_time,
            "window": window
        }
        
        return is_allowed, rate_limit_info
    
    def _cleanup_old_entries(self, current_time: float, window: int):
        """Limpia entradas antiguas."""
        cutoff_time = current_time - (window * 2)  # Mantener 2 ventanas
        
        for identifier in list(self.rate_limits.keys()):
            self.rate_limits[identifier]["requests"] = [
                req_time for req_time in self.rate_limits[identifier]["requests"]
                if req_time >= cutoff_time
            ]
            
            # Eliminar entradas vacías
            if not self.rate_limits[identifier]["requests"]:
                del self.rate_limits[identifier]
    
    def get_rate_limit_info(self, identifier: str, endpoint: str = "api") -> Dict[str, Any]:
        """Obtiene información de rate limit sin incrementar contador."""
        current_time = time.time()
        window = self.default_limits.get(endpoint, self.default_limits["api"])["window"]
        max_requests = self.default_limits.get(endpoint, self.default_limits["api"])["requests"]
        
        if identifier not in self.rate_limits:
            return {
                "limit": max_requests,
                "remaining": max_requests,
                "reset": int(current_time + window),
                "window": window
            }
        
        # Contar requests en la ventana
        window_start = current_time - window
        recent_requests = [
            req_time for req_time in self.rate_limits[identifier]["requests"]
            if req_time >= window_start
        ]
        
        remaining = max(0, max_requests - len(recent_requests))
        reset_time = int(current_time + window)
        
        return {
            "limit": max_requests,
            "remaining": remaining,
            "reset": reset_time,
            "window": window
        }


# Instancias globales
auth_manager = AuthenticationManager()
rate_limiter = RateLimiter()


def get_current_user(token: str) -> User:
    """Obtiene el usuario actual desde el token."""
    payload = auth_manager.verify_token(token)
    user_id = payload.get("sub")
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    user = auth_manager.get_user_by_id(user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    return user


def get_current_user_from_api_key(api_key: str) -> Tuple[User, APIKey]:
    """Obtiene el usuario actual desde la clave API."""
    result = auth_manager.verify_api_key(api_key)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return result


def require_permission(permission: Permission):
    """Decorator para requerir un permiso específico."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # En un entorno real, esto verificaría el permiso del usuario actual
            return func(*args, **kwargs)
        return wrapper
    return decorator


def check_rate_limit(identifier: str, endpoint: str = "api") -> Dict[str, Any]:
    """Verifica rate limit y devuelve información."""
    is_allowed, rate_limit_info = rate_limiter.is_allowed(identifier, endpoint)
    
    if not is_allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(rate_limit_info["limit"]),
                "X-RateLimit-Remaining": str(rate_limit_info["remaining"]),
                "X-RateLimit-Reset": str(rate_limit_info["reset"])
            }
        )
    
    return rate_limit_info


if __name__ == "__main__":
    # Test del sistema de autenticación
    logging.basicConfig(level=logging.INFO)
    
    # Test de autenticación
    user = auth_manager.authenticate_user("admin", "admin123")
    if user:
        print(f"Usuario autenticado: {user.username} ({user.role.value})")
        
        # Crear token
        token = auth_manager.create_access_token(user.user_id)
        print(f"Token creado: {token[:50]}...")
        
        # Verificar token
        payload = auth_manager.verify_token(token)
        print(f"Token verificado: {payload}")
        
        # Crear API key
        api_key = auth_manager.create_api_key(
            user.user_id,
            "Test API Key",
            [Permission.READ, Permission.WRITE],
            expires_days=30
        )
        print(f"API Key creada: {api_key}")
        
        # Verificar API key
        user_from_key, api_key_obj = auth_manager.verify_api_key(api_key)
        print(f"API Key verificada: {user_from_key.username}")
    
    # Test de rate limiting
    identifier = "test_user"
    for i in range(5):
        is_allowed, info = rate_limiter.is_allowed(identifier, "api")
        print(f"Request {i+1}: {is_allowed}, remaining: {info['remaining']}")
    
    print("Tests de autenticación completados")
