// ============================================
// Detección de idioma basada en geolocalización
// ============================================
const SPANISH_SPEAKING_COUNTRIES = [
    'ES', 'MX', 'AR', 'CO', 'PE', 'VE', 'CL', 'EC', 'GT', 'CU', 
    'BO', 'DO', 'HN', 'PY', 'SV', 'NI', 'CR', 'PA', 'UY', 'GQ'
];

// Las traducciones se cargan desde translations.js

// Detectar país del usuario
async function detectUserCountry() {
    try {
        // Primero intentar con la API de geolocalización
        const response = await fetch('https://ipapi.co/json/');
        const data = await response.json();
        return data.country_code;
    } catch (error) {
        // Fallback: usar el idioma del navegador
        const browserLang = navigator.language || navigator.userLanguage;
        if (browserLang.startsWith('es')) {
            return 'ES'; // Asumimos España/Latinoamérica
        }
        return 'US'; // Por defecto inglés
    }
}

// Cambiar idioma de la página
function changeLanguage(lang) {
    document.documentElement.setAttribute('lang', lang);
    document.documentElement.setAttribute('data-lang', lang);
    
    // Guardar preferencia
    localStorage.setItem('preferred-language', lang);
    
    // Actualizar elementos con data-i18n
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        const t = translations[lang];
        
        if (t && t[key]) {
            // Si tiene hijos específicos a traducir, solo cambia el texto directo
            if (element.querySelector('[data-i18n]')) {
                // Es un contenedor con hijos traducibles, no tocar
                return;
            }
            
            // Verificar si es HTML o texto plano
            if (t[key].includes('<')) {
                element.innerHTML = t[key];
            } else {
                element.textContent = t[key];
            }
        }
    });
    
    // Actualizar placeholders
    document.querySelectorAll('[data-i18n-placeholder]').forEach(element => {
        const key = element.getAttribute('data-i18n-placeholder');
        const t = translations[lang];
        
        if (t && t[key]) {
            element.placeholder = t[key];
        }
    });
    
    // Actualizar botones de idioma
    document.querySelectorAll('.lang-btn').forEach(btn => {
        if (btn.getAttribute('data-lang') === lang) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    console.log(`✓ Idioma cambiado a: ${lang === 'es' ? 'Español' : 'English'}`);
}

// Inicializar idioma según geolocalización
async function initializeLanguage() {
    // Primero verificar si hay preferencia guardada
    const savedLang = localStorage.getItem('preferred-language');
    if (savedLang) {
        changeLanguage(savedLang);
        return;
    }
    
    // Si no, detectar por país
    const countryCode = await detectUserCountry();
    const shouldUseSpanish = SPANISH_SPEAKING_COUNTRIES.includes(countryCode);
    const language = shouldUseSpanish ? 'es' : 'en';
    
    changeLanguage(language);
    
    console.log(`🌍 País detectado: ${countryCode}`);
    console.log(`🗣️ Idioma seleccionado: ${language === 'es' ? 'Español' : 'English'}`);
}

// Ejecutar detección de idioma al cargar
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeLanguage);
} else {
    initializeLanguage();
}

// ============================================
// Navegación móvil
// ============================================
const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
const navMenu = document.querySelector('.nav-menu');

if (mobileMenuToggle) {
    mobileMenuToggle.addEventListener('click', () => {
        navMenu.classList.toggle('active');
        
        if (navMenu.classList.contains('active')) {
            navMenu.style.display = 'flex';
            navMenu.style.flexDirection = 'column';
            navMenu.style.position = 'absolute';
            navMenu.style.top = '100%';
            navMenu.style.left = '0';
            navMenu.style.right = '0';
            navMenu.style.background = 'var(--bg-secondary)';
            navMenu.style.padding = '1rem';
            navMenu.style.borderTop = '1px solid var(--border-color)';
        } else {
            navMenu.style.display = '';
        }
    });
}

// ============================================
// Scroll suave para navegación
// ============================================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        const href = this.getAttribute('href');
        
        // Ignorar si es solo '#'
        if (href === '#' || href === '#github') return;
        
        e.preventDefault();
        const target = document.querySelector(href);
        
        if (target) {
            const offset = 80; // Altura del navbar
            const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - offset;
            
            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
            });
            
            // Cerrar menú móvil si está abierto
            if (navMenu && navMenu.classList.contains('active')) {
                navMenu.classList.remove('active');
                navMenu.style.display = '';
            }
        }
    });
});

// ============================================
// Efecto de scroll en navbar
// ============================================
const navbar = document.querySelector('.navbar');
let lastScroll = 0;

window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;
    
    if (currentScroll > 100) {
        navbar.style.boxShadow = 'var(--shadow-lg)';
    } else {
        navbar.style.boxShadow = 'none';
    }
    
    lastScroll = currentScroll;
});

// ============================================
// Animación de entrada para elementos
// ============================================
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observar elementos que deben animarse al entrar en vista
const animatedElements = document.querySelectorAll(
    '.feature-card, .quickstart-step, .script-card, .monitor-card, .trouble-card, .doc-card, .arch-box'
);

animatedElements.forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
});

// ============================================
// Copiar código
// ============================================
window.copyCode = function(button) {
    const codeBlock = button.closest('.code-block');
    const code = codeBlock.querySelector('code').textContent;
    
    navigator.clipboard.writeText(code).then(() => {
        const originalText = button.textContent;
        button.textContent = '✓ Copiado';
        button.style.background = 'var(--success)';
        
        setTimeout(() => {
            button.textContent = originalText;
            button.style.background = '';
        }, 2000);
    }).catch(err => {
        console.error('Error al copiar:', err);
        button.textContent = '✗ Error';
        button.style.background = 'var(--error)';
        
        setTimeout(() => {
            button.textContent = 'Copiar';
            button.style.background = '';
        }, 2000);
    });
};

// ============================================
// Contador animado para stats
// ============================================
const animateValue = (element, start, end, duration, suffix = '') => {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.round(current) + suffix;
    }, 16);
};

// Animar stats cuando entren en vista
const statsObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting && !entry.target.classList.contains('animated')) {
            entry.target.classList.add('animated');
            
            // Aquí podrías añadir animaciones numéricas si los stats fueran números
            entry.target.style.transform = 'scale(1.05)';
            setTimeout(() => {
                entry.target.style.transform = 'scale(1)';
            }, 300);
        }
    });
}, { threshold: 0.5 });

document.querySelectorAll('.stat').forEach(stat => {
    stat.style.transition = 'transform 0.3s ease';
    statsObserver.observe(stat);
});

// ============================================
// Efecto parallax suave en hero
// ============================================
const heroBackground = document.querySelector('.hero-background');

if (heroBackground) {
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const rate = scrolled * 0.5;
        heroBackground.style.transform = `translateY(${rate}px)`;
    });
}

// ============================================
// Loading state para enlaces externos
// ============================================
document.querySelectorAll('a[href^="http"]').forEach(link => {
    link.addEventListener('click', function(e) {
        if (!this.hasAttribute('target')) {
            this.setAttribute('target', '_blank');
            this.setAttribute('rel', 'noopener noreferrer');
        }
    });
});

// ============================================
// Indicador de progreso de lectura
// ============================================
const createReadingProgress = () => {
    const progressBar = document.createElement('div');
    progressBar.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        height: 3px;
        background: var(--gradient-primary);
        z-index: 9999;
        transition: width 0.1s ease;
    `;
    document.body.appendChild(progressBar);
    
    window.addEventListener('scroll', () => {
        const windowHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const scrolled = (window.pageYOffset / windowHeight) * 100;
        progressBar.style.width = scrolled + '%';
    });
};

createReadingProgress();

// ============================================
// Efecto de hover en cards con brillo
// ============================================
document.querySelectorAll('.feature-card, .doc-card, .script-card').forEach(card => {
    card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        card.style.setProperty('--mouse-x', `${x}px`);
        card.style.setProperty('--mouse-y', `${y}px`);
    });
});

// ============================================
// Lazy loading de imágenes (si se añaden más tarde)
// ============================================
if ('IntersectionObserver' in window) {
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.classList.add('loaded');
                observer.unobserve(img);
            }
        });
    });
    
    document.querySelectorAll('img[data-src]').forEach(img => {
        imageObserver.observe(img);
    });
}

// ============================================
// Selector de idioma manual (opcional)
// ============================================
window.switchLanguage = function(lang) {
    if (lang === 'es' || lang === 'en') {
        changeLanguage(lang);
    }
};

// Añadir al objeto window para acceso global
window.capibaraLanguage = {
    switch: switchLanguage,
    current: () => document.documentElement.getAttribute('data-lang')
};

// ============================================
// Console easter egg
// ============================================
console.log('%c🦫 capibara6', 'font-size: 24px; font-weight: bold; background: linear-gradient(135deg, #6366f1 0%, #ec4899 50%, #14b8a6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;');
console.log('%c¡Gracias por tu interés en capibara6!', 'font-size: 14px; color: #94a3b8;');
console.log('%cDesarrollado por Anachroni s.coop - www.anachroni.co', 'font-size: 12px; color: #cbd5e1;');
console.log('%cPara cambiar idioma: capibaraLanguage.switch("es") o capibaraLanguage.switch("en")', 'font-size: 11px; color: #64748b;');

// ============================================
// Función de utilidad para detectar dark mode del sistema
// ============================================
const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');

prefersDarkScheme.addEventListener('change', (e) => {
    // Aquí podrías añadir lógica para cambiar entre temas
    console.log('Preferencia de tema cambiada:', e.matches ? 'dark' : 'light');
});

// ============================================
// Inicialización
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('✅ Capibara6 website loaded successfully');
    
    // Añadir clase loaded al body para posibles animaciones CSS
    setTimeout(() => {
        document.body.classList.add('loaded');
    }, 100);
});

