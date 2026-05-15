#!/usr/bin/env python3
"""Generate synthetic LoRA training data for cunca-gestoria (Galician gestoria/asesoría).

Produces JSONL with {"prompt", "response"} pairs covering:
  - ATRIGA / Xunta tax procedures (IRPF, IXV, transmisións)
  - Administrative procedures (Xunta sede electrónica)
  - Social Security / labor (alta/baixa, contratos)
  - Company formation / mercantil (SL, autónomo)
  - Document drafting in Galician (contratos, recursos, actas)

Supports both Galician (gl) and Spanish (es) prompts — real gestorias
handle both. Responses are always formal administrative Galician/Spanish.

Usage:
    python scripts/generate_gestoria_data.py \
        --output data/finetune/gestoria_gl.jsonl \
        --n 2000 --seed 42

    python scripts/generate_gestoria_data.py \
        --output data/finetune/gestoria_gl.jsonl \
        --n 2000 --lang gl   # Galician-only prompts
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import textwrap
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gestoria_gen")


TEMPLATES: list[dict] = [

    {
        "category": "atriga_irpf",
        "lang": "gl",
        "prompt": "Como declaro os rendementos do traballo no IRPF sendo residente en Galicia?",
        "response": textwrap.dedent("""\
            ## Declaración do IRPF — Rendementos do traballo (Galicia)

            **Obriga de declarar:** Están obrigados a declarar os contribuíntes con rendementos
            do traballo superiores a **22.000 €** brutos anuais dun só pagador, ou **15.000 €**
            con máis dun pagador cando o segundo supere os 1.500 €.

            **Prazo:** Campaña xeral abril–xuño de cada exercicio.

            **Pasos:**
            1. Obtén o borrador en **Axencia Tributaria** (Renda WEB) ou en **ATRIGA** para
               tributos autonómicos (dedución por familia numerosa, discapacidade, etc.).
            2. Revisa os datos do certificado de retencións do teu empregador (modelo 190).
            3. Aplica as **deducións autonómicas de Galicia**:
               - Nacemento/adopción: 300 € por fillo.
               - Familia numerosa: ata 750 € segundo categoría.
               - Aluguer de vivenda habitual: 15% das cotas, máx. 300 €.
               - Investimento en empresas de nova creación: 30%, máx. 6.000 €.
            4. Presenta telemáticamente ou en oficina da AXENCIA TRIBUTARIA / ATRIGA.

            **Documentación necesaria:**
            - DNI / NIE
            - Certificado de retencións do empregador
            - Xustificantes de deducións aplicadas

            Para máis información: atriga.gal · Teléfono ATRIGA: 981 545 555
            """),
    },
    {
        "category": "atriga_irpf",
        "lang": "es",
        "prompt": "¿Qué deducciones autonómicas de Galicia puedo aplicar en el IRPF?",
        "response": textwrap.dedent("""\
            ## Deducciones autonómicas de Galicia en el IRPF

            La Comunidad Autónoma de Galicia establece las siguientes deducciones en la
            cuota autonómica del IRPF (normativa vigente Ley 9/2008 y modificaciones):

            | Concepto | Importe | Límite |
            |---|---|---|
            | Nacimiento/adopción 1.er hijo | 300 € | — |
            | Nacimiento/adopción 2.º hijo | 360 € | — |
            | Nacimiento/adopción 3.er hijo o más | 1.200 € | — |
            | Familia numerosa general | 250 € | — |
            | Familia numerosa especial | 750 € | — |
            | Alquiler vivienda habitual (≤35 años o discapacidad) | 15% cuotas | 300 € |
            | Inversión empresa nueva creación | 30% | 6.000 € |
            | Gastos guardería (0-3 años) | 30% | 400 € |
            | Cuidado familiares discapacitados | 10% | 600 € |

            **Requisito general:** residencia fiscal en Galicia durante el ejercicio.

            Fuente: ATRIGA — atriga.gal/tributos/irpf/
            """),
    },
    {
        "category": "atriga_ixv",
        "lang": "gl",
        "prompt": "Que é o IXV e quen ten que pagalo en Galicia?",
        "response": textwrap.dedent("""\
            ## Imposto sobre as Vendas Retallistas de Determinados Hidrocarburos (IXV)

            O **IXV** é un tributo cedido ás Comunidades Autónomas que grava as vendas
            a retallo de gasolinas, gasóleos, fuelóleos e outros hidrocarburos.

            **Suxeitos pasivos:** Os vendedores a retallo (gasolineiras, distribuidores)
            son os obrigados tributarios. O consumidor final non presenta declaración.

            **Xestión en Galicia:** Corresponde á **ATRIGA** (Axencia Tributaria de Galicia).

            **Modelos:**
            - Modelo 569: declaración-liquidación trimestral
            - Modelo 570: declaración anual de operacións

            Máis info: atriga.gal/tributos/ixv/
            """),
    },
    {
        "category": "atriga_transmisions",
        "lang": "gl",
        "prompt": "Canto hai que pagar polo Imposto de Transmisións Patrimoniais nunha compravenda en Galicia?",
        "response": textwrap.dedent("""\
            ## Imposto de Transmisións Patrimoniais (ITP) — Galicia

            **Tipo xeral en Galicia:** **10%** sobre o valor de transmisión.

            **Tipos reducidos:**
            - **8%** para familias numerosas (vivenda habitual, valor ≤ 200.000 €).
            - **8%** para menores de 36 anos (vivenda habitual, valor ≤ 150.000 €,
              base impoñible IRPF ≤ 22.000 €).
            - **4%** para persoas con discapacidade ≥ 65% (vivenda habitual).

            **Prazo:** 30 días hábiles desde a escritura.
            **Modelo:** 600 — ATRIGA ou telemático.

            **Exemplo:** Piso de 120.000 € → ITP = 120.000 × 10% = **12.000 €**
            """),
    },
    {
        "category": "atriga_sucesions",
        "lang": "gl",
        "prompt": "Que bonificacións existen no Imposto de Sucesións en Galicia?",
        "response": textwrap.dedent("""\
            ## Imposto sobre Sucesións e Doazóns — Bonificacións en Galicia

            **Grupo I e II (descendentes, ascendentes, cónxuxe):**
            - **Redución de 1.000.000 €** por herdeiro entre cónxuxes, fillos e netos.
            - Na práctica, herdanzas familiares normais quedan con cota cero.

            **Vivenda habitual:**
            - Redución adicional do **95%**, límite de 600.000 € por herdeiro.
            - Requírese manter a vivenda 5 anos.

            **Empresa familiar:**
            - Redución do **99%** do valor de negocio familiar ou participacións.

            **Prazo:** 6 meses desde o falecemento (prorrogable 6 meses máis).
            Modelo: 650 · Oficinas ATRIGA ou telemático.
            """),
    },
    {
        "category": "xunta_procedemento",
        "lang": "gl",
        "prompt": "Como solicito unha axuda para a rehabilitación da miña vivenda a través da Xunta de Galicia?",
        "response": textwrap.dedent("""\
            ## Axudas á rehabilitación de vivendas — Xunta de Galicia

            O **IGVS** xestiona programas de axudas para rehabilitación enmarcados no
            Plan Estatal de Vivenda.

            **Procedemento xeral (VI408A):**
            1. Accede a **sede.xunta.gal** → VI408A.
            2. Identifícate con certificado dixital, Chave365 ou eIDAS.
            3. Achega: proxecto técnico, escritura de propiedade, DNI/NIE e declaracións
               de renda da unidade familiar.
            4. Presenta e garda o xustificante de rexistro.

            **Prazo:** Convocatoria anual, xeralmente xaneiro–marzo.
            Consulta: igvs.xunta.gal · Tel.: 981 541 900
            """),
    },
    {
        "category": "xunta_procedemento",
        "lang": "es",
        "prompt": "¿Cómo presento un recurso de alzada ante la Xunta de Galicia?",
        "response": textwrap.dedent("""\
            ## Recurso de Alzada ante la Xunta de Galicia

            **Plazo:** 1 mes desde la notificación (actos expresos) o 3 meses desde
            el silencio administrativo.

            **Presentación:**
            1. Telemática: sede.xunta.gal → PR004A. Requiere certificado digital / Chave365.
            2. Presencial: Oficinas de Registro de la Xunta.
            3. Correo certificado al órgano competente.

            **Contenido mínimo:** identificación del recurrente, acto recurrido,
            motivos de impugnación, petición concreta, fecha y firma.

            **Resolución:** 3 meses. Sin resolución → silencio desestimatorio.
            """),
    },
    {
        "category": "ss_autonomo",
        "lang": "gl",
        "prompt": "Como me dou de alta como autónomo en Galicia? Que trámites son necesarios?",
        "response": textwrap.dedent("""\
            ## Alta como traballador autónomo (RETA) en Galicia

            ### 1. Alta en Facenda (AEAT) — Modelo 036 ou 037
            - **Antes** de iniciar a actividade (ou o mesmo día).
            - Modelo 037 (simplificado) para a maioría de autónomos.
            - Indica o epígrafe IAE da túa actividade.

            ### 2. Alta na Seguridade Social — RETA
            - Prazo: **30 días naturais** desde o inicio.
            - Tramítase en importass.seg-social.es ou en calquera CAISS.
            - Base mínima 2024: **653,59 €/mes** → cota ~230 €/mes.

            ### 3. Licenza municipal (se procede)
            - Actividades con local: licenza de apertura no Concello.

            | Trámite | Prazo | Organismo |
            |---|---|---|
            | Modelo 036/037 | Antes ou o mesmo día | AEAT |
            | Alta RETA | 30 días naturais | TGSS |
            | Licenza | Antes de abrir | Concello |
            """),
    },
    {
        "category": "ss_contrato",
        "lang": "es",
        "prompt": "¿Qué tipos de contrato de trabajo existen en España tras la reforma laboral de 2022?",
        "response": textwrap.dedent("""\
            ## Tipos de contrato de trabajo — Reforma laboral 2022 (RDL 32/2021)

            ### Contratos indefinidos
            - **Indefinido ordinario:** para cualquier actividad sin causa temporal.
            - **Fijo-discontinuo:** para trabajos estacionales o intermitentes.

            ### Contratos temporales (causas tasadas)
            - **Circunstancias de producción:**
              - Ocasional e imprevisible: máx. **6 meses** (12 por convenio).
              - Sustitución vacaciones/ausencias previsibles: máx. **90 días/año**.
            - **Sustitución de trabajador** con reserva de puesto: duración = causa.

            ### Contratos formativos
            - **Formación en alternancia:** 3 meses–2 años.
            - **Práctica profesional** (tras título): 6 meses–1 año.

            **Encadenamiento:** 2 contratos temporales en 24 meses → indefinido.
            Registro en SEPE en plazo de 10 días.
            """),
    },
    {
        "category": "ss_baja",
        "lang": "gl",
        "prompt": "Como se tramita a baixa por enfermidade común para un traballador por conta allea?",
        "response": textwrap.dedent("""\
            ## Baixa por incapacidade temporal (IT) por enfermidade común

            **1. Parte médico de baixa**
            - Emíteo o médico do SERGAS ou mutua.
            - Desde abril 2023: comunicado directamente polo INSS/SERGAS á empresa.

            **2. Prestación económica**
            - Días 1–3: sen prestación (salvo mellora por convenio).
            - Días 4–15: paga a empresa (60% base reguladora).
            - Desde o día 16: paga o INSS/Mutua (60% ata día 20; 75% desde día 21).
            - Base reguladora: promedio bases de cotización 12 meses anteriores ÷ 365.

            **3. Obrigas da empresa**
            - Comunicar a baixa no Sistema RED no prazo de 3 días.
            - Non pode despedir ao traballador de baixa (nulidade).

            **Duración máxima:** 365 días (prorrogable 180 máis polo INSS).
            """),
    },
    {
        "category": "mercantil_sl",
        "lang": "gl",
        "prompt": "Que pasos hai que seguir para constituír unha Sociedade de Responsabilidade Limitada (SRL) en Galicia?",
        "response": textwrap.dedent("""\
            ## Constitución dunha SRL en Galicia

            **1. Certificación negativa de denominación** — Rexistro Mercantil Central (rmc.es).
            **2. Conta bancaria e capital** — mínimo 3.000 € (ou 1 € en formación sucesiva).
            **3. Estatutos e escritura notarial** — con Estatutos Tipo: 24-48h telemático.
            **4. NIF provisional** — AEAT, modelo 036.
            **5. Modelo 600 en ATRIGA** — constitución exenta de ITP.
            **6. Inscrición no Rexistro Mercantil Provincial** — prazo 2 meses.
            **7. Alta en Facenda** — modelo 036 definitivo, réxime IVE, epígrafe IAE.

            | Concepto | Importe orientativo |
            |---|---|
            | Notaría | 200–400 € |
            | Rexistro Mercantil | 100–200 € |
            | Total aprox. | 300–600 € |

            Trámite CIRCE: eugo.es
            """),
    },
    {
        "category": "mercantil_autonomo",
        "lang": "es",
        "prompt": "¿Cuál es la diferencia entre trabajar como autónomo y constituir una SL?",
        "response": textwrap.dedent("""\
            ## Autónomo vs. Sociedad Limitada (SL)

            | Aspecto | Autónomo | SL |
            |---|---|---|
            | Responsabilidad | Ilimitada | Limitada al capital |
            | Capital mínimo | No requiere | 1 € / 3.000 € |
            | Constitución | Casi nulos | 300–600 € |
            | Tributación | IRPF 19%–47% | IS 25% (15% primeros 2 años) |
            | Cotización SS | Desde ~230 €/mes | RETA similar |
            | Contabilidad | Simplificada | Mercantil obligatoria |

            **Conviene SL cuando:** beneficio neto > 60.000–80.000 €, riesgo patrimonial
            elevado, varios socios, o contratos con grandes empresas/administraciones.

            **Conviene autónomo cuando:** inicio de actividad, ingresos inciertos,
            actividad de bajo riesgo, beneficio < 40.000 €/año.
            """),
    },
    {
        "category": "documento_contrato",
        "lang": "gl",
        "prompt": "Redacta un contrato de arrendamento de vivenda para uso habitual en Galicia.",
        "response": textwrap.dedent("""\
            ## CONTRATO DE ARRENDAMENTO DE VIVENDA

            En **[LOCALIDADE]**, a **[DATA]**.

            **REUNIDOS**

            **Da unha parte (ARRENDADOR/A):**
            **[NOME COMPLETO]**, con DNI/NIE núm. **[XXX]**, con domicilio en **[ENDEREZO]**.

            **Da outra parte (ARRENDATARIO/A):**
            **[NOME COMPLETO]**, con DNI/NIE núm. **[XXX]**, con domicilio en **[ENDEREZO]**.

            **EXPOÑEN** que o/a ARRENDADOR/A é propietario/a do inmoble sito en
            **[ENDEREZO COMPLETO DO INMOBLE]**, e acordan subscribir o presente
            **contrato de arrendamento de vivenda para uso habitual e permanente**,
            ao abeiro da **Lei 29/1994, do 24 de novembro, de Arrendamentos Urbanos (LAU)**.

            ## CLÁUSULAS

            ### Artigo 1. Obxecto
            Cédese en arrendamento o inmoble descrito para **vivenda habitual e permanente**.

            ### Artigo 2. Duración
            Duración de **[X] anos**, con inicio o **[DATA]** e vencemento o **[DATA]**.
            Prórroga obrigatoria anual ata 5 anos (7 se persoa xurídica).

            ### Artigo 3. Renda
            Renda mensual: **[IMPORTE] €**, pagadeira nos primeiros 7 días do mes,
            por transferencia á conta **[IBAN]**. Actualización anual conforme ao IGC.

            ### Artigo 4. Fianza
            Fianza legal obrigatoria: **[IMPORTE] €** (1 mensualidade).
            O/a arrendador/a depositarao no **IGVS** no prazo dun mes.

            ### Artigo 5. Gastos
            Por conta do/a arrendatario/a: auga, luz, gas e servizos individuais.
            Por conta do/a arrendador/a: cota de comunidade e IBI.

            ### Artigo 6. Obras
            Sen consentimento escrito do/a arrendador/a, non se poderán facer obras
            que modifiquen a configuración do inmoble.

            ### Artigo 7. Xurisdición
            Xulgados e Tribunais de **[LOCALIDADE]**.

            Asinan en duplicado, no lugar e data indicados.

            **O/A ARRENDADOR/A** &nbsp;&nbsp;&nbsp;&nbsp; **O/A ARRENDATARIO/A**

            _______________________ &nbsp;&nbsp;&nbsp;&nbsp; _______________________

            *Nota: Depositar a fianza no IGVS (igvs.xunta.gal) no prazo dun mes.*
            """),
    },
    {
        "category": "documento_recurso",
        "lang": "es",
        "prompt": "Redacta un escrito de recurso de reposición contra una sanción de tráfico.",
        "response": textwrap.dedent("""\
            ## RECURSO DE REPOSICIÓN

            **[LOCALIDAD], [FECHA]**

            **AL JEFE/A DE TRÁFICO DE [PROVINCIA]**

            **D./D.ª [NOMBRE Y APELLIDOS]**, DNI/NIE **[XXX]**, domicilio en
            **[DIRECCIÓN COMPLETA]**,

            **EXPONE:** Que con fecha **[FECHA]** recibió notificación de la resolución
            sancionadora núm. **[EXPEDIENTE]**, por la que se impone sanción de
            **[IMPORTE] €** por infracción del artículo **[XXX]** del RDLeg 6/2015.

            Disconforme con dicha resolución, dentro del plazo de **un mes** (art. 123
            Ley 39/2015), interpone:

            ## RECURSO DE REPOSICIÓN

            ### HECHOS
            **Primero.** [Descripción de los hechos contradiciendo la denuncia.]
            **Segundo.** [Elementos probatorios: testigos, imágenes, documentación.]

            ### FUNDAMENTOS DE DERECHO
            **Primero.** Art. 123 Ley 39/2015 — derecho a recurso de reposición.
            **Segundo.** [Argumentos: falta de prueba, error de identificación, caducidad.]
            **Tercero.** Principio de presunción de inocencia (art. 24.2 CE).

            ### SOLICITA
            Que se **anule o revoque** la sanción impugnada.

            En [Localidad], a [Fecha].

            Firma: _______________________
            D./D.ª [Nombre] · DNI: [XXX]
            """),
    },
    {
        "category": "documento_acta",
        "lang": "gl",
        "prompt": "Redacta unha acta de xunta ordinaria dunha comunidade de propietarios.",
        "response": textwrap.dedent("""\
            ## ACTA DA XUNTA XERAL ORDINARIA
            ## COMUNIDADE DE PROPIETARIOS [NOME/ENDEREZO]

            **En [LOCALIDADE], sendo as [HORA] horas do día [DATA]**, reuníuse a Xunta
            Xeral Ordinaria da Comunidade de Propietarios de **[ENDEREZO]**, presidida
            por **[NOME DO PRESIDENTE/A]**.

            **Secretario/a:** [NOME] · **Administrador/a:** [NOME]

            **Asistentes:** [X] propietarios presentes + [X] representados.
            **Total cotas:** [X]% — quórum suficiente en [primeira/segunda] convocatoria.

            ---

            ### ORDE DO DÍA

            **1. Aprobación da acta anterior.**
            Aprobada por [unanimidade / maioría de X votos a favor, X en contra].

            **2. Contas do exercicio [ANO] e orzamento [ANO+1].**

            | Concepto | Importe |
            |---|---|
            | Ingresos por cotas | [X] € |
            | Gastos mantemento | [X] € |
            | Gastos subministración | [X] € |
            | **Saldo** | **[X] €** |

            Orzamento [ANO+1]: **[X] €** (cota mensual: **[X] €**).
            **Votación:** Aprobado por [unanimidade / maioría].

            **3. [PUNTO ADICIONAL]**
            [Desenvolvemento e resultado da votación.]

            **4. Rogos e preguntas.**
            [Resumo das intervencións.]

            ---

            Levántase a sesión sendo as **[HORA] horas**.

            **O/A PRESIDENTE/A** &nbsp;&nbsp;&nbsp;&nbsp; **O/A SECRETARIO/A**

            _______________________ &nbsp;&nbsp;&nbsp;&nbsp; _______________________
            """),
    },
    {
        "category": "ive_trimestral",
        "lang": "gl",
        "prompt": "Como presento a declaración trimestral do IVE como autónomo?",
        "response": textwrap.dedent("""\
            ## Declaración trimestral do IVE — Modelo 303

            | Trimestre | Período | Prazo |
            |---|---|---|
            | 1.º T | Xaneiro–marzo | 1–20 de abril |
            | 2.º T | Abril–xuño | 1–20 de xullo |
            | 3.º T | Xullo–setembro | 1–20 de outubro |
            | 4.º T | Outubro–decembro | 1–30 de xaneiro |

            **Como cubrir o Modelo 303:**
            1. Accede a sede.agenciatributaria.gob.es → Modelo 303.
            2. Identifícate con certificado dixital, Cl@ve PIN ou número de referencia.
            3. Declara:
               - **IVE repercutido:** suma cotas 21%/10%/4% das facturas emitidas.
               - **IVE soportado:** suma cotas IVE das facturas recibidas afectas.
               - Resultado positivo → pagas; negativo → compensas ou solicitas devolución.

            **Libros obrigatorios:** facturas emitidas, recibidas e bens de investimento.
            """),
    },
    {
        "category": "ive_intracomunitario",
        "lang": "es",
        "prompt": "¿Cómo funciona el IVA intracomunitario para servicios digitales prestados a otros países de la UE?",
        "response": textwrap.dedent("""\
            ## IVA intracomunitario — Servicios digitales (OSS)

            Desde julio 2021, los servicios digitales a consumidores finales de la UE
            tributan en el **país del destinatario**.

            **Régimen OSS (Ventanilla Única):** declaración trimestral única para toda la UE.
            Alta en sede.agenciatributaria.gob.es → OSS → Régimen de la Unión.

            | País | Tipo general |
            |---|---|
            | Alemania | 19% |
            | Francia | 20% |
            | Italia | 22% |
            | Portugal | 23% |
            | España | 21% |

            **Umbral:** ventas UE < 10.000 €/año → aplica IVA español sin registrarse en OSS.

            Plazo declaración OSS: último día del mes siguiente al trimestre.
            """),
    },
]


def _augment(templates: list[dict], rng: random.Random, target: int) -> list[dict]:
    result = list(templates)
    reformulations_gl = [
        ("Como", "De que xeito"),
        ("Que", "Cal é o"),
        ("hai que", "é necesario"),
        ("podo", "é posible"),
        ("en Galicia", "na Comunidade Autónoma de Galicia"),
    ]
    reformulations_es = [
        ("¿Cómo", "¿De qué manera"),
        ("¿Qué", "¿Cuál es"),
        ("hay que", "es necesario"),
        ("puedo", "es posible"),
        ("en España", "en el territorio nacional"),
    ]
    while len(result) < target:
        tpl = rng.choice(templates)
        prompt = tpl["prompt"]
        refs = reformulations_gl if tpl["lang"] == "gl" else reformulations_es
        for old, new in rng.sample(refs, k=min(2, len(refs))):
            if old in prompt:
                prompt = prompt.replace(old, new, 1)
                break
        if prompt != tpl["prompt"]:
            result.append({**tpl, "prompt": prompt})
    return result[:target]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", default="data/finetune/gestoria_gl.jsonl")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--lang", choices=["gl", "es", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    templates = TEMPLATES
    if args.lang != "all":
        templates = [t for t in templates if t["lang"] == args.lang]
    if not templates:
        logger.error("No templates for lang=%s", args.lang)
        return

    examples = _augment(templates, rng, args.n)
    rng.shuffle(examples)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(
                {"prompt": ex["prompt"], "response": ex["response"],
                 "category": ex["category"], "lang": ex["lang"]},
                ensure_ascii=False,
            ) + "\n")

    logger.info("Saved %d examples -> %s", len(examples), out)
    by_cat: dict[str, int] = {}
    for ex in examples:
        by_cat[ex["category"]] = by_cat.get(ex["category"], 0) + 1
    for k, v in sorted(by_cat.items()):
        logger.info("  %-30s %d", k, v)


if __name__ == "__main__":
    main()
