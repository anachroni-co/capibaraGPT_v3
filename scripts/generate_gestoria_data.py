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


# ── Templates by category ─────────────────────────────────────────────────────

TEMPLATES: list[dict] = [

    # ── ATRIGA / Impostos autonómicos ─────────────────────────────────────────
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

            O **IXV** (Imposto sobre as Vendas Retallistas de Determinados Hidrocarburos)
            é un tributo cedido ás Comunidades Autónomas que grava as vendas a retallistas
            de gasolinas, gasóleos, fuelóleos e outros hidrocarburos.

            **Suxeitos pasivos:** Os vendedores a retallo (gasolineiras, distribuidores)
            son os obrigados tributarios. O consumidor final non presenta declaración.

            **Xestión en Galicia:** A recadación e inspección correspóndelle á **ATRIGA**
            (Axencia Tributaria de Galicia).

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

            Na compravenda de inmobles de segunda man entre particulares aplícase o
            **Imposto sobre Transmisións Patrimoniais e Actos Xurídicos Documentados (ITP-AXD)**.

            **Tipo xeral en Galicia:** **10%** sobre o valor de transmisión (ou valor de
            referencia do Catastro se for superior).

            **Tipos reducidos:**
            - **8%** para familias numerosas (vivenda habitual, valor ≤ 200.000 €).
            - **8%** para menores de 36 anos (vivenda habitual, valor ≤ 150.000 €,
              base impoñible IRPF ≤ 22.000 €).
            - **4%** para persoas con discapacidade ≥ 65% (vivenda habitual).

            **Prazo de presentación:** 30 días hábiles desde a escritura.

            **Modelo:** 600 — preséntase en calquera oficina de ATRIGA ou telemáticamente.

            **Exemplo:** Piso de 120.000 € → ITP = 120.000 × 10% = **12.000 €**

            ATRIGA: atriga.gal/tributos/transmisions-patrimoniais/
            """),
    },
    {
        "category": "atriga_sucesions",
        "lang": "gl",
        "prompt": "Que bonificacións existen no Imposto de Sucesións en Galicia?",
        "response": textwrap.dedent("""\
            ## Imposto sobre Sucesións e Doazóns — Bonificacións en Galicia

            Galicia ten unha das regulacións máis favorables de España para herdanzas
            entre familiares directos.

            **Grupo I e II (descendentes, ascendentes, cónxuxe):**
            - **Redución de 1.000.000 €** por herdeiro en herdanzas entre cónxuxes,
              fillos e netos (Grupo I e II).
            - Base impoñible residual tributa ao **tipo marxinal** da tarifa autonómica.
            - Na práctica, herdanzas familiares normais quedan con cota cero.

            **Vivenda habitual:**
            - Redución adicional do **95%** do valor da vivenda habitual do causante,
              con límite de 600.000 € por herdeiro (grupo I, II e irmáns).
            - Requírese manter a vivenda 5 anos.

            **Empresa familiar:**
            - Redución do **99%** do valor de negocio familiar ou participacións,
              cumprindo requisitos de exención en IP.

            **Prazo:** 6 meses desde o falecemento (prorrogable 6 meses máis).

            Modelo: 650 (sucesións) · Oficinas ATRIGA ou telemático.
            """),
    },

    # ── Xunta sede electrónica / procedementos ────────────────────────────────
    {
        "category": "xunta_procedemento",
        "lang": "gl",
        "prompt": "Como solicito unha axuda para a rehabilitación da miña vivenda a través da Xunta de Galicia?",
        "response": textwrap.dedent("""\
            ## Axudas á rehabilitación de vivendas — Xunta de Galicia

            O **Instituto Galego da Vivenda e Solo (IGVS)** xestiona anualmente programas
            de axudas para rehabilitación, enmarcados no Plan Estatal de Vivenda.

            **Tipos de axudas habituais:**
            - Rehabilitación enerxética (illamento, caldeiras, ventás).
            - Accesibilidade (ascensores, rampas).
            - Rehabilitación de elementos comúns en comunidades de propietarios.

            **Procedemento xeral (VI408A):**
            1. Accede a **sede.xunta.gal** → Buscador de procedementos → VI408A.
            2. Identifícate con **certificado dixital, Chave365 ou eIDAS**.
            3. Cubre o formulario de solicitude e achega:
               - Proxecto técnico ou orzamento asinado por técnico competente.
               - Escritura de propiedade ou contrato de arrendamento.
               - DNI/NIE do solicitante.
               - Últimas declaracións de renda dos membros da unidade familiar.
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

            El **recurso de alzada** es el recurso administrativo ordinario contra
            resoluciones que no agotan la vía administrativa (art. 121 Ley 39/2015).

            **Plazo:** 1 mes desde la notificación del acto (actos expresos) o
            3 meses desde que se produzca el silencio administrativo.

            **Órgano competente:** El superior jerárquico del que dictó el acto.

            **Presentación:**
            1. **Telemática (recomendada):** sede.xunta.gal → PR004A (Recurso de alzada).
               Requiere certificado digital / Chave365.
            2. **Presencial:** Oficinas de Registro de la Xunta (Edificios Administrativos).
            3. **Correo certificado:** dirigido al órgano competente.

            **Contenido mínimo del escrito:**
            - Identificación del recurrente (nombre, DNI/NIE, dirección).
            - Acto que se recurre (número de expediente, fecha y órgano).
            - Motivos de impugnación (hechos y fundamentos de derecho).
            - Petición concreta.
            - Lugar, fecha y firma.

            **Resolución:** 3 meses. Transcurrido ese plazo sin resolución → silencio
            administrativo desestimatorio (salvo excepciones).

            Modelo de escrito disponible en: sede.xunta.gal/detalle-procedemento?codigo=PR004A
            """),
    },

    # ── Seguridade Social / laboral ───────────────────────────────────────────
    {
        "category": "ss_autonomo",
        "lang": "gl",
        "prompt": "Como me dou de alta como autónomo en Galicia? Que trámites son necesarios?",
        "response": textwrap.dedent("""\
            ## Alta como traballador autónomo (RETA) en Galicia

            Para darse de alta como autónomo hai que realizar trámites ante a **AEAT**,
            a **Seguridade Social** e, segundo a actividade, no **Concello** ou noutros
            rexistros.

            ### 1. Alta en Facenda (AEAT) — Modelo 036 ou 037
            - **Antes** de iniciar a actividade (ou o mesmo día).
            - Modelo 037 (simplificado) para a maioría de autónomos sen obrigas especiais.
            - Indica o epígrafe IAE da túa actividade.

            ### 2. Alta na Seguridade Social — RETA
            - Prazo: **30 días naturais** desde o inicio da actividade.
            - Tramítase en **Sede Electrónica da Seguridade Social** (importass.seg-social.es)
              ou en calquera Centro de Atención e Información (CAISS).
            - Escolle a base de cotización (mínima 2024: **653,59 €/mes** → cota ~230 €/mes).
            - Sistema de cotización por ingresos reais: actualiza a base segundo a previsión.

            ### 3. Licenza municipal (se procede)
            - Actividades con local: licenza de apertura no Concello.

            ### 4. Rexistro específico (se procede)
            - Sanidade, educación, construción, etc. requiren inscrición en rexistros
              profesionais da Xunta de Galicia.

            **Resumo de prazos:**
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

            La reforma laboral de 2022 simplificó la contratación reduciendo la
            temporalidad. Los contratos vigentes son:

            ### Contratos indefinidos
            - **Indefinido ordinario:** para cualquier actividad sin causa temporal.
            - **Indefinido fijo-discontinuo:** para trabajos estacionales o de prestación
              intermitente, incluso en empresas de trabajo temporal.

            ### Contratos temporales (causas tasadas)
            - **Por circunstancias de la producción:**
              - Ocasional e imprevisible: máx. **6 meses** (ampliable a 12 por convenio).
              - Sustitución de vacaciones u otras ausencias previsibles:
                máx. **90 días/año** (no consecutivos).
            - **Por sustitución de persona trabajadora:** con reserva de puesto
              (IT, maternidad, etc.) — duración = duración de la causa.

            ### Contratos formativos
            - **Formación en alternancia:** combina trabajo y formación reglada.
              Duración: 3 meses–2 años. Cotización reducida.
            - **Obtención de práctica profesional:** tras título universitario/FP.
              Duración: 6 meses–1 año.

            **Encadenamiento de contratos:** 2 contratos temporales en 24 meses
            → conversión automática en indefinido.

            **Forma:** todos pueden ser verbales (salvo excepciones), pero se recomienda
            escritura. Registro en **SEPE** (Servicio Público de Empleo Estatal) en el
            plazo de 10 días.
            """),
    },
    {
        "category": "ss_baja",
        "lang": "gl",
        "prompt": "Como se tramita a baixa por enfermidade común para un traballador por conta allea?",
        "response": textwrap.dedent("""\
            ## Baixa por incapacidade temporal (IT) por enfermidade común

            ### Procedemento

            **1. Parte médico de baixa**
            - Emíteo o médico do **Servizo Galego de Saúde (SERGAS)** ou mutua.
            - Desde o 1 de abril de 2023: os partes son comunicados directamente
              polo INSS/SERGAS á empresa (o traballador xa non ten que entregalos).

            **2. Duración e control**
            - Partes de confirmación: emíteos o médico segundo duración estimada
              (1 parte cada 3/7/14/30 días segundo proceso).
            - Alta médica: cando desapareza a causa da IT.

            **3. Prestación económica**
            - **Días 1–3:** sen prestación (salvo mellora por convenio colectivo).
            - **Días 4–15:** paga a empresa (60% base reguladora).
            - **A partir do día 16:** paga o INSS/Mutua (60% ata día 20; 75% desde día 21).
            - **Base reguladora:** promedio das bases de cotización dos 12 meses anteriores ÷ 365.

            **4. Obrigas da empresa**
            - Comunicar a baixa no **Sistema RED** (TGSS) no prazo de 3 días.
            - Aboar o salario dos días 4–15.
            - Non pode despedir ao traballador por estar de baixa (nulidade).

            **Duración máxima:** 365 días (prorrogable 180 máis polo INSS).
            """),
    },

    # ── Sociedades / Mercantil ────────────────────────────────────────────────
    {
        "category": "mercantil_sl",
        "lang": "gl",
        "prompt": "Que pasos hai que seguir para constituír unha Sociedade de Responsabilidade Limitada (SRL) en Galicia?",
        "response": textwrap.dedent("""\
            ## Constitución dunha SRL (Sociedade de Responsabilidade Limitada) en Galicia

            ### Pasos

            **1. Certificación negativa de denominación social**
            - Solicítase no **Rexistro Mercantil Central** (rmc.es) — telemático.
            - Propón 3 nomes por orde de preferencia. Prazo: 2-5 días.
            - Vixencia: 3 meses (prorrogable 3 meses máis).

            **2. Conta bancaria e achega de capital**
            - Capital mínimo: **1 € simbólico** (SRL de formación sucesiva) ou
              **3.000 €** (SRL ordinaria, totalmente desembolsado).
            - Ingresa o capital e obtén o certificado do banco.

            **3. Estatutos sociais e escritura ante notario**
            - Redacta os Estatutos (ou usa os Estatutos Tipo do RD 421/2015 para
              simplificar e acelerar).
            - Outorga escritura de constitución ante **Notario**.
            - Con Estatutos Tipo e trámite telemático (CIRCE): **24-48 horas** e custos reducidos.

            **4. Número de Identificación Fiscal (NIF) provisional**
            - Solicítase na **AEAT** (modelo 036) simultaneamente ou tras a escritura.

            **5. Liquidación do ITPAXD (exento)**
            - As constitucións de sociedades están **exentas** do Imposto de
              Transmisións (art. 45 RDLeg 1/1993).
            - Aínda así, hai que presentar o modelo 600 en ATRIGA marcando a exención.

            **6. Inscrición no Rexistro Mercantil Provincial**
            - Prazo: **2 meses** desde a escritura.
            - Documentación: escritura, certificado bancario, NIF provisional.

            **7. Alta en Facenda (AEAT) e inicio de actividade**
            - Modelo 036: alta definitiva, elección de réxime IVE, epígrafe IAE.

            **Custos orientativos (SRL 3.000 €):**
            | Concepto | Importe |
            |---|---|
            | Notaría | 200–400 € |
            | Rexistro Mercantil | 100–200 € |
            | Rexistro Mercantil Central | 14 € |
            | Total aprox. | 300–600 € |

            Trámite CIRCE: eugo.es (Ventanilla Única Empresarial)
            """),
    },
    {
        "category": "mercantil_autonomo",
        "lang": "es",
        "prompt": "¿Cuál es la diferencia entre trabajar como autónomo y constituir una SL?",
        "response": textwrap.dedent("""\
            ## Autónomo vs. Sociedad Limitada (SL) — Comparativa

            | Aspecto | Autónomo (RETA) | Sociedad Limitada |
            |---|---|---|
            | **Responsabilidad** | Ilimitada (patrimonio personal) | Limitada al capital social |
            | **Capital mínimo** | No requiere | 1 € (formación sucesiva) / 3.000 € |
            | **Costes constitución** | Casi nulos | 300–600 € |
            | **Tributación** | IRPF (tramos 19%–47%) | IS al 25% (15% primeros 2 años) |
            | **Cotización SS** | Desde ~230 €/mes | Socio-administrador: RETA similar |
            | **Imagen comercial** | Menor en grandes contratos | Mayor credibilidad |
            | **Gestión contable** | Simplificada | Contabilidad mercantil obligatoria |
            | **Reparto beneficios** | Todo al titular | Vía dividendos (tributación adicional) |

            **¿Cuándo conviene la SL?**
            - Beneficio neto anual > 60.000–80.000 € (ahorro fiscal vs. IRPF).
            - Actividad con riesgo patrimonial elevado.
            - Varios socios.
            - Contratos con grandes empresas o administraciones.

            **¿Cuándo conviene el autónomo?**
            - Inicio de actividad con ingresos inciertos.
            - Actividad de bajo riesgo.
            - Simplicidad de gestión preferida.
            - Beneficio neto < 40.000 €/año.

            Recomendación: consulta con tu asesor fiscal para simular la carga
            tributaria en ambos casos con tus cifras concretas.
            """),
    },

    # ── Documentos / redacción ────────────────────────────────────────────────
    {
        "category": "documento_contrato",
        "lang": "gl",
        "prompt": "Redacta un contrato de arrendamento de vivenda para uso habitual en Galicia.",
        "response": textwrap.dedent("""\
            ## CONTRATO DE ARRENDAMENTO DE VIVENDA

            En **[LOCALIDADE]**, a **[DATA]**.

            **REUNIDOS**

            **Da unha parte (ARRENDADOR/A):**
            **[NOME COMPLETO]**, con DNI/NIE núm. **[XXX]**, con domicilio en
            **[ENDEREZO]**.

            **Da outra parte (ARRENDATARIO/A):**
            **[NOME COMPLETO]**, con DNI/NIE núm. **[XXX]**, con domicilio en
            **[ENDEREZO]**.

            Ambas as partes recoñécense con capacidade legal suficiente para subscribir
            o presente contrato, e

            **EXPOÑEN**

            Que o/a ARRENDADOR/A é propietario/a do inmoble sito en **[ENDEREZO
            COMPLETO DO INMOBLE]**, inscrito no Rexistro da Propiedade de **[REXISTRO]**,
            tomo **[X]**, libro **[X]**, folio **[X]**, finca núm. **[X]**.

            Que ambas as partes acordan subscribir o presente **contrato de arrendamento
            de vivenda para uso habitual e permanente**, ao abeiro da **Lei 29/1994, do
            24 de novembro, de Arrendamentos Urbanos (LAU)**, e conforme ás seguintes:

            ---

            ## CLÁUSULAS

            ### Artigo 1. Obxecto
            O/A ARRENDADOR/A cede en arrendamento ao/á ARRENDATARIO/A o inmoble
            descrito no expositivo, para destinalo exclusivamente a **vivenda habitual
            e permanente** do/a arrendatario/a e da súa unidade familiar.

            ### Artigo 2. Duración
            O presente contrato terá unha duración de **[X] anos**, con inicio o día
            **[DATA DE INICIO]** e vencemento o **[DATA DE FIN]**.

            Chegado o vencemento, o contrato prorrogarase obrigatoriamente por prazos
            anuais ata un máximo de **5 anos** (7 se o arrendador é persoa xurídica),
            salvo que o/a arrendatario/a comunique a súa vontade de non renovar con
            30 días de antelación.

            ### Artigo 3. Renda
            A renda mensual pactada é de **[IMPORTE] €**, pagadeira dentro dos primeiros
            **7 días** de cada mes, mediante transferencia bancaria á conta
            **[IBAN ARRENDADOR/A]**.

            A renda actualizarase anualmente conforme ao **Índice de Garantía de
            Competitividade (IGC)** ou o índice que estableza a normativa vixente.

            ### Artigo 4. Fianza
            O/A ARRENDATARIO/A entrega neste acto a cantidade de **[IMPORTE] €**
            en concepto de fianza legal obrigatoria (equivalente a **1 mensualidade**),
            que o/a ARRENDADOR/A se obriga a depositar no **IGVS** (Instituto Galego
            da Vivenda e Solo) no prazo dun mes.

            ### Artigo 5. Gastos e subministracións
            Serán por conta do/a ARRENDATARIO/A os gastos de subministración de auga,
            electricidade, gas e outros servizos individuais do inmoble.

            A cota da comunidade de propietarios e o **IBI** serán por conta do/a
            **ARRENDADOR/A**, salvo pacto expreso en contrario.

            ### Artigo 6. Obras
            O/A ARRENDATARIO/A non poderá realizar obras que modifiquen a configuración
            do inmoble sen consentimento escrito do/a ARRENDADOR/A.

            ### Artigo 7. Xurisdición
            Para cantas cuestións xurdan do presente contrato, as partes, con renuncia
            ao seu foro propio se o tivesen, sométense aos **Xulgados e Tribunais de
            [LOCALIDADE]**.

            ---

            E en proba de conformidade, asinan o presente contrato por duplicado e a
            un só efecto, no lugar e data indicados na cabeceira.

            **O/A ARRENDADOR/A** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            **O/A ARRENDATARIO/A**

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
            Jefatura Provincial de Tráfico de [Provincia]
            [Dirección]

            **D./D.ª [NOMBRE Y APELLIDOS]**, con DNI/NIE núm. **[XXX]**, con domicilio
            a efectos de notificaciones en **[DIRECCIÓN COMPLETA]**, actuando en su
            propio nombre y derecho,

            **EXPONE:**

            Que con fecha **[FECHA DE NOTIFICACIÓN]** recibió notificación de la
            resolución sancionadora núm. **[EXPEDIENTE]**, dictada por esa Jefatura,
            por la que se impone una sanción de **[IMPORTE] €** y/o la detracción de
            **[X] puntos**, por infracción del artículo **[XXX]** del RDLeg 6/2015,
            de 30 de octubre, por el que se aprueba el texto refundido de la Ley sobre
            Tráfico, Circulación de Vehículos a Motor y Seguridad Vial.

            Que disconforme con dicha resolución, dentro del plazo legal de **un mes**
            establecido en el artículo 123 de la Ley 39/2015, de 1 de octubre, del
            Procedimiento Administrativo Común de las Administraciones Públicas,
            interpone el presente:

            ## RECURSO DE REPOSICIÓN

            Basado en los siguientes:

            ### HECHOS

            **Primero.** [Descripción de los hechos tal como ocurrieron, contradiciendola versión del denunciante.]

            **Segundo.** [Aportar elementos probatorios: testigos, imágenes, documentación.]

            ### FUNDAMENTOS DE DERECHO

            **Primero.** El artículo 123 de la Ley 39/2015 reconoce el derecho a
            interponer recurso de reposición contra actos que no agoten la vía
            administrativa, en el plazo de un mes.

            **Segundo.** [Argumentos jurídicos: falta de prueba suficiente, error en
            la identificación del conductor, caducidad del procedimiento, etc.]

            **Tercero.** El principio de presunción de inocencia (art. 24.2 CE) exige
            que la Administración acredite la infracción de forma suficiente.

            ### SOLICITA

            Que tenga por presentado este escrito, lo admita y, previos los trámites
            legales oportunos, dicte resolución por la que se **anule o revoque** la
            sanción impugnada, con todos los efectos legales favorables para esta parte.

            En [Localidad], a [Fecha].

            Firma: _______________________

            D./D.ª [Nombre y apellidos]
            DNI: [XXX]
            """),
    },
    {
        "category": "documento_acta",
        "lang": "gl",
        "prompt": "Redacta unha acta de xunta ordinaria dunha comunidade de propietarios.",
        "response": textwrap.dedent("""\
            ## ACTA DA XUNTA XERAL ORDINARIA
            ## COMUNIDADE DE PROPIETARIOS [NOME/ENDEREZO]

            **En [LOCALIDADE], sendo as [HORA] horas do día [DATA]**, reuníuse en
            **[LUGAR: local social / portal / etc.]** a Xunta Xeral Ordinaria da
            Comunidade de Propietarios de **[ENDEREZO COMPLETO]**, presidida por
            **[NOME DO PRESIDENTE/A]**.

            ---

            ### ASISTENTES

            **Propietarios presentes:**
            - [Nome], piso [X][LETRA] — [X] cotas de participación
            - [Nome], piso [X][LETRA] — [X] cotas de participación

            **Representados:**
            - [Nome], representado/a por [Nome representante], piso [X][LETRA]

            **Total cotas representadas:** [X]% (quórum suficiente para a válida
            constitución da xunta en [primeira/segunda] convocatoria).

            Actúa como **Secretario/a**: **[NOME]** · **Administrador/a**: **[NOME]**

            ---

            ### ORDE DO DÍA

            **1. Lectura e aprobación, se procede, da acta da xunta anterior.**

            Lida a acta da xunta celebrada o **[DATA ANTERIOR]**, é aprobada por
            **unanimidade / maioría de [X] votos a favor, [X] en contra, [X]
            abstencións**.

            ---

            **2. Exame e aprobación das contas do exercicio [ANO] e do orzamento
            para [ANO+1].**

            O/a administrador/a presenta o estado de contas do exercicio **[ANO]**:

            | Concepto | Importe |
            |---|---|
            | Ingresos por cotas | [X] € |
            | Gastos de mantemento | [X] € |
            | Gastos de portería | [X] € |
            | Gastos de subministración (luz zonas comúns) | [X] € |
            | **Saldo a [DATA]** | **[X] €** |

            Proposta de orzamento para **[ANO+1]**: **[X] €** (cota mensual por piso:
            **[X] €**).

            **Votación:** Aprobado por [unanimidade / maioría].

            ---

            **3. [PUNTO ADICIONAL DA ORDE DO DÍA]**

            [Desenvolvemento do punto e resultado da votación.]

            ---

            **4. Rogos e preguntas.**

            [Resumo das intervencións, se as houber.]

            ---

            ### PECHE

            Sen máis asuntos que tratar, o/a Presidente/a levantou a sesión sendo
            as **[HORA] horas**, da que se estende a presente acta, que asinan o/a
            Presidente/a e o/a Secretario/a para constancia.

            **O/A PRESIDENTE/A** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            **O/A SECRETARIO/A — ADMINISTRADOR/A**

            _______________________ &nbsp;&nbsp;&nbsp;&nbsp; _______________________

            [Nome] &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            [Nome]
            """),
    },

    # ── IVE / IVA ─────────────────────────────────────────────────────────────
    {
        "category": "ive_trimestral",
        "lang": "gl",
        "prompt": "Como presento a declaración trimestral do IVE como autónomo?",
        "response": textwrap.dedent("""\
            ## Declaración trimestral do IVE — Modelo 303

            Os autónomos e empresarios en réxime xeral do IVE presentan o
            **Modelo 303** con periodicidade trimestral.

            ### Prazos de presentación

            | Trimestre | Período | Prazo |
            |---|---|---|
            | 1.º T | Xaneiro–marzo | 1–20 de abril |
            | 2.º T | Abril–xuño | 1–20 de xullo |
            | 3.º T | Xullo–setembro | 1–20 de outubro |
            | 4.º T | Outubro–decembro | 1–30 de xaneiro |

            ### Como cubrir o Modelo 303

            1. Accede a **sede.agenciatributaria.gob.es** → Modelo 303.
            2. Identifícate con certificado dixital, Cl@ve PIN ou número de referencia.
            3. Cubre:
               - **IVE repercutido (vendas):** suma das bases impoñibles e cotas
                 ao 21%, 10% e 4% das túas facturas emitidas.
               - **IVE soportado deducible (compras):** suma das cotas do IVE
                 das facturas recibidas afectas á actividade.
               - **Resultado:** IVE repercutido − IVE soportado.
                 - Se positivo → pagas á AEAT.
                 - Se negativo → compensa en trimestres seguintes (ou solicitas
                   devolución no 4.º T mediante Modelo 303 anual).

            ### Tipos de IVE máis comúns
            - **21%:** tipo xeral (servizos profesionais, material de oficina...).
            - **10%:** alimentos elaborados, hostalería, transporte de viaxeiros.
            - **4%:** alimentos básicos, libros, medicamentos.
            - **0% / exento:** actividades médicas, educativas, seguros...

            ### Libros rexistro obrigatorios
            - Libro de facturas emitidas.
            - Libro de facturas recibidas.
            - Libro de bens de investimento (se procede).
            """),
    },
    {
        "category": "ive_intracomunitario",
        "lang": "es",
        "prompt": "¿Cómo funciona el IVA intracomunitario para servicios digitales prestados a otros países de la UE?",
        "response": textwrap.dedent("""\
            ## IVA intracomunitario — Servicios digitales (OSS/MOSS)

            Desde el 1 de julio de 2021, los servicios digitales (streaming, software,
            apps, e-learning...) prestados a **consumidores finales** de otros países
            de la UE tributan en el **país del destinatario** (destino).

            ### Régimen OSS (One Stop Shop) — Ventanilla Única

            Permite declarar y pagar el IVA de toda la UE desde un único Estado miembro.

            **Alta en OSS (España):**
            1. Accede a **sede.agenciatributaria.gob.es** → OSS.
            2. Registra tu actividad en el Régimen de la Unión.

            **Presentación:**
            - Declaración **trimestral** (modelo OSS).
            - Plazo: último día del mes siguiente al trimestre.
            - Declaras las ventas país a país con el tipo del IVA de cada Estado miembro.

            **Tipos de IVA en los principales países:**
            | País | Tipo general | Tipo reducido (digital) |
            |---|---|---|
            | Alemania | 19% | 7% |
            | Francia | 20% | — |
            | Italia | 22% | — |
            | Portugal | 23% | 6% |
            | España | 21% | — |

            **Umbral:** Si las ventas a consumidores UE son < 10.000 €/año, puedes
            aplicar el IVA español en todas ellas (sin registrarte en OSS).

            **Facturas:** deben indicar el tipo de IVA del país del cliente y
            el importe correspondiente.
            """),
    },
]


# ── Augmentation: variations on existing templates ───────────────────────────

def _augment(templates: list[dict], rng: random.Random, target: int) -> list[dict]:
    """Create paraphrased variations to reach target count."""
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", default="data/finetune/gestoria_gl.jsonl")
    parser.add_argument("--n", type=int, default=2000,
                        help="Target number of examples (default: 2000)")
    parser.add_argument("--lang", choices=["gl", "es", "all"], default="all",
                        help="Language filter (default: all)")
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
            record = {
                "prompt": ex["prompt"],
                "response": ex["response"],
                "category": ex["category"],
                "lang": ex["lang"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Saved %d examples → %s", len(examples), out)

    by_cat: dict[str, int] = {}
    for ex in examples:
        by_cat[ex["category"]] = by_cat.get(ex["category"], 0) + 1
    for k, v in sorted(by_cat.items()):
        logger.info("  %-30s %d", k, v)


if __name__ == "__main__":
    main()
