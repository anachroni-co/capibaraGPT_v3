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

    # ── IRPF autónomo — pagos fraccionados ───────────────────────────────────
    {
        "category": "irpf_autonomo_130",
        "lang": "gl",
        "prompt": "Como funciona o modelo 130 de pagos fraccionados do IRPF para autónomos?",
        "response": textwrap.dedent("""\
            ## Modelo 130 — Pagos fraccionados do IRPF (autónomos en estimación directa)

            Os autónomos en **estimación directa** (normal ou simplificada) presentan
            o **Modelo 130** trimestralmente para anticipar o IRPF.

            ### Prazos
            | Trimestre | Período | Prazo |
            |---|---|---|
            | 1.º T | Xaneiro–marzo | 1–20 de abril |
            | 2.º T | Abril–xuño | 1–20 de xullo |
            | 3.º T | Xullo–setembro | 1–20 de outubro |
            | 4.º T | Outubro–decembro | 1–30 de xaneiro |

            ### Cálculo
            **20%** do rendemento neto acumulado desde o inicio do ano, menos
            os pagos fraccionados xa ingresados en trimestres anteriores e as
            retencións sufridas.

            ```
            Cota = 20% × (ingresos acumulados − gastos deducibles acumulados)
                 − retencións acumuladas
                 − pagos fraccionados anteriores do mesmo ano
            ```

            Se o resultado é **negativo ou cero**, preséntas o modelo con cota 0
            (sen devolución).

            ### Non están obrigados a presentar o 130
            - Autónomos cunha taxa de retención ≥ 70% nos seus ingresos
              (profesionais con retención do 15% que facturen case todo a empresas).

            ### Presentación
            - Telemática: **sede.agenciatributaria.gob.es** → Modelo 130.
            - Certificado dixital, Cl@ve PIN ou número de referencia.
            """),
    },
    {
        "category": "irpf_autonomo_modulos",
        "lang": "es",
        "prompt": "¿En qué consiste el régimen de estimación objetiva (módulos) del IRPF para autónomos?",
        "response": textwrap.dedent("""\
            ## Estimación objetiva (módulos) en el IRPF — Autónomos

            El régimen de **estimación objetiva** (coloquialmente «módulos») calcula
            el rendimiento neto de la actividad a partir de parámetros objetivos
            (empleados, metros del local, potencia instalada, etc.) en lugar de
            ingresos y gastos reales.

            ### ¿Quién puede acogerse?
            - Actividades incluidas en la Orden de Módulos anual (Hacienda).
            - Volumen de rendimientos del año anterior < **250.000 €** (actividades
              agrícolas/ganaderas: < 250.000 €; resto: < 150.000 € desde 2023
              en función del sector).
            - Volumen de compras < **250.000 €**.
            - No estar excluido por renuncia previa (3 años mínimo en directa).

            ### Pago fraccionado — Modelo 131
            - **Trimestral**, plazo igual al Modelo 130.
            - Cota = % fijo sobre el rendimiento neto módulo (varía por actividad).

            ### Ventajas
            - Simplicidad contable: no es obligatorio llevar libro de ingresos/gastos
              (sí facturas emitidas/recibidas).
            - Puede convenir si el rendimiento real es superior al módulo.

            ### Inconvenientes
            - Pagas aunque tengas pérdidas reales.
            - Reducción de umbrales en los últimos años → muchos autónomos
              han sido excluidos.

            ### IVA en módulos (régimen simplificado)
            Si también tributas en IVA simplificado, el IVA a ingresar se calcula
            por cuotas devengadas fijadas en la Orden de Módulos.
            """),
    },

    # ── ITP-AXD — Actos Xurídicos Documentados ───────────────────────────────
    {
        "category": "itpaxd_axd",
        "lang": "gl",
        "prompt": "Canto se paga en Actos Xurídicos Documentados (AXD) ao asinar unha hipoteca en Galicia?",
        "response": textwrap.dedent("""\
            ## Actos Xurídicos Documentados (AXD) — Hipotecas en Galicia

            Desde a **Lei 5/2019 de Contratos de Crédito Inmobiliario**, o suxeito
            pasivo do AXD nas escrituras de hipoteca é o **banco** (non o cliente).
            O comprador xa **non paga** este imposto nas hipotecas novas.

            ### AXD que si paga o comprador
            - **Escritura de compravenda** (documentos notariais): 1,5% en Galicia
              sobre a base impoñible (valor declarado do inmoble).
            - **Actas e outros documentos notariais** suxeitos a inscrición rexistral.

            ### Base impoñible
            - Para compravenda: valor declarado ou valor de referencia do Catastro
              (o maior dos dous).
            - Para hipotecas (pagada polo banco): responsabilidade hipotecaria total
              (capital + xuros + costas).

            ### Prazo de presentación
            - **30 días hábiles** desde a sinatura da escritura.
            - Modelo **600** en ATRIGA (atriga.gal) ou oficinas tributarias.

            ### Exemplo compravenda + escritura
            Piso 150.000 € → AXD escritura = 150.000 × 1,5% = **2.250 €**
            (a cargo do comprador; o ITP ou IVE vai separado).

            Consulta: atriga.gal/tributos/transmisions-patrimoniais/
            """),
    },
    {
        "category": "itpaxd_axd",
        "lang": "es",
        "prompt": "¿Qué impuesto se paga al constituir una hipoteca en Galicia? ¿Lo paga el banco o el cliente?",
        "response": textwrap.dedent("""\
            ## Impuesto AJD en hipotecas (Galicia) — ¿Quién paga?

            Desde la **Ley 5/2019 de Contratos de Crédito Inmobiliario** (art. 29
            Real Decreto Legislativo 1/1993, modificado), el sujeto pasivo del
            **Impuesto de Actos Jurídicos Documentados (AJD)** en las escrituras
            de préstamo hipotecario es la **entidad financiera (banco)**.

            **El cliente no paga AJD por la hipoteca.**

            ### ¿Qué sí paga el comprador?
            - **AJD por la escritura de compraventa:** tipo 1,5% en Galicia sobre
              el valor del inmueble.
            - **ITP** (si es segunda mano): 10% general (tipos reducidos para
              jóvenes, familias numerosas, discapacidad).
            - **IVA + AJD** (si es obra nueva de promotor):
              IVA 10% + AJD 1,5% sobre el precio.

            ### Resumen de costes de adquisición (segunda mano, Galicia)
            | Concepto | % / importe |
            |---|---|
            | ITP | 10% (general) |
            | AJD escritura compraventa | 1,5% |
            | Notaría | 0,2%–0,5% aprox. |
            | Registro de la Propiedad | 0,1%–0,3% aprox. |
            | Gestoría | 300–600 € |

            Plazo ITP/AJD: **30 días hábiles** desde escritura → Modelo 600 en ATRIGA.
            """),
    },

    # ── Axudas Xunta — Kit Digital ────────────────────────────────────────────
    {
        "category": "xunta_kit_digital",
        "lang": "gl",
        "prompt": "Como solicito o Kit Dixital para a miña pequena empresa en Galicia?",
        "response": textwrap.dedent("""\
            ## Kit Dixital — Axudas para a dixitalización de pemes

            O **Kit Dixital** é un programa do Goberno de España (fondos Next Generation EU)
            xestionado pola **Red.es** para subvencionar a adopción de solucións dixitais
            en autónomos e pemes.

            ### Segmentos e contías (convocatoria vixente)
            | Segmento | Empregados | Axuda máxima |
            |---|---|---|
            | Segmento I | 10–49 | 12.000 € |
            | Segmento II | 3–9 | 6.000 € |
            | Segmento III | 0–2 (autónomos) | 2.000 € |

            ### Solucións subvencionables (categorías)
            - Sitio web e presenza en internet.
            - Comercio electrónico.
            - Xestión de redes sociais.
            - Xestión de clientes (CRM).
            - Business Intelligence e analítica.
            - Xestión de procesos (ERP).
            - Facturación electrónica.
            - Ciberseguridade.
            - Comunicacións seguras.
            - Oficina virtual.

            ### Pasos para solicitar
            1. Accede a **acelerapyme.gob.es** → Solicitar bono Kit Dixital.
            2. Identifícate con certificado dixital ou Cl@ve.
            3. Supera o **test de diagnóstico dixital** (Autodiagnóstico).
            4. Presenta a solicitude e obtén o **bono dixital**.
            5. Escolle un **Axente Dixitalizador** homologado e asina o acordo de
               prestación de servizos.

            **Importante:** Debes estar ao corrente de pagos con AEAT e SS.
            Non están admitidas empresas en crise nin en concurso.
            """),
    },
    {
        "category": "xunta_bono_aluguer",
        "lang": "gl",
        "prompt": "Existe algunha axuda da Xunta para o aluguer de vivenda para mozos en Galicia?",
        "response": textwrap.dedent("""\
            ## Bono Alugueiro Novo — Xunta de Galicia

            O **Bono Alugueiro Novo** (procedemento **VI481B**) é unha subvención do
            **IGVS** (Instituto Galego da Vivenda e Solo) dirixida a facilitar o
            acceso á vivenda en réxime de arrendamento a persoas mozas.

            ### Requisitos habituais do beneficiario
            - Idade: entre **18 e 35 anos** (inclusive) na data da solicitude.
            - Residencia efectiva na vivenda arrendada en Galicia.
            - Renda arrendataria: renda bruta anual ≤ **22.000 €** (unidade familiar
              ≤ 30.000 €).
            - Renda do aluguer ≤ **600 €/mes** (municipios con mercado tenso: ata 900 €).
            - Non ser propietario/a de ningunha vivenda en España.
            - Estar ao corrente nas obrigas tributarias e de SS.

            ### Contía da axuda
            - **50% da renda mensual** do aluguer, ata un máximo de **300 €/mes**.
            - Duración: 2 anos (prorrogable segundo convocatoria).

            ### Como solicitar
            1. **sede.xunta.gal** → Buscador → VI481B.
            2. Certificado dixital ou Chave365.
            3. Documentación: contrato de arrendamento, DNI, declaración de renda,
               xustificante de domicilio, certificado bancario.
            4. Prazo: convocatoria anual (xeralmente primeiro semestre).

            Información: igvs.xunta.gal · Tel.: 981 541 900
            """),
    },

    # ── Nóminas ───────────────────────────────────────────────────────────────
    {
        "category": "nominas_calculo",
        "lang": "es",
        "prompt": "¿Cómo se calcula la nómina de un trabajador? Explica las cotizaciones a la Seguridad Social y la retención de IRPF.",
        "response": textwrap.dedent("""\
            ## Cálculo de nómina — Cotizaciones SS y retención IRPF

            ### Estructura básica de una nómina

            ```
            Salario bruto mensual                          X.XXX,XX €
            + Complementos (plus transporte, dietas...)    +   XXX,XX €
            ─────────────────────────────────────────────────────────
            DEVENGOS TOTALES                               X.XXX,XX €

            − Cuota SS trabajador                         −   XXX,XX €
            − Retención IRPF                              −   XXX,XX €
            ─────────────────────────────────────────────────────────
            LÍQUIDO A PERCIBIR                             X.XXX,XX €
            ```

            ### Cotizaciones SS a cargo del trabajador (2024)
            | Contingencia | Tipo trabajador |
            |---|---|
            | Contingencias comunes | 4,70% |
            | Desempleo (contrato indefinido) | 1,55% |
            | Formación profesional | 0,10% |
            | **Total aprox.** | **6,35%** |

            Base de cotización = salario bruto + prorrata pagas extras.

            ### Cuotas SS a cargo de la empresa (referencia)
            | Contingencia | Tipo empresa |
            |---|---|
            | Contingencias comunes | 23,60% |
            | Desempleo (indefinido) | 5,50% |
            | FOGASA | 0,20% |
            | Formación profesional | 0,60% |
            | **Total aprox.** | **29,90%** |

            ### Retención IRPF
            - Calculada por la empresa al inicio del año con el **modelo 145**
              (comunicación de datos del trabajador).
            - Depende del salario bruto anual, situación familiar (hijos, cónyuge,
              discapacidad) y otras rentas.
            - Herramienta de cálculo: AEAT → «Retenciones sobre rendimientos
              del trabajo» (servicio de cálculo en línea).

            **Ingreso a Hacienda:**
            - Retenciones → Modelo 111 (trimestral) o Modelo 111 mensual (grandes empresas).
            - Cuotas SS → Sistema RED (SILTRA), pago domiciliado en los primeros 29 días del mes.
            """),
    },
    {
        "category": "nominas_calculo",
        "lang": "gl",
        "prompt": "Que retencións do IRPF lle corresponden a un traballador con salario de 25.000 € brutos anuais en Galicia?",
        "response": textwrap.dedent("""\
            ## Retención de IRPF sobre rendementos do traballo — Galicia

            Para un traballador con **25.000 € brutos anuais** (sen outros ingresos nin
            circunstancias familiares especiais), a retención estímase do xeito seguinte:

            ### Cálculo simplificado (solteiro/a, sen fillos, 2024)

            | Concepto | Importe |
            |---|---|
            | Salario bruto anual | 25.000 € |
            | − Cotización SS traballador (~6,35%) | − 1.588 € |
            | − Redución por rendementos do traballo | − 2.000 € aprox. |
            | = Base liquidable | ~21.412 € |

            **Tarifa estatal + autonómica (Galicia):**
            - Tramo ata 12.450 €: tipo ~9,5% (estatal 9,5% + autonómico 9%)
            - Tramo 12.450–21.412 €: tipo ~24% (escalas combinadas)
            - Cota íntegra aproximada: **~3.900 €**
            - Menos dedución por traballo (mínimo persoal ~1.514 €)
            - **Retención aproximada: ~14–15%**

            ### Nota importante
            O tipo real varía segundo:
            - Estado civil e número de fillos/descendentes.
            - Discapacidade recoñecida.
            - Segundo pagador.
            - Cotizacións a mutualidades ou plans de pensións.

            Utiliza a ferramenta oficial da AEAT:
            **sede.agenciatributaria.gob.es → Retenciones sobre rendimientos del trabajo**
            para calcular o tipo exacto coas túas circunstancias persoais.
            """),
    },

    # ── Modelos tributarios 111 / 115 / 190 ──────────────────────────────────
    {
        "category": "modelo_111",
        "lang": "gl",
        "prompt": "Para que serve o modelo 111 e cando hai que presentalo?",
        "response": textwrap.dedent("""\
            ## Modelo 111 — Retencións e ingresos a conta do IRPF (traballo e actividades)

            O **Modelo 111** é a declaración mediante a que as empresas e autónomos
            con persoal ingresam na AEAT as retencións practicadas sobre:
            - **Rendementos do traballo** (salarios dos empregados).
            - **Actividades económicas** (honorarios de profesionais con retención do 15%
              ou do 7% para profesionais de inicio de actividade).
            - Premios, propiedade intelectual e outros rendementos suxeitos a retención.

            ### Periodicidade e prazos
            | Obrigado | Periodicidade | Prazo |
            |---|---|---|
            | Pemes e autónomos (volume < 6M €) | Trimestral | 1–20 do mes seguinte ao trimestre |
            | Grandes empresas (volume ≥ 6M €) | Mensual | 1–20 do mes seguinte |

            ### Como presentar
            1. **sede.agenciatributaria.gob.es** → Modelo 111.
            2. Certificado dixital ou Cl@ve PIN.
            3. Indica o importe total das retencións e o número de perceptores.
            4. Paga (se hai cota positiva) mediante domiciliación ou cargo en conta.

            ### Relación co Modelo 190
            O **Modelo 190** é o resumo anual das retencións declaradas en cada
            Modelo 111 do exercicio. Preséntase en **xaneiro** do ano seguinte.
            """),
    },
    {
        "category": "modelo_115",
        "lang": "es",
        "prompt": "¿Qué es el modelo 115 y quién está obligado a presentarlo?",
        "response": textwrap.dedent("""\
            ## Modelo 115 — Retenciones e ingresos a cuenta por arrendamientos

            El **Modelo 115** recoge las retenciones practicadas sobre los alquileres
            de **inmuebles urbanos** satisfechos a arrendadores personas físicas o
            jurídicas (siempre que no estén excluidas de retención).

            ### ¿Quién debe presentarlo?
            Toda empresa o autónomo que pague alquileres de locales u oficinas a un
            arrendador y esté obligado a practicar retención (arrendador no es
            SOCIMI, fondo de inversión, etc. exentos).

            ### Tipo de retención
            - **19%** sobre la renta del alquiler pagada.

            ### ¿Cuándo no hay retención?
            - Arrendador es una SICAV, fondo de inversión inmobiliaria o persona
              jurídica que tributa por el IS y está exenta.
            - Renta anual del arrendatario a ese arrendador < **900 €**.
            - El arrendador acredita que la actividad de arrendamiento está afecta
              a una actividad económica con local propio y empleado.

            ### Plazos (igual que el 111)
            - Trimestral (pymes): 1–20 del mes siguiente al trimestre.
            - Mensual (grandes empresas): 1–20 del mes siguiente.

            ### Resumen anual
            **Modelo 180** — resumen anual de las retenciones de arrendamientos,
            se presenta en enero del año siguiente.
            """),
    },
    {
        "category": "modelo_190",
        "lang": "gl",
        "prompt": "Que é o modelo 190 e en que prazo hai que presentalo?",
        "response": textwrap.dedent("""\
            ## Modelo 190 — Resumo anual de retencións do traballo e actividades

            O **Modelo 190** é o **resumo anual** das retencións e ingresos a conta
            do IRPF sobre rendementos do traballo persoal e de actividades económicas
            declaradas trimestralmente no **Modelo 111**.

            ### Contido
            - Relación nominalizada de todos os perceptores (empregados, profesionais)
              con retención practicada durante o exercicio.
            - Importes brutos abonados e retencións practicadas por cada perceptor.
            - Clave de percepción (A = traballo, G = actividades profesionais, etc.).

            ### Prazo de presentación
            - **1–31 de xaneiro** do ano seguinte ao exercicio.
            - Presentación exclusivamente telemática para obrigados.

            ### Como presentar
            1. **sede.agenciatributaria.gob.es** → Modelo 190.
            2. Podes importar o ficheiro de datos xerado polo teu programa de nóminas
               (formato TXT/BOE) ou cubrilo manualmente.

            ### Utilidade para o traballador
            Os datos do Modelo 190 son a fonte do **certificado de retencións** que a
            empresa entrega ao traballador e que a AEAT usa para xerar o borrador da
            declaración da renda.

            **Modelo equivalente para arrendamentos:** Modelo 180 (resumo anual
            do Modelo 115).
            """),
    },

    # ── Plusvalía municipal (IIVTNU) ──────────────────────────────────────────
    {
        "category": "plusvalia_municipal",
        "lang": "gl",
        "prompt": "Como funciona a plusvalía municipal ao vender un piso en Galicia?",
        "response": textwrap.dedent("""\
            ## Plusvalía municipal (IIVTNU) — Imposto sobre o Incremento de Valor dos
            ## Terreos de Natureza Urbana

            A **plusvalía municipal** é un tributo local que grava o incremento de valor
            experimentado polos terreos urbanos desde a adquisición ata a transmisión.

            **Xestiónano os Concellos** (non a Xunta nin a AEAT).

            ### Quen paga
            - **Compravenda:** o vendedor.
            - **Herdanza/doazón:** o adquirente (herdeiro/donatario).

            ### Dous métodos de cálculo (tras STC 182/2021)
            O contribuínte pode escoller o que resulte **menor**:

            **Método obxectivo (coeficientes):**
            ```
            Base impoñible = Valor catastral do terreo × Coeficiente (segundo anos)
            ```
            Os coeficientes son fixados anualmente por RD (máximos legais) e
            aprobados polo Concello.

            **Método real (incremento efectivo):**
            ```
            Base impoñible = Prezo venda − Prezo compra (proporción terreo)
            ```
            Útil cando a ganancia real é inferior ao cálculo obxectivo.

            ### Sen plusvalía
            - **Non hai feito impoñible** se non houbo incremento real de valor
              (vendes ao mesmo prezo ou con perda). Debes acreditalo con escrituras.

            ### Prazo de presentación
            - **30 días hábiles** desde a transmisión (compravenda).
            - **6 meses** (prorrogables a 1 ano) para herdanzas.
            - Ante o Concello onde está o inmoble.
            """),
    },

    # ── Certificado dixital / Cl@ve ───────────────────────────────────────────
    {
        "category": "certificado_dixital",
        "lang": "gl",
        "prompt": "Como obteño un certificado dixital para facer trámites online coa administración en Galicia?",
        "response": textwrap.dedent("""\
            ## Certificado dixital — Opcións para trámites coa administración

            Existen varias formas de identificarse dixitalmente ante as administracións
            (AEAT, Xunta, Seguridade Social...):

            ### 1. Certificado FNMT (Fábrica Nacional de Moeda e Timbre)
            - **Gratuíto.** O máis usado para autónomos e empresas.
            - **Pasos:**
              1. Accede a **sede.fnmt.es** → Certificados → Persoa Física → Solicitude.
              2. Obtén o **código de solicitude** no navegador.
              3. Acredita a túa identidade **presencialmente** nunha oficina habilitada
                 (AEAT, SS, Concellos adheridos, Delegacións do Goberno).
              4. Descarga o certificado no mesmo navegador onde fixeches a solicitude.
            - Vixencia: **4 anos** (renovable sen acudir presencialmente nos últimos
              60 días de vixencia).

            ### 2. DNI electrónico (DNIe)
            - Incorporado no DNI desde 2006. Require lector de tarxetas e software
              oficial da DXP (dnielectronico.es).
            - Vixencia ligada ao DNI físico.

            ### 3. Cl@ve PIN / Cl@ve Permanente
            - Sistema do Goberno central para cidadáns sen certificado.
            - **Cl@ve PIN:** código de uso único enviado por SMS. Válido para a maioría
              de trámites estatais.
            - **Cl@ve Permanente:** contrasinal de uso continuado. Require rexistro
              previo (presencial ou por carta de invitación).
            - Rexistro: clave.gob.es

            ### 4. Chave365 (Xunta de Galicia)
            - Sistema propio da Xunta para a sede electrónica galega (sede.xunta.gal).
            - Rexistro: chave365.xunta.gal

            ### Para representantes de empresa (persoa xurídica)
            - Certificado de **representante de persoa xurídica** emitido pola FNMT.
            - Require escritura de poder notarial e inscrición no Rexistro Mercantil.
            """),
    },

    # ── Sucesions — detalle grupos e prezentación ─────────────────────────────
    {
        "category": "atriga_sucesions_detalle",
        "lang": "gl",
        "prompt": "Cal é o prazo para presentar o Imposto de Sucesións en Galicia e como se tramita na ATRIGA?",
        "response": textwrap.dedent("""\
            ## Imposto sobre Sucesións — Prazo e tramitación en Galicia (ATRIGA)

            ### Prazo de presentación
            - **6 meses** contados desde o día do falecemento.
            - **Prórroga:** solicítase antes de que vezan os 6 meses iniciais;
              concédense outros 6 meses máis (total: 12 meses), pero durante a
              prórroga devénganse **xuros de demora** (tipo vixente ~4,0625%).

            ### Modelos
            | Modelo | Feito impoñible |
            |---|---|
            | **650** | Herdanzas (mortis causa) |
            | **651** | Doazóns e demais transmisións lucrativas inter vivos |

            ### Presentación
            1. **Telemática (recomendada):** atriga.gal → Oficina Virtual → Modelo 650.
               Require certificado dixital ou Cl@ve.
            2. **Presencial:** calquera oficina territorial de ATRIGA en Galicia
               (A Coruña, Lugo, Ourense, Pontevedra, Santiago, Vigo, Ferrol).

            ### Documentación necesaria
            - Certificado de defunción e de últimas vontades.
            - Copia autorizada do testamento (ou declaración de herdeiros ab intestato).
            - Inventario de bens e dereitos do causante con valoracións.
            - DNI/NIE dos herdeiros.
            - Escrituras de inmobles, certificados bancarios, valores mobiliarios.

            ### Bonificacións principais (Galicia 2024)
            - Grupos I e II (descendentes, ascendentes, cónxuxe): redución de
              **1.000.000 €** por herdeiro → cota cero na práctica na maioría dos casos.
            - Vivenda habitual: redución adicional **95%** (máx. 600.000 €).
            - Empresa familiar: redución **99%**.
            - Grupo III (colaterais 2.º e 3.º grao): redución 8.000 € + tarifa normal.
            - Grupo IV (estranhos): sen reducción específica autonómica.

            ATRIGA: atriga.gal · Tel.: 981 545 555
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
        ("Cal é o", "Cales son os"),
        ("hai que", "é necesario"),
        ("podo", "é posible"),
        ("en Galicia", "na Comunidade Autónoma de Galicia"),
        ("Redacta", "Elabora"),
        ("Elabora", "Prepara"),
        ("dunha", "para unha"),
        ("ao vender", "á hora de vender"),
        ("ao asinar", "á hora de asinar"),
        ("Como obteño", "Como consigo"),
        ("Como solicito", "Como podo solicitar"),
        ("Como se tramita", "Como funciona o trámite de"),
        ("Como presento", "Como se presenta"),
        ("Como funciona", "En que consiste"),
    ]
    reformulations_es = [
        ("¿Cómo", "¿De qué manera"),
        ("¿Qué", "¿Cuál es"),
        ("¿Cuál es", "¿Qué implica"),
        ("hay que", "es necesario"),
        ("puedo", "es posible"),
        ("en España", "en el territorio nacional"),
        ("Redacta", "Elabora"),
        ("Elabora", "Prepara"),
        ("¿En qué consiste", "¿Qué es"),
        ("¿Qué es", "¿Cómo funciona"),
        ("¿Cómo funciona", "¿En qué consiste"),
        ("¿Cómo se calcula", "¿Cuál es el método de cálculo de"),
        ("¿Quién está obligado", "¿A quién obliga"),
        ("al constituir", "cuando se constituye"),
        ("al vender", "cuando se vende"),
        ("diferencia entre", "comparativa entre"),
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
