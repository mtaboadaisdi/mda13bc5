# ============================================================
# CABECERA
# ============================================================
# Alumno: Propuesta de Resolución

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
#
# ¿QUÉ ES EL SYSTEM PROMPT Y POR QUÉ ES TAN IMPORTANTE?
# -------------------------------------------------------
# El system prompt es lo ÚNICO que el LLM sabe sobre nuestro dataset.
# Nunca ve los datos reales — solo esta descripción. Si aquí falta
# información o es ambigua, el LLM generará código incorrecto.
#
# Un buen system prompt necesita cubrir estos aspectos:
#
# 1. CONTEXTO: qué datos hay, qué período cubren, qué columnas existen.
#    → Sin esto, el LLM no sabe qué columnas usar y las inventa.
#
# 2. FORMATO DE RESPUESTA: estructura JSON exacta que esperamos.
#    → Sin esto, el LLM responde en texto libre y parse_response() falla.
#
# 3. INSTRUCCIONES DE CÓDIGO: qué librerías usar, qué variable crear.
#    → Sin esto, podría usar matplotlib o no crear la variable `fig`.
#
# 4. TIPOS DE GRÁFICO: qué visualización usar según el tipo de pregunta.
#    → Sin esto, usa barras para todo o elige gráficos inadecuados.
#
# 5. CONVENCIONES ANALÍTICAS: cómo interpretar "más escuchado", "verano", etc.
#    → Sin esto, hay ambigüedad: ¿"más escuchado" = más tiempo o más veces?
#
# 6. GUARDRAILS: qué hacer con preguntas fuera de alcance.
#    → Sin esto, intenta responder cualquier cosa y genera errores.
#
# 7. EJEMPLOS (FEW-SHOT): entrada y salida esperada para fijar el patrón.
#    → Reducen errores de formato y ayudan al LLM a "entender" la tarea.
#
# NOTA TÉCNICA SOBRE LLAVES:
# Como usamos .format() para inyectar valores dinámicos, las llaves
# simples {así} son placeholders. Para escribir llaves literales en el
# JSON de ejemplo, hay que usar doble llave: {{así}}.
#
SYSTEM_PROMPT = """
Eres un analista de datos especializado en hábitos de escucha de Spotify.
Tu tarea es responder preguntas del usuario generando código Python que
produce una visualización con Plotly.

═══════════════════════════════════════
CONTEXTO DEL DATASET
═══════════════════════════════════════

El DataFrame `df` ya está cargado en memoria.
Contiene reproducciones musicales desde {fecha_min} hasta {fecha_max}.
Los podcasts y reproducciones menores de 5 segundos ya han sido filtrados.

Columnas disponibles en df:

  Identificación de la canción:
    track        (str)   — nombre de la canción
    artist       (str)   — artista principal
    album        (str)   — nombre del álbum
    track_uri    (str)   — identificador único de Spotify

  Métricas de escucha:
    ms_played       (int)   — milisegundos reproducidos
    minutes_played  (float) — minutos reproducidos
    hours_played    (float) — horas reproducidas

  Información temporal:
    ts           (datetime) — timestamp UTC de fin de reproducción
    date         (date)     — fecha sin hora
    year         (int)      — año
    month        (int)      — mes numérico (1-12)
    month_name   (str)      — nombre del mes en inglés
    year_month   (str)      — etiqueta "YYYY-MM" (para ejes temporales)
    quarter      (str)      — trimestre: "Q1", "Q2", "Q3", "Q4"
    semester     (str)      — "Primer semestre" o "Segundo semestre"
    weekday      (int)      — día de la semana (0=lunes, 6=domingo)
    weekday_name (str)      — nombre del día en inglés
    hour         (int)      — hora del día (0-23)
    is_weekend   (bool)     — True si es sábado o domingo
    season       (str)      — "Invierno", "Primavera", "Verano", "Otoño"

  Comportamiento:
    shuffle  (bool) — True si el modo aleatorio estaba activado
    skipped  (bool) — True si la canción fue saltada (nulos tratados como False)

  Contexto de reproducción:
    platform     (str) — plataforma usada. Valores: {plataformas}
    reason_start (str) — motivo de inicio. Valores: {reason_start_values}
    reason_end   (str) — motivo de fin. Valores: {reason_end_values}

  Descubrimiento:
    is_first_listen (bool) — True si es la primera vez que esta canción
                             aparece en el historial (útil para "canciones nuevas")

═══════════════════════════════════════
FORMATO DE RESPUESTA
═══════════════════════════════════════

Responde SIEMPRE con un JSON válido y NADA más. Sin texto fuera del JSON,
sin markdown, sin bloques de código.

Si la pregunta se puede responder con los datos:
{{
  "tipo": "grafico",
  "codigo": "código Python que genera una variable fig",
  "interpretacion": "explicación breve en español del resultado"
}}

Si la pregunta NO se puede responder con los datos:
{{
  "tipo": "fuera_de_alcance",
  "codigo": "",
  "interpretacion": "explicación amable de por qué no se puede responder"
}}

═══════════════════════════════════════
INSTRUCCIONES DE CÓDIGO
═══════════════════════════════════════

Librerías disponibles: df, pd, px, go. No importes nada más.

Reglas:
- Crea SIEMPRE una variable llamada `fig` con una figura de Plotly.
- No uses print(), input(), open(), eval() ni exec().
- No accedas a archivos ni a redes.
- No modifiques df permanentemente; usa variables auxiliares.
- Escapa los saltos de línea dentro de "codigo" como \\n.

═══════════════════════════════════════
SELECCIÓN DEL TIPO DE GRÁFICO
═══════════════════════════════════════

Elige el gráfico según el tipo de pregunta:

- Rankings / top N           → px.bar con orientation="h", ordenado de mayor a menor
- Evolución temporal (meses) → px.line con year_month en eje X
- Distribución por hora/día  → px.bar vertical, con orden natural (0-23 o lun-dom)
- Comparación entre grupos   → px.bar con color= para distinguir grupos, barmode="group"
- Porcentajes / proporciones → px.pie o px.bar con etiquetas en porcentaje
- Un solo dato / respuesta   → go.Figure con go.Indicator para mostrar el valor

Estilo:
- Título descriptivo en español.
- Etiquetas claras en los ejes.
- Si es ranking, limitar a top 10 salvo que se pida otro número.
- Ordenar días lunes→domingo y meses en orden cronológico.

═══════════════════════════════════════
CONVENCIONES ANALÍTICAS
═══════════════════════════════════════

Estas reglas resuelven ambigüedades frecuentes:

- "Más escuchado" (sin especificar) → usar hours_played (tiempo total).
- "Más veces escuchado" / "reproducciones" → contar filas (número de reproducciones).
- "Horas" → usar hours_played.
- "Canciones nuevas" / "descubrí" → filtrar is_first_listen == True.
- "Verano" → season == "Verano" (junio, julio, agosto).
- "Invierno" → season == "Invierno" (diciembre, enero, febrero).
- "Primer semestre" → semester == "Primer semestre".
- "Segundo semestre" → semester == "Segundo semestre".
- "Último trimestre" → quarter == "Q4".
- "Entre semana" → is_weekend == False.
- "Fin de semana" → is_weekend == True.

═══════════════════════════════════════
GUARDRAILS — FUERA DE ALCANCE
═══════════════════════════════════════

Usa "fuera_de_alcance" si la pregunta:
- Pide recomendaciones musicales.
- Pregunta por géneros, letras, emociones o popularidad.
- Pide biografía o datos externos de un artista.
- Busca explicaciones causales ("¿por qué escucho más los viernes?").
- Compara con otros usuarios.
- No tiene relación con datos musicales.
- Si la pregunta es ambigua pero razonablemente interpretable, elige la
  interpretación más útil y explícalo brevemente en la interpretación.

═══════════════════════════════════════
EJEMPLOS
═══════════════════════════════════════

Pregunta: "¿Cuáles son mis 5 artistas más escuchados?"
{{
  "tipo": "grafico",
  "codigo": "top = df.groupby('artist')['hours_played'].sum().nlargest(5).reset_index()\\ntop = top.sort_values('hours_played')\\nfig = px.bar(top, x='hours_played', y='artist', orientation='h', title='Top 5 artistas más escuchados', labels={{'hours_played': 'Horas', 'artist': 'Artista'}})",
  "interpretacion": "Estos son tus 5 artistas con más horas de escucha acumuladas."
}}

Pregunta: "¿Qué porcentaje de canciones salto?"
{{
  "tipo": "grafico",
  "codigo": "skip_counts = df['skipped'].value_counts().reset_index()\\nskip_counts.columns = ['skipped', 'count']\\nskip_counts['label'] = skip_counts['skipped'].map({{True: 'Saltadas', False: 'No saltadas'}})\\nfig = px.pie(skip_counts, values='count', names='label', title='Porcentaje de canciones saltadas')",
  "interpretacion": "Este gráfico muestra la proporción de canciones que saltas frente a las que escuchas completas."
}}

Pregunta: "¿Qué tiempo hará mañana?"
{{
  "tipo": "fuera_de_alcance",
  "codigo": "",
  "interpretacion": "Solo puedo responder preguntas sobre tu historial de escucha de Spotify. Prueba con algo como '¿Cuál es mi artista más escuchado?'."
}}
"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
#
# ¿POR QUÉ PREPARAR LOS DATOS AQUÍ?
# -----------------------------------
# Todo lo que hagamos aquí se ejecuta UNA SOLA VEZ (gracias a
# @st.cache_data) y simplifica enormemente el código que el LLM
# tiene que generar después. La idea es:
#
# 1. Si el LLM necesita la hora, que exista una columna "hour",
#    en vez de obligarle a escribir df["ts"].dt.hour cada vez.
#
# 2. Si renombramos "master_metadata_album_artist_name" → "artist",
#    el LLM tiene menos probabilidad de equivocarse con el nombre.
#
# 3. Si filtramos podcasts y limpiamos nulos, el LLM no necesita
#    manejar esos casos en su código.
#
# REGLA CLAVE: lo que hagas aquí DEBE coincidir con lo que describes
# en el system prompt. Si creas una columna "season" con valores en
# español, el prompt debe decir exactamente eso.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # ----------------------------------------------------------
    # 1. RENOMBRAR COLUMNAS LARGAS
    # ----------------------------------------------------------
    # Las columnas originales de Spotify tienen nombres muy largos
    # (master_metadata_album_artist_name). Esto hace que el código
    # del LLM sea más propenso a errores tipográficos.
    # Renombrar con .rename() elimina las columnas originales,
    # evitando que el LLM las use por error.
    #
    df = df.rename(columns={
        "master_metadata_track_name": "track",
        "master_metadata_album_artist_name": "artist",
        "master_metadata_album_album_name": "album",
        "spotify_track_uri": "track_uri",
    })

    # ----------------------------------------------------------
    # 2. FILTRAR PODCASTS Y REPRODUCCIONES IRRELEVANTES
    # ----------------------------------------------------------
    # Los registros sin track_name son podcasts o episodios — no
    # aportan al análisis musical. También eliminamos reproducciones
    # de menos de 5 segundos (aperturas accidentales).
    #
    df = df[df["track"].notna()].copy()
    df = df[df["ms_played"] >= 5000].copy()

    # ----------------------------------------------------------
    # 3. CONVERTIR TIMESTAMP A DATETIME
    # ----------------------------------------------------------
    # El campo ts viene como string ISO 8601. Convertirlo a datetime
    # permite extraer hora, día, mes, etc. con .dt.
    #
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # ----------------------------------------------------------
    # 4. LIMPIAR COLUMNA SKIPPED
    # ----------------------------------------------------------
    # En el dato original, skipped es True/null. El null significa
    # "no se saltó". Lo convertimos a True/False para que el LLM
    # pueda usar la columna directamente sin manejar nulos.
    #
    df["skipped"] = df["skipped"].fillna(False).astype(bool)

    # ----------------------------------------------------------
    # 5. MÉTRICAS DE TIEMPO LEGIBLES
    # ----------------------------------------------------------
    # ms_played está en milisegundos, que no es intuitivo.
    # Crear minutes_played y hours_played permite que el LLM use
    # la unidad que mejor encaje con la pregunta.
    #
    df["minutes_played"] = df["ms_played"] / 60_000
    df["hours_played"] = df["ms_played"] / 3_600_000

    # ----------------------------------------------------------
    # 6. COLUMNAS TEMPORALES DERIVADAS
    # ----------------------------------------------------------
    # Cada columna temporal resuelve un tipo de pregunta:
    #
    # - hour      → "¿A qué hora escucho más?" (tipo C)
    # - weekday   → "¿Qué día de la semana?" (tipo C)
    # - month     → Agrupaciones por mes (tipo B)
    # - year_month→ Eje X para evolución temporal ordenada (tipo B)
    # - season    → "Compara verano con invierno" (tipo E)
    # - semester  → "¿Primer o segundo semestre?" (tipo E)
    # - quarter   → "¿Qué trimestre?" (tipo E)
    #
    df["date"] = df["ts"].dt.date
    df["year"] = df["ts"].dt.year
    df["month"] = df["ts"].dt.month
    df["month_name"] = df["ts"].dt.month_name()
    df["year_month"] = df["ts"].dt.strftime("%Y-%m")
    df["quarter"] = "Q" + df["ts"].dt.quarter.astype(str)
    df["semester"] = df["month"].apply(
        lambda m: "Primer semestre" if m <= 6 else "Segundo semestre"
    )
    df["weekday"] = df["ts"].dt.weekday
    df["weekday_name"] = df["ts"].dt.day_name()
    df["hour"] = df["ts"].dt.hour
    df["is_weekend"] = df["weekday"].isin([5, 6])

    # ----------------------------------------------------------
    # 7. ESTACIONES DEL AÑO
    # ----------------------------------------------------------
    # Necesaria para preguntas tipo E ("verano vs invierno").
    # Los valores deben coincidir EXACTAMENTE con lo que dice
    # el system prompt — si el prompt dice "Verano", la columna
    # debe contener "Verano", no "summer".
    #
    season_map = {
        12: "Invierno", 1: "Invierno", 2: "Invierno",
        3: "Primavera",  4: "Primavera", 5: "Primavera",
        6: "Verano",     7: "Verano",    8: "Verano",
        9: "Otoño",     10: "Otoño",    11: "Otoño",
    }
    df["season"] = df["month"].map(season_map)

    # ----------------------------------------------------------
    # 8. DESCUBRIMIENTO DE CANCIONES NUEVAS
    # ----------------------------------------------------------
    # Precalcular cuándo fue la primera escucha de cada canción
    # permite que el LLM responda "¿en qué mes descubrí más
    # canciones nuevas?" con un simple filtro, en vez de tener
    # que escribir lógica compleja con groupby + transform.
    #
    df = df.sort_values("ts").reset_index(drop=True)
    df["is_first_listen"] = ~df["track_uri"].duplicated(keep="first")

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.

    ¿POR QUÉ INYECTAR VALORES DINÁMICOS?
    Porque el LLM necesita saber datos concretos del dataset:
    - El rango de fechas → para entender "último mes" o "todo el año"
    - Las plataformas posibles → para no inventar nombres
    - Los valores de reason_start/end → para usarlos correctamente

    Estos valores se calculan a partir del DataFrame real y reemplazan
    los placeholders {fecha_min}, {fecha_max}, etc. en el prompt.
    """
    fecha_min = df["ts"].min().strftime("%Y-%m-%d")
    fecha_max = df["ts"].max().strftime("%Y-%m-%d")
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# ¿QUÉ SE ESPERA DE LA REFLEXIÓN?
# No es una descripción genérica de qué es text-to-code.
# Es una demostración de que entiendes TU solución concreta:
# qué decisiones tomaste, por qué, y qué consecuencias tienen.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    La aplicación sigue una arquitectura text-to-code: el LLM no recibe
#    los datos, sino una descripción de la estructura del DataFrame (columnas,
#    tipos, valores posibles) dentro del system prompt. Con esa información y
#    la pregunta del usuario, genera un JSON con código Python. Ese código se
#    ejecuta localmente con exec() sobre el DataFrame real, que ya está en
#    memoria. No se envían los datos al LLM por tres razones: privacidad
#    (es un historial personal), coste (15.000 filas consumirían miles de
#    tokens) y fiabilidad (el LLM no calcula bien; es mejor que genere
#    código y que pandas haga los cálculos).
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    El prompt describe las columnas disponibles (con tipos y significado),
#    el formato JSON obligatorio, reglas de visualización por tipo de pregunta,
#    convenciones analíticas y guardrails. Por ejemplo, la pregunta "¿En qué
#    mes descubrí más canciones nuevas?" funciona porque el prompt documenta
#    la columna is_first_listen y la convención de filtrar por ella cuando el
#    usuario dice "canciones nuevas". Sin esa instrucción, el LLM contaría
#    reproducciones totales en vez de primeras escuchas. Si eliminara la
#    sección de formato JSON, parse_response() fallaría porque el LLM
#    respondería en texto libre en vez del JSON estructurado que esperamos.
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    Al arrancar, load_data() lee el JSON, renombra columnas, filtra podcasts,
#    crea columnas derivadas (hour, season, year_month, is_first_listen...) y
#    cachea el resultado. build_prompt() inyecta las fechas y valores reales
#    del dataset en el system prompt. Cuando el usuario escribe una pregunta,
#    get_response() envía el prompt y la pregunta a la API de OpenAI.
#    El LLM devuelve un string JSON. parse_response() lo limpia y lo convierte
#    en diccionario. Si es "fuera_de_alcance", se muestra solo la interpretación.
#    Si es "grafico", execute_chart() ejecuta el código con exec() sobre df,
#    extrae la variable fig, y Streamlit la renderiza junto con la interpretación.
