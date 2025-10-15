import os
import streamlit as st
import google.generativeai as genai
from anthropic import Anthropic
from PyPDF2 import PdfReader
import pandas as pd
from datetime import datetime
from huggingface_hub import InferenceClient

# Carpeta donde están los archivos
DATA_DIR = "data"

# Leer todos los archivos TXT de la carpeta
def leer_docs_txt(data_dir):
    textos = []
    for archivo in os.listdir(data_dir):
        if archivo.endswith(".txt"):
            ruta = os.path.join(data_dir, archivo)
            with open(ruta, "r", encoding="utf-8") as f:
                textos.append(f.read())
    return "\n".join(textos)

# Guardar el contexto de los documentos
contexto = leer_docs_txt(DATA_DIR)

# Mostrar los archivos cargados
#st.write("Archivos cargados:", os.listdir(DATA_DIR))
#st.write("Contexto cargado (primeros 500 caracteres):", contexto[:500])


# =====================
# Configuración de claves
# =====================
google_key = st.secrets["GOOGLE_API_KEY"]
hf_token = st.secrets["HF_TOKEN"]

os.environ["GOOGLE_API_KEY"] = google_key
client = InferenceClient(token=hf_token)

# Inicialización de APIs

try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    #st.success("✅ Google Generative AI inicializado correctamente")
except Exception as e:
    st.error(f"❌ Error Google: {e}")

try:
    anthropic_client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
    #st.success("✅ Anthropic inicializado correctamente")
except Exception as e:
    st.error(f"❌ Error Anthropic: {e}")


# =====================
# Interfaz Streamlit
# =====================
#st.set_page_config(page_title="Comparador de modelos IA", layout="wide")
st.title("🤖 Comparador de modelos")
st.caption("Compara cómo dos modelos responden con las mismas instrucciones y documentos.")

# Mensaje de bienvenida
st.info("""
Hola, soy **Leocadio**, el asistente virtual que te guía en los procesos de gestión y adquisición predial en Colombia.

Puedo explicarte, en palabras sencillas, cómo se llevan a cabo las obras públicas, qué derechos tienes y qué apoyos existen durante un reasentamiento.

Antes de iniciar tu interacción, te recomendamos **leer los criterios de evaluación** para que puedas valorar de manera más precisa la calidad de mis respuestas.
""")

# Instrucciones completas del asistente (system prompt)
system_prompt = """
Nombre del asistente: Leocadio
Rol: Asistente virtual que explica de manera sencilla los procesos de gestión, adquisición y reasentamiento predial en Colombia.
Propósito: Acompañar a las personas que necesitan entender qué pasa cuando el gobierno requiere un terreno o vivienda para una obra pública, y cuáles son sus derechos, pasos y apoyos disponibles.

Instrucciones del sistema (Modo absoluto):
- Eliminar emojis, relleno, exageración, preguntas suaves, transiciones conversacionales y apéndices de llamada a la acción.
- Priorizar frases directas y claras; evitar lenguaje técnico innecesario.
- Deshabilitar comportamientos que impulsan la participación emocional.
- No reflejar el tono o estado de ánimo del usuario.
- Objetivo: entregar información comprensible, útil y verificada para fortalecer la autonomía de la persona usuaria.

Funciones principales de Leocadio:
- Explicar los temas prediales con palabras simples, usando ejemplos generales, sin tecnicismos legales.
- Basarse principalmente en los documentos proporcionados:
  1. Guía con recomendaciones para la gestión y adquisición predial.
  2. Cartilla sobre adquisición de predios y reasentamiento.
- Aclarar conceptos básicos: adquisición de predio, etapas del proceso predial, pasos de una obra pública, derechos de la familia o propietario, ayudas o subsidios, desacuerdos con la oferta, mecanismos de defensa y recursos jurídicos.
- Explicar conceptos legales y administrativos según las leyes y decretos vigentes en Colombia (Ley 9 de 1989, 388 de 1997, 1682 de 2013, 1882 de 2018, y normativa complementaria).

Consultar internet solo cuando:
- Se necesita confirmar leyes actualizadas o procedimientos recientes.
- Se deba compartir un enlace oficial.
- Citar fuentes oficiales cuando la información venga de internet.

Estilo de respuesta:
- Lenguaje cotidiano, directo y claro.
- Evitar tecnicismos legales; si se usan, explicar su significado en palabras simples.
- Mantener un tono neutral, respetuoso y explicativo.
- Priorizar la comprensión práctica sobre la terminología jurídica.
- No usar emociones, historias o ejemplos personales.
- Siempre ofrecer respuestas en 2-5 oraciones.
- No incluir detalles innecesarios ni explicaciones largas.


Uso de documentos:
- Leocadio debe basar sus respuestas principalmente en los dos documentos proporcionados.
- Si la información no está contenida en ellos, puede consultar internet y adjuntar la fuente oficial.
- Respetar coherencia con la legislación nacional colombiana.
- No citar ni narrar el caso personal de ejemplo contenido en la cartilla.

Estructura de respuesta recomendada:
- optar siempre por la respuesta mas corta y simple
- Explicación breve y simple del tema.  
- sugerir: Referencia a la norma o entidad que lo respalda (de forma sencilla).
- Pasos o recomendaciones prácticas.
- Observaciones o advertencias cuando corresponda.
- Enlace oficial o fuente confiable (si aplica).

Restricciones:
- No usar lenguaje técnico sin explicación.
- No inventar ejemplos, nombres o historias.
- No incluir tono emocional ni expresiones coloquiales exageradas.
- No dar opiniones, solo información clara y comprobable.
"""

# Documentos
prompt = st.text_area("✏️ Pregunta:", "¿Qué derechos tiene una familia durante un proceso de adquisición predial?")

# =====================
# Botón para consultar
# =====================

# Inicializamos las variables en session_state si no existen
if "respuestas" not in st.session_state:
    st.session_state.respuestas = []

# Cuando se hace clic en el botón
if st.button("Comparar modelos"):
    with st.spinner("Consultando modelos..."):

        # ==== Gemini ====
        def gemini_resp():
            try:
                m = genai.GenerativeModel("models/gemini-2.5-flash")
                full_prompt = f"{system_prompt}\n\nContexto de documentos:\n{contexto}\n\nPregunta:\n{prompt}"
                r = m.generate_content(full_prompt)
                return r.text
            except Exception as e:
                return f"❌ Error Gemini: {e}"

        # ==== Mistral (Hugging Face) ====
        def mistral_resp():
            try:
                full_prompt = f"{system_prompt}\n\nContexto de documentos:\n{contexto}\n\nPregunta:\n{prompt}"
                response = client.chat.completions.create(
                    model="mistralai/Mistral-7B-Instruct-v0.3",
                    messages=[
                        {"role": "system", "content": "Eres un asistente útil y claro."},
                        {"role": "user", "content": full_prompt},
                    ],
                    max_tokens=400,
                    temperature=0.7,
                )
                return response.choices[0].message["content"]
            except Exception as e:
                return f"❌ Error Mistral: {e}"


        # Generar respuestas
        respuesta_gemini = gemini_resp()
        respuesta_mistral = mistral_resp()

        # Guardar las respuestas en session_state
        st.session_state.respuestas.append({
            "pregunta": prompt,
            "gemini": respuesta_gemini,
            "mistral": respuesta_mistral
        })

# =====================
# Mostrar historial
# =====================
if st.session_state.respuestas:
    for idx, item in enumerate(reversed(st.session_state.respuestas)):
        st.markdown(f"### 🔹 Pregunta {len(st.session_state.respuestas) - idx}: {item['pregunta']}")
        colB, colC = st.columns(2)
        with colB:
            st.subheader("Chatbot 1 ") #(Gemini)
            st.markdown(f'<p style="text-align: justify;">{item["gemini"]}</p>', unsafe_allow_html=True)
        with colC:
            st.subheader("Chatbot 2 ") #(Mistral HF)
            st.markdown(f'<p style="text-align: justify;">{item["mistral"]}</p>', unsafe_allow_html=True)




st.header("📊Evaluación de bots")


# =====================
# Configuración inicial
# =====================
archivo_csv = "evaluaciones_chatbots.csv"  # archivo central

# Campos de identificación
st.subheader("👤 Información del evaluador")
rol = st.selectbox(
    "Rol del evaluador:",
    ["Estudiante", "Docente", "Analista", "Investigador", "Otro"]
)
experto = st.selectbox(
    "¿Cuál es su nivel de conocimiento en el tema evaluado?",
    ["Experto", "Intermedio", "Básico", "Sin conocimiento"]
)

# =====================
# Criterios de evaluación
# =====================
criterios = [
    ("Claridad de la respuesta", 
     "La respuesta es comprensible, está bien redactada y se adapta al nivel del usuario."),

    ("Relevancia y coherencia con el contexto", 
     "La respuesta se ajusta al tema o pregunta planteada y evita información innecesaria."),

    ("Tono y vocabulario adecuados", 
     "El lenguaje, tono y vocabulario son apropiados para el público objetivo y el entorno educativo o profesional."),

    ("Capacidad para responder correctamente", 
     "El chatbot proporciona información completa, precisa y útil para resolver la inquietud del usuario."),

    ("Ofrecimiento de recursos adicionales", 
     "El chatbot sugiere materiales, enlaces o ejemplos complementarios cuando se le solicita o son pertinentes.")
]
opciones = ["Sí", "A veces", "No"]


def combinar(nombre, desc):
    return f"{nombre}: {desc}"

df_eval = pd.DataFrame({
    "Criterio": [combinar(c[0], c[1]) for c in criterios],
    "Bot 1": [None]*len(criterios),
    "Bot 2": [None]*len(criterios)
})

st.write("### Criterios de evaluación")
st.caption("Selecciona la valoración para cada criterio y bot")




# Editor con soporte HTML (solo lectura visual del texto)
edited_df = st.data_editor(
    df_eval,
    column_config={
        "Criterio": st.column_config.Column("Criterio"),
        "Bot 1": st.column_config.SelectboxColumn("Bot 1", options=opciones),
        "Bot 2": st.column_config.SelectboxColumn("Bot 2", options=opciones),
    },
    num_rows="fixed",
    use_container_width=True
)

# =====================
# Guardar evaluaciones
# =====================
if st.button("💾 Guardar evaluación"):
    if not rol:
        st.warning("⚠️ Por favor, selecciona tu rol antes de guardar.")
    else:
        # Agregar metadatos
        edited_df["Rol"] = rol
        edited_df["Experto"] = experto
        edited_df["Fecha"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Si el archivo ya existe, cargarlo y agregar nuevos datos
        if os.path.exists(archivo_csv):
            df_existente = pd.read_csv(archivo_csv)
            df_final = pd.concat([df_existente, edited_df], ignore_index=True)
        else:
            df_final = edited_df

        # Guardar de nuevo
        df_final.to_csv(archivo_csv, index=False)
        st.success(f"✅ Evaluación guardada correctamente")

# =====================
# Mostrar resultados acumulados
# =====================
if os.path.exists(archivo_csv):
    df_mostrado = pd.read_csv(archivo_csv)
    total_registros = len(df_mostrado)
    st.write(f"### 📈 {total_registros} evaluaciones registradas hasta ahora")
    #st.dataframe(df_mostrado, use_container_width=True)
else:
    st.info("📭 Aún no hay evaluaciones registradas.")


