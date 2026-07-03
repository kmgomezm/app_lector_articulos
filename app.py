"""
Lector de Artículos Científicos
--------------------------------
Sube un PDF, detecta secciones, traduce al español (si hace falta) con Groq,
genera un resumen breve por sección y permite escuchar todo con gTTS.

Modelos usados (verificar vigencia en https://console.groq.com/docs/models,
Groq cambia/retira modelos con frecuencia):
    - LLM (traducción + resúmenes): openai/gpt-oss-120b  (gratis en Groq)
    - TTS: gTTS (Google Text-to-Speech no oficial, 100% gratis, sin API key)
"""

import io
import json
import re

import fitz  # PyMuPDF
import streamlit as st
from gtts import gTTS
from langdetect import detect
from groq import Groq

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------

MODEL_NAME = "openai/gpt-oss-120b"  # cambiar aquí si Groq retira el modelo

st.set_page_config(page_title="Lector de Artículos Científicos", page_icon="🔊", layout="wide")

SECTION_KEYWORDS = [
    "abstract", "resumen",
    "introduction", "introducción", "introduccion",
    "background", "antecedentes",
    "related work", "trabajos relacionados",
    "literature review", "revisión de literatura", "revision de literatura",
    "materials and methods", "material and methods", "materiales y métodos",
    "methodology", "methods", "method", "método", "métodos", "metodologia", "metodología",
    "results", "resultados",
    "discussion", "discusión", "discusion",
    "results and discussion", "resultados y discusión",
    "conclusion", "conclusions", "conclusión", "conclusiones",
    "limitations", "limitaciones",
    "future work", "trabajo futuro", "trabajos futuros",
    "acknowledgments", "acknowledgements", "agradecimientos",
    "references", "referencias",
    "bibliography", "bibliografía", "bibliografia",
]


# --------------------------------------------------------------------------
# Extracción y segmentación del PDF
# --------------------------------------------------------------------------
#
# Estrategia principal: detección por ESTILO TIPOGRÁFICO. La mayoría de los
# artículos y capítulos usan un tamaño/peso de fuente distinto para los
# títulos de sección frente al cuerpo del texto (p. ej. 16pt en negrita vs.
# 13pt normal). Esto es mucho más robusto que buscar palabras clave fijas
# como "Introducción" o "Métodos", porque funciona con encabezados libres
# ("Presentación", "Antecedentes...", etc.), como en informes, capítulos de
# libro o reportes que no siguen la estructura clásica IMRaD.
#
# Si el PDF no tiene información de estilo aprovechable (por ejemplo, un
# PDF plano sin variación de fuente), se usa como respaldo el método anterior
# basado en palabras clave típicas de artículos científicos (Abstract,
# Methods, Results...).

BOLD_HINTS = ("bold", "black", "heavy", "semibold")  # "medium" se excluye a propósito:
# en muchos PDFs se usa para bylines/autores, no para títulos de sección.

PYMUPDF_BOLD_FLAG = 1 << 4  # bit "bold" en el campo `flags` de cada span de PyMuPDF


def _is_bold_font(fontname: str, flags: int) -> bool:
    f = (fontname or "").lower()
    if any(h in f for h in BOLD_HINTS):
        return True
    return bool(flags & PYMUPDF_BOLD_FLAG)


def extract_styled_lines(pdf_bytes: bytes):
    """Extrae cada línea del PDF con su página, texto, tamaño de fuente y si es negrita.

    Devuelve (lines, meta_title, total_pages), donde lines es una lista de
    dicts en orden de lectura: {"page", "text", "size", "bold"}.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    lines = []
    for pno, page in enumerate(doc):
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                # Usar el span más largo como representativo del tamaño/fuente de la línea
                main_span = max(spans, key=lambda s: len(s.get("text", "")))
                lines.append(
                    {
                        "page": pno,
                        "text": text,
                        "size": round(main_span.get("size", 0), 1),
                        "bold": _is_bold_font(main_span.get("font", ""), main_span.get("flags", 0)),
                    }
                )
    meta_title = (doc.metadata.get("title") or "").strip()
    total_pages = doc.page_count
    doc.close()
    return lines, meta_title, total_pages


def _remove_running_headers(lines, total_pages: int):
    """Quita encabezados/pies de página repetidos y números de página sueltos."""
    freq = {}
    for l in lines:
        freq[l["text"]] = freq.get(l["text"], 0) + 1
    threshold = max(3, int(total_pages * 0.25))
    noise = {t for t, c in freq.items() if c >= threshold}

    def is_page_number(t):
        return bool(re.fullmatch(r"\d{1,4}", t.strip()))

    return [l for l in lines if l["text"] not in noise and not is_page_number(l["text"])]


def _body_font_size(lines):
    """Tamaño de fuente predominante (ponderado por caracteres) = tamaño del cuerpo del texto."""
    weighted = {}
    for l in lines:
        weighted[l["size"]] = weighted.get(l["size"], 0) + len(l["text"])
    if not weighted:
        return 12.0
    return max(weighted.items(), key=lambda kv: kv[1])[0]


def parse_sections_by_style(pdf_bytes: bytes):
    """Segmenta el PDF en secciones usando el tamaño/negrita de la fuente.

    Devuelve (preamble, sections, meta_title, debug_info) o None si no logra
    detectar encabezados confiables (para que el llamador use el método de
    respaldo). `debug_info` sirve para mostrar en la interfaz qué se detectó.
    """
    lines, meta_title, total_pages = extract_styled_lines(pdf_bytes)
    if not lines:
        return None

    clean = _remove_running_headers(lines, total_pages)
    body_size = _body_font_size(clean)

    def is_heading_line(l):
        if len(l["text"]) >= 150:
            return False
        bold_and_bigger = l["bold"] and l["size"] >= body_size + 1
        # Señal de respaldo: salto de tamaño notorio aunque no se detecte negrita
        # (acotado por arriba para no capturar títulos de portada, mucho más grandes).
        size_jump_only = body_size + 2 <= l["size"] <= body_size + 12
        return bold_and_bigger or size_jump_only

    # Agrupar líneas de encabezado consecutivas (títulos que ocupan 2+ líneas)
    merged_headings = []  # (idx_inicio, texto_encabezado_completo)
    i = 0
    while i < len(clean):
        if is_heading_line(clean[i]):
            start = i
            parts = [clean[i]["text"]]
            j = i + 1
            while j < len(clean) and is_heading_line(clean[j]):
                parts.append(clean[j]["text"])
                j += 1
            merged_headings.append((start, " ".join(parts)))
            i = j
        else:
            i += 1

    debug_info = {
        "metodo": "estilo (tamaño/negrita de fuente)",
        "body_size": body_size,
        "n_lineas": len(clean),
        "n_encabezados": len(merged_headings),
        "encabezados": [h for _, h in merged_headings],
    }

    if len(merged_headings) < 2:
        return None, debug_info  # señal insuficiente: usar método de respaldo

    preamble_lines = clean[: merged_headings[0][0]]
    preamble = "\n".join(l["text"] for l in preamble_lines).strip()

    sections = []
    for idx, (start, heading) in enumerate(merged_headings):
        # saltar las líneas que componen el propio encabezado (pueden ser varias)
        n_heading_lines = 1
        k = start + 1
        while k < len(clean) and is_heading_line(clean[k]):
            n_heading_lines += 1
            k += 1
        content_start = start + n_heading_lines
        content_end = merged_headings[idx + 1][0] if idx + 1 < len(merged_headings) else len(clean)
        content_lines = clean[content_start:content_end]
        content = "\n".join(l["text"] for l in content_lines).strip()
        if content:
            sections.append({"heading": heading.strip(), "text": content})

    if not sections:
        return None, debug_info

    return (preamble, sections, meta_title), debug_info


# --- Método de respaldo (palabras clave), para PDFs sin estilo aprovechable ---

def _clean_line(line: str) -> str:
    l = line.strip()
    l = re.sub(r"^[ivxlcdm]+[\.\)]\s*", "", l, flags=re.IGNORECASE)
    l = re.sub(r"^\d+(\.\d+)*[\.\)]?\s*", "", l)
    return l.strip()


def _match_heading_keyword(line: str):
    line = line.strip()
    if not (3 <= len(line) <= 80):
        return None
    cleaned = _clean_line(line)
    low = cleaned.lower().rstrip(":").strip()
    for kw in SECTION_KEYWORDS:
        if low == kw or low == kw + "s":
            return cleaned if cleaned else line
    return None


def parse_sections_by_keywords(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n"
    meta_title = (doc.metadata.get("title") or "").strip()
    doc.close()

    lines = full_text.split("\n")
    headings_idx = [(i, h) for i, l in enumerate(lines) if (h := _match_heading_keyword(l))]

    debug_info = {
        "metodo": "palabras clave (respaldo)",
        "n_encabezados": len(headings_idx),
        "encabezados": [h for _, h in headings_idx],
    }

    if not headings_idx:
        return "", [{"heading": "Contenido completo", "text": full_text.strip()}], meta_title, debug_info

    preamble = "\n".join(lines[: headings_idx[0][0]]).strip()
    sections = []
    for idx, (line_i, heading) in enumerate(headings_idx):
        start = line_i + 1
        end = headings_idx[idx + 1][0] if idx + 1 < len(headings_idx) else len(lines)
        content = "\n".join(lines[start:end]).strip()
        if content:
            sections.append({"heading": heading, "text": content})

    if not sections:
        return preamble, [{"heading": "Contenido completo", "text": full_text.strip()}], meta_title, debug_info

    return preamble, sections, meta_title, debug_info


def extract_and_segment(pdf_bytes: bytes):
    """Punto de entrada único: intenta el método por estilo y, si no es
    confiable, recurre al método por palabras clave.

    Devuelve (preamble, sections, meta_title, debug_info).
    """
    result, debug_info = parse_sections_by_style(pdf_bytes)
    if result is not None:
        preamble, sections, meta_title = result
        return preamble, sections, meta_title, debug_info
    preamble, sections, meta_title, debug_info_fallback = parse_sections_by_keywords(pdf_bytes)
    debug_info_fallback["motivo_respaldo"] = (
        f"El método por estilo solo encontró {debug_info.get('n_encabezados', 0)} "
        f"encabezado(s) confiables (se requieren al menos 2)."
    )
    debug_info_fallback["debug_estilo"] = debug_info
    return preamble, sections, meta_title, debug_info_fallback


def extract_title(preamble: str, meta_title: str) -> str:
    if meta_title and 4 < len(meta_title) < 300:
        return meta_title
    for line in preamble.split("\n"):
        l = line.strip()
        if 15 < len(l) < 200:
            return l
    return "Título no identificado"


# --------------------------------------------------------------------------
# Groq: traducción y resumen
# --------------------------------------------------------------------------

def get_groq_client():
    api_key = st.session_state.get("groq_api_key")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def _chunk_text(text: str, max_chars: int = 6000):
    paragraphs = text.split("\n")
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) + 1 > max_chars:
            if current:
                chunks.append(current)
            current = p
        else:
            current = f"{current}\n{p}" if current else p
    if current:
        chunks.append(current)
    return chunks or [text]


def translate_text(client: Groq, text: str, needs_translation: bool) -> str:
    """Traduce texto (posiblemente largo) del inglés al español, por fragmentos."""
    if not text or not text.strip():
        return text
    if not needs_translation:
        return text

    system_prompt = (
        "Traduce al español el siguiente fragmento de un artículo científico. "
        "Responde ÚNICAMENTE con la traducción, sin comentarios, notas ni encabezados "
        "adicionales, conservando párrafos y todo el contenido original."
    )
    translated_chunks = []
    for chunk in _chunk_text(text):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk},
                ],
                temperature=0.2,
            )
            translated_chunks.append(resp.choices[0].message.content.strip())
        except Exception as e:
            st.warning(f"No se pudo traducir un fragmento (se deja el original): {e}")
            translated_chunks.append(chunk)
    return "\n\n".join(translated_chunks)


def summarize_text(client: Groq, text: str, sentences: int = 3) -> str:
    """Genera un resumen breve en español de una sección (ya en español)."""
    if not text or not text.strip():
        return ""
    system_prompt = (
        f"Lee el siguiente texto, una sección de un artículo científico en español, "
        f"y escribe un resumen de exactamente {sentences} oraciones en español, claro "
        f"y directo, explicando de qué trata la sección. Responde solo con el resumen, "
        f"sin introducciones ni comillas."
    )
    excerpt = text[:8000]
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": excerpt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"No se pudo generar el resumen de una sección: {e}")
        return ""


# --------------------------------------------------------------------------
# Texto a voz (gTTS)
# --------------------------------------------------------------------------

def text_to_speech_bytes(text: str, lang: str = "es"):
    if not text or not text.strip():
        return None
    try:
        tts = gTTS(text=text, lang=lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf
    except Exception as e:
        st.warning(f"No se pudo generar el audio: {e}")
        return None


# --------------------------------------------------------------------------
# Interfaz de usuario
# --------------------------------------------------------------------------

st.title("🔊 Lector de Artículos Científicos")
st.caption(
    "Sube un artículo en PDF, tradúcelo al español por secciones y escúchalo, "
    "con un resumen previo de cada sección."
)

with st.sidebar:
    st.header("Configuración")
    default_key = ""
    try:
        default_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
    api_key_input = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        help="Clave gratuita en https://console.groq.com/keys",
    )
    if api_key_input:
        st.session_state["groq_api_key"] = api_key_input

    st.markdown("---")
    st.markdown(f"**Modelo de texto:** `{MODEL_NAME}` (Groq, gratuito)")
    st.markdown("**Voz:** Google TTS — gTTS (gratuito, sin API key)")
    st.markdown("---")
    n_sentences = st.slider("Oraciones por resumen de sección", 1, 6, 3)

uploaded_file = st.file_uploader("Sube el artículo en PDF", type=["pdf"])

if uploaded_file and st.button("📄 Procesar artículo", type="primary"):
    client = get_groq_client()
    if client is None:
        st.error("Ingresa tu Groq API Key en la barra lateral antes de procesar.")
        st.stop()

    with st.spinner("Extrayendo texto del PDF..."):
        pdf_bytes = uploaded_file.read()
        preamble, raw_sections, meta_title, debug_info = extract_and_segment(pdf_bytes)

    with st.expander("🔍 Diagnóstico de detección de secciones", expanded=False):
        st.write(f"**Método usado:** {debug_info.get('metodo')}")
        if "body_size" in debug_info:
            st.write(f"**Tamaño de fuente del cuerpo detectado:** {debug_info['body_size']} pt")
        st.write(f"**Encabezados detectados:** {debug_info.get('n_encabezados', 0)}")
        if debug_info.get("encabezados"):
            st.write(debug_info["encabezados"])
        if "motivo_respaldo" in debug_info:
            st.warning(debug_info["motivo_respaldo"])
            if debug_info.get("debug_estilo", {}).get("encabezados"):
                st.write("Candidatos que sí detectó el método por estilo (pero fueron menos de 2):")
                st.write(debug_info["debug_estilo"]["encabezados"])

    sample = (preamble + " " + " ".join(s["text"] for s in raw_sections))[:1000]
    try:
        lang = detect(sample) if sample.strip() else "es"
    except Exception:
        lang = "en"
    needs_translation = lang != "es"

    title = extract_title(preamble, meta_title)

    # Separar el abstract/resumen y descartar listas de referencias/bibliografía:
    # no aportan valor al escucharlas y consumirían llamadas a Groq innecesariamente.
    REFERENCE_HEADINGS = (
        "references", "referencias", "bibliography", "bibliografía", "bibliografia",
        "works cited", "obras citadas", "reference list",
    )
    abstract_text = ""
    body_sections = []
    omitted_reference_sections = []
    for s in raw_sections:
        low_heading = s["heading"].lower().strip(":")
        if low_heading in ("abstract", "resumen") and not abstract_text:
            abstract_text = s["text"]
        elif any(low_heading == kw or low_heading.startswith(kw) for kw in REFERENCE_HEADINGS):
            omitted_reference_sections.append(s["heading"])
        else:
            body_sections.append(s)

    if omitted_reference_sections:
        st.caption(
            "ℹ️ Se omitió del procesamiento la(s) sección(es) de referencias/bibliografía: "
            + ", ".join(omitted_reference_sections)
        )

    with st.spinner("Traduciendo título..."):
        title_es = translate_text(client, title, needs_translation)

    with st.spinner("Traduciendo el resumen (abstract)..."):
        if abstract_text:
            abstract_es = translate_text(client, abstract_text, needs_translation)
        else:
            abstract_es = "No se identificó un resumen (abstract) independiente en el artículo."

    processed_sections = []
    progress = st.progress(0.0, text="Procesando secciones...")
    total = max(len(body_sections), 1)
    for i, s in enumerate(body_sections):
        heading_es = translate_text(client, s["heading"], needs_translation)
        text_es = translate_text(client, s["text"], needs_translation)
        summary = summarize_text(client, text_es, sentences=n_sentences)
        processed_sections.append(
            {
                "heading": heading_es,
                "text": text_es,
                "summary": summary,
            }
        )
        progress.progress((i + 1) / total, text=f"Sección {i + 1}/{total} procesada")
    progress.empty()

    st.session_state["article"] = {
        "title": title_es,
        "abstract": abstract_es,
        "sections": processed_sections,
    }
    st.session_state["audio_cache"] = {}
    st.success("¡Artículo procesado! Desplázate hacia abajo para leerlo y escucharlo.")

# --------------------------------------------------------------------------
# Mostrar artículo procesado
# --------------------------------------------------------------------------

if "article" in st.session_state:
    article = st.session_state["article"]
    audio_cache = st.session_state.setdefault("audio_cache", {})

    st.header(article["title"])

    st.subheader("Resumen (Abstract)")
    st.write(article["abstract"])
    if st.button("🔊 Escuchar resumen", key="btn_abstract"):
        if "abstract" not in audio_cache:
            with st.spinner("Generando audio..."):
                audio_cache["abstract"] = text_to_speech_bytes(article["abstract"])
    if audio_cache.get("abstract"):
        st.audio(audio_cache["abstract"], format="audio/mp3")

    st.markdown("---")
    st.subheader("Contenido del artículo")

    if not article["sections"]:
        st.info("No se detectaron secciones adicionales en este artículo.")

    for idx, sec in enumerate(article["sections"]):
        with st.expander(f"{idx + 1}. {sec['heading']}", expanded=(idx == 0)):
            st.markdown(f"**Esta sección trata de:** {sec['summary']}")
            st.write(sec["text"])

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔊 Escuchar resumen", key=f"btn_sum_{idx}"):
                    key = f"sum_{idx}"
                    if key not in audio_cache:
                        with st.spinner("Generando audio..."):
                            intro = f"{sec['heading']}. Esta sección trata de: {sec['summary']}"
                            audio_cache[key] = text_to_speech_bytes(intro)
                if audio_cache.get(f"sum_{idx}"):
                    st.audio(audio_cache[f"sum_{idx}"], format="audio/mp3")
            with col2:
                if st.button("🔊 Escuchar sección completa", key=f"btn_full_{idx}"):
                    key = f"full_{idx}"
                    if key not in audio_cache:
                        with st.spinner("Generando audio (puede tardar en secciones largas)..."):
                            full_audio_text = f"{sec['heading']}. {sec['text']}"
                            audio_cache[key] = text_to_speech_bytes(full_audio_text)
                if audio_cache.get(f"full_{idx}"):
                    st.audio(audio_cache[f"full_{idx}"], format="audio/mp3")
else:
    st.info("Sube un PDF y presiona 'Procesar artículo' para comenzar.")
