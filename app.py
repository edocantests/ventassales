import streamlit as st
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import tempfile
import os
import time

# Configuración de la página
st.set_page_config(
    page_title="YouTube Transcript Generator",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS personalizado para el diseño minimalista
st.markdown("""
<style>
    /* Estilos base */
    .stApp {
        background-color: #F8F9FA;
    }
    
    /* Contenedor principal */
    .main-container {
        background-color: white;
        padding: 64px;
        border-radius: 8px;
        box-shadow: 0 8px 24px rgba(17, 24, 39, 0.08);
        max-width: 768px;
        margin: 0 auto;
        margin-top: 40px;
    }
    
    /* Título principal */
    .main-title {
        color: #111827;
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 8px;
        text-align: center;
    }
    
    /* Subtítulo */
    .sub-title {
        color: #4B5563;
        font-size: 20px;
        font-weight: 500;
        margin-bottom: 32px;
        text-align: center;
    }
    
    /* Etiquetas */
    .form-label {
        color: #111827;
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 8px;
    }
    
    /* Campo de entrada */
    .stTextInput > div > div > input {
        background-color: #F8F9FA;
        border: 1px solid #D1D5DB;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 16px;
        color: #111827;
        height: 48px;
    }
    
    .stTextInput > div > div > input:focus {
        border: 2px solid #3B82F6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
        outline: none;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #9CA3AF;
    }
    
    /* Botón principal */
    .stButton > button {
        background-color: #3B82F6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 500;
        height: 48px;
        width: 100%;
        transition: all 250ms ease-out;
    }
    
    .stButton > button:hover {
        background-color: #2563EB;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
    }
    
    .stButton > button:active {
        transform: translateY(0px) scale(0.98);
    }
    
    /* Área de texto */
    .transcript-container {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 32px;
        margin-top: 32px;
        border: 1px solid #E5E7EB;
    }
    
    .transcript-title {
        color: #111827;
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 16px;
    }
    
    .transcript-text {
        color: #4B5563;
        font-size: 16px;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    
    /* Mensajes de estado */
    .success-message {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid #A7F3D0;
        margin: 16px 0;
    }
    
    .error-message {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid #FCA5A5;
        margin: 16px 0;
    }
    
    /* Información del video */
    .video-info {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 24px;
        border: 1px solid #93C5FD;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-container {
            padding: 32px;
            margin-top: 20px;
        }
        
        .main-title {
            font-size: 28px;
        }
        
        .sub-title {
            font-size: 18px;
        }
    }
</style>
""", unsafe_allow_html=True)

def extract_video_id(url):
    """Extrae el ID del video de una URL de YouTube"""
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
        r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)',
        r'(?:https?://)?(?:m\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def get_video_transcript(video_id):
    """Obtiene la transcripción de un video de YouTube"""
    try:
        # Intentar obtener transcripción en español primero, luego en inglés
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_list.find_transcript(['es', 'es-ES'])
        except:
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
            except:
                # Si no hay transcripciones automáticas, intentar con transcripción manual
                try:
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                    transcript = transcript_list.find_transcript(['es', 'en'])
                except:
                    return None, "No se encontraron transcripciones para este video"
        
        return transcript, None
        
    except Exception as e:
        return None, f"Error al acceder a la transcripción: {str(e)}"

def main():
    # Contenedor principal
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Título y subtítulo
    st.markdown('<h1 class="main-title">📝 YouTube Transcript Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Pega un enlace de YouTube y obtén la transcripción completa del video</p>', unsafe_allow_html=True)
    
    # Campo de entrada para la URL
    st.markdown('<p class="form-label">URL del video de YouTube:</p>', unsafe_allow_html=True)
    youtube_url = st.text_input(
        label="URL del video de YouTube",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Pega aquí el enlace completo del video de YouTube"
    )
    
    # Inicializar variables de estado
    if 'transcript' not in st.session_state:
        st.session_state.transcript = None
    if 'video_id' not in st.session_state:
        st.session_state.video_id = None
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    
    # Botón para generar transcripción
    if st.button("🎬 Generar Transcripción", key="generate_btn"):
        if not youtube_url:
            st.markdown('<div class="error-message">⚠️ Por favor, ingresa una URL de YouTube válida</div>', unsafe_allow_html=True)
        else:
            # Mostrar indicador de carga
            with st.spinner("🔄 Obteniendo transcripción..."):
                # Extraer ID del video
                video_id = extract_video_id(youtube_url)
                
                if not video_id:
                    st.markdown('<div class="error-message">❌ URL de YouTube no válida. Asegúrate de que el enlace sea correcto.</div>', unsafe_allow_html=True)
                else:
                    # Obtener transcripción
                    transcript, error = get_video_transcript(video_id)
                    
                    if error:
                        st.markdown(f'<div class="error-message">❌ {error}</div>', unsafe_allow_html=True)
                        st.session_state.transcript = None
                        st.session_state.video_id = None
                    else:
                        # Formatear transcripción
                        formatter = TextFormatter()
                        transcript_text = formatter.format_transcript(transcript.fetch())
                        
                        # Guardar en estado de sesión
                        st.session_state.transcript = transcript_text
                        st.session_state.video_id = video_id
                        
                        # Mostrar mensaje de éxito
                        st.markdown('<div class="success-message">✅ Transcripción generada exitosamente</div>', unsafe_allow_html=True)
    
    # Mostrar transcripción si existe
    if st.session_state.transcript:
        st.markdown('<div class="transcript-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="transcript-title">📄 Transcripción del Video</h3>', unsafe_allow_html=True)
        st.markdown(f'<div class="transcript-text">{st.session_state.transcript}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Opción de descarga
        if st.session_state.transcript:
            # Crear archivo temporal para descarga
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
            temp_file.write(st.session_state.transcript)
            temp_file.close()
            
            st.markdown('<br>', unsafe_allow_html=True)
            
            # Botón de descarga
            with open(temp_file.name, 'r', encoding='utf-8') as file:
                st.download_button(
                    label="💾 Descargar Transcripción (.txt)",
                    data=file.read(),
                    file_name=f"transcripcion_youtube_{st.session_state.video_id}.txt",
                    mime="text/plain",
                    key="download_btn"
                )
            
            # Limpiar archivo temporal
            try:
                os.unlink(temp_file.name)
            except:
                pass
    
    # Información adicional
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #F3F4F6; padding: 16px; border-radius: 8px; border-left: 4px solid #3B82F6;">
        <h4 style="color: #111827; margin: 0 0 8px 0; font-size: 16px;">💡 Características:</h4>
        <ul style="color: #4B5563; margin: 0; padding-left: 20px; font-size: 14px;">
            <li>Admite videos con transcripciones automáticas en español e inglés</li>
            <li>Genera transcripciones completas y formateadas</li>
            <li>Opción de descargar la transcripción en archivo de texto</li>
            <li>Interfaz limpia y fácil de usar</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Cerrar contenedor principal
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
