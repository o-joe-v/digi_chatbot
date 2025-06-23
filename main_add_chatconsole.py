import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv
import speech_recognition as sr
from datetime import datetime
import io
import time
import threading
from queue import Queue
import logging
import pyaudio
import wave
import tempfile
import azure.cognitiveservices.speech as speechsdk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging for audio processing"""
    log_handler = logging.StreamHandler()
    log_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)

def load_config():
    """Load configuration from environment variables"""
    load_dotenv()
    
    config = {
        'azure_oai_endpoint': os.getenv("AZURE_OAI_ENDPOINT"),
        'azure_oai_key': os.getenv("AZURE_OAI_KEY"),
        'azure_oai_deployment': os.getenv("AZURE_OAI_DEPLOYMENT"),
        'azure_search_endpoint': os.getenv("AZURE_SEARCH_ENDPOINT"),
        'azure_search_key': os.getenv("AZURE_SEARCH_KEY"),
        'azure_search_index': os.getenv("AZURE_SEARCH_INDEX"),
        'azure_api_version': os.getenv("AZURE_API_VERSION"),
        # Text-to-Speech configuration
        'azure_speech_key': os.getenv("AZURE_SPEECH_KEY"),
        'azure_speech_region': os.getenv("AZURE_SPEECH_REGION"),
        'azure_speech_voice': os.getenv("AZURE_SPEECH_VOICE")
    }
    
    return config

def validate_config(config):
    """Validate required configuration parameters"""
    required_params = ['azure_oai_endpoint', 'azure_oai_key', 'azure_oai_deployment']
    missing_params = [param for param in required_params if not config[param]]
    
    if missing_params:
        st.error(f"Missing required configuration: {', '.join(missing_params)}")
        return False
    
    return True

def text_to_speech(text, config):
    """Convert text to speech using Azure Speech Services"""
    try:
        logger.info(f"Converting text to speech: {text[:50]}...")
        
        # Create speech config
        speech_config = speechsdk.SpeechConfig(
            subscription=config['azure_speech_key'], 
            region=config['azure_speech_region']
        )
        speech_config.speech_synthesis_voice_name = config['azure_speech_voice']
        
        # Create speech synthesizer
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        
        # Perform text-to-speech
        result = speech_synthesizer.speak_text_async(text).get()
        
        # Check result
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info("Speech synthesis completed successfully")
            return True
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logger.error(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                logger.error(f"Error details: {cancellation_details.error_details}")
            return False
        else:
            logger.error(f"Speech synthesis failed with reason: {result.reason}")
            return False
            
    except Exception as e:
        logger.error(f"Text-to-speech error: {e}")
        return False

def call_azure_openai_with_search_rest(endpoint, api_key, deployment, search_endpoint, search_key, search_index, query, api_version):
    """Make a direct REST API call to Azure OpenAI with search integration"""
    
    # Clean the endpoint URL
    endpoint = endpoint.rstrip('/')
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    # Updated payload structures that should work with different API versions
    payloads_to_try = [
        # Structure 1: For API version 2024-02-15-preview and later
        {
            "messages": [
                {"role": "system", "content": "You are a helpful Loan agent that responds in Thai language"},
                {"role": "user", "content": query}
            ],
            "temperature": 0.0,
            "max_tokens": 1000,
            "data_sources": [{
                "type": "azure_search",
                "parameters": {
                    "endpoint": search_endpoint,
                    "index_name": search_index,
                    "authentication": {
                        "type": "api_key",
                        "key": search_key
                    },
                    "query_type": "simple",
                    "in_scope": True,
                    "top_n_documents": 5,
                    "role_information": "You are a helpful Loan agent that responds in Thai language"
                }
            }]
        },
        # Structure 2: Alternative format for different API versions
        {
            "messages": [
                {"role": "system", "content": "You are a helpful Loan agent that responds in Thai language"},
                {"role": "user", "content": query}
            ],
            "temperature": 0.0,
            "max_tokens": 1000,
            "data_sources": [{
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": search_endpoint,
                    "index_name": search_index,
                    "authentication": {
                        "type": "api_key",
                        "key": search_key
                    }
                }
            }]
        },
        # Structure 3: Simplified format
        {
            "messages": [
                {"role": "system", "content": "You are a helpful Loan agent that responds in Thai language"},
                {"role": "user", "content": query}
            ],
            "temperature": 0.0,
            "max_tokens": 1000,
            "data_sources": [{
                "type": "azure_search",
                "parameters": {
                    "endpoint": search_endpoint,
                    "index_name": search_index,
                    "key": search_key
                }
            }]
        },
        # Structure 4: Without search integration (fallback)
        {
            "messages": [
                {"role": "system", "content": "You are a helpful Loan agent that responds in Thai language"},
                {"role": "user", "content": query}
            ],
            "temperature": 0.0,
            "max_tokens": 1000
        }
    ]
    
    last_error = None
    
    for i, payload in enumerate(payloads_to_try, 1):
        try:
            logger.info(f"Attempting payload structure {i}")
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                logger.info("Successfully received response from Azure OpenAI")
                if i == 4:
                    logger.warning("Used fallback mode without search integration")
                return result['choices'][0]['message']['content']
            else:
                error_msg = f"Payload {i} failed with status code: {response.status_code}, Response: {response.text}"
                logger.warning(error_msg)
                last_error = error_msg
                continue
                
        except Exception as e:
            error_msg = f"Payload {i} failed with error: {str(e)}"
            logger.error(error_msg)
            last_error = error_msg
            continue
    
    # If all structures fail, provide detailed error information
    error_details = f"All API payload structures failed. Last error: {last_error}"
    logger.error(error_details)
    raise Exception(error_details)

def record_audio(duration=5):
    """Record audio from microphone"""
    try:
        logger.info("Starting microphone recording")
        
        # Audio recording parameters
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Open stream
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
        
        logger.info(f"Recording for {duration} seconds...")
        frames = []
        
        # Record audio
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        # Stop recording
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        logger.info("Recording completed")
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_filename = temp_file.name
        temp_file.close()
        
        # Save audio to file
        wf = wave.open(temp_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return temp_filename
        
    except Exception as e:
        logger.error(f"Error recording audio: {e}")
        return None

def transcribe_audio(audio_file_path):
    """Transcribe audio file to text using speech recognition"""
    try:
        logger.info("Starting audio transcription")
        r = sr.Recognizer()
        
        # Load audio file
        with sr.AudioFile(audio_file_path) as source:
            logger.info("Reading audio file")
            audio = r.record(source)
        
        # Try to recognize speech using Google Speech Recognition
        logger.info("Attempting speech recognition")
        text = r.recognize_google(audio, language='th-TH')  # Thai language
        logger.info(f"Successfully transcribed: {text}")
        return text
        
    except sr.UnknownValueError:
        logger.error("Could not understand audio")
        return "ไม่สามารถเข้าใจเสียงพูดได้"
    except sr.RequestError as e:
        logger.error(f"Speech recognition service error: {e}")
        return f"เกิดข้อผิดพลาดในการรู้จำเสียง: {e}"
    except Exception as e:
        logger.error(f"Audio transcription error: {e}")
        return f"เกิดข้อผิดพลาดในการแปลงเสียง: {e}"

def main():
    st.set_page_config(
        page_title="น้องบุญช่วย",
        page_icon="🤖",
        layout="wide"
    )
    
    setup_logging()
    
    # Header with gear button
    col1, col2 = st.columns([6, 1])
    
    with col1:
        st.title("🤖 น้องบุญช่วย")
    
    with col2:
        # Initialize session state for settings visibility
        if 'show_settings' not in st.session_state:
            st.session_state.show_settings = False
        
        # Gear button
        if st.button("⚙️", key="settings_toggle", help="Settings"):
            st.session_state.show_settings = not st.session_state.show_settings
    
    st.markdown("---")
    
    # Load configuration
    config = load_config()
    
    # Settings section (toggleable)
    if st.session_state.show_settings:
        with st.expander("⚙️ Configuration Settings", expanded=True):
            if validate_config(config):
                st.success("✅ Configuration OK")
                st.info(f"**Endpoint:** {config['azure_oai_endpoint']}")
                st.info(f"**Deployment:** {config['azure_oai_deployment']}")
                st.info(f"**API Version:** {config['azure_api_version']}")
                
                # Additional configuration details
                if config['azure_search_endpoint']:
                    st.info(f"**Search Endpoint:** {config['azure_search_endpoint']}")
                    st.info(f"**Search Index:** {config['azure_search_index']}")
                else:
                    st.warning("⚠️ Search integration not configured")
                
                # Text-to-Speech settings
                st.info(f"**Speech Region:** {config['azure_speech_region']}")
                st.info(f"**Speech Voice:** {config['azure_speech_voice']}")
                
            else:
                st.error("❌ Configuration Error")
                st.error("Please check your environment variables (.env file)")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    
    # Chat history with scrollable container
    st.subheader("💬 Chat")
    
    # Add custom CSS for chat styling
    st.markdown("""
        <style>
        .stContainer > div {
            max-height: 400px;
            overflow-y: auto;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Create scrollable chat container
    chat_container = st.container(height=400)
    
    with chat_container:
        # Display messages using Streamlit chat messages
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message['content'])
                    st.caption(f"⏰ {message.get('timestamp', '')}")
            else:
                with st.chat_message("assistant"):
                    st.write(message['content'])
                    st.caption(f"⏰ {message.get('timestamp', '')}")
        
        # Auto-scroll to bottom when new message is added
        if st.session_state.messages:
            st.empty()  # This helps trigger scroll to bottom
    
    # Input methods
    st.markdown("---")
    st.subheader("📝 Input Methods")
    
    input_tab1, input_tab2 = st.tabs(["🎤 Voice Input", "💬 Text Input"])
    
    with input_tab1:
        st.info("🎤 Click to record audio from microphone")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            duration = st.selectbox("Recording Duration", [3, 5, 7, 10], index=1, key="duration_select")
        
        with col2:
            if st.button("🔴 Start Recording", key="start_recording"):
                if validate_config(config):
                    # Initialize recording state
                    if 'recording' not in st.session_state:
                        st.session_state.recording = False
                    
                    if not st.session_state.recording:
                        st.session_state.recording = True
                        
                        with st.spinner(f"🎤 Recording for {duration} seconds..."):
                            audio_file_path = record_audio(duration)
                            
                        st.session_state.recording = False
                        
                        if audio_file_path:
                            st.success("✅ Recording completed!")
                            
                            # Transcribe audio
                            with st.spinner("🔄 Converting speech to text..."):
                                transcribed_text = transcribe_audio(audio_file_path)
                            
                            # Clean up temporary file
                            try:
                                os.remove(audio_file_path)
                            except:
                                pass
                            
                            if transcribed_text and not transcribed_text.startswith("ไม่สามารถ") and not transcribed_text.startswith("เกิดข้อผิดพลาด"):
                                st.success(f"📝 Transcribed: {transcribed_text}")
                                
                                # Auto-send the transcribed text
                                process_query(transcribed_text, config)
                            else:
                                st.error("❌ Could not transcribe audio. Please try again.")
                        else:
                            st.error("❌ Failed to record audio. Please check your microphone.")
        
        with col3:
            if st.button("🔄 Reset", key="reset_recording"):
                if 'recording' in st.session_state:
                    st.session_state.recording = False
                st.success("Reset completed")
    
    with input_tab2:
        text_input = st.text_area("พิมพ์คำถามของคุณ:", height=100, key="text_input")
        if st.button("📤 Send", key="send_text"):
            if text_input and validate_config(config):
                process_query(text_input, config)
    
    # Logs section
    with st.expander("📋 System Logs"):
        if st.session_state.logs:
            for log in st.session_state.logs[-10:]:  # Show last 10 logs
                st.text(log)
        else:
            st.info("No logs yet")
        
        if st.button("Clear Logs"):
            st.session_state.logs = []
            st.rerun()

def process_query(query, config):
    """Process user query and get response from Azure OpenAI"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add user message
    st.session_state.messages.append({
        "role": "user", 
        "content": query,
        "timestamp": timestamp
    })
    
    # Add log
    log_message = f"[{timestamp}] User Query: {query}"
    st.session_state.logs.append(log_message)
    logger.info(log_message)
    
    try:
        with st.spinner("กำลังประมวลผล..."):
            # Check if search parameters are available
            if not all([config['azure_search_endpoint'], config['azure_search_key'], config['azure_search_index']]):
                response = "ระบบค้นหาไม่พร้อมใช้งาน กรุณาตรวจสอบการตั้งค่า"
                log_message = f"[{timestamp}] Warning: Search parameters missing"
            else:
                response = call_azure_openai_with_search_rest(
                    config['azure_oai_endpoint'],
                    config['azure_oai_key'],
                    config['azure_oai_deployment'],
                    config['azure_search_endpoint'],
                    config['azure_search_key'],
                    config['azure_search_index'],
                    query,
                    config['azure_api_version']
                )
                log_message = f"[{timestamp}] Successfully received AI response"
            
            st.session_state.logs.append(log_message)
            logger.info(log_message)
            
            # Add assistant response
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
    except Exception as e:
        error_message = f"เกิดข้อผิดพลาด: {str(e)}"
        log_message = f"[{timestamp}] Error: {str(e)}"
        
        st.session_state.logs.append(log_message)
        logger.error(log_message)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": error_message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        st.error(error_message)
    
    st.rerun()

if __name__ == "__main__":
    main()