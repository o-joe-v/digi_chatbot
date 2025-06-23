<p align="center">
  <img src="https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/images/streamlit-logo-secondary-light.svg" alt="Streamlit Logo" width="200"/>
</p>

# 🤖 น้องบุญช่วย (Nong Boon Chuay) - AI Chatbot with Azure OpenAI & RAG

น้องบุญช่วย คือ AI Chatbot ที่พัฒนาด้วย Streamlit ซึ่งใช้ประโยชน์จาก Azure OpenAI Service สำหรับการสนทนา และ Azure AI Search สำหรับการดึงข้อมูล (Retrieval Augmented Generation - RAG) เพื่อให้คำตอบที่แม่นยำและมีบริบทมากขึ้น Chatbot นี้รองรับทั้งการป้อนข้อมูลด้วยเสียงและข้อความ รวมถึงการตอบกลับด้วยเสียง (Text-to-Speech) โดยเน้นการใช้งานในภาษาไทย

## ✨ คุณสมบัติหลัก

-   **อินเทอร์เฟซการสนทนาแบบโต้ตอบ:** สร้างด้วย Streamlit เพื่อประสบการณ์ผู้ใช้ที่ใช้งานง่าย
-   **การป้อนข้อมูลด้วยเสียง (Speech-to-Text):** รองรับการบันทึกเสียงจากไมโครโฟนและแปลงเป็นข้อความด้วย `speech_recognition` (Google Speech Recognition)
-   **การตอบกลับด้วยเสียง (Text-to-Speech):** แปลงข้อความตอบกลับจาก AI เป็นเสียงพูดด้วย Azure Cognitive Services Speech SDK
-   **การผสานรวม Azure OpenAI:** ใช้สำหรับสร้างคำตอบในการสนทนา
-   **Retrieval Augmented Generation (RAG):** ดึงข้อมูลที่เกี่ยวข้องจาก Azure AI Search เพื่อให้คำตอบที่อ้างอิงจากแหล่งข้อมูลที่กำหนด
-   **การตั้งค่าที่ยืดหยุ่น:** สามารถกำหนดค่า Endpoint, API Key, Deployment, และการตั้งค่าอื่นๆ ผ่านไฟล์ `.env`
-   **บันทึกระบบ (System Logs):** แสดงบันทึกการทำงานเพื่อช่วยในการ Debug และตรวจสอบ
-   **รองรับภาษาไทย:** ออกแบบมาเพื่อการใช้งานในภาษาไทยเป็นหลัก

## 🚀 เทคโนโลยีที่ใช้

-   **Python**
-   **Streamlit:** สำหรับสร้าง Web UI
-   **Azure OpenAI Service:** สำหรับโมเดลภาษาขนาดใหญ่ (LLM)
-   **Azure AI Search:** สำหรับการจัดการและดึงข้อมูล
-   **Azure Cognitive Services Speech SDK:** สำหรับ Speech-to-Text และ Text-to-Speech
-   `python-dotenv`: สำหรับจัดการ Environment Variables
-   `speech_recognition`: สำหรับ Speech-to-Text
-   `pyaudio`: สำหรับการบันทึกเสียง
-   `requests`: สำหรับการเรียกใช้ REST API

## ⚙️ การติดตั้งและตั้งค่า

### ข้อกำหนดเบื้องต้น

-   Python 3.8+
-   `pip` (Python package installer)
-   บัญชี Azure ที่มีบริการ Azure OpenAI, Azure AI Search และ Azure Cognitive Services Speech

### Environment Variables

สร้างไฟล์ `.env` ใน Root Directory ของโปรเจกต์ และเพิ่มข้อมูลการตั้งค่าดังนี้:

```
AZURE_OAI_ENDPOINT="https://[your-openai-resource-name].openai.azure.com/"
AZURE_OAI_KEY="[your-openai-api-key]"
AZURE_OAI_DEPLOYMENT="[your-openai-deployment-name]"
AZURE_API_VERSION="2024-06-01" # หรือเวอร์ชันที่เหมาะสมกับ deployment ของคุณ

AZURE_SEARCH_ENDPOINT="https://[your-search-service-name].search.windows.net"
AZURE_SEARCH_KEY="[your-search-api-key]"
AZURE_SEARCH_INDEX="[your-search-index-name]"

AZURE_SPEECH_KEY="[your-speech-service-key]"
AZURE_SPEECH_REGION="[your-speech-service-region]" # เช่น eastus2, southeastasia
AZURE_SPEECH_VOICE="th-TH-PremwadaNeural" # หรือเสียงอื่นๆ ที่ต้องการ เช่น th-TH-PremwadeeNeural
```

### การติดตั้ง

1.  **Clone Repository:**
    ```bash
    git clone [URL_TO_YOUR_REPO]
    cd digi_chatbot
    ```

2.  **สร้าง Virtual Environment (แนะนำ):**
    ```bash
    python -m venv venv
    # สำหรับ Windows
    .\venv\Scripts\activate
    # สำหรับ macOS/Linux
    source venv/bin/activate
    ```

3.  **ติดตั้ง Dependencies:**
    ```bash
    pip install -r requirements.txt
    # หากไม่มี requirements.txt คุณสามารถติดตั้งทีละแพ็คเกจได้:
    # pip install streamlit python-dotenv speechrecognition pyaudio requests azure-cognitiveservices-speech
    ```
    *หมายเหตุ: การติดตั้ง `pyaudio` อาจต้องใช้ขั้นตอนเพิ่มเติมบนบางระบบปฏิบัติการ หากพบปัญหา โปรดดูเอกสารของ `pyaudio`*

## 🚀 วิธีใช้งาน

หลังจากตั้งค่า Environment Variables และติดตั้ง Dependencies เรียบร้อยแล้ว ให้รันแอปพลิเคชัน Streamlit:

```bash
streamlit run main_voice_chat_console.py
```

แอปพลิเคชันจะเปิดขึ้นในเบราว์เซอร์ของคุณ

### การใช้งานในแอป

-   **ส่วน Chat:** แสดงประวัติการสนทนา
-   **Voice Input:** กดปุ่ม "🔴 Start Recording" เพื่อบันทึกเสียงคำถามของคุณ
-   **Text Input:** พิมพ์คำถามของคุณในช่องข้อความแล้วกด "📤 Send"
-   **Configuration Settings:** กดปุ่ม "⚙️" ที่มุมขวาบนเพื่อเปิด/ปิดส่วนการตั้งค่า ซึ่งคุณสามารถ "🔍 Test Connection" เพื่อตรวจสอบการเชื่อมต่อกับ Azure Services ได้
-   **System Logs:** ขยายส่วนนี้เพื่อดูบันทึกการทำงานของระบบ ซึ่งมีประโยชน์ในการ Debug

## 🤝 การมีส่วนร่วม

ยินดีต้อนรับการมีส่วนร่วม! หากคุณมีข้อเสนอแนะ, พบ Bug, หรือต้องการเพิ่มคุณสมบัติใหม่ๆ สามารถเปิด Issue หรือส่ง Pull Request ได้เลยครับ

## 📄 License

โปรเจกต์นี้อยู่ภายใต้ MIT License