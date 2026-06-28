# 🚁 Drone Security Analyst Agent

An end-to-end AI-powered drone surveillance system that processes drone footage using AI, Computer Vision, and Large Language Models to automatically detect suspicious activities, generate alerts, and enable semantic search over surveillance videos.

---

## ✨ Features

- 🎥 Automated drone video processing
- 🧠 YOLOv8 object detection
- 🎯 ByteTrack object tracking
- 👁️ Claude Haiku Vision scene understanding
- 🔍 Semantic search using OpenCLIP + ChromaDB
- 🤖 LangGraph AI Security Agent
- 🚨 Intelligent threat detection using Claude Sonnet
- 💬 Natural language Q&A over surveillance footage
- 📊 Streamlit Dashboard
- ⚡ FastAPI backend with asynchronous processing

---

## 🏗️ System Architecture

```
Drone Video
     │
     ▼
Frame Extraction (OpenCV)
     │
     ▼
YOLOv8 Object Detection
     │
     ▼
ByteTrack Tracking
     │
     ▼
Claude Haiku Vision
     │
     ▼
OpenCLIP Embeddings
     │
     ▼
ChromaDB + MongoDB
     │
     ▼
LangGraph Security Agent
     │
     ▼
Claude Sonnet Alert Reasoning
     │
     ▼
Streamlit Dashboard
```

---

## 🛠 Tech Stack

| Category | Technology |
|-----------|------------|
| Backend | FastAPI |
| Dashboard | Streamlit |
| Detection | YOLOv8 |
| Tracking | ByteTrack |
| Vision LLM | Claude Haiku |
| Reasoning LLM | Claude Sonnet |
| Agent Framework | LangGraph |
| Embeddings | OpenCLIP ViT-B/32 |
| Vector DB | ChromaDB |
| Database | MongoDB |
| Image Processing | OpenCV |
| Cloud | AWS Bedrock |

---

## 📂 Project Structure

```text
drone-security-agent/
│
├── agents/
├── database/
├── routes/
├── utils/
├── input_folder/
├── output/
├── main.py
├── app.py
├── streamlit_app.py
├── requirements.txt
└── .env.example
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/drone-security-agent.git

cd drone-security-agent
```

### Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file.

```env
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=

MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=drone_agent
```

---

## ▶️ Run the Application

### Start FastAPI

```bash
python main.py
```

FastAPI

```
http://localhost:8000
```

Swagger

```
http://localhost:8000/docs
```

### Start Streamlit

```bash
streamlit run streamlit_app.py
```

Dashboard

```
http://localhost:8501
```

---

## 🚀 Workflow

1. Place drone videos inside `input_folder/`
2. Start the FastAPI server
3. Launch Streamlit
4. Click **Start Pipeline**
5. Monitor processing progress
6. View generated alerts
7. Perform semantic search
8. Ask natural language questions about surveillance footage

---

## 📡 API Endpoints

### Pipeline

- POST `/api/process`
- GET `/api/job/{job_id}`
- GET `/api/jobs`
- GET `/api/status`
- POST `/api/reprocess/{video}`
- POST `/api/redo-step/{video}/{step}`

### Alerts

- POST `/api/alerts/run/{video}`
- GET `/api/alerts/all`
- GET `/api/alerts/video/{video}`

### Search

- POST `/api/search`
- POST `/api/chat`

---

## 🤖 AI Components

- YOLOv8
- ByteTrack
- Claude Haiku (AWS Bedrock)
- Claude Sonnet (AWS Bedrock)
- LangGraph
- OpenCLIP
- ChromaDB
- MongoDB

---

## 📈 Pipeline

```
Video
   │
   ▼
Frame Extraction
   │
   ▼
YOLO Detection
   │
   ▼
ByteTrack
   │
   ▼
Claude Vision
   │
   ▼
OpenCLIP Embeddings
   │
   ▼
Vector Storage
   │
   ▼
LangGraph Agent
   │
   ▼
Alert Generation
```

---

## 📷 Dashboard

The Streamlit dashboard provides:

- Pipeline Monitoring
- Live Job Progress
- Alert Dashboard
- Semantic Search
- AI Chat Assistant

---

## 🔮 Future Improvements

- Real-time RTSP stream support
- Multi-drone monitoring
- Email & SMS alerts
- Kubernetes deployment
- Docker support
- Role-based authentication
- AWS deployment

---

## 👨‍💻 Author

**Karan Karale**

Senior GenAI Tech Lead

LinkedIn: https://www.linkedin.com/in/karankarale

---

