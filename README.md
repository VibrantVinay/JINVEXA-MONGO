# 🧠 Jinvexa Learning AI

Jinvexa is an AI-powered personalized learning platform that transforms how you acquire new skills. It doesn't just give you a list of links; it builds a custom "university" tailored to your goals, current knowledge, and preferred learning style.

> **License Notice**
> This project is proprietary. No use, copying, modification, or distribution is permitted without prior written permission.
> See the [LICENSE](LICENSE) file for full terms.

## 🏗️ System Architecture

The platform is built in layers, moving from raw data extraction to personalized mentorship.

### 1. Input Layer (Data Extraction) ✅
**Status: Complete**
Responsible for ingesting information from various sources:
- 📺 **YouTube**: Transcripts and video metadata.
- 🌐 **Websites**: Article and documentation parsing.
- 📄 **Documents**: PDF and DOCX processing.
- 🖼️ **Images**: OCR for text extraction from images.

### 2. Planning Layer (Discovery) ✅
**Status: Complete**
The "Brain" of the system that analyzes content and user profiles:
- **Concept Extraction**: Identifying core topics and subtopics.
- **Dependency Mapping**: Determining the logical order of learning.
- **Knowledge Gap Analysis**: Comparing user's current knowledge with the goal.
- **Learning Plan Generation**: Creating a structured roadmap.
- **Memory Handler**: Persistent storage of user profiles and session history.

### 3. Teaching Layer (Execution) ✅
**Status: Complete**
Generates full courses from learning plans with AI-driven format selection and parallel processing:
- **AI Format Selection**: LLM decides per-topic whether to generate text, female voice audio, or male voice audio based on content type and learner level.
- **Multithreaded Generation**: Uses `ThreadPoolExecutor` (max 5 workers) to generate all lessons in parallel — dramatically faster than sequential generation.
- **Text Lessons**: Comprehensive markdown-formatted lessons with introduction, core concepts, examples, exercises, and next steps.
- **Audio Lessons**: TTS-generated audio files using `edge-tts` with gender-specific voices (female for warm/beginner topics, male for professional/advanced topics).
- **Structured Output**: All content organized under `learn_files/`:
  - `learn_files/lessons/` — Text lesson files (`.txt`)
  - `learn_files/audio/` — Audio lesson files (`.mp3`)
  - `learn_files/manifests/` — JSON manifest + human-readable watch order
- **Manifest System**: Generates a JSON manifest with complete watch order, content types, file paths, and format reasons — enabling playlist-style consumption.
- **Course Metadata**: Saves course metadata to session storage for status tracking and resumption.

### 4. Assessment Layer (Evaluation) ✅
**Status: Complete**
Generates and evaluates personalized assignments from course content with AI-driven auto-configuration:
- **Auto-Configuration**: LLM analyzes course complexity (number of lessons, content length, phases) and user profile (level, goals, past performance) to automatically determine:
  - Number of MCQ questions (3-10)
  - Number of written/essay questions (1-4)
  - Difficulty level (beginner / intermediate / advanced)
  - Passing score (60-80%)
- **MCQ Generation**: AI generates multiple-choice questions with 4 options, correct answer index, topic tags, and explanations.
- **Written Question Generation**: AI creates open-ended essay questions with scoring rubrics (clarity, correctness, examples).
- **Auto-Grading**: 
  - MCQ answers are auto-graded instantly by comparing against correct indices.
  - Written answers are evaluated by the LLM with detailed feedback and scores.
- **Progress Tracking**: Tracks user performance across all assignments with:
  - Average, best, and latest scores
  - Performance trends (improving / consistent / needs attention)
  - Weak area identification
  - Personalized recommendations
- **Certificate Eligibility**: Automatically checks if a user qualifies for a certificate (3+ assignments with 70%+ average).
- **Structured Storage**: All assignments and results organized under `learn_files/assignments/`:
  - `sessions/` — Generated assignment JSON files
  - `results/` — Evaluated results per user
  - `progress/` — Progress summary files
- **Fallback System**: If LLM configuration fails, an intelligent rule-based fallback adjusts question counts based on lesson count and content length.

### 5. Mentoring Layer (Guidance) ✅
**Status: Complete**
AI-powered mentoring chatbot that provides personalized guidance and conversation-based learning:
- **Two Modes**:
  - **Session Mode**: Chat about one specific course — the mentor has full context of that course's content, lessons, and learning plan.
  - **Full Mode**: Chat about all your learning across all sessions — the mentor sees your entire learning journey.
- **Conversation Memory**: SQLite-based persistent storage for conversation history with:
  - Auto-garbage collection (cleans conversations older than 7 days)
  - Message limit management (truncates overly long conversations)
  - Session content caching for fast access
- **Interactive Commands**:
  - `quit` — End the mentoring session
  - `clear` — Reset conversation history
  - `summary` — View conversation stats (message count, duration)
- **Context-Aware Responses**: LLM-powered answers that reference actual course content, learning plans, and user progress.
- **History & Resume**: View past conversations and continue them from where you left off.
- **Structured Storage**: All mentoring data organized under `learn_files/mentoring/`:
  - `mentoring_memory.db` — SQLite database with conversations, messages, and content cache

---

## 🚀 Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/nosij-playz/Jinvexa.git
   cd Jinvexa
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your `.env` file:
   ```env
   OLLAMA_MODEL=gemma4:31b-cloud
   STORAGE_DIR=memory_storage
   STORAGE_TYPE=json
   ```

### Running the App
```bash
python app.py
```

### 📋 Available Modes
| # | Mode | Description |
|---|------|-------------|
| 1 | Goal-Based Learning | Interactive discovery conversation to build a learning plan from a goal |
| 2 | Reference-Based Learning | Extract content from URL/video/doc and build a learning plan |
| 3 | Teaching Layer | Generate full course with text & audio lessons from a learning plan |
| 4 | Assignment Layer | Take AI-generated MCQ & written assignments with auto-grading |
| 5 | Mentoring Layer | Chat with AI Mentor (session-specific or full learning context) |
| 6 | Mentoring History | View and resume past mentoring conversations |
| 7 | View Progress | Track assignment scores, trends, and certificate eligibility |
| 8 | View Stats | See user learning statistics and knowledge progress |
| 9 | View Sessions | Browse all learning sessions |
| 10 | Continue Conversation | Resume a previous discovery conversation |
| 11 | Teaching Status | Check course generation status for a session |
| 12 | Model Info | Display current Ollama model information |
| 13 | Change Model | Switch Ollama model and view capabilities |
| 14 | Exit | Quit the application |

## License

This project is proprietary.

Copyright © 2026 Jison Joseph Sebastian. All Rights Reserved.

No use, copying, modification, or distribution is permitted without prior written permission.

For licensing or collaboration requests, please contact:

Email: jisonjosephsebastian7007@gmail.com
contact form:https://jisonjosephsebastian.work.gd/contact
