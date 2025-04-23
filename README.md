# 🍳 Recipe App Backend

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.12-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📖 About

A powerful recipe management backend API built with FastAPI and Python. This application leverages advanced AI models to enhance recipe management and generation capabilities.

## 🚀 Features

- 🤖 AI-powered recipe generation
- 📝 Recipe management (CRUD operations)
- 🔍 Advanced recipe search
- 🎯 Multiple AI model support (MT5, GPT-2, BLOOM)
- 🔒 Secure API endpoints

## 🛠 Tech Stack

- **Framework:** FastAPI
- **Language:** Python 3.9+
- **AI Models:**
  - MT5
  - GPT-2
  - BLOOM
- **Development Tools:**
  - Uvicorn (ASGI Server)
  - Pydantic (Data Validation)
  - Python-dotenv (Environment Management)

## 🏗 Project Structure

```
recipe-app-backend/
├── main.py              # Main FastAPI application
├── main_hf.py          # HuggingFace models integration
├── main_openai.py      # OpenAI integration
├── openai_client.py    # OpenAI client configuration
├── train.py            # Model training utilities
├── requirements.txt    # Project dependencies
└── .env               # Environment variables (not in git)
```

## 🚦 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. Clone the repository
```bash
git clone [your-repository-url]
cd recipe-app-backend
```

2. Create and activate virtual environment
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create .env file
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Run the application
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

Once the application is running, you can access:
- Interactive API documentation (Swagger UI): `http://localhost:8000/docs`
- Alternative API documentation (ReDoc): `http://localhost:8000/redoc`

## 🧪 Testing

To run the tests:
```bash
pytest
```

## 🔐 Environment Variables

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `MODEL_PATH`: Path to local AI models
- Other configuration variables (see .env.example)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📫 Contact

Your Name - [@your_twitter](https://twitter.com/your_twitter) - email@example.com

Project Link: [https://github.com/yourusername/recipe-app-backend](https://github.com/yourusername/recipe-app-backend) 