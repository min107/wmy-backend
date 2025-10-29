from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import google.generativeai as genai

# 환경변수 로드
load_dotenv()

app = Flask(__name__)
CORS(app)

# 제미나이 설정
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

@app.route('/')
def home():
    return jsonify({"message": "제미나이 서버 실행중!"})

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message')
        
        if not message:
            return jsonify({"error": "메시지를 입력해주세요"}), 400
        
        # gemini-1.5-flash-latest 시도 (2024년 최신 무료 모델)
        model = genai.GenerativeModel('models/gemini-2.5-pro')
        response = model.generate_content(message)
        
        return jsonify({
            "success": True,
            "response": response.text
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)