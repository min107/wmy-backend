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

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "서버가 정상 작동중입니다"})

@app.route('/api/generate-image', methods=['POST'])
def generate_image():
    try:
        data = request.get_json()
        payload = data.get('payload')
        
        if not payload:
            return jsonify({"error": "payload가 필요합니다"}), 400
        
        # payload에서 contents 추출
        contents = payload.get('contents', [])
        if not contents:
            return jsonify({"error": "contents가 비어있습니다"}), 400
        
        # Gemini Vision 모델 사용
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # contents를 Gemini API 형식으로 변환
        gemini_contents = []
        for content in contents:
            parts = content.get('parts', [])
            for part in parts:
                if 'text' in part:
                    gemini_contents.append(part['text'])
                elif 'inlineData' in part:
                    # 이미지 데이터 처리
                    inline_data = part['inlineData']
                    import base64
                    image_data = base64.b64decode(inline_data['data'])
                    gemini_contents.append({
                        'mime_type': inline_data['mimeType'],
                        'data': image_data
                    })
        
        # Gemini API 호출
        response = model.generate_content(gemini_contents)
        
        return jsonify({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": response.text
                    }],
                    "role": "model"
                }
            }]
        })
        
    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "details": "이미지 생성 중 오류가 발생했습니다"
            }
        }), 500

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