from openai import OpenAI

client = OpenAI()

class ResponseService():
    def __init__(self):
        pass
    
    def generate_response(self, facts, user_question):
        try:
            # 사용자의 질문과 사실을 기반으로 답장
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                {"role": "user", "content": 
                    f'Please answer the following question in the same language as the question itself.' +
                    ' If the question is in Korean, answer in Korean. If the question is in English, answer in English.' +
                    f' QUESTION: {user_question}. FACTS: {facts}'}
            ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"An error occurred in generate_response: {e}")
            return ""
