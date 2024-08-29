from openai import OpenAI

client = OpenAI()

class IntentService():
    def __init__(self):
        pass
    
    def get_intent(self, user_question: str):
        try:
            # 사용자의 질문으로부터 keyword 추출
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": f'Extract the main keywords from the following question: "{user_question}". ' +
                                     'If the question is in Korean, extract keywords in Korean. Do not answer anything else, only the keywords. Remove honorifics like "님". ' +
                                     'If the question is in English, change question to Korean and extract keywords in Korean. Do not answer anything else, only the keywords. Remove honorifics like "님".'}
                ]
            )
            # print("OpenAI API response:", response)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"An error occurred in get_intent: {e}")
            return ""
