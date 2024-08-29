from dotenv import load_dotenv
load_dotenv()
from intentservice import IntentService
from responseservice import ResponseService
from dataservice import DataService
import gradio as gr

data_service = DataService()
intent_service = IntentService()
response_service = ResponseService()

# 마지막으로 로드된 PDF 파일의 경로를 저장할 변수
last_loaded_pdf = None

def initialize(pdf):
    global last_loaded_pdf
    if last_loaded_pdf != pdf.name:  # PDF 파일이 변경된 경우에만 초기화
        last_loaded_pdf = pdf.name
        # redis에 존재하던 embedding 데이터 삭제
        data_service.drop_redis_data()
        # pdf의 텍스트 -> 임베딩
        data = data_service.pdf_to_embeddings(pdf.name)
        # 임베딩을 redis에 load
        data_service.load_data_to_redis(data)

def runChatBot(pdf, question):
    initialize(pdf)
    intents = intent_service.get_intent(question)
    facts = data_service.search_redis(intents)
    answer = response_service.generate_response(facts, question)
    return answer

def main():
    gr.Interface(
        fn=runChatBot,
        inputs=[gr.File(label="강의계획서 PDF파일"), gr.Textbox(label="질문", placeholder="궁금한 내용을 입력하세요.", container=True, scale=7)],
        outputs=gr.Textbox(label="답변", placeholder="답변내용이 출력됩니다.", container=True, scale=7),
        title="강의계획서 ChatBot",
        description="강의계획서에 대해 대답해주는 챗봇",
        theme="Monochrome",
        allow_flagging="never",
    ).launch()

if __name__ == "__main__":
    main()
