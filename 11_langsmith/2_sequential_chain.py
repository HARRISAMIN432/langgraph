from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
model2 = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

config = {
    'tags': {'llm app', 'report generation', 'summarization'}, 'metadata': {'model1': 'gemini-2.5-flash', 'model1_temp': 0.7, 'parser': 'stroutputparser'}
}


parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)



print(result)
