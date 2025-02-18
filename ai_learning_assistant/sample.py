from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from fpdf import FPDF
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
# Watson Machine Learning credentials
credentials = {
    'url': "<<url here>>",
    'apikey': "<key here>"
}
project_id = "<project here>"
# Parameters for language models
params = {
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
    GenParams.TEMPERATURE: 0.5,
    GenParams.TOP_K: 5,
    GenParams.TOP_P: 1
}
# Prompt templates
pt1 = PromptTemplate(
    input_variables=["topic"],
    template="Enter your question (or type 'quit' to exit) {topic}: Question: ")

pt2 = PromptTemplate(
    input_variables=["question"],
    template="Enter your question (or type 'quit' to exit): {question}")
# Initialize IBM Watson Language Models
flan_ul2_model = Model(
    model_id='google/flan-ul2',
    credentials=credentials,
    params=params,
    project_id=project_id)
flan_ul2_llm = WatsonxLLM(model=flan_ul2_model)
flan_t5_model = Model(
    model_id="google/flan-t5-xxl",
    credentials=credentials,
    project_id=project_id)
flan_t5_llm = WatsonxLLM(model=flan_t5_model)
granite_model = Model(
    model_id="ibm/granite-13b-instruct-v2",
    credentials=credentials,
    project_id=project_id)
granite_llm = WatsonxLLM(model=granite_model)

# Create language model chains
prompt_to_flan_ul2 = LLMChain(llm=flan_ul2_llm, prompt=pt1)
flan_ul2_to_flan_t5 = LLMChain(llm=flan_t5_llm, prompt=pt2)

prompt_to_granite = LLMChain(llm=granite_llm, prompt=pt1)
# Combine chains
qa = SimpleSequentialChain(chains=[prompt_to_granite,prompt_to_flan_ul2, flan_ul2_to_flan_t5], verbose=True)
# embedding model
embed_params = {
     EmbedParams.TRUNCATE_INPUT_TOKENS: 3,
     EmbedParams.RETURN_OPTIONS: {
     'input_text': True
     }
 }
embedding = Embeddings(
     model_id=EmbeddingTypes.IBM_SLATE_30M_ENG,
     params=embed_params,
     credentials=Credentials(
         api_key = credentials["apikey"],
         url = credentials["url"]),
     project_id=project_id
     )
# Load PDF using PyPDFLoader
pdf_folder_path = '/root/ai_learning_assistant/'
loader = PyPDFLoader("/root/ai_learning_assistant/abcd.pdf")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
data = loader.load()
texts = text_splitter.split_documents(data)
# Create in-memory embedding database from the doc using Chroma
docsearch = Chroma.from_documents(texts, embedding,collection_metadata={"source": str(i) for i in range(len(texts))})
# Logic for multiple questions
def ask_question_dynamically():
    query = input("Enter your question (or type 'quit' to exit): ").strip()
    return query
# Initialize PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
# Main loop
while True:
    query = ask_question_dynamically()
    if query.lower() == 'quit':
        print("Exiting... Thank You!!!")
        break
    print("Query:", query)
    # Perform search based on the question
    docs = docsearch.similarity_search(query,k=1)
    # print(docs)
    # Use LLM to get answering
    context = docs[0].page_content if docs else "No relevant documents found."
    answer = qa.batch(inputs=[f"Context: {context}\nQuestion: {query}"])
    # answer = qa.run(input=query)
    # Display immediate answer
    print("Answer:", answer)
    # Add question and immediate answer to PDF
    # pdf.cell(200, 10, txt=f"Question: {query}\nAnswer: {answer}\n", ln=True, align='L')
# Save PDF
pdf.output("qa_results.pdf")


