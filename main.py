import os

from langchain import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
import pinecone

pinecone.init(api_key="5df75cbb-ec2a-4732-8de8-767bb2360e9b",environment="gcp-starter")

if __name__=="__main__":
    print("Hello LangChain!")
    file_path = "C:/Users/GUNNI ASSOCIATES/Desktop/intro-to-vector-db/mediumblogs/mediumblog1.txt"
    loader = TextLoader(file_path,encoding = 'UTF-8')
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    # print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Pinecone.from_documents(texts,embeddings,index_name="medium-blogs-embedding-index")

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever()
    )

    query = "What is Vector Database? explain me this in beginner friendly language."

    result = qa({"query":query})

    print(result)