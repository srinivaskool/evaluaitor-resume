import os
from flask import Flask, request
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

def pdf_to_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=200, length_function=len
    )

    chunks = text_splitter.split_text(text=text)
    return chunks


def resume_summary(query_with_chunks):
    query = f''' need to detailed summarization of below resume and finally conclude them

                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                {query_with_chunks}
                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                '''
    return query


def resume_strength(query_with_chunks):
    query = f'''need to detailed analysis and explain of the strength of below resume and finally conclude them
                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                {query_with_chunks}
                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                '''
    return query


def resume_weakness(query_with_chunks):
    query = f'''need to detailed analysis and explain of the weakness of below resume details and how to improve make a better resume

                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                {query_with_chunks}
                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                '''
    return query


def job_title_suggestion(query_with_chunks):
    query = f''' what are the job roles i apply to likedin based on below?
                  
                  """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                  {query_with_chunks}
                  """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                '''
    return query


def job_description_matching(query_with_chunks, job_description):
    query = f''' I want you to evaluate the job description and my resume, u can verify the qualifications, technologies or languages required for the role and what i know mentioned in resume. 
                 First tell me if they are aligning or not.
                 Give a clear answers if it aligning or not. Then give me suggestions how to improve my resume to match the job description of
                  """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                  {job_description}
                  And my resume is 
                  """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                  {query_with_chunks}
                  """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                '''
    return query


def openai(query):
    openai_api_key = os.environ['API_KEY']
    print(openai_api_key)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstores = FAISS.from_texts(query, embedding=embeddings)

    docs = vectorstores.similarity_search(query=query, k=3)

    llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query)
    return response


@app.route("/evaluate", methods=["POST"])
def process():
    print("started post request")
    pdf = request.json.get("resume", "")
    job_description = request.json.get("job_description", "")
    pdf_chunks = pdf_to_chunks(pdf)

    summary = resume_summary(query_with_chunks=pdf_chunks)
    result_summary = openai(query=summary)

    strength = resume_strength(query_with_chunks=result_summary)
    result_strength = openai(query=strength)

    weakness = resume_weakness(query_with_chunks=result_summary)
    result_weakness = openai(query=weakness)

    job_suggestion = job_title_suggestion(query_with_chunks=result_summary)
    result_suggestion = openai(query=job_suggestion)

    job_matching = job_description_matching(query_with_chunks=result_summary, job_description=job_description)
    result_matching = openai(query=job_matching)

    return {
        "summary": result_summary,
        "strength": result_strength,
        "weakness": result_weakness,
        "suggestion": result_suggestion,
        "matching": result_matching,
    }

if __name__ == "__main__":
    app.run(debug=True, port=5001)
