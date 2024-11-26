import time
from flask import Flask, request, jsonify 
import os
import re
import pypdf
import pandas as pd
import chromadb
import google.generativeai as palm
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import List
import speech_recognition as sr
import json
import google.generativeai as genai
from pypdf import PdfReader
from dotenv import load_dotenv
import os
import shutil

load_dotenv()
if os.path.isfile("RAG/chroma.sqlite3"):
    shutil.rmtree("RAG")
def load_pdf(file_path):
    """
    Reads the text content from a PDF file and returns it as a single string.

    Parameters:
    - file_path (str): The file path to the PDF file.

    Returns:
    - str: The concatenated text content of all pages in the PDF.
    """
    # Logic to read pdf
    reader = PdfReader(file_path)

    # Loop over each page and store it in a variable
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return text

# replace the path with your file path
pdf_text = load_pdf(file_path="mati.pdf")

def split_text(text: str):
    """
    Splits a text string into a list of non-empty substrings based on the specified pattern.
    The "\n \n" pattern will split the document para by para
    Parameters:
    - text (str): The input text to be split.

    Returns:
    - List[str]: A list containing non-empty substrings obtained by splitting the input text.

    """
    split_text = re.split('\n \n', text)
    return [i for i in split_text if i != ""]

chunked_text = split_text(text=pdf_text)

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function using the Gemini AI API for document retrieval.

    This class extends the EmbeddingFunction class and implements the __call__ method
    to generate embeddings for a given set of documents using the Gemini AI API.

    Parameters:
    - input (Documents): A collection of documents to be embedded.

    Returns:
    - Embeddings: Embeddings generated for the input documents.
    """
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model,
                                content=input,
                                task_type="retrieval_document",
                                title=title)["embedding"]

def create_chroma_db(documents:List, path:str, name:str):
    """
    Creates a Chroma database using the provided documents, path, and collection name.

    Parameters:
    - documents: An iterable of documents to be added to the Chroma database.
    - path (str): The path where the Chroma database will be stored.
    - name (str): The name of the collection within the Chroma database.

    Returns:
    - Tuple[chromadb.Collection, str]: A tuple containing the created Chroma Collection and its name.
    """
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())

    for i, d in enumerate(documents):
        db.add(documents=d, ids=str(i))

    return db, name

db,name =create_chroma_db(documents=chunked_text, 
                        path="RAG", #replace with your path
                        name="rag_experiment")

def load_chroma_collection(path, name):
    """
    Loads an existing Chroma collection from the specified path with the given name.

    Parameters:
    - path (str): The path where the Chroma database is stored.
    - name (str): The name of the collection within the Chroma database.

    Returns:
    - chromadb.Collection: The loaded Chroma Collection.
    """
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())

    return db

db=load_chroma_collection(path="RAG", name="rag_experiment")

def get_relevant_passage(query, db, n_results):
    passage = db.query(query_texts=[query], n_results=n_results)['documents'][0]
    return passage

#Example usage
relevant_text = get_relevant_passage(query="Intellykeys",db=db,n_results=3)

# def escuchar_mic():
#     # recognizer = sr.Recognizer()
#     # with sr.Microphone() as source:
#     #     print("Escuchando...")
#     #     try:
#     #         audio = recognizer.listen(source, phrase_time_limit=5)
#     #         texto = recognizer.recognize_google(audio, language="es-ES")
#     #         print(f"Texto reconocido: {texto}")
#     #         return texto
#     #     except sr.UnknownValueError:
#     #         print("No se pudo reconocer el audio.")
#     #         return None
#     #     except sr.RequestError as e:
#     #         print(f"Error en la solicitud de reconocimiento: {e}")
#     #         return None
#     texto = "Hola"
#     return texto

# MICROPHONE CONTEXT: '{escaped_texto_mic}'
# def make_rag_prompt(query, texto_mic, relevant_passage):
def make_rag_prompt(query, relevant_passage):
    escaped_passage = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    # escaped_texto_mic = texto_mic.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = ("""You are a helpful SPANISH and knowledgeable assistant that helps complete text using the reference passage.
If the passage is 'hola', your responses must be: \
1- Hola \
2- Hola, como te va? \
3- Hola, como estas? \

Ensure your response is fluent, coherent, and relevant to the user's text, adding value and depth where appropriate.
Focus on clarity, providing thoughtful and well-structured continuations or expansions of the given text in a friendly, conversational tone.
Generate ALWAYS three distinct options of the completed text, each not exceeding 10 words. Ensure that each option varies in structure and approach.
If the passage does not provide relevant information, complete the user's text based on your understanding while maintaining relevance.
Your responses should follow this format:

1- user_text + your response1
2- user_text + your response2
3- user_text + your response3
USER TEXT: '{query}'
PASSAGE: '{escaped_passage}'

COMPLETION:
    """).format(query=query, escaped_passage=escaped_passage)
                # escaped_texto_mic=escaped_texto_mic
                

    return prompt

def generate_answer_by_prompt(prompt):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text

def generate_response(text_user):
    def generate_answer(db,query):
        #retrieve top 3 relevant text chunks
        # texto_mic = escuchar_mic()
        # print(texto_mic)
        relevant_text = get_relevant_passage(query,db,n_results=3)
        # print(relevant_text)
        prompt = make_rag_prompt(query, relevant_passage="".join(relevant_text)) # joining the relevant chunks to create a single passage
                                # texto_mic,
        # print(prompt)
        answer = generate_answer_by_prompt(prompt)

        return answer

    db=load_chroma_collection(path="RAG", #replace with path of your persistent directory
                            name="rag_experiment") #replace with the collection name

    answer = generate_answer(db,text_user)

    # rag_path = "RAG"  # Path de la carpeta RAG
    # if os.path.exists(rag_path):
    #     shutil.rmtree(rag_path)  # Eliminar la carpeta y su contenido

    return answer

# print(generate_response(input("Enter text: ")))