import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chat_models import ChatOllama
from langchain.chains import RetrievalQA

# Configuración de la página
st.set_page_config(page_title="Asistente Técnico para Motocicletas")

st.title("Asistente Técnico para Motocicletas")
st.write("Carga un manual en PDF y realiza consultas técnicas.\nEsto no será como una conversación, solo puedes hacer una pregunta a la vez. Sé específico y consulta lo que desees.")
st.write("Nota: Luego de cargar el documento se mostrará el cuadro de consultas, puede tomar unos momentos.")
         
# Carga del archivo PDF
uploaded_file = st.file_uploader("Selecciona un archivo PDF", type="pdf")

if uploaded_file is not None:
    # Guardar el archivo temporalmente
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Cargar y procesar el documento
    loader = PyMuPDFLoader("temp.pdf")
    documents = loader.load()

    # Generar embeddings y crear el vectorstore
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents, embedding=embeddings)

    # Configurar el modelo de lenguaje
    llm = ChatOllama(model="llama3")


    # Crear la cadena de preguntas y respuestas
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # Entrada del usuario
    query = st.text_input("Ingresa tu consulta técnica:")

    if query:
        # Obtener la respuesta
        response = qa_chain.run(query)
        st.write("Respuesta:")
        st.write(response)
