"""
Carga la base vectorial ya creada
y ejecuta el chatbot RAG del HotelMar usando Gradio
"""

import gradio as gr
from langchain_ollama import (OllamaLLM,OllamaEmbeddings)
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


# CONFIGURACIÓN
DB_NAME = "hotelmar_db"
EMBED_MODEL = "nomic-embed-text"

# EMBEDDINGS
embeddings = OllamaEmbeddings(
    model=EMBED_MODEL
)

# CARGAR BASE VECTORIAL
vectorstore = Chroma(
    persist_directory=DB_NAME,
    embedding_function=embeddings,
    collection_name="hotel_knowledge"
)

# MEMORIA DEL CHAT
memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True,
    output_key='answer'
)

# PROMPT 
template = """
Eres el asistente virtual del HotelMar.
Responde de forma amable, profesional y breve.
Usa únicamente la información proporcionada
en el contexto.
Si no sabes algo, di:
"No dispongo de esa información actualmente."

Nunca inventes:
- precios
- reservas
- disponibilidad
- datos privados

Contexto:
{context}

Pregunta:
{question}

Respuesta:
"""

QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# FUNCIÓN DEL CHAT
def chat_action(message, _history, model_name, k):

    try:

        # MODELO
        llm = OllamaLLM(
            model=model_name,
            temperature=0.2
        )

        # RETRIEVER
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

        # CADENA RAG
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={
                "prompt": QA_PROMPT
            }
        )

        # RESPUESTA
        result = conversation_chain.invoke({
            "question": message
        })

        return result["answer"]

    except Exception as e:

        return f"Error: {str(e)}"


# INTERFAZ
with gr.Blocks(title="HotelMar AI") as interface:

    gr.Markdown("""
    # 🌊 HotelMar AI
    Asistente virtual inteligente para atención al cliente.
    """)

    model_sel = gr.Dropdown(
        choices=["llama3.2", "mistral"],
        value="llama3.2",
        label="Modelo"
    )

    k_slider = gr.Slider(
        minimum=1,
        maximum=10,
        value=3,
        step=1,
        label="Número de fragmentos"
    )

    gr.ChatInterface(
        fn=chat_action,
        additional_inputs=[
            model_sel,
            k_slider
        ]
    )

# EJECUTAR
if __name__ == "__main__":
    interface.launch()