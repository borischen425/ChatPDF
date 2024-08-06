import streamlit as st
import google.generativeai as genai
import os
from pypdf import PdfMerger
os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key = os.environ['GOOGLE_API_KEY'])

# Model Configuration
MODEL_CONFIG = {
  "temperature": 0.2,
  "top_p": 1,
  "top_k": 32,
  "max_output_tokens": 4096,
}

## Safety Settings of Model
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  }
]

model = genai.GenerativeModel(model_name = "gemini-1.5-flash",
                              generation_config = MODEL_CONFIG,
                              safety_settings = safety_settings)

from pathlib import Path

def image_format(image_path):
    img = Path(image_path)

    if not img.exists():
        raise FileNotFoundError(f"Could not find image: {img}")

    image_parts = [
        {
            "mime_type": "application/pdf", ## Mime type are PNG - image/png. JPEG - image/jpeg. WEBP - image/webp
            "data": img.read_bytes()
        }
    ]
    return image_parts

def gemini_output(image_path, system_prompt, user_prompt):

    image_info = image_format(image_path)
    input_prompt= [system_prompt, image_info[0], user_prompt]
    response = model.generate_content(input_prompt)
    return response.text

st.header("ChatPDF")
pdfs = []
uploaded_file = st.sidebar.file_uploader("Upload your PDF File", type="pdf", accept_multiple_files=True)

for i in range(len(uploaded_file)):
    pdfs.append(uploaded_file[i])
if uploaded_file:
    merger = PdfMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write("./temp.pdf")
    merger.close()

image_path = "./temp.pdf"

# Init chat message history
if "messages" not in st.session_state.keys():
    st.session_state.messages = []

# Re-draw history / all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

system_prompt = """
               You are a specialist in comprehending pdf file.
               Input file in the form of pdf will be provided to you,
               and your task is to respond to questions based on the content of the input file.
               Answering factual questions using your knowledge base  and supplementing information with web search results.
               Domain-Specific Formatting: Provide output guidelines (code blocks, text, tables, etc.) suited to the LLM’s designated domain(s).
               Answer in the user's content language.
               """



#user_prompt = "作者是誰?"
#gemini_output(image_path, system_prompt, user_prompt)


if user_prompt := st.chat_input("message"):

        
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    #query_engine = st.session_state.query_engine
    with st.chat_message("assistant"):
        response = gemini_output(image_path, system_prompt, user_prompt)
        # Somehow streaming conflicts with instrumentation
        # st.write_stream(streaming_response.response_gen)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
