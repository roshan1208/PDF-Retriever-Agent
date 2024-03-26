import streamlit as st
from PIL import Image
import fitz  # PyMuPDF
import pandas as pd
import os
from tqdm import tqdm
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings


### Open source embeddings
embeddings = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs = {'device': 'cpu'})
# embeddings = SentenceTransformerEmbeddings(model_name="intfloat/e5-mistral-7b-instruct", model_kwargs = {'device': 'cpu'})

headerTabel_db = Chroma(collection_name = 'Chromadb_only_header_table_v4', persist_directory="Chromadb_only_header_table_v4/", embedding_function = embeddings)
full_text_db = Chroma(collection_name = 'Chromadb_open_src_emb_pdf_v3', persist_directory="Chromadb_open_src_emb_pdf_v3/", embedding_function = embeddings)

headertabel_retriever = headerTabel_db.as_retriever( search_kwargs ={
                                                   "k": 3,
                                               })
full_text_retriever = full_text_db.as_retriever( search_kwargs ={
                                                   "k": 2,
                                               })
def processed_output_function(query):
    headertabel_retrieved_docs = headertabel_retriever.get_relevant_documents(query)
    full_text_retrieved_docs = full_text_retriever.get_relevant_documents(query)

    page_no_list = []
    # full_context = ''
    for doc in headertabel_retrieved_docs:
        # full_context += doc.page_content
        header, start, end = str(doc.metadata['header_pageNumber']).split('_')
        if int(float(start)) == int(float(end)):
            page_no_list.append([header, int(float(start)), doc.page_content])
        else:
            page_no_list.append([header, int(float(start)), doc.page_content])
            page_no_list.append([header, int(float(end)), doc.page_content])
    
    for doc in full_text_retrieved_docs:
        # full_context += doc.page_content
        header, start, end = str(doc.metadata['header_pageNumber']).split('_')
        if int(float(start)) == int(float(end)):
            page_no_list.append([header, int(float(start)), doc.page_content])
        else:
            page_no_list.append([header, int(float(start)), doc.page_content])
            page_no_list.append([header, int(float(end)), doc.page_content])

    return page_no_list
# print(full_context)


# Function to display the specified page
def display_pdf_page(pdf_path, page_number):
    try:
        # Open the PDF file
        pdf_document = fitz.open(pdf_path)

        # Get the specified page
        page = pdf_document[page_number]

        # Render the page as an image
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Display the image using Streamlit
        st.image(img, use_column_width=True)

        # Close the PDF document
        pdf_document.close()

    except Exception as e:
        st.error(f"Error: {e}")

# Streamlit UI
def main():
    header_html = """
        <style>
            .header {
                background-color: #3498db;
                padding: 1rem;
                color: white;
                font-size: 2rem;
                font-weight: bold;
                text-align: center;
                margin-bottom: 1rem;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
        </style>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    st.title("Medical Writing Assistant")


    # User input for the question
    user_question = st.text_input('Ask question here: ')

    # Process the user input and get page numbers
    if st.button("Retrieve excerpts"):
        page_numbers_list = processed_output_function(user_question)

        # Display the specified page if page numbers are found
        st.subheader("Your Question:")
        st.write(user_question)
        for idx, page_numbers in enumerate(page_numbers_list):
            st.subheader(f"Option: {idx+1}")
            st.success(f"Header: {page_numbers[0]}, Page number: {page_numbers[1]}")
            display_pdf_page('nejmoa2207940_appendix.pdf', page_numbers[1])
            # st.success("Response: {}".format(page_numbers[2]))

if __name__ == "__main__":
    main()