import streamlit as st
import PyPDF2
from text_processsing import generate_summary, generate_questions, read_txt_file, read_pdf_file, scrape_wikipedia
import io
def main():
    st.title("Text Summarization and Question Generation")
    st.sidebar.title("Input Options")
    text_input = None
    input_method = st.sidebar.radio("Select Input Method", ("Type your Text", "Upload .txt file", "Upload .pdf file", "From Wikipedia URL"))
    
    if input_method == "Type your Text":
        text_input = st.text_area("Enter your text here:")
    elif input_method == "Upload .txt file":
        txt_file = st.file_uploader("Upload a .txt file", type=["txt"])
        if txt_file is not None:
            text_input = txt_file.read().decode("utf-8")  # Read the uploaded file contents as string
    elif input_method == "Upload .pdf file":
        pdf_file = st.file_uploader("Upload a .pdf file", type=["pdf"])
        if pdf_file is not None:
            a=PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
            text_input=""
            for i in range(len(a.pages)):
                page=a.pages[i]
                text_input+=page.extract_text()
             # Process PDF file and extract text
    elif input_method == "From Wikipedia URL":
        wiki_url = st.text_input("Enter Wikipedia URL:")
        if st.button("Generate Summary"):
            text_input= scrape_wikipedia(wiki_url)
            summary = generate_summary(text_input, 10)
           #num_sentences = st.sidebar.slider("Number of sentences in  summary:", min_value=1, max_value=10, value=3)  

           # summary = generate_summary(text_input, 10)
            st.write(summary)
            text_input = ""
            
        else:
             text_input = ""

    if text_input:
        num_sentences = st.sidebar.slider("Number of sentences in summary:", min_value=1, max_value=10, value=3)
         
         
        if st.button("Generate Summary"):
            summary = generate_summary(text_input, num_sentences)
             
             
            st.subheader("Summary:")
            st.write(summary)
        
        if st.button("Generate Questions"):
            questions = generate_questions(text_input)
            st.subheader("Generated Questions:")
            for i, question in enumerate(questions, 1):
                st.write(f"{i}. {question}")
    
         

if __name__ == "__main__":
    main()

