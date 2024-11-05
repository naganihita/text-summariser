import sys
import math
import bs4 as bs
from bs4 import BeautifulSoup
 
import urllib.request
import re
import PyPDF2
import nltk
from nltk.stem import WordNetLemmatizer
import spacy
import heapq
from transformers import pipeline
# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()

# Initialize spaCy for English
nlp = spacy.load('en_core_web_sm')

# Function to read .txt file and return its text 
def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
       text = file.read().replace('\n', '')
    return text
import PyPDF2


# Function to read PDF file and return its text
def read_pdf_file(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        text = ''
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()
        return text


# Function to scrape text from a Wikipedia page
def scrape_wikipedia(url):
    try:
        page = urllib.request.urlopen(url)
        soup = BeautifulSoup(page, 'lxml')
        paragraphs = soup.find_all('p')
        article_text = ''
        for paragraph in paragraphs:
            article_text += paragraph.text
        return article_text
    except Exception as e:
        print(f"Error scraping Wikipedia: {e}")
        return None

# Function to preprocess text (remove special characters, tokenize, etc.)
def preprocess_text(text):
    text = re.sub(r'\[[0-9]*\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Function to generate questions from text using spaCy
def generate_questions(text):
    # Process text using spaCy
    doc = nlp(text)
    
    # Initialize lists to store questions
    questions = []
    
    # Extract sentences for analysis
    sentences = list(doc.sents)
    
    for sentence in sentences:
        # Skip very short sentences
        if len(sentence) < 10:
            continue
        
        # Tokenize sentence
        tokens = nlp(sentence.text)
        
        # Identify subject, object, and key elements in the sentence
        subject, object_, root = None, None, None
        for token in tokens:
            if token.dep_ == 'nsubj':
                subject = token.text
            elif token.dep_ == 'dobj':
                object_ = token.text
            elif token.dep_ == 'ROOT':
                root = token.text
        
        # Generate basic question templates
        if subject and root:
            questions.append(f"What is {subject} {root}?")
            questions.append(f"What does {subject} {root}?")
        if object_ and root:
            questions.append(f"What {root} {object_}?")
            questions.append(f"What does {object_} {root}?")
        
        # Add more sophisticated question structures based on context
        
        # Example: Identify named entities and generate context-based questions
        for ent in tokens.ents:
            if ent.label_ == 'PERSON':
                questions.append(f"Who is {ent.text}?")
            elif ent.label_ == 'DATE':
                questions.append(f"When did {ent.text} occur?")
            # Add more entity types as needed
        
    return questions

# Function to calculate term frequency (TF)
def calculate_tf(sentence):
    word_frequency = {}
    for word in tokenize_words(sentence):
        if word not in word_frequency.keys():
            word_frequency[word] = 1
        else:
            word_frequency[word] += 1

    # Normalize the term frequencies
    max_frequency = max(word_frequency.values())
    for word in word_frequency.keys():
        word_frequency[word] = word_frequency[word] / max_frequency

    return word_frequency

# Function to calculate inverse document frequency (IDF)
def calculate_idf(sentences):
    idf_values = {}
    words = [tokenize_words(sentence) for sentence in sentences]

    for sentence in words:
        for word in sentence:
            if word not in idf_values:
                idf_values[word] = 0
            idf_values[word] += 1

    for word, value in idf_values.items():
        idf_values[word] = math.log10(len(sentences) / float(value))

    return idf_values

# Function to calculate TF-IDF scores
def calculate_tfidf(sentences):
    tfidf_scores = {}
    idf_values = calculate_idf(sentences)

    for sentence in sentences:
        tf_scores = calculate_tf(sentence)
        for word, tf in tf_scores.items():
            if word not in tfidf_scores:
                tfidf_scores[word] = {}
            tfidf_scores[word][sentence] = tf * idf_values[word]

    return tfidf_scores

# Function to tokenize sentences
def tokenize_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

# Function to tokenize words and lemmatize
def tokenize_words(sentence):
    words = nltk.word_tokenize(sentence)
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# Function to generate summary based on TF-IDF scores
def generate_summary(text, num_sentences):
    preprocessed_text = preprocess_text(text)
    sentences = tokenize_sentences(preprocessed_text)
    tfidf_scores = calculate_tfidf(sentences)

    sentence_scores = {}
    for sentence in sentences:
        score = 0
        words = tokenize_words(sentence)
        for word in words:
            if word in tfidf_scores and sentence in tfidf_scores[word]:  # Check if sentence exists in tfidf_scores[word]
                score += tfidf_scores[word][sentence]
        sentence_scores[sentence] = score

    # Use the specified number of summarized sentences
    summarized_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summarized_sentences)

    return summary
 

def answer_questions(text, questions):
    # Load the pre-trained question-answering model
    nlp = pipeline("question-answering")

    answers = []
    for question in questions:
        result = nlp(question=question, context=text)
        answers.append(result['answer'])
    
    return answers
# Example usage:
if __name__ == "__main__":
    example_text = "Replace this with your text or use the functions to read from files or URLs."
    summary = generate_summary(example_text, 3)  # Default to 3 sentences for example
    print("Summary:")
    print(summary)
