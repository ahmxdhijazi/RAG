import chromadb
from chromadb.utils import embedding_functions
import json
from dotenv import load_dotenv
import os
load_dotenv()

"""
1. Learn how to leverage Retrieval Augmented Generation (RAG) with OpenAI embeddings.
2. Learn how to leverage ChomaDB to leverage external information for better answers.
"""

def get_context_chunks(data):
    """
    Extracts all context chunks from each question (where it exists)
    """
    results = []
    # Iterate through the main 'data' list (Topics)
    for title in data['data']:
        # Iterate through 'paragraphs' in that topic
        for paragraph in title['paragraphs']:
            results.append(paragraph['context'])
    return results

def create_chromadb_collection(documents):
    """
    Creates a ChromaDB collection and adds the context chunks to it.
    """
    # Initialize ChromaDB client
    client = chromadb.Client()
    
    # Set up embedding functions
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )

    # Create a new collection and now set the embedding function for the collection
    collection_name = "context_chunks_collection"
    collection = client.create_collection(name=collection_name, embedding_function=embedding_function)

    # Add documents to the collection
    collection.add(documents=documents, ids=[str(i) for i in range(len(documents))]) # Generate unique IDs for each document
    return collection

def get_possible_qas(data, limit=500):
    """
    Extracts the first 'limit' questions where is_impossible is False.
    """
    results = []
    # Iterate through the main 'data' list (Topics)
    for title in data['data']:
        # Iterate through 'paragraphs' in that topic
        for paragraph in title['paragraphs']:
            # Iterate through the specific questions/answers
            for qa in paragraph['qas']:
                # skip the 'impossible' questions
                if qa['is_impossible'] is False:
                    #Create a clean dictionary entry to ignore unnecessary fields
                    entry = {
                        "id": qa['id'],
                        "question": qa['question'],
                        # Save ALL valid answers as a list of strings
                        "answers": [ans['text'] for ans in qa['answers']]
                    }
                    results.append(entry) # add the entry to results
        
                # check limit of 500 possible questions
                if len(results) >= limit:
                    return results # immediately stop and return results
    return results

def main():
    print("Loading JSON data from file (:")
    try: #attempt to retrieve the context chunks, if the file is not found, catch the error and update the user
        with open('dev-v2.0.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        # Call the function and update the user
        context_chunks = get_context_chunks(data)
        print(f"Successfully loaded {len(context_chunks)} context chunks!") # All context chunks are stored
    except FileNotFoundError:
        print("Warning: 'dev-v2.0.json' not found. Ensure it is in the directory.")
        data = None
        return # Exit the main function if the file is not found


    # If data is successfully loaded, proceed to create the ChromaDB collection2
    print(f"Building Database with {len(context_chunks)} chunks...")
    collection = create_chromadb_collection(context_chunks)
    print("Database Ready!")

    # Extract 500 possible questions
    questions = get_possible_qas(data)
    print(f"Loaded {len(questions)} questions for testing.") #Length should be 500 to confirm

if __name__ == "__main__":
    main()