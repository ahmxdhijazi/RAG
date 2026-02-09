import chromadb
from chromadb.utils import embedding_functions
import json
from dotenv import load_dotenv
import os
from openai import OpenAI
import argparse

load_dotenv()
client = OpenAI() #default client for batch
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

    REUSED FROM PREVIOUS HOMEWORK

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

def retrieve_context(collection, question):
    """
    Queries ChromaDB for the 5 most relevant chunks to the question.
    """
    #Query the collection
    results = collection.query(
        query_texts=[question],
        n_results=5
    )
    #Chroma returns a dict: Need the first list inside 'documents'
    documents = results['documents'][0]
    
    #Join them with newlines
    return "\n\n".join(documents)

def create_batch_file(batch_data, jsonl_filename, collection):
    '''
    PREPARE BATCH REQUEST FILE WITH RAG
    '''
    print(f"Creating batch file '{jsonl_filename}'")

    with open(jsonl_filename, 'w', encoding='utf-8') as file:
        for i, entry in enumerate(batch_data):
            #RETRIEVE CONTEXT: We use the collection to find relevant chunks for this specific question
            context_str = retrieve_context(collection, entry['question'])
            
            #COMBINE (RAG PROMPT): We explicitly label the Context and the Question for the model
            rag_content = f"Context:\n{context_str}\n\nQuestion:\n{entry['question']}"

            #Construct the API request
            request_object = {
                "custom_id": entry['id'],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5-nano",
                    "reasoning_effort": "minimal",
                    "messages": [
                        #Prompting model to use the context effectively, in hopes of improving answer quality...
                        {"role": "system", "content": "You are a helpful assistant. Answer the question using the provided context. Answer using only a short phrase, date, or entity."},
                        {"role": "user", "content": rag_content} 
                    ],
                    "max_completion_tokens": 750 #increased token limit 
                }
            }
            file.write(json.dumps(request_object) + '\n')

def submit_batch_job(jsonl_filename, description="squad-rag-homework"):
    """
    UPLOAD BATCH FILE AND CREATE BATCH JOB
    """
    print("Uploading file to OpenAI.")
    batch_file = client.files.create(
      file=open(jsonl_filename, "rb"),
      purpose="batch"
    )

    print(f"Creating Batch Job with File ID: {batch_file.id}...")
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description}
    )

    print(f"Batch id {batch_job.id} \nSubmitted!\n")
    return batch_job


def run_serial_openrouter(questions, collection):
    """
    Run questions serially using OpenRouter with RAG context.
    """
    print(f"Starting serial run for {len(questions)} items.")
    
    # Specific client for OpenRouter
    or_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

    results = []

    for i, entry in enumerate(questions):

        #Update user on progress per question... takes a while to run serially, so updates were cool
        print(f"Processing {i+1}/{len(questions)}: ID {entry['id']}")
        
        #RETRIEVE CONTEXT: We use the collection to find relevant chunks for this specific question
        context_str = retrieve_context(collection, entry['question'])
        
        #COMBINE (RAG PROMPT): We explicitly label the Context and the Question for the model
        rag_content = f"Context:\n{context_str}\n\nQuestion:\n{entry['question']}"
        
        answer_text = ""
        try:
            completion = or_client.chat.completions.create(
                model="qwen/qwen3-8b",
                messages=[
                    # Prompting model to use the context effectively...
                    {"role": "system", "content": "You are a helpful assistant. Answer the question using the provided context. Answer using only a short phrase, date, or entity."},
                    {"role": "user", "content": rag_content} 
                ]
            )
            answer_text = completion.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            
        results.append({
            "id": entry['id'],
            "model_answer": answer_text
        })

    # Save Results
    output_filename = "qwen-rag-answers.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Saved serial results to {output_filename}")


def main():
    #CLI Arguments Setup
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument('--mode', type=str, choices=['batch', 'serial'], required=True)
    args = parser.parse_args()

    print("Loading JSON data from file (:")
    try: #attempt to retrieve the context chunks, if the file is not found, catch the error and update the user
        with open('dev-v2.0.json', 'r', encoding='utf-8') as file: data = json.load(file)
    except FileNotFoundError:
        print("Warning: 'dev-v2.0.json' not found. Ensure it is in the directory.")
        return # Exit the main function if the file is not found

    
    context_chunks = get_context_chunks(data) #Call the function and update the user
    print(f"Successfully loaded {len(context_chunks)} context chunks!") # All context chunks are stored

    #Proceed to create the ChromaDB collection2
    print(f"Building Database with {len(context_chunks)} chunks.")
    collection = create_chromadb_collection(context_chunks)
    print("Database Ready!")
    
    # Extract 500 possible questions
    questions = get_possible_qas(data)
    print(f"Loaded {len(questions)} questions for testing.") 

    #Logic for RAG BATCH PIPELINE
    if args.mode == 'batch':
        #Pass collection
        create_batch_file(questions, 'batch_requests.jsonl', collection)
        #call submit batch job function
        submit_batch_job('batch_requests.jsonl')
    #Logic for RAG SERIAL PIPELINE
    elif args.mode == 'serial':
        run_serial_openrouter(questions, collection)
        #run_serial_openrouter(questions[:10], collection) #For testing

if __name__ == "__main__":
    main()