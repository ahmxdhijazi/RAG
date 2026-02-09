import chromadb
from chromadb.utils import embedding_functions
import json
from dotenv import load_dotenv
import os
from openai import OpenAI
import argparse
from datetime import datetime

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

def create_grading_batch(squad_data, student_file, output_filename):
    """
    Reads student answers, compares them to SQuAD ground truth, 
    and writes a Batch File for the Judge (GPT-5-mini) using Appendix 1 Prompt.
    """
    print(f"Preparing grading batch: {output_filename}")
    
    #Build Answer Key (Ground Truth)
    ground_truth = {}
    for title in squad_data['data']:
        for paragraph in title['paragraphs']:
            for qa in paragraph['qas']:
                if not qa['is_impossible']:
                    ground_truth[qa['id']] = {
                        "question": qa['question'],
                        "correct_answers": [ans['text'] for ans in qa['answers']]
                    }

    #Load Student Answers (Handle both JSONL and JSON formats)
    student_answers = {}
    
    #Case 1: Deal with Batch Results (JSONL from GPT-5-nano)
    if student_file.endswith('.jsonl'): 
        try:
            with open(student_file, 'r') as f:
                for line in f:
                    resp = json.loads(line)
                    #Parse the deep batch API response structure
                    ans = resp['response']['body']['choices'][0]['message']['content']
                    q_id = resp['custom_id']
                    student_answers[q_id] = ans
        except FileNotFoundError:
            print(f"Skipping {student_file} (File not found)")
            return

    #Case 2: Deal with Serial Results (JSON form Qwen)
    else: 
        try:
            with open(student_file, 'r') as f:
                data = json.load(f)
                for entry in data:
                    student_answers[entry['id']] = entry['model_answer']
        except FileNotFoundError:
            print(f"Skipping {student_file} (File not found)")
            return

    #Write Judge Requests
    with open(output_filename, 'w') as f:
        count = 0
        for q_id, student_ans in student_answers.items():
            if q_id not in ground_truth: continue 

            truth = ground_truth[q_id]
            
            #Prompt from Appendix 1: Explicitly instructing the "judge"
            user_content = f"""You are a teacher tasked with determining whether a student’s answer to a question was correct, based on a set of possible correct answers. You must only use the provided possible correct answers to determine if the student’s response was correct. Question: {truth['question']} Student’s Response: {student_ans} Possible Correct Answers: {truth['correct_answers']} Your response should only be a valid Json as shown below:
{{
"explanation" (str): A short explanation of why the student’s answer was correct or
incorrect.,
"score" (bool): true if the student’s answer was correct, false if it was incorrect
}}
Your response: """

            #JSON Schema (include 'explanation' as requested in Appendix)
            json_schema = {
                "name": "grading_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "explanation": {"type": "string"},
                        "score": {"type": "boolean"}
                    },
                    "required": ["explanation", "score"],
                    "additionalProperties": False
                }
            }

            #Constructing Request
            req = {
                "custom_id": q_id, 
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5-mini", #The Judge
                    "messages": [
                        {"role": "user", "content": user_content}
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": json_schema
                    }
                }
            }
            f.write(json.dumps(req) + '\n')
            count += 1
            
    print(f"Prepared {count} grading requests.")

def check_and_download_batch(batch_id, output_filename):
    """
    Checks if a batch job is complete and downloads the results.
    """
    batch_job = client.batches.retrieve(batch_id)
    print(f"Checking Batch {batch_id}... Status: {batch_job.status}")
    
    #If complete we should download the results, if failed we should print the error, otherwise we just say not ready yet.
    if batch_job.status == 'completed' and batch_job.output_file_id:
        print("Downloading results...")
        content = client.files.content(batch_job.output_file_id).content
        with open(output_filename, 'wb') as f:
            f.write(content)
        print(f"Saved to {output_filename}")
        return True
    elif batch_job.status == 'failed':
        print(f"Batch Failed: {batch_job.errors}")
        return False
    else:
        print("Job not ready yet.")
        return False
    

def main():
    #CLI Arguments Setup
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument('--mode', type=str, choices=['batch', 'serial', 'grade', 'download'], required=True)
    parser.add_argument('--batch_id', type=str)
    parser.add_argument('--model_name', type=str, choices=['nano', 'qwen'])
    args = parser.parse_args()

    # Define Rubric Filenames
    date_str = datetime.now().strftime("%Y-%m-%d")
    nano_final_file = f"gpt-5-nano-RAG-{date_str}-hw3.json"
    qwen_final_file = f"qwen-3-8b-RAG-{date_str}-hw3.json"


    print("Loading JSON data from file (:")
    try: #attempt to retrieve the context chunks, if the file is not found, catch the error and update the user
        with open('dev-v2.0.json', 'r', encoding='utf-8') as file: data = json.load(file)
    except FileNotFoundError:
        print("Warning: 'dev-v2.0.json' not found. Ensure it is in the directory.")
        return # Exit the main function if the file is not found


    if args.mode == 'batch':
        # DB Setup needed for generation steps
        context_chunks = get_context_chunks(data)
        print(f"Building Database with {len(context_chunks)} chunks...")
        collection = create_chromadb_collection(context_chunks)
        # Extract 500 possible questions
        questions = get_possible_qas(data)
        print(f"Extracted {len(questions)} possible questions for the batch job.")
        create_batch_file(questions, 'batch_requests.jsonl', collection)
        submit_batch_job('batch_requests.jsonl')

    # RAG Serial Pipeline(Qwen)
    elif args.mode == 'serial':
        context_chunks = get_context_chunks(data)
        print(f"Building Database with {len(context_chunks)} chunks...")
        collection = create_chromadb_collection(context_chunks)
        # Extract 500 possible questions
        questions = get_possible_qas(data)
        print(f"Extracted {len(questions)} possible questions for the serial run.")
        
        run_serial_openrouter(questions, collection)

    elif args.mode == 'grade':
        print("\nGrader Activated.")
        
        #Grade GPT-5-Nano (Batch Results)
        #Assumes you downloaded the batch output to 'gpt-5-rag-answers.jsonl'
        if os.path.exists('gpt-5-rag-answers.jsonl'):
            create_grading_batch(data, 'gpt-5-rag-answers.jsonl', 'grade_nano.jsonl')
            submit_batch_job('grade_nano.jsonl', description="grading-nano")
        else:
            print("error: File not found")

        #Grade Qwen (Serial Results) catch mistakes
        if os.path.exists('qwen-rag-answers.json'):
            create_grading_batch(data, 'qwen-rag-answers.json', 'grade_qwen.jsonl')
            submit_batch_job('grade_qwen.jsonl', description="grading-qwen")
        else:
            print("error: File not found")
    #Download Grading Results, required batch_id
    elif args.mode == 'download':
        if not args.batch_id or not args.model_name:
            print("Error: You must provide --batch_id AND --model_name (nano or qwen)")
            return

        # Determine the correct filename based on the model argument
        if args.model_name == 'nano':
            target_file = nano_final_file
        elif args.model_name == 'qwen':
            target_file = qwen_final_file
        
        print(f"Attempting to download Batch {args.batch_id}...")
        print(f"Target Filename: {target_file}")
        
        # Reusing check_and_download function, saves the final JSON directly with the rubric name
        check_and_download_batch(args.batch_id, target_file)


if __name__ == "__main__":
    main()