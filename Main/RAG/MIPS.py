# Import necessary libraries for embedding, vector search, and data handling
from FlagEmbedding import FlagModel
from tqdm import tqdm
from util.json_method import write_dict_to_jsonl
from util.json_method import read_json_file, read_jsonl
import chromadb
from RAG.MIPS import save_recall_results
from FlagEmbedding import FlagModel
from typing import List, Dict
import os

def save_recall_results(query_text: List[str], metadatas, documents, jsonl_path, answers: None | List[str],
                        keys: None | List[str]):
    """Save recall results to JSONL file with structured format
    
    Args:
        query_text: List of query strings
        metadatas: List of metadata dictionaries for each query
        documents: List of retrieved documents for each query
        jsonl_path: Path to save the output JSONL file
        answers: List of ground truth answers (optional)
        keys: List of additional metadata keys to include (optional)
    """
    # Process each query-document-answer triplet
    for query, metadata, document, answer in zip(query_text, metadatas, documents, answers):
        # Create base data structure
        data_dict = {
            "query": query,
            "document": document,
            "answer": answer,
        }
        
        # Add additional metadata if specified
        if keys is not None:
            for key in keys:
                temp = []
                for item in metadata:
                    temp.append(item[key])
                data_dict[key] = temp
                
        # Write to JSONL file
        write_dict_to_jsonl(data_dict, jsonl_path)


class MIPS:
    """Maximum Inner Product Search implementation for efficient document retrieval
    
    This class provides an interface for semantic search using embedding models
    and ChromaDB vector database for fast similarity search.
    
    Attributes:
        embedding_model: Model for encoding queries into embeddings
        database: ChromaDB collection containing document vectors
        top_k: Number of top documents to retrieve (default: 50)
    """
    
    def __init__(self, embedding_model: FlagModel, chroma_database_collection, top_k=50):
        """Initialize MIPS with embedding model and database
        
        Args:
            embedding_model: FlagEmbedding model for query encoding
            chroma_database_collection: ChromaDB collection with indexed documents
            top_k: Number of documents to retrieve per query (default: 50)
        """
        self.embedding_model = embedding_model
        self.database = chroma_database_collection
        self.top_k = top_k

    def query(self, query_text: List[str] | str):
        """Retrieve documents using semantic similarity search
        
        Implements batch processing for large query sets and uses vector database
        for efficient similarity search.
        
        Args:
            query_text: Single query string or list of queries
            
        Returns:
            Tuple containing:
                - List of metadata dictionaries for retrieved documents
                - List of document text content
        """
        # Convert single query to list format
        if type(query_text) is str:
            query_text = [query_text]
            
        # Handle large query sets with batch processing
        if len(query_text) >= 64:
            recall_results = {
                "metadatas": [],
                "documents": []
            }
            batch_size = 64  # Process in batches to manage memory
            
            # Process queries in batches with progress bar
            for i in tqdm(range(0, len(query_text), batch_size)):
                batch_query_text = query_text[i:min(i + batch_size, len(query_text))]
                # Encode batch of queries
                batch_query_embeddings = self.embedding_model.encode(batch_query_text)
                # Search in vector database
                batch_recall_results = self.database.query(batch_query_embeddings, n_results=self.top_k)
                # Accumulate results
                recall_results["metadatas"].extend(batch_recall_results["metadatas"])
                recall_results["documents"].extend(batch_recall_results["documents"])
        else:
            # Direct processing for small query sets
            query_embeddings = self.embedding_model.encode(query_text)
            recall_results = self.database.query(query_embeddings, n_results=self.top_k)
            
        return recall_results["metadatas"], recall_results["documents"]
    # Initialize the BGE-large embedding model for semantic search
embedding_model = FlagModel(
    "BAAI/bge-large-en-v1.5",
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
    use_fp16=True,  # Use half precision for memory efficiency
)

def get_mips_result(multi_qa_dataset: List[Dict], mips_method, path, question_key="query", answer_key="answer", keys=None):
    """Process a QA dataset using MIPS and save results
    
    Args:
        multi_qa_dataset: List of QA dictionaries
        mips_method: MIPS instance for document retrieval
        path: Output path for saving results
        question_key: Key for accessing questions in dataset (default: "query")
        answer_key: Key for accessing answers in dataset (default: "answer")
        keys: Additional metadata keys to include (optional)
    """
    # Extract questions and answers from dataset
    questions = [qa[question_key] for qa in multi_qa_dataset]
    answers = [qa[answer_key] for qa in multi_qa_dataset]
    
    # Retrieve documents using MIPS
    meta_results, documents = mips_method.query(questions)
    
    # Save results with structured format
    save_recall_results(
        query_text=questions, 
        documents=documents, 
        metadatas=meta_results, 
        jsonl_path=path, 
        keys=keys, 
        answers=answers
    )
# Example usage configuration
if __name__ == "__main__":
    # Initialize ChromaDB client with persistent storage
    chroma_client = chromadb.PersistentClient(path="../database")
    
    # Load evaluation dataset
    dataset = read_json_file("../dataset/pqa/pqal/evaluate_dataset.json")
    
    # Get the medical QA collection
    collection = chroma_client.get_collection("MedPQA")
    
    # Initialize MIPS with embedding model and database collection
    mips_model = MIPS(embedding_model, collection)
    
    # Process dataset and save results
    get_mips_result(
        dataset, 
        mips_model, 
        path="dataset/mips_recall_results_PQA.jsonl",
        question_key="QUESTION", 
        answer_key="final_decision", 
        keys=["id"]
    )