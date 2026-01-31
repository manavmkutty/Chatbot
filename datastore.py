from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import os
import shutil
import chromadb
import uuid


REBUILD_INDEX = True

chroma_path = "./data/chromadb_data"

if REBUILD_INDEX and os.path.exists(chroma_path):
    shutil.rmtree(chroma_path)

client = chromadb.PersistentClient(path = chroma_path)

collection = client.get_or_create_collection(name="documents_collection" , 
                                             metadata = {"hnsw:space": "cosine"})


def load_data_to_chromadb():
    if not REBUILD_INDEX:
        print("Index already exists. Skipping data loading.")
        return
    
    documents = load_documents()
    chunks = chunking_docs(documents)

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')   #initialised an embedding model for embedding process

    for each_chunk in chunks:   #loop through each chunk to generate embeddings for each chunk
        embeddings = embedding_function(each_chunk.page_content , embedding_model)  #this each embeddings can be stored in chromadb with the chunk metadata
        #store each_chunk and embeddings to chromadb here
        store_in_chromadb(each_chunk , embeddings , collection)

    print(f"Stored {collection.count()} chunks in ChromaDB collection.")

#method to load docs from the directory
def load_documents():
    loader_obj = DirectoryLoader(path="./data" , glob="*.pdf" , loader_cls= PyPDFLoader)
    return loader_obj.load()

#method to chunk the loaded documents
def chunking_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 150,
        length_function = len , 
        add_start_index = True
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Splitted {len(documents)} into {len(chunks)} chunks")
    print("\n\n")

    # print("Sample chunk")
    # print(chunks[60].page_content)
    # print(f"page number: {chunks[60].metadata['page']}")

    return chunks 



#method to generate embeddings for the chunks may use for encoding querries also 
def embedding_function(text_to_be_embedded , embedding_model):
    return embedding_model.encode(text_to_be_embedded)

def store_in_chromadb(chunk , embeddings , collection):

    doc_id = str(uuid.uuid4())

    collection.add(
        ids = [doc_id],
        documents = [chunk.page_content] , 
        embeddings = [embeddings.tolist()] , 
        metadatas = [chunk.metadata]
    )