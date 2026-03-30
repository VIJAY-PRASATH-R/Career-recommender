import os
from getpass import getpass

os.environ["HF_KEY"]=getpass("Enter THE HF KEY:")
os.environ["GROQ_KEY"]=getpass("Enter the Groq Key:")

import os
from getpass import getpass
from llama_index.llms.groq import Groq
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from google.colab import files
import io

uploaded = files.upload()
for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

llm = Groq(model="llama-3.1-8b-instant",api_key=os.environ.get("GROQ_KEY"))

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

file_path = list(uploaded.keys())[0]

with open(file_path, "wb") as f:
  f.write(uploaded[file_path])

documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

index = VectorStoreIndex.from_documents(documents,embed_model=embed_model)

query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("Give the skills of the resume")

print("\n Skills found: ", response)
