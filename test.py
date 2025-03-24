from langchain_ollama import OllamaEmbeddings


embed = OllamaEmbeddings(base_url='localhost:11434', model='nomic-embed-text')
input_text = "The meaning of life is 42"
vector = embed.embed_query(input_text)
print(vector[:3])