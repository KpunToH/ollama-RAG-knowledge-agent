# ollama-RAG-knowledge-agent
Ground for students project on knowledge agents for non-formalized data extraction 

# Installation
1) Install Anaconda suitable for your OS: https://www.anaconda.com/download/
2) Install Ollama for local LLM inference https://ollama.com/download
3) I advise you to create a separate virtual enviroment for the project:
```bash
	conda create -n knowledge_agent python=3.11
	conda activate knowledge_agent
```
4) install the requirements from file
```bash
	cd your/notebook/directory
	pip install -r requirements.txt
```
5) If you have troubles with activating Nvidia GPU instead of CPU you may need to install torch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade --force-reinstall
```


# How to use
1) Change the path to your documents for RAG
2) Change your question for LLM+RAG
3) Run all cells in provided jupyter notebook
4) Evaluate the answer

#What to tinker
You can easily change the method for chunk separation, local LLM model and prompt, vector database thanks to Langchain

# What to read?

This example was created with the help of https://medium.com/@imabhi1216/implementing-rag-using-langchain-and-ollama-93bdf4a9027c:
- You can read more about chunking strategies here: https://medium.com/@zilliz_learn/experimenting-with-different-chunking-strategies-via-langchain-694a4bd9f7a5
- You can find the current SOTA local models on bechmarks : https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
- You can find the best embedding models for your language here: https://huggingface.co/spaces/mteb/leaderboard
- You can find additional providers for every component, already integrated into Langchain here: https://python.langchain.com/v0.2/docs/integrations/platforms/
