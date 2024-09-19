# Sarvam.AI_RAG
Building a RAG system
RAG systems are one of the most widely used patterns which is powering a lot of AI applications. The basic idea is using an external data source in a vector database along with an LLM. You have to build a RAG system which works on a medium sized dataset. In this case you would be working with NCERT PDF text.
Build a RAG system and serve it using a FastAPI endpoint. You should be able to send a query and get a response back.
You can use any frontend you see fit to showcase this endpoint.

Building an Agent that can perform smart actions based on the user’s query. Extend the service and host another endpoint for the agent.
The agent should be reliably able to decide when to call the VectorDB and when to not. Ex: Hello -> should not call the VectorDB.
Secondly, introduce at least ONE more action / tool in your system that the Agent can invoke based on the user’s query. Bonus points for more creative actions!
