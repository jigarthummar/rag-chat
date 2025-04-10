# gradio_app.py
import os
import sys
import gc
import gradio as gr
from typing import List, Dict, Any, Tuple

# Import necessary modules from your RAG application
import modules.config as config
from modules.document_processor import DocumentProcessor
from modules.embeddings import EmbeddingGenerator
from modules.document_store import DocumentStore
from modules.query_enhancer import QueryEnhancer
from modules.retriever import Retriever
from modules.llm_interface import LLMInterface
from modules.conversation_history import ConversationHistory

class GradioRAGApplication:
    def __init__(self):
        """Initialize the RAG application with all its components."""
        # Initialize components
        print("Initializing RAG application components...")
        
        self.document_processor = DocumentProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        self.embedding_generator = EmbeddingGenerator(
            model_name=config.EMBEDDING_MODEL
        )
        
        self.document_store = DocumentStore(
            persist_directory=config.PERSIST_DIRECTORY
        )
        
        try:
            # Initialize the query enhancer
            self.query_enhancer = QueryEnhancer(api_key=config.OPENAI_API_KEY)
            
            # Initialize the enhanced retriever
            self.retriever = Retriever(
                embedding_generator=self.embedding_generator,
                document_store=self.document_store,
                query_enhancer=self.query_enhancer,
                top_k=config.TOP_K,
                similarity_threshold=config.SIMILARITY_THRESHOLD,
                use_query_enhancement=True,
                max_docs_per_source=config.MAX_DOCS_PER_SOURCE
            )
            
            # Initialize the enhanced LLM interface with chat capabilities
            self.llm = LLMInterface(
                api_key=config.OPENAI_API_KEY,
                model=config.LLM_MODEL
            )
            
            # Dictionary to store conversation histories by ID
            self.conversations = {}
            self.current_conversation_id = None
            
            print("RAG application components initialized successfully")
            
        except ValueError as e:
            print(f"Error initializing components: {e}")
            print("Please set your OPENAI_API_KEY in a .env file or as an environment variable.")
            sys.exit(1)
    
    def get_or_create_conversation(self, conversation_id: str = None) -> Tuple[str, ConversationHistory]:
        """Get an existing conversation or create a new one."""
        if conversation_id and conversation_id in self.conversations:
            return conversation_id, self.conversations[conversation_id]
        
        # Create a new conversation
        from datetime import datetime
        new_id = conversation_id or datetime.now().strftime("%Y%m%d%H%M%S")
        self.conversations[new_id] = ConversationHistory(
            max_history=config.MAX_HISTORY_TURNS
        )
        return new_id, self.conversations[new_id]
    
    def upload_document(self, file_obj) -> str:
        """Process and upload a document to the vector store."""
        try:
            if file_obj is None:
                return "No file selected."
            
            # Get the file path
            file_path = file_obj.name
            file_name = os.path.basename(file_path)
            
            # Check if the file exists
            if not os.path.exists(file_path):
                return f"Error: File not found: {file_path}"
                
            # Check the file extension
            if not file_name.lower().endswith(('.pdf', '.txt')):
                return "Only PDF and TXT files are supported."
                
            # Check the file size to determine processing method
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            
            # Process the document
            print(f"Processing document: {file_name}, Size= {file_size:.2f}MB")
            doc_chunks = self.document_processor.process_document(file_path)
                
            # Generate embeddings
            print(f"Generating embeddings for {len(doc_chunks)} chunks...")
            doc_chunks_with_embeddings = self.embedding_generator.process_document_chunks(doc_chunks)
                
            # Add to vector store
            print("Adding to vector store...")
            self.document_store.add_documents(doc_chunks_with_embeddings)
                
            success_message = f"Successfully uploaded document: {file_name} ({len(doc_chunks)} chunks)"
            print(success_message)
            return success_message
        
        except Exception as e:
            error_message = f"Error uploading document: {str(e)}"
            print(error_message)
            return error_message
        finally:
            # Force garbage collection
            gc.collect()
    
    def process_message(self, message: str, chatbot: List[List[str]]) -> Tuple[List[List[str]], str]:
        """
        Process a user message in the context of a conversation.
        
        Args:
            message: User's message
            chatbot: Gradio chatbot component state
            
        Returns:
            Updated chatbot and sources information
        """
        try:
            # Make sure we have a current conversation
            conv_id, conversation = self.get_or_create_conversation(self.current_conversation_id)
            self.current_conversation_id = conv_id
            
            # Add user message to chatbot
            chatbot.append([message, None])
            
            # Retrieve relevant documents with the retriever
            print(f"Retrieving relevant information for: '{message}'")
            retrieved_docs = self.retriever.retrieve(message)
            
            if not retrieved_docs:
                response_text = "I couldn't find any relevant information to answer your question."
                sources_text = "No relevant sources found."
                
                # Add to conversation history
                conversation.add_interaction(
                    user_message=message,
                    system_response=response_text,
                    retrieved_docs=[]
                )
                
                # Update chatbot with assistant response
                chatbot[-1][1] = response_text
                return chatbot, sources_text
            
            # Get conversation history for context
            chat_history = None
            if len(conversation.history) > 0:
                chat_history = conversation.get_messages_for_llm()
            
            # Generate response using the LLM interface
            print("Generating response...")
            result = self.llm.generate_response(
                query=message,
                context_docs=retrieved_docs,
                max_tokens=config.MAX_TOKENS,
                conversation_history=chat_history
            )
            
            # Extract and format source information
            sources = []
            for doc in result.get("used_context", []):
                if "metadata" in doc:
                    source = doc["metadata"].get("source", "Unknown")
                    # Format the source to show just the filename rather than the full path
                    source = os.path.basename(source)
                    similarity = doc.get("similarity", "Unknown")
                    if isinstance(similarity, float):
                        similarity_str = f"{similarity:.2f}"
                    else:
                        similarity_str = str(similarity)
                    
                    sources.append(f"{source} (Relevance: {similarity_str})")
            
            # Format sources text
            if sources:
                sources_text = "Sources:\n- " + "\n- ".join(list(dict.fromkeys(sources)))  # Remove duplicates
            else:
                sources_text = "No source information available."
            
            # Add to conversation history
            conversation.add_interaction(
                user_message=message,
                system_response=result.get("response", ""),
                retrieved_docs=retrieved_docs
            )
            
            # Update chatbot with assistant response
            chatbot[-1][1] = result.get("response", "")
            
            return chatbot, sources_text
        
        except Exception as e:
            error_message = f"Error processing your message: {str(e)}"
            print(f"Error: {error_message}")
            
            # Add error to conversation history
            if self.current_conversation_id in self.conversations:
                self.conversations[self.current_conversation_id].add_interaction(
                    user_message=message,
                    system_response=error_message,
                    retrieved_docs=[]
                )
            
            # Update chatbot with error
            chatbot[-1][1] = error_message
            return chatbot, "Error occurred. No sources available."
            
        finally:
            # Force garbage collection
            gc.collect()
    
    def clear_conversation(self) -> Tuple[List[List[str]], str]:
        """Clear the current conversation and start a new one."""
        # Create a new conversation
        self.current_conversation_id, _ = self.get_or_create_conversation()
        print(f"Started new conversation with ID: {self.current_conversation_id}")
        
        # Return empty chatbot and sources
        return [], ""
    
    def save_conversation(self) -> str:
        """Save the current conversation to a file."""
        if not self.current_conversation_id or self.current_conversation_id not in self.conversations:
            return "No active conversation to save."
        
        try:
            # Get the current conversation
            conversation = self.conversations[self.current_conversation_id]
            
            # Save to file
            file_path = conversation.save_to_file()
            return f"Conversation saved to: {file_path}"
        except Exception as e:
            return f"Error saving conversation: {str(e)}"


# Create Gradio interface
def create_gradio_interface():
    # Initialize the RAG application
    rag_app = GradioRAGApplication()
    
    # Define interface components
    with gr.Blocks(title="RAG Chat Application") as demo:
        gr.Markdown("# RAG Chat Application")
        gr.Markdown("Upload documents and ask questions about them. The system will retrieve relevant information to answer your questions.")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(height=500, label="Conversation")
                
                with gr.Row():
                    msg = gr.Textbox(placeholder="Type your question here...", label="Message", scale=4)
                    submit_btn = gr.Button("Submit", variant="primary")
                
                with gr.Row():
                    clear_btn = gr.Button("New Conversation")
                    save_btn = gr.Button("Save Conversation")
            
            with gr.Column(scale=1):
                # Document upload area
                file_upload = gr.File(label="Upload Document (PDF or TXT)")
                upload_btn = gr.Button("Process Document")
                upload_output = gr.Textbox(label="Upload Status", interactive=False)
                
                # Sources display
                sources_display = gr.Textbox(label="Sources", interactive=False, lines=10)
        
        # Set up event handlers
        upload_btn.click(
            fn=rag_app.upload_document,
            inputs=[file_upload],
            outputs=[upload_output]
        )
        
        def process_and_update(message, chatbot):
            return rag_app.process_message(message, chatbot)
        
        msg.submit(
            fn=process_and_update,
            inputs=[msg, chatbot],
            outputs=[chatbot, sources_display],
            api_name="chat"
        ).then(
            fn=lambda: "",
            outputs=[msg]
        )
        
        submit_btn.click(
            fn=process_and_update,
            inputs=[msg, chatbot],
            outputs=[chatbot, sources_display]
        ).then(
            fn=lambda: "",
            outputs=[msg]
        )
        
        clear_btn.click(
            fn=rag_app.clear_conversation,
            outputs=[chatbot, sources_display]
        )
        
        save_btn.click(
            fn=rag_app.save_conversation,
            outputs=[upload_output]
        )
        
        # Initial setup: start with a new conversation
        demo.load(
            fn=rag_app.clear_conversation,
            outputs=[chatbot, sources_display]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True)  # Set share=False in production

