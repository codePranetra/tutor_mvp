"""
LangChain RAG-based chatbot service with Flask web interface.
"""
import os
import logging
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Store chat histories per user
user_chat_histories = {}

# Global variables for vector store and retriever
vector_store = None
retriever = None


class LangChainRAGService:
    """Service class for managing LangChain RAG-based AI interactions."""
    
    def __init__(self):
        """Initialize LangChain RAG service with OpenAI LLM and vector store."""
        try:
            logger.info("=" * 60)
            logger.info("Starting RAG service initialization...")
            logger.info("=" * 60)
            
            # Check API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY environment variable is not set!")
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            
            logger.info(f"API Key found: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '***'}")
            
            # Ensure API key is in environment for LangChain to pick up
            os.environ["OPENAI_API_KEY"] = api_key
            
            # Initialize LLM
            logger.info("Initializing ChatOpenAI LLM...")
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.7
                )
                logger.info("✓ ChatOpenAI LLM initialized successfully")
            except Exception as e:
                logger.error(f"✗ Error initializing ChatOpenAI with parameters: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Fallback: initialize with minimal parameters
                logger.info("Trying fallback initialization...")
                try:
                    self.llm = ChatOpenAI(model="gpt-4o-mini")
                    logger.info("✓ ChatOpenAI LLM initialized with fallback method")
                except Exception as e2:
                    logger.error(f"✗ Fallback initialization also failed: {e2}")
                    raise
            
            # Initialize embeddings
            logger.info("Initializing OpenAIEmbeddings...")
            try:
                self.embeddings = OpenAIEmbeddings()
                logger.info("✓ OpenAIEmbeddings initialized successfully")
            except Exception as e:
                logger.error(f"✗ Error initializing OpenAIEmbeddings: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Fallback: try with explicit API key
                logger.info("Trying fallback initialization with explicit API key...")
                try:
                    self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                    logger.info("✓ OpenAIEmbeddings initialized with fallback method")
                except Exception as e2:
                    logger.error(f"✗ Fallback initialization also failed: {e2}")
                    raise
            
            # Load and process documents
            logger.info("Loading and processing documents...")
            self._load_documents()
            
            # Create the RAG chain
            logger.info("Creating RAG chain...")
            self._create_rag_chain()
            logger.info("✓ RAG chain created successfully")
            
            logger.info("=" * 60)
            logger.info("RAG service initialization completed successfully!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error("RAG service initialization FAILED!")
            logger.error("=" * 60)
            logger.error(f"Error: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
    
    def _load_documents(self):
        """Load documents from data.txt and create vector store."""
        global vector_store, retriever
        
        try:
            logger.info("  → Reading data.txt file...")
            # Load the text file directly (Windows compatible)
            with open("data.txt", "r", encoding="utf-8") as f:
                text = f.read()
            
            file_size = len(text)
            logger.info(f"  ✓ File loaded: {file_size} characters")
            
            # Create a document from the text
            logger.info("  → Creating document...")
            documents = [Document(page_content=text)]
            logger.info(f"  ✓ Document created")
            
            # Split documents into chunks
            logger.info("  → Splitting documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            splits = text_splitter.split_documents(documents)
            logger.info(f"  ✓ Split into {len(splits)} chunks")
            
            # Create vector store
            logger.info("  → Creating FAISS vector store...")
            vector_store = FAISS.from_documents(splits, self.embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            logger.info("  ✓ Vector store created successfully")
            
        except FileNotFoundError as e:
            logger.error(f"✗ File not found: {e}")
            logger.error("  Make sure data.txt exists in the project directory")
            raise
        except Exception as e:
            logger.error(f"✗ Error loading documents: {e}")
            logger.error(f"  Error type: {type(e).__name__}")
            import traceback
            logger.error(f"  Traceback:\n{traceback.format_exc()}")
            raise
    
    def _create_rag_chain(self):
        """Create the RAG chain for question answering."""
        try:
            logger.info("  → Creating prompt template...")
            # Define the prompt template
            template = """You are a friendly, warm, and helpful educational assistant that helps students learn from the knowledge base (data.txt). Your primary goal is to facilitate learning and understanding while maintaining a friendly, approachable, and encouraging tone.

**LANGUAGE DETECTION AND RESPONSE:**
- **CRITICAL**: Detect the language/mixing style of the user's question
- If user writes in **Hindi** (Devanagari script), respond entirely in **Hindi**
- If user writes in **English**, respond entirely in **English**
- If user writes in **Hinglish** (mix of Hindi and English), respond in **Hinglish** (same mixing pattern)
- Match the user's language style exactly - this is very important for effective communication

**CONVERSATION HANDLING:**

1. **GREETINGS AND CASUAL CONVERSATION** (Very Important):
   - When user greets you (hi, hello, namaste, नमस्ते, कैसे हो, how are you, etc.), respond warmly and friendly
   - Use a cheerful, welcoming tone
   - Briefly introduce yourself as a learning assistant
   - Gently guide the conversation toward learning: "I'm here to help you learn about banking and economics! What would you like to know?"
   - Examples:
     * Hindi: "नमस्ते! मैं आपकी सहायता करने के लिए यहाँ हूँ। मैं बैंकिंग और अर्थशास्त्र के बारे में आपकी मदद कर सकता हूँ। आप क्या जानना चाहेंगे?"
     * English: "Hello! I'm here to help you learn. I can assist you with banking and economics topics. What would you like to learn about?"
     * Hinglish: "Hi! Main aapki help karne ke liye yahan hoon. Main banking aur economics ke topics par aapki madad kar sakta hoon. Aap kya janna chahenge?"

2. **OFF-TOPIC CONVERSATIONS**:
   - If user asks about something not related to the knowledge base (weather, general chat, etc.), respond friendly but gently redirect
   - Acknowledge their question briefly
   - Politely guide them back: "That's interesting! I'm here to help you learn about banking and economics. Would you like to explore any topics from that area?"
   - Always maintain a friendly, non-pushy tone
   - Never be rude or dismissive

3. **LEARNING ASSISTANT MODE** (Default):
   - When students ask about topics from the knowledge base, provide comprehensive explanations
   - Use information from the knowledge base to answer accurately
   - Structure answers with clear explanations, examples, and key points
   - Help students understand concepts step-by-step
   - If information is not in knowledge base, politely say so and suggest related topics you can help with
   - Always maintain an encouraging and supportive tone

4. **PRACTICE MODE** (When user requests practice):
   - When user asks for practice questions (e.g., "give me practice questions", "मुझे अभ्यास प्रश्न दो", "practice questions चाहिए")
   - Generate 3-5 relevant practice questions based on the knowledge base content
   - Format questions clearly with numbering
   - Wait for user to provide answers
   - After user submits answers, rate each answer on a scale of 1-10
   - Provide constructive feedback for each answer
   - Explain what was correct and what could be improved
   - Give the correct answer if the student's answer was incorrect or incomplete
   - Always be encouraging, even when correcting mistakes

**ANSWER RATING CRITERIA:**
- **9-10**: Excellent - Complete, accurate, well-explained
- **7-8**: Good - Mostly correct with minor gaps
- **5-6**: Average - Partially correct but missing important points
- **3-4**: Below Average - Some understanding but significant gaps
- **1-2**: Poor - Minimal understanding or incorrect

**CRITICAL FORMATTING INSTRUCTIONS:**
1. **For multi-part questions**: Answer each part separately with clear headings or labels
2. **Paragraphs**: Use double line breaks (\\n\\n) to separate paragraphs
3. **Lists**: 
   - Use bullet points (- or •) for unordered lists
   - Use numbered lists (1., 2., 3.) for sequential steps or ordered information
   - Always put each list item on a new line
4. **Headings**: Use **bold** for section headings, important terms, or key concepts
5. **Structure**: 
   - Start with a brief introduction if the question is complex
   - Break down complex answers into clear sections
   - Use sub-points when explaining multiple aspects
   - End with a brief summary if appropriate
6. **Spacing**: Add blank lines between major sections for readability
7. **Examples/Calculations**: 
   - Show calculations step-by-step with clear labels
   - Format examples with proper indentation or bullet points
8. **Definitions**: When defining terms, use bold for the term followed by a clear explanation

**PRACTICE QUESTION FORMAT:**
When generating practice questions, use this format:

**अभ्यास प्रश्न (Practice Questions)** or **Practice Questions**

1. [Question 1]
2. [Question 2]
3. [Question 3]
...

**ANSWER RATING FORMAT:**
When rating answers, use this format:

**आपके उत्तरों का मूल्यांकन (Answer Evaluation)** or **Answer Evaluation**

**प्रश्न 1 (Question 1):**
- आपका उत्तर (Your Answer): [student's answer]
- रेटिंग (Rating): [X]/10
- फीडबैक (Feedback): [detailed feedback]
- सही उत्तर (Correct Answer): [correct answer if needed]

**प्रश्न 2 (Question 2):**
...

Use the following context from the knowledge base to answer the question:
{context}

Previous conversation history:
{chat_history}

Question: {question}

Answer (detect user's language and respond in the same language/style. Be friendly and warm. If it's a greeting, respond warmly and guide toward learning. If it's off-topic, acknowledge it friendly and gently redirect to learning topics. Format your response with proper structure, clear sections, paragraphs, lists, and formatting. Make it easy to read and understand. Always maintain an encouraging and supportive tone):"""
            
            self.prompt = ChatPromptTemplate.from_template(template)
            logger.info("  ✓ Prompt template created")
            
            # Create the RAG chain
            logger.info("  → Building RAG chain...")
            self.rag_chain = (
                {
                    "context": lambda x: self._format_docs(retriever.invoke(x["question"])),
                    "question": RunnablePassthrough(),
                    "chat_history": lambda x: x.get("chat_history", "")
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            logger.info("  ✓ RAG chain built successfully")
            
        except Exception as e:
            logger.error(f"✗ Error creating RAG chain: {e}")
            logger.error(f"  Error type: {type(e).__name__}")
            import traceback
            logger.error(f"  Traceback:\n{traceback.format_exc()}")
            raise
    
    def _format_docs(self, docs):
        """Format retrieved documents into a string."""
        return "\n\n".join([doc.page_content for doc in docs])
    
    def get_or_create_chat_history(self, user_id, history=None):
        """
        Get or create a chat history for a user.
        
        Args:
            user_id: User identifier
            history: Optional list of persisted messages from database
        
        Returns:
            ChatMessageHistory object
        """
        chat_history = ChatMessageHistory()

        if history:
            logger.info(f"Loading {len(history)} messages from history for user {user_id}")
            for entry in history:
                role = entry.get("role")
                content = entry.get("message", "")
                if not content:
                    continue
                if role == "user":
                    chat_history.add_user_message(content)
                elif role == "assistant":
                    chat_history.add_ai_message(content)
        else:
            # Check if we have in-memory history
            if user_id in user_chat_histories and len(user_chat_histories[user_id].messages) > 0:
                return user_chat_histories[user_id]

        user_chat_histories[user_id] = chat_history
        return chat_history
    
    def process_message(self, user_id, message_text, history=None):
        """
        Process incoming message with LangChain RAG.
        
        Args:
            user_id: User identifier
            message_text: User's message text
            history: Optional list of persisted messages
        
        Returns:
            Response text from the RAG chain
        """
        try:
            # Get or create chat history for this user
            chat_history = self.get_or_create_chat_history(user_id, history=history)
            
            # Build chat history string and detect practice mode
            chat_history_str = ""
            practice_mode_active = False
            
            # Check if we're in practice mode (user asked for practice questions)
            for msg in chat_history.messages:
                if isinstance(msg, HumanMessage):
                    msg_lower = msg.content.lower()
                    # Check if user asked for practice
                    if any(phrase in msg_lower for phrase in [
                        'practice question', 'अभ्यास प्रश्न', 'practice questions', 
                        'मुझे अभ्यास', 'give me practice', 'practice चाहिए',
                        'practice questions दो', 'अभ्यास करना'
                    ]):
                        practice_mode_active = True
                    chat_history_str += f"Human: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    # Check if assistant provided practice questions
                    if 'अभ्यास प्रश्न' in msg.content or 'Practice Question' in msg.content or 'practice question' in msg.content.lower():
                        practice_mode_active = True
                    chat_history_str += f"Assistant: {msg.content}\n"
            
            # Add context about practice mode to the question
            enhanced_question = message_text
            if practice_mode_active:
                enhanced_question += "\n\n[IMPORTANT: The user is in PRACTICE MODE. If they are providing answers to practice questions, rate their answers on a scale of 1-10 and provide detailed feedback for each answer. Include the correct answer if their answer was incorrect or incomplete.]"
            
            # Run the RAG chain
            response_text = self.rag_chain.invoke({
                "question": enhanced_question,
                "chat_history": chat_history_str
            })
            
            # Add messages to history
            chat_history.add_user_message(message_text)
            chat_history.add_ai_message(response_text)
            
            return response_text
            
        except Exception as e:
            import traceback
            logger.error("=" * 60)
            logger.error(f"Error processing message for user {user_id}: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Message: {message_text}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            logger.error("=" * 60)
            return f"Sorry, something went wrong: {str(e)}. Please try again."


# Initialize the RAG service
rag_service = None
try:
    logger.info("\n" + "=" * 60)
    logger.info("APPLICATION STARTUP")
    logger.info("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("✗ OPENAI_API_KEY environment variable is not set!")
        logger.error("  Please set it using one of the following methods:")
        logger.error("  1. PowerShell: $env:OPENAI_API_KEY='your-key-here'")
        logger.error("  2. Create a .env file with: OPENAI_API_KEY=your-key-here")
        logger.error("  3. Set it as a system environment variable")
    else:
        logger.info("✓ OPENAI_API_KEY found in environment")
        logger.info("Starting RAG service initialization...")
        rag_service = LangChainRAGService()
        logger.info("\n" + "=" * 60)
        logger.info("✓ RAG SERVICE READY")
        logger.info("=" * 60 + "\n")
except Exception as e:
    logger.error("\n" + "=" * 60)
    logger.error("✗ FAILED TO INITIALIZE RAG SERVICE")
    logger.error("=" * 60)
    logger.error(f"Error: {str(e)}")
    logger.error(f"Error type: {type(e).__name__}")
    import traceback
    logger.error(f"\nFull traceback:\n{traceback.format_exc()}")
    logger.error("=" * 60 + "\n")
    rag_service = None


@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages from the web interface."""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        user_id = data.get('user_id', 'default_user')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        if rag_service is None:
            return jsonify({
                'error': 'RAG service not initialized. Please check server logs for details.',
                'status': 'error'
            }), 500
        
        # Process the message
        response = rag_service.process_message(user_id, user_message)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"Error in /api/chat endpoint: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        logger.error("=" * 60)
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'rag_service_initialized': rag_service is not None
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
