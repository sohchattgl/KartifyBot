"""
Kartify Order Query ChatBot - Streamlit Application
Multi-Agent System with LangGraph for Customer Support
"""

import streamlit as st
import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, Sequence
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Kartify Customer Support",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        align-self: flex-end;
    }
    .assistant-message {
        background-color: #f5f5f5;
        align-self: flex-start;
    }
    .order-details {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# STATE DEFINITION
# ============================================================================
class ChatBotState(TypedDict):
    """State shared across all agents"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    customer_query: str
    order_id: str | None
    order_data: dict | None
    product_data: dict | None
    quality_check_result: dict | None
    replacement_result: dict | None
    next_agent: str
    final_response: str | None

# ============================================================================
# CONFIGURATION LOADER
# ============================================================================
@st.cache_resource
def load_config():
    """Load API credentials from config.json"""
    try:
        with open('config.json', 'r') as file:
            config = json.load(file)
            API_KEY = config.get("API_KEY")
            OPENAI_API_BASE = config.get("OPENAI_API_BASE")
            
            os.environ['OPENAI_API_KEY'] = API_KEY
            if OPENAI_API_BASE:
                os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
            
            return True
    except Exception as e:
        st.error(f"⚠️ Error loading config: {str(e)}")
        return False

# ============================================================================
# DATABASE SERVICES
# ============================================================================
class SQLOrderService:
    """Service for querying orders from SQLite database"""
    
    def __init__(self, db_path='orders.db'):
        self.db_path = db_path
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    
    def get_order(self, order_id: str) -> dict | None:
        """Get order details using SQL query"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT
                    o.order_id, o.order_date, o.status, o.delivery_date,
                    o.total_amount, o.shipping_address, o.payment_method,
                    c.name as customer_name, c.email as customer_email
                FROM orders o
                JOIN customers c ON o.customer_id = c.customer_id
                WHERE o.order_id = ?
            ''', (order_id,))
            
            order_row = cursor.fetchone()
            if not order_row:
                conn.close()
                return None
            
            cursor.execute('''
                SELECT oi.product_id, p.name, oi.quantity, oi.price_at_purchase
                FROM order_items oi
                JOIN products p ON oi.product_id = p.product_id
                WHERE oi.order_id = ?
            ''', (order_id,))
            
            items_rows = cursor.fetchall()
            
            order_data = {
                "order_id": order_row["order_id"],
                "customer_name": order_row["customer_name"],
                "customer_email": order_row["customer_email"],
                "order_date": order_row["order_date"],
                "status": order_row["status"],
                "delivery_date": order_row["delivery_date"],
                "total": order_row["total_amount"],
                "shipping_address": order_row["shipping_address"],
                "payment_method": order_row["payment_method"],
                "items": [
                    {
                        "product_id": row["product_id"],
                        "name": row["name"],
                        "quantity": row["quantity"],
                        "price": row["price_at_purchase"]
                    }
                    for row in items_rows
                ]
            }
            
            conn.close()
            return order_data
            
        except Exception as e:
            st.error(f"Error querying order: {str(e)}")
            return None

class SQLProductService:
    """Service for querying products from SQLite database"""
    
    def __init__(self, db_path='orders.db'):
        self.db_path = db_path
    
    def get_product(self, product_id: str) -> dict | None:
        """Get product details using SQL query"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT product_id, name, description, price, warranty_period,
                       return_policy, category, stock_quantity
                FROM products
                WHERE product_id = ?
            ''', (product_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            return {
                "product_id": row["product_id"],
                "name": row["name"],
                "description": row["description"],
                "price": row["price"],
                "warranty": row["warranty_period"],
                "return_policy": row["return_policy"],
                "category": row["category"],
                "stock_quantity": row["stock_quantity"]
            }
            
        except Exception as e:
            st.error(f"Error querying product: {str(e)}")
            return None

class ReplacementService:
    """Service for creating replacement orders"""
    
    def __init__(self, db_path='orders.db'):
        self.db_path = db_path
    
    def create_replacement(self, order_id: str, reason: str) -> dict:
        """Create a replacement order"""
        replacement_id = f"REP{order_id[3:]}"
        
        return {
            "replacement_id": replacement_id,
            "original_order_id": order_id,
            "status": "Initiated",
            "estimated_delivery": (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"),
            "tracking_number": f"TRK{replacement_id}",
            "reason": reason
        }

# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

class OrchestratorAgent:
    """LLM-driven orchestrator for Kartify support — recursion-safe, context-aware routing"""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are the orchestrator for Kartify's support workflow.
                Decide which agent should act next based on the customer's message and context.

                Agents:
                - order_retrieval → Fetch order info (only if missing)
                - product_info → Get product details (only if missing)
                - quality_check → Check damage/replacement eligibility
                - replacement_processing → Create replacement or reorder
                - response_generation → Generate final reply

                Rules:
                - Never call order_retrieval if order_data already exists.
                - If order_data exists and the issue seems resolved or non-specific → go to response_generation.
                - If the query confirms an action (like 'yes', 'please assist', 'go ahead') → route to replacement_processing.
                - If the query is vague, greeting, or unrelated → route to response_generation.
                - Always output ONLY the agent name (lowercase, underscore format).
                """
            ),
            (
                "human",
                """Customer Query:
                {query}

                Context:
                - Has order data: {has_order}
                - Has product data: {has_product}
                - Has quality check: {has_quality}
                - Has replacement: {has_replacement}

                Next agent:"""
            )
        ])

    def process(self, state: ChatBotState) -> ChatBotState:
        query = state.get("customer_query", "").strip().lower()
        has_order = state.get("order_data") is not None
        has_product = state.get("product_data") is not None
        has_quality = state.get("quality_check_result") is not None
        has_replacement = state.get("replacement_result") is not None

        # --- Explicit handling for confirmations and vague responses ---
        confirm_words = [
            "yes", "sure", "okay", "ok", "please assist", "go ahead",
            "do it", "help me", "confirm", "proceed", "yeah"
        ]
        vague_words = [
            "hi", "hello", "thanks", "thank you", "fine", "good", "no"
        ]

        if any(word in query for word in confirm_words):
            next_agent = "replacement_processing"
        elif any(word in query.split() for word in vague_words) or not query:
            next_agent = "response_generation"
        else:
            # --- Use LLM for complex routing decisions ---
            chain = self.prompt | self.llm
            try:
                response = chain.invoke({
                    "query": query,
                    "has_order": has_order,
                    "has_product": has_product,
                    "has_quality": has_quality,
                    "has_replacement": has_replacement
                })
                next_agent = response.content.strip().lower().replace(" ", "_")
            except Exception:
                next_agent = "response_generation"

        # --- Guardrails to prevent loops ---
        allowed_agents = [
            "order_retrieval", "product_info", "quality_check",
            "replacement_processing", "response_generation"
        ]
        if next_agent not in allowed_agents:
            next_agent = "response_generation"

        # --- Final logic cleanup ---
        if has_order and next_agent == "order_retrieval":
            next_agent = "response_generation"

        # If all info exists and no issue keywords → go straight to response_generation
        if has_order and not any(
            k in query for k in ["damage", "broken", "replace", "cancel", "reorder"]
        ):
            if next_agent != "replacement_processing":
                next_agent = "response_generation"

        # ✅ Save the next step
        state["next_agent"] = next_agent
        return state


class OrderRetrievalAgent:
    """Retrieves order information from SQL database"""

    def __init__(self, llm):
        self.llm = llm
        self.order_service = SQLOrderService()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract the order ID from the customer query.
            Look for patterns like ORD followed by numbers (e.g., ORD12345).
            Return ONLY the order ID in format ORDXXXXX or the word 'NOT_FOUND' if no order ID is present.
            Do not include any other text."""),
            ("human", "{query}")
        ])

    def process(self, state: ChatBotState) -> ChatBotState:
        chain = self.prompt | self.llm
        response = chain.invoke({"query": state["customer_query"]})
        order_id = response.content.strip()

        if order_id != "NOT_FOUND" and "ORD" in order_id.upper():
            order_data = self.order_service.get_order(order_id.upper())
            state["order_id"] = order_id.upper()
            state["order_data"] = order_data

        state["next_agent"] = "orchestrator"
        return state

class ProductInfoAgent:
    """Retrieves product information from SQL database"""

    def __init__(self):
        self.product_service = SQLProductService()

    def process(self, state: ChatBotState) -> ChatBotState:
        if state.get("order_data"):
            items = state["order_data"].get("items", [])
            product_data = []

            for item in items:
                product_id = item.get("product_id")
                if product_id:
                    product = self.product_service.get_product(product_id)
                    if product:
                        product_data.append(product)

            state["product_data"] = product_data

        state["next_agent"] = "orchestrator"
        return state

class QualityCheckAgent:
    """Checks if order qualifies for replacement"""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze if the customer query mentions any product issues.
            Look for keywords like: damaged, defective, broken, wrong item, not working, faulty, quality issues.
            Respond with only 'YES' if issues are mentioned, or 'NO' if not."""),
            ("human", "{query}")
        ])

    def process(self, state: ChatBotState) -> ChatBotState:
        if not state.get("order_data"):
            state["quality_check_result"] = {
                "eligible": False,
                "reason": "No order data available",
                "issues": []
            }
        else:
            order_data = state["order_data"]
            is_delivered = order_data.get("status") == "Delivered"

            delivery_date_str = order_data.get("delivery_date")
            within_window = False
            if delivery_date_str:
                delivery_date = datetime.strptime(delivery_date_str, "%Y-%m-%d")
                days_since_delivery = (datetime.now() - delivery_date).days
                within_window = days_since_delivery <= 30

            chain = self.prompt | self.llm
            response = chain.invoke({"query": state["customer_query"]})
            has_valid_reason = response.content.strip().upper() == "YES"

            eligible = is_delivered and within_window and has_valid_reason

            issues = []
            if has_valid_reason:
                query_lower = state["customer_query"].lower()
                if "damaged" in query_lower or "damage" in query_lower:
                    issues.append("Product damaged")
                if "defective" in query_lower or "broken" in query_lower:
                    issues.append("Product defective")
                if "wrong" in query_lower:
                    issues.append("Wrong item received")
                if not issues:
                    issues.append("Quality issue reported")

            state["quality_check_result"] = {
                "eligible": eligible,
                "reason": f"Delivered: {is_delivered}, Within window: {within_window}, Valid reason: {has_valid_reason}",
                "issues": issues,
                "days_since_delivery": (datetime.now() - delivery_date).days if delivery_date_str else None
            }

        state["next_agent"] = "orchestrator"
        return state

class ReplacementProcessingAgent:
    """Processes replacement or reorder requests safely"""

    def __init__(self):
        self.replacement_service = ReplacementService()

    def process(self, state: ChatBotState) -> ChatBotState:
        # --- Retrieve current state data safely ---
        quality_check = state.get("quality_check_result") or {}
        order_data = state.get("order_data")
        order_id = state.get("order_id")

        # --- Guard: ensure order details exist ---
        if not order_data or not order_id:
            state["final_response"] = (
                "I couldn’t find your order details. Could you please provide your order ID so I can help with the replacement?"
            )
            state["next_agent"] = "response_generation"
            return state

        # --- Determine reason and eligibility ---
        eligible = isinstance(quality_check, dict) and quality_check.get("eligible", False)
        reason = ", ".join(quality_check.get("issues", [])) if quality_check else "Customer requested reorder"

        # --- Create replacement or reorder record ---
        if eligible or "reorder" in state.get("customer_query", "").lower() or "assist" in state.get("customer_query", "").lower():
            replacement_result = self.replacement_service.create_replacement(order_id, reason)
            state["replacement_result"] = replacement_result
        else:
            # Not eligible — fallback message
            state["replacement_result"] = {
                "replacement_id": None,
                "status": "Not Eligible",
                "reason": reason
            }
            state["final_response"] = (
                "It looks like this order may not be eligible for a replacement. "
                "Could you please confirm if you'd like to reorder these items instead?"
            )

        # --- Always proceed to response generation ---
        state["next_agent"] = "response_generation"
        return state


class ResponseGenerationAgent:
    """Generates the final customer-facing response for Kartify support.
    Always produces a helpful, empathetic, and complete answer — even with minimal data.
    """

    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a warm, professional customer support representative for Kartify.
                Your goal is to provide a clear, empathetic, and helpful response based on all available context.

                Guidelines:
                - Be natural, polite, and solution-oriented.
                - Always acknowledge the customer’s concern.
                - If information is incomplete or unclear, politely ask for clarification.
                - If the query is vague (e.g., greetings or thanks), respond appropriately and keep it brief.
                - If the issue is resolved (replacement or confirmation given), close the conversation positively.
                - NEVER say you are an AI — respond as a human Kartify support agent.
                - Always end on a reassuring, customer-friendly note, but DO NOT include a personal name or signature line.

                Information you can use:
                - Customer Query: {query}
                - Order Data: {order_data}
                - Product Data: {product_data}
                - Quality Check Result: {quality_check}
                - Replacement Result: {replacement_result}
                """
            ),
            (
                "human",
                """Based on the above context, write a clear, kind, and helpful response for the customer.
                The message should sound natural, empathetic, and directly address their concern."""
            )
        ])

    def process(self, state: ChatBotState) -> ChatBotState:
        """Generate a friendly, fallback-safe customer response."""

        query = state.get("customer_query", "").strip()
        order_data = str(state.get("order_data", "No order data"))
        product_data = str(state.get("product_data", "No product data"))
        quality_check = str(state.get("quality_check_result", "No quality check performed"))
        replacement_result = str(state.get("replacement_result", "No replacement created"))

        # 🛡️ Handle vague or empty queries before calling LLM
        vague_terms = ["hi", "hello", "hey", "thanks", "thank you", "ok", "okay"]
        if not query or query.lower().strip() in vague_terms:
            response_text = "Hi there! 😊 How can I assist you with your Kartify order today?"
        else:
            try:
                chain = self.prompt | self.llm
                response = chain.invoke({
                    "query": query,
                    "order_data": order_data,
                    "product_data": product_data,
                    "quality_check": quality_check,
                    "replacement_result": replacement_result
                })
                response_text = response.content.strip()
            except Exception as e:
                # 🧩 Safe fallback if LLM call fails
                response_text = (
                    "I'm sorry, something went wrong while preparing your response. "
                    "Could you please rephrase or provide a bit more detail about your concern?"
                )

        # 🧠 Ensure a final response always exists
        if not response_text or response_text.strip() == "":
            response_text = "I’m here to help with your Kartify order. Could you please clarify your request?"

        # 💬 Save the final message and signal end of flow
        state["final_response"] = response_text
        state["next_agent"] = None  # 'None' or 'end' → terminate the graph safely

        return state
# ============================================================================
# BUILD GRAPH
# ============================================================================
@st.cache_resource
def build_graph():
    """Build and compile the LangGraph workflow"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    orchestrator = OrchestratorAgent(llm)
    order_agent = OrderRetrievalAgent(llm)
    product_agent = ProductInfoAgent()
    quality_agent = QualityCheckAgent(llm)
    replacement_agent = ReplacementProcessingAgent()
    response_agent = ResponseGenerationAgent(llm)
    
    def orchestrator_node(state: ChatBotState):
        return orchestrator.process(state)
    
    def order_node(state: ChatBotState):
        return order_agent.process(state)
    
    def product_node(state: ChatBotState):
        return product_agent.process(state)
    
    def quality_node(state: ChatBotState):
        return quality_agent.process(state)
    
    def replacement_node(state: ChatBotState):
        return replacement_agent.process(state)
    
    def response_node(state: ChatBotState):
        return response_agent.process(state)
    
    workflow = StateGraph(ChatBotState)
    
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("order_retrieval", order_node)
    workflow.add_node("product_info", product_node)
    workflow.add_node("quality_check", quality_node)
    workflow.add_node("replacement_processing", replacement_node)
    workflow.add_node("response_generation", response_node)
    
    workflow.add_edge("order_retrieval", "orchestrator")
    workflow.add_edge("product_info", "orchestrator")
    workflow.add_edge("quality_check", "orchestrator")
    workflow.add_edge("replacement_processing", "response_generation")
    workflow.add_edge("response_generation", END)
    
    def route_next(state: ChatBotState):
        return state["next_agent"]
    
    workflow.add_conditional_edges(
        "orchestrator",
        route_next,
        {
            "order_retrieval": "order_retrieval",
            "product_info": "product_info",
            "quality_check": "quality_check",
            "replacement_processing": "replacement_processing",
            "response_generation": "response_generation",
            "end": END
        }
    )
    
    workflow.set_entry_point("orchestrator")
    return workflow.compile()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'chatbot_state' not in st.session_state:
    st.session_state.chatbot_state = {
        "messages": [],
        "customer_query": None,
        "order_id": None,
        "order_data": None,
        "product_data": None,
        "quality_check_result": None,
        "replacement_result": None,
        "next_agent": "orchestrator",
        "final_response": None
    }

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">📦 Kartify Customer Support</h1>', unsafe_allow_html=True)
    
    # Load configuration
    if not load_config():
        st.error("❌ Failed to load configuration. Please check your config.json file.")
        st.stop()
    
    # Build graph
    try:
        graph = build_graph()
    except Exception as e:
        st.error(f"❌ Failed to initialize chatbot: {str(e)}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        **Kartify Customer Support Bot**
        
        This AI-powered chatbot helps you with:
        - Order status inquiries
        - Product information
        - Return & replacement requests
        - Delivery tracking
        - Order cancellations
        
        Simply type your question below!
        """)
        
        if st.button("🔄 Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.chatbot_state = {
                "messages": [],
                "customer_query": None,
                "order_id": None,
                "order_data": None,
                "product_data": None,
                "quality_check_result": None,
                "replacement_result": None,
                "next_agent": "orchestrator",
                "final_response": None
            }
            st.rerun()
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-message user-message">👤 **You:** {msg["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message">🤖 **Kartify Support:** {msg["content"]}</div>', 
                           unsafe_allow_html=True)
                
                # Display order details if available
                if msg.get("order_data"):
                    with st.expander("📦 Order Details"):
                        order = msg["order_data"]
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Order ID:** {order['order_id']}")
                            st.write(f"**Customer:** {order['customer_name']}")
                            st.write(f"**Status:** {order['status']}")
                        with col2:
                            st.write(f"**Order Date:** {order['order_date']}")
                            st.write(f"**Total:** ${order['total']}")
                            if order['delivery_date']:
                                st.write(f"**Delivery Date:** {order['delivery_date']}")
                        
                        st.write("**Items:**")
                        for item in order['items']:
                            st.write(f"- {item['name']} (Qty: {item['quantity']}) - ${item['price']}")
    
    # Chat input
    user_input = st.chat_input("Type your message here... (e.g., 'What's the status of order ORD1001?')")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Update chatbot state
        st.session_state.chatbot_state["customer_query"] = user_input
        st.session_state.chatbot_state["messages"].append({"role": "user", "content": user_input})
        
        # Process with graph
        with st.spinner("🤔 Thinking..."):
            try:
                st.session_state.chatbot_state = graph.invoke(st.session_state.chatbot_state)
                
                # Add assistant response
                response = st.session_state.chatbot_state["final_response"]
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response,
                    "order_data": st.session_state.chatbot_state.get("order_data")
                })
                
                # Update messages
                st.session_state.chatbot_state["messages"].append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                st.error(f"❌ Error processing request: {str(e)}")
        
        st.rerun()

if __name__ == "__main__":
    main()