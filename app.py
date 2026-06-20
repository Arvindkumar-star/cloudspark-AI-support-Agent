import gradio as gr
from src.agent_workflow import SupportAgent
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the Support Agent
agent = SupportAgent()

# Gradio Theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_width="1px",
    block_title_text_color="*primary_400",
)

def agent_predict(message, history):
    try:
        result = agent.process_query(message)
        
        # Prepare bot message with persona info and styling
        bot_msg = f"### Detected Persona: {result['persona']}\n\n{result['response']}"
        
        if result['sources']:
            bot_msg += f"\n\n---\n**Sources:** {', '.join(set(result['sources']))}"
            
        if result['is_escalated']:
            bot_msg += "\n\n⚠️ **Escalated to Human Agent**"
            if result['handoff_summary']:
                bot_msg += f"\n\n**Handoff Summary:**\n```json\n{result['handoff_summary']}\n```"
        
        return bot_msg
    except Exception as e:
        return f"❌ **Error processing request:** {str(e)}"

with gr.Blocks(title="CloudSpark AI Support") as demo:
    gr.Markdown("# ⚡ CloudSpark AI Support")
    gr.Markdown("Persona-Adaptive Customer Support Agent powered by Gemini & LangChain")
    
    with gr.Row():
        with gr.Column(scale=4):
            chat_interface = gr.ChatInterface(
                fn=agent_predict,
                fill_height=True,
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### Agent Dashboard")
            status_display = gr.Markdown("Status: **Online**")
            
            with gr.Accordion("Knowledge Base Info", open=True):
                gr.Markdown("Documents Ingested: 15")
                gr.Markdown("Vector DB: ChromaDB")
            
            gr.Markdown("---")
            gr.Markdown("### Features")
            gr.Markdown("- ✅ Persona Adaptation")
            gr.Markdown("- ✅ RAG Grounding")
            gr.Markdown("- ✅ Auto-Escalation")

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860,
        theme=theme
    )
