import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

class EscalationManager:
    def __init__(self, confidence_threshold=0.5):
        self.threshold = confidence_threshold
        self.llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

    def should_escalate(self, user_query, retrieval_scores, chat_history):
        # 1. Check retrieval confidence
        avg_score = sum([score for _, score in retrieval_scores]) / len(retrieval_scores) if retrieval_scores else 0
        if avg_score < self.threshold:
            return True, f"Low retrieval confidence ({avg_score:.2f} < {self.threshold})"
        
        # 2. Check for sensitive keywords
        sensitive_keywords = ["billing", "legal", "account hack", "sue", "lawsuit", "refund", "emergency"]
        if any(word in user_query.lower() for word in sensitive_keywords):
            return True, "Sensitive topic detected (Billing/Legal/Security)"
        
        # 3. Check for repeated frustration in history (simplified)
        frustration_count = sum(1 for msg in chat_history[-3:] if "persona" in msg and msg["persona"] == "Frustrated User")
        if frustration_count >= 2:
            return True, "Repeated user frustration"

        return False, "Standard query"

    def generate_handoff_summary(self, persona, user_query, history, retrieved_docs, attempted_actions):
        prompt = ChatPromptTemplate.from_template(
            "Generate a structured handoff summary for a human support agent based on the following conversation details.\n"
            "Detected Persona: {persona}\n"
            "Last Message: {query}\n"
            "Attempted Actions: {actions}\n"
            "Context Used: {docs}\n"
            "History Summary: {history}\n\n"
            "Output the summary as a JSON object with these keys: persona, issue, documents_used, attempted_steps, recommendation."
        )
        
        # Formulate a summary of history
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]])
        doc_names = list(set([doc.metadata.get('source', 'Unknown') for doc, _ in retrieved_docs]))
        
        chain = prompt | self.llm
        result = chain.invoke({
            "persona": persona,
            "query": user_query,
            "actions": attempted_actions,
            "docs": doc_names,
            "history": history_text
        })
        
        try:
            # Try to parse the JSON output from Gemini
            summary_raw = result.content
            if isinstance(summary_raw, list):
                summary_raw = "".join([c.get("text", "") for c in summary_raw if isinstance(c, dict)])
            
            summary_json = summary_raw.strip()
            if "```json" in summary_json:
                summary_json = summary_json.split("```json")[1].split("```")[0].strip()
            elif "```" in summary_json:
                summary_json = summary_json.split("```")[1].split("```")[0].strip()
            
            return json.loads(summary_json)
        except Exception as e:
            print(f"Error parsing handoff summary: {e}")
            return {
                "persona": persona,
                "issue": user_query,
                "documents_used": doc_names,
                "attempted_steps": attempted_actions,
                "recommendation": "Investigate immediately"
            }
