from src.rag_engine import RAGEngine
from src.persona_engine import PersonaEngine
from src.escalation_logic import EscalationManager

class SupportAgent:
    def __init__(self):
        self.rag = RAGEngine()
        self.persona = PersonaEngine()
        self.escalation = EscalationManager()
        self.history = []

    def process_query(self, user_query):
        # 1. Detect Persona
        persona_result = self.persona.detect_persona(user_query)
        detected_persona = persona_result.get("persona", "Neutral")
        
        # 2. Retrieve Context
        retrieved_results = self.rag.retrieve(user_query)
        context_text = "\n\n".join([doc.page_content for doc, score in retrieved_results])
        
        # 3. Check Escalation
        should_escalate, reason = self.escalation.should_escalate(user_query, retrieved_results, self.history)
        
        response_content = ""
        handoff_summary = None
        
        if should_escalate:
            # Generate adaptive response but include escalation notice
            raw_response = self.persona.generate_adaptive_response(user_query, detected_persona, context_text)
            response_content = f"{raw_response}\n\n[SYSTEM] Escalating to human agent. Reason: {reason}"
            handoff_summary = self.escalation.generate_handoff_summary(
                detected_persona, 
                user_query, 
                self.history, 
                retrieved_results, 
                ["RAG Query", "Persona Adaptation"]
            )
        else:
            response_content = self.persona.generate_adaptive_response(user_query, detected_persona, context_text)

        # Update history
        self.history.append({"role": "user", "content": user_query, "persona": detected_persona})
        self.history.append({"role": "assistant", "content": response_content})
        
        return {
            "persona": detected_persona,
            "persona_reasoning": persona_result.get("reasoning", ""),
            "response": response_content,
            "sources": [doc.metadata.get('source') for doc, _ in retrieved_results],
            "is_escalated": should_escalate,
            "escalation_reason": reason if should_escalate else None,
            "handoff_summary": handoff_summary
        }
