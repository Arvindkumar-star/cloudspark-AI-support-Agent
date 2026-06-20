from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

class PersonaDetection(BaseModel):
    persona: str = Field(description="The detected persona: 'Technical Expert', 'Frustrated User', or 'Business Executive'")
    reasoning: str = Field(description="Brief justification for the detected persona")

class PersonaEngine:
    def __init__(self, model_name="gemini-flash-latest"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self.parser = JsonOutputParser(pydantic_object=PersonaDetection)
        
    def detect_persona(self, user_query):
        prompt = ChatPromptTemplate.from_template(
            "Analyze the following user query and classify the user into one of these three personas:\n"
            "1. Technical Expert: Uses technical jargon, asks for logs/APIs/config, wants depth.\n"
            "2. Frustrated User: Uses emotional language, complains, urgent, annoyed.\n"
            "3. Business Executive: Outcome-focused, concise, cares about business impact/timeline.\n\n"
            "User Query: {query}\n\n"
            "{format_instructions}"
        )
        
        chain = prompt | self.llm | self.parser
        try:
            result = chain.invoke({
                "query": user_query, 
                "format_instructions": self.parser.get_format_instructions()
            })
            return result
        except Exception as e:
            print(f"Error detecting persona: {e}")
            return {"persona": "Neutral", "reasoning": "Fallback due to error."}

    def get_persona_instructions(self, persona):
        instructions = {
            "Technical Expert": (
                "Response style: Detailed, technical, includes root cause analysis, and step-by-step troubleshooting. "
                "Use precise terminology. If logs or API details are mentioned in context, use them."
            ),
            "Frustrated User": (
                "Response style: Highly empathetic, simple language, reassuring, and action-oriented. "
                "Acknowledge the frustration and provide a clear, immediate step forward."
            ),
            "Business Executive": (
                "Response style: Concise, outcome-focused, minimal technical jargon. "
                "Highlight the resolution timeline and business impact. Use bullet points."
            )
        }
        return instructions.get(persona, "Response style: Professional, helpful, and concise.")

    def generate_adaptive_response(self, user_query, persona, context):
        persona_style = self.get_persona_instructions(persona)
        
        prompt = ChatPromptTemplate.from_template(
            "You are an intelligent customer support agent. Your goal is to provide a grounded response based ONLY on the provided context.\n"
            "If the information is not in the context, clearly state that you don't have that information and suggest escalation.\n\n"
            "Context:\n{context}\n\n"
            "User Persona: {persona}\n"
            "Adaptation Instructions: {style}\n\n"
            "User Query: {query}\n\n"
            "Response:"
        )
        
        chain = prompt | self.llm
        result = chain.invoke({
            "query": user_query,
            "persona": persona,
            "style": persona_style,
            "context": context
        })
        content = result.content
        if isinstance(content, list):
            content = "".join([c.get("text", "") for c in content if isinstance(c, dict)])
        return content
