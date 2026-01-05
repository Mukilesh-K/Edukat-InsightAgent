import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

class IntentClassification:
    def __init__(self, model_name="gpt-4o-mini"):
        # Initialize the ChatOpenAI model
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        self.intents = [
            "get_class_information",
            "get_person_details",
            "check_attendance",
            "check_assignment",
            "get_subject_details"
        ]
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an intelligent intent classifier. "
                       "Classify the user's input into one of the following intents: {intents}. "
                       "Return the result in JSON format with a single key 'intent'."),
            ("user", "{text}")
        ])
        
        # Create the chain: Prompt -> LLM -> Output Parser
        self.chain = self.prompt | self.llm | StrOutputParser()

    def intent_classification(self, text):
        """
        Classifies the intent of the given text using LangChain and OpenAI.
        """
        try:
            # Invoke the chain
            response_str = self.chain.invoke({
                "intents": ", ".join(self.intents),
                "text": text
            })
            
            # Parse the JSON string result
            result = json.loads(response_str)
            return result.get("intent")
            
        except Exception as e:
            print(f"Error classifying intent: {e}")
            return None

if __name__ == "__main__":
    intent_classifier = IntentClassification()
    
    test_queries = [
        "What are the details of the math class?",
        "Who is the person in charge?",
        "Is the assignment due today?",
        "Show me the subject details."
    ]
    
    print("Testing Intent Classifier with LangChain...")
    for query in test_queries:
        intent = intent_classifier.intent_classification(query)
        print(f"Query: '{query}' -> Intent: {intent}")