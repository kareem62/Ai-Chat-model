from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

def is_url_or_soap_request(question):
    # Keywords that indicate URL or SOAP request queries
    url_keywords = ["url", "link", "website", "site"]
    soap_keywords = ["request", "soap", "xml", "template", "getcabinet", "getcontact"]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in url_keywords + soap_keywords)

template_for_urls_and_soap = """
You are a helpful AI assistant that provides information about URLs and SOAP request templates.

Available items:
{reviews}

Question: {question}

Instructions for your response:
1. If the user asks about a specific URL, provide that exact matching URL and its description
2. If the user asks about a SOAP request, provide the complete SOAP request template
3. Format XML with proper indentation for readability
4. Be direct and precise in your response
"""

template_for_general = """
You are a helpful AI assistant that answers general knowledge questions.

Question: {question}

Instructions for your response:
1. Provide a clear, direct answer to the question
2. Use your general knowledge to give accurate information
3. Do not mention or include any URLs or SOAP requests in your response
4. Keep the response focused and relevant to the question
"""

prompt_urls_soap = ChatPromptTemplate.from_template(template_for_urls_and_soap)
prompt_general = ChatPromptTemplate.from_template(template_for_general)

def format_documents(docs):
    formatted_text = ""
    for doc in docs:
        if doc.metadata.get('type') == 'url':
            url = doc.metadata.get('url', '')
            content = doc.page_content.replace('URL: ', '').replace('Description: ', '')
            description = content.split('\n')[1] if '\n' in content else ''
            formatted_text += f"Type: URL\nURL: {url}\nDescription: {description}\n\n"
        else:  # SOAP request
            content = doc.page_content.split('\n')
            request_name = content[0].replace('RequestName: ', '')
            sample_request = '\n'.join(content[2:])  # Skip the "SampleRequest:" line
            formatted_text += f"Type: SOAP Request\nRequest Name: {request_name}\nSample Request:\n{sample_request}\n\n"
    return formatted_text

while True:
    print("\n\n-------------------------------")
    print("You can ask about:")
    print("1. URLs (e.g., 'What is the Oracle URL?')")
    print("2. SOAP Requests (e.g., 'Show me the request template for GetContactCallStatus')")
    print("3. General Questions (e.g., 'What is the capital of France?')")
    question = input("\nWhat would you like to know? (q to quit): ")
    print("\n\n")
    if question.lower() == "q":
        break
    
    if is_url_or_soap_request(question):
        # Only retrieve and format URLs/SOAP requests for relevant queries
        relevant_items = retriever.invoke(question)
        formatted_items = format_documents(relevant_items)
        result = chain = prompt_urls_soap | model
        response = result.invoke({"reviews": formatted_items, "question": question})
    else:
        # For general questions, use the general template without any URLs/SOAP data
        chain = prompt_general | model
        response = chain.invoke({"question": question})
    
    print(response)