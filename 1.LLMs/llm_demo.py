import langchain_openai import OpenAI

#This means:
#"LangChain OpenAI package se OpenAI class ko import karo."
#So now Python knows about OpenAI, and you can create/use an OpenAI model.


from dotenv import load_dotenv

#This imports load_dotenv() from the python-dotenv library.
#Its job is to load variables stored in your .env file into your Python environment.


load_dotenv()
#This actually loads the .env file.
 

 #now making an object of Openai 
llm=OpenAI(model='gpt-3.5-turbo-instruct')


#invoke() means send a prompt/input to the model and get its response.
result=llm.invoke("What is teh captical of india")
print(result)